from typing import Optional, Union, List, Tuple
import torch
import torch.nn as nn
from transformers import CLIPVisionConfig

from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear, QKVParallelLinear
from nanovllm.layers.activation import get_act_fn
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.attention import Attention
import torch.nn.functional as F


class CLIPEncoderInfo:
    
    def __init__(self, vision_config: CLIPVisionConfig):
        self.vision_config = vision_config
    
    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int:
        # CLIP uses CLS token + patch tokens
        return self.get_patch_grid_length() ** 2 + 1
    
    def get_image_size(self) -> int:
        return self.vision_config.image_size
    
    def get_patch_size(self) -> int:
        return self.vision_config.patch_size
    
    def get_patch_grid_length(self) -> int:
        image_size = self.get_image_size()
        patch_size = self.get_patch_size()
        assert image_size % patch_size == 0
        return image_size // patch_size


class CLIPVisionEmbeddings(nn.Module):
    
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        assert self.image_size % self.patch_size == 0
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        return embeddings


class CLIPAttention(nn.Module):
    
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got embed_dim: {self.embed_dim} and num_heads: {self.num_heads})"
            )
        
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.num_heads,
            self.num_heads,  # num_kv_heads = num_heads for CLIP
            bias=True,
        )
        
        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scale,
            num_kv_heads=self.num_heads,  # CLIP uses standard MHA, not GQA
        )
        
        self.use_flash_attn = getattr(config, 'use_flash_attention', True)
        
        self.out_proj = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            bias=True
        )
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, None]:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_heads * self.head_dim  # Same as q_size for CLIP
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        if self.use_flash_attn and q.is_cuda:
            attn_output = self.attn(q, k, v)
        else:
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size * seq_len, self.embed_dim)
        
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        
        attn_output = self.out_proj(attn_output)
        
        return attn_output, None


class CLIPMLP(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        
        self.activation_fn = get_act_fn(config.hidden_act)
        
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True
        )
        
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.self_attn = CLIPAttention(config)
        self.mlp = CLIPMLP(config)
        
        use_fused_ln = getattr(config, 'use_fused_layer_norm', False)
        if use_fused_ln:
            try:
                from apex.normalization import FusedLayerNorm
                self.layer_norm1 = FusedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
                self.layer_norm2 = FusedLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            except ImportError:
                self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
                self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(hidden_states)
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def _forward_with_checkpointing(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint
        
        def attn_block(x):
            residual = x
            x = self.layer_norm1(x)
            x, _ = self.self_attn(x)
            return residual + x
        
        def mlp_block(x):
            residual = x
            x = self.layer_norm2(x)
            x = self.mlp(x)
            return residual + x
        
        hidden_states = checkpoint(attn_block, hidden_states)
        hidden_states = checkpoint(mlp_block, hidden_states)
        return hidden_states


class CLIPEncoder(nn.Module):
    
    def __init__(
        self,
        config: CLIPVisionConfig,
        num_hidden_layers_override: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override
        
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(config) for _ in range(num_hidden_layers)
        ])
    
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        return_all_hidden_states: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        hidden_states_pool = [inputs_embeds] if return_all_hidden_states else []
        hidden_states = inputs_embeds
        
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
            if return_all_hidden_states:
                hidden_states_pool.append(hidden_states)
        
        if return_all_hidden_states:
            return hidden_states_pool
        return hidden_states


class CLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        config: CLIPVisionConfig,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
    ):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = CLIPVisionEmbeddings(config)
        
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        
        self.encoder = CLIPEncoder(config, num_hidden_layers_override)
        
        num_hidden_layers = config.num_hidden_layers
        if require_post_norm is None:
            require_post_norm = len(self.encoder.layers) == num_hidden_layers
        
        if require_post_norm:
            self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        else:
            self.post_layernorm = None
    
    def load_weights(self, weights):
        loaded = set()
        
        layer_weights = {}
        regular_weights = []
        
        for name, weight in weights:
            if name.startswith("vision_tower."):
                name = name[len("vision_tower."):]
            
            if any(proj in name for proj in ['.q_proj.', '.k_proj.', '.v_proj.']):
                match = re.search(r'layers\.(\d+)\.', name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx not in layer_weights:
                        layer_weights[layer_idx] = {}
                    
                    if '.q_proj.weight' in name:
                        layer_weights[layer_idx]['q_weight'] = weight
                    elif '.k_proj.weight' in name:
                        layer_weights[layer_idx]['k_weight'] = weight
                    elif '.v_proj.weight' in name:
                        layer_weights[layer_idx]['v_weight'] = weight
                    elif '.q_proj.bias' in name:
                        layer_weights[layer_idx]['q_bias'] = weight
                    elif '.k_proj.bias' in name:
                        layer_weights[layer_idx]['k_bias'] = weight
                    elif '.v_proj.bias' in name:
                        layer_weights[layer_idx]['v_bias'] = weight
                    loaded.add(name)
            else:
                regular_weights.append((name, weight))
        
        for layer_idx, weights_dict in layer_weights.items():
            if 'q_weight' in weights_dict and 'k_weight' in weights_dict and 'v_weight' in weights_dict:
                q_weight = weights_dict['q_weight']
                k_weight = weights_dict['k_weight']
                v_weight = weights_dict['v_weight']
                
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                
                param_name = f"vision_model.encoder.layers.{layer_idx}.self_attn.qkv_proj.weight"
                for pname, param in self.named_parameters():
                    if pname == param_name:
                        param.data.copy_(qkv_weight)
                        break
                
                if 'q_bias' in weights_dict and 'k_bias' in weights_dict and 'v_bias' in weights_dict:
                    q_bias = weights_dict['q_bias']
                    k_bias = weights_dict['k_bias']
                    v_bias = weights_dict['v_bias']
                    
                    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                    
                    param_name = f"vision_model.encoder.layers.{layer_idx}.self_attn.qkv_proj.bias"
                    for pname, param in self.named_parameters():
                        if pname == param_name:
                            param.data.copy_(qkv_bias)
                            break
        
        for name, weight in regular_weights:
            for pname, param in self.named_parameters():
                if pname == name:
                    if param.shape == weight.shape:
                        param.data.copy_(weight)
                        loaded.add(name)
                    break
        
        return loaded
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        feature_sample_layers: Optional[List[int]] = None
    ) -> torch.Tensor:
        
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        
        return_all_hidden_states = feature_sample_layers is not None
        encoder_outputs = self.encoder(
            hidden_states,
            return_all_hidden_states=return_all_hidden_states
        )
        
        if feature_sample_layers is not None:
            selected_layers = []
            for layer_idx in feature_sample_layers:
                if layer_idx < 0:
                    layer_idx = len(encoder_outputs) + layer_idx
                selected_layers.append(encoder_outputs[layer_idx])
            
            encoder_outputs = torch.stack(selected_layers, dim=0)
            
            if self.post_layernorm is not None:
                encoder_outputs = self.post_layernorm(encoder_outputs)
        else:
            if self.post_layernorm is not None:
                encoder_outputs = self.post_layernorm(encoder_outputs)
        
        return encoder_outputs


class CLIPVisionModel(nn.Module):
    def __init__(
        self,
        config: CLIPVisionConfig,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
    ):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(
            config=config,
            num_hidden_layers_override=num_hidden_layers_override,
            require_post_norm=require_post_norm,
        )
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        feature_sample_layers: Optional[List[int]] = None
    ) -> torch.Tensor:
        
        return self.vision_model(pixel_values, feature_sample_layers)
    
    def load_weights(self, weights: List[Tuple[str, torch.Tensor]]) -> set:
        loaded_params = set()
        params_dict = dict(self.named_parameters())
        
        qkv_weights = {}  # layer_idx -> {q: tensor, k: tensor, v: tensor}
        qkv_biases = {}   # layer_idx -> {q: tensor, k: tensor, v: tensor}
        regular_weights = []
        
        for name, loaded_weight in weights:
            if (name.startswith("vision_model.post_layernorm") and 
                self.vision_model.post_layernorm is None):
                continue
            
            if "self_attn.q_proj" in name or "self_attn.k_proj" in name or "self_attn.v_proj" in name:
                import re
                layer_match = re.search(r'layers\.(\d+)\.', name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    
                    if name.endswith('.weight'):
                        if layer_idx not in qkv_weights:
                            qkv_weights[layer_idx] = {}
                        
                        if "q_proj" in name:
                            qkv_weights[layer_idx]['q'] = loaded_weight
                        elif "k_proj" in name:
                            qkv_weights[layer_idx]['k'] = loaded_weight
                        elif "v_proj" in name:
                            qkv_weights[layer_idx]['v'] = loaded_weight
                    
                    elif name.endswith('.bias'):
                        if layer_idx not in qkv_biases:
                            qkv_biases[layer_idx] = {}
                        
                        if "q_proj" in name:
                            qkv_biases[layer_idx]['q'] = loaded_weight
                        elif "k_proj" in name:
                            qkv_biases[layer_idx]['k'] = loaded_weight
                        elif "v_proj" in name:
                            qkv_biases[layer_idx]['v'] = loaded_weight
                    
                    loaded_params.add(name)
            else:
                regular_weights.append((name, loaded_weight))
        
        for layer_idx, weights_dict in qkv_weights.items():
            if 'q' in weights_dict and 'k' in weights_dict and 'v' in weights_dict:
                q_weight = weights_dict['q']
                k_weight = weights_dict['k']
                v_weight = weights_dict['v']
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                
                param_name = f"vision_model.encoder.layers.{layer_idx}.self_attn.qkv_proj.weight"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    param.data.copy_(qkv_weight.to(param.device))
                    loaded_params.add(param_name)
        
        for layer_idx, biases_dict in qkv_biases.items():
            if 'q' in biases_dict and 'k' in biases_dict and 'v' in biases_dict:
                q_bias = biases_dict['q']
                k_bias = biases_dict['k']
                v_bias = biases_dict['v']
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                
                param_name = f"vision_model.encoder.layers.{layer_idx}.self_attn.qkv_proj.bias"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    param.data.copy_(qkv_bias.to(param.device))
                    loaded_params.add(param_name)
        
        for name, loaded_weight in regular_weights:
            if name in params_dict:
                param = params_dict[name]
                param.data.copy_(loaded_weight.to(param.device))
                loaded_params.add(name)
        
        return loaded_params