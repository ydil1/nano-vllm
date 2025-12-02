"""
Llama model implementation for nano-vLLM
Optimized for LLaVA's text-only inference
"""
import torch
from torch import nn
import torch.distributed as dist
from typing import Optional
from transformers import LlamaConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class LlamaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 10000, 
        rope_scaling: dict | None = None,
        layer_idx: int = 0,
        use_kv_cache: bool = False,
    ) -> None:
        super().__init__()
        # Get tensor parallel size safely
        try:
            tp_size = dist.get_world_size() if dist.is_initialized() else 1
        except:
            tp_size = 1
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.layer_idx = layer_idx
        self.use_kv_cache = use_kv_cache
        self.kv_cache = None  # Will be initialized when needed

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        
        # RoPE
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional['LayerKVCache'] = None,
    ) -> torch.Tensor:
        # Handle both 2D (flattened) and 3D tensor inputs
        original_shape = hidden_states.shape
        if len(hidden_states.shape) == 2:
            # Assume batch_size = 1 for 2D inputs
            batch_size = 1
            seq_len = hidden_states.shape[0]
            hidden_dim = hidden_states.shape[1]
            hidden_states = hidden_states.unsqueeze(0)  # Add batch dimension
        else:
            batch_size, seq_len, _ = hidden_states.shape

        # Handle positions tensor shape
        # positions can be [seq_len] for single sequence or [batch_size, seq_len] for batch
        if positions.ndim == 1:
            # If 1D, expand to match batch_size
            if positions.size(0) == seq_len:
                positions = positions.unsqueeze(0).expand(batch_size, -1)
            # else assume it's already [batch_size * seq_len]

        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        # Use reshape instead of view for non-contiguous tensors (e.g., after padding)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary embedding
        # The rotary embedding expects [batch_size * seq_len, num_heads, head_dim]
        # So we need to reshape temporarily
        q_rot = q.reshape(batch_size * seq_len, self.num_heads, self.head_dim)
        k_rot = k.reshape(batch_size * seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary with flattened positions - ensure positions matches num_tokens
        positions_flat = positions.reshape(-1)
        q_rot, k_rot = self.rotary_emb(positions_flat, q_rot, k_rot)

        # Reshape back to [batch_size, seq_len, num_heads, head_dim]
        q = q_rot.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k_rot.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Use KV cache if provided
        if kv_cache is not None:
            # Transpose for cache: [batch_size, seq_len, num_heads, head_dim]
            k_cache, v_cache = kv_cache.update(k, v, positions)
            # k_cache and v_cache are [batch_size, num_heads, cache_len, head_dim]
            k = k_cache.transpose(1, 2)  # Back to [batch_size, cache_len, num_heads, head_dim]
            v = v_cache.transpose(1, 2)
            # Update seq_len to cached length
            cache_seq_len = k_cache.shape[2]
        else:
            cache_seq_len = seq_len
        
        # For single-token or vLLM-style attention, use the existing attention layer
        # For multi-token sequences, we need proper attention computation
        if seq_len == 1 and kv_cache is None:
            # Single token without cache - use existing optimized path
            q = q.squeeze(1)  # Remove seq_len dimension
            k = k.squeeze(1)
            v = v.squeeze(1)
            o = self.attn(q, k, v)
            o = o.unsqueeze(1)  # Add back seq_len dimension
        else:
            # Multi-token sequence - compute attention properly
            import torch.nn.functional as F
            
            # Transpose for attention: [batch_size, num_heads, seq_len, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Handle multi-query attention if needed
            if self.num_kv_heads != self.num_heads:
                repeat_factor = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)
            
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
            
            # Apply causal mask - handle cached sequences
            if kv_cache is not None and cache_seq_len > seq_len:
                # Attending to cached sequence
                mask = torch.ones(seq_len, cache_seq_len, dtype=torch.bool, device=q.device)
                # Allow attention to all previous cached positions
                start_pos = cache_seq_len - seq_len
                mask[:, :start_pos + seq_len] = False
                mask = torch.triu(mask, diagonal=start_pos + 1)
            else:
                # Normal causal mask
                mask = torch.triu(torch.ones(seq_len, cache_seq_len, dtype=torch.bool, device=q.device), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))
            
            # Softmax
            attn_weights = F.softmax(scores, dim=-1)
            
            # Apply attention to values
            o = torch.matmul(attn_weights, v)
            
            # Transpose back: [batch_size, seq_len, num_heads, head_dim]
            o = o.transpose(1, 2)
            
            # Reshape to [batch_size, seq_len, hidden_size]
            o = o.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Output projection
        output = self.o_proj(o)
        return output


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu", f"Llama only supports silu, got {hidden_act}"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = LlamaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            head_dim=getattr(config, 'head_dim', None),
            layer_idx=layer_idx,
        )
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        kv_cache: Optional['LayerKVCache'] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions, hidden_states, kv_cache)
        
        # MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        # For LLaVA, we may need to handle image tokens beyond vocab_size
        # If image_token_index >= vocab_size, extend the vocabulary
        vocab_size = config.vocab_size
        if hasattr(config, 'image_token_index') and config.image_token_index >= vocab_size:
            vocab_size = config.image_token_index + 1
        self.embed_tokens = VocabParallelEmbedding(vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        kv_caches: Optional[list] = None,
    ) -> torch.Tensor:
        # Handle both 2D and 3D inputs
        original_shape = None
        if input_ids is not None:
            original_shape = input_ids.shape
            if len(input_ids.shape) == 1:
                # Flattened input - add batch dimension
                input_ids = input_ids.unsqueeze(0)
        
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        
        # Ensure hidden_states is 3D for processing
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states.unsqueeze(0)
        
        residual = None
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            hidden_states, residual = layer(positions, hidden_states, residual, layer_cache)
        
        hidden_states, _ = self.norm(hidden_states, residual)
        
        # Return in original format if needed
        if original_shape is not None and len(original_shape) == 1:
            hidden_states = hidden_states.squeeze(0)
        
        return hidden_states


class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        # Match lm_head vocab size to embed_tokens
        vocab_size = config.vocab_size
        if hasattr(config, 'image_token_index') and config.image_token_index >= vocab_size:
            vocab_size = config.image_token_index + 1
        self.lm_head = ParallelLMHead(vocab_size, config.hidden_size)
        
        # Only tie embeddings if explicitly requested
        # Note: LLaVA has separate lm_head weights, so we don't tie them
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        kv_caches: Optional[list] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, inputs_embeds, kv_caches)
        logits = self.compute_logits(hidden_states)
        return logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def load_weights(self, weights: list[tuple[str, torch.Tensor]]):
        loaded = set()
        
        # First pass: collect weights that need to be packed
        qkv_weights = {}  # layer_idx -> {q: tensor, k: tensor, v: tensor}
        gate_up_weights = {}  # layer_idx -> {gate: tensor, up: tensor}
        regular_weights = []
        
        for name, weight in weights:
            # Skip rotary embeddings
            if "rotary_emb" in name:
                continue
            
            # Skip vision tower weights - they have different dimensions
            if "vision_tower" in name or "vision_model" in name:
                continue
            
            # Check for QKV weights (language model only)
            if ".q_proj." in name or ".k_proj." in name or ".v_proj." in name:
                # Extract layer index
                import re
                layer_match = re.search(r'layers\.(\d+)\.', name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    if layer_idx not in qkv_weights:
                        qkv_weights[layer_idx] = {}
                    
                    if ".q_proj." in name:
                        qkv_weights[layer_idx]['q'] = weight
                    elif ".k_proj." in name:
                        qkv_weights[layer_idx]['k'] = weight
                    elif ".v_proj." in name:
                        qkv_weights[layer_idx]['v'] = weight
                    loaded.add(name)
            # Check for gate/up weights (language model only)
            elif (".gate_proj." in name or ".up_proj." in name) and "vision" not in name:
                # Extract layer index
                import re
                layer_match = re.search(r'layers\.(\d+)\.', name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    if layer_idx not in gate_up_weights:
                        gate_up_weights[layer_idx] = {}
                    
                    if ".gate_proj." in name:
                        gate_up_weights[layer_idx]['gate'] = weight
                    elif ".up_proj." in name:
                        gate_up_weights[layer_idx]['up'] = weight
                    loaded.add(name)
            else:
                regular_weights.append((name, weight))
        
        # Pack QKV weights
        for layer_idx, weights_dict in qkv_weights.items():
            if 'q' in weights_dict and 'k' in weights_dict and 'v' in weights_dict:
                q_weight = weights_dict['q']
                k_weight = weights_dict['k']
                v_weight = weights_dict['v']
                
                # Find the qkv_proj parameter
                param_name = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
                param = None
                for pname, p in self.named_parameters():
                    if pname == param_name:
                        param = p
                        break
                
                if param is not None:
                    # Pack Q, K, V into single tensor
                    # The expected shape is [q_size + kv_size + kv_size, hidden_size]
                    q_size = q_weight.shape[0]
                    kv_size = k_weight.shape[0]
                    
                    try:
                        param.data[:q_size] = q_weight.to(param.device)
                        param.data[q_size:q_size+kv_size] = k_weight.to(param.device)
                        param.data[q_size+kv_size:] = v_weight.to(param.device)
                    except RuntimeError as e:
                        print(f"Warning: Could not load QKV weights for layer {layer_idx}: {e}")
        
        # Pack gate-up weights
        for layer_idx, weights_dict in gate_up_weights.items():
            if 'gate' in weights_dict and 'up' in weights_dict:
                gate_weight = weights_dict['gate']
                up_weight = weights_dict['up']
                
                # Find the gate_up_proj parameter
                param_name = f"model.layers.{layer_idx}.mlp.gate_up_proj.weight"
                param = None
                for pname, p in self.named_parameters():
                    if pname == param_name:
                        param = p
                        break
                
                if param is not None:
                    # Pack gate and up into single tensor
                    gate_size = gate_weight.shape[0]
                    
                    try:
                        param.data[:gate_size] = gate_weight.to(param.device)
                        param.data[gate_size:] = up_weight.to(param.device)
                    except RuntimeError as e:
                        print(f"Warning: Could not load gate-up weights for layer {layer_idx}: {e}")
        
        # Load regular weights
        for name, weight in regular_weights:
            # Try different name mappings
            param = None
            
            # Try as-is
            for pname, p in self.named_parameters():
                if pname == name:
                    param = p
                    break
            
            # Try with model. prefix
            if param is None and not name.startswith("model."):
                test_name = "model." + name
                for pname, p in self.named_parameters():
                    if pname == test_name:
                        param = p
                        break
            
            # Try without model. prefix
            if param is None and name.startswith("model."):
                test_name = name[6:]
                for pname, p in self.named_parameters():
                    if pname == test_name:
                        param = p
                        break
            
            if param is not None:
                # Special handling for vocab size mismatch (32000 vs 32001 for LLaVA)
                if (name == 'model.embed_tokens.weight' or name == 'lm_head.weight') and \
                   param.shape[0] == weight.shape[0] + 1 and param.shape[1] == weight.shape[1]:
                    # Pad the weight with zeros for the extra image token
                    param.data[:weight.shape[0]] = weight.to(param.device)
                    param.data[weight.shape[0]:] = 0  # Initialize extra token to zeros
                    loaded.add(name)
                elif param.shape == weight.shape:
                    param.data.copy_(weight.to(param.device))
                    loaded.add(name)
                else:
                    print(f"Shape mismatch for {name}: expected {param.shape}, got {weight.shape}")
        
        return loaded