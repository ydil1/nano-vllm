import torch
from torch import nn
import torch.nn.functional as F
from transformers import LlavaConfig, CLIPVisionConfig
from typing import Optional, List, Tuple, Union

from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from nanovllm.layers.activation import SiluAndMul
from nanovllm.models.llama import LlamaForCausalLM
from nanovllm.models.clip import CLIPVisionModel
from nanovllm.models.clip_image_processing import CLIPImageProcessor
from nanovllm.models.llava_fusion import OptimizedImageTextFusion
try:
    from nanovllm.multimodal import (
        MULTIMODAL_REGISTRY,
        LlavaProcessingInfo,
        LlavaMultiModalProcessor,
        LlavaDummyInputsBuilder,
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False


class LlavaMultiModalProjector(nn.Module):
    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        projector_hidden_act: str = "gelu",
    ):
        super().__init__()
        self.linear_1 = ColumnParallelLinear(
            vision_hidden_size,
            text_hidden_size,
            bias=True,
        )
        
        if projector_hidden_act == "gelu":
            self.act = nn.GELU()
        elif projector_hidden_act == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {projector_hidden_act}")
        
        self.linear_2 = RowParallelLinear(
            text_hidden_size,
            text_hidden_size,
            bias=True,
        )
    
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states



class LlavaForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.config = config
        num_hidden_layers_override = self._get_num_hidden_layers(config)
        self.vision_tower = CLIPVisionModel(
            config.vision_config,
            num_hidden_layers_override=num_hidden_layers_override,
            require_post_norm=False
        )

        self.multi_modal_projector = LlavaMultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act,
        )
        text_config = config.text_config
        if hasattr(config, 'vocab_size'):
            text_config.vocab_size = config.vocab_size 
        self.language_model = LlamaForCausalLM(text_config)
        
        self.image_token_index = config.image_token_index
        
        self.fusion = OptimizedImageTextFusion(
            image_token_index=config.image_token_index,
            use_memory_efficient=True
        )
    
    def _get_num_hidden_layers(self, config: LlavaConfig) -> int:
        vision_feature_layer = config.vision_feature_layer
        num_hidden_layers = config.vision_config.num_hidden_layers
        
        if isinstance(vision_feature_layer, int):
            if vision_feature_layer < 0:
                return num_hidden_layers + vision_feature_layer + 1
            return vision_feature_layer
        elif isinstance(vision_feature_layer, (list, tuple)):
            max_layer = max(
                (layer + num_hidden_layers + 1 if layer < 0 else layer)
                for layer in vision_feature_layer
            )
            return max_layer
        else:
            return num_hidden_layers
    
    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, embed_dim = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        
        input_ids_flat = input_ids.view(-1)
        inputs_embeds_flat = inputs_embeds.view(-1, embed_dim)
        image_token_mask = input_ids_flat == self.image_token_index
        
        if image_token_mask.any():
            num_image_tokens = image_token_mask.sum().item()
            if image_features.dim() == 3:
                image_features_flat = image_features.view(-1, image_features.shape[-1])
            else:
                image_features_flat = image_features
            
            if image_features_flat.shape[0] < num_image_tokens:
                repeats = (num_image_tokens // image_features_flat.shape[0]) + 1
                image_features_flat = image_features_flat.repeat(repeats, 1)
            
            inputs_embeds_flat[image_token_mask] = image_features_flat[:num_image_tokens].to(dtype)
        
        inputs_embeds = inputs_embeds_flat.view(batch_size, sequence_length, embed_dim)
        
        return inputs_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        has_vision = pixel_values is not None or image_embeds is not None
        
        if has_vision:
            if len(input_ids.shape) == 1:
                batch_size = 1
                input_ids = input_ids.unsqueeze(0)
                positions = positions.unsqueeze(0) if len(positions.shape) == 1 else positions
            else:
                batch_size = input_ids.shape[0]
            input_ids_for_embed = input_ids.clone()
            image_token_mask = input_ids == self.image_token_index
            if image_token_mask.any():
                input_ids_for_embed[image_token_mask] = 0
            
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids_for_embed)
            
            if pixel_values is not None:
                with torch.cuda.amp.autocast(enabled=False):
                    image_features = self.vision_tower(pixel_values)
                if hasattr(self.config, 'vision_feature_select_strategy'):
                    strategy = self.config.vision_feature_select_strategy
                else:
                    strategy = "default"
                
                if strategy == "default":
                    image_features = image_features[:, 1:]
                elif strategy == "full":
                    pass
                image_features = self.multi_modal_projector(image_features)
            else:
                image_features = image_embeds
            
            image_token_mask = input_ids == self.image_token_index
            
            if len(input_ids.shape) == 1:
                num_image_tokens_per_sample = image_token_mask.sum()
            else:
                num_image_tokens_per_sample = image_token_mask.sum(dim=1)
            num_image_features_per_sample = image_features.shape[1] if image_features.dim() == 3 else image_features.shape[0]
            new_embeds_list = []
            new_positions_list = []
            
            for b in range(batch_size):
                sample_input_ids = input_ids[b]
                sample_embeds = inputs_embeds[b]
                sample_positions = positions[b] if positions.dim() > 1 else positions
                sample_image_features = image_features[b] if image_features.dim() == 3 else image_features
                image_token_indices = (sample_input_ids == self.image_token_index).nonzero(as_tuple=True)[0]
                
                if len(image_token_indices) > 0:
                    first_image_idx = image_token_indices[0].item()
                    last_image_idx = image_token_indices[-1].item() + 1
                    new_embeds = torch.cat([
                        sample_embeds[:first_image_idx],
                        sample_image_features,
                        sample_embeds[last_image_idx:]
                    ], dim=0)
                    if positions.dim() == 1:
                        new_seq_len = new_embeds.shape[0]
                        new_positions = torch.arange(new_seq_len, device=positions.device, dtype=positions.dtype)
                    else:
                        new_positions = torch.cat([
                            sample_positions[:first_image_idx],
                            torch.arange(first_image_idx, first_image_idx + num_image_features_per_sample,
                                       device=positions.device, dtype=positions.dtype),
                            sample_positions[last_image_idx:] + (num_image_features_per_sample - (last_image_idx - first_image_idx))
                        ])
                    
                    new_embeds_list.append(new_embeds)
                    new_positions_list.append(new_positions)
                else:
                    new_embeds_list.append(sample_embeds)
                    new_positions_list.append(sample_positions)
            
            if batch_size > 1:
                max_len = max(e.shape[0] for e in new_embeds_list)
                padded_embeds = []
                padded_positions = []
                
                for embeds, pos in zip(new_embeds_list, new_positions_list):
                    if embeds.shape[0] < max_len:
                        pad_len = max_len - embeds.shape[0]
                        embeds = torch.cat([embeds, torch.zeros(pad_len, embeds.shape[1], device=embeds.device, dtype=embeds.dtype)])
                        pos = torch.cat([pos, pos[-1].expand(pad_len)])
                    padded_embeds.append(embeds)
                    padded_positions.append(pos)
                
                inputs_embeds = torch.stack(padded_embeds)
                positions = torch.stack(padded_positions) if positions.dim() > 1 else padded_positions[0]
            else:
                inputs_embeds = new_embeds_list[0].unsqueeze(0)
                positions = new_positions_list[0]
            if batch_size == 1 and len(positions.shape) > 1:
                positions = positions.squeeze(0)
            if batch_size == 1 and len(inputs_embeds.shape) > 2:
                inputs_embeds = inputs_embeds.squeeze(0)
            
            hidden_states = self.language_model.model(
                input_ids=None, 
                positions=positions, 
                inputs_embeds=inputs_embeds
            )
        else:
            hidden_states = self.language_model.model(input_ids, positions)
        
        return hidden_states
    
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)
    
    def load_weights(self, weights: List[Tuple[str, torch.Tensor]]):
        language_model_weights = []
        projector_weights = []
        vision_weights = []
        
        for name, weight in weights:
            if "rotary_emb" in name:
                continue
            elif name.startswith("model.vision_tower."):
                vision_weights.append((name, weight))
            elif name.startswith("model.mm_projector.") or name.startswith("model.multi_modal_projector."):
                if "model.mm_projector.0." in name:
                    new_name = name.replace("model.mm_projector.0.", "multi_modal_projector.linear_1.")
                elif "model.mm_projector.2." in name:
                    new_name = name.replace("model.mm_projector.2.", "multi_modal_projector.linear_2.")
                else:
                    new_name = name.replace("model.mm_projector.", "multi_modal_projector.")
                    new_name = new_name.replace("model.multi_modal_projector.", "multi_modal_projector.")
                projector_weights.append((new_name, weight))
            else:
                language_model_weights.append((name, weight))
        
        loaded = set()
        if language_model_weights:
            cleaned_weights = []
            for name, weight in language_model_weights:
                if name.startswith('language_model.'):
                    cleaned_name = name[len('language_model.'):]
                else:
                    cleaned_name = name
                cleaned_weights.append((cleaned_name, weight))
            
            lm_loaded = self.language_model.load_weights(cleaned_weights)
            if lm_loaded:
                loaded.update(lm_loaded)
        
        for name, weight in projector_weights:
            try:
                if name.startswith("multi_modal_projector."):
                    param_name = name[len("multi_modal_projector."):]
                else:
                    param_name = name
                param = None
                for pname, p in self.multi_modal_projector.named_parameters():
                    if pname == param_name:
                        param = p
                        break
                
                if param is not None:
                    param.data.copy_(weight.to(param.device))
                    loaded.add(name)
                else:
                    print(f"Warning: Could not find projector param {param_name} for weight {name}")
            except Exception as e:
                print(f"Error loading projector weight {name}: {e}")
        
        for name, weight in vision_weights:
            try:
                if name.startswith("model.vision_tower."):
                    param_name = name.replace("model.vision_tower.", "vision_tower.vision_model.")
                    parts = param_name.split('.')
                    module = self
                    for part in parts[:-1]:
                        if hasattr(module, part):
                            module = getattr(module, part)
                        else:
                            break
                    if hasattr(module, parts[-1]):
                        param = getattr(module, parts[-1])
                        if isinstance(param, nn.Parameter):
                            param.data.copy_(weight.to(param.device))
                            loaded.add(name)
            except Exception as e:
                pass
        
        return loaded


if REGISTRY_AVAILABLE:
    LlavaForConditionalGeneration = MULTIMODAL_REGISTRY.register_processor(
        processor=LlavaMultiModalProcessor,
        info=LlavaProcessingInfo,
        dummy_inputs=LlavaDummyInputsBuilder,
    )(LlavaForConditionalGeneration)