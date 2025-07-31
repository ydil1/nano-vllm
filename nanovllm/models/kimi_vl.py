"""
KimiVL Model Implementation for nano-vllm

This is a multi-modal vision-language model that combines:
1. MoonViT vision encoder (to be implemented in moonvit.py)
2. Multi-modal projector to map vision features to text space
3. DeepSeekV2-based language model
4. Support for interleaved image and text inputs

Architecture Overview:
- Vision Tower: Processes images into feature representations
- Projector: Maps vision features to language model dimension
- Language Model: Processes combined text and image embeddings
- LM Head: Generates output tokens
"""

import math
import torch
from torch import nn
import torch.distributed as dist
from dataclasses import dataclass
from typing import Literal, Union, TypedDict, Optional, List
from transformers import KimiVLConfig

# nano-vllm layers
from nanovllm.layers.embed_head import ParallelLMHead
from .moonvit import MoonViTPretrainedModel
from .kimi_vl_layers import (
    GELUActivation,
    merge_multimodal_embeddings,
    get_num_image_tokens,
    KimiVLModel,
)

# For dummy input only
@dataclass
class MaxImageTokenMeta:
    width: int = 1024
    height: int = 1024

class KimiVLMultiModalProjector(nn.Module):

    def __init__(self, config: KimiVLConfig):
        super().__init__()

        self.hidden_size = (config.vision_config.hidden_size *
                            config.vision_config.merge_kernel_size[0] *
                            config.vision_config.merge_kernel_size[1])

        self.pre_norm = torch.nn.LayerNorm(config.vision_config.hidden_size,
                                           eps=1e-5)
        self.linear_1 = nn.Linear(self.hidden_size,
                                  self.hidden_size,
                                  bias=True)
        self.act = GELUActivation()
        self.linear_2 = nn.Linear(self.hidden_size,
                                  config.text_config.hidden_size,
                                  bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(image_features).view(
            -1, self.hidden_size)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class KimiVLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: Union[torch.Tensor, list[torch.Tensor]]
    """
    Shape:`(num_patches, num_channels, patch_size, patch_size)`
    """

    image_grid_hws: torch.Tensor
    """Shape:`(num_images, 2)`"""


# TODO: support embeds too
# We only support pixel input for kimi-vl now
KimiVLImageInputs = KimiVLImagePixelInputs
# Type alias for nested tensors
NestedTensors = List[torch.Tensor]


# Main Model Class for KimiVL

class KimiVLForConditionalGeneration(nn.Module):
    # Mapping for loading weights with merged projections
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    
    def __init__(self, config: KimiVLConfig):
        super().__init__()
        self.config = config
        
        # Vision components
        self.vision_tower = MoonViTPretrainedModel(config.vision_config)
        
        # Multi-modal projector
        self.multi_modal_projector = KimiVLMultiModalProjector(config)
        
        # Language model
        self.language_model = KimiVLModel(config.text_config)
        
        # LM Head for output generation
        self.lm_head = ParallelLMHead(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
        )
        
        # Special tokens
        self.media_placeholder_token_id = config.media_placeholder_token_id
        
        # Cache configuration parameters
        self.patch_size = config.vision_config.patch_size
        self.merge_kernel_size = config.vision_config.merge_kernel_size
        self.in_token_limit = getattr(config.vision_config, 'in_token_limit', 4096)
    
    def _validate_and_reshape_mm_tensor(
        self,
        mm_input: object,
        name: str
    ) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                           f"Got type: {type(mm_input)}")
        
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(f"{name} should be 2D or batched 3D tensor. "
                               f"Got ndim: {mm_input.ndim} "
                               f"(shape={mm_input.shape})")
            return mm_input.reshape(-1, mm_input.shape[-1])
        else:
            return torch.concat(mm_input)
    
    def _parse_and_validate_image_input(
        self,
        **kwargs: object
    ) -> Optional[KimiVLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_hws = kwargs.pop("image_grid_hws", None)
        
        if pixel_values is None:
            return None
        
        # Validate and reshape grid sizes
        image_grid_hws = self._validate_and_reshape_mm_tensor(
            image_grid_hws, "image grid hws"
        )
        
        # Process pixel values
        num_channels = 3
        patch_size = self.patch_size
        
        if isinstance(pixel_values, list):
            pixel_values = torch.cat([
                x.reshape(-1, num_channels, patch_size, patch_size)
                for x in pixel_values
            ])
        else:
            pixel_values = pixel_values.reshape(-1, num_channels, patch_size,
                                              patch_size)
        
        # Convert to appropriate dtype if vision tower exists
        if self.vision_tower is not None:
            pixel_values = pixel_values.to(self.vision_tower.dtype)
        
        # Validate grid_hws shape
        assert image_grid_hws.ndim == 2, \
            f"unexpected shape for image_grid_hws: {image_grid_hws.shape}"
        
        return KimiVLImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_hws=image_grid_hws,
        )
    
    @torch.inference_mode()
    def _process_image_pixels(
        self,
        inputs: KimiVLImagePixelInputs
    ) -> List[torch.Tensor]:
        assert self.vision_tower is not None, "Vision tower is not initialized"
        
        pixel_values = inputs["pixel_values"]
        image_grid_hws = inputs["image_grid_hws"]
        return self.vision_tower(pixel_values, image_grid_hws)
    
    def _process_image_input(
        self,
        image_input: KimiVLImageInputs
    ) -> List[torch.Tensor]:
        assert image_input["type"] == "pixel_values"
        
        # Get vision features
        image_features = self._process_image_pixels(image_input)
        assert isinstance(image_features, list)
        
        # Project features to language model space
        lengths = [x.shape[0] for x in image_features]
        concatenated = torch.cat(image_features)
        projected = self.multi_modal_projector(concatenated)
        
        # Split back into individual images
        return projected.split(lengths)
    
    def get_multimodal_embeddings(
        self,
        **kwargs: object
    ) -> Optional[NestedTensors]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        
        # Process images through vision encoder and projector
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings
    
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        # Get text embeddings
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        
        # Merge multi-modal embeddings if provided
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=self.media_placeholder_token_id
            )
        
        return inputs_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            
            if image_input is None:
                # Text-only input
                inputs_embeds = None
            else:
                # Multi-modal input
                inputs_embeds = self.get_input_embeddings(input_ids)
                image_embeds = self._process_image_input(image_input)
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids,
                    inputs_embeds,
                    image_embeds,
                    placeholder_token_id=self.media_placeholder_token_id,
                )
                input_ids = None 
        
        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )
        
        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
    
    def load_weights(self, weights):
        params_dict = dict(self.named_parameters())
        
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            
            if "vision" in name and name in params_dict:
                param = params_dict[name]
                param.data.copy_(loaded_weight)
                continue
            
            loaded = False
            for source_name, (target_name, shard_id) in self.packed_modules_mapping.items():
                if source_name in name:
                    new_name = name.replace(source_name, target_name)
                    if new_name in params_dict:
                        param = params_dict[new_name]
                        if hasattr(param, "weight_loader"):
                            param.weight_loader(param, loaded_weight, shard_id)
                        else:
                            param.data.copy_(loaded_weight)
                        loaded = True
                        break
            
            if not loaded and name in params_dict:
                param = params_dict[name]
                if hasattr(param, "weight_loader"):
                    param.weight_loader(param, loaded_weight)
                else:
                    param.data.copy_(loaded_weight)



