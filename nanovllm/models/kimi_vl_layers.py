"""
Language Model Components for KimiVL

This file contains the core components for the KimiVL language model:
- Attention layers
- MLP layers
- Decoder layers
- Helper functions
"""

import math
import torch
from torch import nn
from typing import List, Optional, Tuple

from nanovllm.layers.linear import (
    QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
)
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import VocabParallelEmbedding



class GELUActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(input, approximate="tanh")


def merge_multimodal_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: List[torch.Tensor],
    placeholder_token_id: int,
) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape
    embed_dim = inputs_embeds.shape[-1]
    
    placeholder_mask = (input_ids == placeholder_token_id)
    
    if not placeholder_mask.any() or len(multimodal_embeddings) == 0:
        return inputs_embeds
    total_len = seq_len
    for emb in multimodal_embeddings:
        total_len += emb.shape[0] - 1
    
    output_embeds = torch.zeros(
        batch_size, total_len, embed_dim,
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device
    )
    
    for batch_idx in range(batch_size):
        src_idx = 0
        dst_idx = 0
        img_idx = 0
        
        while src_idx < seq_len:
            if placeholder_mask[batch_idx, src_idx] and img_idx < len(multimodal_embeddings):
                img_emb = multimodal_embeddings[img_idx]
                img_len = img_emb.shape[0]
                output_embeds[batch_idx, dst_idx:dst_idx + img_len] = img_emb
                dst_idx += img_len
                img_idx += 1
                src_idx += 1
            else:
                output_embeds[batch_idx, dst_idx] = inputs_embeds[batch_idx, src_idx]
                dst_idx += 1
                src_idx += 1
    
    return output_embeds[:, :dst_idx] 


def get_num_image_tokens(
    image_width: int,
    image_height: int,
    patch_size: int,
    merge_kernel_size: tuple,
    in_token_limit: int = 4096
) -> int:
    # Check if image needs resizing due to token limit
    num_patches = (image_width // patch_size) * (image_height // patch_size)
    
    if num_patches > in_token_limit:
        # Resize image to fit within token limit
        scale = math.sqrt(in_token_limit / num_patches)
        image_width = int(image_width * scale)
        image_height = int(image_height * scale)
    
    kernel_height, kernel_width = merge_kernel_size
    
    # Calculate padding needed for kernel alignment
    pad_height = (kernel_height * patch_size - image_height % 
                  (kernel_height * patch_size)) % (kernel_height * patch_size)
    pad_width = (kernel_width * patch_size - image_width % 
                 (kernel_width * patch_size)) % (kernel_width * patch_size)
    
    # Calculate final token dimensions after merge
    token_height = (image_height + pad_height) // (kernel_height * patch_size)
    token_width = (image_width + pad_width) // (kernel_width * patch_size)
    
    return int(token_height * token_width)



class KimiVLAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        rope_scaling: dict = None,
        rms_norm_eps: float = 1e-6,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        self.tp_size = torch.distributed.get_world_size()
        self.num_heads = num_heads // self.tp_size
        self.num_kv_heads = num_kv_heads // self.tp_size

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            num_heads,
            num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        
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
            num_kv_heads=self.num_kv_heads,
        )
        
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        
        # Split into Q, K, V
        # q_size = num_heads * head_dim (per GPU)
        # kv_size = num_kv_heads * head_dim (per GPU)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        
        # Apply RMSNorm to Q and K if enabled
        if self.use_qk_norm:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            q = self.q_norm(q).view(-1, q_size)
            k = self.k_norm(k).view(-1, kv_size)
        
        # Apply rotary embeddings
        q, k = self.rotary_emb(positions, q, k)
        
        # Compute attention
        attn_output = self.attn(q, k, v)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output


class KimiVLMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
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
        # SwiGLU activation
        self.act_fn = SiluAndMul()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class KimiVLDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = KimiVLAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=getattr(config, "use_qk_norm", True),
        )
        self.mlp = KimiVLMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm with residual connection
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        # Self attention
        hidden_states = self.self_attn(positions, hidden_states)
        
        # Pre-norm for MLP with residual
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        # MLP
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual

class KimiVLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            KimiVLDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        positions: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        # Get embeddings
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        
        # Pass through decoder layers
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        # Final norm
        hidden_states, _ = self.norm(hidden_states, residual)
        
        return hidden_states