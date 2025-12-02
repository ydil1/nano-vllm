import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass


@dataclass
class FusionOutput:
    embeddings: torch.Tensor
    positions: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    image_positions: Optional[List[Tuple[int, int]]] = None  # (start, end) for each image


class OptimizedImageTextFusion(nn.Module):
    def __init__(
        self,
        image_token_index: int = 32000,
        use_memory_efficient: bool = True,
    ):
        super().__init__()
        self.image_token_index = image_token_index
        self.use_memory_efficient = use_memory_efficient
    
    @torch.jit.script_if_tracing
    def _find_image_token_positions(
        self,
        input_ids: torch.Tensor,
        image_token_index: int,
    ) -> List[Tuple[int, int]]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        positions = []
        for batch_idx in range(input_ids.shape[0]):
            seq = input_ids[batch_idx]
            mask = seq == image_token_index
            
            if not mask.any():
                continue
            indices = mask.nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                start = indices[0].item()
                end = start
                
                for i in range(1, len(indices)):
                    if indices[i] == indices[i-1] + 1:
                        end = indices[i].item()
                    else:
                        positions.append((start, end + 1))
                        start = indices[i].item()
                        end = start
                
                positions.append((start, end + 1))
        
        return positions
    
    def _efficient_merge(
        self,
        text_embeds: torch.Tensor,
        image_features: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> FusionOutput:
        batch_size = text_embeds.shape[0] if text_embeds.dim() > 2 else 1
        device = text_embeds.device
        dtype = text_embeds.dtype
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(0)
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(0)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        
        seq_len = text_embeds.shape[1]
        embed_dim = text_embeds.shape[2]
        num_image_tokens = image_features.shape[1]
        image_positions = self._find_image_token_positions(input_ids, self.image_token_index)
        
        if not image_positions:
            return FusionOutput(
                embeddings=text_embeds.squeeze(0) if batch_size == 1 else text_embeds,
                positions=positions.squeeze(0) if batch_size == 1 else positions,
                image_positions=None,
            )
        num_image_placeholders = sum(end - start for start, end in image_positions)
        new_seq_len = seq_len - num_image_placeholders + num_image_tokens * len(image_positions)
        merged_embeds = torch.empty(
            batch_size, new_seq_len, embed_dim,
            device=device, dtype=dtype
        )
        new_positions = torch.empty(
            batch_size, new_seq_len,
            device=device, dtype=positions.dtype
        )
        for b in range(batch_size):
            src_idx = 0
            dst_idx = 0
            
            for start, end in image_positions:
                if start > src_idx:
                    chunk_len = start - src_idx
                    merged_embeds[b, dst_idx:dst_idx + chunk_len] = text_embeds[b, src_idx:start]
                    new_positions[b, dst_idx:dst_idx + chunk_len] = positions[b, src_idx:start]
                    dst_idx += chunk_len
                img_features = image_features[b] if image_features.shape[0] > 1 else image_features[0]
                merged_embeds[b, dst_idx:dst_idx + num_image_tokens] = img_features

                if positions.dim() > 1:
                    start_pos = positions[b, start] if start < seq_len else positions[b, -1]
                    end_pos = positions[b, min(end, seq_len - 1)] if end < seq_len else positions[b, -1] + 1
                    new_positions[b, dst_idx:dst_idx + num_image_tokens] = torch.linspace(
                        start_pos, end_pos, num_image_tokens, device=device, dtype=positions.dtype
                    )
                else:
                    new_positions[b, dst_idx:dst_idx + num_image_tokens] = torch.arange(
                        dst_idx, dst_idx + num_image_tokens, device=device, dtype=positions.dtype
                    )
                
                dst_idx += num_image_tokens
                src_idx = end
            if src_idx < seq_len:
                remaining = seq_len - src_idx
                merged_embeds[b, dst_idx:dst_idx + remaining] = text_embeds[b, src_idx:]
                new_positions[b, dst_idx:dst_idx + remaining] = positions[b, src_idx:] if positions.dim() > 1 else torch.arange(dst_idx, dst_idx + remaining, device=device, dtype=positions.dtype)

        if batch_size == 1:
            merged_embeds = merged_embeds.squeeze(0)
            new_positions = new_positions.squeeze(0)
        
        return FusionOutput(
            embeddings=merged_embeds,
            positions=new_positions,
            image_positions=image_positions,
        )
    
    def forward(
        self,
        text_embeds: torch.Tensor,
        image_features: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> FusionOutput:
        if self.use_memory_efficient:
            output = self._efficient_merge(
                text_embeds, image_features, input_ids, positions
            )
        else:
            output = self._simple_merge(
                text_embeds, image_features, input_ids, positions
            )
        if attention_mask is not None:
            output.attention_mask = self._update_attention_mask(
                attention_mask, output.image_positions, image_features.shape[1]
            )
        
        return output
    
    def _simple_merge(
        self,
        text_embeds: torch.Tensor,
        image_features: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> FusionOutput:
        image_mask = input_ids == self.image_token_index
        
        if not image_mask.any():
            return FusionOutput(embeddings=text_embeds, positions=positions)
        
        merged = []
        pos_list = []
        
        for i, token_id in enumerate(input_ids):
            if token_id == self.image_token_index:
                merged.append(image_features)
                img_positions = torch.arange(
                    len(pos_list),
                    len(pos_list) + image_features.shape[-2],
                    device=positions.device,
                    dtype=positions.dtype
                )
                pos_list.append(img_positions)
            else:
                if text_embeds.dim() == 2:
                    merged.append(text_embeds[i:i+1])
                else:
                    merged.append(text_embeds[:, i:i+1])
                pos_list.append(positions[i:i+1] if positions.dim() == 1 else positions[:, i:i+1])
        
        # Concatenate
        merged_embeds = torch.cat(merged, dim=-2 if text_embeds.dim() == 2 else 1)
        new_positions = torch.cat(pos_list, dim=0 if positions.dim() == 1 else 1)
        
        return FusionOutput(embeddings=merged_embeds, positions=new_positions)
    
    def _update_attention_mask(
        self,
        attention_mask: torch.Tensor,
        image_positions: List[Tuple[int, int]],
        num_image_tokens: int,
    ) -> torch.Tensor:
        if not image_positions:
            return attention_mask

        batch_size = attention_mask.shape[0]
        old_len = attention_mask.shape[1]
        num_replacements = len(image_positions)
        tokens_removed = sum(end - start for start, end in image_positions)
        tokens_added = num_image_tokens * num_replacements
        new_len = old_len - tokens_removed + tokens_added
        new_mask = torch.zeros(batch_size, new_len, dtype=attention_mask.dtype, device=attention_mask.device)
        
        for b in range(batch_size):
            src_idx = 0
            dst_idx = 0
            
            for start, end in image_positions:
                if start > src_idx:
                    chunk_len = start - src_idx
                    new_mask[b, dst_idx:dst_idx + chunk_len] = attention_mask[b, src_idx:start]
                    dst_idx += chunk_len
                new_mask[b, dst_idx:dst_idx + num_image_tokens] = 1
                dst_idx += num_image_tokens
                src_idx = end
            if src_idx < old_len:
                remaining = old_len - src_idx
                new_mask[b, dst_idx:dst_idx + remaining] = attention_mask[b, src_idx:]
        
        return new_mask


def create_optimized_fusion(
    image_token_index: int = 32000,
    use_memory_efficient: bool = True,
) -> OptimizedImageTextFusion:
    return OptimizedImageTextFusion(
        image_token_index=image_token_index,
        use_memory_efficient=use_memory_efficient,
    )