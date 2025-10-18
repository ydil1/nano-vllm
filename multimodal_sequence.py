from typing import List, Optional, Dict, Any, Tuple
import torch
from dataclasses import dataclass

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence


IMAGE_TOKEN_ID = -200 
VIDEO_TOKEN_ID = -201 
AUDIO_TOKEN_ID = -202 
DEFAULT_IMAGE_PATCHES = 576  # CLIP的patch数量
#TODO:我还需要找到这里patch的计算方法，比如视频和音频的patch数量
DEFAULT_VIDEO_PATCHES = 0
DEFAULT_AUDIO_PATCHES = 0

@dataclass(frozen=True)
class PlaceholderRange:
    """
    占位符范围定义（基于 vLLM 的设计）

    Example:
        文本: "看这张 <image> 图片"
        token_ids: [101, 102, IMAGE_TOKEN_ID, 103]

        占位符将记录:
        - offset: 2 (IMAGE_TOKEN_ID 的位置)
        - length: 576 (展开后占576个位置)
    """
    offset: int      
    length: int    
    modality: str 

    @property
    def end_offset(self) -> int:
        return self.offset + self.length

class MultimodalSequence(Sequence):
    def __init__(self,
                 token_ids: List[int],
                 images: Optional[List[Any]] = None,
                 sampling_params: SamplingParams = SamplingParams()):
        super().__init__(token_ids, sampling_params)

        self.images = images or []
        #TODO:我需要找到这里video和audio的占位符的计算方法
        self.placeholders: Dict[str, List[PlaceholderRange]] = {
            "image": []
        }

        self._init_placeholders()

        self._validate()

    def _init_placeholders(self):
        for idx, token_id in enumerate(self.token_ids):
            if token_id == IMAGE_TOKEN_ID:
                placeholder = PlaceholderRange(
                    offset=idx,
                    length=DEFAULT_IMAGE_PATCHES, 
                    modality="image"
                )
                self.placeholders["image"].append(placeholder)
            #TODO:后续可以添加video和audio的创建方法

    def _validate(self):
        num_image_placeholders = len(self.placeholders["image"])
        num_images = len(self.images)

        if num_image_placeholders != num_images:
            raise ValueError(
                f"({num_images}) and ({num_image_placeholders}) do not match"
            )

    @property
    def total_length(self) -> int:
        total = len(self.token_ids)
        for modality_placeholders in self.placeholders.values():
            for placeholder in modality_placeholders:
                total += placeholder.length - 1

        return total

    def get_expanded_token_ids(self) -> List[int]:
        expanded = []
        placeholder_map = {p.offset: p for p in self.placeholders["image"]}
        placeholder_idx = 0
        for idx, token_id in enumerate(self.token_ids):
            if idx in placeholder_map:
                placeholder = placeholder_map[placeholder_idx]
                placeholder_idx += 1
                expanded.extend([token_id] * placeholder.length)
            else:
                expanded.append(token_id)

        return expanded

    def get_position_ids(self, device) -> torch.Tensor:
        positions = []
        current_pos = 0
        placeholder_map = {p.offset: p for p in self.placeholders["image"]}

        placeholder_idx = 0
        for idx, token_id in enumerate(self.token_ids):
            if token_id in placeholder_map:
                placeholder = placeholder_map[placeholder_idx]
                placeholder_idx += 1
                positions.extend(range(current_pos, current_pos + placeholder.length))
                current_pos += placeholder.length
            else:
                positions.append(current_pos)
                current_pos += 1

        return torch.tensor(positions, dtype=torch.long, device=device)

    def merge_multimodal_embeddings(self,
                                   text_embeddings: torch.Tensor,
                                   image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embeddings: 文本嵌入 [num_text_tokens, hidden_dim]
            image_features: 图像特征 [num_images, num_patches, hidden_dim]

        Returns:
            合并后的嵌入 [total_length, hidden_dim]
        """
        device = text_embeddings.device
        dtype = text_embeddings.dtype
        hidden_dim = text_embeddings.shape[-1]

        merged = torch.zeros(self.total_length, hidden_dim, device=device, dtype=dtype)

        merged_idx = 0
        text_idx = 0
        image_idx = 0

        placeholder_map = {p.offset: p for p in self.placeholders["image"]}
        placeholder_idx = 0
        for idx, token_id in enumerate(self.token_ids):
            if token_id in placeholder_map:
                placeholder = placeholder_map[placeholder_idx]
                placeholder_idx += 1
                # 获取对应图像的特征
                if len(image_features.shape) == 3:
                    # [num_images, num_patches, hidden_dim]
                    current_image_features = image_features[image_idx]  # [num_patches, hidden_dim]
                else:
                    start = image_idx * placeholder.length
                    end = start + placeholder.length
                    current_image_features = image_features[start:end]

                # 这里是处理图像的方法,我想的是嵌入应该是plceholder.length个位置
                merged[merged_idx:merged_idx + placeholder.length] = current_image_features
                merged_idx += placeholder.length
                image_idx += 1
            else:
                #这里是处理文本的方法,我想的是可以直接嵌入,文本只有一个占位符
                merged[merged_idx] = text_embeddings[text_idx]
                merged_idx += 1
                text_idx += 1

        return merged

    def get_placeholder_positions(self, modality: str = "image") -> List[Tuple[int, int]]:
        positions = []
        current_offset = 0
        placeholder_idx = 0
        placeholder_map = {p.offset: p for p in self.placeholders[modality]}

        for idx,token_id in enumerate(self.token_ids):
            if token_id in placeholder_map:
                placeholder = placeholder_map[placeholder_idx]
                placeholder_idx += 1
                positions.append((current_offset, current_offset + placeholder.length))
                current_offset += placeholder.length
            else:
                positions.append((current_offset, current_offset))
                current_offset += 1

        return positions