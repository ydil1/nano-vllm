from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from copy import copy
from itertools import count
from enum import Enum, auto
from nanovllm.sampling_params import SamplingParams
import torch

IMAGE_TOKEN_ID = -200 
VIDEO_TOKEN_ID = -201 
AUDIO_TOKEN_ID = -202 
#TODO:我还需要找到这里patch的计算方法，比如视频和音频的patch数量
IMAGE_TOKEN_ID = -200
DEFAULT_IMAGE_PATCHES = 576 

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

@dataclass(frozen=True)
class PlaceholderRange:
    offset: int
    length: int
    modality: str  

    @property
    def end_offset(self) -> int:
        return self.offset + self.length


class MultimodalSequence:
    block_size = 16  
    counter = count()  

    def __init__(self,
                 token_ids: List[int],
                 images: Optional[List[Any]] = None,
                 sampling_params: Optional[SamplingParams] = None,
                 seq_id: Optional[int] = None):
        self.seq_id = seq_id if seq_id is not None else next(MultimodalSequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1] if token_ids else None
        self.num_prompt_tokens = len(token_ids) 
        self.num_cached_tokens = 0
        self.block_table: List[int] = []
        self.sampling_params = SamplingParams()
        self.temperature = self.sampling_params.temperature
        self.max_tokens = self.sampling_params.max_tokens
        self.ignore_eos = self.sampling_params.ignore_eos
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
        
    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def total_length(self) -> int:
        total = len(self.token_ids)
        for placeholder in self.placeholders["image"]:
            total += placeholder.length - 1  
        return total

    @property
    def num_tokens(self) -> int:
        return self.total_length

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> List[int]:
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> List[int]:
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self) -> int:
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self) -> int:
        return (self.total_length + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        return self.total_length - (self.num_blocks - 1) * self.block_size


    def block(self, i: int) -> List[int]:
        assert 0 <= i < self.num_blocks
        expanded_tokens = self.get_expanded_token_ids()
        start = i * self.block_size
        end = min(start + self.block_size, len(expanded_tokens))
        return expanded_tokens[start:end]

    def get_expanded_token_ids(self) -> List[int]:
        expanded = []
        placeholder_map = {p.offset: p for p in self.placeholders["image"]}

        for idx, token_id in enumerate(self.token_ids):
            if idx in placeholder_map:
                expanded.extend([token_id] * DEFAULT_IMAGE_PATCHES)
            else:
                expanded.append(token_id)

        return expanded

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        if token_id == IMAGE_TOKEN_ID:
            placeholder = PlaceholderRange(
                offset=len(self.token_ids) - 1,
                length=DEFAULT_IMAGE_PATCHES,
                modality="image"
            )
            self.placeholders["image"].append(placeholder)


    def get_position_ids(self, device: str = 'cpu') -> torch.Tensor:
        positions = []
        current_pos = 0
        placeholder_map = {p.offset: p for p in self.placeholders["image"]}

        for idx, token_id in enumerate(self.token_ids):
            if idx in placeholder_map:
                placeholder = placeholder_map[idx]
                positions.extend(range(current_pos, current_pos + placeholder.length))
                current_pos += placeholder.length
            else:
                positions.append(current_pos)
                current_pos += 1

        return torch.tensor(positions, dtype=torch.long, device=device)

    def merge_multimodal_embeddings(self,
                                   text_embeddings,
                                   image_features):
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
        for idx, token_id in enumerate(self.token_ids):
            if idx in placeholder_map:
                placeholder = placeholder_map[idx]
                if len(image_features.shape) == 3:
                    # [num_images, num_patches, hidden_dim]
                    current_image_features = image_features[image_idx]  # [num_patches, hidden_dim]
                else:
                    start = image_idx * placeholder.length
                    end = start + placeholder.length
                    current_image_features = image_features[start:end]
                merged[merged_idx:merged_idx + placeholder.length] = current_image_features
                merged_idx += placeholder.length
                image_idx += 1
            else:
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
            if idx in placeholder_map:
                placeholder = placeholder_map[idx]
                positions.append((current_offset, current_offset + placeholder.length))
                current_offset += placeholder.length
            else:
                positions.append((current_offset, current_offset))
                current_offset += 1

        return positions

    @property
    def get_image_positions(self) -> List[Tuple[int, int]]:
        return self.get_placeholder_positions("image")
    @property
    def has_images(self) -> bool:
        return bool(self.images)
    @property
    def num_images(self) -> int:
        return len(self.images)
