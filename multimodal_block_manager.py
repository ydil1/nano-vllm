from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from collections import deque
import xxhash
import numpy as np

IMAGE_TOKEN_ID = -200
DEFAULT_IMAGE_PATCHES = 576  
BLOCK_SIZE = 16

class MultimodalBlock:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        self.is_multimodal = False

    def update(self, hash: int, token_ids: List[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
        self.is_multimodal = False


@dataclass
class Segment:
    type: str
    start_pos: int
    length: int
    blocks_needed: int
    token_ids: Optional[List[int]] = None


class MultimodalBlockManager:
    def __init__(self, num_blocks: int, block_size: Optional[int] = None):
        assert num_blocks > 0
        if block_size is None:
            block_size = 16
        self.block_size = block_size
        self.blocks: List[MultimodalBlock] = [MultimodalBlock(i) for i in range(num_blocks)]
        self.hash_to_block_id: Dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: Set[int] = set()
        self.multimodal_segments: Dict[int, List[Segment]] = {}
        self.image_block_ids: Set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: List[int], prefix: int = -1) -> int:
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> MultimodalBlock:
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} is already in use"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        if block_id in self.image_block_ids:
            self.image_block_ids.remove(block_id)

    def analyze(self, seq) -> List[Segment]:
        segments = []
        current_pos = 0
        text_start = None
        text_tokens = []

        for i, token_id in enumerate(seq.token_ids):
            if token_id == IMAGE_TOKEN_ID:
                if text_tokens:
                    blocks_needed = (len(text_tokens) + self.block_size - 1) // self.block_size
                    segments.append(Segment(
                        type='text',
                        start_pos=text_start,
                        length=len(text_tokens),
                        blocks_needed=blocks_needed,
                        token_ids=text_tokens.copy()
                    ))
                    text_tokens = []
                    text_start = None
                segments.append(Segment(
                    type='image',
                    start_pos=current_pos,
                    length=DEFAULT_IMAGE_PATCHES,
                    blocks_needed=DEFAULT_IMAGE_PATCHES // self.block_size  # 36块
                ))
                current_pos += DEFAULT_IMAGE_PATCHES
            else:
                # 文本token
                if text_start is None:
                    text_start = current_pos
                text_tokens.append(token_id)
                current_pos += 1

        if text_tokens:
            blocks_needed = (len(text_tokens) + self.block_size - 1) // self.block_size
            segments.append(Segment(
                type='text',
                start_pos=text_start,
                length=len(text_tokens),
                blocks_needed=blocks_needed,
                token_ids=text_tokens.copy()
            ))

        return segments

    def can_allocate(self, seq) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq):
        assert not seq.block_table, f"Sequence {seq.seq_id} already has blocks"
        if hasattr(seq, 'images') and seq.images:
            segments = self.analyze(seq)
            self.multimodal_segments[seq.seq_id] = segments
        h = -1
        cache_miss = False

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            is_image_block = self._is_image_block(seq, i) if hasattr(seq, 'images') else False

            if is_image_block:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
                block.is_multimodal = True
                self.image_block_ids.add(block_id)
                h = -1  
            else:
                if len(token_ids) == self.block_size:
                    h = self.compute_hash(token_ids, h)
                    block_id = self.hash_to_block_id.get(h, -1)

                    if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                        cache_miss = True
                else:
                    h = -1
                    cache_miss = True

                if cache_miss:
                    block_id = self.free_block_ids[0]
                    block = self._allocate_block(block_id)
                else:
                    seq.num_cached_tokens += self.block_size
                    if block_id in self.used_block_ids:
                        block = self.blocks[block_id]
                        block.ref_count += 1
                    else:
                        block = self._allocate_block(block_id)

                if h != -1:
                    block.update(h, token_ids)
                    self.hash_to_block_id[h] = block_id

            seq.block_table.append(block_id)

    def _is_image_block(self, seq, block_idx: int) -> bool:
        if seq.seq_id not in self.multimodal_segments:
            return False

        segments = self.multimodal_segments[seq.seq_id]
        block_start = block_idx * self.block_size
        block_end = block_start + self.block_size
        for segment in segments:
            if segment.type == 'image':
                if block_start < segment.start_pos + segment.length and block_end > segment.start_pos:
                    return True
        return False


    def deallocate(self, seq):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        seq.num_cached_tokens = 0
        seq.block_table.clear()

        if seq.seq_id in self.multimodal_segments:
            del self.multimodal_segments[seq.seq_id]

    def can_append(self, seq) -> bool:
        if seq.seq_id in self.multimodal_segments:
            return False
        return len(self.free_block_ids) >= (len(seq.token_ids) % self.block_size == 1)

    def may_append(self, seq):
        if seq.seq_id in self.multimodal_segments:
            raise NotImplementedError("Multimodal sequences don't support appending")

        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        if len(seq.token_ids) % self.block_size == 1:
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        elif len(seq.token_ids) % self.block_size == 0:
            last_block_idx = len(block_table) - 1
            start = last_block_idx * self.block_size
            token_ids = seq.token_ids[start:start + self.block_size]
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id

        else:
            assert last_block.hash == -1, "Last block should be incomplete"

    def get_stats(self) -> Dict:
        return {
            'total_blocks': len(self.blocks),
            'free_blocks': len(self.free_block_ids),
            'used_blocks': len(self.used_block_ids),
            'image_blocks': len(self.image_block_ids),
            'cached_hashes': len(self.hash_to_block_id),
            'multimodal_sequences': len(self.multimodal_segments)
        }