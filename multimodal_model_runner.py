import pickle
import torch
import torch.distributed as dist
import os
import json
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from typing import List, Optional, Dict, Any, Tuple
from nanovllm.config import Config
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from PIL import Image
from nanovllm.models.llava import LlavaForConditionalGeneration
from nanovllm.models.clip_image_processing import CLIPImageProcessor
from transformers import AutoTokenizer, AutoConfig
from transformers import CLIPImageProcessor as HFCLIPImageProcessor
from multimodal_sequence import MultimodalSequence, IMAGE_TOKEN_ID, DEFAULT_IMAGE_PATCHES
from multimodal_block_manager import MultimodalBlockManager

class MultimodalModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        self.model = self._load_multimodal_model(config)
        self.image_processor = self._init_image_processor(config)
        self.tokenizer = self._init_tokenizer(config)

        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()

        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def _load_multimodal_model(self, config):
        model_path = config.model or "liuhaotian/llava-v1.5-7b" #hardcode 一下,之前报了个错

        if hasattr(config, 'model_type') and config.model_type == 'llava':
            from nanovllm.models.llava import LlavaForConditionalGeneration
            from transformers import LlavaConfig

            llava_config = LlavaConfig.from_pretrained(model_path)
            model = LlavaForConditionalGeneration(llava_config)

            if os.path.exists(model_path):
                state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
        else:
            model = load_model(config)

        return model

    def _init_tokenizer(self, config):
        model_path = config.model or "liuhaotian/llava-v1.5-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})
        return tokenizer

    def _init_image_processor(self, config):
        model_path = config.model or "liuhaotian/llava-v1.5-7b"
        if hasattr(config, 'model_type') and config.model_type == 'llava':
            from nanovllm.models.clip_image_processing import CLIPImageProcessor
            return CLIPImageProcessor(
                size=336,
                crop_size=336,
                do_normalize=True,
                do_resize=True,
                do_center_crop=False,
                do_convert_rgb=True,
            )
        else:
            try:
                return HFCLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
            except:
                return None

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        warmup_seqs = []
        for i in range(num_seqs):
            if i % 2 == 0: 
                token_ids = [0] * (max_model_len // 2) + [IMAGE_TOKEN_ID] + [0] * (max_model_len // 2 - 1)
                dummy_image = torch.randn(3, 336, 336)
                seq = MultimodalSequence(token_ids, images=[dummy_image])
            else:
                seq = MultimodalSequence([0] * max_model_len)
            warmup_seqs.append(seq)

        self.run(warmup_seqs, is_prefill=True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size *
                      num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize)

        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        self.kv_cache = torch.zeros(
            2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
            self.block_size, num_kv_heads, hf_config.head_dim
        )

        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: List[MultimodalSequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_multimodal_inputs(self, seqs: List[MultimodalSequence]) -> Dict[str, Any]:
        multimodal_inputs = {
            'image_features': [],
            'image_positions': [],
            'has_images': [],
            'pixel_values': None 
        }

        all_pixel_values = []
        for seq in seqs:
            if seq.has_images:
                processed_images = []
                for image in seq.images:
                    if self.image_processor is not None:
                        if isinstance(image, torch.Tensor):
                            if image.dim() == 3 and image.shape[0] == 3:
                                image = image.permute(1, 2, 0)
                            image_pil = Image.fromarray((image.cpu().numpy() * 255).astype('uint8'))
                        elif isinstance(image, Image.Image):
                            image_pil = image
                        else:
                            image_pil = image

                        # Process the image
                        processed = self.image_processor(image_pil, return_tensors="pt")
                        pixel_values = processed["pixel_values"]
                        all_pixel_values.append(pixel_values)
                        processed_images.append(pixel_values)

                multimodal_inputs['image_features'].append(processed_images)
                multimodal_inputs['image_positions'].append(seq.get_image_positions)
                multimodal_inputs['has_images'].append(True)
            else:
                multimodal_inputs['has_images'].append(False)

        if all_pixel_values:
            multimodal_inputs['pixel_values'] = torch.cat(all_pixel_values, dim=0)

        return multimodal_inputs

    def prepare_prefill(self, seqs: List[MultimodalSequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        for seq in seqs:
            seqlen = seq.total_length  
            expanded_ids = seq.get_expanded_token_ids()
            input_ids.extend(expanded_ids[seq.num_cached_tokens:])
            position_ids = seq.get_position_ids(device='cpu')
            positions.extend(position_ids[seq.num_cached_tokens:].tolist())

            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                   slot_mapping, None, block_tables)

        multimodal_inputs = self.prepare_multimodal_inputs(seqs)

        return input_ids, positions, multimodal_inputs

    def prepare_decode(self, seqs: List[MultimodalSequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            total_len = seq.total_length
            positions.append(total_len)
            context_lens.append(total_len)
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        multimodal_inputs = None

        return input_ids, positions, multimodal_inputs

    def prepare_sample(self, seqs: List[MultimodalSequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor,
                  multimodal_inputs: Optional[Dict] = None, is_prefill: bool = False):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            model_kwargs = {
                'input_ids': input_ids,
                'positions': positions
            }

            if multimodal_inputs is not None and any(multimodal_inputs['has_images']):
                if multimodal_inputs['pixel_values'] is not None:
                    model_kwargs['pixel_values'] = multimodal_inputs['pixel_values']
            if hasattr(self.model, 'vision_tower'):
                hidden_states = self.model(**model_kwargs)
                return self.model.compute_logits(hidden_states)
            else:
                # Standard model
                outputs = self.model(**model_kwargs)
                if hasattr(outputs, 'logits'):
                    return outputs.logits
                else:
                    return self.model.compute_logits(outputs)
        else:
            return self._run_with_cudagraph(input_ids, positions)

    def run(self, seqs: List[MultimodalSequence], is_prefill: bool) -> List[int]:
        if is_prefill:
            input_ids, positions, multimodal_inputs = self.prepare_prefill(seqs)
        else:
            input_ids, positions, multimodal_inputs = self.prepare_decode(seqs)

        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, multimodal_inputs, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        reset_context()
        return token_ids

    def merge_multimodal_embeddings(self, seqs: List[MultimodalSequence],
                                   text_embeddings: torch.Tensor,
                                   image_features_list: List[torch.Tensor]) -> torch.Tensor:
        merged_embeddings = []
        text_offset = 0

        for seq_idx, seq in enumerate(seqs):
            if seq.has_images:
                num_text_tokens = len(seq.token_ids) - len(seq.images)
                seq_text_embeds = text_embeddings[text_offset:text_offset + num_text_tokens]
                seq_image_features = image_features_list[seq_idx]
                merged = seq.merge_multimodal_embeddings(seq_text_embeds, seq_image_features)
                merged_embeddings.append(merged)

                text_offset += num_text_tokens
            else:
                num_tokens = len(seq.token_ids)
                seq_embeds = text_embeddings[text_offset:text_offset + num_tokens]
                merged_embeddings.append(seq_embeds)
                text_offset += num_tokens

        return torch.cat(merged_embeddings, dim=0)

    @torch.inference_mode()
    def capture_cudagraph(self):
        # TODO: 实现CUDA graph
        pass

    def _run_with_cudagraph(self, input_ids: torch.Tensor, positions: torch.Tensor):
        # TODO: 缺少实现
        pass

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)