import torch
import threading
import time
import os
import json
import sys
import numpy as np
from PIL import Image
from queue import Queue, Empty, PriorityQueue
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch.multiprocessing as mp
from nanovllm.models.clip import CLIPVisionModel
from nanovllm.models.llama import LlamaForCausalLM
from nanovllm.models.llava import LlavaMultiModalProjector
from transformers import CLIPVisionConfig, LlamaConfig, AutoTokenizer
from nanovllm.models.clip_image_processing import CLIPImageProcessor
from multimodal_sequence import MultimodalSequence, IMAGE_TOKEN_ID, DEFAULT_IMAGE_PATCHES
from multimodal_block_manager import MultimodalBlockManager


@dataclass
class MultimodalTask:
    task_id: str
    sequence: MultimodalSequence
    priority: float
    images: Optional[List[Any]] = None
    prompt: Optional[str] = None
    image_features: Optional[torch.Tensor] = None
    timestamp: float = 0.0


class VisionEncoder:
    def __init__(self, model_path: str, device: str = "cuda:0",
                 batch_size: int = 4, verbose: bool = False):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose

        self._init_vision_model()

    def _init_vision_model(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        config_path = os.path.join(self.model_path, "config.json") if os.path.isdir(self.model_path) else None
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                vision_config_dict = config.get("vision_config", {})
        else:
            vision_config_dict = {
                "hidden_size": 1024,
                "image_size": 336,
                "intermediate_size": 4096,
                "num_attention_heads": 16,
                "num_hidden_layers": 24,
                "patch_size": 14,
            }

        vision_config = CLIPVisionConfig(
            hidden_size=vision_config_dict.get("hidden_size", 1024),
            image_size=vision_config_dict.get("image_size", 336),
            intermediate_size=vision_config_dict.get("intermediate_size", 4096),
            num_attention_heads=vision_config_dict.get("num_attention_heads", 16),
            num_hidden_layers=vision_config_dict.get("num_hidden_layers", 24),
            patch_size=vision_config_dict.get("patch_size", 14),
            num_channels=3,
        )

        self.vision_model = CLIPVisionModel(vision_config)
        self.vision_model.to(self.device, dtype=torch.float16)
        self.vision_model.eval()

        self.image_processor = CLIPImageProcessor(
            size=336,
            crop_size=336,
            do_normalize=True,
            do_resize=True,
        )

        torch.cuda.empty_cache()

        if self.verbose:
            print(f"[VisionEncoder] Vision model initialized")

    def process_batch(self, images: List[Any]) -> torch.Tensor:

        # Process images
        processed_images = []
        for img in images:
            if isinstance(img, torch.Tensor):
                # Convert tensor to PIL Image
                if img.dim() == 3 and img.shape[0] == 3:
                    img = img.permute(1, 2, 0)
                img_pil = Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8))
            elif isinstance(img, Image.Image):
                img_pil = img
            else:
                img_pil = img

            processed_images.append(img_pil)

        # Process through CLIP
        pixel_values = self.image_processor(processed_images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device, dtype=torch.float16)

        with torch.no_grad():
            # Get vision features
            image_features = self.vision_model(pixel_values)

            # For LLaVA, typically use features without CLS token
            if len(image_features.shape) == 3:
                # Remove CLS token (first token)
                image_features = image_features[:, 1:, :]

        return image_features.cpu()


class LLMDecoder:
    def __init__(self, model_path: str, device: str = "cuda:1",
                 batch_size: int = 4, verbose: bool = False):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose

        self._init_language_model()

    def _init_language_model(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        config_path = os.path.join(self.model_path, "config.json") if os.path.isdir(self.model_path) else None
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                full_config = json.load(f)
                text_config_dict = full_config.get("text_config", {})
                vision_config_dict = full_config.get("vision_config", {})
        else:
            # Default Llama config
            text_config_dict = {
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "num_key_value_heads": 32,
                "vocab_size": 32001,  # LLaVA extends vocabulary
                "rms_norm_eps": 1e-6,
            }
            vision_config_dict = {
                "hidden_size": 1024,
            }
        text_config = LlamaConfig(**text_config_dict)
        self.language_model = LlamaForCausalLM(text_config)
        self.language_model.to(self.device, dtype=torch.float16)
        self.language_model.eval()

        self.mm_projector = LlavaMultiModalProjector(
            vision_hidden_size=vision_config_dict.get("hidden_size", 1024),
            text_hidden_size=text_config_dict.get("hidden_size", 4096),
            projector_hidden_act="gelu",
        )


        self.mm_projector.to(self.device, dtype=torch.float16)
        self.mm_projector.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        self.image_token_index = IMAGE_TOKEN_ID
        torch.cuda.empty_cache()
        if self.verbose:
            print(f"[LLMDecoder] Language model initialized")

    def generate(self, prompt: str, image_features: torch.Tensor,max_new_tokens: int = 128, temperature: float = 0.7) -> str:
        
        image_features = image_features.to(self.device, dtype=torch.float16)
        image_features = self.mm_projector(image_features)
        prompt = prompt.replace("<image>", str(self.image_token_index))

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        image_token_mask = input_ids == self.image_token_index

        if image_token_mask.any():
            batch_size, seq_len = input_ids.shape
            new_seq_len = seq_len - 1 + image_features.shape[1] 

            new_embeds = torch.zeros(
                batch_size, new_seq_len, inputs_embeds.shape[-1],
                device=self.device, dtype=inputs_embeds.dtype
            )

            image_pos = image_token_mask.nonzero(as_tuple=True)[1][0].item()
            new_embeds[:, :image_pos] = inputs_embeds[:, :image_pos]
            new_embeds[:, image_pos:image_pos + image_features.shape[1]] = image_features

            if image_pos + 1 < seq_len:
                new_embeds[:, image_pos + image_features.shape[1]:] = inputs_embeds[:, image_pos + 1:]

            inputs_embeds = new_embeds

        with torch.no_grad():
            positions = torch.arange(inputs_embeds.shape[1], device=self.device)
            hidden_states = self.language_model.model(
                input_ids=None,
                positions=positions,
                inputs_embeds=inputs_embeds
            )

            logits = self.language_model.compute_logits(hidden_states)
            generated_tokens = []
            for _ in range(max_new_tokens):
                if temperature > 0:
                    probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                generated_tokens.append(next_token.item())

                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                next_embeds = self.language_model.model.embed_tokens(next_token)
                inputs_embeds = torch.cat([inputs_embeds, next_embeds], dim=1)
                positions = torch.arange(inputs_embeds.shape[1], device=self.device)

                hidden_states = self.language_model.model(
                    input_ids=None,
                    positions=positions,
                    inputs_embeds=inputs_embeds
                )
                logits = self.language_model.compute_logits(hidden_states)

        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response


class MultimodalScheduler:
    def __init__(self, model_path: str,
                 vision_device: str = "cuda:0",
                 language_device: str = "cuda:1",
                 batch_size: int = 4,
                 pipeline_depth: int = 3,
                 verbose: bool = False):
        self.model_path = model_path
        self.vision_device = vision_device
        self.language_device = language_device
        self.batch_size = batch_size
        self.pipeline_depth = pipeline_depth
        self.verbose = verbose
        self.vision_queue = PriorityQueue(maxsize=100)
        self.language_queue = PriorityQueue(maxsize=100)
        self.result_dict = {}

        self.running = True
        self.vision_thread = None
        self.language_thread = None
        self._init_processors()
        self._start_threads()

    def _init_processors(self):
        self.vision_encoder = VisionEncoder(
            self.model_path,
            device=self.vision_device,
            batch_size=self.batch_size,
            verbose=self.verbose
        )

        self.llm_decoder = LLMDecoder(
            self.model_path,
            device=self.language_device,
            batch_size=self.batch_size,
            verbose=self.verbose
        )

    def _start_threads(self):
        self.vision_thread = threading.Thread(target=self._vision_worker, daemon=True)
        self.language_thread = threading.Thread(target=self._language_worker, daemon=True)

        self.vision_thread.start()
        self.language_thread.start()

    def _vision_worker(self):
        while self.running:
            batch_tasks = self._collect_vision_batch()
            if not batch_tasks:
                time.sleep(0.01)
                continue

            all_images = []
            for task in batch_tasks:
                if task.images:
                    all_images.extend(task.images)

            if all_images:
                image_features = self.vision_encoder.process_batch(all_images)
                feature_idx = 0
                for task in batch_tasks:
                    if task.images:
                        num_images = len(task.images)
                        task.image_features = image_features[feature_idx:feature_idx + num_images]
                        feature_idx += num_images

                    self.language_queue.put((task.priority, time.time(), task))

    def _language_worker(self):
        while self.running:
            try:
                _, _, task = self.language_queue.get(timeout=1.0)

                if task.image_features is not None and task.prompt:
                    response = self.llm_decoder.generate(
                        task.prompt,
                        task.image_features,
                        max_new_tokens=128
                    )

                    self.result_dict[task.task_id] = {
                        'response': response,
                        'success': True
                    }
                else:
                    self.result_dict[task.task_id] = {
                        'response': 'Processing failed',
                        'success': False
                    }

            except Empty:
                continue
            except Exception as e:
                if self.verbose:
                    print(f"[LanguageWorker] Error: {e}")
                    import traceback
                    traceback.print_exc()
                time.sleep(0.1)

    def _collect_vision_batch(self):
        batch = []

        try:
            priority, timestamp, task = self.vision_queue.get(timeout=0.1)
            batch.append(task)
        except Empty:
            return batch

        start_time = time.time()
        while len(batch) < self.batch_size and time.time() - start_time < 0.05:
            try:
                priority, timestamp, task = self.vision_queue.get(timeout=0.01)
                batch.append(task)
            except Empty:
                break

        return batch

    def submit_task(self, task_id: str, sequence: MultimodalSequence, prompt: str, priority: float = None) -> str:
        if priority is None:
            priority = time.time()

        task = MultimodalTask(
            task_id=task_id,
            sequence=sequence,
            priority=priority,
            images=sequence.images if sequence.has_images else None,
            prompt=prompt,
            timestamp=time.time()
        )

        if task.images:
            self.vision_queue.put((priority, time.time(), task))
        else:
            task.image_features = None
            self.language_queue.put((priority, time.time(), task))

        return task_id

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[Dict]:
        start_time = time.time()
        while True:
            if task_id in self.result_dict:
                return self.result_dict.pop(task_id)

            if timeout is not None and time.time() - start_time >= timeout:
                break

            time.sleep(0.05)

        return None

    def shutdown(self):
        self.running = False

        if self.vision_thread and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=5)

        if self.language_thread and self.language_thread.is_alive():
            self.language_thread.join(timeout=5)

        torch.cuda.empty_cache()
