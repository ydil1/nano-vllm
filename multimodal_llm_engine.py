import time
import uuid
import threading
import torch
import torch.multiprocessing as mp
from queue import Queue, Empty, Full
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import json
import os

from multimodal_sequence import MultimodalSequence, IMAGE_TOKEN_ID
from multimodal_block_manager import MultimodalBlockManager
from multimodal_scheduler import MultimodalScheduler
from multimodal_model_runner import MultimodalModelRunner


@dataclass
class MultimodalRequest:
    """Request for multimodal processing"""
    request_id: str
    prompt: str
    images: Optional[List[Union[Image.Image, torch.Tensor, str]]] = None
    sampling_params: Optional[Dict[str, Any]] = None
    priority: float = 0.0
    arrival_time: float = 0.0


@dataclass
class MultimodalResponse:
    """Response from multimodal processing"""
    request_id: str
    text: str
    metadata: Dict[str, Any] = None


class RequestMetrics:
    """Track request processing metrics"""

    def __init__(self):
        self.arrival_time: Optional[float] = None
        self.vision_start_time: Optional[float] = None
        self.vision_end_time: Optional[float] = None
        self.llm_start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.completion_time: Optional[float] = None
        self.token_times: List[float] = []

    def calculate_ttft(self) -> Optional[float]:
        """Time to first token"""
        if self.arrival_time and self.first_token_time:
            return self.first_token_time - self.arrival_time
        return None

    def calculate_e2e_latency(self) -> Optional[float]:
        """End-to-end latency"""
        if self.arrival_time and self.completion_time:
            return self.completion_time - self.arrival_time
        return None

    def calculate_vision_time(self) -> Optional[float]:
        """Vision processing time"""
        if self.vision_start_time and self.vision_end_time:
            return self.vision_end_time - self.vision_start_time
        return None

    def calculate_llm_time(self) -> Optional[float]:
        """LLM processing time"""
        if self.llm_start_time and self.completion_time:
            return self.completion_time - self.llm_start_time
        return None


class PipelineManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vision_queue = Queue(maxsize=config.get('vision_queue_size', 100))
        self.llm_queue = Queue(maxsize=config.get('llm_queue_size', 100))
        self.vision_buffer = {}  # Store processed vision features
        self.request_tracker = {}  # Track request status
        self.metrics = {}  # Request metrics

        self.running = True
        self.monitor_thread = None

    def _start_monitor(self):
        self.monitor_thread = threading.Thread(target=self._monitor_pipeline, daemon=True)
        self.monitor_thread.start()

    def _monitor_pipeline(self):
        while self.running:
            time.sleep(5.0)  # Report every 5 seconds

            vision_pending = self.vision_queue.qsize()
            llm_pending = self.llm_queue.qsize()
            buffer_size = len(self.vision_buffer)

            print(f"[Pipeline Monitor] Vision Queue: {vision_pending}, "
                  f"LLM Queue: {llm_pending}, Buffer: {buffer_size}")

            # Check for bottlenecks
            if vision_pending > 50:
                print("[Pipeline Monitor] WARNING: Vision queue backlog detected")
            if llm_pending > 50:
                print("[Pipeline Monitor] WARNING: LLM queue backlog detected")

    def should_send_to_llm(self, request_id: str) -> bool:
        # Check if vision features are ready
        if request_id not in self.vision_buffer:
            return False

        # Check LLM queue capacity
        if self.llm_queue.full():
            return False

        # Can add more sophisticated scheduling logic here
        return True

    def update_metrics(self, request_id: str, stage: str, timestamp: float = None):
        """Update request metrics"""
        if timestamp is None:
            timestamp = time.time()

        if request_id not in self.metrics:
            self.metrics[request_id] = RequestMetrics()

        metrics = self.metrics[request_id]

        if stage == 'arrival':
            metrics.arrival_time = timestamp
        elif stage == 'vision_start':
            metrics.vision_start_time = timestamp
        elif stage == 'vision_end':
            metrics.vision_end_time = timestamp
        elif stage == 'llm_start':
            metrics.llm_start_time = timestamp
        elif stage == 'first_token':
            metrics.first_token_time = timestamp
        elif stage == 'completion':
            metrics.completion_time = timestamp

    def get_metrics_summary(self, request_id: str) -> Dict[str, Any]:
        """Get metrics summary for a request"""
        if request_id not in self.metrics:
            return {}

        metrics = self.metrics[request_id]
        return {
            'ttft': metrics.calculate_ttft(),
            'e2e_latency': metrics.calculate_e2e_latency(),
            'vision_time': metrics.calculate_vision_time(),
            'llm_time': metrics.calculate_llm_time(),
        }

    def shutdown(self):
        """Shutdown pipeline manager"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)


class MultimodalLLMEngine:
    """
    Main engine for multimodal LLM inference
    Coordinates between vision encoder, LLM decoder, and manages the pipeline
    """

    def __init__(self, model_path: str,
                 vision_device: str = "cuda:0",
                 language_device: str = "cuda:1",
                 num_kv_blocks: int = 1024,
                 block_size: int = 16,
                 batch_size: int = 4,
                 pipeline_depth: int = 3,
                 enable_pipeline_parallel: bool = True,
                 verbose: bool = False):
        """
        Initialize the multimodal LLM engine

        Args:
            model_path: Path to the model (e.g., LLaVA)
            vision_device: Device for vision encoder
            language_device: Device for language decoder
            num_kv_blocks: Number of KV cache blocks
            block_size: Size of each block
            batch_size: Batch size for processing
            pipeline_depth: Depth of pipeline
            enable_pipeline_parallel: Enable pipeline parallelism
            verbose: Verbose output
        """
        self.model_path = model_path
        self.vision_device = vision_device
        self.language_device = language_device
        self.batch_size = batch_size
        self.pipeline_depth = pipeline_depth
        self.enable_pipeline_parallel = enable_pipeline_parallel
        self.verbose = verbose

        # Initialize block manager
        self.block_manager = MultimodalBlockManager(
            num_blocks=num_kv_blocks,
            block_size=block_size
        )

        # Initialize scheduler
        self.scheduler = MultimodalScheduler(
            model_path=model_path,
            vision_device=vision_device,
            language_device=language_device,
            batch_size=batch_size,
            pipeline_depth=pipeline_depth,
            verbose=verbose
        )

        # Initialize pipeline manager if enabled
        if enable_pipeline_parallel:
            pipeline_config = {
                'vision_queue_size': 100,
                'llm_queue_size': 100,
                'verbose': verbose
            }
            self.pipeline_manager = PipelineManager(pipeline_config)
        else:
            self.pipeline_manager = None

        # Request tracking
        self.pending_requests = {}
        self.completed_requests = {}
        self.request_lock = threading.Lock()

        # Processing threads
        self.running = True
        self.process_thread = threading.Thread(target=self._process_requests, daemon=True)
        self.process_thread.start()


    def _process_requests(self):
        """Process pending requests"""
        while self.running:
            try:
                # Check for completed tasks
                with self.request_lock:
                    for request_id in list(self.pending_requests.keys()):
                        result = self.scheduler.get_result(request_id, timeout=0)
                        if result:
                            request = self.pending_requests.pop(request_id)

                            # Update metrics
                            if self.pipeline_manager:
                                self.pipeline_manager.update_metrics(request_id, 'completion')

                            # Create response
                            response = MultimodalResponse(
                                request_id=request_id,
                                text=result.get('response', ''),
                                metadata={
                                    'success': result.get('success', False),
                                    'metrics': self.pipeline_manager.get_metrics_summary(request_id)
                                    if self.pipeline_manager else {}
                                }
                            )

                            self.completed_requests[request_id] = response


                time.sleep(0.01)

            except Exception as e:
                time.sleep(0.1)

    def add_request(self, prompt: str,
                    images: Optional[List[Union[Image.Image, torch.Tensor, str]]] = None,
                    sampling_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new request to the engine

        Args:
            prompt: Text prompt with optional <image> placeholders
            images: List of images (PIL, tensor, or paths)
            sampling_params: Sampling parameters

        Returns:
            request_id: Unique request ID
        """
        request_id = str(uuid.uuid4())
        arrival_time = time.time()

        # Create request
        request = MultimodalRequest(
            request_id=request_id,
            prompt=prompt,
            images=images,
            sampling_params=sampling_params or {},
            priority=arrival_time,  # Use arrival time as priority (FIFO)
            arrival_time=arrival_time
        )

        # Update metrics
        if self.pipeline_manager:
            self.pipeline_manager.update_metrics(request_id, 'arrival', arrival_time)

        # Create multimodal sequence
        token_ids = self._tokenize_prompt(prompt)
        sequence = MultimodalSequence(
            token_ids=token_ids,
            images=images if images else None,
            sampling_params=None,  # Will use defaults
            seq_id=request_id
        )

        # Check if we can allocate blocks
        if not self.block_manager.can_allocate(sequence):
            return None

        # Allocate blocks
        self.block_manager.allocate(sequence)

        # Track request
        with self.request_lock:
            self.pending_requests[request_id] = request

        # Submit to scheduler
        self.scheduler.submit_task(
            task_id=request_id,
            sequence=sequence,
            prompt=prompt,
            priority=arrival_time
        )


        return request_id

    def _tokenize_prompt(self, prompt: str) -> List[int]:
        """Tokenize prompt (simplified version)"""
        # In a real implementation, use the actual tokenizer
        # For now, create dummy token IDs
        tokens = []
        words = prompt.split()

        for word in words:
            if word == "<image>":
                tokens.append(IMAGE_TOKEN_ID)
            else:
                # Dummy tokenization
                tokens.extend([hash(word) % 32000 for _ in range(len(word) // 3 + 1)])

        return tokens

    def get_response(self, request_id: str, timeout: Optional[float] = None) -> Optional[MultimodalResponse]:
        """
        Get response for a request

        Args:
            request_id: Request ID
            timeout: Timeout in seconds

        Returns:
            Response or None if not ready/timeout
        """
        start_time = time.time()

        while timeout is None or time.time() - start_time < timeout:
            with self.request_lock:
                if request_id in self.completed_requests:
                    response = self.completed_requests.pop(request_id)

                    # Deallocate blocks
                    # Note: In real implementation, need to track sequence properly
                    # self.block_manager.deallocate(sequence)

                    return response

            time.sleep(0.05)

        return None

    def batch_add_requests(self, requests: List[Tuple[str, Optional[List]]]) -> List[str]:
        """
        Add multiple requests in batch

        Args:
            requests: List of (prompt, images) tuples

        Returns:
            List of request IDs
        """
        request_ids = []

        for prompt, images in requests:
            request_id = self.add_request(prompt, images)
            if request_id:
                request_ids.append(request_id)

        return request_ids

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        with self.request_lock:
            num_pending = len(self.pending_requests)
            num_completed = len(self.completed_requests)

        block_stats = self.block_manager.get_stats()

        stats = {
            'pending_requests': num_pending,
            'completed_requests': num_completed,
            'block_manager': block_stats,
        }

        if self.pipeline_manager:
            stats['pipeline'] = {
                'vision_queue': self.pipeline_manager.vision_queue.qsize(),
                'llm_queue': self.pipeline_manager.llm_queue.qsize(),
                'vision_buffer': len(self.pipeline_manager.vision_buffer),
            }

        return stats

    def shutdown(self):
        """Shutdown the engine"""
        self.running = False

        # Wait for processing thread
        if self.process_thread.is_alive():
            self.process_thread.join(timeout=5)

        # Shutdown scheduler
        self.scheduler.shutdown()

        # Shutdown pipeline manager
        if self.pipeline_manager:
            self.pipeline_manager.shutdown()

        torch.cuda.empty_cache()


def create_engine(model_path: str,
                  vision_device: str = "cuda:0",
                  language_device: str = "cuda:1",
                  **kwargs) -> MultimodalLLMEngine:
    """
    Factory function to create multimodal LLM engine

    Args:
        model_path: Path to model
        vision_device: Device for vision
        language_device: Device for language
        **kwargs: Additional arguments

    Returns:
        MultimodalLLMEngine instance
    """
    return MultimodalLLMEngine(
        model_path=model_path,
        vision_device=vision_device,
        language_device=language_device,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Create engine
    engine = create_engine(
        model_path="liuhaotian/llava-v1.5-7b",
        vision_device="cuda:0",
        language_device="cuda:1",
        batch_size=4,
        verbose=True
    )

    # Add a request
    request_id = engine.add_request(
        prompt="<image>\nWhat do you see in this image?",
        images=[Image.new('RGB', (336, 336), color='red')]  # Dummy red image
    )

    print(f"Submitted request: {request_id}")

    # Wait for response
    response = engine.get_response(request_id, timeout=30)

    if response:
        print(f"Response: {response.text}")
        print(f"Metrics: {response.metadata}")
    else:
        print("Request timed out")

    # Get stats
    stats = engine.get_stats()
    print(f"Engine stats: {stats}")

    # Shutdown
    engine.shutdown()
