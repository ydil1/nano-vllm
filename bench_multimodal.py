import os
import time
import torch
import numpy as np
from PIL import Image
from random import randint, seed, choice, shuffle
from typing import List, Tuple, Dict, Any
import json
import glob

from multimodal_llm_engine import create_engine, MultimodalLLMEngine
from multimodal_sequence import MultimodalSequence


LLAVA_SYSTEM_PROMPT = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

# Questions for LLaVA (will be wrapped in conversation format)
LLAVA_QUESTIONS = [
    "Describe the image in detail.",
    "What is shown in this image?",
    "Can you describe what you see in this image?",
    "Provide a detailed description of this image.",
    "What are the main elements in this image?",
    "Describe the scene depicted in this image.",
    "What is happening in this image?",
    "Please describe the contents of this image.",
    "What can you tell me about this image?",
    "Explain what you observe in this image.",
    "Describe the objects and their relationships in this image.",
    "What details can you identify in this image?",
    "Provide a comprehensive description of this image.",
    "What is the main subject of this image?",
    "Describe the visual elements present in this image.",
]

def format_llava_prompt(question: str) -> str:
    """Format a question into LLaVA v1 conversation format"""
    # LLaVA v1 uses TWO separator style with sep=" " and sep2="</s>"
    # Format: SYSTEM USER: <image>\nquestion ASSISTANT:
    return f"{LLAVA_SYSTEM_PROMPT} USER: <image>\n{question} ASSISTANT:"

# For backward compatibility, create the prompts list
SHAREGPT4V_PROMPTS = [format_llava_prompt(q) for q in LLAVA_QUESTIONS]


def load_coco_images(coco_dir: str, num_images: int = 100) -> List[str]:
    """Load image paths from COCO dataset directory"""
    image_pattern = os.path.join(coco_dir, "*.jpg")
    all_images = sorted(glob.glob(image_pattern))

    if not all_images:
        print(f"Warning: No images found in {coco_dir}")
        return []

    # Shuffle and select
    shuffle(all_images)
    selected = all_images[:min(num_images, len(all_images))]

    print(f"Found {len(all_images)} images in {coco_dir}, selected {len(selected)}")
    return selected


def generate_test_prompts_from_coco(
    coco_dir: str,
    num_prompts: int = 32
) -> List[Tuple[str, List[Image.Image], str]]:
    """Generate test prompts using real COCO images

    Returns:
        List of (prompt, images, image_path) tuples
    """
    image_paths = load_coco_images(coco_dir, num_prompts)

    if not image_paths:
        print("No COCO images found, falling back to dummy images")
        return generate_test_prompts_dummy(num_prompts)

    test_cases = []

    for i, image_path in enumerate(image_paths):
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')

            # Select a random prompt
            prompt = choice(SHAREGPT4V_PROMPTS)

            test_cases.append((prompt, [image], image_path))

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

    # If we need more prompts than available images, cycle through
    while len(test_cases) < num_prompts and image_paths:
        idx = len(test_cases) % len(image_paths)
        image_path = image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            prompt = choice(SHAREGPT4V_PROMPTS)
            test_cases.append((prompt, [image], image_path))
        except:
            break

    return test_cases


def generate_test_prompts_dummy(num_prompts: int) -> List[Tuple[str, List[Image.Image], str]]:
    """Generate test prompts with dummy images (fallback)"""
    test_cases = []

    for i in range(num_prompts):
        # Create random colored image
        rgb = np.random.randint(0, 256, (336, 336, 3), dtype=np.uint8)
        image = Image.fromarray(rgb)

        prompt = choice(SHAREGPT4V_PROMPTS)
        test_cases.append((prompt, [image], f"dummy_image_{i}"))

    return test_cases


def calculate_tokens_per_dollar(total_tokens: int, total_output_tokens: int,
                                 total_runtime_seconds: float,
                                 hardware_cost_per_hour: float = 0.646):
    """
    Calculate tokens per dollar based on hardware cost

    Args:
        total_tokens: Total tokens processed (input + output)
        total_output_tokens: Total output tokens generated
        total_runtime_seconds: Total runtime in seconds
        hardware_cost_per_hour: Hardware cost per hour (default: $0.646 for 2x A10G)

    Returns:
        Dict with cost analysis
    """
    total_runtime_hours = total_runtime_seconds / 3600
    total_cost = hardware_cost_per_hour * total_runtime_hours

    tokens_per_dollar = total_tokens / total_cost if total_cost > 0 else 0
    output_tokens_per_dollar = total_output_tokens / total_cost if total_cost > 0 else 0

    print(f"\n{'='*60}")
    print("Tokens Per Dollar Analysis")
    print("="*60)
    print(f"Total processed tokens:    {total_tokens:,}")
    print(f"Total output tokens:       {total_output_tokens:,}")
    print(f"Total runtime:             {total_runtime_hours:.4f} hours ({total_runtime_seconds:.2f}s)")
    print(f"Hardware cost/hour:        ${hardware_cost_per_hour:.3f}")
    print(f"Total cost:                ${total_cost:.4f}")
    print(f"Tokens per dollar:         {tokens_per_dollar:,.0f}")
    print(f"Output tokens per dollar:  {output_tokens_per_dollar:,.0f}")
    print("="*60)

    return {
        'tokens_per_dollar': tokens_per_dollar,
        'output_tokens_per_dollar': output_tokens_per_dollar,
        'total_cost': total_cost,
        'total_tokens': total_tokens,
        'total_output_tokens': total_output_tokens,
        'total_runtime_hours': total_runtime_hours,
        'hardware_cost_per_hour': hardware_cost_per_hour,
    }


def benchmark_throughput_with_output(
    engine: MultimodalLLMEngine,
    coco_dir: str,
    num_requests: int = 32,
    max_batch_size: int = 4,
    verbose: bool = True,
    print_outputs: bool = True
):
    """Benchmark throughput with multiple concurrent requests and print outputs"""

    print(f"\n{'='*60}")
    print(f"Throughput Benchmark: {num_requests} requests")
    print(f"Using COCO images from: {coco_dir}")
    print(f"{'='*60}")

    # Generate test cases from COCO
    test_cases = generate_test_prompts_from_coco(coco_dir, num_requests)

    if not test_cases:
        print("Error: No test cases could be generated")
        return None

    print(f"Generated {len(test_cases)} test cases")

    # Track metrics
    request_ids = []
    start_times = {}
    max_tokens_map = {}
    image_paths_map = {}
    prompts_map = {}
    input_tokens_map = {}  # Track input tokens

    # Submit all requests
    submit_start = time.time()

    for i, (prompt, images, image_path) in enumerate(test_cases):
        max_tokens = randint(50, 150)  # Variable output length
        request_id = engine.add_request(
            prompt=prompt,
            images=images,
            sampling_params={
                'temperature': 0.7,
                'max_tokens': max_tokens,
            }
        )

        if request_id:
            request_ids.append(request_id)
            start_times[request_id] = time.time()
            max_tokens_map[request_id] = max_tokens
            image_paths_map[request_id] = image_path
            prompts_map[request_id] = prompt
            # Estimate input tokens (prompt words + image patches ~576)
            input_tokens_map[request_id] = len(prompt.split()) + 576

            if verbose and (i + 1) % 10 == 0:
                print(f"  Submitted {i+1}/{len(test_cases)} requests...")

    submit_time = time.time() - submit_start
    print(f"\nSubmission completed: {len(request_ids)} requests in {submit_time:.2f}s")

    # Wait for all responses and collect results
    responses = {}
    completion_times = {}
    actual_output_tokens = {}

    print("\nWaiting for responses...\n")
    wait_start = time.time()

    for i, request_id in enumerate(request_ids):
        response = engine.get_response(request_id, timeout=120)  # 2 min timeout
        completion_time = time.time()

        if response:
            responses[request_id] = response
            completion_times[request_id] = completion_time

            # Get response text and count actual output tokens
            if hasattr(response, 'text'):
                output_text = response.text
            elif hasattr(response, 'output'):
                output_text = response.output
            elif isinstance(response, dict):
                output_text = response.get('text', response.get('output', str(response)))
            else:
                output_text = str(response)

            # Estimate actual output tokens (rough approximation: words * 1.3)
            actual_output_tokens[request_id] = int(len(output_text.split()) * 1.3)

            # Print each request's output
            if print_outputs:
                latency = completion_time - start_times[request_id]
                image_path = image_paths_map[request_id]
                prompt = prompts_map[request_id]

                print(f"{'='*60}")
                print(f"[Request {i+1}/{len(request_ids)}]")
                print(f"[Image]: {os.path.basename(image_path)}")
                print(f"[Prompt]: {prompt}")
                print(f"[Output]: {output_text[:300]}{'...' if len(output_text) > 300 else ''}")
                print(f"[Latency]: {latency:.3f}s")
                print(f"{'='*60}\n")
        else:
            print(f"[Request {i+1}] TIMEOUT or FAILED")

    total_time = time.time() - submit_start
    processing_time = time.time() - wait_start

    # Calculate metrics
    successful = len(responses)
    failed = len(request_ids) - successful

    # Calculate latencies
    latencies = []
    for request_id in responses:
        latency = completion_times[request_id] - start_times[request_id]
        latencies.append(latency)

    if latencies:
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
    else:
        avg_latency = p50_latency = p95_latency = p99_latency = 0

    # Calculate token statistics
    total_input_tokens = sum(input_tokens_map[request_id] for request_id in responses.keys())
    total_output_tokens = sum(actual_output_tokens.get(request_id, max_tokens_map[request_id])
                              for request_id in responses.keys())
    total_tokens = total_input_tokens + total_output_tokens

    throughput = total_output_tokens / processing_time if processing_time > 0 else 0
    requests_per_second = successful / processing_time if processing_time > 0 else 0

    # Print results
    print(f"\n{'='*60}")
    print(f"Throughput Benchmark Results:")
    print(f"{'='*60}")
    print(f"Total requests:        {len(request_ids)}")
    print(f"Successful:            {successful}")
    print(f"Failed:                {failed}")
    print(f"Total time:            {total_time:.2f}s")
    print(f"Processing time:       {processing_time:.2f}s")
    print(f"")
    print(f"Token Statistics:")
    print(f"  Total input tokens:  {total_input_tokens:,}")
    print(f"  Total output tokens: {total_output_tokens:,}")
    print(f"  Total tokens:        {total_tokens:,}")
    print(f"")
    print(f"Throughput:            {throughput:.2f} output tokens/s")
    print(f"Requests/second:       {requests_per_second:.2f}")
    print(f"")
    print(f"Latency Statistics:")
    print(f"  Average:             {avg_latency:.3f}s")
    print(f"  P50:                 {p50_latency:.3f}s")
    print(f"  P95:                 {p95_latency:.3f}s")
    print(f"  P99:                 {p99_latency:.3f}s")

    # Get engine stats
    try:
        stats = engine.get_stats()
        print(f"\nEngine Statistics:")
        print(f"  Pending requests:    {stats['pending_requests']}")
        print(f"  Block usage:         {stats['block_manager']['used_blocks']}/{stats['block_manager']['total_blocks']}")
    except:
        pass

    return {
        'total_requests': len(request_ids),
        'successful': successful,
        'failed': failed,
        'total_time': total_time,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_tokens': total_tokens,
        'throughput_tokens_per_sec': throughput,
        'requests_per_sec': requests_per_second,
        'avg_latency': avg_latency,
        'p50_latency': p50_latency,
        'p95_latency': p95_latency,
        'p99_latency': p99_latency,
    }


def benchmark_latency_with_output(
    engine: MultimodalLLMEngine,
    coco_dir: str,
    num_iterations: int = 10,
    print_outputs: bool = True
):
    """Benchmark single request latency with output printing"""

    print(f"\n{'='*60}")
    print(f"Latency Benchmark: {num_iterations} iterations")
    print(f"{'='*60}")

    # Load some COCO images
    test_cases = generate_test_prompts_from_coco(coco_dir, num_iterations)

    latencies = []

    for i, (prompt, images, image_path) in enumerate(test_cases[:num_iterations]):
        # Submit request
        start_time = time.time()
        request_id = engine.add_request(
            prompt=prompt,
            images=images,
            sampling_params={
                'temperature': 0.7,
                'max_tokens': 100,
            }
        )

        # Wait for response
        response = engine.get_response(request_id, timeout=30)
        end_time = time.time()

        if response:
            latency = end_time - start_time
            latencies.append(latency)

            # Get response text
            if hasattr(response, 'text'):
                output_text = response.text
            elif hasattr(response, 'output'):
                output_text = response.output
            elif isinstance(response, dict):
                output_text = response.get('text', response.get('output', str(response)))
            else:
                output_text = str(response)

            if print_outputs:
                print(f"\n[Iteration {i+1}]")
                print(f"  Image: {os.path.basename(image_path)}")
                print(f"  Prompt: {prompt}")
                print(f"  Output: {output_text[:200]}{'...' if len(output_text) > 200 else ''}")
                print(f"  Latency: {latency:.3f}s")
        else:
            print(f"  Iteration {i+1}: FAILED")

    # Calculate statistics
    if latencies:
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        std_latency = np.std(latencies)

        print(f"\n{'='*60}")
        print(f"Latency Benchmark Results:")
        print(f"{'='*60}")
        print(f"Average latency:       {avg_latency:.3f}s")
        print(f"Min latency:           {min_latency:.3f}s")
        print(f"Max latency:           {max_latency:.3f}s")
        print(f"Std deviation:         {std_latency:.3f}s")

    return {
        'iterations': num_iterations,
        'avg_latency': avg_latency if latencies else 0,
        'min_latency': min_latency if latencies else 0,
        'max_latency': max_latency if latencies else 0,
        'std_latency': std_latency if latencies else 0,
    }


def main():
    """Main benchmark function"""
    import argparse

    parser = argparse.ArgumentParser(description='Multimodal LLaVA Benchmark with COCO images')
    parser.add_argument('--model_path', type=str,
                        default=os.path.expanduser("~/huggingface/llava-v1.5-7b/"),
                        help='Path to LLaVA model')
    parser.add_argument('--coco_dir', type=str,
                        default="./coco/train2017/",
                        help='Path to COCO train2017 directory')
    parser.add_argument('--num_requests', type=int, default=16,
                        help='Number of requests for throughput benchmark')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for processing')
    parser.add_argument('--vision_device', type=str, default="cuda:0",
                        help='Device for vision encoder')
    parser.add_argument('--language_device', type=str, default="cuda:1",
                        help='Device for language model')
    parser.add_argument('--hardware_cost', type=float, default=0.646,
                        help='Hardware cost per hour in dollars (default: $0.646 for 2x A10G)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--no_print_outputs', action='store_true',
                        help='Disable printing individual outputs')

    args = parser.parse_args()

    seed(42)  # For reproducibility

    print(f"{'='*60}")
    print(f"Multimodal LLaVA Benchmark with COCO Images")
    print(f"Model: {args.model_path}")
    print(f"COCO Dir: {args.coco_dir}")
    print(f"{'='*60}")

    # Check COCO directory
    if not os.path.exists(args.coco_dir):
        print(f"Warning: COCO directory not found at {args.coco_dir}")
        print("Will use dummy images instead")

    # Create engine
    print("\nInitializing multimodal engine...")
    engine = create_engine(
        model_path=args.model_path,
        vision_device=args.vision_device,
        language_device=args.language_device,
        num_kv_blocks=1024,
        block_size=16,
        batch_size=args.batch_size,
        pipeline_depth=3,
        enable_pipeline_parallel=True,
        verbose=args.verbose
    )

    print("Engine initialized successfully!")

    # Warmup with a COCO image
    print("\nWarming up...")
    warmup_cases = generate_test_prompts_from_coco(args.coco_dir, 1)
    if warmup_cases:
        warmup_prompt, warmup_images, _ = warmup_cases[0]
    else:
        warmup_prompt = "<image>\nWhat is in this image?"
        rgb = np.random.randint(0, 256, (336, 336, 3), dtype=np.uint8)
        warmup_images = [Image.fromarray(rgb)]

    warmup_id = engine.add_request(warmup_prompt, warmup_images)
    warmup_response = engine.get_response(warmup_id, timeout=30)

    if warmup_response:
        print("Warmup completed successfully!")
    else:
        print("Warmup failed!")

    # Run benchmarks
    benchmark_start_time = time.time()

    try:
        print_outputs = not args.no_print_outputs

        # 1. Latency benchmark
        latency_results = benchmark_latency_with_output(
            engine,
            args.coco_dir,
            num_iterations=5,
            print_outputs=print_outputs
        )

        # 2. Throughput benchmark
        throughput_results = benchmark_throughput_with_output(
            engine,
            args.coco_dir,
            num_requests=args.num_requests,
            verbose=args.verbose,
            print_outputs=print_outputs
        )

        # Calculate total runtime
        total_runtime = time.time() - benchmark_start_time

        # 3. Tokens per dollar analysis
        cost_results = None
        if throughput_results and throughput_results.get('total_tokens', 0) > 0:
            cost_results = calculate_tokens_per_dollar(
                total_tokens=throughput_results['total_tokens'],
                total_output_tokens=throughput_results['total_output_tokens'],
                total_runtime_seconds=total_runtime,
                hardware_cost_per_hour=args.hardware_cost
            )

        # Save results
        all_results = {
            'model': args.model_path,
            'coco_dir': args.coco_dir,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'vision_device': args.vision_device,
                'language_device': args.language_device,
                'batch_size': args.batch_size,
                'num_requests': args.num_requests,
                'hardware_cost_per_hour': args.hardware_cost,
            },
            'latency': latency_results,
            'throughput': throughput_results,
            'cost_analysis': cost_results,
            'total_runtime_seconds': total_runtime,
        }

        # Save to JSON
        output_file = 'benchmark_results.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Benchmark completed!")
        print(f"Total runtime: {total_runtime:.2f}s")
        print(f"Results saved to: {output_file}")
        print(f"{'='*60}")

    finally:
        # Cleanup
        print("\nShutting down engine...")
        engine.shutdown()
        print("Done!")


if __name__ == "__main__":
    main()
