"""Benchmark different batch sizes.

This script compares transcription performance across different batch sizes
to identify optimal batching configurations.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from faster_gigaam import FastGigaAM
from faster_gigaam.profiler import MemoryProfiler


def generate_test_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate synthetic audio for testing.
    
    Args:
        duration: Audio duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Audio samples as numpy array
    """
    num_samples = int(duration * sample_rate)
    audio = np.random.randn(num_samples).astype(np.float32)
    return audio


def benchmark_batch_size(
    model_name: str,
    device: str,
    batch_size: int,
    audio_duration: float,
    num_runs: int = 3,
):
    """Benchmark transcription with a specific batch size.
    
    Args:
        model_name: GigaAM model name
        device: Device to use
        batch_size: Batch size to test
        audio_duration: Audio duration in seconds
        num_runs: Number of runs for averaging
        
    Returns:
        Dictionary of results or None if failed
    """
    print(f"\nTesting batch_size={batch_size}...")
    
    # Initialize model
    compute_type = "float16" if device == "cuda" else "float32"
    
    try:
        model = FastGigaAM(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
        )
    except Exception as e:
        print(f"  Failed to initialize: {e}")
        return None
    
    # Generate test audio
    audio = generate_test_audio(audio_duration)
    
    memory_profiler = MemoryProfiler(device=torch.device(device) if device == "cuda" else None)
    
    # Warm-up run
    try:
        _, _ = model.transcribe(audio)
    except Exception as e:
        print(f"  Warm-up failed: {e}")
        return None
    
    # Benchmark runs
    run_times = []
    memory_stats = []
    
    for run in range(num_runs):
        # Reset memory stats
        memory_profiler.reset_peak_stats()
        
        # Time transcription
        start_time = time.time()
        try:
            segments, info = model.transcribe(audio)
            elapsed = time.time() - start_time
            run_times.append(elapsed)
            
            # Get memory stats
            mem_stats = memory_profiler.get_memory_stats()
            if mem_stats:
                memory_stats.append(mem_stats)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Out of memory with batch_size={batch_size}")
                return None
            print(f"  Run {run+1} failed: {e}")
            return None
        except Exception as e:
            print(f"  Run {run+1} failed: {e}")
            return None
    
    if not run_times:
        return None
    
    # Calculate statistics
    avg_time = np.mean(run_times)
    std_time = np.std(run_times)
    rtf = avg_time / audio_duration
    throughput = audio_duration / avg_time
    
    result = {
        "batch_size": batch_size,
        "avg_time": avg_time,
        "std_time": std_time,
        "rtf": rtf,
        "throughput": throughput,
        "num_chunks": info.num_chunks,
    }
    
    # Add memory stats if available
    if memory_stats:
        avg_peak_mem = np.mean([s.max_allocated_mb for s in memory_stats])
        result["peak_memory_mb"] = avg_peak_mem
    
    # Print result
    print(f"  Avg time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"  RTF: {rtf:.3f}")
    print(f"  Throughput: {throughput:.1f}x realtime")
    if "peak_memory_mb" in result:
        print(f"  Peak memory: {result['peak_memory_mb']:.1f}MB")
    
    return result


def print_summary(results, audio_duration):
    """Print summary of results.
    
    Args:
        results: List of result dictionaries
        audio_duration: Audio duration tested
    """
    if not results:
        print("\nNo successful results to summarize")
        return
    
    print(f"\n{'='*70}")
    print(f"Batch Size Comparison ({audio_duration}s audio)")
    print(f"{'='*70}\n")
    
    print(f"{'Batch Size':<12} {'Time (s)':<12} {'RTF':<12} {'Throughput':<15} {'Memory (MB)':<15}")
    print("-" * 70)
    
    baseline_time = results[0]["avg_time"]
    
    for result in results:
        speedup = baseline_time / result["avg_time"]
        memory_str = f"{result['peak_memory_mb']:.1f}" if "peak_memory_mb" in result else "N/A"
        
        print(
            f"{result['batch_size']:<12} "
            f"{result['avg_time']:<12.3f} "
            f"{result['rtf']:<12.3f} "
            f"{result['throughput']:<8.1f}x ({speedup:.2f}x) "
            f"{memory_str:<15}"
        )
    
    print()
    
    # Find optimal batch size
    best_result = min(results, key=lambda r: r["avg_time"])
    print(f"Optimal batch size: {best_result['batch_size']} "
          f"(RTF: {best_result['rtf']:.3f}, throughput: {best_result['throughput']:.1f}x)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark different batch sizes")
    parser.add_argument(
        "--model",
        type=str,
        default="v3_ctc",
        help="Model name (default: v3_ctc)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="Batch sizes to test (default: 1 2 4 8 16 32)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Audio duration in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per batch size (default: 3)",
    )
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    print("faster-gigaam Batch Size Benchmark")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Audio duration: {args.duration}s")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Runs per batch size: {args.runs}")
    
    results = []
    
    for batch_size in args.batch_sizes:
        result = benchmark_batch_size(
            model_name=args.model,
            device=args.device,
            batch_size=batch_size,
            audio_duration=args.duration,
            num_runs=args.runs,
        )
        
        if result:
            results.append(result)
        else:
            print(f"  Skipping batch_size={batch_size} due to errors")
            # Stop testing larger batch sizes if we hit OOM
            if batch_size > 1:
                print(f"  Stopping batch size testing (likely memory limit reached)")
                break
    
    # Print summary
    print_summary(results, args.duration)


if __name__ == "__main__":
    main()
