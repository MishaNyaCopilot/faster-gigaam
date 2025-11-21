"""Benchmark CPU vs GPU performance.

This script compares transcription performance between CPU and GPU
for various audio durations.
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
from faster_gigaam.profiler import MemoryProfiler, PerformanceProfiler


def generate_test_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate synthetic audio for testing.
    
    Args:
        duration: Audio duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Audio samples as numpy array
    """
    num_samples = int(duration * sample_rate)
    # Generate pink noise (more realistic than white noise)
    audio = np.random.randn(num_samples).astype(np.float32)
    return audio


def benchmark_device(
    model_name: str,
    device: str,
    audio_durations: list,
    batch_size: int = 1,
    num_runs: int = 3,
):
    """Benchmark transcription on a specific device.
    
    Args:
        model_name: GigaAM model name
        device: Device to use ("cpu" or "cuda")
        audio_durations: List of audio durations to test
        batch_size: Batch size to use
        num_runs: Number of runs per duration for averaging
        
    Returns:
        Dictionary of results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {device.upper()} (batch_size={batch_size})")
    print(f"{'='*60}\n")
    
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
        print(f"Failed to initialize model on {device}: {e}")
        return None
    
    results = []
    memory_profiler = MemoryProfiler(device=torch.device(device) if device == "cuda" else None)
    
    for duration in audio_durations:
        print(f"Testing {duration}s audio...")
        
        # Generate test audio
        audio = generate_test_audio(duration)
        
        # Warm-up run
        try:
            _, _ = model.transcribe(audio)
        except Exception as e:
            print(f"  Warm-up failed: {e}")
            continue
        
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
                
            except Exception as e:
                print(f"  Run {run+1} failed: {e}")
                continue
        
        if not run_times:
            print(f"  All runs failed for {duration}s audio")
            continue
        
        # Calculate statistics
        avg_time = np.mean(run_times)
        std_time = np.std(run_times)
        rtf = avg_time / duration
        throughput = duration / avg_time
        
        result = {
            "duration": duration,
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
        
        results.append(result)
        
        # Print result
        print(f"  Duration: {duration:.1f}s")
        print(f"  Avg time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"  RTF: {rtf:.3f}")
        print(f"  Throughput: {throughput:.1f}x realtime")
        print(f"  Chunks: {info.num_chunks}")
        if "peak_memory_mb" in result:
            print(f"  Peak memory: {result['peak_memory_mb']:.1f}MB")
        print()
    
    return results


def compare_results(cpu_results, gpu_results):
    """Compare CPU and GPU results.
    
    Args:
        cpu_results: Results from CPU benchmark
        gpu_results: Results from GPU benchmark
    """
    if not cpu_results or not gpu_results:
        print("Cannot compare - missing results")
        return
    
    print(f"\n{'='*60}")
    print("CPU vs GPU Comparison")
    print(f"{'='*60}\n")
    
    print(f"{'Duration':<12} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<12} {'GPU RTF':<12}")
    print("-" * 60)
    
    for cpu_res, gpu_res in zip(cpu_results, gpu_results):
        if cpu_res["duration"] != gpu_res["duration"]:
            continue
        
        speedup = cpu_res["avg_time"] / gpu_res["avg_time"]
        
        print(
            f"{cpu_res['duration']:<12.1f} "
            f"{cpu_res['avg_time']:<12.3f} "
            f"{gpu_res['avg_time']:<12.3f} "
            f"{speedup:<12.2f}x "
            f"{gpu_res['rtf']:<12.3f}"
        )
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU vs GPU performance")
    parser.add_argument(
        "--model",
        type=str,
        default="v3_ctc",
        help="Model name (default: v3_ctc)",
    )
    parser.add_argument(
        "--durations",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 30.0, 60.0, 120.0],
        help="Audio durations to test in seconds (default: 5 10 30 60 120)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per duration (default: 3)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Only benchmark CPU",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Only benchmark GPU",
    )
    
    args = parser.parse_args()
    
    print("faster-gigaam CPU vs GPU Benchmark")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Runs per duration: {args.runs}")
    print(f"Durations: {args.durations}")
    
    cpu_results = None
    gpu_results = None
    
    # Benchmark CPU
    if not args.gpu_only:
        cpu_results = benchmark_device(
            model_name=args.model,
            device="cpu",
            audio_durations=args.durations,
            batch_size=args.batch_size,
            num_runs=args.runs,
        )
    
    # Benchmark GPU
    if not args.cpu_only and torch.cuda.is_available():
        gpu_results = benchmark_device(
            model_name=args.model,
            device="cuda",
            audio_durations=args.durations,
            batch_size=args.batch_size,
            num_runs=args.runs,
        )
    elif not args.cpu_only:
        print("\nGPU not available, skipping GPU benchmark")
    
    # Compare results
    if cpu_results and gpu_results:
        compare_results(cpu_results, gpu_results)


if __name__ == "__main__":
    main()
