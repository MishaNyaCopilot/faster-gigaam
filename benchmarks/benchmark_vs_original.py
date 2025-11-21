"""Benchmark faster-gigaam vs original GigaAM implementation.

This script compares the performance of faster-gigaam against the
original GigaAM implementation to demonstrate the improvements.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gigaam
from faster_gigaam import FastGigaAM
from faster_gigaam.profiler import MemoryProfiler


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    audio = gigaam.load_audio(audio_path, sample_rate=16000)
    return len(audio) / 16000.0


def benchmark_original_gigaam(
    audio_path: str,
    model_name: str,
    device: str,
    num_runs: int = 3,
):
    """Benchmark original GigaAM implementation.
    
    Args:
        audio_path: Path to audio file
        model_name: Model name
        device: Device to use
        num_runs: Number of runs
        
    Returns:
        Dictionary of results
    """
    print(f"\n{'='*60}")
    print(f"Original GigaAM - {device.upper()}")
    print(f"{'='*60}\n")
    
    # Get audio duration
    audio_duration = get_audio_duration(audio_path)
    print(f"Audio duration: {audio_duration:.2f}s")
    
    # Load model
    print(f"Loading model '{model_name}'...")
    try:
        model = gigaam.load_model(
            model_name=model_name,
            device=device,
            fp16_encoder=(device == "cuda"),
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    
    memory_profiler = MemoryProfiler(device=torch.device(device) if device == "cuda" else None)
    
    # Warm-up
    print("Warm-up run...")
    try:
        _ = model.transcribe(audio_path)
    except Exception as e:
        print(f"Warm-up failed: {e}")
        return None
    
    # Benchmark runs
    print(f"Running {num_runs} benchmark iterations...")
    run_times = []
    memory_stats = []
    
    for run in range(num_runs):
        memory_profiler.reset_peak_stats()
        
        start_time = time.time()
        try:
            transcription = model.transcribe(audio_path)
            elapsed = time.time() - start_time
            run_times.append(elapsed)
            
            mem_stats = memory_profiler.get_memory_stats()
            if mem_stats:
                memory_stats.append(mem_stats)
            
            print(f"  Run {run+1}: {elapsed:.3f}s")
            
        except Exception as e:
            print(f"  Run {run+1} failed: {e}")
            continue
    
    if not run_times:
        return None
    
    # Calculate statistics
    avg_time = np.mean(run_times)
    std_time = np.std(run_times)
    rtf = avg_time / audio_duration
    throughput = audio_duration / avg_time
    
    result = {
        "implementation": "Original GigaAM",
        "device": device,
        "audio_duration": audio_duration,
        "avg_time": avg_time,
        "std_time": std_time,
        "rtf": rtf,
        "throughput": throughput,
    }
    
    if memory_stats:
        result["peak_memory_mb"] = np.mean([s.max_allocated_mb for s in memory_stats])
    
    print(f"\nResults:")
    print(f"  Avg time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"  RTF: {rtf:.4f}")
    print(f"  Throughput: {throughput:.1f}x realtime")
    if "peak_memory_mb" in result:
        print(f"  Peak memory: {result['peak_memory_mb']:.1f}MB")
    
    return result


def benchmark_faster_gigaam(
    audio_path: str,
    model_name: str,
    device: str,
    batch_size: int,
    num_runs: int = 3,
):
    """Benchmark faster-gigaam implementation.
    
    Args:
        audio_path: Path to audio file
        model_name: Model name
        device: Device to use
        batch_size: Batch size
        num_runs: Number of runs
        
    Returns:
        Dictionary of results
    """
    print(f"\n{'='*60}")
    print(f"faster-gigaam - {device.upper()} (batch_size={batch_size})")
    print(f"{'='*60}\n")
    
    # Get audio duration
    audio_duration = get_audio_duration(audio_path)
    print(f"Audio duration: {audio_duration:.2f}s")
    
    # Load model
    print(f"Loading model '{model_name}'...")
    compute_type = "float16" if device == "cuda" else "float32"
    
    try:
        model = FastGigaAM(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    
    memory_profiler = MemoryProfiler(device=torch.device(device) if device == "cuda" else None)
    
    # Warm-up
    print("Warm-up run...")
    try:
        _, _ = model.transcribe(audio_path)
    except Exception as e:
        print(f"Warm-up failed: {e}")
        return None
    
    # Benchmark runs
    print(f"Running {num_runs} benchmark iterations...")
    run_times = []
    memory_stats = []
    
    for run in range(num_runs):
        memory_profiler.reset_peak_stats()
        
        start_time = time.time()
        try:
            segments, info = model.transcribe(audio_path)
            elapsed = time.time() - start_time
            run_times.append(elapsed)
            
            mem_stats = memory_profiler.get_memory_stats()
            if mem_stats:
                memory_stats.append(mem_stats)
            
            print(f"  Run {run+1}: {elapsed:.3f}s")
            
        except Exception as e:
            print(f"  Run {run+1} failed: {e}")
            continue
    
    if not run_times:
        return None
    
    # Calculate statistics
    avg_time = np.mean(run_times)
    std_time = np.std(run_times)
    rtf = avg_time / audio_duration
    throughput = audio_duration / avg_time
    
    result = {
        "implementation": "faster-gigaam",
        "device": device,
        "batch_size": batch_size,
        "audio_duration": audio_duration,
        "avg_time": avg_time,
        "std_time": std_time,
        "rtf": rtf,
        "throughput": throughput,
        "num_chunks": info.num_chunks,
    }
    
    if memory_stats:
        result["peak_memory_mb"] = np.mean([s.max_allocated_mb for s in memory_stats])
    
    print(f"\nResults:")
    print(f"  Avg time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"  RTF: {rtf:.4f}")
    print(f"  Throughput: {throughput:.1f}x realtime")
    print(f"  Chunks: {info.num_chunks}")
    if "peak_memory_mb" in result:
        print(f"  Peak memory: {result['peak_memory_mb']:.1f}MB")
    
    return result


def print_comparison_table(results):
    """Print comparison table of all results.
    
    Args:
        results: List of result dictionaries
    """
    if not results:
        print("No results to compare")
        return
    
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    # Header
    print(f"{'Implementation':<20} {'Device':<8} {'Batch':<8} {'Time (s)':<12} {'RTF':<10} {'Throughput':<15} {'Memory (MB)':<12}")
    print("-" * 80)
    
    # Find baseline (original CPU)
    baseline = None
    for r in results:
        if r["implementation"] == "Original GigaAM" and r["device"] == "cpu":
            baseline = r
            break
    
    # Print results
    for result in results:
        impl = result["implementation"]
        device = result["device"].upper()
        batch = str(result.get("batch_size", "N/A"))
        time_val = result["avg_time"]
        rtf = result["rtf"]
        throughput = result["throughput"]
        memory = result.get("peak_memory_mb", 0.0)
        
        # Calculate speedup vs baseline
        if baseline and result != baseline:
            speedup = baseline["avg_time"] / time_val
            throughput_str = f"{throughput:.1f}x ({speedup:.2f}x)"
        else:
            throughput_str = f"{throughput:.1f}x"
        
        print(
            f"{impl:<20} {device:<8} {batch:<8} "
            f"{time_val:<12.3f} {rtf:<10.4f} {throughput_str:<15} "
            f"{memory:<12.1f}"
        )
    
    print()
    
    # Print key insights
    print(f"{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}\n")
    
    if baseline:
        # Find best faster-gigaam result
        faster_results = [r for r in results if r["implementation"] == "faster-gigaam"]
        if faster_results:
            best_faster = min(faster_results, key=lambda r: r["avg_time"])
            speedup = baseline["avg_time"] / best_faster["avg_time"]
            
            print(f"Audio Duration: {baseline['audio_duration']:.2f}s")
            print(f"\nOriginal GigaAM (CPU):")
            print(f"  Time: {baseline['avg_time']:.3f}s")
            print(f"  RTF: {baseline['rtf']:.4f}")
            print(f"  Throughput: {baseline['throughput']:.1f}x realtime")
            
            print(f"\nBest faster-gigaam ({best_faster['device'].upper()}, batch_size={best_faster.get('batch_size', 'N/A')}):")
            print(f"  Time: {best_faster['avg_time']:.3f}s")
            print(f"  RTF: {best_faster['rtf']:.4f}")
            print(f"  Throughput: {best_faster['throughput']:.1f}x realtime")
            
            print(f"\n🚀 SPEEDUP: {speedup:.2f}x faster than original GigaAM!")
            print(f"   ({baseline['avg_time']:.3f}s → {best_faster['avg_time']:.3f}s)")
            
            # Calculate time saved for 1 hour of audio
            time_saved_per_hour = (baseline['avg_time'] - best_faster['avg_time']) * (3600 / baseline['audio_duration'])
            print(f"\n💡 For 1 hour of audio, faster-gigaam saves ~{time_saved_per_hour:.1f}s of processing time")


def main():
    parser = argparse.ArgumentParser(
        description="Compare faster-gigaam vs original GigaAM"
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to audio file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="v3_ctc",
        help="Model name (default: v3_ctc)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8],
        help="Batch sizes to test for faster-gigaam (default: 1 8)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs for averaging (default: 3)",
    )
    parser.add_argument(
        "--skip-cpu",
        action="store_true",
        help="Skip CPU benchmarks",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip GPU benchmarks",
    )
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not Path(args.audio_path).exists():
        print(f"Error: Audio file not found: {args.audio_path}")
        return
    
    print("="*80)
    print("faster-gigaam vs Original GigaAM Benchmark")
    print("="*80)
    print(f"Audio file: {args.audio_path}")
    print(f"Model: {args.model}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Runs per test: {args.runs}")
    
    results = []
    
    # Benchmark original GigaAM on CPU
    if not args.skip_cpu:
        result = benchmark_original_gigaam(
            audio_path=args.audio_path,
            model_name=args.model,
            device="cpu",
            num_runs=args.runs,
        )
        if result:
            results.append(result)
    
    # Benchmark original GigaAM on GPU
    if not args.skip_gpu and torch.cuda.is_available():
        result = benchmark_original_gigaam(
            audio_path=args.audio_path,
            model_name=args.model,
            device="cuda",
            num_runs=args.runs,
        )
        if result:
            results.append(result)
    
    # Benchmark faster-gigaam on CPU
    if not args.skip_cpu:
        for batch_size in args.batch_sizes:
            result = benchmark_faster_gigaam(
                audio_path=args.audio_path,
                model_name=args.model,
                device="cpu",
                batch_size=batch_size,
                num_runs=args.runs,
            )
            if result:
                results.append(result)
    
    # Benchmark faster-gigaam on GPU
    if not args.skip_gpu and torch.cuda.is_available():
        for batch_size in args.batch_sizes:
            result = benchmark_faster_gigaam(
                audio_path=args.audio_path,
                model_name=args.model,
                device="cuda",
                batch_size=batch_size,
                num_runs=args.runs,
            )
            if result:
                results.append(result)
    
    # Print comparison table
    print_comparison_table(results)


if __name__ == "__main__":
    main()
