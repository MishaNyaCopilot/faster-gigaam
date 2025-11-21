"""Benchmark with real audio files.

This script tests transcription performance with actual audio files
to validate that performance holds up with real speech data.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from faster_gigaam import FastGigaAM
from faster_gigaam.profiler import MemoryProfiler, PerformanceProfiler
import gigaam


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    audio = gigaam.load_audio(audio_path, sample_rate=16000)
    return len(audio) / 16000.0


def benchmark_audio_file(
    audio_path: str,
    model_name: str,
    device: str,
    batch_size: int,
    num_runs: int = 3,
):
    """Benchmark transcription on a real audio file.
    
    Args:
        audio_path: Path to audio file
        model_name: GigaAM model name
        device: Device to use
        batch_size: Batch size
        num_runs: Number of runs for averaging
        
    Returns:
        Dictionary of results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {audio_path}")
    print(f"Device: {device.upper()}, Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Get audio duration
    try:
        audio_duration = get_audio_duration(audio_path)
        print(f"Audio duration: {audio_duration:.2f}s")
    except Exception as e:
        print(f"Failed to load audio: {e}")
        return None
    
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
        print(f"Failed to initialize model: {e}")
        return None
    
    memory_profiler = MemoryProfiler(device=torch.device(device) if device == "cuda" else None)
    
    # Warm-up run
    print("\nWarm-up run...")
    try:
        segments, info = model.transcribe(audio_path)
        print(f"Transcription preview: {segments[0].text[:100] if segments else 'No output'}...")
    except Exception as e:
        print(f"Warm-up failed: {e}")
        return None
    
    # Benchmark runs
    print(f"\nRunning {num_runs} benchmark iterations...")
    run_times = []
    memory_stats = []
    all_segments = []
    
    for run in range(num_runs):
        # Reset memory stats
        memory_profiler.reset_peak_stats()
        
        # Time transcription
        start_time = time.time()
        try:
            segments, info = model.transcribe(audio_path)
            elapsed = time.time() - start_time
            run_times.append(elapsed)
            
            if run == 0:
                all_segments = segments
            
            # Get memory stats
            mem_stats = memory_profiler.get_memory_stats()
            if mem_stats:
                memory_stats.append(mem_stats)
            
            print(f"  Run {run+1}: {elapsed:.3f}s")
            
        except Exception as e:
            print(f"  Run {run+1} failed: {e}")
            continue
    
    if not run_times:
        print("All runs failed")
        return None
    
    # Calculate statistics
    import numpy as np
    avg_time = np.mean(run_times)
    std_time = np.std(run_times)
    min_time = np.min(run_times)
    max_time = np.max(run_times)
    rtf = avg_time / audio_duration
    throughput = audio_duration / avg_time
    
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Audio duration:    {audio_duration:.2f}s")
    print(f"Avg time:          {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"Min/Max time:      {min_time:.3f}s / {max_time:.3f}s")
    print(f"RTF:               {rtf:.4f}")
    print(f"Throughput:        {throughput:.1f}x realtime")
    print(f"Chunks processed:  {info.num_chunks}")
    
    if memory_stats:
        avg_peak_mem = np.mean([s.max_allocated_mb for s in memory_stats])
        print(f"Peak GPU memory:   {avg_peak_mem:.1f}MB")
    
    # Show transcription
    print(f"\n{'='*60}")
    print("Transcription")
    print(f"{'='*60}")
    for segment in all_segments[:5]:  # Show first 5 segments
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
    if len(all_segments) > 5:
        print(f"... ({len(all_segments) - 5} more segments)")
    
    return {
        "audio_path": audio_path,
        "audio_duration": audio_duration,
        "avg_time": avg_time,
        "std_time": std_time,
        "min_time": min_time,
        "max_time": max_time,
        "rtf": rtf,
        "throughput": throughput,
        "num_chunks": info.num_chunks,
        "device": device,
        "batch_size": batch_size,
    }


def compare_devices(audio_path: str, model_name: str, batch_size: int, num_runs: int):
    """Compare CPU vs GPU performance on the same audio file."""
    print("\n" + "="*60)
    print("CPU vs GPU Comparison")
    print("="*60)
    
    # Benchmark CPU
    cpu_result = benchmark_audio_file(
        audio_path=audio_path,
        model_name=model_name,
        device="cpu",
        batch_size=batch_size,
        num_runs=num_runs,
    )
    
    # Benchmark GPU if available
    gpu_result = None
    if torch.cuda.is_available():
        gpu_result = benchmark_audio_file(
            audio_path=audio_path,
            model_name=model_name,
            device="cuda",
            batch_size=batch_size,
            num_runs=num_runs,
        )
    else:
        print("\nGPU not available, skipping GPU benchmark")
    
    # Compare results
    if cpu_result and gpu_result:
        speedup = cpu_result["avg_time"] / gpu_result["avg_time"]
        
        print(f"\n{'='*60}")
        print("Comparison Summary")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'CPU':<15} {'GPU':<15} {'Speedup':<10}")
        print("-" * 60)
        print(f"{'Time':<20} {cpu_result['avg_time']:<15.3f} {gpu_result['avg_time']:<15.3f} {speedup:<10.2f}x")
        print(f"{'RTF':<20} {cpu_result['rtf']:<15.4f} {gpu_result['rtf']:<15.4f}")
        print(f"{'Throughput':<20} {cpu_result['throughput']:<15.1f}x {gpu_result['throughput']:<15.1f}x")
        print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark with real audio files")
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
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "both"],
        help="Device to use (default: both if CUDA available, else cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs for averaging (default: 3)",
    )
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not Path(args.audio_path).exists():
        print(f"Error: Audio file not found: {args.audio_path}")
        return
    
    print("faster-gigaam Real Audio Benchmark")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Runs: {args.runs}")
    
    # Determine which devices to test
    if args.device is None:
        args.device = "both" if torch.cuda.is_available() else "cpu"
    
    if args.device == "both":
        compare_devices(args.audio_path, args.model, args.batch_size, args.runs)
    else:
        benchmark_audio_file(
            audio_path=args.audio_path,
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            num_runs=args.runs,
        )


if __name__ == "__main__":
    main()
