# faster-gigaam Benchmarks

This directory contains benchmark scripts for evaluating the performance of faster-gigaam.

## Available Benchmarks

### 1. CPU vs GPU Comparison (`benchmark_cpu_vs_gpu.py`)

Compares transcription performance between CPU and GPU across various audio durations.

**Usage:**
```bash
# Basic usage
python benchmarks/benchmark_cpu_vs_gpu.py

# Custom durations
python benchmarks/benchmark_cpu_vs_gpu.py --durations 10 30 60 120

# With batch processing
python benchmarks/benchmark_cpu_vs_gpu.py --batch-size 8

# CPU only
python benchmarks/benchmark_cpu_vs_gpu.py --cpu-only

# GPU only
python benchmarks/benchmark_cpu_vs_gpu.py --gpu-only
```

**Options:**
- `--model`: Model name (default: v3_ctc)
- `--durations`: Audio durations to test in seconds (default: 5 10 30 60 120)
- `--batch-size`: Batch size to use (default: 1)
- `--runs`: Number of runs per duration for averaging (default: 3)
- `--cpu-only`: Only benchmark CPU
- `--gpu-only`: Only benchmark GPU

**Example Output:**
```
Duration     CPU Time     GPU Time     Speedup      GPU RTF     
------------------------------------------------------------
5.0          2.450        0.850        2.88x        0.170       
10.0         4.820        1.650        2.92x        0.165       
30.0         14.350       4.920        2.92x        0.164       
60.0         28.680       9.840        2.91x        0.164       
```

### 2. Batch Size Comparison (`benchmark_batch_sizes.py`)

Compares transcription performance across different batch sizes to identify optimal configurations.

**Usage:**
```bash
# Basic usage
python benchmarks/benchmark_batch_sizes.py

# Custom batch sizes
python benchmarks/benchmark_batch_sizes.py --batch-sizes 1 4 8 16 32

# Longer audio
python benchmarks/benchmark_batch_sizes.py --duration 120

# On CPU
python benchmarks/benchmark_batch_sizes.py --device cpu
```

**Options:**
- `--model`: Model name (default: v3_ctc)
- `--device`: Device to use (default: cuda if available, else cpu)
- `--batch-sizes`: Batch sizes to test (default: 1 2 4 8 16 32)
- `--duration`: Audio duration in seconds (default: 60.0)
- `--runs`: Number of runs per batch size (default: 3)

**Example Output:**
```
Batch Size   Time (s)     RTF          Throughput      Memory (MB)    
----------------------------------------------------------------------
1            9.840        0.164        6.1x (1.00x)    1250.5         
2            5.420        0.090        11.1x (1.82x)   1450.2         
4            3.180        0.053        18.9x (3.09x)   1850.8         
8            2.240        0.037        26.8x (4.39x)   2650.4         
16           1.920        0.032        31.3x (5.13x)   4250.1         

Optimal batch size: 16 (RTF: 0.032, throughput: 31.3x)
```

## Performance Metrics

### Real-Time Factor (RTF)
The ratio of processing time to audio duration. Lower is better.
- RTF < 1.0: Faster than real-time (can process audio faster than it plays)
- RTF = 1.0: Real-time processing
- RTF > 1.0: Slower than real-time

### Throughput
Audio seconds processed per wall-clock second. Higher is better.
- Throughput = 1 / RTF
- Example: Throughput of 10x means processing 10 seconds of audio per second

### Speedup
Relative performance improvement compared to baseline (usually batch_size=1 or CPU).
- Speedup = baseline_time / current_time
- Example: 3x speedup means 3 times faster than baseline

## Interpreting Results

### CPU vs GPU
- **Expected speedup**: 2-4x on GPU for audio > 10 seconds
- **GPU advantage increases** with longer audio and larger batch sizes
- **CPU may be faster** for very short audio (< 5 seconds) due to GPU overhead

### Batch Size
- **Larger batches** generally improve throughput but increase memory usage
- **Optimal batch size** depends on:
  - Available GPU memory
  - Audio duration (longer audio = more chunks = more benefit from batching)
  - Latency requirements (larger batches = higher latency)
- **Diminishing returns** typically occur beyond batch_size=16-32

### Memory Usage
- Memory usage scales roughly linearly with batch size
- Monitor peak memory to avoid OOM errors
- Leave headroom for other processes (aim for < 80% GPU memory usage)

## Tips for Optimal Performance

1. **Use GPU when available**: 2-4x speedup for typical workloads
2. **Enable FP16**: Use `compute_type="float16"` on GPU for additional speedup
3. **Tune batch size**: Start with 8, increase until memory limit or diminishing returns
4. **Batch multiple files**: Use `transcribe_batch()` for processing multiple files
5. **Monitor memory**: Use smaller batch sizes if hitting OOM errors

## System Requirements

- **CPU benchmarks**: Any system with Python 3.8+
- **GPU benchmarks**: CUDA-capable GPU with 4GB+ VRAM recommended
- **Memory**: 8GB+ RAM recommended for larger batch sizes

## Troubleshooting

### Out of Memory Errors
- Reduce batch size: `--batch-size 4` or `--batch-size 2`
- Use FP32 instead of FP16: `compute_type="float32"`
- Close other GPU applications

### Slow Performance
- Ensure CUDA is properly installed and detected
- Check GPU utilization with `nvidia-smi`
- Verify model is loaded on GPU (check logs)
- Try larger batch sizes for better GPU utilization

### Inconsistent Results
- Increase number of runs: `--runs 5`
- Ensure system is not under load during benchmarking
- Close background applications
