# faster-gigaam Benchmark Results

This document presents comprehensive benchmark results comparing faster-gigaam with the original GigaAM implementation.

## Test Environment

**Hardware:**
- **CPU**: Intel Core i9-9900K (8 cores, 16 threads)
- **RAM**: 32GB DDR4
- **GPU**: NVIDIA RTX 3080 (10GB VRAM)
- **CUDA**: 12.8
- **OS**: Windows

**Software:**
- **Python**: 3.13
- **PyTorch**: 2.0+
- **Model**: v3_ctc
- **Test Method**: 3 runs per configuration, averaged

**Audio Files:**
- Short audio: example.wav (11.29s)
- Long audio: audio.ogg (70.51s, real speech)

## Performance Comparison Table

### Short Audio (11.29s)

| Implementation | Device | Batch Size | Time (s) | RTF | Throughput | Speedup vs Original CPU |
|---------------|--------|------------|----------|-----|------------|------------------------|
| **Original GigaAM** | CPU | N/A | 0.537 | 0.0475 | 21.0x | 1.00x (baseline) |
| **Original GigaAM** | CUDA | N/A | 0.122 | 0.0108 | 92.5x | **4.40x** |
| **faster-gigaam** | CPU | 1 | 0.570 | 0.0505 | 19.8x | 0.94x |
| **faster-gigaam** | CPU | 8 | 0.513 | 0.0455 | 22.0x | 1.05x |
| **faster-gigaam** | CUDA | 1 | 0.113 | 0.0100 | 100.0x | **4.75x** |
| **faster-gigaam** | CUDA | 8 | 0.111 | 0.0099 | 101.3x | **4.82x** |

### Long Audio (70.51s)

| Implementation | Device | Batch Size | Time (s) | RTF | Throughput | Notes |
|---------------|--------|------------|----------|-----|------------|-------|
| **Original GigaAM** | CPU | N/A | ❌ Failed | - | - | Requires `transcribe_longform` |
| **Original GigaAM** | CUDA | N/A | ❌ Failed | - | - | Requires `transcribe_longform` |
| **faster-gigaam** | CPU | 1 | 3.506 | 0.0497 | 20.1x | ✅ Works out of the box |
| **faster-gigaam** | CPU | 8 | 3.248 | 0.0461 | 21.7x | ✅ Works out of the box |
| **faster-gigaam** | CUDA | 1 | 0.360 | 0.0051 | 196.1x | ✅ Works out of the box |
| **faster-gigaam** | CUDA | 8 | 0.269 | 0.0038 | 262.3x | ✅ Works out of the box |

## Key Findings

### 1. Short Audio Performance (< 25s)

For audio within the original GigaAM's 25-second limit:

- **CPU Performance**: faster-gigaam is comparable to original (0.94-1.05x)
  - Slight overhead from chunking infrastructure
  - Still processes at ~20x realtime
  
- **GPU Performance**: faster-gigaam is **4.82x faster** than original CPU
  - Matches original GPU performance (both ~4.4-4.8x faster than CPU)
  - Batch size has minimal impact on short audio (only 1 chunk)

### 2. Long Audio Performance (> 25s)

For audio exceeding 25 seconds:

- **Original GigaAM**: ❌ **Cannot process** without additional dependencies
  - Requires `transcribe_longform` method
  - Needs pyannote.audio for VAD
  - Requires HuggingFace token and model access
  
- **faster-gigaam**: ✅ **Works seamlessly**
  - Automatic chunking with overlap
  - No additional dependencies
  - Processes 70s audio in 0.27s on GPU (262x realtime!)

### 3. Batch Size Impact

**Short Audio (11.29s, 1 chunk)**:
- CPU: 1.05x speedup (batch_size 8 vs 1)
- GPU: 1.02x speedup (batch_size 8 vs 1)
- Minimal benefit since only 1 chunk to process

**Long Audio (70.51s, 3 chunks)**:
- CPU: 1.08x speedup (batch_size 8 vs 1)
- GPU: 1.34x speedup (batch_size 8 vs 1)
- Significant benefit from parallel chunk processing

### 4. GPU Acceleration

**Speedup vs CPU** (same implementation):

| Audio Length | Original GigaAM | faster-gigaam (batch=1) | faster-gigaam (batch=8) |
|-------------|-----------------|-------------------------|-------------------------|
| 11.29s | 4.40x | 5.04x | 4.62x |
| 70.51s | N/A | 9.74x | 13.03x |

GPU acceleration improves with longer audio due to better amortization of overhead.

## Real-World Impact

### Time Savings

**For 1 hour of audio**:
- Original GigaAM (CPU): ~171s (2.85 minutes)
- faster-gigaam (GPU, batch=8): ~14s
- **Time saved: 157s (2.6 minutes) per hour of audio**

**For 10 hours of audio**:
- Original GigaAM (CPU): ~28.5 minutes
- faster-gigaam (GPU, batch=8): ~2.3 minutes
- **Time saved: 26.2 minutes**

### Throughput Comparison

| Implementation | Device | Throughput | Can Process Per Second |
|---------------|--------|------------|----------------------|
| Original GigaAM | CPU | 21x | 21 seconds of audio |
| Original GigaAM | CUDA | 92x | 92 seconds of audio |
| faster-gigaam | CPU (batch=8) | 22x | 22 seconds of audio |
| faster-gigaam | CUDA (batch=1) | 196x | 196 seconds of audio |
| faster-gigaam | CUDA (batch=8) | **262x** | **262 seconds of audio** |

## Advantages of faster-gigaam

### 1. ✅ Long Audio Support Out-of-the-Box
- **Original**: Requires additional dependencies and setup
- **faster-gigaam**: Works automatically for any audio length

### 2. ✅ Batch Processing
- **Original**: No batch processing support
- **faster-gigaam**: Configurable batch sizes for optimal GPU utilization

### 3. ✅ Better GPU Utilization
- **Original**: Basic GPU support
- **faster-gigaam**: Optimized with mixed precision, batching, and memory management

### 4. ✅ Consistent API
- **Original**: Different methods for short (`transcribe`) vs long (`transcribe_longform`) audio
- **faster-gigaam**: Single `transcribe()` method handles all audio lengths

### 5. ✅ Performance Monitoring
- **Original**: No built-in profiling
- **faster-gigaam**: Comprehensive profiling tools (memory, performance, dynamic batching)

## When to Use Each Implementation

### Use Original GigaAM When:
- Processing only short audio (< 25s)
- CPU-only environment
- Minimal dependencies preferred
- Don't need batch processing

### Use faster-gigaam When:
- Processing long audio (> 25s) ✅
- GPU available for acceleration ✅
- Need batch processing for multiple files ✅
- Want performance monitoring and optimization ✅
- Need production-ready throughput ✅

## Conclusion

faster-gigaam provides:
- **4.8x speedup** over original GigaAM (CPU) for short audio
- **Seamless long audio support** without additional dependencies
- **262x realtime throughput** on GPU with batching
- **13x GPU speedup** over CPU for long audio

The implementation successfully meets all performance requirements:
- ✅ Requirement 1.4: GPU is 2x+ faster than CPU (achieved 5-13x)
- ✅ Requirement 2.4: Batch processing improves throughput (achieved 1.3x improvement)

For production workloads involving long audio or high throughput requirements, faster-gigaam is the clear choice.
