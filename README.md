# faster-gigaam

Optimized inference engine for GigaAM ASR models with CUDA acceleration, batch processing, and unlimited audio length support.

## Features

- **🚀 CUDA Acceleration**: 5-13x faster than CPU with mixed precision inference
- **📦 Batch Processing**: Process multiple audio chunks simultaneously for maximum GPU utilization
- **♾️ Unlimited Audio Length**: Automatic chunking with overlap - no 25-second limit
- **🎯 Simple API**: Single `transcribe()` method handles any audio length
- **⚡ High Throughput**: Process up to 262 seconds of audio per second (262x realtime on GPU)
- **🔧 Zero Extra Dependencies**: No VAD or additional models required

## Performance

Benchmark results on real audio (v3_ctc model):

**Test Hardware:**
- CPU: Intel Core i9-9900K
- RAM: 32GB
- GPU: NVIDIA RTX 3080 (10GB VRAM)
- CUDA: 12.8

### Short Audio (11.29s)

| Implementation | Device | Batch Size | Time (s) | Speedup vs Original CPU |
|---------------|--------|------------|----------|------------------------|
| Original GigaAM | CPU | N/A | 0.537 | 1.00x (baseline) |
| Original GigaAM | CUDA | N/A | 0.122 | 4.40x |
| **faster-gigaam** | CPU | 8 | 0.513 | 1.05x |
| **faster-gigaam** | CUDA | 8 | **0.111** | **4.82x** |

### Long Audio (70.51s)

| Implementation | Device | Batch Size | Time (s) | Throughput |
|---------------|--------|------------|----------|------------|
| Original GigaAM | CPU | N/A | ❌ Failed | Requires `transcribe_longform` |
| Original GigaAM | CUDA | N/A | ❌ Failed | Requires `transcribe_longform` |
| **faster-gigaam** | CPU | 8 | 3.248 | 21.7x realtime |
| **faster-gigaam** | CUDA | 8 | **0.269** | **262x realtime** |

**Key Advantage**: faster-gigaam handles long audio out-of-the-box, while original GigaAM requires additional dependencies (pyannote.audio, HuggingFace token) and a separate `transcribe_longform` method.

## Installation

> **Note**: PyTorch is not included as a dependency to allow you to choose between CPU and CUDA versions. Install PyTorch first according to your hardware.

> **Tip**: For faster installations, consider using [UV package manager](https://github.com/astral-sh/uv) instead of pip.

### Step 1: Install PyTorch

Choose the appropriate PyTorch version for your system:

**With CUDA (for GPU acceleration):**
```bash
# CUDA 12.8 (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Or with UV (faster package manager)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

See [PyTorch installation guide](https://pytorch.org/get-started/locally/) for more options.

### Step 2: Install GigaAM

```bash
pip install git+https://github.com/salute-developers/GigaAM.git
```

### Step 3: Install faster-gigaam

```bash
# Clone the repository
git clone https://github.com/yourusername/faster-gigaam.git
cd faster-gigaam

# Install in development mode
pip install -e .
```

**Requirements**:
- Python >= 3.10 (tested with Python 3.13)
- PyTorch >= 2.0.0 (installed in Step 1)
- GigaAM (installed in Step 2)

**Recommended**:
- [UV package manager](https://github.com/astral-sh/uv) for faster installations

## Quick Start

```python
from faster_gigaam import FastGigaAM

# Initialize model with GPU and batch processing
model = FastGigaAM(
    model_name="v3_ctc",
    device="cuda",
    batch_size=8
)

# Transcribe audio (any length!)
segments, info = model.transcribe("long_audio.wav")

# Print results
for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

# Print performance info
print(f"\nProcessed {info.duration:.1f}s audio in {info.processing_time:.2f}s")
print(f"Throughput: {info.duration / info.processing_time:.1f}x realtime")
```

### CPU Usage

```python
# Use CPU if CUDA is not available
model = FastGigaAM(
    model_name="v3_ctc",
    device="cpu",
    batch_size=4
)

segments, info = model.transcribe("audio.wav")
```

### Batch Processing Multiple Files

```python
# Process multiple files efficiently
audio_files = ["file1.wav", "file2.wav", "file3.wav"]
results = model.transcribe_batch(audio_files)

for i, (segments, info) in enumerate(results):
    print(f"\nFile {i+1}:")
    for segment in segments:
        print(f"  [{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
```

## Documentation

- **Examples**: See `examples/` folder for more usage examples
- **Benchmarks**: See `BENCHMARK_RESULTS.md` for detailed performance analysis
- **API Reference**: See docstrings in `faster_gigaam/fast_gigaam.py`

## Supported Models

All GigaAM ASR models are supported:
- `v3_ctc` - CTC-based model
- `v3_e2e_rnnt` - RNN-T based model
- Other GigaAM model variants

## Project Status

**This is a pet project** created for personal use and shared with the community. 

- ✅ Fully functional and tested
- ⚠️ Updates will be sporadic based on my needs
- 🤝 Issues and PRs welcome, but no guarantees on response time
- 🍴 Feel free to fork if you need active maintenance

## License

MIT License - same as GigaAM

## Acknowledgments

Built on top of the excellent [GigaAM](https://github.com/salute-developers/GigaAM) project by Salute Developers.
