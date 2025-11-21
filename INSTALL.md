# Installation Guide

## Quick Install (3 steps)

### 1. Install PyTorch

**For GPU (CUDA 12.8 - recommended):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Using UV (faster):**
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**For CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install GigaAM

```bash
pip install git+https://github.com/salute-developers/GigaAM.git

# Or with UV
uv pip install git+https://github.com/salute-developers/GigaAM.git
```

### 3. Install faster-gigaam

```bash
git clone https://github.com/yourusername/faster-gigaam.git
cd faster-gigaam
pip install -e .

# Or with UV
uv pip install -e .
```

## Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import gigaam
print(f"GigaAM installed: ✓")

from faster_gigaam import FastGigaAM
print(f"faster-gigaam installed: ✓")
```

## Troubleshooting

### CUDA not available
- Make sure you installed the CUDA version of PyTorch
- Check your NVIDIA driver version: `nvidia-smi`
- CUDA 12.8 requires driver >= 525.60.13

### GigaAM installation fails
- Make sure you have git installed
- Try: `pip install --upgrade pip setuptools wheel`
- Then retry GigaAM installation

### Import errors
- Make sure you're in the faster-gigaam directory when running `pip install -e .`
- Try: `pip install -e . --force-reinstall`

## System Requirements

**Minimum:**
- **Python**: 3.10 or higher (tested with Python 3.13)
- **CPU**: Multi-core processor
- **RAM**: 8GB
- **GPU (optional)**: NVIDIA GPU with CUDA 12.8 support
- **VRAM**: 4GB for small models

**Recommended (for best performance):**
- **CPU**: Intel i7/i9 or AMD Ryzen 7/9
- **RAM**: 16GB+
- **GPU**: NVIDIA RTX 3070 or better
- **VRAM**: 8GB+ (10GB for large models)

**Tested Configuration:**
- CPU: Intel Core i9-9900K
- RAM: 32GB DDR4
- GPU: NVIDIA RTX 3080 (10GB VRAM)
- CUDA: 12.8
- OS: Windows

## Using UV Package Manager (Optional)

[UV](https://github.com/astral-sh/uv) is a fast Python package manager that can significantly speed up installations:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Use UV for all pip commands
uv pip install <package>  # Much faster than pip!
```
