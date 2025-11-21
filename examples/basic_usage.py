"""Basic usage example for faster-gigaam.

This example demonstrates:
1. Simple transcription of audio files
2. Handling both short and long audio
3. GPU vs CPU comparison
4. Understanding the output format

Requirements: 4.1, 4.3
"""

import time
import torch
from faster_gigaam import FastGigaAM

# =============================================================================
# Example 1: Basic Transcription
# =============================================================================
print("=" * 70)
print("Example 1: Basic Transcription")
print("=" * 70)

# Initialize the model
# - model_name: GigaAM model version (e.g., "v3_ctc", "v3_e2e_rnnt")
# - device: "cuda" for GPU acceleration, "cpu" for CPU-only
# - batch_size: Number of chunks to process simultaneously
model = FastGigaAM(
    model_name="v3_e2e_rnnt",  # or "v3_ctc" for CTC model
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=8,
)

# Transcribe an audio file
# Replace with your actual audio file path
audio_path = "audio.wav"  # Your audio file here

try:
    segments, info = model.transcribe(audio_path)
    
    # Print transcription metadata
    print(f"\nAudio duration: {info.duration:.2f}s")
    print(f"Processing time: {info.processing_time:.2f}s")
    print(f"Real-time factor: {info.processing_time / info.duration:.2f}x")
    print(f"Processed in {info.num_chunks} chunk(s)")
    print(f"Device: {info.device}")
    print(f"Compute type: {info.compute_type}")
    print()
    
    # Print each segment with timestamps
    print("Transcription:")
    print("-" * 70)
    for segment in segments:
        print(f"[{segment.start:6.2f}s - {segment.end:6.2f}s] {segment.text}")
    
except FileNotFoundError:
    print(f"Audio file '{audio_path}' not found. Please provide a valid audio file.")
except Exception as e:
    print(f"Error during transcription: {e}")

# =============================================================================
# Example 2: Short vs Long Audio Handling
# =============================================================================
print("\n" + "=" * 70)
print("Example 2: Short vs Long Audio Handling")
print("=" * 70)

# Short audio (< 25 seconds) - processed in a single chunk
print("\nShort audio (< 25 seconds):")
print("-" * 70)
print("For audio shorter than 25 seconds, faster-gigaam processes it")
print("in a single chunk without splitting.")
print()
print("Example:")
print("  audio_duration = 15.3s")
print("  num_chunks = 1")
print("  Output: Single segment with full transcription")

# Long audio (> 25 seconds) - automatically chunked
print("\nLong audio (> 25 seconds):")
print("-" * 70)
print("For longer audio, faster-gigaam automatically:")
print("  1. Splits audio into 25-second chunks with 1-second overlap")
print("  2. Processes chunks in batches (batch_size=8)")
print("  3. Merges overlapping segments for seamless output")
print()
print("Example:")
print("  audio_duration = 180.5s")
print("  num_chunks = 8")
print("  Output: Multiple segments with continuous timestamps")

# You can customize chunking behavior:
model_custom = FastGigaAM(
    model_name="v3_e2e_rnnt",
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=8,
    chunk_length=30,      # Use 30-second chunks instead of 25
    chunk_overlap=2.0,    # Use 2-second overlap instead of 1
)
print("\nCustom chunking parameters:")
print(f"  chunk_length = {model_custom.chunk_length}s")
print(f"  chunk_overlap = {model_custom.chunk_overlap}s")

# =============================================================================
# Example 3: GPU vs CPU Comparison
# =============================================================================
print("\n" + "=" * 70)
print("Example 3: GPU vs CPU Comparison")
print("=" * 70)

# Check if CUDA is available
if not torch.cuda.is_available():
    print("\nCUDA not available. Skipping GPU comparison.")
    print("To use GPU acceleration:")
    print("  1. Install CUDA toolkit")
    print("  2. Install PyTorch with CUDA support")
    print("  3. Ensure you have a compatible NVIDIA GPU")
else:
    print("\nComparing CPU vs GPU performance...")
    print("-" * 70)
    
    # Use a test audio file (replace with your file)
    test_audio = "audio.wav"
    
    try:
        # CPU inference
        print("\n[1/2] Running on CPU...")
        cpu_model = FastGigaAM(
            model_name="v3_ctc",  # Using CTC for faster comparison
            device="cpu",
            batch_size=1,  # CPU typically uses smaller batches
        )
        
        cpu_start = time.time()
        cpu_segments, cpu_info = cpu_model.transcribe(test_audio)
        cpu_time = time.time() - cpu_start
        
        print(f"  Duration: {cpu_info.duration:.2f}s")
        print(f"  Processing time: {cpu_time:.2f}s")
        print(f"  Real-time factor: {cpu_time / cpu_info.duration:.2f}x")
        
        # GPU inference
        print("\n[2/2] Running on GPU...")
        gpu_model = FastGigaAM(
            model_name="v3_ctc",
            device="cuda",
            batch_size=8,  # GPU can handle larger batches
            compute_type="float16",  # FP16 for faster inference
        )
        
        gpu_start = time.time()
        gpu_segments, gpu_info = gpu_model.transcribe(test_audio)
        gpu_time = time.time() - gpu_start
        
        print(f"  Duration: {gpu_info.duration:.2f}s")
        print(f"  Processing time: {gpu_time:.2f}s")
        print(f"  Real-time factor: {gpu_time / gpu_info.duration:.2f}x")
        
        # Comparison
        print("\n" + "-" * 70)
        print("Comparison:")
        print(f"  CPU time: {cpu_time:.2f}s")
        print(f"  GPU time: {gpu_time:.2f}s")
        print(f"  GPU speedup: {cpu_time / gpu_time:.2f}x faster")
        print()
        print("Key takeaways:")
        print("  - GPU is significantly faster for audio > 10 seconds")
        print("  - GPU can process larger batches (batch_size=8 vs 1)")
        print("  - GPU supports FP16 for additional speedup")
        print("  - For short audio (< 5s), CPU may be sufficient")
        
    except FileNotFoundError:
        print(f"\nAudio file '{test_audio}' not found.")
        print("Please provide a valid audio file to run the comparison.")
    except Exception as e:
        print(f"\nError during comparison: {e}")

# =============================================================================
# Example 4: Working with Numpy Arrays
# =============================================================================
print("\n" + "=" * 70)
print("Example 4: Working with Numpy Arrays")
print("=" * 70)

import numpy as np

# You can also transcribe from numpy arrays instead of files
# This is useful when you've already loaded audio or are processing
# audio from a non-file source (e.g., microphone, network stream)

print("\nTranscribing from numpy array...")
print("-" * 70)

# Generate sample audio (replace with your actual audio data)
# Audio should be 16kHz mono float32
sample_rate = 16000
duration_seconds = 10
audio_array = np.random.randn(sample_rate * duration_seconds).astype(np.float32)

print(f"Audio array shape: {audio_array.shape}")
print(f"Audio duration: {len(audio_array) / sample_rate:.2f}s")
print(f"Sample rate: {sample_rate}Hz")

try:
    segments, info = model.transcribe(audio_array)
    print(f"\nTranscribed {info.duration:.2f}s of audio in {info.processing_time:.2f}s")
    print(f"Number of segments: {len(segments)}")
except Exception as e:
    print(f"Error: {e}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
faster-gigaam provides a simple API for transcribing audio with GigaAM models:

1. Initialize model:
   model = FastGigaAM(model_name, device, batch_size)

2. Transcribe audio:
   segments, info = model.transcribe(audio_path)

3. Access results:
   - segments: List of Segment objects with start, end, text
   - info: TranscriptionInfo with duration, processing_time, etc.

Key features:
  ✓ Automatic handling of long audio (> 25 seconds)
  ✓ CUDA acceleration for faster processing
  ✓ Batch processing for improved throughput
  ✓ Support for both file paths and numpy arrays
  ✓ Detailed timing and metadata

For more examples, see:
  - long_audio.py: Processing very long audio files
  - batch_processing.py: Transcribing multiple files efficiently
  - advanced_usage.py: Error handling, memory management, etc.
""")
