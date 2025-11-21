"""faster-gigaam: Optimized inference for GigaAM ASR models.

This module provides CUDA-accelerated batch processing and long audio
support for GigaAM models.

Example:
    >>> from faster_gigaam import FastGigaAM
    >>> model = FastGigaAM("v3_ctc", device="cuda", batch_size=8)
    >>> segments, info = model.transcribe("audio.wav")
    >>> for segment in segments:
    ...     print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
"""

from .batch_processor import BatchProcessor
from .chunker import AudioChunker
from .data_models import AudioChunk, Segment, TranscriptionInfo
from .dynamic_batcher import DynamicBatcher
from .fast_gigaam import FastGigaAM
from .profiler import (
    MemoryProfiler,
    MemoryStats,
    PerformanceProfiler,
    PerformanceStats,
    cuda_memory_manager,
)

__version__ = "0.1.0"

__all__ = [
    "AudioChunk",
    "AudioChunker",
    "BatchProcessor",
    "DynamicBatcher",
    "FastGigaAM",
    "MemoryProfiler",
    "MemoryStats",
    "PerformanceProfiler",
    "PerformanceStats",
    "Segment",
    "TranscriptionInfo",
    "cuda_memory_manager",
]
