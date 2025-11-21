"""Memory and performance profiling utilities.

This module provides tools for profiling memory usage and performance
characteristics of the faster-gigaam pipeline.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MemoryStats:
    """Memory usage statistics.
    
    Attributes:
        allocated_mb: Currently allocated memory in MB
        reserved_mb: Reserved memory in MB
        max_allocated_mb: Peak allocated memory in MB
        total_mb: Total available memory in MB
    """
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    total_mb: float
    
    def __str__(self) -> str:
        return (
            f"Memory: {self.allocated_mb:.1f}MB allocated, "
            f"{self.reserved_mb:.1f}MB reserved, "
            f"{self.max_allocated_mb:.1f}MB peak, "
            f"{self.total_mb:.1f}MB total"
        )


@dataclass
class PerformanceStats:
    """Performance statistics for transcription.
    
    Attributes:
        audio_duration: Total audio duration in seconds
        processing_time: Wall-clock processing time in seconds
        rtf: Real-time factor (processing_time / audio_duration)
        throughput: Audio seconds processed per wall-clock second
        num_chunks: Number of chunks processed
        batch_size: Batch size used
        device: Device used for processing
    """
    audio_duration: float
    processing_time: float
    rtf: float
    throughput: float
    num_chunks: int
    batch_size: int
    device: str
    
    def __str__(self) -> str:
        return (
            f"Performance: {self.audio_duration:.1f}s audio in {self.processing_time:.2f}s "
            f"(RTF: {self.rtf:.3f}, throughput: {self.throughput:.1f}x, "
            f"chunks: {self.num_chunks}, batch_size: {self.batch_size}, device: {self.device})"
        )


class MemoryProfiler:
    """Profiles GPU memory usage during inference.
    
    Tracks memory allocation, reservation, and peak usage on CUDA devices.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize memory profiler.
        
        Args:
            device: Device to profile (default: current CUDA device)
        """
        self.device = device
        if self.device is None and torch.cuda.is_available():
            self.device = torch.device("cuda")
    
    def get_memory_stats(self) -> Optional[MemoryStats]:
        """Get current memory statistics.
        
        Returns:
            MemoryStats object if CUDA is available, None otherwise
        """
        if self.device is None or self.device.type != "cuda":
            return None
        
        if not torch.cuda.is_available():
            return None
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**2
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**2
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**2
        
        return MemoryStats(
            allocated_mb=allocated,
            reserved_mb=reserved,
            max_allocated_mb=max_allocated,
            total_mb=total,
        )
    
    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if self.device is not None and self.device.type == "cuda":
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
    
    @contextmanager
    def profile(self):
        """Context manager for profiling memory usage.
        
        Yields:
            MemoryStats object with peak memory usage after context exits
            
        Example:
            >>> profiler = MemoryProfiler()
            >>> with profiler.profile() as stats:
            ...     result = model.transcribe("audio.wav")
            >>> print(stats)
        """
        self.reset_peak_stats()
        
        try:
            yield
        finally:
            pass
        
        # Get stats after execution
        stats = self.get_memory_stats()
        return stats


class PerformanceProfiler:
    """Profiles transcription performance.
    
    Tracks timing, throughput, and real-time factor for transcription tasks.
    """
    
    @staticmethod
    def calculate_stats(
        audio_duration: float,
        processing_time: float,
        num_chunks: int,
        batch_size: int,
        device: str,
    ) -> PerformanceStats:
        """Calculate performance statistics.
        
        Args:
            audio_duration: Total audio duration in seconds
            processing_time: Wall-clock processing time in seconds
            num_chunks: Number of chunks processed
            batch_size: Batch size used
            device: Device used for processing
            
        Returns:
            PerformanceStats object with calculated metrics
        """
        rtf = processing_time / audio_duration if audio_duration > 0 else 0.0
        throughput = audio_duration / processing_time if processing_time > 0 else 0.0
        
        return PerformanceStats(
            audio_duration=audio_duration,
            processing_time=processing_time,
            rtf=rtf,
            throughput=throughput,
            num_chunks=num_chunks,
            batch_size=batch_size,
            device=device,
        )
    
    @staticmethod
    @contextmanager
    def profile_transcription(
        audio_duration: float,
        num_chunks: int,
        batch_size: int,
        device: str,
    ):
        """Context manager for profiling transcription.
        
        Args:
            audio_duration: Total audio duration in seconds
            num_chunks: Number of chunks to process
            batch_size: Batch size being used
            device: Device being used
            
        Yields:
            PerformanceStats object after transcription completes
            
        Example:
            >>> with PerformanceProfiler.profile_transcription(10.0, 1, 8, "cuda") as stats:
            ...     result = model.transcribe("audio.wav")
            >>> print(stats)
        """
        start_time = time.time()
        
        try:
            yield
        finally:
            pass
        
        processing_time = time.time() - start_time
        
        stats = PerformanceProfiler.calculate_stats(
            audio_duration=audio_duration,
            processing_time=processing_time,
            num_chunks=num_chunks,
            batch_size=batch_size,
            device=device,
        )
        
        return stats


@contextmanager
def cuda_memory_manager():
    """Context manager for CUDA memory management.
    
    Ensures GPU memory is cleared after operations complete.
    
    Example:
        >>> with cuda_memory_manager():
        ...     result = process_large_batch(data)
    """
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
