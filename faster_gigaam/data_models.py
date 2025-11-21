"""Core data models for faster-gigaam.

This module defines the data structures used throughout the faster-gigaam
pipeline for representing audio chunks, transcription segments, and
transcription metadata.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Segment:
    """Represents a transcribed segment with timing information.
    
    Attributes:
        id: Unique identifier for the segment
        start: Start time in seconds relative to original audio
        end: End time in seconds relative to original audio
        text: Transcribed text content
    """
    id: int
    start: float
    end: float
    text: str


@dataclass
class AudioChunk:
    """Represents a chunk of audio with metadata.
    
    Used for processing long audio files by splitting them into
    manageable chunks with overlap for continuity.
    
    Attributes:
        audio: Audio samples as numpy array
        start_time: Start time in seconds relative to original audio
        end_time: End time in seconds relative to original audio
        chunk_index: Index in the sequence of chunks (0-based)
    """
    audio: np.ndarray
    start_time: float
    end_time: float
    chunk_index: int


@dataclass
class TranscriptionInfo:
    """Metadata about the transcription process.
    
    Contains information about how the audio was processed and
    performance metrics.
    
    Attributes:
        duration: Total audio duration in seconds
        num_chunks: Number of chunks the audio was split into
        batch_size: Batch size used for processing
        device: Device used for inference ("cpu" or "cuda")
        compute_type: Precision used ("float16" or "float32")
        processing_time: Total wall-clock time for processing in seconds
    """
    duration: float
    num_chunks: int
    batch_size: int
    device: str
    compute_type: str
    processing_time: float
