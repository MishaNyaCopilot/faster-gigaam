"""Audio chunking for long audio support.

This module provides functionality to split long audio files into
overlapping chunks for processing, and merge the resulting segments
back together while handling overlaps.
"""

from typing import List

import numpy as np

from .data_models import AudioChunk, Segment


class AudioChunker:
    """Handles splitting long audio into processable chunks.
    
    The AudioChunker splits audio longer than chunk_length into overlapping
    chunks to enable processing of arbitrarily long audio files. The overlap
    helps maintain continuity at chunk boundaries.
    
    Attributes:
        chunk_length: Duration of each chunk in seconds
        overlap: Overlap duration between consecutive chunks in seconds
        sample_rate: Audio sample rate in Hz
    """
    
    def __init__(
        self,
        chunk_length: int = 25,
        overlap: float = 1.0,
        sample_rate: int = 16000,
    ):
        """Initialize audio chunker.
        
        Args:
            chunk_length: Chunk duration in seconds (default: 25)
            overlap: Overlap duration in seconds (default: 1.0)
            sample_rate: Audio sample rate in Hz (default: 16000)
            
        Raises:
            ValueError: If chunk_length <= overlap or if values are non-positive
        """
        if chunk_length <= 0:
            raise ValueError(
                f"chunk_length must be positive, got {chunk_length}"
            )
        if overlap < 0:
            raise ValueError(
                f"overlap must be non-negative, got {overlap}"
            )
        if overlap >= chunk_length:
            raise ValueError(
                f"overlap ({overlap}s) must be less than chunk_length ({chunk_length}s)"
            )
        if sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be positive, got {sample_rate}"
            )
        
        self.chunk_length = chunk_length
        self.overlap = overlap
        self.sample_rate = sample_rate
        
    def chunk_audio(
        self,
        audio: np.ndarray,
    ) -> List[AudioChunk]:
        """Split audio into overlapping chunks.
        
        For audio shorter than chunk_length, returns a single chunk.
        For longer audio, creates overlapping chunks with the specified
        overlap duration.
        
        Args:
            audio: Audio samples as numpy array (1D)
            
        Returns:
            List of AudioChunk objects with audio data and metadata
            
        Raises:
            ValueError: If audio is empty or has invalid shape
        """
        if audio.ndim != 1:
            raise ValueError(
                f"audio must be 1-dimensional, got shape {audio.shape}"
            )
        if len(audio) == 0:
            raise ValueError("audio cannot be empty")
        
        audio_duration = len(audio) / self.sample_rate
        
        # If audio is shorter than or equal to chunk_length, return single chunk
        if audio_duration <= self.chunk_length:
            return [
                AudioChunk(
                    audio=audio,
                    start_time=0.0,
                    end_time=audio_duration,
                    chunk_index=0,
                )
            ]
        
        # Calculate chunk parameters
        chunk_samples = int(self.chunk_length * self.sample_rate)
        overlap_samples = int(self.overlap * self.sample_rate)
        stride_samples = chunk_samples - overlap_samples
        
        chunks = []
        chunk_index = 0
        start_sample = 0
        
        while start_sample < len(audio):
            end_sample = min(start_sample + chunk_samples, len(audio))
            
            chunk_audio = audio[start_sample:end_sample]
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            
            chunks.append(
                AudioChunk(
                    audio=chunk_audio,
                    start_time=start_time,
                    end_time=end_time,
                    chunk_index=chunk_index,
                )
            )
            
            chunk_index += 1
            start_sample += stride_samples
            
            # If we've reached the end, break
            if end_sample >= len(audio):
                break
        
        return chunks
    
    def merge_segments(
        self,
        segments: List[Segment],
        chunks: List[AudioChunk],
    ) -> List[Segment]:
        """Merge segments from overlapping chunks, handling duplicates.
        
        When chunks overlap, segments in the overlap region may be duplicated.
        This method deduplicates segments by keeping only segments that fall
        primarily within their source chunk's non-overlapping region.
        
        Args:
            segments: List of segments from all chunks (must be sorted by start time)
            chunks: List of chunks that produced the segments
            
        Returns:
            Deduplicated list of segments with continuous timestamps
            
        Raises:
            ValueError: If segments and chunks are inconsistent
        """
        if not segments:
            return []
        
        if not chunks:
            raise ValueError("chunks cannot be empty when segments exist")
        
        # If only one chunk, no merging needed
        if len(chunks) == 1:
            return segments
        
        # Sort segments by start time to ensure proper ordering
        sorted_segments = sorted(segments, key=lambda s: s.start)
        
        # Build overlap boundaries for each chunk
        # For each chunk, we keep segments that start before the overlap with next chunk
        merged = []
        used_indices = set()  # Track which segments we've already added
        
        for i, chunk in enumerate(chunks):
            # Determine the cutoff time for this chunk
            # Keep segments that start before the midpoint of overlap with next chunk
            if i < len(chunks) - 1:
                # There's a next chunk - calculate overlap midpoint
                next_chunk = chunks[i + 1]
                overlap_start = next_chunk.start_time
                overlap_end = chunk.end_time
                cutoff_time = (overlap_start + overlap_end) / 2.0
            else:
                # Last chunk - keep all remaining segments
                cutoff_time = float('inf')
            
            # Find segments from this chunk
            chunk_start = chunk.start_time
            
            for seg_idx, segment in enumerate(sorted_segments):
                # Skip if we've already added this segment
                if seg_idx in used_indices:
                    continue
                
                # Segment belongs to this chunk if it starts within chunk's range
                # and before the cutoff time
                if chunk_start <= segment.start < cutoff_time:
                    merged.append(segment)
                    used_indices.add(seg_idx)
        
        # Re-assign segment IDs to be sequential
        for idx, segment in enumerate(merged):
            segment.id = idx
        
        return merged

