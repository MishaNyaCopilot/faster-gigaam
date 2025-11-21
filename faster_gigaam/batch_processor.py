"""Batch processing for parallel inference.

This module provides functionality to process multiple audio segments
simultaneously using batching to maximize GPU utilization and improve
throughput.
"""

from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from gigaam.model import GigaAMASR


class BatchProcessor:
    """Manages batched inference for multiple audio segments.
    
    The BatchProcessor handles padding of variable-length audio segments
    to enable parallel processing on GPU, and ensures outputs are returned
    in the correct order.
    
    Attributes:
        model: GigaAM model instance for inference
        batch_size: Maximum number of segments to process simultaneously
    """
    
    def __init__(
        self,
        model: GigaAMASR,
        batch_size: int = 1,
    ):
        """Initialize batch processor.
        
        Args:
            model: GigaAM model instance
            batch_size: Maximum batch size (default: 1)
            
        Raises:
            ValueError: If batch_size is not a positive integer
            TypeError: If batch_size is not an integer
        """
        if not isinstance(batch_size, int):
            raise TypeError(
                f"batch_size must be int, got {type(batch_size).__name__}"
            )
        if batch_size < 1:
            raise ValueError(
                f"batch_size must be positive integer, got {batch_size}"
            )
        
        self.model = model
        self.batch_size = batch_size
        self._device = next(model.parameters()).device
    
    @torch.inference_mode()
    def process_batch(
        self,
        audio_chunks: List[torch.Tensor],
        language: str = "ru",
        beam_size: int = 1,
        temperature: float = 0.0,
    ) -> List[str]:
        """Process multiple audio chunks in parallel.
        
        Takes a list of audio tensors, pads them to the same length,
        processes them through the model in batches, and returns
        transcriptions in the same order as the input.
        
        Args:
            audio_chunks: List of audio tensors (1D, variable length)
            language: Language code (default: "ru")
            beam_size: Beam width for beam search (default: 1, greedy decoding)
            temperature: Temperature for sampling (default: 0.0, greedy decoding)
            
        Returns:
            List of transcriptions for each chunk in input order
            
        Raises:
            ValueError: If audio_chunks is empty or contains invalid tensors
            RuntimeError: If GPU out of memory
        
        Note:
            language, beam_size, and temperature parameters are currently not used
            by the underlying GigaAM decoder, which only supports greedy decoding.
            These parameters are accepted for API compatibility and future extensibility.
        """
        if not audio_chunks:
            raise ValueError("audio_chunks cannot be empty")
        
        # Validate all chunks are 1D tensors
        for i, chunk in enumerate(audio_chunks):
            if not isinstance(chunk, torch.Tensor):
                raise ValueError(
                    f"audio_chunks[{i}] must be torch.Tensor, "
                    f"got {type(chunk).__name__}"
                )
            if chunk.ndim != 1:
                raise ValueError(
                    f"audio_chunks[{i}] must be 1-dimensional, "
                    f"got shape {chunk.shape}"
                )
            if len(chunk) == 0:
                raise ValueError(f"audio_chunks[{i}] cannot be empty")
        
        all_transcriptions = []
        
        # Process in batches
        for batch_start in range(0, len(audio_chunks), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(audio_chunks))
            batch = audio_chunks[batch_start:batch_end]
            
            try:
                # Pad batch and get lengths
                padded_batch, lengths = self._pad_batch(batch)
                
                # Move to model device
                padded_batch = padded_batch.to(self._device)
                lengths = lengths.to(self._device)
                
                # Use autocast for GPU inference with mixed precision
                if self._device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        # Forward pass through model
                        encoded, encoded_len = self.model.forward(padded_batch, lengths)
                else:
                    # CPU inference without autocast
                    encoded, encoded_len = self.model.forward(padded_batch, lengths)
                
                # Decode to text
                transcriptions = self.model.decoding.decode(
                    self.model.head, encoded, encoded_len
                )
                
                all_transcriptions.extend(transcriptions)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Get current GPU memory usage if available
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated(self._device) / 1024**3
                        reserved = torch.cuda.memory_reserved(self._device) / 1024**3
                        total = torch.cuda.get_device_properties(self._device).total_memory / 1024**3
                        
                        # Clear cache and raise informative error
                        torch.cuda.empty_cache()
                        
                        raise RuntimeError(
                            f"GPU out of memory (allocated: {allocated:.2f}GB, "
                            f"reserved: {reserved:.2f}GB, total: {total:.2f}GB). "
                            f"Try reducing batch_size from {self.batch_size} to "
                            f"{max(1, self.batch_size // 2)}"
                        ) from e
                    else:
                        raise RuntimeError(
                            f"Out of memory. Try reducing batch_size from "
                            f"{self.batch_size} to {max(1, self.batch_size // 2)}"
                        ) from e
                raise
        
        # Clear GPU cache after processing all batches
        if self._device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return all_transcriptions
    
    def _pad_batch(
        self,
        tensors: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Pad tensors to same length for batching.
        
        Pads all tensors in the list to match the length of the longest
        tensor, enabling them to be stacked into a single batch tensor.
        
        Args:
            tensors: List of 1D tensors with variable lengths
            
        Returns:
            padded_tensor: Batched tensor with shape [batch, max_length]
            lengths: Original lengths of each tensor as 1D tensor
            
        Raises:
            ValueError: If tensors list is empty
        """
        if not tensors:
            raise ValueError("tensors cannot be empty")
        
        # Get original lengths
        lengths = torch.tensor([len(t) for t in tensors], dtype=torch.long)
        
        # Pad sequences to same length
        # pad_sequence expects list of tensors and returns [max_len, batch]
        # We transpose to get [batch, max_len]
        padded = pad_sequence(tensors, batch_first=True, padding_value=0.0)
        
        return padded, lengths
