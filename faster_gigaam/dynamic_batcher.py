"""Dynamic batching for similar-length chunks.

This module provides functionality to group audio chunks of similar length
together to minimize padding waste and improve GPU utilization.
"""

from typing import List, Tuple

import torch


class DynamicBatcher:
    """Groups audio chunks by similar length for efficient batching.
    
    Dynamic batching reduces padding waste by grouping chunks of similar
    length together. This improves GPU utilization and throughput compared
    to naive batching which may group very different length chunks.
    
    Attributes:
        batch_size: Maximum number of chunks per batch
        length_tolerance: Maximum length difference ratio within a batch (0.0-1.0)
    """
    
    def __init__(
        self,
        batch_size: int = 8,
        length_tolerance: float = 0.2,
    ):
        """Initialize dynamic batcher.
        
        Args:
            batch_size: Maximum batch size (default: 8)
            length_tolerance: Maximum length difference ratio (default: 0.2)
                A value of 0.2 means chunks can differ by up to 20% in length
                
        Raises:
            ValueError: If parameters are invalid
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not 0.0 <= length_tolerance <= 1.0:
            raise ValueError(
                f"length_tolerance must be in [0.0, 1.0], got {length_tolerance}"
            )
        
        self.batch_size = batch_size
        self.length_tolerance = length_tolerance
    
    def create_batches(
        self,
        tensors: List[torch.Tensor],
    ) -> List[Tuple[List[int], List[torch.Tensor]]]:
        """Group tensors into batches by similar length.
        
        Sorts tensors by length and groups similar-length tensors together
        to minimize padding waste. Returns batches with their original indices
        to enable reordering outputs.
        
        Args:
            tensors: List of 1D tensors with variable lengths
            
        Returns:
            List of (indices, batch_tensors) tuples where:
                - indices: Original indices of tensors in this batch
                - batch_tensors: Tensors for this batch
                
        Raises:
            ValueError: If tensors is empty
        """
        if not tensors:
            raise ValueError("tensors cannot be empty")
        
        # Create list of (index, length, tensor) tuples
        indexed_tensors = [
            (i, len(tensor), tensor)
            for i, tensor in enumerate(tensors)
        ]
        
        # Sort by length
        indexed_tensors.sort(key=lambda x: x[1])
        
        batches = []
        current_batch_indices = []
        current_batch_tensors = []
        current_batch_min_len = None
        current_batch_max_len = None
        
        for idx, length, tensor in indexed_tensors:
            # Check if this tensor fits in current batch
            if current_batch_min_len is None:
                # First tensor in batch
                current_batch_min_len = length
                current_batch_max_len = length
                current_batch_indices.append(idx)
                current_batch_tensors.append(tensor)
            else:
                # Check if adding this tensor would exceed tolerance
                new_max_len = max(current_batch_max_len, length)
                new_min_len = current_batch_min_len
                length_ratio = (new_max_len - new_min_len) / new_max_len if new_max_len > 0 else 0.0
                
                # Also check if batch is full
                batch_full = len(current_batch_tensors) >= self.batch_size
                tolerance_exceeded = length_ratio > self.length_tolerance
                
                if batch_full or tolerance_exceeded:
                    # Start new batch
                    batches.append((current_batch_indices, current_batch_tensors))
                    current_batch_indices = [idx]
                    current_batch_tensors = [tensor]
                    current_batch_min_len = length
                    current_batch_max_len = length
                else:
                    # Add to current batch
                    current_batch_indices.append(idx)
                    current_batch_tensors.append(tensor)
                    current_batch_max_len = new_max_len
        
        # Add final batch
        if current_batch_tensors:
            batches.append((current_batch_indices, current_batch_tensors))
        
        return batches
    
    def calculate_padding_waste(
        self,
        tensors: List[torch.Tensor],
    ) -> Tuple[float, float]:
        """Calculate padding waste for naive vs dynamic batching.
        
        Compares the amount of padding required for naive batching (grouping
        tensors sequentially) vs dynamic batching (grouping by length).
        
        Args:
            tensors: List of 1D tensors with variable lengths
            
        Returns:
            naive_waste: Padding waste ratio for naive batching (0.0-1.0)
            dynamic_waste: Padding waste ratio for dynamic batching (0.0-1.0)
            
        Note:
            Waste ratio is calculated as: (padded_elements - actual_elements) / padded_elements
        """
        if not tensors:
            return 0.0, 0.0
        
        total_elements = sum(len(t) for t in tensors)
        
        # Calculate naive batching waste
        naive_padded = 0
        for i in range(0, len(tensors), self.batch_size):
            batch = tensors[i:i + self.batch_size]
            max_len = max(len(t) for t in batch)
            naive_padded += max_len * len(batch)
        
        naive_waste = (naive_padded - total_elements) / naive_padded if naive_padded > 0 else 0.0
        
        # Calculate dynamic batching waste
        batches = self.create_batches(tensors)
        dynamic_padded = 0
        for _, batch_tensors in batches:
            max_len = max(len(t) for t in batch_tensors)
            dynamic_padded += max_len * len(batch_tensors)
        
        dynamic_waste = (dynamic_padded - total_elements) / dynamic_padded if dynamic_padded > 0 else 0.0
        
        return naive_waste, dynamic_waste
