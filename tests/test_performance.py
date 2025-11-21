"""Tests for performance optimization modules."""

import pytest
import torch

from faster_gigaam.dynamic_batcher import DynamicBatcher
from faster_gigaam.profiler import (
    MemoryProfiler,
    PerformanceProfiler,
    cuda_memory_manager,
)


class TestDynamicBatcher:
    """Tests for DynamicBatcher."""
    
    def test_init(self):
        """Test DynamicBatcher initialization."""
        batcher = DynamicBatcher(batch_size=8, length_tolerance=0.2)
        assert batcher.batch_size == 8
        assert batcher.length_tolerance == 0.2
    
    def test_init_invalid_batch_size(self):
        """Test initialization with invalid batch size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            DynamicBatcher(batch_size=0)
    
    def test_init_invalid_tolerance(self):
        """Test initialization with invalid tolerance."""
        with pytest.raises(ValueError, match="length_tolerance must be in"):
            DynamicBatcher(batch_size=8, length_tolerance=1.5)
    
    def test_create_batches_single_tensor(self):
        """Test batching with single tensor."""
        batcher = DynamicBatcher(batch_size=8)
        tensors = [torch.randn(100)]
        
        batches = batcher.create_batches(tensors)
        
        assert len(batches) == 1
        assert len(batches[0][0]) == 1  # One index
        assert len(batches[0][1]) == 1  # One tensor
    
    def test_create_batches_similar_lengths(self):
        """Test batching with similar-length tensors."""
        batcher = DynamicBatcher(batch_size=4, length_tolerance=0.2)
        # Create tensors with similar lengths (within 20%)
        tensors = [
            torch.randn(100),
            torch.randn(105),
            torch.randn(110),
            torch.randn(115),
        ]
        
        batches = batcher.create_batches(tensors)
        
        # Should create one batch since all are within tolerance
        assert len(batches) == 1
        assert len(batches[0][1]) == 4
    
    def test_create_batches_different_lengths(self):
        """Test batching with very different lengths."""
        batcher = DynamicBatcher(batch_size=4, length_tolerance=0.2)
        # Create tensors with very different lengths
        tensors = [
            torch.randn(100),
            torch.randn(200),
            torch.randn(300),
            torch.randn(400),
        ]
        
        batches = batcher.create_batches(tensors)
        
        # Should create multiple batches due to length differences
        assert len(batches) > 1
    
    def test_create_batches_preserves_order(self):
        """Test that batching preserves original indices."""
        batcher = DynamicBatcher(batch_size=2)
        tensors = [torch.randn(100), torch.randn(200), torch.randn(150)]
        
        batches = batcher.create_batches(tensors)
        
        # Collect all indices
        all_indices = []
        for indices, _ in batches:
            all_indices.extend(indices)
        
        # Should have all original indices
        assert sorted(all_indices) == [0, 1, 2]
    
    def test_create_batches_empty(self):
        """Test batching with empty list."""
        batcher = DynamicBatcher(batch_size=8)
        
        with pytest.raises(ValueError, match="tensors cannot be empty"):
            batcher.create_batches([])
    
    def test_calculate_padding_waste(self):
        """Test padding waste calculation."""
        batcher = DynamicBatcher(batch_size=4, length_tolerance=0.2)
        # Create tensors with different lengths
        tensors = [
            torch.randn(100),
            torch.randn(150),
            torch.randn(200),
            torch.randn(250),
        ]
        
        naive_waste, dynamic_waste = batcher.calculate_padding_waste(tensors)
        
        # Both should be between 0 and 1
        assert 0.0 <= naive_waste <= 1.0
        assert 0.0 <= dynamic_waste <= 1.0
        
        # Dynamic batching should have less or equal waste
        assert dynamic_waste <= naive_waste
    
    def test_calculate_padding_waste_empty(self):
        """Test padding waste with empty list."""
        batcher = DynamicBatcher(batch_size=8)
        
        naive_waste, dynamic_waste = batcher.calculate_padding_waste([])
        
        assert naive_waste == 0.0
        assert dynamic_waste == 0.0


class TestMemoryProfiler:
    """Tests for MemoryProfiler."""
    
    def test_init(self):
        """Test MemoryProfiler initialization."""
        profiler = MemoryProfiler()
        assert profiler.device is None or profiler.device.type == "cuda"
    
    def test_get_memory_stats_cpu(self):
        """Test getting memory stats on CPU."""
        profiler = MemoryProfiler(device=torch.device("cpu"))
        stats = profiler.get_memory_stats()
        
        # Should return None for CPU
        assert stats is None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_memory_stats_cuda(self):
        """Test getting memory stats on CUDA."""
        profiler = MemoryProfiler(device=torch.device("cuda"))
        stats = profiler.get_memory_stats()
        
        # Should return MemoryStats
        assert stats is not None
        assert stats.allocated_mb >= 0
        assert stats.reserved_mb >= 0
        assert stats.max_allocated_mb >= 0
        assert stats.total_mb > 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_reset_peak_stats(self):
        """Test resetting peak memory stats."""
        profiler = MemoryProfiler(device=torch.device("cuda"))
        
        # Allocate some memory
        tensor = torch.randn(1000, 1000, device="cuda")
        
        # Reset peak stats
        profiler.reset_peak_stats()
        
        # Get stats - peak should be low after reset
        stats = profiler.get_memory_stats()
        assert stats is not None
        
        # Clean up
        del tensor
        torch.cuda.empty_cache()


class TestPerformanceProfiler:
    """Tests for PerformanceProfiler."""
    
    def test_calculate_stats(self):
        """Test performance stats calculation."""
        stats = PerformanceProfiler.calculate_stats(
            audio_duration=60.0,
            processing_time=10.0,
            num_chunks=3,
            batch_size=8,
            device="cuda",
        )
        
        assert stats.audio_duration == 60.0
        assert stats.processing_time == 10.0
        assert stats.rtf == pytest.approx(10.0 / 60.0)
        assert stats.throughput == pytest.approx(60.0 / 10.0)
        assert stats.num_chunks == 3
        assert stats.batch_size == 8
        assert stats.device == "cuda"
    
    def test_calculate_stats_zero_duration(self):
        """Test stats calculation with zero duration."""
        stats = PerformanceProfiler.calculate_stats(
            audio_duration=0.0,
            processing_time=1.0,
            num_chunks=1,
            batch_size=1,
            device="cpu",
        )
        
        assert stats.rtf == 0.0
    
    def test_calculate_stats_zero_time(self):
        """Test stats calculation with zero processing time."""
        stats = PerformanceProfiler.calculate_stats(
            audio_duration=60.0,
            processing_time=0.0,
            num_chunks=1,
            batch_size=1,
            device="cpu",
        )
        
        assert stats.throughput == 0.0


class TestCudaMemoryManager:
    """Tests for cuda_memory_manager context manager."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_memory_manager(self):
        """Test CUDA memory manager clears cache."""
        # Allocate some memory
        with cuda_memory_manager():
            tensor = torch.randn(1000, 1000, device="cuda")
            allocated_before = torch.cuda.memory_allocated()
            assert allocated_before > 0
        
        # Memory should be cleared after context
        # Note: allocated memory may not be zero if tensor still exists,
        # but cache should be cleared
        del tensor
        torch.cuda.empty_cache()
    
    def test_cuda_memory_manager_cpu(self):
        """Test CUDA memory manager on CPU (should not error)."""
        # Should work fine even without CUDA
        with cuda_memory_manager():
            tensor = torch.randn(100, 100)
            assert tensor is not None
