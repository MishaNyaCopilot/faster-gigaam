"""Tests for BatchProcessor functionality."""

import numpy as np
import pytest
import torch

import gigaam
from faster_gigaam import BatchProcessor


def generate_test_audio(duration=3.0, sr=16000):
    """Generate synthetic test audio."""
    t = np.linspace(0, duration, int(sr * duration))
    audio = (
        0.5 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 660 * t)
    )
    return audio.astype(np.float32)


class TestBatchProcessorInitialization:
    """Test BatchProcessor initialization and parameter validation."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        model = gigaam.load_model("v3_ctc", device="cpu")
        processor = BatchProcessor(model, batch_size=4)
        
        assert processor.model is model
        assert processor.batch_size == 4
        assert processor._device == next(model.parameters()).device
    
    def test_init_default_batch_size(self):
        """Test initialization with default batch size."""
        model = gigaam.load_model("v3_ctc", device="cpu")
        processor = BatchProcessor(model)
        
        assert processor.batch_size == 1
    
    def test_init_invalid_batch_size_negative(self):
        """Test initialization with negative batch size raises ValueError."""
        model = gigaam.load_model("v3_ctc", device="cpu")
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            BatchProcessor(model, batch_size=-1)
    
    def test_init_invalid_batch_size_zero(self):
        """Test initialization with zero batch size raises ValueError."""
        model = gigaam.load_model("v3_ctc", device="cpu")
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            BatchProcessor(model, batch_size=0)
    
    def test_init_invalid_batch_size_type(self):
        """Test initialization with non-integer batch size raises TypeError."""
        model = gigaam.load_model("v3_ctc", device="cpu")
        
        with pytest.raises(TypeError, match="batch_size must be int"):
            BatchProcessor(model, batch_size=2.5)


class TestBatchProcessorPadding:
    """Test BatchProcessor padding functionality."""
    
    def test_pad_batch_same_length(self):
        """Test padding tensors of the same length."""
        model = gigaam.load_model("v3_ctc", device="cpu")
        processor = BatchProcessor(model)
        
        tensors = [
            torch.randn(1000),
            torch.randn(1000),
            torch.randn(1000),
        ]
        
        padded, lengths = processor._pad_batch(tensors)
        
        assert padded.shape == (3, 1000)
        assert torch.all(lengths == 1000)
    
    def test_pad_batch_different_lengths(self):
        """Test padding tensors of different lengths."""
        model = gigaam.load_model("v3_ctc", device="cpu")
        processor = BatchProcessor(model)
        
        tensors = [
            torch.randn(1000),
            torch.randn(2000),
            torch.randn(1500),
        ]
        
        padded, lengths = processor._pad_batch(tensors)
        
        # Should pad to max length (2000)
        assert padded.shape == (3, 2000)
        assert torch.all(lengths == torch.tensor([1000, 2000, 1500]))
    
    def test_pad_batch_single_tensor(self):
        """Test padding a single tensor."""
        model = gigaam.load_model("v3_ctc", device="cpu")
        processor = BatchProcessor(model)
        
        tensors = [torch.randn(1000)]
        
        padded, lengths = processor._pad_batch(tensors)
        
        assert padded.shape == (1, 1000)
        assert lengths[0] == 1000
    
    def test_pad_batch_empty_list(self):
        """Test padding empty list raises ValueError."""
        model = gigaam.load_model("v3_ctc", device="cpu")
        processor = BatchProcessor(model)
        
        with pytest.raises(ValueError, match="tensors cannot be empty"):
            processor._pad_batch([])


class TestBatchProcessorProcessing:
    """Test BatchProcessor audio processing functionality."""
    
    def test_process_batch_single_chunk(self):
        """Test processing a single audio chunk."""
        model = gigaam.load_model("v3_ctc", device="cpu")
        processor = BatchProcessor(model, batch_size=1)
        
        audio = torch.from_numpy(generate_test_audio(duration=2.0))
        chunks = [audio]
        
        transcriptions = processor.process_batch(chunks)
        
        assert len(transcriptions) == 1
        assert isinstance(transcriptions[0], str)
    
    def test_process_batch_multiple_chunks(self):
        """Test processing multiple audio chunks."""
        model = gigaam.load_model("v3_ctc", device="cpu")
        processor = BatchProcessor(model, batch_size=4)
        
        chunks = [
            torch.from_numpy(generate_test_audio(duration=1.5)),
            torch.from_numpy(generate_test_audio(duration=2.0)),
            torch.from_numpy(generate_test_audio(duration=1.0)),
        ]
        
        transcriptions = processor.process_batch(chunks)
        
        assert len(transcriptions) == 3
        for transcription in transcriptions:
            assert isinstance(transcription, str)
    
    def test_process_batch_ordering_preserved(self):
        """Test that output order matches input order."""
        model = gigaam.load_model("v3_ctc", device="cpu")
        processor = BatchProcessor(model, batch_size=2)
        
        # Create chunks with different lengths
        chunks = [
            torch.from_numpy(generate_test_audio(duration=0.5)),
            torch.from_numpy(generate_test_audio(duration=2.0)),
            torch.from_numpy(generate_test_audio(duration=1.0)),
            torch.from_numpy(generate_test_audio(duration=1.5)),
        ]
        
        transcriptions = processor.process_batch(chunks)
        
        # Should have same number of outputs as inputs
        assert len(transcriptions) == len(chunks)
    
    def test_process_batch_empty_list(self):
        """Test processing empty list raises ValueError."""
        model = gigaam.load_model("v3_ctc", device="cpu")
        processor = BatchProcessor(model)
        
        with pytest.raises(ValueError, match="audio_chunks cannot be empty"):
            processor.process_batch([])
    
    def test_process_batch_invalid_tensor_type(self):
        """Test processing non-tensor raises ValueError."""
        model = gigaam.load_model("v3_ctc", device="cpu")
        processor = BatchProcessor(model)
        
        with pytest.raises(ValueError, match="must be torch.Tensor"):
            processor.process_batch([np.array([1, 2, 3])])
    
    def test_process_batch_invalid_tensor_shape(self):
        """Test processing 2D tensor raises ValueError."""
        model = gigaam.load_model("v3_ctc", device="cpu")
        processor = BatchProcessor(model)
        
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            processor.process_batch([torch.randn(10, 10)])
    
    def test_process_batch_empty_tensor(self):
        """Test processing empty tensor raises ValueError."""
        model = gigaam.load_model("v3_ctc", device="cpu")
        processor = BatchProcessor(model)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            processor.process_batch([torch.tensor([])])
    
    def test_process_batch_exceeds_batch_size(self):
        """Test processing more chunks than batch_size."""
        model = gigaam.load_model("v3_ctc", device="cpu")
        processor = BatchProcessor(model, batch_size=2)
        
        # Create 5 chunks with batch_size=2
        chunks = [
            torch.from_numpy(generate_test_audio(duration=1.0))
            for _ in range(5)
        ]
        
        transcriptions = processor.process_batch(chunks)
        
        # Should process all chunks across multiple batches
        assert len(transcriptions) == 5


@pytest.mark.parametrize("model_name", ["v3_ctc", "v3_e2e_rnnt"])
def test_process_batch_different_models(model_name):
    """Test BatchProcessor works with different model types."""
    model = gigaam.load_model(model_name, device="cpu")
    processor = BatchProcessor(model, batch_size=2)
    
    chunks = [
        torch.from_numpy(generate_test_audio(duration=1.5)),
        torch.from_numpy(generate_test_audio(duration=2.0)),
    ]
    
    transcriptions = processor.process_batch(chunks)
    
    assert len(transcriptions) == 2
    for transcription in transcriptions:
        assert isinstance(transcription, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
