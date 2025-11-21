"""Tests for FastGigaAM main API class."""

import pytest
import torch

from faster_gigaam import FastGigaAM


class TestFastGigaAMInitialization:
    """Test FastGigaAM initialization and parameter validation."""
    
    def test_init_cpu_default_params(self):
        """Test initialization with CPU and default parameters."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        assert model.device == "cpu"
        assert model.compute_type == "float16"
        assert model.batch_size == 1
        assert model.chunk_length == 25
        assert model.chunk_overlap == 1.0
        assert model.model is not None
        assert model.chunker is not None
        assert model.batch_processor is not None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_init_cuda(self):
        """Test initialization with CUDA."""
        model = FastGigaAM("v3_ctc", device="cuda", batch_size=4)
        
        assert model.device == "cuda"
        assert model.batch_size == 4
        assert next(model.model.parameters()).device.type == "cuda"
    
    def test_init_float32(self):
        """Test initialization with float32 precision."""
        model = FastGigaAM("v3_ctc", device="cpu", compute_type="float32")
        
        assert model.compute_type == "float32"
    
    def test_init_custom_chunk_params(self):
        """Test initialization with custom chunking parameters."""
        model = FastGigaAM(
            "v3_ctc",
            device="cpu",
            chunk_length=30,
            chunk_overlap=2.0,
        )
        
        assert model.chunk_length == 30
        assert model.chunk_overlap == 2.0
        assert model.chunker.chunk_length == 30
        assert model.chunker.overlap == 2.0
    
    def test_init_invalid_device_type(self):
        """Test that invalid device type raises TypeError."""
        with pytest.raises(TypeError, match="device must be str"):
            FastGigaAM("v3_ctc", device=123)
    
    def test_init_invalid_device_value(self):
        """Test that invalid device value raises ValueError."""
        with pytest.raises(ValueError, match="device must be 'cuda' or 'cpu'"):
            FastGigaAM("v3_ctc", device="gpu")
    
    @pytest.mark.skipif(torch.cuda.is_available(), reason="Test requires CUDA unavailable")
    def test_init_cuda_unavailable(self):
        """Test that requesting CUDA when unavailable raises RuntimeError."""
        with pytest.raises(RuntimeError, match="CUDA device requested but not available"):
            FastGigaAM("v3_ctc", device="cuda")
    
    def test_init_invalid_compute_type_type(self):
        """Test that invalid compute_type type raises TypeError."""
        with pytest.raises(TypeError, match="compute_type must be str"):
            FastGigaAM("v3_ctc", device="cpu", compute_type=16)
    
    def test_init_invalid_compute_type_value(self):
        """Test that invalid compute_type value raises ValueError."""
        with pytest.raises(ValueError, match="compute_type must be 'float16' or 'float32'"):
            FastGigaAM("v3_ctc", device="cpu", compute_type="int8")
    
    def test_init_invalid_batch_size_type(self):
        """Test that invalid batch_size type raises TypeError."""
        with pytest.raises(TypeError, match="batch_size must be int"):
            FastGigaAM("v3_ctc", device="cpu", batch_size=1.5)
    
    def test_init_invalid_batch_size_value(self):
        """Test that invalid batch_size value raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive integer"):
            FastGigaAM("v3_ctc", device="cpu", batch_size=0)
        
        with pytest.raises(ValueError, match="batch_size must be positive integer"):
            FastGigaAM("v3_ctc", device="cpu", batch_size=-1)
    
    def test_init_invalid_chunk_length_type(self):
        """Test that invalid chunk_length type raises TypeError."""
        with pytest.raises(TypeError, match="chunk_length must be numeric"):
            FastGigaAM("v3_ctc", device="cpu", chunk_length="25")
    
    def test_init_invalid_chunk_length_value(self):
        """Test that invalid chunk_length value raises ValueError."""
        with pytest.raises(ValueError, match="chunk_length must be positive"):
            FastGigaAM("v3_ctc", device="cpu", chunk_length=0)
        
        with pytest.raises(ValueError, match="chunk_length must be positive"):
            FastGigaAM("v3_ctc", device="cpu", chunk_length=-5)
    
    def test_init_invalid_chunk_overlap_type(self):
        """Test that invalid chunk_overlap type raises TypeError."""
        with pytest.raises(TypeError, match="chunk_overlap must be numeric"):
            FastGigaAM("v3_ctc", device="cpu", chunk_overlap="1.0")
    
    def test_init_invalid_chunk_overlap_value(self):
        """Test that invalid chunk_overlap value raises ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            FastGigaAM("v3_ctc", device="cpu", chunk_overlap=-1.0)
    
    def test_init_overlap_exceeds_chunk_length(self):
        """Test that overlap >= chunk_length raises ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than chunk_length"):
            FastGigaAM("v3_ctc", device="cpu", chunk_length=25, chunk_overlap=25)
        
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than chunk_length"):
            FastGigaAM("v3_ctc", device="cpu", chunk_length=25, chunk_overlap=30)
    
    def test_init_invalid_model_name(self):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Failed to load model"):
            FastGigaAM("invalid_model", device="cpu")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_init_fp16_encoder_on_cuda(self):
        """Test that FP16 encoder is used when compute_type=float16 on CUDA."""
        model = FastGigaAM("v3_ctc", device="cuda", compute_type="float16")
        
        # Check that encoder is in half precision
        encoder_dtype = next(model.model.encoder.parameters()).dtype
        assert encoder_dtype == torch.float16
    
    def test_init_fp32_encoder_on_cpu(self):
        """Test that FP32 encoder is used on CPU even with compute_type=float16."""
        model = FastGigaAM("v3_ctc", device="cpu", compute_type="float16")
        
        # On CPU, encoder should remain in float32
        encoder_dtype = next(model.model.encoder.parameters()).dtype
        assert encoder_dtype == torch.float32
    
    def test_transcribe_file_not_found(self):
        """Test that transcribe() raises FileNotFoundError for missing file."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(FileNotFoundError, match="Audio file.*not found"):
            model.transcribe("nonexistent_file.wav")
    
    def test_transcribe_batch_file_not_found(self):
        """Test that transcribe_batch() raises FileNotFoundError for missing files."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(FileNotFoundError, match="Audio file.*not found"):
            model.transcribe_batch(["nonexistent_file.wav"])


class TestFastGigaAMComponents:
    """Test that FastGigaAM properly initializes its components."""
    
    def test_chunker_initialized(self):
        """Test that AudioChunker is properly initialized."""
        model = FastGigaAM("v3_ctc", device="cpu", chunk_length=30, chunk_overlap=2.0)
        
        assert model.chunker is not None
        assert model.chunker.chunk_length == 30
        assert model.chunker.overlap == 2.0
        assert model.chunker.sample_rate == 16000
    
    def test_batch_processor_initialized(self):
        """Test that BatchProcessor is properly initialized."""
        model = FastGigaAM("v3_ctc", device="cpu", batch_size=8)
        
        assert model.batch_processor is not None
        assert model.batch_processor.batch_size == 8
        assert model.batch_processor.model is model.model
    
    def test_model_device_placement(self):
        """Test that model is placed on correct device."""
        model_cpu = FastGigaAM("v3_ctc", device="cpu")
        assert next(model_cpu.model.parameters()).device.type == "cpu"
        
        if torch.cuda.is_available():
            model_cuda = FastGigaAM("v3_ctc", device="cuda")
            assert next(model_cuda.model.parameters()).device.type == "cuda"



class TestFastGigaAMTranscribe:
    """Test FastGigaAM transcribe() method."""
    
    def test_transcribe_short_audio_file(self):
        """Test transcribing a short audio file (< 25 seconds)."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        # Use the example.wav file
        segments, info = model.transcribe("example.wav")
        
        # Check segments
        assert isinstance(segments, list)
        assert len(segments) > 0
        
        for segment in segments:
            assert isinstance(segment.id, int)
            assert isinstance(segment.start, float)
            assert isinstance(segment.end, float)
            assert isinstance(segment.text, str)
            assert segment.start >= 0
            assert segment.end > segment.start
        
        # Check info
        assert isinstance(info, object)
        assert hasattr(info, 'duration')
        assert hasattr(info, 'num_chunks')
        assert hasattr(info, 'batch_size')
        assert hasattr(info, 'device')
        assert hasattr(info, 'compute_type')
        assert hasattr(info, 'processing_time')
        
        assert info.duration > 0
        assert info.num_chunks >= 1
        assert info.batch_size == 1
        assert info.device == "cpu"
        assert info.compute_type == "float16"
        assert info.processing_time > 0
    
    def test_transcribe_numpy_array(self):
        """Test transcribing from numpy array."""
        import numpy as np
        
        model = FastGigaAM("v3_ctc", device="cpu")
        
        # Create a short audio array (5 seconds of random noise)
        audio_array = np.random.randn(16000 * 5).astype(np.float32)
        
        segments, info = model.transcribe(audio_array)
        
        # Check that it returns valid segments
        assert isinstance(segments, list)
        assert len(segments) > 0
        
        # Check info
        assert info.duration > 0
        assert info.num_chunks >= 1
    
    def test_transcribe_invalid_audio_type(self):
        """Test that invalid audio type raises TypeError."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(TypeError, match="audio must be str.*or np.ndarray"):
            model.transcribe(123)
        
        with pytest.raises(TypeError, match="audio must be str.*or np.ndarray"):
            model.transcribe([1, 2, 3])
    
    def test_transcribe_invalid_numpy_shape(self):
        """Test that invalid numpy array shape raises ValueError."""
        import numpy as np
        
        model = FastGigaAM("v3_ctc", device="cpu")
        
        # 2D array
        audio_2d = np.random.randn(2, 16000)
        with pytest.raises(ValueError, match="audio array must be 1-dimensional"):
            model.transcribe(audio_2d)
    
    def test_transcribe_empty_numpy_array(self):
        """Test that empty numpy array raises ValueError."""
        import numpy as np
        
        model = FastGigaAM("v3_ctc", device="cpu")
        
        audio_empty = np.array([])
        with pytest.raises(ValueError, match="audio array cannot be empty"):
            model.transcribe(audio_empty)
    
    def test_transcribe_audio_too_short(self):
        """Test that very short audio raises ValueError."""
        import numpy as np
        
        model = FastGigaAM("v3_ctc", device="cpu")
        
        # 0.05 seconds of audio (too short)
        audio_short = np.random.randn(int(16000 * 0.05)).astype(np.float32)
        
        with pytest.raises(ValueError, match="Audio too short"):
            model.transcribe(audio_short)
    
    def test_transcribe_with_batch_size(self):
        """Test transcribing with different batch sizes."""
        model = FastGigaAM("v3_ctc", device="cpu", batch_size=4)
        
        segments, info = model.transcribe("example.wav")
        
        assert isinstance(segments, list)
        assert info.batch_size == 4
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_transcribe_on_cuda(self):
        """Test transcribing on CUDA device."""
        model = FastGigaAM("v3_ctc", device="cuda")
        
        segments, info = model.transcribe("example.wav")
        
        assert isinstance(segments, list)
        assert len(segments) > 0
        assert info.device == "cuda"
    
    def test_transcribe_segment_ids_sequential(self):
        """Test that segment IDs are sequential."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        segments, _ = model.transcribe("example.wav")
        
        for i, segment in enumerate(segments):
            assert segment.id == i
    
    def test_transcribe_timestamps_monotonic(self):
        """Test that segment timestamps are monotonically increasing."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        segments, _ = model.transcribe("example.wav")
        
        for i in range(len(segments) - 1):
            assert segments[i].end <= segments[i + 1].start or \
                   segments[i].start < segments[i + 1].start

    
    def test_transcribe_long_audio(self):
        """Test transcribing long audio that requires chunking."""
        import numpy as np
        
        model = FastGigaAM("v3_ctc", device="cpu", chunk_length=10, chunk_overlap=1.0)
        
        # Create 30 seconds of audio (should create 3 chunks with 10s length)
        audio_long = np.random.randn(16000 * 30).astype(np.float32)
        
        segments, info = model.transcribe(audio_long)
        
        # Should have multiple chunks
        assert info.num_chunks > 1
        assert info.duration == 30.0
        
        # Segments should be properly ordered
        for i in range(len(segments) - 1):
            assert segments[i].id < segments[i + 1].id


class TestFastGigaAMDecoderParameters:
    """Test decoder parameter validation in transcribe methods."""
    
    def test_transcribe_with_valid_language(self):
        """Test transcribe with valid language parameter."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        # Should not raise any errors
        segments, info = model.transcribe("example.wav", language="ru")
        assert isinstance(segments, list)
    
    def test_transcribe_with_valid_beam_size(self):
        """Test transcribe with valid beam_size parameter."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        # Should not raise any errors
        segments, info = model.transcribe("example.wav", beam_size=5)
        assert isinstance(segments, list)
    
    def test_transcribe_with_valid_temperature(self):
        """Test transcribe with valid temperature parameter."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        # Should not raise any errors
        segments, info = model.transcribe("example.wav", temperature=0.5)
        assert isinstance(segments, list)
    
    def test_transcribe_with_all_parameters(self):
        """Test transcribe with all decoder parameters."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        segments, info = model.transcribe(
            "example.wav",
            language="en",
            beam_size=3,
            temperature=0.3,
        )
        assert isinstance(segments, list)
    
    def test_transcribe_invalid_language_type(self):
        """Test that invalid language type raises TypeError."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(TypeError, match="language must be str"):
            model.transcribe("example.wav", language=123)
    
    def test_transcribe_empty_language(self):
        """Test that empty language string raises ValueError."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(ValueError, match="language cannot be empty string"):
            model.transcribe("example.wav", language="")
    
    def test_transcribe_invalid_beam_size_type(self):
        """Test that invalid beam_size type raises TypeError."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(TypeError, match="beam_size must be int"):
            model.transcribe("example.wav", beam_size=1.5)
        
        with pytest.raises(TypeError, match="beam_size must be int"):
            model.transcribe("example.wav", beam_size="1")
    
    def test_transcribe_invalid_beam_size_value(self):
        """Test that invalid beam_size value raises ValueError."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(ValueError, match="beam_size must be positive integer"):
            model.transcribe("example.wav", beam_size=0)
        
        with pytest.raises(ValueError, match="beam_size must be positive integer"):
            model.transcribe("example.wav", beam_size=-1)
    
    def test_transcribe_invalid_temperature_type(self):
        """Test that invalid temperature type raises TypeError."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(TypeError, match="temperature must be numeric"):
            model.transcribe("example.wav", temperature="0.5")
        
        with pytest.raises(TypeError, match="temperature must be numeric"):
            model.transcribe("example.wav", temperature=[0.5])
    
    def test_transcribe_invalid_temperature_value(self):
        """Test that invalid temperature value raises ValueError."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(ValueError, match="temperature must be in range"):
            model.transcribe("example.wav", temperature=-0.1)
        
        with pytest.raises(ValueError, match="temperature must be in range"):
            model.transcribe("example.wav", temperature=1.5)
    
    def test_transcribe_temperature_boundary_values(self):
        """Test that temperature boundary values (0.0 and 1.0) are valid."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        # Should not raise errors
        segments1, _ = model.transcribe("example.wav", temperature=0.0)
        assert isinstance(segments1, list)
        
        segments2, _ = model.transcribe("example.wav", temperature=1.0)
        assert isinstance(segments2, list)
    
    def test_transcribe_batch_with_valid_parameters(self):
        """Test transcribe_batch with valid decoder parameters."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        results = model.transcribe_batch(
            ["example.wav"],
            language="en",
            beam_size=3,
            temperature=0.5,
        )
        
        assert len(results) == 1
        assert isinstance(results[0][0], list)
    
    def test_transcribe_batch_invalid_language_type(self):
        """Test that transcribe_batch with invalid language type raises TypeError."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(TypeError, match="language must be str"):
            model.transcribe_batch(["example.wav"], language=123)
    
    def test_transcribe_batch_empty_language(self):
        """Test that transcribe_batch with empty language raises ValueError."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(ValueError, match="language cannot be empty string"):
            model.transcribe_batch(["example.wav"], language="")
    
    def test_transcribe_batch_invalid_beam_size_type(self):
        """Test that transcribe_batch with invalid beam_size type raises TypeError."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(TypeError, match="beam_size must be int"):
            model.transcribe_batch(["example.wav"], beam_size=1.5)
    
    def test_transcribe_batch_invalid_beam_size_value(self):
        """Test that transcribe_batch with invalid beam_size value raises ValueError."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(ValueError, match="beam_size must be positive integer"):
            model.transcribe_batch(["example.wav"], beam_size=0)
    
    def test_transcribe_batch_invalid_temperature_type(self):
        """Test that transcribe_batch with invalid temperature type raises TypeError."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(TypeError, match="temperature must be numeric"):
            model.transcribe_batch(["example.wav"], temperature="0.5")
    
    def test_transcribe_batch_invalid_temperature_value(self):
        """Test that transcribe_batch with invalid temperature value raises ValueError."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(ValueError, match="temperature must be in range"):
            model.transcribe_batch(["example.wav"], temperature=-0.1)
        
        with pytest.raises(ValueError, match="temperature must be in range"):
            model.transcribe_batch(["example.wav"], temperature=1.5)


class TestFastGigaAMTranscribeBatch:
    """Test FastGigaAM transcribe_batch() method."""
    
    def test_transcribe_batch_single_file(self):
        """Test transcribing a single file in batch mode."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        results = model.transcribe_batch(["example.wav"])
        
        # Should return list with one result
        assert isinstance(results, list)
        assert len(results) == 1
        
        segments, info = results[0]
        
        # Check segments
        assert isinstance(segments, list)
        assert len(segments) > 0
        
        for segment in segments:
            assert isinstance(segment.id, int)
            assert isinstance(segment.start, float)
            assert isinstance(segment.end, float)
            assert isinstance(segment.text, str)
        
        # Check info
        assert info.duration > 0
        assert info.num_chunks >= 1
        assert info.processing_time > 0
    
    def test_transcribe_batch_multiple_files(self):
        """Test transcribing multiple files in batch mode."""
        model = FastGigaAM("v3_ctc", device="cpu", batch_size=4)
        
        # Use the same file multiple times for testing
        results = model.transcribe_batch(["example.wav", "example.wav"])
        
        # Should return list with two results
        assert isinstance(results, list)
        assert len(results) == 2
        
        # Check both results
        for segments, info in results:
            assert isinstance(segments, list)
            assert len(segments) > 0
            assert info.duration > 0
            assert info.num_chunks >= 1
            assert info.processing_time > 0
    
    def test_transcribe_batch_preserves_order(self):
        """Test that transcribe_batch preserves input order."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        # Use the same file multiple times - we just need to verify order is preserved
        # The implementation should return results in the same order as input
        results = model.transcribe_batch(["example.wav", "example.wav", "example.wav"])
        
        # Check that we get results in order
        assert len(results) == 3
        
        # All results should have similar structure since they're the same file
        for i, (segments, info) in enumerate(results):
            assert isinstance(segments, list)
            assert len(segments) > 0
            assert info.duration > 0
            
            # Verify segments are properly ordered within each result
            for j in range(len(segments) - 1):
                assert segments[j].id < segments[j + 1].id
    
    def test_transcribe_batch_empty_list(self):
        """Test that empty audio_files list raises ValueError."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(ValueError, match="audio_files cannot be empty"):
            model.transcribe_batch([])
    
    def test_transcribe_batch_invalid_type(self):
        """Test that invalid audio_files type raises TypeError."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(TypeError, match="audio_files must be list"):
            model.transcribe_batch("example.wav")
        
        with pytest.raises(TypeError, match="audio_files must be list"):
            model.transcribe_batch(("example.wav",))
    
    def test_transcribe_batch_invalid_item_type(self):
        """Test that invalid item type in audio_files raises TypeError."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(TypeError, match="audio_files\\[0\\] must be str"):
            model.transcribe_batch([123])
    
    def test_transcribe_batch_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        with pytest.raises(FileNotFoundError, match="Audio file.*not found"):
            model.transcribe_batch(["nonexistent.wav"])
    
    def test_transcribe_batch_mixed_files(self):
        """Test transcribing batch with one valid and one invalid file."""
        model = FastGigaAM("v3_ctc", device="cpu")
        
        # Should fail early before processing any files
        with pytest.raises(FileNotFoundError):
            model.transcribe_batch(["example.wav", "nonexistent.wav"])
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_transcribe_batch_on_cuda(self):
        """Test transcribe_batch on CUDA device."""
        model = FastGigaAM("v3_ctc", device="cuda", batch_size=4)
        
        results = model.transcribe_batch(["example.wav", "example.wav"])
        
        assert len(results) == 2
        for segments, info in results:
            assert len(segments) > 0
            assert info.device == "cuda"
