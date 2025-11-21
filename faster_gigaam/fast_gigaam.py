"""Main API class for faster-gigaam.

This module provides the FastGigaAM class, which is the primary interface
for using faster-gigaam. It orchestrates model loading, parameter validation,
and the transcription pipeline.
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

import gigaam
from gigaam.model import GigaAMASR

from .batch_processor import BatchProcessor
from .chunker import AudioChunker
from .data_models import Segment, TranscriptionInfo

logger = logging.getLogger(__name__)


class FastGigaAM:
    """Main interface for faster-gigaam functionality.
    
    FastGigaAM provides an optimized inference engine for GigaAM ASR models
    with CUDA acceleration, batch processing, and long audio support.
    
    Example:
        >>> model = FastGigaAM("v3_ctc", device="cuda", batch_size=8)
        >>> segments, info = model.transcribe("audio.wav")
        >>> for segment in segments:
        ...     print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
    
    Attributes:
        model: Loaded GigaAM model instance
        device: Device being used for inference
        compute_type: Precision type being used
        batch_size: Number of chunks to process simultaneously
        chunk_length: Length of each chunk in seconds
        chunk_overlap: Overlap between chunks in seconds
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 1,
        chunk_length: int = 25,
        chunk_overlap: float = 1.0,
        download_root: Optional[str] = None,
    ):
        """Initialize faster-gigaam model.
        
        Args:
            model_name: GigaAM model version (e.g., "v3_ctc", "v3_e2e_rnnt")
            device: Device to run on ("cuda" or "cpu")
            compute_type: Precision ("float16" or "float32")
            batch_size: Number of chunks to process simultaneously
            chunk_length: Length of each chunk in seconds
            chunk_overlap: Overlap between chunks in seconds
            download_root: Optional directory for model downloads
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If CUDA is requested but not available
        """
        # Validate device parameter
        if not isinstance(device, str):
            raise TypeError(
                f"device must be str, got {type(device).__name__}"
            )
        if device not in ["cuda", "cpu"]:
            raise ValueError(
                f"device must be 'cuda' or 'cpu', got '{device}'"
            )
        
        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but not available. "
                "Install CUDA toolkit or use device='cpu'"
            )
        
        # Validate compute_type parameter
        if not isinstance(compute_type, str):
            raise TypeError(
                f"compute_type must be str, got {type(compute_type).__name__}"
            )
        if compute_type not in ["float16", "float32"]:
            raise ValueError(
                f"compute_type must be 'float16' or 'float32', got '{compute_type}'"
            )
        
        # Validate batch_size parameter
        if not isinstance(batch_size, int):
            raise TypeError(
                f"batch_size must be int, got {type(batch_size).__name__}"
            )
        if batch_size < 1:
            raise ValueError(
                f"batch_size must be positive integer, got {batch_size}"
            )
        
        # Validate chunk_length parameter
        if not isinstance(chunk_length, (int, float)):
            raise TypeError(
                f"chunk_length must be numeric, got {type(chunk_length).__name__}"
            )
        if chunk_length <= 0:
            raise ValueError(
                f"chunk_length must be positive, got {chunk_length}"
            )
        
        # Validate chunk_overlap parameter
        if not isinstance(chunk_overlap, (int, float)):
            raise TypeError(
                f"chunk_overlap must be numeric, got {type(chunk_overlap).__name__}"
            )
        if chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be non-negative, got {chunk_overlap}"
            )
        if chunk_overlap >= chunk_length:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}s) must be less than "
                f"chunk_length ({chunk_length}s)"
            )
        
        # Store parameters
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.chunk_length = chunk_length
        self.chunk_overlap = chunk_overlap
        
        # Load model
        logger.info(f"Loading model '{model_name}' on device '{device}'")
        
        # Determine if we should use FP16 for encoder
        fp16_encoder = (compute_type == "float16" and device == "cuda")
        
        try:
            self.model = gigaam.load_model(
                model_name=model_name,
                fp16_encoder=fp16_encoder,
                use_flash=False,
                device=device,
                download_root=download_root,
            )
        except ValueError as e:
            # Re-raise with more helpful message
            raise ValueError(
                f"Failed to load model '{model_name}'. {str(e)}"
            ) from e
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model '{model_name}' not found. Run gigaam.load_model('{model_name}') "
                f"to download, or check the model name. {str(e)}"
            ) from e
        
        # Verify model is ASR model
        if not isinstance(self.model, GigaAMASR):
            raise ValueError(
                f"Model '{model_name}' is not an ASR model. "
                f"faster-gigaam only supports ASR models (CTC/RNNT)."
            )
        
        # Initialize components
        self.chunker = AudioChunker(
            chunk_length=chunk_length,
            overlap=chunk_overlap,
            sample_rate=16000,
        )
        
        self.batch_processor = BatchProcessor(
            model=self.model,
            batch_size=batch_size,
        )
        
        logger.info(
            f"FastGigaAM initialized: device={device}, "
            f"compute_type={compute_type}, batch_size={batch_size}"
        )
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: str = "ru",
        beam_size: int = 1,
        temperature: float = 0.0,
    ) -> Tuple[List[Segment], TranscriptionInfo]:
        """Transcribe audio file or array.
        
        Args:
            audio: Audio file path or numpy array
            language: Language code (default: "ru")
            beam_size: Beam width for beam search (default: 1, greedy decoding)
            temperature: Temperature for sampling (default: 0.0, greedy decoding)
        
        Returns:
            segments: List of transcribed segments with timestamps
            info: Transcription metadata
            
        Raises:
            FileNotFoundError: If audio file is not found
            ValueError: If audio format is invalid or parameters are invalid
            TypeError: If parameters have invalid types
        
        Note:
            beam_size and temperature parameters are validated but not yet used
            by the underlying GigaAM decoder, which currently only supports
            greedy decoding. These parameters are included for API compatibility
            and future extensibility.
        """
        import time
        import os
        
        # Validate language parameter
        if not isinstance(language, str):
            raise TypeError(
                f"language must be str, got {type(language).__name__}"
            )
        if not language:
            raise ValueError("language cannot be empty string")
        
        # Validate beam_size parameter
        if not isinstance(beam_size, int):
            raise TypeError(
                f"beam_size must be int, got {type(beam_size).__name__}"
            )
        if beam_size < 1:
            raise ValueError(
                f"beam_size must be positive integer, got {beam_size}"
            )
        
        # Validate temperature parameter
        if not isinstance(temperature, (int, float)):
            raise TypeError(
                f"temperature must be numeric, got {type(temperature).__name__}"
            )
        if temperature < 0.0 or temperature > 1.0:
            raise ValueError(
                f"temperature must be in range [0.0, 1.0], got {temperature}"
            )
        
        start_time = time.time()
        
        # Load audio
        if isinstance(audio, str):
            # Audio is a file path
            if not os.path.exists(audio):
                raise FileNotFoundError(
                    f"Audio file '{audio}' not found. Check file path and permissions"
                )
            
            try:
                # Use gigaam's load_audio function
                audio_tensor = gigaam.load_audio(audio, sample_rate=16000)
            except Exception as e:
                raise ValueError(
                    f"Failed to load audio file '{audio}'. "
                    f"Check that the file is a valid audio format. Error: {str(e)}"
                ) from e
            
            # Convert to numpy for chunking
            audio_array = audio_tensor.cpu().numpy()
        
        elif isinstance(audio, np.ndarray):
            # Audio is already a numpy array
            if audio.ndim != 1:
                raise ValueError(
                    f"audio array must be 1-dimensional, got shape {audio.shape}"
                )
            if len(audio) == 0:
                raise ValueError("audio array cannot be empty")
            
            audio_array = audio
        
        else:
            raise TypeError(
                f"audio must be str (file path) or np.ndarray, "
                f"got {type(audio).__name__}"
            )
        
        # Calculate audio duration
        audio_duration = len(audio_array) / 16000.0
        
        # Validate audio is not too short
        if audio_duration < 0.1:
            raise ValueError(
                f"Audio too short ({audio_duration:.2f}s). "
                f"Minimum duration is 0.1 seconds"
            )
        
        # Chunk the audio
        chunks = self.chunker.chunk_audio(audio_array)
        
        # Convert chunks to tensors and prepare for batch processing
        chunk_tensors = []
        for chunk in chunks:
            chunk_tensor = torch.from_numpy(chunk.audio).float()
            chunk_tensors.append(chunk_tensor)
        
        # Process chunks in batches
        transcriptions = self.batch_processor.process_batch(
            chunk_tensors,
            language=language,
            beam_size=beam_size,
            temperature=temperature,
        )
        
        # Create segments from transcriptions
        segments = []
        for i, (chunk, transcription) in enumerate(zip(chunks, transcriptions)):
            segment = Segment(
                id=i,
                start=chunk.start_time,
                end=chunk.end_time,
                text=transcription,
            )
            segments.append(segment)
        
        # Merge segments if there were multiple chunks
        if len(chunks) > 1:
            segments = self.chunker.merge_segments(segments, chunks)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create transcription info
        info = TranscriptionInfo(
            duration=audio_duration,
            num_chunks=len(chunks),
            batch_size=self.batch_size,
            device=self.device,
            compute_type=self.compute_type,
            processing_time=processing_time,
        )
        
        return segments, info
    
    def transcribe_batch(
        self,
        audio_files: List[str],
        language: str = "ru",
        beam_size: int = 1,
        temperature: float = 0.0,
    ) -> List[Tuple[List[Segment], TranscriptionInfo]]:
        """Transcribe multiple audio files in batch.
        
        This method processes multiple audio files efficiently by loading
        them all and processing their chunks together in batches. This
        maximizes GPU utilization and improves throughput compared to
        processing files sequentially.
        
        Args:
            audio_files: List of audio file paths
            language: Language code (default: "ru")
            beam_size: Beam width for beam search (default: 1, greedy decoding)
            temperature: Temperature for sampling (default: 0.0, greedy decoding)
        
        Returns:
            List of (segments, info) tuples for each audio file, in the
            same order as the input audio_files list
            
        Raises:
            FileNotFoundError: If any audio file is not found
            ValueError: If audio format is invalid, audio_files is empty, or parameters are invalid
            TypeError: If audio_files is not a list or parameters have invalid types
        
        Note:
            beam_size and temperature parameters are validated but not yet used
            by the underlying GigaAM decoder, which currently only supports
            greedy decoding. These parameters are included for API compatibility
            and future extensibility.
        """
        import time
        import os
        
        # Validate language parameter
        if not isinstance(language, str):
            raise TypeError(
                f"language must be str, got {type(language).__name__}"
            )
        if not language:
            raise ValueError("language cannot be empty string")
        
        # Validate beam_size parameter
        if not isinstance(beam_size, int):
            raise TypeError(
                f"beam_size must be int, got {type(beam_size).__name__}"
            )
        if beam_size < 1:
            raise ValueError(
                f"beam_size must be positive integer, got {beam_size}"
            )
        
        # Validate temperature parameter
        if not isinstance(temperature, (int, float)):
            raise TypeError(
                f"temperature must be numeric, got {type(temperature).__name__}"
            )
        if temperature < 0.0 or temperature > 1.0:
            raise ValueError(
                f"temperature must be in range [0.0, 1.0], got {temperature}"
            )
        
        # Validate input
        if not isinstance(audio_files, list):
            raise TypeError(
                f"audio_files must be list, got {type(audio_files).__name__}"
            )
        
        if not audio_files:
            raise ValueError("audio_files cannot be empty")
        
        # Validate all files exist before processing
        for i, audio_file in enumerate(audio_files):
            if not isinstance(audio_file, str):
                raise TypeError(
                    f"audio_files[{i}] must be str, got {type(audio_file).__name__}"
                )
            if not os.path.exists(audio_file):
                raise FileNotFoundError(
                    f"Audio file '{audio_file}' not found. Check file path and permissions"
                )
        
        start_time = time.time()
        
        # Load all audio files
        audio_arrays = []
        audio_durations = []
        
        for audio_file in audio_files:
            try:
                # Use gigaam's load_audio function
                audio_tensor = gigaam.load_audio(audio_file, sample_rate=16000)
            except Exception as e:
                raise ValueError(
                    f"Failed to load audio file '{audio_file}'. "
                    f"Check that the file is a valid audio format. Error: {str(e)}"
                ) from e
            
            # Convert to numpy for chunking
            audio_array = audio_tensor.cpu().numpy()
            
            # Validate audio
            if len(audio_array) == 0:
                raise ValueError(f"Audio file '{audio_file}' is empty")
            
            audio_duration = len(audio_array) / 16000.0
            
            if audio_duration < 0.1:
                raise ValueError(
                    f"Audio file '{audio_file}' too short ({audio_duration:.2f}s). "
                    f"Minimum duration is 0.1 seconds"
                )
            
            audio_arrays.append(audio_array)
            audio_durations.append(audio_duration)
        
        # Chunk all audio files
        all_chunks = []
        file_chunk_mapping = []  # Track which chunks belong to which file
        
        for file_idx, audio_array in enumerate(audio_arrays):
            chunks = self.chunker.chunk_audio(audio_array)
            
            # Store mapping of chunk indices to file index
            chunk_start_idx = len(all_chunks)
            all_chunks.extend(chunks)
            chunk_end_idx = len(all_chunks)
            
            file_chunk_mapping.append((chunk_start_idx, chunk_end_idx, chunks))
        
        # Convert all chunks to tensors
        chunk_tensors = []
        for chunk in all_chunks:
            chunk_tensor = torch.from_numpy(chunk.audio).float()
            chunk_tensors.append(chunk_tensor)
        
        # Process all chunks in batches
        all_transcriptions = self.batch_processor.process_batch(
            chunk_tensors,
            language=language,
            beam_size=beam_size,
            temperature=temperature,
        )
        
        # Assemble results for each file
        results = []
        
        for file_idx, (chunk_start, chunk_end, original_chunks) in enumerate(file_chunk_mapping):
            # Get transcriptions for this file's chunks
            file_transcriptions = all_transcriptions[chunk_start:chunk_end]
            
            # Create segments from transcriptions
            segments = []
            for i, (chunk, transcription) in enumerate(zip(original_chunks, file_transcriptions)):
                segment = Segment(
                    id=i,
                    start=chunk.start_time,
                    end=chunk.end_time,
                    text=transcription,
                )
                segments.append(segment)
            
            # Merge segments if there were multiple chunks
            if len(original_chunks) > 1:
                segments = self.chunker.merge_segments(segments, original_chunks)
            
            # Create transcription info for this file
            info = TranscriptionInfo(
                duration=audio_durations[file_idx],
                num_chunks=len(original_chunks),
                batch_size=self.batch_size,
                device=self.device,
                compute_type=self.compute_type,
                processing_time=0.0,  # Will be updated after all files are processed
            )
            
            results.append((segments, info))
        
        # Calculate total processing time
        total_processing_time = time.time() - start_time
        
        # Update processing time for all results
        # Distribute time proportionally based on duration
        total_duration = sum(audio_durations)
        for i, (segments, info) in enumerate(results):
            # Proportional time allocation
            file_proportion = audio_durations[i] / total_duration if total_duration > 0 else 1.0 / len(results)
            info.processing_time = total_processing_time * file_proportion
        
        return results
