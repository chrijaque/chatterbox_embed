import runpod
import time  
import torchaudio 
import os
import tempfile
import base64
import torch
import logging
from chatterbox.tts import ChatterboxTTS
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from google.cloud import storage
import numpy as np  # Added for MP3 conversion

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the handler from the forked repository
try:
    from chatterbox.vc import ChatterboxVC
    FORKED_HANDLER_AVAILABLE = True
    logger.info("✅ Successfully imported ChatterboxVC from forked repository")
except ImportError as e:
    FORKED_HANDLER_AVAILABLE = False
    logger.warning(f"⚠️ Could not import ChatterboxVC from forked repository: {e}")

# Try to import optional dependencies
try:
    import nltk
    from nltk.tokenize.punkt import PunktSentenceTokenizer
    
    # Force download and setup NLTK punkt tokenizer
    logger.info("🔧 Setting up NLTK punkt tokenizer...")
    try:
        # Download punkt tokenizer explicitly
        nltk.download('punkt', quiet=True)

        # Verify it's available
        nltk.data.find('tokenizers/punkt')
        NLTK_AVAILABLE = True
        logger.info("✅ NLTK punkt tokenizer downloaded and verified")

        # Test tokenization to ensure it works
        test_text = "This is a test. It has multiple sentences. Let's verify NLTK works."
        tokenizer = PunktSentenceTokenizer()
        test_sentences = tokenizer.tokenize(test_text)
        logger.info(f"✅ NLTK test successful: {len(test_sentences)} sentences tokenized")

    except Exception as e:
        logger.error(f"❌ NLTK setup failed: {e}")
        NLTK_AVAILABLE = False
        logger.warning("⚠️ Will use simple text splitting instead")
    
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("⚠️ nltk not available - will use simple text splitting")

try:
    from pydub import AudioSegment, effects
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("⚠️ pydub not available - will use torchaudio for audio processing")

# Pre-load ChatterboxTTS model at module level (avoids re-initialization)
logger.info("🔧 Pre-loading ChatterboxTTS model...")
try:
    model = ChatterboxTTS.from_pretrained(device='cuda')
    logger.info("✅ ChatterboxTTS model pre-loaded successfully")
    
    # Initialize the forked repository handler if available
    if FORKED_HANDLER_AVAILABLE:
        logger.info("🔧 Pre-loading ChatterboxVC...")
        try:
            forked_handler = ChatterboxVC(
                s3gen=model.s3gen,
                device=model.device
            )
            logger.info("✅ ChatterboxVC pre-loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to pre-load ChatterboxVC: {e}")
            forked_handler = None
    else:
        logger.warning("⚠️ ChatterboxVC not available - will use fallback methods")
        forked_handler = None
        
except Exception as e:
    logger.error(f"❌ Failed to pre-load ChatterboxTTS model: {e}")
model = None
forked_handler = None

# Local directory paths (use absolute paths for RunPod deployment)
VOICE_PROFILES_DIR = Path("/voice_profiles")
VOICE_SAMPLES_DIR = Path("/voice_samples")  # For voice clone samples
TTS_GENERATED_DIR = Path("/tts_generated")  # For TTS story generation
TEMP_VOICE_DIR = Path("/temp_voice")

# Log directory status (don't create them as they already exist in RunPod)
logger.info(f"Using existing directories:")
logger.info(f"  VOICE_PROFILES_DIR: {VOICE_PROFILES_DIR}")
logger.info(f"  VOICE_SAMPLES_DIR: {VOICE_SAMPLES_DIR}")
logger.info(f"  TTS_GENERATED_DIR: {TTS_GENERATED_DIR}")
logger.info(f"  TEMP_VOICE_DIR: {TEMP_VOICE_DIR}")

# Check if directories exist
logger.info(f"Directory existence check:")
logger.info(f"  VOICE_PROFILES_DIR exists: {VOICE_PROFILES_DIR.exists()}")
logger.info(f"  VOICE_SAMPLES_DIR exists: {VOICE_SAMPLES_DIR.exists()}")
logger.info(f"  TTS_GENERATED_DIR exists: {TTS_GENERATED_DIR.exists()}")
logger.info(f"  TEMP_VOICE_DIR exists: {TEMP_VOICE_DIR.exists()}")

# Initialize Firebase storage client
storage_client = None
bucket = None

# -------------------------------------------------------------------
# 🎵 MP3 Conversion Utilities
# -------------------------------------------------------------------
def tensor_to_mp3_bytes(audio_tensor, sample_rate, bitrate="96k"):
    """
    Convert audio tensor directly to MP3 bytes.
    
    :param audio_tensor: PyTorch audio tensor
    :param sample_rate: Audio sample rate
    :param bitrate: MP3 bitrate (e.g., "96k", "128k", "160k")
    :return: MP3 bytes
    """
    if PYDUB_AVAILABLE:
        try:
            # Convert tensor to AudioSegment
            audio_segment = tensor_to_audiosegment(audio_tensor, sample_rate)
            # Export to MP3 bytes
            mp3_file = audio_segment.export(format="mp3", bitrate=bitrate)
            # Read the bytes from the file object
            mp3_bytes = mp3_file.read()
            return mp3_bytes
        except Exception as e:
            logger.warning(f"Direct MP3 conversion failed: {e}, falling back to WAV")
            return tensor_to_wav_bytes(audio_tensor, sample_rate)
    else:
        logger.warning("pydub not available, falling back to WAV")
        return tensor_to_wav_bytes(audio_tensor, sample_rate)

def tensor_to_audiosegment(audio_tensor, sample_rate):
    """
    Convert PyTorch audio tensor to pydub AudioSegment.
    
    :param audio_tensor: PyTorch audio tensor
    :param sample_rate: Audio sample rate
    :return: pydub AudioSegment
    """
    if not PYDUB_AVAILABLE:
        raise ImportError("pydub is required for audio conversion")
    
    # Convert tensor to numpy array
    if audio_tensor.dim() == 2:
        # Stereo: (channels, samples)
        audio_np = audio_tensor.numpy()
    else:
        # Mono: (samples,) -> (1, samples)
        audio_np = audio_tensor.unsqueeze(0).numpy()
    
    # Convert to int16 for pydub
    audio_np = (audio_np * 32767).astype(np.int16)
    
    # Create AudioSegment
    audio_segment = AudioSegment(
        audio_np.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit
        channels=audio_np.shape[0]
    )
    
    return audio_segment

def tensor_to_wav_bytes(audio_tensor, sample_rate):
    """
    Convert audio tensor to WAV bytes (fallback).
    
    :param audio_tensor: PyTorch audio tensor
    :param sample_rate: Audio sample rate
    :return: WAV bytes
    """
    # Save to temporary WAV file
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(temp_wav.name, audio_tensor, sample_rate)
    
    # Read WAV bytes
    with open(temp_wav.name, 'rb') as f:
        wav_bytes = f.read()
    
    # Clean up temp file
    os.unlink(temp_wav.name)
    
    return wav_bytes

def convert_audio_file_to_mp3(input_path, output_path, bitrate="160k"):
    """
    Convert audio file to MP3 with specified bitrate.
    
    :param input_path: Path to input audio file
    :param output_path: Path to output MP3 file
    :param bitrate: MP3 bitrate
    """
    if not PYDUB_AVAILABLE:
        raise ImportError("pydub is required for audio conversion")
    
    try:
        # Load audio file
        audio = AudioSegment.from_file(input_path)
        # Export as MP3
        audio.export(output_path, format="mp3", bitrate=bitrate)
        logger.info(f"✅ Converted {input_path} to MP3: {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to convert {input_path} to MP3: {e}")
        raise

# -------------------------------------------------------------------
# 🐞  Firebase / GCS credential debug helper
# -------------------------------------------------------------------
def _debug_gcs_creds():
    """Minimal Firebase credential check - removed extensive debugging since voice cloning is working"""
    import os
    logger.info("🔍 Firebase credentials check")
    
    # Check if RunPod secret is available
    firebase_secret_path = os.getenv('RUNPOD_SECRET_Firebase')
    if firebase_secret_path:
        if firebase_secret_path.startswith('{'):
            logger.info("✅ Using RunPod Firebase secret (JSON content)")
        else:
            logger.info("✅ Using RunPod Firebase secret (file path)")
    else:
        logger.warning("⚠️ No RunPod Firebase secret found")

def initialize_firebase():
    """Initialize Firebase storage client"""
    global storage_client, bucket
    
    # Firebase initialization
    
    try:
        # Check if we're in RunPod and have the secret
        firebase_secret_path = os.getenv('RUNPOD_SECRET_Firebase')
        
        if firebase_secret_path:
            if firebase_secret_path.startswith('{'):
                # It's JSON content, create a temporary file
                logger.info("✅ Using RunPod Firebase secret as JSON content")
                import tempfile
                import json
                
                # Validate JSON first
                try:
                    creds_data = json.loads(firebase_secret_path)
                    logger.info(f"✅ Valid JSON with project_id: {creds_data.get('project_id', 'unknown')}")
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Invalid JSON in RUNPOD_SECRET_Firebase: {e}")
                    raise
                
                # Create temporary file with the JSON content
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                    json.dump(creds_data, tmp_file)
                    tmp_path = tmp_file.name
                
                logger.info(f"✅ Created temporary credentials file: {tmp_path}")
                storage_client = storage.Client.from_service_account_json(tmp_path)
                
            elif os.path.exists(firebase_secret_path):
                # It's a file path
                logger.info(f"✅ Using RunPod Firebase secret file: {firebase_secret_path}")
                storage_client = storage.Client.from_service_account_json(firebase_secret_path)
            else:
                logger.warning(f"⚠️ RUNPOD_SECRET_Firebase exists but is not JSON content or valid file path")
                # Fallback to GOOGLE_APPLICATION_CREDENTIALS
                logger.info("🔄 Using GOOGLE_APPLICATION_CREDENTIALS fallback")
                storage_client = storage.Client()
        else:
            # No RunPod secret, fallback to GOOGLE_APPLICATION_CREDENTIALS
            logger.info("🔄 Using GOOGLE_APPLICATION_CREDENTIALS fallback")
            storage_client = storage.Client()
        
        bucket = storage_client.bucket("godnathistorie-a25fa.firebasestorage.app")
        logger.info("✅ Firebase storage client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Firebase storage: {e}")
        return False

def upload_to_firebase(data: bytes, destination_blob_name: str, content_type: str = "application/octet-stream", metadata: dict = None) -> Optional[str]:
    """
    Upload data directly to Firebase Storage with metadata
    
    :param data: Binary data to upload
    :param destination_blob_name: Destination path in Firebase
    :param content_type: MIME type of the file
    :param metadata: Optional metadata to store with the file
    :return: Public URL or None if failed
    """
    global bucket
    
    logger.info(f"🔍 Starting upload: {destination_blob_name} ({len(data)} bytes, {content_type})")
    
    if bucket is None:
        logger.info("🔍 Bucket is None, initializing Firebase...")
        if not initialize_firebase():
            logger.error("❌ Firebase not initialized, cannot upload")
            return None
    
    try:
        logger.info(f"🔍 Creating blob: {destination_blob_name}")
        blob = bucket.blob(destination_blob_name)
        logger.info(f"🔍 Uploading {len(data)} bytes...")
        
        # Set metadata if provided, otherwise use default CORS headers
        if metadata:
            blob.metadata = metadata
        else:
            blob.metadata = {
                'Access-Control-Allow-Origin': '*',
                'Cache-Control': 'public, max-age=3600'
            }
        
        blob.upload_from_string(data, content_type=content_type)
        
        # Make the blob public so it can be accessed via URL
        blob.make_public()
        
        public_url = blob.public_url
        logger.info(f"✅ Uploaded to Firebase: {destination_blob_name} -> {public_url}")
        
        # Verify upload
        try:
            blob.reload()
            logger.info(f"🔍 Upload verified: {blob.name} ({blob.size} bytes)")
        except Exception as verify_error:
            logger.warning(f"⚠️ Could not verify upload: {verify_error}")
        
        return public_url
    except Exception as e:
        logger.error(f"❌ Failed to upload to Firebase: {e}")
        return None

# Chunking and processing class right
class TTSProcessor:
    def __init__(self, model: ChatterboxTTS, voice_profile_path: str, pause_ms: int = 100, max_chars: int = 500, forked_handler=None):
        """
        Initializes the TTSProcessor with a TTS model and a voice profile.

        :param model: The ChatterboxTTS model instance
        :param voice_profile_path: Path to the voice profile (.npy)
        :param pause_ms: Milliseconds of silence between chunks
        :param max_chars: Maximum number of characters per chunk
        :param forked_handler: Optional VoiceCloneHandler from forked repository
        """
        self.model = model
        self.forked_handler = forked_handler
        self.pause_ms = pause_ms
        self.max_chars = max_chars
        self.voice_profile_path = voice_profile_path  # Store the path for later use
        
        # Load the voice profile using the best available method
        try:
            if forked_handler is not None and hasattr(forked_handler, 'set_target_voice'):
                # ChatterboxVC doesn't have load_voice_profile, so use model method
                if hasattr(model, 'load_voice_profile'):
                    self.voice_profile = model.load_voice_profile(voice_profile_path)
                    logger.info(f"✅ Voice profile loaded using model method from: {voice_profile_path}")
                else:
                    # Fallback: store path for old method
                    self.voice_profile = voice_profile_path
                    logger.warning(f"⚠️ Model doesn't have load_voice_profile - using path: {voice_profile_path}")
            elif hasattr(model, 'load_voice_profile'):
                # Use enhanced method from forked repository
                self.voice_profile = model.load_voice_profile(voice_profile_path)
                logger.info(f"✅ Voice profile loaded as ref_dict from: {voice_profile_path}")
            else:
                # Fallback: store path for old method
                self.voice_profile = voice_profile_path
                logger.warning(f"⚠️ Model doesn't have load_voice_profile - using path: {voice_profile_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load voice profile: {e}")
            # Fallback: store path for old method
            self.voice_profile = voice_profile_path

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits input text into sentence-aligned chunks based on max character count.

        :param text: Full story text
        :return: List of text chunks
        """
        if NLTK_AVAILABLE:
            try:
                # Use NLTK for proper sentence tokenization
                logger.info("📝 Using NLTK sentence tokenization")
                tokenizer = PunktSentenceTokenizer()
                sentences = tokenizer.tokenize(text)
                logger.info(f"📝 NLTK tokenization successful: {len(sentences)} sentences")
            except Exception as e:
                logger.warning(f"⚠️ NLTK tokenization failed: {e} - using fallback")
                sentences = self._simple_sentence_split(text)
        else:
            # Fallback to simple sentence splitting
            logger.info("📝 Using fallback sentence splitting (NLTK not available)")
            sentences = self._simple_sentence_split(text)

        chunks, current = [], ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= self.max_chars:
                current += (" " + sent).strip()
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = sent
        if current.strip():
            chunks.append(current.strip())

        logger.info(f"📦 Text chunking: {len(sentences)} sentences → {len(chunks)} chunks")
        return chunks
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """
        Simple sentence splitting fallback when NLTK is not available.
        
        :param text: Text to split into sentences
        :return: List of sentences
        """
        # Clean up the text
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Split on sentence endings
        sentence_endings = ['.', '!', '?']
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in sentence_endings:
                sentence = current.strip()
                if sentence:
                    sentences.append(sentence)
                current = ""
        
        # Add any remaining text
        if current.strip():
            sentences.append(current.strip())
        
        # Filter out empty sentences
        sentences = [s for s in sentences if s.strip()]
        
        return sentences

    def generate_chunks(self, chunks: List[str]) -> List[str]:
        """
        Generates speech from each text chunk and stores temporary WAV files.

        :param chunks: List of text chunks
        :return: List of file paths to temporary WAV files
        """
        wav_paths = []
        
        # Voice profile is already loaded in __init__
        
        for i, chunk in enumerate(chunks):
            logger.info(f"🔄 Generating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            
            # Retry logic for each chunk
            chunk_success = False
            retry_count = 0
            max_retries = 2  # 2 retries = 3 total attempts
            
            while not chunk_success and retry_count <= max_retries:
                try:
                    # Clear GPU cache before each attempt
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Generate audio tensor using the best available method
                    audio_tensor = None
                    
                    # Use the correct high-level method: ChatterboxTTS.generate() with voice_profile_path
                    try:
                        # For TTS, we'll use the standard generate method with the voice profile path
                        # Since we're in the TTSProcessor, we need to pass the voice profile path
                        # that was used to initialize this processor
                        
                        audio_tensor = self.model.generate(
                            text=chunk,
                            voice_profile_path=self.voice_profile_path,  # Use the path passed to __init__
                            temperature=0.8,
                            exaggeration=0.5,
                            cfg_weight=0.5
                        )
                        logger.info(f"✅ Chunk {i+1} generated using standard generate with voice_profile_path")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Standard generate with voice_profile_path failed for chunk {i+1}: {e}")
                        
                        # Fallback: Use the old generate method
                        logger.info(f"🔄 Falling back to generate method for chunk {i+1}")
                        audio_tensor = self.model.generate(chunk, temperature=0.7)
                        logger.info(f"✅ Chunk {i+1} generated using fallback generate method")
                    
                    # Save to temporary file
                    temp_wav = tempfile.NamedTemporaryFile(suffix=f"_chunk_{i}.wav", delete=False)
                    torchaudio.save(temp_wav.name, audio_tensor, self.model.sr)
                    wav_paths.append(temp_wav.name)
                    
                    logger.info(f"✅ Chunk {i+1} generated | Shape: {audio_tensor.shape}")
                    chunk_success = True
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count <= max_retries:
                        logger.warning(f"⚠️ Chunk {i+1} failed (attempt {retry_count}/{max_retries + 1}): {e}")
                        logger.info(f"🔄 Retrying chunk {i+1}...")
                    else:
                        # Final failure - stop processing
                        logger.error(f"❌ Chunk {i+1} failed after {max_retries + 1} attempts: {e}")
                        logger.error(f"❌ Stopping TTS processing due to chunk failure")
                        
                        # Clean up any successfully generated chunks
                        self.cleanup(wav_paths)
                        
                        # Raise exception to stop processing
                        raise RuntimeError(f"Chunk {i+1} failed after {max_retries + 1} attempts: {e}")
        
        return wav_paths

    def stitch_and_normalize(self, wav_paths: List[str], output_path: str) -> tuple:
        """
        Stitches WAV chunks together with pause and normalizes audio levels.

        :param wav_paths: List of temporary WAV file paths
        :param output_path: Final path to export the combined WAV file
        :return: Tuple of (audio_tensor, sample_rate, duration_seconds)
        """
        logger.info(f"🔍 stitch_and_normalize called with output_path: {output_path}")
        logger.info(f"🔍 Output path absolute: {Path(output_path).absolute()}")
        logger.info(f"🔍 Output directory exists: {Path(output_path).parent.exists()}")
        
        if PYDUB_AVAILABLE:
            # Use pydub for professional audio processing
            logger.info(f"🔍 Using pydub for audio processing")
            final = AudioSegment.empty()
            for p in wav_paths:
                seg = AudioSegment.from_wav(p)
                final += seg + AudioSegment.silent(self.pause_ms)
            normalized = effects.normalize(final)
            logger.info(f"🔍 About to export to: {output_path}")
            normalized.export(output_path, format="wav")
            logger.info(f"🔍 Export completed. File exists: {Path(output_path).exists()}")
            
            # Load the saved file to get the tensor
            audio_tensor, sample_rate = torchaudio.load(output_path)
            duration = len(normalized) / 1000.0  # Convert ms to seconds
            logger.info(f"🔍 Loaded audio tensor shape: {audio_tensor.shape}")
            return audio_tensor, sample_rate, duration
        else:
            # Fallback to torchaudio concatenation
            audio_chunks = []
            sample_rate = None
            for wav_path in wav_paths:
                audio_tensor, sr = torchaudio.load(wav_path)
                if sample_rate is None:
                    sample_rate = sr
                audio_chunks.append(audio_tensor)
                
                # Add silence between chunks
                silence_duration = int(self.pause_ms * sample_rate / 1000)
                silence = torch.zeros(1, silence_duration)
                audio_chunks.append(silence)
            
            # Concatenate all chunks
            if audio_chunks:
                final_audio = torch.cat(audio_chunks, dim=-1)
                torchaudio.save(output_path, final_audio, sample_rate)
                duration = final_audio.shape[-1] / sample_rate
                return final_audio, sample_rate, duration
            else:
                raise RuntimeError("No audio chunks to concatenate")

    def cleanup(self, wav_paths: List[str]):
        """
        Deletes temporary WAV files to clean up disk space.

        :param wav_paths: List of paths to temporary WAV files
        """
        for path in wav_paths:
            try:
                os.remove(path)
                logger.debug(f"🧹 Cleaned up temporary file: {path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete {path} — {e}")

    def process(self, text: str, output_path: str) -> tuple:
        """
        Full TTS pipeline: chunk → generate → stitch → clean.

        :param text: Input text to synthesize
        :param output_path: Path to save the final audio file
        :return: Tuple of (audio_tensor, sample_rate, metadata_dict)
        """
        logger.info(f"🎵 Starting TTS processing for {len(text)} characters")
        logger.info(f"🔍 Output path: {output_path}")
        logger.info(f"🔍 Output path type: {type(output_path)}")
        logger.info(f"🔍 Output path absolute: {Path(output_path).absolute()}")
        
        chunks = self.chunk_text(text)
        logger.info(f"📦 Split into {len(chunks)} chunks")
        
        wav_paths = self.generate_chunks(chunks)
        if not wav_paths:
            raise RuntimeError("Failed to generate any audio chunks")
        
        logger.info(f"🔗 Stitching {len(wav_paths)} audio chunks...")
        logger.info(f"🔍 Stitching to output path: {output_path}")
        audio_tensor, sample_rate, total_duration = self.stitch_and_normalize(wav_paths, output_path)
        
        self.cleanup(wav_paths)
        
        logger.info(f"✅ TTS processing completed | Duration: {total_duration:.2f}s")
        logger.info(f"🔍 Final output path: {output_path}")
        logger.info(f"🔍 Output file exists: {Path(output_path).exists()}")
        
        metadata = {
            "chunk_count": len(chunks),
            "output_path": output_path,
            "duration_sec": total_duration,
            "successful_chunks": len(wav_paths),
            "sample_rate": sample_rate
        }
        
        return audio_tensor, sample_rate, metadata

def initialize_model():
    global model, forked_handler
    
    if model is not None:
        logger.info("Model already initialized")
        return model
    
    logger.info("Initializing S3Token2Wav model...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    try:
        # Minimal initialization - focus on core functionality
        model = ChatterboxTTS.from_pretrained(device='cuda')
        logger.info("✅ ChatterboxTTS model initialized")

        # Initialize the forked repository handler if available
        if FORKED_HANDLER_AVAILABLE:
            logger.info("🔧 Initializing ChatterboxVC...")
            try:
                forked_handler = ChatterboxVC(
                    s3gen=model.s3gen,
                    device=model.device
                )
                logger.info("✅ ChatterboxVC initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize ChatterboxVC: {e}")
                forked_handler = None
        else:
            logger.warning("⚠️ ChatterboxVC not available - will use fallback methods")

        # Additional model introspection logs
        import inspect
        logger.info(f"📦 Model class: {model.__class__}")
        logger.info(f"📁 Model module: {model.__class__.__module__}")
        logger.info(f"📂 Loaded model from file: {inspect.getfile(model.__class__)}")
        logger.info(f"🧠 Model dir(): {dir(model)}")
        logger.info(f"🔎 Has method load_voice_profile: {hasattr(model, 'load_voice_profile')}")

        # List all methods that contain 'voice' or 'profile'
        voice_methods = [method for method in dir(model) if 'voice' in method.lower() or 'profile' in method.lower()]
        logger.info(f"🔍 Voice/Profile related methods: {voice_methods}")

        # Fast-fail check for required method
        assert hasattr(model, 'load_voice_profile'), "🚨 Loaded model is missing `load_voice_profile`. Wrong class?"

        # Verify s3gen module source
        logger.info("🔍 ===== S3GEN VERIFICATION =====")
        if hasattr(model, "s3gen"):
            logger.info(f"📂 s3gen module path: {model.s3gen.__class__.__module__}")
            logger.info(f"📂 s3gen class: {model.s3gen.__class__}")
            logger.info(f"📂 s3gen class file: {model.s3gen.__class__.__module__}")
            
            # Check s3gen module file path
            try:
                import chatterbox.models.s3gen.s3gen as s3gen_module
                logger.info(f"📂 s3gen module file: {s3gen_module.__file__}")
                
                if 'chatterbox_embed' in s3gen_module.__file__:
                    logger.info("🎯 s3gen module is from FORKED repository")
                else:
                    logger.warning("⚠️ s3gen module is from ORIGINAL repository")
            except Exception as e:
                logger.warning(f"⚠️ Could not check s3gen module file: {e}")
            
            # Check if inference_from_text exists and its source
            if hasattr(model.s3gen, 'inference_from_text'):
                method = getattr(model.s3gen, 'inference_from_text')
                logger.info(f"📂 inference_from_text source: {method.__code__.co_filename}")
                logger.info(f"📂 inference_from_text line: {method.__code__.co_firstlineno}")
                
                if 'chatterbox_embed' in method.__code__.co_filename:
                    logger.info("🎯 inference_from_text is from FORKED repository")
                else:
                    logger.warning("⚠️ inference_from_text is from ORIGINAL repository")
            else:
                logger.warning("⚠️ inference_from_text method does NOT exist")
                
            # List all methods on s3gen
            s3gen_methods = [method for method in dir(model.s3gen) if not method.startswith('_')]
            logger.info(f"📋 Available s3gen methods: {s3gen_methods}")
        else:
            logger.warning("⚠️ Model does not have s3gen attribute")
        
        logger.info("🔍 ===== END S3GEN VERIFICATION =====")
        
        # Attach the T3 text‑to‑token encoder to S3Gen so that
        # s3gen.inference_from_text() works
        if hasattr(model, "s3gen") and hasattr(model, "t3"):
            model.s3gen.text_encoder = model.t3
            logger.info("📌 Attached text_encoder (model.t3) to model.s3gen")
        logger.info("✅ Model initialized on CUDA")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise

def list_files_for_debug():
    """List files in our directories for debugging"""
    logger.info("📂 Directory contents:")
    for directory in [VOICE_PROFILES_DIR, VOICE_SAMPLES_DIR, TTS_GENERATED_DIR, TEMP_VOICE_DIR]:
        if directory.exists():
            files = list(directory.glob("*"))
            logger.info(f"  {directory}: {[f.name for f in files]} ({len(files)} files)")
        else:
            logger.info(f"  {directory}: [DIRECTORY NOT FOUND]")
    
    # Also check if TTS_GENERATED_DIR subdirectories exist
    if TTS_GENERATED_DIR.exists():
        logger.info(f"📂 TTS_GENERATED_DIR subdirectories:")
        for subdir in TTS_GENERATED_DIR.iterdir():
            if subdir.is_dir():
                logger.info(f"  - {subdir.name}/")
                for subsubdir in subdir.iterdir():
                    if subsubdir.is_dir():
                        logger.info(f"    - {subsubdir.name}/")

def generate_voice_sample(voice_id, text, profile_base64, language, is_kids_voice, temp_profile_path, start_time):
    """Generate voice clone sample"""
    global model, forked_handler
    
    logger.info("🎤 ===== VOICE SAMPLE GENERATION =====")
    
    # Create output filename for voice sample (MP3)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_dir = VOICE_SAMPLES_DIR
    tts_filename = local_dir / f"{voice_id}_{voice_id}_sample_{timestamp}.mp3"  # Changed to .mp3
    
    logger.info(f"🎯 Voice sample local path: {tts_filename}")
    
    try:
        logger.info("🔄 Starting voice sample processing...")
        
        # Use TTSProcessor for robust chunking and generation
        processor = TTSProcessor(
            model=model,
            voice_profile_path=str(temp_profile_path),
            pause_ms=150,
            max_chars=600,
            forked_handler=forked_handler
        )
        
        # Process the text (will generate MP3 directly)
        audio_tensor, sample_rate, result = processor.process(text, str(tts_filename))
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Voice sample generated in {generation_time:.2f}s")
        
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(f"❌ Failed to generate voice sample after {generation_time:.2f}s")
        logger.error(f"❌ Error: {str(e)}")
        
        # Try with smaller chunks if it's a CUDA error
        if "CUDA error" in str(e) and len(text) > 300:
            logger.info("🔄 Retrying with smaller chunks due to CUDA error...")
            try:
                processor = TTSProcessor(
                    model=model,
                    voice_profile_path=str(temp_profile_path),
                    pause_ms=100,
                    max_chars=300
                )
                audio_tensor, sample_rate, result = processor.process(text, str(tts_filename))
                generation_time = time.time() - start_time
                logger.info(f"✅ Voice sample generated with smaller chunks in {generation_time:.2f}s")
            except Exception as retry_error:
                logger.error(f"❌ Retry also failed: {retry_error}")
                return {"status": "error", "message": f"Failed to generate voice sample even with chunking: {retry_error}"}
        else:
            return {"status": "error", "message": f"Failed to generate voice sample: {e}"}
    
    # Upload voice sample to Firebase
    if is_kids_voice:
        audio_path_firebase = f"audio/voices/{language}/kids/samples/{voice_id}_{voice_id}_sample_{timestamp}.mp3"
    else:
        audio_path_firebase = f"audio/voices/{language}/samples/{voice_id}_{voice_id}_sample_{timestamp}.mp3"
    
    logger.info(f"🎯 Voice sample Firebase path: {audio_path_firebase}")
    
    try:
        with open(tts_filename, 'rb') as f:
            mp3_bytes = f.read()
        
        # Store metadata with the voice sample file
        sample_metadata = {
            'voice_id': voice_id,
            'voice_name': voice_id.replace('voice_', ''),
            'file_type': 'voice_sample',
            'language': language,
            'is_kids_voice': str(is_kids_voice),
            'format': '96k_mp3',
            'timestamp': timestamp,
            'created_date': datetime.now().isoformat(),
            'text_length': len(text),
            'generation_time': str(generation_time),
            'Access-Control-Allow-Origin': '*',
            'Cache-Control': 'public, max-age=3600'
        }
        
        audio_uploaded = upload_to_firebase(
            mp3_bytes,
            audio_path_firebase,
            "audio/mpeg",  # Changed from "audio/x-wav"
            sample_metadata
        )
        logger.info(f"🎵 Voice sample uploaded: {audio_path_firebase}")
        
        # Return response
        response = {
            "status": "success",
            "audio_path": audio_path_firebase if audio_uploaded else None,
            "metadata": {
                "voice_id": voice_id,
                "voice_name": voice_id.replace('voice_', ''),
                "text_input": text[:500] + "..." if len(text) > 500 else text,
                "generation_time": generation_time,
                "sample_rate": model.sr,
                "audio_shape": list(audio_tensor.shape),
                "tts_file": str(tts_filename),
                "timestamp": timestamp,
                "response_type": "firebase_path",
                "format": "96k_mp3"
            }
        }
        
        logger.info(f"📤 Voice sample response ready | Time: {generation_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"❌ Failed to upload voice sample to Firebase: {e}")
        return {"status": "error", "message": f"Failed to upload voice sample to Firebase: {e}"}

def generate_tts_story(voice_id, text, profile_base64, language, story_type, is_kids_voice, temp_profile_path, start_time):
    """Generate TTS story"""
    global model, forked_handler
    
    logger.info("📖 ===== TTS STORY GENERATION =====")
    logger.info(f"🔍 Parameters received:")
    logger.info(f"  voice_id: {voice_id}")
    logger.info(f"  language: {language}")
    logger.info(f"  story_type: {story_type}")
    logger.info(f"  is_kids_voice: {is_kids_voice}")
    logger.info(f"  TTS_GENERATED_DIR: {TTS_GENERATED_DIR}")
    
    # Create output filename for TTS story (MP3)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_dir = TTS_GENERATED_DIR / language / story_type
    logger.info(f"🔍 Local directory path: {local_dir}")
    
    # Create directory if it doesn't exist
    logger.info(f"🔍 Creating directory: {local_dir}")
    logger.info(f"🔍 Directory parent exists: {local_dir.parent.exists()}")
    logger.info(f"🔍 Directory parent: {local_dir.parent}")
    
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"🔍 Directory created/exists: {local_dir.exists()}")
    logger.info(f"🔍 Directory absolute path: {local_dir.absolute()}")
    
    # Test write permissions
    try:
        test_file = local_dir / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        logger.info(f"🔍 Write permissions: OK")
    except Exception as e:
        logger.error(f"🔍 Write permissions: FAILED - {e}")
    
    tts_filename = local_dir / f"TTS_{voice_id}_{timestamp}.mp3"  # Changed to .mp3
    logger.info(f"🎯 TTS story local path: {tts_filename}")
    logger.info(f"🔍 Full absolute path: {tts_filename.absolute()}")
    
    try:
        logger.info("🔄 Starting TTS story processing...")
        
        # Use TTSProcessor for robust chunking and generation
        processor = TTSProcessor(
            model=model,
            voice_profile_path=str(temp_profile_path),
            pause_ms=150,
            max_chars=600,
            forked_handler=forked_handler
        )
        
        # Process the text (will generate MP3 directly)
        logger.info(f"🔍 Calling TTSProcessor.process with output path: {str(tts_filename)}")
        audio_tensor, sample_rate, result = processor.process(text, str(tts_filename))
        
        generation_time = time.time() - start_time
        logger.info(f"✅ TTS story generated in {generation_time:.2f}s")
        logger.info(f"🔍 TTSProcessor result: {result}")
        
        logger.info(f"🔍 Final output path: {tts_filename}")
        logger.info(f"🔍 Output file exists: {tts_filename.exists()}")
        logger.info(f"🔍 File size: {tts_filename.stat().st_size if tts_filename.exists() else 'N/A'} bytes")
        
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(f"❌ Failed to generate TTS story after {generation_time:.2f}s")
        logger.error(f"❌ Error type: {type(e)}")
        logger.error(f"❌ Error message: {str(e)}")
        logger.error(f"❌ Error details: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        
        # Try with smaller chunks if it's a CUDA error
        if "CUDA error" in str(e) and len(text) > 300:
            logger.info("🔄 Retrying with smaller chunks due to CUDA error...")
            try:
                processor = TTSProcessor(
                    model=model,
                    voice_profile_path=str(temp_profile_path),
                    pause_ms=100,
                    max_chars=300
                )
                audio_tensor, sample_rate, result = processor.process(text, str(tts_filename))
                generation_time = time.time() - start_time
                logger.info(f"✅ TTS story generated with smaller chunks in {generation_time:.2f}s")
            except Exception as retry_error:
                logger.error(f"❌ Retry also failed: {retry_error}")
                return {"status": "error", "message": f"Failed to generate TTS story even with chunking: {retry_error}"}
        else:
            return {"status": "error", "message": f"Failed to generate TTS story: {e}"}
    
    # Upload TTS story to Firebase
    audio_path_firebase = f"audio/stories/{language}/{story_type}/TTS_{voice_id}_{timestamp}.mp3"  # Changed to .mp3
    
    logger.info(f"🎯 TTS story Firebase path: {audio_path_firebase}")
    
    try:
        logger.info(f"🔍 About to read file: {tts_filename}")
        logger.info(f"🔍 File exists before reading: {tts_filename.exists()}")
        logger.info(f"🔍 File absolute path: {tts_filename.absolute()}")
        
        with open(tts_filename, 'rb') as f:
            mp3_bytes = f.read()
        
        logger.info(f"🔍 Successfully read {len(mp3_bytes)} bytes from file")
        
        # Store metadata with the TTS story file
        tts_metadata = {
            'voice_id': voice_id,
            'voice_name': voice_id.replace('voice_', ''),
            'file_type': 'tts_story',
            'language': language,
            'story_type': story_type,
            'is_kids_voice': str(is_kids_voice),
            'format': '96k_mp3',
            'timestamp': timestamp,
            'created_date': datetime.now().isoformat(),
            'text_length': len(text),
            'generation_time': str(generation_time),
            'model': 'chatterbox_tts',
            'Access-Control-Allow-Origin': '*',
            'Cache-Control': 'public, max-age=3600'
        }
        
        audio_uploaded = upload_to_firebase(
            mp3_bytes,
            audio_path_firebase,
            "audio/mpeg",  # Changed from "audio/x-wav"
            tts_metadata
        )
        logger.info(f"🎵 TTS story uploaded: {audio_path_firebase}")
        
        # Return response
        response = {
            "status": "success",
            "voice_id": voice_id,
            "audio_path": audio_path_firebase if audio_uploaded else None,
            "audio_url": audio_path_firebase if audio_uploaded else None,  # For compatibility
            "generation_time": generation_time,
            "model": "chatterbox_tts",
            "metadata": {
                "voice_id": voice_id,
                "voice_name": voice_id.replace('voice_', ''),
                "text_input": text[:500] + "..." if len(text) > 500 else text,
                "generation_time": generation_time,
                "sample_rate": model.sr,
                "audio_shape": list(audio_tensor.shape),
                "tts_file": str(tts_filename),
                "timestamp": timestamp,
                "response_type": "firebase_path",
                "format": "96k_mp3"
            }
        }
        
        logger.info(f"📤 TTS story response ready | Time: {generation_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"❌ Failed to upload TTS story to Firebase: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        logger.error(f"❌ Error details: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {"status": "error", "message": f"Failed to upload TTS story to Firebase: {e}"}

def handler(event, responseFormat="base64"):
    """Handle TTS generation requests using saved voice embeddings"""
    global model, forked_handler
    
    logger.info("🚀 ===== TTS HANDLER STARTED =====")
    logger.info(f"📥 Received event: {type(event)}")
    logger.info(f"📥 Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
    
    input = event.get('input', {})
    logger.info(f"📥 Input type: {type(input)}")
    logger.info(f"📥 Input keys: {list(input.keys()) if isinstance(input, dict) else 'Not a dict'}")
    
    # 🔍 EXTENSIVE DEBUGGING: Log all input parameters
    logger.info("🔍 ===== INPUT PARAMETER DEBUG =====")
    for key, value in input.items():
        if key == 'profile_base64':
            logger.info(f"  {key}: {len(str(value))} chars (base64 data)")
        else:
            logger.info(f"  {key}: {value}")
    logger.info("🔍 ===== END INPUT PARAMETER DEBUG =====")
    
    # Check if this is a file download request
    if input.get('action') == 'download_file':
        return handle_file_download(input)
    
    # Check if this is a file listing request
    if input.get('action') == 'list_files':
        return list_available_files()
    
    # Extract TTS parameters
    text = input.get('text')
    voice_id = input.get('voice_id')
    profile_base64 = input.get('profile_base64')
    responseFormat = input.get('responseFormat', 'base64')
    
    # Extract story context parameters
    language = input.get('language', 'en')
    story_type = input.get('story_type', 'user')
    is_kids_voice = input.get('is_kids_voice', False)
    
    logger.info(f"📋 Extracted parameters:")
    logger.info(f"  - text: {text[:50]}{'...' if text and len(text) > 50 else ''} (length: {len(text) if text else 0})")
    logger.info(f"  - voice_id: {voice_id}")
    logger.info(f"  - has_profile_base64: {bool(profile_base64)}")
    logger.info(f"  - profile_size: {len(profile_base64) if profile_base64 else 0}")
    logger.info(f"  - responseFormat: {responseFormat}")
    logger.info(f"  - language: {language}")
    logger.info(f"  - story_type: {story_type}")
    logger.info(f"  - is_kids_voice: {is_kids_voice}")
    
    if not text or not voice_id or not profile_base64:
        logger.error("❌ Missing required parameters")
        logger.error(f"  - text provided: {bool(text)}")
        logger.error(f"  - voice_id provided: {bool(voice_id)}")
        logger.error(f"  - profile_base64 provided: {bool(profile_base64)}")
        return {"status": "error", "message": "text, voice_id, and profile_base64 are required"}
    
    logger.info(f"🎤 TTS request validated: voice_id={voice_id}, text_length={len(text)}")
    
    # Debug directory contents
    list_files_for_debug()
    
    try:
        logger.info("🔍 ===== VOICE EMBEDDING PROCESSING =====")
        
        # Check if model is initialized
        if model is None:
            logger.error("❌ Model not initialized")
            return {"status": "error", "message": "Model not initialized"}
        
        logger.info(f"✅ Model is initialized: {type(model)}")
        logger.info(f"✅ Model device: {getattr(model, 'device', 'Unknown')}")
        logger.info(f"✅ Model sample rate: {getattr(model, 'sr', 'Unknown')}")
        logger.info(f"✅ Forked handler available: {forked_handler is not None}")
        
        # Decode the voice profile data
        logger.info("🔄 Decoding voice profile data...")
        try:
            profile_data = base64.b64decode(profile_base64)
            logger.info(f"✅ Voice profile data decoded: {len(profile_data)} bytes")
        except Exception as e:
            logger.error(f"❌ Failed to decode voice profile data: {e}")
            return {"status": "error", "message": f"Failed to decode voice profile data: {e}"}
        
        # Save the voice profile data to a temporary file
        logger.info("🔄 Saving voice profile data to temporary file...")
        temp_profile_path = TEMP_VOICE_DIR / f"{voice_id}_temp.npy"
        
        try:
            with open(temp_profile_path, 'wb') as f:
                f.write(profile_data)
            logger.info(f"✅ Temporary profile file created: {temp_profile_path}")
            logger.info(f"✅ File size: {temp_profile_path.stat().st_size} bytes")
        except Exception as e:
            logger.error(f"❌ Failed to save temporary profile file: {e}")
            return {"status": "error", "message": f"Failed to save temporary profile file: {e}"}
        
        # Check if model has the required method
        logger.info(f"🔍 Checking model capabilities:")
        logger.info(f"  - has load_voice_profile: {hasattr(model, 'load_voice_profile')}")
        logger.info(f"  - has generate: {hasattr(model, 'generate')}")
        logger.info(f"  - has save_voice_profile: {hasattr(model, 'save_voice_profile')}")
        logger.info(f"  - forked handler available: {forked_handler is not None}")
        
        # Load the profile using the best available method
        if forked_handler is not None and hasattr(forked_handler, 'set_target_voice'):
            logger.info("🔄 Loading profile using model method (ChatterboxVC doesn't have load_voice_profile)...")
            if hasattr(model, 'load_voice_profile'):
                profile = model.load_voice_profile(str(temp_profile_path))
                logger.info(f"✅ Voice profile loaded using model method")
            else:
                logger.error("❌ Model doesn't have load_voice_profile method")
                return {"status": "error", "message": "Voice profile support not available"}
        elif hasattr(model, 'load_voice_profile'):
            logger.info("🔄 Loading profile using model load_voice_profile method...")
            profile = model.load_voice_profile(str(temp_profile_path))
            logger.info(f"✅ Voice profile loaded successfully")
        else:
            logger.error("❌ Model doesn't have load_voice_profile method")
            logger.error("❌ This suggests the forked repository features are not available")
            return {"status": "error", "message": "Voice profile support not available"}
        
        logger.info(f"✅ Profile type: {type(profile)}")
        if hasattr(profile, 'shape'):
            logger.info(f"✅ Profile shape: {profile.shape}")
        if hasattr(profile, 'dtype'):
            logger.info(f"✅ Profile dtype: {profile.dtype}")
        
        # Generate speech using the profile
        logger.info(f"🎵 TTS: {voice_id} | Text length: {len(text)}")
        logger.info(f"🔍 DEBUG: Model loaded: {model is not None}")
        logger.info(f"🔍 DEBUG: Voice profile loaded: {profile is not None}")
        logger.info(f"🔍 DEBUG: Voice profile type: {type(profile)}")
        logger.info(f"🔍 DEBUG: Temp profile path exists: {temp_profile_path.exists()}")
        
        # Safety check for extremely long texts
        if len(text) > 13000:
            logger.warning(f"⚠️ Very long text ({len(text)} chars) - truncating to safe length")
            text = text[:13000] + "... [truncated]"
            logger.info(f"📝 Truncated text to {len(text)} characters")
        
        start_time = time.time()
        
        # Route to appropriate generation function based on story_type
        logger.info(f"🎯 ===== ROUTING DECISION =====")
        logger.info(f"  story_type: '{story_type}' (type: {type(story_type)})")
        logger.info(f"  story_type == 'sample': {story_type == 'sample'}")
        logger.info(f"  story_type in ['user', 'app']: {story_type in ['user', 'app']}")
        
        if story_type == "sample":
            logger.info("🎤 Routing to voice sample generation...")
            response = generate_voice_sample(voice_id, text, profile_base64, language, is_kids_voice, temp_profile_path, start_time)
        else:
            logger.info("📖 Routing to TTS story generation...")
            logger.info(f"  Language: {language}")
            logger.info(f"  Story Type: {story_type}")
            logger.info(f"  Is Kids Voice: {is_kids_voice}")
            response = generate_tts_story(voice_id, text, profile_base64, language, story_type, is_kids_voice, temp_profile_path, start_time)
        
        # Clean up temporary profile file
        try:
            if temp_profile_path.exists():
                os.unlink(temp_profile_path)
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Failed to clean up temp file: {cleanup_error}")
        
        logger.info("🎉 TTS completed successfully")
        return response
        
    except Exception as e:
        logger.error("💥 ===== TTS HANDLER FAILED =====")
        logger.error(f"❌ TTS request failed: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        logger.error(f"❌ Error details: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

def handle_file_download(input):
    """Handle file download requests"""
    logger.info("📁 ===== FILE DOWNLOAD REQUEST =====")
    
    file_path = input.get('file_path')
    if not file_path:
        logger.error("❌ No file_path provided in download request")
        return {"status": "error", "message": "No file_path provided"}
    
    logger.info(f"📁 Requested file: {file_path}")
    
    # Method 1: Try direct file access
    logger.info("🔍 Method 1: Direct file system access")
    try:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            logger.info(f"✅ File exists: {file_path}")
            logger.info(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            
            # Read file and convert to base64
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            audio_base64 = base64.b64encode(file_data).decode('utf-8')
            logger.info(f"✅ File read successfully: {len(audio_base64):,} base64 chars")
            
            return {
                "status": "success",
                "method": "direct_file_access",
                "file_path": file_path,
                "file_size_bytes": file_size,
                "file_size_mb": file_size/1024/1024,
                "audio_base64": audio_base64,
                "message": "File downloaded via direct file system access"
            }
        else:
            logger.error(f"❌ File not found: {file_path}")
            return {"status": "error", "message": f"File not found: {file_path}"}
            
    except Exception as e:
        logger.error(f"❌ Direct file access failed: {e}")
        return {"status": "error", "message": f"Direct file access failed: {e}"}

def list_available_files():
    """List all available TTS files for debugging"""
    logger.info("📂 ===== LISTING AVAILABLE FILES =====")
    
    try:
        if TTS_GENERATED_DIR.exists():
            # List files from both directories
            # Voice samples (keep existing flat structure)
            sample_files = list(VOICE_SAMPLES_DIR.glob("*_sample_*.wav"))  # Look for sample files
            
            # TTS files (new nested structure)
            tts_files = []
            if TTS_GENERATED_DIR.exists():
                # Recursively find all TTS files in language/story_type subdirectories
                tts_files = list(TTS_GENERATED_DIR.rglob("TTS_*.wav"))
            
            files = tts_files + sample_files
            logger.info(f"📂 Found {len(files)} TTS files:")
            
            for file in files:
                file_size = file.stat().st_size
                logger.info(f"  📄 {file.name}: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            
            return {
                "status": "success",
                "files": [{"name": f.name, "size_bytes": f.stat().st_size, "size_mb": f.stat().st_size/1024/1024} for f in files]
            }
        else:
            logger.error(f"❌ TTS directory not found: {TTS_GENERATED_DIR}")
            return {"status": "error", "message": f"TTS directory not found: {TTS_GENERATED_DIR}"}
            
    except Exception as e:
        logger.error(f"❌ Failed to list files: {e}")
        return {"status": "error", "message": f"Failed to list files: {e}"}



if __name__ == '__main__':
    logger.info("🚀 TTS Handler starting...")
    
    try:
        logger.info("🔧 Initializing model...")
        initialize_model()
        logger.info("✅ Model ready")
        
        logger.info("🚀 Starting RunPod serverless handler...")
        runpod.serverless.start({'handler': handler })
        
    except Exception as e:
        logger.error(f"💥 TTS Handler startup failed: {e}")
        raise 