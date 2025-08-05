from pathlib import Path
from typing import Optional, Dict, Tuple, List
import os
import tempfile
import logging
import numpy as np
import torchaudio
import time
import random
import string

import librosa
import torch
import perth
import torchaudio
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen, VoiceProfile
from .models.voice_encoder import VoiceEncoder
from .models.t3 import T3
from .models.tokenizers.tokenizer import EnTokenizer

# Configure logging
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
    logger.info("✅ pydub available for audio processing")
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("⚠️ pydub not available - will use torchaudio for audio processing")

REPO_ID = "ResembleAI/chatterbox"


def generate_unique_voice_id(voice_name: str, length: int = 8, max_attempts: int = 10) -> str:
    """
    Generate a unique voice ID with random alphanumeric characters to prevent naming collisions.
    Checks Firebase to ensure uniqueness.
    
    Args:
        voice_name: The base voice name
        length: Length of the random alphanumeric suffix (default: 8)
        max_attempts: Maximum attempts to generate unique ID (default: 10)
        
    Returns:
        Unique voice ID in format: voice_{voice_name}_{random_alphanumeric}
        
    Example:
        generate_unique_voice_id("christestclone") -> "voice_christestclone_A7b2K9x1"
    """
    from google.cloud import storage
    
    # Initialize Firebase client
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket("godnathistorie-a25fa.firebasestorage.app")
        logger.info("✅ Firebase client initialized for uniqueness check")
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize Firebase client: {e}")
        logger.warning("⚠️ Proceeding without uniqueness check")
        bucket = None
    
    for attempt in range(max_attempts):
        # Generate random alphanumeric suffix (letters + numbers)
        random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        
        # Create voice ID
        voice_id = f"voice_{voice_name}_{random_suffix}"
        
        # Check if this voice_id already exists in Firebase
        if bucket is not None:
            try:
                # Check in all language folders for existing profiles
                languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']  # Add more as needed
                exists = False
                
                for lang in languages:
                    # Check regular profiles
                    profile_path = f"audio/voices/{lang}/profiles/{voice_id}.npy"
                    blob = bucket.blob(profile_path)
                    if blob.exists():
                        logger.info(f"🔄 Voice ID {voice_id} already exists in {lang}/profiles, retrying...")
                        exists = True
                        break
                    
                    # Check kids profiles
                    kids_profile_path = f"audio/voices/{lang}/kids/profiles/{voice_id}.npy"
                    kids_blob = bucket.blob(kids_profile_path)
                    if kids_blob.exists():
                        logger.info(f"🔄 Voice ID {voice_id} already exists in {lang}/kids/profiles, retrying...")
                        exists = True
                        break
                
                if not exists:
                    logger.info(f"✅ Generated unique voice ID: {voice_id}")
                    return voice_id
                    
            except Exception as e:
                logger.warning(f"⚠️ Error checking Firebase uniqueness: {e}")
                logger.info(f"✅ Generated voice ID (no uniqueness check): {voice_id}")
                return voice_id
        else:
            # No Firebase check available, return the generated ID
            logger.info(f"✅ Generated voice ID (no uniqueness check): {voice_id}")
            return voice_id
    
    # If we've exhausted all attempts, add timestamp to ensure uniqueness
    timestamp = str(int(time.time()))[-6:]  # Last 6 digits of timestamp
    voice_id = f"voice_{voice_name}_{random_suffix}_{timestamp}"
    logger.warning(f"⚠️ Exhausted uniqueness attempts, using timestamp: {voice_id}")
    return voice_id


class ChatterboxVC:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict=None,
    ):
        logger.info(f"🔧 ChatterboxVC.__init__ called")
        logger.info(f"  - s3gen type: {type(s3gen)}")
        logger.info(f"  - device: {device}")
        logger.info(f"  - ref_dict: {ref_dict is not None}")
        
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        self.watermarker = perth.PerthImplicitWatermarker()
        
        if ref_dict is None:
            self.ref_dict = None
            logger.info(f"  - ref_dict set to None")
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }
            logger.info(f"  - ref_dict set with {len(self.ref_dict)} keys")
        
        logger.info(f"✅ ChatterboxVC initialized successfully")
        logger.info(f"  - Available methods: {[m for m in dir(self) if not m.startswith('_')]}")
        
        # Debug: Check for specific methods
        expected_vc_methods = [
            'create_voice_clone',
            'save_voice_profile', 
            'load_voice_profile',
            'set_voice_profile',
            'tensor_to_mp3_bytes',
            'tensor_to_audiosegment',
            'tensor_to_wav_bytes',
            'convert_audio_file_to_mp3'
        ]
        
        available_methods = [m for m in dir(self) if not m.startswith('_')]
        missing_methods = [m for m in expected_vc_methods if m not in available_methods]
        
        logger.info(f"🔍 VC Method Check:")
        logger.info(f"  - Expected methods: {expected_vc_methods}")
        logger.info(f"  - Available methods: {available_methods}")
        logger.info(f"  - Missing methods: {missing_methods}")
        
        if missing_methods:
            logger.error(f"❌ MISSING METHODS: {missing_methods}")
        else:
            logger.info(f"✅ All expected methods are available!")

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxVC':
        logger.info(f"🔧 ChatterboxVC.from_local called")
        logger.info(f"  - ckpt_dir: {ckpt_dir}")
        logger.info(f"  - device: {device}")
        
        ckpt_dir = Path(ckpt_dir)
        
        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
            logger.info(f"  - map_location set to: {map_location}")
        else:
            map_location = None
            logger.info(f"  - map_location set to: None (CUDA)")
            
        # Load VoiceEncoder
        logger.info(f"  - Loading VoiceEncoder...")
        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()
        logger.info(f"  - VoiceEncoder loaded and moved to {device}")



        # Load S3Gen
        logger.info(f"  - Loading S3Gen model...")
        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()
        logger.info(f"  - S3Gen model loaded and moved to {device}")


            
        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            logger.info(f"  - Loading builtin voice from: {builtin_voice}")
            states = torch.load(builtin_voice, map_location=map_location)
            ref_dict = states['gen']
            logger.info(f"  - Builtin voice loaded with {len(ref_dict)} keys")
        else:
            logger.info(f"  - No builtin voice found at: {builtin_voice}")

        logger.info(f"  - Creating ChatterboxVC instance...")
        result = cls(s3gen, device, ref_dict=ref_dict)
        logger.info(f"✅ ChatterboxVC.from_local completed")
        return result

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxVC':
        logger.info(f"🔧 ChatterboxVC.from_pretrained called")
        logger.info(f"  - device: {device}")
        
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logger.warning("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                logger.warning("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"
            logger.info(f"  - device changed to: {device}")
            
        logger.info(f"  - Downloading model files...")
        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)
            logger.info(f"  - Downloaded: {fpath} -> {local_path}")

        logger.info(f"  - Calling from_local...")
        result = cls.from_local(Path(local_path).parent, device)
        logger.info(f"✅ ChatterboxVC.from_pretrained completed")
        return result

    def set_target_voice(self, wav_fpath):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)


    def generate(
        self,
        audio,
        target_voice_path=None,
    ):
        if target_voice_path:
            self.set_target_voice(target_voice_path)
        else:
            assert self.ref_dict is not None, "Please `prepare_conditionals` first or specify `target_voice_path`"

        with torch.inference_mode():
            audio_16, _ = librosa.load(audio, sr=S3_SR)
            audio_16 = torch.from_numpy(audio_16).float().to(self.device)[None, ]

            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            wav, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                ref_dict=self.ref_dict,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    # ------------------------------------------------------------------
    # NEW: text‑to‑speech with an in‑memory voice profile (no file I/O)
    # ------------------------------------------------------------------
    def tts(
        self,
        text: str,
        *,
        finalize: bool = True,
    ) -> torch.Tensor:
        """
        Synthesise ``text`` directly using the current voice profile
        (``self.ref_dict``) without any intermediate audio prompt.

        This leverages the new ``S3Token2Wav.inference_from_text`` helper
        that was patched into *s3gen.py*.

        Parameters
        ----------
        text : str
            The text to speak.
        finalize : bool, optional
            Whether this is the final chunk (affects streaming).  Defaults
            to ``True`` for one-shot synthesis.

        Returns
        -------
        torch.Tensor
            A (1, samples) waveform tensor at ``self.sr``.
        """
        logger.info(f"📝 ChatterboxVC.tts called")
        logger.info(f"  - text length: {len(text)}")
        logger.info(f"  - text preview: {text[:100]}...")
        logger.info(f"  - finalize: {finalize}")
        
        if self.ref_dict is None:
            error_msg = "ChatterboxVC.tts(): no voice profile loaded. Call `set_target_voice()` or construct with `ref_dict`."
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

        # Ensure the S3Gen model has a text encoder attached.
        if not hasattr(self.s3gen, "text_encoder"):
            error_msg = "ChatterboxVC.tts(): `self.s3gen` has no `text_encoder`. Attach one with `self.s3gen.text_encoder = my_encoder`."
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

        logger.info(f"  - Starting TTS generation...")
        with torch.inference_mode():
            wav = self.s3gen.inference_from_text(
                text,
                ref_dict=self.ref_dict,
                finalize=finalize,
            )
            wav_np = wav.cpu().numpy()
            watermarked = self.watermarker.apply_watermark(
                wav_np, sample_rate=self.sr
            )
        
        result = torch.from_numpy(watermarked).unsqueeze(0)
        logger.info(f"✅ ChatterboxVC.tts completed successfully")
        logger.info(f"  - Result tensor shape: {result.shape}")
        return result

    # ------------------------------------------------------------------
    # Voice Profile Management
    # ------------------------------------------------------------------
    def save_voice_profile(self, audio_file_path: str, save_path: str):
        """
        Save a complete voice profile including embedding, prompt features, and tokens.
        
        :param audio_file_path: Path to the reference audio file
        :param save_path: Path to save the voice profile (.npy)
        """
        logger.info(f"💾 ChatterboxVC.save_voice_profile called")
        logger.info(f"  - audio_file_path: {audio_file_path}")
        logger.info(f"  - save_path: {save_path}")
        
        try:
            # Load reference audio
            logger.info(f"  - Loading reference audio...")
            ref_wav, sr = librosa.load(audio_file_path, sr=None)
            ref_wav = torch.from_numpy(ref_wav).float()
            logger.info(f"    - Audio loaded: shape={ref_wav.shape}, sr={sr}")
            
            # Get the full reference dictionary from s3gen
            logger.info(f"  - Extracting voice embedding...")
            ref_dict = self.s3gen.embed_ref(ref_wav, sr, device=self.device)
            logger.info(f"    - Embedding extracted: {len(ref_dict)} keys")
            
            # Create and save the complete voice profile
            logger.info(f"  - Creating voice profile...")
            profile = VoiceProfile(
                embedding=ref_dict["embedding"],
                prompt_feat=ref_dict["prompt_feat"],
                prompt_feat_len=ref_dict.get("prompt_feat_len"),
                prompt_token=ref_dict["prompt_token"],
                prompt_token_len=ref_dict["prompt_token_len"],
            )
            
            # Convert to numpy format for saving
            logger.info(f"  - Converting to numpy format...")
            profile_data = {
                "embedding": profile.embedding.detach().cpu().numpy(),
            }
            if profile.prompt_feat is not None:
                profile_data["prompt_feat"] = profile.prompt_feat.detach().cpu().numpy()
            if profile.prompt_feat_len is not None:
                profile_data["prompt_feat_len"] = profile.prompt_feat_len
            if profile.prompt_token is not None:
                profile_data["prompt_token"] = profile.prompt_token.detach().cpu().numpy()
            if profile.prompt_token_len is not None:
                profile_data["prompt_token_len"] = profile.prompt_token_len.detach().cpu().numpy()
            
            # Save profile
            logger.info(f"  - Saving profile to {save_path}...")
            np.save(save_path, profile_data)
            logger.info(f"✅ ChatterboxVC.save_voice_profile completed successfully")
            
        except Exception as e:
            logger.error(f"❌ ChatterboxVC.save_voice_profile failed: {e}")
            logger.error(f"  - Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"  - Full traceback: {traceback.format_exc()}")
            raise

    def load_voice_profile(self, path: str):
        """
        Load a complete voice profile with custom format.
        
        :param path: Path to the voice profile (.npy)
        :return: VoiceProfile object
        """
        logger.info(f"📂 Loading voice profile from {path}")
        
        try:
            data = np.load(path, allow_pickle=True).item()
            
            # Create VoiceProfile object
            profile = VoiceProfile(
                embedding=torch.tensor(data["embedding"]).to(self.device),
                prompt_feat=torch.tensor(data["prompt_feat"]).to(self.device) if "prompt_feat" in data else None,
                prompt_feat_len=data.get("prompt_feat_len"),
                prompt_token=torch.tensor(data["prompt_token"]).to(self.device) if "prompt_token" in data else None,
                prompt_token_len=torch.tensor(data["prompt_token_len"]).to(self.device) if "prompt_token_len" in data else None,
            )
            
            logger.info(f"✅ Voice profile loaded from {path}")
            return profile
            
        except Exception as e:
            logger.error(f"❌ Failed to load voice profile: {e}")
            raise

    def set_voice_profile(self, voice_profile_path: str):
        """
        Set the voice profile from a saved file and update the internal ref_dict.
        
        :param voice_profile_path: Path to the voice profile (.npy)
        """
        logger.info(f"🎯 ChatterboxVC.set_voice_profile called")
        logger.info(f"  - voice_profile_path: {voice_profile_path}")
        
        try:
            # Load the voice profile
            logger.info(f"  - Loading voice profile...")
            profile = self.load_voice_profile(voice_profile_path)
            logger.info(f"    - Voice profile loaded successfully")
            
            # Create ref_dict from the loaded profile
            logger.info(f"  - Creating ref_dict from profile...")
            self.ref_dict = {
                "prompt_token": profile.prompt_token.to(self.device),
                "prompt_token_len": profile.prompt_token_len.to(self.device),
                "prompt_feat": profile.prompt_feat.to(self.device),
                "prompt_feat_len": profile.prompt_feat_len,
                "embedding": profile.embedding.to(self.device),
            }
            logger.info(f"    - ref_dict created with {len(self.ref_dict)} keys")
            logger.info(f"    - ref_dict keys: {list(self.ref_dict.keys())}")
            

            
            logger.info(f"✅ ChatterboxVC.set_voice_profile completed successfully")
            
        except Exception as e:
            logger.error(f"❌ ChatterboxVC.set_voice_profile failed: {e}")
            logger.error(f"  - Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"  - Full traceback: {traceback.format_exc()}")
            raise

    # ------------------------------------------------------------------
    # Audio Processing Utilities
    # ------------------------------------------------------------------
    def tensor_to_mp3_bytes(self, audio_tensor: torch.Tensor, sample_rate: int, bitrate: str = "96k") -> bytes:
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
                audio_segment = self.tensor_to_audiosegment(audio_tensor, sample_rate)
                # Export to MP3 bytes
                mp3_file = audio_segment.export(format="mp3", bitrate=bitrate)
                # Read the bytes from the file object
                mp3_bytes = mp3_file.read()
                return mp3_bytes
            except Exception as e:
                logger.warning(f"Direct MP3 conversion failed: {e}, falling back to WAV")
                return self.tensor_to_wav_bytes(audio_tensor, sample_rate)
        else:
            logger.warning("pydub not available, falling back to WAV")
            return self.tensor_to_wav_bytes(audio_tensor, sample_rate)

    def tensor_to_audiosegment(self, audio_tensor: torch.Tensor, sample_rate: int):
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

    def tensor_to_wav_bytes(self, audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
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

    def convert_audio_file_to_mp3(self, input_path: str, output_path: str, bitrate: str = "160k"):
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

    # ------------------------------------------------------------------
    # Voice Cloning Pipeline
    # ------------------------------------------------------------------
    def upload_to_firebase(self, file_path: str, destination_blob_name: str, content_type: str = "application/octet-stream", metadata: dict = None) -> str:
        """
        Upload a file to Firebase Storage
        
        :param file_path: Path to the file to upload
        :param destination_blob_name: Destination path in Firebase
        :param content_type: MIME type of the file
        :param metadata: Optional metadata to store with the file
        :return: Public URL
        """
        try:
            from google.cloud import storage
            
            # Initialize Firebase storage client
            storage_client = storage.Client()
            bucket = storage_client.bucket("godnathistorie-a25fa.firebasestorage.app")
            
            logger.info(f"🔍 Starting Firebase upload: {destination_blob_name}")
            
            # Read file data
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Create blob and upload
            blob = bucket.blob(destination_blob_name)
            
            # Set metadata if provided
            if metadata:
                blob.metadata = metadata
            
            # Upload the data
            blob.upload_from_string(data, content_type=content_type)
            
            # Make the blob publicly accessible
            blob.make_public()
            
            public_url = blob.public_url
            logger.info(f"✅ Uploaded to Firebase: {destination_blob_name} -> {public_url}")
            
            return public_url
            
        except Exception as e:
            logger.error(f"❌ Failed to upload to Firebase: {e}")
            # Return the path as fallback
            return f"https://storage.googleapis.com/godnathistorie-a25fa.firebasestorage.app/{destination_blob_name}"

    def create_voice_clone(self, audio_file_path: str, voice_id: str = None, voice_name: str = None, metadata: Dict = None, sample_text: str = None) -> Dict:
        """
        Create voice clone from audio file.
        
        Args:
            audio_file_path: Path to the audio file
            voice_id: Unique voice identifier (optional, will generate if not provided)
            voice_name: Voice name (required if voice_id not provided)
            metadata: Optional metadata
            sample_text: Custom text for sample generation (optional, uses original audio if not provided)
            
        Returns:
            Dict with status, voice_id, profile_path, recorded_audio_path, sample_audio_path, generation_time
        """
        import time
        start_time = time.time()
        
        # Generate voice_id if not provided
        if voice_id is None:
            if voice_name is None:
                raise ValueError("Either voice_id or voice_name must be provided")
            voice_id = generate_unique_voice_id(voice_name)
        
        logger.info(f"🎤 ChatterboxVC.create_voice_clone called")
        logger.info(f"  - audio_file_path: {audio_file_path}")
        logger.info(f"  - voice_id: {voice_id}")
        logger.info(f"  - voice_name: {voice_name}")
        logger.info(f"  - metadata: {metadata}")
        logger.info(f"  - sample_text: {sample_text}")
        
        try:
            # Step 1: Save voice profile from audio
            logger.info(f"  - Step 1: Saving voice profile...")
            profile_path = f"{voice_id}.npy"
            self.save_voice_profile(audio_file_path, profile_path)
            logger.info(f"    - Voice profile saved: {profile_path}")
            
            # Step 2: Set voice profile
            logger.info(f"  - Step 2: Setting voice profile...")
            self.set_voice_profile(profile_path)
            logger.info(f"    - Voice profile set successfully")
            
            # Step 3: Generate sample audio
            logger.info(f"  - Step 3: Generating sample audio...")
            if sample_text:
                # Generate custom sample text
                logger.info(f"    - Using custom sample text: '{sample_text}'")
                sample_audio = self.tts(sample_text)
            else:
                # Generate default sample text using voice profile
                default_sample_text = f"Hello, this is the voice profile of {voice_name or 'this voice'}. I can be used to narrate whimsical stories and fairytales."
                logger.info(f"    - Using default sample text: '{default_sample_text}'")
                sample_audio = self.tts(default_sample_text)
            logger.info(f"    - Sample audio generated, shape: {sample_audio.shape}")
            
            # Step 4: Convert sample to MP3 and save
            logger.info(f"  - Step 4: Converting sample to MP3...")
            sample_mp3_bytes = self.tensor_to_mp3_bytes(sample_audio, self.sr, "96k")
            sample_audio_path = f"{voice_id}_sample.mp3"
            
            # Save sample to file
            with open(sample_audio_path, 'wb') as f:
                f.write(sample_mp3_bytes)
            logger.info(f"    - Sample audio saved: {sample_audio_path}")
            
            # Step 5: Convert original audio to MP3 and save
            logger.info(f"  - Step 5: Converting original audio to MP3...")
            recorded_audio_path = f"{voice_id}_recorded.mp3"
            self.convert_audio_file_to_mp3(audio_file_path, recorded_audio_path, "160k")
            logger.info(f"    - Recorded audio saved: {recorded_audio_path}")
            
            generation_time = time.time() - start_time
            
            # Upload files to Firebase
            logger.info(f"  - Uploading files to Firebase...")
            
            # Get language and is_kids_voice from metadata or default to 'en' and False
            language = (metadata or {}).get('language', 'en')
            is_kids_voice = (metadata or {}).get('is_kids_voice', False)
            logger.info(f"    - Using language: {language}")
            logger.info(f"    - Is kids voice: {is_kids_voice}")
            
            # Determine the base path based on is_kids_voice
            if is_kids_voice:
                base_path = f"audio/voices/{language}/kids"
                logger.info(f"    - Using kids folder: {base_path}")
            else:
                base_path = f"audio/voices/{language}"
                logger.info(f"    - Using regular folder: {base_path}")
            
            # Upload sample audio to correct bucket path
            self.upload_to_firebase(
                sample_audio_path, 
                f"{base_path}/samples/{voice_id}_sample.mp3",
                content_type="audio/mpeg"
            )
            
            # Upload recorded audio to correct bucket path
            self.upload_to_firebase(
                recorded_audio_path, 
                f"{base_path}/recorded/{voice_id}_recorded.mp3",
                content_type="audio/mpeg"
            )
            
            # Upload voice profile to correct bucket path
            self.upload_to_firebase(
                profile_path, 
                f"{base_path}/profiles/{voice_id}.npy",
                content_type="application/octet-stream"
            )
            
            # Return JSON-serializable response
            result = {
                "status": "success",
                "voice_id": voice_id,
                "profile_path": f"{voice_id}.npy",
                "recorded_audio_path": f"{voice_id}_recorded.mp3",
                "sample_audio_path": f"{voice_id}_sample.mp3",
                "generation_time": generation_time,
                "metadata": metadata or {},
                "language": language
            }
            
            logger.info(f"✅ Voice clone created successfully!")
            logger.info(f"  - Profile path: {profile_path}")
            logger.info(f"  - Recorded audio path: {recorded_audio_path}")
            logger.info(f"  - Sample audio path: {sample_audio_path}")
            logger.info(f"  - Generation time: {generation_time:.2f}s")
            
            return result
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"❌ ChatterboxVC.create_voice_clone failed: {e}")
            logger.error(f"  - Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"  - Full traceback: {traceback.format_exc()}")
            
            return {
                "status": "error",
                "voice_id": voice_id,
                "error": str(e),
                "generation_time": generation_time
            }

    def generate_voice_sample(self, voice_profile_path: str, text: str = None) -> Tuple[torch.Tensor, bytes]:
        """
        Generate a voice sample using a saved voice profile.
        
        :param voice_profile_path: Path to the voice profile (.npy)
        :param text: Text to synthesize (optional, uses default if not provided)
        :return: Tuple of (audio_tensor, mp3_bytes)
        """
        logger.info(f"🎵 Generating voice sample from {voice_profile_path}")
        
        try:
            # Set the voice profile
            self.set_voice_profile(voice_profile_path)
            
            # Use provided text or default template
            if text is None:
                text = "Hello, this is a demonstration of voice cloning with Chatterbox."
            
            # Generate audio
            audio_tensor = self.tts(text)
            
            # Convert to MP3 bytes
            mp3_bytes = self.tensor_to_mp3_bytes(audio_tensor, self.sr, "96k")
            
            logger.info(f"✅ Voice sample generated successfully")
            return audio_tensor, mp3_bytes
            
        except Exception as e:
            logger.error(f"❌ Failed to generate voice sample: {e}")
            raise



