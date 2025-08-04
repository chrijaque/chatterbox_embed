from pathlib import Path
from typing import Optional, Dict, Tuple, List
import os
import tempfile
import logging
import numpy as np

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


class T3TextEncoder:
    """
    Wrapper for T3 model that provides the expected interface for s3gen.
    Uses the actual voice profile embedding from ChatterboxVC.ref_dict.
    """
    
    def __init__(self, t3_model: T3, tokenizer: EnTokenizer, vc_instance=None):
        self.t3 = t3_model
        self.tokenizer = tokenizer
        self.vc_instance = vc_instance  # Reference to ChatterboxVC instance
        
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text into speech tokens using T3 model.
        Uses the actual voice profile embedding if available.
        """
        import torch
        from .models.t3.modules.cond_enc import T3Cond
        
        # Tokenize the text
        text_tokens = self.tokenizer.encode(text)
        text_tokens = torch.tensor(text_tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        
        # Add start and stop tokens as required by T3
        start_token = torch.tensor([[255]], dtype=torch.long, device=text_tokens.device)  # start_text_token
        stop_token = torch.tensor([[0]], dtype=torch.long, device=text_tokens.device)     # stop_text_token
        
        text_tokens = torch.cat([start_token, text_tokens, stop_token], dim=1)
        
        # Use actual voice profile embedding if available
        if (self.vc_instance and 
            hasattr(self.vc_instance, 'ref_dict') and 
            self.vc_instance.ref_dict and 
            'embedding' in self.vc_instance.ref_dict):
            
            # ✅ Use real voice profile embedding
            original_emb = self.vc_instance.ref_dict['embedding']
            
            # Resize embedding to match T3's expected size (256)
            if original_emb.shape[-1] != 256:
                logger.info(f"🎤 Resizing speaker embedding from {original_emb.shape[-1]} to 256")
                if original_emb.dim() == 1:
                    original_emb = original_emb.unsqueeze(0)  # Add batch dimension
                
                # Create a simple linear projection to resize
                resize_layer = torch.nn.Linear(original_emb.shape[-1], 256, bias=False).to(self.t3.device)
                speaker_emb = resize_layer(original_emb)
            else:
                speaker_emb = original_emb
                
            logger.info(f"🎤 Using real voice profile embedding (size: {speaker_emb.shape})")
        else:
            # ⚠️ Fallback to dummy embedding (should not happen in production)
            speaker_emb = torch.zeros(1, 256, device=self.t3.device)
            logger.warning(f"⚠️ No voice profile available, using dummy embedding")
        
        emotion_adv = torch.tensor(0.5, device=self.t3.device)
        t3_cond = T3Cond(speaker_emb=speaker_emb, emotion_adv=emotion_adv)
        
        # Use T3's inference method to generate speech tokens
        speech_tokens = self.t3.inference(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            max_new_tokens=512,  # Adjust as needed
            temperature=0.8,
            do_sample=True,
            stop_on_eos=True
        )
        
        # Clamp tokens to valid range for S3Gen (SPEECH_VOCAB_SIZE = 6561)
        speech_tokens = torch.clamp(speech_tokens, 0, 6560)
        
        return speech_tokens.squeeze(0)  # Remove batch dimension


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

        # Load T3 text encoder
        logger.info(f"  - Loading T3 text encoder...")
        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()
        logger.info(f"  - T3 text encoder loaded and moved to {device}")

        # Load S3Gen
        logger.info(f"  - Loading S3Gen model...")
        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()
        logger.info(f"  - S3Gen model loaded and moved to {device}")

        # Load tokenizer
        logger.info(f"  - Loading tokenizer...")
        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )
        logger.info(f"  - Tokenizer loaded")

        # Create T3TextEncoder wrapper and attach to S3Gen (required for TTS)
        logger.info(f"  - Creating T3TextEncoder wrapper...")
        text_encoder = T3TextEncoder(t3, tokenizer, vc_instance=None)  # Will be set after VC instance creation
        s3gen.text_encoder = text_encoder
        logger.info(f"  - T3TextEncoder wrapper attached to S3Gen")
            
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
        
        # Set the VC instance reference in the T3TextEncoder
        if hasattr(result.s3gen, 'text_encoder') and hasattr(result.s3gen.text_encoder, 'vc_instance'):
            result.s3gen.text_encoder.vc_instance = result
            logger.info(f"  - VC instance reference set in T3TextEncoder")
        
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
    def create_voice_clone(self, audio_file_path: str, voice_id: str, voice_name: str = None, output_dir: str = None, metadata: Dict = None, **kwargs) -> Dict:
        """
        Complete voice cloning pipeline:
        1. Convert voice recording to voice profile
        2. Save voice profile and recorded audio to Firebase
        3. Use voice profile to generate voice sample
        4. Save voice sample to Firebase
        """
        logger.info(f"🎤 ChatterboxVC.create_voice_clone called")
        logger.info(f"  - audio_file_path: {audio_file_path}")
        logger.info(f"  - voice_id: {voice_id}")
        logger.info(f"  - voice_name: {voice_name}")
        logger.info(f"  - metadata: {metadata}")
        
        # Use provided metadata or defaults
        if metadata is None:
            metadata = {}
        
        if voice_name is not None:
            voice_name = voice_name
        else:
            voice_name = metadata.get('name', voice_id.replace('voice_', ''))
        
        language = metadata.get('language', 'en')
        is_kids_voice = metadata.get('is_kids_voice', False)
        custom_template = metadata.get('template_message')
        
        try:
            # Step 1: Convert voice recording to voice profile
            logger.info(f"  - Step 1: Converting voice recording to voice profile...")
            if output_dir:
                profile_path = Path(output_dir) / f"{voice_id}.npy"
            else:
                profile_path = Path(f"{voice_id}.npy")
            
            self.save_voice_profile(audio_file_path, str(profile_path))
            logger.info(f"    - Voice profile created: {profile_path}")
            
            # Step 2: Save voice profile and recorded audio to Firebase
            logger.info(f"  - Step 2: Preparing files for Firebase upload...")
            
            # Convert original audio to MP3
            if output_dir:
                recorded_mp3_path = Path(output_dir) / f"{voice_id}_recorded.mp3"
            else:
                recorded_mp3_path = Path(f"{voice_id}_recorded.mp3")
            
            self.convert_audio_file_to_mp3(audio_file_path, str(recorded_mp3_path), "160k")
            
            # Read recorded MP3 bytes
            with open(recorded_mp3_path, 'rb') as f:
                recorded_mp3_bytes = f.read()
            
            # Read voice profile bytes
            with open(profile_path, 'rb') as f:
                profile_bytes = f.read()
            
            # Clean up temporary MP3 file
            if recorded_mp3_path.exists():
                os.unlink(recorded_mp3_path)
            
            # Step 3: Use voice profile to generate voice sample
            logger.info(f"  - Step 3: Generating voice sample...")
            self.set_voice_profile(str(profile_path))
            
            # Use custom template or default
            if custom_template:
                template_message = custom_template
            else:
                template_message = f"Hello, this is the voice clone of {voice_name}. This voice is used to narrate whimsical stories and fairytales."
            
            logger.info(f"    - Template message: {template_message[:100]}...")
            
            # Generate sample audio using TTS
            sample_audio = self.tts(template_message)
            logger.info(f"    - Sample audio generated, shape: {sample_audio.shape}")
            
            # Convert to MP3 bytes
            sample_mp3_bytes = self.tensor_to_mp3_bytes(sample_audio, self.sr, "96k")
            logger.info(f"    - Sample MP3 bytes size: {len(sample_mp3_bytes)}")
            
            # Step 4: Return all data for Firebase upload
            result = {
                "voice_id": voice_id,
                "status": "success",
                "voice_profile_bytes": profile_bytes,
                "recorded_audio_bytes": recorded_mp3_bytes,
                "sample_audio_bytes": sample_mp3_bytes,
                "metadata": {
                    "voice_name": voice_name,
                    "language": language,
                    "is_kids_voice": is_kids_voice,
                    "template_message": template_message
                },
                "message": "Voice clone created successfully"
            }
            
            logger.info(f"✅ Voice clone pipeline completed!")
            logger.info(f"  - Voice profile size: {len(profile_bytes)} bytes")
            logger.info(f"  - Recorded audio size: {len(recorded_mp3_bytes)} bytes")
            logger.info(f"  - Sample audio size: {len(sample_mp3_bytes)} bytes")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ ChatterboxVC.create_voice_clone failed: {e}")
            logger.error(f"  - Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"  - Full traceback: {traceback.format_exc()}")
            
            return {
                "voice_id": voice_id,
                "status": "error",
                "error": str(e),
                "message": "Voice clone creation failed"
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