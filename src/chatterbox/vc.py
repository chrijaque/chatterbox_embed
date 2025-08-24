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
import scipy.signal

import librosa
import torch
import perth
import torchaudio
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen, VoiceProfile
from .models.voice_encoder import VoiceEncoder
from .models.t3 import T3
from .models.tokenizers.tokenizer import EnTokenizer
from .models.t3.modules.cond_enc import T3Cond
from .tts import punc_norm

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


def make_safe_slug(value: str) -> str:
    """Create a filesystem and URL-safe slug from a string."""
    if value is None:
        return ""
    import re as _re
    slug = value.strip().lower()
    slug = _re.sub(r"\s+", "_", slug)
    slug = _re.sub(r"[^a-z0-9_-]", "", slug)
    slug = slug.strip("_-")
    return slug or "voice"


def build_voice_id_with_user(voice_name: str, user_id: str) -> str:
    """Build voice id in the format voice_{name}_{userID} using sanitized parts."""
    name_part = make_safe_slug(voice_name or "voice")
    user_part = make_safe_slug(user_id or "")
    if user_part:
        return f"voice_{name_part}_{user_part}"
    return f"voice_{name_part}"


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
        t3,  # Add T3 model
        s3gen: S3Gen,
        ve,  # Add VoiceEncoder
        tokenizer,  # Add tokenizer
        device: str,
        ref_dict: dict=None,
    ):
        logger.info(f"🔧 ChatterboxVC.__init__ called")
        logger.info(f"  - t3 type: {type(t3)}")
        logger.info(f"  - s3gen type: {type(s3gen)}")
        logger.info(f"  - ve type: {type(ve)}")
        logger.info(f"  - tokenizer type: {type(tokenizer)}")
        logger.info(f"  - device: {device}")
        logger.info(f"  - ref_dict: {ref_dict is not None}")
        
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.watermarker = perth.PerthImplicitWatermarker()
        
        if ref_dict is None:
            self.ref_dict = None
            self.ve_embedding = None
            logger.info(f"  - ref_dict set to None")
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }
            # Extract VoiceEncoder embedding if available
            self.ve_embedding = ref_dict.get('ve_embedding')
            if self.ve_embedding is not None and torch.is_tensor(self.ve_embedding):
                self.ve_embedding = self.ve_embedding.to(device)
            logger.info(f"  - ref_dict set with {len(self.ref_dict)} keys")
            logger.info(f"  - ve_embedding available: {self.ve_embedding is not None}")
        
        logger.info(f"✅ ChatterboxVC initialized successfully")
        logger.info(f"  - Available methods: {[m for m in dir(self) if not m.startswith('_')]}")

        # Final loudness normalization configuration for generated samples (default OFF to avoid impact)
        self.enable_loudness_normalization = False
        self.loudness_target_lufs = -19.4  # Integrated loudness (LUFS)
        self.loudness_target_tp = -1.0     # True peak (dBTP)
        self.loudness_target_lra = 11.0    # Loudness range (LU)
        self.loudness_method = "ffmpeg"    # "ffmpeg" with two-pass loudnorm (preferred)

        # Audio cleaning in cloning pipeline (enabled; this was the previous behavior)
        self.enable_audio_cleaning = True
        
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

        # Load T3 model
        logger.info(f"  - Loading T3 model...")
        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()
        logger.info(f"  - T3 model loaded and moved to {device}")

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
            
        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            logger.info(f"  - Loading builtin voice from: {builtin_voice}")
            states = torch.load(builtin_voice, map_location=map_location)
            ref_dict = states['gen']
            logger.info(f"  - Builtin voice loaded with {len(ref_dict)} keys")
        else:
            logger.info(f"  - No builtin voice found at: {builtin_voice}")

        logger.info(f"  - Creating ChatterboxVC instance...")
        result = cls(t3, s3gen, ve, tokenizer, device, ref_dict=ref_dict)
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

    def _ffmpeg_available(self) -> bool:
        try:
            import shutil
            return shutil.which("ffmpeg") is not None
        except Exception:
            return False

    def _run_ffmpeg_loudnorm(self, input_path: str, output_path: str) -> bool:
        """Run two-pass ffmpeg loudnorm to achieve target LUFS/TP/LRA. Returns True on success."""
        import subprocess
        import json
        import re

        # Pass 1: Measure
        measure_cmd = [
            "ffmpeg", "-hide_banner", "-nostats", "-y",
            "-i", input_path,
            "-af", f"loudnorm=I={self.loudness_target_lufs}:TP={self.loudness_target_tp}:LRA={self.loudness_target_lra}:print_format=json",
            "-f", "null", "-"
        ]
        try:
            proc = subprocess.run(measure_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stderr = proc.stderr or ""
            json_matches = list(re.finditer(r"\{[\s\S]*?\}", stderr))
            if not json_matches:
                return False
            stats = json.loads(json_matches[-1].group(0))
            measured_I = stats.get("input_i")
            measured_LRA = stats.get("input_lra")
            measured_TP = stats.get("input_tp")
            measured_thresh = stats.get("input_thresh")
            offset = stats.get("target_offset")
            if any(v is None for v in [measured_I, measured_LRA, measured_TP, measured_thresh, offset]):
                return False

            # Pass 2: Apply with measured parameters
            apply_cmd = [
                "ffmpeg", "-hide_banner", "-nostats", "-y",
                "-i", input_path,
                "-af",
                (
                    "loudnorm="
                    f"I={self.loudness_target_lufs}:TP={self.loudness_target_tp}:LRA={self.loudness_target_lra}:"
                    f"measured_I={measured_I}:measured_LRA={measured_LRA}:measured_TP={measured_TP}:"
                    f"measured_thresh={measured_thresh}:offset={offset}:linear=true:print_format=summary"
                ),
                output_path,
            ]
            proc2 = subprocess.run(apply_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return proc2.returncode == 0
        except Exception:
            return False

    def _fallback_simple_loudness(self, input_path: str, output_path: str) -> bool:
        """Fallback loudness step: increase level and cap peaks. Not true EBU R128 but safe."""
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_wav(input_path)
            # Conservative gain towards -19.4 LUFS target (~+3.6 dB from -23)
            seg = seg.apply_gain(3.6)
            peak_over = seg.max_dBFS + 1.0
            if peak_over > 0:
                seg = seg.apply_gain(-peak_over)
            seg.export(output_path, format="wav")
            return True
        except Exception:
            return False

    def apply_loudness_normalization_tensor(self, audio_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Normalize loudness of a tensor to target LUFS using ffmpeg two-pass (preferred)."""
        if not self.enable_loudness_normalization:
            return audio_tensor
        import tempfile
        import os as _os
        try:
            # Write to temp WAV
            tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_in.close()
            torchaudio.save(tmp_in.name, audio_tensor, sample_rate)

            # Normalize to temp out
            base, _ = _os.path.splitext(tmp_in.name)
            tmp_out = base + "_loudnorm.wav"

            ok = False
            if self.loudness_method == "ffmpeg" and self._ffmpeg_available():
                ok = self._run_ffmpeg_loudnorm(tmp_in.name, tmp_out)
            if not ok:
                ok = self._fallback_simple_loudness(tmp_in.name, tmp_out)

            # Load result
            out_path = tmp_out if ok else tmp_in.name
            audio_norm, _ = torchaudio.load(out_path)

            # Cleanup
            for p in [tmp_in.name, tmp_out]:
                try:
                    if _os.path.exists(p):
                        _os.unlink(p)
                except Exception:
                    pass

            return audio_norm
        except Exception:
            return audio_tensor

    def set_target_voice(self, wav_fpath):
        """Set target voice from audio file path, creating both S3Gen and VoiceEncoder embeddings."""
        logger.info(f"🎯 ChatterboxVC.set_target_voice called with: {wav_fpath}")
        
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_wav_full, orig_sr = librosa.load(wav_fpath, sr=None)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)
        
        # Also compute VoiceEncoder embedding for T3 conditioning
        ref_16k_wav = librosa.resample(ref_wav_full, orig_sr=orig_sr, target_sr=S3_SR)
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        self.ve_embedding = ve_embed.mean(axis=0, keepdim=True).to(self.device)
        
        logger.info(f"✅ Target voice set with S3Gen and VoiceEncoder embeddings")


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
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Synthesise text using the proper T3 → S3Gen pipeline with voice profile.

        Parameters
        ----------
        text : str
            The text to speak.
        finalize : bool, optional
            Whether this is the final chunk (affects streaming).
        exaggeration : float
            Voice exaggeration factor.
        cfg_weight : float
            Classifier-free guidance weight.
        temperature : float
            Generation temperature.
        repetition_penalty : float
            Repetition penalty factor.
        min_p : float
            Minimum probability threshold.
        top_p : float
            Top-p sampling threshold.

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
            error_msg = "ChatterboxVC.tts(): no voice profile loaded. Call `set_target_voice()` or `set_voice_profile()`."
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

        if self.ve_embedding is None:
            error_msg = "ChatterboxVC.tts(): no VoiceEncoder embedding available. Voice profile missing ve_embedding."
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

        logger.info(f"  - Starting TTS generation with proper T3 → S3Gen pipeline...")
        
        with torch.inference_mode():
            # Step 1: Prepare T3 conditioning
            logger.info(f"  - Preparing T3 conditioning...")
            
            # Create speech condition prompt tokens if available
            t3_cond_prompt_tokens = None
            if plen := self.t3.hp.speech_cond_prompt_len:
                if 'prompt_token' in self.ref_dict:
                    # Use prompt tokens from ref_dict, limited to required length
                    prompt_tokens = self.ref_dict['prompt_token']
                    if prompt_tokens.shape[1] >= plen:
                        t3_cond_prompt_tokens = prompt_tokens[:, :plen].to(self.device)
                    else:
                        t3_cond_prompt_tokens = prompt_tokens.to(self.device)

            # Create T3 conditioning
            t3_cond = T3Cond(
                speaker_emb=self.ve_embedding,
                cond_prompt_speech_tokens=t3_cond_prompt_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1).to(self.device),
            ).to(device=self.device)
            
            # Step 2: Process text and tokenize
            logger.info(f"  - Processing and tokenizing text...")
            text = punc_norm(text)
            text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

            if cfg_weight > 0.0:
                text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

            sot = self.t3.hp.start_text_token
            eot = self.t3.hp.stop_text_token
            text_tokens = torch.nn.functional.pad(text_tokens, (1, 0), value=sot)
            text_tokens = torch.nn.functional.pad(text_tokens, (0, 1), value=eot)
            
            # Step 3: T3 inference (text → speech tokens)
            logger.info(f"  - T3 inference: text → speech tokens...")
            speech_tokens = self.t3.inference(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            
            # Extract only the conditional batch if CFG was used
            if cfg_weight > 0.0:
                speech_tokens = speech_tokens[0]

            # Clean up tokens
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens[speech_tokens < 6561]
            speech_tokens = speech_tokens.to(self.device)
            
            logger.info(f"  - Generated speech tokens: {speech_tokens.shape}")
            
            # Step 4: S3Gen inference (speech tokens → waveform)
            logger.info(f"  - S3Gen inference: speech tokens → waveform...")
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.ref_dict,
                finalize=finalize,
            )
            
            # Step 5: Apply watermarking
            logger.info(f"  - Applying watermarking...")
            wav_np = wav.squeeze(0).detach().cpu().numpy()
            watermarked = self.watermarker.apply_watermark(wav_np, sample_rate=self.sr)
        
        result = torch.from_numpy(watermarked).unsqueeze(0)
        logger.info(f"✅ ChatterboxVC.tts completed successfully")
        logger.info(f"  - Result tensor shape: {result.shape}")
        return result

    # ------------------------------------------------------------------
    # Audio Cleaning and Preprocessing
    # ------------------------------------------------------------------
    def clean_audio(self, audio_file_path: str, output_path: str = None) -> str:
        """
        High-quality audio cleaning with spectral noise reduction.
        Optimized for best voice cloning results on GPU serverless.
        
        :param audio_file_path: Path to input audio file
        :param output_path: Path to save cleaned audio (optional)
        :return: Path to cleaned audio file
        """
        logger.info(f"🔬 High-quality audio cleaning: {audio_file_path}")
        
        if output_path is None:
            base, _ext = os.path.splitext(audio_file_path)
            output_path = f"{base}_cleaned.wav"
        
        try:
            # Import required libraries
            import noisereduce as nr
            import soundfile as sf
            from scipy.signal import butter, filtfilt
            logger.info("  - Using advanced spectral noise reduction")
        except ImportError as e:
            logger.error(f"❌ Required library missing: {e}")
            logger.error("  - Install with: pip install noisereduce soundfile")
            return audio_file_path  # Return original on failure
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_file_path, sr=None)
            logger.info(f"  - Loaded audio: {len(audio)} samples @ {sr}Hz, duration: {len(audio)/sr:.2f}s")
            
            original_length = len(audio)
            
            # 1. Initial trim to remove obvious silence
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=15)
            logger.info(f"  - Initial trim: {len(audio_trimmed)} samples ({len(audio_trimmed)/sr:.2f}s)")
            
            # 2. Spectral noise reduction (most important for quality)
            audio_denoised = nr.reduce_noise(
                y=audio_trimmed, 
                sr=sr,
                stationary=False,      # Handle varying background noise
                prop_decrease=0.85     # Aggressive noise reduction
            )
            logger.info(f"  - Applied spectral noise reduction")
            
            # 3. High-pass filter to remove low-frequency rumble
            nyquist = sr / 2
            low_cutoff = 85  # Remove below 85Hz (preserves voice fundamentals)
            low = low_cutoff / nyquist
            b, a = butter(6, low, btype='high')  # 6th order for steep rolloff
            audio_filtered = filtfilt(b, a, audio_denoised)
            logger.info(f"  - Applied high-pass filter @ {low_cutoff}Hz")
            
            # 4. Gentle normalization to preserve dynamics
            peak = np.max(np.abs(audio_filtered))
            if peak > 0:
                # Normalize to -3dB to prevent clipping while preserving dynamics
                target_level = 0.707  # -3dB
                audio_normalized = audio_filtered * (target_level / peak)
            else:
                audio_normalized = audio_filtered
            
            # 5. Final precision trim to remove any remaining silence
            audio_final, trim_idx = librosa.effects.trim(
                audio_normalized, 
                top_db=25,           # More sensitive trim
                frame_length=2048,   # Longer frames for stability
                hop_length=512
            )
            
            # 6. Quality checks
            final_duration = len(audio_final) / sr
            if final_duration < 0.5:  # Less than 0.5 seconds
                logger.warning(f"  - Audio very short after cleaning: {final_duration:.2f}s")
            elif final_duration < 2.0:  # Less than 2 seconds
                logger.info(f"  - Short audio after cleaning: {final_duration:.2f}s")
            
            # Always save cleaned audio as high-quality WAV (lossless) for best embedding quality
            sf.write(output_path, audio_final, sr, format='WAV', subtype='PCM_24')
            
            logger.info(f"✅ High-quality cleaning completed: {output_path}")
            logger.info(f"  - Original: {original_length/sr:.2f}s → Cleaned: {len(audio_final)/sr:.2f}s")
            logger.info(f"  - Reduction: {(1 - len(audio_final)/original_length)*100:.1f}%")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Audio cleaning failed: {e}")
            import traceback
            logger.error(f"  - Full traceback: {traceback.format_exc()}")
            return audio_file_path  # Return original on failure

    # ------------------------------------------------------------------
    # Voice Profile Management
    # ------------------------------------------------------------------
    def save_voice_profile(self, audio_file_path: str, save_path: str):
        """
        Save a complete voice profile including embedding, prompt features, tokens, and VoiceEncoder embedding.
        
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
            logger.info(f"  - Extracting S3Gen voice embedding...")
            ref_dict = self.s3gen.embed_ref(ref_wav, sr, device=self.device)
            logger.info(f"    - S3Gen embedding extracted: {len(ref_dict)} keys")
            
            # Also compute voice encoder embedding for T3
            logger.info(f"  - Extracting VoiceEncoder embedding...")
            ref_16k_wav = librosa.resample(ref_wav.numpy(), orig_sr=sr, target_sr=S3_SR)
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)
            logger.info(f"    - VoiceEncoder embedding extracted: shape={ve_embed.shape}")
            
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
                "ve_embedding": ve_embed.detach().cpu().numpy(),  # Add VoiceEncoder embedding
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
        Load a complete voice profile with custom format including VoiceEncoder embedding.
        
        :param path: Path to the voice profile (.npy)
        :return: VoiceProfile object with ve_embedding attribute
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
            
            # Add VoiceEncoder embedding as an attribute
            if "ve_embedding" in data:
                profile.ve_embedding = torch.tensor(data["ve_embedding"]).to(self.device)
                logger.info(f"  - VoiceEncoder embedding loaded: shape={profile.ve_embedding.shape}")
            else:
                # Fallback for old profiles without voice encoder embedding
                profile.ve_embedding = None
                logger.warning(f"  - No VoiceEncoder embedding found in profile (old format)")
            
            logger.info(f"✅ Voice profile loaded from {path}")
            return profile
            
        except Exception as e:
            logger.error(f"❌ Failed to load voice profile: {e}")
            raise

    def set_voice_profile(self, voice_profile_path: str):
        """
        Set the voice profile from a saved file and update the internal ref_dict and ve_embedding.
        
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
            
            # Set VoiceEncoder embedding for T3 conditioning
            if hasattr(profile, 've_embedding') and profile.ve_embedding is not None:
                self.ve_embedding = profile.ve_embedding.to(self.device)
                logger.info(f"    - VoiceEncoder embedding set: shape={self.ve_embedding.shape}")
            else:
                logger.warning(f"    - No VoiceEncoder embedding available in profile")
                self.ve_embedding = None
            
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
            
            # Get file size for logging
            file_size = os.path.getsize(file_path)
            logger.info(f"🔍 Starting Firebase upload: {destination_blob_name} ({file_size:,} bytes)")
            
            # Create blob and upload directly from file (more efficient for large files)
            blob = bucket.blob(destination_blob_name)
            
            # Set metadata if provided
            if metadata:
                blob.metadata = metadata
            
            # Upload directly from file path (more memory efficient)
            blob.upload_from_filename(file_path, content_type=content_type)
            
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
        Create voice clone from audio file with high-quality audio cleaning.
        
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
        
        # Strict: require server-provided voice_id; do not invent
        if voice_id is None:
            raise ValueError("voice_id is required and must be provided by the server")
        
        logger.info(f"🎤 ChatterboxVC.create_voice_clone called")
        logger.info(f"  - audio_file_path: {audio_file_path}")
        logger.info(f"  - voice_id: {voice_id}")
        logger.info(f"  - voice_name: {voice_name}")
        logger.info(f"  - metadata: {metadata}")
        logger.info(f"  - sample_text: {sample_text}")
        
        try:
            # Step 0: Optional high-quality audio cleaning (disabled by default)
            if self.enable_audio_cleaning:
                logger.info(f"  - Step 0: Applying high-quality audio cleaning...")
                processed_audio_path = self.clean_audio(audio_file_path)
                logger.info(f"    - Audio cleaning completed: {processed_audio_path}")
            else:
                processed_audio_path = audio_file_path
                logger.info(f"  - Step 0: Skipping audio cleaning (using original reference)")
            
            # Enforce deterministic filenames from voice_id
            user_id_meta = str((metadata or {}).get("user_id", ""))
            profile_filename = (metadata or {}).get("profile_filename") or f"{voice_id}.npy"
            sample_filename = (metadata or {}).get("sample_filename") or f"{voice_id}.mp3"
            # recorded filename is only used when no pointer was provided
            safe_name = make_safe_slug(voice_name or "voice")
            safe_user = make_safe_slug(user_id_meta or "user")
            recorded_filename = (metadata or {}).get("recorded_filename") or f"recording_{safe_name}_{safe_user}.wav"

            # Step 1: Save voice profile from (cleaned) audio (local temp), will upload with standardized name
            logger.info(f"  - Step 1: Saving voice profile...")
            profile_local_path = f"{voice_id}.npy"
            self.save_voice_profile(processed_audio_path, profile_local_path)
            logger.info(f"    - Voice profile saved: {profile_local_path}")
            
            # Step 2: Set voice profile
            logger.info(f"  - Step 2: Setting voice profile...")
            self.set_voice_profile(profile_local_path)
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

            # Apply final loudness normalization to sample
            try:
                sample_audio = self.apply_loudness_normalization_tensor(sample_audio, self.sr)
                logger.info("    - Sample loudness normalized to target LUFS")
            except Exception as _e_ln:
                logger.warning(f"    - Sample loudness normalization skipped: {_e_ln}")
            
            # Step 4: Convert sample to MP3 and save (local temp)
            logger.info(f"  - Step 4: Converting sample to MP3...")
            sample_mp3_bytes = self.tensor_to_mp3_bytes(sample_audio, self.sr, "96k")
            sample_local_path = f"{voice_id}.mp3"
            
            # Save sample to file
            with open(sample_local_path, 'wb') as f:
                f.write(sample_mp3_bytes)
            logger.info(f"    - Sample audio saved: {sample_local_path}")
            
            # Step 5: Prepare recorded audio path semantics
            logger.info(f"  - Step 5: Preparing recorded audio path semantics...")
            recorded_audio_path_local = None
            recorded_pointer = None
            recorded_in_metadata = (metadata or {}).get('recorded_path')
            if recorded_in_metadata:
                # A client-supplied recorded Storage path exists → do NOT upload duplicate
                recorded_pointer = str(recorded_in_metadata)
                logger.info(f"    - Provided recorded_path pointer: {recorded_pointer}")
            else:
                # Fallback to previous behavior: keep cleaned audio locally for upload
                recorded_audio_path_local = recorded_filename
                try:
                    # If source is already WAV, copy/move; otherwise transcode to WAV
                    _src_ext = os.path.splitext(processed_audio_path)[1].lower()
                    if _src_ext == ".wav":
                        if processed_audio_path != recorded_audio_path_local:
                            import shutil
                            shutil.move(processed_audio_path, recorded_audio_path_local)
                        logger.info(f"    - Recorded source is WAV → {recorded_audio_path_local}")
                    else:
                        logger.info(f"    - Transcoding recorded source {_src_ext} → WAV")
                        audio_tensor, sr = torchaudio.load(processed_audio_path)
                        torchaudio.save(recorded_audio_path_local, audio_tensor, sr)
                        logger.info(f"    - Transcoded to WAV: {recorded_audio_path_local} @ {sr}Hz")
                except Exception as _e_conv:
                    logger.warning(f"    - Failed to convert to WAV, fallback to move: {_e_conv}")
                    import shutil
                    shutil.move(processed_audio_path, recorded_audio_path_local)
            
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
            
            # Firebase Storage Structure (standardized by voiceId):
            # /{base_path}/profiles/{voiceId}.npy
            # /{base_path}/samples/{voiceId}.mp3
            # recorded path remains client-provided or recording_{safeName}_{userId}.ext
            
            # Pre-create Firestore doc to surface entry immediately while uploads run
            try:
                client_pre = _init_firestore_client()
                if client_pre:
                    from google.cloud.firestore import SERVER_TIMESTAMP  # type: ignore
                    doc_ref_pre = client_pre.collection("voice_profiles").document(voice_id)
                    doc_ref_pre.set({
                        "userId": (metadata or {}).get("user_id", ""),
                        "voiceId": voice_id,
                        "name": voice_name or voice_id,
                        "language": language,
                        "isKidsVoice": is_kids_voice,
                        "status": "processing",
                        "createdAt": SERVER_TIMESTAMP,
                        "updatedAt": SERVER_TIMESTAMP,
                        "metadata": metadata or {},
                    }, merge=True)
                    logger.info(f"🗄️  Firestore voice_profiles/{voice_id} pre-created (processing)")
            except Exception as e:
                logger.warning(f"⚠️ Failed to pre-create Firestore voice_profiles doc: {e}")

            # Common metadata for all files
            user_id_meta = str((metadata or {}).get("user_id", ""))
            logger.info(f"  - metadata.user_id for Storage: '{user_id_meta}'")
            common_meta = {
                "user_id": user_id_meta,
                "voice_id": voice_id,
                "voice_name": voice_name or "",
                "language": language,
                "is_kids_voice": str(is_kids_voice).lower(),
                "model_type": (metadata or {}).get("model_type", "chatterbox"),
            }

            # Upload sample audio to standardized bucket path
            self.upload_to_firebase(
                sample_local_path,
                f"{base_path}/samples/{sample_filename}",
                content_type="audio/mpeg",
                metadata={**common_meta, "file_kind": "sample"}
            )
            
            # Upload recorded audio only if we didn't receive a pointer
            rec_ext = None
            mime_map = {
                ".wav": "audio/wav",
                ".flac": "audio/flac",
                ".ogg": "audio/ogg",
                ".aiff": "audio/aiff",
                ".aif": "audio/aiff",
                ".mp3": "audio/mpeg",
                ".m4a": "audio/mp4",
                ".aac": "audio/aac",
                ".wma": "audio/x-ms-wma",
                ".opus": "audio/opus",
            }
            recorded_path_for_response = None
            if recorded_pointer:
                # Use pointer directly in outputs; skip upload
                recorded_path_for_response = recorded_pointer
                logger.info("    - Skipping recorded upload (client-provided recorded_path)")
                # Try to infer extension for Firestore doc
                _lp = recorded_pointer.lower()
                for ext in mime_map.keys():
                    if _lp.endswith(ext):
                        rec_ext = ext
                        break
                if not rec_ext:
                    rec_ext = ".wav"
            else:
                # Proceed with upload as before
                rec_ext = ".wav"
                rec_ct = mime_map.get(rec_ext, "application/octet-stream")
                self.upload_to_firebase(
                    recorded_audio_path_local,
                    f"{base_path}/recorded/{recorded_filename}",
                    content_type=rec_ct,
                    metadata={**common_meta, "file_kind": "recorded"}
                )
                recorded_path_for_response = recorded_filename
            
            # Upload voice profile to standardized bucket path
            self.upload_to_firebase(
                profile_local_path,
                f"{base_path}/profiles/{profile_filename}",
                content_type="application/octet-stream",
                metadata={**common_meta, "file_kind": "profile"}
            )
            
            # Return JSON-serializable response
            result = {
                "status": "success",
                "voice_id": voice_id,
                "profile_path": profile_filename,
                "recorded_audio_path": recorded_path_for_response,
                "sample_audio_path": sample_filename,
                "generation_time": generation_time,
                "metadata": metadata or {},
                "language": language
            }
            
            logger.info(f"✅ Voice clone created successfully!")
            logger.info(f"  - Profile (storage): {profile_filename}")
            logger.info(f"  - Recorded (storage): {recorded_path_for_response}")
            logger.info(f"  - Sample (storage): {sample_filename}")
            logger.info(f"  - Generation time: {generation_time:.2f}s")

            # Also write/update Firestore voice_profiles document so the main app's /voices list updates in realtime
            try:
                client = _init_firestore_client()
                if client:
                    from google.cloud.firestore import SERVER_TIMESTAMP  # type: ignore
                    doc_id = voice_id
                    doc_ref = client.collection("voice_profiles").document(doc_id)
                    doc_ref.set({
                        "userId": (metadata or {}).get("user_id", ""),
                        "voiceId": voice_id,
                        "name": voice_name or voice_id,
                        "language": language,
                        "isKidsVoice": is_kids_voice,
                        "status": "ready",
                        "samplePath": f"{base_path}/samples/{sample_filename}",
                        "profilePath": f"{base_path}/profiles/{profile_filename}",
                        "recordedPath": (
                            recorded_pointer if recorded_pointer else f"{base_path}/recorded/{recorded_filename}"
                        ),
                        "createdAt": SERVER_TIMESTAMP,
                        "updatedAt": SERVER_TIMESTAMP,
                        "metadata": metadata or {},
                    }, merge=True)
                    result["firestore_profile_id"] = doc_id
                    logger.info(f"🗄️  Firestore voice_profiles/{doc_id} upserted")
                else:
                    logger.warning("⚠️ Firestore client not initialized; skipping Firestore write")
            except Exception as e:
                logger.warning(f"⚠️ Failed to write Firestore voice_profiles doc: {e}")
            
            # Clean up local files after Firebase upload (keep serverless environment clean)
            try:
                local_cleanup = [profile_local_path, sample_local_path]
                if recorded_audio_path_local:
                    local_cleanup.append(recorded_audio_path_local)
                for file_path in local_cleanup:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"  - Cleaned up local file: {file_path}")
            except Exception as e:
                logger.warning(f"  - Failed to clean up some local files: {e}")
            
            # Attempt app callback on success if provided
            try:
                cb_url = (metadata or {}).get('callback_url')
                if cb_url:
                    import hmac, hashlib, time, json
                    from urllib.parse import urlparse
                    from urllib.request import Request, urlopen
                    # Build canonical storage paths
                    kids_segment = 'kids/' if is_kids_voice else ''
                    profile_path = f"audio/voices/{language}/{kids_segment}profiles/{voice_id}.npy"
                    sample_path = f"audio/voices/{language}/{kids_segment}samples/{voice_id}.mp3"
                    payload = {
                        'status': 'success',
                        'user_id': user_id_meta,
                        'voice_id': voice_id,
                        'voice_name': voice_name or '',
                        'language': language,
                        'profile_path': profile_path,
                        'sample_path': sample_path,
                    }
                    secret = os.getenv('DAEZEND_API_SHARED_SECRET')
                    if secret:
                        parsed = urlparse(cb_url)
                        path = parsed.path or '/api/voice-clone/callback'
                        ts = str(int(time.time() * 1000))
                        body = json.dumps(payload).encode('utf-8')
                        prefix = f"POST\n{path}\n{ts}\n".encode('utf-8')
                        sig = hmac.new(secret.encode('utf-8'), prefix + body, hashlib.sha256).hexdigest()
                        req = Request(cb_url, data=body, headers={
                            'Content-Type': 'application/json',
                            'X-Daezend-Timestamp': ts,
                            'X-Daezend-Signature': sig,
                        }, method='POST')
                        try:
                            with urlopen(req, timeout=15) as resp:
                                _ = resp.read()
                        except Exception as _cb_e:
                            logger.warning(f"⚠️ Success callback failed: {_cb_e}")
            except Exception:
                pass

            return result
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"❌ ChatterboxVC.create_voice_clone failed: {e}")
            logger.error(f"  - Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"  - Full traceback: {traceback.format_exc()}")
            # Attempt error callback
            try:
                cb_url = (metadata or {}).get('callback_url')
                if cb_url:
                    import hmac, hashlib, time, json
                    from urllib.parse import urlparse
                    from urllib.request import Request, urlopen
                    payload = {
                        'status': 'error',
                        'user_id': (metadata or {}).get('user_id', ''),
                        'voice_id': voice_id or '',
                        'voice_name': voice_name or '',
                        'language': (metadata or {}).get('language', 'en'),
                        'error': str(e),
                    }
                    secret = os.getenv('DAEZEND_API_SHARED_SECRET')
                    if secret:
                        parsed = urlparse(cb_url)
                        path = parsed.path or '/api/voice-clone/callback'
                        ts = str(int(time.time() * 1000))
                        body = json.dumps(payload).encode('utf-8')
                        prefix = f"POST\n{path}\n{ts}\n".encode('utf-8')
                        sig = hmac.new(secret.encode('utf-8'), prefix + body, hashlib.sha256).hexdigest()
                        req = Request(cb_url, data=body, headers={
                            'Content-Type': 'application/json',
                            'X-Daezend-Timestamp': ts,
                            'X-Daezend-Signature': sig,
                        }, method='POST')
                        try:
                            with urlopen(req, timeout=15) as resp:
                                _ = resp.read()
                        except Exception as _cb_e:
                            logger.warning(f"⚠️ Error callback failed: {_cb_e}")
            except Exception:
                pass
            
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
            # Apply final loudness normalization for sample generation API
            try:
                audio_tensor = self.apply_loudness_normalization_tensor(audio_tensor, self.sr)
                logger.info("    - Normalized voice sample loudness to target LUFS")
            except Exception as _e_ln:
                logger.warning(f"    - Loudness normalization skipped: {_e_ln}")
            
            # Convert to MP3 bytes
            mp3_bytes = self.tensor_to_mp3_bytes(audio_tensor, self.sr, "96k")
            
            logger.info(f"✅ Voice sample generated successfully")
            return audio_tensor, mp3_bytes
            
        except Exception as e:
            logger.error(f"❌ Failed to generate voice sample: {e}")
            raise


def _init_firestore_client():
    """Initialize Firestore using explicit service account if provided.

    Prefers RUNPOD_SECRET_Firebase (JSON) to build credentials explicitly so we
    don't rely on ambient ADC in the RunPod runtime. Falls back to default client.
    """
    try:
        import os
        import json
        from google.cloud import firestore
        from google.oauth2 import service_account  # type: ignore

        sa_json_str = os.environ.get("RUNPOD_SECRET_Firebase")
        if sa_json_str:
            try:
                sa_info = json.loads(sa_json_str)
                credentials = service_account.Credentials.from_service_account_info(sa_info)
                project_id = sa_info.get("project_id")
                client = firestore.Client(project=project_id, credentials=credentials)
                return client
            except Exception as inner_e:
                logger.warning(f"⚠️ Failed to init Firestore from RUNPOD_SECRET_Firebase; falling back to default ADC: {inner_e}")

        # Fallback to default ADC
        return firestore.Client()
    except Exception as e:
        logger.error(f"❌ Failed to initialize Firestore client: {e}")
        return None


def clone_voice(
    *,
    name: str,
    audio_bytes: bytes,
    audio_format: str = "wav",
    language: str = "en",
    is_kids_voice: bool = False,
    model_type: str = "chatterbox",
    user_id: str = "",
    profile_id: Optional[str] = None,
) -> Dict:
    """Module-level helper used by Redis worker to clone a voice and update Firestore.

    Returns a dict mirroring create_voice_clone plus Firestore update result.
    """
    try:
        # Persist bytes to a temp file
        import tempfile
        import os as _os
        tmp = tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False)
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()

        # Load models and run clone
        device = "cpu"
        vc = ChatterboxVC.from_pretrained(device)
        result = vc.create_voice_clone(audio_file_path=tmp.name, voice_name=name, metadata={
            "language": language,
            "is_kids_voice": is_kids_voice,
            "model_type": model_type,
            "user_id": user_id,
        })

        # Clean up temp file
        try:
            _os.unlink(tmp.name)
        except Exception:
            pass

        # Write Firestore voice_profiles/{docId}
        client = _init_firestore_client()
        if client and result.get("status") == "success":
            doc_id = profile_id or result.get("voice_id") or name
            doc_ref = client.collection("voice_profiles").document(doc_id)
            from google.cloud.firestore import SERVER_TIMESTAMP  # type: ignore
            base_path = f"audio/voices/{language}{'/kids' if is_kids_voice else ''}"
            # Match recorded file extension returned by create_voice_clone
            rec_ext = _os.path.splitext(result.get('recorded_audio_path', ''))[1] or '.wav'
            doc_ref.set({
                "userId": user_id,
                "name": name,
                "language": language,
                "isKidsVoice": is_kids_voice,
                "status": "ready",
                "samplePath": f"{base_path}/samples/{result.get('voice_id')}_sample.mp3",
                "profilePath": f"{base_path}/profiles/{result.get('voice_id')}.npy",
                "recordedPath": f"{base_path}/recorded/{result.get('voice_id')}_recorded{rec_ext}",
                "createdAt": SERVER_TIMESTAMP,
                "updatedAt": SERVER_TIMESTAMP,
                "metadata": result.get("metadata", {}),
            }, merge=True)
            result["firestore_profile_id"] = doc_id

        return result
    except Exception as e:
        logger.exception("clone_voice failed")
        return {"status": "error", "error": str(e)}


