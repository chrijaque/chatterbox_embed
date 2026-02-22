from pathlib import Path
import os
import tempfile
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Dict, Union, Any
from datetime import datetime

import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torchaudio
import numpy as np

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen, VoiceProfile
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond

# Import from new modules
from .utils import PYDUB_AVAILABLE, _get_git_sha, _maybe_log_seg_levels, REPO_ID
from .text.normalization import punc_norm
from .storage.bucket_resolver import resolve_bucket_name, is_r2_bucket
from .storage.r2_storage import upload_to_r2, download_from_r2
from .audio.conversion import tensor_to_mp3_bytes, tensor_to_audiosegment, tensor_to_wav_bytes
from .conditionals import Conditionals
from .chunking import ContentType, ChunkInfo, SmartChunker, AdvancedTextSanitizer
from .parameters import AdaptiveParameterManager
from .quality import QualityScore, ChunkQualityAnalyzer
from .stitching import AdvancedStitcher

# Configure logging
logger = logging.getLogger(__name__)
CHATTERBOX_RUNTIME_VERSION = "2026-02-22-testA-750-350"


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()
        
        # Initialize smart chunking and adaptive parameters
        self.smart_chunker = SmartChunker()
        self.param_manager = AdaptiveParameterManager()
        
        # Initialize advanced features
        self.text_sanitizer = AdvancedTextSanitizer()
        self.quality_analyzer = ChunkQualityAnalyzer()
        self.advanced_stitcher = AdvancedStitcher()
        
        # Parallel processing settings
        self.max_parallel_workers = 1  # Disabled for single-user processing
        self.enable_parallel_processing = False  # Disabled as requested
        # Quality Analysis toggle (disable to remove overhead)
        # Default to disabled; can be enabled via env CHATTERBOX_ENABLE_QUALITY_ANALYSIS=true
        self.enable_quality_analysis = os.getenv("CHATTERBOX_ENABLE_QUALITY_ANALYSIS", "false").lower() == "true"
        self.experiment_config = self._init_experiment_config()
        if self.experiment_config.get("enabled", False):
            logger.warning(
                "🧪 Experiment instrumentation active | version=2026-02-18b name=%s",
                self.experiment_config.get("name", "default"),
            )
        
        # Phase 1: Conditional Caching
        self._cached_conditionals = None
        self._cached_voice_profile_path = None
        self._cached_exaggeration = None
        self._cached_saved_voice_path = None
        self._cached_audio_prompt_path = None
        self._conditional_cache_hits = 0
        self._conditional_cache_misses = 0
        
        logger.info(f"✅ ChatterboxTTS initialized successfully with conditional caching")
        logger.info(f"  - Available methods: {[m for m in dir(self) if not m.startswith('_')]}")
        
        # Debug: Check for specific methods
        expected_tts_methods = [
            'generate_tts_story',
            'generate_long_text',
            'chunk_text',
            'generate_chunks',
            'stitch_and_normalize',
            'cleanup_chunks',
            'upload_to_storage'
        ]
        
        available_methods = [m for m in dir(self) if not m.startswith('_')]
        missing_methods = [m for m in expected_tts_methods if m not in available_methods]
        
        logger.info(f"🔍 TTS Method Check:")
        logger.info(f"  - Expected methods: {expected_tts_methods}")
        logger.info(f"  - Available methods: {available_methods}")
        logger.info(f"  - Missing methods: {missing_methods}")
        
        if missing_methods:
            logger.error(f"❌ MISSING METHODS: {missing_methods}")
        else:
            logger.info(f"✅ All expected methods are available!")

        # Startup environment diagnostics
        try:
            logger.info(
                "🧪 Audio env | PYDUB_AVAILABLE=%s, loudnorm_enabled=%s, loudnorm_method=%s, ffmpeg_available=%s, git_sha=%s",
                PYDUB_AVAILABLE,
                getattr(self.advanced_stitcher, "enable_loudness_normalization", False),
                getattr(self.advanced_stitcher, "loudness_method", ""),
                self.advanced_stitcher._ffmpeg_available() if hasattr(self.advanced_stitcher, "_ffmpeg_available") else False,
                _get_git_sha(),
            )
        except Exception:
            pass

    def _env_bool(self, key: str, default: bool = False) -> bool:
        raw = os.getenv(key)
        if raw is None:
            return default
        return str(raw).strip().lower() in ("1", "true", "yes", "on")

    def _log_experiment(self, message: str, *args) -> None:
        """Promote experiment diagnostics to warning level for visibility in worker logs."""
        exp_cfg = getattr(self, "experiment_config", {}) or {}
        if exp_cfg.get("enabled", False):
            logger.warning(message, *args)
        else:
            logger.info(message, *args)

    def _init_experiment_config(self) -> Dict[str, Any]:
        """Build experiment toggles so we can isolate one theory per run."""
        cfg: Dict[str, Any] = {
            "enabled": self._env_bool("CHATTERBOX_EXPERIMENT_MODE", False),
            "name": os.getenv("CHATTERBOX_EXPERIMENT_NAME", "default"),
            "issue_only_mode": self._env_bool("CHATTERBOX_EXPERIMENT_ISSUE_ONLY_MODE", False),
            "enable_token_guards": self._env_bool("CHATTERBOX_EXPERIMENT_ENABLE_TOKEN_GUARDS", True),
            "enable_silence_gate": self._env_bool("CHATTERBOX_EXPERIMENT_ENABLE_SILENCE_GATE", True),
            "enable_qa_regen": self._env_bool("CHATTERBOX_EXPERIMENT_ENABLE_QA_REGEN", True),
            "enable_retry_param_drift": self._env_bool("CHATTERBOX_EXPERIMENT_ENABLE_RETRY_PARAM_DRIFT", True),
            "enable_adaptive_voice_params": self._env_bool("CHATTERBOX_EXPERIMENT_ENABLE_ADAPTIVE_VOICE_PARAMS", True),
            "verbose_chunk_logs": self._env_bool("CHATTERBOX_EXPERIMENT_VERBOSE_CHUNK_LOGS", True),
            "show_sampling_progress": self._env_bool("CHATTERBOX_EXPERIMENT_SHOW_SAMPLING_PROGRESS", False),
            "force_adaptive_blend": None,
        }

        force_blend_raw = os.getenv("CHATTERBOX_EXPERIMENT_FORCE_ADAPTIVE_BLEND")
        if force_blend_raw is not None and str(force_blend_raw).strip() != "":
            try:
                cfg["force_adaptive_blend"] = max(0.0, min(1.0, float(force_blend_raw)))
            except Exception:
                logger.warning(
                    "⚠️ Invalid CHATTERBOX_EXPERIMENT_FORCE_ADAPTIVE_BLEND=%s (expected float 0..1). Ignoring.",
                    force_blend_raw,
                )
                cfg["force_adaptive_blend"] = None

        # Issue-only mode strips non-essential process heuristics so we can
        # isolate the real failure path.
        if not cfg["enabled"]:
            cfg["name"] = "off"
            cfg["issue_only_mode"] = False
            cfg["enable_token_guards"] = True
            cfg["enable_silence_gate"] = True
            cfg["enable_qa_regen"] = True
            cfg["enable_retry_param_drift"] = True
            cfg["enable_adaptive_voice_params"] = True
            cfg["force_adaptive_blend"] = None
        elif cfg["issue_only_mode"]:
            cfg["enable_retry_param_drift"] = False
            cfg["enable_adaptive_voice_params"] = False
            cfg["enable_qa_regen"] = False

        self._log_experiment(
            "🧪 Experiment config | enabled=%s name=%s issue_only=%s token_guards=%s silence_gate=%s qa_regen=%s retry_param_drift=%s adaptive_voice_params=%s force_blend=%s",
            cfg["enabled"],
            cfg["name"],
            cfg["issue_only_mode"],
            cfg["enable_token_guards"],
            cfg["enable_silence_gate"],
            cfg["enable_qa_regen"],
            cfg["enable_retry_param_drift"],
            cfg["enable_adaptive_voice_params"],
            cfg["force_adaptive_blend"],
        )
        return cfg

    def _get_or_prepare_conditionals(self, voice_profile_path: str = None,
                                   saved_voice_path: str = None,
                                   audio_prompt_path: str = None,
                                   exaggeration: float = 0.5) -> Conditionals:
        """
        Get cached conditionals or prepare new ones.
        This is the core optimization that eliminates redundant conditional preparation.
        
        Args:
            voice_profile_path: Path to voice profile (.npy)
            saved_voice_path: Path to saved voice embedding
            audio_prompt_path: Path to audio prompt file
            exaggeration: Voice exaggeration factor
            
        Returns:
            Conditionals object ready for use
        """
        # Determine cache key based on input parameters
        if voice_profile_path:
            cache_key = ('voice_profile', voice_profile_path, exaggeration)
            current_path = voice_profile_path
            current_type = 'voice_profile'
        elif saved_voice_path and audio_prompt_path:
            cache_key = ('saved_voice', saved_voice_path, audio_prompt_path, exaggeration)
            current_path = f"{saved_voice_path}+{audio_prompt_path}"
            current_type = 'saved_voice'
        elif audio_prompt_path:
            cache_key = ('audio_prompt', audio_prompt_path, exaggeration)
            current_path = audio_prompt_path
            current_type = 'audio_prompt'
        else:
            raise ValueError("Must provide one of: voice_profile_path, (saved_voice_path + audio_prompt_path), or audio_prompt_path")
        
        # Check if we have valid cached conditionals
        if (self._cached_conditionals is not None and 
            cache_key == self._get_cache_key()):
            
            self._conditional_cache_hits += 1
            logger.debug(f"♻️ Conditional cache HIT ({self._conditional_cache_hits} hits, {self._conditional_cache_misses} misses)")
            logger.debug(f"   - Reusing conditionals for: {current_type} = {current_path}")
            return self._cached_conditionals
        
        # Cache miss - prepare new conditionals
        self._conditional_cache_misses += 1
        logger.info(f"🔄 Conditional cache MISS - preparing new conditionals")
        logger.info(f"   - Type: {current_type}")
        logger.info(f"   - Path: {current_path}")
        logger.info(f"   - Exaggeration: {exaggeration}")
        
        # Prepare conditionals based on type
        if voice_profile_path:
            self.prepare_conditionals_with_voice_profile(voice_profile_path, exaggeration)
        elif saved_voice_path and audio_prompt_path:
            self.prepare_conditionals_with_saved_voice(saved_voice_path, audio_prompt_path, exaggeration)
        elif audio_prompt_path:
            self.prepare_conditionals_with_audio_prompt(audio_prompt_path, exaggeration)
        
        # Store cache key and return conditionals
        self._cached_voice_profile_path = voice_profile_path
        self._cached_saved_voice_path = saved_voice_path
        self._cached_audio_prompt_path = audio_prompt_path
        self._cached_exaggeration = exaggeration
        
        return self._cached_conditionals
    
    def _get_cache_key(self):
        """Get current cache key"""
        if self._cached_voice_profile_path:
            return ('voice_profile', self._cached_voice_profile_path, self._cached_exaggeration)
        elif self._cached_saved_voice_path and self._cached_audio_prompt_path:
            return ('saved_voice', self._cached_saved_voice_path, self._cached_audio_prompt_path, self._cached_exaggeration)
        elif self._cached_audio_prompt_path:
            return ('audio_prompt', self._cached_audio_prompt_path, self._cached_exaggeration)
        return None
    
    def prepare_conditionals_with_voice_profile(self, voice_profile_path: str, exaggeration: float = 0.5):
        """Prepare conditionals using a complete voice profile for most accurate TTS"""
        # Load the voice profile
        profile = self.load_voice_profile(voice_profile_path)
        
        # Create s3gen ref_dict from the loaded profile
        s3gen_ref_dict = dict(
            prompt_token=profile.prompt_token.to(self.device),
            prompt_token_len=profile.prompt_token_len.to(self.device),
            prompt_feat=profile.prompt_feat.to(self.device),
            prompt_feat_len=profile.prompt_feat_len,
            embedding=profile.embedding.to(self.device),
        )
        
        # For T3, we can use the prompt tokens from the profile for conditioning
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            # Use the prompt tokens from the profile, but limit to the required length
            t3_cond_prompt_tokens = profile.prompt_token[:, :plen].to(self.device)
        else:
            t3_cond_prompt_tokens = None
        
        # Use the voice encoder embedding from the profile for T3 conditioning
        if hasattr(profile, 've_embedding') and profile.ve_embedding is not None:
            ve_embed = profile.ve_embedding.to(self.device)
        else:
            raise ValueError("Voice profile missing ve_embedding")
        
        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        
        self._cached_conditionals = Conditionals(t3_cond, s3gen_ref_dict)
        # Also update self.conds for backward compatibility with generate_chunks
        self.conds = self._cached_conditionals
        logger.info(f"✅ Conditionals prepared using voice profile from {voice_profile_path}")
    
    def prepare_conditionals_with_saved_voice(self, saved_voice_path: str, prompt_audio_path: str, exaggeration=0.5):
        """Prepare conditionals using a pre-saved voice embedding for faster processing"""
        import librosa
        ## Load saved voice embedding
        saved_embedding = self.s3gen.load_voice_clone(saved_voice_path)
        
        ## Load prompt reference wav for tokens and features (still needed)
        s3gen_ref_wav, _sr = librosa.load(prompt_audio_path, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        
        # Generate mel features and tokens from prompt audio
        ref_wav_24 = torch.from_numpy(s3gen_ref_wav).float().to(self.device)
        if len(ref_wav_24.shape) == 1:
            ref_wav_24 = ref_wav_24.unsqueeze(0)
        ref_mels_24 = self.s3gen.mel_extractor(ref_wav_24).transpose(1, 2).to(self.device)
        
        # Tokenize 16khz reference for prompt tokens
        ref_16k_wav_tensor = torch.from_numpy(ref_16k_wav).float().to(self.device)[None, ]
        ref_speech_tokens, ref_speech_token_lens = self.s3gen.tokenizer(ref_16k_wav_tensor)
        
        # Make sure mel_len = 2 * stoken_len
        if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
            ref_speech_tokens = ref_speech_tokens[:, :ref_mels_24.shape[1] // 2]
            # Ensure tensor is not in inference mode before in-place modification
            ref_speech_token_lens = ref_speech_token_lens.clone().detach()
            ref_speech_token_lens[0] = ref_speech_tokens.shape[1]
        
        # Create s3gen ref_dict with saved embedding
        s3gen_ref_dict = dict(
            prompt_token=ref_speech_tokens.to(self.device),
            prompt_token_len=ref_speech_token_lens,
            prompt_feat=ref_mels_24,
            prompt_feat_len=None,
            embedding=saved_embedding,  # Use the pre-saved embedding!
        )

        # Speech cond prompt tokens for T3
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding for T3 (different from CAMPPlus)
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        
        self._cached_conditionals = Conditionals(t3_cond, s3gen_ref_dict)
        # Also update self.conds for backward compatibility with generate_chunks
        self.conds = self._cached_conditionals
        logger.info(f"✅ Conditionals prepared using saved voice from {saved_voice_path}")

    def prepare_conditionals_with_audio_prompt(self, wav_fpath, exaggeration=0.5):
        """Prepare conditionals using an audio prompt file"""
        import librosa
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self._cached_conditionals = Conditionals(t3_cond, s3gen_ref_dict)
        # Also update self.conds for backward compatibility with generate_chunks
        self.conds = self._cached_conditionals
        logger.info(f"✅ Conditionals prepared using audio prompt from {wav_fpath}")
    
    def clear_conditional_cache(self):
        """Clear the conditional cache to free memory"""
        logger.info("🧹 Clearing conditional cache")
        self._cached_conditionals = None
        self._cached_voice_profile_path = None
        self._cached_saved_voice_path = None
        self._cached_audio_prompt_path = None
        self._cached_exaggeration = None
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("🧹 GPU cache cleared")
    
    def get_conditional_cache_stats(self):
        """Get conditional cache statistics"""
        total_requests = self._conditional_cache_hits + self._conditional_cache_misses
        hit_rate = (self._conditional_cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self._conditional_cache_hits,
            'misses': self._conditional_cache_misses,
            'total_requests': total_requests,
            'hit_rate_percent': hit_rate,
            'cache_size': 1 if self._cached_conditionals is not None else 0
        }

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        try:
            ve.to(device).eval()
        except Exception as e:
            logger.warning(f"Failed to move VoiceEncoder to device '{device}': {e}. Falling back to CPU.")
            device = "cpu"
            ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        try:
            t3.to(device).eval()
        except Exception as e:
            logger.warning(f"Failed to move T3 to device '{device}': {e}. Falling back to CPU.")
            device = "cpu"
            t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        try:
            s3gen.to(device).eval()
        except Exception as e:
            logger.warning(f"Failed to move S3Gen to device '{device}': {e}. Falling back to CPU.")
            device = "cpu"
            s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    def save_voice_clone(self, audio_file_path: str, save_path: str):
        """Save a voice embedding from an audio file for fast reuse"""
        import librosa
        ref_wav, sr = librosa.load(audio_file_path, sr=None)
        ref_wav = torch.from_numpy(ref_wav).float()
        self.s3gen.save_voice_clone(ref_wav, sr, save_path)
        print(f"Voice clone saved to {save_path}")

    def save_voice_profile(self, audio_file_path: str, save_path: str):
        """Save a complete voice profile including embedding, prompt features, and tokens for more accurate TTS"""
        import librosa
        ref_wav, sr = librosa.load(audio_file_path, sr=None)
        ref_wav = torch.from_numpy(ref_wav).float()
        
        # Get the full reference dictionary from s3gen
        ref_dict = self.s3gen.embed_ref(ref_wav, sr, device=self.device)
        
        # Also compute voice encoder embedding for T3
        ref_16k_wav = librosa.resample(ref_wav.numpy(), orig_sr=sr, target_sr=S3_SR)
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)
        
        # Create and save the complete voice profile
        profile = VoiceProfile(
            embedding=ref_dict["embedding"],  # CAMPPlus embedding for S3Gen
            prompt_feat=ref_dict["prompt_feat"],
            prompt_feat_len=ref_dict.get("prompt_feat_len"),
            prompt_token=ref_dict["prompt_token"],
            prompt_token_len=ref_dict["prompt_token_len"],
        )
        
        # Add voice encoder embedding to the profile data
        profile_data = {
            "embedding": profile.embedding.detach().cpu().numpy(),
            "ve_embedding": ve_embed.detach().cpu().numpy(),  # Voice encoder embedding for T3
        }
        if profile.prompt_feat is not None:
            profile_data["prompt_feat"] = profile.prompt_feat.detach().cpu().numpy()
        if profile.prompt_feat_len is not None:
            profile_data["prompt_feat_len"] = profile.prompt_feat_len
        if profile.prompt_token is not None:
            profile_data["prompt_token"] = profile.prompt_token.detach().cpu().numpy()
        if profile.prompt_token_len is not None:
            profile_data["prompt_token_len"] = profile.prompt_token_len.detach().cpu().numpy()
        
        import numpy as np
        np.save(save_path, profile_data)
        print(f"✅ Full voice profile saved to {save_path}")

    def load_voice_clone(self, path: str):
        """Load a pre-saved voice embedding"""
        return self.s3gen.load_voice_clone(path)

    def load_voice_profile(self, path: str):
        """Load a complete voice profile with custom format including voice encoder embedding"""
        import numpy as np
        logger.info(f"📂 Loading voice profile from {path}")
        data = np.load(path, allow_pickle=True).item()
        
        # Log what keys are present for debugging
        logger.info(f"  - Profile keys: {list(data.keys())}")
        
        # Create VoiceProfile object
        profile = VoiceProfile(
            embedding=torch.from_numpy(data["embedding"]).to(self.device),
            prompt_feat=torch.from_numpy(data["prompt_feat"]).to(self.device) if "prompt_feat" in data else None,
            prompt_feat_len=data.get("prompt_feat_len"),
            prompt_token=torch.from_numpy(data["prompt_token"]).to(self.device) if "prompt_token" in data else None,
            prompt_token_len=torch.from_numpy(data["prompt_token_len"]).to(self.device) if "prompt_token_len" in data else None,
        )
        
        # Add voice encoder embedding if present
        if "ve_embedding" in data:
            profile.ve_embedding = torch.from_numpy(data["ve_embedding"]).to(self.device)
            logger.info(f"  - VoiceEncoder embedding loaded: shape={profile.ve_embedding.shape}")
        else:
            logger.warning(f"  - No VoiceEncoder embedding found in profile")
            profile.ve_embedding = None
        
        logger.info(f"✅ Voice profile loaded successfully")
        logger.info(f"  - Embedding shape: {profile.embedding.shape}")
        logger.info(f"  - Prompt token shape: {profile.prompt_token.shape if profile.prompt_token is not None else None}")
        logger.info(f"  - Prompt feat shape: {profile.prompt_feat.shape if profile.prompt_feat is not None else None}")
        
        return profile

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        # Resolve/validate requested device, fallback to CPU when unavailable
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available or visible. Falling back to CPU.")
            device = "cpu"
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def generate(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        audio_prompt_path=None,
        saved_voice_path=None,
        voice_profile_path=None,
        exaggeration=0.5,
        cfg_weight=0.3,
        temperature=0.6,
    ):
        # Use conditional caching/preparation only when necessary. If conditionals
        # are already prepared, reuse them to avoid heavy recomputation per chunk.
        if self.conds is None:
            if voice_profile_path:
                self.prepare_conditionals_with_voice_profile(voice_profile_path, exaggeration=exaggeration)
            elif saved_voice_path and audio_prompt_path:
                self.prepare_conditionals_with_saved_voice(saved_voice_path, audio_prompt_path, exaggeration=exaggeration)
            elif audio_prompt_path:
                self.prepare_conditionals_with_audio_prompt(audio_prompt_path, exaggeration=exaggeration)
            else:
                raise RuntimeError("Conditionals are not prepared. Provide voice_profile_path, (saved_voice_path + audio_prompt_path), or audio_prompt_path.")
        
        logger.info("✅ Conditionals prepared for TTS generation")

        # Norm and tokenize text
        # Text is assumed sanitized earlier in the pipeline; avoid re-normalizing here
        # to prevent double processing and preserve pacing.
        text = text
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=750,  # Test A default: reduce long-tail degeneration risk
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            
            speech_tokens = speech_tokens[speech_tokens < 6561]
            token_count = int(speech_tokens.numel()) if speech_tokens is not None else 0
            self._log_experiment(
                "🧪 T3 token diagnostics | mode=single_generate token_count=%s",
                token_count,
            )
            if self.experiment_config.get("enable_token_guards", True):
                if speech_tokens is None or speech_tokens.numel() == 0:
                    raise RuntimeError("T3 produced empty speech token sequence (likely early EOS)")
                if token_count < 8:
                    raise RuntimeError(f"T3 produced too few speech tokens after filtering ({token_count} < 8)")

            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
        return torch.from_numpy(wav).unsqueeze(0)

    def _generate_with_prepared_conditionals(
        self,
        text,
        conditionals: Conditionals,
        exaggeration=None,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        cfg_weight=0.3,
        temperature=0.6,
        max_new_tokens_override: Optional[int] = None,
        return_token_count: bool = False,
        diagnostics_chunk_id: Optional[int] = None,
    ):
        """
        Generate audio with provided conditionals, optionally overriding exaggeration per chunk.
        """
        # Validate conditionals
        if conditionals is None:
            raise RuntimeError("Conditionals must be provided to _generate_with_prepared_conditionals.")
        
        # Create a deep copy of conditionals to avoid modifying the original
        import copy
        chunk_conditionals = copy.deepcopy(conditionals)
        
        # Optional per-chunk emotion override without re-prepping conditionals
        # This allows AdaptiveParameterManager to vary 'exaggeration' per chunk.
        if exaggeration is not None and hasattr(chunk_conditionals, "t3") and hasattr(chunk_conditionals.t3, "emotion_adv"):
            try:
                chunk_conditionals.t3.emotion_adv = (torch.ones(1, 1, 1, device=self.device) * float(exaggeration))
            except Exception as e:
                logger.warning(f"Could not set per-chunk emotion_adv: {e}")

        # Text is assumed sanitized earlier in the pipeline
        text = text
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=chunk_conditionals.t3,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens_override or 750,  # Test A default cap
                show_progress=(
                    bool(self.experiment_config.get("show_sampling_progress", False))
                    if (self.experiment_config or {}).get("enabled", False)
                    else True
                ),
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            
            speech_tokens = speech_tokens[speech_tokens < 6561]
            token_count = int(speech_tokens.numel()) if speech_tokens is not None else 0
            if diagnostics_chunk_id is not None:
                self._log_experiment(
                    "🧪 Chunk %s token diagnostics | token_count=%s",
                    diagnostics_chunk_id,
                    token_count,
                )
            else:
                self._log_experiment("🧪 T3 token diagnostics | token_count=%s", token_count)
            if self.experiment_config.get("enable_token_guards", True):
                if speech_tokens is None or speech_tokens.numel() == 0:
                    raise RuntimeError("T3 produced empty speech token sequence (likely early EOS)")
                if token_count < 8:
                    raise RuntimeError(f"T3 produced too few speech tokens after filtering ({token_count} < 8)")

            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=chunk_conditionals.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        if return_token_count:
            return wav_tensor, token_count
        return wav_tensor

    def chunk_text(self, text: str, max_chars: int = 500) -> List[ChunkInfo]:
        """
        Smart text chunking with advanced sanitization and content awareness.

        :param text: Full story text
        :param max_chars: Maximum number of characters per chunk
        :return: List of ChunkInfo objects with full analysis
        """
        # Step 1: Detect story breaks before sanitization
        story_break_positions = []
        if '⁂' in text:
            # Find all story break positions in the original text
            start_pos = 0
            while True:
                pos = text.find('⁂', start_pos)
                if pos == -1:
                    break
                story_break_positions.append(pos)
                start_pos = pos + 1
            logger.info(f"🎭 Found {len(story_break_positions)} story break(s) at positions: {story_break_positions}")
        
        # Step 2: Advanced text sanitization
        logger.debug(f"🧹 Applying advanced text sanitization to {len(text)} characters")
        sanitized_text = self.text_sanitizer.deep_clean(text)
        
        if len(sanitized_text) != len(text):
            logger.debug(f"🧹 Text sanitization: {len(text)} → {len(sanitized_text)} characters")
        
        # Step 3: Smart chunking with optional lead-sentence split
        target_chars = int(max_chars * 0.8)  # Target 80% of max for better quality

        # Simple chunking without complex opener logic
        chunk_infos = self.smart_chunker.smart_chunk(sanitized_text, target_chars, max_chars)
        
        # Step 4: Mark chunks that should have story break pauses
        if story_break_positions:
            self._mark_story_break_chunks(chunk_infos, story_break_positions, text)

        # Ensure chunk_infos is always defined and not empty
        if not chunk_infos:
            logger.warning("⚠️ No chunks generated, falling back to basic chunking")
            chunk_infos = self.smart_chunker.smart_chunk(sanitized_text, target_chars, max_chars)
        
        # Log detailed chunk analysis
        if chunk_infos:
            total_chars = sum(chunk.char_count for chunk in chunk_infos)
            avg_chars = total_chars / len(chunk_infos)
            complexity_scores = [chunk.complexity_score for chunk in chunk_infos]
            avg_complexity = sum(complexity_scores) / len(complexity_scores)
            
            logger.debug(f"📊 Advanced Chunk Analysis:")
            logger.debug(f"   - Total chunks: {len(chunk_infos)}")
            logger.debug(f"   - Avg chars/chunk: {avg_chars:.1f}")
            logger.debug(f"   - Avg complexity: {avg_complexity:.1f}/10")
            logger.debug(f"   - Content distribution: {self.smart_chunker._get_content_type_distribution(chunk_infos)}")
            
            # Sanitization impact analysis
            original_problematic = sum(1 for c in text if ord(c) > 127)
            sanitized_problematic = sum(1 for c in sanitized_text if ord(c) > 127)
            if original_problematic > sanitized_problematic:
                logger.debug(f"   - Problematic chars removed: {original_problematic - sanitized_problematic}")
        
        return chunk_infos
    
    def _mark_story_break_chunks(self, chunk_infos: List[ChunkInfo], story_break_positions: List[int], original_text: str) -> None:
        """
        Mark chunks that should have story break pauses based on original text positions.
        
        :param chunk_infos: List of chunk info objects
        :param story_break_positions: Positions of '⁂' symbols in original text
        :param original_text: Original text before sanitization
        """
        if not story_break_positions or not chunk_infos:
            return
        
        # Calculate cumulative character positions for each chunk in original text
        cumulative_chars = 0
        chunk_boundaries = []
        
        for chunk in chunk_infos:
            # Find this chunk's text in the original text (approximate)
            chunk_start = cumulative_chars
            chunk_end = cumulative_chars + chunk.char_count
            chunk_boundaries.append((chunk_start, chunk_end))
            cumulative_chars += chunk.char_count
        
        # Mark chunks that contain story breaks
        for break_pos in story_break_positions:
            for i, (start, end) in enumerate(chunk_boundaries):
                if start <= break_pos <= end:
                    chunk_infos[i].has_story_break = True
                    logger.info(f"🎭 Marked chunk {i} for story break pause")
                    break
        
        # Log summary
        story_break_chunks = sum(1 for chunk in chunk_infos if chunk.has_story_break)
        logger.info(f"🎭 Story break analysis: {len(story_break_positions)} breaks → {story_break_chunks} chunks marked")
    
    def simple_sentence_split(self, text: str) -> List[str]:
        """
        Simple sentence splitting using punctuation marks.
        
        :param text: Text to split
        :return: List of sentences
        """
        logger.info(f"📝 Simple sentence splitting for text length: {len(text)}")
        
        # Simple sentence endings
        sentence_endings = ['.', '!', '?', '\n']
        
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in sentence_endings:
                # Clean up the sentence
                sentence = current_sentence.strip()
                if sentence:
                    sentences.append(sentence)
                current_sentence = ""
        
        # Add any remaining text as a sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        logger.info(f"✅ Split into {len(sentences)} sentences")
        return sentences
    
    def _generate_single_chunk_with_quality(self, chunk_info: ChunkInfo, voice_profile_path: str,
                                          pre_prepared_conditionals: Conditionals = None) -> Tuple[str, QualityScore]:
        """Generate a single chunk with quality analysis and retry logic"""
        adaptive_params = self.param_manager.get_adaptive_parameters(chunk_info)
        max_retries = 3
        
        # Fast path: QA disabled -> single attempt, no analysis
        if not getattr(self, "enable_quality_analysis", True):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if pre_prepared_conditionals:
                    audio_tensor = self._generate_with_prepared_conditionals(
                        text=chunk_info.text,
                        conditionals=pre_prepared_conditionals,
                        exaggeration=adaptive_params.get("exaggeration"),
                        temperature=adaptive_params.get("temperature", 0.6),
                        cfg_weight=adaptive_params.get("cfg_weight", 0.3),
                        repetition_penalty=adaptive_params.get("repetition_penalty", 1.2),
                        min_p=adaptive_params.get("min_p", 0.05),
                        top_p=adaptive_params.get("top_p", 1.0),
                    )
                else:
                    audio_tensor = self.generate(
                        text=chunk_info.text,
                        voice_profile_path=voice_profile_path,
                        temperature=adaptive_params["temperature"],
                        exaggeration=adaptive_params["exaggeration"],
                        cfg_weight=adaptive_params["cfg_weight"],
                        repetition_penalty=adaptive_params["repetition_penalty"],
                        min_p=adaptive_params["min_p"],
                        top_p=adaptive_params["top_p"]
                    )

                temp_wav = tempfile.NamedTemporaryFile(suffix=f"_chunk_{chunk_info.id}.wav", delete=False)
                temp_wav_path = temp_wav.name
                temp_wav.close()
                torchaudio.save(temp_wav_path, audio_tensor, self.sr)
                return temp_wav_path, QualityScore(
                    overall_score=100.0,
                    issues=[],
                    duration=audio_tensor.shape[-1] / self.sr,
                    silence_ratio=0.0,
                    peak_db=0.0,
                    rms_db=0.0,
                    should_regenerate=False
                )
            except Exception as e:
                logger.warning(f"⚠️ Chunk {chunk_info.id} generation failed with QA disabled: {e}")
                raise

        for attempt in range(max_retries):
            try:
                # Clear GPU cache before each attempt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Generate audio
                if pre_prepared_conditionals:
                    audio_tensor = self._generate_with_prepared_conditionals(
                        text=chunk_info.text,
                        conditionals=pre_prepared_conditionals,
                        exaggeration=adaptive_params.get("exaggeration"),
                        temperature=adaptive_params.get("temperature", 0.6),
                        cfg_weight=adaptive_params.get("cfg_weight", 0.3),
                        repetition_penalty=adaptive_params.get("repetition_penalty", 1.2),
                        min_p=adaptive_params.get("min_p", 0.05),
                        top_p=adaptive_params.get("top_p", 1.0),
                    )
                else:
                    audio_tensor = self.generate(
                        text=chunk_info.text,
                        voice_profile_path=voice_profile_path,
                        temperature=adaptive_params["temperature"],
                        exaggeration=adaptive_params["exaggeration"],
                        cfg_weight=adaptive_params["cfg_weight"],
                        repetition_penalty=adaptive_params["repetition_penalty"],
                        min_p=adaptive_params["min_p"],
                        top_p=adaptive_params["top_p"]
                    )

                # Save to temporary file for quality analysis
                temp_wav = tempfile.NamedTemporaryFile(suffix=f"_chunk_{chunk_info.id}.wav", delete=False)
                temp_wav_path = temp_wav.name
                temp_wav.close()
                torchaudio.save(temp_wav_path, audio_tensor, self.sr)
                
                # Analyze quality
                quality_score = self.quality_analyzer.analyze_chunk_quality(temp_wav_path, chunk_info)
                
                # Check if regeneration is needed
                if not quality_score.should_regenerate or attempt == max_retries - 1:
                    return temp_wav_path, quality_score
                
                # Clean up failed attempt
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)
                
                logger.warning(f"⚠️ Chunk {chunk_info.id} attempt {attempt + 1}/{max_retries} failed quality check: {quality_score.issues}")
                
            except Exception as e:
                logger.error(f"❌ Chunk {chunk_info.id} generation attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise
        
        # Should never reach here, but just in case
        raise RuntimeError(f"Failed to generate chunk {chunk_info.id} after {max_retries} attempts")
    
    def generate_chunks_parallel(self, chunk_infos: List[ChunkInfo], voice_profile_path: str,
                               pre_prepared_conditionals: Conditionals = None) -> List[Tuple[str, QualityScore]]:
        """Generate chunks in parallel with quality analysis"""
        logger.info(f"🚀 Starting parallel chunk generation ({self.max_parallel_workers} workers)")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(self._generate_single_chunk_with_quality, chunk_info, voice_profile_path, pre_prepared_conditionals): chunk_info
                for chunk_info in chunk_infos
            }
            
            # Collect results as they complete
            for future in future_to_chunk:
                chunk_info = future_to_chunk[future]
                try:
                    wav_path, quality_score = future.result()
                    results.append((chunk_info.id, wav_path, quality_score))
                    
                    logger.info(f"✅ Chunk {chunk_info.id + 1} | "
                               f"Quality: {quality_score.overall_score:.1f}/100 | "
                               f"Duration: {quality_score.duration:.2f}s | "
                               f"Issues: {len(quality_score.issues)}")
                    
                except Exception as e:
                    logger.error(f"❌ Chunk {chunk_info.id + 1} failed completely: {e}")
                    raise
        
        # Sort results by chunk ID to maintain order
        results.sort(key=lambda x: x[0])
        return [(wav_path, quality_score) for _, wav_path, quality_score in results]
    
    def generate_chunks(
        self,
        chunk_infos: List[ChunkInfo],
        voice_profile_path: str,
        base_temperature: float = 0.6,
        base_exaggeration: float = 0.5,
        base_cfg_weight: float = 0.3,
        *,
        adaptive_voice_param_blend: float = 0.2,
    ) -> List[str]:
        """
        Advanced chunk generation with parallel processing and quality analysis.
        Now optimized with conditional caching for improved performance.

        :param chunk_infos: List of ChunkInfo objects with full analysis
        :param voice_profile_path: Path to the voice profile (.npy)
        :param base_temperature: Base temperature (will be adapted per chunk)
        :param base_exaggeration: Base exaggeration (will be adapted per chunk)
        :param base_cfg_weight: Base CFG weight (will be adapted/blended per chunk)
        :param adaptive_voice_param_blend: Blend factor for adaptive per-chunk voice params (temp/exag/cfg).
            - 0.2 (default): mostly stable base voice with light adaptive movement
            - 1.0: fully adaptive per chunk
            - 0.0: fully respect base_temperature/base_exaggeration/base_cfg_weight for ALL chunks
        :return: List of file paths to temporary WAV files
        """
        generation_start = time.time()
        try:
            adaptive_voice_param_blend = float(adaptive_voice_param_blend)
        except Exception:
            adaptive_voice_param_blend = 1.0
        adaptive_voice_param_blend = max(0.0, min(1.0, adaptive_voice_param_blend))
        exp_cfg = self.experiment_config or {}
        force_blend = exp_cfg.get("force_adaptive_blend")
        if force_blend is not None:
            adaptive_voice_param_blend = float(force_blend)
        
        # Prepare conditionals once and reuse for all chunks to avoid recomputation
        logger.info(f"🎯 Preparing conditionals once for {len(chunk_infos)} chunks")
        try:
            # If a full voice profile (.npy) is provided, use the dedicated loader
            if isinstance(voice_profile_path, str) and voice_profile_path.lower().endswith('.npy'):
                self.prepare_conditionals_with_voice_profile(voice_profile_path, exaggeration=base_exaggeration)
            else:
                # Otherwise treat it as an audio prompt path
                self.prepare_conditionals_with_audio_prompt(voice_profile_path, exaggeration=base_exaggeration)
        except Exception:
            logger.exception("❌ Failed to prepare conditionals")
            raise

        logger.info(
            f"🔄 Using sequential processing with adaptive parameters for {len(chunk_infos)} chunks "
            f"(adaptive_voice_param_blend={adaptive_voice_param_blend:.2f})"
        )
        wav_paths = []
        quality_scores: List[QualityScore] = []

        # QA/retry configuration (protect against "silent chunk" failures)
        try:
            max_chunk_regen_attempts = int(os.getenv("CHATTERBOX_CHUNK_REGEN_ATTEMPTS", "4"))
        except Exception:
            max_chunk_regen_attempts = 4
        max_chunk_regen_attempts = max(1, min(6, max_chunk_regen_attempts))
        fail_on_bad_chunk = os.getenv("CHATTERBOX_FAIL_ON_BAD_CHUNK", "true").lower() == "true"
        silence_peak_threshold = 1e-6
        silence_rms_threshold = 1e-7
        self._log_experiment(
            "🧪 Chunk regen config | quality_analysis=%s fail_on_bad_chunk=%s max_attempts=%d silence_gate=(peak<%.1e,rms<%.1e)",
            getattr(self, "enable_quality_analysis", False),
            fail_on_bad_chunk,
            max_chunk_regen_attempts,
            silence_peak_threshold,
            silence_rms_threshold,
        )
        self._log_experiment(
            "🧪 Experiment run | enabled=%s name=%s issue_only=%s token_guards=%s silence_gate=%s qa_regen=%s retry_param_drift=%s adaptive_voice_params=%s blend=%.2f",
            exp_cfg.get("enabled", False),
            exp_cfg.get("name", "default"),
            exp_cfg.get("issue_only_mode", False),
            exp_cfg.get("enable_token_guards", True),
            exp_cfg.get("enable_silence_gate", True),
            exp_cfg.get("enable_qa_regen", True),
            exp_cfg.get("enable_retry_param_drift", True),
            exp_cfg.get("enable_adaptive_voice_params", True),
            adaptive_voice_param_blend,
        )

        for i, chunk_info in enumerate(chunk_infos):
            logger.info(f"🎵 Generating chunk {i+1}/{len(chunk_infos)}: {chunk_info.text[:50]}...")

            # Get adaptive parameters based on chunk content, position, and complexity
            adaptive_params = self.param_manager.get_adaptive_parameters(chunk_info)
            
            # Determine the effective per-chunk parameters.
            # NOTE: adaptive_params always contains temperature/exaggeration/cfg_weight, so base_* would
            # otherwise be ignored. We blend adaptive with base to make external overrides meaningful.
            try:
                adaptive_temp = float(adaptive_params.get("temperature", base_temperature))
            except Exception:
                adaptive_temp = base_temperature
            try:
                adaptive_exag = float(adaptive_params.get("exaggeration", base_exaggeration))
            except Exception:
                adaptive_exag = base_exaggeration
            try:
                adaptive_cfg = float(adaptive_params.get("cfg_weight", base_cfg_weight))
            except Exception:
                adaptive_cfg = base_cfg_weight

            use_adaptive_voice_params = bool(exp_cfg.get("enable_adaptive_voice_params", True))
            if use_adaptive_voice_params:
                temp_used = (base_temperature * (1.0 - adaptive_voice_param_blend)) + (adaptive_temp * adaptive_voice_param_blend)
                exag_used = (base_exaggeration * (1.0 - adaptive_voice_param_blend)) + (adaptive_exag * adaptive_voice_param_blend)
                cfg_used = (base_cfg_weight * (1.0 - adaptive_voice_param_blend)) + (adaptive_cfg * adaptive_voice_param_blend)
            else:
                temp_used = float(base_temperature)
                exag_used = float(base_exaggeration)
                cfg_used = float(base_cfg_weight)

            # Log effective params for visibility (first chunk + occasional sampling)
            if chunk_info.is_first_chunk or (i % 8 == 0):
                logger.info(
                    f"🎛️ Chunk params (effective): temp={temp_used:.2f}, exag={exag_used:.2f}, cfg={cfg_used:.2f} "
                    f"| base(temp={base_temperature:.2f}, exag={base_exaggeration:.2f}, cfg={base_cfg_weight:.2f}) "
                    f"| adaptive(temp={adaptive_temp:.2f}, exag={adaptive_exag:.2f}, cfg={adaptive_cfg:.2f}) "
                    f"| blend={adaptive_voice_param_blend:.2f}"
                )

            # Use adaptive parameters for varied narration
            # Generate + (optional) QA gate with retries to prevent silent chunks.
            temp_wav = tempfile.NamedTemporaryFile(suffix=f"_chunk_{chunk_info.id}.wav", delete=False)
            temp_wav_path = temp_wav.name
            temp_wav.close()

            # Base params for this chunk (we may adjust during retries)
            rep_pen = float(adaptive_params.get("repetition_penalty", 1.2))
            min_p = float(adaptive_params.get("min_p", 0.05))
            top_p = float(adaptive_params.get("top_p", 1.0))

            last_qs: Optional[QualityScore] = None
            for attempt in range(1, max_chunk_regen_attempts + 1):
                # Stabilize on retries: reduce randomness, increase adherence.
                # This specifically targets cases where the model emits mostly silence or low-energy output.
                if attempt == 1:
                    temp_try = temp_used
                    exag_try = exag_used
                    cfg_try = cfg_used
                else:
                    if exp_cfg.get("enable_retry_param_drift", True):
                        temp_try = max(0.5, temp_used - (0.08 * (attempt - 1)))
                        cfg_try = min(0.8, cfg_used + (0.08 * (attempt - 1)))
                        exag_try = max(0.1, exag_used - (0.05 * (attempt - 1)))
                    else:
                        temp_try = temp_used
                        cfg_try = cfg_used
                        exag_try = exag_used

                try:
                    audio_tensor, token_count = self._generate_with_prepared_conditionals(
                        text=chunk_info.text,
                        conditionals=self.conds,
                        exaggeration=exag_try,
                        temperature=temp_try,
                        cfg_weight=cfg_try,
                        repetition_penalty=rep_pen,
                        min_p=min_p,
                        top_p=top_p,
                        return_token_count=True,
                        diagnostics_chunk_id=chunk_info.id,
                    )

                    # Hard silence gate before writing WAV to disk.
                    x = audio_tensor.detach().cpu().numpy().ravel()
                    peak = float(np.max(np.abs(x))) if x.size else 0.0
                    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0
                    self._log_experiment(
                        "🧪 Chunk %s diagnostics | attempt=%s/%s token_count=%s peak=%.3e rms=%.3e",
                        chunk_info.id,
                        attempt,
                        max_chunk_regen_attempts,
                        token_count,
                        peak,
                        rms,
                    )

                    if exp_cfg.get("enable_silence_gate", True) and (
                        x.size == 0 or (peak < silence_peak_threshold and rms < silence_rms_threshold)
                    ):
                        retry_reason = (
                            f"silent_output(size={x.size}, peak={peak:.3e}, rms={rms:.3e})"
                        )
                        logger.warning(
                            f"⚠️ Chunk {chunk_info.id} retry reason={retry_reason} "
                            f"(attempt {attempt}/{max_chunk_regen_attempts})"
                        )
                        if attempt == max_chunk_regen_attempts and fail_on_bad_chunk:
                            raise RuntimeError(
                                f"Chunk {chunk_info.id} failed silence gate after "
                                f"{max_chunk_regen_attempts} attempts: {retry_reason}"
                            )
                        continue

                    torchaudio.save(temp_wav_path, audio_tensor, self.sr)

                    qa_regen_enabled = bool(exp_cfg.get("enable_qa_regen", True))
                    if not qa_regen_enabled or not getattr(self, "enable_quality_analysis", False):
                        last_qs = QualityScore(
                            overall_score=100.0,
                            issues=[],
                            duration=float(audio_tensor.shape[-1]) / float(self.sr),
                            silence_ratio=0.0,
                            peak_db=0.0,
                            rms_db=0.0,
                            should_regenerate=False,
                        )
                        break

                    qs = self.quality_analyzer.analyze_chunk_quality(temp_wav_path, chunk_info)
                    last_qs = qs

                    if not qs.should_regenerate:
                        break

                    retry_reason = f"quality_silence_gate(issues={qs.issues})"
                    logger.warning(
                        f"⚠️ Chunk {chunk_info.id} retry reason={retry_reason} "
                        f"(attempt {attempt}/{max_chunk_regen_attempts}) "
                        f"silence_ratio={qs.silence_ratio:.2f} dur={qs.duration:.2f}s "
                        f"-> retrying with temp={temp_try:.2f} cfg={cfg_try:.2f} exag={exag_try:.2f}"
                    )

                    if attempt == max_chunk_regen_attempts and fail_on_bad_chunk:
                        raise RuntimeError(
                            f"Chunk {chunk_info.id} failed QA after {max_chunk_regen_attempts} attempts: {qs.issues}"
                        )
                except Exception as e:
                    retry_reason = f"generation_error({type(e).__name__}: {e})"
                    logger.warning(
                        f"⚠️ Chunk {chunk_info.id} retry reason={retry_reason} "
                        f"(attempt {attempt}/{max_chunk_regen_attempts})"
                    )
                    if attempt == max_chunk_regen_attempts:
                        raise
                    continue

            if exp_cfg.get("verbose_chunk_logs", True):
                logger.info(
                    "🧪 Chunk %s done | qa_score=%s issues=%s",
                    chunk_info.id,
                    f"{last_qs.overall_score:.1f}" if last_qs is not None else "n/a",
                    last_qs.issues if last_qs is not None else [],
                )

            if not os.path.exists(temp_wav_path):
                raise RuntimeError(
                    f"Chunk {chunk_info.id}: no valid audio was produced after {max_chunk_regen_attempts} attempts"
                )
            wav_paths.append(temp_wav_path)
            if last_qs is not None:
                quality_scores.append(last_qs)

        # Optional summary logging
        try:
            if getattr(self, "enable_quality_analysis", False) and quality_scores:
                total_generation_time = time.time() - generation_start
                self._log_quality_analysis(chunk_infos, quality_scores, total_generation_time)
        except Exception:
            pass

        return wav_paths
    
    def _log_quality_analysis(self, chunk_infos: List[ChunkInfo], quality_scores: List[QualityScore], total_time: float):
        """Log comprehensive quality analysis results"""
        if not quality_scores:
            return
        
        # Overall quality metrics
        avg_quality = np.mean([qs.overall_score for qs in quality_scores])
        min_quality = min(qs.overall_score for qs in quality_scores)
        max_quality = max(qs.overall_score for qs in quality_scores)
        
        # Issue analysis
        all_issues = []
        for qs in quality_scores:
            all_issues.extend(qs.issues)
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Duration analysis
        total_audio_duration = sum(qs.duration for qs in quality_scores)
        avg_duration = total_audio_duration / len(quality_scores)
        
        # Content type quality breakdown
        content_quality = {}
        for chunk_info, quality_score in zip(chunk_infos, quality_scores):
            content_type = chunk_info.content_type.value
            if content_type not in content_quality:
                content_quality[content_type] = []
            content_quality[content_type].append(quality_score.overall_score)
        
        avg_content_quality = {
            ct: np.mean(scores) for ct, scores in content_quality.items()
        }
        
        # Log comprehensive analysis
        logger.info(f"🔍 Comprehensive Quality Analysis:")
        logger.info(f"   - Overall quality: {avg_quality:.1f}/100 (range: {min_quality:.1f}-{max_quality:.1f})")
        logger.info(f"   - Total audio duration: {total_audio_duration:.2f}s")
        logger.info(f"   - Average chunk duration: {avg_duration:.2f}s")
        logger.info(f"   - Generation time: {total_time:.2f}s")
        logger.info(f"   - Audio/Generation ratio: {total_audio_duration/total_time:.2f}x")
        
        if issue_counts:
            logger.info(f"   - Quality issues: {dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True))}")
        
        if len(avg_content_quality) > 1:
            logger.info(f"   - Content type quality: {avg_content_quality}")
        
        # Quality warnings
        poor_quality_chunks = [i for i, qs in enumerate(quality_scores) if qs.overall_score < 70]
        if poor_quality_chunks:
            logger.warning(f"⚠️ {len(poor_quality_chunks)} chunks have quality scores < 70: {poor_quality_chunks}")
        
        excellent_quality_chunks = [i for i, qs in enumerate(quality_scores) if qs.overall_score >= 90]
        if excellent_quality_chunks:
            logger.info(f"🌟 {len(excellent_quality_chunks)} chunks have excellent quality (≥90): {excellent_quality_chunks}")

    def stitch_and_normalize(self, wav_paths: List[str], chunk_infos: List[ChunkInfo],
                           output_path: str, pause_ms: int = 100) -> Tuple[torch.Tensor, int, float]:
        """
        Advanced stitching with smart pauses, fades, and quality-aware normalization.

        :param wav_paths: List of temporary WAV file paths
        :param chunk_infos: List of ChunkInfo objects for context-aware stitching
        :param output_path: Final path to export the combined WAV file
        :param pause_ms: Base pause duration (will be adjusted by advanced stitcher)
        :return: Tuple of (audio_tensor, sample_rate, duration_seconds)
        """
        logger.info(f"🎼 Advanced stitching with context-aware transitions")
        
        # Use advanced stitcher for intelligent combining
        return self.advanced_stitcher.advanced_stitch(wav_paths, chunk_infos, output_path)

    def cleanup_chunks(self, wav_paths: List[str]):
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

    def generate_long_text(
        self,
        text: str,
        voice_profile_path: str,
        output_path: str,
        max_chars: int = 500,
        pause_ms: int = 100,
        temperature: float = 0.6,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        pause_scale: float = 1.0,
        *,
        adaptive_voice_param_blend: float = 0.2,
    ) -> Tuple[torch.Tensor, int, Dict]:
        """
        Full TTS pipeline for long texts: chunk → generate → stitch → clean.

        :param text: Input text to synthesize
        :param voice_profile_path: Path to the voice profile (.npy)
        :param output_path: Path to save the final audio file
        :param max_chars: Maximum number of characters per chunk
        :param pause_ms: Milliseconds of silence between chunks
        :param temperature: Generation temperature
        :param exaggeration: Voice exaggeration factor
        :param cfg_weight: CFG weight for generation
        :return: Tuple of (audio_tensor, sample_rate, metadata_dict)
        """
        logger.info(f"🎵 Starting TTS processing for {len(text)} characters")
        logger.info(f"🔍 Output path: {output_path}")
        
        # Safety check for extremely long texts
        if len(text) > 13000:
            logger.warning(f"⚠️ Very long text ({len(text)} chars) - truncating to safe length")
            text = text[:13000] + "... [truncated]"
            logger.info(f"📝 Truncated text to {len(text)} characters")
        
        chunk_infos = self.chunk_text(text, max_chars)
        logger.info(f"📦 Split into {len(chunk_infos)} intelligent chunks")
        
        # Apply global pace factor to advanced stitcher
        try:
            self.advanced_stitcher.global_pause_factor = max(0.5, min(2.0, float(pause_scale)))
            logger.info(f"🕰️ Global pause scale set to {self.advanced_stitcher.global_pause_factor}")
        except Exception:
            logger.warning("⚠️ Failed to apply pause_scale; using default 1.0")
        
        wav_paths = self.generate_chunks(
            chunk_infos,
            voice_profile_path,
            temperature,
            exaggeration,
            cfg_weight,
            adaptive_voice_param_blend=adaptive_voice_param_blend,
        )
        if not wav_paths:
            raise RuntimeError("Failed to generate any audio chunks")
        
        logger.info(f"🔗 Advanced stitching {len(wav_paths)} audio chunks...")
        audio_tensor, sample_rate, total_duration = self.stitch_and_normalize(wav_paths, chunk_infos, output_path, pause_ms)

        # Apply watermark once on final audio
        try:
            final_np = audio_tensor.squeeze(0).detach().cpu().numpy()
            final_np = self.watermarker.apply_watermark(final_np, sample_rate=sample_rate)
            audio_tensor = torch.from_numpy(final_np).unsqueeze(0)
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply final watermark: {e}")
        
        self.cleanup_chunks(wav_paths)
        
        logger.info(f"✅ TTS processing completed | Duration: {total_duration:.2f}s")
        logger.info(f"🔍 Final output path: {output_path}")
        logger.info(f"🔍 Output file exists: {Path(output_path).exists()}")
        
        # Enhanced metadata with cache performance
        cache_stats = self.get_conditional_cache_stats()
        metadata = self._create_generation_metadata(chunk_infos, total_duration, sample_rate, text, max_chars, 
                                                  pause_ms, pause_scale, cache_stats)
        metadata["output_path"] = output_path
        metadata["successful_chunks"] = len(wav_paths)
        
        return audio_tensor, sample_rate, metadata


    def upload_to_storage(self, data: bytes, destination_blob_name: str, content_type: str = "application/octet-stream", metadata: dict = None) -> Optional[str]:
        """
        Upload data directly to R2 storage with metadata.
        Only R2 storage is supported.
        
        :param data: Binary data to upload
        :param destination_blob_name: Destination path in R2 (e.g., "private/users/{user_id}/stories/audio/{lang}/{story_id}/{version_id}.mp3")
        :param content_type: MIME type of the file
        :param metadata: Optional metadata to store with the file
        :return: Public URL
        """
        try:
            # Resolve region-aware bucket from metadata hints
            bucket_hint = (metadata or {}).get('bucket_name') if isinstance(metadata, dict) else None
            country_hint = (metadata or {}).get('country_code') if isinstance(metadata, dict) else None
            resolved_bucket = resolve_bucket_name(bucket_hint, country_hint)

            # Basic destination path sanitization
            dest_name = str(destination_blob_name or "").lstrip("/")

            # Only R2 is supported - verify bucket is R2
            if not is_r2_bucket(resolved_bucket):
                error_msg = f"Only R2 storage is supported. Bucket '{resolved_bucket}' is not an R2 bucket. Expected 'minstraly-storage'."
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            logger.info(f"✅ Using R2 upload for bucket: {resolved_bucket}")
            return upload_to_r2(data, dest_name, content_type, metadata)
            
        except Exception as e:
            logger.error(f"❌ Failed to upload: {e}")
            return None

    def generate_tts_story(self, text: str, voice_id: str, profile_base64: str = "", 
                           language: str = 'en', story_type: str = 'user', 
                           is_kids_voice: bool = False, metadata: Dict = None, pause_scale: float = 1.15,
                           *, user_id: str = "", story_id: str = "", profile_path: str = "", voice_name: str = "",
                           # New optional TTS parameters
                           temperature: float = None, exaggeration: float = None, 
                          cfg_weight: float = None,
                          adaptive_voice_param_blend: float = 0.2) -> Dict:
        """
        Generate TTS story with voice profile from base64.
        
        Args:
            text: Text to synthesize
            voice_id: Unique voice identifier
            profile_base64: Voice profile as base64 string
            language: Language code
            story_type: Type of story
            is_kids_voice: Whether it's a kids voice
            metadata: Optional metadata (for API compatibility)
            temperature: Generation temperature (default: 0.8)
            exaggeration: Voice exaggeration factor (default: 0.5)
            cfg_weight: CFG weight for generation (default: 0.5)
            pause_scale: Global pause scaling factor (default: 1.15)
            
        Returns:
            Dict with status, audio_data, storage_url (R2), storage_path (R2), story_type, and generation_time
        """
        import time
        import base64
        import tempfile
        import os
        
        start_time = time.time()
        
        # Extract voice_name from metadata if not provided directly
        if not voice_name and metadata and isinstance(metadata, dict) and 'voice_name' in metadata:
            voice_name = metadata['voice_name']
        
        # Fallback to voice_id if no voice_name provided
        if not voice_name:
            voice_name = voice_id
        
        logger.info(f"📚 ChatterboxTTS.generate_tts_story called")
        logger.info(f"  - text length: {len(text)}")
        logger.info(f"  - voice_id: {voice_id}")
        logger.info(f"  - voice_name: {voice_name}")
        logger.info(f"  - language: {language}")
        logger.info(f"  - story_type: {story_type}")
        logger.info(f"  - is_kids_voice: {is_kids_voice}")
        logger.info(f"  - metadata: {metadata}")
        
        try:
            # Step 1: Load voice profile from base64 or R2 storage path
            if profile_base64:
                logger.info(f"  - Step 1: Loading voice profile from base64...")
                profile_bytes = base64.b64decode(profile_base64)
            elif profile_path:
                logger.info(f"  - Step 1: Loading voice profile from storage path: {profile_path}")
                try:
                    # Resolve bucket from metadata hints (non-R2 bucket names are ignored)
                    bucket_hint = (metadata or {}).get('bucket_name') if isinstance(metadata, dict) else None
                    country_hint = (metadata or {}).get('country_code') if isinstance(metadata, dict) else None
                    resolved_bucket = resolve_bucket_name(bucket_hint, country_hint)
                    
                    # resolve_bucket_name() always returns an R2 bucket (ignores non-R2 bucket names)
                    logger.info(f"    - Using R2 download (bucket={resolved_bucket})")
                    profile_bytes = download_from_r2(profile_path)
                    if not profile_bytes:
                        raise ValueError(f"Failed to download profile from R2: {profile_path}")
                    logger.info(f"    - Voice profile downloaded from R2 (bucket={resolved_bucket})")
                except Exception as storage_e:
                    logger.error(f"❌ Failed to download profile from storage: {storage_e}")
                    raise
            else:
                raise ValueError("Either profile_base64 or profile_path must be provided")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as temp_file:
                temp_file.write(profile_bytes)
                temp_profile_path = temp_file.name
            
            logger.info(f"    - Voice profile loaded")
            
            # Step 2: Generate TTS audio
            # Set defaults if not provided
            final_temperature = temperature if temperature is not None else 0.8
            final_exaggeration = exaggeration if exaggeration is not None else 0.5
            final_cfg_weight = cfg_weight if cfg_weight is not None else 0.5
            
            logger.info(f"  - Step 2: Generating TTS audio...")
            logger.info(f"  - TTS Parameters: temp={final_temperature}, exag={final_exaggeration}, cfg={final_cfg_weight}, pause_scale={pause_scale}")
            audio_tensor, sample_rate, generation_metadata = self.generate_long_text(
                text=text,
                voice_profile_path=temp_profile_path,
                output_path="./temp_tts_output.wav",
                max_chars=350,
                pause_ms=150,
                temperature=final_temperature,
                exaggeration=final_exaggeration,
                cfg_weight=final_cfg_weight,
                pause_scale=pause_scale,
                adaptive_voice_param_blend=adaptive_voice_param_blend,
            )
            
            # Step 3: Convert to MP3 bytes
            logger.info(f"  - Step 3: Converting to MP3...")
            mp3_bytes = tensor_to_mp3_bytes(audio_tensor, sample_rate, "96k")
            
            # Step 4: Upload directly to R2
            logger.info(f"  - Step 4: Uploading directly to R2...")
            
            # Extract story_type from metadata or use direct parameter
            final_story_type = story_type  # Start with direct parameter
            if metadata and isinstance(metadata, dict) and 'story_type' in metadata:
                final_story_type = metadata['story_type']  # Override with metadata if available
            
            # Ensure story_type is valid (user or app)
            if final_story_type not in ['user', 'app']:
                logger.warning(f"Invalid story_type '{final_story_type}', defaulting to 'user'")
                final_story_type = 'user'
            
            # Determine storage path based on admin generation flag
            is_admin_gen = (metadata or {}).get('is_admin_generation', False) if isinstance(metadata, dict) else False
            storage_path_hint = (metadata or {}).get('storage_path', '') if isinstance(metadata, dict) else ''
            
            if is_admin_gen and storage_path_hint:
                # Admin generation: use provided storage path (R2)
                import random
                suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=4))
                unique_filename = f"{voice_id}_{suffix}.mp3"
                r2_path = f"{storage_path_hint.rstrip('/')}/{unique_filename}"
                version_id = suffix  # Use suffix as version_id for admin generation
            else:
                # Regular user story generation: use new R2 path structure
                # Generate versionId (timestamp-based)
                version_id = f"{int(time.time() * 1000)}"
                
                # Validate required fields for R2 path
                if not user_id or not story_id or not language:
                    raise ValueError(f"Missing required fields for R2 path: user_id={user_id}, story_id={story_id}, language={language}")
                
                # Use new R2 path structure: private/users/{userId}/stories/audio/{language}/{storyId}/{versionId}.mp3
                r2_path = f"private/users/{user_id}/stories/audio/{language}/{story_id}/{version_id}.mp3"
            
            logger.info(f"    - R2 path: {r2_path}")
            logger.info(f"    - Story type: {final_story_type}")
            logger.info(f"    - Is admin generation: {is_admin_gen}")
            logger.info(f"    - Version ID: {version_id}")
            
            # Upload directly to R2
            r2_url = None
            try:
                # Always use R2 bucket for user stories
                r2_url = self.upload_to_storage(
                    data=mp3_bytes,
                    destination_blob_name=r2_path,
                    content_type="audio/mpeg",
                    metadata={
                        "bucket_name": "minstraly-storage",  # Always R2 for user stories
                        "user_id": user_id,
                        "story_id": story_id,
                        "voice_id": voice_id,
                        "voice_name": voice_name,
                        "language": language,
                        "story_type": final_story_type,
                        "text_length": len(text),
                        "generation_time": time.time() - start_time,
                        "audio_size": len(mp3_bytes),
                        "duration": generation_metadata.get("duration_sec", 0),
                        "version_id": version_id,  # Add version ID to metadata for easier discovery
                    }
                )
                if r2_url:
                    logger.info(f"    - Uploaded successfully to R2: {r2_url}")
                else:
                    logger.warning("⚠️ R2 upload did not return URL")
            except Exception as upload_error:
                logger.error(f"❌ R2 upload failed: {upload_error}")
                import traceback
                logger.error(f"❌ R2 upload traceback: {traceback.format_exc()}")
                r2_url = None
                raise  # Re-raise to fail the generation
            
            # Convert to base64
            audio_base64 = base64.b64encode(mp3_bytes).decode('utf-8')
            
            # Clean up temporary file
            os.unlink(temp_profile_path)
            
            generation_time = time.time() - start_time
            
            # Return result with R2 URL and path
            result = {
                "status": "success",
                "audio_data": audio_base64,
                "storage_url": r2_url,  # R2 public URL (primary)
                "storage_path": r2_path,  # R2 path (primary)
                "r2_path": r2_path,  # Explicit R2 path for callback validation
                "r2_url": r2_url,  # Explicit R2 URL
                "audio_url": r2_url,  # Alias for compatibility
                # Backward compatibility keys (deprecated, use storage_url/storage_path)
                "firebase_url": r2_url,
                "firebase_path": r2_path,
                "version_id": version_id,
                "story_type": final_story_type,
                "generation_time": generation_time,
                "duration": generation_metadata.get("duration_sec", 0)
            }

            # Firestore updates should happen via callback to the app.
            # Direct worker writes are opt-in for environments with configured ADC.
            enable_direct_firestore_update = os.getenv("CHATTERBOX_ENABLE_DIRECT_FIRESTORE_UPDATE", "false").lower() == "true"
            if enable_direct_firestore_update:
                try:
                    from google.cloud import firestore
                    client = firestore.Client()
                    from google.cloud.firestore import SERVER_TIMESTAMP  # type: ignore
                    if story_id:
                        doc = client.collection("stories").document(story_id)
                        # Build new audio version entry
                        # Use version_id from path generation (already generated above)
                        new_version = {
                            "id": version_id,
                            "voiceId": voice_id,
                            "voiceName": voice_name,
                            "audioUrl": r2_url or "",
                            "url": r2_url or "",
                            "createdAt": SERVER_TIMESTAMP,
                            "updatedAt": SERVER_TIMESTAMP,
                            "service": "chatterbox",
                            "metadata": {
                                "format": "mp3",
                                "size": len(mp3_bytes),
                                "duration": generation_metadata.get("duration_sec", 0) if 'generation_metadata' in locals() else 0,
                                "voiceName": voice_name,
                                "r2Path": r2_path,  # Store R2 path in metadata
                            },
                        }
                        # Update atomically: set status and push new version
                        doc.set({
                            "audioStatus": "ready",
                            "audioUrl": r2_url or "",
                            "updatedAt": SERVER_TIMESTAMP,
                        }, merge=True)
                        # Firestore array union append (fallback to read-modify-write if needed)
                        try:
                            from google.cloud.firestore_v1 import ArrayUnion
                            doc.update({"audioVersions": ArrayUnion([new_version])})
                        except Exception:
                            # Read-modify-write fallback
                            snap = doc.get()
                            existing = []
                            if snap.exists and isinstance(snap.to_dict().get("audioVersions"), list):
                                existing = snap.to_dict()["audioVersions"]
                            existing.append(new_version)
                            doc.set({"audioVersions": existing}, merge=True)
                        result["firestore_story_id"] = story_id
                except Exception as fe:
                    logger.warning(f"⚠️ Firestore update for story failed: {fe}")
            else:
                logger.info("ℹ️ Skipping direct Firestore write in worker (callback endpoint is source of truth)")
            
            logger.info(f"✅ TTS story generated successfully!")
            logger.info(f"  - Audio size: {len(mp3_bytes)} bytes")
            logger.info(f"  - Generation time: {generation_time:.2f}s")
            
            return result
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"❌ ChatterboxTTS.generate_tts_story failed: {e}")
            logger.error(f"  - Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"  - Full traceback: {traceback.format_exc()}")
            
            return {
                "status": "error",
                "error": str(e),
                "generation_time": generation_time
            }

    def generate_chunks_with_saved_voice(self, chunk_infos: List[ChunkInfo], saved_voice_path: str, 
                                       audio_prompt_path: str, base_temperature: float = 0.6, 
                                       base_exaggeration: float = 0.5, base_cfg_weight: float = 0.3) -> List[str]:
        """
        Generate chunks using saved voice + audio prompt with conditional caching optimization.
        
        :param chunk_infos: List of ChunkInfo objects with full analysis
        :param saved_voice_path: Path to saved voice embedding
        :param audio_prompt_path: Path to audio prompt file
        :param base_temperature: Base temperature (will be adapted per chunk)
        :param base_exaggeration: Base exaggeration (will be adapted per chunk)
        :param base_cfg_weight: Base CFG weight (will be adapted per chunk)
        :return: List of file paths to temporary WAV files
        """
        generation_start = time.time()
        
        # Use original sequential generation logic
        logger.info(f"🔄 Using original sequential processing for {len(chunk_infos)} chunks (saved voice + audio prompt)")
        wav_paths = []
        quality_scores = []
        
        for chunk_info in chunk_infos:
            wav_path, quality_score = self._generate_single_chunk_with_quality(
                chunk_info, None, None  # No pre-prepared conditionals
            )
            wav_paths.append(wav_path)
            quality_scores.append(quality_score)
        
        # Log comprehensive quality analysis
        total_generation_time = time.time() - generation_start
        self._log_quality_analysis(chunk_infos, quality_scores, total_generation_time)
        
        return wav_paths
    
    def generate_chunks_with_audio_prompt(self, chunk_infos: List[ChunkInfo], audio_prompt_path: str,
                                        base_temperature: float = 0.6, base_exaggeration: float = 0.5, 
                                        base_cfg_weight: float = 0.3) -> List[str]:
        """
        Generate chunks using audio prompt with conditional caching optimization.
        
        :param chunk_infos: List of ChunkInfo objects with full analysis
        :param audio_prompt_path: Path to audio prompt file
        :param base_temperature: Base temperature (will be adapted per chunk)
        :param base_exaggeration: Base exaggeration (will be adapted per chunk)
        :param base_cfg_weight: Base CFG weight (will be adapted per chunk)
        :return: List of file paths to temporary WAV files
        """
        generation_start = time.time()
        
        # Use original sequential generation logic
        logger.info(f"🔄 Using original sequential processing for {len(chunk_infos)} chunks (audio prompt)")
        wav_paths = []
        quality_scores = []
        
        for chunk_info in chunk_infos:
            wav_path, quality_score = self._generate_single_chunk_with_quality(
                chunk_info, None, None  # No pre-prepared conditionals
            )
            wav_paths.append(wav_path)
            quality_scores.append(quality_score)
        
        # Log comprehensive quality analysis
        total_generation_time = time.time() - generation_start
        self._log_quality_analysis(chunk_infos, quality_scores, total_generation_time)
        
        return wav_paths
    
    def _generate_chunks_with_prepared_conditionals(self, chunk_infos: List[ChunkInfo], 
                                                  pre_prepared_conditionals: Conditionals,
                                                  generation_start: float) -> List[str]:
        """
        Internal method to generate chunks with pre-prepared conditionals.
        This centralizes the generation logic for all conditional types.
        
        :param chunk_infos: List of ChunkInfo objects
        :param pre_prepared_conditionals: Pre-prepared conditionals
        :param generation_start: Start time for performance tracking
        :return: List of file paths to temporary WAV files
        """
        if self.enable_parallel_processing and len(chunk_infos) > 1:
            # Use parallel processing with quality analysis
            logger.info(f"🚀 Using parallel processing for {len(chunk_infos)} chunks")
            chunk_results = self.generate_chunks_parallel(chunk_infos, None, pre_prepared_conditionals)
            wav_paths = [wav_path for wav_path, _ in chunk_results]
            quality_scores = [quality_score for _, quality_score in chunk_results]
        else:
            # Use sequential processing (fallback or single chunk)
            logger.info(f"🔄 Using sequential processing for {len(chunk_infos)} chunks")
            wav_paths = []
            quality_scores = []
            
            for chunk_info in chunk_infos:
                wav_path, quality_score = self._generate_single_chunk_with_quality(
                    chunk_info, None, pre_prepared_conditionals
                )
                wav_paths.append(wav_path)
                quality_scores.append(quality_score)
        
        # Log comprehensive quality analysis
        total_generation_time = time.time() - generation_start
        self._log_quality_analysis(chunk_infos, quality_scores, total_generation_time)
        
        return wav_paths

    def generate_long_text_with_saved_voice(self, text: str, saved_voice_path: str, audio_prompt_path: str, 
                                          output_path: str, max_chars: int = 500, pause_ms: int = 100, 
                                          temperature: float = 0.6, exaggeration: float = 0.5, 
                                          cfg_weight: float = 0.3, pause_scale: float = 1.0) -> Tuple[torch.Tensor, int, Dict]:
        """
        Generate long text TTS using saved voice + audio prompt with conditional caching optimization.
        
        :param text: Input text to synthesize
        :param saved_voice_path: Path to saved voice embedding
        :param audio_prompt_path: Path to audio prompt file
        :param output_path: Path to save the final audio file
        :param max_chars: Maximum number of characters per chunk
        :param pause_ms: Milliseconds of silence between chunks
        :param temperature: Generation temperature
        :param exaggeration: Voice exaggeration factor
        :param cfg_weight: CFG weight for generation
        :param pause_scale: Global pause scaling factor
        :return: Tuple of (audio_tensor, sample_rate, metadata_dict)
        """
        logger.info(f"🎵 Starting optimized TTS processing for {len(text)} characters (saved voice + audio prompt)")
        logger.info(f"🔍 Output path: {output_path}")
        
        # Safety check for extremely long texts
        if len(text) > 13000:
            logger.warning(f"⚠️ Very long text ({len(text)} chars) - truncating to safe length")
            text = text[:13000] + "... [truncated]"
            logger.info(f"📝 Truncated text to {len(text)} characters")
        
        chunk_infos = self.chunk_text(text, max_chars)
        logger.info(f"📦 Split into {len(chunk_infos)} intelligent chunks")
        
        # Apply global pace factor to advanced stitcher
        try:
            self.advanced_stitcher.global_pause_factor = max(0.5, min(2.0, float(pause_scale)))
            logger.info(f"🕰️ Global pause scale set to {self.advanced_stitcher.global_pause_factor}")
        except Exception:
            logger.warning("⚠️ Failed to apply pause_scale; using default 1.0")
        
        # Prepare conditionals once for saved voice + prompt
        self.prepare_conditionals_with_saved_voice(saved_voice_path, audio_prompt_path, exaggeration=exaggeration)

        # Generate using centralized prepared-conditional path
        generation_start = time.time()
        wav_paths: List[str] = []
        quality_scores: List[QualityScore] = []
        for chunk_info in chunk_infos:
            wav_path, quality_score = self._generate_single_chunk_with_quality(
                chunk_info, None, self.conds
            )
            wav_paths.append(wav_path)
            quality_scores.append(quality_score)
        if not wav_paths:
            raise RuntimeError("Failed to generate any audio chunks")
        
        logger.info(f"🔗 Advanced stitching {len(wav_paths)} audio chunks...")
        audio_tensor, sample_rate, total_duration = self.stitch_and_normalize(wav_paths, chunk_infos, output_path, pause_ms)

        # Apply watermark once on final audio
        try:
            final_np = audio_tensor.squeeze(0).detach().cpu().numpy()
            final_np = self.watermarker.apply_watermark(final_np, sample_rate=sample_rate)
            audio_tensor = torch.from_numpy(final_np).unsqueeze(0)
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply final watermark: {e}")
        
        self.cleanup_chunks(wav_paths)
        
        logger.info(f"✅ Optimized TTS processing completed | Duration: {total_duration:.2f}s")
        logger.info(f"🔍 Final output path: {output_path}")
        logger.info(f"🔍 Output file exists: {Path(output_path).exists()}")
        
        # Enhanced metadata with cache performance
        cache_stats = self.get_conditional_cache_stats()
        metadata = self._create_generation_metadata(chunk_infos, total_duration, sample_rate, text, max_chars, 
                                                  pause_ms, pause_scale, cache_stats)
        
        return audio_tensor, sample_rate, metadata
    
    def generate_long_text_with_audio_prompt(self, text: str, audio_prompt_path: str, output_path: str,
                                           max_chars: int = 500, pause_ms: int = 100, temperature: float = 0.6,
                                           exaggeration: float = 0.5, cfg_weight: float = 0.3, 
                                           pause_scale: float = 1.0) -> Tuple[torch.Tensor, int, Dict]:
        """
        Generate long text TTS using audio prompt with conditional caching optimization.
        
        :param text: Input text to synthesize
        :param audio_prompt_path: Path to audio prompt file
        :param output_path: Path to save the final audio file
        :param max_chars: Maximum number of characters per chunk
        :param pause_ms: Milliseconds of silence between chunks
        :param temperature: Generation temperature
        :param exaggeration: Voice exaggeration factor
        :param cfg_weight: CFG weight for generation
        :param pause_scale: Global pause scaling factor
        :return: Tuple of (audio_tensor, sample_rate, metadata_dict)
        """
        logger.info(f"🎵 Starting optimized TTS processing for {len(text)} characters (audio prompt)")
        logger.info(f"🔍 Output path: {output_path}")
        
        # Safety check for extremely long texts
        if len(text) > 13000:
            logger.warning(f"⚠️ Very long text ({len(text)} chars) - truncating to safe length")
            text = text[:13000] + "... [truncated]"
            logger.info(f"📝 Truncated text to {len(text)} characters")
        
        chunk_infos = self.chunk_text(text, max_chars)
        logger.info(f"📦 Split into {len(chunk_infos)} intelligent chunks")
        
        # Apply global pace factor to advanced stitcher
        try:
            self.advanced_stitcher.global_pause_factor = max(0.5, min(2.0, float(pause_scale)))
            logger.info(f"🕰️ Global pause scale set to {self.advanced_stitcher.global_pause_factor}")
        except Exception:
            logger.warning("⚠️ Failed to apply pause_scale; using default 1.0")
        
        # Prepare conditionals once for audio prompt
        self.prepare_conditionals_with_audio_prompt(audio_prompt_path, exaggeration=exaggeration)

        # Generate using centralized prepared-conditional path
        wav_paths: List[str] = []
        quality_scores: List[QualityScore] = []
        for chunk_info in chunk_infos:
            wav_path, quality_score = self._generate_single_chunk_with_quality(
                chunk_info, None, self.conds
            )
            wav_paths.append(wav_path)
            quality_scores.append(quality_score)
        if not wav_paths:
            raise RuntimeError("Failed to generate any audio chunks")
        
        logger.info(f"🔗 Advanced stitching {len(wav_paths)} audio chunks...")
        audio_tensor, sample_rate, total_duration = self.stitch_and_normalize(wav_paths, chunk_infos, output_path, pause_ms)

        # Apply watermark once on final audio
        try:
            final_np = audio_tensor.squeeze(0).detach().cpu().numpy()
            final_np = self.watermarker.apply_watermark(final_np, sample_rate=sample_rate)
            audio_tensor = torch.from_numpy(final_np).unsqueeze(0)
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply final watermark: {e}")
        
        self.cleanup_chunks(wav_paths)
        
        logger.info(f"✅ Optimized TTS processing completed | Duration: {total_duration:.2f}s")
        logger.info(f"🔍 Final output path: {output_path}")
        logger.info(f"🔍 Output file exists: {Path(output_path).exists()}")
        
        # Enhanced metadata with cache performance
        cache_stats = self.get_conditional_cache_stats()
        metadata = self._create_generation_metadata(chunk_infos, total_duration, sample_rate, text, max_chars, 
                                                  pause_ms, pause_scale, cache_stats)
        
        return audio_tensor, sample_rate, metadata
    
    def _create_generation_metadata(self, chunk_infos: List[ChunkInfo], total_duration: float,
                                  sample_rate: int, text: str, max_chars: int, pause_ms: int, 
                                  pause_scale: float, cache_stats: Dict) -> Dict:
        """Create comprehensive metadata for generation results"""
        if chunk_infos:
            content_type_dist = self.smart_chunker._get_content_type_distribution(chunk_infos)
            avg_complexity = sum(c.complexity_score for c in chunk_infos) / len(chunk_infos)
            avg_chunk_chars = sum(c.char_count for c in chunk_infos) / len(chunk_infos)
            dialogue_chunks = sum(1 for c in chunk_infos if c.content_type == ContentType.DIALOGUE)
            paragraph_breaks = sum(1 for c in chunk_infos if c.paragraph_break_after)
        else:
            content_type_dist = {}
            avg_complexity = 0
            avg_chunk_chars = 0
            dialogue_chunks = 0
            paragraph_breaks = 0
        
        return {
            # Basic metrics
            "chunk_count": len(chunk_infos),
            "duration_sec": total_duration,
            "sample_rate": sample_rate,
            "text_length": len(text),
            "max_chars_per_chunk": max_chars,
            "pause_ms": pause_ms,
            "pause_scale": pause_scale,
            
            # Smart chunking analysis
            "avg_chunk_chars": round(avg_chunk_chars, 1),
            "avg_complexity_score": round(avg_complexity, 2),
            "content_type_distribution": content_type_dist,
            "dialogue_chunk_count": dialogue_chunks,
            "paragraph_breaks": paragraph_breaks,
            "chunking_method": "smart_content_aware",
            
            # Advanced features
            "text_sanitization": "advanced_unicode_normalization",
            "parallel_processing": self.enable_parallel_processing,
            "max_parallel_workers": self.max_parallel_workers if self.enable_parallel_processing else 1,
            "quality_analysis": "comprehensive_audio_validation",
            "stitching_method": "advanced_context_aware_transitions",
            
            # Performance metrics
            "audio_chars_per_second": round(len(text) / max(total_duration, 0.1), 1),
            "audio_efficiency_ratio": round(total_duration / max(len(text) * 0.08, 1), 2),
            
            # Conditional caching performance
            "conditional_cache_hits": cache_stats['hits'],
            "conditional_cache_misses": cache_stats['misses'],
            "conditional_cache_hit_rate": round(cache_stats['hit_rate_percent'], 1),
            "conditional_cache_total_requests": cache_stats['total_requests'],
            "optimization_enabled": True,
        }
