from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
import logging
from typing import List, Optional, Tuple, Dict

import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torchaudio

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen, VoiceProfile
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond

# Configure logging
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import nltk
    from nltk.tokenize.punkt import PunktSentenceTokenizer
    # Force download and setup NLTK punkt tokenizer
    try:
        nltk.download('punkt', quiet=True)
        nltk.data.find('tokenizers/punkt')
        NLTK_AVAILABLE = True
        logger.info("✅ NLTK punkt tokenizer available")
    except Exception as e:
        logger.warning(f"⚠️ NLTK setup failed: {e}")
        NLTK_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("⚠️ nltk not available - will use simple text splitting")

try:
    from pydub import AudioSegment, effects
    PYDUB_AVAILABLE = True
    logger.info("✅ pydub available for audio processing")
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("⚠️ pydub not available - will use torchaudio for audio processing")

REPO_ID = "ResembleAI/chatterbox"


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


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
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
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
        data = np.load(path, allow_pickle=True).item()
        
        # Create VoiceProfile object
        profile = VoiceProfile(
            embedding=torch.tensor(data["embedding"]).to(self.device),
            prompt_feat=torch.tensor(data["prompt_feat"]).to(self.device) if "prompt_feat" in data else None,
            prompt_feat_len=data.get("prompt_feat_len"),
            prompt_token=torch.tensor(data["prompt_token"]).to(self.device) if "prompt_token" in data else None,
            prompt_token_len=torch.tensor(data["prompt_token_len"]).to(self.device) if "prompt_token_len" in data else None,
        )
        
        # Add voice encoder embedding as an attribute
        if "ve_embedding" in data:
            profile.ve_embedding = torch.tensor(data["ve_embedding"]).to(self.device)
        else:
            # Fallback for old profiles without voice encoder embedding
            profile.ve_embedding = None
            
        return profile

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
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

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
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
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def prepare_conditionals_with_saved_voice(self, saved_voice_path: str, prompt_audio_path: str, exaggeration=0.5):
        """Prepare conditionals using a pre-saved voice embedding for faster processing"""
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
        
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)
        print(f"Conditionals prepared using saved voice from {saved_voice_path}")

    def prepare_conditionals_with_voice_profile(self, voice_profile_path: str, exaggeration=0.5):
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
        if plen := self.t3.hp.speech_cond_prompt_len:
            # Use the prompt tokens from the profile, but limit to the required length
            t3_cond_prompt_tokens = profile.prompt_token[:, :plen].to(self.device)
        else:
            t3_cond_prompt_tokens = None
        
        # Use the voice encoder embedding from the profile for T3 conditioning
        if hasattr(profile, 've_embedding') and profile.ve_embedding is not None:
            ve_embed = profile.ve_embedding.to(self.device)
        else:
            # Fallback to random embedding if no voice encoder embedding
            ve_embed = torch.randn(1, 256).to(self.device)
        
        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)
        print(f"✅ Conditionals prepared using voice profile from {voice_profile_path}")

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
        cfg_weight=0.5,
        temperature=0.8,
    ):
        if voice_profile_path:
            # Use complete voice profile for most accurate TTS
            self.prepare_conditionals_with_voice_profile(voice_profile_path, exaggeration=exaggeration)
        elif saved_voice_path and audio_prompt_path:
            # Use saved voice embedding with fresh prompt audio for prosody
            self.prepare_conditionals_with_saved_voice(saved_voice_path, audio_prompt_path, exaggeration=exaggeration)
        elif audio_prompt_path:
            # Traditional method: compute everything fresh
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first, specify `audio_prompt_path`, or provide `voice_profile_path`, or provide both `saved_voice_path` and `audio_prompt_path`"

        # Norm and tokenize text
        text = punc_norm(text)
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
                max_new_tokens=1000,  # TODO: use the value in config
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

            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def chunk_text(self, text: str, max_chars: int = 500) -> List[str]:
        """
        Splits input text into sentence-aligned chunks based on max character count.

        :param text: Full story text
        :param max_chars: Maximum number of characters per chunk
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
            if len(current) + len(sent) + 1 <= max_chars:
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

    def generate_chunks(self, chunks: List[str], voice_profile_path: str, temperature: float = 0.8, 
                       exaggeration: float = 0.5, cfg_weight: float = 0.5) -> List[str]:
        """
        Generates speech from each text chunk and stores temporary WAV files.

        :param chunks: List of text chunks
        :param voice_profile_path: Path to the voice profile (.npy)
        :param temperature: Generation temperature
        :param exaggeration: Voice exaggeration factor
        :param cfg_weight: CFG weight for generation
        :return: List of file paths to temporary WAV files
        """
        wav_paths = []
        
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
                    
                    # Generate audio tensor using the voice profile
                    audio_tensor = self.generate(
                        text=chunk,
                        voice_profile_path=voice_profile_path,
                        temperature=temperature,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight
                    )
                    
                    # Save to temporary file
                    temp_wav = tempfile.NamedTemporaryFile(suffix=f"_chunk_{i}.wav", delete=False)
                    torchaudio.save(temp_wav.name, audio_tensor, self.sr)
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
                        self.cleanup_chunks(wav_paths)
                        
                        # Raise exception to stop processing
                        raise RuntimeError(f"Chunk {i+1} failed after {max_retries + 1} attempts: {e}")
        
        return wav_paths

    def stitch_and_normalize(self, wav_paths: List[str], output_path: str, pause_ms: int = 100) -> Tuple[torch.Tensor, int, float]:
        """
        Stitches WAV chunks together with pause and normalizes audio levels.

        :param wav_paths: List of temporary WAV file paths
        :param output_path: Final path to export the combined WAV file
        :param pause_ms: Milliseconds of silence between chunks
        :return: Tuple of (audio_tensor, sample_rate, duration_seconds)
        """
        logger.info(f"🔍 stitch_and_normalize called with output_path: {output_path}")
        
        if PYDUB_AVAILABLE:
            # Use pydub for professional audio processing
            logger.info(f"🔍 Using pydub for audio processing")
            final = AudioSegment.empty()
            for p in wav_paths:
                seg = AudioSegment.from_wav(p)
                final += seg + AudioSegment.silent(pause_ms)
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
                silence_duration = int(pause_ms * sample_rate / 1000)
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

    def generate_long_text(self, text: str, voice_profile_path: str, output_path: str, 
                          max_chars: int = 500, pause_ms: int = 100, temperature: float = 0.8,
                          exaggeration: float = 0.5, cfg_weight: float = 0.5) -> Tuple[torch.Tensor, int, Dict]:
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
        
        chunks = self.chunk_text(text, max_chars)
        logger.info(f"📦 Split into {len(chunks)} chunks")
        
        wav_paths = self.generate_chunks(chunks, voice_profile_path, temperature, exaggeration, cfg_weight)
        if not wav_paths:
            raise RuntimeError("Failed to generate any audio chunks")
        
        logger.info(f"🔗 Stitching {len(wav_paths)} audio chunks...")
        audio_tensor, sample_rate, total_duration = self.stitch_and_normalize(wav_paths, output_path, pause_ms)
        
        self.cleanup_chunks(wav_paths)
        
        logger.info(f"✅ TTS processing completed | Duration: {total_duration:.2f}s")
        logger.info(f"🔍 Final output path: {output_path}")
        logger.info(f"🔍 Output file exists: {Path(output_path).exists()}")
        
        metadata = {
            "chunk_count": len(chunks),
            "output_path": output_path,
            "duration_sec": total_duration,
            "successful_chunks": len(wav_paths),
            "sample_rate": sample_rate,
            "text_length": len(text),
            "max_chars_per_chunk": max_chars,
            "pause_ms": pause_ms
        }
        
        return audio_tensor, sample_rate, metadata

    # ------------------------------------------------------------------
    # Complete TTS Pipeline with Firebase Upload
    # ------------------------------------------------------------------
    def _tensor_to_mp3_bytes(self, audio_tensor: torch.Tensor, sample_rate: int, bitrate: str = "96k") -> bytes:
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
                audio_segment = self._tensor_to_audiosegment(audio_tensor, sample_rate)
                # Export to MP3 bytes
                mp3_file = audio_segment.export(format="mp3", bitrate=bitrate)
                # Read the bytes from the file object
                mp3_bytes = mp3_file.read()
                return mp3_bytes
            except Exception as e:
                logger.warning(f"Direct MP3 conversion failed: {e}, falling back to WAV")
                return self._tensor_to_wav_bytes(audio_tensor, sample_rate)
        else:
            logger.warning("pydub not available, falling back to WAV")
            return self._tensor_to_wav_bytes(audio_tensor, sample_rate)

    def _tensor_to_audiosegment(self, audio_tensor: torch.Tensor, sample_rate: int):
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

    def _tensor_to_wav_bytes(self, audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
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

    def _upload_to_firebase(self, data: bytes, destination_blob_name: str, content_type: str = "application/octet-stream", metadata: dict = None) -> str:
        """
        Upload data directly to Firebase Storage with metadata
        
        :param data: Binary data to upload
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
            
            logger.info(f"🔍 Starting Firebase upload: {destination_blob_name} ({len(data)} bytes)")
            
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

    def generate_tts_story(self, text: str, voice_profile_path: str, voice_id: str, 
                          language: str = 'en', story_type: str = 'user', 
                          is_kids_voice: bool = False, metadata: Dict = None) -> Dict:
        """
        Complete TTS pipeline: generate → convert → upload to Firebase → return URLs
        
        :param text: Text to synthesize
        :param voice_profile_path: Path to voice profile
        :param voice_id: Unique voice identifier
        :param language: Language for Firebase organization
        :param story_type: Type of story (user, sample, etc.)
        :param is_kids_voice: Whether this is a kids voice
        :param metadata: Optional additional metadata
        :return: Dictionary with Firebase URLs and metadata
        """
        from datetime import datetime
        
        logger.info(f"🎵 Starting TTS story generation for {voice_id}")
        
        try:
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create output path
            output_path = f"/tmp/tts_generated/{voice_id}_{timestamp}.wav"
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Use the long text TTS functionality
            audio_tensor, sample_rate, tts_metadata = self.generate_long_text(
                text=text,
                voice_profile_path=voice_profile_path,
                output_path=output_path,
                max_chars=500,
                pause_ms=150,
                temperature=0.8,
                exaggeration=0.5,
                cfg_weight=0.5
            )
            
            # Convert to MP3 bytes
            mp3_bytes = self._tensor_to_mp3_bytes(audio_tensor, sample_rate, "96k")
            
            # Upload to Firebase
            if is_kids_voice:
                firebase_path = f"audio/stories/{language}/kids/{story_type}/TTS_{voice_id}_{timestamp}.mp3"
            else:
                firebase_path = f"audio/stories/{language}/{story_type}/TTS_{voice_id}_{timestamp}.mp3"
            
            # Prepare Firebase metadata
            firebase_metadata = {
                'voice_id': voice_id,
                'language': language,
                'story_type': story_type,
                'is_kids_voice': str(is_kids_voice),
                'format': '96k_mp3',
                'timestamp': timestamp,
                'model': 'chatterbox_tts',
                'created_date': datetime.now().isoformat(),
                'text_length': len(text),
                'chunk_count': tts_metadata.get('chunk_count', 0),
                'duration_sec': tts_metadata.get('duration_sec', 0),
                'Access-Control-Allow-Origin': '*',
                'Cache-Control': 'public, max-age=3600'
            }
            
            # Add optional metadata
            if metadata:
                firebase_metadata.update(metadata)
            
            # Upload to Firebase
            firebase_url = self._upload_to_firebase(
                mp3_bytes,
                firebase_path,
                "audio/mpeg",
                firebase_metadata
            )
            
            # Clean up temporary file
            try:
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Failed to clean up temp file: {cleanup_error}")
            
            logger.info(f"✅ TTS story generation completed for {voice_id}")
            
            return {
                "status": "success",
                "voice_id": voice_id,
                "audio_path": firebase_path,
                "audio_url": firebase_url,
                "generation_time": tts_metadata.get("duration_sec", 0),
                "model": "chatterbox_tts",
                "metadata": {
                    **tts_metadata,
                    "voice_id": voice_id,
                    "language": language,
                    "story_type": story_type,
                    "is_kids_voice": is_kids_voice,
                    "firebase_path": firebase_path,
                    "firebase_url": firebase_url,
                    "text_length": len(text),
                    "timestamp": timestamp
                }
            }
            
        except Exception as e:
            logger.error(f"❌ TTS story generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "voice_id": voice_id
            }