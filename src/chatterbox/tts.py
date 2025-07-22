from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen, VoiceProfile
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


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
        self.s3gen.save_voice_profile(ref_wav, sr, save_path)
        print(f"✅ Full voice profile saved to {save_path}")

    def load_voice_clone(self, path: str):
        """Load a pre-saved voice embedding"""
        return self.s3gen.load_voice_clone(path)

    def load_voice_profile(self, path: str):
        """Load a complete voice profile"""
        return self.s3gen.load_voice_profile(path)

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
        
        # For voice encoder embedding, we can use the CAMPPlus embedding from the profile
        # as a reasonable approximation for T3 conditioning
        ve_embed = profile.embedding.to(self.device)
        
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