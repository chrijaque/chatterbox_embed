"""Load original / Turbo / MTL Chatterbox models from CHATTERBOX_MODEL_FAMILY."""
from __future__ import annotations

import os

FAMILY_ORIGINAL = "original"
FAMILY_TURBO = "turbo"
FAMILY_MTL = "mtl"

_FAMILY_TO_MODEL_TYPE = {
    FAMILY_ORIGINAL: "chatterbox",
    FAMILY_TURBO: "chatterbox-turbo",
    FAMILY_MTL: "chatterbox-mtl",
}

_MODEL_TYPE_TO_FAMILY = {v: k for k, v in _FAMILY_TO_MODEL_TYPE.items()}


def get_model_family(raw: str | None = None) -> str:
    value = (raw or os.getenv("CHATTERBOX_MODEL_FAMILY") or FAMILY_ORIGINAL).strip().lower()
    if value in ("original", "legacy", "chatterbox"):
        return FAMILY_ORIGINAL
    if value in (FAMILY_TURBO, FAMILY_MTL):
        return value
    raise ValueError(f"Unknown CHATTERBOX_MODEL_FAMILY '{value}'. Expected original, turbo, or mtl.")


def model_type_for_family(family: str | None = None) -> str:
    return _FAMILY_TO_MODEL_TYPE[get_model_family(family)]


def allowed_payload_model_types(family: str | None = None) -> set[str]:
    resolved = get_model_family(family)
    return {_FAMILY_TO_MODEL_TYPE[resolved]}


def family_for_model_type(model_type: str | None) -> str | None:
    if not model_type:
        return None
    return _MODEL_TYPE_TO_FAMILY.get(str(model_type).strip().lower())


def assert_payload_model_type(model_type: str | None, family: str | None = None) -> str:
    resolved_family = get_model_family(family)
    expected = model_type_for_family(resolved_family)
    provided = (model_type or "").strip().lower()
    if not provided:
        return expected
    if provided not in allowed_payload_model_types(resolved_family):
        raise ValueError(
            f"This worker is CHATTERBOX_MODEL_FAMILY={resolved_family} "
            f"(model_type={expected}) and cannot run model_type={provided}."
        )
    return provided


def load_tts_model(device: str, family: str | None = None):
    resolved = get_model_family(family)
    if resolved == FAMILY_TURBO:
        from .tts_turbo import ChatterboxTurboTTS
        return ChatterboxTurboTTS.from_pretrained(device=device)
    if resolved == FAMILY_MTL:
        from .mtl_tts import ChatterboxMultilingualTTS
        return ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
    from .tts import ChatterboxTTS
    return ChatterboxTTS.from_pretrained(device=device)


def load_vc_model(device: str, family: str | None = None):
    from .vc import ChatterboxVC
    return ChatterboxVC.from_pretrained(device=device, family=family)


SAMPLE_TEXT_BY_LANGUAGE = {
    "ar": "مرحبا، هذا استنساخ صوتي لرواية القصص. يمكنني سرد قصص ما قبل النوم.",
    "da": "Hej, dette er en stemmeklon til historiefortælling. Jeg kan fortælle godnathistorier.",
    "de": "Hallo, das ist ein Stimmklon fürs Vorlesen. Ich kann Gute-Nacht-Geschichten erzählen.",
    "el": "Γεια σας, αυτό είναι ένα αντίγραφο φωνής για αφήγηση. Μπορώ να λέω ιστορίες πριν τον ύπνο.",
    "en": "Hello, this is a voice clone for storytelling. I can narrate bedtime stories.",
    "es": "Hola, este es un clon de voz para contar historias. Puedo narrar cuentos para dormir.",
    "fi": "Hei, tämä on ääniklooni tarinankerrontaan. Voin kertoa iltasatuja.",
    "fr": "Bonjour, ceci est un clone vocal pour raconter des histoires. Je peux narrer des histoires du soir.",
    "he": "שלום, זה שיבוט קול לסיפור סיפורים. אני יכול לספר סיפורי לילה.",
    "hi": "नमस्ते, यह कहानी सुनाने के लिए एक आवाज़ क्लोन है। मैं सोने की कहानियाँ सुना सकता हूँ।",
    "it": "Ciao, questo è un clone vocale per raccontare storie. Posso narrare fiabe della buonanotte.",
    "ja": "こんにちは。これは物語のための音声クローンです。寝物語を語ることができます。",
    "ko": "안녕하세요. 이것은 이야기용 음성 클론입니다. 잠자리 이야기를 들려드릴 수 있습니다.",
    "ms": "Helo, ini ialah klon suara untuk bercerita. Saya boleh menceritakan kisah sebelum tidur.",
    "nl": "Hallo, dit is een stemkloon voor verhalen. Ik kan slaapverhaaltjes vertellen.",
    "no": "Hei, dette er en stemmeklon for historiefortelling. Jeg kan fortelle godnatt-historier.",
    "pl": "Cześć, to klon głosu do opowiadania historii. Mogę opowiadać bajki na dobranoc.",
    "pt": "Olá, este é um clone de voz para contar histórias. Posso narrar histórias de ninar.",
    "ru": "Здравствуйте, это голосовой клон для рассказов. Я могу читать сказки на ночь.",
    "sv": "Hej, det här är en röstklon för historieberättande. Jag kan berätta godnattsagor.",
    "sw": "Habari, hii ni kloni ya sauti ya kusimulia hadithi. Naweza kusimulia hadithi za kulala.",
    "tr": "Merhaba, bu hikâye anlatımı için bir ses klonu. Uyku masalları anlatabilirim.",
    "zh": "你好，这是一个用于讲故事的声音克隆。我可以讲述睡前故事。",
}


def sample_text_for_language(language: str, voice_name: str | None = None) -> str:
    code = (language or "en").strip().lower()
    text = SAMPLE_TEXT_BY_LANGUAGE.get(code) or SAMPLE_TEXT_BY_LANGUAGE["en"]
    if voice_name and code == "en":
        return f"Hello, this is the voice profile of {voice_name}. I can be used to narrate whimsical stories and fairytales."
    return text
