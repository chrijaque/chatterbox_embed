import importlib.util
import os
from pathlib import Path

FACTORY_PATH = Path(__file__).resolve().parents[1] / "src" / "chatterbox" / "factory.py"
spec = importlib.util.spec_from_file_location("chatterbox_factory", FACTORY_PATH)
factory = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(factory)

SAMPLE_TEXT_BY_LANGUAGE = factory.SAMPLE_TEXT_BY_LANGUAGE
allowed_payload_model_types = factory.allowed_payload_model_types
assert_payload_model_type = factory.assert_payload_model_type
get_model_family = factory.get_model_family
model_type_for_family = factory.model_type_for_family
sample_text_for_language = factory.sample_text_for_language


def test_factory_routing():
    assert get_model_family("original") == "original"
    assert get_model_family("chatterbox") == "original"
    assert get_model_family("turbo") == "turbo"
    assert get_model_family("mtl") == "mtl"
    assert model_type_for_family("turbo") == "chatterbox-turbo"
    assert model_type_for_family("mtl") == "chatterbox-mtl"
    assert model_type_for_family("original") == "chatterbox"
    assert allowed_payload_model_types("turbo") == {"chatterbox-turbo"}
    assert assert_payload_model_type("chatterbox-turbo", "turbo") == "chatterbox-turbo"
    assert assert_payload_model_type(None, "mtl") == "chatterbox-mtl"
    try:
        assert_payload_model_type("chatterbox-mtl", "turbo")
        raise AssertionError("expected mismatch")
    except ValueError as exc:
        assert "cannot run model_type=chatterbox-mtl" in str(exc)
    assert "da" in SAMPLE_TEXT_BY_LANGUAGE
    da_sample = sample_text_for_language("da").lower()
    assert "historie" in da_sample or "stemme" in da_sample
    previous = os.environ.get("CHATTERBOX_MODEL_FAMILY")
    os.environ["CHATTERBOX_MODEL_FAMILY"] = "mtl"
    try:
        assert get_model_family() == "mtl"
    finally:
        if previous is None:
            os.environ.pop("CHATTERBOX_MODEL_FAMILY", None)
        else:
            os.environ["CHATTERBOX_MODEL_FAMILY"] = previous
    print("factory routing tests passed")


if __name__ == "__main__":
    test_factory_routing()
