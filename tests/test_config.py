"""Tests for config.py — presets, loading, resolved_encoder, etc."""

import platform

import pytest

from cleancut.config import Config, PRESETS


def test_load_defaults_populates_wordlists():
    config = Config.load_defaults()
    assert config.wordlists, "Wordlists should be non-empty"
    assert "profanity" in config.wordlists
    assert "drugs" in config.wordlists


def test_load_defaults_populates_replacements():
    config = Config.load_defaults()
    assert config.replacements, "Replacements should be non-empty"


def test_apply_preset_fast():
    config = Config.load_defaults()
    config.apply_preset("fast")
    assert config.whisper_model == "base"
    assert config.density_enabled is False


def test_apply_preset_thorough_enables_everything():
    config = Config.load_defaults()
    config.apply_preset("thorough")
    assert config.density_enabled is True
    assert config.llm_enabled is True
    assert config.vlm_enabled is True
    assert config.audio_events_enabled is True


def test_apply_unknown_preset_raises():
    config = Config.load_defaults()
    with pytest.raises(ValueError):
        config.apply_preset("nonexistent")


def test_resolved_encoder_explicit():
    config = Config.load_defaults()
    config.encoder = "libx264"
    assert config.resolved_encoder() == "libx264"


def test_resolved_encoder_auto_on_mac():
    config = Config.load_defaults()
    config.encoder = "auto"
    # We can't easily mock platform.system, so just verify it returns SOMETHING
    enc = config.resolved_encoder()
    assert enc in ("videotoolbox", "libx264")
    if platform.system() == "Darwin":
        assert enc == "videotoolbox"


def test_default_actions_violence_keep():
    config = Config.load_defaults()
    assert config.actions["violence"] == "keep"


def test_default_enabled_categories_excludes_violence():
    config = Config.load_defaults()
    assert "violence" not in config.enabled_categories


def test_all_presets_keys_present():
    config = Config.load_defaults()
    for preset_name in PRESETS:
        config = Config.load_defaults()
        config.apply_preset(preset_name)
        # All preset fields should map to real config attributes
        for k in PRESETS[preset_name]:
            assert hasattr(config, k), f"Preset {preset_name} sets unknown field {k}"
