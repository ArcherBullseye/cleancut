"""Tests for the JSON-stripping helpers in classify_dialogue and classify_visual."""

from cleancut.classify_visual import _flagged_categories, VLMParams
from cleancut.llm_utils import strip_to_json


def test_strip_dialogue_handles_clean_json():
    s = '{"category": "drugs", "should_cut": true}'
    assert strip_to_json(s) == s


def test_strip_dialogue_handles_chatty_prefix():
    s = 'Sure! Here is the result:\n{"category": "drugs"}\nLet me know if you need more.'
    assert strip_to_json(s) == '{"category": "drugs"}'


def test_strip_dialogue_handles_no_json():
    s = "Just text, no json"
    # Falls back to the input
    assert strip_to_json(s) == s


def test_strip_visual_handles_clean_json():
    s = '{"explicit": false}'
    assert strip_to_json(s) == s


def test_flagged_categories_multi_signal():
    result = {"explicit": True, "drug_use": True, "confidence": 0.9}
    cats = _flagged_categories(result, VLMParams())
    assert "nudity" in cats
    assert "drugs" in cats


def test_flagged_categories_drops_below_threshold():
    result = {"explicit": True, "confidence": 0.3}
    assert _flagged_categories(result, VLMParams(min_confidence=0.55)) == []


def test_flagged_categories_violence_only_when_in_cut_on():
    # Default VLMParams: cut_on=("explicit", "drug_use", "violence") — violence included
    result = {"violence": True, "confidence": 0.8}
    assert _flagged_categories(result, VLMParams()) == ["violence"]


def test_flagged_categories_empty_when_nothing_flagged():
    result = {"intimate": False, "explicit": False, "confidence": 0.9}
    assert _flagged_categories(result, VLMParams()) == []
