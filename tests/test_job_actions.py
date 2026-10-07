from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from cleancut.config import Config
from cleancut.editor_ranges import Range
from cleancut.edl import EditDecision, EditDecisionList, resolve_action
from cleancut.speech import eligible_words, prepare_replacements
from cleancut.subtitles import scan_words
from cleancut.transcribe import Word
from tests.test_speech import word_decision
from webapp import jobs, settings
from webapp.app import app


@pytest.fixture
def local_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(jobs, "DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(jobs, "ensure_dirs", lambda: None)
    monkeypatch.setattr(jobs, "job_dir", lambda job_id: tmp_path / "jobs" / str(job_id))
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path)
    jobs.init_db()
    return app.test_client()


def test_scan_remembers_choices_and_keeps_existing_job_snapshot(local_jobs, tmp_path):
    first, second = tmp_path / "one.mp4", tmp_path / "two.mp4"
    first.touch()
    second.touch()
    choices = {
        "path": str(first), "preset": "fast", "categories": ["profanity", "nudity"],
        "actions": {"profanity": "replace", "nudity": "cut"}, "prefer_language": "fra",
        "auto_render": True, "use_visual": False, "use_local_ai": False,
        "allow_solo_visual": True, "word_end_padding_ms": 300,
    }
    with patch("webapp.app.library.resolve_safe", side_effect=lambda path: first if path == str(first) else second):
        response = local_jobs.post("/api/scan", json=choices)
        assert response.status_code == 200
        job_id = response.json["job_id"]
        remembered = local_jobs.get("/api/settings").json["settings"]
        assert remembered["actions"]["profanity"] == "replace"
        assert remembered["categories"] == choices["categories"]
        assert remembered["preset"] == "fast" and remembered["prefer_language"] == "fra"
        assert remembered["auto_render"] and remembered["allow_solo_visual"]
        assert remembered["word_end_padding_ms"] == 300
        assert not remembered["use_visual"] and not remembered["local_ai_enabled"]
        # A fresh read from disk / restarted job database retains defaults.
        jobs.init_db()
        next_response = local_jobs.post("/api/scan", json={"path": str(second)})
        second_job = jobs.get_job(next_response.json["job_id"])
        second_opts = json.loads(second_job["options"])
        assert second_opts["actions"]["profanity"] == "replace"
        assert second_opts["categories"] == choices["categories"]
        assert not second_opts["use_visual"] and not second_opts["use_llm"]
        assert second_opts["auto_render"]
        assert second_opts["word_end_padding_ms"] == 300
        command = jobs.build_scan_command(second_job)
        assert command[command.index("--word-end-padding-ms") + 1] == "300"
    settings.save({"actions": {"profanity": "mute"}})
    assert json.loads(jobs.get_job(job_id)["options"])["actions"]["profanity"] == "replace"


@pytest.mark.parametrize("body", [
    {"actions": {"nudity": "replace"}}, {"actions": {"profanity": "bogus"}},
    {"actions": []}, {"categories": ["bogus"]}, {"categories": None},
    {"word_end_padding_ms": 501}, {"word_end_padding_ms": -1}, {"word_end_padding_ms": "200"},
])
def test_invalid_scan_choices_do_not_persist(local_jobs, tmp_path, body):
    video = tmp_path / "movie.mp4"
    video.touch()
    with patch("webapp.app.library.resolve_safe", return_value=video):
        assert local_jobs.post("/api/scan", json={"path": str(video), **body}).status_code == 400
    assert not settings._SETTINGS_PATH.exists()
    assert jobs.list_jobs() == []


def test_empty_categories_are_saved_not_replaced_with_defaults(local_jobs, tmp_path):
    video = tmp_path / "movie.mp4"
    video.touch()
    with patch("webapp.app.library.resolve_safe", return_value=video):
        response = local_jobs.post("/api/scan", json={"path": str(video), "categories": []})
    assert response.status_code == 200
    assert settings.load()["categories"] == []
    assert json.loads(jobs.get_job(response.json["job_id"])["options"])["categories"] == []


def test_legacy_global_switch_migrates_next_job_default_once(local_jobs):
    settings._SETTINGS_PATH.write_text(json.dumps({"profanity_audio": "replace", "actions": {"nudity": "cut"}}))
    cfg = settings.load()
    assert cfg["actions"]["profanity"] == "replace"
    assert "profanity_audio" not in cfg
    settings.save({"actions": {"profanity": "mute"}})
    assert settings.load()["actions"]["profanity"] == "mute"
    assert "profanity_audio" not in json.loads(settings._SETTINGS_PATH.read_text())


def test_legacy_switch_does_not_override_cut_default(local_jobs):
    settings._SETTINGS_PATH.write_text(json.dumps({"profanity_audio": "replace", "actions": {"profanity": "cut"}}))
    assert settings.load()["actions"]["profanity"] == "cut"


def test_render_uses_edl_actions_not_global_or_future_defaults(local_jobs, tmp_path):
    video = tmp_path / "movie.mp4"
    video.touch()
    scan_id = jobs.create_job("scan", str(video), options={"actions": {"profanity": "replace"}, "output_location": "source"})
    path = tmp_path / "edl.json"
    EditDecisionList(decisions=[word_decision(action="replace"), word_decision(4, 4.5)]).to_json(path)
    jobs.update_job(scan_id, status=jobs.DONE, edl_path=str(path))
    settings.save({"actions": {"profanity": "mute"}, "speech_host": "http://192.168.1.20:8765"})
    response = local_jobs.post(f"/api/job/{scan_id}/render")
    render = jobs.get_job(response.json["job_id"])
    opts = json.loads(render["options"])
    assert opts["profanity_audio"] == "mute"  # no blanket conversion of Mute
    command = jobs.build_render_command(render)
    assert command[command.index("--profanity-audio") + 1] == "mute"
    assert command[command.index("--speech-host") + 1] == "http://192.168.1.20:8765"
    assert [d.action for d in EditDecisionList.from_json(path)] == ["replace", "mute"]


def test_review_replace_action_round_trip_and_summary(local_jobs, tmp_path):
    path = tmp_path / "edl.json"
    EditDecisionList(decisions=[word_decision(), EditDecision(5, 6, "cut", "nudity")]).to_json(path)
    job_id = jobs.create_job("scan", str(tmp_path / "movie.mp4"), edl_path=str(path))
    response = local_jobs.post(f"/api/job/{job_id}/edl/decision", json={"index": 0, "action": "replace"})
    assert response.status_code == 200
    assert response.json["summary"]["replacements"] == 1
    assert response.json["summary"]["mutes"] == 0
    assert EditDecisionList.from_json(path).decisions[0].action == "replace"
    bad = local_jobs.post(f"/api/job/{job_id}/edl/decision", json={"index": 1, "action": "replace"})
    assert bad.status_code == 400
    assert settings.load()["actions"]["profanity"] == "mute"  # individual edits aren't next-job defaults


def test_scan_and_cli_emit_explicit_replace_with_word_precision(cli_args):
    from cleancut.cli import _apply_common

    config = Config.load_defaults()
    _apply_common(cli_args("scan", "movie.mp4", "--action", "profanity=replace"), config)
    edl = scan_words([Word(1, 1.3, "damn")], config).pad(.15).merge_overlapping(.5)
    assert edl.decisions[0].action == "replace"
    assert edl.decisions[0].start == pytest.approx(.96)
    assert edl.decisions[0].end == pytest.approx(1.5)
    assert edl.decisions[0].word_edits
    flags = jobs._category_flags({"categories": ["profanity"], "actions": {"profanity": "replace"}})
    assert "profanity=replace" in flags
    with pytest.raises(SystemExit, match="only supported"):
        _apply_common(cli_args("scan", "movie.mp4", "--action", "nudity=replace"), config)


def test_mixed_audio_actions_remain_distinct_and_overlap_fails_closed():
    first = replace(word_decision(1, 1.3), start=1, end=1.3, action="replace")
    for start in [1.3, 1.5]:
        second = replace(word_decision(start, start+.3), start=start, end=start+.3)
        merged = EditDecisionList(decisions=[first, second]).merge_overlapping(.5)
        assert [d.action for d in merged] == ["replace", "mute"]
    overlap = replace(word_decision(), start=1.2, end=1.6)
    merged = EditDecisionList(decisions=[first, overlap]).merge_overlapping(.5)
    assert merged.decisions[0].action == "mute"
    assert resolve_action("profanity+sex", {"profanity": "replace", "sex": "mute"}) == "mute"


def test_only_selected_replace_words_generate_and_background_supports_both(tmp_path):
    from cleancut.background import eligible_mutes

    edl = EditDecisionList(decisions=[word_decision(action="replace"), word_decision(4, 4.5)])
    assert len(eligible_words(edl, [], include_mutes=False)) == 1
    assert len(eligible_mutes(edl, [])) == 2
    with patch("cleancut.speech.check_service", side_effect=OSError("offline")) as check:
        assert prepare_replacements(tmp_path / "video", edl, [], Config(), tmp_path) == []
        check.assert_called_once()
    assert [d.action for d in edl] == ["replace", "mute"]
    with patch("cleancut.speech.check_service") as check:
        only_mute = EditDecisionList(decisions=[word_decision()])
        assert prepare_replacements(tmp_path / "video", only_mute, [], Config(), tmp_path) == []
        check.assert_not_called()


def test_replacement_never_overlays_an_explicit_overlapping_mute():
    edl = EditDecisionList(decisions=[word_decision(action="replace"), EditDecision(2, 3, "mute", "sex")])
    assert eligible_words(edl, [], include_mutes=False) == []
    assert eligible_words(EditDecisionList(decisions=[word_decision(action="replace")]), [Range(2, 3)]) == []


def test_settings_page_has_connection_not_replacement_mode(local_jobs):
    html = local_jobs.get("/settings").get_data(as_text=True)
    assert 'id="profanity-audio"' not in html
    assert 'id="speech-host"' in html
    assert 'id="actions"' not in html
    assert "last scan selections" in html


def test_edl_replace_loads_reference_even_when_subtitles_disabled(tmp_path):
    from cleancut.pipeline import PipelineOptions, run_full

    path = tmp_path / "edl.json"
    edl = EditDecisionList(decisions=[word_decision(action="replace")])
    edl.to_json(path)
    opts = PipelineOptions(video=tmp_path / "movie.mp4", output=tmp_path / "out.mp4",
                           edl_in=path, burn_subs=False, soft_subs=False)
    with patch("cleancut.pipeline._get_subtitles_and_words", return_value=(["reference"], [])) as read, \
         patch("cleancut.pipeline.render", return_value=opts.output) as render:
        run_full(opts, Config())
    read.assert_called_once()
    assert render.call_args.args[1] == ["reference"]


def test_existing_active_job_does_not_change_defaults(local_jobs, tmp_path):
    video = tmp_path / "movie.mp4"
    video.touch()
    active = jobs.create_job("scan", str(video))
    with patch("webapp.app.library.resolve_safe", return_value=video):
        response = local_jobs.post("/api/scan", json={"path": str(video), "actions": {"profanity": "replace"}})
    assert response.json["job_id"] == active and response.json["existing"]
    assert not settings._SETTINGS_PATH.exists()


def test_settings_ignore_invalid_actions_and_keep_other_preferences(local_jobs):
    settings.save({"actions": {"profanity": "replace", "nudity": "replace", "bogus": "cut"},
                   "categories": [["profanity"]], "speech_host": "http://localhost:8765"})
    cfg = settings.load()
    assert cfg["actions"]["profanity"] == "replace"
    assert cfg["actions"]["nudity"] == "cut" and "bogus" not in cfg["actions"]
    assert cfg["categories"] == settings.DEFAULTS["categories"]
    assert cfg["speech_host"] == "http://localhost:8765"
