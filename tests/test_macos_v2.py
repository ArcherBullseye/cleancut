from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from cleancut.config import Config
from cleancut.editor import _run_with_encoder_fallback, _video_encoder_args
from cleancut.probe import Stream, probe_streams


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_mac_install_uses_whisper_release_without_legacy_pkg_resources_build():
    requirements = (PROJECT_ROOT / "requirements.txt").read_text()
    assert "openai-whisper==20250625" in requirements
    assert "openai-whisper==20240930" not in requirements


def test_mac_launcher_detects_an_incomplete_virtualenv_before_startup():
    launcher = (PROJECT_ROOT / "macos" / "run.sh").read_text()
    assert "importlib.util.find_spec" in launcher
    assert "CleanCut installation is incomplete" in launcher


def _video(**overrides) -> Stream:
    fields = {
        "index": 0,
        "codec_name": "hevc",
        "codec_type": "video",
        "pix_fmt": "yuv420p10le",
        "profile": "Main 10",
        "level": 150,
        "codec_tag_string": "hvc1",
        "width": 3840,
        "height": 2160,
        "avg_frame_rate": "24000/1001",
        "color_range": "tv",
        "color_space": "bt2020nc",
        "color_transfer": "smpte2084",
        "color_primaries": "bt2020",
    }
    fields.update(overrides)
    return Stream(**fields)


def test_probe_keeps_dimensions_framerate_and_hdr_metadata(tmp_path):
    payload = {"streams": [{
        "index": 0,
        "codec_name": "hevc",
        "codec_type": "video",
        "pix_fmt": "yuv420p10le",
        "profile": "Main 10",
        "level": 150,
        "codec_tag_string": "hvc1",
        "width": 3840,
        "height": 2160,
        "avg_frame_rate": "24000/1001",
        "color_range": "tv",
        "color_space": "bt2020nc",
        "color_transfer": "smpte2084",
        "color_primaries": "bt2020",
    }]}
    with patch("cleancut.probe.shutil.which", return_value="/usr/bin/ffprobe"), \
         patch("cleancut.probe.subprocess.check_output", return_value=json.dumps(payload).encode()):
        stream = probe_streams(tmp_path / "hdr.mp4")[0]

    assert (stream.width, stream.height) == (3840, 2160)
    assert stream.avg_frame_rate == "24000/1001"
    assert stream.profile == "Main 10"
    assert stream.level == 150
    assert stream.codec_tag_string == "hvc1"
    assert stream.is_hdr is True


def test_auto_encoder_uses_hevc_videotoolbox_for_hdr_on_mac(tmp_path):
    config = Config.load_defaults()
    config.encoder = "auto"
    with patch("cleancut.config.platform.system", return_value="Darwin"), \
         patch("cleancut.probe.probe_streams", return_value=[_video()]):
        assert config.resolved_encoder(tmp_path / "hdr.mov") == "hevc_videotoolbox"


def test_hevc_videotoolbox_is_main10_hvc1_and_preserves_color_description():
    args = _video_encoder_args("hevc_videotoolbox", 20, _video())
    assert args[args.index("-c:v") + 1] == "hevc_videotoolbox"
    assert args[args.index("-pix_fmt") + 1] == "p010le"
    assert args[args.index("-profile:v") + 1] == "main10"
    assert args[args.index("-tag:v") + 1] == "hvc1"
    assert args[args.index("-allow_sw") + 1] == "1"
    assert args[args.index("-color_trc") + 1] == "smpte2084"
    assert args[args.index("-r") + 1] == "24000/1001"
    assert args[args.index("-fps_mode") + 1] == "cfr"


def test_videotoolbox_failure_retries_with_software_encoder(tmp_path):
    source = _video(codec_name="h264", pix_fmt="yuv420p", color_transfer="")
    hardware_args = _video_encoder_args("videotoolbox", 20, source)
    partial = tmp_path / ".output.partial.mp4"
    partial.write_bytes(b"failed render")
    cmd = ["ffmpeg", "-i", "input.mp4", *hardware_args, str(partial)]

    with patch(
        "cleancut.editor.subprocess.run",
        side_effect=[subprocess.CalledProcessError(1, cmd), subprocess.CompletedProcess(cmd, 0)],
    ) as run:
        _run_with_encoder_fallback(
            cmd,
            encoder="videotoolbox",
            encoder_args=hardware_args,
            source=source,
            quality=20,
            partial=partial,
        )

    assert not partial.exists()
    retry = run.call_args_list[1].args[0]
    assert retry[retry.index("-c:v") + 1] == "libx264"
    assert "h264_videotoolbox" not in retry


def test_analysis_proxy_is_720p_atomic_and_hardware_encoded(tmp_path, monkeypatch):
    from cleancut import cache
    from cleancut.proxy import analysis_source

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"proxy")
        return subprocess.CompletedProcess(cmd, 0)

    with patch("cleancut.proxy.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("cleancut.proxy.probe_streams", return_value=[_video()]), \
         patch("cleancut.proxy._encoder_available", return_value=True), \
         patch("cleancut.proxy.platform.system", return_value="Darwin"), \
         patch("cleancut.proxy.subprocess.run", side_effect=fake_run):
        result = analysis_source(source, 720)

    assert result.exists()
    cmd = calls[-1]
    assert "scale=1280:720:flags=fast_bilinear,format=yuv420p" in cmd
    assert cmd[cmd.index("-c:v") + 1] == "h264_videotoolbox"
    assert cmd[cmd.index("-fps_mode") + 1] == "vfr"
    assert ".partial.mp4" in cmd[-1]


def test_analysis_proxy_skips_sources_already_below_limit(tmp_path):
    from cleancut.proxy import analysis_source

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    with patch("cleancut.proxy.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch(
             "cleancut.proxy.probe_streams",
             return_value=[_video(width=1280, height=720, color_transfer="")],
         ):
        assert analysis_source(source, 720) == source


def test_cache_clear_removes_proxy_artifact_and_metadata(tmp_path, monkeypatch):
    from cleancut import cache

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    key = cache.config_hash(height=720)
    artifact = cache.artifact_path(source, "analysis_proxy", key, ".mp4")
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"proxy")
    cache.save(source, "analysis_proxy", key, {"path": str(artifact)})

    assert cache.clear(source, "analysis_proxy") == 2
    assert not artifact.exists()


def test_quick_validation_rejects_duration_mismatch(tmp_path):
    from cleancut.media_validation import validate_media

    output = tmp_path / "output.mp4"
    output.write_bytes(b"video")
    with patch("cleancut.media_validation.probe_duration", return_value=80.0), \
         patch(
             "cleancut.media_validation.probe_streams",
             return_value=[_video(width=1920, height=1080)],
         ), pytest.raises(RuntimeError, match="duration mismatch"):
        validate_media(output, expected_duration=100.0, mode="quick")


def test_quick_validation_rejects_incompatible_apple_codec_level(tmp_path):
    from cleancut.media_validation import validate_media

    output = tmp_path / "output.mp4"
    output.write_bytes(b"video")
    with patch("cleancut.media_validation.probe_duration", return_value=10.0), \
         patch(
             "cleancut.media_validation.probe_streams",
             return_value=[_video(codec_name="hevc", level=255)],
         ), pytest.raises(RuntimeError, match="codec level"):
        validate_media(output, expected_duration=10.0, mode="quick")


def test_full_validation_rejects_decode_errors(tmp_path):
    from cleancut.media_validation import validate_media

    output = tmp_path / "output.mp4"
    output.write_bytes(b"video")
    error = subprocess.CalledProcessError(1, ["ffmpeg"], stderr="corrupt frame")
    with patch("cleancut.media_validation.probe_duration", return_value=10.0), \
         patch(
             "cleancut.media_validation.probe_streams",
             return_value=[_video(width=1920, height=1080)],
         ), \
         patch("cleancut.media_validation.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("cleancut.media_validation.subprocess.run", side_effect=error), \
         pytest.raises(RuntimeError, match="corrupt frame"):
        validate_media(output, expected_duration=10.0, mode="full")


def test_web_commands_carry_proxy_and_render_validation_settings():
    from webapp.jobs import build_render_command, build_scan_command

    scan = {
        "id": 7,
        "video_path": "/video/movie.mp4",
        "preset": "balanced",
        "options": json.dumps({
            "analysis_proxy": True,
            "analysis_height": 720,
            "nudity_model": "accurate",
            "categories": [],
            "actions": {},
            "ollama_host": "http://127.0.0.1:11434",
            "llm_model": "qwen3.5:9b",
            "vlm_model": "qwen3.5:9b",
            "use_llm": True,
            "use_vlm": True,
        }),
    }
    scan_cmd = build_scan_command(scan)
    assert scan_cmd[scan_cmd.index("--analysis-height") + 1] == "720"
    assert scan_cmd[scan_cmd.index("--nudity-model") + 1] == "accurate"
    assert scan_cmd[scan_cmd.index("--llm-host") + 1] == "http://127.0.0.1:11434"
    assert scan_cmd[scan_cmd.index("--llm-model") + 1] == "qwen3.5:9b"
    assert scan_cmd[scan_cmd.index("--vlm-model") + 1] == "qwen3.5:9b"
    assert "--use-llm" in scan_cmd
    assert "--use-vlm" in scan_cmd

    render = {
        "id": 8,
        "video_path": "/video/movie.mp4",
        "edl_path": "/data/movie.edl.json",
        "output_path": "/data/movie.clean.mp4",
        "options": json.dumps({
            "encoder": "auto",
            "quality": 20,
            "render_validation": "full",
        }),
    }
    render_cmd = build_render_command(render)
    assert render_cmd[render_cmd.index("--encoder") + 1] == "auto"
    assert render_cmd[render_cmd.index("--verify-render") + 1] == "full"
    assert render_cmd[render_cmd.index("-o") + 1].endswith("/8/rendered.mp4")
    assert render_cmd[render_cmd.index("-o") + 1] != render["output_path"]


def test_native_settings_migrate_legacy_umbrel_ollama_defaults():
    from webapp import settings

    old = {
        "ollama_host": "http://ollama_ollama_1:11434",
        "llm_model": "llama3.2:3b",
        "vlm_model": "moondream",
    }
    with patch("webapp.settings._IS_MAC", True), \
         patch("webapp.settings._read", return_value=old):
        migrated = settings.load()
    assert migrated["ollama_host"] == "http://127.0.0.1:11434"
    assert migrated["llm_model"] == "qwen3.5:9b"
    assert migrated["vlm_model"] == "qwen3.5:9b"


def test_publish_render_atomically_copies_local_file_to_nas(tmp_path):
    from webapp.jobs import publish_render

    local = tmp_path / "local" / "rendered.mp4"
    remote = tmp_path / "nas" / "Movie.clean.mp4"
    local.parent.mkdir()
    remote.parent.mkdir()
    local.write_bytes(b"verified-local-render")
    remote.write_bytes(b"previous-clean-copy")

    with patch("cleancut.media_validation.validate_media") as validate:
        publish_render(local, remote)

    assert remote.read_bytes() == b"verified-local-render"
    assert local.exists()
    assert not (remote.parent / ".Movie.clean.cleancut-uploading.mp4").exists()
    validate.assert_called_once()


def test_failed_nas_publish_preserves_previous_file(tmp_path):
    from webapp.jobs import publish_render

    local = tmp_path / "rendered.mp4"
    remote = tmp_path / "nas" / "Movie.clean.mp4"
    remote.parent.mkdir()
    local.write_bytes(b"new-render")
    remote.write_bytes(b"previous-clean-copy")

    with patch("webapp.jobs.shutil.copyfileobj", side_effect=OSError("NAS disconnected")), \
         pytest.raises(OSError, match="NAS disconnected"):
        publish_render(local, remote)

    assert remote.read_bytes() == b"previous-clean-copy"
    assert not (remote.parent / ".Movie.clean.cleancut-uploading.mp4").exists()


def test_queue_render_can_return_cleaned_video_beside_nas_source(tmp_path):
    from webapp import jobs

    nas = tmp_path / "Mounted NAS"
    nas.mkdir()
    video = nas / "Movie.mkv"
    video.write_bytes(b"source")
    scan = {
        "id": 12,
        "kind": "scan",
        "video_path": str(video),
        "title": "Movie",
        "preset": "balanced",
        "edl_path": str(tmp_path / "movie.edl.json"),
        "options": json.dumps({"output_location": "source"}),
    }
    with patch("webapp.jobs.get_job", return_value=scan), \
         patch("webapp.jobs.settings_store.load", return_value={}), \
         patch("webapp.jobs.create_job", return_value=13) as create:
        assert jobs.queue_render(12) == 13

    assert create.call_args.kwargs["output_path"] == str(nas / "Movie.clean.mp4")
