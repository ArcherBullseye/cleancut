"""Job queue: runs cleancut as a subprocess, one job at a time.

A scan is tens of minutes to hours on a 4-core box, so jobs must survive a
browser refresh and an app restart. State lives in SQLite under DATA_DIR and
the work runs in a child process -- which means an OOM or a hard crash inside
a detector takes down the job, not the web server, and cancelling is a signal
rather than an un-interruptible thread.
"""

from __future__ import annotations

import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from webapp import settings as settings_store
from webapp.paths import CACHE_DIR, DB_PATH, MODELS_DIR, OUTPUT_DIR, ensure_dirs, job_dir

# The directory the `cleancut` package sits in, which is what the child process
# needs on its path. Derived rather than hardcoded to /app so the server runs
# the same way outside the container.
APP_ROOT = Path(os.environ.get("APP_ROOT", str(Path(__file__).resolve().parent.parent)))

QUEUED, RUNNING, DONE, FAILED, CANCELED = "queued", "running", "done", "failed", "canceled"
ACTIVE_STATUSES = (QUEUED, RUNNING)

# Log line -> stage label shown in the UI. Ordered: first match wins, so more
# specific markers must come before general ones.
_STAGE_MARKERS: list[tuple[str, str]] = [
    ("Reading subtitles", "Reading subtitles"),
    ("Found sidecar", "Reading subtitles"),
    ("Extracting embedded", "Extracting subtitles"),
    ("Whisper transcript loaded from cache", "Transcript (cached)"),
    ("Transcribing with Whisper", "Transcribing speech"),
    ("Saved transcript", "Transcript saved"),
    ("Scanning", "Scanning dialogue"),
    ("Detecting shot boundaries", "Detecting shots"),
    ("Visual scan", "Visual scan"),
    ("LLM dialogue scan", "LLM dialogue scan"),
    ("Audio event scan", "Audio event scan"),
    ("Density clustering", "Density clustering"),
    ("VLM scene scan", "VLM scene scan"),
    ("Suppressed", "Corroboration"),
    ("Wrote EDL", "Writing EDL"),
    ("Applying", "Applying cuts"),
    ("Muting", "Muting and subtitles"),
    ("Encoder", "Encoding"),
    ("Wrote report", "Writing report"),
]

_PERCENT_RE = re.compile(r"(\d{1,3})%")
# ffmpeg reports position, never a percentage -- against the probed duration
# that becomes one. Without it the bar sits at zero for the whole encode.
_FFMPEG_TIME_RE = re.compile(r"time=(\d+):(\d\d):(\d\d)(?:\.(\d+))?")
# Transient redraw lines -- ffmpeg's status and tqdm's bars (Whisper's model
# download alone emits thousands). Worth a progress update, not a log entry each:
# a 60-second clip wrote a 209 KB log before this covered tqdm too.
_FFMPEG_STATUS_RE = re.compile(r"^(frame|size)=\s*\S|^\s*\d{1,3}%\|")

_procs: dict[int, subprocess.Popen] = {}
_procs_lock = threading.Lock()
_wake = threading.Event()
_worker_started = False
_worker_lock = threading.Lock()


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    ensure_dirs()
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                kind         TEXT NOT NULL,
                status       TEXT NOT NULL,
                video_path   TEXT NOT NULL,
                title        TEXT NOT NULL DEFAULT '',
                preset       TEXT NOT NULL DEFAULT '',
                options      TEXT NOT NULL DEFAULT '{}',
                edl_path     TEXT NOT NULL DEFAULT '',
                output_path  TEXT NOT NULL DEFAULT '',
                stage        TEXT NOT NULL DEFAULT '',
                progress     REAL NOT NULL DEFAULT 0,
                error        TEXT NOT NULL DEFAULT '',
                parent_id    INTEGER,
                duration     REAL NOT NULL DEFAULT 0,
                created_at   REAL NOT NULL,
                started_at   REAL,
                finished_at  REAL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        # A job left RUNNING can only be a crash or a restart mid-run -- the
        # child process is long gone, so never leave it spinning in the UI.
        conn.execute(
            "UPDATE jobs SET status=?, error=? WHERE status=?",
            (FAILED, "Interrupted by an app restart.", RUNNING),
        )


def create_job(
    kind: str,
    video_path: str,
    *,
    title: str = "",
    preset: str = "",
    options: dict[str, Any] | None = None,
    edl_path: str = "",
    output_path: str = "",
    parent_id: int | None = None,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO jobs (kind, status, video_path, title, preset, options,
                              edl_path, output_path, parent_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kind, QUEUED, video_path, title or Path(video_path).stem, preset,
                json.dumps(options or {}), edl_path, output_path, parent_id, time.time(),
            ),
        )
        job_id = int(cur.lastrowid)
    job_dir(job_id).mkdir(parents=True, exist_ok=True)
    _wake.set()
    return job_id


def update_job(job_id: int, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with _connect() as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))


def get_job(job_id: int) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(limit: int = 100) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def active_job_for(video_path: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE video_path=? AND status IN (?, ?) ORDER BY id DESC LIMIT 1",
            (video_path, QUEUED, RUNNING),
        ).fetchone()
    return dict(row) if row else None


def delete_job(job_id: int) -> None:
    cancel_job(job_id)
    with _connect() as conn:
        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    import shutil
    shutil.rmtree(job_dir(job_id), ignore_errors=True)


def cancel_job(job_id: int) -> bool:
    job = get_job(job_id)
    if not job or job["status"] not in ACTIVE_STATUSES:
        return False
    update_job(job_id, status=CANCELED, finished_at=time.time(), stage="Canceled")
    with _procs_lock:
        proc = _procs.get(job_id)
    if proc and proc.poll() is None:
        # The child spawns ffmpeg helpers; killing the group takes those too.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
    return True


def log_path(job_id: int) -> Path:
    return job_dir(job_id) / "job.log"


def read_log(job_id: int, max_bytes: int = 60_000) -> str:
    path = log_path(job_id)
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
            fh.readline()  # drop the partial line
        return fh.read().decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# command construction
# --------------------------------------------------------------------------

def _probe_duration(video: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, timeout=60,
        )
        return float(out.stdout.strip())
    except (ValueError, OSError, subprocess.SubprocessError):
        return 0.0


def transcript_path(job_id: int) -> Path:
    return job_dir(job_id) / "transcript.srt"


def edl_path_for(job_id: int) -> Path:
    return job_dir(job_id) / "edl.json"


def _category_flags(opts: dict[str, Any]) -> list[str]:
    """Translate the UI's category/action selections into CLI flags."""
    args: list[str] = []
    all_categories = ("profanity", "drugs", "sex", "violence", "nudity")
    enabled = set(opts.get("categories") or [])
    for cat in all_categories:
        args += ["--enable-category" if cat in enabled else "--disable-category", cat]
    for cat, action in (opts.get("actions") or {}).items():
        if cat in all_categories and action in ("mute", "cut", "keep"):
            args += ["--action", f"{cat}={action}"]
    return args


def _ollama_flags(opts: dict[str, Any], preset: str) -> list[str]:
    """LLM/VLM need Ollama; without a host they must be switched off explicitly
    or the `thorough` preset would enable them and every scan would stall
    retrying a dead connection."""
    host = (opts.get("ollama_host") or "").strip()
    if not host:
        return ["--no-llm", "--no-vlm"]
    args = ["--llm-host", host]
    if opts.get("llm_model"):
        args += ["--llm-model", opts["llm_model"]]
    if opts.get("vlm_model"):
        args += ["--vlm-model", opts["vlm_model"]]
    # Presets other than thorough leave these off; an explicit opt-in overrides.
    if opts.get("use_llm"):
        args.append("--use-llm")
    elif opts.get("use_llm") is False:
        args.append("--no-llm")
    if opts.get("use_vlm"):
        args.append("--use-vlm")
    elif opts.get("use_vlm") is False:
        args.append("--no-vlm")
    return args


def build_scan_command(job: dict[str, Any]) -> list[str]:
    opts = json.loads(job["options"] or "{}")
    job_id = job["id"]
    cmd = [
        sys.executable, "-u", "-m", "cleancut", "scan", job["video_path"],
        "--preset", job["preset"] or "balanced",
        "-o", str(edl_path_for(job_id)),
        "--save-transcript", str(transcript_path(job_id)),
        "--prefer-language", opts.get("prefer_language") or "eng",
    ]
    cmd += _category_flags(opts)
    cmd += _ollama_flags(opts, job["preset"])
    if opts.get("use_audio_events") is False:
        cmd.append("--no-audio-events")
    elif opts.get("use_audio_events") is True:
        cmd.append("--use-audio-events")
    if not opts.get("use_visual", True):
        cmd.append("--no-visual")
    if opts.get("allow_solo_visual"):
        cmd.append("--allow-solo-visual")
    if opts.get("audio_track") not in (None, ""):
        cmd += ["--audio-track", str(opts["audio_track"])]
    return cmd


def build_render_command(job: dict[str, Any]) -> list[str]:
    opts = json.loads(job["options"] or "{}")
    cmd = [
        sys.executable, "-u", "-m", "cleancut", "clean", job["video_path"],
        "--edl", job["edl_path"],
        "-o", job["output_path"],
        "--encoder", opts.get("encoder") or "libx264",
        "--quality", str(opts.get("quality", 20)),
        "--prefer-language", opts.get("prefer_language") or "eng",
    ]
    # Reuse the scan's transcript. Without it, `clean` re-runs Whisper purely to
    # have subtitles to soften -- hours of work already done.
    subs = opts.get("subs_path") or ""
    if subs and Path(subs).exists():
        cmd += ["--subs", subs]
    # With no saved transcript the subtitles are left for cleancut to resolve --
    # a sidecar or embedded track costs nothing to re-read, and if the scan had
    # to run Whisper it both saved a transcript and cached the result, so this
    # cannot turn into a second hours-long transcription.
    mode = opts.get("subtitle_mode", "soft")
    if mode == "none":
        cmd.append("--no-burn-subs")
    elif mode == "soft":
        cmd.append("--soft-subs")
    return cmd


def _child_env() -> dict[str, str]:
    """Pin every model/cache location into DATA_DIR.

    Whisper, HuggingFace AST and torch all download weights on first use. Left
    at their defaults those land in the container filesystem and are lost --
    and re-downloaded, gigabytes at a time -- on every app update.
    """
    env = dict(os.environ)
    env.update({
        "CLEANCUT_CACHE_DIR": str(CACHE_DIR / "cleancut"),
        "XDG_CACHE_HOME": str(CACHE_DIR),
        "HF_HOME": str(MODELS_DIR / "huggingface"),
        "TORCH_HOME": str(MODELS_DIR / "torch"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(APP_ROOT),
        # No GPU on an Umbrel Home; oversubscribing 4 cores makes it slower.
        "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "4"),
        "TOKENIZERS_PARALLELISM": "false",
        "TERM": "dumb",
        "NO_COLOR": "1",
        "COLUMNS": "200",
    })
    return env


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def _stage_for(line: str) -> str | None:
    for marker, label in _STAGE_MARKERS:
        if marker in line:
            return label
    return None


def _run_job(job: dict[str, Any]) -> None:
    job_id = job["id"]
    jd = job_dir(job_id)
    jd.mkdir(parents=True, exist_ok=True)
    duration = _probe_duration(Path(job["video_path"]))
    update_job(job_id, status=RUNNING, started_at=time.time(), stage="Starting",
               progress=0, error="", duration=duration)

    cmd = build_scan_command(job) if job["kind"] == "scan" else build_render_command(job)
    log = log_path(job_id)
    with log.open("w", encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(cmd)}\n\n")
        fh.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(APP_ROOT),
                env=_child_env(),
                start_new_session=True,  # own process group, so cancel kills ffmpeg too
                bufsize=0,
            )
        except OSError as e:
            update_job(job_id, status=FAILED, finished_at=time.time(),
                       error=f"Failed to start: {e}", stage="Failed")
            fh.write(f"\nFailed to start: {e}\n")
            return

        with _procs_lock:
            _procs[job_id] = proc

        state = {"stage": "", "progress": 0.0, "flushed": 0.0, "logged": 0.0}

        def handle(line: str) -> None:
            now = time.time()
            status_line = bool(_FFMPEG_STATUS_RE.match(line))
            # ffmpeg emits a status line several times a second for hours. Keep
            # a heartbeat in the log without writing gigabytes of it.
            if not status_line or now - state["logged"] > 15.0:
                if status_line:
                    state["logged"] = now
                fh.write(line + "\n")
                fh.flush()

            new_stage = _stage_for(line)
            if new_stage and new_stage != state["stage"]:
                state["stage"] = new_stage
                state["progress"] = 0.0
                update_job(job_id, stage=new_stage, progress=0)

            pct = None
            tm = _FFMPEG_TIME_RE.search(line)
            if tm and duration > 0:
                position = int(tm.group(1)) * 3600 + int(tm.group(2)) * 60 + int(tm.group(3))
                pct = min(100.0, position / duration * 100.0)
            else:
                pm = _PERCENT_RE.search(line)
                if pm:
                    pct = min(100.0, float(pm.group(1)))
            if pct is not None and abs(pct - state["progress"]) >= 1.0:
                state["progress"] = pct
                if now - state["flushed"] > 2.0:
                    state["flushed"] = now
                    update_job(job_id, progress=pct)

        try:
            assert proc.stdout is not None
            fd = proc.stdout.fileno()
            buffer = b""
            while True:
                try:
                    chunk = os.read(fd, 8192)
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk
                # Split on \r as well as \n: ffmpeg and tqdm redraw in place with
                # carriage returns and can go hours without ever writing a
                # newline, which would freeze the log and the progress bar.
                parts = re.split(rb"[\r\n]", buffer)
                buffer = parts.pop()
                for part in parts:
                    text = part.decode("utf-8", errors="replace").rstrip()
                    if text:
                        handle(text)
            if buffer.strip():
                handle(buffer.decode("utf-8", errors="replace").rstrip())
        finally:
            code = proc.wait()
            with _procs_lock:
                _procs.pop(job_id, None)

    current = get_job(job_id) or {}
    if current.get("status") == CANCELED:
        return
    if code == 0:
        _finish_ok(job_id, job)
    else:
        tail = read_log(job_id, 4000).strip().splitlines()[-6:]
        update_job(job_id, status=FAILED, finished_at=time.time(), stage="Failed",
                   error="\n".join(tail) or f"exited with code {code}")


def _finish_ok(job_id: int, job: dict[str, Any]) -> None:
    fields: dict[str, Any] = {
        "status": DONE, "finished_at": time.time(), "stage": "Complete", "progress": 100,
    }
    if job["kind"] == "scan":
        edl = edl_path_for(job_id)
        fields["edl_path"] = str(edl) if edl.exists() else ""
        if not edl.exists():
            fields.update(status=FAILED, stage="Failed", error="Scan produced no EDL.")
    else:
        out = Path(job["output_path"])
        # ffmpeg creates the output file before it writes to it, so a render
        # that dies mid-encode leaves a zero-byte file behind. Existence alone
        # is not proof of success.
        if not out.exists() or out.stat().st_size == 0:
            fields.update(status=FAILED, stage="Failed",
                          error="Render finished but produced no output file.")
    update_job(job_id, **fields)

    if fields.get("status") == DONE and job["kind"] == "scan":
        opts = json.loads(job["options"] or "{}")
        if opts.get("auto_render"):
            queue_render(job_id)


def queue_render(scan_job_id: int, *, overrides: dict[str, Any] | None = None) -> int | None:
    """Queue a render of a finished scan's EDL."""
    scan = get_job(scan_job_id)
    if not scan or scan["kind"] != "scan" or not scan["edl_path"]:
        return None
    cfg = settings_store.load()
    opts = json.loads(scan["options"] or "{}")
    transcript = transcript_path(scan_job_id)
    video = Path(scan["video_path"])
    out_dir = Path(opts.get("output_dir") or cfg.get("output_dir") or "") or OUTPUT_DIR
    if not out_dir.is_absolute():
        out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{video.stem}.clean.mp4"

    render_opts = {
        "encoder": cfg.get("encoder", "libx264"),
        "quality": cfg.get("quality", 20),
        "subtitle_mode": cfg.get("subtitle_mode", "soft"),
        "prefer_language": opts.get("prefer_language", "eng"),
        "subs_path": str(transcript) if transcript.exists() else "",
    }
    render_opts.update(overrides or {})
    return create_job(
        "render", str(video), title=scan["title"], preset=scan["preset"],
        options=render_opts, edl_path=scan["edl_path"], output_path=str(output),
        parent_id=scan_job_id,
    )


def _next_queued() -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status=? ORDER BY id ASC LIMIT 1", (QUEUED,)
        ).fetchone()
    return dict(row) if row else None


def _worker_loop() -> None:
    while True:
        job = _next_queued()
        if job is None:
            _wake.wait(timeout=5)
            _wake.clear()
            continue
        try:
            _run_job(job)
        except Exception as e:  # a worker crash must not stop the queue
            update_job(job["id"], status=FAILED, finished_at=time.time(),
                       stage="Failed", error=f"{type(e).__name__}: {e}")


def start_worker() -> None:
    """Start the single background worker. Idempotent."""
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        _worker_started = True
    threading.Thread(target=_worker_loop, name="cleancut-worker", daemon=True).start()
