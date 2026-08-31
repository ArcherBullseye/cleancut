"""The `python -m cleancut` entry point must propagate failures.

webapp/jobs.py drives the CLI as a subprocess and decides a job's fate from the
exit code. __main__.py called main() and discarded its return value, so the
process exited 0 no matter what: a render that died inside ffmpeg was recorded
as Complete, next to a zero-byte output file.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_module(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    return subprocess.run(
        [sys.executable, "-m", "cleancut", *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, timeout=120,
    )


class TestModuleEntryPointExitCode:
    def test_failure_propagates_nonzero(self):
        r = _run_module("scan", "/nonexistent/definitely-not-here.mkv")
        assert r.returncode != 0, (
            "python -m cleancut exited 0 on a failing command; every subprocess "
            f"caller will read this as success.\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        )

    def test_help_still_succeeds(self):
        assert _run_module("--help").returncode == 0
