#!/usr/bin/env python3
"""Native macOS readiness check. Makes no changes to the system."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def status(ok: bool, label: str, detail: str) -> None:
    mark = "OK" if ok else "FAIL"
    print(f"[{mark:4}] {label}: {detail}")


def warning(label: str, detail: str) -> None:
    print(f"[WARN] {label}: {detail}")


def command_output(*args: str) -> str:
    return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True)


def main() -> int:
    failures = 0
    warnings = 0
    native = platform.system() == "Darwin" and platform.machine() == "arm64"
    status(native, "platform", f"{platform.system()} {platform.machine()}")
    failures += not native

    py_ok = sys.version_info >= (3, 10)
    status(py_ok, "python", platform.python_version())
    failures += not py_ok

    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(name)
        status(path is not None, name, path or "not found on PATH")
        failures += path is None

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            encoders = command_output(ffmpeg, "-hide_banner", "-encoders")
            for encoder in ("h264_videotoolbox", "hevc_videotoolbox"):
                ok = encoder in encoders
                status(ok, encoder, "available" if ok else "missing from FFmpeg build")
                failures += not ok
        except (OSError, subprocess.SubprocessError) as e:
            status(False, "FFmpeg encoders", str(e))
            failures += 1

    for module in ("flask", "waitress", "cv2", "nudenet", "whisper", "torch"):
        ok = importlib.util.find_spec(module) is not None
        status(ok, f"python:{module}", "installed" if ok else "missing")
        failures += not ok

    if importlib.util.find_spec("torch") is not None:
        import torch

        available = bool(torch.backends.mps.is_available())
        if available:
            status(True, "PyTorch Metal", "available")
        else:
            warning("PyTorch Metal", "unavailable; model inference will use the CPU")
            warnings += 1

    data_dir = Path(os.environ.get(
        "DATA_DIR", str(Path.home() / "Library" / "Application Support" / "CleanCut")
    )).expanduser()
    disk_target = data_dir if data_dir.exists() else data_dir.parent
    free_gib = shutil.disk_usage(disk_target).free / (1024 ** 3)
    disk_ok = free_gib >= 50
    if disk_ok:
        status(True, "free disk", f"{free_gib:.1f} GiB at {disk_target}")
    else:
        warning(
            "free disk",
            f"{free_gib:.1f} GiB at {disk_target}; 50 GiB is recommended for 4K jobs",
        )
        warnings += 1

    print()
    if failures:
        print(f"CleanCut Mac is not ready: {failures} required check(s) failed.")
        return 1
    suffix = f" with {warnings} warning(s)" if warnings else ""
    print(f"CleanCut Mac is ready for native Apple Silicon processing{suffix}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
