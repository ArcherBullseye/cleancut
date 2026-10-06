#!/bin/zsh
set -euo pipefail

script_dir=${0:A:h}
project_dir=${script_dir:h}
venv_dir=$project_dir/.venv-macos

if [[ ! -x $venv_dir/bin/python ]]; then
  print -u2 "CleanCut is not installed. Run macos/install.sh first."
  exit 2
fi

missing_packages=$("$venv_dir/bin/python" -c '
import importlib.util
packages = ("flask", "waitress", "whisper", "torch")
print(", ".join(name for name in packages if importlib.util.find_spec(name) is None))
')
if [[ -n $missing_packages ]]; then
  print -u2 "CleanCut installation is incomplete (missing: $missing_packages)."
  print -u2 "From $project_dir, run ./macos/install.sh again."
  exit 2
fi

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
export DATA_DIR=${CLEANCUT_DATA_DIR:-"$HOME/Library/Application Support/CleanCut"}
export OUTPUT_DIR=${CLEANCUT_OUTPUT_DIR:-"$DATA_DIR/output"}
export MEDIA_ROOTS=${CLEANCUT_MEDIA_ROOTS:-"$HOME/Movies:/Volumes"}
export CLEANCUT_CACHE_DIR="$DATA_DIR/cache/cleancut"
export XDG_CACHE_HOME="$DATA_DIR/cache"
export HF_HOME="$DATA_DIR/models/huggingface"
export TORCH_HOME="$DATA_DIR/models/torch"
export CLEANCUT_MODEL_DIR="$DATA_DIR/models/cleancut"
export OLLAMA_HOST=${CLEANCUT_OLLAMA_HOST:-http://127.0.0.1:11434}
export HOST=${CLEANCUT_HOST:-127.0.0.1}
export PORT=${CLEANCUT_PORT:-3000}
export CLEANCUT_VERSION=2.0.0-mac-beta.7
export PYTHONPATH=$project_dir
export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$CLEANCUT_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$CLEANCUT_MODEL_DIR"
cd "$project_dir"

print "CleanCut Mac: http://$HOST:$PORT"
print "Media roots: $MEDIA_ROOTS"
print "Data: $DATA_DIR"

# Keep the Mac awake while the server or a queued job is active. Closing a
# laptop lid can still suspend it unless macOS is in supported clamshell mode.
exec caffeinate -dimsu "$venv_dir/bin/python" -m webapp.app
