#!/bin/zsh
set -euo pipefail
script_dir=${0:A:h}
project_dir=${script_dir:h}
speech_venv=$project_dir/.venv-speech
if [[ ! -x $speech_venv/bin/python ]]; then
  print -u2 "Run ./macos/install-speech.sh on this AI Mac first."
  exit 2
fi
missing_packages=$("$speech_venv/bin/python" -c '
import importlib.util
packages = ("flask", "waitress", "mlx_audio", "huggingface_hub", "srt")
print(", ".join(name for name in packages if importlib.util.find_spec(name) is None))
')
if [[ -n $missing_packages ]]; then
  print -u2 "Speech installation is incomplete (missing: $missing_packages)."
  print -u2 "Run ./macos/install-speech.sh again."
  exit 2
fi
export HF_HOME=${CLEANCUT_SPEECH_MODEL_DIR:-"$HOME/Library/Application Support/CleanCut/speech-models"}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$project_dir
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
cd "$project_dir"
exec caffeinate -dimsu "$speech_venv/bin/python" -u -m cleancut.speech_server
