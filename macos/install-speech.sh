#!/bin/zsh
set -euo pipefail
script_dir=${0:A:h}
project_dir=${script_dir:h}
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
  print -u2 "CleanCut speech requires an Apple Silicon Mac."
  exit 2
fi
macos_version=$(sw_vers -productVersion)
if [[ ${macos_version%%.*} -lt 14 ]]; then
  print -u2 "The speech runtime requires macOS 14 (Sonoma) or newer."
  exit 2
fi
if ! command -v brew >/dev/null 2>&1; then
  print -u2 "Install Homebrew from https://brew.sh first."
  exit 2
fi
brew install python@3.12 ffmpeg
python_bin=$(brew --prefix python@3.12)/bin/python3.12
speech_venv=$project_dir/.venv-speech
"$python_bin" -m venv "$speech_venv"
"$speech_venv/bin/python" -m pip install --upgrade pip wheel
"$speech_venv/bin/python" -m pip install -r "$project_dir/requirements-speech.txt"
export HF_HOME=${CLEANCUT_SPEECH_MODEL_DIR:-"$HOME/Library/Application Support/CleanCut/speech-models"}
export CLEANCUT_SPEECH_MODEL=${CLEANCUT_SPEECH_MODEL:-mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit}
"$speech_venv/bin/python" -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ["CLEANCUT_SPEECH_MODEL"])'
print "Speech installed. Run ./macos/run-speech.sh on this AI Mac."
