#!/bin/zsh
set -euo pipefail
script_dir=${0:A:h}
project_dir=${script_dir:h}
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
  print -u2 "CleanCut's separation installer requires an Apple Silicon Mac."
  exit 2
fi
if ! command -v brew >/dev/null 2>&1; then
  print -u2 "Install Homebrew from https://brew.sh first."
  exit 2
fi
brew install python@3.12 ffmpeg
python_bin=$(brew --prefix python@3.12)/bin/python3.12
separation_venv=$project_dir/.venv-separation
"$python_bin" -m venv "$separation_venv"
"$separation_venv/bin/python" -m pip install --upgrade pip wheel 'setuptools<81'
"$separation_venv/bin/python" -m pip install -r "$project_dir/requirements-separation.txt"
export PYTHONPATH=$project_dir
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
separation_data=${CLEANCUT_DATA_DIR:-"$HOME/Library/Application Support/CleanCut"}
separation_models=${CLEANCUT_SEPARATION_MODEL_DIR:-"$separation_data/separation-models"}
"$separation_venv/bin/python" -m cleancut.separation_worker --install "$separation_models"
"$separation_venv/bin/python" -m cleancut.separation_worker --check "$separation_models"
print "Separation installed. Enable Preserve background during word mutes in CleanCut Settings."
print "No separate server is needed. Run this installer on the Mac running CleanCut."
