#!/bin/zsh
set -euo pipefail

script_dir=${0:A:h}
project_dir=${script_dir:h}

if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
  print -u2 "CleanCut Mac requires an Apple Silicon Mac."
  exit 2
fi

if ! command -v brew >/dev/null 2>&1; then
  print -u2 "Homebrew is required to install Python 3.12 and FFmpeg."
  print -u2 "Install it from https://brew.sh and run this script again."
  exit 2
fi

brew install python@3.12 ffmpeg

python_bin=$(brew --prefix python@3.12)/bin/python3.12
venv_dir=$project_dir/.venv-macos
"$python_bin" -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install --upgrade pip wheel 'setuptools<81'
PIP_CONSTRAINT=$project_dir/macos/pip-constraints.txt \
  "$venv_dir/bin/python" -m pip install -r "$project_dir/requirements-macos.txt"
"$venv_dir/bin/python" -m pip install --no-deps --editable "$project_dir"

mkdir -p "$HOME/Library/Application Support/CleanCut/output"

print
"$venv_dir/bin/python" "$project_dir/macos/doctor.py"
print
print "Installation complete. Double-click macos/Start CleanCut.command"
