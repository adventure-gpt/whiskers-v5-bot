#!/usr/bin/env sh
# Whiskers V5: create the virtual environment, install the dependencies and run the self-check.
set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python was not found. Install Python 3.12 (or set PYTHON to its path) and run this again."
    exit 1
fi

if [ ! -x .venv/bin/python ]; then
    echo "Creating the virtual environment in .venv ..."
    "$PYTHON" -m venv .venv
fi

echo "Installing the dependencies (the CPU build of PyTorch is about 200 MB) ..."
.venv/bin/python -m pip install --upgrade pip >/dev/null
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Self-check:"
.venv/bin/python src/bot.py --selfcheck

echo
echo "Done. In the RLBot launcher, add src/bot.toml and put Whiskers in a match."
