#!/bin/bash
# AlphaWOTR on a Mac (or Linux): the one-time setup.
#   1  a Python environment of its own in .venv, with the packages of requirements.txt
#   2  the two search libraries built for this PC (they ship as sources; Windows has them prebuilt)
#   3  the AI server started once, without the game client, to see that it comes up
# Needs: Python 3.10 or newer, and a C++ compiler (on a Mac:  xcode-select --install ).
# Afterwards: Play_vs_AI.command (the game client needs Java 8 or newer).
cd "$(dirname "$0")" || exit 1
PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>/dev/null; then
        PY="$c"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "  [X] Python 3.10 or newer is needed (https://www.python.org/downloads/)."
    exit 1
fi
if [ "$(uname)" = "Darwin" ] && ! xcode-select -p >/dev/null 2>&1; then
    echo "  [X] The C++ compiler is missing.  Run:  xcode-select --install   and then this again."
    exit 1
fi
if ! command -v c++ >/dev/null 2>&1; then
    echo "  [X] A C++ compiler (c++) is needed to build the search libraries."
    exit 1
fi
echo "  $("$PY" --version): making .venv and installing the packages ..."
"$PY" -m venv .venv || exit 1
.venv/bin/python -m pip install --quiet --upgrade pip || exit 1
.venv/bin/python -m pip install --quiet -r requirements.txt || exit 1
for box in native_backend/live_search_oct19 turn_search/native_w2x/box; do
    echo "  Building the search library in $box (a few minutes) ..."
    .venv/bin/python -B "$box/build_portable.py" || exit 1
done
.venv/bin/python -B wotr_bridge/play.py --self-check || exit 1
echo
echo "  Ready.  Start the game with Play_vs_AI.command."
