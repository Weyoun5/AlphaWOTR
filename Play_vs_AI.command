#!/bin/bash
# AlphaWOTR on a Mac: play against the AI.  The Mac download (AlphaWOTR-mac.zip) carries its own Python, its packages
# and the two search libraries; the game client needs Java 8 or newer.
#   Play_vs_AI.command            asks for the AI's side and net
#   Play_vs_AI.command SA COR     the AI's side (SA or FP), then its net (WIN or MIL for FP, WIN or COR for SA);
#                                 AUTO MIL COR watches AI vs AI
# What it does, step for step: wotr_bridge/play.py
cd "$(dirname "$0")" || exit 1
# macOS marks every file of a download from the internet and then refuses to run the programs among them.  Opening
# this file was your word that you trust the download: the mark is taken off this folder, and off nothing else.
xattr -dr com.apple.quarantine . 2>/dev/null
PY="python/bin/python3"
if [ -n "$ALPHAWOTR_PYTHON" ]; then
    PY="$ALPHAWOTR_PYTHON"
fi
if [ ! -x "$PY" ]; then
    echo "  [X] No Python here ($PY).  The Mac download carries one: AlphaWOTR-mac.zip"
    echo "      (your own Python instead:  export ALPHAWOTR_PYTHON=/path/to/python3 )"
    exit 1
fi
exec "$PY" -B wotr_bridge/play.py "$@"
