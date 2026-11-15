#!/bin/bash
# AlphaWOTR on a Mac (or Linux): play against the AI.  Run Setup_Mac.command once first.
#   Play_vs_AI.command            asks for the AI's side and net
#   Play_vs_AI.command SA COR     the AI's side (SA or FP), then its net (WIN or MIL for FP, WIN or COR for SA);
#                                 AUTO MIL COR watches AI vs AI
# What it does, step for step: wotr_bridge/play.py
cd "$(dirname "$0")" || exit 1
PY=".venv/bin/python"
if [ -n "$ALPHAWOTR_PYTHON" ]; then
    PY="$ALPHAWOTR_PYTHON"
fi
if [ ! -x "$PY" ]; then
    echo "  [X] No Python environment here yet: run Setup_Mac.command first."
    exit 1
fi
exec "$PY" -B wotr_bridge/play.py "$@"
