#!/usr/bin/env bash
# Gemini をメイン LLM として起動する。
# Save and restore terminal settings around the Qt process.
# PySide6/Qt can leave the terminal in raw mode on exit, making stdin
# unresponsive. The trap restores settings even if the process crashes.
set -e
cd "$(dirname "$0")"
if [ -t 0 ]; then
    _saved_stty=$(stty -g 2>/dev/null || true)
    trap '[ -n "$_saved_stty" ] && stty "$_saved_stty" 2>/dev/null || stty sane 2>/dev/null || true' EXIT
fi
env FAMILIAR_ENV_FILE=.env.gemini uv run familiar --gui "$@"
