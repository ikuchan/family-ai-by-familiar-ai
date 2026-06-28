#!/usr/bin/env bash
# Gemini をメイン LLM として起動する。
set -e
cd "$(dirname "$0")"
exec env FAMILIAR_ENV_FILE=.env.gemini uv run familiar --gui "$@"
