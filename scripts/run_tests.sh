#!/usr/bin/env bash
# Run the full test suite against the test DB.
# Usage: ./scripts/run_tests.sh [pytest args...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── Start test DB ──────────────────────────────────────────────────────────
echo "Starting test DB..."
docker compose --profile test up -d db-test

# Wait until Docker healthcheck reports "healthy" (max 60 s)
echo "Waiting for test DB to be ready..."
CONTAINER="family-ai-by-familiar-ai-db-test-1"
for i in $(seq 1 60); do
    STATUS=$(docker inspect --format '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo "missing")
    if [ "$STATUS" = "healthy" ]; then
        echo "Test DB ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: test DB did not become ready in time (status: $STATUS)." >&2
        docker compose --profile test stop db-test
        exit 1
    fi
    sleep 1
done

# ── Run tests ──────────────────────────────────────────────────────────────
EXIT_CODE=0
uv run pytest -q "$@" || EXIT_CODE=$?

# ── Stop test DB ───────────────────────────────────────────────────────────
echo "Stopping test DB..."
docker compose --profile test stop db-test

exit "$EXIT_CODE"
