#!/usr/bin/env bash
# Run the full test suite against the test DB.
#
# Usage:
#   ./scripts/run_tests.sh -m "commit message" [pytest args...]
#   ./scripts/run_tests.sh -f [pytest args...]
#
# A commit message (-m) is REQUIRED. If all tests pass, tracked changes are
# committed and the new commit hash becomes the version shown in the TUI.
# To run the tests WITHOUT committing, you must pass -f/--force explicitly.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── Parse args ─────────────────────────────────────────────────────────────
COMMIT_MSG=""
FORCE=0
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--message)
            COMMIT_MSG="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=1
            shift
            ;;
        *)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Require -m unless -f is given ───────────────────────────────────────────
if [[ -z "$COMMIT_MSG" && "$FORCE" -ne 1 ]]; then
    echo "ERROR: コミットメッセージ (-m \"...\") が必要です。" >&2
    echo "       コミットせずにテストだけ実行する場合は -f を付けてください。" >&2
    echo "Usage: ./scripts/run_tests.sh -m \"commit message\" [pytest args...]" >&2
    echo "       ./scripts/run_tests.sh -f [pytest args...]" >&2
    exit 2
fi

# ── Start test DB ──────────────────────────────────────────────────────────
echo "Starting test DB..."
docker compose --profile test up -d db-test

# Wait until Docker healthcheck reports "healthy" (max 120 s).
# WAL recovery after an unclean shutdown can take 60-90 s on a busy test DB,
# so give it twice the old window before declaring failure.
echo "Waiting for test DB to be ready..."
CONTAINER="family-ai-by-familiar-ai-db-test-1"
for i in $(seq 1 120); do
    STATUS=$(docker inspect --format '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo "missing")
    if [ "$STATUS" = "healthy" ]; then
        echo "Test DB ready."
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: test DB did not become ready in time (status: $STATUS)." >&2
        docker compose --profile test stop --timeout 60 db-test
        exit 1
    fi
    sleep 1
done

# ── Run tests ──────────────────────────────────────────────────────────────
EXIT_CODE=0
uv run pytest -q "${PYTEST_ARGS[@]}" || EXIT_CODE=$?

# ── Stop test DB ───────────────────────────────────────────────────────────
# --timeout 60: give PostgreSQL up to 60 s to flush WAL before Docker sends
# SIGKILL. Without this the default 10 s is often too short after a full test
# run, leaving the data directory dirty and forcing WAL recovery on next start
# — which takes longer than the healthcheck window and marks the container
# unhealthy.
echo "Stopping test DB..."
docker compose --profile test stop --timeout 60 db-test

# ── Commit on success ──────────────────────────────────────────────────────
if [ "$EXIT_CODE" -eq 0 ] && [ -n "$COMMIT_MSG" ]; then
    echo ""
    echo "All tests passed. Committing..."
    git add -u
    if git diff --cached --quiet; then
        echo "Nothing to commit — working tree clean."
    else
        git commit -m "${COMMIT_MSG}

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
        VERSION=$(uv run python -c "from familiar_agent import __version__; print(__version__)" 2>/dev/null || echo "unknown")
        echo "Committed. Version: ${VERSION}"
    fi
elif [ "$EXIT_CODE" -ne 0 ]; then
    echo "" >&2
    echo "Tests failed — no commit made." >&2
fi

exit "$EXIT_CODE"
