#!/usr/bin/env bash
# Daily backup of familiar_ai PostgreSQL database via docker compose.
# Keeps the last KEEP_DAYS days of backups, then rotates older ones.
#
# Usage:
#   ./scripts/backup_db.sh
#
# Environment overrides:
#   DB_NAME    database name   (default: familiar_ai)
#   BACKUP_DIR backup directory (default: ~/.familiar_ai/backups)
#   KEEP_DAYS  days to retain   (default: 7)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DB_NAME="${DB_NAME:-familiar_ai}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/.familiar_ai/backups}"
KEEP_DAYS="${KEEP_DAYS:-7}"

mkdir -p "$BACKUP_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/${DB_NAME}_${TIMESTAMP}.sql.gz"
LOG_PREFIX="[$(date +%Y-%m-%dT%H:%M:%S)]"

echo "$LOG_PREFIX Starting backup → $BACKUP_FILE"

docker compose -f "$PROJECT_DIR/docker-compose.yml" exec -T db \
    pg_dump -U familiar "$DB_NAME" | gzip > "$BACKUP_FILE"

SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "$LOG_PREFIX Done: $SIZE"

# Rotate: delete files older than KEEP_DAYS days
find "$BACKUP_DIR" -name "${DB_NAME}_*.sql.gz" -mtime +"$KEEP_DAYS" -delete
echo "$LOG_PREFIX Rotated backups older than $KEEP_DAYS days"
