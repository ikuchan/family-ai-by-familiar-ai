#!/usr/bin/env bash
# Daily backup of familiar_ai PostgreSQL database via docker compose.
# Keeps the last KEEP_DAYS days of local backups, then rotates older ones.
# If GDRIVE_REMOTE is set and rclone is configured, uploads to Google Drive.
#
# Usage:
#   ./scripts/backup_db.sh
#
# Environment overrides:
#   DB_NAME        database name          (default: familiar_ai)
#   BACKUP_DIR     local backup directory  (default: ~/.familiar_ai/backups)
#   KEEP_DAYS      local retention days    (default: 7)
#   GDRIVE_REMOTE  rclone remote path      (default: gdrive:familiar_ai_backups)
#   GDRIVE_DAYS    cloud retention days    (default: 30)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DB_NAME="${DB_NAME:-familiar_ai}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/.familiar_ai/backups}"
KEEP_DAYS="${KEEP_DAYS:-7}"
GDRIVE_REMOTE="${GDRIVE_REMOTE:-google drive:familiar_ai_backups}"
GDRIVE_DAYS="${GDRIVE_DAYS:-30}"

mkdir -p "$BACKUP_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/${DB_NAME}_${TIMESTAMP}.sql.gz"
LOG_PREFIX="[$(date +%Y-%m-%dT%H:%M:%S)]"

echo "$LOG_PREFIX Starting backup → $BACKUP_FILE"

docker compose -f "$PROJECT_DIR/docker-compose.yml" exec -T db \
    pg_dump -U familiar "$DB_NAME" | gzip > "$BACKUP_FILE"

SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "$LOG_PREFIX Done: $SIZE"

# Rotate local backups older than KEEP_DAYS
find "$BACKUP_DIR" -name "${DB_NAME}_*.sql.gz" -mtime +"$KEEP_DAYS" -delete
echo "$LOG_PREFIX Rotated local backups older than $KEEP_DAYS days"

# Upload to Google Drive if rclone is configured
RCLONE_REMOTE_NAME="${GDRIVE_REMOTE%%:*}"
if command -v rclone &>/dev/null && rclone listremotes 2>/dev/null | grep -q "^${RCLONE_REMOTE_NAME}:"; then
    echo "$LOG_PREFIX Uploading to ${GDRIVE_REMOTE} ..."
    rclone copy "$BACKUP_FILE" "$GDRIVE_REMOTE/" --no-update-modtime
    echo "$LOG_PREFIX Upload complete"

    # Remove cloud files older than GDRIVE_DAYS days
    rclone delete "$GDRIVE_REMOTE/" \
        --min-age "${GDRIVE_DAYS}d" \
        --include "${DB_NAME}_*.sql.gz" 2>/dev/null || true
    echo "$LOG_PREFIX Cloud rotation done (kept last ${GDRIVE_DAYS} days)"
else
    echo "$LOG_PREFIX rclone remote '${RCLONE_REMOTE_NAME}' not configured — skipping cloud upload"
fi
