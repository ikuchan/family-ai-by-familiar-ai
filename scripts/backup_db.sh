#!/usr/bin/env bash
# Daily backup of familiar_ai PostgreSQL database via docker compose.
# Keeps the last KEEP_DAYS days of local backups, then rotates older ones.
# Uploads to Google Drive via rclone. If GDRIVE_REMOTE names a remote that rclone
# does not have, the script fails after taking the local backup.
#
# Usage:
#   ./scripts/backup_db.sh
#
# Environment overrides:
#   DB_NAME        database name           (default: familiar_ai)
#   BACKUP_DIR     local backup directory  (default: ~/.familiar_ai/backups)
#   KEEP_DAYS      local retention days    (default: 7)
#   GDRIVE_REMOTE  rclone remote path      (default: google drive:familiar_ai_backups)
#                  Set to "" to run without an off-machine copy.
#   GDRIVE_DAYS    cloud retention days    (default: 30)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DB_NAME="${DB_NAME:-familiar_ai}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/.familiar_ai/backups}"
KEEP_DAYS="${KEEP_DAYS:-7}"
# 空文字は「機外へ出さない」の意思表示として扱うため、:- ではなく - を使う。
GDRIVE_REMOTE="${GDRIVE_REMOTE-google drive:familiar_ai_backups}"
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

# Upload to Google Drive.
#
# 機外の控えは、機内が壊れたときに残る唯一のものになる。リモートが見つからないときに
# 警告だけ出して正常終了すると、systemd はユニットを成功と表示し、上がっていないことに
# 誰も気づかない。実際に 2026-08-30 の初回実行がそれで、リモート名の取り違えを
# journal を読むまで見落とした。設定されているのに使えない場合は失敗させる。
#
# ローカルのダンプはこの時点で既に出来ている。失敗させるのは「機外に無い」を報せるためで、
# 機内のぶんを捨てるわけではない。
RCLONE_REMOTE_NAME="${GDRIVE_REMOTE%%:*}"
if [ -z "$GDRIVE_REMOTE" ]; then
    echo "$LOG_PREFIX GDRIVE_REMOTE is empty — running without an off-machine copy"
elif ! command -v rclone &>/dev/null; then
    echo "$LOG_PREFIX ERROR: GDRIVE_REMOTE=${GDRIVE_REMOTE} is set but rclone is not installed." >&2
    echo "$LOG_PREFIX Local backup kept at $BACKUP_FILE, but there is no off-machine copy." >&2
    exit 1
elif ! rclone listremotes 2>/dev/null | grep -q "^${RCLONE_REMOTE_NAME}:"; then
    echo "$LOG_PREFIX ERROR: rclone remote '${RCLONE_REMOTE_NAME}' is not configured (GDRIVE_REMOTE=${GDRIVE_REMOTE})." >&2
    echo "$LOG_PREFIX Local backup kept at $BACKUP_FILE, but there is no off-machine copy." >&2
    exit 1
else
    echo "$LOG_PREFIX Uploading to ${GDRIVE_REMOTE} ..."
    rclone copy "$BACKUP_FILE" "$GDRIVE_REMOTE/" --no-update-modtime
    echo "$LOG_PREFIX Upload complete"

    # Remove cloud files older than GDRIVE_DAYS days
    rclone delete "$GDRIVE_REMOTE/" \
        --min-age "${GDRIVE_DAYS}d" \
        --include "${DB_NAME}_*.sql.gz" 2>/dev/null || true
    echo "$LOG_PREFIX Cloud rotation done (kept last ${GDRIVE_DAYS} days)"
fi
