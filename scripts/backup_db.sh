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
#   BACKUP_LOG     freshness log path
#                  (default: ~/.familiar_ai/backups/backup.log)

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

# エージェント自身がバックアップの鮮度を見ている（agent.py の _backup_status_note）。
# 最後の "Done:" から25時間を超えると、自分で「backup was Nh ago」と言う。
#
# 読む先は BACKUP_DIR ではなく、この固定パスである。$BACKUP_DIR/backup.log へ書いても
# 届かない。旧環境ではこの追記を systemd タイマー側が行っていたため、2026年8月に
# タイマーごと失われた。スクリプト側で書けば、手で叩いたときも記録が残る。
#
# なお log が存在しないとき _backup_status_note() は黙る。書かなければ、
# バックアップが止まっていても誰も気づかない。
BACKUP_LOG="${BACKUP_LOG:-$HOME/.familiar_ai/backups/backup.log}"
mkdir -p "$(dirname "$BACKUP_LOG")"

log() {
    echo "$LOG_PREFIX $*"
    echo "$LOG_PREFIX $*" >> "$BACKUP_LOG"
}

log_err() {
    echo "$LOG_PREFIX $*" >&2
    echo "$LOG_PREFIX $*" >> "$BACKUP_LOG"
}

log "Starting backup → $BACKUP_FILE"

docker compose -f "$PROJECT_DIR/docker-compose.yml" exec -T db \
    pg_dump -U familiar "$DB_NAME" | gzip > "$BACKUP_FILE"

SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
log "Done: $SIZE"

# Rotate local backups older than KEEP_DAYS
find "$BACKUP_DIR" -name "${DB_NAME}_*.sql.gz" -mtime +"$KEEP_DAYS" -delete
log "Rotated local backups older than $KEEP_DAYS days"

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
    log "GDRIVE_REMOTE is empty — running without an off-machine copy"
elif ! command -v rclone &>/dev/null; then
    log_err "ERROR: GDRIVE_REMOTE=${GDRIVE_REMOTE} is set but rclone is not installed."
    log_err "Local backup kept at $BACKUP_FILE, but there is no off-machine copy."
    exit 1
elif ! rclone listremotes 2>/dev/null | grep -q "^${RCLONE_REMOTE_NAME}:"; then
    log_err "ERROR: rclone remote '${RCLONE_REMOTE_NAME}' is not configured (GDRIVE_REMOTE=${GDRIVE_REMOTE})."
    log_err "Local backup kept at $BACKUP_FILE, but there is no off-machine copy."
    exit 1
else
    log "Uploading to ${GDRIVE_REMOTE} ..."
    rclone copy "$BACKUP_FILE" "$GDRIVE_REMOTE/" --no-update-modtime
    log "Upload complete"

    # Remove cloud files older than GDRIVE_DAYS days
    rclone delete "$GDRIVE_REMOTE/" \
        --min-age "${GDRIVE_DAYS}d" \
        --include "${DB_NAME}_*.sql.gz" 2>/dev/null || true
    log "Cloud rotation done (kept last ${GDRIVE_DAYS} days)"
fi
