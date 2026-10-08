#!/usr/bin/env bash
# Copy the gitignored files that cannot be regenerated to Google Drive via rclone.
#
# The repository holds the code; this holds what the repository deliberately does not:
# the familiar's personality, the family description, the secrets, and the recovery notes.
# In August 2026 ME.md and FAMILY.md were lost with the old disk because they were
# gitignored and had no off-machine copy anywhere.
#
# Usage:
#   ./scripts/backup_config.sh
#
# Environment overrides:
#   CONFIG_REMOTE  rclone remote path  (default: pajubackup:familiar_ai_config)
#                  Set to "" to run without an off-machine copy.
#   RESTORE_DIR    recovery notes      (default: ~/familiar_ai_restore)
#   MEMORY_DIR     agent memory dir    (default: derived from the project path)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG_REMOTE="${CONFIG_REMOTE-pajubackup:familiar_ai_config}"
RESTORE_DIR="${RESTORE_DIR:-$HOME/familiar_ai_restore}"
# Claude Code は作業ディレクトリの / を - に置き換えた名前で記憶を置く。
# 内部の規約なので変わりうる。見つからなければ下で名前を挙げて報せる。
MEMORY_DIR="${MEMORY_DIR:-$HOME/.claude/projects/$(echo "$PROJECT_DIR" | tr '/' '-')/memory}"

LOG_PREFIX="[$(date +%Y-%m-%dT%H:%M:%S)]"
log()     { echo "$LOG_PREFIX $*"; }
log_err() { echo "$LOG_PREFIX $*" >&2; }

# リポジトリ直下で gitignore されていて、失うと戻らないもの（人が書く設定・2026-10-05 に MUSIC.md を足し、
# 読む仕組みごと撤去した ROUTINES.md を外した）。無ければ「missing」と言う。
REPO_FILES=(.env ME.md FAMILY.md MUSIC.md)
# 作っていなくてもよいもの（あれば送る・無くても黙る）。
OPTIONAL_REPO_FILES=(PEOPLE.md JEV_正解.md)
# `~/.familiar_ai/` の小さなもの（Spotify の鍵・作業メモ）。撮った写真（captures/）は大きいので送らない。
HOME_DIR="${HOME_DIR:-$HOME/.familiar_ai}"
HOME_FILES=(spotify_token.json FAMILY_答え_作業中.md)

# ── 対象を数える ────────────────────────────────────────────────────────────
present_files=()
for f in "${REPO_FILES[@]}"; do
    if [ -f "$PROJECT_DIR/$f" ]; then
        present_files+=("$f")
    else
        # 黙って抜けると、ME.md がある日から無くなっても誰も気づかない。
        log "missing (not backed up): $f"
    fi
done
for f in "${OPTIONAL_REPO_FILES[@]}"; do
    [ -f "$PROJECT_DIR/$f" ] && present_files+=("$f")
done
present_home=()
for f in "${HOME_FILES[@]}"; do
    if [ -f "$HOME_DIR/$f" ]; then
        present_home+=("$f")
    else
        log "missing (not backed up): $HOME_DIR/$f"
    fi
done
[ -d "$RESTORE_DIR" ] || log "missing (not backed up): $RESTORE_DIR"
[ -d "$MEMORY_DIR" ]  || log "missing (not backed up): $MEMORY_DIR"

if [ "${#present_files[@]}" -eq 0 ] && [ "${#present_home[@]}" -eq 0 ] && [ ! -d "$RESTORE_DIR" ] && [ ! -d "$MEMORY_DIR" ]; then
    log "nothing to back up — no target exists"
    exit 0
fi

# ── リモートを確かめる ──────────────────────────────────────────────────────
# 見つからないまま黙って飛ばすと、上げたつもりで上がっていない状態が続く。
# backup_db.sh と同じく、設定されているのに使えない場合は失敗させる。
if [ -z "$CONFIG_REMOTE" ]; then
    log "CONFIG_REMOTE is empty — running without an off-machine copy"
    exit 0
fi
RCLONE_REMOTE_NAME="${CONFIG_REMOTE%%:*}"
if ! command -v rclone &>/dev/null; then
    log_err "ERROR: CONFIG_REMOTE=${CONFIG_REMOTE} is set but rclone is not installed."
    exit 1
fi
if ! rclone listremotes 2>/dev/null | grep -q "^${RCLONE_REMOTE_NAME}:"; then
    log_err "ERROR: rclone remote '${RCLONE_REMOTE_NAME}' is not configured (CONFIG_REMOTE=${CONFIG_REMOTE})."
    exit 1
fi

# ── 転送する ───────────────────────────────────────────────────────────────
# copy であって sync ではない。手元で消したものを向こうでも消すと、
# 誤って消した一晩あとに機外の控えまで失う。
#
# check には --one-way を付ける。copy なので向こうには古いファイルが残り、
# 双方向で比べると「手元に無い」を差分として数えてしまう。
# 確かめたいのは「いま手元にあるものが、向こうに同じ内容で在るか」だけである。
transfer() {
    local label="$1" src="$2" dest="$3"; shift 3
    log "copying ${label} → ${dest}"
    rclone copy "$src" "$dest" "$@"
    rclone check "$src" "$dest" --one-way "$@"
    log "verified ${label}"
}

FILE_LIST="$(mktemp)"
HOME_LIST="$(mktemp)"
trap 'rm -f "$FILE_LIST" "$HOME_LIST"' EXIT
if [ "${#present_files[@]}" -gt 0 ]; then
    printf '%s\n' "${present_files[@]}" > "$FILE_LIST"
    transfer "repo files (${present_files[*]})" "$PROJECT_DIR" "$CONFIG_REMOTE/repo" --files-from "$FILE_LIST"
fi
if [ "${#present_home[@]}" -gt 0 ]; then
    printf '%s\n' "${present_home[@]}" > "$HOME_LIST"
    transfer "home files (${present_home[*]})" "$HOME_DIR" "$CONFIG_REMOTE/home" --files-from "$HOME_LIST"
fi
[ -d "$RESTORE_DIR" ] && transfer "$RESTORE_DIR" "$RESTORE_DIR" "$CONFIG_REMOTE/restore"
[ -d "$MEMORY_DIR" ]  && transfer "$MEMORY_DIR"  "$MEMORY_DIR"  "$CONFIG_REMOTE/memory"

log "config backup complete"
