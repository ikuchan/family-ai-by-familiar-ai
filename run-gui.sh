#!/usr/bin/env bash
# Save and restore terminal settings around the Qt process.
# PySide6/Qt can leave the terminal in raw mode on exit, making stdin
# unresponsive. The trap restores settings even if the process crashes.
#
# 落ちたら起動し直す（環-ah・2026-10-10 本人の決定）。終了コードで「止めた」と「落ちた」を見分ける。
#   0（画面を閉じた）・130（Ctrl+C）・143（kill <pid>）は止めたので終わる。それ以外（例外の 1・メモリ不足で OS に
#   止められた 137・segfault など）は落ちたので、待ってから起動し直す。待つ時間は 10 秒から始め、落ちるたびに 1.5 倍
#   （上限なし）。10 分以上動いてから落ちたら 10 秒に戻す。kill -9 は 137 になり起動し直すので、止めるときは kill <pid>。
# 毎回 PYTHONFAULTHANDLER=1 を付け、パジュのエラー出力を crash.log に追記する（Python の外の部品で落ちたときに全スレッドの
# スタックが残る）。落ちた時刻・終了コード・次に待つ秒数は restart.log に残す。渡した引数（--debug など）は毎回渡す。
# 試験では FAMILIAR_CMD・FAMILIAR_LOG_DIR・FAMILIAR_RESTART_BASE_SEC・FAMILIAR_STABLE_SEC で差し替える（ふだんは渡さない）。
set -u
cd "$(dirname "$0")"
if [ -t 0 ]; then
    _saved_stty=$(stty -g 2>/dev/null || true)
    trap '[ -n "$_saved_stty" ] && stty "$_saved_stty" 2>/dev/null || stty sane 2>/dev/null || true' EXIT
fi

read -r -a cmd <<< "${FAMILIAR_CMD:-uv run familiar --gui}"
log_dir="${FAMILIAR_LOG_DIR:-$HOME/.cache/familiar-ai}"
base="${FAMILIAR_RESTART_BASE_SEC:-10}"
stable="${FAMILIAR_STABLE_SEC:-600}"
mkdir -p "$log_dir"
crash_log="$log_dir/crash.log"
restart_log="$log_dir/restart.log"

now() { date '+%Y-%m-%d %H:%M:%S'; }

next_wait="$base"
while true; do
    echo "===== $(now) 起動した（引数：$*）=====" >> "$crash_log"
    started=$(date +%s.%N)
    code=0
    PYTHONFAULTHANDLER=1 "${cmd[@]}" "$@" 2>> "$crash_log" || code=$?
    case "$code" in
        0 | 130 | 143) exit "$code" ;;
    esac
    elapsed=$(awk -v a="$started" -v b="$(date +%s.%N)" 'BEGIN { print b - a }')
    ran=$(awk -v e="$elapsed" 'BEGIN { printf "%.0f", e }')
    if awk -v e="$elapsed" -v s="$stable" 'BEGIN { exit !(e >= s) }'; then
        wait_sec="$base"
    else
        wait_sec="$next_wait"
    fi
    next_wait=$(awk -v w="$wait_sec" 'BEGIN { printf "%g", w * 1.5 }')
    line="$(now) 落ちた（終了コード ${code}・動いた ${ran} 秒）。${wait_sec} 秒待って起動し直す"
    echo "$line" | tee -a "$restart_log"
    echo "===== $line =====" >> "$crash_log"
    sleep "$wait_sec"
done
