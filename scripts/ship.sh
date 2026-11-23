#!/usr/bin/env bash
# 変更を出す：整形 → 検査 → パジュを止める → 全体テスト（緑ならコミット）→ push → パジュを起動し直す（2026-11-23）。
#
#   ./scripts/ship.sh "fix(知-xx): …"
#
# 手でつないでいたときのつまずき（2026-11-23）を、ここで防ぐ：
#   - 整形の前に試験を回し、pre-commit の ruff format がコミットを止めた → 変えた .py を先に整形し、それでも止まったら
#     同じメッセージでもう一度コミットする。
#   - 緑とコミットを確かめずにパジュを起動し、コミットしていないコードで動かした → HEAD が進み、作業ツリーが空のときだけ
#     push と起動に進む。
#   - 起動の環境：デスクトップのいまの環境（DBUS など）を借り、リモートデスクトップの音の設定（PIPEWIRE_REMOTE・PULSE_*）を
#     外す（外さないと声が YAMAHA に出ない）。
# パジュが止まっていても動く（止めるものが無ければ飛ばす）。テストが赤なら、パジュを元どおり起動し直して 1 で終わる。
set -uo pipefail

msg="${1:-}"
if [[ -z "$msg" ]]; then
  echo "使い方: $0 \"コミットメッセージ\"" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
log="$HOME/.cache/familiar-ai/app.log"

paju_pid() { ps -eo pid,cmd | grep "[.]venv/bin/python .*familiar --gui" | awk '{print $1}' | head -1; }

start_paju() {
  local desk
  desk=$(pgrep -u "$USER" -x cinnamon | head -1)
  if [[ -n "$desk" && -r /proc/$desk/environ ]]; then
    for v in DBUS_SESSION_BUS_ADDRESS DISPLAY XAUTHORITY XDG_RUNTIME_DIR XDG_SESSION_TYPE; do
      val=$(tr '\0' '\n' <"/proc/$desk/environ" | grep "^$v=" | cut -d= -f2-)
      [[ -n "$val" ]] && export "$v=$val"
    done
  fi
  unset PIPEWIRE_REMOTE PULSE_SERVER PULSE_SINK PULSE_SOURCE
  setsid nohup ./run-gui.sh </dev/null >/dev/null 2>&1 &
  for _ in $(seq 1 30); do
    [[ -n "$(paju_pid)" ]] && break
    sleep 2
  done
  local want
  want=$(git rev-parse --short=5 HEAD)
  for _ in $(seq 1 30); do
    grep -q "\[v0\.$want" "$log" 2>/dev/null && break
    sleep 2
  done
  if [[ -z "$(paju_pid)" ]]; then
    echo "パジュが起動しなかった" >&2
    return 1
  fi
  local errors
  errors=$(awk -v v="[v0.$want" 'index($0, v){on=1} on && /\[ERROR\]/{n++} END{print n+0}' "$log")
  echo "パジュを起動した（v0.$want・起動してからの ERROR $errors 件）"
}

stop_paju() {
  local pid
  pid=$(paju_pid)
  [[ -z "$pid" ]] && return 0
  kill "$pid"  # kill -9 は使わない（run-gui.sh が「落ちた」と見て起こし直す）
  for _ in $(seq 1 30); do
    [[ -z "$(paju_pid)" ]] && return 0
    sleep 1
  done
  echo "パジュが止まらなかった" >&2
  return 1
}

# 1. 変えた .py だけを整形する（全体に掛けると関係の無いファイルが巻き添えになる）。
mapfile -t changed < <({ git diff --name-only; git diff --cached --name-only; git ls-files --others --exclude-standard; } | grep '\.py$' | sort -u)
existing=()
for f in "${changed[@]}"; do [[ -f "$f" ]] && existing+=("$f"); done
if ((${#existing[@]})); then
  uv run ruff format -q "${existing[@]}"
fi

# 2. 検査。
uv run ruff check . -q || { echo "ruff が赤" >&2; exit 1; }
uv run --group dev mypy src/familiar_agent >/dev/null || { uv run --group dev mypy src/familiar_agent | tail -5; echo "mypy が赤" >&2; exit 1; }

# 3. パジュを止め、全体テスト（緑なら run_tests.sh がコミットする）。
was_running=$([[ -n "$(paju_pid)" ]] && echo 1 || echo 0)
stop_paju || exit 1
before=$(git rev-parse HEAD)
out=$(./scripts/run_tests.sh -m "$msg" 2>&1)
# 全文を残す（末尾だけでは、赤のときにどれが落ちたか分からず、全体を回し直すことになった・2026-11-23）。
printf '%s\n' "$out" >"$HOME/.cache/familiar-ai/last_test.log"
tail -25 <<<"$out"
grep -E "^(FAILED|ERROR) " <<<"$out" | head -20
echo "試験の全文：$HOME/.cache/familiar-ai/last_test.log"

# 4. 緑だったのに pre-commit の整形でコミットが止まったときだけ、同じメッセージでもう一度（赤ならコミットしない）。
if grep -q "All tests passed" <<<"$out" && [[ "$(git rev-parse HEAD)" == "$before" ]]; then
  if [[ -n "$(git status --porcelain)" ]] && git add -A && git commit -q -m "$msg"; then
    echo "pre-commit の整形のあと、もう一度コミットした"
  fi
fi

if [[ "$(git rev-parse HEAD)" == "$before" || -n "$(git status --porcelain)" ]]; then
  echo "コミットされなかった（テストが赤か、作業ツリーが残っている）。push しない" >&2
  [[ "$was_running" == 1 ]] && start_paju
  exit 1
fi

# 5. 外へ退避する。
git push -q && echo "push した：$(git log --oneline -1 | cut -c1-60)"

# 6. パジュを起動し直す（止める前に動いていなくても、いまの版で起こす）。
start_paju
