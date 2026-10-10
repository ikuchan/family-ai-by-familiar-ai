"""落ちたパジュを起動し直す（環-ah・2026-10-10・本人の決定）。

`run-gui.sh` は終了コードで「止めた」と「落ちた」を見分ける。0（画面を閉じた）・130（Ctrl+C）・143（`kill <pid>`）は
止めたので終わる。それ以外（例外の 1・メモリ不足で OS に止められた 137・segfault など）は落ちたので、待ってから起動し
直す。待つ時間は 10 秒から始め、落ちるたびに 1.5 倍（上限なし）。10 分以上動いてから落ちたら 10 秒に戻す。落ちた時刻・
終了コード・次に待つ秒数を `restart.log` に残す。毎回 `PYTHONFAULTHANDLER=1` を付け、パジュのエラー出力を `crash.log`
に追記する（Python の外の部品で落ちたときに全スレッドのスタックが残る）。渡した引数（`--debug` など）は毎回渡す。

試験では起動するコマンド・ログの置き場・秒数を環境変数で差し替え、決めた終了コードを順に返す偽のコマンドで本物の
`run-gui.sh` を動かす。DB は使わない。
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "run-gui.sh"

#: 偽のコマンド。`codes` の先頭行（「終了コード」か「終了コード:動く秒」）を取り、呼ばれた引数と環境を `calls` に残す。
_FAKE = r"""#!/usr/bin/env bash
dir="$(dirname "$0")"
line="$(head -n 1 "$dir/codes")"
tail -n +2 "$dir/codes" > "$dir/codes.next" && mv "$dir/codes.next" "$dir/codes"
echo "args=$*|faulthandler=${PYTHONFAULTHANDLER:-}" >> "$dir/calls"
echo "偽のパジュのエラー出力" >&2
code="${line%%:*}"
case "$line" in *:*) sleep "${line#*:}" ;; esac
exit "$code"
"""


def _run(tmp_path: Path, codes: "list[str]", *args: str, stable: str = "600"):
    fake = tmp_path / "fake.sh"
    fake.write_text(_FAKE, encoding="utf-8")
    fake.chmod(0o755)
    (tmp_path / "codes").write_text("\n".join(codes) + "\n", encoding="utf-8")
    logs = tmp_path / "logs"
    # 守り：`PATH` の先頭に偽の `uv` を置く。差し替え（`FAMILIAR_CMD`）が効かなくても本物のパジュは起動しない
    # （2026-10-10 の RED で、差し替えを読まない run-gui.sh が本物を本番の .env で 3 つ起動し、OOM でパジュが落ちた）。
    guard = tmp_path / "bin"
    guard.mkdir()
    (guard / "uv").write_text(
        f'#!/usr/bin/env bash\necho "本物の uv が呼ばれた $*" >> "{tmp_path}/calls"\nexit 0\n',
        encoding="utf-8",
    )
    (guard / "uv").chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{guard}{os.pathsep}{os.environ.get('PATH', '')}",
        "FAMILIAR_CMD": str(fake),
        "FAMILIAR_LOG_DIR": str(logs),
        "FAMILIAR_RESTART_BASE_SEC": "0.01",
        "FAMILIAR_STABLE_SEC": stable,
    }
    env.pop("PYTHONFAULTHANDLER", None)
    done = subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    calls_file = tmp_path / "calls"
    calls = calls_file.read_text(encoding="utf-8").splitlines() if calls_file.exists() else []
    return done, calls, logs


def _waits(logs: Path) -> "list[float]":
    text = (logs / "restart.log").read_text(encoding="utf-8")
    return [float(x) for x in re.findall(r"([0-9.]+) 秒待って", text)]


@pytest.mark.parametrize("code", ["0", "130", "143"])
def test_a_stop_is_not_restarted(tmp_path, code):
    done, calls, _ = _run(tmp_path, [code, "0"])
    assert len(calls) == 1
    assert done.returncode == int(code)


def test_a_crash_is_restarted_until_it_stops(tmp_path):
    _, calls, logs = _run(tmp_path, ["1", "137", "0"])
    assert len(calls) == 3
    text = (logs / "restart.log").read_text(encoding="utf-8")
    assert "終了コード 1" in text and "終了コード 137" in text


def test_the_wait_grows_by_half_each_crash(tmp_path):
    _run(tmp_path, ["1", "1", "1", "0"])
    w = _waits(tmp_path / "logs")
    assert len(w) == 3
    assert w[0] == pytest.approx(0.01)
    assert w[1] == pytest.approx(0.015)
    assert w[2] == pytest.approx(0.0225)


def test_a_long_run_resets_the_wait(tmp_path):
    _run(tmp_path, ["1", "1", "1:0.4", "1", "0"], stable="0.3")
    w = _waits(tmp_path / "logs")
    assert w == pytest.approx([0.01, 0.015, 0.01, 0.015])


def test_the_arguments_are_passed_on_every_start(tmp_path):
    _, calls, _ = _run(tmp_path, ["1", "0"], "--debug")
    assert len(calls) == 2
    assert all(c.startswith("args=--debug|") for c in calls)


def test_crashes_leave_a_native_trace_in_the_crash_log(tmp_path):
    _, calls, logs = _run(tmp_path, ["137", "0"])
    assert all(c.endswith("faulthandler=1") for c in calls)
    crash = (logs / "crash.log").read_text(encoding="utf-8")
    assert crash.count("偽のパジュのエラー出力") == 2  # 追記（上書きしない）
    assert crash.count("起動した") == 2  # 起動ごとの区切り
    assert "終了コード 137" in crash  # 落ちた行
