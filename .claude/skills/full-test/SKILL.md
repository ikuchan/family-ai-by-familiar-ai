---
name: full-test
description: 全体テストを回してコミットし、外の Git へ退避し、止めた本体を元に戻す手順。src/・tests/・migration/ を変えて全体テスト（./scripts/run_tests.sh）を回す前、docs だけをコミットする前、マイグレーションを置く前、パジュを立ち上げ直すときに使う。
---

# 全体テストとコミットと退避

CLAUDE.md の完了条件の、具体的なやり方。順に行う。

## 1. 差分を見直す

`src/`・`tests/`・`migration/` を変えたら、全体テストの前に `/code-review` で差分を見直す。
docs だけの変更は見直さない。

- ふだんは medium。マイグレーション・DB への書き込み・声の道を変えたときは high。
- 指摘は、直すか、直さない理由を書いて閉じる。
- 3 回まで。2 回目からは、前回から変わった差分だけを見る。
- 同じ指摘が 2 回出たら、CLAUDE.md・スキル・lint のどれかへ移す。

## 2. 本体を止める

本体（パジュ）とテストは、メモリ・GPU・DB を取り合う。動いたまま回すと、メモリが足りずにテストの
worker が死に（`worker crashed`）、テストが揺れる。

1. `free -h` で空きを見る。
2. 本体を探す：`ps -eo pid,cmd | grep -E "[.]venv/bin/python3 .*familiar --gui"`
3. 見つかった pid を `kill <pid>` で止める。`kill -9` は使わない（`run-gui.sh` は終了コード 137 を
   「落ちた」と見て起動し直す。ふつうの `kill` の 143 なら止まったままになる）。
4. 2 の結果が 0 件になるまで待つ。`sbv2_server` と、ほかの `psql` は残ってよい。

テストが揺れたら、テストの中身より先に資源を疑う。まず `free -h` を見る。

## 3. 検査してから回す

1. `ruff format` は、変えたファイルにだけかける（全体にかけると、関係のない 150〜240 ファイルが変わる）。
2. `uv run ruff check .`
3. `uv run --group dev mypy src/familiar_agent`
4. 全体テストはバックグラウンドで回す：
   `./scripts/run_tests.sh -m "<開発内容>" 2>&1 | tail -25`
   - 緑なら、スクリプトが `git add -A` してコミットする。回っているあいだは、ほかのファイルを編集しない
     （コミットに巻き込まれる）。
   - 出力を `tail -3` や `grep` で絞らない。失敗の中身も、コミットされたかも見えなくなる。

docs の md だけを変えたときは、本体を止めず、全体テストも回さない。
`uv run pytest -q -p no:cacheprovider tests/test_docs_are_consistent.py` を回してから、
`git add` と `git commit` で直接コミットする。

## 4. 外の Git へ退避する

コミットしたら、すぐ `git push` する。この機械の外にある控えは push 先だけである
（2026-08 に開発機が壊れ、push していなかった 20 日分のコードを失った）。

## 5. 本体を元に戻す

1. デスクトップの画面につなぐ環境変数（`DBUS_SESSION_BUS_ADDRESS`・`DISPLAY`・`XAUTHORITY`・
   `XDG_RUNTIME_DIR`・`XDG_SESSION_TYPE`）を、デスクトップの側から受け取ってから起動する。
2. リポジトリの直下で起動する：`setsid nohup ./run-gui.sh </dev/null >/dev/null 2>&1 &`
   - `--debug` は付けない（ログが 1 日 1 GB になる）。
3. 確かめる：
   - 2 の `ps` の結果が 1 件であること。
   - `grep -c " | ERROR | " ~/.cache/familiar-ai/app.log` が 0 であること。
   - 起動した版が、コミットした版と同じであること。

いまのログは `~/.cache/familiar-ai/app.log` にある。`logs/` の下は、起動のときに退避された過去のぶんである。

## マイグレーションを置くとき

本体は DB へ接続するたびに、`migration/` の未適用のぶんを本番 DB へ当てる（`db.py` の
`apply_or_hold_migrations`）。ファイルを置いた時点で、動いている本体が次に接続したときに当たる。置く前に
本体を止める。止められないときは、`HOLD_MIGRATIONS=1` を渡して起動し直しておく。
