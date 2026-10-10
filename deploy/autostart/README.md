# パジュの自動起動

パソコンが起動したら、パジュを自動で立ち上げるための設定を置く（環-ah・2026-10-10）。

## なぜここに置くか

置き場所（`~/.config/autostart/`）はリポジトリの外にある。開発機を作り直したときに失わないよう、写しをここに置く。

## 立ち上がる流れ

1. パソコンが起動すると、Chrome リモートデスクトップのサービス（`chrome-remote-desktop@<ユーザー>`）が仮想の画面
   （`DISPLAY=:20`・Cinnamon）を作る。誰もログインしなくても始まる。
2. Cinnamon の自動起動が `familiar-ai.desktop` を読み、ターミナルで `./run-gui.sh` を開く。
3. `run-gui.sh` がパジュを起動する。落ちたら起動し直す。

`run-gui.sh` が持つこと：

| 終わり方 | 終了コード | どうするか |
|---|---|---|
| 画面を閉じた | 0 | 終わる |
| Ctrl+C | 130 | 終わる |
| `kill <pid>` | 143 | 終わる |
| それ以外（例外・メモリ不足・segfault など） | 1・137 など | 待ってから起動し直す |

- 待つ時間は 10 秒から始め、落ちるたびに 1.5 倍にする（上限なし）。10 分以上動いてから落ちたら 10 秒に戻す。
- 毎回 `PYTHONFAULTHANDLER=1` を付け、パジュのエラー出力を `~/.cache/familiar-ai/crash.log` に追記する。
- 落ちた時刻・終了コード・次に待つ秒数を `~/.cache/familiar-ai/restart.log` に残す。
- `kill -9` で止めると 137 になり、起動し直す。止めるときは `kill <pid>` を使う。

## 設置

```bash
mkdir -p ~/.config/autostart
# リポジトリの場所は印（@REPO@）にしてある。リポジトリ直下で、この機械の場所に置き換えて置く。
sed "s|@REPO@|$PWD|g" deploy/autostart/familiar-ai.desktop > ~/.config/autostart/familiar-ai.desktop
```

外すときは `~/.config/autostart/familiar-ai.desktop` を消す。
