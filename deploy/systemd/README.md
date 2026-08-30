# 日次バックアップの systemd unit

`scripts/backup_db.sh` を毎晩 03:00 に走らせるための unit を2本置く。

## なぜここに置くか

2026年8月に開発機が故障したとき、`scripts/backup_db.sh` はリポジトリにあったので残った。
消えたのは、それを毎晩叩くタイマーのほうだった。新しい機械では、時刻の根拠から
docker グループの扱いまで、ゼロから決め直すことになった。

unit には値だけでなく判断の理由がコメントとして入っている。ファイルを失うと、
値と一緒に理由も失う。だからリポジトリに置く。

## 中身

unit は2組ある。守る対象が違い、片方が壊れてももう片方は動く。

| ファイル | 役割 |
|---|---|
| `familiar-ai-backup.timer` | いつ走るか。毎晩 03:00。逃したぶんは次の起動時に取り返す |
| `familiar-ai-backup.service` | 何を走らせるか。`scripts/backup_db.sh`（記憶の入った DB） |
| `familiar-ai-config-backup.timer` | 毎晩 03:10。DB の 03:00 とずらす |
| `familiar-ai-config-backup.service` | `scripts/backup_config.sh`（gitignore された設定と人格） |

## 設置

```bash
sudo cp deploy/systemd/familiar-ai-backup.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now familiar-ai-backup.timer
```

タイマーを待たずに1回試すとき。

```bash
sudo systemctl start familiar-ai-backup.service
journalctl -u familiar-ai-backup.service -n 20 --no-pager
ls -la /data/backups/nightly/
rclone ls pajubackup:familiar_ai_backups     # ここに出て初めて機外にある
systemctl list-timers familiar-ai-backup.timer
```

## 機械ごとに直すところ

unit の値は 2026-08-30 時点の開発機に合わせてある。別の機械へ移すときは次を見直す。

- `User=` と `Group=` と `Environment=HOME=` のユーザー名
- `ExecStart=` と `Documentation=` のリポジトリの絶対パス
- `Environment=BACKUP_DIR=` の保存先
- `Environment=GDRIVE_REMOTE=` の rclone リモート名。`rclone listremotes` で確認する

## 決めたことと、その理由

- **system タイマーにする。** user タイマーは `Linger=no` のままだと、ログインしていない夜に
  走らない。ただし `User=` を書いて本人の権限で実行するので、バックアップの所有者は本人のまま、
  rclone の設定も `~/.config/rclone` をそのまま使える。
- **`SupplementaryGroups=docker` を明示する。** `docker.sock` は `root:docker` の 660 で、
  `User=` を書いても補助グループは付かない。これが無いと docker に触れず失敗する。
- **実行時刻は 03:00。** 旧環境のダンプ名が `familiar_ai_20260821_030002.sql.gz` であることから、
  旧環境もこの時刻だったと分かる。推測ではなく残った証拠に合わせている。
- **`Persistent=true`。** 電源が落ちて 03:00 を逃したぶんは次の起動時に走る。一晩ぶん飛ばすより、
  昼に HDD が数十秒回るほうがましだと判断した。
- **`BACKUP_DIR` は `/data/backups` 直下ではなく `nightly/`。** `backup_db.sh` のローテーションは
  `find "$BACKUP_DIR" -name "${DB_NAME}_*.sql.gz" -mtime +"$KEEP_DAYS" -delete` で、`find` は再帰する。
  直下を渡すと `restore-source/` に置いた保全用のダンプが名前に一致し、7日後に消える。
  読み取り専用にしても、削除の可否は親ディレクトリの権限で決まるので止まらない。
- **ログは journald へ。** `/data` は HDD で、夜間に回り続けると子どもが寝る部屋で音が出る。
  常時追記するものを `/data/logs` へ置かない。

## 失敗の見え方

`scripts/backup_db.sh` は、`GDRIVE_REMOTE` が設定されているのにリモートが見つからないとき、
ローカルのダンプを残したうえで `exit 1` する。systemd はユニットを failed にする。

以前は警告を出して終了コード 0 で終わっていた。そのため 2026-08-30 の初回実行で
リモート名を取り違えていたことに気づけず、機外へ上がっていないまま成功と表示されていた。
機外の控えは、機内が壊れたときに残る唯一のものなので、黙って飛ばさない。

機外へ出さない運用にするときは `GDRIVE_REMOTE=""` を渡して明示する。

## 設定と人格のバックアップ

リポジトリはコードを持つ。こちらは、リポジトリが意図して持たないものを持つ。

```
pajubackup:familiar_ai_config/
  repo/     ← .env, ME.md, FAMILY.md, ROUTINES.md
  restore/  ← ~/familiar_ai_restore/（復旧メモ・スキーマ差分・再現スクリプト）
  memory/   ← ~/.claude/projects/<作業ディレクトリ>/memory/
```

2026年8月に `ME.md` と `FAMILY.md` を失った。gitignore されていて、GitHub にも Drive にも
控えが無かったためである。コードは残ったが、この子の名前と性格と話し方は残らなかった。

### 決めたことと、その理由

- **`copy` であって `sync` ではない。** 手元で消したものを向こうでも消すと、誤って消した
  一晩あとに機外の控えまで失う。古いファイルは向こうに残り続けるが、対象は 180K 程度なので
  容量は問題にならない。
- **`rclone check` に `--one-way` を付ける。** `copy` なので向こうには古いファイルが残る。
  双方向で比べると「手元に無い」を差分として数えてしまう。確かめたいのは
  「いま手元にあるものが、向こうに同じ内容で在るか」だけである。
- **転送後に必ず検証する。** 上げたつもりで上がっていない、を潰す。不一致なら `exit 1`。
- **`.env` は平文で載る。** この Drive には既に家族の記憶が入った DB ダンプが載っているため、
  暗号化リモートを挟んでも実質的な基準は変わらない。構成を増やさない側を選んだ。
- **`rclone.conf` は含めない。** Drive へ入るための鍵を Drive に置いても、復旧時は先に
  `rclone config` で入り直すことになる。実用上の価値が無い。
- **見つからない対象は名前を挙げて log に残す。** 黙って抜けると、`ME.md` がある日から
  無くなっても誰も気づかない。

### 設置

```bash
sudo cp deploy/systemd/familiar-ai-config-backup.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now familiar-ai-config-backup.timer
sudo systemctl start familiar-ai-config-backup.service
journalctl -u familiar-ai-config-backup.service -n 20 --no-pager
rclone ls pajubackup:familiar_ai_config
```

## バックアップの鮮度は本人が見ている

`src/familiar_agent/agent.py` の `_backup_status_note()` は、最後の成功から25時間を超えると
「last database backup was Nh ago — may need attention」と自分で言う。

読む先は次の**固定パス**であり、`BACKUP_DIR` とは無関係である。

```
~/.familiar_ai/backups/backup.log
```

`BACKUP_DIR=/data/backups/nightly` を渡していても、log はこの場所へ書かなければ届かない。
`scripts/backup_db.sh` がここへ追記する。旧環境ではこの追記を systemd タイマー側が
行っていたため、2026年8月にタイマーごと失われ、本人からバックアップが見えなくなっていた。

**log が存在しないとき `_backup_status_note()` は黙る。** バックアップが止まっていても、
log が無ければ何も言わない。この挙動は未修正である。
