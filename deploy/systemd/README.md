# 日次バックアップの systemd unit

`scripts/backup_db.sh` を毎晩 03:00 に走らせるための unit を2本置く。

## なぜここに置くか

2026年8月に開発機が故障したとき、`scripts/backup_db.sh` はリポジトリにあったので残った。
消えたのは、それを毎晩叩くタイマーのほうだった。新しい機械では、時刻の根拠から
docker グループの扱いまで、ゼロから決め直すことになった。

unit には値だけでなく判断の理由がコメントとして入っている。ファイルを失うと、
値と一緒に理由も失う。だからリポジトリに置く。

## 中身

| ファイル | 役割 |
|---|---|
| `familiar-ai-backup.timer` | いつ走るか。毎晩 03:00。逃したぶんは次の起動時に取り返す |
| `familiar-ai-backup.service` | 何を走らせるか。`scripts/backup_db.sh` を本人の権限で実行する |

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
