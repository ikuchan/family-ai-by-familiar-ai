#!/bin/bash
# ObsidianMemo（Vault）の DB を、このコンテナの初回起動時（pgdata が空のとき）に自動で用意する。
# docker-entrypoint-initdb.d は *.sh / *.sql を直接の階層でしか自動実行しないので、
# 実体の SQL は vault-schema/（サブディレクトリ）に置き、ここから順番に流す。
# 既存の pgdata が残っている再起動では、initdb.d 自体が呼ばれないので実行されない
# （postgres 公式イメージの仕様。作り直すときは `docker compose down` して volume を消してから up）。
set -euo pipefail
DIR="$(dirname "$0")/vault-schema"

echo "vault: 000_create.sql（ロール・vault DB 作成）"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres -f "$DIR/000_create.sql"

for f in "$DIR"/00[1-9]_*.sql; do
  [ "$(basename "$f")" = "999_drop.sql" ] && continue
  echo "vault: $(basename "$f")"
  psql -v ON_ERROR_STOP=1 --username vault_admin --dbname vault -f "$f"
done
echo "vault: スキーマ適用ここまで"
