# デプロイ手順

## 前提条件

| ソフトウェア | バージョン |
|---|---|
| Python | 3.10 以上 |
| PostgreSQL | 15 以上 |
| pgvector | 0.7 以上 |
| uv | 最新 |
| ffmpeg | 任意 (カメラ使用時) |

---

## 1. PostgreSQL のセットアップ

### インストール

```bash
# Ubuntu / Debian
sudo apt install -y postgresql postgresql-contrib

# macOS (Homebrew)
brew install postgresql@16
brew services start postgresql@16

# Raspberry Pi (Ubuntu)
sudo apt install -y postgresql
```

### pgvector のインストール

```bash
# Ubuntu / Debian
sudo apt install -y postgresql-server-dev-all build-essential git
git clone --depth 1 https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install

# macOS
brew install pgvector
```

### データベースとユーザーの作成

```bash
sudo -u postgres psql << 'SQL'
CREATE USER familiar WITH PASSWORD 'familiar';
CREATE DATABASE familiar_ai OWNER familiar;
GRANT ALL PRIVILEGES ON DATABASE familiar_ai TO familiar;
\c familiar_ai
CREATE EXTENSION IF NOT EXISTS vector;
SQL
```

接続確認:
```bash
psql -U familiar -d familiar_ai -c "SELECT extversion FROM pg_extension WHERE extname='vector';"
```

---

## 2. リポジトリのセットアップ

```bash
git clone https://github.com/<your-fork>/familiar-ai
cd familiar-ai

# uv でインストール
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                          # 基本依存
uv sync --group camera           # カメラ使用時
uv sync --group voice            # 音声使用時
# uv sync --group recognition    # 顔/声紋認識 (optional, 重い)
```

---

## 3. 環境設定

```bash
cp .env.example .env
```

`.env` を編集:

```env
PLATFORM=anthropic
API_KEY=sk-ant-...

DATABASE_URL=postgresql://familiar:familiar@localhost:5432/familiar_ai

# カメラがある場合
CAMERA_HOST=192.168.1.xxx
CAMERA_USER=your-camera-user
CAMERA_PASS=your-camera-pass

# TTS がある場合
ELEVENLABS_API_KEY=sk_...

# 複数人を使う場合 (起動時に自動登録)
FAMILIAR_PERSONS=alice,bob
```

---

## 4. ペルソナの作成

```bash
cp persona-template/ja.md ME.md
# ME.md を編集してキャラクターを記述
```

---

## 5. 人物の事前登録 (オプション)

顔認識を使う場合:

```bash
# 顔画像を登録
uv run python -c "
from familiar_agent.recognition.face import register_face
register_face('alice', '/path/to/alice.jpg')
register_face('bob',   '/path/to/bob.jpg')
"
```

声紋を登録する場合:

```bash
uv run python -c "
from familiar_agent.tools.memory import ObservationMemory
from familiar_agent.person_memory_manager import PersonMemoryManager
from familiar_agent.recognition.voice import VoiceIdentifier

mem = ObservationMemory()
mgr = PersonMemoryManager(mem)

alice_id = mgr.register_person('alice', 'アリス')
vi = VoiceIdentifier(mgr)
vi.register_voice(alice_id, '/path/to/alice_sample.wav')
"
```

---

## 6. 初回起動とマイグレーション

マイグレーションは起動時に自動実行される。
手動実行する場合:

```bash
uv run python -c "
import os; os.environ.setdefault('DATABASE_URL', 'postgresql://familiar:familiar@localhost:5432/familiar_ai')
from familiar_agent.db import get_db
from familiar_agent.db_migrations import apply_migrations, default_migration_dir
db = get_db()
with db.lock:
    n = apply_migrations(db.conn(), default_migration_dir())
    db.commit()
print(f'{n} migrations applied')
"
```

---

## 7. 起動

```bash
./run.sh              # TUI (推奨)
./run.sh --no-tui     # プレーン REPL
```

---

## 8. Raspberry Pi での運用

```bash
# サービスとして登録
sudo tee /etc/systemd/system/familiar.service << 'EOF'
[Unit]
Description=familiar-ai agent
After=network.target postgresql.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/familiar-ai
EnvironmentFile=/home/pi/familiar-ai/.env
ExecStart=/home/pi/.local/bin/uv run familiar --no-tui
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable familiar
sudo systemctl start familiar
sudo journalctl -u familiar -f   # ログ確認
```

---

## 9. 既存 SQLite データの移行

元の familiar-ai (v0.5) からデータを移行する場合:

```bash
# スクリプトを実行
SQLITE_PATH=~/.familiar_ai/observations.db \
DATABASE_URL=postgresql://familiar:familiar@localhost:5432/familiar_ai \
uv run python scripts/migrate_sqlite_to_pg.py
```

`scripts/migrate_sqlite_to_pg.py` の内容:

```python
import sqlite3, psycopg2, os, struct
from pathlib import Path

TABLES = [
    "observations", "obs_embeddings", "memory_events", "memory_jobs",
    "semantic_facts", "behavior_policies", "memory_revisions",
    "memory_links", "episodes", "episode_memories",
    "memory_activation", "unfinished_business",
    "scene_entities", "scene_events", "exploration_state",
    "relationship_state",
]

src = sqlite3.connect(os.environ["SQLITE_PATH"])
src.row_factory = sqlite3.Row
dst = psycopg2.connect(os.environ["DATABASE_URL"])

for table in TABLES:
    rows = src.execute(f"SELECT * FROM {table}").fetchall()
    if not rows:
        continue
    cols = list(rows[0].keys())
    ph   = ", ".join(["%s"] * len(cols))
    sql  = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({ph}) ON CONFLICT DO NOTHING"
    with dst.cursor() as cur:
        for row in rows:
            vals = tuple(bytes(v) if isinstance(v, memoryview) else v for v in row)
            cur.execute(sql, vals)
    dst.commit()
    print(f"✓ {table}: {len(rows)} rows")

src.close(); dst.close()
print("Migration complete.")
```

---

## 10. 環境変数リファレンス

| 変数 | 説明 | デフォルト |
|---|---|---|
| `DATABASE_URL` | PostgreSQL 接続文字列 | `postgresql://familiar:familiar@localhost:5432/familiar_ai` |
| `PLATFORM` | LLM プラットフォーム | `anthropic` |
| `API_KEY` | LLM API キー | 必須 |
| `AGENT_NAME` | TUI に表示される名前 | `Familiar` |
| `CAMERA_HOST` | RTSP カメラの IP | なし |
| `FAMILIAR_PERSONS` | カンマ区切りの初期登録人物名 | なし |
| `FAMILIAR_PRESENCE_INTERVAL` | カメラポーリング間隔 (秒) | `5` |
| `FAMILIAR_PRESENCE_TIMEOUT` | 退席判定タイムアウト (秒) | `30` |
| `FAMILIAR_EMBEDDING_PREWARM` | 起動時モデル先読み | `1` |
| `FAMILIAR_AI_MIGRATION_DIR` | マイグレーションディレクトリ | 自動検出 |
