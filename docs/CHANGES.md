# 変更点ドキュメント (v0.5 → v0.6)

本ドキュメントは `lifemate-ai/familiar-ai` のフォークにおける
設計上の変更点を説明する。

---

## 1. データベース: SQLite → PostgreSQL + pgvector

### 背景
元実装は `~/.familiar_ai/observations.db` に SQLite を使用していた。
複数人対応・並行書き込みの安全性・ベクトル検索のスケーラビリティの観点から
PostgreSQL へ移行し、`pgvector` 拡張でベクトル型の SQL 検索を有効化した。

### 主な変更
| 変更前 | 変更後 |
|---|---|
| `sqlite3` + `BLOB` | `psycopg2` + `BYTEA` / `vector(384)` |
| `?` プレースホルダ | `%s` プレースホルダ |
| `INSERT OR IGNORE` | `INSERT ... ON CONFLICT DO NOTHING` |
| `PRAGMA journal_mode=WAL` | 不要（PostgreSQL MVCC） |
| `sqlite3.Row` | `psycopg2.extras.RealDictCursor` |
| `strftime('%m-%d', date)` | `TO_CHAR(date::DATE, 'MM-DD')` |

### 新規ファイル
- `src/familiar_agent/db.py` — シングルトン接続管理
- `src/familiar_agent/db_migrations.py` — PostgreSQL マイグレーションランナー

---

## 2. マルチパーソン記憶 (設計α)

### 概念的な変更
「人物ごとの記憶空間」ではなく「AIが管理する人物別コンテキストストア」
として設計。他者の記憶に直接アクセスするのではなく、
**視点ベクトル** (perspective vector) を通じた推定として実装する。

### 予約 person_id
| ID | 用途 |
|---|---|
| `00000000-0000-0000-0000-000000000000` | `AGENT_SELF_ID` — エージェント自身 |
| `00000000-0000-0000-0000-000000000001` | `DEFAULT_PERSON_ID` — 旧来のデフォルト（後方互換） |

### 新規テーブル
```sql
persons              -- 人物レジストリ + perspective_vec (BYTEA)
situated_embeddings  -- 人物視点済みベクトル (vector(384), HNSWインデックス)
```

### 既存テーブルへの追加カラム
`observations`, `memory_events`, `semantic_facts`, `behavior_policies`,
`memory_revisions`, `episodes`, `unfinished_business`, `relationship_state`
に `person_id TEXT NOT NULL DEFAULT '<DEFAULT_PERSON_ID>'` を追加。

### `observations` への追加カラム
```sql
writer_id         TEXT    -- このメモリを「持っている」人
subject_id        TEXT    -- このメモリが「について」の人
participants_json TEXT    -- その場にいた全員 (JSON配列)
scope             TEXT    -- "speaker"|"witnessed"|"scene"|"all"
```

---

## 3. 視点ベクトル (Perspective Vector)

### アーキテクチャ
- 各人物は `persons.perspective_vec (BYTEA)` に 384次元ベクトルを保持
- メモリ書き込み時に `situated_embeddings` テーブルへ全登録人物分の
  視点済みベクトルを先行計算して格納する
- 計算式: `situated = normalise(mem_vec + 0.3 * person_vec)`
- 人物ベクトルは `ObservationMemory._update_perspective_vec()` で
  移動平均 (lr=0.05) により徐々に更新される

### 検索
```sql
SELECT ..., 1 - (s.vector <=> %s::vector) AS score
FROM situated_embeddings s
JOIN observations o ON o.id = s.obs_id
WHERE s.person_id = %s AND o.superseded_by IS NULL
ORDER BY s.vector <=> %s::vector
LIMIT %s
```
Python 側でのフルスキャンがなくなり、HNSW インデックスによる
近似近傍探索 (O(log N)) が利用できる。

---

## 4. PersonMemoryManager

新規ファイル `src/familiar_agent/person_memory_manager.py`。

```
present_persons : dict[person_id, PersonPresence]
    その場にいる全員

current_speaker_id : str | None
    今話している人
```

### メモリのルーティング
| 操作 | ルーティング先 |
|---|---|
| `remember(scope="speaker")` | current_speaker のメモリ |
| `remember(scope="witnessed")` | その場にいる全員のメモリ |
| `remember(scope="scene")` | AGENT_SELF のメモリ |
| `remember(scope="all")` | 上記すべて |
| `recall()` | present 全員 + AGENT_SELF を横断検索 |
| `recall_self_model()` | 常に AGENT_SELF_ID のみ |
| `recall_curiosities()` | 常に AGENT_SELF_ID のみ |

---

## 5. PersonTool (新規)

エージェントが人物を宣言・管理するための LLM ツール群。
`identify_person` を以下の3ツールに分割した。

| ツール名 | 用途 |
|---|---|
| `declare_speaker` | 今話しているのが誰かを宣言 |
| `note_person_arrived` | 誰かが来たことを記録 |
| `note_person_left` | 誰かが去ったことを記録 |
| `who_is_present` | 現在の存在状況を返す |
| `ask_who_is_speaking` | 名前を尋ねる文を生成 |

---

## 6. 存在検知パイプライン (新規)

`src/familiar_agent/recognition/` 以下に3モジュールを追加。

| モジュール | 機能 | 必要な追加依存 |
|---|---|---|
| `face.py` | deepface による顔照合 | `deepface` (optional) |
| `voice.py` | resemblyzer による声紋照合 | `resemblyzer` (optional) |
| `presence_watcher.py` | カメラバックグラウンドポーリング | なし |

いずれもオプション依存であり、未インストールの場合はフォールバックする。

---

## 7. 削除・廃止

| 廃止されたもの | 理由 |
|---|---|
| `src/familiar_agent/sqlite_migrations.py` | `db_migrations.py` に置換 |
| `tools/memory.py` の `active_person_id` | `PersonMemoryManager` の `current_speaker_id` に置換 |
| `identify_person` ツール | `declare_speaker` + `note_person_arrived` に分割 |

---

## 8. マイグレーション一覧

| ファイル | 内容 |
|---|---|
| `001`〜`005` | 既存テーブル (PostgreSQL 構文に変換) |
| `006` | no-op (001 にマージ済み) |
| `007`〜`009` | 既存テーブル (PostgreSQL 構文に変換) |
| `010` | `persons` テーブル + 全テーブルへ `person_id` 追加 |
| `011` | `situated_embeddings` + `pgvector` 拡張 + HNSW インデックス |
| `012` | `observations` へ `writer_id`, `subject_id`, `participants_json`, `scope` 追加 |
