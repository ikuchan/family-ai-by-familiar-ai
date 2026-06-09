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

---

## 9. 思考モードのランタイム切替

### 概要

`AnthropicBackend.thinking_mode` はインスタンス変数であり、プロセス再起動なしに書き換えられる。
`agent.py` の `run()` 入り口でコマンドを検出し、LLM を呼ばずに即座に切り替える。

### コマンド一覧

| 入力 | 効果 |
|---|---|
| `/think` | `adaptive` ↔ `disabled` をトグル |
| `/think on` | adaptive 思考を有効化 |
| `/think off` | 思考なし（高速）に戻す |
| `/think status` | 現在のモードを表示（変更なし）|
| `深く考えて` / `深く考えてください` | adaptive に切り替え |
| `考えなくていい` / `考えなくていいです` | disabled に切り替え |
| `thinking on` / `enable thinking` | adaptive に切り替え（英語） |
| `thinking off` / `disable thinking` | disabled に切り替え（英語） |

### 自動思考（AI が自律判断）

メッセージ長 > 200 文字、または分析・設計・なぜ・どうして・explain・analyze
などのキーワードを含む場合、そのターンだけ自動的に `adaptive` に切り替わり、
ターン終了後に元のモードへ戻る。

ユーザーが `/think on` で明示的にオンにした場合（`_thinking_user_override = True`）は
自動リセットは行われず、ユーザー設定が優先される。

### セッション間の挙動

- `/think` による変更は **そのセッション限り**
- セッション開始時（`first_turn`）に `.env` の `THINKING_MODE` へ自動リセット
- デフォルトは `THINKING_MODE=disabled`（高速・低コスト）

### 実装箇所

- `src/familiar_agent/agent.py`
  - `_THINK_COMMAND_RE`, `_THINK_ON_EXACT`, `_THINK_OFF_EXACT` — 定数
  - `_COMPLEX_QUERY_RE` — 自動思考トリガーパターン
  - `_handle_thinking_command()` — コマンド処理
  - `_configure_backend_for_turn()` — per-turn 自動思考ロジック
  - `_is_complex_query()` — 複雑度判定

---

## 10. 複数人関係モデルと話者識別

### 概要

`RelationshipTracker` が `state_key` パラメータで複数人の関係状態を同一 SQLite DB
に格納できるように拡張された。新しい `PersonRegistry` クラスが人物ごとの
`RelationshipTracker` インスタンスを管理し、エージェントは常に「アクティブな話者」
のトラッカーを参照する。

### DB スキーマ（変更なし）

`relationship_state` テーブルの `state_key` カラムを人物名として使用する。
プライマリコンパニオンは後方互換のため `state_key = 'default'` を維持。

```
relationship_state
  state_key  TEXT PRIMARY KEY   ← "default" (主コンパニオン) | 人物名
  value_json TEXT               ← RelationshipTracker の状態 (JSON)
  updated_at TEXT
```

### 話者指定の方法

**1ターンだけ話者を指定（メッセージプレフィックス）：**

```
[太郎] こんにちは、元気？
@Yuki: 昨日の宿題終わった？
```

**セッション中ずっとその人として話す（スラッシュコマンド）：**

```
/speaker 太郎          ← 太郎に切り替え
/speaker               ← 現在の話者と既知の人物一覧を表示
```

**セッション終了時の動作：**
- アクティブ話者はセッション開始時（`first_turn`）にデフォルト（`COMPANION_NAME`）へリセット

### システムプロンプトへの注入

毎ターン、variable 部分に以下のコンテキストが追加される：

```
(speaker :name "太郎")
(known-persons "Yuki" "花子")
```

話者に切り替わると、その人物の関係状態（信頼度・親密度・好み・傾向など）が
自動的に `context_for_prompt()` から読み込まれる。

### 実装箇所

- `src/familiar_agent/relationship.py`
  - `RelationshipTracker.__init__` に `state_key: str = "default"` を追加
  - `_load_from_db()`, `_save()` でパラメータ化クエリを使用
  - `PersonRegistry` クラス（新規）
- `src/familiar_agent/agent.py`
  - `_SPEAKER_PREFIX_RE`, `_SPEAKER_COMMAND_RE` — 定数
  - `self._persons: PersonRegistry` — `self._relationship` を置き換え
  - `_relationship` property + setter — 既存呼び出し箇所への透過的エイリアス
  - `_handle_speaker_command()` — `/speaker` コマンド処理
  - `_extract_speaker_prefix()` — `[name]` / `@name:` プレフィックス解析
  - `_system_prompt()` — speaker / known-persons コンテキスト注入

---

## 11. PersonMemoryManager の配線と FAMILY.md 分割

### 概要

`PersonMemoryManager`（UUID ベースの人物識別レイヤー）が `agent.py` に完全に配線され、
`PersonRegistry`（名前ベースの話者管理）と連携して動作するようになった。
また、AIペルソナ記述を `ME.md` とファミリー記述を `FAMILY.md` に分離した。

### PersonMemoryManager の配線

| 変更内容 | 概要 |
|----------|------|
| `self._pmm = PersonMemoryManager(self._memory)` | `__init__` で初期化済み |
| `MemoryTool(self._pmm)` | 旧 `MemoryTool(self._memory)` から修正 |
| `self._presence_watcher` | カメラ利用時に `CameraPresenceWatcher` を起動 |
| `_apply_face_hint(img_path)` | `see` ツール後に顔認識を非同期実行 |

`see` ツールがカメラ画像を返すと、保存パスを抽出して `recognize_face_async()` を
バックグラウンドで実行。認識が閾値（0.75）を超えると `PersonRegistry` の
アクティブ話者が自動更新される。

### ME.md / FAMILY.md 分割

| ファイル | 役割 |
|----------|------|
| `ME.md` | AIのペルソナ（名前・性格・話し方） |
| `FAMILY.md` | 一緒に暮らす人の記述（名前・外見・関係） |
| `ME-template.md` | ME.md のサンプル |
| `FAMILY-template.md` | FAMILY.md のサンプル（新規） |

両ファイルは起動時に1回読み込まれ、安定キャッシュ部分に結合される。
一方が存在しなくても起動可能。どちらも `.gitignore` に含まれる。

### 実装箇所

- `src/familiar_agent/agent.py`
  - `self._family_md: str = self._load_family_md()` — `__init__` に追加
  - `_load_family_md()` — `_load_me_md()` と同パターンで実装
  - `stable_parts = [self._me_md, self._family_md, base]` — 安定プロンプトに追加
  - `_apply_face_hint(img_path)` — 顔認識→話者同期ヘルパー
  - `see` ツール結果処理で `_apply_face_hint` を `ensure_future` 呼び出し
- `FAMILY-template.md` — 新規作成
- `.gitignore` — `FAMILY.md` を追加
