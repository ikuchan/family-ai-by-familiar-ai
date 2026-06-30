# 埋め込みモデル移行レポート: multilingual-e5-small → bge-m3

**実施日**: 2026-06-29  
**対象ブランチ**: develop-ikuchan  
**チケット**: familiar-ai_チケット_bge-m3移行_v0_2.md

---

## 背景

課題7の計測で「e5-small は段階的関連を分けきれない」と判明。  
recall 品質改善のため、埋め込みモデルを **BAAI/bge-m3（1024次元）** へ移行した。  
新設計（課題8）を待たず、現行システムへの先行保守作業として実施。

---

## 前提条件（BUG-1）の確認

本移行の着手前提として BUG-1（A+B）の完了を確認した。

| パート | 内容 | コミット |
|---|---|---|
| A | `save_async_with_id` の時間窓冪等化（`_CONTENT_DEDUP_WINDOW_SECS`） | `42e5dca` 重複発声テスト |
| B | 既存重複の purge マイグレーション（019） | 同上 |

---

## 実施したコード変更

### `src/familiar_agent/tools/memory.py`

| 変更箇所 | 変更前 | 変更後 |
|---|---|---|
| `EMBEDDING_MODEL`（:40） | `"intfloat/multilingual-e5-small"` | `"BAAI/bge-m3"` |
| `EMBEDDING_DIM`（:41） | `384` | `1024` |
| `encode_document` のプレフィックス | `f"passage: {t}"` | プレフィックス除去（素のテキスト） |
| `encode_query` のプレフィックス | `f"query: {t}"` | プレフィックス除去（素のテキスト） |
| `_get_perspective_vec_with_conn` | `_decode_vector(...)` をそのまま返す | `_coerce_to_embedding_dim(...)` を通して返す（次元不一致クラッシュ防止） |

bge-m3 は e5 の `passage:` / `query:` プレフィックス規約を持たない。付けたままだと品質が落ちるため除去が必須。

### `migration/2026-06-29-019_purge_utterance_duplicates.py`（バグ修正）

`apply_migrations` ランナーがデフォルトカーソル（タプル行）を使うのに対し、マイグレーション内が `row["ids"]` のような名前アクセスをしていたため適用時にクラッシュしていた。カーソルを内部で `RealDictCursor` に明示して修正。

### `migration/2026-06-29-020_bge_m3_situated_embeddings.py`（新規）

```
HNSW index drop
→ TRUNCATE situated_embeddings
→ TRUNCATE obs_embeddings
→ ALTER TABLE: DROP COLUMN vector / ADD COLUMN vector vector(1024)
→ CREATE INDEX USING hnsw (vector vector_cosine_ops)
```

スキーマ変更のみ。データの再生成は `scripts/reembed_all.py` で行う。

### `scripts/reembed_all.py`（新規）

全 observations を bge-m3 で再エンコードし、`obs_embeddings`（BYTEA）と `situated_embeddings`（vector(1024)）を一括 upsert するバッチスクリプト。

- `--db-url` / `--batch-size` オプション対応
- `ON CONFLICT DO UPDATE` により冪等（中断・再実行可）
- 旧 384 次元の `persons.perspective_vec` を NULL リセット（完了後）

---

## テストの追加・修正

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `tests/test_bge_m3_migration.py` | 新規 | モデル名・次元定数、プレフィックス無し、`_pad_or_truncate`、DB マイグレーション確認 |
| `tests/test_bug1_utterance_dedup.py` | 修正 | fixture の `np.zeros(384)` → `np.zeros(1024)` |
| `tests/test_memory_consolidation.py` | 修正 | `np.ones(384, ...)` → `np.ones(1024, ...)` （2箇所） |

---

## データマイグレーション実績（本番 DB: familiar_ai）

実施コマンド:

```bash
# マイグレーション適用（019 + 020）
uv run python -c "..."   # apply_migrations 経由

# 再埋め込み
uv run python scripts/reembed_all.py
```

### 結果

| 項目 | 値 |
|---|---|
| 適用済みマイグレーション | 019, 020 |
| `situated_embeddings.vector` の次元 | **1024** |
| obs_embeddings 件数 | **2,519**（observations 件数と一致） |
| situated_embeddings 件数 | **10,076**（2,519 × 4 persons） |
| 旧次元 perspective_vec の残存 | **0**（全 persons で NULL リセット済み） |

### スキーマ完了条件チェック

- [x] `grep multilingual-e5-small` → 0件（コード内現役参照なし）
- [x] `grep "EMBEDDING_DIM = 384"` → 0件
- [x] `passage:` / `query:` プレフィックスが bge-m3 経路に残存しない
- [x] `situated_embeddings` の全行が 1024 次元（`atttypmod = 1024`）
- [x] HNSW インデックスが 1024 次元で再作成済み
- [x] obs_embeddings 件数 = observations 件数（孤児ゼロ）
- [x] `persons.perspective_vec` の旧次元ベクトルを NULL リセット

---

## 副次的バグ修正

**`_get_perspective_vec_with_conn` の次元不一致防止**

モデル移行後に旧 384 次元の `perspective_vec` が DB に残っていた場合、`mem_vec (1024) + ALPHA * p_vec (384)` で `ValueError` が発生する。`_coerce_to_embedding_dim` を挟むことで次元を揃えてクラッシュを防ぐよう修正した（次回以降の移行時にも有効な安全網）。

---

## 次のステップ（チケット手順 7・8）

1. **mu 再計算**: bge-m3 ベクトルで person_id 別の平均ベクトルを再算出（計測スクリプト）
2. **c_lo / c_hi 再測定**: 計測指示書 v0.5 の計測1を bge-m3 で実施し、意味分離が改善したか確認
