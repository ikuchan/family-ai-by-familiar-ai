# familiar-ai 復旧記録：失われた20日のスキーマ差分（v0.1）

## この文書が記録すること

2026年8月21日に開発機が故障し、ディスクは回収できなかった。リポジトリの最終コミットは
8月1日 22:36 で、そこから故障までの20日分のコードは残っていない。残ったのは、毎晩の
バックアップが機外へ出していた 8月21日 03:00 時点のデータベースのダンプ1本だけである。

この文書は、そのダンプを新しい開発機へ戻した記録と、**ダンプのスキーマが 8月1日の
コードに対して持っている差**を確定させたものである。差は 16 本のマイグレーションぶんある。
失われた20日に何を作ったかを知る手がかりは、いまのところこの差しかない。

以後の復元作業は、この差を1本ずつ埋める形で進む。

## 何が失われ、何が残ったか

コードは GitHub にも残っていない。`git ls-remote` で確かめたところ、`develop-ikuchan` の
先端は最終コミットと同じで、`develop` と `main-ikuchan`、`dev-postgres-and-mutlitple-persons`
のいずれにも 8月1日より後のコミットは無い。push していなかったぶんが、そのまま消えた。

記憶そのものは無事である。ダンプには観測が 6433 件あり、範囲は 2026年6月8日から
8月20日 17:08 までを覆う。人ごとの記憶は次のように残っている。

| 人 | `situated_memories` の件数 |
|---|---|
| `__self__`（エージェント自身） | 6805 |
| いくながゆうすけ | 1397 |
| いくながたいき | 368 |
| いくながこうき | 69 |
| いくながたえこ | 43 |

失ったのはコードであって、記憶ではない。復旧作業の本体は「データベースを 8月1日へ戻す」
ことではなく、**8月1日のコードを 8月21日のスキーマへ追いつかせる**ことになる。

## ダンプを戻すまでに引っかかったこと

新しい開発機で最初に止まったのは、docker グループが現ログインセッションに効いていない点
だった。ユーザーはグループに登録済みだが、その登録はセッション開始より後に行われている。
このマシンには `sg` も `newgrp` も実体が無く、`sudo` はパスワードを求める。作業していた
tmux サーバもグループ追加より前に起動しており、その子プロセスは全て古い資格を引き継いでいた。

再起動で解消した。同時に `systemd --user` も作り直されるため、日次バックアップを
systemd のユーザータイマーとして登録したときに docker のソケットで弾かれる、という罠も
このとき一緒に消えている。

## リストアと、その検証

ダンプは平文の SQL であり、PostgreSQL 16.14 が出力したものだった。冒頭に `\restrict`
が入るため、流し込む側の psql が古いと弾かれる。ここで使ったイメージは 16.15 で、
サーバもクライアントも新しい側にあたる。

```
psql -v ON_ERROR_STOP=1  終了コード 0
ERROR / WARNING / FATAL   0 件
```

行数の検証は、片方向の確認では足りない。流し込む前にダンプの `COPY` ブロックから 28 表の
行数を数えておき、戻したあとはデータベース側から表の集合ごと取り直して突き合わせた。
この形なら、表が欠けても増えても差として出る。結果は 28 表すべて、行数まで一致した。
索引は 66、外部キーは 23、`hnsw` 索引は 2 で、`vector` 拡張も入っている。

## 失われた16本のマイグレーション

ダンプの `schema_migrations` には 54 件の適用記録が残っていた。リポジトリにあるのは 038
までなので、039 から 054 までの 16 件が失われたぶんにあたる。名前と適用時刻は、その20日間の
作業を順に並べたものとして読める。

| 適用時刻 | 名前 |
|---|---|
| 08-03 05:54 | `039_drop_dead_columns` `040_drop_dedupe_key` `041_drop_unfinished_business` `042_drop_observations_person_id` |
| 08-11 12:41〜13:08 | `043_drop_recall_count` `044_situated_memories` `045_drop_perspective_vec` `046_situated_index_names` `047_situated_roles` |
| 08-11 23:00 | `048_inner_records_belong_to_the_agent` |
| 08-12 14:39 | `049_emotion_vec_cosine_index` |
| 08-14 07:10 | `050_emotion_pad_may_be_unmeasured` |
| 08-15 21:44 | `051_remove_the_per_turn_self_model` `052_fold_the_filler_utterances` `053_drop_the_dangling_rows` `054_retire_the_fillers` |

039 と 041 は、8月1日のコミット（課題8 v0.52 に `importance` と `unfinished_business` 表の
撤去を課題として足したもの）の翌日にあたる。課題として書いた撤去を、そのまま実行している。

## スキーマ差分

差分は手で読むと取りこぼす。実際、ダンプの DDL を目で追った最初の版では、
`observations.groundedness_n` の消滅と、`idx_obs_emotion_vec` の演算子クラスの変更を
落としていた。そのため、テスト用のデータベースへ 8月1日のマイグレーション 38 本を流し、
両者の `pg_dump --schema-only` を機械的に突き合わせた形に置き換えてある。本番側へは
書いていない。

### 表の増減

- 消滅：`situated_embeddings`（5列）、`unfinished_business`（9列）
- 新設：`situated_memories`（8列）、`observations_removed_fillers`（19列）、`observations_removed_self_model`（19列）

`situated_memories` は `situated_embeddings` の改名にあたり、`content` と
`last_recalled_at`、`groundedness_n` の3列が増えている。新設の2表は、消した観測の退避先である。
行が捨てられたわけではなく、自己モデルぶん 1068 件と繋ぎ発話ぶん 337 件が、そこに移されている。

### 列の差

`observations` から6列が消えた。

| 列 | 8月1日の定義 |
|---|---|
| `importance` | `real DEFAULT 1.0 NOT NULL` |
| `person_id` | `text DEFAULT '00000000-0000-0000-0000-000000000001'::text NOT NULL` |
| `scope` | `text DEFAULT 'speaker'::text NOT NULL` |
| `recall_count` | `integer DEFAULT 0 NOT NULL` |
| `last_recalled_at` | `timestamp with time zone` |
| `groundedness_n` | `integer DEFAULT 0 NOT NULL` |

`last_recalled_at` と `groundedness_n` は消えたのではなく `situated_memories` へ移っている。
想起の回数と接地の度合いは、観測そのものではなく、人ごとの記憶に紐づく量になった。

情動の4軸のうち3軸で `NOT NULL` が外れた。`emotion_p` と `emotion_pn`、`emotion_dom` は
`double precision DEFAULT 0.5` になり、`emotion_a` だけが `NOT NULL` のまま残っている。
覚醒は常に測れるが、快と正負、支配は測れないことがある、という区別だと読める。この非対称は
復元のときに再現する必要がある。

このほか、`memory_events.dedupe_key` と `persons.perspective_vec` が消えた。

### 索引と制約

索引は8つ消え、3つ増え、1つ変わった。変わったのは `idx_obs_emotion_vec` で、演算子クラスが
`vector_l2_ops` から `vector_cosine_ops` になっている。情動ベクトルの近さを、長さを含む距離
ではなく向きだけで測るようになった、ということである。

新設は `situated_memories` に対する3つで、うち `idx_situated_recency` は
`(person_id, last_recalled_at)` の複合になっている。消えた8つのうち4つは、消滅した2表に
付いていたものである。

制約は、消滅した2表ぶんの8つが消え、`situated_memories` ぶんの4つが増えた。一意制約は
`(obs_id, person_id, relation_key)` のままで、名前だけが表の改名に追随している。

## 8月1日のコードは、この記憶に触れない

戻したデータベースに対して、いまのコードはそのままでは動かない。無くなった列と表を、
`src/familiar_agent/` が次の件数だけ参照している。

```
importance 13 / scope 35 / recall_count 12 / last_recalled_at 18
dedupe_key 17 / perspective_vec 19 / unfinished_business 14 / situated_embeddings 15
```

観測を書き込む経路が、データベースに無い `scope` 列へ値を入れようとする。8月1日に作った
ばかりの記憶接続 OIF も、`recall_count` と `importance` を触っている。

マイグレーションの自動適用そのものは害を出さない。適用済みの id を `schema_migrations` から
読んで未適用ぶんだけを流す作りなので、38 本はすべて記録済みとして飛ばされる。壊れるのは
実行時の SQL のほうである。

## 検索機能については何も分かっていない

8月4日に検索機能が動いた記録が別にある。しかしスキーマには、その日に対応する変更が無い。
039 から 042 が 8月3日に適用されたあと、次の適用は 8月11日まで空く。

したがって、検索は既存の人ごとの埋め込みと pgvector の上でコードだけを書いて実現したか、
`content` 列を足した 8月11日の 044 で完成したかのどちらかになる。ダンプはこの点について
証言しない。`self_narrative_log` に 8月4日前後の一人称の記録が 621 件のうちいくらか残って
いるので、そこから当たりを付ける余地はある。

## 差分を取り直す手順

テスト用のデータベース（5433）に使い捨てのデータベースを作り、`migration/` の 38 本を
`apply_migrations` で流してから、両者のスキーマだけを出して比べる。本番（5432）へは書かない。

```bash
docker compose --profile test up -d db-test
uv run --no-project --python 3.11 --with psycopg2-binary --with numpy python <スクリプト>
docker compose exec -T db      pg_dump -U familiar --schema-only --no-owner --no-acl familiar_ai
docker compose exec -T db-test pg_dump -U familiar --schema-only --no-owner --no-acl familiar_schema_0801
```

依存は psycopg2 と numpy の2つで足りる。マイグレーションが `familiar_agent` から import
するのは時刻の補助だけで、それは標準ライブラリのみで動くため、プロジェクト全体の依存を
同期する必要はない。

生成物と、ダンプから抜いたスキーマ一式は、リポジトリ外の `~/familiar_ai_restore/` に置いてある。

## 更新履歴

> v0.1：8月21日のダンプを新しい開発機へ戻した記録と、8月1日のコードに対するスキーマ差分を新設。
> 失われた16本のマイグレーションの名前と適用時刻、表と列と索引と制約の差、いまのコードが
> 動かない箇所を確定させた。差分は手読みではなく、38本を流したデータベースとの機械的な突き合わせによる。
