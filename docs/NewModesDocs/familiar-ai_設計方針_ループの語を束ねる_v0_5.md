# familiar-ai 設計方針：ループの語を束ねる（環-g・v0.5）

## この文書の位置づけ

`loop/event_loop.py` で、**同じものに複数の名前が付き、同じ事実が複数の入れ物に入っている**。
その重複を束ね、語を1つに決めるための設計方針である。

**環-e（口を作り、実体を切り出す）とは別の課題である。** 環-e は「挙動を変えない」を守りと
しているが、本課題は**重複を消す＝どちらかを正にする**ので挙動に触れる。混ぜると、挙動が
変わったときの原因を切り分けられなくなる。

**順序は 環-g が先で、環-e-に（実体を切り出す）が後である。** 束に器を与えるとき、いまの
6つの入れ物のうちどれが正でどれが写しかが決まっていないと、器の中身を決められない。

実装はこの文書の承認後に始める。

## 1. なぜ束ねるか

### 実測（2026-09-08）

`InformationProcessing` は 1クラス・47メソッド・1,674行。`__init__` が置く可変状態は **38 個**で、
そのうち **20 個は書き手が3つ以上**ある。状態は寿命で4つに分かれる（`モジュール分割設計`）。

- イ．装置（10個）／ロ．求めの寿命（18個）／ハ．飛行中の調べもの（9個）／ニ．反復の寿命（1個）

### 拠る原則

複雑さの源は **state と control** であり、最大のものは可変状態である。複雑さは
**essential**（問題そのものに内在）と **accidental**（解き方が持ち込んだ）に分かれる
（Moseley & Marks, *Out of the Tar Pit*, 2006）。

**同じ事実を2つの入れ物に持つことは、essential ではない。** 1件の調べものが「どの動作で」
「何という語で」投げられたかは1つの事実であり、それを3通りに持つのは解き方が持ち込んだ
複雑さである。手で揃えているかぎり、ずれは時間の問題でしかない。

## 2. 束ねる対象（6つ）

### ① 調べもの — 6つの入れ物が、同じ2つの事実を持つ

| いまの入れ物 | 中身 |
|---|---|
| `_in_flight_lookups: list[(動作, 語, 通し番号)]` | **動作・語**・番号 |
| `_lookup_action_by_query: dict[語 → 動作]` | **語 → 動作** |
| `_lookup_index_by_query: dict[語 → 通し番号]` | **語 → 番号** |
| `_lookup_generation: dict[語 → 世代]` | 語 → 世代 |
| `_lookup_results: list[(番号, 動作, 語, 結果)]` | 番号・**動作・語**・結果 |
| `_inflight: int` | 飛行中の数だけ |

`_in_flight_lookups` の1要素が、そのまま `_lookup_action_by_query` と
`_lookup_index_by_query` の中身である。

**`_inflight` と `_in_flight_lookups` は、名前も意味もほぼ同じで、5箇所で別々に動かしている。**

| 場所 | `_inflight` | `_in_flight_lookups` |
|---|---|---|
| `_dispatch_lookup`（同じ語を既に調べた） | `+= 1` | **積まない** |
| `_dispatch_lookup`（普通に投げる） | `+= 1` | `append` |
| `_intake`（完了1件） | `-= 1` | 該当語を `del` |
| `_abort_investigation` | `= 0` | `clear()` |
| `_finish` | **触らない** | `clear()` |

**現に壊れてはいない。** 「既に調べた語」の経路は、数だけ増やしたあと完了を積むので釣り合う。
ただしその釣り合いは**コメントで説明されているだけ**で、機械が守っているわけではない。

#### 案：1件を表す器を作り、列を1本にする

```python
@dataclass
class Lookup:
    index: int              # この求めの中の通し番号（1から）
    action: str             # see / look / recall / search_deferred / fetch_deferred
    query: str              # 探す語
    generation: int         # 投げたときの求めの世代
    result: str | None = None   # 届いた結果（未着は None）
```

`_lookups: list[Lookup]` の1本にする。いまの6つは、すべてここから導ける。

| 旧 | 新しい引き方 |
|---|---|
| `_in_flight_lookups` | `result is None` のもの |
| `_lookup_action_by_query` | `query` で引いて `.action` |
| `_lookup_index_by_query` | `query` で引いて `.index` |
| `_lookup_generation` | `query` で引いて `.generation` |
| `_lookup_results` | `result is not None` のもの |
| `_inflight` | `result is None` の数 |

「既に調べた語」の経路も、**器へ1件積んで即座に結果を入れる**形にすれば、数の釣り合いは
導出で保たれる。手で揃える箇所が5つから0になる。

### ② 「いま生きている記録」を追う仕組みが2本ある

まず語の食い違いがある。同じ「連鎖／鎖」が3つの別物を指す。

| いまの名前 | 実際に指すもの |
|---|---|
| `_chain` | **反復の回数**（コメントは「発話が出るまでの連鎖長」・上限5） |
| `_chain_head_id` | **記録の鎖の先頭**（進む） |
| `_parent_id` のコメント「この連鎖を起こした求め」 | **求めそのもの**（進まない） |

**しかし本体は語ではない。「いま生きている記録」を追う変数が2つある。**

`_write_version` は、版を書くたびに**同じ id を2つの変数へ入れている**。

```python
if version_id:
    if self._version_id and self._version_id != version_id:
        mark_superseded(self._version_id, version_id, kind=KIND_REVISION)
    self._version_id = version_id
    self._chain_head_id = version_id        # ← 同じ id を、もう1つの変数へ
    self._chain_head_content = content
```

畳む仕組みも2本ある。

| | 畳む種類 | 進める先 |
|---|---|---|
| `_advance_chain` | `前進`（`KIND_ADVANCE`） | `_chain_head_id` |
| `_write_version` | `改訂`（`KIND_REVISION`） | `_version_id` **と** `_chain_head_id` |

#### `前進` が実際に発火する条件（実測）

`_advance_chain` の呼び手は**2箇所だけ**で、どちらも新しい求めの始まりである。

| 呼び手 | 場面 |
|---|---|
| `_begin_affect` | 情動で新しい求めを始める |
| `_begin_device` | 人の出入りで新しい求めを始める |

中身は次のとおりで、`if` に入るのは `_chain_head_id` が残っているときだけである。

```python
if self._chain_head_id and self._chain_head_id != new_id:
    mark_superseded(..., kind=KIND_ADVANCE)
self._chain_head_id = new_id
```

`_finish` と `_abort_investigation` は `_chain_head_id` を `None` にする。したがって
**`前進` が書かれるのは「前の求めが閉じないまま、情動または機器で次が始まった」ときだけ**
である。そのとき畳まれるのは、**前の求めの起点と、次の求めの起点**——別々の出来事どうし
である。

**始め方も揃っていない。** 人の発話（`run_iteration`）だけは `_advance_chain` を通らず、
`_chain_head_id` へ直に代入する。同じ場面でも `前進` は書かれない。

#### 決定：`前進` は要らない（2026-09-08・承認済み）

畳む（supersede）とは、**役割 `旧` を付けて「この記録はもう現行ではない」と印すこと**で
ある。記録は消えない——昔のこととしては辿れる。種類は**理由を言うだけ**で、想起の絞りは
種類を見ない。

種類ごとに、畳んでよい条件は違う。

| 種類 | 畳んでよい条件 |
|---|---|
| **改訂** | 同じものの新しい状態が書かれた。古い版は「いまの状態」としては誤りになる |
| **畳み込み** | 中身が別の記録に吸われた。畳まれた側は誤りではない |
| **解決** | 保留していたことが果たされた |
| **前進** | 「記録の鎖が一つ進んだ」 |

**別々の求めの起点どうしは、そのどれにも当たらない。** 同じものの新しい状態でもなく、
中身が吸われたのでもなく、保留が果たされたのでもない。`前進` の本当の目的は
「**想起の枠をループ自身の記録で埋めないため**」（`__init__` のコメント）であって、
「現行でない」を意味していない。**その都合のために畳むという印を使っていた。**

求めの中は `改訂`（版チェーン）が担う。**求めをまたいで畳む理由はない。** 閉じなかった
求めの起点は、閉じなかった記録として残るのが正しい。人の発話の起点が最初から鎖の外に
あって畳まれないのと、同じ扱いになる。

`KIND_ADVANCE` の定数は残す（既存の記録が参照している）。**これから書かれなくなる。**

### ③ `_pending_intent` は、①の器そのもの

- `_open_intent()` は「open 意図を O に残し」と書かれているが、**O へ何も書かない**。
  `_pending_intent` に控えて、タスクを投げるだけである。
- `_write_intent_and_dispatch()` は、投げてから `_write_version()` を呼ぶ。
  **実際に O へ書かれるのは版である。**

「意図」という語が、O に書かれるものの名前としても、控えの名前としても使われている。

**そして中身は①の器そのものである。**

```python
self._pending_intent: tuple[str, dict, str] = ("", {}, "recall")
#                            発話    道具入力  動作
```

`_write_intent_and_dispatch` はここから `query = _query_label(action, tool_input)` を作る。
`action` と `query` は `Lookup` の持ち物であり、別の入れ物に分けている理由がない。
**改名ではなく、①の器へ吸収する。**

### ④ 「(id, 内容)」の組が2つある

`_origin_text`（求めの文面）・`_parent_id`（求めの O の id）・`_chain_head_id`（鎖の先頭・進む）・
`_begin_origin()`（やりとりの起点を控える）。

**中身は違う**（`_chain_head_id` は進むが `_parent_id` は動かない）。しかし**名前からは
違いが読めない**。

**そして同じ形が2組ある。**

| 組 | id | 内容 |
|---|---|---|
| 求め | `_parent_id` | `_origin_text` |
| いま生きている記録 | `_chain_head_id` | `_chain_head_content` |

②で `_chain_head_id` を撤去すれば、後者は `_version_id` ＋ 内容の組になる。
**2組を同じ形で扱えるようにする**（改名だけでは済まない）。

### ⑤ 「どこから」のカーソルが2つ ── 重複ではない（調べた・2026-09-08）

| | `_exchange_from` | `_show_from` |
|---|---|---|
| 型 | `int`（`_wr_ids` の添字） | `str \| None`（記録の id） |
| 指すもの | いまのターンで作った記録が、並びの**どこから始まるか** | **直前に閉じたやりとり**の起点の記録 |
| 寿命 | 求めの中 | **求めをまたぐ**。起動直後は DB から引く |
| 読み手 | `_close_exchange` の区間の切り出し | `_recent_ctx` が `recent_exchanges()` へ渡す |

`_close_exchange` の中で、**片方からもう片方が導かれる**。

```python
members = self._wr_ids[self._exchange_from :]   # 添字で区間を切る
self._exchange_from = len(self._wr_ids)
for obs_id, role in members:
    if role == "起点":
        self._show_from = obs_id                # その区間の起点を id で控える
        break
```

「いま閉じた区間」→「その起点の id」という順で導かれるので、**同じ事実の2通りの持ち方では
ない。改名で足りる。**

#### ただし `_show_seeded` は、真偽値へ潰れた状態である

```python
if not self._show_seeded:
    self._show_seeded = True
    self._show_from = agent._memory.latest_exchange_origin()
```

**一度立つと二度と戻らない。** 起動から一度きり DB を引き、あとは `_close_exchange` が
更新する。`_show_from` が `None` のまま `_show_seeded` が立つと（DB にやりとりが1件も
無い・引くのに失敗した）、次に `_close_exchange` が値を入れるまで直近のやりとりは載らない。

`_close_exchange` は必ず呼ばれるので実害は小さいと見ているが、**「一度きり」を真偽値で
表しているのは、状態機械が真偽値へ潰れた跡**である（Statecharts）。`_show_seeded` を消し、
`_show_from is None` で引き直す形にすれば変数が1つ減る。**ただし DB が空のときに毎ターン
引きに行くようになるので、これは挙動の変更である。**

### ⑥ 「世代」が2つ

`_generation`（求めの世代・打ち切りで進む）と `_lookup_generation`（語→そのときの世代）。
後者は ① の器へ吸収する。

## 3. 改名の対応表（v0.4 で確定）

**改名を先にやる。** 名前が正しくなってから中身を変えるほうが、変える対象を見誤らない。
改名は「旧名で引いて0件」で証明でき、挙動を変えないので、先にやっても危険が増えない。

**消えると決まっているものは改名しない。** 器へ吸収されるもの（下の「対象外」）を改名するのは
無駄である。

### g-い：撤去済みの略語 `WR` を落とす

用語一覧は **WRDB を撤去した**（060）と記している。WR は種類 `共起` の関係として
`relations` に載るようになった。**にもかかわらず `WR` を名前に持つコードが残っている。**

しかも `_wr_ids` の実体は、いまや**「このターンが作った記録と、その役割」**である。拡散想起の
母集合にも使い、やりとりの関係の項にもなる。**WR は2つの用のうち片方の、しかも撤去された
呼び名**である。

| 旧名 | 新名 | 場所 | 呼び手（src／tests） |
|---|---|---|---|
| `_wr_ids` | `_turn_records` | `loop/event_loop.py` | 9／6 |
| `_note_wr` | `_note_record` | `loop/event_loop.py` | 8／0 |
| `combine_wr_ids` | `combine_cooccurring_ids` | `store/relations.py` | 3／3 |
| `_record_wr` | `_record_cooccurrence` | `agent.py` | 2／0 |
| `extra_wr_ids` | `extra_cooccurring_ids` | `agent.py` | 3／2 |

あわせて `agent.py` のコメント2箇所と `loop/coherence.py` の docstring 1箇所を直す。

**完了条件：`grep -rn "wr_ids\|_note_wr\|_record_wr" src/ tests/` が0件。**
`config.py` の `[D-WR拡散想起]` は設計文書の固定 ID なので残す（**理由を明示して除外する**）。

### g-ろ：紛らわしい名前

#### 属性

| 旧名 | 新名 | なぜ |
|---|---|---|
| `_loop` | `_asyncio_loop` | このクラス自体がイベントループなので、`_loop` がどちらを指すか読めない |
| `_inbox` | `_drained_completions` | 一般名詞すぎて、3つのキューのどれとも読める。実体は完了キューから取り出した控え |
| `_tasks` | `_background_tasks` | 何のタスクか読めない。実体は投げっぱなしの背景タスク |
| `_origin_kind` | `_trigger_kind` | `_origin_text` と対に見えるが別物。さらに `origin` はやりとりの役割「起点」と衝突する |
| `_origin_text` | `_request_text` | 実体は**求めの文面**であって、反復の起点ではない |
| `_parent_id` | `_request_id` | 「親」を連想させるが、実体は**求めの O の id** |
| `_chain` | `_iterations` | 「連鎖」が3つの意味を持つ。実体は**反復の回数** |
| `_capped_hit` | `_iterations_capped` | 何が capped か読めない |
| `_generation` | `_request_generation` | 何の世代か読めない。実体は**求めの世代** |
| `_version_id` | `_live_version_id` | 「いま生きている版」であることが名前に無い |
| `_exclude_from_lookup` | `_recall_exclude_id` | 除外するのは `recall` の検索であって lookup 全般ではない |
| `_w_index` | `_w_id_map` | `index` が索引か添字か読めない。実体は 12桁 id → 完全な id の対応表 |
| `_exchange_from` | `_exchange_start` | `from` が値なのか位置なのか読めない。実体は並びの添字 |
| `_show_from` | `_recent_cursor` | 同上。実体は「直近のやりとりをどこから見せるか」の記録 id |
| `_progress_pending` | `_slow_notice_received` | 「進捗が保留」と読めるが、実体は**「まだかかっている」を受けた**という印 |
| `_released_speech` | `_speech_to_deliver` | `released` が済んだことか、これからかが読めない。実体は**これから W へ流す分** |

**変えないもの**：`_agent`・`_dif`・`_driver`・`_on_text`・`_on_action`・3つのキュー・
`_utterance`・`_said_fillers`。どれも名前と実体が合っている。

#### メソッド・module 関数

| 旧名 | 新名 | なぜ |
|---|---|---|
| `run_iteration` | `begin_request` | **公開面。** 名前は「1反復を回す」だが、実体は**求めを始める**（反復は中で複数回りうる）。呼び手は src 3／tests 28 |
| `_begin_origin` | `_note_origin` | `begin` だが、実際は控えるだけ（`_note_record` を1回呼ぶ） |
| `_abort_investigation` | `_abort_lookups` | `investigation` は用語一覧にない。実体は**調べもの**の打ち切り |
| `_action_of` | `_action_of_query` | 何の何かが読めない |
| `_open_intent` | `_start_lookup` | **O へ何も書かない**のに「O に残す」と読める |
| `_write_intent_and_dispatch` | `_dispatch_and_write_version` | 書くのは意図でなく**版**。しかも順序は「投げてから書く」 |
| `_settled`（module） | `_result_or_none` | 何が settled なのか読めない |
| `_when`（module） | `_elapsed_label` | 実体は「経過時間（時刻）」の文字列 |

**変えないもの**：`_iterate`（1反復・用語と合う）・`_intake`・`_compose_workspace`・
`_recent_ctx`・`_link_follows`・`_apply_memory_verdicts`・`_emit`・`set_output`・`start`・
`close`・`push_*` 3つ・`_ensure_driver`・`_drive`・`_begin_affect`・`_begin_device`・
`_release_pending_speech`・`_coherence_violation`・`_speak`・`_say_filler`・
`_delivery_block_reason`・`_accept_silence`・`_present_names`・`_hold_speech`・`_finish`・
`_write_version`・`_write_seen_mark`・`_version_content`・`_open_ids`・`_close_exchange`・
`_tools`・`_dispatch_lookup`・`_watch_slow_lookup`・`_run_lookup`・`_query_label`・
`_log_recall_weights`。

#### 触らないもの

**`_run_camera`・`_camera_tool_def`・`_current_pose_name`。** 知-c が実機で挙動を追っている
最中で、改名は挙動を変えないが、**diff に現れると切り分けの邪魔になる**（環-e-は 段3 と同じ理由）。

#### 対象外（あとの段で消えるので改名しない）

`_inflight`・`_in_flight_lookups`・`_lookup_action_by_query`・`_lookup_index_by_query`・
`_lookup_generation`・`_lookup_results`・`_lookup_seq`・`_pending_intent`（g-は で器へ）／
`_chain_head_id`・`_advance_chain`（g-に で撤去）／`_chain_head_content`（g-ほ で扱う）／
`_show_seeded`（g-へ で消す）／`_next_lookup_index`（g-は で消える見込み）

### 日本語の語（用語一覧 v0.55 に追記済み）

| 語 | 意味 | この意味で使わない語 |
|---|---|---|
| **求め** | 始まりから発話で閉じるまでの1単位 | 「連鎖」 |
| **反復** | 求めの中の1回。1反復＝1出力 | 「連鎖長」 |
| **記録の鎖** | O の supersede の連なり | 「鎖」単独 |
| **版** | 求めの状態を書いた O の記録 | 「意図O」 |
| **調べもの** | 1件の外部呼び出し | 「調査」「検索」の混用 |
| **やりとり** | 1ターンの記録を順序つきで束ねた関係 | — |
| **起点** | やりとりの中で、そのターンを始めた記録の役割 | 求めや鎖の先頭を「起点」と呼ばない |

## 4. 段取り

**改名を先に、挙動の変更をあとに。** 名前が正しくなってから中身を変えるほうが、変える対象を
見誤らない。改名は「旧名で引いて0件」で証明でき、挙動を変えないので、先にやっても危険が
増えない。

| 段 | 中身 | 挙動 |
|---|---|---|
| **g-い** | **撤去済みの略語 `WR` を落とす**（5ファイル・6箇所） | 変わらない |
| **g-ろ** | **紛らわしい名前を直す**（属性16・メソッドと module 関数8） | 変わらない |
| **g-は** | **調べものを1つの器へ**（① ③ ⑥） | **変わる**・**完了**（2026-09-08） |
| **g-に** | **いま生きている記録を1本に**。`_chain_head_id` と `_advance_chain` を撤去。`前進` を書かなくなる。始め方3つを揃える（②） | **変わる** |
| **g-ほ** | **(id, 内容) の組**を同じ形に（④） | 変わりうる |
| **g-へ** | **`_show_seeded` を消す**（⑤の付随） | 変わりうる |

**改名を2つに割る。** `WR` は**撤去済みの略語が残っている**という別の性質の問題で、
`store/relations.py` と `agent.py` にも及ぶ。混ぜると grep の範囲が広がりすぎる。

**1段ずつコミットする。** 改名の段は、旧名で引いて0件を確かめてから閉じる。

## 5. 挙動が変わるところ

### g-は の実装（2026-09-08・完了）

`Lookup`（`index`・`action`・`query`・`generation`・`result`）を `loop/event_loop.py` に置き、
`_lookups: list[Lookup]` の1本にした。**飛行中は `result is None`** で表す。

| 旧 | 新しい引き方 |
|---|---|
| `_in_flight_lookups` | `lk.in_flight` のもの |
| `_lookup_action_by_query` | `_lookup_of(query).action` |
| `_lookup_index_by_query` | `_lookup_of(query).index` |
| `_lookup_generation` | `_lookup_of(query).generation` |
| `_lookup_results` | `result is not None` のもの |
| `_inflight` | `_in_flight_count`（**導出**） |
| `_lookup_seq` | `_next_lookup_index()`＝`len(_lookups) + 1`（**導出**） |
| `_pending_intent` | `_pending_lookup`（名前だけ・器へは入れない。投げる前の控えで、`Lookup` はまだ無い） |

**手で揃える箇所が5つから0になった。**

#### 挙動が変わったところ（2つ）

**① 「既に調べた語」で器を増やさない。** 以前は `_inflight` だけ増やして列に積まず、完了を
積んで取込が減らすことで釣り合わせていた。いまは数が導出なので、その必要がない。器を2つ
作ると、語で引いたときどちらが返るか決まらなくなる。**完了だけを積む**（投げずに黙って
帰ると、完了も時間切れも来ないまま駆動体が待ち続ける）。

**② 通し番号は呼ぶだけでは増えない。** 以前は `_lookup_seq` を持ち、`_next_lookup_index()`
が呼ぶたびに繰り上げていた。いまは `len(_lookups) + 1` なので、**器を1件足したときにだけ**
繰り上がる。

### g-は（挙動の変化・当初の見込み）

同じ結果になるように作るが、作り方が変わる以上、実機で確かめるまで同じだと断定しない。

1. **「既に調べた語」の経路。** いまは `_inflight` だけ増やして列に積まない。器へ統一すると、
   1件積んで即座に結果を入れる形になる。飛行中の数は導出になるので、釣り合いは機械が守る。
2. **`_finish` が列だけ空にして数を触らない点。** 器に統一すれば、空にすれば数も0になる。

### g-に（いま生きている記録を1本に）

**`前進`（`KIND_ADVANCE`）の関係が、これから書かれなくなる。**

いま書かれるのは「求めが閉じないまま、情動または機器で次が始まった」ときだけである。その
場面で、いまは前の求めの起点が次の求めの起点に畳まれる。寄せると**畳まれなくなり、前の
起点は生きたまま残る**。

**これは意図した変更である**（上記②の決定）。閉じなかった求めの起点は、閉じなかった記録
として残る。人の発話の起点が最初から鎖の外にあって畳まれないのと、同じ扱いになる。

**既存の記録は変わらない。** すでに `前進` で畳まれている記録は、そのままである。

**始め方が揃う。** いまは人の発話だけ `_advance_chain` を通らない。揃えることで、3つの
入口が同じ手順で求めを始める。

### g-ほ（(id, 内容) の組）

2組を同じ形にするだけで、値の意味は変えない。**変わりうる**としているのは、いま片方が
`_write_version` の中で暗黙に更新されており、明示にすると更新の時点がずれる可能性がある
ためである。実装で確かめる。

### g-へ（`_show_seeded` を消す）

**DB にやりとりが1件も無いあいだ、毎ターン `latest_exchange_origin()` を引きに行く**
ようになる。いまは起動から一度きりである。

1件でもあれば一度引いて `_show_from` が埋まり、以後は `_close_exchange` が更新するので、
引き直しは起きない。**空の DB は初回起動のときだけ**なので、増える呼び出しは限られる
見込みだが、実測していない。

### g-い・g-ろ（改名）

**挙動を変えない。** 旧名で引いて0件になることが完了条件である。

## 6. 完了条件

- **旧名で引いて0件**（数え上げたリストで代えない。除外するなら理由を1件ずつ明示する）
- `ruff` / `mypy` / 全体テストが緑
- g-い・g-ろ は、**旧名で引いて0件**。除外するものは理由を1件ずつ明示する
- g-は は、**飛行中の数が導出になったことをテストで見る**（手で揃える箇所が0）
- g-に は、**`_advance_chain` と `_chain_head_id` の grep が0件**であること。あわせて、**3つの入口が同じ手順で求めを始める**ことをテストで見る

- 用語一覧（`用語_略語一覧`）へ、上の日本語の語を追記する
- `モジュール分割設計` の 環-e-に の節へ、環-g が先であることを書く

## 7. 環-e との関係

環-g が済んだあと、環-e-に（実体を切り出す）へ戻る。**器の中身は、名前が束ねられてからで
ないと決められない。**

環-e-に の段3（束に器を与える）は、本課題の g-い が「調べもの」の器を作るので、残るのは
「求めの寿命（18個）」の器だけになる。
