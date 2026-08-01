# familiar-ai モジュール分割設計（v0.7）

## この文書が決めること

`agent.py`・`tools/memory.py`・`loop/event_loop.py` を、どの単位でファイルへ切り出すかを決める。行数を減らすことが目的ではない。**コードを開いた人が設計図と突き合わせて読める状態**にすることが目的で、行数はその結果として下がる。

置き場所と順序は課題8 が決めている。本書はその中身にあたる。

## 分割の基準

**基準1：設計が定めたコンポーネントに名前を合わせる。**

このシステムには自前の分割がすでにある（[D-I内部]／[D-周期]／[D-B分離]／[D-記憶単一化]）。ファイル構成をその名前へ合わせれば、設計図とコードが一対一で照らせる。逆に、メソッド名の似たものを寄せる分け方は、設計に無い区切りを新しく作ることになり、読む人が覚える対応表が増える。

**基準2：変わりそうな判断を隠す単位で切る。**

Parnas が「On the Criteria To Be Used in Decomposing Systems into Modules」(1972) で示した基準で、処理の流れの順に切るのではなく、**変更が起きたときに1つのモジュールに閉じる**ように切る。本プロジェクトには Phase 3 から Phase 6 という変更予定表があるため、この基準を具体的に適用できる。撤去が決まっているものを1箇所へ隔離しておけば、Phase 6 の撤去は「そのディレクトリを消す」に近づく。

**基準3：依存は内側へ向ける。**

Ports and Adapters（Cockburn）と Clean Architecture の依存規則にあたる。中核（O／MI／W／T）は外部機器や LLM の実装を知らない。設計の [D-内外境界]（外部は物理世界だけで、LLM と MCP は内部資源）と同じ線引きである。

## 設計が定めているコンポーネント

```text
T（周期駆動）  レジスタ＝drive／mood／norm／presence、TIF、G（知覚の連続側）
   │ AIF（T 接続専用：情動を受け、Nudge を送る）
I（イベント駆動）
   ├ メイン＝ループ管理（3キューを drain し、順序づけ、未解決を確認して送出）
   │    └ 内包＝調停器（Arbiter）、想起（Recall・取り込み時に O から W を作る）
   ├ 生成器      指示構築とフルLLM と解釈
   ├ 評価器      驚き、感情、予測、値踏み、意味づけ、要約（軽量LLM）
   ├ 動作器      動作要求を DIF と実行担当へ渡す
   └ 統合保守器  REST 内省（near-dup 統合、supersede、活性減衰）
   │ DIF（機器IF：カメラ、スピーカー、マイク、音楽）
   │ 資源ハンドラ（LLM担当と実行担当＝MCP、検索、ツール）
O（記憶ストア・追記と supersede）   W は O からの派生ビュー
C（完了キュー）   SS（自己状態）   Config（全調整可能定数）
```

## 現状のずれ

大きいファイルがあるのは行数の問題ではなく、**設計上5つに分かれている実体が1ファイルに同居している**ためである。

旧 `run()` は 環-c で撤去され、いまその位置には `loop/event_loop.py`（1,444 行）がある。同居の形は移ったが、解けてはいない。

| 設計上の実体 | いまの居場所 | 行数 | 主なもの |
|---|---|---|---|
| 生成器 | `event_loop.py` | 350 | `_iterate`（205 行）・`_compose_workspace`・`_apply_memory_verdicts`・`_finish` |
| ループ管理 | `event_loop.py` | 325 | `_drive`・`_intake`・`run_iteration`・`_begin_affect`・`_begin_device`・`push_*` |
| 動作器／資源ハンドラ | `event_loop.py` | 200 | `_dispatch_lookup`・`_run_lookup`・`_run_camera`・`_tools` |
| 発話ゲート | `event_loop.py` | 190 | `_speak`・`_delivery_block_reason`・`_hold_speech`・`_accept_silence` |
| O 書込 | `event_loop.py` | 136 | `_write_version`・`_write_seen_mark`・`_open_intent` |
| 評価器 | `loop/evaluator.py`・`loop/arbiter.py` | 566 | **切り出し済み** |
| O のアクセス層 | `store/` | — | **切り出し済み** |
| W 構築の採点 | `tools/memory.py` の `_score_breakdown` 周辺 | — | `core/recall_score.py` へは未抽出 |

**資源は I が直接呼んでいる。** 設計は LLM 担当（LLH）・実行担当（TEX）・動作器（ACT）を別の実体として定めるが、実装ではどれも存在せず、`event_loop.py` が身体と外部サービスを直接叩く。

| 資源 | 呼んでいる箇所 |
|---|---|
| カメラ | `event_loop.py:472` `agent._camera.call(...)` |
| VLM（意味づけ） | 同 `:485` `extract_entities(..., agent._scene_backend, ...)` |
| 検索・取得 | 同 `:422` `agent._deferred_search` ／ `_deferred_fetch` |
| 記憶 | 同 `:445` `agent._memory_tool.call("recall", ...)` |
| 軽量LLM | 同 `:1091` `agent._utility_backend` |
| TTS | 同 `:1245`・`:1261` `agent._tts.call("say", ...)` |

同居が実際に不具合を隠した例がある。2026-07-20 の「`say()` で話したターンが永続化されない」は、永続化の判定が `run()` の約 590 行目に埋もれていたために3週間気づかれなかった。設計上は「O 書込の呼び出し口」という独立した関心であり、同居させる理由が無い。

## 目標のファイル構成

```text
core/     設計の中核。外部 I/O を知らない
  mental_item.py    MI と PI の器、行から MI への変換
  recall_score.py   W 構築の5軸スコアラ
  mood_register.py  drive_register.py  tif.py   （既存・T レジスタ）
  config.py         [D-設定]
store/    O。SQL はここにだけ置く
  observations.py   追記、supersede、取り出し（by_kind／by_situated／by_date／by_vector）
  situated.py       視点と在席相関のベクトル
  embedding.py      埋め込みモデルとベクトル符号化
  jobs.py           memory_events と memory_jobs
  db_compat.py      接続ラッパ
  clock.py          時刻の一元化
loop/     I
  loop_manager.py   3キューの drain（Phase 5 で本格化）
  recall.py         想起（O から W へ）
  generator.py      生成器
  evaluator.py      評価器
  actor.py          動作器
  maintenance.py    統合保守器（REST）
  persistence.py    ターン永続化（O 書込の呼び出し口）
  silence_gate.py   発話ゲート（在席、静穏時間、宛先選択）
io/       出入口
  aif.py            T 接続
  dif/              camera.py  tts.py  stt.py  music.py
  resources/        llm.py  mcp.py  search.py
legacy/   Phase 6 で撤去予定を隔離
  semantic_facts.py  behavior_policies.py  memory_links.py  workspace.py  tape.py ほか
```

## 第一弾の範囲

全体を一度に動かさない。**いま切ると効き、かつ Phase 5 で捨てない**ものに絞る。

| 順 | 切り出し | 根拠 |
|---|---|---|
| 0 | 生存確認の不変条件 | 分割の安全網。壊れても分からない状態で大工事をしない |
| 1 | `store/`（O とアクセス層、`clock.py` を含む） | 課題8 v0.6 が Phase 2 の宿題として指定済み。Phase 3 が書き込み経路に触る。時刻の一元化で 2026-07-20 の9時間ずれの真因を封じる |
| 2 | `core/recall_score.py`（未作成） | 5軸合成は現在 `tools/memory.py` 内の `_score_breakdown`（正本）にあり `min_score` 是正も実装済み。`core/` への抽出は内部R（D 後）へ |
| 3 | `loop/evaluator.py` | 設計に名前のある実体が `agent.py` に散っている。Phase 4 が PAD から声色を作るときに参照する |
| 4 | `loop/persistence.py` | 永続化の判定を独立させる。Phase 3 と Phase 4 が積み増す前に |
| 5 | `legacy/` への隔離 | Phase 6 の撤去を、ディレクトリの削除に近づける |

## 触らないもの（v0.6 の判断・前提が変わった）

v0.6 は「`run()` の制御構造（ループ管理、生成器、動作器）には手を入れない。ターン内イテレーションの入れ子は [D-反復出力] で撤去が決まっており、撤去される構造を整えても捨てることになる。作り替えは Phase 5 で行う」と決めていた。

**この前提はもう成り立たない**（2026-08-01）。`run()` は 環-c で撤去され、Phase 5 の作り替えは済んでいる。いまの `loop/event_loop.py` は捨てる構造ではなく、これから残る構造である。したがって「作り替えのときに一緒にやる」という先送りの根拠が無くなった。

切り出しは 環-e として立てる（下記）。

## 完了条件の考え方

各段で、移動元に旧名が残っていないことを grep で確認する（件数を数えるのではなく0件を示す）。挙動は変えないので、全体テストが緑であることが不変の条件になる。分割の前後で実機の体感が変わらないことも確かめる。

## 実施結果（store の切り出し）

`store/` の切り出しを次の順で行った。いずれも挙動不変で、各段ごとに全体テストが緑になってからコミットした。

| 段 | 内容 | 主なファイル |
|---|---|---|
| S1 | 時刻の一元化 | `store/clock.py` |
| S2 | 接続ラッパ | `store/db_compat.py` |
| S3 | 埋め込みとベクトル符号化 | `store/embedding.py` |
| 隔離 | 撤去予定の意味・信念層と明示リンク | `legacy/semantic_layer.py` |
| S4 | 非同期の書き込みキュー | `store/jobs.py` |
| S5 | 視点ベクトルと situated 行 | `store/situated.py` |
| S6a | 観測の読み出し層（by_kind／by_situated／by_date） | `store/observations.py` |
| S6b | 観測の書き込み本体 | 同上 |
| C1〜C3 | 継承から合成への組み替え、層ごとの単体テスト | `store/context.py` ほか |
| S6c | 想起の類似検索を `by_vector` へ、Config を層から外す | 同上 |
| S6d | 人物レジストリと観測系の残り | `store/persons.py` ほか |

結果の構成。SQL は `store/`（と `legacy/`）にだけ残る。

```text
store/  context.py     層が共有する道具（StoreContext）
        clock.py       時刻の一元化
        db_compat.py   接続ラッパ
        embedding.py   埋め込みとベクトル符号化
        persons.py     人物レジストリ
        situated.py    視点ベクトルと situated 行
        jobs.py        非同期の書き込みキュー
        observations.py 観測の読み書き（by_kind／by_situated／by_date／by_vector）
legacy/ semantic_layer.py 撤去予定（Phase 6）
tools/  memory.py      ファサード（ObservationMemory）＝公開面を委譲で集める
```

## 継承をやめて合成にした理由

当初は各層を mixin として `ObservationMemory` に継承させた。しかし mixin は宿主の名前空間を共有するため、層どうしが同名メソッドで衝突しうる。実際 S6b で、キュー層に置いた「宿主から借りる」宣言が観測層の実体化本体を MRO で覆い隠し、実体化が例外になる状態を作った（生存確認の不変条件が気づいた）。

そこで合成へ組み替えた。各層は普通のクラスにし、`StoreContext`（接続・ロック・person・埋め込み器の4つだけ）と、層をまたぐ依存を**引数で受け取る**。依存の向きは `jobs → observations → situated / legacy` の一方向である。`ObservationMemory` はファサードとして公開面だけを委譲する。**層の内部ヘルパーは委譲しない**（層の内側が外から触れる状態を残さないため）。委譲は `*a, **kw` でなく実際の署名を書く（何を受けるかがファサードから読めるように）。自動転送（`__getattr__`）は使わない。

合成にしたことで、各層を `StoreContext` だけで単独に組み立てられるようになった。これまで `ObservationMemory` 越しにしか触れなかった振る舞いに、層ごとの単体テストを当てた。

Config は層が持たない。設定は呼び出し側（ファサード）が `MemoryConfig` から読み、層は引数で受け取る（例：dedup の窓）。ワーカー的なクラスが設定を抱えると、同じ層を別の設定で使えなくなるためである。

## 作り替え予定で層へ移さなかった一群

`create_episode`／`append_to_episode`／`recall_divergent`（episodes）、`refresh_working_memory`／`get_working_memory`（memory_salience）、`open_unfinished_business`／`list_unfinished_business`（unfinished_business）は `tools/memory.py` に残した。W（作業記憶）は O からの派生ビューで毎ターン作り直す（[D-記憶単一化]）ので `memory_salience` に溜める形自体が変わり、エピソードと明示リンクは WR 拡散想起へ置き換わる（[D-WR拡散想起]）。Phase 5 で作り替えるため、いま層へ移しても捨てることになる。撤去が確定していないので `legacy/` にも入れない。行き先は作り替えの形が決まった段で決める。

## agent.py の切り出し方針

`store/` と違い、`agent.py` の対象は `EmbodiedAgent` のメソッドで、宿主の状態を多く参照する。切り出しの目的（一度に読める大きさにして依存を見えるようにする）を達するには、**依存が引数に束ねられるものだけ**を対象にする。別ファイルへ動かしても宿主参照を丸ごと渡すのでは、依存が減らず目的を達しない。

`self.` 属性の参照を実測して二つを比べた。

- **evaluator（`_emotion_for_turn`／`_summarize_exchange`／`_infer_companion_mood`／`_check_response_coherence`）**：触るのは実質 `_utility_backend`（軽量LLM）と `backend`（メインLLM）の二つ。`_evaluate_emotion_pad` は既にモジュール関数。設計の「評価器＝軽量LLM」（[D-I内部]）に名前が一致し、Phase 4 が PAD→声色で参照する先になる。**切り出す。**
- **persistence（`_run_post_response_pipeline`・192 行）**：`_prediction`／`_memory`／`_scene`／`_exploration`／`_relationship`／`_persons`／`_pmm`／各種 `_cached_*` など **28 種の `self.` 属性**に触る。エージェント状態のほぼ全域で、`StoreContext` のようには束ねられない。かつ Phase 5 で `run()` ごと作り替えが決まっている入れ子の一部である。**見送る**（いま切っても境界にならず、Phase 5 で作り直す）。

## 環-e（I の切り出し）

`loop/event_loop.py` を、設計が定める実体ごとのファイルへ分ける。目標構成のうち未作成なのは `loop_manager.py`・`generator.py`・`actor.py`・`persistence.py`・`silence_gate.py` と、`io/dif/`・`io/resources/` である。

**挙動を変えない。** この課題は移動だけを行う。守るのは次の4つである。

- **移すだけで、中身のロジックは変えない。** 名前も変えない（改名が要るなら別の作業として立てる）
- 移動元に旧名が残っていないことを **grep で0件**にする
- 全体テストが緑であることを不変の条件にする。挙動を変えないので、テスト側の修正は import 経路に限る
- 1段ずつコミットする

**段の順序は、依存の少ないものから決める。** 他から呼ばれるだけで他を呼ばない塊を先に出せば、残りへの影響が小さい。上の内訳では O 書込（136 行）と発話ゲート（190 行）がそれに当たり、生成器（`_iterate` 205 行を含む）が最も絡む。**具体的な順序は未定である**（切り出しの単位を決めてから確定する）。

**`see`・`look` から始めない。** 見回りは 知-c で挙動そのものを追っている最中で、同じ箇所を切り出しと機能変更の両方で触ると、どちらが原因で挙動が変わったのかを切り分けられなくなる。

## 残り

`min_score` の是正（生コサインの閾値から合成スコアの床へ）は実装済み。`core/recall_score.py` への抽出は D の後（内部R）に置く。persistence の切り出しは、v0.6 では「Phase 5 の `run()` 作り替えの中で行う」としていたが、その作り替えが済んだので **環-e** が引き取る。

---

## 更新履歴

> v0.7：**I の切り出しを 環-e として立てた**（2026-08-01）。「現状のずれ」を `agent.py` 基準から `loop/event_loop.py`（1,444 行）を含む形へ差し替え、設計上の実体ごとの行数を実測で載せた。生成器 350・ループ管理 325・動作器／資源ハンドラ 200・発話ゲート 190・O 書込 136 で、評価器と O のアクセス層は切り出し済みである。**資源は I が直接呼んでいる**ことを箇所つきで記録した（カメラ・VLM・検索・記憶・軽量LLM・TTS）。v0.6 の「`run()` の制御構造には触らない」は、`run()` が 環-c で撤去され Phase 5 の作り替えが済んだため前提が成り立たなくなったので、判断を書き直した。環-e は**挙動を変えず移動だけ**を行い、段の順序は依存の少ないものから決める（具体的な順序は未定）。`see`・`look` からは始めない（知-c で挙動を追っている最中のため）。

> v0.6：**用語の分離（6概念）を反映**した。`activation`・`a`・`score` に相乗りしていた量を、日本語・英語・記号の頭文字をすべて分けた（根づき groundedness g／高ぶり arousal a／勢い dynamism d／地力 merit m／顕著性 salience s／適合度 fit f）。旧称「覚醒」「喚起」は高ぶりへ統一した。定義は `用語_略語一覧` にある。

> v0.5：境界切り出しの実績と、順序方針のリファインメント（段取り v0.24）を反映。リファクタリングを 境界R（core/store/loop/io/legacy のつなぎ目）→ D（store 境界の中でのデータモデル整理）→ 内部R（中身整理）の3段に割り、loop に触る persistence 等は後回し（Phase 5）と位置づける。現状の分割実態＝`store/`（observations／situated／persons／jobs／context／clock／embedding／db_compat）、`loop/`（evaluator・history）、`legacy/`（semantic_layer）は実在。`io/`・`core/` は未作成。`tools/memory.py` は現在 1,348 行（agent.py は 3,826 行）。`min_score` の合成床化は実装済み。`core/recall_score.py` は未作成で、ハイブリッド5軸合成は `tools/memory.py` 内の `_score_breakdown`（正本）にある（抽出は内部R＝D 後へ）。
> v0.4：v0.3 の方針どおり **evaluator を `loop/evaluator.py` へ切り出した**（挙動保存）。`agent.py` から感情（`_emotion_for_turn`）・要約（`_summarize_exchange`）・相手気分（`_infer_companion_mood`）・整合性チェック（`_check_response_coherence`）の4メソッドと、値踏みゲート `A_GATE`・PAD 評価関数 `_evaluate_emotion_pad`・各プロンプト・`_companion_mood_heuristic` を移し、履歴走査 `_flatten_history` は `loop/history.py` へ分けた（評価器と要約が共有・循環 import 回避）。`agent.py` 側は薄い委譲だけ残す（テストの差し替え点でもある）。`EmbodiedAgent._evaluator` は、内部欲求ターンでメイン backend が utility へ一時スワップされても追随するよう、現在の `self.backend` と `_utility_backend` から導出する派生プロパティにした（スナップショットしない）。`store/` と同じく合成で、Config を持たず注入された backend だけに依存する。`loop/persistence.py` は v0.3 のとおり見送り（Phase 5 で `run()` ごと作り替える）。
> v0.3：`agent.py` の切り出し方針を、実測に基づいて絞った。当初は `loop/evaluator.py` と `loop/persistence.py` の二つを第一弾に入れていたが、`_run_post_response_pipeline` が 28 種の `self.` 属性（エージェント状態のほぼ全域）に触ると分かったため、**persistence は見送る**。切り出しても依存を束ねられず（`StoreContext` のようにいかない）、かつ Phase 5 で `run()` ごと作り替えるものだからである。**evaluator だけを切り出す**（依存は軽量LLM とメインバックエンドの二つに収まり、設計に名前があり、Phase 4 が参照する残るものである）。
> v0.2：`store/` の切り出しを実施した（S1〜S6d）。`tools/memory.py` は 2,594 行から 1,238 行へ減り、SQL は `store/`（と撤去予定の `legacy/`）にだけ残る形になった。途中で**継承（mixin）をやめて合成へ組み替えた**（C1〜C3）。mixin は宿主の名前空間を共有するため層どうしが名前で衝突し、実際に実体化が MRO で覆い隠される事故が出たためである。各層は `StoreContext` から共有の道具（接続・ロック・person・埋め込み器）を受け取り、層をまたぐ依存は引数に出す。あわせて層ごとの単体テストを足し（合成にしたことで層を単独で組み立てられる）、層が Config を直接読まない形にした。実施結果と、作り替え予定で層へ移さなかった一群を下に記す。
> v0.1：Phase 2 の締めに置く境界切り出し（課題8 v0.20）の、分割単位そのものを決める。判断の基準を「設計が定めたコンポーネントに合わせる」と「変わりそうな判断を隠す」の二つに置き、目標のファイル構成と第一弾の範囲を確定した。実装は未着手。
