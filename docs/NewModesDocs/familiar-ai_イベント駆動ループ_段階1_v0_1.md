# familiar-ai イベント駆動ループ（#11・段階1）（v0.3）

> v0.3：`run_iteration` に診断ログを追加（反復番号つき debug、ターン終了の info 総括、上限空終了の warning）。挙動不変。
> v0.2：スライス2（内部ツール recall を QC 経由で連鎖）を実装。I（情報処理機構）を `InformationProcessing` クラスとして起こし、QC（完了キュー）と LPM（ループ核＝`run_iteration`）を実体化。
> v0.1：段階1スライス1（人の発言→拡散込み想起→1反復1出力）を実装。正本＝`I内部設計根拠`・`設計図_Mermaid` ③。

## 位置づけ

#11 は現行のターン駆動 ReAct ループを、目標の**純イベント駆動ループ**（3キュー AIF/DIF/完了・1反復1出力・軽量LLM 調停）へ差し替える大工事。前提（起動源＝Drive・拡散想起）は実装済み。**増分・新ループを `loop/` に並行実装し `EVENT_LOOP` で現行 `run()` と排他切替**する（既定 off）。段階を薄い縦切りで進める。

## 段階1・スライス1（実装済み）

**人の発言（会話入力 trigger）→ 拡散込み想起（W）→ 1反復で1発話**。ツールを渡さず1出力を保証（発話のみ）。多段 ReAct はしない。

- **`loop/prompt.py`**：案B（クリーン最小・日本語ルール）の system プロンプト。
  - 静的核 `EVENT_SYSTEM_PROMPT`＝`(agent :type embodied (body eyes/neck/voice/**net**・**足なし**))`＋`(loop :one-output-per-iteration)`＋`(identity :id family-bond)`（家族の一員・切れない関係・時に厳しく・誰よりも愛する）＋rules（正直・**no-raw-internal-metrics**〔social_policy から移植〕・一人称視点取り・validation・bid・人格・言語）。net＝`search_deferred`/`fetch_deferred`（結果は後の反復で届く）。
  - `build_event_system_prompt(...)`＝静的核＋**自己認識 MI（ME.md＋FAMILY.md＋capabilities）**＋在席＋**PI（mood/drive 定性・生値なし）**＋W。**撤去対象（social_policy・mental_snapshot・interoception・relationship スカラ）は載せない**。
- **`loop/event_loop.py`**：`run_iteration(agent, utterance)`＝`recall`（5軸＋拡散込み）で W→system 構築→`stream_turn(tools=[])` で1発話→永続化は既存 `_run_post_response_pipeline`（**utility LLM のみ・フルLLM 不使用**）を spawn。
- **Config**：`EVENT_LOOP`（既定 off）。**`run()` 先頭で on の user turn のみ `run_iteration` へ排他切替**（`is True` 厳格判定・自発ターンは対象外）。

**挙動不変**：`EVENT_LOOP` off（既定）で現行 `run()` 経路のまま・全体テスト緑。

## 段階1・スライス2（実装済み）

**内部ツール `recall` を QC（完了キュー）経由で連鎖**させる、`設計図_Mermaid` ③ I 詳細図の最小縦切り。外部I/Oの無い `recall` を題材に、取込→O→W の連鎖機構だけを先に作る。

- **`InformationProcessing`（I：情報処理機構）**：`loop/event_loop.py` に新設。属性 `_completion_queue`（**QC**・`asyncio.Queue`）と、メソッド `run_iteration`（**LPM**：ループ核の drain 反復）を持つ。O・C（Config）・W・RH 相当のツール実行は既存実体を持つ `agent` を当面参照。AIF/DIF/QA/QD、ARB/APR/ACT/MNT のクラス分離は後続段階（stub しない）。
- **反復フロー**（④シーケンス整合）：(1) **取込**＝QC を drain し完了結果を **O 書込**（`save_async_with_id`, kind=`observation`）、(2) **REC**＝`recall_async` で W 構築、(3) **GEN**＝`stream_turn([say, recall])`。**say**→発話して終了／**recall**→**RH**（`_memory_tool.call`）が実行し結果を **QC へ enqueue**→次反復。相関ID は使わず結果は O→W で再会（[D-単一想起]）。
- **上限**：`event_max_iterations`（env `EVENT_MAX_ITERATIONS`・既定3）で打ち切る安全弁。
- **supersede**：消化した完了 O の id を `_run_post_response_pipeline(superseded_ids=...)` へ渡し、ターン観察保存後にその id で `mark_superseded`（完了結果の中間 O を想起から外す）。

**挙動不変**：`EVENT_LOOP` off（既定）で現行 `run()` 経路のまま。

## 診断ログ（実装済み）

`run_iteration` は、反復の進行と結末を後からログだけで再構成できるよう記録する。挙動（分岐や出力）は変えず、記録だけを足している。

- **反復ごと**（`debug`）：先頭に `iter=N/M`（N は 1 始まりの反復番号、M は上限）を付ける。反復頭 `event-loop iter=N/M 開始`、QC 取込 `event-loop iter=N/M QC取込=K件`（取込があった反復のみ）、決定直後 `event-loop iter=N/M 決定=say|recall|none`。どの反復のトレースかが一目で分かる。
- **ターン終了**（`info`）：`event-loop 終了: 反復=N/M 結末=発話|沈黙|空 text_len=L` を1行。本番（INFO）でも反復数と結末が残る。**結末**は、say で終われば「発話」、say も recall も無い素テキストで終われば「沈黙」、どちらも決まらず上限まで回り切れば「空」。
- **上限空終了**（`warning`）：結末が「空」のとき `event-loop 反復上限 M に達し発話未決のまま終了（空応答）` を出す。最終反復が recall で打ち切られ発話が未決のまま終わった経路を名指しする。

会話や記憶の中身はログに出さない（`text_len` の長さのみ）。

## 次の段階（未実装）

- スライス3：機器・net ツール（`see`/`search_deferred` 等）を QC 連鎖に載せる（`search_deferred` は投げっぱなしで完了が後の反復で QC に届く）。
- 段階2：軽量LLM 調停の3分岐（(a)軽量で閉じる／(b)軽量つなぎ→フル／(c)定型）。
- 段階3：AIF/DIF/完了 の3キュー結線（drive 発火・動体・deferred を各キューへ移設）。
- 段階4：二段生成（軽量つなぎ即答＋フル本応答）。
- 段階5：旧 `run()`・GUI アイドル分岐の撤去（Phase 6）。

## 確定した方針

- 言語：ルール文は日本語、骨格キーワード・ツール名は英語。
- 身体：実機に脚なし＝legs/walk 無し。net（インターネット）を body-tool に。音楽 MCP は実装時に body へ追加（動的検出が理想）。
- 家族：FAMILY.md（名簿）は自己認識 MI として既に注入済み。家族への根本的な立ち位置は `identity :id family-bond` として静的に持たせた。
