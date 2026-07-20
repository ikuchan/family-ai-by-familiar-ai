# familiar-ai モジュール分割設計（v0.1）

> v0.1：Phase 2 の締めに置く境界切り出し（課題8 v0.20）の、分割単位そのものを決める。判断の基準を「設計が定めたコンポーネントに合わせる」と「変わりそうな判断を隠す」の二つに置き、目標のファイル構成と第一弾の範囲を確定した。実装は未着手。

## この文書が決めること

`agent.py`（4,025行）と `tools/memory.py`（2,594行）を、どの単位でファイルへ切り出すかを決める。行数を減らすことが目的ではない。**コードを開いた人が設計図と突き合わせて読める状態**にすることが目的で、行数はその結果として下がる。

置き場所と順序は課題8 v0.20 が決めている。本書はその中身にあたる。

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

`agent.py` が大きいのは行数の問題ではなく、**設計上5つに分かれている実体が1ファイルに同居している**ためである。

| 設計上の実体 | いまの居場所 |
|---|---|
| 評価器 | `agent.py` に散在（`_evaluate_emotion_pad`／`_summarize_exchange`／`_infer_companion_mood`／`_check_response_coherence`） |
| 生成器 | `run()` の中に埋没 |
| 動作器 | `_execute_tool` と `run()` のツール実行分岐に混在 |
| 統合保守器 | 日次処理（`_generate_day_summary`／`_maybe_adapt_values` ほか）に前身が散在 |
| ループ管理 | `run()` 781行 |
| O のアクセス層 | `memory.py` に4本だけ実装、残りは生 SQL のまま同居 |
| W 構築の採点 | `memory.py` の `_compute_final_score` 周辺 |

2026-07-20 に見つけた不具合のうち、`say()` で話したターンが永続化されない件は、永続化の判定が `run()` の約590行目に埋もれていたために3週間気づかれなかった。設計上は「O 書込の呼び出し口」という独立した関心であり、同居させる理由が無い。

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
| 2 | `core/recall_score.py` | Phase 2 の本体。次の `min_score` 是正がこの上に乗る |
| 3 | `loop/evaluator.py` | 設計に名前のある実体が `agent.py` に散っている。Phase 4 が PAD から声色を作るときに参照する |
| 4 | `loop/persistence.py` | 永続化の判定を独立させる。Phase 3 と Phase 4 が積み増す前に |
| 5 | `legacy/` への隔離 | Phase 6 の撤去を、ディレクトリの削除に近づける |

## 触らないもの

`run()` の制御構造（ループ管理、生成器、動作器）には手を入れない。ターン内イテレーションの入れ子は [D-反復出力] で撤去が決まっており、撤去される構造を整えても捨てることになる。作り替えは Phase 5 で行う。

## 完了条件の考え方

各段で、移動元に旧名が残っていないことを grep で確認する（件数を数えるのではなく0件を示す）。挙動は変えないので、全体テストが緑であることが不変の条件になる。分割の前後で実機の体感が変わらないことも確かめる。
