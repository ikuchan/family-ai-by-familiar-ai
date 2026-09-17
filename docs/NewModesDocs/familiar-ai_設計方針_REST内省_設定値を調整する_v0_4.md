# familiar-ai 設計方針：REST 内省・層 3「設定値を調整する」（v0.4）

設定値は REST 内省の 4 層（出来事 → 自己像 → 設定値 → 能力・`用語一覧` v0.70）の第 3 層で、機構の振る舞いを
決める値を持つ。**登録制・範囲つき**で、REST 内省が計測ログを読み、登録済みの範囲で値を動かす（動詞は
「調整する」）。対象は開発とともに増える。

## 1. 外形（2026-09-14 決定・4 層共通）

- 現在値は **DB**（`agent_state.config_overrides`・既存の器 `config_overrides.py`）。既定値は `config.py`（GitHub）。
- **優先順位は DB > 既定**。旧「env > DB > 既定」から `.env` を外す——`.env` は機密と機体固有のものだけを持つ。
  登録した値と同名の環境変数があれば起動時に WARNING を 1 行出し、値は使わない（黙って無視しない）。
- 接続情報（鍵・URL・機器 id 等）は対象外のまま（`is_protected`）。式の骨格（`rate`・`theta_fire`）も対象外。

## 2. 登録

登録が無い値は動かせない。登録には次を**必ず**添える（`RANGES` を辞書から表へ）：

| 項目 | 内容 |
|---|---|
| 完全名 | `MemoryConfig.recent_exchanges_main` のように |
| 範囲・刻み | 下限・上限・1 晩に動かせる幅（既定は 1 刻み） |
| 見る計測 | 計測ログのどの種別・どの欄を集計するか |
| 動かす規則 | 機械で決まるもの（閾値の比較）か、LLM が数字を見て提案するものか |
| 根拠 | `計測・設定値 根拠台帳` の節 |

**登録候補**（2026-09-14）：

| 値 | 範囲〔仮〕 | 見る計測 | 規則 |
|---|---|---|---|
| `MemoryConfig.distill_min_a0` | 0.20〜0.70（登録済み） | 蒸留の材料の a0 分布 | LLM |
| `MemoryConfig.recent_exchanges_arbiter`／`_main` | 1〜10 | 申告の深さ（`referred`／`important` が直近の何往復目か） | 機械：窓の端に参照が 20% 以上なら +1、末尾が一度も参照されなければ −1（`課題5` D 章） |
| `AgentConfig.arbiter_timeout_sec` | 1.0〜10.0 | 調停の秒数・時間切れの割合 | LLM（時間切れの割合と p90） |
| `MemoryConfig.recall_half_life_days`（$HL$・登録済み） | 1〜30 日 | 層 1 の $I$（毎晩 1 行） | LLM（$I$ の目標との差） |
| `MemoryConfig.diffuse_far_share`（登録済み） | 0〜1 | 関連想起で載った記録の申告（遠い順由来／思い出していない順由来） | 機械：参照された側へ寄せる |
| `MemoryConfig.info_target_bits`（$I^\*$・登録済み・2026-09-15） | $2^{14}$〜$2^{20}$ | 層 1 の $I$（毎晩 1 行） | LLM（未接続・規則は核の固めの後に） |
| `MemoryConfig.core_same_cos`／`core_bundle_cos`／`core_bundle_min`／`core_bundles_per_night`（登録済み・2026-09-15・核の固め） | 0.90〜1.00／0.30〜0.80／2〜6／1〜20 | 層 1（固め） | LLM（未接続・`出来事を畳む` §3c） |
| `InnerStateConfig.mood_*_p10/p30/p70/p90`（16）・`drive_p70_*`（5）（登録済み・情-f） | 0.0〜1.0 | `気分`・`欲求` の行（毎ターン） | 機械：等頻度になるよう分位を取り直す |

## 3. 材料は計測ログだけ

- `rest_logs/measure.log`（記-i）。機械が種別ごとに集計し、**数字**（件数・中央・p90・最大・時系列の傾向）を渡す。
  生の行を LLM に渡さない（数百行から分位点を読ませると誤る）。
- 集計は**前回の改名以降**の行だけ。読み終えたら `measure.log.<読んだ時刻>` へ改名する（回転は REST）。
- 計測ログに足す行（書き手は `core/measure.record`）：

| 種別 | 欄 | 用途 |
|---|---|---|
| `続き先` | 結末・相手・起点（記-i・済） | 判定の健全性 |
| `気分`・`欲求` | PAD 4 軸・欲求 5 軸の生値・発火（情-f・済） | 内部状態の言葉の境目 |
| `調停` | 秒・分岐・時間切れ | `arbiter_timeout_sec` |
| `申告` | 記録 id・判定・直近の何往復目か | $n$ の見直し |
| `層1` | $I$・核の情報量・畳んだ件数・固めた件数・$\Delta$ | $I^\*$・$HL$・閾値 |

## 4. 動かし方

1. 値ごとに、登録した規則で「上げる・下げる・動かさない」を決める。
2. **1 晩に 1 値につき 1 刻み**。範囲の端で止まる。
3. 動かしたら、計測ログ（`設定値 名前=… 前=… 後=… 根拠=…`）と `direction='内省'` の記録に残す。
4. 人が既定に戻したいときは DB の値を消す（`config_overrides` の削除）。

## 4b. 実装（v0.2・2026-09-14・記-a-に）

- 登録の表：`core/settings.py` の `REGISTRY`（`Setting(field, lo, hi, step, measure_kind, rule, note)`・26 件）。
  `config_overrides.RANGES` はここから導く。
- 優先順位：`config._resolve_setting`（**DB > 既定**・env は読まず WARNING）に一本化（旧 `resolve_float`／`_resolve_float`
  は撤去）。`distill_min_a0`・`recall_half_life_days`・`diffuse_far_share`・`arbiter_timeout_sec`・`recent_exchanges_*`・
  内部状態の境目 21 個がこれを通る。
- 計測ログの行：`調停 秒 分岐 時間切れ`（`arbiter.arbitrate`）、`直近 窓 端 外`（`Workspace.build`）、
  `申告 important useless referred unused`（`apply_memory_verdicts`）、`関連 遠い 掘り`（拡散想起）、
  `設定値 名前 前 後`（動かしたとき）。既存の `続き先`・`気分`・`欲求`・`層1`・`層2`。
- 読み手と集計：`core/measure.read_rows`・`summarize_arbiter`（件数・中央・p90・最大・時間切れ）・
  `summarize_window`（続きの相手が端か外だった回数）・`summarize_inner_state`（軸ごとの分位）・
  `summarize_relation`（参照された数を並びごと）・`rotate`（`measure.log.<時刻>` へ改名）。
- 規則：`loop/rest_settings.py`——`adjust_window`（端か外の参照 ≥20% で +1・続きがあるのに 0 で −1）、
  `adjust_inner_state`（計測した分位へ 1 刻み）、`adjust_far_share`（参照された側へ 0.1）、`adjust_timeout`
  （LLM に数字を渡し、提案を 1 刻み・範囲内に丸める）、`apply`（`save_override`・`内省` の記録・計測ログ）、
  `adjust_settings`（読む → 集計 → 動かす → 改名）。`rest.run_rest_pass` が層 2 の後に呼ぶ。

## 5. 未決〔仮〕

| 項目 | 仮値 | 決め方 |
|---|---|---|
| 各値の範囲・刻み | 上表 | 登録時に `根拠台帳` へ |
| 「窓の端に 20%」の閾値 | 20% | 数晩の実測 |
| LLM が提案する値の呼び方 | 内省の 1 往復に同梱 | 層 1・2 の実装で決まる内省ループの形に合わせる |

## 最小の標本数（v0.3・記-k・2026-09-15）

規則ごとに標本が足りなければその晩は動かさない（`rest_settings.MIN_SAMPLES`〔仮〕）：内部状態 50 行（`気分`／`欲求`）・
窓 20 行（`続き先` の続き）・時間切れ 20 行（`調停`）・関連 20 行（`申告`）。実機で計測ログ 121 行のうち `気分`／`欲求` が
4 行だったのに、境目 9 件が 1 刻みずつ動いた。集計（`core/measure.summarize_*`）が `件数` を持つ。

## 更新履歴

> v0.4：登録表に 2026-09-15 に登録した 5 つ（$I^\*$ と核の固めの 4 つ）を足し、「出来事を畳む v0.2 の未決」の行を置き換えた（`REGISTRY` と照合・2026-09-17）。
> v0.3：規則ごとの最小の標本数（記-k・2026-09-15）。

> v0.2：**実装**（2026-09-14・記-a-に）。§4b に在りか。
> v0.1 追記 2（同日）：$HL$ と `diffuse_far_share` を登録済みに（記-a-ろ-い）。
> v0.1 追記（同日）：情-f の境目 21 個を登録候補と計測ログの行に足した。
> v0.1：新規作成（2026-09-14）。外形（DB > 既定・`.env` を読まない）、登録の必須項目、登録候補、材料は計測ログだけ、
> 動かし方（1 晩 1 刻み・記録を残す）。
