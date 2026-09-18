# familiar-ai 設計方針：ストップウォッチ（v0.1）

> **測る**もの。「今から測って」で始め、「止めて」で止め、測った長さを答える。タイマー（短い時間を**待つ**）とも
> アラーム（遠い時刻に**起こす**）とも別物として作る（知-u・2026-09-18）。共有するのは `予定` の記録と T の tick だけ。
> 09-16 に主LLM が始めた 2 本（「何のために時間を測るか不明」「パパのタイマー」）が 2 日間動き続け、`/timer stop` の
> 返事「2 本止めた」で何を止めたか分からなかった。タイマーと同じ表・道具・命令・枠に混ざっていたことが原因。

## 1. タイマー・アラームとの違い（一覧）

| | タイマー（`設計方針_タイマー`） | アラーム（`設計方針_アラーム`） | ストップウォッチ（この資料） |
|---|---|---|---|
| 意味 | 短い時間を待つ（その間は集中） | 遠い時刻に起こす・知らせる | **今からどれだけかかるかを測る** |
| 表 | `timers`（065・066） | `alarms`（067） | **`stopwatches`**（068） |
| 道具 | `set_timer`・`cancel_timer`・`pause_timer`・`resume_timer` | `set_alarm`・`cancel_alarm` | **`start_stopwatch(label)`・`stop_stopwatch(id か all)`** |
| 命令 | `/timer stop`・`/timer pause`・`/timer resume`・`/mic on` | `/alarm stop [id]` | **`/stopwatch stop [id]`** |
| 掛ける前の確認 | `TIMER_CONFIRM`＋静穏時間・沈黙中は常に | 静穏時間に鳴るときだけ | **しない**（鳴らないので） |
| 黙る・聞かない | `TIMER_SILENCE`・`TIMER_MIC_CLOSE` | しない | **しない** |
| 同時 | 1 本 | 5 本〔仮〕 | **1 本** |
| 一時停止・再開 | あり | なし | **なし**（止めるだけ） |
| 鳴らす | `timer_watch`・`TIMER_RING_SEC` | `alarm_watch`・`ALARM_RING_SEC` | **鳴らない** |
| 寿命 | 鳴る時刻 | 鳴る時刻 | **`STOPWATCH_MAX_SEC`（6 時間）で T が止める** |
| 枠 | `[タイマー]` | `[アラーム]` | **`[ストップウォッチ]`**（経過・寿命が近ければ「あと N 分で自動で止まる」） |
| 規則 | `core/timer_rules.py` | `core/alarm_rules.py` | **`core/stopwatch_rules.py`**（経過の言い方・枠） |
| 記録（O） | `予定`「…のタイマーを掛けた／やめた」 | `予定`「…にアラームを掛けた／やめた」 | `予定`「…から「…」を測り始めた」「「…」のストップウォッチを 6 時間で止めた」 |

## 2. 置き場

表 **`stopwatches`**（068）：`id`・`label`・`started_at`・`stopped_at`・`asked_by`・`obs_id`・`expired`（寿命で T が止めた印）。
器は `store/stopwatches.py`（`add`・`active`・`recently_stopped`・`stop`・`stop_all`・`expire`）。
`timers` に残っていた `due` が NULL の行（2 本・止め済み）は移さない。`timers` の `active` は `due` のある行だけを返す。

## 3. 始め方・止め方（`tools/stopwatch.py`）

- `start_stopwatch(label)`：同時 1 本。2 本目は「いま「…」（id=…・経過 …）を測っている。止めてから」。起点は人が言った瞬間
  （`Request.began_at`・タイマーと同じ）。
- `stop_stopwatch(id か "all")`：返りは **「ストップウォッチ「お風呂」を止めた（58 分 12 秒 経過）」**——種類と経過を必ず言う。
  `all` で複数なら「・」でつなぐ。動いていなければ「動いているストップウォッチは無い」。
- **調停（軽量LLM）の候補と主LLM の道具の両方**に載る（`event_loop._STOPWATCH_ACTIONS`・`arbiter._EXTRA_ACTIONS`）。
  調停が `start_stopwatch` に `{"id": …}` と書けば `stop_stopwatch` に直す（`arbiter._parse`）。帰りの反復は想起なし
  （`workspace.RETURN_WITHOUT_RECALL`・`イベント駆動ループ` v0.84）。
- 経過の言い方（`stopwatch_rules.elapsed_text`）：1 分未満は秒、1 時間未満は分＋秒、1 日未満は時間＋分、それ以上は日＋時間。

## 4. 寿命（`loop/stopwatch_watch.py`・T）

T が 60 秒に 1 度 `expire` を呼び、`STOPWATCH_MAX_SEC`（6 時間）を超えて動いているものを止めて `expired` を立て、O に
`予定`「「…」のストップウォッチを 6 時間で止めた（止められないまま動いていた）」を書く。**声は出さない**（枠に
「寿命で自動で止めた」が 180 秒残るだけ）。寿命が 30 分〔仮〕を切ると枠に「あと N 分で自動で止まる」を添える。

## 5. 名前の検め（未着手・知-u の残り）

`label` は主LLM が書く。文（「何のために時間を測るか不明」）が入っても機械は検めない。上限の字数〔仮 12〕を超えるか
「不明」を含めば「測る」に置き換える案は、仮値が未承認のため入れていない。

## 6. 仮値

同時 1 本・寿命 6 時間（`STOPWATCH_MAX_SEC`・確定）・止めた後の表示 180 秒・寿命の予告 30 分。

## 更新履歴

> v0.1：新設（知-u・2026-09-18）。タイマーから分離——表 `stopwatches`（068）・道具 2 本・`/stopwatch stop`・`stopwatch_watch`・`stopwatch_rules`・鳴らない・確認しない・黙らない・同時 1 本・寿命 6 時間。
