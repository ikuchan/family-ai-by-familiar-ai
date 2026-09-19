# familiar-ai 設計方針：アラーム（v0.4）

> 遠い時刻に**起こす・知らせる**もの。タイマー（短い時間を**待つ**・測る）とは別物として作る（知-q・2026-09-18）。
> 共有するのは基盤だけ——音の再生（`DIF.ring`）、T の tick、`予定` の記録、静穏時間の判定（`quiet_hours_rule`）。
> 2026-09-15 の知-n では `set_timer(at=…)` の引数の違いだけで同じ表・同じ規則を通っていたが、タイマーの規則
> （掛けているあいだ黙る・聞かない・同時 1 本・一時停止）をアラームに当てると「7 時に起こして」で翌朝まで黙って
> 聞かない、朝のアラームが入っているとキッチンタイマーが掛けられない、になる。依存を全部切った。

## 1. タイマーとの違い（一覧）

| | タイマー（`設計方針_タイマー`） | アラーム（この資料） |
|---|---|---|
| 意味 | 短い時間を待つ（その間は集中）。測る | 遠い時刻に起こす・知らせる。それまでは普通に暮らす |
| 表 | `timers`（065・066） | **`alarms`**（067） |
| 道具 | `set_timer(after_minutes)`・`pause_timer`・`resume_timer`・`cancel_timer`（ストップウォッチは別物・`設計方針_ストップウォッチ`） | **`set_alarm(at, label)`・`cancel_alarm(id か all)`** |
| 命令 | `/timer stop`・`/timer pause`・`/timer resume`・`/mic on` | **`/alarm stop [id]`** |
| 掛ける前の確認 | `TIMER_CONFIRM`（既定 true）＋静穏時間・沈黙中は常に | **静穏時間に鳴るときだけ**（`alarm_rules.needs_confirmation`） |
| 黙る・聞かない | `TIMER_SILENCE`・`TIMER_MIC_CLOSE` | **しない**（設定も無い） |
| 同時 | タイマー 1 本・ストップウォッチ 1 本 | **5 本**〔仮〕（朝と昼など） |
| 一時停止・再開 | あり | なし（中止だけ） |
| 鳴らす | `timer_watch`・`TIMER_RING_SEC` | **`alarm_watch`**（同じ tick から）・`ALARM_RING_SEC`（30）・機器の求め「アラーム」。山谷（8 秒で 0.1・25 秒で戻す）はタイマーと同じ（`設計方針_タイマー` §5a） |
| 枠 | `[タイマー]` | **`[アラーム]`**（次に鳴る時刻・今日／明日） |
| 規則 | `core/timer_rules.py`（分数だけ） | **`core/alarm_rules.py`**（`resolve_at`・「7 時半」「21:00」・過ぎていれば翌日） |
| 記録（O） | `予定`「…のタイマーを掛けた／やめた」 | `予定`「…にアラームを掛けた／やめた」 |

## 2. 置き場

表 **`alarms`**（067）：`id`・`label`・`at`（timestamptz・鳴る時刻）・`set_at`・`fired_at`・`cancelled_at`・`asked_by`・`obs_id`・`passes_quiet`。
器は `store/alarms.py`（`add`・`active`・`due_now`・`recently_fired`・`mark_fired`・`cancel`・`cancel_all`）。
`timers` に残っていた `at` 由来の行は区別できず、未来の未発火は無いので移さない。

## 3. 掛け方（`tools/alarm.py`）

- `set_alarm(at, label, confirmed=false)`：`at` はローカル時刻（`7:00`・`21時半`・全角可）、過ぎていれば翌日。
  静穏時間に鳴るなら預かり（`confirm_state.PendingConfirm`・`設計方針_タイマー` §10 と同じ器）を置いて「確かめて」を返し、主LLM か調停が本人に一度聞くだけ。「いい」は機械の `confirm` が `call(..., confirmed=True)` で掛ける（`passes_quiet` が立つ）。`confirmed` は LLM の引数に無い（v0.2）。
- `cancel_alarm(id か "all")`：止める前に鳴っている音を止める（`on_cancel` → `agent._stop_timer_ring`・音の口は共有）。
- **調停（軽量LLM）の候補と主LLM の道具の両方**に載る。どちらに行っても同じ道具（`event_loop._ALARM_ACTIONS`・`arbiter._EXTRA_ACTIONS`）。
  調停が `set_timer` に `{"at": …}` と書いても `set_alarm` に直す（`arbiter._parse`）。`query` に「7 時 起こす」と書いたときも同じ。
- `set_timer` に `at` を渡すと「何時に、はアラーム（set_alarm）」で断る（`set_timer` から `at` を撤去・2026-09-18）。

## 4. 鳴らし方（`loop/alarm_watch.py`・T）

T が毎 tick `due_now` を拾い、**先に `fired_at` を打ってから**音（`DIF.ring`・`ALARM_RING_SEC`・倍率 `TIMER_VOICE_GAIN`）を鳴らし、
`機器` の求め「アラーム：「起こす」の時刻（パパに頼まれたもの）」を積む（`passes_gate=passes_quiet`）。静穏時間の音は確かめて掛けた
ものだけ（声も同じ）。落ちていた間に過ぎたものは「N 分遅れて鳴っている」を添える。音は `cancel_alarm`・`/alarm stop`・時間切れで止まる。

## 5. 仮値

同時 5 本・`ALARM_RING_SEC` 30 秒・鳴った後の表示 180 秒・「遅れて」を添える閾 60 秒。

## 更新履歴

> v0.4：鳴り方の山谷はタイマーと同じ（2026-09-19）。
> v0.3：道具の行からストップウォッチを外した（知-u で別物に・2026-09-18）。
> v0.2：確認を機械の状態に（出-y・2026-09-18）——`confirmed` を道具から撤去、預かりと `confirm`／`decline` はタイマーと共通。
> v0.1：新設（知-q・2026-09-18）。タイマーから分離——表 `alarms`（067）・道具 2 本・`alarm_watch`・`alarm_rules`・`/alarm stop`・確認は静穏時間だけ・黙らない・同時 5 本。
