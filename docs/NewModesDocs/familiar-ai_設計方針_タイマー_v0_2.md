# familiar-ai 設計方針：タイマー（アラーム・タイマー・ストップウォッチ）（v0.2）

「7 時に起こして」「3 分測って」「今から測って」に応える。**途中で停められること**を軸に設計する
（知-n・2026-09-15）。用語一覧の定義（期限つきの意図・`set_timer`・T が due で発火・I は時計を持たない）を実装に落とした。

## 1. 止め方（主）

| 口 | 何が起きるか |
|---|---|
| **声**：「タイマー止めて」「やっぱりいい」 | 主LLM の道具 `cancel_timer(id か "all")`。W に**動いているタイマーの枠** `[タイマー]`（id・何のため・残り／経過）が毎反復載るので、主LLM はどれを止めるか分かる |
| **画面**：`/timer stop [id]` | LLM を通さず即止め（確実な非常口）。区切りは全角空白・中黒でも通る |
| **鳴ったあと** | 発話は 1 回だけ（鳴り続けない）。鳴った後 180 秒〔仮〕は枠に「id=… は N 分前に鳴った（もう止まっている）」を残し、「止めて」に「もう止まっている」と答えられる |

止めたことは O に「やめた：「…」のタイマー」の記録（`予定`）として残す（supersede しない）。

## 2. 置き場

表 **`timers`**（065）：`id`・`label`・`due`（timestamptz・ストップウォッチは NULL）・`started_at`・`fired_at`・`cancelled_at`・`asked_by`・`obs_id`・`passes_quiet`。
状態は列で持ち、content の時刻を読まない。再起動をまたいで残る。器は `store/timers.py`（`add`・`active`・`due_now`・`recently_fired`・`mark_fired`・`cancel`・`cancel_all`）。
あわせて O に `direction='予定'` の記録「パパに頼まれて、19:33 に「パスタ」のタイマーを掛けた」を書く（想起に載る・「約束した」が記憶になる）。

## 3. 掛け方（`tools/timer.py`・主LLM の道具 3 本）

- `set_timer(after_minutes か at, label, confirmed=false)`：どちらか一方。`at` はローカル時刻（`7:00`・`21時半`・全角可）、過ぎていれば翌日（`core/timer_rules.resolve_due`）。返りに id と鳴る時刻。
- `start_stopwatch(label)`：due 無し。鳴らない。
- `cancel_timer(id か "all")`。
- 同時に **5 本〔仮〕**まで。**調停（軽量LLM）が自分で掛ける**：候補に `set_timer`／`start_stopwatch`／`cancel_timer` を載せ、調停は `{"branch":"action","action":"set_timer","tool_input":{"after_minutes":3,"label":"パスタ"},"text":"3分ね、測るよ"}` のように**道具の入力を `tool_input` に書く**（`Decision.tool_input`・`_tool_input_of`）。道具は即実行され、完了が戻ると調停が light で「掛けたよ」と言える——主LLM は起きない。「確かめて」が返れば light で聞き、「いい」なら `confirmed:true` で掛け直す。それでも調停が light を選んだときは、道具が要る頼み（`arbiter.needs_tools`）を full へ倒す（最後の砦）——実機（2026-09-15 22:47）で light が「タイマーをセットしました」と言うだけで掛かっていなかった。道具は `_ACTIONS`／`_FULL_ACTIONS`／`_LOOKUP_ACTIONS` に載り、`recall` と同じく結果はその場で返って次の反復が言葉にする。見出しは入力ごとに別（「タイマーを掛ける「パスタ」」）で、同じ求めで掛けて止めるができる。

## 4. 通り抜けと、その前の確認（2026-09-15 決定）

鳴るときの発話は**在席ゲート・静穏時間・沈黙の依頼を通り抜けて 1 回だけ話す**——頼んだ本人がいて、1 回だけだから。
ただし**通り抜ける可能性があるものは、いきなり登録せず一度確かめる**：

- `set_timer` は due を解いたあと、**静穏時間（`QUIET_HOURS`・既定 23〜7）に入る／沈黙の依頼が生きている**なら登録せず「まだ掛けていない。確かめてから：…」を返す（`timer_rules.needs_confirmation`・機械が判定）。主LLM はその理由を本人に伝えて一度だけ聞き、「いい」なら `confirmed=true` で呼び直す。登録時に `passes_quiet` を立てる。
- 通り抜けない時間帯のものは確認なしで即登録。
- 静穏時間の端（7:00 は 23〜7 の外）は通り抜けの対象ではない。

## 5. 鳴らし方（`loop/timer_watch.py`・T）

T が毎 tick（0.5 秒）`due_now` を拾い、**先に `fired_at` を打ってから**（二度鳴らさない）`機器` の求め「タイマー：「…」の時間（パパに頼まれたもの）」を積む（`DIF.device(…, passes_gate=passes_quiet)`）。求めは `Request.passes_gate` を持ち、`_delivery_block_reason()` の先頭で通り抜ける。落ちていた間に due を過ぎたものは起動後の最初の tick で「N 分遅れて鳴っている（HH:MM の予定だった）」を添えて鳴る（60 秒以上の遅れ）。

## 5a. 音で鳴る（知-n-ろ・v0.2・2026-09-17）

「タイマーです」の一言でなく、**タイマーらしい音**で知らせる。音は自作（拾ってきた音源は許諾の確認と出典の
記録が要る）——`scripts/gen_timer_alarm.py` が numpy で「ピピピッ」（1 kHz と 1.3 kHz の短音 3 連＋休み・
1 周期 1.0 秒・48 kHz・16 bit）を合成し `src/familiar_agent/sounds/timer_alarm.wav` に置く。

- **鳴らす**：`fire_due` が機器の求めを積むのと同時に `DIF.ring_timer(seconds=TIMER_RING_SEC, gain=TIMER_VOICE_GAIN)`。
  wav を `seconds` のあいだ繰り返す（`asyncio.Task`・再生は `tools/tts._play_via_sounddevice`）。
- **止める**：`cancel_timer`（声の「止めて」・`TimerTool` の `on_cancel` → `agent._stop_timer_ring`）、`/timer stop`（同じ道具）、時間切れ。
  鳴った時点でタイマーは `active` に無いので、`_cancel` は**先に**音を止めてから表を見る。
- **マイクは閉じない**：声の口（`DIF.speak`）を通らないので `tts_active` の門が立たず、鳴っている最中の「止めて」が届く。
  音をマイクが拾う分は STT の幻聴の門（`no_speech_prob`）で落ちる想定——**実機で確かめる**（`課題8` 束 C）。
- **静穏時間**：確かめて掛けたもの（`passes_quiet`）だけ音も鳴る（声と同じ扱い）。
- **声**：機器の求め（沈黙を解く・O・`[届いた知らせ]`）は変えない。調停の `_LEAD_DEVICE` に「音でも知らせているので、
  聞かれない限り黙っていてよい」を足した。
- `.env`：`TIMER_RING_SEC`（既定 30〔仮〕・`0` で音を鳴らさず声だけ＝これまでどおり）。

## 5b. 起点は人が言った瞬間（2026-09-15 夜）

調停・道具・言葉にするまでに 3 秒ほどかかる。「3 分測って」「はい」「始めて」と**言った瞬間**（求めの入口
`push_utterance` の時刻・`Request.began_at`）を `set_timer`／`start_stopwatch` の起点にする（`TimerTool.call(…, now=)`）。
due＝言った時刻＋N 分、ストップウォッチの `started_at`＝言った時刻。

## 6. 今何分？

主LLM が `[タイマー]` の枠（残り／経過は枠を組むときに計算・`timer_rules.render_frame`）から答える。発火は要らない。枠は `[反復]` の後ろに添え、動いているものも直前に鳴ったものも無ければ出さない。

## 7. 仮値

同時 5 本・鳴った後の表示 180 秒・「遅れて」を添える閾 60 秒。

## 更新履歴

> v0.2：**音で鳴る**（知-n-ろ・2026-09-17）——自作の wav を `TIMER_RING_SEC` 繰り返す・止めるのは `cancel_timer`／`/timer stop`／時間切れ・マイクは閉じない・静穏時間は確かめたものだけ。§5a。

> v0.1 追記 2（2026-09-15 夜）：起点は人が言った瞬間（§5b）。実機の 4 件（dict の cursor・`writer_id`・`query` からの入力・動作の無い action）を直した。
> v0.1 追記（2026-09-15 夜）：調停が自分でタイマーを掛ける（候補 3 つ・`tool_input`）。道具が要る頼みで light が残ったら full へ倒す（`needs_tools`）。
> v0.1：新規（2026-09-15・知-n）。設計方針と実装計画の承認のうえ実装した内容だけを書いた。


> 追記（2026-09-16・実装済み）：**鳴った知らせだけ声を大きくする**。`.env` の `TIMER_VOICE_GAIN`（既定 1.0・この機体は 1.5）を、タイマー起点（`[タイマー]`）の求めの発話にだけ掛ける（`InformationProcessing._voice_gain` → `DIF.speak(gain=)` → `TTSTool.say(gain=)` → sounddevice の再生で PCM に倍率・飽和）。機器の音量には触らず、その 1 回の再生データにだけ効くので、鳴り終わった次の発話は元の音量で戻す処理は要らない。

> 追記（2026-09-16・実装済み）：**掛けているあいだは黙り、鳴ったら戻す**（`TIMER_SILENCE`・既定 有効・`false` でこれまでどおり）。`set_timer` で掛けた瞬間に、既存の沈黙依頼の仕組みで `SilenceRequest(person=頼んだ人, until=鳴る時刻, reason="timer:<id>")` を保存する（`silence_state.hush_for_timer`・純関数）。鳴る時刻＝期限なので鳴る知らせは何もしなくても通り、`timer_watch` は知らせに `release_pending=True` を付けて溜めていた返事も一緒に配る。`cancel_timer` で止めたらそのタイマー由来の沈黙を解く（`unhush_timer`・`all` はタイマー由来すべて）。人が明示的に頼んだ沈黙（`reason` が空）は上書きしない。複数のタイマーは遅いほうの鳴る時刻まで。ストップウォッチは対象外。タイマー由来の沈黙は、次のタイマーの「確かめて」の条件（`_silence_active_now`）には数えない。調停の `[いま]` には「タイマーが鳴るまで黙っている（あと約 n 分）」。

> 追記（2026-09-16・情-h）：タイマー由来の沈黙も、人の「黙って」も、**黙っているあいだは誰の声にも返事をせず、入口で聞くだけ**にする（案イ撤回・`イベント駆動ループ` v0.73）。鳴った知らせの求めに、黙っていたあいだに聞いたこと・起きたこと・湧いたことが列挙され、主LLM が 1 回でまとめて答える。`release_pending` による返事の保留はこの経路では使わない。
