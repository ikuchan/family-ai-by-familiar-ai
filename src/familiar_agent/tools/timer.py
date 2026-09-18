"""タイマーの道具（知-n・2026-09-15・`設計方針_タイマー` v0.1）。

主LLM が呼ぶ 4 本——`set_timer`（何分後・アラームは別物 `tools/alarm.py`・ストップウォッチも別物
`tools/stopwatch.py`・2026-09-18 知-u）・`cancel_timer`（止める・**「途中で停められる」の主な口**）・
`pause_timer`・`resume_timer`。状態は表 `timers`（`store/timers.py`）、記憶には `予定` の記録（登録）と
「やめた」の記録（取消）を書く。

**掛ける前に確かめる（`TIMER_CONFIRM`）・静穏時間に掛かる／黙っているよう頼まれているときは、
いきなり登録しない。** 「確かめて」の判定では預かり（`core/confirm_state.py`・`ask`）を置いて確認文
だけを返し、LLM は本人に一度聞くだけ。「いい」は機械（`confirm`）が預かった入力で `call(...,
confirmed=True)` と呼び直す。`confirmed` は LLM の入力に無く、書いても効かない（出-y・2026-09-18）。
静穏時間のものは印（`passes_quiet`）で鳴るときに配信ゲートを通り抜ける。
"""

from __future__ import annotations

from dataclasses import dataclass

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from ..core import timer_rules
from ..core.label_rules import clean_label
from ..io.oif import MI
from ..person_memory_manager import AGENT_SELF_ID

if TYPE_CHECKING:
    from ..core.confirm_state import PendingConfirm

logger = logging.getLogger(__name__)

# 同時に動かせるのは**タイマー 1 本**（2026-09-18・フラグに関係ない規則）。以前は合わせて 5 本〔仮〕だった。
# 掛け直すなら止めてから。
MAX_ACTIVE = 1
RECENT_SEC = 180.0  # 鳴った後も枠に残す秒数〔仮〕（「止めて」に「もう止まっている」と答える）

_DEFS: list[dict] = [
    {
        "name": "set_timer",
        "description": (
            "タイマーを掛ける（何分後に鳴る・「3分測って」は after_minutes=3）。何時に、ならアラーム（set_alarm）。"
            "label には何のためかを短く。返りに id と鳴る時刻。同時に 1 本。"
            "返りが「確かめて」なら、まだ掛かっていない——その文を相手に伝えて一度だけ聞く。"
            "掛け直しは要らない（「いい」と言われたら confirm が掛ける）。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "after_minutes": {"type": "number", "description": "今から何分後（0 より大きい）"},
                "label": {"type": "string", "description": "何のため（例「パスタ」「お茶」）"},
            },
            "required": ["after_minutes", "label"],
        },
    },
    {
        "name": "cancel_timer",
        "description": (
            "動いているタイマーを止める（「タイマー止めて」「やっぱりいい」）。ストップウォッチは stop_stopwatch。"
            'id は [タイマー] の枠にある番号。全部止めるなら "all"。'
        ),
        "input_schema": {
            "type": "object",
            "properties": {"id": {"description": '番号か "all"'}},
            "required": ["id"],
        },
    },
    # 一時停止・再開（知-o 段 4・2026-09-18）。何度でも。止めている間は残りが動かず鳴らない。
    {
        "name": "pause_timer",
        "description": "動いているタイマーを一時停止する（「一時停止」「ちょっと止めといて」）。ストップウォッチは対象外。",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"description": '番号か "all"'}},
            "required": ["id"],
        },
    },
    {
        "name": "resume_timer",
        "description": "一時停止中のタイマーを再開する（「再開」「続けて」）。止めていた分だけ鳴る時刻が遅れる。",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"description": '番号か "all"'}},
            "required": ["id"],
        },
    },
]


@dataclass(frozen=True)
class TimerFlags:
    """タイマーの振る舞い 3 つ（`設計方針_タイマー` v0.3）。"""

    silence: bool  # 掛けているあいだ黙る
    mic_close: bool  # 掛けているあいだ聞かない（操作の言葉だけ通す）
    confirm: bool  # 掛ける前に一度確かめる


class TimerTool:
    def __init__(
        self,
        *,
        store: Callable[[], Any],
        oif: Any,
        speaker: Callable[[], str],
        quiet: Callable[[], Any],
        silence_active: Callable[[], bool],
        now: "Callable[[], datetime] | None" = None,
        hush: "Callable[[str, datetime, int], None] | None" = None,
        unhush: "Callable[[int | str], None] | None" = None,
        on_cancel: "Callable[[], None] | None" = None,
        ask: "Callable[[PendingConfirm], None] | None" = None,
    ) -> None:
        self._store = store
        self._ask = (
            ask  # 「確かめて」の預かりを置く口（`agent.ask_confirm`）。無ければ確認文だけ返す
        )
        self._on_cancel = (
            on_cancel  # 止める頼みで音を止める（鳴った後は active に無いので先に呼ぶ）
        )
        self._oif = oif
        self._speaker = speaker
        self._quiet = quiet
        self._silence_active = silence_active
        self._now = now or (lambda: datetime.now(timezone.utc).astimezone())
        # 掛けているあいだ黙る（`TIMER_SILENCE`・2026-09-16）。`hush(誰, 鳴る時刻, id)`／`unhush(id|"all")`。
        self._hush = hush
        self._unhush = unhush

    def flags(self) -> "TimerFlags":
        """振る舞いの設定 3 つを**呼ぶたびに** `.env`（`os.environ`）から読む。

        設定画面で変えて保存した瞬間に効かせるため（作り直さない）。`AgentConfig()` を作らないのは、
        層 3 の登録値を DB から引きに行くから（道具の 1 呼び出しごとに DB を叩かない）。
        """
        from ..config import _bool_env

        return TimerFlags(
            silence=_bool_env("TIMER_SILENCE", default=True),
            mic_close=_bool_env("TIMER_MIC_CLOSE", default=True),
            confirm=_bool_env("TIMER_CONFIRM", default=True),
        )

    def store(self):
        """器（`TimerStore`）。T が鳴らすときに使う。"""
        return self._store()

    def get_tool_definitions(self) -> list[dict]:
        return [dict(d) for d in _DEFS]

    def frame(self) -> str:
        """`[タイマー]` の枠。無ければ空。"""
        now = self._now()
        store = self._store()
        return timer_rules.render_frame(
            store.active(now=now),
            store.recently_fired(now=now, within_sec=RECENT_SEC),
            now=now,
            recently_stopped=store.recently_stopped(now=now, within_sec=RECENT_SEC),
            not_listening=self.listening_closed(now=now),
        )

    def listening_closed(self, *, now: "datetime | None" = None) -> str:
        """聞かない状態か（`TIMER_MIC_CLOSE`・段 5）。閉じていればその理由（タイマーの名）、聞くなら空。

        **動いているタイマーから導く**（新しい状態は持たない・再起動をまたぐ）：due があり・
        一時停止でなく・`listen`（`/mic on`）が立っていないものが 1 本でもあれば閉じる。
        """
        if not self.flags().mic_close:
            return ""
        now = now or self._now()
        for r in self._store().active(now=now):
            if r.get("paused_at") is not None or r.get("listen"):
                continue
            return f"タイマー「{r['label']}」"
        return ""

    async def _listen(self) -> tuple[str, bool]:
        """`/mic on`：動いているタイマーはそのままに、聞く状態へ戻す（そのタイマー限り）。"""
        now = self._now()
        store = self._store()
        rows = [r for r in store.active(now=now) if r.get("due") is not None]
        if not rows:
            return "動いているタイマーは無い（聞いている）", True
        for r in rows:
            store.set_listen(int(r["id"]), True)
        logger.info("タイマー中でも聞く：%s", "・".join(str(r["id"]) for r in rows))
        return "聞く：" + "・".join(
            f"id={r['id']} 「{r['label']}」（鳴るまで）" for r in rows
        ), True

    async def call(
        self,
        name: str,
        tool_input: dict,
        *,
        now: "datetime | None" = None,
        confirmed: bool = False,
    ) -> tuple[str, bool]:
        """(文面, 使えたか)。`now` は起点（人が言った瞬間・無ければいま）。

        `confirmed` は**機械だけが立てる**（`agent.resolve_confirm`・「いい」と言われた預かりの呼び直し）。
        LLM の `tool_input` に同じ名前があっても読まない。
        """
        try:
            if name == "set_timer":
                return await self._set(tool_input, now=now, confirmed=confirmed)
            if name == "cancel_timer":
                return await self._cancel(tool_input)
            if name in ("pause_timer", "resume_timer"):
                return await self._pause_or_resume(name, tool_input)
            if name == "listen":  # 道具でなく命令（`/mic on`）からだけ
                return await self._listen()
        except Exception as e:  # noqa: BLE001
            logger.exception("タイマーの道具に失敗: %s", e)
            return f"タイマーの道具が使えなかった：{e}", False
        return f"そんな道具は無い：{name}", False

    def _keep(self, inp: dict, ask: str, what: str) -> None:
        """「確かめて」の預かりを置く（`confirm_state.PendingConfirm`）。口が無ければ何もしない。"""
        if self._ask is None:
            return
        from ..core.confirm_state import PendingConfirm

        kept = {k: v for k, v in inp.items() if k != "confirmed"}  # LLM が書いた印は預からない
        self._ask(
            PendingConfirm(
                action="set_timer", tool_input=kept, asked_at=time.time(), text=ask, what=what
            )
        )

    async def _set(
        self, inp: dict, *, now: "datetime | None" = None, confirmed: bool = False
    ) -> tuple[str, bool]:
        label = clean_label(inp.get("label"), "タイマー")  # 名前の検め（知-u）
        now = (now or self._now()).astimezone()
        try:
            if inp.get("at"):
                return "何時に、はアラーム（set_alarm）。タイマーは何分後だけ", False
            due = timer_rules.resolve_due(
                after_minutes=(
                    float(inp["after_minutes"]) if inp.get("after_minutes") is not None else None
                ),
                now=now,
            )
        except (ValueError, TypeError) as e:
            return f"時刻を読めない：{e}", False
        store = self._store()
        running = [r for r in store.active(now=now) if r.get("due") is not None]
        if running:
            # 同時に 1 本。確認より先に見る（確認だけして掛からない、を避ける）。
            r = running[0]
            return (
                f"いま「{r['label']}」（id={r['id']}{timer_rules.measure_at(r, now)}）が動いている。"
                "止めるか、鳴るのを待ってから",
                False,
            )
        flags = self.flags()
        reason = timer_rules.needs_confirmation(
            due, quiet=self._quiet(), silence_active=self._silence_active()
        )
        minutes = (due - now).total_seconds() / 60.0
        what = f"タイマーを掛ける「{label}」（{minutes:g} 分）"
        if not reason and flags.confirm and not confirmed:
            # 掛ける前に一度確かめる（`TIMER_CONFIRM`・既定 true）。静穏時間・沈黙中の確認は下で（常に）。
            ask = timer_rules.confirm_text(
                minutes, silence=flags.silence, mic_close=flags.mic_close
            )
            self._keep(inp, ask, what)
            return f"まだ掛けていない。本人に一度聞く：「{ask}」", True
        if reason and not confirmed:
            ask = f"{reason}。{due:%H:%M} に「{label}」で鳴らしていい？"
            self._keep(inp, ask, what)
            return f"まだ掛けていない。確かめてから：{reason}——本人に一度聞く：「{ask}」", True
        who = self._speaker() or ""
        obs_id = await self._write(
            f"{who or '誰か'}に頼まれて、{due:%H:%M} に「{label}」のタイマーを掛けた"
        )
        tid = store.add(
            label=label, due=due, asked_by=who, obs_id=obs_id, passes_quiet=bool(reason), now=now
        )
        logger.info(
            "タイマーを掛けた id=%d %s due=%s 通り抜け=%s",
            tid,
            label,
            due.isoformat(),
            bool(reason),
        )
        hushed = ""
        if self.flags().silence and self._hush is not None and who:
            # 掛けた瞬間から鳴るまで黙る。鳴る時刻＝期限なので、鳴る知らせは何もしなくても通る。
            self._hush(who, due, tid)
            hushed = "。鳴るまで黙っている"
        return f"掛けた：id={tid} 「{label}」 {due:%H:%M} に鳴る" + (
            "（静かな時間でも鳴らす）" if reason else ""
        ) + hushed, True

    async def _cancel(self, inp: dict) -> tuple[str, bool]:
        if self._on_cancel is not None:
            self._on_cancel()
        now = self._now()
        store = self._store()
        target = inp.get("id")
        if str(target).strip().lower() == "all":
            rows = store.active(now=now)
            if not rows:
                return "動いているタイマーは無い", True
            n = store.cancel_all(now=now)
            if self._unhush is not None:
                self._unhush("all")
            await self._write("やめた：" + "・".join(f"「{r['label']}」" for r in rows))
            logger.info("タイマーを全部止めた（%d 本）", n)
            return f"{n} 本止めた：" + "・".join(f"「{r['label']}」" for r in rows), True
        try:
            tid = int(str(target))
        except (TypeError, ValueError):
            return f"id を読めない：{target}", False
        row = next((r for r in store.active(now=now) if int(r["id"]) == tid), None)
        if row is None or not store.cancel(tid, now=now):
            return f"id={tid} のタイマーは動いていない", False
        if self._unhush is not None:
            self._unhush(tid)
        measured = timer_rules.measure_at(row, now)  # 止めた瞬間の残り
        await self._write(f"やめた：「{row['label']}」{measured}")
        logger.info("タイマーを止めた id=%d %s %s", tid, row["label"], measured)
        return f"止めた：id={tid} 「{row['label']}」{measured}", True

    async def _pause_or_resume(self, name: str, inp: dict) -> tuple[str, bool]:
        """一時停止／再開（due つきのタイマーだけ・`all` は動いている 1 本）。"""
        now = self._now()
        store = self._store()
        pausing = name == "pause_timer"
        rows = store.active(now=now)
        target = inp.get("id")
        if str(target).strip().lower() == "all":
            pick = rows
        else:
            try:
                tid = int(str(target))
            except (TypeError, ValueError):
                return f"id を読めない：{target}", False
            pick = [r for r in rows if int(r["id"]) == tid]
            if not pick:
                return f"id={tid} のタイマーは動いていない", False
        if not pick:
            return "動いているタイマーは無い", True
        done: list[str] = []
        for r in pick:
            ok = (
                store.pause(int(r["id"]), now=now)
                if pausing
                else store.resume(int(r["id"]), now=now)
            )
            if not ok:
                state = "もう止めている" if pausing else "止めていない"
                return f"id={r['id']} 「{r['label']}」は{state}", False
            fresh = next((x for x in store.active(now=now) if int(x["id"]) == int(r["id"])), r)
            done.append(f"id={r['id']} 「{r['label']}」{timer_rules.measure_at(fresh, now)}")
            logger.info(
                "タイマーを%s id=%s %s", "一時停止" if pausing else "再開", r["id"], r["label"]
            )
        await self._write(("止めておく：" if pausing else "再開した：") + "・".join(done))
        return ("止めておく：" if pausing else "再開した：") + "・".join(done), True

    async def _write(self, content: str) -> "str | None":
        try:
            return await self._oif.write(
                MI(id="", content=content[:500], timestamp=None, direction="予定"),
                writer_id=AGENT_SELF_ID,  # 書き手は自分（`OIF.write` の必須の材料・実機で落ちた）
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("予定の記録を書けなかった: %s", e)
            return None
