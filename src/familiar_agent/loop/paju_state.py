"""パジュの状態図（環-aw 第 1 段・`設計方針_パジュの状態` v0.3）。いまは頭と耳の 2 つの領域だけを持つ。

- 頭：何もしていない（`idle`）、考えている（`thinking`）、答えを言っている（`answering`）。考えているは子を 5 つ持つ
  （思い出している `recalling`、決めている `deciding`、返りを待っている `awaiting`、確かめている `checking`、
  書き留めている `noting`・決定チ）。答えを言っているは考えているの兄弟（決定ツ）。
- 耳：聞いている（`hearing`）、考えているあいだ聞かない（`deaf`）。

**耳は事象で動かす。頭に入るときの動作には付けない。** 答えのあと、頭は書き留めのために考えているへ戻るが、
そのとき耳は聞いたままにする（決定モ）。考えているに入るときの動作で耳を閉じると、ここでまた聞かなくなる。

状態を変えるのは事象だけで、送る口は `fire` の 1 つにする。移りの無い事象は捨て、事象の名前と各領域の状態を
1 行残す（決定ユ）。言葉の中身は書かない。

考えているの合計には上限がある（決定ミ）。考えているに入ると数え始め、答えを言い始めたら、その求めではもう数えない
（環-aw 第 1 段 項目 8・2026-11-10 本人の決定：答えたあとの書き留めは 3 分を過ぎてもよい）。何もしていないに入ると
やめ、次の求めは初めから数える。上限に届いたら `limit_reached` を送り、ループへ知らせる（`on_limit`）。残り時間は
この図の中だけで持つ。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Callable

from statemachine import State, StateChart

from ..core.aio import spawn_outside
from statemachine.exceptions import TransitionNotAllowed

logger = logging.getLogger(__name__)

#: 考えているの子（決定チ）。
CHILDREN = ("recalling", "deciding", "awaiting", "checking", "noting")

_NAMES = {
    "idle": "何もしていない",
    "thinking": "考えている",
    "recalling": "思い出している",
    "deciding": "決めている",
    "awaiting": "返りを待っている",
    "checking": "確かめている",
    "noting": "書き留めている",
    "answering": "答えを言っている",
    "hearing": "聞いている",
    "deaf": "聞かない",
}


class PajuState(StateChart):
    """パジュの状態図。`start` を待ってから使う（作っただけでは状態が空で、そのあいだは聞いているとみなす）。"""

    allow_event_without_transition = False

    def __init__(
        self,
        on_child: "Callable[[str], None] | None" = None,
        on_idle: "Callable[[], None] | None" = None,
        thinking_limit_sec: "float | None" = None,
        on_limit: "Callable[[], None] | None" = None,
    ) -> None:
        #: 考えているの子に入ったときに呼ぶ口（子の名前を渡す）。ループが電子音を鳴らす（決定テ）。
        self._on_child = on_child
        #: 頭が何もしていないに入ったときに呼ぶ口。ループが電子音の仕事を片付ける（求めの閉じ方によらない）。
        self._on_idle = on_idle
        #: 考えているの合計の上限（秒・決定ミ）。無ければ数えない。
        self._limit_sec = thinking_limit_sec
        #: 上限に届いたときに呼ぶ口。ループが時間切れの答えを始める（項目 8）。数える仕事からだけ呼ぶ——電子音を
        #: 鳴らせないときに送る `limit_reached` では呼ばない（そのときは求めを閉じず、耳を戻すだけ・本人の決定）。
        self._on_limit = on_limit
        #: この求めで考えているに使える残り（秒）。この求めでもう届いた、または答え始めたら None。
        self._limit_left: "float | None" = thinking_limit_sec
        #: 数えている仕事と、数え始めた時刻（`time.monotonic`）。
        self._limit_task: "asyncio.Task[None] | None" = None
        self._limit_since = 0.0
        super().__init__()

    class paju(State.Parallel):
        class head(State.Compound):
            idle = State(initial=True)

            class thinking(State.Compound):
                recalling = State(initial=True)
                deciding = State()
                awaiting = State()
                checking = State()
                noting = State()

            answering = State()

            request_opened = idle.to(thinking)
            # 子へは親の中で移る（`internal`）。親を出入りしないので、考えているに入るときの動作は繰り返さない。
            recall_began = thinking.to(thinking.recalling, internal=True)
            decide_began = thinking.to(thinking.deciding, internal=True)
            wait_began = thinking.to(thinking.awaiting, internal=True)
            check_began = thinking.to(thinking.checking, internal=True)
            note_began = thinking.to(thinking.noting, internal=True)
            speech_began = thinking.to(answering)
            speech_ended = answering.to(thinking.noting)
            request_closed = thinking.to(idle) | answering.to(idle)

        class ear(State.Compound):
            hearing = State(initial=True)
            deaf = State()

            request_opened = hearing.to(deaf)
            # 言っているあいだは、いつも聞かない（決定モ）。途中の確認の問いのあとで聞いていても、次の答えでまた閉じる。
            speech_began = hearing.to(deaf)
            speech_ended = deaf.to(hearing)
            request_closed = deaf.to(hearing)
            limit_reached = deaf.to(hearing)

    async def after_transition(self, event: str) -> None:
        # 非同期の動作を 1 つ持つことで、ライブラリが非同期の担い手を選ぶ（無いと同期で動き、`start` を待てない）。
        logger.debug("状態図 %s で移った（%s）", event, self.describe())

    def on_enter_state(self, state: State) -> None:
        # 考えているの子に入るときの動作と、何もしていないに入るときの動作。子へは事象で移るので、鳴らし始めも片付けも
        # 事象に付いて起きる（決定テ）。
        # ライブラリは動作の例外を内部の事象に変えて DEBUG にしか残さないので、ここで捕まえて残す（移りは止めない）。
        try:
            self._count_limit(state.id)
        except Exception:  # noqa: BLE001
            logger.warning(
                "状態図 %s で考えているの上限を数えられなかった", state.id, exc_info=True
            )
        try:
            if self._on_child is not None and state.id in CHILDREN:
                self._on_child(state.id)
            elif self._on_idle is not None and state.id == "idle":
                self._on_idle()
        except Exception:  # noqa: BLE001
            logger.warning("状態図 %s に入るときの動作が失敗した", state.id, exc_info=True)

    def _count_limit(self, state_id: str) -> None:
        """考えているの合計を数える（決定ミ）。考えているに入ると数え、答えを言い始めたらその求めでは終え、何もしていないで
        初めに戻す。"""
        if self._limit_sec is None:
            return
        if state_id == "thinking":
            not_counting = (
                self._limit_task is None or self._limit_task.done()
            )  # 外で止められた仕事も数えない
            if self._limit_left is not None and not_counting:
                with contextlib.suppress(RuntimeError):  # 走っている loop が無ければ数えない
                    # 入るときの動作は処理の段の中なので、外の仕事として立てる（そのままでは送る事象が列に残る）
                    self._limit_task = spawn_outside(
                        self._wait_limit(self._limit_left, self._limit_sec)
                    )
                    self._limit_since = time.monotonic()
        elif state_id == "answering":
            self.stop_counting()
            self._limit_left = None  # 答えたあとは、その求めでもう数えない（項目 8）
        elif state_id == "idle":
            self.stop_counting()
            self._limit_left = self._limit_sec

    def stop_counting(self) -> None:
        """数えるのを止め、使った時間を残りから引く。残りは 0 で止める（届いた刻みに止めても、戻ったときに送る）。"""
        task, self._limit_task = self._limit_task, None
        if task is None:
            return
        task.cancel()
        if self._limit_left is not None:
            self._limit_left = max(self._limit_left - (time.monotonic() - self._limit_since), 0.0)

    def end_counting(self) -> None:
        """もう数えない（ループを閉じるとき）。このあと考えているに入っても数え始めない。"""
        self.stop_counting()
        self._limit_sec = None

    async def _wait_limit(self, seconds: float, limit: float) -> None:
        # 送り終えるまで `_limit_task` に残す（閉じるときに、送っている途中でも止められる）。
        try:
            await asyncio.sleep(seconds)
            self._limit_left = None  # この求めでは、もう送らない（何もしていないに入ると戻す）
            if not self.is_deaf():
                # 電子音を鳴らせず、耳はもう聞いている。移りの無い事象を送ると、ライブラリはそのとき列に待っていた
                # ほかの事象も退けて列を空にするので、送らない。
                logger.info("状態図 考えているが上限 %.1f 秒に届いたが、耳は聞いている", limit)
            else:
                logger.info("状態図 考えているが上限 %.1f 秒に届いた", limit)
                await self.fire("limit_reached")
            if self._on_limit is not None:
                self._on_limit()  # 考えすぎたことは、耳によらず知らせる
        except Exception:  # noqa: BLE001
            logger.warning("状態図 考えているの上限を送れなかった", exc_info=True)
        finally:
            if self._limit_task is asyncio.current_task():
                self._limit_task = None

    async def start(self) -> None:
        """初めの状態に入る。入っていれば何もしない。"""
        if not self.configuration:
            await self.activate_initial_state()

    async def fire(self, event: str) -> None:
        """事象を送る、ただ 1 つの口。移りが無ければ捨て、1 行残す（決定ユ）。"""
        try:
            await self.send(event)
        except TransitionNotAllowed as e:
            # 同じ処理の中で別の送り手の事象が捨てられると、ライブラリは列に待っていた送り手みなに同じ例外を返す。
            # そのときは、送った事象と捨てられた事象の両方を残す（どちらか一方では取り違える）。
            dropped = getattr(e.event, "id", None) or event
            if dropped == event:
                logger.info("状態図 %s を受ける移りが無いので捨てた（%s）", event, self.describe())
            else:
                logger.info(
                    "状態図 %s を送ったが、%s を受ける移りが無く捨てられた（%s）",
                    event,
                    dropped,
                    self.describe(),
                )

    def is_deaf(self) -> bool:
        """耳が、考えているあいだ聞かないにいるか。活性化の前は聞いているとみなす。"""
        return any(s.id == "deaf" for s in self.configuration)

    def head_state(self) -> str:
        """頭の今の状態（`idle`・`thinking`・`answering`）。活性化の前は `idle` とみなす。"""
        ids = {s.id for s in self.configuration}
        return next((n for n in ("thinking", "answering") if n in ids), "idle")

    def thinking_child(self) -> str:
        """考えているの今の子の名前。考えていなければ空。"""
        ids = {s.id for s in self.configuration}
        return next((n for n in CHILDREN if n in ids), "")

    def describe(self) -> str:
        """各領域の今の状態を 1 行で（ログ用）。"""
        ids = {s.id for s in self.configuration}
        head = next((n for n in ("idle", "thinking", "answering") if n in ids), "")
        if head == "thinking":
            child = self.thinking_child()
            head_text = f"{_NAMES[head]}/{_NAMES[child]}" if child else _NAMES[head]
        else:
            head_text = _NAMES.get(head, "")
        ear = next((n for n in ("hearing", "deaf") if n in ids), "")
        return f"頭={head_text} 耳={_NAMES.get(ear, '')}"
