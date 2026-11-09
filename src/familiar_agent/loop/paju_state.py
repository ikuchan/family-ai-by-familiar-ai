"""パジュの状態図（環-aw 第 1 段・`設計方針_パジュの状態` v0.3）。いまは頭と耳の 2 つの領域だけを持つ。

- 頭：何もしていない（`idle`）、考えている（`thinking`）、答えを言っている（`answering`）。考えているは子を 5 つ持つ
  （思い出している `recalling`、決めている `deciding`、返りを待っている `awaiting`、確かめている `checking`、
  書き留めている `noting`・決定チ）。答えを言っているは考えているの兄弟（決定ツ）。
- 耳：聞いている（`hearing`）、考えているあいだ聞かない（`deaf`）。

**耳は事象で動かす。頭に入るときの動作には付けない。** 答えのあと、頭は書き留めのために考えているへ戻るが、
そのとき耳は聞いたままにする（決定モ）。考えているに入るときの動作で耳を閉じると、ここでまた聞かなくなる。

状態を変えるのは事象だけで、送る口は `fire` の 1 つにする。移りの無い事象は捨て、事象の名前と各領域の状態を
1 行残す（決定ユ）。言葉の中身は書かない。
"""

from __future__ import annotations

import logging
from typing import Callable

from statemachine import State, StateChart
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
    ) -> None:
        #: 考えているの子に入ったときに呼ぶ口（子の名前を渡す）。ループが電子音を鳴らす（決定テ）。
        self._on_child = on_child
        #: 頭が何もしていないに入ったときに呼ぶ口。ループが電子音の仕事を片付ける（求めの閉じ方によらない）。
        self._on_idle = on_idle
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
            if self._on_child is not None and state.id in CHILDREN:
                self._on_child(state.id)
            elif self._on_idle is not None and state.id == "idle":
                self._on_idle()
        except Exception:  # noqa: BLE001
            logger.warning("状態図 %s に入るときの動作が失敗した", state.id, exc_info=True)

    async def start(self) -> None:
        """初めの状態に入る。入っていれば何もしない。"""
        if not self.configuration:
            await self.activate_initial_state()

    async def fire(self, event: str) -> None:
        """事象を送る、ただ 1 つの口。移りが無ければ捨て、1 行残す（決定ユ）。"""
        try:
            await self.send(event)
        except TransitionNotAllowed:
            logger.info("状態図 %s を受ける移りが無いので捨てた（%s）", event, self.describe())

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
