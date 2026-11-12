"""パジュの状態図（環-aw 第 1 段・第 2 段・`設計方針_パジュの状態` v0.5）。頭、耳、口の 3 つの領域を持つ。

- 頭：何もしていない（`idle`）、考えている（`thinking`）、答えを言っている（`answering`）。考えているは子を 5 つ持つ
  （思い出している `recalling`、決めている `deciding`、返りを待っている `awaiting`、確かめている `checking`、
  書き留めている `noting`・決定チ）。答えを言っているは考えているの兄弟（決定ツ）。
- 耳：並行の 2 つの領域を持つ。考えているあいだ（`listening`）は、聞いている（`hearing`）、考えているあいだ聞かない
  （`deaf`）。窓（`window`）は、閉じている（`shut`）、開いている（`open`）、仮の呼びかけ（`tentative`）（第 2 段）。
- 口：いまは黙るか（`hush`）の 1 つの領域だけを持つ。話してよい（`may_speak`）、黙っている（`silent`）（第 2 段。
  黙っているあいだも耳は聞いていて、答えを声にしないだけなので、口に置く）。

同じ状態へ戻る移りは、外へ出て入り直す形（`x.to.itself()`）にする。`internal=True` の形では、並行の領域の中で
ほかの領域の状態が二重になった（python-statemachine 3.2.2）。

窓、仮の呼びかけ、黙るかは、状態とデータ（`wake`、`tentative_at`、`silence`）の両方を持つ。門はデータで決め、状態は
移りとログのために持つ。データを変えるのはこの図の口（`open_window`、`call_tentatively`、`hush_with` など）だけ。時間で
切れる移り（窓、仮の呼びかけ、黙るの期限）は、この図の中の数える仕事が、状態を見てから送る（決定ヰ）。黙る依頼の
持ち主はこの図で、DB の `agent_state` の行は再起動で戻すための控え（入ると書き、出ると消し、`start` で戻す・決定ヱ）。

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
from typing import TYPE_CHECKING, Callable

from statemachine import State, StateChart

from ..core.aio import spawn_outside
from ..core.wake_window import WAKE_WINDOW_SEC, WakeWindow
from statemachine.exceptions import TransitionNotAllowed

if TYPE_CHECKING:
    from ..silence_state import SilenceRequest

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
    "shut": "閉じている",
    "open": "開いている",
    "tentative": "仮の呼びかけ",
    "may_speak": "話してよい",
    "silent": "黙っている",
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
        on_silence_saved: "Callable[[SilenceRequest], None] | None" = None,
        on_silence_cleared: "Callable[[], None] | None" = None,
        restore_silence: "Callable[[], SilenceRequest | None] | None" = None,
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
        #: 窓のデータ（第 2 段）。窓の中だったかは、届いた時刻（話し始め）でこのデータから決める（決定 2）。
        #: 状態（閉じている、開いている）は、移りとログのために持ち、時間で切れるのは数える仕事が送る。
        self.wake = WakeWindow()
        #: 窓の時間を数える仕事。話しているあいだ（`speaking` が 1 以上）は立てない。
        self._window_task: "asyncio.Task[None] | None" = None
        #: 数える仕事が `window_lapsed` を送っているあいだ真。そのあいだは仕事を止めない（送り終えてから見直す）。
        self._window_sending = False
        #: 仮の呼びかけの届いた時刻（`time.monotonic`・知-bs）。名前だけで確かさの足りなかった声。続きかは届いた時刻から
        #: 10 秒（`WAKE_WINDOW_SEC`）のうちかで決めるので、時間で閉じているへ戻っても、窓が開いても残す。捨てると None。
        self.tentative_at: "float | None" = None
        #: 仮の呼びかけの 10 秒を数える仕事と、それが `tentative_lapsed` を送っているあいだ真の印（窓と同じ作り）。
        self._tentative_task: "asyncio.Task[None] | None" = None
        self._tentative_sending = False
        #: 黙っている依頼（第 2 段 項目 4）。持ち主はこの図で、DB の `agent_state` の行は控え（再起動で戻すため）。
        #: 黙っているに入ると `on_silence_saved` で書き、出ると `on_silence_cleared` で消す。`start` のとき
        #: `restore_silence` で読み、生きていれば残りの期限で黙り、過ぎていれば消す。
        self.silence: "SilenceRequest | None" = None
        self._on_silence_saved = on_silence_saved
        self._on_silence_cleared = on_silence_cleared
        self._restore_silence = restore_silence
        #: 黙るの期限を数える仕事と、それが `silence_ended` を送っているあいだ真の印（窓と同じ作り）。
        self._silence_task: "asyncio.Task[None] | None" = None
        self._silence_sending = False
        #: 初めの状態に入り、控えを戻し終えたら済む。同時に待つ読み手も、戻し終えるまで待つ。
        self._starting: "asyncio.Future[None] | None" = None
        #: ループを閉じたら真。もう時間を数える仕事を立てない。
        self._timers_ended = False
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

        class ear(State.Parallel):
            class listening(State.Compound):
                hearing = State(initial=True)
                deaf = State()

                request_opened = hearing.to(deaf)
                # 言っているあいだは、いつも聞かない（決定モ）。途中の確認の問いのあとで聞いていても、次の答えでまた閉じる。
                speech_began = hearing.to(deaf)
                speech_ended = deaf.to(hearing)
                request_closed = deaf.to(hearing)
                limit_reached = deaf.to(hearing)

            class window(State.Compound):
                shut = State(initial=True)
                open = State()
                tentative = State()

                # 同じ状態へ戻る移りは、外へ出て入り直す形にする。`internal=True` の形では、並行の領域の中でほかの
                # 領域の状態が二重になった（python-statemachine 3.2.2）。
                window_opened = shut.to(open) | tentative.to(open) | open.to.itself()
                window_lapsed = open.to(shut)
                tentative_called = shut.to(tentative) | tentative.to.itself()
                tentative_lapsed = tentative.to(shut)
                tentative_dropped = tentative.to(shut)

        class mouth(State.Parallel):
            class hush(State.Compound):
                may_speak = State(initial=True)
                silent = State()

                silence_began = may_speak.to(silent) | silent.to.itself()
                silence_ended = silent.to(may_speak)

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

    async def open_window(self, at: float, *, resume: bool = False) -> None:
        """名前を聞いた、打たれた、返事を言い終えた。窓を `at` から開け（`WakeWindow.open`）、開いているへ移る。

        仮の呼びかけからも 1 回で開いているへ移る。届いた時刻は捨てない（2026-11-12 本人の決定。窓のあいだ門は窓で受けるので、
        預かりは使われない）。
        """
        self.wake.open(at, resume=resume)
        await self._to_open()

    async def extend_window(self, at: float) -> None:
        """窓の中の入力、返事、つなぎ。`at` に窓が開いていたときだけ延ばす（`WakeWindow.extend`）。

        状態がもう閉じていても、`at` がデータの上で窓の中なら延ばして開いているへ戻す（切れる直前に話し始めた声）。
        開いていたかは `WakeWindow.extend` が見る。閉じた窓は延びないので、`_to_open` も移さない。
        """
        self.wake.extend(at)
        await self._to_open()

    async def hold_window(self) -> None:
        """話し始め（窓が開いているときだけ呼ぶ）。言い終える（`release_window`）まで数えない。"""
        self.wake.hold()
        self._stop_window_task()

    async def release_window(self, at: float) -> None:
        """言い終えた。声がみな鳴り終わったら、そこから 10 秒を数える（`WakeWindow.release`）。"""
        self.wake.release(at)
        await self._to_open()

    def window_open_at(self, at: float) -> bool:
        """`at`（`time.monotonic` の秒）に窓が開いていたか。門はこの 1 つで決める（状態では決めない）。"""
        return self.wake.is_open(at)

    def end_timers(self) -> None:
        """時間を数える仕事をみな止める（ループを閉じるとき）。このあとは立てない。"""
        self._timers_ended = True
        self._stop_window_task()
        self._stop_tentative_task()
        self._stop_silence_task()

    async def hush_with(self, req: "SilenceRequest") -> None:
        """黙っている依頼を受けた（人の「黙って」、タイマー）。データを置き換え、黙っているへ移り、控えを書く。

        名前を `hush` にしないのは、口の領域 `hush` と重なり、ライブラリが許さないため。

        どの依頼にするか（タイマーは人の依頼を上書きしない、遅いタイマーは延ばす）は、呼び手が `silence_state` の
        純関数で決める。期限を過ぎた依頼では黙らない。
        """
        await self.start()
        if time.time() >= req.until:
            return
        await self._enter_silence(req)
        if self._on_silence_saved is not None:
            try:
                self._on_silence_saved(req)
            except Exception:  # noqa: BLE001
                logger.warning("状態図 黙っている依頼の控えを書けなかった", exc_info=True)

    async def unhush(self) -> None:
        """黙るのを解く（「話していいよ」、止めたタイマー）。黙っていなければ何も送らない。"""
        await self.start()
        if self.silence is None and not self._in("silent"):
            return
        # データが無くても状態が黙っているなら戻す（期限で解けたことを送れなかったとき）。
        self.silence = None
        self._stop_silence_task()
        self._clear_silence_backup()
        if self._silence_sending:
            return  # 期限で解けたことを送っている。送り終えれば話してよいにいる
        try:
            if self._in("silent"):
                await self.fire("silence_ended")
        except Exception:  # noqa: BLE001
            logger.warning("状態図 黙るのを解けなかった", exc_info=True)

    def silence_request(self) -> "SilenceRequest | None":
        """いまの黙っている依頼。門は、これを処理した時刻で `is_silenced` に掛けて決める。無ければ None。"""
        return self.silence

    def is_silent(self) -> bool:
        """口が黙っているにいるか。"""
        return self._in("silent")

    async def _enter_silence(self, req: "SilenceRequest") -> None:
        self.silence = req
        try:
            if not self._silence_sending:
                await self.fire("silence_began")
                self._arm_silence()
        except Exception:  # noqa: BLE001
            logger.warning("状態図 黙っているへ移せなかった", exc_info=True)

    def _clear_silence_backup(self) -> None:
        if self._on_silence_cleared is not None:
            try:
                self._on_silence_cleared()
            except Exception:  # noqa: BLE001
                logger.warning("状態図 黙っている依頼の控えを消せなかった", exc_info=True)

    def _stop_silence_task(self) -> None:
        if self._silence_sending:
            return  # 送っている仕事は止めない。送り終えてから、データを見て移し直す
        task, self._silence_task = self._silence_task, None
        if task is not None:
            task.cancel()

    def _arm_silence(self) -> None:
        """黙るの期限を数え直す（依頼が変わりうるので、生きている仕事も立て直す）。"""
        if self._silence_sending:
            return
        self._stop_silence_task()
        if self._timers_ended or self.silence is None or not self._in("silent"):
            return
        with contextlib.suppress(RuntimeError):  # 走っている loop が無ければ数えない
            self._silence_task = spawn_outside(self._wait_silence())

    async def _wait_silence(self) -> None:
        try:
            # 期限は壁の時計（`time.time`）の秒。依頼が置き換わったら、新しい期限まで待つ。
            while (req := self.silence) is not None and (left := req.until - time.time()) > 0:
                await asyncio.sleep(left)
            if req is None or not self._in("silent"):
                return  # 送る前に状態を見る（移りの無い事象を送らない）
            logger.info("状態図 黙る期限が来たので解いた")
            self.silence = None
            self._clear_silence_backup()
            self._silence_sending = True
            try:
                await self.fire("silence_ended")
            finally:
                self._silence_sending = False
            if not self._timers_ended and (nxt := self.silence) is not None:
                # 送っているあいだに新しい依頼が届いた。黙っているへ戻して数え直す。自分を止めないよう、先に手放す。
                self._silence_task = None
                await self._enter_silence(nxt)
        except Exception:  # noqa: BLE001
            logger.warning("状態図 黙る期限で解けたことを送れなかった", exc_info=True)
        finally:
            if self._silence_task is asyncio.current_task():
                self._silence_task = None

    async def call_tentatively(self, at: float) -> None:
        """名前だけで確かさの足りない声が `at` に届いた（知-bs）。届いた時刻を預け、仮の呼びかけへ移る。

        10 秒は届いた時刻から数える。もう 10 秒たった声では移らない（移ってすぐ戻らない）。届いた時刻は、いつも預ける
        （門は届いた時刻の窓で決めるので、状態が開いていても預ける）。状態が開いているときは移らず、窓が切れたら移る。
        """
        self.tentative_at = at
        await self._to_tentative()

    async def drop_tentative(self) -> None:
        """仮の呼びかけを捨てる（鳴っているあいだの音楽の言葉を窓を開けずに受けた）。いなければ何も送らない。"""
        self.tentative_at = None
        self._stop_tentative_task()
        if self._tentative_sending:
            return  # 切れたことを送っている。送り終えれば閉じているにいる（ここで送ると移りが無い）
        try:
            if self._in("tentative"):
                await self.fire("tentative_dropped")
        except Exception:  # noqa: BLE001
            logger.warning("状態図 仮の呼びかけを捨てられなかった", exc_info=True)

    def tentative_since(self) -> "float | None":
        """仮の呼びかけの届いた時刻。門はこの時刻で続きかを決める（状態では決めない）。無ければ None。"""
        return self.tentative_at

    def _tentative_live(self) -> bool:
        """データの上で、仮の呼びかけが届いてから 10 秒たっていないか（門の `_follows_tentative_call` と同じ下限と上限）。"""
        at = self.tentative_at
        return at is not None and 0 <= time.monotonic() - at < WAKE_WINDOW_SEC

    async def _to_tentative(self) -> None:
        """データの上で仮の呼びかけが生きていて、窓の状態が開いていなければ、仮の呼びかけへ移して数え直す。
        状態図が落ちても、会話は止めない（警告を 1 行残す）。"""
        try:
            if self._tentative_sending:
                return  # 送り終えてから、仕事がデータを見て移し直す
            if self._tentative_live() and not self._in("open"):
                await self.fire("tentative_called")
            self._arm_tentative()
        except Exception:  # noqa: BLE001
            logger.warning("状態図 仮の呼びかけへ移せなかった", exc_info=True)

    def _stop_tentative_task(self) -> None:
        if self._tentative_sending:
            return  # 送っている仕事は止めない。送り終えてから、データを見て移し直す
        task, self._tentative_task = self._tentative_task, None
        if task is not None:
            task.cancel()

    def _arm_tentative(self) -> None:
        """仮の呼びかけの 10 秒を数え直す（届いた時刻が変わりうるので、生きている仕事も立て直す）。"""
        if self._tentative_sending:
            return
        self._stop_tentative_task()
        if self._timers_ended or not self._in("tentative"):
            return
        with contextlib.suppress(RuntimeError):  # 走っている loop が無ければ数えない
            self._tentative_task = spawn_outside(self._wait_tentative())

    async def _wait_tentative(self) -> None:
        try:
            while (at := self.tentative_at) is not None and (
                left := at + WAKE_WINDOW_SEC - time.monotonic()
            ) > 0:
                await asyncio.sleep(left)
            # 送る前に状態を見る（移りの無い事象を送らない）。届いた時刻は残す。
            if self._in("tentative"):
                logger.debug("状態図 仮の呼びかけが切れた")
                self._tentative_sending = True
                try:
                    await self.fire("tentative_lapsed")
                finally:
                    self._tentative_sending = False
                if not self._timers_ended and self._tentative_live():
                    # 送っているあいだに新しい呼びかけが届いた。仮の呼びかけへ戻して数え直す。自分を止めないよう、先に手放す。
                    self._tentative_task = None
                    await self._to_tentative()
        except Exception:  # noqa: BLE001
            logger.warning("状態図 仮の呼びかけが切れたことを送れなかった", exc_info=True)
        finally:
            if self._tentative_task is asyncio.current_task():
                self._tentative_task = None

    def _in(self, state_id: str) -> bool:
        return any(s.id == state_id for s in self.configuration)

    async def _to_open(self) -> None:
        """いまの時刻でデータの上の窓が開いていれば、開いているへ移して数え直す。古い届いた時刻で開けても、もう
        切れている窓は開いているへ移さない。状態図が落ちても、会話は止めない（警告を 1 行残す）。"""
        try:
            if not self._in("open") and self.wake.is_open(time.monotonic()):
                await self.fire("window_opened")
                # 仮の呼びかけからも移るので、その 10 秒は数えない（届いた時刻は残す。窓が切れたとき生きていれば移し直す）
                self._stop_tentative_task()
            self._arm_window()
        except Exception:  # noqa: BLE001
            logger.warning("状態図 窓を開いているへ移せなかった", exc_info=True)

    def _stop_window_task(self) -> None:
        if self._window_sending:
            return  # 送っている仕事は止めない。送り終えてから、データを見て数え直す
        task, self._window_task = self._window_task, None
        if task is not None:
            task.cancel()

    def _arm_window(self) -> None:
        """窓の時間を数え直す。開いていないか、話しているあいだは数えない。"""
        if self._window_sending:
            return
        if self._timers_ended or self.wake.speaking > 0 or not self._in("open"):
            self._stop_window_task()
            return
        task = self._window_task
        if task is not None and not task.done():
            return  # 生きている仕事は、切れる時刻を読み直すので、そのまま使う（待つあいだは毎秒延ばす）
        with contextlib.suppress(RuntimeError):  # 走っている loop が無ければ数えない
            self._window_task = spawn_outside(self._wait_window())

    async def _wait_window(self) -> None:
        try:
            while (left := self.wake.until - time.monotonic()) > 0:
                await asyncio.sleep(left)
            # 送る前に状態を見る。もう閉じているのに送ると、移りが無く、ライブラリが列のほかの事象も退ける。
            if self.wake.speaking == 0 and self._in("open"):
                logger.debug("状態図 窓が切れた")
                self._window_sending = True
                try:
                    await self.fire("window_lapsed")
                finally:
                    self._window_sending = False
                if not self._timers_ended and self.wake.is_open(time.monotonic()):
                    # 送っているあいだに開け直された（`_to_open` は、まだ開いていると見て送らなかった）。開いているへ
                    # 戻して数え直す。自分を止めないよう、先に手放す。
                    self._window_task = None
                    await self._to_open()
                elif not self._timers_ended:
                    # 窓が開いているあいだに届いた仮の呼びかけが、まだ生きていれば移す（切れる間際に届いた声）。
                    await self._to_tentative()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning("状態図 窓が切れたことを送れなかった", exc_info=True)
        finally:
            if self._window_task is asyncio.current_task():
                self._window_task = None

    async def start(self) -> None:
        """初めの状態に入り、黙っている依頼の控えを戻す。済んでいれば何もしない。

        戻し終えるまでは、同時に待つ読み手（門、トニック）も待たせる（戻す前に読むと、黙っていないと読む）。
        """
        while (starting := self._starting) is not None:
            if starting.done():
                return
            # 戻している読み手を待つ。落ちたら `_starting` は None に戻るので、もう一度見て自分で戻し直す。
            await asyncio.shield(starting)
        self._starting = starting = asyncio.get_running_loop().create_future()
        try:
            # ほかの口が先に事象を送ると、ライブラリはひとりでに初めの状態に入る。そのときも控えは戻す。
            if not self.configuration:
                await self.activate_initial_state()
            await self._restore()
        except BaseException:
            self._starting = None  # 途中で落ちた、止められた。次の `start` でやり直す
            raise
        finally:
            starting.set_result(None)

    async def _restore(self) -> None:
        """再起動しなおし：控えに生きた依頼があれば残りの期限で黙る。過ぎていれば控えを消す。控えは書き直さない。"""
        if self._restore_silence is None:
            return
        from ..silence_state import is_silenced

        try:
            req = self._restore_silence()
        except Exception:  # noqa: BLE001
            logger.warning("状態図 黙っている依頼の控えを読めなかった", exc_info=True)
            return
        if req is None:
            return
        if is_silenced(req, now=time.time()):
            logger.info(
                "状態図 黙っている依頼を控えから戻した（残り %.0f 分）",
                (req.until - time.time()) / 60,
            )
            await self._enter_silence(req)
        else:
            self._clear_silence_backup()

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
        window = next((n for n in ("shut", "open", "tentative") if n in ids), "")
        mouth = next((n for n in ("may_speak", "silent") if n in ids), "")
        return (
            f"頭={head_text} 耳={_NAMES.get(ear, '')}/窓={_NAMES.get(window, '')}"
            f" 口={_NAMES.get(mouth, '')}"
        )
