"""パジュの状態図（環-aw 第 1 段・第 2 段・`設計方針_パジュの状態` v0.7）。頭、耳、口の 3 つの領域を持つ。

- 頭：何もしていない（`idle`）、考えている（`thinking`）、答えを言っている（`answering`）。考えているは子を 5 つ持つ
  （思い出している `recalling`、決めている `deciding`、返りを待っている `awaiting`、確かめている `checking`、
  書き留めている `noting`・決定チ）。答えを言っているは考えているの兄弟（決定ツ）。
- 耳：並行の 3 つの領域を持つ。考えているあいだ（`listening`）は、聞いている（`hearing`）、考えているあいだ聞かない
  （`deaf`）。窓（`window`）は、閉じている（`shut`）、開いている（`open`）、仮の呼びかけ（`tentative`）（第 2 段）。
  待っている答え（`awaited`）は、答えなし（`no_answer`）、確認（`confirm_answer`）、申し出（`offer_answer`）、
  聞き返し（`heard_answer`）（第 3 段）。
- 口：並行の 2 つの領域を持つ。黙るか（`hush`）は、話してよい（`may_speak`）、黙っている（`silent`）（第 2 段。
  黙っているあいだも耳は聞いていて、答えを声にしないだけなので、口に置く）。声（`voice`）は、いま口から何が出ているか
  で、なし（`no_voice`）、考え中の音（`thinking_cue`・機械の音A）、合図（`cue`・効果音B、はいっ、えっと、黙る音）、
  話している（`speaking`・作った声も録音の声も、パジュの声のすべて）（第 5 段）。声の事象を送るのは dif だけで、
  声は dif の声の鍵で 1 本ずつ通す（状態は数を持たないので、重なった声を数えられない）。

同じ状態へ戻る移りは、外へ出て入り直す形（`x.to.itself()`）にする。`internal=True` の形では、並行の領域の中で
ほかの領域の状態が二重になった（python-statemachine 3.2.2）。

窓、仮の呼びかけ、黙るかは、状態とデータ（`wake`、`tentative_at`、`silence`）の両方を持つ。門はデータで決め、状態は
移りとログのために持つ。データを変えるのはこの図の口（`open_window`、`call_tentatively`、`hush_with` など）だけ。時間で
切れる移り（窓、仮の呼びかけ、黙るの期限）は、この図の中の数える仕事が、状態を見てから送る（決定ヰ）。黙る依頼の
持ち主はこの図で、DB の `agent_state` の行は再起動で戻すための控え（依頼を置くと書き、解くと消し、`start` で戻す・決定ヱ）。

待っている答えの持ち主もこの図で、問いの中身はデータ（`question`）に持つ（決定グ）。置く口はデータを待つ前に置き、
順番が来たら状態をデータに合わせる。なしへ戻す仕事は、自分の問いのデータがまだ置かれているときだけ消す。移りは
打ち切りから守って送る（決定ズ）。

話しかけない求め（seeking・safety の発火）は `quiet_request_opened` で開く。頭は考えているへ移るが、耳の領域には
この事象の移りが無いので、耳は聞いたまま（決定ザ・2026-11-15）。声にしない求めのあいだに届いた声を捨てないため。

**耳は事象で動かす。頭に入るときの動作には付けない。** 答えのあと、頭は書き留めのために考えているへ戻るが、
そのとき耳は聞いたままにする（決定モ）。考えているに入るときの動作で耳を閉じると、ここでまた聞かなくなる。

状態を変えるのは事象だけで、外から送る口は `fire` の 1 つにする。`fire` は順番を待って送る。順番の中からは `_fire` で送る（決定ン）。移りの無い事象は捨て、事象の名前と各領域の状態を
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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

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
    "sleeping": "寝ている",
    "hearing": "聞いている",
    "deaf": "聞かない",
    "shut": "閉じている",
    "open": "開いている",
    "tentative": "仮の呼びかけ",
    "may_speak": "話してよい",
    "silent": "黙っている",
    "no_voice": "なし",
    "thinking_cue": "考え中の音",
    "cue": "合図",
    "speaking": "話している",
    "no_answer": "なし",
    "confirm_answer": "確認",
    "offer_answer": "申し出",
    "heard_answer": "聞き返し",
    "ear_open": "開いている",
    "music_only": "音楽の言葉だけ通す",
    "timer_only": "タイマーの言葉だけ通す",
    "alarm_quiet": "止まっている",
    "alarm_ringing": "鳴っている",
    "song_stopped": "止まっている",
    "song_playing": "鳴っている",
    "song_ducked": "絞っている",
    "body_still": "止まっている",
    "neck_moving": "首を動かしている",
    "patrolling": "見回っている",
}

# 寝ているあいだに求めを開こうとした口が、起きたかを見直す間隔。眠ると起きるは自律機構の輪の上でしか変わらないので、
# 輪の間隔（`tonic.TONIC_PERIOD_SEC`）と同じにする（案 A′・2026-11-18 本人の決定）
WAKE_POLL_SEC = 0.5

#: 窓を保つときの上限（秒）。保っているあいだは窓の閉じる時刻をいまからこの先に置く。求めの閉じや声の終わりが届かなくても
#: （状態図へ送り損ねた）、窓を開けたままにしない控え（端-b 段 1-い）。考えているの上限（3 分）と長い答えより長くする。
HOLD_CAP_SEC = 600.0

#: 入ると窓を保つか見直す状態（頭と体）。保つかは `PajuState._request_holds` が決める。
_HOLD_OR_RELEASE = frozenset({"thinking", "answering", "noting", "idle", "patrolling"})

#: 口の声の状態（第 5 段）。
VOICE_STATES = ("no_voice", "thinking_cue", "cue", "speaking")

#: 体の状態（第 7 段）。
BODY_STATES = ("body_still", "neck_moving", "patrolling")

#: 口の歌の状態（第 6 段 項目 3）と、その状態へ入る事象（写しは、いまの状態から行き先へ入る事象を送る）。
SONG_STATES = ("song_stopped", "song_playing", "song_ducked")
_SONG_ENTERS = {
    ("song_stopped", "song_playing"): "song_began",
    ("song_ducked", "song_playing"): "song_restored",
    ("song_stopped", "song_ducked"): "song_quieted",
    ("song_playing", "song_ducked"): "song_quieted",
    ("song_playing", "song_stopped"): "song_ended",
    ("song_ducked", "song_stopped"): "song_ended",
}


def song_of(music: Any) -> str:
    """音楽の状態（`MusicState`）から、歌の状態を読む（持ち主は音楽の側・状態図はその写し）。

    鳴っているかは `core/music_state.is_playing` の 1 か所で決める。鳴っていて絞りの基準（`ducked_base`）があれば絞っている。
    家族が再生の機器で直接かけた曲は、印が立たないので歌に入らない（決定フ）。
    絞る側は音量を下げる前に基準を置き、下げられなければ消す（`music_watch.duck_while_speaking`）。そのあいだに輪が回ると、
    一瞬だけ絞っているに写る（受け入れる。印を増やさない）。寝ているあいだは輪が止まるので、起きてから写す。
    """
    from ..core.music_state import is_playing

    if not is_playing(music):
        return "song_stopped"
    return "song_ducked" if getattr(music, "ducked_base", None) is not None else "song_playing"


#: 耳の入口の状態の名前（決定ハ）。門（`event_loop._ear_drops`）と状態の行は、これで比べる。
EAR_OPEN = _NAMES["ear_open"]
EAR_MUSIC = _NAMES["music_only"]
EAR_TIMER = _NAMES["timer_only"]
#: 入口の状態ごとの、入る事象（決定ガ）。
_GATE_EVENTS = {"ear_open": "gate_opened", "music_only": "music_gated", "timer_only": "timer_gated"}

#: 待っている答えの種類ごとの、状態と入る事象（決定ノ）。
_ANSWER_STATES = {"confirm": "confirm_answer", "offer": "offer_answer", "heard": "heard_answer"}
_ANSWER_EVENTS = {"confirm": "confirm_asked", "offer": "offer_made", "heard": "heard_asked"}


@dataclass
class AwaitedAnswer:
    """待っている答えの問い 1 つ（決定グ）。データを変えるのは状態図の口だけ。

    中身は、確認なら預かり（`PendingConfirm`）、申し出なら申し出た文と深く調べる入力、聞き返しなら見立て。
    """

    kind: str  # "confirm"・"offer"・"heard"
    content: Any
    #: 置いた時刻（`time.monotonic`）。確認と聞き返しの寿命はここから数える（決定ゲ）。
    placed_at: float
    #: 申し出を預けたときの窓の番号。答えた求めの窓の番号と同じときだけ使う（決定ゲ）。
    session: int = 0
    #: 確認の決まった促しを最後に言った時刻（`time.time`・知-o）。新しい確認では空（決定グ）。
    prompted_at: "float | None" = None


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
        confirm_ttl_sec: float = 300.0,
        heard_ttl_sec: float = 30.0,
        on_speech_ended: "Callable[[bool], None] | None" = None,
    ) -> None:
        #: 考えているの子に入ったときに呼ぶ口（子の名前を渡す）。ループが電子音を鳴らす（決定テ）。
        self._on_child = on_child
        #: 頭が何もしていないに入ったときに呼ぶ口。ループが電子音の仕事を片付ける（求めの閉じ方によらない）。
        self._on_idle = on_idle
        #: 声の話しているを出たときに呼ぶ口（鳴ったかを渡す）。ループが自分の声の見張りの 1.2 秒を数えさせる（第 5 段）。
        self._on_speech_ended = on_speech_ended
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
        #: 窓の時間を数える仕事。窓を保っているあいだは上限（`HOLD_CAP_SEC`）まで数え、保つときが終わったら立て直す。
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
        #: 依頼を置くと `on_silence_saved` で書き、解くと `on_silence_cleared` で消す。`start` のとき
        #: `restore_silence` で読み、生きていれば残りの期限で黙り、過ぎていれば消す。
        self.silence: "SilenceRequest | None" = None
        self._on_silence_saved = on_silence_saved
        self._on_silence_cleared = on_silence_cleared
        self._restore_silence = restore_silence
        #: 黙るの期限を数える仕事と、それが `silence_ended` を送っているあいだ真の印（窓と同じ作り）。
        self._silence_task: "asyncio.Task[None] | None" = None
        self._silence_sending = False
        #: 待っている答えの問い（第 3 段・決定グ）。無ければ None。寿命は確認 `confirm_ttl_sec`、聞き返し
        #: `heard_ttl_sec`（決定ゲ）で、数える仕事が切る。申し出は秒数では切らない。
        self.question: "AwaitedAnswer | None" = None
        self._confirm_ttl = confirm_ttl_sec
        self._heard_ttl = heard_ttl_sec
        self._answer_task: "asyncio.Task[None] | None" = None
        #: 入口がタイマーの言葉だけ通すのとき、入口を閉じているタイマーの名前（例「タイマー「パスタ」」・決定ゾ）。ほかは空。
        #: 入口は写しで、持ち主はタイマーの行と音楽の印（決定ガ）。変えるのは写す口（`copy_gate`）だけ。
        self.gate_timer = ""
        #: 送る口の順番待ち（決定ン）。状態を見て送る口は、この中で見る、データを変える、送るを続けて行う。
        #: 錠は作ったループに縛られるので、走っているループが変わったら作り直す（試験の装置と画面は、状態図をループより
        #: 長く使う。`InformationProcessing._ensure_driver` と同じ）。
        self._turn_lock: "asyncio.Lock | None" = None
        self._turn_loop: "asyncio.AbstractEventLoop | None" = None
        #: 声の鍵（第 5 段）。dif が声（作った声、録音の声）を 1 本ずつ鳴らすための順番待ち。声の状態は数を持たないので、
        #: 2 つの声が重なると、先に終わった声の「話し終えた」で、まだ鳴っている声を「なし」にしてしまう。ループの dif と
        #: 自律機構の dif が同じ鍵を使うよう、ただ 1 つの状態図に置く。取る順番は、声の鍵 → `_turn` だけ。
        self._voice_lock: "asyncio.Lock | None" = None
        self._voice_loop: "asyncio.AbstractEventLoop | None" = None
        #: 初めの状態に入り、控えを戻し終えたら済む。同時に待つ読み手も、戻し終えるまで待つ。
        self._starting: "asyncio.Future[None] | None" = None
        #: ループを閉じたら真。もう時間を数える仕事を立てない。
        self._timers_ended = False
        #: 眠い（rest が発火したが、まだ眠っていない・決定ナの持ち越し）。頭が何もしていないになったら眠る。
        self.sleepy = False
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
            # 内省のあいだ（決定ダ、ク）。何もしていないからだけ入る（決定ナ）。求めを開く移りは持たない。
            sleeping = State()

            request_opened = idle.to(thinking)
            # 話しかけない求め（seeking・safety の発火）を開いた。耳の領域には移りを置かない（耳は聞いたまま・決定ザ）。
            quiet_request_opened = idle.to(thinking)
            # 子へは親の中で移る（`internal`）。親を出入りしないので、考えているに入るときの動作は繰り返さない。
            recall_began = thinking.to(thinking.recalling, internal=True)
            decide_began = thinking.to(thinking.deciding, internal=True)
            wait_began = thinking.to(thinking.awaiting, internal=True)
            check_began = thinking.to(thinking.checking, internal=True)
            note_began = thinking.to(thinking.noting, internal=True)
            speech_began = thinking.to(answering)
            speech_ended = answering.to(thinking.noting)
            request_closed = thinking.to(idle) | answering.to(idle)
            sleep_began = idle.to(sleeping)  # 眠くなった（自律機構の rest の発火・`fall_asleep`）
            woke = sleeping.to(idle)  # 起きた（内省の層を全部終えた。落ちても送る）

        class ear(State.Parallel):
            class listening(State.Compound):
                hearing = State(initial=True)
                deaf = State()

                # 禁止の条件：頭が寝ているあいだは、求めを開いても耳を移さない（頭にも移りが無いので、状態図が捨てる・決定エ、ダ）。
                # 求めを開く口（`open_request`）は寝ているあいだ送らないので、これは `fire` で直に送る口への備え
                request_opened = hearing.to(deaf, unless="is_sleeping")
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

            class awaited(State.Compound):
                no_answer = State(initial=True)
                confirm_answer = State()
                offer_answer = State()
                heard_answer = State()

                # 新しい問いは前の問いを上書きする（決定ノ）。同じ種類の上書きはデータだけ替え、状態は移さない。
                confirm_asked = (
                    no_answer.to(confirm_answer)
                    | offer_answer.to(confirm_answer)
                    | heard_answer.to(confirm_answer)
                )
                offer_made = (
                    no_answer.to(offer_answer)
                    | confirm_answer.to(offer_answer)
                    | heard_answer.to(offer_answer)
                )
                heard_asked = (
                    no_answer.to(heard_answer)
                    | confirm_answer.to(heard_answer)
                    | offer_answer.to(heard_answer)
                )
                answer_cleared = (
                    confirm_answer.to(no_answer)
                    | offer_answer.to(no_answer)
                    | heard_answer.to(no_answer)
                )

            class entrance(State.Compound):
                # 入口の写し（決定ガ）。持ち主を読んだ口が、違うときだけ送る。名前だけの違いはデータを写し直す（決定ゾ）。
                ear_open = State(initial=True)
                music_only = State()
                timer_only = State()

                gate_opened = music_only.to(ear_open) | timer_only.to(ear_open)
                music_gated = ear_open.to(music_only) | timer_only.to(music_only)
                timer_gated = ear_open.to(timer_only) | music_only.to(timer_only)

        class mouth(State.Parallel):
            class alarm(State.Compound):
                # 知らせの音（第 6 段・決定ニ）。タイマーとアラームの音。持ち主は dif の鳴らしている仕事で、これはその写し
                # （自律機構の輪が `copy_ring` で写す・入口と同じ形）。声とは再生の鍵で交互に鳴るが、別の領域なので、状態としては同時に成り立つ。
                alarm_quiet = State(initial=True)
                alarm_ringing = State()

                ring_began = alarm_quiet.to(alarm_ringing)
                ring_ended = alarm_ringing.to(alarm_quiet)

            class song(State.Compound):
                # 歌（第 6 段 項目 3・決定オ、ヌ、フ）。パジュがかけた音楽。持ち主は音楽の状態（鳴っている印と絞りの基準）で、
                # これはその写し（自律機構の輪が `copy_song` で写す・知らせの音と同じ形）。絞る・戻すは音楽の側が持ち主を読む。
                song_stopped = State(initial=True)
                song_playing = State()
                song_ducked = State()

                song_began = song_stopped.to(song_playing)
                song_quieted = song_playing.to(song_ducked) | song_stopped.to(song_ducked)
                song_restored = song_ducked.to(song_playing)
                song_ended = song_playing.to(song_stopped) | song_ducked.to(song_stopped)

            class hush(State.Compound):
                may_speak = State(initial=True)
                silent = State()

                silence_began = may_speak.to(silent) | silent.to.itself()
                silence_ended = silent.to(may_speak)

            class voice(State.Compound):
                # 口から何が出ているか（第 5 段）。考え中の音は機械音 A、合図は効果音 B・はいっ・えっと・黙る音、話しているは
                # パジュの声のすべて（作った声、録音の声）。送り元は dif だけで、声は声の鍵で 1 本ずつ通す（`voice_turn`）。
                no_voice = State(initial=True)
                thinking_cue = State()
                cue = State()
                speaking = State()

                # 合図の最中に次の求めの考え中の音が始まれば、後から始まった音の状態にする（古い合図の終わりは捨てる）
                thinking_cue_began = no_voice.to(thinking_cue) | cue.to(thinking_cue)
                thinking_cue_ended = thinking_cue.to(no_voice)
                # 合図 → 合図は外へ出て入り直す（はいっ のあとの えっと など、合図が続くとき）
                cue_began = no_voice.to(cue) | thinking_cue.to(cue) | cue.to.itself()
                cue_ended = cue.to(no_voice)
                # 黙る依頼が効いていて、求めの外の声（`through_silence` なし）なら移らない（環-bb・2026-11-21 本人の決定
                # 「黙っているあいだ話さない」を Lean で証明するため）。判定は `silenced_outside` の 1 か所で、依頼のデータを
                # 条件を見る時刻で読む。dif は鍵を待つ前にも同じ口で見て止め、話し始めたを断られたら鳴らさない。
                voice_began = (
                    no_voice.to(speaking, unless="silenced_outside")
                    | thinking_cue.to(speaking, unless="silenced_outside")
                    | cue.to(speaking, unless="silenced_outside")
                )
                voice_ended = speaking.to(no_voice)

        class body(State.Compound):
            # 体（第 7 段・決定イ、ウ）。首を動かしているは事象で動く（声と同じ形）：持ち主はカメラの `move_to` で、動かし始めと
            # 終わりに送る。見回っているは、話しかけない求め（seeking・safety の発火）が開いているあいだで、頭と同じ事象で動く。
            # 見回っているあいだは首の事象で移らない（自分へ戻る・図のとおり〔仮〕）。遅れて届いた首の終わりも捨てない。
            body_still = State(initial=True)
            neck_moving = State()
            patrolling = State()

            neck_began = (
                body_still.to(neck_moving) | neck_moving.to.itself() | patrolling.to.itself()
            )
            neck_ended = (
                neck_moving.to(body_still) | patrolling.to.itself() | body_still.to.itself()
            )
            # 頭が受けたときだけ（頭は何もしていないからだけ開く。頭が断った求めを、体だけが見回りにしない）
            quiet_request_opened = body_still.to(patrolling, cond="is_head_idle") | neck_moving.to(
                patrolling, cond="is_head_idle"
            )
            request_closed = patrolling.to(body_still)

    async def after_transition(self, event: str) -> None:
        # 非同期の動作を 1 つ持つことで、ライブラリが非同期の担い手を選ぶ（無いと同期で動き、`start` を待てない）。
        # 情報の段に残す（環-bb 二段目・あとから再生する）。入るのは事象と状態の名前だけ（`data` は出さない）。
        logger.info("状態図 %s で移った（%s）", event, self.describe())

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
            if state.id in _HOLD_OR_RELEASE:
                # 答えるまでの求めのあいだは窓を閉じない（端-b 段 1-い）。保つかは `_request_holds` の 1 か所で決め、
                # ここは見直すきっかけ（頭の考えている・答えている・書き留めている・何もしていない、体の見回っている）だけ持つ
                self._review_window()
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
            async with self._turn():  # 見てから送るまでに、ほかの口を割り込ませない（決定ン）
                self._limit_left = None  # この求めでは、もう送らない（何もしていないに入ると戻す）
                if not self.is_deaf():
                    # 耳はもう聞いている（電子音を鳴らせなかった、または話しかけない求め・決定ザ）。移りの無い事象は送らない（決定ユ）。
                    logger.info(
                        "状態図 考えているが上限 %.1f 秒に届いた（耳は聞いているので移りは送らない）",
                        limit,
                    )
                else:
                    logger.info("状態図 考えているが上限 %.1f 秒に届いた", limit)
                    await self._fire("limit_reached")
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
        self._hold_while_speaking()
        await self._to_open()

    async def extend_window(self, at: float) -> None:
        """窓の中の入力、返事、つなぎ。`at` に窓が開いていたときだけ延ばす（`WakeWindow.extend`）。

        状態がもう閉じていても、`at` がデータの上で窓の中なら延ばして開いているへ戻す（切れる直前に話し始めた声）。
        開いていたかは `WakeWindow.extend` が見る。閉じた窓は延びないので、`_to_open` も移さない。
        """
        self.wake.extend(at)
        self._hold_while_speaking()
        await self._to_open()

    # ── 声と窓（第 5 段）、求めと窓（端-b 段 1-い）───────────────────────────────────────
    # 窓が開いているあいだの声は、どれも（答え、録音の声）窓を保ち、話し終えてから 10 秒で閉じる（本人「話し終わって
    # から１０秒」）。保っているあいだは窓の閉じる時刻（`wake.until`）を上限（`HOLD_CAP_SEC`）の先に置く。数で保つと、門（`window_open_at`）と
    # 窓の会話の番号（`WakeWindow.open`）がどちらも `is_open` を読むので、そこから見えなくなる。
    # 求めが開いているあいだ（頭が考えている、答えている）も窓を保つ（2026-11-25 本人の決定・課題9 §7.4）。排他の仕組みを
    # 窓だけにするため（調べもので長く考えると、答えより前に窓が閉じ、その隙にほかの口が入れた）。保つかは状態から読み、
    # 印は持たない（`_holding`）。

    def _request_holds(self) -> bool:
        """求めが窓を保つときか。答えを言っているか、答えより前に考えているあいだ（書き留めているは除く）。

        話しかけない求め（見回り）は保たない。見分けは体の見回っている（`quiet_request_opened` で入る）で、耳では見ない
        （待ちの電子音を鳴らせないと、考え始めに耳が開く）。答えのあとの書き留め（`noting`）と、そのあとの道具のあいだも
        保たない（答えのあと 10 秒で閉じるのはいまどおり・課題9 §7.4）。
        """
        head = self.head_state()
        if head == "answering":
            return True
        return head == "thinking" and not self._in("noting") and not self._in("patrolling")

    def _holding(self) -> bool:
        """窓を保つときか。声が話しているか、求めが保つとき（`_request_holds`）。"""
        return self._in("speaking") or self._request_holds()

    def _held(self) -> bool:
        """いま窓を保っているか（閉じる時刻が、上限の半分より先にある）。印は持たず、閉じる時刻から読む。

        境目を「ふつうの窓の長さ」にすると、長く開けた窓（試験など）を保っていると取り違えて縮める。保っているあいだは
        閉じる時刻を上限の先に置き、保つのは考えているの上限（3 分）と答えのあいだなので、上限の半分を割らない。
        """
        return self.wake.until - time.monotonic() > HOLD_CAP_SEC / 2

    def _hold_window(self) -> None:
        """窓が開いていて保つときなら、閉じる時刻を上限（`HOLD_CAP_SEC`）の先へ置く。閉じていれば触らない（声や求めで
        窓を開けない）。もう保っていれば動かさない（上限は保ち始めから数える）。"""
        now = time.monotonic()
        if self._holding() and self.wake.is_open(now) and not self._held():
            self.wake.until = now + HOLD_CAP_SEC
            self._stop_window_task()
            self._arm_window()  # 上限で閉じる控えとして数える

    def _release_window(self, *, voice_ends: bool = False) -> None:
        """保つときが終わった。保っていて、もう保つときでなければ、いまから 10 秒にして数え直す（事象は数える仕事が送る）。

        `voice_ends` は話し終えた口（出るときの動作で、構成にはまだ話しているが残る）。声は見ず、求めだけを見る。
        """
        holding = self._request_holds() if voice_ends else self._holding()
        if self._held() and not holding:
            self.wake.until = time.monotonic() + WAKE_WINDOW_SEC
            self._stop_window_task()  # 上限まで眠っている仕事を、新しい閉じる時刻で立て直す
            self._arm_window()

    def _review_window(self) -> None:
        """いまの状態で、窓を保つか手放すかを見直す。"""
        if self._holding():
            self._hold_window()
        else:
            self._release_window()

    def on_enter_speaking(self) -> None:
        """話し始めた。窓が開いていれば、話し終えるまで閉じない。閉じていれば触らない（声で窓を開けない）。"""
        self._hold_window()

    def on_exit_speaking(self, played: bool = False) -> None:
        """話し終えた（`played` は鳴ったか）。求めが保つときでなければ、いまから 10 秒にして数え直す。見張りにも知らせる
        （事象ではなく、ただの呼び出し）。出るときの動作なので、構成にはまだ話しているが残る（`_holding` でなく
        `_request_holds` を見る）。"""
        self._release_window(voice_ends=True)
        if self._on_speech_ended is not None:
            try:
                self._on_speech_ended(played)
            except Exception:  # noqa: BLE001
                logger.warning("状態図 話し終えたことを見張りへ知らせられなかった", exc_info=True)

    def _hold_while_speaking(self) -> None:
        """話しているあいだか求めのあいだに窓を開けた・延ばしたら、保つときが終わるまで閉じない。"""
        if self._holding():
            self._hold_window()

    def window_open_at(self, at: float) -> bool:
        """`at`（`time.monotonic` の秒）に窓が開いていたか。門はこの 1 つで決める（状態では決めない）。"""
        return self.wake.is_open(at)

    def end_timers(self) -> None:
        """時間を数える仕事をみな止める（ループを閉じるとき）。このあとは立てない。"""
        self._timers_ended = True
        self._stop_window_task()
        self._stop_tentative_task()
        self._stop_silence_task()
        self._stop_answer_task()

    async def hush_with(self, req: "SilenceRequest") -> None:
        """黙っている依頼を受けた（人の「黙って」、タイマー）。データを置き換え、控えを書き、黙っているへ移る。

        名前を `hush` にしないのは、口の領域 `hush` と重なり、ライブラリが許さないため。

        どの依頼にするか（タイマーは人の依頼を上書きしない、遅いタイマーは延ばす）は、呼び手が `silence_state` の
        純関数で決める。期限を過ぎた依頼では黙らない。
        """
        await self.start()
        if time.time() >= req.until:
            return
        # 呼び手が読んで決めたその場で、待たずに依頼と控えを置く。待つと、そのあいだに読んだ値が古くなる（決定ン）
        self.silence = req
        if self._on_silence_saved is not None:
            try:
                self._on_silence_saved(req)
            except Exception:  # noqa: BLE001
                logger.warning("状態図 黙っている依頼の控えを書けなかった", exc_info=True)
        await self._settle_silence()

    async def unhush(self) -> None:
        """黙るのを解く（「話していいよ」、止めたタイマー）。黙っていなければ何も送らない。"""
        await self.start()
        if self.silence is None and not self._in("silent"):
            return
        # 依頼と控えは待たずに消す。データが無くても状態が黙っているなら戻す（期限で解けたことを送れなかったとき）
        self.silence = None
        self._stop_silence_task()
        self._clear_silence_backup()
        await self._settle_silence()

    def silence_request(self) -> "SilenceRequest | None":
        """いまの黙っている依頼。門は、これを処理した時刻で `is_silenced` に掛けて決める。無ければ None。"""
        return self.silence

    def silenced_outside(self, through_silence: bool = False) -> bool:
        """黙る依頼がいま効いていて、声が求めの外のもの（`through_silence` なし）か（`voice_began` の条件・環-bb）。"""
        from ..silence_state import is_silenced

        return not through_silence and is_silenced(self.silence_request(), now=time.time())

    def is_silent(self) -> bool:
        """口が黙っているにいるか。"""
        return self._in("silent")

    async def _settle_silence(self) -> None:
        """順番が来たら依頼を読み直し、口の状態を依頼に合わせる。黙る口と解く口が重なっても、最後に置いた依頼に合う
        （決定ン）。依頼があれば期限を数え直す。"""
        async with self._turn():
            try:
                if self.silence is not None and not self._in("silent"):
                    await self._fire("silence_began")
                elif self.silence is None and self._in("silent"):
                    await self._fire("silence_ended")
            except Exception:  # noqa: BLE001
                logger.warning("状態図 口の状態を黙っている依頼に合わせられなかった", exc_info=True)
            self._arm_silence()  # 移りが落ちても数える。期限で依頼と控えを消す役を残す

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
        """黙るの期限を数え直す（依頼が変わりうるので、生きている仕事も立て直す）。依頼があれば数える。口の状態には
        従わない。黙っているへ移るのが落ちても、期限で依頼と控えを消すため。"""
        if self._silence_sending:
            return
        self._stop_silence_task()
        if self._timers_ended or self.silence is None:
            return
        with contextlib.suppress(RuntimeError):  # 走っている loop が無ければ数えない
            self._silence_task = spawn_outside(self._wait_silence())

    async def _wait_silence(self) -> None:
        try:
            # 期限は壁の時計（`time.time`）の秒。依頼が置き換わったら、新しい期限まで待つ。
            while True:
                while (req := self.silence) is not None and (left := req.until - time.time()) > 0:
                    await asyncio.sleep(left)
                async with self._turn():  # 見てから送るまでに、ほかの口を割り込ませない（決定ン）
                    req = self.silence
                    if req is not None and req.until - time.time() > 0:
                        continue  # 順番を待つあいだに依頼が置き換わった。新しい期限まで待つ
                    if req is None:
                        return
                    # 期限が来たら依頼と控えをいつも消し、黙っているにいるときだけ送る（移りの無い事象を送らない）
                    logger.info("状態図 黙る期限が来たので解いた")
                    self.silence = None
                    self._clear_silence_backup()
                    if self._in("silent"):
                        self._silence_sending = True
                        try:
                            await self._fire("silence_ended")
                        finally:
                            self._silence_sending = False
                break
            if not self._timers_ended and self.silence is not None:
                # 送っているあいだに新しい依頼が届いた。黙っているへ戻して数え直す。自分を止めないよう、先に手放す。
                self._silence_task = None
                await self._settle_silence()
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
        # 届いた時刻は、すぐ消す。門はすぐ読み、仮に呼ぶ口（`call_tentatively`）も、すぐ書く。順番の中で消すと、
        # 順番を待つあいだにあとから届いた呼びかけまで消す。
        self.tentative_at = None
        self._stop_tentative_task()
        async with self._turn():  # 見てから送るまでに、ほかの口を割り込ませない（決定ン）
            if self._tentative_sending:
                return  # 切れたことを送っている。送り終えれば閉じているにいる（ここで送ると移りが無い）
            try:
                if self._in("tentative"):
                    await self._fire("tentative_dropped")
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
            async with self._turn():  # 見てから送るまでに、ほかの口を割り込ませない（決定ン）
                if self._tentative_sending:
                    return  # 送り終えてから、仕事がデータを見て移し直す
                if self._tentative_live() and not self._in("open"):
                    await self._fire("tentative_called")
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
            while True:
                while (at := self.tentative_at) is not None and (
                    left := at + WAKE_WINDOW_SEC - time.monotonic()
                ) > 0:
                    await asyncio.sleep(left)
                async with self._turn():  # 見てから送るまでに、ほかの口を割り込ませない（決定ン）
                    if self._tentative_live():
                        continue  # 順番を待つあいだに新しい呼びかけが届いた。その 10 秒まで待つ
                    # 送る前に状態を見る（移りの無い事象を送らない）。届いた時刻は残す。
                    lapsed = self._in("tentative")
                    if lapsed:
                        logger.debug("状態図 仮の呼びかけが切れた")
                        self._tentative_sending = True
                        try:
                            await self._fire("tentative_lapsed")
                        finally:
                            self._tentative_sending = False
                break
            if lapsed:
                # 送り終えてから見直す。順番は手放してから呼ぶ（錠は二重に取れない）。
                if not self._timers_ended and self._tentative_live():
                    # 送っているあいだに新しい呼びかけが届いた。仮の呼びかけへ戻して数え直す。自分を止めないよう、先に手放す。
                    self._tentative_task = None
                    await self._to_tentative()
        except Exception:  # noqa: BLE001
            logger.warning("状態図 仮の呼びかけが切れたことを送れなかった", exc_info=True)
        finally:
            if self._tentative_task is asyncio.current_task():
                self._tentative_task = None

    # ── 待っている答え（第 3 段・決定ノ、グ、ゲ、ズ、ゼ）────────────────────────────────

    async def ask_confirm(self, pc: Any) -> None:
        """道具が「確かめて」と判定した預かりを置く（出-y）。前の問いを上書きし、促した時刻は空から始める。"""
        await self._place(AwaitedAnswer("confirm", pc, time.monotonic()))

    async def make_offer(self, question: str, *, session: int, deep: "dict | None" = None) -> None:
        """申し出を預ける（出-dj）。`session` は預けたときの窓の番号。"""
        content = {"question": question, "deep": deep}
        await self._place(AwaitedAnswer("offer", content, time.monotonic(), session=session))

    async def ask_heard(self, check: dict) -> None:
        """見立てつきの聞き返しを預ける（出-cm）。"""
        await self._place(AwaitedAnswer("heard", check, time.monotonic()))

    def confirm(self) -> Any:
        """生きている確認の預かり。無い、切れた（決定ゼ）なら None。読み手は状態でなくデータを読む（決定ズの 1）。"""
        q = self.question
        if q is None or q.kind != "confirm":
            return None
        from ..core.confirm_state import alive

        return q.content if alive(q.content, now=time.time(), ttl=self._confirm_ttl) else None

    def confirm_prompted_at(self) -> "float | None":
        """確認の決まった促しを最後に言った時刻。確認が無ければ None。"""
        q = self.question
        return q.prompted_at if q is not None and q.kind == "confirm" else None

    def note_confirm_prompt(self, at: float) -> None:
        """確認の決まった促しを言った時刻を残す（知-o）。確認が無ければ何もしない。"""
        q = self.question
        if q is not None and q.kind == "confirm":
            q.prompted_at = at

    async def take_confirm(self) -> Any:
        """確認を解くために取り出す。生きていれば預かりを返し、どちらでも確認をなしへ戻す。"""
        pc = self.confirm()
        await self._take("confirm")
        return pc

    async def take_offer(self, *, session: int) -> "tuple[str, dict | None] | None":
        """申し出を取り出す（使っても使わなくても消す）。答えた求めの窓の番号が預けたときと違えば None（決定ゲ）。"""
        q = await self._take("offer")
        if q is None:
            return None
        if not session or session != q.session:
            logger.info("状態図 申し出は窓が変わったので使わない")
            return None
        return str(q.content.get("question") or ""), q.content.get("deep")

    # ── 入口の写し（第 3 段・決定ハ、ガ、ギ、ゾ）──────────────────────────────────────────

    def gate(self) -> str:
        """入口のいまの状態の名前（`EAR_OPEN`・`EAR_MUSIC`・`EAR_TIMER`）。活性化の前は開いているとみなす。"""
        return _NAMES[self._gate_state()]

    def _gate_state(self) -> str:
        ids = {s.id for s in self.configuration}
        return next((n for n in ("music_only", "timer_only") if n in ids), "ear_open")

    async def copy_ring(self, ringing: "Callable[[], bool]") -> None:
        """知らせの音の持ち主（dif の鳴らしている仕事）を写す（第 6 段）。写すのは自律機構の輪（0.5 秒ごと）だけ。
        持ち主は順番が来てから読み、違うときだけ送る。同じなら何もせず、順番も待たない。
        """
        if self.is_ringing() == bool(ringing()):
            return
        try:
            await self.start()
        except Exception:  # noqa: BLE001
            logger.warning("状態図 知らせの音を写せなかった（初めの状態に入れない）", exc_info=True)
            return
        async with self._turn():  # 読んでから送るまでに、ほかの写しを割り込ませない（決定ン）
            try:
                now = bool(ringing())
                if self.is_ringing() != now:
                    await self._fire("ring_began" if now else "ring_ended")
                    logger.info(
                        "状態図 知らせの音を写した：%s",
                        _NAMES["alarm_ringing" if now else "alarm_quiet"],
                    )
            except Exception:  # noqa: BLE001
                logger.warning("状態図 知らせの音を写せなかった", exc_info=True)

    async def copy_song(self, read: "Callable[[], str]") -> None:
        """歌の持ち主（音楽の状態・`song_of`）を写す（第 6 段 項目 3）。写すのは自律機構の輪（0.5 秒ごと）だけ。
        持ち主は順番が来てから読み、違うときだけ送る。同じなら何もせず、順番も待たない。
        """
        try:
            if self.song_state() == read():
                return
            await self.start()
        except Exception:  # noqa: BLE001
            logger.warning("状態図 歌を写せなかった", exc_info=True)
            return
        async with self._turn():  # 読んでから送るまでに、ほかの写しを割り込ませない（決定ン）
            try:
                now = read()
                event = _SONG_ENTERS.get((self.song_state(), now))
                if event:
                    await self._fire(event)
                    logger.info("状態図 歌を写した：%s", _NAMES[now])
            except Exception:  # noqa: BLE001
                logger.warning("状態図 歌を写せなかった", exc_info=True)

    async def copy_gate(self, reason: str) -> None:
        """持ち主の理由（`agent.mic_gate_reason`）を入口へ写す。空なら開いている、`MUSIC_GATE_REASON` なら音楽、
        ほかはタイマーで、理由がそのタイマーの名前。状態も名前も同じなら何もせず、順番も待たない（輪は 0.5 秒ごと）。"""
        from ..agent import MUSIC_GATE_REASON  # 理由を写しへ直すのはここ 1 か所

        if not reason:
            target, name = "ear_open", ""
        elif reason == MUSIC_GATE_REASON:
            target, name = "music_only", ""
        else:
            target, name = "timer_only", reason
        if self._gate_state() == target and self.gate_timer == name:
            return
        try:
            await self.start()
        except Exception:  # noqa: BLE001
            logger.warning("状態図 入口を写せなかった（初めの状態に入れない）", exc_info=True)
            return  # 門は止めない。写しは前のまま
        async with self._turn():  # 見てから送るまでに、ほかの写しを割り込ませない（決定ン）
            before = self.gate_timer
            # 名前は移る前に置く（移っている途中に読んだ状態の行も、名前の空いたタイマーを見ない・決定ズの 1 と同じ）
            self.gate_timer = name
            try:
                if self._gate_state() != target:
                    await self._fire(_GATE_EVENTS[target])
                # 捨てられた（`_fire` は例外を出さない）。名前も戻す
                if self._gate_state() != target:
                    self.gate_timer = before
                    return
                logger.info("状態図 入口を写した：%s %s", self.gate(), self.gate_timer)
            except Exception:  # noqa: BLE001
                self.gate_timer = before  # 移れなければ名前も戻す（写しの状態と名前をずらさない）
                logger.warning("状態図 入口を写せなかった", exc_info=True)

    def offer_waiting(self) -> bool:
        """申し出の答えを待っているか。データを読む（決定ズの 1）。"""
        q = self.question
        return q is not None and q.kind == "offer"

    async def drop_offer(self) -> None:
        """申し出だけを消す（申し出でない返事を話した・出-dj）。ほかの問いには触らない。"""
        await self._take("offer")

    async def take_heard(self) -> "dict | None":
        """聞き返しの見立てを取り出す（使っても使わなくても消す）。30 秒を過ぎていれば None（決定ラ）。"""
        q = await self._take("heard")
        if q is None:
            return None
        if time.monotonic() - q.placed_at > self._heard_ttl:
            logger.info("状態図 聞き返しの見立ては切れていたので使わない")
            return None
        return q.content

    async def _take(self, kind: str) -> "AwaitedAnswer | None":
        """`kind` の問いが置かれていれば消してなしへ戻し、その問いを返す。ほかの問いには触らない。"""
        q = self.question
        if q is None or q.kind != kind:
            return None
        self.question = None
        await asyncio.shield(self._settle_answer())
        return q

    async def _place(self, q: AwaitedAnswer) -> None:
        # データは待つ前に置く（決定ズの 1）。待つあいだに走ったほかの道具も、置いた問いを読む。
        self.question = q
        # 打ち切られても状態をデータに合わせきる（決定ズの 3）
        await asyncio.shield(self._settle_answer())

    def _answer_state(self) -> str:
        ids = {s.id for s in self.configuration}
        return next(
            (n for n in ("confirm_answer", "offer_answer", "heard_answer") if n in ids), "no_answer"
        )

    async def _settle_answer(self) -> None:
        """順番が来たらデータを読み直し、待っている答えの状態をデータに合わせる。置く口と取り出す口が重なっても、
        最後に置いたデータに合う（決定ン）。寿命を数え直す。初めの状態に入れなくても、置いたデータは残す（決定ズの 1）。"""
        try:
            await self.start()
        except Exception:  # noqa: BLE001
            logger.warning(
                "状態図 待っている答えをデータに合わせられなかった（初めの状態に入れない）",
                exc_info=True,
            )
            return
        async with self._turn():
            try:
                q = self.question
                target = _ANSWER_STATES[q.kind] if q is not None else "no_answer"
                if self._answer_state() != target:
                    await self._fire(_ANSWER_EVENTS[q.kind] if q is not None else "answer_cleared")
            except Exception:  # noqa: BLE001
                logger.warning("状態図 待っている答えをデータに合わせられなかった", exc_info=True)
            self._arm_answer()

    def _stop_answer_task(self) -> None:
        task, self._answer_task = self._answer_task, None
        if task is not None:
            task.cancel()

    def _arm_answer(self) -> None:
        """いまの問いの寿命を数え直す。確認と聞き返しだけを数える（申し出は窓と求めで戻す・決定ゲ）。"""
        self._stop_answer_task()
        q = self.question
        if self._timers_ended or q is None:
            return
        ttl = {"confirm": self._confirm_ttl, "heard": self._heard_ttl}.get(q.kind)
        if ttl is None:
            return
        with contextlib.suppress(RuntimeError):  # 走っている loop が無ければ数えない
            self._answer_task = spawn_outside(self._wait_answer(q, q.placed_at + ttl))

    async def _wait_answer(self, q: AwaitedAnswer, until: float) -> None:
        try:
            if (left := until - time.monotonic()) > 0:
                await asyncio.sleep(left)
            async with self._turn():  # 見てから送るまでに、ほかの口を割り込ませない（決定ン）
                if self.question is not q:
                    return  # 上書きされたか使われた。自分の問いでなければ触らない（決定ズの 2）
                logger.info("状態図 %sの答えを待つのが切れた", _NAMES[_ANSWER_STATES[q.kind]])
                self.question = None
                if self._answer_state() != "no_answer":
                    await self._fire("answer_cleared")
        except Exception:  # noqa: BLE001
            logger.warning("状態図 答えを待つのが切れたことを送れなかった", exc_info=True)
        finally:
            if self._answer_task is asyncio.current_task():
                self._answer_task = None

    async def _drop_stale_offer(self, event: str) -> None:
        """申し出をなしへ戻す（決定ゲ）。窓が閉じたとき頭が何もしていないか、求めを閉じたとき窓が開いていなければ。

        答えのあとの書き留めのあいだや、上限で窓が閉じても、窓の中で受けた返事のために残す（答えるまでの求めのあいだは、
        窓を保つ・端-b 段 1-い）。順番の中で呼ぶ。
        """
        q = self.question
        if q is None or q.kind != "offer":
            return
        # 寝ているあいだも、何もしていないと同じく戻す（窓に答える人を待っていない・決定ダ）
        if event == "window_lapsed" and self.head_state() in ("thinking", "answering"):
            return
        if event == "request_closed" and self._in("open"):
            return
        logger.info("状態図 申し出の答えを待つのをやめた（%s）", event)
        self.question = None
        if self._answer_state() != "no_answer":
            await self._fire("answer_cleared")

    def _in(self, state_id: str) -> bool:
        return any(s.id == state_id for s in self.configuration)

    async def _to_open(self) -> None:
        """いまの時刻でデータの上の窓が開いていれば、開いているへ移して数え直す。古い届いた時刻で開けても、もう
        切れている窓は開いているへ移さない。状態図が落ちても、会話は止めない（警告を 1 行残す）。"""
        try:
            async with self._turn():  # 見てから送るまでに、ほかの口を割り込ませない（決定ン）
                if not self._in("open") and self.wake.is_open(time.monotonic()):
                    await self._fire("window_opened")
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
        """窓の時間を数え直す。開いていなければ数えない。保っているあいだは上限（`HOLD_CAP_SEC`）を数える。"""
        if self._window_sending:
            return
        if self._timers_ended or not self._in("open"):
            self._stop_window_task()
            return
        task = self._window_task
        if task is not None and not task.done():
            return  # 生きている仕事は、切れる時刻を読み直すので、そのまま使う（待つあいだは毎秒延ばす）
        with contextlib.suppress(RuntimeError):  # 走っている loop が無ければ数えない
            self._window_task = spawn_outside(self._wait_window())

    async def _wait_window(self) -> None:
        try:
            while True:
                while (left := self.wake.until - time.monotonic()) > 0:
                    # 保っているあいだは上限まで眠る。保つときが終わったら、短い閉じる時刻で立て直す（`_release_window`）
                    await asyncio.sleep(left)
                async with self._turn():  # 見てから送るまでに、ほかの口を割り込ませない（決定ン）
                    if self.wake.until - time.monotonic() > 0:
                        continue  # 順番を待つあいだに延ばされた。新しい切れる時刻まで待つ
                    # 送る前に状態を見る。もう閉じているのに送ると、移りが無い（決定ユ）。
                    lapsed = self._in("open")
                    if lapsed:
                        logger.debug("状態図 窓が切れた")
                        self._window_sending = True
                        try:
                            await self._fire("window_lapsed")
                        finally:
                            self._window_sending = False
                break
            if lapsed:
                # 送り終えてから見直す。順番は手放してから呼ぶ（錠は二重に取れない）。
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
            self.silence = req
            await self._settle_silence()
        else:
            self._clear_silence_backup()

    def _turn(self) -> asyncio.Lock:
        """いま走っているループの順番待ち（決定ン）。ループが変わっていたら作り直す。"""
        loop = asyncio.get_running_loop()
        if self._turn_lock is None or self._turn_loop is not loop:
            self._turn_lock = asyncio.Lock()
            self._turn_loop = loop
        return self._turn_lock

    def voice_turn(self) -> asyncio.Lock:
        """声の鍵（第 5 段）。いま走っているループのもので、ループが変わっていたら作り直す（`_turn` と同じ）。"""
        loop = asyncio.get_running_loop()
        if self._voice_lock is None or self._voice_loop is not loop:
            self._voice_lock = asyncio.Lock()
            self._voice_loop = loop
        return self._voice_lock

    async def fire(self, event: str, **data: Any) -> None:
        """外から事象を送る口（ループの `_to_state`）。順番を待ってから送る（決定ン）。

        ライブラリは事象の処理の途中で動作を `asyncio.gather` で呼び、ほかの仕事に順番を譲る。順番を待たずに送ると、
        並んだ事象はまだ処理されておらず、ほかの口から見える状態が古い。古い状態で送った事象は移りが無く、ライブラリは
        列に並んでいたほかの事象も一緒に退ける。順番を待てば、ライブラリの列にはいつも 1 つしか並ばない。

        状態図に入るときの動作の中からは待たない（順番を持った送り手が、その動作の終わりを待っているので止まる）。
        動作の中で送るときは、外の仕事として立てる（`spawn_outside`）。

        `data` は事象と一緒に動作へ渡す（例：話し終えた `voice_ended` の、鳴ったかの `played`）。
        """
        async with self._turn():
            await self._fire(event, **data)

    async def _fire(self, event: str, **data: Any) -> None:
        """事象を送る、ただ 1 つの口。順番を持っているあいだに呼ぶ。移りが無ければ捨て、1 行残す（決定ユ）。"""
        try:
            await self.send(event, **data)
            if event in ("window_lapsed", "request_closed"):
                await self._drop_stale_offer(event)
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
        """頭の今の状態（`idle`・`thinking`・`answering`・`sleeping`）。活性化の前は `idle` とみなす。"""
        ids = {s.id for s in self.configuration}
        return next((n for n in ("thinking", "answering", "sleeping") if n in ids), "idle")

    # ── 頭の寝ている（第 4 段・決定ダ、ナ、ク）────────────────────────────────────────────

    def is_head_idle(self) -> bool:
        """頭が何もしていないか（体が見回りに入る条件。移りを選ぶ時点の頭を読む）。"""
        return self.head_state() == "idle"

    def is_sleeping(self) -> bool:
        """頭が寝ているか。"""
        return self.head_state() == "sleeping"

    def feel_sleepy(self) -> None:
        """眠いを置く（自律機構の rest の発火）。眠るのは、頭が何もしていないになってから（`fall_asleep`）。"""
        self.sleepy = True

    async def fall_asleep(self) -> bool:
        """眠くて、頭が何もしていないなら寝ているへ移り、眠いを消して真を返す。考えているか答えを言っているあいだは、
        眠いを残して偽（求めを閉じるまで持ち越す・決定ナ）。"""
        if not self.sleepy:
            return False
        await self.start()
        async with self._turn():  # 見てから送るまでに、求めを開く事象を割り込ませない（決定ン）
            if self.head_state() != "idle":
                return False
            await self._fire("sleep_began")
            if self.head_state() != "sleeping":
                return False  # 捨てられた（`_fire` は例外を出さない）。眠いは残し、次の輪で見直す
            self.sleepy = False
            return True

    async def open_request(self, event: str) -> None:
        """求めを開く（`request_opened` か `quiet_request_opened`）。頭が寝ていれば、起きるまで待ってから開く。

        眠る口（`fall_asleep`）と同じ順番の中で見て送るので、どちらが先でも、寝ていると考えているは同時に成り立たない。
        捨てずに待たせるのは、機器の知らせが積む前に済ませていることがあるから（タイマーは鳴った印を付けてから積む・
        案 A′・2026-11-18 本人の決定）。寝ているあいだに届いた言葉と命令は、ここへ来る前に `agent.run` が捨てる。
        """
        await self.start()
        told = False
        while True:
            async with self._turn():
                if self.head_state() != "sleeping":
                    await self._fire(event)
                    return
            if not told:
                logger.info("頭が寝ているので、起きるまで求めを始めずに待つ（%s）", event)
                told = True
            await asyncio.sleep(WAKE_POLL_SEC)

    async def wake_up(self) -> None:
        """起きた（内省を終えた。落ちても呼ぶ）。"""
        await self.fire("woke")

    def voice_state(self) -> str:
        """口の声の今の状態（`VOICE_STATES` のどれか）。活性化の前は `no_voice` とみなす。"""
        ids = {s.id for s in self.configuration}
        return next((n for n in VOICE_STATES if n in ids), "no_voice")

    def is_ringing(self) -> bool:
        """口の知らせの音が鳴っているか（第 6 段）。活性化の前は止まっているとみなす。"""
        return "alarm_ringing" in {s.id for s in self.configuration}

    def body_state(self) -> str:
        """体の今の状態（`BODY_STATES` のどれか）。活性化の前は止まっているとみなす。"""
        ids = {s.id for s in self.configuration}
        return next((n for n in BODY_STATES if n in ids), "body_still")

    def song_state(self) -> str:
        """口の歌の今の状態（`SONG_STATES` のどれか）。活性化の前は止まっているとみなす。"""
        ids = {s.id for s in self.configuration}
        return next((n for n in SONG_STATES if n in ids), "song_stopped")

    def thinking_child(self) -> str:
        """考えているの今の子の名前。考えていなければ空。"""
        ids = {s.id for s in self.configuration}
        return next((n for n in CHILDREN if n in ids), "")

    def describe(self) -> str:
        """各領域の今の状態を 1 行で（ログ用）。"""
        ids = {s.id for s in self.configuration}
        head = next((n for n in ("idle", "thinking", "answering", "sleeping") if n in ids), "")
        if head == "thinking":
            child = self.thinking_child()
            head_text = f"{_NAMES[head]}/{_NAMES[child]}" if child else _NAMES[head]
        else:
            head_text = _NAMES.get(head, "")
        ear = next((n for n in ("hearing", "deaf") if n in ids), "")
        window = next((n for n in ("shut", "open", "tentative") if n in ids), "")
        mouth = next((n for n in ("may_speak", "silent") if n in ids), "")
        voice = next((n for n in VOICE_STATES if n in ids), "")
        answer = next((n for n in _NAMES if n.endswith("_answer") and n in ids), "")
        gate = next((n for n in ("ear_open", "music_only", "timer_only") if n in ids), "")
        alarm = self.is_ringing()
        return (
            f"頭={head_text} 耳={_NAMES.get(ear, '')}/窓={_NAMES.get(window, '')}"
            f"/答え={_NAMES.get(answer, '')}/入口={_NAMES.get(gate, '')} 口={_NAMES.get(mouth, '')}"
            f"/声={_NAMES.get(voice, '')}/知らせの音={_NAMES['alarm_ringing' if alarm else 'alarm_quiet']}"
            f"/歌={_NAMES[self.song_state()]} 体={_NAMES[self.body_state()]}"
        )
