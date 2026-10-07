"""ウェイクワードと窓（出-as・出-au・2026-09-26・`設計方針_判定の段` v0.1 §2.1）。純関数と小さな器。

名前を聞くまでは、入力を会話として受けない。家族どうしの話・食事のあいさつ・幻聴の定型文にまで返事を
していた（実機 2026-09-26 朝）。名前を聞いたら 10 秒の窓を開き、窓の中の入力・返事・つなぎで、そこから
10 秒へ延ばす。**キーボードも声と同じ**で、名前で窓を開ける（出-au で改めた）。窓は届いた時刻で判定する。

**窓は出来事で開け閉めする。** 声が鳴ったか・マイクで聞いたかは見ない——`.env.quiet`（声を出さない・
マイクを聞かない）でも同じ動きにするため。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

from .silence_rules import names_me

#: 窓の長さ（出-as の 1 分 → 30 秒・2026-09-26・`設計方針_判定の段` §2.1 → 10 秒・2026-10-07 本人の決定）。
WAKE_WINDOW_SEC = 10.0


def heard_name(text: str, names: "list[str]") -> bool:
    """ウェイクワードを聞いたか。**文頭に**名前があれば（1 字違いまで・`names_me`・2026-09-30 に「文のどこか」から改めた）。

    **名前が設定されていなければ、声は何も聞かない**（本人の決定 2026-09-26）。呼ばれたと分かる材料が
    無いのに受けると、周りの会話にまで返事をする元に戻る。キーボードの入力は別の口で、いつでも受ける。
    """
    if not names:
        return False
    return names_me(text, names)


class InputText(str):
    """届いた入力の印。中身はただの文字列で、**出どころと届いた時刻**を運ぶ（出-as 段 2・出-au 段 1-1）。

    積むところ（声の書き起こし・キーボードの 3 か所）が包み、`agent.run` の先頭が読む。窓は**届いた時刻**で
    判定する（`設計方針_判定の段` §2 原則 3）——画面は前の `run()` が返るまで次の入力を取らないので、取り出した
    時刻で見ると、窓の中で言った声を窓が切れた後として捨てることがある。画面と待ち行列の型は `str` のまま。
    """

    source = "keyboard"
    at: float

    def __new__(cls, text: str, at: "float | None" = None) -> "InputText":
        obj = super().__new__(cls, text)
        obj.at = time.monotonic() if at is None else float(at)
        return obj


class VoiceText(InputText):
    """声の書き起こし（`realtime_stt_session._committed_relay`・TUI の録音）。

    `voice` はその区切りの声の特徴（ECAPA・知-ae）。ローカルの書き起こしが 1.5 秒以上の区切りにだけ載せる。
    短い断片・ElevenLabs・TUI の録音には無い（None）。
    """

    source = "voice"
    voice: "Any" = None
    #: 書き起こしの無音らしさ（区切りのいちばん大きい値）と確かさ（いちばん低い値）。名前で起きる基準が見る
    #: （2026-10-05）。ローカルの書き起こしだけが載せる。
    no_speech: "float | None" = None
    logprob: "float | None" = None

    def __new__(
        cls,
        text: str,
        at: "float | None" = None,
        voice: "Any" = None,
        no_speech: "float | None" = None,
        logprob: "float | None" = None,
    ) -> "VoiceText":
        obj = cast("VoiceText", super().__new__(cls, text, at))
        obj.voice = voice
        obj.no_speech = no_speech
        obj.logprob = logprob
        return obj


class KeyText(InputText):
    """キーボードで打たれた入力（GUI・TUI・CUI）。"""

    source = "keyboard"


def source_of(text: str) -> str:
    """入力の出どころ。印があればそれ、無ければ `keyboard`（`.env.quiet` の入力もこれ）。"""
    return text.source if isinstance(text, InputText) else "keyboard"


def voice_of(text: str) -> "Any":
    """入力に載っている声の特徴（知-ae）。声でない入力や、特徴の無い声は None。"""
    return getattr(text, "voice", None) if isinstance(text, VoiceText) else None


def measures_of(text: str) -> "tuple[float | None, float | None]":
    """入力に載っている書き起こしの（無音らしさ, 確かさ）。無ければ (None, None)。"""
    if isinstance(text, VoiceText):
        return text.no_speech, text.logprob
    return None, None


def arrived_at(text: str, *, now: "float | None" = None) -> float:
    """入力が届いた時刻（`time.monotonic()` の秒）。印が無ければ呼んだ時刻。"""
    if isinstance(text, InputText):
        return text.at
    return time.monotonic() if now is None else now


@dataclass
class WakeWindow:
    """会話として受ける窓。`until` は窓が閉じる時刻（`time.monotonic()` の秒）。"""

    until: float = 0.0
    #: いま鳴っている声の数。鳴っているあいだは `until` を過ぎても開いている（2026-10-07 実機 23:18）。
    speaking: int = 0

    def is_open(self, now: float) -> bool:
        return self.speaking > 0 or now < self.until

    def hold(self) -> None:
        """話し始め（窓が開いているときだけ呼ぶ）。話し終わる（`release`）まで閉じない。

        窓を残り時間で数えていると、長い返事を話しているうちに切れ、聞き返しへの返事を窓の外として捨てた
        （2026-10-07 実機 23:18：「何の曲にしますか？」の後の「いいですよ」を 3 回）。
        """
        self.speaking += 1

    def release(self, now: float) -> None:
        """話し終わり。声がみな鳴り終わったら、そこから `WAKE_WINDOW_SEC`（本人「話し終わってから１０秒」）。"""
        if self.speaking <= 0:
            return
        self.speaking -= 1
        if self.speaking == 0:
            self.until = max(self.until, now + WAKE_WINDOW_SEC)

    def open(self, now: float) -> None:
        """名前を聞いた・キーボードで打たれた。そこから `WAKE_WINDOW_SEC`（縮めない）。"""
        self.until = max(self.until, now + WAKE_WINDOW_SEC)

    def extend(self, now: float) -> None:
        """窓の中の入力・返事・つなぎ。**窓が開いているときだけ**そこから `WAKE_WINDOW_SEC` へ延ばす。

        切れた後の返事は話さない（独り言）ので、延ばして開け直さない。
        """
        if self.is_open(now):
            self.open(now)

    def close(self) -> None:
        self.until = 0.0
        self.speaking = 0
