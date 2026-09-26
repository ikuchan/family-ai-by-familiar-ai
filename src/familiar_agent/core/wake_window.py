"""ウェイクワードと 1 分の窓（出-as・2026-09-26・`設計方針_話していいかの決まり` v0.1 §2.3）。純関数と小さな器。

名前を聞くまでは、声を会話として受けない。家族どうしの話・食事のあいさつ・幻聴の定型文にまで返事を
していた（実機 2026-09-26 朝）。名前を聞いたら 1 分の窓を開き、窓の中の入力・返事・つなぎで、そこから
1 分へ延ばす。キーボードの入力はいつでも受けて窓を開ける（紛れ込みようがない）。

**窓は出来事で開け閉めする。** 声が鳴ったか・マイクで聞いたかは見ない——`.env.quiet`（声を出さない・
マイクを聞かない）でも同じ動きにするため。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .silence_rules import names_me

#: 窓の長さ（本人の決まり「ウェイクワードを聞いてから 1 分以内」）。
WINDOW_SEC = 60.0


def heard_name(text: str, names: "list[str]") -> bool:
    """ウェイクワードを聞いたか。文の**どこかに**名前があれば（1 字違いまで・`names_me`）。

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
    """声の書き起こし（`realtime_stt_session._committed_relay`・TUI の録音）。"""

    source = "voice"


class KeyText(InputText):
    """キーボードで打たれた入力（GUI・TUI・CUI）。"""

    source = "keyboard"


def source_of(text: str) -> str:
    """入力の出どころ。印があればそれ、無ければ `keyboard`（`.env.quiet` の入力もこれ）。"""
    return text.source if isinstance(text, InputText) else "keyboard"


def arrived_at(text: str, *, now: "float | None" = None) -> float:
    """入力が届いた時刻（`time.monotonic()` の秒）。印が無ければ呼んだ時刻。"""
    if isinstance(text, InputText):
        return text.at
    return time.monotonic() if now is None else now


@dataclass
class WakeWindow:
    """会話として受ける窓。`until` は窓が閉じる時刻（`time.monotonic()` の秒）。"""

    until: float = 0.0

    def is_open(self, now: float) -> bool:
        return now < self.until

    def open(self, now: float) -> None:
        """名前を聞いた・キーボードで打たれた。そこから 1 分（縮めない）。"""
        self.until = max(self.until, now + WINDOW_SEC)

    def extend(self, now: float) -> None:
        """窓の中の入力・返事・つなぎ。**窓が開いているときだけ**そこから 1 分へ延ばす。

        切れた後の返事は話さない（独り言）ので、延ばして開け直さない。
        """
        if self.is_open(now):
            self.open(now)

    def close(self) -> None:
        self.until = 0.0
