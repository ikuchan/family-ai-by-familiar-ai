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

from .silence_rules import _loose, _one_insertion_apart, _within_one_edit, names_me

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


def _plain(s: str) -> str:
    """呼び方の違いだけを均す：カタカナをひらがなに、長音を落とす（濁点と小書きは残す）。"""
    out = []
    for ch in s:
        if ch in ("ー", "〜", "～"):
            continue
        o = ord(ch)
        out.append(chr(o - 0x60) if 0x30A1 <= o <= 0x30F6 else ch)
    return "".join(out)


def _head_at(text: str) -> int:
    """文頭の空白と記号を飛ばした位置（`silence_rules._head` と同じ飛ばし方）。"""
    import unicodedata

    i = 0
    while i < len(text) and (text[i].isspace() or unicodedata.category(text[i])[0] in "PSZ"):
        i += 1
    return i


def fix_name(text: str, names: "list[str]") -> "tuple[str, str]":
    """文頭の聞き違いの名前を、当たった名前に直す（知-ap・2026-10-10・本人の決定）。返りは（先へ渡す文, 画面の文）。

    窓の門は文頭の 1 字違いまで名前とみなす（`names_me`）が、文は書き起こしのまま O・調停・主LLM に渡っていた。
    10/08 18:16「バージュ音楽をかけて」に、主LLM が「『バージュ』っていう曲かアーティストですか？」と聞き返した。
    直すのは聞き違えた部分だけで、画面には聞こえたままの形に名前を括弧で添える（`バージュ（パジュ）音楽をかけて`）。

    長音とかなの違いだけ（「パジュー」「ぱじゅ」）は直さない。聞き違いではなく呼び方の違いで、主LLM も読める。
    名前が文頭に無ければ、何も変えない。
    """
    if not names or not names_me(text, names):
        return text, text
    i0 = _head_at(text)
    rest = text[i0:]
    for raw in names:
        n = _loose(raw)
        if not n:
            continue
        # 当たり方の近い順に探す：ゆるい読みが同じ → 同じ長さで 1 字違い → 1 字の抜け・足し（`names_me` と同じ許し方）。
        # 同じ段では長いほうを取り、名前の後ろの長音まで聞き違いに含める。
        tiers: "list[list[int]]" = [[], [], []]
        for k in range(1, len(rest) + 1):
            lp = _loose(rest[:k])
            if lp == n:
                tiers[0].append(k)
            elif len(n) >= 3 and len(lp) == len(n) and _within_one_edit(lp, n):
                tiers[1].append(k)
            elif len(n) >= 3 and abs(len(lp) - len(n)) == 1 and _one_insertion_apart(lp, n):
                tiers[2].append(k)
        for found in tiers:
            if not found:
                continue
            k = max(found)
            heard = rest[:k]
            if _plain(heard) == _plain(raw):
                return text, text  # 呼び方の違いだけ
            return (
                text[:i0] + raw + rest[k:],
                text[:i0] + heard + f"（{raw}）" + rest[k:],
            )
    return text, text


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
