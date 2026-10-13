"""話しかけたときの合図（出-bg・2026-10-10・本人の決定）。

声で話しかけた言葉に、パジュが聞いて反応したかを音ですぐ伝える。Jev が判断を始めたら機械音 A（`THINKING_WAV`）を
鳴らし続け、決めたら止めて、決めた動作に合わせて流す。ここはどの合図かを決めるのと、音のファイルの置き場だけを持つ
（鳴らすのは `io/dif`）。音は `scripts/gen_reaction_sounds.py` で作る。

| Jev が決めた動作 | 合図 |
|---|---|
| 黙る | なし |
| 軽く返す・聞き返す・状態を伝える | 効果音 B（返事がすぐ続くので、声の相づちを重ねない） |
| 道具を使う | 作り置きの声（はい・うん） |
| 考えて返す | 作り置きの声（えっと・うーん） |
"""

from __future__ import annotations

import random
from pathlib import Path

NONE = "なし"
ACK = "効果音"
TOOL = "道具の声"
THINK = "考える声"

_SOUNDS = Path(__file__).resolve().parent.parent / "sounds"
#: Jev が判断しているあいだ鳴らし続ける機械音 A（1 周期ぶん。鳴らす側が繰り返す）。
THINKING_WAV = _SOUNDS / "react_thinking.wav"
#: 軽く返す・聞き返すときの効果音 B。
ACK_WAV = _SOUNDS / "react_ack.wav"
#: 作り置きの声（パジュの声・SBV2）。毎回その中から選ぶ。
TOOL_WORDS = ("はい", "うん")  # 「りょ」は「キョ」と聞こえたので外した（2026-10-12・本人）
THINK_WORDS = ("えっと", "うーん")  # 「んー」は変に聞こえたので外した（出-ce・2026-10-13・本人）
TOOL_WAVS = tuple(_SOUNDS / f"react_tool_{i}.wav" for i in range(len(TOOL_WORDS)))
THINK_WAVS = tuple(_SOUNDS / f"react_think_{i}.wav" for i in range(len(THINK_WORDS)))

#: 効果音 B を流す動作（軽量LLM の一言がすぐ続くもの）。黙る・考えて返す以外の、言葉だけで返すもの。
_LIGHT = frozenset(
    {"ask_back", "reply_light", "state_light", "tell_light", "talk_light", "music_now"}
)


def cue_for(final: str) -> str:
    """最終の動作 → 合図。黙るはなし・言葉だけで返すものは効果音・考えて返すは考える声・ほか（道具）は道具の声。"""
    if final == "silent":
        return NONE
    if final in _LIGHT:
        return ACK
    if final in ("reply_full", "talk_full"):
        return THINK
    return TOOL


def word_for(path: Path) -> str:
    """合図の音のファイル → 言った言葉（出-bt）。機械音・効果音は言葉が無いので空。

    ファイル名で比べる。どの声のフォルダ（`voiced`）でも、「はい」は `react_tool_0.wav` である。
    """
    for words, wavs in ((TOOL_WORDS, TOOL_WAVS), (THINK_WORDS, THINK_WAVS)):
        names = [w.name for w in wavs]
        if path.name in names:
            return words[names.index(path.name)]
    return ""


def voiced(path: Path) -> Path:
    """作り置きの声を、いまの声のもので返す（出-bv）。返事（Gemini）と声をそろえる。

    主の担い手が gemini で、`<置き場>/voices/<声>/` に同じ名前のファイルがあればそれ、無ければ元のファイル（SBV2 版）。
    主が SBV2 なら SBV2 版を使う。声は `.env` の `GEMINI_TTS_VOICE` で切り替える。
    """
    from ..config import TTSConfig

    cfg = TTSConfig()
    if cfg.engine != "gemini":
        return path
    candidate = path.parent / "voices" / cfg.gemini_voice / path.name
    return candidate if candidate.exists() else path


def sound_for(cue: str) -> "Path | None":
    """合図 → 流すファイル。作り置きの声は毎回その中から選ぶ。なしは None。"""
    if cue == ACK:
        return ACK_WAV
    if cue == TOOL:
        return voiced(random.choice(TOOL_WAVS))
    if cue == THINK:
        return voiced(random.choice(THINK_WAVS))
    return None


def sound_files() -> "list[Path]":
    """合図の音のファイル全部（作る道具と試験が見る）。"""
    return [THINKING_WAV, ACK_WAV, *TOOL_WAVS, *THINK_WAVS]
