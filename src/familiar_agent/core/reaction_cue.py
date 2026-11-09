"""話しかけたときの合図（出-bg・2026-10-10・本人の決定）。

声で話しかけた言葉に、パジュが聞いて反応したかを音ですぐ伝える。Jev が判断を始めたら機械音 A（`THINKING_WAV`）を
鳴らし続け、決めたら止めて、決めた動作に合わせて流す。ここはどの合図かを決めるのと、音のファイルの置き場だけを持つ
（鳴らすのは `io/dif`）。音は `scripts/gen_reaction_sounds.py` で作る。

| Jev が決めた動作 | 合図 |
|---|---|
| 黙る | なし |
| 軽く返す・聞き返す・状態を伝える | 効果音 B（返事がすぐ続くので、声の相づちを重ねない） |
| 道具を使う | 作り置きの声（はいっ） |
| 考えて返す | 作り置きの声（えっと） |

合図を鳴らした後で、軽量LLM の判定か主LLM の印で黙ると決まったら、黙る音（`SILENT_WAV`・下がっていく 2 つの和音）を 1 回鳴らす
（出-dc・2026-10-26・本人の決定）。合図だけ鳴って何も言わない、を「聞こえたけど返事はやめた」と分かるようにする。
"""

from __future__ import annotations

import random
from pathlib import Path

NONE = "なし"
ACK = "効果音"
TOOL = "道具の声"
THINK = "考える声"
SILENT = "黙る音"

_SOUNDS = Path(__file__).resolve().parent.parent / "sounds"
#: Jev が判断しているあいだ鳴らし続ける機械音 A（1 周期ぶん。鳴らす側が繰り返す）。
THINKING_WAV = _SOUNDS / "react_thinking.wav"
#: 考えているの子ごとの電子音（環-aw 決定テ、ヤ。音は第 1 段 項目 6・2026-11-09・本人が聞いて決めた）。決めているは
#: 機械音 A と同じ音。ここを差し替えれば、鳴らす側は変えずに済む。
BEEP_WAVS: "dict[str, Path]" = {
    "recalling": _SOUNDS / "react_beep_recalling.wav",
    "deciding": THINKING_WAV,
    "awaiting": _SOUNDS / "react_beep_awaiting.wav",
    "checking": _SOUNDS / "react_beep_checking.wav",
    "noting": _SOUNDS / "react_beep_noting.wav",
}
#: 軽く返す・聞き返すときの効果音 B。
ACK_WAV = _SOUNDS / "react_ack.wav"
#: 合図の後で黙ると決まったときの黙る音（出-dc）。
SILENT_WAV = _SOUNDS / "react_silent.wav"
#: 作り置きの声（パジュの声・SBV2）。いまは 1 つずつ（出-de）。増やせば、鳴らす側が毎回その中から選ぶ（`sound_for`）。
TOOL_WORDS = ("はいっ",)
# 最初の反応は道具の声 1 つと「えっと」だけにした（出-de・2026-10-31・本人）。前は道具の声が はい・はーい！・待ってね・
# オッケー、考える声が えっと・うーん だった（出-ce）。道具の声は「はーい！」から「はいっ」に変えた（環-aw 第 1 段 項目 6・
# 2026-11-09・本人が聞いて決めた）。2 つの声の大きさは、作る道具が決まった大きさに揃える（出-df）。
THINK_WORDS = ("えっと",)
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
    if cue == SILENT:
        return SILENT_WAV
    if cue == TOOL:
        return voiced(random.choice(TOOL_WAVS))
    if cue == THINK:
        return voiced(random.choice(THINK_WAVS))
    return None


def sound_files() -> "list[Path]":
    """合図の音のファイル全部（作る道具と試験が見る）。"""
    beeps = [p for p in BEEP_WAVS.values() if p != THINKING_WAV]
    return [THINKING_WAV, ACK_WAV, SILENT_WAV, *beeps, *TOOL_WAVS, *THINK_WAVS]
