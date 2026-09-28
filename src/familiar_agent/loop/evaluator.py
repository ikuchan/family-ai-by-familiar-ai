"""評価器 — ターン評価の集合（W2b-2）。判定は Jev、文章は軽量LLM（出-au 段 5・`core/jev_judges`）。

agent.py から分離した、次の3つを持つ（発話前の検査・続き先・記憶の申告は `core/jev_judges` へ移した）。

- emotion_for_turn: ターンの感情を PAD で評価し派生ラベルを返す（値踏みゲート込み・判定は Jev）
- summarize_exchange: やり取りを1文へ蒸留（記憶保存用・軽量LLM が書く）
- infer_companion_mood: 相手の気分を分類（判定は Jev・使えなければ既定の absent）

依存は構築時に注入する utility_backend・backend・jev。mood レジスタは
`load_current_mood()` で読むだけ（書かない）。
"""

from __future__ import annotations

import logging

from .._i18n import _t
from ..core import jev_judges
from ..core.context_parts import Stance as _Stance
from ..emotion_pad import label_from_pad
from ..mood_register import MoodPAD, load_current_mood

logger = logging.getLogger(__name__)


# ── プロンプトとパラメータ ───────────────────────────────────────────────────

# 発話の前に規則違反を見る（出-f）。**規則の写しをここに置かない。** 正本は
# `EVENT_SYSTEM_PROMPT` の `(rules ...)` で、システム文として渡る。写しを持てば、正本が
# 変わったときにここだけ古くなる。
# 値踏みゲート（課題5・Config 差し替え可）。A<A_GATE は評価器を呼ばず P/Pn/Dom＝M。
A_GATE = 0.25

# Conversation save prompt — distill what happened into one sentence
_SUMMARY_PROMPT = """\
Summarize this exchange in one sentence that captures the emotional core. \
Write in {lang}.
Speaker: {user}
Agent: {agent}

One sentence only."""


class Evaluator:
    """軽量LLM を使うターン評価をまとめる。

    構築時に utility_backend（評価用の安い経路）と backend（主経路）を注入する。
    専用の utility backend が無い（utility is backend）環境では、余計な LLM 往復を
    避けて発見的手法やスキップに落とす。
    """

    def __init__(self, utility_backend, backend, *, context=None, jev=None) -> None:
        """`context(stance)` は立ち位置と文脈を返す（出-e）。

        渡さなければ立ち位置を渡さない（いままでと同じ）。材料が欠けたときも `None` を
        返してよい——`FAMILY.md` が無い機体でターンを落とさない。
        """
        self._utility_backend = utility_backend
        self.backend = backend
        self._context = context
        # 判定は Jev（出-au 段 5-6：感情の評価と気分の見立て）。無ければ未測定・既定の気分。
        self._jev = jev

    def _stance(self, stance) -> "str | None":
        """立ち位置と文脈を引く。提供者が無ければ `None`。"""
        if self._context is None:
            return None
        return self._context(stance)

    async def emotion_for_turn(
        self, text: str, arousal: float, *, mood: "MoodPAD | None" = None
    ) -> "tuple[MoodPAD | None, float, str]":
        """ターンの感情を PAD で評価し、A と派生ラベルを合わせて返す（W2b-2）。

        返すのは `(PAD, A, ラベル)` で、**測れなかったときの PAD は `None`**（050）。
        現在の mood をベースに評価器（軽量LLM）が P/Pn/Dom を出す（A は機械 arousal・
        A_gate 未満は呼ばない）。mood は読みだけ。

        **未測定のラベルは `neutral`。** `observations.emotion` は NOT NULL で、粗い分類に
        すぎない（正は PAD である）。ラベルまで欠かせると既存の読み手が全部 None を扱う
        ことになり、得るものより失うものが大きい。
        """
        # **感情を作るのはパジュである**（出-e）。判定は Jev（出-au 段 5-6）。
        pad, a = await jev_judges.judge_emotion(
            self._jev,
            text=text,
            mood=load_current_mood() if mood is None else mood,
            arousal=arousal,
            a_gate=A_GATE,
        )
        return pad, a, ("neutral" if pad is None else label_from_pad(pad))

    async def infer_companion_mood(self, text: str) -> str:
        """相手の言葉から相手の気分を見立てる（判定は Jev・出-au 段 5-6）。

        言葉が短すぎれば `absent`。Jev が使えないときも見立てずに既定の `absent`（本人の決定 2026-09-27）。
        """
        if not text or len(text.strip()) < 3:
            return "absent"
        return await jev_judges.judge_companion_mood(self._jev, text=text)

    async def summarize_exchange(self, user_input: str, agent_response: str) -> str:
        """Distill an exchange into one sentence for memory storage."""
        # 自分の記憶に残す言葉なので、一人称で立つ。
        result = await self._utility_backend.complete(
            _SUMMARY_PROMPT.format(
                lang=_t("summary_lang"),
                user=user_input[:200],
                agent=agent_response[:200],
            ),
            max_tokens=80,
            system=self._stance(_Stance.PAJU),
        )
        return result or agent_response[:100]
