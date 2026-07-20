"""評価器 — 軽量LLM（utility backend）を使うターン評価の集合（W2b-2）。

agent.py から分離した、次の4つを持つ。

- emotion_for_turn: ターンの感情を PAD で評価し派生ラベルを返す（値踏みゲート込み）
- summarize_exchange: やり取りを1文へ蒸留（記憶保存用）
- infer_companion_mood: 相手の気分を分類（専用軽量backend が無ければキーワード発見的手法）
- check_response_coherence: 応答の論理的自己矛盾・規則違反を配信前に検出

依存は構築時に注入する utility_backend と backend のみ。mood レジスタは
`load_current_mood()` で読むだけ（書かない）。
"""

from __future__ import annotations

import logging
import re

from .._i18n import _t
from ..emotion_pad import label_from_pad
from ..mood_register import MoodPAD, load_current_mood
from .history import _flatten_history

logger = logging.getLogger(__name__)


# ── プロンプトとパラメータ ───────────────────────────────────────────────────

# Response coherence check — catch logical self-contradictions before delivery
_COHERENCE_CHECK_PROMPT = """\
You are a logical consistency checker. Given the recent conversation and the agent's \
planned response, determine whether the response contains a logical error, rule violation, \
or self-contradiction.

Examples of violations:
- In shiritori (word-chain game): responding with a word that ends in 'ん'
- Claiming something that directly contradicts what was just said
- Giving an answer that violates the stated rules of an ongoing activity

Recent conversation:
{context}

Agent's planned response:
{response}

If the response is logically consistent, reply with exactly: OK
If there is a violation, reply with a brief description of the violation (one sentence)."""

# 値踏みゲート（課題5・Config 差し替え可）。A<A_GATE は評価器を呼ばず P/Pn/Dom＝M。
A_GATE = 0.25

# 評価器（軽量LLM）へ観測の感情を P/Pn/Dom で出させるプロンプト（W2b-2・論点5b）。
# A 軸は機械 arousal なので尋ねない。最終的には自己認識 MI のシステムプロンプトへ統合する。
_EMOTION_PAD_PROMPT = """\
Rate the emotion of this exchange on three axes, each 0.0 to 1.0:
- P  (pleasure):    0 none, 0.5 neutral, 1 very pleasant
- Pn (displeasure): 0 none, 0.5 neutral, 1 very unpleasant
- Dom (dominance):  0 powerless, 0.5 neutral, 1 fully in control

Current mood baseline (P Pn Dom): {mood}
Move from this baseline only as the exchange warrants.

Text:
{text}

Reply with exactly three decimals in order P Pn Dom, space-separated \
(e.g. "0.7 0.2 0.6"). Nothing else."""

# Conversation save prompt — distill what happened into one sentence
_SUMMARY_PROMPT = """\
Summarize this exchange in one sentence that captures the emotional core. \
Write in {lang}.
Speaker: {user}
Agent: {agent}

One sentence only."""

# Companion mood prompt — classify companion's emotional state from their message
_COMPANION_MOOD_PROMPT = """\
Read this message and pick the single best label for the sender's mood:
engaged / tired / frustrated / absent / happy

Message: {text}

Reply with the label only (one English word)."""


async def _evaluate_emotion_pad(backend, text: str, mood: "MoodPAD", arousal: float,
                                *, a_gate: float = A_GATE) -> "MoodPAD":
    """観測の感情を PAD で評価する（W2b-2）。

    A は機械 arousal。arousal<a_gate は評価器を呼ばず P/Pn/Dom＝M（mood）。以上は
    評価器（軽量LLM）へ投げ、固定順3数値を正規表現で拾い [0,1] クランプして
    MoodPAD(p, pn, a=arousal, dom) にする。3つ未満・例外は mood フォールバック。
    """
    a = min(1.0, max(0.0, arousal))
    if a < a_gate:
        return MoodPAD(p=mood.p, pn=mood.pn, a=a, dom=mood.dom)
    try:
        mood_str = f"{mood.p:.2f} {mood.pn:.2f} {mood.dom:.2f}"
        raw = await backend.complete(
            _EMOTION_PAD_PROMPT.format(text=text[:400], mood=mood_str), max_tokens=20
        )
        nums = re.findall(r"-?[0-9]*\.?[0-9]+", raw)
        if len(nums) < 3:
            raise ValueError("evaluator did not return three numbers")
        p, pn, dom = (min(1.0, max(0.0, float(nums[i]))) for i in range(3))
        return MoodPAD(p=p, pn=pn, a=a, dom=dom)
    except Exception as e:
        logger.warning("emotion PAD evaluation failed, falling back to mood: %s", e)
        return MoodPAD(p=mood.p, pn=mood.pn, a=a, dom=mood.dom)


def _companion_mood_heuristic(text: str) -> str:
    """Fast keyword-based mood classifier used when no dedicated utility backend exists.

    Covers the common explicit expressions; defaults to "engaged" for ambiguous text.
    Labels: engaged / tired / frustrated / absent / happy
    """
    t = text.lower()

    # tired
    if any(
        w in t
        for w in [
            "疲れ",
            "つかれ",
            "しんど",
            "眠い",
            "ねむ",
            "だるい",
            "きつい",
            "tired",
            "exhausted",
            "sleepy",
            "worn out",
            "drained",
        ]
    ):
        return "tired"

    # frustrated
    if any(
        w in t
        for w in [
            "むかつ",
            "いらいら",
            "うざ",
            "最悪",
            "ムカつ",
            "怒",
            "frustrated",
            "annoyed",
            "angry",
            "not working",
            "isn't working",
            "doesn't work",
            "won't work",
            "can't",
            "ugh",
            "argh",
        ]
    ):
        return "frustrated"

    # happy
    if any(
        w in t
        for w in [
            "嬉しい",
            "うれし",
            "楽しい",
            "たのし",
            "やったー",
            "やった",
            "最高",
            "最強",
            "好き",
            "すき",
            "幸せ",
            "しあわせ",
            ":)",
            "😊",
            "😄",
            "🎉",
            "笑",
            "www",
            "ｗ",
            "happy",
            "great",
            "perfect",
            "worked",
            "excellent",
            "wonderful",
            "love",
            "yay",
            "awesome",
        ]
    ):
        return "happy"

    # absent (very short / punctuation only)
    if len(text.strip()) < 4:
        return "absent"

    return "engaged"


class Evaluator:
    """軽量LLM を使うターン評価をまとめる。

    構築時に utility_backend（評価用の安い経路）と backend（主経路）を注入する。
    専用の utility backend が無い（utility is backend）環境では、余計な LLM 往復を
    避けて発見的手法やスキップに落とす。
    """

    def __init__(self, utility_backend, backend) -> None:
        self._utility_backend = utility_backend
        self.backend = backend

    async def emotion_for_turn(self, text: str, arousal: float) -> tuple[MoodPAD, str]:
        """ターンの感情を PAD で評価し、派生ラベルと合わせて返す（W2b-2）。

        現在の mood をベースに評価器（軽量LLM）が P/Pn/Dom を出し（A は機械 arousal・
        A_gate 未満は mood）、`label_from_pad` で最近傍ラベルを導く。mood は読みだけ。
        """
        pad = await _evaluate_emotion_pad(
            self._utility_backend, text, load_current_mood(), arousal
        )
        return pad, label_from_pad(pad)

    async def infer_companion_mood(self, text: str) -> str:
        """Classify companion's emotional state from their message. Returns mood label.

        When the utility backend is the same as the main backend (e.g. both are Kimi),
        uses a fast keyword heuristic to avoid an extra LLM round-trip every turn.
        Falls back to the LLM when a dedicated utility backend is configured.
        """
        if not text or len(text.strip()) < 3:
            return "absent"
        # Skip LLM call when utility == main backend (no dedicated cheap model available).
        if self._utility_backend is self.backend:
            return _companion_mood_heuristic(text)
        label = await self._utility_backend.complete(
            _COMPANION_MOOD_PROMPT.format(text=text[:300]), max_tokens=10
        )
        label = label.strip().lower()
        valid = {"engaged", "tired", "frustrated", "absent", "happy"}
        return label if label in valid else "engaged"

    async def check_response_coherence(self, response: str, messages: list) -> str | None:
        """Check whether the agent's response contains a logical error or rule violation.

        Uses the utility backend for a lightweight reflection pass.  Returns None if
        the response is coherent, or a short violation description if not.
        Skipped when no dedicated utility backend exists (same heuristic as TAPE).

        `messages` は生の会話履歴（ネスト list を含みうる）。走査前に flatten する。
        """
        if self._utility_backend is self.backend:
            return None
        if not response or response == "(no response)":
            return None

        # Build a compact context from the last few messages (user + assistant text only)
        context_parts: list[str] = []
        for msg in _flatten_history(messages[-6:]):  # tool結果はネストlist。走査前に展開
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                context_parts.append(f"{role}: {content[:200]}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        context_parts.append(f"{role}: {block['text'][:200]}")
                        break
        context = "\n".join(context_parts[-6:])

        try:
            result = await self._utility_backend.complete(
                _COHERENCE_CHECK_PROMPT.format(context=context, response=response[:300]),
                max_tokens=60,
            )
            result = result.strip()
            if result.upper().startswith("OK"):
                return None
            logger.info("Coherence check caught violation: %s", result)
            return result
        except Exception as e:
            logger.debug("Coherence check failed (non-critical): %s", e)
            return None

    async def summarize_exchange(self, user_input: str, agent_response: str) -> str:
        """Distill an exchange into one sentence for memory storage."""
        result = await self._utility_backend.complete(
            _SUMMARY_PROMPT.format(
                lang=_t("summary_lang"),
                user=user_input[:200],
                agent=agent_response[:200],
            ),
            max_tokens=80,
        )
        return result or agent_response[:100]
