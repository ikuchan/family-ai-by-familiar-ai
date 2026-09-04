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

from .._i18n import _t
from ..core.structured_ask import ask_choice, ask_numbers
from ..core.context_parts import Stance as _Stance
from ..emotion_pad import label_from_pad
from ..mood_register import MoodPAD, load_current_mood
from .history import _flatten_history

logger = logging.getLogger(__name__)

#: 相手の気分として選ばせる5つ。`_companion_mood_heuristic` が返す語と揃える。
_COMPANION_MOODS = frozenset({"engaged", "tired", "frustrated", "absent", "happy"})


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
# **感情を作るのはパジュである**（出-e・2026-09-05）。外から採点する計器ではないので、
# 「このやり取りを採点せよ」ではなく「自分が何を感じたか」を聞く。
#
# **評価や推論の指示を置かない。** 軸の目盛り・P と Pn が独立であること・平静の位置・
# いまの気分・出力の形式だけを渡す。旧版は「正直に見れば全部の軸が低くなる。中程度では
# ない」と**答えの側を指示**しており、場面のあいだの幅が 0.77 から 0.46 へ潰れた。
#
# **平静の位置は伝える。** 事実であって指示ではない。これが無いと中立の場面が 0.41 まで
# 上がる（伝えると 0.21 へ戻り、幅は保たれる・`根拠台帳` §25.8）。
#
# **誰も見ないことを伝える。** 人に読ませる言葉として整えると、角が丸まって真ん中へ寄る。
_EMOTION_PAD_PROMPT = """\
いまのやり取りで、あなた自身が何を感じたかを3つの数で書く。各 0.0〜1.0。

- P  （快）  ： 心地よさの大きさ。0＝まったく無い、1＝とても大きい
- Pn （不快）： 嫌さの大きさ。    0＝まったく無い、1＝とても大きい
- Dom（掌握）： 0＝どうにもできない、0.5＝ふつう、1＝すっかり手の内

P と Pn は別々の量で、1本の尺度の両端ではない。両方とも大きいことも、両方とも小さい
こともある。

この3軸は、何も起きていないとき P=0.10 / Pn=0.10 / Dom=0.50 へ落ち着く。それが平静である。
いまのあなたの気分（P Pn Dom）：{mood}

この3つの数は、あなた以外だれも見ない。人に読ませる言葉ではないので、整えず、
感じたままの大きさをそのまま書く。

やり取り：
{text}

P Pn Dom の順に小数を3つ、空白で区切って書く（例 "0.7 0.2 0.6"）。ほかには何も書かない。"""

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
                                *, a_gate: float = A_GATE,
                                system: str | None = None) -> "tuple[MoodPAD | None, float]":
    """観測の感情を PAD で評価する（W2b-2）。**測れたかどうかを返り値で表す**（050）。

    返すのは `(PAD, A)` で、**測れなかったときの PAD は `None`** である。A は機械 arousal
    なので常に返る（内容の新規性 novelty から作る）。

    `arousal < a_gate` は評価器を呼ばない。以上は評価器（軽量LLM）へ投げ、固定順3数値を
    正規表現で拾い [0,1] クランプして `MoodPAD(p, pn, a=arousal, dom)` にする。

    **測れないときに気分で埋めない**（050）。埋めた値は測ったものではないので、感情軸の
    母集合に混ぜてはいけない。実データでは、この「埋める」が 6433 行のうち 2941 行を
    PAD 全部 0.5＝ゼロベクトルに潰し、感情軸が候補を並べ替えられなくなっていた。
    0.5 が入っていると「測ったのか埋めたのか」を後から見分けられず、REST 内省が埋め直す
    余地も消える。3つ未満・例外も同じく未測定にする（測れなかったのは同じである）。
    """
    a = min(1.0, max(0.0, arousal))
    if a < a_gate:
        return None, a
    mood_str = f"{mood.p:.2f} {mood.pn:.2f} {mood.dom:.2f}"
    # 数値を拾って範囲へ丸めるのは口が持つ（出-d）。3つ揃わなければ `None` が返り、
    # ここは未測定として扱う。呼び出しが落ちても口は例外を投げない。
    nums = await ask_numbers(
        backend,
        _EMOTION_PAD_PROMPT.format(text=text[:400], mood=mood_str),
        count=3,
        max_tokens=20,
        system=system,
    )
    if nums is None:
        return None, a
    p, pn, dom = nums
    return MoodPAD(p=p, pn=pn, a=a, dom=dom), a


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

    def __init__(self, utility_backend, backend, *, context=None) -> None:
        """`context(stance, *, with_rules)` は立ち位置と文脈を返す（出-e）。

        渡さなければ立ち位置を渡さない（いままでと同じ）。材料が欠けたときも `None` を
        返してよい——`FAMILY.md` が無い機体でターンを落とさない。
        """
        self._utility_backend = utility_backend
        self.backend = backend
        self._context = context

    def _stance(self, stance, *, with_rules: bool = False) -> "str | None":
        """立ち位置と文脈を引く。提供者が無ければ `None`。"""
        if self._context is None:
            return None
        return self._context(stance, with_rules=with_rules)

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
        # **感情を作るのはパジュである**（出-e）。外から採点する計器ではない。
        pad, a = await _evaluate_emotion_pad(
            self._utility_backend, text,
            load_current_mood() if mood is None else mood, arousal,
            system=self._stance(_Stance.PAJU),
        )
        return pad, a, ("neutral" if pad is None else label_from_pad(pad))

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
        # 家族の気持ちを読むのはパジュがすることなので、一人称で立つ。
        label = await ask_choice(
            self._utility_backend,
            _COMPANION_MOOD_PROMPT.format(text=text[:300]),
            choices=_COMPANION_MOODS,
            max_tokens=10,
            system=self._stance(_Stance.PAJU),
        )
        # **読めなかったときに「乗り気」と断定しない**（出-d）。同じファイルにある語ベースの
        # 判定へ落とす。「読めなかったから既定」より根拠がある。
        return label if label is not None else _companion_mood_heuristic(text)

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
            # **自分で自分は検査できない。** ここだけ外から測る立ち位置で、規則を
            # システム文で受け取る（規則の正本は `EVENT_SYSTEM_PROMPT` の `(rules ...)`）。
            result = await self._utility_backend.complete(
                _COHERENCE_CHECK_PROMPT.format(context=context, response=response[:300]),
                max_tokens=60,
                system=self._stance(_Stance.INSTRUMENT, with_rules=True),
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
