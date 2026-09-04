"""文脈を組む口（出-e）。モデルへ**何を与えるか**の形を揃える。

出-d が「何を求めるか」（`structured_ask`）を揃えたのに対し、ここは与える側である。

同じ組み立てが `loop/prompt.py` の `build_event_system_prompt` と `loop/arbiter.py` の
`ARBITER_PROMPT` にあり、**同じ不変条件を別々に手で守っていた**。起動中ほぼ変わらないもの
（人格・家族・規則）を先に置き、変わるもの（時刻・在席・反復・作業状態）を後ろへ置く。
前方一致キャッシュが効く条件で、崩れると実機で調停が時間切れになり沈黙依頼が読まれなかった。

**部品の中身はここで持たない。** 人格は `ME.md`、できることは `capability_state.load_summary()`、
家族は `FAMILY.md`、規則は `loop.prompt.rules_section()` が正本である。呼ぶ側がそこから取って
渡す。ここに写しを置くと、正本が変わったときにここだけ古くなる。

ここが持つのは**立ち位置の言葉**（既存のどこにも無い）と**並びの規則**だけである。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

#: パジュとして立つときの言葉。**感情を作るのはパジュである**（2026-09-05 決定）。
#: 一人称は単独では成り立たないので、自己認識と家族を必ず伴う（`build_context` が確かめる）。
_AS_PAJU = "あなたはパジュである。以下は、あなた自身と、あなたが暮らす家族の記述である。"

#: 外から測るときの言葉。**自分で自分は検査できない**ので、整合チェックはこちらで立つ。
#: パジュとして整合チェックをさせると、違反18件中0〜1件しか捕まえなかった（根拠台帳 §25.8）。
_AS_INSTRUMENT = (
    "あなたは、パジュという伴侶エージェントのやり取りを読んで判定する計器である。\n"
    "パジュ自身ではない。パジュとして答えず、道具を呼ばず、問いが求めた形式以外で答えない。"
)


class Stance(Enum):
    """どの立ち位置で読むか。仕事によって分かれる。"""

    #: PAD 評価・満たされた欲求・一言要約・相手の気分の分類・REST の感情の再評価。
    PAJU = "paju"
    #: 整合チェック・同一意図の判定。
    INSTRUMENT = "instrument"


@dataclass(frozen=True)
class Context:
    """組み上がった文脈。**安定部と可変部に分けて返す**。

    1本の文字列にすると、backend がキャッシュの境目を置けない。Anthropic の
    `cache_control` を付ける先が `stable` である。
    """

    stable: str
    variable: str


def build_context(
    *,
    stance: Stance,
    core: str = "",
    self_understanding: str = "",
    family: str = "",
    rules: str = "",
    now: str = "",
    presence: str = "",
    inner_state: str = "",
    iteration: str = "",
    workspace: str = "",
) -> Context:
    """部品を選び、安定 → 可変の順に組む。

    並びは呼ぶ側が決められない。順序は前方一致キャッシュが効く条件であり、呼ぶたびに手で
    守らせると、いつか崩れる。可変部は **いま → 在席 → 内部状態 → 反復 → 作業状態**。

    `stance=PAJU` は `self_understanding` と `family` を要る。自分が誰で誰と暮らして
    いるかを知らなければ、パジュにはなれない。

    **`core`（静的核）を渡すなら、立ち位置の一文は置かない。** 静的核の `(identity ...)` が
    同じことを厚く言っており、二度書くことになる。立ち位置の一文は、静的核を渡さない相手
    （軽量LLM）のための代用である。

    可変部の各部品は**自分で名乗る**（`(now …)`・`(present …)`・`[内部状態(PI)]`・
    `[反復]`・`[過去の記憶…]`）ので、ここでは見出しを足さない。
    """
    # 要求が効くのは、立ち位置の一文しか身元が無いとき（＝静的核を渡さない軽量LLM）。
    # 静的核があれば `(identity ...)` が身元を担うので、欠けていても組める。`FAMILY.md` が
    # 無い機体や、自己認識をまだ生成していない初回起動で、主LLM のターンごと落とさない。
    if stance is Stance.PAJU and not (core and core.strip()):
        missing = [n for n, v in (("自己認識", self_understanding), ("家族", family)) if not v.strip()]
        if missing:
            raise ValueError(
                "パジュとして立つには " + "と".join(missing) + " が要る（一人称は単独では成り立たない）"
            )

    stable_parts: list[str] = []
    if core and core.strip():
        stable_parts.append("[身体と決まり]\n" + core.strip())
    else:
        stable_parts.append(_AS_PAJU if stance is Stance.PAJU else _AS_INSTRUMENT)
    for label, text in (
        ("[あなたは誰か]", self_understanding),
        ("[一緒に暮らす人たち]", family),
        ("[守っている決まり]", rules),
    ):
        if text and text.strip():
            stable_parts.append(label + "\n" + text.strip())

    variable_parts = [
        t.strip() for t in (now, presence, inner_state, iteration, workspace) if t and t.strip()
    ]

    return Context(
        stable="\n\n".join(stable_parts),
        variable="\n\n".join(variable_parts),
    )
