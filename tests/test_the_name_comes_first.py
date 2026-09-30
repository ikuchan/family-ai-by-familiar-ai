"""名前で呼ばれたとみなすのは、文頭に名前があるときだけ（2026-09-30・本人の決定）。

以前は文の**どこかに**名前があれば当たり、3 字以上の名前は 1 字違いまで許していたので、テレビの台詞や
名前を話題にしただけの言葉で窓が開いた。残っている書き起こし 662 件のうち名前に当たった 96 件で、
文頭に無かった 10 件はどれも呼ばれていない誤検出だった。文頭の前の空白と記号は飛ばす。呼びかけの
言葉（ねえ・あのさ など）は飛ばさない（本人の決定「許さない」）。窓の門・黙る依頼を解く言葉・
鳴っているあいだの音楽の門が、同じ判定を使う。
"""

from __future__ import annotations

import pytest

from familiar_agent.core.silence_hold import lifts
from familiar_agent.core.silence_rules import names_me
from familiar_agent.core.wake_window import heard_name

NAMES = ["パジュ"]

#: 実機の書き起こしで、文頭に名前が無いのに当たっていたもの（2026-09-20〜26 のログ）。
MISHEARD = [
    "それを吹きかけるわけないシュートにも",
    "スパークはけないシュートを1本という",
    "聞いておいているパジュのことです。",
    "そして、アジア記録保持者のオージュ 選手もいましたね、すごい。",
    "聞いてくれないパチューが出ました",
    "はい、いえい、もはじめと",
    "めん、おどん、まい、げん、む、はじめん",
    "オートキョウジュ!オートキョウジュ!",
    "エンドウトク、ジュッパイ、ジュニティスター",
    "さっきこそ本当にパチューが使いこなしてた",
]


@pytest.mark.parametrize(
    "text",
    [
        "パジュ、明日の天気は？",
        "、パジュ、聞こえる？",
        "　パジュ 3分測って",
        "はじゅ、静かにして",
        "バジュ、聞こえる？",
        "パジュー、黙って",
    ],
)
def test_a_name_at_the_head_is_heard(text):
    assert names_me(text, NAMES)
    assert heard_name(text, NAMES)


@pytest.mark.parametrize(
    "text",
    [
        "明日の天気、パジュに聞いてみよう",
        "静かにしてぱじゅ",
        "ねえパジュ、聞こえる？",
        "あのさ、パジュ",
    ],
)
def test_a_name_elsewhere_is_not_heard(text):
    assert not names_me(text, NAMES)
    assert not heard_name(text, NAMES)


@pytest.mark.parametrize("text", MISHEARD)
def test_what_was_misheard_is_no_longer_heard(text):
    assert not heard_name(text, NAMES)


def test_lifting_needs_the_name_at_the_head():
    assert lifts("会話入力", "パジュ、話していいよ", names=NAMES)
    assert not lifts("会話入力", "話していいよ、パジュ", names=NAMES)
