"""沈黙依頼は、名前を呼ばれたときだけ受ける（長さの指定つき）。

周囲の会話が書き起こされてターンを起こしている（実機で観測：本人の「パジュ黙って」は届かず、
別の人の会話が入力になっていた）。名前を呼ばれていない「黙って」まで拾うと、無関係な会話で
黙り込む。**呼びかけ語を要求するのは沈黙依頼だけ**で、普通の会話は従来どおり名前なしで通る。

長さも受け取る。真偽値では「5分だけ黙って」に応えられず、いつも既定（60分）になっていた。
`silence` を数値（分）へ広げ、**0 を「黙らない」**に当てる（項目を増やさない）。

名前は `ME.md` が持つ（`.env` の `AGENT_NAME` は撤去した）。
"""

from __future__ import annotations

from familiar_agent.loop.arbiter import Arbiter, ArbiterInput
from familiar_agent.loop.arbiter import assemble


# --- 調停の返り値 ---------------------------------------------------------


def test_a_silence_request_carries_its_length():
    d = assemble({"branch": "light", "text": "はい", "silence_minutes": 5})
    assert d.silence_minutes == 5


def test_no_request_means_zero_minutes():
    d = assemble({"branch": "light", "text": "はい"})
    assert d.silence_minutes == 0


def test_a_request_without_a_length_falls_back_to_the_default():
    # 「黙って」とだけ言われた場合。軽量LLM は既定値を知らないので、-1 で「長さの指定なし」
    # を表し、受け側が Config の既定（15分）を当てる。
    d = assemble({"branch": "light", "text": "はい", "silence_minutes": -1})
    assert d.silence_minutes == -1


def test_a_broken_length_does_not_silence():
    # 読めない値で黙り込むと、解けるまで何も言えなくなる。
    d = assemble({"branch": "light", "text": "はい", "silence_minutes": "ずっと"})
    assert d.silence_minutes == 0


# --- Jev への問い（出-au 段 5-7d）------------------------------------------


def _questions(**kw):
    """発話の問い（段 4-4c から、黙る依頼は意味「時間に関する依頼」の動作「quiet」と、何分かの問い）。"""
    from familiar_agent.core import utterance_meaning as um

    return um.fanout_questions(confirming=False, music=False, camera=False, family=[])


def test_the_name_is_not_passed_separately():
    """名前は `ME.md`（「名前： …」）にある。Jev に送る文へ別枠で渡さない。"""
    state = Arbiter(jev=None, writer=None)._state(ArbiterInput(utterance="x", workspace_ctx=""))
    assert "{agent_name}" not in state


def test_the_name_is_checked_by_the_window_not_by_the_question():
    """名前の関門は入口の窓（名前を聞いてから 30 秒）にまとめた（出-as §2.5）。Jev は名前を知らないので問わない。"""
    asked = _questions()["action_time"]["criteria"]["quiet"]
    assert "名前" not in asked
    assert "黙っていて" in asked


def test_the_question_asks_for_a_length_rather_than_a_flag():
    minutes = _questions()["quiet_minutes"]
    assert minutes["type"] == "choice"
    assert "default" in minutes["criteria"]  # 長さを言っていない → 既定


# --- 長さの適用 -----------------------------------------------------------


def test_an_unspecified_length_becomes_the_configured_default():
    from familiar_agent.silence_state import resolve_minutes

    assert resolve_minutes(-1, default=15, maximum=60) == 15


def test_a_specified_length_is_used_as_asked():
    from familiar_agent.silence_state import resolve_minutes

    assert resolve_minutes(5, default=15, maximum=60) == 5


def test_a_length_beyond_the_cap_is_rounded_down_rather_than_refused():
    """「3時間黙って」に黙らないより、上限まで黙るほうが意図に近い。"""
    from familiar_agent.silence_state import resolve_minutes

    assert resolve_minutes(180, default=15, maximum=60) == 60


def test_zero_means_no_request():
    from familiar_agent.silence_state import resolve_minutes

    assert resolve_minutes(0, default=15, maximum=60) == 0
