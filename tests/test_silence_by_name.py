"""沈黙依頼は、名前を呼ばれたときだけ受ける（長さの指定つき）。

周囲の会話が書き起こされてターンを起こしている（実機で観測：本人の「パジュ黙って」は届かず、
別の人の会話が入力になっていた）。名前を呼ばれていない「黙って」まで拾うと、無関係な会話で
黙り込む。**呼びかけ語を要求するのは沈黙依頼だけ**で、普通の会話は従来どおり名前なしで通る。

長さも受け取る。真偽値では「5分だけ黙って」に応えられず、いつも既定（60分）になっていた。
`silence` を数値（分）へ広げ、**0 を「黙らない」**に当てる（項目を増やさない）。

名前は `ME.md` が持つ（`.env` の `AGENT_NAME` は撤去した）。
"""

from __future__ import annotations

from familiar_agent.loop.arbiter import ARBITER_PROMPT, _parse


# --- 調停の返り値 ---------------------------------------------------------


def test_a_silence_request_carries_its_length():
    d = _parse('{"branch":"light","text":"はい","silence_minutes":5}')
    assert d.silence_minutes == 5


def test_no_request_means_zero_minutes():
    d = _parse('{"branch":"light","text":"はい"}')
    assert d.silence_minutes == 0


def test_a_request_without_a_length_falls_back_to_the_default():
    # 「黙って」とだけ言われた場合。軽量LLM は既定値を知らないので、-1 で「長さの指定なし」
    # を表し、受け側が Config の既定（15分）を当てる。
    d = _parse('{"branch":"light","text":"はい","silence_minutes":-1}')
    assert d.silence_minutes == -1


def test_a_broken_length_does_not_silence():
    # 読めない値で黙り込むと、解けるまで何も言えなくなる。
    d = _parse('{"branch":"light","text":"はい","silence_minutes":"ずっと"}')
    assert d.silence_minutes == 0


# --- プロンプト -----------------------------------------------------------


def test_the_name_comes_from_who_you_are_rather_than_a_separate_slot():
    """名前は `ME.md`（「名前： …」）にあり、`[あなたは誰か]` に丸ごと入っている。

    別枠でもう一度渡すと同じ情報が2箇所になる。規則のほうから、そこを指す。
    """
    assert "{agent_name}" not in ARBITER_PROMPT
    assert "{me}" in ARBITER_PROMPT


def test_the_prompt_requires_the_name_for_a_silence_request():
    # 呼ばれていなければ黙らない、と明記されていること。
    assert "呼ばれ" in ARBITER_PROMPT


def test_the_prompt_asks_for_minutes_rather_than_a_flag():
    assert "silence_minutes" in ARBITER_PROMPT
    assert '"silence"' not in ARBITER_PROMPT        # 旧い真偽値の項目が残っていないこと


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
