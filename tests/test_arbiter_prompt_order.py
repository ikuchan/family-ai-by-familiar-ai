"""調停プロンプトの並び：固定のものを先、変わるものを後へ。

プロンプトキャッシュは**前方一致**で効く。変わりうるものが固定のものより前にあると、それが
変わるたびに後ろ全部が作り直しになる。

実機で、調停が 2 秒で返らず時間切れになった（沈黙依頼が読まれないまま `full` へ倒れた）。
渡すものを増やしてきたので、キャッシュが効く形に組み直す。

名前は `[あなたは誰か]`（`ME.md`）に既に入っている。同じものを別枠でもう一度渡さない。
"""

from __future__ import annotations

from familiar_agent.loop.arbiter import ARBITER_PROMPT


def _pos(marker: str) -> int:
    i = ARBITER_PROMPT.find(marker)
    assert i >= 0, f"見つからない: {marker}"
    return i


def test_who_you_are_comes_before_anything_that_changes():
    # `[あなたは誰か]` と `[一緒に暮らす人たち]` は起動中ほぼ変わらない。
    assert _pos("[あなたは誰か]") < _pos("[いま]")
    assert _pos("[一緒に暮らす人たち]") < _pos("[いま]")


def test_the_iteration_cap_note_is_not_in_the_stable_part():
    """上限の但し書きは5反復に1回だけ現れる（通常は空文字）。

    固定のものより前に置くと、現れた回に後ろ全部が作り直しになる。
    """
    assert _pos("{capped_note}") > _pos("[あなたは誰か]")
    assert _pos("{capped_note}") > _pos("[一緒に暮らす人たち]")


def test_what_the_person_said_comes_last():
    # 毎回変わるものほど後ろ。
    assert _pos("[人の言葉]") > _pos("[いま誰が居るか]")
    assert _pos("[いまの作業状態]") > _pos("[人の言葉]")


def test_the_name_is_not_passed_separately():
    # `ME.md` の「名前： …」が `[あなたは誰か]` に入っている。二重に渡さない。
    assert "{agent_name}" not in ARBITER_PROMPT


def test_the_silence_rule_points_at_who_you_are():
    # 名前を別枠で渡さない代わりに、どこに書いてあるかを指す。
    assert "あなたは誰か" in ARBITER_PROMPT.split("silence_minutes")[0][-600:]
