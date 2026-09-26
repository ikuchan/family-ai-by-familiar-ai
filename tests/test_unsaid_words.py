"""言いたかったのに言えなかったこと（出-as 段 7・2026-09-26・`設計方針_話していいかの決まり` §2.6）。

話そうとして止められた発話（誰も映っていない情動・窓が切れた後の返事）は、保留して後で配るのをやめ、独り言
として記録する。想起で上がってくるように、**相手の面**に「〇〇に言いたかったこと：…」として書き、根づきを 2 に
する（相手が分からなければパジュ自身の面に「誰かに言いたかったこと：…」）。返事を声に出した求めで、その記録を
「使った」と申告したら、**畳んで根づきを普通に戻す**。
"""

from __future__ import annotations

from types import SimpleNamespace

from familiar_agent.core import unsaid


def test_the_words_name_who_it_was_for():
    assert unsaid.content("パパ", "明日は雨だよ") == "パパに言いたかったこと：明日は雨だよ"


def test_an_unknown_listener_is_someone():
    assert unsaid.content("", "明日は雨だよ") == "誰かに言いたかったこと：明日は雨だよ"


def test_the_weight_is_two():
    assert unsaid.GROUNDEDNESS == 2


def test_it_is_recognised_and_other_monologues_are_not():
    assert unsaid.is_unsaid("パパに言いたかったこと：明日は雨だよ")
    assert unsaid.is_unsaid("[そばに居た] 誰かに言いたかったこと：ねえ")  # 想起の頭が付いても
    assert not unsaid.is_unsaid("考えたが言わなかった：静かにしておこう")
    assert not unsaid.is_unsaid("伝えた：パパに言いたかったこと：明日は雨だよ")  # 畳んだあと


def _rec(obs_id, content):
    return SimpleNamespace(mi=SimpleNamespace(obs_id=obs_id, content=content))


def test_the_told_ones_are_picked_from_the_verdicts():
    memories = [
        _rec("aaaaaaaaaaaa-1", "パパに言いたかったこと：明日は雨だよ"),
        _rec("bbbbbbbbbbbb-2", "パパに言いたかったこと：金木犀が咲いた"),
        _rec("cccccccccccc-3", "昨日の運動会"),
    ]
    w_id_map = {
        "aaaaaaaaaaaa": "aaaaaaaaaaaa-1",
        "bbbbbbbbbbbb": "bbbbbbbbbbbb-2",
        "cccccccccccc": "cccccccccccc-3",
    }
    raw = [
        {"id": "aaaaaaaaaaaa", "verdict": "referred"},  # 使った → 畳む
        {"id": "bbbbbbbbbbbb", "verdict": "unused"},  # 使っていない → 残す
        {"id": "cccccccccccc", "verdict": "important"},  # 言いたかったことではない
    ]
    assert unsaid.told(raw, w_id_map, memories) == [
        ("aaaaaaaaaaaa-1", "パパに言いたかったこと：明日は雨だよ")
    ]


def test_nothing_is_told_without_verdicts():
    assert unsaid.told(None, {}, []) == []
