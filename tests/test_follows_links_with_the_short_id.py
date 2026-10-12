"""続き先の判定が返す短い id を、W の対応表で O の完全な id に直してから辺を張る（出-bo・2026-10-12）。

判定は W の行の 12 桁の短い id を返すが、`link_follows` はそれを対応表の値（完全な id）と比べていた。
本物の id は長いので一度も一致せず、9/14 から計測ログの「続き」は 0 件・「不一致」が 173 件だった。
試験の偽の id（`m1`）は短い形と完全な形が同じ文字列で、これが見えなかった。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop import workspace
from familiar_agent.store.relations import KIND_SUCCESSION

PAST = "2449f300-6662-4c1a-9b7e-0d3f5a6b7c8d"
ORIGIN = "4b5becd5-39d0-4e2f-8a1b-2c3d4e5f6a7b"


def _short(full: str) -> str:
    return full.replace("-", "")[:12]


def _agent():
    return SimpleNamespace(_oif=MagicMock())


def test_the_short_id_is_turned_into_the_full_one_before_linking():
    a = _agent()
    req = SimpleNamespace(request_id=ORIGIN)
    w_id_map = {_short(PAST): PAST, _short(ORIGIN): ORIGIN}
    assert workspace.link_follows(a, req, w_id_map, _short(PAST)) is True
    a._oif.link.assert_called_once_with(KIND_SUCCESSION, [(PAST, "前", 0), (ORIGIN, "後", 1)])


def test_an_id_not_in_the_workspace_is_not_linked():
    """**反証の側**：対応表に無い id は張らない。"""
    a = _agent()
    req = SimpleNamespace(request_id=ORIGIN)
    assert workspace.link_follows(a, req, {_short(ORIGIN): ORIGIN}, _short(PAST)) is False
    a._oif.link.assert_not_called()


def test_its_own_origin_is_not_linked():
    """**反証の側**：自分の起点を指したら張らない（さかのぼりが止まらなくなる）。"""
    a = _agent()
    req = SimpleNamespace(request_id=ORIGIN)
    w_id_map = {_short(PAST): PAST, _short(ORIGIN): ORIGIN}
    assert workspace.link_follows(a, req, w_id_map, _short(ORIGIN)) is False
    a._oif.link.assert_not_called()


def test_a_real_shaped_id_is_recorded_as_a_continuation(tmp_path):
    """ループを通して、本物の形の id で「続き」と残る。"""
    from tests.test_the_judgement_outcome_goes_to_the_rest_log import _loop, _outcomes, _run

    a, path = _loop(tmp_path, verdict=_short(PAST))
    a._active_memory().recall_async = AsyncMock(
        return_value=[{"memory_id": PAST, "summary": "明日の天気", "fit": 0.9, "confidence": 0.8}]
    )
    _run(a, utterance="さっきの話だけど")
    assert _outcomes(path) == ["続き"]
