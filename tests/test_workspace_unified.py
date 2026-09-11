"""W の組み立てを候補集合へ統合する（改造方針 S4）。

正本 [D-想起起動] は「O に乗った後は共通の流れ（O → 根づき → W 構築〔5軸採点〕→ 調停）
で1本」と定める。実装は、想起で拾った記録だけが union と採点を通り、ループ自身が O へ
書いた記録（意図 O・完了 O）は採点を通らず**手組みの文字列**として W へ連結されていた。
そのため記録が W に載るかどうかが「畳むか畳まないか」で決まり、優先度の計算が効かなかった。

ここでは (1) 取込 O を候補から外さないこと、(2) 手組みの連結が消えたこと、(3) 1件の途中で
切らないこと、(4) 字数枠を超えたら適合度の低い件から丸ごと落とすこと、(5) 落とした件数を
残すこと、を見る。
"""

from __future__ import annotations


from familiar_agent.loop import workspace
from familiar_agent.loop.request import Request

import logging

import pytest

from familiar_agent.backends import ToolCall
from familiar_agent.config import MemoryConfig

from tests.test_event_loop import _agent, _run, _turn


def _rec(obs_id="m1", content="昔の話", fit=0.5, conf=0.8, direction="発話"):
    """想起は口から `Recalled` で来る（環-e-い）。"""
    from datetime import datetime

    from familiar_agent.io.oif import MI, Recalled

    return Recalled(
        mi=MI(
            id=obs_id,
            obs_id=obs_id,
            content=content,
            timestamp=datetime(2026, 9, 11, 15, 0),
            direction=direction,
        ),
        fit=fit,
        groundedness=1.0,
        confidence=conf,
    )


def test_workspace_char_budget_default() -> None:
    """W 全体の字数枠の既定は 40000。

    1つの求めで届きうる完了 O が全部同時に全文で載る大きさから決めた。反復上限 5 で
    5反復目は調べられないので完了は最大4件、1件の上限は 8192 字で 32768 字。余裕を見た値。
    """
    assert MemoryConfig().workspace_max_chars == 40000


def test_intake_is_not_excluded_from_recall() -> None:
    """取込 O を想起の候補から外さない。

    手がかりは取込の content そのものなので、候補に入れば必ず上位に来る。これは枠を
    食っているのではなく、いちばん重要なものが1位に来ている状態である。
    """
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    _run(a, utterance="おはよう")
    kwargs = a._active_memory().recall_async.call_args.kwargs
    assert not kwargs.get("exclude_ids"), "取込 O を候補から外している"


def test_handmade_blocks_are_gone() -> None:
    """調べたものの一覧と待ち一覧の手組みが消えている。

    中身は意図 O と完了 O として O にあるので、候補集合の一員として W へ入る。
    """
    import inspect

    src = inspect.getsource(workspace.compose)
    for gone in ("この求めのために調べたもの", "いま返事を待っている調べもの"):
        assert gone not in src, f"手組みの文字列『{gone}』が残っている"


def _oif_stub():
    """主体を返さない口（W の1行は核が組む・環-e-い）。"""
    from unittest.mock import MagicMock

    return MagicMock(actors=MagicMock(return_value={}))


def test_items_are_never_truncated() -> None:
    """1件の途中で切らない。

    切ると、調べた結果の枕だけが残って中身が消える（実機で `「目の前を見る」を see で
    調べた結果が届いた：` だけが W に載った）。
    """

    long_body = "あ" * 3000
    memories = [_rec("m1", long_body, fit=0.9)]
    out, _ = workspace.compose(_oif_stub(), memories, Request())
    # W の1行は 120 字で丸める（`_lines`）。**枠で落とさなかった**ことを見る。
    assert "あ" * 120 in out, "1件が枠で落とされている"


def test_overflow_drops_whole_items_lowest_fit_first(caplog) -> None:
    """字数枠を超えたら、適合度の低い件から丸ごと落とす。落とした件数を残す。"""

    budget = MemoryConfig().workspace_max_chars
    big = "大" * (budget // 2 + 100)  # 2件で枠を超える大きさ
    # 印は**先頭**に置く。W の1行は 120 字で丸めるので、末尾の印は出ない。
    memories = [_rec("hi", "上位" + big, fit=0.9), _rec("lo", "下位" + big, fit=0.1)]
    with caplog.at_level(logging.INFO, logger="familiar_agent.loop.workspace"):
        out, _ = workspace.compose(_oif_stub(), memories, Request())

    assert "上位" in out, "適合度の高い件が落ちている"
    assert "下位" not in out, "枠を超えたのに落ちていない"
    assert len(out) <= budget + 400, "枠を大きく超えている"
    assert any("W に入らなかった" in r.getMessage() for r in caplog.records), (
        "落とした件数がログに残っていない"
    )


def test_within_budget_keeps_everything() -> None:
    """枠に収まるなら何も落とさない（普通の会話では当たらない）。"""

    memories = [_rec(f"m{i}", f"みじかい記憶{i}") for i in range(7)]
    out, _ = workspace.compose(_oif_stub(), memories, Request())
    for i in range(7):
        assert f"みじかい記憶{i}" in out, f"{i} 件目が落ちている"


@pytest.mark.parametrize("budget", [1000, 20000])
def test_budget_is_configurable(monkeypatch, budget) -> None:
    """字数枠は Config（env）で差し替えられる。"""
    monkeypatch.setenv("WORKSPACE_MAX_CHARS", str(budget))
    assert MemoryConfig().workspace_max_chars == budget
