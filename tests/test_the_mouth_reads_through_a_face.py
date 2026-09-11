"""**読むときの視点を、口が受け取れるようにする**（環-e-い・想起は後）。

`situated_memories` は人ごとで、想起は**話者の面**を通る（`agent._active_memory()`）。
ところが `agent._oif` が持つのは**基底の記憶**で、視点は `__self__` に寄る。そのまま
`agent._oif.recall(...)` へ呼び替えると、**想起が話者の面から `__self__` の面へ移る**——
記-f で直したばかりのところを、逆向きに壊すことになる。

`View` は docstring で「**読むときの視点**」と名乗っていたのに、欄には**誰の面かが無かった**
（`k`・`floor`・`weights`・`present`・`time_ref`・`time_span_days`）。足す。

**呼び替えはここではやらない。** 器が違う（`Recalled` 対 辞書の並び）ので、`compose`・
`format_for_context`・申告の突き合わせ・共起・`link_follows` の5つが動く。口が正しい面を
引けるようにするところまでを、この段で扱う。
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.io.oif import OIF, Cue, View


def test_the_view_carries_whose_face_to_read_through():
    assert "viewpoint" in View.__dataclass_fields__
    assert View().viewpoint == ""  # 既定は「口が持つ記憶のまま」


def test_without_a_viewpoint_it_reads_through_the_memory_it_holds():
    base = MagicMock(recall_async=AsyncMock(return_value=[]))
    oif = OIF(base)
    asyncio.run(oif.recall(Cue(text="手がかり")))
    base.recall_async.assert_awaited_once()


def test_with_a_viewpoint_it_reads_through_that_persons_face():
    """**人ごとの実体は呼び手が持つ。** `pmm` が1人につき1つ持っており、口が
    `for_person` を都度呼ぶと実体が増える。"""
    base = MagicMock(recall_async=AsyncMock(return_value=[]))
    face = MagicMock(recall_async=AsyncMock(return_value=[]))
    resolve = MagicMock(return_value=face)
    oif = OIF(base, for_person=resolve)
    asyncio.run(oif.recall(Cue(text="手がかり"), View(viewpoint="ゆうすけ")))
    resolve.assert_called_once_with("ゆうすけ")
    face.recall_async.assert_awaited_once()
    base.recall_async.assert_not_awaited()


def test_without_a_resolver_it_falls_back_to_the_memorys_own_view():
    base = MagicMock(recall_async=AsyncMock(return_value=[]))
    base.for_person.return_value = MagicMock(recall_async=AsyncMock(return_value=[]))
    oif = OIF(base)
    asyncio.run(oif.recall(Cue(text="手がかり"), View(viewpoint="ゆうすけ")))
    base.for_person.assert_called_once_with("ゆうすけ")


def test_the_agent_hands_the_mouth_the_cached_faces():
    """`pmm` が人ごとに1つ持つ実体を渡す（作り直さない）。"""
    from familiar_agent.agent import EmbodiedAgent

    src = inspect.getsource(EmbodiedAgent.__init__)
    assert "for_person=self._pmm.get_memory_for" in src


def test_the_recall_is_not_switched_over_yet():
    """**残したものを黙って落とさない。** 呼び替えは器の移行とセットで行う。"""
    import pathlib

    w = (
        pathlib.Path(__file__).resolve().parent.parent
        / "src"
        / "familiar_agent"
        / "loop"
        / "workspace.py"
    ).read_text(encoding="utf-8")
    assert "mem.recall_async(" in w, "想起が口へ移っている（残りの記述を直すこと）"
