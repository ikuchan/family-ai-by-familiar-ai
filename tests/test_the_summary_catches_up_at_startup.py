"""`ME.md` を書き換えたら、次の起動で `[あなたは誰か]` に届く（環-x・2026-09-24）。

実機 16:01、パジュは「**どの県にいるかまでは分からない**」と答えた。`ME.md` には
`茨城県守谷市本町の家。1 階の和室に居る。` と書いてある。

主LLM が見ているのは `ME.md` ではなく**要約**である（`load_summary() or self._me_md`）。
その要約は **2026-09-13 に書かれたきりの 749 字**で、9/21 に書き直した `ME.md` が一つも
入っていなかった。作り直すのは REST 内省の層 4 で、それが回るのは「**誰も居ない晩**」だけ
だからである（層 4 は実機で一度も回っていない・`課題8` C-12）。

`ME.md` を書き換えるのは**アプリを止めているとき**なので、**起動が届ける機会になる**。
出-ao で「`ME.md` が書き換わった」を引き金に足したが、引き金を見る機会が夜しか無かった。

**一覧（`capabilities.yaml`）には触らない。** 道具の増減で変わるもので、起動とは関係がない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from familiar_agent.loop import rest_capabilities as rc

ME = "名前：パジュ\n一人称：ぼく"


def _agent(me_md: str = ME):
    a = MagicMock()
    a._me_md = me_md
    a.backend = MagicMock()
    a.backend.complete = AsyncMock(return_value=me_md + "\n\n## 私にできること\n- 記憶を探せる")
    return a


# ── 古ければ作り直し、新しければ触らない ──────────────────────────────────


def test_an_old_summary_is_rebuilt():
    """要約の先頭が `ME.md` と食い違えば古い（出-ao の判じ方をそのまま使う）。"""
    with (
        patch.object(rc, "load_summary", return_value="まったく別の古い要約"),
        patch.object(rc, "load_capabilities", return_value="capabilities:\n  - id: memory\n"),
        patch.object(rc, "save_summary") as save,
    ):
        reason = asyncio.run(rc.catch_up_summary(_agent()))
    assert reason is None
    save.assert_called_once()


def test_a_fresh_summary_costs_nothing():
    """**LLM を呼ばない。** `ME.md` が変わっていない起動で費用を払わない。"""
    a = _agent()
    with (
        patch.object(rc, "load_summary", return_value=ME + "\n\n## 私にできること\n- あ"),
        patch.object(rc, "save_summary") as save,
    ):
        reason = asyncio.run(rc.catch_up_summary(a))
    assert reason is None
    save.assert_not_called()
    a.backend.complete.assert_not_called()


def test_a_missing_summary_is_built():
    with (
        patch.object(rc, "load_summary", return_value=""),
        patch.object(rc, "load_capabilities", return_value="capabilities:\n  - id: memory\n"),
        patch.object(rc, "save_summary") as save,
    ):
        reason = asyncio.run(rc.catch_up_summary(_agent()))
    assert reason is None
    save.assert_called_once()


def test_a_refused_rebuild_is_reported_not_raised():
    """検査に落ちても起動は続く。理由は返して記録に残す。"""
    a = _agent()
    a.backend.complete = AsyncMock(return_value="ME.md を書き換えてしまった要約")
    with (
        patch.object(rc, "load_summary", return_value="古い"),
        patch.object(rc, "load_capabilities", return_value="capabilities:\n  - id: memory\n"),
        patch.object(rc, "save_summary") as save,
    ):
        reason = asyncio.run(rc.catch_up_summary(a))
    assert reason and "そのまま" in reason
    save.assert_not_called()


def test_no_me_md_does_nothing():
    """`ME.md` が無い構成では判じる材料が無い。触らない。"""
    a = _agent("")
    with (
        patch.object(rc, "load_summary", return_value="なにかの要約"),
        patch.object(rc, "save_summary") as save,
    ):
        reason = asyncio.run(rc.catch_up_summary(a))
    assert reason is None
    save.assert_not_called()
    a.backend.complete.assert_not_called()


# ── 起動の並びに入る ──────────────────────────────────────────────────────


def test_the_agent_starts_the_catch_up_in_the_background():
    """**起動を塞がない。** 温めや SBV2 起こしと同じく、投げるだけ。"""
    import inspect

    from familiar_agent.agent import EmbodiedAgent

    src = inspect.getsource(EmbodiedAgent._start_background_services)
    assert "catch_up_summary" in src
    assert "await catch_up_summary" not in src


def test_it_sits_below_the_once_only_gate():
    """人の発話のたびに走らせない（環-k・2026-09-14 と同じ守り）。"""
    import inspect

    from familiar_agent.agent import EmbodiedAgent

    src = inspect.getsource(EmbodiedAgent._start_background_services)
    assert src.index("_services_primed") < src.index("catch_up_summary")
