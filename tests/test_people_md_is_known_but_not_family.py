"""家族以外の知っている人は `PEOPLE.md` に書く（知-ab・2026-10-01・本人の決定）。

祖父母・親戚・子どもの友だち・来客を書く場所が無く、`FAMILY.md` に足すと同居の家族と同じ扱い（人物表に
登録され、話者になり、その人の記憶の空間ができる）になった。`PEOPLE.md` は `FAMILY.md` と同じ書き方で、
システム文の `[家族以外で知っている人]` と写真の読み取りにだけ渡す。**人物表には入れない**（本人の決定ア：
その人専用の記憶は溜めない）。名乗り・口調の機械の口にも入れない（入れると話者になる）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core.context_parts import Stance, build_context

PEOPLE = "## おばあちゃん\n\n- **名前**：花子\n- **呼び方**：おばあちゃん\n- **関係**：母方の祖母・月に一度来る\n"


def test_the_frame_sits_after_the_family():
    ctx = build_context(
        stance=Stance.PAJU, self_understanding="パジュ", family="[家族] パパ", people=PEOPLE
    )
    assert "[家族以外で知っている人]\n## おばあちゃん" in ctx.stable
    assert ctx.stable.index("[一緒に暮らす人たち]") < ctx.stable.index("[家族以外で知っている人]")


def test_no_frame_without_people():
    ctx = build_context(stance=Stance.PAJU, self_understanding="パジュ", family="[家族] パパ")
    assert "[家族以外で知っている人]" not in ctx.stable


def test_the_main_llm_knows_the_people():
    from familiar_agent.backends import ToolCall
    from tests.test_event_loop import _agent, _run, _turn

    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    a._people_md = PEOPLE
    _run(a, utterance="パジュ、おばあちゃんっていつ来る？")
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "[家族以外で知っている人]" in system and "月に一度来る" in system


def test_the_arbiters_writer_knows_the_people():
    from tests._arbiter_fakes import decide, jev_says, system_of, writer_says

    writer = writer_says({"text": "はい"})
    asyncio.run(
        decide(jev=jev_says("light"), writer=writer, utterance="こんにちは", people_md=PEOPLE)
    )
    assert "[家族以外で知っている人]" in system_of(writer)


def test_the_evaluator_context_knows_the_people():
    from familiar_agent.agent import EmbodiedAgent
    from familiar_agent.core.context_parts import Stance as _S

    a = MagicMock(spec=EmbodiedAgent)
    a._me_md = "パジュ"
    a._family_md = "[家族] パパ"
    a._people_md = PEOPLE
    text = EmbodiedAgent._stance_context(a, _S.PAJU)
    assert text is not None and "[家族以外で知っている人]" in text


def test_the_file_is_looked_for_like_family(tmp_path, monkeypatch):
    from familiar_agent.agent import EmbodiedAgent

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    a = MagicMock(spec=EmbodiedAgent)
    assert EmbodiedAgent._load_person_md(a, "PEOPLE.md") == ""
    (tmp_path / "home" / ".familiar_ai").mkdir(parents=True)
    (tmp_path / "home" / ".familiar_ai" / "PEOPLE.md").write_text("家の外", encoding="utf-8")
    assert EmbodiedAgent._load_person_md(a, "PEOPLE.md") == "家の外"
    (tmp_path / "PEOPLE.md").write_text("直下", encoding="utf-8")
    assert EmbodiedAgent._load_person_md(a, "PEOPLE.md") == "直下"  # 直下が先


def test_reload_reads_people_too():
    from familiar_agent.agent import EmbodiedAgent

    a = MagicMock(spec=EmbodiedAgent)
    a._me_md, a._family_md, a._people_md = "me", "fam", ""
    a._load_me_md = MagicMock(return_value="me")
    a._load_family_md = MagicMock(return_value="fam")
    a._load_person_md = MagicMock(side_effect=lambda name: PEOPLE if name == "PEOPLE.md" else "")
    out = EmbodiedAgent._handle_reload_command(a, "/reload")
    assert a._people_md == PEOPLE
    assert "PEOPLE.md を更新しました" in out
    a._register_family_from_md.assert_not_called()  # 家族は変わっていない。人物表は触らない


def test_people_are_not_registered_as_persons():
    from familiar_agent.agent import EmbodiedAgent

    a = MagicMock(spec=EmbodiedAgent)
    a._family_md = "## パパ\n\n- **名前**：雄輔\n- **呼び方**：パパ\n"
    a._people_md = PEOPLE
    a._pmm = MagicMock()
    a._persons = MagicMock()
    EmbodiedAgent._register_family_from_md(a)
    names = [c.args[0] for c in a._pmm.register_person.call_args_list]
    assert names == ["雄輔"]  # 家族だけ。花子（おばあちゃん）は登録しない


@pytest.mark.asyncio
async def test_the_photo_reader_is_told_both():
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a._family_md = "## パパ\n- **名前**：雄輔"
    a._people_md = PEOPLE
    ip = InformationProcessing(a)
    seen: dict = {}

    async def read_photo(image_b64, backend, *, family_md):
        seen["family_md"] = family_md
        return ["机"], []

    import familiar_agent.loop.event_loop as el

    orig = el.read_photo
    el.read_photo = read_photo
    try:
        ip._quick_labels = AsyncMock(return_value=[])  # type: ignore[method-assign]
        await ip._read_photo_into_state("", image_b64="b64", image_path=None, prefix="")
    finally:
        el.read_photo = orig
        await ip.close()
    assert "雄輔" in seen["family_md"] and "[家族以外で知っている人]" in seen["family_md"]
