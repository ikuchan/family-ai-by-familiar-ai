"""自己像（層 2）の器・種・検査・文（記-a-へ・2026-09-14・`設計方針_REST内省_自己像` v0.2）。

状態は DB（`agent_state.self_image`）、種は repo の `defaults/self_image.yaml`（DB に無いときだけ）。
3 欄（望み 8×60・気がかり 8×60・価値 10×50・合計 1,500 字）。各行は出典（自己エピソードの id）と
入った日を持つ。文にするときは数値も id も出さない。
"""

from __future__ import annotations

from datetime import date

from familiar_agent.core import self_image as si


def _clear():
    from familiar_agent.db import get_db

    db = get_db()
    with db.lock:
        conn = db.conn()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM agent_state WHERE state_key = %s", (si.STATE_KEY,))
        conn.commit()


def test_the_seed_is_read_when_the_db_has_nothing_and_then_saved():
    _clear()
    image = si.load()
    assert len(image.hopes) == 8 and len(image.concerns) == 8 and len(image.values) == 10
    assert image.hopes[0].text.startswith("家族それぞれが好きなものを")
    assert all(line.sources == () for line in image.hopes)  # 種には出典が無い
    # 種を読んだら DB に置く（以後は DB が現在値）。
    again = si.load()
    assert again == image
    assert si.stored() is not None


def test_save_and_load_round_trip_keep_since_and_sources():
    _clear()
    image = si.load()
    new = image.replace_line(
        "望み", 0, si.Line(text="新しい望み。", since=date(2026, 9, 14), sources=("ep-1",))
    )
    si.store(new)
    got = si.load()
    assert got.hopes[0] == si.Line(text="新しい望み。", since=date(2026, 9, 14), sources=("ep-1",))
    assert got.hopes[1] == image.hopes[1]


def test_check_enforces_rows_chars_and_total():
    ok = si.load()
    assert si.check(ok) is None
    too_many = ok.replace_field("価値", list(ok.values) + [si.Line("一つ多い。", date.today(), ())])
    assert si.check(too_many) is not None
    too_long = ok.replace_line("望み", 0, si.Line("あ" * 61, date.today(), ()))
    assert si.check(too_long) is not None
    long_value = ok.replace_line("価値", 0, si.Line("い" * 51, date.today(), ()))
    assert si.check(long_value) is not None
    empty = ok.replace_field("望み", [])
    assert si.check(empty) is not None  # 欄は空にしない


def test_render_is_a_frame_without_numbers_or_ids():
    _clear()
    image = si.load()
    image = image.replace_line(
        "気がかり",
        0,
        si.Line("疲れている人に無理に話しかけていないか。", date(2026, 9, 10), ("ep-9",)),
    )
    text = si.render(image, today=date(2026, 9, 14))
    assert text.startswith("[いまの自分]")
    assert "望み：" in text and "気がかり：" in text and "価値：" in text
    assert "（9/10 から）" in text  # 古さの印
    assert "ep-9" not in text
    assert "帰ってきた人を、最初に迎える存在でいたい。" in text


# ── 読み手：主LLM のシステム文と調停に、規則と一緒に渡る ──────────────────────


def test_the_event_system_prompt_carries_the_self_image_after_the_rules():
    from familiar_agent.loop.prompt import build_event_system_prompt

    stable, _variable = build_event_system_prompt(
        self_understanding="ぼくはパジュ",
        family_md="## パパ\n名前: ゆうすけ",
        present_ctx="",
        pi_ctx="",
        workspace_ctx="",
        iter_ctx="",
        self_image="[いまの自分]\n望み：\n- 帰ってきた人を、最初に迎える存在でいたい。",
    )
    # 主LLM の規則は `[身体と決まり]`（静的核）の中にある。自己像は同じ安定部で、家族の後・可変部の前。
    assert stable.index("[一緒に暮らす人たち]") < stable.index("[いまの自分]")
    assert "帰ってきた人を、最初に迎える存在でいたい。" in stable


def test_the_loop_hands_the_current_self_image_to_the_prompt(monkeypatch):
    """`_build_system` は DB の自己像（無ければ種）を渡す。"""
    from unittest.mock import MagicMock

    from familiar_agent.loop.event_loop import InformationProcessing

    _clear()
    a = MagicMock()
    a._me_md, a._family_md = "ぼくはパジュ", "## パパ\n名前: ゆうすけ"
    a._tts = None
    monkeypatch.setattr("familiar_agent.capability_state.load_summary", lambda: "")
    ip = InformationProcessing.__new__(InformationProcessing)
    ip._agent = a
    from familiar_agent.loop.request import Request

    ip._req = Request()
    ip._dif = MagicMock(understands_tags=False)
    stable, _ = ip._build_system(present_ctx="", workspace_ctx="", iter_ctx="")
    assert "[いまの自分]" in stable and "帰ってきた人を、最初に迎える存在でいたい。" in stable


def test_the_arbiter_gets_the_same_self_image_in_its_system(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from familiar_agent.loop.arbiter import arbitrate

    b = MagicMock(spec=["complete"])
    b.complete = AsyncMock(return_value='{"branch": "full"}')
    asyncio.run(
        arbitrate(
            b,
            utterance="x",
            workspace_ctx="",
            self_understanding="ぼくはパジュ",
            family_md="家族",
            self_image="[いまの自分]\n価値：\n- 悔しがる気持ちを笑わない。",
        )
    )
    assert "悔しがる気持ちを笑わない。" in b.complete.await_args.kwargs["system"]
