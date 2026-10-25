"""要るスキルだけを、書き手と主LLM に渡す（環-au 段 B・2026-10-25・本人の決定）。

調停の Jev の問いに、スキルごとに「要る／要らない」の 2 択を足す（待ちは増えない・決定ア）。選ばれたスキルは、窓の中なら
窓が閉じるまで、窓の外ならその求めのあいだ載せる（決定ア）。書き手（軽量LLM）の文では [いま誰が居るか] の後・窓の会話の前、
主LLM では system の最後の部分（毎回変わる部分）に、同じ [やり方] の枠で載せる。
"""

from __future__ import annotations

import asyncio
import time

from familiar_agent.core import utterance_meaning as um
from familiar_agent.core.skills import Skill, skills_frame
from tests._arbiter_fakes import _pick, decide, jev_says, writer_says

SHIRITORI = Skill("しりとり", "しりとりをしているとき", "- 「ん」で終わったら負け。")
BANANA = Skill("マジカルバナナ", "マジカルバナナをしているとき", "- 連想する。")
SKILLS = (SHIRITORI, BANANA)
TALK = "[窓の会話（19:00:00 から・古い順・全部）]\n- 19:00:00 パパ：しりとりしよう"


# ── 問いと枠 ─────────────────────────────────────────────────


def test_each_skill_gets_one_need_question():
    qs = um.fanout_questions(music=False, camera=False, family=[], skills=SKILLS)
    assert set(qs["skill_0"]["criteria"]) == {"need", "no"}
    assert "しりとり" in qs["skill_0"]["instructions"]
    assert "しりとりをしているとき" in qs["skill_0"]["instructions"]
    assert "マジカルバナナ" in qs["skill_1"]["instructions"]


def test_no_skills_means_no_questions():
    """**反証の側**：スキルが無ければ、問いは足さない（いままでの問いのまま）。"""
    qs = um.fanout_questions(music=False, camera=False, family=[])
    assert not any(k.startswith("skill_") for k in qs)


def test_the_chosen_skills_are_the_ones_answered_need():
    got = {"skill_0": _pick("need", 0.8), "skill_1": _pick("no", 0.9)}
    assert um.chosen_skills(got, SKILLS) == ["しりとり"]
    assert um.chosen_skills({}, SKILLS) == []  # 答えが無ければ選ばない


def test_the_frame_holds_only_the_named_skills():
    text = skills_frame(SKILLS, ["しりとり"])
    assert (
        text.startswith("[やり方]") and "「ん」で終わったら負け" in text and "連想する" not in text
    )
    assert skills_frame(SKILLS, []) == ""


# ── 調停：書き手に渡す ─────────────────────────────────────────────


def _writer_prompt(writer) -> str:
    """書き手の呼び出し（返すか黙るかの判定ではない方）のプロンプト。"""
    prompts = [c.args[0] for c in writer.complete.await_args_list if "[書くもの]" in c.args[0]]
    assert prompts, "書き手が呼ばれていない"
    return prompts[-1]


def _decided(*, need: bool, active: "tuple[str, ...]" = ()):
    jev = jev_says("light")
    jev.ask.return_value.answers["skill_0"] = _pick("need" if need else "no", 0.9)
    jev.ask.return_value.answers["skill_1"] = _pick("no", 0.9)
    writer = writer_says({"text": "りんご！"})
    d = asyncio.run(
        decide(
            jev=jev,
            writer=writer,
            utterance="しりとりしよう",
            present_ctx="（顔ぶれ）",
            window_ctx=TALK,
            skills=SKILLS,
            active_skills=active,
        )
    )
    return d, writer


def test_a_skill_jev_says_is_needed_reaches_the_writer_before_the_window_talk():
    d, writer = _decided(need=True)
    prompt = _writer_prompt(writer)
    assert "「ん」で終わったら負け" in prompt
    assert prompt.index("[いま誰が居るか]") < prompt.index("[やり方]") < prompt.index(TALK)
    assert d.skills == ("しりとり",)


def test_no_skill_needed_and_none_active_means_no_frame():
    """**反証の側**：要らないと答え、前から載っているものも無ければ、[やり方] の枠ごと出ない。"""
    d, writer = _decided(need=False)
    assert "[やり方]" not in _writer_prompt(writer)
    assert d.skills == ()


def test_an_active_skill_stays_even_when_jev_says_no():
    d, writer = _decided(need=False, active=("しりとり",))
    assert "「ん」で終わったら負け" in _writer_prompt(writer)
    assert d.skills == ()  # 新しく選んだものは無い（貯めるのはループ）


# ── ループ：貯める・消える ─────────────────────────────────────────


def _ip(*, window: bool):
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._current_speaker_name = lambda: "パパ"
    if window:
        ip._wake_window().open(time.monotonic())
        ip._req.window_session = ip._wake_window().session
    return ip


def test_in_the_window_a_chosen_skill_stays_until_the_window_closes():
    ip = _ip(window=True)
    ip._keep_skills(("しりとり",))
    assert ip._active_skills() == ("しりとり",)
    ip._keep_skills(())  # 次の求めで Jev が要らないと答えても
    assert ip._active_skills() == ("しりとり",)
    ip._wake_window().close()
    ip._wake_window().open(time.monotonic())  # 新しい窓
    ip._req.window_session = ip._wake_window().session
    assert ip._active_skills() == ()


def test_outside_the_window_a_chosen_skill_lasts_only_for_the_request():
    ip = _ip(window=False)
    ip._keep_skills(("しりとり",))
    assert ip._active_skills() == ("しりとり",)
    asyncio.run(
        ip._begin_request(kind="情動", text="次の求め")
    )  # 求めの器は作り直さず、欄ごとに戻す
    assert ip._active_skills() == ()


def test_skills_are_added_not_replaced():
    ip = _ip(window=True)
    ip._keep_skills(("しりとり",))
    ip._keep_skills(("マジカルバナナ",))
    assert ip._active_skills() == ("しりとり", "マジカルバナナ")


# ── 主LLM ─────────────────────────────────────────────────


def test_the_main_llm_gets_the_skills_in_the_changing_part():
    from familiar_agent.backends.anthropic import AnthropicBackend
    from familiar_agent.loop.prompt import build_event_system_prompt

    frame = skills_frame(SKILLS, ["しりとり"])
    parts = build_event_system_prompt(
        self_understanding="[ME] ぼく",
        family_md="[FAMILY] パパ・ママ",
        present_ctx="",
        pi_ctx="",
        workspace_ctx="[想起]昔の話",
        window=("[窓の会話（19:00:00 から）]", "- 19:00:00 パパ：しりとりしよう"),
        skills=frame,
    )
    assert "「ん」で終わったら負け" in parts[-1]
    assert all("「ん」で終わったら負け" not in p for p in parts[:-1])
    blocks = AnthropicBackend._build_system_param(parts)
    assert [i for i, b in enumerate(blocks) if "cache_control" in b] == [0, 2]


def test_the_loop_gives_the_active_skills_to_the_main_llm(monkeypatch):
    from familiar_agent.core import skills as skills_mod

    monkeypatch.setattr(skills_mod, "load_skills", lambda root=None: list(SKILLS))
    ip = _ip(window=True)
    ip._keep_skills(("しりとり",))
    assert "「ん」で終わったら負け" in asyncio.run(ip._skills_text())
    ip._wake_window().close()
    ip._wake_window().open(time.monotonic())
    ip._req.window_session = ip._wake_window().session
    assert asyncio.run(ip._skills_text()) == ""


def test_decide_offers_the_skills_and_keeps_the_chosen_ones(monkeypatch):
    from familiar_agent.core import skills as skills_mod
    from familiar_agent.loop.arbiter import Arbiter, Decision
    from tests._arbiter_fakes import ws_of

    monkeypatch.setattr(skills_mod, "load_skills", lambda root=None: list(SKILLS))
    got: list = []

    async def capture(self, inp, on_decided=None):
        got.append(inp)
        return Decision(branch="light", text="", skills=("しりとり",))

    monkeypatch.setattr(Arbiter, "decide", capture)
    ip = _ip(window=True)
    asyncio.run(
        ip._decide(utterance="しりとりしよう", ws=ws_of(), present_ctx="", capped=False, round_=1)
    )
    assert got[0].skills == SKILLS and got[0].active_skills == ()
    assert ip._active_skills() == ("しりとり",)


# --- 見直しの指摘（1 回目）---


def test_an_active_skill_is_not_asked_again():
    qs = um.fanout_questions(
        music=False, camera=False, family=[], skills=SKILLS, active=("しりとり",)
    )
    assert "skill_0" not in qs and "skill_1" in qs


def test_a_writer_time_out_does_not_write_skills_into_the_shared_fallback():
    """書き手が時間切れで倒し先を返しても、共有の倒し先にスキルを書き込まない（次の別の求めに漏れる）。"""
    from familiar_agent.loop import arbiter

    jev = jev_says("light")
    jev.ask.return_value.answers["skill_0"] = _pick("need", 0.9)
    d = asyncio.run(
        decide(
            jev=jev,
            writer=writer_says({"text": "x"}, delay=1.0),
            timeout=0.05,
            utterance="しりとりしよう",
            skills=SKILLS,
        )
    )
    assert d.skills == ("しりとり",)
    assert arbiter._FALLBACK.skills == ()


def test_an_unreadable_folder_means_no_skills(tmp_path, caplog):
    """本当に読めないフォルダで確かめる（`glob` は黙って空を返すので、置き場の警告が出ることまで見る）。"""
    import logging
    import os

    import pytest

    from familiar_agent.core.skills import load_skills

    if os.geteuid() == 0:
        pytest.skip("root では読めないフォルダを作れない")
    folder = tmp_path / "skills"
    folder.mkdir()
    (folder / "a.md").write_text("# しりとり\nいつ要る：遊ぶとき\n- 決まり\n", encoding="utf-8")
    os.chmod(folder, 0)
    try:
        with caplog.at_level(logging.WARNING):
            assert load_skills(folder) == []
    finally:
        os.chmod(folder, 0o755)
    assert "置き場を読めなかった" in caplog.text


def test_the_speech_check_also_knows_the_rules(monkeypatch):
    """主LLM へ投げるとき、発話前の検査に渡る直近にもスキルの枠が入る（実際に渡った値で見る）。"""
    from familiar_agent.core import skills as skills_mod
    from familiar_agent.loop import event_loop
    from familiar_agent.loop.arbiter import Arbiter, Decision
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests._arbiter_fakes import writer_says
    from tests.test_event_loop import _agent
    from tests.test_iteration_characterization import _run_until

    monkeypatch.setattr(skills_mod, "load_skills", lambda root=None: list(SKILLS))
    monkeypatch.setattr(event_loop, "main_llm_cache_on", lambda: True)

    async def full(self, inp, on_decided=None):
        return Decision(branch="full", skills=("しりとり",))

    monkeypatch.setattr(Arbiter, "decide", full)
    a = _agent(stream_returns=[])
    a._utility_backend = writer_says({"text": ""})
    sent: list = []
    ip = InformationProcessing(a)
    monkeypatch.setattr(ip, "_dispatch_main_llm", lambda **kw: sent.append(kw))
    ip._current_speaker_name = lambda: "パパ"
    ip._wake_window().open(time.monotonic())
    ip._wake_window().until += 3600
    _run_until(ip, lambda: sent, push=lambda: ip.push_affect("CALM", "落ち着いている"))
    assert "「ん」で終わったら負け" in sent[0]["recent_frame"]
    assert "「ん」で終わったら負け" in sent[0]["system"][-1]
