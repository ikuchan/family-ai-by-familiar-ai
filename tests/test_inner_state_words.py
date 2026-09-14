"""内部状態（気分・欲求）を主LLM へ渡す言葉（情-f・2026-09-14）。

**気分は 4 軸 × 5 段**。境目はその軸の実測分布の分位（p10／p30／p70／p90）で、等頻度なので
どの段も実際に出る。以前は 12 点表への最近傍 1 語だったが、記-g で測り直した PAD の幅
（P 0.10〜0.35）は表の座標（P 0.85 など）と別のスケールで、15 回の更新が 15 回 `neutral`
だった（情報量 0）。

**欲求は発火した軸を明示し、ほかは p70 超だけ弱く**。固定の 0.5／0.75 では REST が常に「高」、
BOND・ESTEEM が常に「低」で情報が無かった。
"""

from __future__ import annotations

from familiar_agent.core import inner_state
from familiar_agent.core.inner_state import DriveP70, MoodBands
from familiar_agent.drive_register import AiDrivers
from familiar_agent.mood_register import MoodPAD

_BANDS = MoodBands(
    p=(0.10, 0.10, 0.25, 0.35),
    pn=(0.10, 0.10, 0.10, 0.15),
    a=(0.36, 0.50, 0.50, 0.50),
    dom=(0.50, 0.50, 0.55, 0.60),
)
_P70 = DriveP70(seeking=0.66, rest=0.95, bond=0.03, safety=0.75, esteem=0.07)


def test_each_mood_axis_is_a_word_on_a_five_step_scale():
    words = inner_state.mood_words(MoodPAD(p=0.30, pn=0.10, a=0.50, dom=0.62), _BANDS)
    assert words == "うれしさ 高い・つらさ ふつう・高ぶり ふつう・余裕 とても高い"


def test_the_ends_and_the_boundaries():
    # p10 未満＝とても低い、p10 以上 p30 未満＝低い、p30 以上 p70 未満＝ふつう、
    # p70 以上 p90 未満＝高い、p90 以上＝とても高い。境目は上の段に入る。
    assert inner_state.level(0.05, (0.10, 0.10, 0.25, 0.35)) == "とても低い"
    assert inner_state.level(0.10, (0.10, 0.10, 0.25, 0.35)) == "ふつう"  # p10=p30 なら「低い」は空
    assert inner_state.level(0.25, (0.10, 0.10, 0.25, 0.35)) == "高い"
    assert inner_state.level(0.35, (0.10, 0.10, 0.25, 0.35)) == "とても高い"
    assert inner_state.level(0.40, (0.36, 0.50, 0.50, 0.50)) == "低い"


def test_an_unmeasured_mood_says_so():
    assert inner_state.mood_words(None, _BANDS) == "気分は測れていない"


def test_the_fired_drive_comes_first_and_others_only_when_above_their_p70():
    drives = AiDrivers(seeking=0.9, rest=0.95, bond=0.01, safety=0.8, esteem=0.06)
    words = inner_state.drive_words(drives, fired="seeking", p70=_P70)
    assert words.startswith("発火：SEEKING（探索したい）")
    assert "（ほかに 安心したさ がやや）" in words
    assert "REST" not in words and "休みたさ" not in words  # 0.95 は自分の p70 を超えない


def test_no_firing_and_nothing_above_p70():
    drives = AiDrivers(seeking=0.3, rest=0.9, bond=0.0, safety=0.2, esteem=0.05)
    assert inner_state.drive_words(drives, fired="", p70=_P70) == "発火：なし"


def test_the_prompt_line_has_no_raw_numbers():
    line = inner_state.pi_line(
        MoodPAD(p=0.30, pn=0.10, a=0.50, dom=0.62),
        AiDrivers(seeking=0.9, rest=0.95, bond=0.01, safety=0.8, esteem=0.06),
        fired="seeking",
        bands=_BANDS,
        p70=_P70,
    )
    assert line.startswith("[内部状態(PI)] 気分：")
    assert "発火：SEEKING（探索したい）" in line
    import re

    assert not re.search(r"\d\.\d", line), line


# ── ループ側：発火した軸を求めが持ち、`_pi_ctx` が言葉にし、数値は計測ログへ ──────


def _agent_with_state(monkeypatch, tmp_path, *, mood=None, drives=None):
    from unittest.mock import MagicMock

    from familiar_agent.core import measure

    measure.setup(base_dir=tmp_path)
    monkeypatch.setattr(
        "familiar_agent.mood_register.load_current_mood",
        lambda: mood if mood is not None else MoodPAD(p=0.30, pn=0.10, a=0.50, dom=0.62),
    )
    monkeypatch.setattr(
        "familiar_agent.drive_register.load_current_drives",
        lambda: (
            drives
            if drives is not None
            else AiDrivers(seeking=0.9, rest=0.95, bond=0.01, safety=0.8, esteem=0.06)
        ),
    )
    a = MagicMock()
    return a, tmp_path / "rest_logs" / "measure.log"


def test_the_request_remembers_which_drive_fired():
    from familiar_agent.loop.request import Request

    req = Request()
    assert req.fired_axis == ""


def test_pi_ctx_uses_the_words_and_writes_the_numbers_to_the_measure_log(monkeypatch, tmp_path):
    from familiar_agent.loop.generator import _pi_ctx
    from familiar_agent.loop.request import Request

    _a, path = _agent_with_state(monkeypatch, tmp_path)
    req = Request()
    req.fired_axis = "seeking"
    line = _pi_ctx(req)
    assert line.startswith("[内部状態(PI)] 気分：うれしさ 高い")
    assert "発火：SEEKING（探索したい）" in line and "安心したさ がやや" in line
    assert "0.9" not in line and "0.30" not in line
    logged = path.read_text(encoding="utf-8")
    assert " 気分 P=0.30" in logged and "Dom=0.62" in logged
    assert " 欲求 SEEKING=0.90" in logged and "発火=seeking" in logged


def test_a_conversation_turn_says_no_firing(monkeypatch, tmp_path):
    from familiar_agent.loop.generator import _pi_ctx
    from familiar_agent.loop.request import Request

    _agent_with_state(
        monkeypatch,
        tmp_path,
        drives=AiDrivers(seeking=0.3, rest=0.95, bond=0.0, safety=0.2, esteem=0.05),
    )
    assert "発火：なし" in _pi_ctx(Request())


def test_the_fired_axis_is_set_by_the_affect_entry_and_cleared_at_close():
    import asyncio

    from familiar_agent.backends import ToolCall
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent, _turn

    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    ip = InformationProcessing(a)
    seen: list[str] = []
    original = ip._iterate

    async def spy():
        seen.append(ip._req.fired_axis)
        return await original()

    ip._iterate = spy  # type: ignore[method-assign]

    async def scenario():
        ip.push_affect("seeking", "探索したい気持ちが湧いている")
        ip.start()
        for _ in range(400):
            if a._run_post_response_pipeline.called:  # 求めが閉じた（`_finish` まで来た）
                break
            await asyncio.sleep(0.005)
        await ip.close()

    asyncio.run(scenario())
    assert seen and seen[0] == "seeking"
    assert ip._req.fired_axis == ""


def test_the_old_fixed_bands_are_gone():
    import familiar_agent.core.drive_autonomy as da
    from familiar_agent.config import DriveConfig

    assert not hasattr(da, "drive_snapshot") and not hasattr(da, "qualitative_level")
    assert not hasattr(DriveConfig(), "drive_level_mid")
