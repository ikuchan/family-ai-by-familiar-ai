"""主LLM のキャッシュを、REST が直近 7 日の様子で入り切りする（出-bs 段 2・2026-10-21・本人の決定）。

10/19〜10/21 の計測で、反復 11 回のうち当たったのは 5 分以内に続いた 2 回だけだった。外れた 9 回は約 8,000
トークンを毎回書き直し、書き込みの割増し（1.25 倍）だけを払っていた——キャッシュなしより約 3% 重い。会話が
たて続く週は当たるので、どちらがよいかは時期で変わる。層 3 が毎晩、直近 7 日の続き方から安いほうへ寄せる。

当たりは**時刻で数える**（前の反復から 5 分以内・道具の数が同じ・本人の決定ア）。使わない週は本物の当たりが
記録されないので、どちらの週も同じ物差しで数える。境目は料金の倍率から計算する（本人の決定イ）。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.backends.types import TurnResult
from familiar_agent.core import measure, settings
from familiar_agent.core.measure import Row
from familiar_agent.loop import rest_settings

FIELD = "AgentConfig.main_llm_cache"
T0 = datetime(2026, 10, 20, 12, 0, 0)


def _rows(gaps: "list[float]", tools: "list[int] | None" = None) -> list[Row]:
    """最初の行は T0、続きは前の行から `gaps` 秒ずつ。道具の数は既定で 1。"""
    tools = tools or [1] * (len(gaps) + 1)
    out, t = [Row(T0, "主LLM", {"道具": str(tools[0])})], T0
    for g, n in zip(gaps, tools[1:]):
        t = t + timedelta(seconds=g)
        out.append(Row(t, "主LLM", {"道具": str(n)}))
    return out


# ── 数え方 ─────────────────────────────────────────────────────────────────


def test_a_call_within_five_minutes_with_the_same_tools_is_a_hit():
    s = measure.summarize_main_llm(_rows([300.0]))
    assert (s["件数"], s["当たり"]) == (2, 1)


def test_a_call_after_five_minutes_is_a_miss():
    s = measure.summarize_main_llm(_rows([301.0]))
    assert (s["件数"], s["当たり"]) == (2, 0)


def test_a_different_tool_set_is_a_miss():
    s = measure.summarize_main_llm(_rows([60.0], tools=[1, 28]))
    assert s["当たり"] == 0


def test_other_kinds_are_not_counted():
    rows = _rows([60.0]) + [Row(T0, "調停", {"秒": "1.0"})]
    assert measure.summarize_main_llm(rows)["件数"] == 2


def test_no_rows_is_zero_without_dividing():
    s = measure.summarize_main_llm([])
    assert (s["件数"], s["当たり"], s["割合"]) == (0, 0, 0.0)


# ── 境目と判定 ─────────────────────────────────────────────────────────────


def test_the_threshold_comes_from_the_price_ratios():
    assert rest_settings.cache_threshold(1.25, 0.1) == pytest.approx(0.25 / 1.15)


def _summary(n: int, hits: int) -> dict:
    return {"件数": n, "当たり": hits, "割合": hits / n}


def _at(monkeypatch, current: "float | None"):
    over = {} if current is None else {FIELD: current}
    monkeypatch.setattr("familiar_agent.config_overrides.load_overrides", lambda: over)


def test_a_low_hit_rate_turns_the_cache_off(monkeypatch):
    _at(monkeypatch, None)  # 既定は使う
    got = rest_settings.adjust_main_llm_cache(_summary(22, 4), write=1.25, read=0.1)  # 18%
    assert got == [(FIELD, 1.0, 0.0)]


def test_a_high_hit_rate_turns_it_back_on(monkeypatch):
    _at(monkeypatch, 0.0)
    got = rest_settings.adjust_main_llm_cache(_summary(20, 6), write=1.25, read=0.1)  # 30%
    assert got == [(FIELD, 0.0, 1.0)]


def test_too_few_calls_leave_it_alone(monkeypatch):
    _at(monkeypatch, None)
    assert rest_settings.adjust_main_llm_cache(_summary(19, 0), write=1.25, read=0.1) == []


def test_already_on_the_cheaper_side_changes_nothing(monkeypatch):
    """**反証の側**：もう安い側なら動かさない（毎晩同じ記録を書かない）。"""
    _at(monkeypatch, 0.0)
    assert rest_settings.adjust_main_llm_cache(_summary(20, 0), write=1.25, read=0.1) == []
    _at(monkeypatch, None)
    assert rest_settings.adjust_main_llm_cache(_summary(20, 10), write=1.25, read=0.1) == []


def test_the_switch_is_registered_for_layer_three():
    s = settings.get(FIELD)
    assert s is not None
    assert (s.lo, s.hi, s.step, s.measure_kind, s.rule) == (0.0, 1.0, 1.0, "主LLM", "機械")


# ── 7 日分を読む ───────────────────────────────────────────────────────────


def _write(path, lines: "list[str]") -> None:
    path.write_text("".join(f"{x}\n" for x in lines), encoding="utf-8")


def test_seven_days_are_read_across_rotated_logs(tmp_path):
    d = tmp_path / measure.DIR_NAME
    d.mkdir()
    since = datetime(2026, 10, 14, 12, 0, 0)
    # 7 日より前に改名した：読まない
    _write(d / "measure.log.20261013T030000", ["2026-10-13T02:00:00 主LLM 道具=1"])
    # 7 日の境目をまたぐ：境目より前の行は数えない
    _write(
        d / "measure.log.20261015T030000",
        ["2026-10-14T11:59:59 主LLM 道具=1", "2026-10-14T12:00:00 主LLM 道具=1"],
    )
    _write(
        d / measure.FILE_NAME,
        ["2026-10-21T06:33:19 主LLM 道具=1", "2026-10-21T06:38:12 調停 秒=1.0"],
    )
    rows = measure.read_rows_since(since, base_dir=tmp_path)
    got = sorted((r.when.isoformat(), r.kind) for r in rows)
    assert got == [
        ("2026-10-14T12:00:00", "主LLM"),
        ("2026-10-21T06:33:19", "主LLM"),
        ("2026-10-21T06:38:12", "調停"),
    ]


def test_the_rest_rule_uses_the_week_and_the_ratios(monkeypatch):
    """`adjust_settings` は、この規則にだけ 7 日分の行と設定の倍率を渡す。"""
    seen: dict = {}
    monkeypatch.setattr(rest_settings.measure, "read_rows", lambda: [])
    monkeypatch.setattr(rest_settings.measure, "rotate", lambda: None)

    def since(when, base_dir=None):
        seen["since"] = when
        return _rows([60.0] * 25)

    monkeypatch.setattr(rest_settings.measure, "read_rows_since", since)

    def adjust(summary, *, write, read):
        seen["args"] = (summary["件数"], write, read)
        return []

    monkeypatch.setattr(rest_settings, "adjust_main_llm_cache", adjust)
    monkeypatch.setattr(rest_settings, "adjust_timeout", AsyncMock(return_value=[]))
    agent = MagicMock()
    agent.config.prompt_cache_write_ratio = 1.25
    agent.config.prompt_cache_read_ratio = 0.1
    before = datetime.now()
    asyncio.run(rest_settings.adjust_settings(agent))
    assert seen["args"] == (26, 1.25, 0.1)
    assert (
        before - timedelta(days=7, seconds=5) <= seen["since"] <= datetime.now() - timedelta(days=7)
    )


def test_the_price_ratios_are_plain_defaults():
    """倍率は層 3 に登録しない（料金が変わったら人が写す）。"""
    from familiar_agent.config import AgentConfig

    fields = AgentConfig.__dataclass_fields__
    w = fields["prompt_cache_write_ratio"].default
    r = fields["prompt_cache_read_ratio"].default
    assert (w, r) == (1.25, 0.1)
    assert settings.get("AgentConfig.prompt_cache_write_ratio") is None


# ── 主LLM の呼び方 ─────────────────────────────────────────────────────────


def _dispatch(monkeypatch, over: dict):
    from tests.test_main_llm_is_dispatched import _ip

    monkeypatch.setattr("familiar_agent.config_overrides.load_overrides", lambda: over)
    recorded: list = []
    monkeypatch.setattr(
        "familiar_agent.loop.event_loop.measure_record",
        lambda kind, **f: recorded.append((kind, f)),
        raising=False,
    )
    ip, a = _ip()
    tools = [{"name": "say"}, {"name": "recall"}]
    ip._tools = MagicMock(return_value=tools)
    a.backend.stream_turn = AsyncMock(return_value=(TurnResult("end_turn", ""), None))

    async def scenario():
        ip._dispatch_main_llm(
            messages=[],
            system=("安定", "可変"),
            effort="high",
            capped=False,
            memories=[],
            w_id_map={},
            verdict_view="__self__",
            recent_frame="",
            max_tokens=1024,
        )
        for t in list(ip._background_tasks):
            await t

    asyncio.run(scenario())
    return ip, a, recorded


def test_with_the_cache_on_the_call_is_as_before(monkeypatch):
    """**反証の側**：既定（上書き無し）は今までどおり、組で渡し、道具に印を付ける。"""
    ip, a, recorded = _dispatch(monkeypatch, {})
    assert a.backend.stream_turn.await_args.kwargs["system"] == ("安定", "可変")
    assert ip._tools.call_args.kwargs.get("cache_tools", True) is True
    assert recorded == [("主LLM", {"道具": 2})]


def test_with_the_cache_off_no_marker_is_sent(monkeypatch):
    ip, a, recorded = _dispatch(monkeypatch, {FIELD: 0.0})
    assert a.backend.stream_turn.await_args.kwargs["system"] == "安定\n\n可変"
    assert ip._tools.call_args.kwargs["cache_tools"] is False
    assert recorded == [("主LLM", {"道具": 2})]
