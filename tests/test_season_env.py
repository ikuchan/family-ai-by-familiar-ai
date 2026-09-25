"""いまの季節とまわり——季節の層の中身（知-ac 段 1・2026-09-26・`設計方針_季節の層` v0.1）。

4 つの欄（暦・天気・まわり・家の話題）。暦は日付から機械が毎回計算し、ほかは REST 内省の季節の層が
晩に 1 回書く。置き場は DB（`agent_state` の鍵 `season_env`）。古さで渡し方を変える。
"""

from __future__ import annotations

from datetime import date

from familiar_agent.core import season_env as se


def _env(written_on=date(2026, 9, 26), **rows):
    return se.SeasonEnv(
        written_on=written_on,
        rows={k: tuple(se.Row(*r) for r in v) for k, v in rows.items()},
    )


SEARCH = (
    "茨城県守谷市の天気。この数日は最高24℃前後で、朝晩は15℃ほど。守谷では金木犀が咲き始めました。"
)
GOOD = dict(
    天気=[("この数日は最高 24℃前後、朝晩は 15℃ほど。", "最高24℃前後で、朝晩は15℃ほど")],
    まわり=[("守谷では金木犀が咲き始めた。", "金木犀が咲き始めました")],
    家の話題=[("たいきの運動会が近いと話していた。", "obs-1")],
)


# ── 暦（機械が計算する） ─────────────────────────────────────────────────


def test_the_calendar_names_the_last_and_the_next_solar_term():
    assert se.calendar_line(date(2026, 9, 26)) == "秋分（9/23）を過ぎたころ。次は寒露（10/8）。"


def test_the_calendar_says_today_on_the_day():
    assert se.calendar_line(date(2026, 9, 23)) == "今日は秋分（9/23）。次は寒露（10/8）。"


def test_the_calendar_wraps_around_the_new_year():
    assert se.calendar_line(date(2027, 1, 2)) == "冬至（12/22）を過ぎたころ。次は小寒（1/6）。"
    assert se.calendar_line(date(2026, 12, 30)) == "冬至（12/22）を過ぎたころ。次は小寒（1/6）。"


# ── 検査（通すかどうかは機械） ───────────────────────────────────────────


def _check(env):
    return se.check(env, search_text=SEARCH, material_ids={"obs-1"})


def test_a_good_env_passes():
    assert _check(_env(**GOOD)) is None


def test_a_line_over_forty_chars_is_refused():
    long = "あ" * 41
    assert "40" in _check(_env(**{**GOOD, "まわり": [(long, "金木犀が咲き始めました")]}))


def test_more_than_two_rows_are_refused():
    three = [("金木犀が咲いた。", "金木犀が咲き始めました")] * 3
    assert _check(_env(**{**GOOD, "まわり": three})) is not None


def test_an_unknown_field_is_refused():
    assert _check(_env(**GOOD, 服装=[("長袖。", "")])) is not None


def test_the_calendar_is_not_written_by_the_llm():
    """暦は機械が計算する。LLM が書いてきたら通さない。"""
    assert _check(_env(**GOOD, 暦=[("秋分。", "")])) is not None


def test_a_quote_not_in_the_search_is_refused():
    """天気とまわりは、検索結果から**そのまま引いた文**が本文にあること。"""
    bad = {**GOOD, "天気": [("最高 30℃。", "最高30℃の真夏日")]}
    assert "検索" in _check(_env(**bad))


def test_a_house_topic_without_a_known_event_is_refused():
    bad = {**GOOD, "家の話題": [("運動会が近い。", "obs-999")]}
    assert _check(_env(**bad)) is not None


def test_empty_fields_are_fine():
    """検索できなかった晩は、天気とまわりが空でよい。"""
    assert _check(_env(家の話題=GOOD["家の話題"])) is None


# ── 渡し方（古さで変える） ───────────────────────────────────────────────


def test_a_fresh_env_is_given_whole():
    text = se.render(_env(**GOOD), today=date(2026, 9, 26))
    assert text.startswith("[いまの季節とまわり]（2026-09-26 の晩に書いた）")
    assert "- 暦：秋分（9/23）を過ぎたころ。" in text
    assert "- まわり：守谷では金木犀が咲き始めた。" in text
    assert "- 家の話題：たいきの運動会が近いと話していた。" in text
    assert "obs-1" not in text and "金木犀が咲き始めました" not in text  # 出典は渡さない


def test_seven_days_is_still_fresh():
    assert "日前の様子" not in se.render(_env(**GOOD), today=date(2026, 10, 3))


def test_after_seven_days_the_age_is_told():
    text = se.render(_env(**GOOD), today=date(2026, 10, 6))
    assert "（10 日前の様子）" in text.splitlines()[0]
    assert "金木犀" in text


def test_after_thirty_days_only_the_calendar_is_given():
    text = se.render(_env(**GOOD), today=date(2026, 10, 27))
    assert "金木犀" not in text and "運動会" not in text
    assert "- 暦：霜降（10/24）を過ぎたころ。" in text


def test_nothing_written_gives_only_the_calendar():
    text = se.render(None, today=date(2026, 9, 26))
    assert text == "[いまの季節とまわり]\n- 暦：秋分（9/23）を過ぎたころ。次は寒露（10/8）。"


# ── 置き場（テスト DB） ──────────────────────────────────────────────────


def test_store_read_and_clear_round_trip():
    assert se.stored() is None
    env = _env(**GOOD)
    assert se.store(env) is True
    assert se.stored() == env
    assert se.clear() is True
    assert se.stored() is None
