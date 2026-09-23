"""直近のやりとりは、人との言葉だけ（出-ar・2026-09-24）。

やりとりの**起点**を数えると、発話 309・情動 175・機器 130 で、**人以外が半分**を占めていた。
実機 15:49 の `[直近のやりとり]` は `[タイマー]` の 1 行だけで、**人とのやりとりが 1 行も
無いのに枠の名前は「直近のやりとり」**だった。

機器（タイマー・メモ・入室）は**起きたこと**、情動は**湧いたこと**で、どちらもやりとりでは
ない（本人・2026-09-24）。枠を分ける。保留（話したかったが話せなかったこと）と
「届いたもの」は、もともと別の枠にあるので触らない。

**これは読みやすさの直しである。** 枠を分けても、つなぎのあとの挨拶のやり直し（出-aj #4）は
止まらない（実測 6/6）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from familiar_agent.loop import workspace

_WHEN = datetime(2026, 9, 21, 15, 48, tzinfo=timezone.utc)


def _said(obs_id: str, content: str, *, role: str = "起点", direction: str = "発話"):
    return SimpleNamespace(
        obs_id=obs_id, role=role, content=content, when=_WHEN, direction=direction
    )


class _OIF:
    def voices(self, ids):
        return {}


def _render(rows):
    return workspace.render_recent(_OIF(), [("o", rows)], 3)[1]


# ── 人の言葉だけが「やりとり」に残る ─────────────────────────────────────


def test_a_persons_words_stay_in_the_exchange_frame():
    got = _render([_said("o1", "こんにちは")])
    assert "[直近のやりとり" in got
    assert "こんにちは" in got.split("[そのあいだに起きたこと]")[0]


def test_my_own_reply_stays_with_it():
    got = _render([_said("o1", "こんにちは"), _said("o2", "自分が答えた：やあ", role="答え")])
    head = got.split("[そのあいだに起きたこと]")[0]
    assert "こんにちは" in head and "やあ" in head


# ── 機器と情動は「起きたこと」へ ─────────────────────────────────────────


def test_a_device_line_moves_to_what_happened():
    got = _render([_said("o1", "[タイマー] 「パスタ」の時間", direction="機器")])
    assert "[そのあいだに起きたこと]" in got
    assert "タイマー" in got.split("[そのあいだに起きたこと]")[1]
    assert "タイマー" not in got.split("[そのあいだに起きたこと]")[0]


def test_an_urge_moves_too():
    got = _render([_said("o1", "[内的な促し:SEEKING] 探索したい", direction="情動")])
    assert "探索したい" in got.split("[そのあいだに起きたこと]")[1]


def test_both_frames_appear_in_order():
    """人との言葉が先、起きたことが後。"""
    got = _render(
        [
            _said("o1", "[タイマー] 鳴った", direction="機器"),
            _said("o2", "こんにちは"),
        ]
    )
    assert got.index("[直近のやりとり") < got.index("[そのあいだに起きたこと]")


# ── 空なら枠ごと出さない ──────────────────────────────────────────────────


def test_no_device_lines_means_no_second_frame():
    assert "[そのあいだに起きたこと]" not in _render([_said("o1", "こんにちは")])


def test_no_words_means_no_exchange_frame():
    got = _render([_said("o1", "[タイマー] 鳴った", direction="機器")])
    assert "[直近のやりとり" not in got
    assert "[そのあいだに起きたこと]" in got


def test_nothing_at_all_is_empty():
    assert _render([]) == ""


# ── id の対応表は両方から作る ────────────────────────────────────────────


def test_ids_from_both_frames_are_in_the_map():
    rows = [_said("o1", "こんにちは"), _said("o2", "[タイマー] 鳴った", direction="機器")]
    _, _, id_map = workspace.render_recent(_OIF(), [("o", rows)], 3)
    assert set(id_map.values()) == {"o1", "o2"}
