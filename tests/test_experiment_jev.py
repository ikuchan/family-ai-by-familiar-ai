"""出-au 段 4 の実験の道具（`scripts/experiment_jev.py`）。ログの組み立てと数え方だけを確かめる（外へは出ない）。"""

from __future__ import annotations

from datetime import datetime

from familiar_agent.backends.jev import JevAnswer
from scripts.experiment_jev import Case, is_person_utterance, parse_arbiter_cases, tally

_LOG = """\
2026-09-21 17:50:16,177 [v] [DEBUG] familiar_agent.loop.event_loop: event-loop 調停へ渡す W:
[直近のやりとり]
- 17:47 相手：明日の予定は？
2026-09-21 17:50:17,000 [v] [INFO] familiar_agent.loop.arbiter: 調停 0.9 秒（プロンプト 100 字）
2026-09-21 17:50:17,001 [v] [INFO] familiar_agent.loop.event_loop: event-loop 調停=action effort=low action=family_schedule
2026-09-21 17:51:00,000 [v] [DEBUG] familiar_agent.loop.event_loop: event-loop 調停へ渡す W:
[いま道具から返った]
- 予定の結果
2026-09-21 17:51:01,001 [v] [INFO] familiar_agent.loop.event_loop: event-loop 調停=light effort=low action=-
2026-09-21 17:52:00,000 [v] [DEBUG] familiar_agent.loop.event_loop: event-loop 調停へ渡す W:
（なし）
2026-09-21 17:52:05,000 [v] [WARNING] familiar_agent.loop.arbiter: 調停が 5.0 秒で返らなかったのでフルへ倒す
2026-09-21 17:52:05,001 [v] [INFO] familiar_agent.loop.event_loop: event-loop 調停=full effort=low action=-
2026-09-21 17:53:00,000 [v] [DEBUG] familiar_agent.loop.event_loop: event-loop 調停へ渡す W:
（なし）
2026-09-21 17:53:01,000 [v] [INFO] familiar_agent.loop.arbiter: 調停 light を full へ倒す（道具が要る頼み）：3分のタイマー
2026-09-21 17:53:01,001 [v] [INFO] familiar_agent.loop.event_loop: event-loop 調停=full effort=low action=-
2026-09-21 17:54:00,000 [v] [DEBUG] familiar_agent.loop.event_loop: event-loop 調停へ渡す W:
（なし）
2026-09-21 17:55:00,000 [v] [DEBUG] familiar_agent.loop.event_loop: event-loop 調停へ渡す W:
[直近]
2026-09-21 17:55:01,001 [v] [INFO] familiar_agent.loop.event_loop: event-loop 調停=full effort=medium action=-
"""


def test_it_pairs_each_workspace_with_its_own_decision():
    cases = parse_arbiter_cases(_LOG)
    got = [(c.at.strftime("%H:%M"), c.branch, c.effort, c.guarded) for c in cases]
    assert got == [
        ("17:50", "action", "low", False),  # 道具の帰り（17:51）と時間切れ（17:52）は除く
        ("17:53", "light", "low", True),  # 守りの前の生の答え
        ("17:55", "full", "medium", False),  # 答えの無い W（17:54）は捨て、次の W と取り違えない
    ]
    assert "明日の予定は？" in cases[0].workspace


def test_person_utterances_exclude_own_replies():
    assert is_person_utterance("明日の予定は？")
    assert not is_person_utterance("自分が答えた：晴れだよ")
    assert not is_person_utterance("つなぎに言った：調べるね")
    assert not is_person_utterance("")


def _case(branch, effort="low"):
    return Case(at=datetime(2026, 9, 21), workspace="", branch=branch, effort=effort, action="-")


def _ans(branch, effort="low", conf=0.8):
    return JevAnswer(
        ok=True,
        answers={"branch": {"choice": branch, "confidence": conf}, "effort": {"choice": effort}},
        seconds=0.3,
    )


def test_tally_counts_agreement_and_failures():
    t = tally(
        [
            (_case("light"), _ans("light")),
            (_case("full", "medium"), _ans("full", "low")),
            (_case("action"), _ans("full", conf=0.4)),
            (_case("light"), JevAnswer(ok=False, error="時間切れ")),
        ]
    )
    assert (t.total, t.failed, t.branch_agree) == (4, 1, 2)
    assert (t.effort_total, t.effort_agree) == (1, 0)
    assert t.confusion[("action", "full")] == 1


def test_affect_and_device_requests_are_told_apart():
    from scripts.experiment_jev import not_a_person_request

    assert not_a_person_request("- 2026 id:x (適合度:1.00) conf:1.00 わたし：[内的な促し:BOND] …")
    assert not_a_person_request("- 2026 id:x (適合度:1.00) conf:1.00 きっかけ：[入室] パパ が来た")
    assert not not_a_person_request("- 2026 id:x (適合度:1.00) conf:1.00 相手が言った: 明日は？")


def test_recent_only_keeps_just_the_recent_frame():
    from scripts.experiment_jev import recent_only

    w = "[いま道具から返った]\n- a\n[直近のやりとり（古い順）]\n- b\n- c\n[過去の記憶]\n- d"
    assert recent_only(w) == "[直近のやりとり（古い順）]\n- b\n- c"


def _pair(llm, jev):
    return _case(llm), _ans(jev)


def test_picking_spreads_over_the_ways_they_disagree():
    from scripts.experiment_jev import pick_disagreements

    pairs = (
        [_pair("full", "action")] * 10
        + [_pair("action", "light")] * 10
        + [_pair("light", "light")] * 5
    )
    got = pick_disagreements(pairs, 4)
    assert len(got) == 4
    assert sorted((c.branch, j) for c, j in got) == [
        ("action", "light"),
        ("action", "light"),
        ("full", "action"),
        ("full", "action"),
    ]  # 一致した例は選ばない・割れ方ごとに交互に


def test_labels_are_scored_against_both_and_nothing_is_kept():
    from scripts.experiment_jev import label_interactively, score_labels

    items = [(_case("full"), "action"), (_case("action"), "light"), (_case("light"), "full")]
    answers = iter(["x", "3", "3", "s"])  # 無効な入力は聞き直す
    shown: list = []
    labels = label_interactively(items, ask=lambda _p: next(answers), show=shown.append)
    assert labels == ["action", "action", None]
    out = score_labels(items, labels)
    assert "Jev 1" in out and "軽量LLM 1" in out and "正解を付けた 2 件" in out
    assert not any("軽量LLM" in s or "Jev" in s for s in shown[1:])  # どちらが選んだかは見せない
