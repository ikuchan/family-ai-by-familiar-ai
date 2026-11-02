"""検査の事実で、頼み（きっかけ）を届いた結果から分ける（知-bn・2026-11-03・本人の決定）。

発話前の検査の「この求めで届いた結果（これも根拠）」の欄に、頼みの記録そのものと、組み直した版の頭の行
（「『…』と聞かれた」）の 2 つの道から、頼みが根拠として入っていた。頼みは「根拠ではない」と書いた見出しの下に
1 回だけ書き、結果の欄には届いたもの（調べものの行と見た記録）だけを残す。

- 会話の求めは「この求めの頼み（人が言ったこと・根拠ではない）」、情動などは
  「この求めのきっかけ（パジュの中で起きたこと・根拠ではない）」（本人の決定ア）。
- 機器の知らせ（タイマー・アラームなど）は外から届いた事実で、知らせる返事の根拠なので
  「この求めのきっかけ（機器からの知らせ・これも根拠）」（本人の決定ア）。
"""

from __future__ import annotations

from familiar_agent.loop.event_loop import InformationProcessing, Lookup
from tests.test_event_loop import _agent
from tests.test_workspace_is_the_core import _rec

ASK = "きょうの試合の結果をしらべて"
URGE = "[内的な促し:SAFETY] 確かめたい気持ちが湧いている。見回る。"
ASKED_HEAD = "この求めの頼み（人が言ったこと・根拠ではない）："
CAUSE_HEAD = "この求めのきっかけ（パジュの中で起きたこと・根拠ではない）："
DEVICE_HEAD = "この求めのきっかけ（機器からの知らせ・これも根拠）："
ALARM = "[アラーム] 7時 起きる時間"


def _ip(text=ASK, kind="発話"):
    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._req.request_id = "req-1"
    ip._req.request_text = text
    ip._req.trigger_kind = kind
    return ip


def _searched(ip, n=2):
    ip._req.lookups += [
        Lookup(index=i, action="search_deferred", query=f"語{i}", generation=0, result=f"結果{i}")
        for i in range(1, n + 1)
    ]
    ip._req.live_version_id = "v-live"
    return [
        _rec("req-1", ip._req.request_text, fit=0.9),
        _rec("v-live", f"「{ip._req.request_text}」と聞かれ、1番：…", fit=0.95),
    ]


def _arrived_part(facts: str) -> str:
    i = facts.find("この求めで届いた結果")
    return facts[i:] if i >= 0 else ""


def test_the_request_is_written_once_under_its_own_heading():
    ip = _ip()
    facts = ip._checker_facts(_searched(ip))
    assert facts.count(ASK) == 1
    assert f"{ASKED_HEAD}\n  - {ASK}" in facts
    assert ASK not in _arrived_part(facts)


def test_the_results_count_only_what_arrived():
    ip = _ip()
    facts = ip._checker_facts(_searched(ip))
    assert "この求めで届いた結果（2件・これも根拠）：" in facts
    assert "\n  - 1番：" in facts and "\n  - 2番：" in facts


def test_an_urge_is_written_as_the_cause():
    ip = _ip(text=URGE, kind="情動")
    facts = ip._checker_facts([_rec("req-1", URGE, fit=0.9)])
    assert f"{CAUSE_HEAD}\n  - {URGE}" in facts
    assert ASKED_HEAD not in facts
    assert facts.count(URGE) == 1


def test_what_was_seen_stays_a_result():
    """**反証の側**：この求めで見た記録は、届いたものなので結果の欄に残る。"""
    ip = _ip(text=URGE, kind="情動")
    ip._req.turn_records = [("seen-1", "見た")]
    facts = ip._checker_facts(
        [_rec("req-1", URGE, fit=0.9), _rec("seen-1", "窓の前に誰もいない", fit=0.9)]
    )
    assert "窓の前に誰もいない" in _arrived_part(facts)


def test_the_request_is_written_even_without_results():
    """**反証の側**：調べものが無くても、頼みは書く（結果の欄は出さない）。"""
    ip = _ip()
    facts = ip._checker_facts([_rec("req-1", ASK, fit=0.9)])
    assert f"{ASKED_HEAD}\n  - {ASK}" in facts
    assert "この求めで届いた結果" not in facts


def test_a_device_notice_is_written_as_evidence():
    ip = _ip(text=ALARM, kind="機器")
    facts = ip._checker_facts([_rec("req-1", ALARM, fit=0.9)])
    assert f"{DEVICE_HEAD}\n  - {ALARM}" in facts
    assert CAUSE_HEAD not in facts and ASKED_HEAD not in facts
    assert facts.count(ALARM) == 1


def test_an_empty_kind_is_a_spoken_request():
    """**反証の側**：起点が空なら会話とみなす（`req.trigger_kind or "発話"` と同じ扱い）。"""
    ip = _ip(kind="")
    facts = ip._checker_facts(_searched(ip))
    assert f"{ASKED_HEAD}\n  - {ASK}" in facts
    assert facts.count(ASK) == 1
    assert ASK not in _arrived_part(facts)


def test_a_long_device_notice_reaches_the_checker_in_full():
    """機器の知らせは 1,500 字まで記録に残る（`_DEVICE_MAX`）。`request_text` は 500 字で切れるので、頼みの記録の中身を使う。"""
    notice = "[予定] " + "あ" * 700 + "末尾の予定"
    ip = _ip(text=notice[:500], kind="機器")
    facts = ip._checker_facts([_rec("req-1", notice, fit=0.9)])
    assert "末尾の予定" in facts
    assert facts.count("[予定]") == 1


def test_without_the_record_the_request_text_is_used():
    """**反証の側**：頼みの記録が W に無ければ、`request_text` で書く。"""
    ip = _ip()
    facts = ip._checker_facts([])
    assert f"{ASKED_HEAD}\n  - {ASK}" in facts
