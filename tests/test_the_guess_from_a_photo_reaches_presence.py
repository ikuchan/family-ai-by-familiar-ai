"""写真からの見立てを在席へ流す（出-ae-は・2026-09-22）。

実機 15:57、写真を見た調停が「パパ、おかえりなさい！」と言った一方、機械は
`(present :speaker "unconfirmed")` のままだった。**推し量ること自体は禁じない**（本人の
決定）。推し量ったなら、それが在席にも使われてほしい。

身元が機械に入る道は、顔（`presence_status`）と声の名乗り（`speaker_claim`）の 2 本あり、
**写真からの見立てだけがどこへも行かなかった**。

実機の写真と作業状態で測ると、`seen_people` の欄を足せば 5 回中 5 回書く（足さなければ
`speaker_claim` は空のまま）。ただし**確信度は 1.0 と書いてくる**ので、機械側で上限を掛ける
——見立ては推し量りで、顔で測った値と同じ重さにはしない。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core.seen_people import SEEN_CONFIDENCE_MAX, parse_seen_people

_FAMILY = """# 一緒に暮らす人たち

## ゆうすけ

- **名前**：ゆうすけ
- **呼び方**：パパ、ゆうすけ、おとうさん
- **関係**：家族の父

## たいき

- **名前**：たいき
- **呼び方**：たいき、たいきくん
- **関係**：長男
"""


# ── 見立ての読み取り ──────────────────────────────────────────────────────


def test_a_family_member_is_resolved_to_the_way_we_call_them():
    """調停は `名前` を書いてくることがある（実機は "ゆうすけ"）。呼び方へ直す。"""
    known, unknown = parse_seen_people([{"name": "ゆうすけ", "confidence": 1.0}], _FAMILY)
    assert [n for n, _ in known] == ["パパ"]
    assert unknown == 0


def test_the_confidence_of_a_guess_is_capped():
    """1.0 と書かれても、見立ては顔で測った値と同じ重さにしない。"""
    known, _ = parse_seen_people([{"name": "パパ", "confidence": 1.0}], _FAMILY)
    assert known[0][1] == SEEN_CONFIDENCE_MAX
    assert SEEN_CONFIDENCE_MAX < 1.0


def test_a_lower_confidence_is_kept_as_it_is():
    known, _ = parse_seen_people([{"name": "パパ", "confidence": 0.3}], _FAMILY)
    assert known[0][1] == 0.3


def test_a_nameless_person_is_counted_as_unknown():
    known, unknown = parse_seen_people(
        [{"name": "パパ", "confidence": 0.8}, {"name": "", "confidence": 0.5}], _FAMILY
    )
    assert [n for n, _ in known] == ["パパ"]
    assert unknown == 1


def test_a_name_that_is_not_in_the_family_is_counted_as_unknown():
    """家族に無い名前は人として当てない。**居たことは残す**ので不明に数える。"""
    known, unknown = parse_seen_people([{"name": "だれか", "confidence": 0.9}], _FAMILY)
    assert known == []
    assert unknown == 1


def test_the_same_person_twice_is_one_person():
    known, unknown = parse_seen_people(
        [{"name": "パパ", "confidence": 0.8}, {"name": "ゆうすけ", "confidence": 0.5}], _FAMILY
    )
    assert [n for n, _ in known] == ["パパ"]
    assert unknown == 0


def test_nothing_in_nothing_out():
    assert parse_seen_people([], _FAMILY) == ([], 0)
    assert parse_seen_people(None, _FAMILY) == ([], 0)


def test_a_broken_entry_is_skipped():
    """調停の返りは壊れうる。落ちずに読めるものだけ拾う。"""
    known, unknown = parse_seen_people(
        ["これは辞書ではない", {"confidence": 0.5}, {"name": "パパ"}], _FAMILY
    )
    assert [n for n, _ in known] == ["パパ"]
    assert unknown == 1  # 名前の欄が無いものは不明の 1 人


# ── 調停の返りに欄がある ──────────────────────────────────────────────────


def test_the_arbiter_reads_the_field():
    from familiar_agent.loop.arbiter import _parse

    reply = '{"branch": "light", "text": "おかえりなさい", "seen_people": [{"name": "パパ", "confidence": 1.0}]}'
    got = _parse(reply, can_see=True, origin="機器")
    assert got is not None
    assert got.seen_people == [{"name": "パパ", "confidence": 1.0}]


def test_the_field_may_be_absent():
    from familiar_agent.loop.arbiter import _parse

    got = _parse('{"branch": "light", "text": "おかえりなさい"}', can_see=True, origin="機器")
    assert got is not None
    assert got.seen_people == []


def test_the_prompt_asks_for_the_field():
    from familiar_agent.loop.arbiter import ARBITER_PROMPT

    assert "seen_people" in ARBITER_PROMPT
    assert "写真に人が写って" in ARBITER_PROMPT


# ── 見立てを在席へ流す ────────────────────────────────────────────────────


def _loop_with(family: str, pmm):
    from familiar_agent.loop.event_loop import InformationProcessing

    ip = InformationProcessing.__new__(InformationProcessing)

    class _Agent:
        _family_md = family
        _pmm = pmm
        # 見立てが 1 人なら話者を付ける（出-as 段 9a）。付ける口の偽物。
        _persons = MagicMock()
        _sync_pmm_speaker = AsyncMock()

    ip._agent = _Agent()
    return ip


class _FakePMM:
    """在席の口を写す。**見立ては `set_guessed_present` を通る**（知-ag・2026-09-24）——
    以前は `person_arrived` で足していたが、置き換えないので写真を見るたびに人が増えた。"""

    def __init__(self, known: dict):
        self._known = known
        self.guessed: list | None = None
        self.unknown = None

    def find_person_id_by_name(self, name):
        return self._known.get(name)

    async def set_guessed_present(self, people):
        self.guessed = list(people)

    def note_unknown_present(self, count, confidence=0.5):
        self.unknown = (count, confidence)


def test_a_guessed_family_member_enters_presence():
    import asyncio
    from dataclasses import dataclass

    @dataclass
    class _D:
        seen_people: list

    pmm = _FakePMM({"パパ": "p-yusuke"})
    ip = _loop_with(_FAMILY, pmm)
    asyncio.run(ip._apply_seen_people([{"name": "ゆうすけ", "confidence": 1.0}]))
    assert pmm.guessed == [("p-yusuke", SEEN_CONFIDENCE_MAX)]
    assert pmm.unknown == (0, SEEN_CONFIDENCE_MAX)


def test_a_nameless_person_becomes_an_unknown_present():
    import asyncio
    from dataclasses import dataclass

    @dataclass
    class _D:
        seen_people: list

    pmm = _FakePMM({"パパ": "p-yusuke"})
    ip = _loop_with(_FAMILY, pmm)
    asyncio.run(
        ip._apply_seen_people(
            [{"name": "パパ", "confidence": 0.8}, {"name": "", "confidence": 0.4}]
        )
    )
    assert pmm.guessed == [("p-yusuke", 0.6)]
    assert pmm.unknown == (1, SEEN_CONFIDENCE_MAX)


def test_nothing_happens_without_a_guess():
    """欄が空なら在席を触らない。**写真を見ていない反復で在席を消さない。**"""
    import asyncio
    from dataclasses import dataclass

    @dataclass
    class _D:
        seen_people: list

    pmm = _FakePMM({})
    ip = _loop_with(_FAMILY, pmm)
    asyncio.run(ip._apply_seen_people([]))
    assert pmm.guessed is None
    assert pmm.unknown is None


# ── 主LLM も同じ口で申告する（出-an） ────────────────────────────────────


def test_the_say_tool_takes_the_guess():
    """`say()` の入力に見立ての欄がある（`memory_verdicts` と同じ形）。

    写真は主LLM にも届く。実機 15:57 の写真で測ると、欄が無ければ 6 回とも「パパ」と呼んで
    機械へは何も渡らず、欄を足せば 6 回とも書く（確信度 0.7〜0.9）。
    """
    from familiar_agent.tools.tts import TTSTool

    tts = TTSTool.__new__(TTSTool)
    tts.engine = "elevenlabs"
    tts.elevenlabs_model = "eleven_flash_v2_5"
    props = tts.get_tool_definitions()[0]["input_schema"]["properties"]
    assert "seen_people" in props
    item = props["seen_people"]["items"]["properties"]
    assert set(item) == {"name", "confidence"}


def test_the_main_llm_guess_goes_through_the_same_door():
    """主LLM の申告も `_apply_seen_people` を通る（調停と同じ扱い・同じ上限）。"""
    import asyncio

    pmm = _FakePMM({"パパ": "p-yusuke"})
    ip = _loop_with(_FAMILY, pmm)
    asyncio.run(ip._apply_seen_people([{"name": "パパ", "confidence": 0.9}]))
    assert pmm.guessed == [("p-yusuke", SEEN_CONFIDENCE_MAX)]
