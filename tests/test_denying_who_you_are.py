"""身元を否定されたら顔ぶれから外す（出-am・2026-09-22）。

身元が機械に入る道は 3 本ある（顔・声の名乗り・写真からの見立て）。ところが**消える道は
時間切れと顔の消失だけ**で、声からの打ち消しが届かなかった。誤った見立ては寿命が来るまで
残る。

「パパじゃないよ」と言われたら、**由来に関係なく**その人を顔ぶれから外し、話者を戻す
（本人の決定・2026-09-22）。名前を言わずに否定されたら、いま話者としている人を外す。

（在席が「誰も居ない 60 秒」で顔ぶれ表を消す失効の試験は、知-ai でその失効ごと撤去した。顔ぶれは人ごとの
持ち時間で切れる・`test_presence_holds_a_minute`。）
"""

from __future__ import annotations


from familiar_agent.person_memory_manager import PersonMemoryManager


def _pmm() -> PersonMemoryManager:
    import threading

    m = PersonMemoryManager.__new__(PersonMemoryManager)
    m._present = {}
    m._spaces = {}
    m._speaker_id = None
    m._speaker_source = None
    m._speaker_confidence = None
    m._switch_callbacks = []
    m._lock = threading.RLock()
    return m


# ── 調停が否定を読む ──────────────────────────────────────────────────────


def test_the_arbiter_reads_the_denial():
    from familiar_agent.loop.arbiter import assemble

    got = assemble(
        {"branch": "light", "text": "ごめんなさい", "not_person": "パパ"},
        can_see=False,
        origin="発話",
    )
    assert got is not None
    assert got.not_person == "パパ"


def test_the_denial_may_be_absent():
    from familiar_agent.loop.arbiter import assemble

    got = assemble({"branch": "light", "text": "はい"}, can_see=False, origin="発話")
    assert got is not None
    assert got.not_person == ""


def test_the_prompt_asks_for_the_denial():
    from familiar_agent.loop.arbiter import Arbiter, ArbiterInput

    # 打ち消しは Jev に問う（出-au 段 5-7d）。
    qs = Arbiter(jev=None, writer=None)._questions(ArbiterInput(utterance="x", workspace_ctx=""))
    asked = qs["denies"]["instructions"]
    assert "名前の人ではないと打ち消した" in asked
    assert "denied" in qs
    # 見本は番人の範囲内で置く（「〜」で始まる形か 3 字以内）。見本を全部外すと、
    # 名前を言わない打ち消し（「ちがうよ」）が 3 回中 0 回になった（実測・2026-09-22）。
    assert "〜じゃないよ" in asked


# ── 否定を顔ぶれへ効かせる ──────────────────────────────────────────────────

_FAMILY = (
    "## ゆうすけ\n- **名前**：ゆうすけ\n- **呼び方**：パパ、ゆうすけ、おとうさん\n"
    "\n## たいき\n- **名前**：たいき\n- **呼び方**：たいき、たいきくん\n"
)


class _FakePersons:
    def __init__(self, name=""):
        self.active_name = name
        self.reset = 0

    def reset_to_default(self):
        self.reset += 1
        self.active_name = ""


def _loop(pmm, persons, family=_FAMILY):
    from familiar_agent.loop.event_loop import InformationProcessing

    ip = InformationProcessing.__new__(InformationProcessing)

    class _Agent:
        _family_md = family
        _pmm = pmm
        _persons = persons

    ip._agent = _Agent()
    return ip


class _FakePMM2:
    def __init__(self, known):
        self._known = known
        self.left: list = []
        self.speaker_name = ""

    def find_person_id_by_name(self, name):
        return self._known.get(name)

    async def person_left(self, pid):
        self.left.append(pid)

    def current_speaker_name(self):
        return self.speaker_name


def test_a_denied_person_leaves_presence():
    import asyncio

    pmm = _FakePMM2({"パパ": "p-yusuke"})
    persons = _FakePersons("パパ")
    ip = _loop(pmm, persons)
    asyncio.run(ip._apply_not_person("パパ"))
    assert pmm.left == ["p-yusuke"]
    assert persons.reset == 1


def test_a_denial_without_a_name_uses_the_current_speaker():
    """調停は空でなく「いまの話者」を書いてくる（実測）。空でも困らないようにする。"""
    import asyncio

    pmm = _FakePMM2({"パパ": "p-yusuke"})
    persons = _FakePersons("パパ")
    ip = _loop(pmm, persons)
    asyncio.run(ip._apply_not_person(""))
    assert pmm.left == ["p-yusuke"]


def test_someone_outside_the_family_is_ignored():
    import asyncio

    pmm = _FakePMM2({"パパ": "p-yusuke"})
    persons = _FakePersons("パパ")
    ip = _loop(pmm, persons)
    asyncio.run(ip._apply_not_person("太郎"))
    assert pmm.left == []
    assert persons.reset == 0


def test_nothing_happens_without_a_denial_and_no_speaker():
    import asyncio

    pmm = _FakePMM2({"パパ": "p-yusuke"})
    persons = _FakePersons("")
    ip = _loop(pmm, persons)
    asyncio.run(ip._apply_not_person(""))
    assert pmm.left == []
    assert persons.reset == 0


def test_the_denial_is_applied_before_the_claim():
    """「ちがう、ママだよ」で パパ を外してから ママ を付ける。順序が逆だと付けた人を消す。"""
    import inspect

    from familiar_agent.loop.event_loop import InformationProcessing

    src = inspect.getsource(InformationProcessing._apply_requests)
    assert src.index("_apply_not_person") < src.index("_apply_speaker_claim")
