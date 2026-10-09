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
    from familiar_agent.core import utterance_meaning as um

    # 打ち消しは Jev に問う。段 4-4c から発話の 1 回目の意味（名乗りの否定）で、2 回目はどの呼び方か。
    asked = um.MEANINGS["deny"][1]
    assert "相手ではない" in asked
    # 見本は鉤括弧の外に置く（言わせる見本ではない）。名前を言わない打ち消し（「ちがうよ」）も見本に入れる。
    # 見本を全部外すと、名前を言わない打ち消しが 3 回中 0 回になった（実測・2026-09-22）。
    assert "じゃないよ" in asked and "ちがうよ" in asked


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


def test_an_empty_denial_means_no_denial():
    """空は「打ち消しなし」（出-ax・2026-10-07）。以前は空ならいまの話者を外していたので、調停を通るたびに
    声で付けた話者が外れた（実機 12:49・12:59・13:28 の 6 回・0.4〜2.3 秒後）。名前を言わない打ち消しは、
    Jev が選択肢「いま話者としている人」を選ぶので名前つきで届く（下の試験）。"""
    import asyncio

    pmm = _FakePMM2({"パパ": "p-yusuke"})
    persons = _FakePersons("パパ")
    ip = _loop(pmm, persons)
    asyncio.run(ip._apply_not_person(""))
    assert pmm.left == []
    assert persons.reset == 0


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


# ── 打ち消していないのに外さない（出-ax・2026-10-07 実機）─────────────────────


def test_todays_form_keeps_the_voice_speaker():
    """声でパパを付けた直後、打ち消しの無い入力（「パジュー、今日の天気は?」）で調停が倒れても外さない。"""
    import asyncio

    from familiar_agent.loop.arbiter import _FALLBACK

    pmm = _FakePMM2({"パパ": "p-yusuke"})
    persons = _FakePersons("パパ")
    ip = _loop(pmm, persons)
    ip._apply_silence = lambda decision, utterance="": None

    async def no_claim(decision):
        return None

    ip._apply_speaker_claim = no_claim
    asyncio.run(ip._apply_requests(_FALLBACK, utterance="パジュー、今日の天気は?"))
    assert pmm.left == []
    assert persons.reset == 0


def test_a_denial_without_a_name_arrives_with_the_current_speaker():
    """「ちがうよ」（名前なし）：Jev は「いま話者としている人」を選べるので、その名前で届いて外れる。

    古い問いにあった決まりを、段 5（2026-10-10 本人の決定ア）で新しい問い（否定の 2 回目）に移した。
    """
    import asyncio

    from familiar_agent.backends.jev import JevAnswer
    from tests._arbiter_fakes import decide, writer_says

    class _Jev:
        available = True
        asked: dict = {}

        async def ask(self, state, questions):
            _Jev.asked = questions
            return JevAnswer(
                ok=True,
                answers={
                    "meaning": {"choice": "deny", "confidence": 0.9},
                    "action_deny": {"choice": "パパ", "confidence": 0.9},
                },
            )

    d = asyncio.run(
        decide(
            jev=_Jev(),
            writer=writer_says({"text": "ごめんね"}),
            utterance="ちがうよ",
            family_md=_FAMILY,
            current_speaker="パパ",
        )
    )
    assert "名前を言わず" in _Jev.asked["action_deny"]["criteria"]["パパ"]
    data = {"not_person": d.not_person}
    assert data["not_person"] == "パパ"
    pmm = _FakePMM2({"パパ": "p-yusuke"})
    persons = _FakePersons("パパ")
    asyncio.run(_loop(pmm, persons)._apply_not_person(data["not_person"]))
    assert pmm.left == ["p-yusuke"]
