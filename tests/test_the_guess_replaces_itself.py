"""写真の見立ては、その写真を正として置き換える（知-ag・2026-09-24）。

実機 15:51、センサは `在席：テレビ に 1 人` と言い続けているのに、GUI には
**パパ 0.60 と たいきくん 0.60 の 2 人**が並んだ。0.60 は `SEEN_CONFIDENCE_MAX`——
どちらも**写真からの見立て**で入っている。

ログの並びがそのまま経緯である。

    15:51:28  写真の見立てで在席に入れた：パパ（確信度 0.60）／不明を 0 人に
    15:51:37  写真の見立てで在席に入れた：たいき（確信度 0.60）
    15:51:37  在席の変化：パパ → たいきくん・パパ

**見立ては 2 回とも「1 人」を言っている。** 後のほうが前を上書きせず、足された。

同じ関数（`_apply_seen_people`）の中で扱いが割れていた——名前の分からない人は
`note_unknown_present` で**人数を置き換える**のに、名前の付いた人は `person_arrived` で
**足すだけ**だった。`note_unknown_present` の説明文がその理由をすでに書いている：
「足し続けるのではなく、いまの人数に合わせる。**呼ばれるたびに足すと写真を見た回数だけ
人が増える**」。直すのは、その言い分を名前の付いた人にも当てることである。

**顔で入った人・手で入れた人は消さない。** 写真は部屋の一部しか写さない（カメラは首を
振る）ので、写真を正とできるのは**写真で入れたものだけ**である。
"""

from __future__ import annotations

import asyncio
import threading

from familiar_agent.person_memory_manager import PersonMemoryManager

GUESS = "見立て"


def _pmm() -> PersonMemoryManager:
    m = PersonMemoryManager.__new__(PersonMemoryManager)
    m._present = {}
    m._spaces = {}
    m._speaker_id = None
    m._speaker_source = None
    m._speaker_confidence = None
    m._switch_callbacks = []
    m._lock = threading.RLock()
    return m


def _names(pmm) -> set[str]:
    return {p.person_id for p in pmm._present.values() if not p.anonymous}


# ── 在席が「どうやって入ったか」を持つ ───────────────────────────────────


def test_the_way_someone_arrived_is_remembered():
    pmm = _pmm()
    asyncio.run(pmm.person_arrived("p1", 0.6, source=GUESS))
    assert pmm._present["p1"].source == GUESS


def test_arriving_without_a_source_is_unmarked():
    """顔・声・手入力は、いままでどおり由来を書かない（既定は空）。"""
    pmm = _pmm()
    asyncio.run(pmm.person_arrived("p1", 0.9))
    assert pmm._present["p1"].source == ""


# ── 見立ては見立てを置き換える ───────────────────────────────────────────


def test_a_new_guess_replaces_the_previous_one():
    """**これが今回の欠陥そのもの。** パパを見立てた後にたいきを見立てたら、たいきだけ。"""
    pmm = _pmm()
    asyncio.run(pmm.set_guessed_present([("p-papa", 0.6)]))
    assert _names(pmm) == {"p-papa"}
    asyncio.run(pmm.set_guessed_present([("p-taiki", 0.6)]))
    assert _names(pmm) == {"p-taiki"}


def test_two_people_in_one_photo_both_stay():
    """置き換えであって 1 人に絞るのではない。同じ写真に 2 人なら 2 人。"""
    pmm = _pmm()
    asyncio.run(pmm.set_guessed_present([("p-papa", 0.6), ("p-taiki", 0.6)]))
    assert _names(pmm) == {"p-papa", "p-taiki"}


def test_someone_who_came_in_by_face_is_not_removed():
    """**写真は部屋の一部しか写さない。** 写真で入れていない人を写真で消さない。"""
    pmm = _pmm()
    asyncio.run(pmm.person_arrived("p-face", 0.95))
    asyncio.run(pmm.set_guessed_present([("p-papa", 0.6)]))
    assert _names(pmm) == {"p-face", "p-papa"}
    asyncio.run(pmm.set_guessed_present([("p-taiki", 0.6)]))
    assert _names(pmm) == {"p-face", "p-taiki"}


def test_an_empty_guess_clears_the_guessed_ones_only():
    pmm = _pmm()
    asyncio.run(pmm.person_arrived("p-face", 0.95))
    asyncio.run(pmm.set_guessed_present([("p-papa", 0.6)]))
    asyncio.run(pmm.set_guessed_present([]))
    assert _names(pmm) == {"p-face"}


def test_the_same_person_seen_again_keeps_the_new_confidence():
    pmm = _pmm()
    asyncio.run(pmm.set_guessed_present([("p-papa", 0.6)]))
    asyncio.run(pmm.set_guessed_present([("p-papa", 0.3)]))
    assert pmm._present["p-papa"].confidence == 0.3


# ── ループがその口を通る ──────────────────────────────────────────────────


class _FakePMM:
    def __init__(self, known: dict):
        self._known = known
        self.guessed: list | None = None
        self.unknown = None
        self.arrived: list = []

    def find_person_id_by_name(self, name):
        return self._known.get(name)

    async def person_arrived(self, pid, confidence=1.0, source=""):
        self.arrived.append((pid, confidence, source))

    async def set_guessed_present(self, people):
        self.guessed = list(people)

    def note_unknown_present(self, count, confidence=0.5):
        self.unknown = (count, confidence)


_FAMILY = (
    "## ゆうすけ\n- **名前**：ゆうすけ\n- **呼び方**：パパ、ゆうすけ\n- **関係**：家族の父\n"
    "\n## たいき\n- **名前**：たいき\n- **呼び方**：たいき\n- **関係**：長男\n"
)


def _loop(pmm):
    from familiar_agent.loop.event_loop import InformationProcessing

    ip = InformationProcessing.__new__(InformationProcessing)

    class _Agent:
        _family_md = _FAMILY
        _pmm = pmm

    ip._agent = _Agent()
    return ip


def test_the_loop_replaces_instead_of_adding():
    from familiar_agent.core.seen_people import SEEN_CONFIDENCE_MAX

    pmm = _FakePMM({"パパ": "p-papa", "たいき": "p-taiki"})
    ip = _loop(pmm)
    asyncio.run(ip._apply_seen_people([{"name": "ゆうすけ", "confidence": 1.0}]))
    assert pmm.guessed == [("p-papa", SEEN_CONFIDENCE_MAX)]
    assert pmm.arrived == [], "足す口はもう使わない"
    asyncio.run(ip._apply_seen_people([{"name": "たいき", "confidence": 1.0}]))
    assert pmm.guessed == [("p-taiki", SEEN_CONFIDENCE_MAX)]


def test_an_empty_guess_still_touches_nothing():
    """写真を見ていない反復で在席を消さない（出-ae-は のまま）。"""
    pmm = _FakePMM({})
    ip = _loop(pmm)
    asyncio.run(ip._apply_seen_people([]))
    assert pmm.guessed is None
    assert pmm.unknown is None
