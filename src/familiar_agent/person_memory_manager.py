"""Person-aware memory routing layer.

Architecture
------------
present_persons : dict[person_id, PersonPresence]
    Everyone currently in the room.
current_speaker_id : str | None
    The person actively speaking this turn.

Memory access
-------------
- Writes  → current_speaker's memory space.
- Reads   → situated search across ALL present persons + AGENT_SELF.
- Agent's own memories (self_model, curiosity …) always use AGENT_SELF_ID.

Perspective vectors (design α)
-------------------------------
Situated vectors are keyed by (obs_id, person_id, relation_key).
At memory-write time every registered person gets a situated_embedding
pre-computed as:  normalise(mem_vec - mu)  (045: no per-person term).
At recall time the query is issued directly against situated_memories,
returning sorted results from SQL — no full table scan needed.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    # 実行時には読み込まない。`tools.memory` はこのモジュールを使う側なので、
    # 実行時に import すると循環する。注釈は文字列のままで足りる。
    from .tools.memory import ObservationMemory


logger = logging.getLogger(__name__)

# ── Reserved person IDs ────────────────────────────────────────────────────
AGENT_SELF_ID = "00000000-0000-0000-0000-000000000000"
DEFAULT_PERSON_ID = "00000000-0000-0000-0000-000000000001"

# Perspective vector blend weight (0 = no perspective, 1 = full person bias)
# Auto-switch threshold for non-manual/llm hints
AUTO_SWITCH_THRESHOLD: float = 0.75
# Seconds without camera/voice signal before requesting re-detection
PRESENCE_TIMEOUT_SEC: float = 120.0


# ── Data structures ────────────────────────────────────────────────────────


@dataclass
class PersonPresence:
    person_id: str
    confidence: float = 1.0
    #: 名前の分からない顔ぶれ（出-ae-は・2026-09-22）。`person_id` は顔ぶれ表の中だけの札で、
    #: **人を指していない**。`get_present_ids()` からは外す——あそこは観測の `participants`
    #: になり、人ごとの面を立てる材料なので、誰にも対応しない面ができてしまう。
    anonymous: bool = False
    #: どうやって顔ぶれに入ったか（知-ag・2026-09-24）。`見立て`＝写真からの推し量り、
    #: 空＝顔・声・手入力。**置き換えてよいのは見立てで入ったものだけ**である——写真は
    #: 部屋の一部しか写さない（カメラは首を振る）ので、写真に居ない人を写真で消せない。
    source: str = ""
    arrived_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_signal_at: float = field(default_factory=lambda: time.time())


@dataclass
class RecognitionHint:
    """A single recognition signal from any source."""

    person_id: str
    confidence: float  # 0.0–1.0
    source: str  # "face" | "voice" | "text" | "manual" | "llm"
    reason: str = ""

    # Special source values that bypass the confidence threshold
    IMMEDIATE = frozenset({"manual", "llm"})


# ── Manager ────────────────────────────────────────────────────────────────


class PersonMemoryManager:
    """Central coordinator for person identity and memory routing."""

    def __init__(
        self,
        base_memory: "ObservationMemory",
        switch_thresholds: dict[str, float] | None = None,
    ) -> None:
        self._base = base_memory
        self._present: dict[str, PersonPresence] = {}
        self._speaker_id: str | None = None
        # 現在の話者を決めた認識の由来と確信度（GUI で「話者認識がどうなっているか」を出す）。
        self._speaker_source: str = ""
        self._speaker_confidence: float | None = None
        self._instances: dict[str, Any] = {}  # person_id → ObservationMemory
        self._switch_callbacks: list[Callable[[str | None, str], Awaitable[None]]] = []
        self._lock = threading.Lock()
        # source 別の自動切替しきい値。認識モデルで cosine 尺度が違うため source 別に持つ。
        # 未指定なら RecognitionConfig の既定（顔・声）を読む。表に無い source は既定の
        # AUTO_SWITCH_THRESHOLD へフォールバックする（text/auto 等）。
        if switch_thresholds is None:
            from .config import RecognitionConfig

            _rc = RecognitionConfig()
            switch_thresholds = {
                "face": _rc.face_switch_threshold,
                "voice": _rc.voice_switch_threshold,
            }
        self._switch_thresholds = switch_thresholds

    # ── Presence management ────────────────────────────────────────────────

    async def person_arrived(
        self, person_id: str, confidence: float = 1.0, source: str = ""
    ) -> None:
        """Register that someone has entered the space.

        `source` はどうやって入ったか（知-ag）。書かなければ空＝顔・声・手入力。
        """
        with self._lock:
            was_empty = len(self._present) == 0
            self._present[person_id] = PersonPresence(
                person_id=person_id, confidence=confidence, source=source
            )
        logger.info("Arrived: %s  (total present: %d)", person_id, len(self._present))
        if was_empty:
            await self.set_speaker(person_id, source="auto", confidence=confidence)

    #: 写真からの見立てで入った印（知-ag・2026-09-24）。
    GUESS_SOURCE = "見立て"

    async def set_guessed_present(self, people: "list[tuple[str, float]]") -> None:
        """写真に写った人を顔ぶれに入れるか、持ち時間を数え直す（知-ai・2026-10-05）。

        **置き換えない。** 写真は部屋の一部しか写さない（カメラは首を振る）ので、写っていないことは
        「居ない」の証拠にならない。以前（知-ag）は見立てで入った人をこの並びに置き換えていたので、SEEKING の
        見回りのたびに顔ぶれが入れ替わった。写っていない人は、人ごとの持ち時間で切れる（`expire_presence`）。
        すでに居る人は由来（声・手入力）を変えずに数え直す。
        """
        now = time.time()
        with self._lock:
            for pid, conf in people:
                row = self._present.get(pid)
                if row is None:
                    self._present[pid] = PersonPresence(
                        person_id=pid, confidence=conf, source=self.GUESS_SOURCE
                    )
                else:
                    row.last_signal_at = now
                    if row.source == self.GUESS_SOURCE:
                        row.confidence = conf  # 見立てで入った人は、新しい見立ての確かさにする
        ids = [pid for pid, _ in people]
        logger.info("写真の見立てで顔ぶれに入れた・数え直した：%s", self._names_of(ids))

    def _names_of(self, ids) -> str:
        """ログ用の名前。**名前が引けなくても落ちない**（人物表を持たない器もある）。"""
        out = []
        for pid in ids:
            try:
                out.append(self.get_person_name(pid) or pid)
            except Exception:  # noqa: BLE001
                out.append(pid)
        return "・".join(out) or "誰も居ない"

    def mark_absent(self, person_id: str) -> None:
        """顔ぶれ表からだけ消す（`person_left` との違い）。

        在/不在の層が「誰も居ない」を見続けたときの失効に使う（2026-09-17）。話者の指定は
        別の寿命で切れる（`clear_speaker`・`agent.speaker_known`・知-t・2026-09-18）。
        """
        with self._lock:
            self._present.pop(person_id, None)

    async def person_left(self, person_id: str) -> None:
        """Register that someone has left the space."""
        with self._lock:
            self._present.pop(person_id, None)
            if self._speaker_id == person_id:
                self._speaker_id = None
        logger.info("Left: %s  (total present: %d)", person_id, len(self._present))

    UNKNOWN_KEY_PREFIX = "unknown:"

    def note_unknown_present(self, count: int, confidence: float = 0.5) -> None:
        """写真に名前の分からない人が `count` 人写った（出-ae-は・知-ai・2026-10-05）。

        札は 1 人目から `count` 人目まで持ち時間を数え直し、足りなければ足す。**写っていない札は消さない**
        （名前の付いた人と同じく、持ち時間で切れる）。見立てごとに足し続けないよう、札は番号で使い回す。
        話者にはしない（話者には人を指す id が要る）。
        """
        n = max(0, int(count))
        now = time.time()
        with self._lock:
            for i in range(n):
                key = f"{self.UNKNOWN_KEY_PREFIX}{i + 1}"
                row = self._present.get(key)
                if row is None:
                    self._present[key] = PersonPresence(
                        person_id=key, confidence=confidence, anonymous=True
                    )
                else:
                    row.last_signal_at = now
        if n:
            logger.info(
                "名前の分からない顔ぶれ %d 人を入れた・数え直した（確信度 %.2f）", n, confidence
            )

    def expire_presence(self, hold_sec: float) -> "list[str]":
        """持ち時間（`hold_sec`）が切れた人を顔ぶれから外し、外した鍵を返す（知-ai・2026-10-05）。

        T の tick が呼ぶ。在席（カメラ）は見ない。話者の指定は別の寿命（`speaker_known`・知-t）で切れる。
        """
        now = time.time()
        with self._lock:
            gone = [k for k, p in self._present.items() if now - p.last_signal_at > hold_sec]
            for k in gone:
                self._present.pop(k, None)
        if gone:
            logger.info("顔ぶれの持ち時間が切れた：%s", self._names_of(gone))
        return gone

    def get_present_ids(self) -> list[str]:
        """**人を指す id だけ**を返す。名前の分からない顔ぶれは含めない。"""
        with self._lock:
            return [pid for pid, p in self._present.items() if not p.anonymous]

    def present_keys(self) -> list[str]:
        """顔ぶれ表の鍵を全部（**名前の分からない顔ぶれの札も含む**）。

        `get_present_ids()` との違い：あちらは観測の `participants` になるので人を指す id
        だけを返す。こちらは顔ぶれ表そのものを畳む用（失効・出-am）。
        """
        with self._lock:
            return list(self._present.keys())

    def refresh_signal(self, person_id: str) -> None:
        """Update the last-signal timestamp for a present person."""
        with self._lock:
            if person_id in self._present:
                self._present[person_id].last_signal_at = time.time()

    def stale_present_ids(self) -> list[str]:
        """Return IDs of persons whose last signal is older than PRESENCE_TIMEOUT_SEC."""
        now = time.time()
        with self._lock:
            return [
                pid
                for pid, p in self._present.items()
                if (now - p.last_signal_at) > PRESENCE_TIMEOUT_SEC
            ]

    # ── Speaker management ─────────────────────────────────────────────────

    async def set_speaker(
        self, person_id: str, source: str = "manual", confidence: float | None = None
    ) -> bool:
        """Declare who is speaking. Adds them to present if not already.

        `source`/`confidence` は誰がどう決めたか（顔/声/手動＋確信度）で、GUI 表示に使う。
        """
        if person_id not in self._present:
            await self.person_arrived(person_id)
        else:
            self.refresh_signal(
                person_id
            )  # その人だと分かる印なので、顔ぶれの持ち時間を数え直す（知-ai）
        old = self._speaker_id
        with self._lock:
            self._speaker_id = person_id
            self._speaker_source = source
            self._speaker_confidence = confidence
        if old != person_id:
            logger.info("Speaker: %s → %s  (source=%s)", old, person_id, source)
            for cb in self._switch_callbacks:
                try:
                    await cb(old, person_id)
                except Exception as e:
                    logger.warning("Switch callback error: %s", e)
        return old != person_id

    def clear_speaker(self) -> None:
        """話者の指定を「分からない」に戻す（寿命切れ・知-t・2026-09-18）。顔ぶれ表は触らない。"""
        with self._lock:
            old, self._speaker_id = self._speaker_id, None
            self._speaker_source = "auto"
            self._speaker_confidence = None
        if old is not None:
            logger.info("Speaker: %s → None（指定が切れた）", old)

    @property
    def current_speaker_id(self) -> str | None:
        return self._speaker_id

    # ── Recognition hint processing ────────────────────────────────────────

    async def apply_hint(self, hint: RecognitionHint) -> bool:
        """Apply a recognition signal. Returns True if a switch happened."""
        self.refresh_signal(hint.person_id)
        if hint.source in hint.IMMEDIATE:
            return await self.set_speaker(
                hint.person_id, source=hint.source, confidence=hint.confidence
            )
        threshold = self._switch_thresholds.get(hint.source, AUTO_SWITCH_THRESHOLD)
        if hint.confidence >= threshold:
            return await self.set_speaker(
                hint.person_id, source=hint.source, confidence=hint.confidence
            )
        logger.debug(
            "Hint below threshold: src=%s pid=%s conf=%.2f (th=%.2f)",
            hint.source,
            hint.person_id[:8],
            hint.confidence,
            threshold,
        )
        return False

    # ── Memory access ──────────────────────────────────────────────────────

    def get_speaker_memory(self) -> "ObservationMemory | None":
        """Write target: current speaker's memory. None if no speaker known."""
        if self._speaker_id is None:
            return None
        return self._get_or_create(self._speaker_id)

    def get_memory_for(self, person_id: str) -> "ObservationMemory":
        return self._get_or_create(person_id)

    def get_agent_memory(self) -> "ObservationMemory":
        return self._get_or_create(AGENT_SELF_ID)

    def get_all_present_memories(
        self,
    ) -> list[tuple[str, "ObservationMemory"]]:
        """Read targets: memories of all present persons."""
        return [(pid, self._get_or_create(pid)) for pid in self.get_present_ids()]

    def _get_or_create(self, person_id: str) -> "ObservationMemory":
        if person_id not in self._instances:
            self._instances[person_id] = self._base.for_person(person_id)
        return self._instances[person_id]

    # ── Person registry helpers ────────────────────────────────────────────

    def register_person(self, name: str, display_name: str = "") -> str:
        return self._base.register_person(name, display_name)

    def list_persons(self) -> list[dict]:
        return self._base.list_persons()

    def update_display_name(self, person_id: str, display_name: str) -> bool:
        return self._base.update_display_name(person_id, display_name)

    def get_speaker_info(self) -> dict | None:
        if self._speaker_id is None:
            return None
        persons = {p["id"]: p for p in self.list_persons()}
        return persons.get(self._speaker_id)

    def speaker_status(self) -> dict | None:
        """現在の話者の統合ビュー（GUI 表示用）。話者未定なら None。

        name は表示名、source は誰が決めたか（顔/声/手動/auto）、confidence はその確信度。
        声紋をライブ結線したあとは source/confidence がその結果に置き換わる。
        """
        if self._speaker_id is None:
            return None
        return {
            "person_id": self._speaker_id,
            "name": self.get_person_name(self._speaker_id),
            "source": self._speaker_source,
            "confidence": self._speaker_confidence,
        }

    def presence_status(self) -> list[dict]:
        """顔ぶれの一覧の統合ビュー（GUI 表示用）。name・confidence・is_speaker。"""
        with self._lock:
            present = list(self._present.values())
            speaker = self._speaker_id
        return [
            {
                "person_id": None if p.anonymous else p.person_id,
                "name": "不明" if p.anonymous else self.get_person_name(p.person_id),
                "confidence": p.confidence,
                "is_speaker": (not p.anonymous) and p.person_id == speaker,
            }
            for p in present
        ]

    def get_person_name(self, person_id: str) -> str:
        """呼びかけに使う名前。`display_name` は別名の一覧なので先頭だけを返す。

        `display_name` は FAMILY.md の「呼び方」で "パパ、いくながさん、ゆうすけ" のように
        読点区切りの一覧になる（`find_person_id_by_name` も割って照合している）。一覧のまま
        渡すと顔ぶれの文脈が `(present :speaker "たいきくん、たいき")` になり、呼びかけも
        モデルがどれを選ぶか任せになる。
        """
        persons = {p["id"]: p for p in self.list_persons()}
        display = persons.get(person_id, {}).get("display_name", person_id[:8])
        for sep in ("、", ","):
            display = display.split(sep)[0]
        return display.strip()

    def set_family_md(self, text: str) -> None:
        """いまの `FAMILY.md`。人を指す言葉を名前に直すのに使う（起動時と `/reload` に渡す・知-af）。"""
        self._family_md = text or ""

    def find_person_id_by_name(self, name: str) -> str | None:
        """人を指す言葉（名前・呼びかけ名・呼び方の別名）から、人物の id を引く（知-af・2026-10-02）。

        **`FAMILY.md`（いまの記述）で名前に直してから、人物表を名前の完全一致で引く。** 人物表の
        `display_name` は最初に登録したときの呼び方のまま止まっているので、照らすのには使わない。以前は
        `display_name` の別名でも当て、当たる行が複数あれば作られた順の最初の行を黙って選んでいた——名前を
        書き換えて行が二重になると、記憶の行き先が並び順で決まった。

        言葉が 2 人以上に当たれば誰も選ばず None（ログに残す）。`FAMILY.md` に無い言葉も None。
        `FAMILY.md` が無ければ、名前の完全一致だけで引く。
        """
        from .core.parsing import parse_family_md
        from .core.speaker_claim import aliases_of

        word = (name or "").strip()
        if not word:
            return None
        target = word
        members = parse_family_md(getattr(self, "_family_md", "") or "")
        if members:
            hits = [m for m in members if word in aliases_of(m)]
            if len(hits) > 1:
                logger.error(
                    "人物：「%s」は 2 人以上（%s）に当たるので、誰とも決めない（FAMILY.md の呼び方が重なっている）",
                    word,
                    "・".join(str(m["name"]) for m in hits),
                )
                return None
            if not hits:
                return None
            target = str(hits[0]["name"])
        for p in self.list_persons():
            if str(p.get("id", "")) in (AGENT_SELF_ID, DEFAULT_PERSON_ID):
                continue
            if p.get("name") == target:
                return str(p["id"])
        return None
