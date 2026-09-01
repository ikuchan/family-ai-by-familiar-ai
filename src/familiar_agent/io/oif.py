"""記憶接続 OIF：記憶ストア O との唯一の出入り口（`設計方針_OIF` v0.1）。

`ObservationMemory` の公開面は 72 種あり、口として働いていなかった。外から呼ばれるのは
63 種、うち本番コードから呼ばれるのは 22 種である。触る側も 10 ファイルに散っている。

OIF はその 22 種を **8 つの口**へまとめ、`ObservationMemory`（と、その内側のベクトル
埋め込み）を抱える。通ったものは debug ログに残す。

この段では呼び出し側を付け替えない。口の中身を整えるのと、呼び出し側を寄せるのは別の
作業で、一度にやると挙動の変化と移動の区別がつかなくなる。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING

from ..mood_register import MoodPAD

if TYPE_CHECKING:
    from ..config import RecallWeights


# ── MI ─────────────────────────────────────────────────────────────────────

# `direction` から `kind` を決める表。**`kind` は列に持たない。**
# 実測（本番 DB・2026-08-01）で 12 種の `direction` に対し `kind` は 6 種しかなく、
# 8 つの `direction` がすべて `observation` に落ちる。片方から他方が決まる。
_KIND_OF_DIRECTION: dict[str, str] = {
    "会話": "conversation",
    "内省": "self_model",
    "好奇心": "curiosity",
    "記憶": "day_summary",
}
_DEFAULT_KIND = "observation"


@dataclass
class MI:
    """記憶項目。O の1行に対応する。

    属性は 18 で、読み手ごとに調べて決めた。主想起・拡散想起・プロンプトのいずれからも
    読まれない列は入れず、計算で作れるものも入れない（`kind`・`emotion_vec`・ベクトル・
    採点）。撤去する列（`person_id`・`scope`・`importance`）も入れない。
    """

    id: str
    content: str
    # 書くときは空でよい（O が書込時刻を付ける）。読んだ MI には必ず入っている。
    timestamp: datetime | None
    direction: str

    emotion: str = "neutral"

    # 版。`parent_id` は過去（この記録を起こしたもの）を、`superseded_by` は未来
    # （この記録を置き換えた版）を指す。実データで4通りの組み合わせがすべて存在し、
    # 片方から他方を導けない。
    parent_id: str | None = None
    superseded_by: str | None = None

    pad: MoodPAD = field(default_factory=MoodPAD)

    # 根づき と 新しさ は**素**で持ち、導出は計算する。
    groundedness_g0: float = 1.0
    groundedness_n: int = 0
    last_recalled_at: datetime | None = None

    # 視点。拡散想起のエンティティ辺（`diffuse_store.recall_by_person`）が使う。
    # situated では代替できない（situated の person は「誰の視点で符号化したか」で、
    # ここが要るのは「誰についての記録か」である）。
    writer_id: str = ""
    subject_id: str = ""
    participants: tuple[str, ...] = ()

    image_path: str | None = None
    image_data: str | None = None

    @property
    def kind(self) -> str:
        """`direction` から決まる粗い分類。表に無いものは observation。"""
        return _KIND_OF_DIRECTION.get(self.direction, _DEFAULT_KIND)


# ── 想起の引数と戻り ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Cue:
    """何を手がかりに探すか。

    いま別メソッドになっているもの（日次要約・同じ月日・その日の記録）を、欄の違いで
    表す。`recall_day_summaries` は `Cue(direction="記憶")`、`recall_on_this_day` は
    `Cue(on_month_day=(8, 1))`、`get_observations_for_date` は `Cue(on_date=...)` になる。
    """

    text: str = ""
    direction: str | None = None
    on_date: date | None = None
    on_month_day: tuple[int, int] | None = None
    exclude: tuple[str, ...] = ()
    open_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class View:
    """どう探すか（見方）。省略すれば既定。

    書き込み側の**視点**（`writer_id`・`subject_id`・`participants`）とは別のもので、
    語を分けないと中身が混ざる。
    """

    k: int = 7                                   # W へ載せる上限（課題5 の確定値）
    floor: float = 0.05                          # 合成スコアの床
    weights: "RecallWeights | None" = None       # 5軸の重み（trigger 別）
    present: tuple[str, ...] = ()                # 在席者（p 軸）
    time_ref: float | None = None
    time_span_days: float | None = None


@dataclass(frozen=True)
class Recalled:
    """想起された記憶。MI に、そのときの採点を添えたもの。

    採点は保存しない導出値なので MI には入れない。同じ記録でも問いが変われば変わる。
    """

    mi: MI
    fit: float
    groundedness: float


class Verdict(Enum):
    """使った記憶の申告。参照した MI だけ再評価する。

    想起しただけで更新すると、一度上がった記録が自分を押し上げ続ける（実機で 47 日前の
    挨拶が新しさ 1.000 で居座った）。
    """

    IMPORTANT = "important"     # 根づき +1 ＋ 新しさの起点を更新
    USELESS = "useless"         # 根づき −1 ＋ 同上
    REFERRED = "referred"       # 新しさの起点だけ更新
    UNUSED = "unused"           # 何もしない


@dataclass(frozen=True)
class Span:
    """記憶の広がり。"""

    earliest: date | None = None


@dataclass(frozen=True)
class Health:
    """使える状態か。"""

    ready: bool = False
    failed: bool = False


# ── 口 ─────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ログに載せる本文の長さ。記憶の内容を出すのは debug に限り、そこでも先頭だけにする。
_TRAIL_CHARS = 24


def _head(text: str) -> str:
    return text[:_TRAIL_CHARS].replace("\n", " ")


class OIF:
    """記憶ストア O との唯一の出入り口。

    中で `ObservationMemory` が `store/` の各層とベクトル埋め込みを持つ。外へ出すのは
    8つの口だけで、待ち行列（`memory_jobs`）は内側の都合として隠す。
    """

    def __init__(self, memory) -> None:
        self._memory = memory

    async def write(self, mi: MI, *, now: bool = True) -> str:
        """記憶を1件書き、その id を返す。空欄は書き込み側が埋める。"""
        logger.debug("OIF write ← %s／%d字／%s", mi.direction, len(mi.content),
                     _head(mi.content))
        obs_id, _ = await self._memory.save_async_with_id(
            mi.content,
            direction=mi.direction,
            kind=mi.kind,
            emotion=mi.emotion,
            emotion_pad=mi.pad,
            parent_id=mi.parent_id,
            writer_id=mi.writer_id,
            subject_id=mi.subject_id,
            participants=list(mi.participants),
            image_path=mi.image_path,
            materialize_now=now,
        )
        logger.debug("OIF write → %s", obs_id)
        return str(obs_id or "")

    async def append(self, mi_id: str, note: str) -> bool:
        """既存の記録へ足し、埋め込みを作り直す。足したときだけ True。"""
        logger.debug("OIF append ← %s／%s", mi_id, _head(note))
        got = bool(self._memory.note_lookup_started(mi_id))
        logger.debug("OIF append → %s", got)
        return got

    async def recall(self, cue: Cue, view: View = View()) -> list[Recalled]:
        """手がかりで探し、適合度の高い順に返す。"""
        logger.debug("OIF recall ← %s／k=%d／床=%.2f", _head(cue.text), view.k, view.floor)
        rows = await self._memory.recall_async(
            cue.text,
            n=view.k,
            kind=_KIND_OF_DIRECTION.get(cue.direction or "", None)
            if cue.direction else None,
            min_score=view.floor,
            present_others=list(view.present) or None,
            exclude_ids=list(cue.exclude) or None,
            time_ref=view.time_ref,
            time_span_days=view.time_span_days,
            weights=view.weights,
            open_ids=list(cue.open_ids) or None,
        )
        out = [_to_recalled(r) for r in (rows or [])]
        logger.debug("OIF recall → %d件", len(out))
        return out

    async def novelty(self, content: str) -> float:
        """その内容がどれだけ新しいか（0〜1）。近い記憶があるほど低い。"""
        got = float(await self._memory.content_novelty_async(content))
        logger.debug("OIF novelty ← %d字 → %.3f", len(content), got)
        return got

    def supersede(self, old_id: str, new_id: str) -> bool:
        """old を new の版で置き換える。先着勝ち。"""
        got = bool(self._memory.mark_superseded(old_id, new_id))
        logger.debug("OIF supersede ← %s→%s → %s", old_id, new_id, got)
        return got

    def feedback(self, verdicts: dict[str, Verdict]) -> int:
        """使った記憶の申告を反映する。触れた件数を返す。"""
        plain = {i: v.value for i, v in verdicts.items()}
        got = int(self._memory.apply_verdicts(plain))
        logger.debug("OIF feedback ← %d件 → %d件へ反映", len(plain), got)
        return got

    async def span(self) -> Span:
        """記憶の広がり（最古の日付）。"""
        raw = await self._memory.get_earliest_date_async()
        got = Span(earliest=date.fromisoformat(raw) if raw else None)
        logger.debug("OIF span → %s", got.earliest)
        return got

    def health(self) -> Health:
        """使える状態か（埋め込みが載っているか、失敗していないか）。"""
        got = Health(ready=bool(self._memory.is_embedding_ready()),
                     failed=bool(self._memory.embedding_failed()))
        logger.debug("OIF health → ready=%s failed=%s", got.ready, got.failed)
        return got


def _to_recalled(row: dict) -> Recalled:
    """想起が返す行を MI ＋ 採点へ移す。

    `summary`／`memory_id` は O 側の `content`／`id` と同じものが別の名前で出ている。
    ここで名前を1つに揃える。
    """
    return Recalled(
        mi=MI(
            id=str(row.get("memory_id", "")),
            content=str(row.get("summary", "")),
            timestamp=row.get("timestamp"),
            direction=str(row.get("direction", "")),
            emotion=str(row.get("emotion", "neutral")),
            pad=row.get("emotion_pad") or MoodPAD(),
            image_path=row.get("image_path"),
        ),
        fit=float(row.get("fit", 0.0)),
        groundedness=float(row.get("groundedness", 0.0)),
    )


__all__ = [
    "MI", "OIF", "Cue", "View", "Recalled", "Verdict", "Span", "Health",
]
