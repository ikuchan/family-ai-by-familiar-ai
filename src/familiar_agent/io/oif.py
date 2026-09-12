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
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..mood_register import MoodPAD
from ..store.relations import KIND_REVISION

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
    """記憶項目。**出来事 × 関係の面**の1つに対応する（案3・2026-09-02）。

    044 で situated が索引から記憶になり、面ごとの言葉・時間の起点・根づきの `n` が面へ
    移った。想起も面のベクトルで探している。**MI が指すのは面である。**

    **出来事ごとの量と面ごとの量を混ぜない。** `content`・`groundedness_n`・
    `last_recalled_at` は面のもの、`groundedness_g0`（取込の驚き）・`timestamp`・
    `parent_id` は出来事のものである（`MIデータモデル` §5）。

    表は分けたままにしてある。畳む印が面の側に無いことが「**畳んでも面は残る**」を
    構造として保証しており、版チェーンはこれに依存している。畳む印は関係の側にある
    （`設計方針_MI間の関係` 段 2 で `superseded_by` 列を落とした）。

    読み手ごとに調べて決めた点は変えていない。主想起・拡散想起・プロンプトのいずれからも
    読まれない列は入れず、計算で作れるものも入れない（`kind`・`emotion_vec`・ベクトル・
    採点）。撤去した列（`person_id` の所有者・`scope`・`importance`）も入れない。
    """

    # `id` は面（`situated_memories.id`）で、upsert しても保たれる。
    id: str
    content: str  # 【面】の言葉。面が持たないとき（actor）は出来事の本文
    # 書くときは空でよい（O が書込時刻を付ける）。読んだ MI には必ず入っている。
    timestamp: datetime | None
    direction: str

    # 面の同定。**書くときは空でよい**（まだ面が立っていない）。読んだ MI には必ず入る。
    # 誰がしたこと・誰が居たかは `OIF.write` の引数で渡し、書いた直後に面が立つ。
    obs_id: str = ""
    person_id: str = ""  # 誰との関係か（**所有者ではない**）
    relation_key: str = ""  # どの役割か（actor / present / about / …）

    emotion: str = "neutral"

    # 版。`parent_id` は過去（この記録を起こしたもの）を、`superseded_by` は未来
    # （この記録を置き換えたもの）を指す。後者は関係から引いて渡されたときだけ入る。
    # （この記録を置き換えた版）を指す。実データで4通りの組み合わせがすべて存在し、
    # 片方から他方を導けない。
    parent_id: str | None = None
    superseded_by: str | None = None

    # **未測定でありうる**（050）。`None` は「測っていない」で、中立とは違う。中立で埋めると
    # 測ったのか埋めたのかが後から見分けられず、気分の材料が一点に潰れる。
    pad: "MoodPAD | None" = None

    # 根づき と 新しさ は**素**で持ち、導出は計算する。
    groundedness_g0: float = 1.0
    groundedness_n: int = 0
    last_recalled_at: datetime | None = None

    image_path: str | None = None
    image_data: str | None = None

    @property
    def kind(self) -> str:
        """`direction` から決まる粗い分類。表に無いものは observation。"""
        return _KIND_OF_DIRECTION.get(self.direction, _DEFAULT_KIND)


@dataclass(frozen=True)
class Said:
    """やりとりの中で口に出した1件（関係の項）。

    `role` は `起点`（相手が言った）・`つなぎ`・`答え`（どちらもパジュが言った）。
    `depth` は 0 が渡した起点で、さかのぼるほど大きい。
    """

    content: str
    role: str
    when: "datetime | None"
    depth: int


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

    # **誰の面から引くか。** `situated_memories` は人ごとで、想起は話者の面を通る。
    # 空なら口が持つ記憶のまま（基底＝視点は `__self__` に寄る）。docstring が「読むときの
    # 視点」と名乗っていたのに、この欄が無かった（環-e-い・2026-09-11）。
    viewpoint: str = ""
    k: int = 7  # W へ載せる上限（課題5 の確定値）
    floor: float = 0.05  # 合成スコアの床
    weights: "RecallWeights | None" = None  # 5軸の重み（trigger 別）
    present: tuple[str, ...] = ()  # 在席者（p 軸）
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
    # どれだけ確かか（0〜1）。生コサインを 0〜1 へ移したもので、**保存する値ではない**
    # ので `MI` には入れない。W の1行と整合チェックの材料がこれを読む。
    confidence: float = 0.0


class Verdict(Enum):
    """使った記憶の申告。参照した MI だけ再評価する。

    想起しただけで更新すると、一度上がった記録が自分を押し上げ続ける（実機で 47 日前の
    挨拶が新しさ 1.000 で居座った）。
    """

    IMPORTANT = "important"  # 根づき +1 ＋ 新しさの起点を更新
    USELESS = "useless"  # 根づき −1 ＋ 同上
    REFERRED = "referred"  # 新しさの起点だけ更新
    UNUSED = "unused"  # 何もしない


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

    def __init__(self, memory, *, for_person=None) -> None:
        """`memory`＝既定の記憶。`for_person`＝その人の面を返すもの（`View.viewpoint` 用）。

        **人ごとの実体は呼び手が持つ。** `PersonMemoryManager` が1人につき1つ持っており、
        口が `ObservationMemory.for_person` を都度呼ぶと実体が増える。渡されなければ
        その `for_person` へ落ちる（テストと、人を持たない呼び手のため）。
        """
        self._memory = memory
        self._for_person = for_person or getattr(memory, "for_person", None)

    def _through(self, viewpoint: str):
        """その視点で引く記憶を返す。視点が無ければ既定の記憶のまま。"""
        if not viewpoint or self._for_person is None:
            return self._memory
        return self._for_person(viewpoint)

    async def write(
        self,
        mi: MI,
        *,
        writer_id: str,
        now: bool = True,
        participants: "list[str] | None" = None,
        arousal: "float | None" = None,
    ) -> str:
        """記憶を1件書き、**出来事の** id を返す。空欄は書き込み側が埋める。

        `writer_id`（誰がしたことか）と `participants`（誰が居たか）は**書き込みの材料**で、
        MI の属性ではない（案3）。書いた直後に `actor` と `present` の面が立ち、以後は
        その面が「誰との関係か」を持つ。読むときの MI は面を指すので、視点を属性として
        持ち回る必要がない。

        **書き手は必須である**（記-f・2026-09-11）。既定を置くと、渡し忘れたときに
        `writer = writer_id or self._ctx.person_id` へ落ち、**書く側のインスタンス次第**に
        なる。`好奇心` と `記憶` が実際にその形だった——2つの既定の一致で結果は正しかった
        が、読んでも分からず、記憶を差し替えれば黙って変わる。**渡し忘れたら落ちるほうが、
        黙って別の人の記録になるより良い**（`apply_memory_verdicts` の対応表と同じ判断）。

        空文字も受け取らない。`None` を弾いても空で素通りできれば同じことになる。
        パジュ自身の記録なら `AGENT_SELF_ID` を、人の言葉ならその人の id を渡す。
        """
        if not writer_id:
            raise ValueError("誰の記録かを渡していない（OIF.write の writer_id）")
        # **その日づけは MI が持つ。** 日次要約は「その日のこと」として残すので、書いた
        # 時刻ではなくその日の終わりを時刻にする（`override_date`）。`timestamp` が空なら
        # いまの時刻になる。
        override_date = mi.timestamp.strftime("%Y-%m-%d") if mi.timestamp else None
        logger.debug("OIF write ← %s／%d字／%s", mi.direction, len(mi.content), _head(mi.content))
        obs_id, _ = await self._memory.save_async_with_id(
            mi.content,
            direction=mi.direction,
            kind=mi.kind,
            emotion=mi.emotion,
            emotion_pad=mi.pad,
            parent_id=mi.parent_id,
            writer_id=writer_id,
            participants=participants,
            image_path=mi.image_path,
            materialize_now=now,
            override_date=override_date,
            # そのとき外からどれだけ揺さぶられたか（a0／A の源）。測っていなければ渡さない。
            arousal=arousal,
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
        logger.debug(
            "OIF recall ← %s／k=%d／床=%.2f／面=%s",
            _head(cue.text),
            view.k,
            view.floor,
            view.viewpoint or "既定",
        )
        mem = self._through(view.viewpoint)
        # **手がかりの欄が、どう探すかを言う**（`Cue` の docstring）。以前は別メソッドに
        # 分かれていたものを、欄の違いで表す。
        if cue.on_month_day:
            month, day = cue.on_month_day
            rows = await mem.recall_on_this_day_async(month, day, view.k)
        elif cue.direction == "記憶" and not cue.text:
            # その日のまとめは**新しい順**に採る。手がかりが無いので似ている順では引けない。
            rows = await mem.recall_day_summaries_async(n=view.k)
        else:
            rows = await self._recall_by_cue(mem, cue, view)
        out = [_to_recalled(r) for r in (rows or [])]
        logger.debug("OIF recall → %d件", len(out))
        return out

    async def _recall_by_cue(self, mem, cue: "Cue", view: "View") -> list:
        """手がかりの言葉で、似ている順に探す（いちばん普通の道）。"""
        return await mem.recall_async(
            cue.text,
            n=view.k,
            kind=_KIND_OF_DIRECTION.get(cue.direction, _DEFAULT_KIND) if cue.direction else None,
            min_score=view.floor,
            present_others=list(view.present) or None,
            exclude_ids=list(cue.exclude) or None,
            time_ref=view.time_ref,
            time_span_days=view.time_span_days,
            weights=view.weights,
            open_ids=list(cue.open_ids) or None,
        )

    # ── 関係（MI 間のつながり）──────────────────────────────────────────
    # 関係は 058〜060（2026-09-06〜07）で入った機構で、2026-08-01 に設計したこの口に
    # 面が無かった。呼び手は7箇所あり、うち1つは記憶の私的属性（`_ctx`）を掴んでいた。

    def link(self, kind: str, members: "Sequence[tuple[str, str, int | None]]") -> "int | None":
        """関係を1つ書き、その id を返す。**書く口は1つ**で、種類は `kind` が言う。

        `members` は `(記録の id, 役割, 位置)` の並び。位置は順序を持たない関係で `None`。
        種類の語は `store/relations.py` の定数（`KIND_EXCHANGE` ほか）を使う。

        畳む（`supersede`）だけは別の面である——先着勝ちで、部分一意索引が二本目を落とす。
        """
        got = self._memory.link(kind, members)
        logger.debug("OIF link ← %s／%d項 → %s", kind, len(members), got)
        return got

    def exchanges(self, origin_id: str) -> "list[Said]":
        """継起をさかのぼり、各やりとりで**口に出した**項を古い順に返す。

        `版` と `見た` は内部の作業記録で会話ではないので入らない（store 側の既定）。
        """
        rows = self._memory.recent_exchanges(origin_id)
        got = [
            Said(
                content=str(r.get("content", "")),
                role=str(r.get("role", "")),
                when=r.get("timestamp"),
                depth=int(r.get("depth", 0)),
            )
            for r in (rows or [])
        ]
        logger.debug("OIF exchanges ← %.8s → %d件", origin_id, len(got))
        return got

    def actors(self, obs_ids: "Sequence[str]") -> "dict[str, str]":
        """その記録たちの**主体**を `記録の id → 名前` で返す（`__self__` は `わたし`）。

        「誰がやったか」は `actor` の面が持つ（記-f）。面が立っていない記録は入らない
        ——呼び手は主体を言わない（名前を捏造しない）。
        """
        got = self._memory.actor_names_of(list(obs_ids))
        logger.debug("OIF actors ← %d件 → %d件", len(obs_ids), len(got))
        return got

    def roles(self, obs_ids: "Sequence[str]") -> "dict[str, str]":
        """その記録たちが**やりとり**で取っている役割を `記録の id → 役割` で返す。

        **誰が言ったかは、ここが持つ。** `起点` は相手、`答え`・`つなぎ` はパジュ。`actor` の
        面は「誰の記憶か」であって、話者が解決できないと規則 048 で `__self__` に寄る
        （相手の言葉が「わたしが言った」と印字されていた・2026-09-12）。
        """
        got = self._memory.exchange_roles_of(list(obs_ids))
        logger.debug("OIF roles ← %d件 → %d件", len(obs_ids), len(got))
        return got

    def latest_origin(self) -> "str | None":
        """いちばん新しいやりとりの起点。**繋ぐためではなく、どこから見せるかのカーソル**。"""
        got = self._memory.latest_exchange_origin()
        logger.debug("OIF latest_origin → %s", got)
        return got

    async def novelty(self, content: str) -> float:
        """その内容がどれだけ新しいか（0〜1）。近い記憶があるほど低い。"""
        got = float(await self._memory.content_novelty_async(content))
        logger.debug("OIF novelty ← %d字 → %.3f", len(content), got)
        return got

    def supersede(self, old_id: str, new_id: str, kind: str = KIND_REVISION) -> bool:
        """old を現行から外し、new を新しい側にする。先着勝ち。

        `kind` は畳む理由で、隠すかどうかには効かない（役割 `旧` が決める）。
        """
        got = bool(self._memory.mark_superseded(old_id, new_id, kind=kind))
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
        got = Health(
            ready=bool(self._memory.is_embedding_ready()),
            failed=bool(self._memory.embedding_failed()),
        )
        logger.debug("OIF health → ready=%s failed=%s", got.ready, got.failed)
        return got


def _to_recalled(row: dict) -> Recalled:
    """想起が返す行を MI ＋ 採点へ移す。

    `summary` は面の言葉（面が持たなければ出来事の本文）、`memory_id` は出来事の id、
    `facet_id` は面の id である。**MI の `id` は面**で、出来事は `obs_id` が指す。

    `memory_id` が出来事のままなのは、拡散想起の種・除外・supersede・共起の記録が
    すべて出来事の id で動いているためである。
    """
    return Recalled(
        mi=MI(
            id=str(row.get("facet_id") or row.get("memory_id") or row.get("id", "")),
            obs_id=str(row.get("memory_id") or row.get("id", "")),
            person_id=str(row.get("person_id", "")),
            relation_key=str(row.get("relation_key", "")),
            content=str(row.get("summary") or row.get("content", "")),
            timestamp=row.get("timestamp"),
            direction=str(row.get("direction", "")),
            emotion=str(row.get("emotion", "neutral")),
            pad=row.get("emotion_pad"),  # 未測定なら None のまま（050）
            parent_id=row.get("parent_id"),
            superseded_by=row.get("superseded_by"),
            groundedness_g0=float(row.get("groundedness_g0", 1.0)),
            groundedness_n=int(row.get("groundedness_n", 0)),
            last_recalled_at=row.get("last_recalled_at"),
            image_path=row.get("image_path"),
        ),
        fit=float(row.get("fit", 0.0)),
        groundedness=float(row.get("groundedness", 0.0)),
        confidence=float(row.get("confidence", 0.0)),
    )


__all__ = [
    "MI",
    "OIF",
    "Cue",
    "View",
    "Recalled",
    "Verdict",
    "Span",
    "Health",
]
