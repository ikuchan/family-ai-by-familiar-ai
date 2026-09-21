"""作業記憶 W：反復ごとに組み、反復ごとに捨てる（環-e-に・に-5-に-2）。

`_iterate` と `_act_on_decision` は**殻**である——`_dispatch_main_llm`・`_finish`・`_speak`・
`_write_version` などを **10個と8個**呼び返す（ここへ移す前は 11個と9個）。別 file へ出せば
`InformationProcessing` への逆参照が要り、に-4 で断った「核が殻を呼び返す」形が戻る。
**殻は `event_loop.py` に残す。**

ここに集めたのは、W をめぐる**何も呼び返さない**ぶん（移す前の `event_loop.py` で 174 行）
である（Functional core / Imperative shell）。W の索引（12桁 → 完全な id）を組み、W から引く。

**class を作らずモジュール関数にする**（`loop/generator.py` と同じ形）。W は反復ごとに
作り直すもので、長生きの持ち主に抱えさせると寿命が混ざる（に-2「ニ．反復の寿命は束に
しない。引数と返り値で渡す」）。**求めは引数で受け取り、保持しない。**
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from ..core import measure
from ..io.oif import Cue, Recalled, View
from ..store import clock
from ..store.relations import KIND_SUCCESSION
from .request import Request

logger = logging.getLogger(__name__)


def open_ids(req: Request) -> list[str]:
    """この求めの open な記録（活性に下限を課して W へ浮かせる対象）。

    発話の記録（求めの親）と、**いま生きている版**である。版チェーンでは生きている版は
    常に1つなので、飛行中の意図を別に数える必要はない。トリガ O を求めが閉じるまで open
    扱いにするのは、完了で起きた反復では手がかりが**届いた結果の本文**に変わり、元の人の
    問いとは語彙が重なるとは限らないためである。完了プロファイルは関連を厳しく要求する
    （w_r=1.5）ので、下限が無いと「何のために調べていたか」が W から落ちる。
    """
    ids = [req.request_id] if req.request_id else []
    if req.live_version_id and req.live_version_id not in ids:
        ids.append(req.live_version_id)
    # **この求めで見たもの**（見た印）も浮かせる（2026-09-12 実機で露見）。`see` の帰りは
    # 版に載せない（同じ出来事が2件になって枠を食う）ので、中身は見た印だけが持つ。
    # 似ている順の採点に任せると「何が見えますか？」に `見えたもの：table…` が載らず、
    # 主LLM は見たらしいが何が見えたか書いていない版を渡されて、もう一度 `see` を出した。
    # 新しい欄は作らない——役割は `turn_records` に控えてある。前のやりとりのぶんは
    # `exchange_start` より前なので入らない。
    for obs_id, role in req.turn_records[req.exchange_start :]:
        if role == "見た" and obs_id and obs_id not in ids:
            ids.append(obs_id)
    return ids


def compose(
    oif,
    memories: "list[Recalled]",
    req: Request,
    *,
    exclude: "set[str] | None" = None,
    basis: str = "",
) -> "tuple[str, dict[str, str]]":
    """W を組み、**(W の文字列, 12桁 → 完全な id の対応表) を返す**。

    正本 [D-想起起動] は「O に乗った後は共通の流れ（O → 根づき → W 構築〔5軸採点〕→
    調停）で1本」と定める。以前は想起で拾った記録だけが採点を通り、ループ自身が O へ
    書いた記録（意図 O・完了 O）は採点を通らず手組みの文字列として連結されていた。
    そのため記録が W に載るかどうかが「畳むか畳まないか」で決まり、優先度の計算が
    どこにも効いていなかった。手組みをやめ、中身は候補集合の一員として入る。

    **1件の途中では切らない。** 枠（`workspace_max_chars`）を超えたら適合度の低い件から
    丸ごと落とす。切ると調べた結果の枕だけが残って中身が消える（実機で
    `「目の前を見る」を see で調べた結果が届いた：` だけが W に載った）。

    **対応表は W から導かれる**ので一緒に返す。別々に取れば片方だけ古くなる。フルLLM の
    申告を突き合わせるのに使い、前方一致で当てずっぽうに引くと、写し間違いが黙って別の
    記憶へ適用されてしまう。

    `basis`＝**この想起の引き方**（出-ah・2026-09-21）。面・件数・時期・直近の広さを 1 行で添える。
    引き方が見えないと、主LLM は「足りない」ことに気づけず、引き直す道具（`recall_as` ほか）を
    使うきっかけを持てない。空なら何も足さない。

    `said`（言ったつなぎ）と `held`（配る保留）は手組みのまま残す。どちらも O にあるが、
    `held` は `pending_store` が鮮度と配達を管理しており、想起とは別の規則を持つ。
    """
    from ..config import MemoryConfig

    budget = MemoryConfig().workspace_max_chars

    # 適合度の高い順に、枠へ入るぶんだけ採る。落ちたものは薄れた＝忘れたのであって、
    # 抜けを検出する仕組みは置かない（W は速く薄れる・改めて調べるのが自然な振る舞い）。
    ranked = sorted(memories, key=lambda r: r.fit, reverse=True)
    kept: "list[Recalled]" = []
    used = 0
    for r in ranked:
        size = len(r.mi.content)
        if kept and used + size > budget:
            continue
        kept.append(r)
        used += size
    dropped = len(memories) - len(kept)
    if dropped:
        # 何件落ちたかを残す。枠に収まったのか溢れたのかが分からないと、枠の値を
        # 決められない。記憶の内容は出さない。
        logger.info(
            "event-loop W に入らなかった記録＝%d件（枠 %d 字・載せた %d 件）",
            dropped,
            budget,
            len(kept),
        )
    # 想起が返した順（適合度の降順）を保つ。並べ替えた結果をそのまま渡す。
    memories = kept
    # 直近のやりとりの枠に載った記録は、過去の列には出さない（同じ話が二重に載る）。
    # 対応表には残す——申告は直近の行の id でも来る。黙っていたあいだの列挙に載せたものも同じ（情-h）。
    heard_ids = {h.obs_id for h in req.heard_while_silent if getattr(h, "obs_id", "")}
    shown = [
        r
        for r in memories
        if not (exclude and r.mi.obs_id in exclude) and r.mi.obs_id not in heard_ids
    ]

    id_map = {r.mi.obs_id.replace("-", "")[:12]: r.mi.obs_id for r in memories if r.mi.obs_id}
    # すでに相手へ伝えた一言。これが無いと、同じ言い回しを最初から言い直す
    # （実機で「〜ですね！」で始まる前置きが3回続いた）。
    said = ""
    if req.said_fillers:
        lines = "\n".join(f"- 「{t}」" for t in req.said_fillers)
        said = (
            "すでに相手へ伝えた一言（言った順。次に何か言うなら、"
            "同じ言い回しを繰り返さず、この続きとして自然につなぐ）：\n" + lines
        )
    heard = ""
    if req.heard_while_silent:
        # 黙っていたあいだに届いたもの（情-h）。想起を経ず、この枠で確実に全部（字数上限つき）。
        from ..core.silence_hold import render

        ats = [h.at for h in req.heard_while_silent]
        heard = render(
            list(req.heard_while_silent),
            since=min(ats),
            until=max(ats),
            max_chars=MemoryConfig().silent_heard_max_chars,
        )
    held = ""
    if req.speech_to_deliver:
        held = (
            "聞く相手が居ないあいだに話したかったこと"
            "（いま伝えるなら、そのときのこととして話す）：\n" + "\n".join(req.speech_to_deliver)
        )
    # **誰の記録かを添える。** `actor` の面（`situated_memories`）が「誰がやったか」を
    # 持っており、`direction` も `やりとり` の役割もその代わりにはならない。面が立って
    # いない記録は入らないので、その行は主体を言わない（名前を捏造しない）。
    #
    # **渡された記録を書き換えない。** 主体は印字のためのもので、呼び手が持つ記録
    # （共起・申告が使う）に足す理由がない。写しに載せる。
    names: "dict[str, str]" = {}
    with contextlib.suppress(Exception):
        names = oif.actors([r.mi.obs_id for r in memories])
    # **誰が言ったかは役割が持つ。** 話者が解決できないと `actor` の面は規則 048 で
    # `__self__` に寄り、相手の言葉が「わたしが言った」になる。`起点` は相手である。
    # いまの求めの分はまだ関係に無いので `turn_records` から、閉じた分は関係から。
    roles: "dict[str, str]" = {i: r for i, r in req.turn_records if i}
    with contextlib.suppress(Exception):
        roles = {**oif.roles([r.mi.obs_id for r in memories]), **roles}
    for r in memories:
        if roles.get(r.mi.obs_id) == "起点" and names.get(r.mi.obs_id, "わたし") == "わたし":
            names[r.mi.obs_id] = "相手"
    basis_line = f"[この想起：{basis}]" if basis else ""
    text = "\n\n".join(
        p for p in [said, heard, held, basis_line, _lines(shown, names)] if p and p.strip()
    )
    return text, id_map


#: 帰りの反復で**想起を回さない**道具（出-x・2026-09-18）。掛けた・確かめて・止めた、の返りは
#: 過去の記録を要さず、想起の列（同じ道具の過去の帰り 5〜6 件）が並ぶと調停はそれを「いまの頼み」
#: として読み、確かめてを返した帰りに掛け直した（実機 14:50・15:28・15:42。実験は `根拠台帳` §35）。
#: 検索・見る・思い出すの帰りは従来どおり想起を回す。
RETURN_WITHOUT_RECALL = frozenset(
    {
        "set_timer",
        "start_stopwatch",
        "cancel_timer",
        "pause_timer",
        "resume_timer",
        "set_alarm",
        "cancel_alarm",
        "stop_stopwatch",
        "confirm",
        "decline",
    }
)


def returned_actions(req: Request) -> frozenset[str]:
    """この反復で道具から返った action の集合（`Request.just_returned` の索引を引く）。"""
    idx = set(getattr(req, "just_returned", ()) or ())
    return frozenset(lk.action for lk in req.lookups if lk.index in idx and lk.result is not None)


def just_returned(req: Request) -> str:
    """W の最上部：**この反復で道具から返ったもの**（出-x・2026-09-18）。

    道具の返りは版（過去の列の全文）に載るだけで、調停は「記憶」として読み、いま届いた返りとして
    扱わなかった——`set_timer` が「確かめて」を返したのに聞き返さず掛け直した（実機 14:50）。
    実験（`scripts/experiment_arbiter_confirm.py`・8 回ずつ）：最上部に 1 行載せると 8/8 で
    light の確認文、先導文の差し替えだけでは 6/8。載せるのは取込がこの反復で受けた分だけ。
    """
    idx = set(getattr(req, "just_returned", ()) or ())
    if not idx:
        return ""
    lines = ["[いま道具から返った]"]
    for lk in sorted(req.lookups, key=lambda x: x.index):
        if lk.index in idx and lk.result is not None:
            lines.append(f"- {lk.action}「{lk.query}」→ {lk.result}")
    return "\n".join(lines) if len(lines) > 1 else ""


def recent_chains(oif, n: int) -> "list[tuple[str, list]]":
    """直近 n 往復の起点と、各起点から継起をさかのぼった鎖（新しい順）。引くのは 1 度。"""
    if n <= 0:
        return []
    origins: list[str] = []
    with contextlib.suppress(Exception):
        origins = list(oif.latest_origins(n))
    out: "list[tuple[str, list]]" = []
    for origin in origins:
        chain: list = []
        with contextlib.suppress(Exception):
            chain = list(oif.exchanges(origin))
        out.append((origin, chain))
    return out


def render_recent(
    oif,
    chains: "list[tuple[str, list]]",
    n: int,
    *,
    max_age_sec: int = 0,
    now: "datetime | None" = None,
) -> "tuple[list, str, dict[str, str]]":
    """直近の枠を文にする。窓 n は新しい側から n 往復（鎖ごと）。

    `max_age_sec`（0 で無し）より古い行は、起点でも鎖でさかのぼった行でも載せない（出-ae(2)）。
    往復数だけで切ると再起動をまたいで前の相手の名前が最上部に居続ける。外した行は直近の
    `exclude` から外れるので、点が付けば過去の記憶の列に出る。
    """
    if n <= 0:
        return [], "", {}
    seen: set[str] = set()
    rows: list = []
    for _origin, chain in chains[:n]:
        for r in chain:
            if r.obs_id and r.obs_id not in seen:
                seen.add(r.obs_id)
                rows.append(r)
    if max_age_sec > 0 and rows:
        floor = (now or clock.now_utc()).timestamp() - max_age_sec
        kept = [r for r in rows if r.when and r.when.timestamp() >= floor]
        if len(kept) < len(rows):
            logger.info("直近: %d 秒より古い %d 行を外した", max_age_sec, len(rows) - len(kept))
        rows = kept
    if not rows:
        return [], "", {}
    rows.sort(key=lambda r: r.when.timestamp() if r.when else 0.0)
    names: "dict[str, str]" = {}
    with contextlib.suppress(Exception):
        names = dict(oif.actors([r.obs_id for r in rows]))
    id_map = {r.obs_id.replace("-", "")[:12]: r.obs_id for r in rows}
    lines = [f"[直近のやりとり（古い順・最新 {n} 往復と、それに続く話）]"]
    for r in rows:
        sid = r.obs_id.replace("-", "")[:12]
        if r.role in ("答え", "つなぎ"):
            who = "わたし"
        elif getattr(r, "direction", "発話") in ("情動", "機器"):
            # 人の言葉でない起点（内的な促し・入室）。「相手」と書くと、自分の内側や
            # 機器の出来事が誰かの発言に読める（実機 2026-09-13 21:21）。
            who = "きっかけ"
        else:
            who = names.get(r.obs_id) or "相手"
            who = "相手" if who == "わたし" else who
        lines.append(f"- {clock.ts_to_time(r.when)} id:{sid} {who}：{r.content}")
    return rows, "\n".join(lines), id_map


def recent_window(
    oif, n: int, *, max_age_sec: int = 0, now: "datetime | None" = None
) -> "tuple[list, str, dict[str, str]]":
    """直近のやりとりの枠（記-h・`設計方針_MI間の関係` v0.15 段 4 改訂）。

    **時系列で最新 n 往復（無条件）＋ 各々から継起の辺があるぶんさかのぼった鎖。**
    「最近何があったか」は辺の有無に関係なく要り、「この話題の糸」は辺が担う。
    返りは (載せた項, 文, 12桁→完全な id の対応表)。項は時刻順（古い順）で、同じ記録は
    1 度しか載せない（2 つの往復が同じ根に繋がることがある）。

    **いまの反復の続き先の判定は待たない。** 判定の結果は継起の辺として書かれ、次以降の
    反復がここでさかのぼるのに使う。以前は主LLM だけが判定を待って「続きでなければ
    載せない」としており、調停（判定を待てない）と主LLM で記憶が食い違っていた。
    継起だけを頼っていたため、話題が切り替わった直後は直近が空になり、こうきと話した直後の
    入室の反復で「おかえり、こうき！」と挨拶した（2026-09-13・F）。

    各行に 12 桁の id を印字して対応表に入れる。判定と申告が直近の記録も名指せる。
    """
    return render_recent(oif, recent_chains(oif, n), n, max_age_sec=max_age_sec, now=now)


@dataclass
class Workspace:
    """W——調停と主LLM が受け取る作業状態。**組み方は 1 つ、違うのは窓の幅だけ**（記-h）。

    枠は 3 つ：直近のやりとり（時系列 n 往復＋継起の鎖）→ いまの求めの作業状態（開いている
    版は全文）→ 過去の記憶（想起）。後ろ 2 つは `compose()` が組む。直近は窓の最大幅で
    1 度だけ引き、`render(n)` が狭い側を切り出す。
    """

    oif: object
    memories: "list[Recalled]"
    req: Request
    chains: "list[tuple[str, list]]"
    n_arbiter: int
    n_main: int
    #: 直近の窓の時間の上限（秒・0 で無し・`MemoryConfig.recent_exchanges_max_sec`）。
    max_age_sec: int = 0
    #: この想起の引き方（出-ah）。面・件数・時期・直近の広さを 1 行で W に添える。
    basis: str = ""
    id_map: "dict[str, str]" = field(default_factory=dict)
    # 「いま道具から返った」の枠は**組んだ時点の値を固定**する（`render` は遅延評価で、求めの
    # `just_returned` を組んだ後に空にすると枠が消えた——実機 2026-09-18 15:28）。
    just_returned_text: str = ""
    #: この反復で返った道具（`returned_actions`）。調停は返ってきた道具を候補から外し、先導文を
    #: 道具の返りを受けるものに替える。`_iterate` は W を組んだら `just_returned` を空にするので、
    #: ここに固定する。
    returned_actions: frozenset[str] = frozenset()
    #: 申告（`memory_verdicts`）の母数と照合に使う対応表＝**過去の記憶の列だけ**（出-n 4）。
    #: 直近の枠は無条件に載せたもので、大事／不要を申告させて根づきを動かす意味がない。
    verdict_map: "dict[str, str]" = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        oif,
        memories: "list[Recalled]",
        req: Request,
        *,
        n_arbiter: int,
        n_main: int,
        max_age_sec: int = 0,
        basis: str = "",
    ) -> "Workspace":
        chains = recent_chains(
            oif, max(n_arbiter, n_main) + 1
        )  # 窓の外の次の起点も 1 つ引く（計測用）
        ws = cls(
            oif,
            memories,
            req,
            chains[: max(n_arbiter, n_main)],
            n_arbiter,
            n_main,
            max_age_sec,
            basis,
        )
        # 最上部＝確認待ち（出-y）と、この反復で道具から返ったもの（出-x）。
        ws.just_returned_text = "\n\n".join(
            p for p in (getattr(req, "confirm_frame", ""), just_returned(req)) if p
        )
        # 層 3 の材料（記-a-に）：窓のいちばん古い起点と、窓の外の次の起点。続き先の `相手` と
        # 突き合わせ、端や窓の外が参照されるなら +1、端が一度も参照されなければ −1。
        edge = chains[n_main - 1][0] if len(chains) >= n_main else "-"
        beyond = chains[n_main][0] if len(chains) > n_main else "-"
        measure.record("直近", 窓=n_main, 端=edge, 外=beyond)
        # 対応表は最も広い窓で作る（申告・判定はどちらの窓の id でも来る）。
        _rows, _text, recent_ids = ws._recent(max(n_arbiter, n_main))
        _past, past_ids = compose(
            oif, memories, req, exclude=set(recent_ids.values()), basis=ws.basis
        )
        ws.id_map = {**past_ids, **recent_ids}
        ws.verdict_map = {k: v for k, v in past_ids.items() if v not in set(recent_ids.values())}
        return ws

    def _recent(self, n: int) -> "tuple[list, str, dict[str, str]]":
        return render_recent(self.oif, self.chains, n, max_age_sec=self.max_age_sec)

    def recent_text(self, n: int) -> str:
        return self._recent(n)[1]

    def render(self, n: int) -> str:
        _rows, recent, recent_ids = self._recent(n)
        past, _ = compose(
            self.oif, self.memories, self.req, exclude=set(recent_ids.values()), basis=self.basis
        )
        return "\n\n".join(p for p in (self.just_returned_text, recent, past) if p and p.strip())

    @property
    def for_arbiter(self) -> str:
        return self.render(self.n_arbiter)

    @property
    def for_main(self) -> str:
        return self.render(self.n_main)


#: 確かさがこれを下回ったら印を付ける（`format_for_context` から引き継いだ値）。
_CONF_LOW = 0.55


def _lines(memories: "list[Recalled]", names: "dict[str, str]") -> str:
    """W の1行を組む。**核の仕事**である（環-e-い）。

    **1 行は全文で載せる**（2026-09-20）。以前は過去の記憶だけ 120 字で切り、細部は直近の
    やりとりの枠（逐語）が担うとしていた。その直近を 3／6 往復・**5 分**に狭めたので
    （出-ae(2)）、5 分より前の話は 120 字の断片しかどこにも残らなくなった。量は枠
    （`workspace_max_chars`＝40,000 字）が受ける——枠の計算はもともと全文で行っており、
    載る件数は `recall_k`（7 件）で決まるので、切り詰めは枠のためには要らない。
    切ると中身が落ちることは実機でも起きている（『9月14日(月) 30℃/22℃』が 120 字の外に出て、
    調停は検索し直し、主LLM は「読み取れなかった」と答えた・2026-09-13）。

    以前は `ObservationMemory.format_for_context` が組んでいたが、`Recalled` を受ける形に
    すると**記憶が OIF の器を知る**ことになる（依存が逆向き）。W を組むのは核なので、
    ここへ移した。`agent` 側の呼び手は辞書のままなので、記憶の面は残してある。

    **12桁で指す。** 8桁だと記録が10万件規模でほぼ確実に衝突する。照合は呼び出し側が
    対応表で行うので、写し間違いは一致せず件数のずれに出る。
    """
    if not memories:
        return ""
    # **読み込み時に引かない。** `tools.memory` は埋め込みを抱えるので、W を組むときだけ引く。
    from ..tools.memory import subject_line

    out = ["[過去の記憶（証拠つき）: conf<0.55 は不確か]:"]
    for r in memories:
        mi = r.mi
        low = " low-confidence" if r.confidence < _CONF_LOW else ""
        emo = f" [{mi.emotion}]" if mi.emotion and mi.emotion != "neutral" else ""
        sid = mi.obs_id.replace("-", "")[:12] or "?"
        day = clock.ts_to_date(mi.timestamp) if mi.timestamp else "?"
        at = clock.ts_to_time(mi.timestamp) if mi.timestamp else "?"
        subject = subject_line(mi.direction, names.get(mi.obs_id))
        body = mi.content
        out.append(
            f"- {day} {at} id:{sid} (適合度:{r.fit:.2f}) conf:{r.confidence:.2f}{low}"
            f" {subject}{emo}: {body}"
        )
    return "\n".join(out)


def describe_basis(cfg, *, viewpoint: str, time_ref: "float | None", found: int) -> str:
    """この想起の引き方を 1 行で（出-ah・2026-09-21）。

    W に「7 件・いま基準・直近 5 分・いまの相手の面」と書いておかないと、主LLM は足りないことに
    気づけず、引き直す道具を使うきっかけを持てない。**数と条件だけ**を書き、人の名前は出さない
    （面の名前を出すと、誰か分からないときに名前で呼ぶ材料になる）。
    """
    face = "いまの相手の面" if viewpoint and viewpoint != "__self__" else "共通の面"
    when = "いま基準" if time_ref is None else "時期を移して"
    minutes = int(cfg.recent_exchanges_max_sec // 60)
    return (
        f"{face}・{cfg.recall_k} 件まで（{found} 件）・{when}・"
        f"直近 {minutes} 分／{cfg.recent_exchanges_main} 往復"
    )


async def recall(
    oif,
    cue: str,
    *,
    viewpoint: str = "",
    weights,
    req: Request,
    time_ref: "float | None" = None,
    time_span_days: "float | None" = None,
) -> "Workspace":
    """想起して W を組み、**`Workspace`（W に載った記録・窓ごとの文・対応表）を返す**（に-5-ろ）。

    呼び手は2つ——反復の頭（いまが基準）と、調停が時期を指したときの引き直しである。
    同じ呼び出しが2度書かれていて、片方を直してもう片方を忘れれば、基準を移した反復
    だけ床が効かないといった食い違いが黙って入る（床＝`min_score` は実際に、連想想起
    には渡っていてイベントループにだけ渡っていなかった）。

    記録と W を一緒に返すのは、**W が記録から組まれる派生**だからである。別々に取れば
    片方だけ古くなる。

    **重みは呼び手が持つ。** `jitter_weights` は乱数を足すので、ここで作り直すと
    引き直しのたびに別の重みになる。同じ反復のあいだは同じ重みでなければならない。

    床（`min_score`）を渡す。渡さないと既定 0.0 で床が効かず、無関係な記録まで W の枠を
    埋める。床は正本 [D-想起合成] が「無関係排除の主たる足切り」と定めるものである。
    """
    from ..config import MemoryConfig

    cfg = MemoryConfig()
    # **口を通す**（環-e-い）。`viewpoint` が無いと、口が持つ基底の記憶＝`__self__` の面
    # から引いてしまう（記-f の直しを逆向きに壊す）。
    # 道具（タイマー・アラーム）の帰りは想起なし（`RETURN_WITHOUT_RECALL`）。W＝返り＋直近だけ。
    returned = returned_actions(req)
    memories: "list[Recalled]" = []
    if not returned & RETURN_WITHOUT_RECALL:
        memories = await oif.recall(
            Cue(text=cue, open_ids=tuple(open_ids(req))),
            View(
                viewpoint=viewpoint,
                k=cfg.recall_k,
                floor=cfg.recall_min_score,
                weights=weights,
                time_ref=time_ref,
                time_span_days=time_span_days,
            ),
        )
    # W は「思い出している記憶」ではなく、いまの作業状態。ループ自身の行動も MI として
    # O にあるので、合成ラベル（[取込]・[調査中]）は作らず MI をそのまま並べる。
    # W から落ちたものは薄れた＝忘れたのであって、抜けを検出する仕組みは置かない
    # （W は「速く薄れる」・改めて調べるのが自然な振る舞い）。
    # 直近のやりとりは W の枠として一緒に組む（記-h）。窓は軽量LLM／主LLM で別。
    ws = Workspace.build(
        oif,
        memories,
        req,
        n_arbiter=cfg.recent_exchanges_arbiter,
        n_main=cfg.recent_exchanges_main,
        max_age_sec=cfg.recent_exchanges_max_sec,
        basis=describe_basis(cfg, viewpoint=viewpoint, time_ref=time_ref, found=len(memories)),
    )
    ws.returned_actions = returned
    return ws


def link_follows(agent, req: Request, w_id_map: "dict[str, str]", full: "str | None") -> bool:
    """判定が返した続き先へ、継起の辺を張る（`根拠台帳` §29）。

    **W に無い id は捨てる。** 判定は12桁の形で返すが、実在するかまでは見ていない。
    突き合わせは `memory_verdicts` と同じ対応表を通す。

    自分の起点を指しても繋がない。自己ループはさかのぼりが止まらなくなる。
    """
    if not full or not req.request_id or not w_id_map:
        return False
    if full not in set(w_id_map.values()) or full == req.request_id:
        return False
    logger.info("event-loop このターンは %.8s に続く", full)
    with contextlib.suppress(Exception):
        agent._oif.link(KIND_SUCCESSION, [(full, "前", 0), (req.request_id, "後", 1)])
    return True


#: **軽量LLM へ申告だけを聞く。** 調停の JSON へ足すと、実測で `light` を選ぶ側へ判断が
#: 寄った（light 6/24 → 11/24・2場面が full から移った）。切り離せば調停のプロンプトは
#: 一字も変わらないので、分岐は動かない。
#:
#: 4つの判定に**別々の引き金**を与える（主LLM の `say` で効いた形と同じ）。引き金が無いと
#: 無難な `referred` が全件に並び、W の記憶が一斉に若返る。
_VERDICT_PROMPT = """\
これは口に出す言葉ではなく、自分の中の決めごとである。挨拶や説明はせず、指定の
JSON だけを返す。

いま人から届いた言葉に、自分はこう答えた。並んでいる記憶をどう扱ったかを申告する。

[人の言葉]
{utterance}

[自分の答え]
{reply}

[いまの作業状態]
{workspace}

`id:` が付いた行**すべて**について1件ずつ、`id` はその行のものをそのまま写す。

- `important`：答えに使い、**かつこの反復を越えて効く**（相手が尋ねた／覚えておきたいこと）
- `referred`：答えに使ったが、**この反復だけ**
- `useless`：見たが、ここでは思い出す価値が無かった
- `unused`：まったく使わなかった。**多くはこれになる**

次の形の JSON だけを返す（他には何も書かない）:
{{"memory_verdicts": [{{"id": "…", "verdict": "important|referred|useless|unused"}}]}}
"""


async def ask_verdicts(backend, *, utterance: str, reply: str, workspace_ctx: str) -> list:
    """**軽量LLM** に、いま答えるのに W の記憶をどう使ったかを聞く（出-h-ろ）。

    記憶を見て答える口は2つある——**主LLM**（`say`）と**軽量LLM**（調停の `light`）で、
    申告の口を持っていたのは主LLM だけだった。軽量LLM が答えて閉じた反復では
    `groundedness_n` が何も動かず、**記憶が育つ経路（申告1本）を通らない道**があった。

    **調停とは別に聞く。** 調停の JSON へ足すと `light` を選ぶ側へ判断が寄る（実測）。
    ここで聞けば調停のプロンプトは変わらないので、分岐は動かない。

    **読めない返事は空を返す。** 申告が無いだけで、発話には関わらない。倒す理由がない。
    """
    out = await backend.complete(
        _VERDICT_PROMPT.format(utterance=utterance, reply=reply, workspace=workspace_ctx),
        300,
    )
    match = re.search(r"\{.*\}", out or "", re.S)
    if not match:
        return []
    try:
        raw = json.loads(match.group(0)).get("memory_verdicts")
    except Exception:
        return []
    return raw if isinstance(raw, list) else []


def apply_memory_verdicts(mem, raw, w_id_map: "dict[str, str]") -> None:
    """申告された「想起した記憶の扱い」を反映する（課題5 E節 段2）。

    **照合できたものだけ適用する**。指示しても、落としたり無い id を足したりする。
    欠けた分を「使わなかった」と決めつけると、申告漏れと本当に使わなかったことを
    混同する。件数をログに残し、指示が守られているかを後から確かめられるようにする。

    **対応表に既定値を置かない**（環-h ②）。主LLM は投げっぱなしで、返るまでに別の完了が
    届けばループのいまの対応表は作り直されている。**主LLM が見た W の対応表**でないと、
    12桁が当たってしまったときに黙って別の記憶へ適用される。渡し忘れたら落ちるほうが、
    黙って別の記憶へ当たるより良い。

    **引いた面と書く面を揃える**（出-h-ろ ③）。`situated_memories` は人ごとで、想起は
    `agent._active_memory()`＝話者の面を通る。基底の記憶（`agent._memory`）へ書くと視点が
    `__self__` へ寄り、話者が同定されている場面では `UPDATE ... WHERE person_id = '__self__'`
    が0行を返して**申告が効かない**。だから `agent` ではなく、**想起に使った記憶そのもの**を
    受け取る。申告を背景で当てるとき（軽量LLM の口）も、投げた時点のものを写して渡す。
    """
    if not raw or not w_id_map:
        return
    verdicts: dict[str, str] = {}
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        full = w_id_map.get(str(item.get("id", "")).replace("-", "")[:12])
        verdict = str(item.get("verdict", "")).strip().lower()
        if full and verdict in ("important", "useless", "referred", "unused"):
            verdicts[full] = verdict
    logger.info("event-loop 記憶の判定 %d/%d 件", len(verdicts), len(w_id_map))
    # 層 3 の材料（記-a-に）：判定ごとの id 列。
    by_kind = {
        k: ",".join(i for i, v in verdicts.items() if v == k) or "-"
        for k in ("important", "useless", "referred", "unused")
    }
    measure.record("申告", **by_kind)
    if verdicts:
        with contextlib.suppress(Exception):
            mem.apply_verdicts(verdicts)
