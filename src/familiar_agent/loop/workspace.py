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
import logging

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
    return ids


def compose(mem, memories: list[dict], req: Request) -> "tuple[str, dict[str, str]]":
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

    `said`（言ったつなぎ）と `held`（配る保留）は手組みのまま残す。どちらも O にあるが、
    `held` は `pending_store` が鮮度と配達を管理しており、想起とは別の規則を持つ。
    """
    from ..config import MemoryConfig

    budget = MemoryConfig().workspace_max_chars

    # 適合度の高い順に、枠へ入るぶんだけ採る。落ちたものは薄れた＝忘れたのであって、
    # 抜けを検出する仕組みは置かない（W は速く薄れる・改めて調べるのが自然な振る舞い）。
    ranked = sorted(memories, key=lambda m: float(m.get("fit", 0.0)), reverse=True)
    kept: list[dict] = []
    used = 0
    for m in ranked:
        size = len(str(m.get("summary", "")))
        if kept and used + size > budget:
            continue
        kept.append(m)
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

    id_map = {
        str(m.get("memory_id", "")).replace("-", "")[:12]: str(m.get("memory_id", ""))
        for m in memories
        if m.get("memory_id")
    }
    # すでに相手へ伝えた一言。これが無いと、同じ言い回しを最初から言い直す
    # （実機で「〜ですね！」で始まる前置きが3回続いた）。
    said = ""
    if req.said_fillers:
        lines = "\n".join(f"- 「{t}」" for t in req.said_fillers)
        said = (
            "すでに相手へ伝えた一言（言った順。次に何か言うなら、"
            "同じ言い回しを繰り返さず、この続きとして自然につなぐ）：\n" + lines
        )
    held = ""
    if req.speech_to_deliver:
        held = (
            "聞く相手が居ないあいだに話したかったこと"
            "（いま伝えるなら、そのときのこととして話す）：\n" + "\n".join(req.speech_to_deliver)
        )
    text = "\n\n".join(p for p in [said, held, mem.format_for_context(memories)] if p and p.strip())
    return text, id_map


async def recall(
    mem,
    cue: str,
    *,
    weights,
    req: Request,
    time_ref: "float | None" = None,
    time_span_days: "float | None" = None,
) -> "tuple[list[dict], str, dict[str, str]]":
    """想起して W を組み、**(W に載った記録, 作業状態, 対応表) を返す**（に-5-ろ）。

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
    memories = await mem.recall_async(
        cue,
        n=cfg.recall_k,
        min_score=cfg.recall_min_score,
        weights=weights,
        open_ids=open_ids(req),
        time_ref=time_ref,
        time_span_days=time_span_days,
    )
    # W は「思い出している記憶」ではなく、いまの作業状態。ループ自身の行動も MI として
    # O にあるので、合成ラベル（[取込]・[調査中]）は作らず MI をそのまま並べる。
    # W から落ちたものは薄れた＝忘れたのであって、抜けを検出する仕組みは置かない
    # （W は「速く薄れる」・改めて調べるのが自然な振る舞い）。
    text, id_map = compose(mem, memories, req)
    return memories, text, id_map


def link_follows(agent, req: Request, w_id_map: "dict[str, str]", full: "str | None") -> None:
    """判定が返した続き先へ、継起の辺を張る（`根拠台帳` §29）。

    **W に無い id は捨てる。** 判定は12桁の形で返すが、実在するかまでは見ていない。
    突き合わせは `memory_verdicts` と同じ対応表を通す。

    自分の起点を指しても繋がない。自己ループはさかのぼりが止まらなくなる。
    """
    if not full or not req.request_id or not w_id_map:
        return
    if full not in set(w_id_map.values()) or full == req.request_id:
        return
    logger.info("event-loop このターンは %.8s に続く", full)
    with contextlib.suppress(Exception):
        agent._memory.record_succession(full, req.request_id)


def apply_memory_verdicts(agent, raw, w_id_map: "dict[str, str]") -> None:
    """フルLLM が申告した「想起した記憶の扱い」を反映する（課題5 E節 段2）。

    **照合できたものだけ適用する**。指示しても、落としたり無い id を足したりする。
    欠けた分を「使わなかった」と決めつけると、申告漏れと本当に使わなかったことを
    混同する。件数をログに残し、指示が守られているかを後から確かめられるようにする。

    **対応表に既定値を置かない**（環-h ②）。主LLM は投げっぱなしで、返るまでに別の完了が
    届けばループのいまの対応表は作り直されている。**主LLM が見た W の対応表**でないと、
    12桁が当たってしまったときに黙って別の記憶へ適用される。渡し忘れたら落ちるほうが、
    黙って別の記憶へ当たるより良い。
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
    if verdicts:
        with contextlib.suppress(Exception):
            agent._memory.apply_verdicts(verdicts)
