"""開いている意図の台帳（出-cc・2026-10-13・本人の決定）。

意図（調べもの・道具・主LLM の答え）の器（`Lookup`）は、作ったときに uuid を持ち、ここに載る。完了は uuid で器を引き、
結果が届いた・打ち切った・求めを閉じたで外れる。持ち主はパジュ全体（`InformationProcessing`・装置の寿命）で、
求めをまたいで一対一に引ける。以前は語（query）で引いていたので、閉じた求めの完了が次の求めを回し（10/13 20:00:33）、
前の求めと同じ語の結果が新しい器に当たった。

引き当てはこのメモリ上の台帳で行う。始まりと終わりは記録として `store/intents`（DB・別スレッド）へ渡すだけで、
書けなくても引き当ては止まらない（本人の決定イ）。
"""

from __future__ import annotations

from typing import Any

from ..store import intents as _intents
from .request import Lookup


class IntentLedger:
    def __init__(self, store: Any = _intents) -> None:
        self._open: dict[str, Lookup] = {}
        self._store = store

    def open(self, lk: Lookup, *, request_id: str, origin: str) -> None:
        """器を台帳に載せ、投げたことを記録する。"""
        self._open[lk.intent_id] = lk
        self._store.opened(
            lk.intent_id, request_id=request_id, origin=origin, action=lk.action, query=lk.query
        )

    def reopen(self, lk: Lookup) -> None:
        """閉じた器を台帳に戻す（同じ語をもう一度頼まれ、前の器で前の結果を返すとき）。記録は書き直さない。"""
        self._open[lk.intent_id] = lk

    def get(self, intent_id: "str | None") -> "Lookup | None":
        """開いている器を uuid で引く。無い・None なら None（意図の無い完了）。"""
        return self._open.get(intent_id) if intent_id else None

    def close(
        self, intent_id: str, *, state: str, failed: bool = False, result: "str | None" = None
    ) -> None:
        """器を台帳から外し、どう終わったかを記録する。もう外れていれば何もしない。"""
        if self._open.pop(intent_id, None) is None:
            return
        self._store.ended(
            intent_id,
            state=state,
            failed=failed,
            result_chars=len(result) if result is not None else None,
        )

    def close_request(self, lookups: "list[Lookup]", *, state: str) -> None:
        """求めの器のうち、まだ開いているものをまとめて閉じる（打ち切った・求めと一緒に閉じた）。"""
        for lk in lookups:
            self.close(lk.intent_id, state=state)

    def late(self, intent_id: "str | None") -> None:
        """閉じた後に完了が届いたことを記録する（台帳には載せ直さない）。"""
        if intent_id:
            self._store.ended(intent_id, state=_intents.LATE, failed=False, result_chars=None)
