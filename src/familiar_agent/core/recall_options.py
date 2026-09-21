"""思い出し方の指定（出-ah・2026-09-21）。純関数。

W の「過去の記憶」に何が載るかを決める軸は 10 個あるが、主LLM が動かせるのは**手がかりの言葉**
（`recall(query)`）だけだった。時期を移せるのは調停（`time_ref`）、残りは設定値である。思い出せない
とき、7 件・いまの相手の面・いま基準・直近 5 分、という条件そのものを変える手が無い。

そこで **視点・件数と思い出し方・時期と幅・直近の窓** を道具として渡す。この file は、その指定を
**丸めて軸へ割り当てるだけ**を持つ（引くのは `loop/workspace`）。効き目はその 1 回だけで、以後の
反復の W は元の条件で組む（本人の決定・2026-09-21）。
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..config import RecallWeights

#: 件数の上限。候補として採点しているのが 50 件なので、それ以上は増やせない。1 件は全文で
#: 載るので、上げるほど枠（`workspace_max_chars`＝40,000 字）に当たりやすくなる（本人の決定）。
MAX_K = 20
#: 直近の窓を広げられる上限（分・往復）。既定は 5 分・3／6 往復（`recent_exchanges_max_sec` ほか）。
RECENT_MAX_MIN = 30
RECENT_MAX_TURNS = 20
#: 時期の幅の既定（日）。指定が無ければこの幅で前後を見る。
DEFAULT_SPAN_DAYS = 30.0

#: 指した軸をこの倍率で強め、他をこの割合に薄める（本人の決定・2026-09-21）。
_STRONG = 2.0
_THIN = 2.0 / 3.0

#: 思い出し方の言い方 → 5 軸のどれか。**軸の名前は主LLM に見せない**（人の言葉で指す）。
WAYS: dict[str, str] = {
    "新しい順に": "w_t",
    "印象に残っていることを": "w_e",
    "よく思い出すことを": "w_g",
    "この人との関わりで": "w_p",
    "話に近いものを": "w_r",
}


def weights_for(way: str, base: RecallWeights) -> RecallWeights:
    """思い出し方の言い方で重みを組み替える。知らない言い方なら base のまま。"""
    axis = WAYS.get((way or "").strip())
    if axis is None:
        return base
    values = {
        name: (getattr(base, name) * (_STRONG if name == axis else _THIN))
        for name in ("w_r", "w_t", "w_e", "w_g", "w_p")
    }
    return RecallWeights(**values)


def clamp_k(k: "int | None") -> int:
    """載せる件数を 1〜`MAX_K` に丸める。指定が無ければ上限まで（深く思い出す道具なので）。"""
    if k is None:
        return MAX_K
    return max(1, min(MAX_K, int(k)))


def clamp_recent(minutes: "float | None", turns: "int | None") -> "tuple[int, int]":
    """直近の窓を上限（30 分・20 往復）に丸める。指定が無ければ上限まで。"""
    m = RECENT_MAX_MIN if minutes is None else max(1, min(RECENT_MAX_MIN, int(minutes)))
    t = RECENT_MAX_TURNS if turns is None else max(1, min(RECENT_MAX_TURNS, int(turns)))
    return m, t


def parse_when(date: str, span_days: "float | None") -> "tuple[float, float] | None":
    """ISO の日付（`2026-08-15`）と幅（日）を、想起へ渡す形にする。読めなければ None。

    調停が時期を指すとき（`time_ref`）と同じ仕組みに乗せる。言葉（「去年の夏」）は受けない——
    日付に直すのは言葉を扱う側（主LLM）の仕事である。
    """
    text = (date or "").strip()
    if not text:
        return None
    try:
        when = datetime.fromisoformat(text)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    span = DEFAULT_SPAN_DAYS if span_days is None else max(1.0, float(span_days))
    return when.timestamp(), span
