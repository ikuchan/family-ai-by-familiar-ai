"""写真からの見立てを読む（出-ae-は・2026-09-22）。純関数。

身元が機械に入る道は 2 本あった——顔（`PersonMemoryManager.presence_status`）と、声の
名乗り（調停の `speaker_claim` → `_apply_speaker_claim`）。**写真から推し量った身元だけが
どこへも行かなかった。** 実機 15:57、調停は写真を見て「パパ、おかえりなさい！」と言った
のに、機械は `(present :speaker "unconfirmed")` のままで、以後ずっと食い違った。

推し量ること自体は禁じない（本人の決定・2026-09-22）。**推し量ったなら在席にも使う。**
ここは調停の返り（`seen_people`）を、在席へ入れられる形へ直す場所である。

- 家族の名前・呼び方に当たれば、呼びかけに使う名前へ直す（`resolve_claim`・知-w と同じ口）。
- 当たらない名前と、名前の無い人は**不明の人数**に数える。居たことまでは落とさない。
- 確信度には上限を掛ける。実機の調停は推し量りに 1.0 と書いてくるので、そのまま入れると
  顔で測った値と見分けが付かなくなる。
"""

from __future__ import annotations

from .speaker_claim import resolve_claim

#: 見立ての確信度の上限。顔（実測）より低い値にする。
SEEN_CONFIDENCE_MAX = 0.6


def parse_seen_people(
    people: "list | None", family_md: str
) -> "tuple[list[tuple[str, float]], int]":
    """調停の `seen_people` を（家族の呼び方と確信度の並び, 不明の人数）へ直す。

    同じ人を 2 度挙げていれば 1 人に畳む（先に出たほうの確信度を採る）。
    """
    known: list[tuple[str, float]] = []
    seen: set[str] = set()
    unknown = 0
    for item in people or ():
        if not isinstance(item, dict):
            continue  # 調停の返りは壊れうる。読めるものだけ拾う
        raw = str(item.get("name") or "").strip()
        name = resolve_claim(raw, family_md) if raw else None
        if not name:
            unknown += 1
            continue
        if name in seen:
            continue
        seen.add(name)
        try:
            conf = float(item.get("confidence", SEEN_CONFIDENCE_MAX))
        except (TypeError, ValueError):
            conf = SEEN_CONFIDENCE_MAX
        known.append((name, max(0.0, min(SEEN_CONFIDENCE_MAX, conf))))
    return known, unknown
