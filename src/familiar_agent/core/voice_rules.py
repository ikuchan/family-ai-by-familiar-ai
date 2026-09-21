"""どの声で読むか（環-u・2026-09-21）。純関数。

**ふだんは flash**（合成 0.6 秒）で、`speed` は 0.9（既定の 1.0 は速すぎた・本人が聴いて決めた）。
**v3 を使うのは 1 つの場面だけ**——主LLM が話し、かつ**外へ問い合わせた返り**から起きた反復のとき。
v3 は漢字を読めるのでひらがな化が要らない代わりに、合成に 3.4 秒かかる。その場面は相手がすでに
数秒待っているので、そこへ乗せる（本人の決定）。

調停の短い一言（`light`）と、その場で返る道具（`recall`・タイマー・確認・見る）の帰りは、待ち時間が
無いか短いので flash のままにする。
"""

from __future__ import annotations

#: 外へ問い合わせる道具。返りが来るまでに数秒かかる（本人の決定：deferred と外の道具）。
ASYNC_RETURNS = frozenset(
    {
        "search_deferred",
        "fetch_deferred",
        "house_rules",
        "get_house_rules",
        "family_schedule",
        "get_family_schedule",
        "notion_search",
        "search_notion",
        "journal",
        "get_journal",
        "vault",
    }
)


def careful_voice(branch: str, returned: "frozenset[str] | set[str] | tuple[str, ...]") -> bool:
    """じっくり読む声（v3）で読むか。

    条件は 2 つとも揃ったときだけ——**主LLM の発話**（`full`）であることと、**外へ問い合わせた
    返り**から起きた反復であること。字数は見ない（本人の決定・2026-09-21）。
    """
    if branch != "full":
        return False
    for action in returned or ():
        if action in ASYNC_RETURNS or action.startswith("ask_vault_"):
            return True
    return False
