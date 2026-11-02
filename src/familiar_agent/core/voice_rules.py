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
        if action in ASYNC_RETURNS or action.startswith(("ask_vault_", "get_journal_")):
            return True
    return False


#: 長い返事を分けるかたまりの字数（出-dh 段 1 の直し・2026-11-02・本人の決定イ）。声の道具は 200 字を超えた文を切るので、
#: 分けて続けて話す。短いほど最初の声が早く出る（本人「100 文字以下に分けて流していく形」）。
SPEECH_CHUNK_CHARS = 100
_SENTENCE_ENDS = "。！？!?\n"
_PAUSES = "、，"  # 半角のカンマは数字の位取り（1,000）に使うので区切らない
#: 文の終わりのあとに続いても、前の文に付ける閉じかっこ（次のかたまりの頭に回さない）。
_CLOSERS = "」』）)】〉》\"'”’"


def split_for_speech(text: str, limit: int = SPEECH_CHUNK_CHARS) -> list[str]:
    """声にする文を、`limit` 字以内のかたまりに分ける（何も落とさない・つなげば元に戻る）。

    文の終わりで切り、1 文が長すぎれば読点で、それでも長ければ `limit` 字で切る。かたまりは、入るだけ前へ詰める。
    """
    if len(text) <= limit:
        return [text] if text else []

    def pieces(s: str, ends: str) -> list[str]:
        out, cur = [], ""
        for i, ch in enumerate(s):
            cur += ch
            nxt = s[i + 1] if i + 1 < len(s) else ""
            closes = ch in _CLOSERS and len(cur) >= 2 and cur[-2] in ends
            if (ch in ends or closes) and nxt not in _CLOSERS:
                out.append(cur)
                cur = ""
        return out + ([cur] if cur else [])

    units: list[str] = []
    for sentence in pieces(text, _SENTENCE_ENDS):
        if len(sentence) <= limit:
            units.append(sentence)
            continue
        for clause in pieces(sentence, _PAUSES):
            units += [clause[i : i + limit] for i in range(0, len(clause), limit)]
    chunks: list[str] = []
    for u in units:
        # 音の無いかたまり（「。」「」」・空白だけ）は作らない（合成器が音を返さない・断る）。前のかたまりに付ける
        # （上限を少し超えても落とさない）。文頭なら次のかたまりの頭に付ける。
        if chunks and (len(chunks[-1]) + len(u) <= limit or not _speakable(u)):
            chunks[-1] += u
        elif chunks and not _speakable(chunks[-1]):
            chunks[-1] += u
        else:
            chunks.append(u)
    return chunks


def _speakable(text: str) -> bool:
    """声になる字（文字か数字）を含むか。"""
    return any(ch.isalnum() for ch in text)
