"""出-ay の実験：発話の 1 回目（意味の分類）を Jev に聞いたとき、正解の意味を選べるか（2026-10-08・本人の決定）。

調停を「1 回目に意味、2 回目に動作」へ組み替える前に、1 回目だけを測る。正解は `JEV_正解.md`（リポジトリ直下・版管理の外）
の発話の場面の「正解の意味」。Jev に送るのは、その場の言葉と、ログから分かるその時刻の状況（音楽が鳴っていたか・直前に
自分が言ったこと）だけで、W は入れない（意味は W に関係なく決まる）。

**画面に出すのは場面の番号・意味の名前・確率と一致数だけ。** 言葉の中身は出さない。

使い方：`uv run python scripts/experiment_meaning.py`（`.env` の `JEV_API_KEY` を読む・鍵は出さない）。
"""

from __future__ import annotations

import asyncio
import glob
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "JEV_正解.md"
LOG_DIR = Path.home() / ".cache" / "familiar-ai"

#: 意味の選択肢（明確なものから、あやふやなものの順・本人の決定）。鍵 → (名前, Jev への説明)。
MEANINGS: "dict[str, tuple[str, str]]" = {
    "confirm": (
        "確認待ちへの答え",
        "パジュがいま待っている確認の問いへの答え（いい・お願い・やめて・いらない）",
    ),
    "music": ("音楽に関する依頼", "音楽に関する依頼（かける・止める・次の曲・音量）"),
    "time": (
        "時間に関する依頼",
        "時間に関する依頼（アラーム・タイマー・ストップウォッチ・しばらく黙っていて）",
    ),
    "look": ("見る依頼", "見る依頼（そっちを向いて・何が見える・見て）"),
    "research": (
        "記憶も含めた調査の依頼",
        "記憶も含めた調査の依頼（天気・ニュース・予定・前にあったこと・知っていることを尋ねる）",
    ),
    "other": ("1〜5 以外の依頼と会話", "ほかの依頼と会話（あいさつ・相づち・気持ち・おしゃべり）"),
    "broken_talk": (
        "会話として成立しない言葉",
        "言葉ではあるが、会話として意味が通らない（聞き取りが崩れた・途中で切れた・何を言いたいか分からない）",
    ),
    "not_words": (
        "言葉として成立しない文字",
        "言葉になっていない（くしゃみ・咳・物音・意味の無い音を書き起こしたもの）",
    ),
}
_NAME_TO_KEY = {name: key for key, (name, _) in MEANINGS.items()}


@dataclass
class Case:
    n: str
    at: str  # "2026-10-08 18:16:32"
    words: str
    gold: str  # MEANINGS の鍵


def parse_gold(text: str) -> "list[Case]":
    """`JEV_正解.md` から、発話で「正解の意味」のある場面を読む。"""
    out: list[Case] = []
    for block in re.split(r"\n(?=## )", text):
        head = re.match(r"## (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)（(\d+)）", block)
        if not head or "- 起点：発話" not in block:
            continue
        words = re.search(r"- 言葉：「(.*)」", block)
        gold = re.search(r"- 正解の意味：(.+)", block)
        if not (words and gold) or gold.group(1).strip() not in _NAME_TO_KEY:
            continue
        out.append(
            Case(head.group(2), head.group(1), words.group(1), _NAME_TO_KEY[gold.group(1).strip()])
        )
    return out


def meaning_question(*, confirming: bool, music: bool, camera: bool) -> dict:
    """1 回目の問い。その場で使えない意味（確認待ちでない・音楽が無い・カメラが無い）は並べない。"""
    from familiar_agent.backends.jev import choice

    skip = {"confirm"} if not confirming else set()
    if not music:
        skip.add("music")
    if not camera:
        skip.add("look")
    criteria = {k: desc for k, (_, desc) in MEANINGS.items() if k not in skip}
    return {"meaning": choice("この人の言葉は、次のどれに当たるか", criteria)}


def state_for(words: str, *, now: str, music_playing: bool, last_said: str) -> str:
    return (
        "家のロボット（パジュ）が、家族の言葉を聞いた。その言葉が何を求めているかを分ける。\n\n"
        f"[いま]\n{now}\n\n"
        f"[音楽]\n{'鳴っている' if music_playing else '鳴っていない'}\n\n"
        f"[直前にパジュが言ったこと]\n{last_said or '（なし）'}\n\n"
        f"[人の言葉]\n{words}"
    )


def context_at(at: str) -> "tuple[bool, str]":
    """ログから、その時刻に音楽が鳴っていたかと、直前にパジュが言ったこと。"""
    day, clock = at.split(" ")
    playing, last_said, from_person = False, "", False
    files = sorted(glob.glob(str(LOG_DIR / "logs" / f"app.{day}_*.log"))) + [
        str(LOG_DIR / "app.log")
    ]
    for f in files:
        for line in open(f, encoding="utf-8", errors="replace"):
            if not line.startswith(day) or line[11:19] >= clock:
                continue
            if "かけ始めた" in line and "tools.music" in line:
                playing = True
            if "調べもの stop_music" in line:
                playing = False
            if "GUI 入力を積んだ" in line:
                from_person = True  # 次の吹き出しは人の言葉
                continue
            m = re.search(r"吹き出し \d+件目を足した .*?「([^」]*)", line)
            if m:
                if not from_person and not m.group(1).startswith(("🎤", "✅")):
                    last_said = m.group(1)
                from_person = False
    return playing, last_said


async def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # 鍵だけを読む（表示しない）
    from familiar_agent.backends.jev import JevClient

    cases = parse_gold(GOLD.read_text(encoding="utf-8"))
    client = JevClient.from_env(timeout=15.0)
    if not client.available:
        raise SystemExit("Jev の鍵が無い")
    hits = 0
    print("場面\t正解\tJev の 1 番（確率）\t2 番（確率）\t確信度\t一致")
    for c in cases:
        playing, last_said = context_at(c.at)
        answer = await client.ask(
            state_for(c.words, now=c.at, music_playing=playing, last_said=last_said),
            meaning_question(confirming=False, music=True, camera=True),
        )
        got = (getattr(answer, "answers", None) or {}).get("meaning") or {}
        probs = sorted((got.get("probabilities") or {}).items(), key=lambda kv: -float(kv[1]))
        top = str(got.get("choice") or "—")
        second = (
            f"{MEANINGS[probs[1][0]][0]}（{float(probs[1][1]):.2f}）" if len(probs) > 1 else "—"
        )
        first_p = f"{float(probs[0][1]):.2f}" if probs else "—"
        ok = top == c.gold
        hits += ok
        print(
            f"{c.n}\t{MEANINGS[c.gold][0]}\t{MEANINGS.get(top, (top,))[0]}（{first_p}）\t{second}\t"
            f"{float(got.get('confidence', 0) or 0):.2f}\t{'○' if ok else '×'}"
        )
    print(f"\n一致 {hits}/{len(cases)}")


if __name__ == "__main__":
    asyncio.run(main())
