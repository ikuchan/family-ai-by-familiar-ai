"""別の高度なモデルに、時間をかけて調べて考えさせる（出-dh 段 1・2026-11-02・本人の決定）。

「しっかり考えて」と頼まれたら、調べもの（`search_deferred`）とは別に、Anthropic の側の Web 検索を持たせた別モデル
（既定は Claude Fable 5.1・考える深さ low・`config.py`）に 1 回で調べさせる。答えは主LLM に書き直させず、そのまま声にする
ので、話す前の検査が守っていたこと（思い込みを言い切らない・できないことを約束しない・家族を決めつけない）と、読み上げに
向く書き方（記号を使わない・長さ）を、ここの指示で先に縛る。名前・自分のこと・話し方は `ME.md` から読む。

ここは指示を組むのと 1 回呼ぶだけで、時間の上限と完了はループ（`event_loop._run_deep`）が持つ。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: 別モデルが持つ Web 検索の道具（結果を絞ってから文脈に入れる版・Fable 5.1 と Opus 5.5 で使える・Models API で確認）。
WEB_SEARCH = "web_search_20260209"
#: 続き（`pause_turn`）を受ける回数の上限。検索の回数は `max_uses` が抑える。
_MAX_TURNS = 5
#: 概算の費用（100 万トークンあたりの入力・出力のドル、検索 1 回のドル）。ログに出すだけ（2026-10-06 の表）。
_PRICE = {"claude-fable-5-1": (10.0, 50.0), "claude-opus-5-5": (4.0, 20.0)}
_SEARCH_PRICE = 0.01


@dataclass(frozen=True)
class Answer:
    text: str
    failed: bool
    why: str  # 失敗のわけ（ログと完了の文に使う）
    seconds: float
    searches: int = 0


def system_prompt(me_md: str, *, self_text: str, target: int, limit: int, adult: bool) -> str:
    """別モデルへの指示。名前・話し方は `ME.md` から、自分のこと（`self_text`）は呼び手が入れる（コードに名前を書かない）。"""
    from ..core import parsing
    from ..core.tone import tone_line

    name = (parsing.parse_me_names(me_md) or ["ロボット"])[0]
    tone = tone_line(me_md, adult=adult)
    return "\n".join(
        [
            f"あなたは、この家のロボット「{name}」です。家族に頼まれて、じっくり調べて考えた答えを、声で伝えます。",
            "",
            "[自分のこと]",
            self_text,
            "",
            "[話し方]",
            *([f"- {tone}"] if tone else []),
            "- 声で読み上げるので、話し言葉で書く。記号（太字・箇条書きの印・見出し・表・URL・絵文字）は使わない。"
            "順番があるときは「まず」「次に」「最後に」のように言葉でつなぐ。",
            f"- 長さは {target} 字を目安にし、必ず {limit} 字以内にする。要点は大事な順に 3 つまでにし、"
            "それより細かいことは言わない。",
            "",
            "[調べた先の確かめ]",
            "- 調べた先が信頼できるかを、使う前に確かめる。公的機関、研究や調査、メーカーや団体の公式、報道を先に使う。"
            "個人の感想、広告、出どころの分からないまとめだけを根拠に言い切らない。",
            "- 情報の日付を見る。古いもの、いまと違っていそうなものは使わないか、古いと言い添える。",
            "- 信頼できる出どころどうしで食い違うときは、食い違っていると言う。",
            "",
            "[確かさ]",
            "- 言い切るのは、いま調べて確かめたことだけ。出どころは「〜の調査では」「〜によると」のように言葉で添える。",
            "- 調べても確かめられなかったことは、分からなかったと言う。思い込みや古い知識で埋めない。",
            "- 数字・日付・名前は、調べた中にあったものだけを使う。",
            "- 家族のこと（好み・予定・体調など）を決めつけない。渡された会話に無いことは言わない。",
            "",
            "[終わり方]",
            "- 答えは、調べ終えてから最後にまとめて書く。調べているあいだに文を書かない（途中の文は声にしない）。",
            "- 答えで終える。「〜しましょうか？」のような申し出や、問いかけで終わらない。",
            "- できないこと（予約・購入・連絡など）を、やるとは言わない。",
        ]
    )


def user_message(*, today: str, request: str, recent: str, results: str) -> str:
    """別モデルへ渡す材料（今日の日付・頼み・直前の会話・この頼みですでに調べた結果）。"""
    return (
        f"今日の日付：{today}\n"
        f"頼まれたこと：{request}\n"
        f"直前の会話：{recent or '（なし）'}\n"
        f"この頼みですでに調べた結果：{results or 'なし'}"
    )


async def research(
    client: Any,
    *,
    model: str,
    effort: str,
    max_searches: int,
    max_tokens: int,
    system: str,
    user: str,
) -> Answer:
    """別モデルを 1 回呼ぶ（`pause_turn` なら続ける）。断られた・落ちた・切れた・何も書かなかったは失敗。

    答えは、最後の検索の結果より後に書いた文だけ（検索の前の前置き「調べますね」などは声にしない）。検索の回数は、続きを
    受けても合わせて `max_searches` まで（`max_uses` は 1 回の要求ごとの上限なので、残りを渡す）。出力の上限で切れた
    （`max_tokens`）・続きを受けきれなかった答えは、尻切れのまま話さないよう失敗にする。
    ログに、モデル・秒数・使用量・検索の回数・概算の費用を 1 行残す（本文は出さない）。
    """
    started = time.monotonic()
    messages: list = [{"role": "user", "content": user}]
    text, tin, tout, searches, stop = "", 0, 0, 0, ""
    try:
        for _ in range(_MAX_TURNS):
            left = max_searches - searches
            if left <= 0 and stop == "pause_turn":
                break  # 検索を使い切ったのに続きを求められた。道具を外して頼むと断られうるので、切れた扱いにする
            tools = [{"type": WEB_SEARCH, "name": "web_search", "max_uses": max(1, left)}]
            r = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                tools=tools,
                output_config={"effort": effort},
            )
            u = r.usage
            tin += int(getattr(u, "input_tokens", 0) or 0)
            tout += int(getattr(u, "output_tokens", 0) or 0)
            stu = getattr(u, "server_tool_use", None)
            searches += int(getattr(stu, "web_search_requests", 0) or 0) if stu else 0
            for b in r.content:
                if b.type == "text":
                    text += getattr(b, "text", "")
                elif b.type in ("server_tool_use", "web_search_tool_result"):
                    text = ""  # 検索の前に書いた文は前置き。答えは最後の結果より後に書いた文
            stop = str(r.stop_reason or "")
            if stop != "pause_turn":
                break
            messages = messages + [{"role": "assistant", "content": r.content}]
    except Exception:  # noqa: BLE001
        logger.warning("深く調べる：%s を呼べなかった", model, exc_info=True)
        return Answer("", True, "呼べなかった", time.monotonic() - started, searches)
    seconds = time.monotonic() - started
    pin, pout = _PRICE.get(model, (0.0, 0.0))
    logger.info(
        "深く調べる：%s %s %.0f 秒・終わり=%s・入力 %d・出力 %d・検索 %d 回・約 $%.2f・%d 字",
        model,
        effort,
        seconds,
        stop,
        tin,
        tout,
        searches,
        (tin * pin + tout * pout) / 1e6 + searches * _SEARCH_PRICE,
        len(text.strip()),
    )
    if stop == "refusal":
        return Answer("", True, "断られた", seconds, searches)
    if stop in ("max_tokens", "pause_turn"):
        return Answer("", True, "答えが途中で切れた", seconds, searches)
    if not text.strip():
        return Answer("", True, "答えが無かった", seconds, searches)
    return Answer(text.strip(), False, "", seconds, searches)
