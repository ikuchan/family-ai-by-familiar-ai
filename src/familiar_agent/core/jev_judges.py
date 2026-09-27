"""Jev（判断専用モデル）に決めさせる判定（出-au 段 5・`設計方針_判定の段` §2.2）。

判定ごとに「送る文を組む・質問を組む・答えを読む・倒し先へ倒す」をここに置く。**Jev が使えないとき**（鍵が無い・
失敗・時間切れ）と、**確信度がしきい値（`jev_confidence_min`・0.6〔仮〕）より低いとき**は、判定ごとの倒し先へ倒す。
倒し先は「迷ったときに害が小さい側」（§2.2.1）。判定はここでは例外を投げない。
"""

from __future__ import annotations

import logging
import re

from ..backends.jev import choice

logger = logging.getLogger(__name__)


class JudgeFailed(Exception):
    """Jev が使えない・失敗した。結末を「落ちた」と数えたい判定（続き先）だけが投げる。"""


#: 宛先（§2.2.3）。窓の中の名前の無い声がパジュ宛てか。
TO_PAJU = "パジュ宛て"
TO_FAMILY = "家族どうし"

#: 考え直すか（§2.2.3・§2.4）。
RETHINK = "考え直す"
AS_IS = "そのまま出す"


def picked(answer, key: str, min_conf: float) -> "str | None":
    """Choice の答えを読む。失敗・答えが無い・確信度が `min_conf` 未満なら None（倒し先へ）。"""
    if not getattr(answer, "ok", False):
        return None
    got = (getattr(answer, "answers", None) or {}).get(key) or {}
    if float(got.get("confidence", 0.0) or 0.0) < min_conf:
        return None
    pick = got.get("choice")
    return str(pick) if pick else None


async def _ask(client, state: str, questions: dict):
    """Jev に聞く。使えない・例外は None。"""
    if client is None or not getattr(client, "available", False):
        return None
    try:
        return await client.ask(state, questions)
    except Exception as e:  # noqa: BLE001
        logger.warning("Jev の判定に失敗した：%s", type(e).__name__)
        return None


async def judge_rethink(
    client, *, question: str, draft: str, added: "list[str]", min_conf: float
) -> str:
    """主LLM が考えているあいだに言い足された言葉で、返事を考え直すべきか（§2.4）。倒し先は「そのまま出す」。"""
    lines = "\n".join(f"- 「{t}」" for t in added)
    state = (
        f"[人が最初に言ったこと]\n{question}\n\n"
        f"[家のロボット（パジュ）が書いた返事の下書き]\n{draft}\n\n"
        f"[下書きを考えているあいだに、人が言い足したこと]\n{lines}"
    )
    questions = {
        "rethink": choice(
            "言い足されたことを踏まえて、返事の下書きを書き直すべきか",
            {
                "rethink": "書き直すべき。言い足されたことが問いを変える・絞る・正すので、下書きのままでは合わない",
                "as_is": "そのまま出してよい。言い足されたことは別の話か、下書きの答えを変えない",
            },
        )
    }
    pick = picked(await _ask(client, state, questions), "rethink", min_conf)
    verdict = RETHINK if pick == "rethink" else AS_IS
    logger.info("Jev 判定 考え直すか：%s（言い足し %d 件）", verdict, len(added))
    return verdict


async def judge_addressee(client, *, text: str, recent: str, present: str, min_conf: float) -> str:
    """窓の中の名前の無い声が、パジュ宛てか家族どうしか（§2.2.3）。倒し先は「パジュ宛て」（受ける）。

    無視されたと感じさせるより、返事をするほうが害が小さい。「分からない」も受ける側に数える。
    """
    state = (
        f"[いま聞こえた言葉（家のロボット・パジュの名前は含まない）]\n{text}\n\n"
        f"[直前のやりとり（古い順）]\n{recent or '（なし）'}\n\n"
        f"[いま部屋に居る人]\n{present or '（分からない）'}"
    )
    questions = {
        "to_whom": choice(
            "この言葉は、家のロボット（パジュ）に向けたものか、家族どうしの会話か",
            {
                "paju": "パジュに向けた言葉。直前のパジュとのやりとりの続き、またはパジュへの頼み・問い",
                "family": "家族どうしの会話。パジュには向けていない",
                "unknown": "どちらとも決められない",
            },
        )
    }
    pick = picked(await _ask(client, state, questions), "to_whom", min_conf)
    verdict = TO_FAMILY if pick == "family" else TO_PAJU
    logger.info("Jev 判定 宛先：%s", verdict)
    return verdict


async def judge_speech(
    client, *, response: str, recent: str, facts: str, rules: "dict[str, str]", min_conf: float
) -> "str | None":
    """返事が決まりを破っていないか（発話前の検査・§2.2.3）。破っていれば差し戻しの文、無ければ None。

    選択肢は決まり（id → 説明）と「破っていない」。差し戻しの文は「決まり <id>：<説明>」（本人の決定 ア）。
    倒し先は「破っていない」——迷っただけで主LLM を呼び直さない。
    """
    if not response or not rules:
        return None
    state = (
        f"[機械が確かめた事実]\n{facts or '（なし）'}\n\n"
        f"[直近のやりとり]\n{recent or '（なし）'}\n\n"
        f"[家のロボット（パジュ）がいま言おうとしている返事]\n{response[:600]}"
    )
    criteria = dict(rules)
    criteria["ok"] = (
        "どの決まりも破っていない。書かれた事実だけでは破っているとも言えない場合もこれ"
    )
    questions = {
        "broken": choice(
            "この返事が破っている決まりはどれか（事実に書かれていないことは推測しない）", criteria
        )
    }
    pick = picked(await _ask(client, state, questions), "broken", min_conf)
    if not pick or pick == "ok" or pick not in rules:
        logger.info("Jev 判定 発話前の検査：破っていない")
        return None
    logger.info("Jev 判定 発話前の検査：%s", pick)
    return f"決まり {pick}：{rules[pick]}"


#: W の中で id を持つ行（`id:<12桁>`）。
_W_ROW = re.compile(r"id:([0-9A-Za-z]{1,12})\b")
#: Choice の選択肢は 255 個まで。「どれの続きでもない」の分を残す。
_MAX_ROWS = 254


def workspace_rows(workspace: str) -> "dict[str, str]":
    """W の行から id → その行の文を取り出す（同じ id は最初の行）。多ければ新しい行（後ろ）を残す。"""
    rows: "dict[str, str]" = {}
    for line in workspace.splitlines():
        m = _W_ROW.search(line)
        if m and m.group(1) not in rows:
            rows[m.group(1)] = line.strip()[:300]
    if len(rows) > _MAX_ROWS:
        rows = dict(list(rows.items())[-_MAX_ROWS:])
    return rows


async def judge_follows(client, *, workspace: str, utterance: str, min_conf: float) -> "str | None":
    """いまの言葉が W のどの行の続きか（`根拠台帳` §29・§2.2.3）。続きならその id、無ければ None。

    言葉か W が無ければ聞かない。確信度が低ければ None（続きではない）。**Jev が使えない・失敗したときは
    `JudgeFailed` を投げる**——呼び手は「落ちた」と数え、「途切れ」と混ぜない（辺は書かない＝倒し先と同じ）。
    """
    rows = workspace_rows(workspace or "")
    if not (utterance or "").strip() or not rows:
        return None
    if client is None or not getattr(client, "available", False):
        raise JudgeFailed("Jev が使えない")
    criteria = dict(rows)
    criteria["none"] = "どの行の続きでもない。話題が変わった、またはやりとりを終える言葉"
    state = (
        f"[いま話しかけられた言葉]\n{utterance}\n\n"
        f"[いま頭にある記憶と直近のやりとり（各行の id が選択肢）]\n{workspace}"
    )
    questions = {
        "follows": choice(
            "いまの言葉は、どの行のやりとりの続きか。続きとは、同じ話題・同じ用件・同じ相手のやりとりが"
            "そのまま先へ進んだもの。「さっきの話」「それ」のような指す言葉があれば、指す先の行を選ぶ",
            criteria,
        )
    }
    try:
        answer = await client.ask(state, questions)
    except Exception as e:  # noqa: BLE001
        raise JudgeFailed(type(e).__name__) from e
    if not getattr(answer, "ok", False):
        raise JudgeFailed(getattr(answer, "error", "失敗"))
    pick = picked(answer, "follows", min_conf)
    return pick if pick in rows else None


#: 記憶の申告の 4 通り（主LLM の `say` の `memory_verdicts` と同じ・根づきの更新がこれを使う）。
VERDICT_CRITERIA = {
    "important": "答えに使い、かつこのやりとりを越えて効く（相手が尋ねた・覚えておきたいこと）",
    "referred": "答えに使ったが、このやりとりだけ",
    "useless": "見たが、ここでは思い出す価値が無かった",
    "unused": "まったく使わなかった。多くはこれになる",
}


async def judge_verdicts(
    client, *, utterance: str, reply: str, workspace_ctx: str, ids: "list[str]", min_conf: float
) -> list:
    """軽量LLM が答えて閉じた反復で、W の記憶をどう使ったかを申告する（出-h-ろ・§2.2.3）。

    記憶ごとに 1 問、1 回の呼び出しでまとめて聞く。返りは主LLM の申告と同じ `[{"id", "verdict"}]`。確信度が
    低い記憶は申告しない（間違った「大事」は根づきを誤って動かす）。Jev が使えない・失敗なら空。
    """
    rows = workspace_rows(workspace_ctx or "")
    wanted = [i for i in ids if i in rows]
    if not wanted or not reply:
        return []
    state = f"[人の言葉]\n{utterance or '（なし）'}\n\n[家のロボット（パジュ）の答え]\n{reply}"
    questions = {
        i: choice(f"この記憶を、答えにどう使ったか：{rows[i]}", VERDICT_CRITERIA) for i in wanted
    }
    answer = await _ask(client, state, questions)
    out = []
    for i in wanted:
        pick = picked(answer, i, min_conf)
        if pick in VERDICT_CRITERIA:
            out.append({"id": i, "verdict": pick})
    logger.info("Jev 判定 記憶の申告：%d／%d 件", len(out), len(wanted))
    return out
