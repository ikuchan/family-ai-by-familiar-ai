"""Jev（判断専用モデル）に決めさせる判定（出-au 段 5・`設計方針_判定の段` §2.2）。

判定ごとに「送る文を組む・質問を組む・答えを読む・倒し先へ倒す」をここに置く。**Jev が使えないとき**（鍵が無い・
失敗・時間切れ）と、**確信度がしきい値（`jev_confidence_min`・0.6〔仮〕）より低いとき**は、判定ごとの倒し先へ倒す。
倒し先は「迷ったときに害が小さい側」（§2.2.1）。判定はここでは例外を投げない。
"""

from __future__ import annotations

import logging
import re

from ..backends.jev import choice, score

logger = logging.getLogger(__name__)


#: 宛先（§2.2.3）。窓の中の名前の無い声がパジュ宛てか。
TO_PAJU = "パジュ宛て"
TO_FAMILY = "家族どうし"

#: 考え直すか（§2.2.3・§2.4）。
RETHINK = "考え直す"
AS_IS = "そのまま出す"

#: 確認待ちへの答え（知-o・2026-10-13）。
CONFIRM_YES = "はい"
CONFIRM_NO = "いいえ"
CONFIRM_OTHER = "それ以外"


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


async def judge_confirm_answer(client, *, question: str, text: str, min_conf: float) -> str:
    """確認待ちのあいだに届いた言葉が、確認の問いへの答えか（知-o）。倒し先は「それ以外」（掛けもやめもせず促すだけ）。

    確認待ちのあいだは発話の意味の大きな問いを使わず、この 3 択だけを聞く（10/13 20:00:17 に「はい」を確信度不足で聞き返した）。
    """
    state = f"[家のロボット（パジュ）が聞いた確認の問い]\n{question}\n\n[いま聞こえた言葉]\n{text}"
    questions = {
        "answer": choice(
            "いま聞こえた言葉は、確認の問いへの答えか",
            {
                "yes": "はい。そのとおりにしてよいと答えている（はい・うん・いいよ・お願い）",
                "no": "いいえ。やめてと答えている（いいえ・いや・やめて・いらない）",
                "other": "それ以外。答えになっていない、または別の話・別の頼み",
            },
        )
    }
    pick = picked(await _ask(client, state, questions), "answer", min_conf)
    verdict = {"yes": CONFIRM_YES, "no": CONFIRM_NO}.get(pick or "", CONFIRM_OTHER)
    logger.info("Jev 判定 確認の答え：%s", verdict)
    return verdict


async def judge_bare_reply(client, *, heard: str, text: str, min_conf: float) -> bool:
    """say の無い主LLM の文が、いまの相手への返事か（出-ck・2026-10-15）。倒し先は独り言（偽・差し戻さない）。

    返事なら、呼び手が主LLM へ 1 回だけ「say で言って」と差し戻す。文そのものは声にしない（独り言の文を外に出さない・本人）。
    """
    state = f"[いま届いた言葉]\n{heard or '（なし）'}\n\n[家のロボット（パジュ）が書いた文]\n{text}"
    questions = {
        "reply": choice(
            "パジュが書いた文は、いま届いた言葉の相手に声で届けるための返事か、パジュ自身の考えや独り言か",
            {
                "reply": "相手への返事。相手に向けて答え・気持ち・問いを言っている",
                "monologue": "自分の考えや独り言。相手に向けて言う形になっていない",
            },
        )
    }
    pick = picked(await _ask(client, state, questions), "reply", min_conf)
    logger.info("Jev 判定 say の無い文：%s", "返事" if pick == "reply" else "独り言")
    return pick == "reply"


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


#: 意味を成さないとみなす発話の意味（成立しないもの・文脈に合わない言葉）。
_NO_MEANING = frozenset({"unformed", "off_context"})


async def judge_wake_meaning(client, *, text: str) -> bool:
    """1 字違いでだけ名前に当たった声の言葉が、意味を成すか（出-ch・2026-10-16・本人の決定サ）。倒し先は「成さない」（捨てる）。

    ふつう名前のあとには用件が続く（本人）。発話の意味の一覧（`utterance_meaning.MEANINGS`）で聞き、「成立しないもの」
    「文脈に合わない言葉」以外なら意味を成すとする。確信度では絞らない——実機の 22 件で、本物の「バジルのこのこと…」が
    0.32、崩れた音が 0.61・0.86 だった（jev-1.13.0・`計測_設定値根拠台帳` §59）。
    """
    from . import utterance_meaning as um

    state = (
        f"[パジュが聞いた言葉（音声の書き起こし・名前は崩れて書き起こされることがある）]\n{text}"
    )
    questions = {
        "meaning": choice(
            "この人の言葉は、次のどれに当たるか（直前のやりとりは無い）",
            {k: um.MEANINGS[k][1] for k in um.offered(music=True, camera=True)},
        )
    }
    answer = await _ask(client, state, questions)
    if not getattr(answer, "ok", False):
        logger.info("Jev 判定 1 字違いの名前：答えなし（捨てる）")
        return False
    got = (getattr(answer, "answers", None) or {}).get("meaning") or {}
    pick = str(got.get("choice") or "")
    logger.info(
        "Jev 判定 1 字違いの名前：%s（確信 %.2f）",
        pick or "-",
        float(got.get("confidence", 0.0) or 0.0),
    )
    return bool(pick) and pick not in _NO_MEANING


async def judge_speech(
    client, *, response: str, recent: str, facts: str, rules: "dict[str, str]", min_conf: float
) -> "str | None":
    """返事が決まりを破っていないか（発話前の検査・§2.2.3）。破っていれば差し戻しの文、無ければ None。

    記憶の申告を一緒に聞かない形（`judge_speech_and_verdicts` の `rows` が空）。
    """
    violation, _ = await judge_speech_and_verdicts(
        client,
        response=response,
        recent=recent,
        facts=facts,
        rules=rules,
        rows={},
        min_conf=min_conf,
    )
    return violation


async def judge_speech_and_verdicts(
    client,
    *,
    response: str,
    recent: str,
    facts: str,
    rules: "dict[str, str]",
    rows: "dict[str, str]",
    min_conf: float,
) -> "tuple[str | None, list]":
    """発話前の検査と、記憶の申告を **1 回の呼び出しで** 聞く（記-r・2026-10-14・本人の決定）。

    検査の選択肢は決まり（id → 説明）と「破っていない」。差し戻しの文は「決まり <id>：<説明>」（本人の決定 ア）。
    倒し先は「破っていない」——迷っただけで主LLM を呼び直さない。

    申告は主LLM に書かせず、ここで Jev に聞く。検査は声にする前に同期で走るので、同じ呼び出しに問いを足せば
    待ちは増えない。`rows` は W の過去の記憶（12 桁 → 行の文）。問いと読み方は `judge_verdicts` と同じで、確信度の
    低い記憶は申告しない。返事は切らない——申告は返事のどこで記憶を使ったかを見るので、600 字で切ると後ろで使った
    記憶を見落とす。返りは `(差し戻しの文 | None, [{"id", "verdict"}])`。
    """
    if not response:
        return None, []
    state = (
        f"[機械が確かめた事実]\n{facts or '（なし）'}\n\n"
        f"[直近のやりとり]\n{recent or '（なし）'}\n\n"
        f"[家のロボット（パジュ）がいま言おうとしている返事]\n{response}"
    )
    questions: dict = {}
    if rules:
        criteria = dict(rules)
        criteria["ok"] = (
            "どの決まりも破っていない。書かれた事実だけでは破っているとも言えない場合もこれ"
        )
        questions["broken"] = choice(
            "この返事が破っている決まりはどれか（事実に書かれていないことは推測しない）", criteria
        )
    questions.update(_verdict_questions(rows))
    if not questions:
        return None, []
    answer = await _ask(client, state, questions)
    verdicts = _verdicts_from(answer, list(rows), min_conf)
    if rows:
        logger.info("Jev 判定 記憶の申告（検査と一緒）：%d／%d 件", len(verdicts), len(rows))
    pick = picked(answer, "broken", min_conf)
    if not pick or pick == "ok" or pick not in rules:
        logger.info("Jev 判定 発話前の検査：破っていない")
        return None, verdicts
    logger.info("Jev 判定 発話前の検査：%s", pick)
    return f"決まり {pick}：{rules[pick]}", verdicts


#: W の中で id を持つ行（`id:<12桁>`）。
_W_ROW = re.compile(r"id:([0-9A-Za-z]{1,12})\b")
#: Choice の選択肢は 255 個まで。
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


#: 最終版の続き先で「どれの続きでもない」を表す選択肢（記-t 段 2）。
NOT_FOLLOWING = "none"


async def judge_succession(
    client, *, current: str, candidates: "dict[str, str]", min_conf: float
) -> "str | None":
    """いま閉じた塊（最終版）が、候補の塊（直前に閉じた最終版）のどれの続きか（記-t 段 2）。

    選んだ候補の id を返す。候補が無い・「どれの続きでもない」・確信度が低い・Jev が使えない／失敗したときは
    None（継起を書かない＝倒し先）。候補の文の字数は呼び手が合わせる。
    """
    if not candidates or not (current or "").strip():
        return None
    criteria = dict(candidates)
    criteria[NOT_FOLLOWING] = "どれの続きでもない。別の話題・別の用件のやりとり"
    state = f"[いま終わったやりとり]\n{current}"
    questions = {
        "follows": choice(
            "いま終わったやりとりは、選択肢のどのやりとりの続きか。続きとは、同じ話題・同じ用件のやりとりが"
            "そのまま先へ進んだもの（聞き直し・言い直し・それを受けた次の問いを含む）",
            criteria,
        )
    }
    pick = picked(await _ask(client, state, questions), "follows", min_conf)
    return pick if pick in candidates else None


#: 記憶の申告の 4 通り（Jev が答える・根づきの更新がこれを使う）。主LLM には申告させない（記-r）。
VERDICT_CRITERIA = {
    "important": "答えに使い、かつこのやりとりを越えて効く（相手が尋ねた・覚えておきたいこと）",
    "referred": "答えに使ったが、このやりとりだけ",
    "useless": "見たが、ここでは思い出す価値が無かった",
    "unused": "まったく使わなかった。多くはこれになる",
}


def _verdict_questions(rows: "dict[str, str]") -> dict:
    """記憶ごとに 1 問（id は W の 12 桁）。"""
    return {
        i: choice(f"この記憶を、答えにどう使ったか：{row}", VERDICT_CRITERIA)
        for i, row in rows.items()
    }


def _verdicts_from(answer, ids: "list[str]", min_conf: float) -> list:
    """答えから申告を読む。確信度の低い記憶は申告しない（間違った「大事」は根づきを誤って動かす）。"""
    out = []
    for i in ids:
        pick = picked(answer, i, min_conf)
        if pick in VERDICT_CRITERIA:
            out.append({"id": i, "verdict": pick})
    return out


async def judge_verdicts(
    client,
    *,
    utterance: str,
    reply: str,
    workspace_ctx: str,
    ids: "list[str]",
    min_conf: float,
    rows: "dict[str, str] | None" = None,
) -> list:
    """記憶の申告を背景で聞く（軽量LLM が答えた反復・主LLM の言い直しと検査を切った反復・出-h-ろ・記-r）。

    記憶ごとに 1 問、1 回の呼び出しでまとめて聞く。返りは `[{"id", "verdict"}]`。確信度が低い記憶は申告しない。
    行の文は `rows`（12 桁 → 文）で渡すか、無ければ W の文（`workspace_ctx`）から取る。Jev が使えない・失敗なら空。
    """
    rows = rows if rows is not None else workspace_rows(workspace_ctx or "")
    wanted = {i: rows[i] for i in ids if i in rows}
    if not wanted or not reply:
        return []
    state = f"[人の言葉]\n{utterance or '（なし）'}\n\n[家のロボット（パジュ）の答え]\n{reply}"
    answer = await _ask(client, state, _verdict_questions(wanted))
    out = _verdicts_from(answer, list(wanted), min_conf)
    logger.info("Jev 判定 記憶の申告：%d／%d 件", len(out), len(wanted))
    return out


#: 感情の評価（§2.2.3・本人の決定 5 段）。P と Pn は量、Dom は掌握。
_AMOUNT_LEVELS = ["まったく無い", "少し", "ふつう", "かなり", "とても大きい"]
_DOM_LEVELS = ["どうにもできない", "あまりできない", "ふつう", "かなりできる", "すっかり手の内"]


def _unit(got: dict, levels: int) -> "float | None":
    """Score の値（段の番号を確率で重みづけた 0〜段数-1）を 0〜1 に直す。無ければ None。"""
    v = got.get("score") if isinstance(got, dict) else None
    if v is None:
        return None
    return min(1.0, max(0.0, float(v) / (levels - 1)))


async def judge_emotion(client, *, text: str, mood, arousal: float, a_gate: float):
    """いまのやり取りでパジュ自身が何を感じたかを P・Pn・Dom で測る（W2b-2・§2.2.3）。返りは `(PAD|None, A)`。

    A は機械の高ぶりで、`a_gate` 未満なら聞かない。平静の位置といまの気分は**事実として**送る（答えの側を指示しない・
    出-e）。Jev が使えない・失敗・3 つそろわなければ未測定（`None`・気分で埋めない・050）。確信度では倒さない。
    """
    from ..mood_register import MoodPAD

    a = min(1.0, max(0.0, arousal))
    if a < a_gate:
        return None, a
    state = (
        f"[やり取り]\n{text[:400]}\n\n"
        "[家のロボット（パジュ）のいまの気分]\n"
        f"快 {mood.p:.2f}・不快 {mood.pn:.2f}・掌握 {mood.dom:.2f}"
        "（何も起きていないときは 快 0.10・不快 0.10・掌握 0.50 に落ち着く。快と不快は別々の量で、両方大きいことも"
        "両方小さいこともある）"
    )
    questions = {
        "p": score("このやり取りで、パジュ自身が感じた心地よさ（快）の大きさ", _AMOUNT_LEVELS),
        "pn": score("このやり取りで、パジュ自身が感じた嫌さ（不快）の大きさ", _AMOUNT_LEVELS),
        "dom": score("このやり取りで、パジュ自身が感じた、状況を掌握している度合い", _DOM_LEVELS),
    }
    answer = await _ask(client, state, questions)
    if not getattr(answer, "ok", False):
        return None, a
    got = answer.answers
    p = _unit(got.get("p", {}), len(_AMOUNT_LEVELS))
    pn = _unit(got.get("pn", {}), len(_AMOUNT_LEVELS))
    dom = _unit(got.get("dom", {}), len(_DOM_LEVELS))
    if p is None or pn is None or dom is None:
        return None, a
    return MoodPAD(p=p, pn=pn, a=a, dom=dom), a


#: 気分の見立ての選択肢（`observations` の読み手が使う 5 つ）。
COMPANION_MOODS = {
    "engaged": "乗り気・話に関わっている",
    "tired": "疲れている",
    "frustrated": "苛立っている・困っている",
    "absent": "関わっていない・上の空・分からない",
    "happy": "うれしい・楽しい",
}
#: Jev が使えないときの既定（見立てない・本人の決定 2026-09-27）。
MOOD_DEFAULT = "absent"


async def judge_companion_mood(client, *, text: str) -> str:
    """相手の言葉から相手の気分を見立てる（§2.2.3）。確信度では倒さない。使えないときは既定の `absent`。"""
    questions = {"mood": choice("この言葉を言った人の、いまの気分に最も近いもの", COMPANION_MOODS)}
    answer = await _ask(client, f"[家族の言葉]\n{text[:300]}", questions)
    pick = picked(answer, "mood", 0.0)
    return pick if pick in COMPANION_MOODS else MOOD_DEFAULT


async def judge_music_pick(
    client, *, candidates: "list[dict]", family_now: str, artists: str, previous: str
) -> "dict[str, float] | None":
    """どのおすすめがいまの家族に合うか（知-aq・2026-10-22・本人の決定）。選択肢 `c1`… ごとの確率を返す。

    確信度で倒さない——確率は呼び手が `music_suggestion.pick` で使う（固まったら 2 番目以降から引く）。
    使えない・失敗・確率が無ければ None（倒し先：前日の候補を残す）。
    """
    state = (
        f"[家族のいまの様子]\n{family_now or '（なし）'}\n\n"
        f"[プレイリストのアーティスト]\n{artists or '（なし）'}\n\n"
        f"[前日のおすすめ]\n{previous or '（なし）'}"
    )
    criteria = {
        f"c{n}": f"「{c['title']}」（{c['artist']}）。選んだ理由：{c['reason']}"
        for n, c in enumerate(candidates, 1)
    }
    questions = {
        "pick": choice(
            "家のロボット（パジュ）が家族に 1 曲すすめる。家族のいまの様子とプレイリストの好みに、いちばん合う曲はどれか",
            criteria,
        )
    }
    answer = await _ask(client, state, questions)
    if not getattr(answer, "ok", False):
        logger.info("Jev 判定 おすすめ：答えなかった")
        return None
    got = (getattr(answer, "answers", None) or {}).get("pick") or {}
    probs = {
        str(k): float(v) for k, v in dict(got.get("probabilities") or {}).items() if k in criteria
    }
    if not probs:
        logger.info("Jev 判定 おすすめ：確率が無かった")
        return None
    return probs


#: 季節の層で「どれも合わない」を表す選択肢（知-ac・2026-10-24）。
SEASON_NONE = "none"


def _season_picks(got: dict, keys: "dict[str, object]", most: int) -> list:
    """確率の高い順に `most` 本まで。1 位が「どれも合わない」なら読まない。2 位以降は「どれも合わない」より上だけ。"""
    probs = {str(k): float(v) for k, v in dict(got.get("probabilities") or {}).items()}
    if not probs:
        return []
    order = sorted(probs, key=lambda k: probs[k], reverse=True)
    if order[0] not in keys:
        return []
    none = probs.get(SEASON_NONE, 0.0)
    picks = [order[0]] + [k for k in order[1:] if k in keys and probs[k] > none]
    return [keys[k] for k in picks[:most]]


async def judge_season_reads(
    client, *, today, place: str, news: list, columns: list
) -> "tuple[list, list]":
    """季節の層で読む記事（知-ac・2026-10-24・本人の決定）。ウェザーニュースは 2 本まで、Yahoo!のコラムは 1 本まで。

    題名に「どれも合わない」を足して選ばせ、確率どうしを比べる（しきい値は置かない）。
    使えない・失敗・確率が無ければ何も選ばない（倒し先：節気のページだけ読む）。
    """
    from .season_env import last_term

    news_keys = {f"n{i}": n for i, n in enumerate(news, 1)}
    column_keys = {f"c{i}": c for i, c in enumerate(columns, 1)}
    none = "どれも合わない"
    questions = {}
    if news_keys:
        questions["news"] = choice(
            "家のロボット（パジュ）が、季節の話のきっかけに天気のニュースを読む。住んでいる土地といまの時期に"
            "いちばん合う記事はどれか",
            {**{k: n.title for k, n in news_keys.items()}, SEASON_NONE: none},
        )
    if column_keys:
        questions["column"] = choice(
            "家のロボット（パジュ）が、季節の話のきっかけに歳時記や天気の読み物を読む。いまの時期にいちばん"
            "合う読み物はどれか（季節外れの行事は合わない）",
            {**{k: c.title for k, c in column_keys.items()}, SEASON_NONE: none},
        )
    if not questions:
        return [], []
    state = f"[今日]\n{today.isoformat()}（節気は{last_term(today)}）\n\n[住んでいる土地]\n{place or '（分からない）'}"
    answer = await _ask(client, state, questions)
    if not getattr(answer, "ok", False):
        logger.info("Jev 判定 季節の読み物：答えなかった")
        return [], []
    got = getattr(answer, "answers", None) or {}
    return (
        _season_picks(got.get("news") or {}, news_keys, 2),
        _season_picks(got.get("column") or {}, column_keys, 1),
    )
