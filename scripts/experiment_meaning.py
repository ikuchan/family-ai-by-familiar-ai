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
#: 2026-10-08：「既に受けた作業依頼の確認」を足し、成立しない言葉と文字を 1 つにまとめた（本人）。
MEANINGS: "dict[str, tuple[str, str]]" = {
    "confirm": (
        "確認待ちへの答え",
        "パジュがいま待っている確認の問いへの答え（いい・お願い・やめて・いらない）",
    ),
    "accepted_check": (
        "既に受けた作業依頼の確認",
        "前にパジュに頼んだこと（調べもの・覚えておいて・かけて など）がどうなったか、"
        "やってくれたか、続けてほしいかを確かめる・念を押す",
    ),
    "music": ("音楽に関する依頼", "音楽に関する依頼（かける・止める・次の曲・音量）"),
    "time": (
        "時間に関する依頼",
        "時間に関する依頼（アラーム・タイマー・ストップウォッチ・しばらく黙っていて）",
    ),
    "look": ("見る依頼", "見る依頼（そっちを向いて・何が見える・見て）"),
    # 2026-10-09：調査の依頼を「調べる」と「答えられる」に分けた（本人）。2 回目で「知っていることで答える」と
    # 道具を競わせると確信度が割れた（31「週末は」：道具だけなら 0.96、並べると 0.42〜0.51）。調べ直すかの手順
    # （変わることか・前の答えと比べて・28 の答え）は、ここの説明に書く。
    "research": (
        "調べる必要のある問い",
        "世の中のことや記憶を尋ねていて、その答えをまだ持っていないか、"
        "天気・日程のように変わることで前の答えが古い",
    ),
    "answerable": (
        "知っていることで答えられる問い",
        "尋ねていることの答えが直前のやりとりや思い出したことにあり、"
        "変わらないことか、変わることでもその答えがまだ新しい",
    ),
    "other": (
        "1〜7 以外の依頼と会話",
        "ほかの依頼と会話（あいさつ・相づち・気持ち・おしゃべり）",
    ),
    "unformed": (
        "成立しないもの（文章・言葉・文字・その他）",
        "文章・言葉・文字として成り立たない（聞き取りが崩れた・途中で切れた・意味の無い音を書き起こした）",
    ),
}
_NAME_TO_KEY = {name: key for key, (name, _) in MEANINGS.items()}

#: 2 回目の動作。鍵 → Jev への説明（2026-10-08・本人の決定：どの意味にも「聞き返す」「黙る」を置く。
#: 成立しないものは「黙る」だけ）。
ACTIONS: "dict[str, str]" = {
    "ask_back": "聞き返す（何をしてほしいのかを確かめる）",
    "silent": "黙る（何も言わず、何もしない）",
    "confirm": "確認の問いに「はい」と答えたとして進める",
    "decline": "確認の問いに「いいえ」と答えたとして取りやめる",
    "state_light": "頼まれていたことが今どうなっているかを短く伝える",
    "reply_light": "直前のやりとりをもとに、短く返す",
    "reply_full": "記憶を踏まえて、考えて返す",
    "play_music": "音楽をかける（曲名が無ければ、直前にかかっていた曲やプレイリストを続ける）",
    "stop_music": "音楽を止める",
    "next_track": "次の曲にする",
    "music_volume": "音楽の音量を変える",
    "set_timer": "タイマーを掛ける（何分後に鳴る）",
    "cancel_timer": "タイマーを止める・取り消す",
    "pause_timer": "タイマーを一時停止する",
    "resume_timer": "タイマーを再開する",
    "set_alarm": "アラームを掛ける（何時に鳴る）",
    "cancel_alarm": "アラームを取り消す",
    "start_stopwatch": "ストップウォッチで測り始める",
    "stop_stopwatch": "ストップウォッチを止めて測った長さを言う",
    "quiet": "しばらく黙っていてという頼みを受ける",
    "look": "首を向けて見る",
    "see": "目の前を見る",
    "search_deferred": "インターネットで調べる",
    "recall": "自分の記憶を探す",
    "family_schedule": "家族の予定を見る",
    "house_rules": "家の決まりを見る",
    "notion_search": "家の目次・日次記録を探す",
    "journal": "日ごとの記録を見る",
    "vault": "話している人の記録に聞く",
}
_COMMON = ("ask_back", "silent")
#: 意味ごとに並べる動作（「聞き返す」「黙る」は `_COMMON` で足す・成立しないものは黙るだけ）。
ACTIONS_BY_MEANING: "dict[str, tuple[str, ...]]" = {
    "confirm": ("confirm", "decline", *_COMMON),
    "accepted_check": ("state_light", *_COMMON),
    "music": ("play_music", "stop_music", "next_track", "music_volume", *_COMMON),
    "time": (
        "set_timer",
        "cancel_timer",
        "pause_timer",
        "resume_timer",
        "set_alarm",
        "cancel_alarm",
        "start_stopwatch",
        "stop_stopwatch",
        "quiet",
        *_COMMON,
    ),
    "look": ("look", "see", *_COMMON),
    "research": (
        "search_deferred",
        "recall",
        "family_schedule",
        "house_rules",
        "notion_search",
        "journal",
        "vault",
        *_COMMON,
    ),
    "answerable": ("reply_full", "reply_light", *_COMMON),
    "other": ("reply_light", "reply_full", *_COMMON),
    "unformed": ("silent",),
}
#: 調査の依頼の 2 回目だけ、説明を手順つきに替える（2026-10-08・本人）。調べものは、まず変わることかを考え、
#: 変わることなら前の答え（直前のやりとり・思い出したこと）と比べて調べ直すかを決める（28 の答え）。家の記録の
#: 道具は、何が載っていて何が載っていないかを書く（37 の試合の日程で家の目次を選んだ）。
RESEARCH_TEXT: "dict[str, str]" = {
    "search_deferred": (
        "インターネットで調べる。世の中のこと（天気・ニュース・試合の日程・店や場所）で、"
        "まだ答えを持っていないか、変わることで前の答えが古いとき"
    ),
    "recall": (
        "自分の記憶を探す。家族のことや前にあったこと・前の会話を尋ねられ、"
        "直前のやりとりに答えが無いとき"
    ),
    "family_schedule": "家族の予定表を見る。家族の誰かの今日や明日の予定・何時からかを尋ねられたとき",
    "house_rules": "家の決まり（ゲームをしていい曜日・帰ったときの約束など）を見る",
    "notion_search": "家族が書き溜めた家の目次・日次の記録・やることの一覧を探す（世の中のことは載っていない）",
    "journal": "家族の日ごとの記録（よく眠れたか・調子）を見る",
    "vault": "いま話している人が書き溜めた考え・経緯・検討の記録に聞く",
}

#: 既に受けた作業依頼の確認の 2 回目の説明（2026-10-08・本人：状態を伝える／聞き返すの使い分け）。
ACCEPTED_TEXT: "dict[str, str]" = {
    "state_light": (
        "どの依頼のことかが、直前のやりとりや思い出したことから一つに分かり、"
        "その依頼がどうなっているかを短く伝えられるとき"
    ),
    "ask_back": "どの依頼のことか、何をしてほしいのかが一つに決まらないとき、何のことかを聞き返す",
    "silent": "相づちや独り言で、返事が要らないとき",
}
#: 意味ごとの説明の差し替え。無いものは `ACTIONS` の説明を使う。
MEANING_TEXT: "dict[str, dict[str, str]]" = {
    "research": RESEARCH_TEXT,
    "accepted_check": ACCEPTED_TEXT,
}
#: 聞き返すときに軽量LLM へ渡す文字数の上限（2026-10-08・本人の決定）。本体に組み込むときに使う。
ASK_BACK_MAX_CHARS = 20

#: `JEV_正解.md` の「正解の動作」の書き方 → 鍵。
_ACTION_WORDS = {
    "聞き返す": "ask_back",
    "黙る": "silent",
    "やりとりをもとに軽く返す": "reply_light",
}


@dataclass
class Case:
    n: str
    at: str  # "2026-10-08 18:16:32"
    words: str
    gold: str  # MEANINGS の鍵
    action: str = ""  # ACTIONS の鍵（W に関係なく決まる場面だけ）


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
        act = re.search(r"- 正解の動作：(.+)", block)
        word = act.group(1).strip() if act else ""
        action = _ACTION_WORDS.get(word, word if word in ACTIONS else "")
        out.append(
            Case(
                head.group(2),
                head.group(1),
                words.group(1),
                _NAME_TO_KEY[gold.group(1).strip()],
                action,
            )
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
    # 直前のやりとりの続きとして読む（本人・2026-10-08：31・32・33 はやりとりの続きとして読めばいい）
    return {
        "meaning": choice(
            "この人の言葉を、直前のやりとりの続きとして読むと、次のどれに当たるか", criteria
        )
    }


def action_question(meaning: str) -> "dict | None":
    """2 回目の問い。正解の意味を前提に、その意味の動作だけを並べる。1 つしか無ければ聞かない（None）。"""
    from familiar_agent.backends.jev import choice

    keys = ACTIONS_BY_MEANING[meaning]
    if len(keys) < 2:
        return None
    name = MEANINGS[meaning][0]
    return {
        "action": choice(
            f"この人の言葉は「{name}」だと分かっている。直前のやりとりの続きとして、パジュは次にどうするか",
            {k: MEANING_TEXT.get(meaning, {}).get(k, ACTIONS[k]) for k in keys},
        )
    }


def state_for(
    words: str,
    *,
    now: str,
    music_playing: bool,
    recent: "list[tuple[str, str]]",
    requests: "list[str] | None" = None,
) -> str:
    lines = "\n".join(f"{who}：{text}" for who, text in recent) or "（なし）"
    # 調べていた依頼が想起に上がった場合（`--with-requests`）。本体では W の過去の記憶に「わたしが調べていたこと」
    # として載ることがある（確実ではない・本人「確実でなくていい」）。
    memory = (
        "[思い出したこと]\n"
        + "\n".join(f"- わたしが調べていたこと：{r}" for r in requests)
        + "\n\n"
        if requests
        else ""
    )
    return (
        "家のロボット（パジュ）が、家族の言葉を聞いた。その言葉が何を求めているかを、"
        "直前のやりとりの続きとして読んで分ける。\n\n"
        f"[いま]\n{now}\n\n"
        f"[音楽]\n{'鳴っている' if music_playing else '鳴っていない'}\n\n"
        f"[直前のやりとり（古い順）]\n{lines}\n\n"
        f"{memory}"
        f"[人の言葉]\n{words}"
    )


#: 直前のやりとりとして渡す行数と、さかのぼる秒（窓 10 秒より長く、会話のひとまとまりを拾う）。
RECENT_LINES = 6
RECENT_SEC = 300


def context_at(at: str) -> "tuple[bool, list[tuple[str, str]]]":
    """ログから、その時刻に音楽が鳴っていたかと、直前のやりとり（人とパジュの言葉・古い順）。"""
    from datetime import datetime, timedelta

    day, clock = at.split(" ")
    since = (datetime.fromisoformat(at) - timedelta(seconds=RECENT_SEC)).strftime("%H:%M:%S")
    playing, from_person = False, False
    recent: list[tuple[str, str]] = []
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
                text = m.group(1)
                if line[11:19] >= since and not text.startswith(("🎤", "✅")):
                    recent.append(("人" if from_person else "パジュ", text))
                from_person = False
    return playing, recent[-RECENT_LINES:]


def requests_at(at: str) -> "list[str]":
    """ログから、その時刻の前 `RECENT_SEC` 秒に投げた調べものと、その状態（結果が届いた・調べ中）。"""
    from datetime import datetime, timedelta

    day, clock = at.split(" ")
    since = (datetime.fromisoformat(at) - timedelta(seconds=RECENT_SEC)).strftime("%H:%M:%S")
    out: list[list[str]] = []
    files = sorted(glob.glob(str(LOG_DIR / "logs" / f"app.{day}_*.log"))) + [
        str(LOG_DIR / "app.log")
    ]
    for f in files:
        for line in open(f, encoding="utf-8", errors="replace"):
            t = line[11:19]
            if not line.startswith(day) or not (since <= t < clock):
                continue
            m = re.search(r"調べもの (\w+) [0-9.]+ 秒：(.*?)（", line)
            if m:
                out.append([m.group(1), m.group(2), "調べ中"])
            elif "想起 trigger=完了" in line and out:
                out[-1][2] = "結果が届いた"
    return [f"{what}（{action}・{state}）" for action, what, state in out]


def _row(
    n: str, gold_name: str, got: dict, names: "dict[str, str]", gold: str
) -> "tuple[str, bool]":
    probs = sorted((got.get("probabilities") or {}).items(), key=lambda kv: -float(kv[1]))
    top = str(got.get("choice") or "—")
    second = (
        f"{names.get(probs[1][0], probs[1][0])}（{float(probs[1][1]):.2f}）"
        if len(probs) > 1
        else "—"
    )
    first_p = f"{float(probs[0][1]):.2f}" if probs else "—"
    ok = top == gold
    line = (
        f"{n}\t{gold_name}\t{names.get(top, top)}（{first_p}）\t{second}\t"
        f"{float(got.get('confidence', 0) or 0):.2f}\t{'○' if ok else '×'}"
    )
    return line, ok


#: しきい値（1 回目・2 回目とも・2026-10-08 本人の決定 0.45）。
THRESHOLD = 0.45


def decide(meaning: dict, action: "dict | None") -> "tuple[str, str]":
    """本人が決めた規則で最終の動作を決める（2026-10-08）。返りは (最終の動作の鍵, どう決まったか)。

    - 1 回目が「成立しないもの」なら、確信度に関係なく黙る。
    - 1 回目がそれ以外でしきい値未満なら、倒れる（主LLM に任せる）。
    - 2 回目の 1 番が「聞き返す」なら、確信度に関係なく軽量LLM に聞き返させる。
    - 2 回目がそれ以外でしきい値未満なら、倒れる。
    """
    m = str(meaning.get("choice") or "")
    if m == "unformed":
        return "silent", "成立しないもの→黙る"
    if float(meaning.get("confidence", 0) or 0) < THRESHOLD:
        return "fallback", "1 回目がしきい値未満→倒れる"
    if action is None:
        return ACTIONS_BY_MEANING[m][0], "選択肢が 1 つ"
    a = str(action.get("choice") or "")
    if a == "ask_back":
        return "ask_back", "聞き返す→軽量LLM"
    if float(action.get("confidence", 0) or 0) < THRESHOLD:
        return "fallback", "2 回目がしきい値未満→倒れる"
    return a, "使う"


async def pipeline(client, cases: "list[Case]") -> None:
    """本番と同じ流れ：1 回目の Jev の答えで 2 回目を聞き、規則で最終の動作を出す。"""
    names = {k: name for k, (name, _) in MEANINGS.items()}
    print("場面\t1 回目（確信度）\t2 回目（確信度）\t最終\tどう決まったか\t正解の動作\t一致")
    hits = total = 0
    for c in cases:
        playing, recent = context_at(c.at)
        if recent and recent[-1] == ("人", c.words):
            recent = recent[:-1]
        state = state_for(c.words, now=c.at, music_playing=playing, recent=recent)
        a1 = await client.ask(state, meaning_question(confirming=False, music=True, camera=True))
        meaning = (getattr(a1, "answers", None) or {}).get("meaning") or {}
        m = str(meaning.get("choice") or "")
        action: "dict | None" = None
        q2 = action_question(m) if m in ACTIONS_BY_MEANING else None
        if q2 is not None:
            a2 = await client.ask(state, q2)
            action = (getattr(a2, "answers", None) or {}).get("action") or {}
        final, why = decide(meaning, action)
        first = f"{names.get(m, m)}（{float(meaning.get('confidence', 0) or 0):.2f}）"
        second = (
            f"{action.get('choice')}（{float(action.get('confidence', 0) or 0):.2f}）"
            if action
            else "—"
        )
        shown = "倒れる（主LLM）" if final == "fallback" else final
        if c.action:
            total += 1
            ok = final == c.action
            hits += ok
            mark = "○" if ok else "×"
        else:
            mark = "（W 次第）"
        print(f"{c.n}\t{first}\t{second}\t{shown}\t{why}\t{c.action or '—'}\t{mark}")
    print(f"\n動作が決まる場面の一致 {hits}/{total}")


async def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # 鍵だけを読む（表示しない）
    from familiar_agent.backends.jev import JevClient

    cases = parse_gold(GOLD.read_text(encoding="utf-8"))
    if ACTION_MODE:
        cases = [c for c in cases if c.action]  # 動作が W に関係なく決まる場面だけ
    client = JevClient.from_env(timeout=15.0)
    if not client.available:
        raise SystemExit("Jev の鍵が無い")
    if PIPELINE_MODE:
        await pipeline(client, cases)
        return
    hits = 0
    print("場面\t正解\tJev の 1 番（確率）\t2 番（確率）\t確信度\t一致")
    meaning_names = {k: name for k, (name, _) in MEANINGS.items()}
    for c in cases:
        playing, recent = context_at(c.at)
        if recent and recent[-1] == ("人", c.words):
            recent = recent[:-1]  # いまの言葉そのものは [人の言葉] に置く
        state = state_for(
            c.words,
            now=c.at,
            music_playing=playing,
            recent=recent,
            requests=requests_at(c.at) if WITH_REQUESTS else None,
        )
        if ACTION_MODE:
            question = action_question(c.gold)  # 正解の意味を前提にする（2 回目だけを測る）
            if question is None:
                only = ACTIONS_BY_MEANING[c.gold][0]
                ok = only == c.action
                hits += ok
                print(f"{c.n}\t{c.action}\t{only}（聞かずに決まる）\t—\t—\t{'○' if ok else '×'}")
                continue
            answer = await client.ask(state, question)
            got = (getattr(answer, "answers", None) or {}).get("action") or {}
            line, ok = _row(c.n, c.action, got, {}, c.action)
        else:
            answer = await client.ask(
                state, meaning_question(confirming=False, music=True, camera=True)
            )
            got = (getattr(answer, "answers", None) or {}).get("meaning") or {}
            line, ok = _row(c.n, MEANINGS[c.gold][0], got, meaning_names, c.gold)
        hits += ok
        print(line)
    print(f"\n一致 {hits}/{len(cases)}")


WITH_REQUESTS = False
ACTION_MODE = False
PIPELINE_MODE = False

if __name__ == "__main__":
    import sys

    WITH_REQUESTS = "--with-requests" in sys.argv
    ACTION_MODE = "--actions" in sys.argv  # 2 回目（動作）を測る
    PIPELINE_MODE = "--pipeline" in sys.argv  # 本番と同じ流れで最終の動作まで出す
    asyncio.run(main())
