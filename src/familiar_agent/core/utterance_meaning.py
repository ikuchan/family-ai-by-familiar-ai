"""発話の意味と動作の問い（出-ay 段 4-4a・2026-10-09・`設計方針_判定の段` §2.2.5）。

調停の組み替えで、人の言葉への最初の反復は「1 回目に意味、2 回目に意味ごとの動作」を聞くと決めた（本人）。実験
（`scripts/experiment_meaning.py`・正解 8 場面中 7・1 場面 約 0.3 秒）で決めた問いと決まりを、本体の部品としてここに置く。
ここは問いを組むのと、答えから最終の動作を決めるだけで、Jev も軽量LLM も呼ばない。

- 意味は明確なものから、あやふやなものの順。使えないもの（確認待ちでない・すすめた曲の返事を待っていない・音楽が無い・
  カメラが無い）は並べない。
- 先読みで 1 回に並べる：意味の問いと、意味ごとの「もしこの言葉が○○なら」の問いを同じ回で聞き、返った意味の答えだけ使う
  （Jev にキャッシュは無く、2 回に分けると状態の文を 2 回送る）。
- 「文脈」は直前のやりとり（人とパジュの言葉）のことで、思い出した記憶ではない。
- 決まり：成立しないものは黙る。文脈に合わない言葉は、聞き返すを 0.6 以上で選んだときだけ聞き返し、それ以外は黙る。ほかは
  しきい値 0.6。聞き返すは確信度に関係なく。越えなければ「よく考えるか、軽く聞き返すか」を聞き、それも 0.6 未満なら軽く
  聞き返す（聞き返すは軽量LLM が 20 字までで書く）。

動作の鍵：`ask_back`（聞き返す）・`silent`（黙る）・`state_light`（状態を伝える）・`reply_light`（軽く返す）・
`reply_full`（考えて返す＝主LLM）・`quiet`（黙る依頼を受ける）・`lift_quiet`（黙るのを解く）・道具の名前。名乗りと否定の
2 回目は家族の呼び方（と `other`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: しきい値（1 回目・2 回目・越えなかったときの問い・2026-10-09 本人の決定）。
THRESHOLD = 0.6
#: 聞き返すときに軽量LLM へ渡す文字数の上限（2026-10-08 本人の決定）。
ASK_BACK_MAX_CHARS = 20

#: 意味。鍵 → (名前, Jev への説明)。明確なものから、あやふやなものの順（本人）。
MEANINGS: "dict[str, tuple[str, str]]" = {
    "confirm": (
        "確認待ちへの答え",
        "パジュがいま待っている確認の問いへの答え（いい・お願い・やめて・いらない）",
    ),
    # すすめた曲への返事（段 4-4f・2026-10-10 本人）。返事のあとはかけるか、かけないかだけなので Jev が決める。
    "suggestion": (
        "すすめた曲への返事",
        "パジュがさっきすすめた曲への返事（いいね・かけて・いらない・いまはいい）",
    ),
    "claim": ("名乗り", "自分が誰かを名乗っている（パパだよ・たいきです）"),
    "deny": (
        "名乗りの否定",
        "パジュが思っている相手ではないと言っている（パパじゃないよ・ちがうよ）",
    ),
    "accepted_check": (
        "既に受けた作業依頼の確認",
        "前にパジュに頼んだこと（調べもの・覚えておいて・かけて など）がどうなったか、"
        "やってくれたか、続けてほしいかを確かめる・念を押す",
    ),
    "music": ("音楽に関する依頼", "音楽に関する依頼（かける・止める・次の曲・音量）"),
    "time": (
        "時間に関する依頼",
        "時間に関する依頼（アラーム・タイマー・ストップウォッチ・しばらく黙っていて・もう話していいよ）",
    ),
    "look": ("見る依頼", "見る依頼（そっちを向いて・何が見える・見て）"),
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
        "その他の依頼と会話",
        "ほかの依頼と会話（あいさつ・相づち・気持ち・おしゃべり）",
    ),
    "off_context": (
        "これまでの文脈に合わない言葉",
        "直前のやりとりの続きとして読むと、話の流れに合わない"
        "（パジュに向けた言葉ではなさそう・別の人との話・急に関係の無い話）",
    ),
    "unformed": (
        "成立しないもの（文章・言葉・文字・その他）",
        "文章・言葉・文字として成り立たない（聞き取りが崩れた・途中で切れた・意味の無い音を書き起こした）",
    ),
}

#: 2 回目の動作の説明。
ACTIONS: "dict[str, str]" = {
    "ask_back": "聞き返す（何をしてほしいのかを確かめる）",
    "silent": "黙る（何も言わず、何もしない）",
    "confirm": "確認の問いに「はい」と答えたとして進める",
    "decline": "確認の問いに「いいえ」と答えたとして取りやめる",
    "suggestion_like": "すすめた曲をかける（気に入った・聞いてみたい）",
    "suggestion_decline": "すすめた曲をかけない（いらない・いまはいい）。二度とすすめない",
    "state_light": (
        "どの依頼のことかが、直前のやりとりや思い出したことから一つに分かり、"
        "その依頼がどうなっているかを短く伝えられるとき"
    ),
    "reply_light": "直前のやりとりをもとに、短く返す",
    "reply_full": "記憶を踏まえて、考えて返す",
    "play_music": "音楽をかける（曲名が無ければ、止まっていた続きを鳴らす）",
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
    "lift_quiet": "もう話していいよという言葉を受けて、黙るのを解く",
    "look": "首を向けて見る",
    "see": "目の前を見る",
    "search_deferred": (
        "インターネットで調べる。世の中のこと（天気・ニュース・試合の日程・場所や営業時間）で、"
        "まだ答えを持っていないか、変わることで前の答えが古いとき"
    ),
    "recall": (
        "自分の記憶を探す。家族のことや前にあったこと・前の会話を尋ねられ、直前のやりとりに答えが無いとき"
    ),
    "family_schedule": "家族の予定表を見る。家族の誰かの今日や明日の予定・何時からかを尋ねられたとき",
    "house_rules": "家の決まり（ゲームをしていい曜日・帰ったときの約束など）を見る",
    "notion_search": "家族が書き溜めた家の目次・日次の記録・やることの一覧を探す（世の中のことは載っていない）",
    "journal": "家族の日ごとの記録（よく眠れたか・調子）を見る",
    "vault": "いま話している人が書き溜めた考え・経緯・検討の記録に聞く",
}
_COMMON = ("ask_back", "silent")
#: 意味ごとに並べる動作（本人の表）。名乗りと否定は家族の呼び方から作る（`fanout_questions`）。
ACTIONS_BY_MEANING: "dict[str, tuple[str, ...]]" = {
    "confirm": ("confirm", "decline", *_COMMON),
    "suggestion": ("suggestion_like", "suggestion_decline", *_COMMON),
    "claim": (),
    "deny": (),
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
        "lift_quiet",
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
    "off_context": ("ask_back", "silent"),
    "unformed": ("silent",),
}
#: 名乗り・否定の 2 回目で、家族の誰でもないとき。
OTHER = "other"
#: 確信度に関係なくそのまま使う意味（成立しないもの・文脈に合わない言葉・本人）。
ALWAYS_USE = frozenset({"unformed", "off_context"})
#: 考える深さ（2026-10-09 本人の決定ウ：先読みの 1 回に足す）。考えて返す（主LLM）ときに渡す。いままでの調停と同じ説明。
EFFORTS: "dict[str, str]" = {
    "low": "ふつう。ほとんどの場合",
    "medium": "ひと言で表せない複雑な気持ちを受け止める、4 つ以上の記憶を踏まえて応える、調べた結果をまとめる",
    "high": "人がよく考えるよう明示的に求めた",
}
#: 黙る依頼の長さの選択肢（分）。`default` は長さの指定なし（受け手が既定を当てる）。
QUIET_MINUTES: "dict[str, str]" = {
    "default": "長さを言っていない",
    "5": "5 分",
    "10": "10 分",
    "15": "15 分",
    "30": "30 分",
    "60": "1 時間",
}
#: 越えなかったときに聞く問いの選択肢。
UNSURE_ACTIONS: "dict[str, str]" = {
    "reply_full": "よく考えてみる（記憶を踏まえて、考えて返す）",
    "ask_back": "軽く聞き返す（何をしてほしいのかを短く確かめる）",
}


def offered(
    *, confirming: bool, music: bool, camera: bool, suggesting: bool = False
) -> "list[str]":
    """1 回目に並べる意味。使えないもの（すすめた曲の返事を待っていない、なども）は並べない。"""
    skip = set()
    if not confirming:
        skip.add("confirm")
    if not suggesting:
        skip.add("suggestion")
    if not music:
        skip.add("music")
    if not camera:
        skip.add("look")
    return [k for k in MEANINGS if k not in skip]


def _actions_for(meaning: str, family: "list[str]") -> "dict[str, str]":
    if meaning == "claim":
        return {**{n: f"{n}だと名乗った" for n in family}, OTHER: "家族の誰でもない人だと名乗った"}
    if meaning == "deny":
        return {
            **{n: f"{n}ではないと言った" for n in family},
            OTHER: "どの呼び方を否定したか分からない",
        }
    return {k: ACTIONS[k] for k in ACTIONS_BY_MEANING[meaning]}


def fanout_questions(
    *,
    confirming: bool,
    music: bool,
    camera: bool,
    family: "list[str]",
    suggesting: bool = False,
) -> "dict[str, dict]":
    """意味の問いと、意味ごとの 2 回目の問いを、先読みで 1 回に並べる（返った意味の答えだけ使う）。"""
    from ..backends.jev import choice

    meanings = offered(confirming=confirming, music=music, camera=camera, suggesting=suggesting)
    qs: dict[str, dict] = {
        "meaning": choice(
            "この人の言葉を、直前のやりとりの続きとして読むと、次のどれに当たるか"
            "（思い出したことは文脈に含めない）",
            {k: MEANINGS[k][1] for k in meanings},
        )
    }
    qs["effort"] = choice("考えて答えるなら、どれくらい深く考えるべきか", dict(EFFORTS))
    if "time" in meanings:
        # 黙る依頼の長さ（段 4-4c・いままでと同じ選択肢）。「黙る依頼を受ける」を選んだときだけ使う。
        qs["quiet_minutes"] = choice(
            "もしこの人がしばらく黙っていてと頼んでいるなら、何分か", dict(QUIET_MINUTES)
        )
    for m in meanings:
        actions = _actions_for(m, family)
        if len(actions) < 2:
            continue
        qs[f"action_{m}"] = choice(
            f"もしこの人の言葉が「{MEANINGS[m][0]}」なら、直前のやりとりの続きとして、パジュは次にどうするか",
            actions,
        )
    return qs


@dataclass(frozen=True)
class Outcome:
    """決まりで決めた最終の動作。`unsure` のときは `unsure_question` を聞いて決め直す。"""

    final: str
    name: str = ""  # 名乗り・否定の 2 回目で選んだ呼び方
    why: str = ""


def _conf(answer: "dict[str, Any] | None") -> float:
    return float((answer or {}).get("confidence", 0) or 0)


def decide(meaning: "dict[str, Any]", action: "dict[str, Any] | None") -> Outcome:
    """1 回目と 2 回目の答えから、本人の決まりで最終の動作を決める。"""
    m = str(meaning.get("choice") or "")
    if m == "unformed":
        return Outcome("silent", why="成立しないもの→黙る")
    a = str((action or {}).get("choice") or "")
    if m == "off_context":
        sure_ask = a == "ask_back" and _conf(action) >= THRESHOLD
        return Outcome("ask_back" if sure_ask else "silent", why="文脈に合わない言葉")
    if _conf(meaning) < THRESHOLD:
        return Outcome("unsure", why="1 回目がしきい値未満")
    if not a:
        return Outcome("unsure", why="2 回目の答えが無い")
    if a == "ask_back":
        return Outcome("ask_back", why="聞き返す（確信度に関係なく）")
    if _conf(action) < THRESHOLD:
        return Outcome("unsure", why="2 回目がしきい値未満")
    if m in ("claim", "deny"):
        return Outcome(m, name="" if a == OTHER else a, why="名乗り・否定")
    return Outcome(a, why="使う")


def unsure_question() -> "dict[str, dict]":
    """越えなかったときの問い：よく考えるか、軽く聞き返すか。"""
    from ..backends.jev import choice

    return {
        "action": choice(
            "この人の言葉に、パジュはよく考えて返すか、軽く聞き返すか", dict(UNSURE_ACTIONS)
        )
    }


def resolve_unsure(answer: "dict[str, Any] | None") -> str:
    """越えなかったときの問いの答え。0.6 以上ならそれ、未満なら軽く聞き返す（本人）。"""
    picked = str((answer or {}).get("choice") or "")
    if picked in UNSURE_ACTIONS and _conf(answer) >= THRESHOLD:
        return picked
    return "ask_back"
