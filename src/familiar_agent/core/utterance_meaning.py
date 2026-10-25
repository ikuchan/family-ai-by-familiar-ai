"""発話の意味と動作の問い（出-ay 段 4-4a・2026-10-09・`設計方針_判定の段` §2.2.5）。

調停の組み替えで、人の言葉への最初の反復は「1 回目に意味、2 回目に意味ごとの動作」を聞くと決めた（本人）。実験
（`scripts/experiment_meaning.py`・正解 8 場面中 7・1 場面 約 0.3 秒）で決めた問いと決まりを、本体の部品としてここに置く。
ここは問いを組むのと、答えから最終の動作を決めるだけで、Jev も軽量LLM も呼ばない。

- 意味は明確なものから、あやふやなものの順。使えないもの（確認待ちでない・すすめた曲の返事を待っていない・音楽が無い・
  カメラが無い）は並べない。
- 先読みで 1 回に並べる：意味の問いと、意味ごとの「もしこの言葉が○○なら」の問いを同じ回で聞き、返った意味の答えだけ使う
  （Jev にキャッシュは無く、2 回に分けると状態の文を 2 回送る）。
- 「文脈」は直前のやりとり（人とパジュの言葉）のことで、思い出した記憶ではない。
- 決まり：成立しないものは黙る。文脈に合わない言葉は、名前が無ければ黙り、名前で呼ばれたら次に確率の高い意味で読む
  （出-db 段 B）。ほかは
  確率で決める（出-db 段 A・2026-10-25・本人の決定 1-a）：返す側（軽く返す＋考えて返す）の確率の和が、ほかのどの動作よりも
  大きければ返し、そうでなければいちばん確率の高い動作。名乗り・否定だけは 0.6 を見て、足りなければ聞き返す（聞き返すは
  軽量LLM が 20 字までで書く）。

動作の鍵：`ask_back`（聞き返す）・`silent`（黙る）・`state_light`（状態を伝える）・`reply_light`（軽く返す）・
`reply_full`（考えて返す＝主LLM）・`quiet`（黙る依頼を受ける）・`lift_quiet`（黙るのを解く）・道具の名前。名乗りと否定の
2 回目は家族の呼び方（と `other`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .skills import Skill

#: しきい値（2026-10-09 本人の決定）。出-db 段 A（2026-10-25）から、意味と動作の決まりでは名乗り・否定だけに使う。
#: ほかの意味と動作は確率で決める——確信度は確率の集まり方で、害の無い選択肢に割れただけで聞き返しに倒れていた。見立ての「たぶん」と、調停の確認の答え・聞き取りの確認の答えもこの値を見る。
THRESHOLD = 0.6
#: 聞き返すときに軽量LLM へ渡す文字数の上限（2026-10-08 本人の決定）。
ASK_BACK_MAX_CHARS = 20

#: 意味。鍵 → (名前, Jev への説明)。明確なものから、あやふやなものの順（本人）。確認待ちへの答えは、確認待ちのあいだ調停が
#: 3 択だけで決めるので、ここには置かない（知-o・2026-10-13・`arbiter._confirm_by_rule`）。
MEANINGS: "dict[str, tuple[str, str]]" = {
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
    "music": (
        "音楽に関する依頼",
        "音楽に関する依頼（かける・止める・次の曲・音量・いま何の曲か聞く）。何分・何時といった長さや時刻だけを言っていて、"
        "曲・歌手・音楽という言葉が無ければ、音楽ではない",
    ),
    # 長さや時刻の手がかり（出-cb・10/13 19:58「3分のダイマーをかけて」を音楽と読み、音楽をかけようとして失敗した）。
    "time": (
        "時間に関する依頼",
        "時間に関する依頼（アラーム・タイマー・ストップウォッチ）。何分・何秒・何時といった長さや時刻を言っていれば、"
        "ほかの語の書き起こしが崩れていても時間",
    ),
    # 黙る依頼は時間から独立させた（出-cn・2026-10-15・本人の決定カ2）。時間の下に置くと「黙って」は「その他の会話」に
    # 読まれ、動作では「黙る（何も言わず、何もしない）」を選んで、黙る依頼にならなかった（実験 1・2：18 回中 0 回）。
    "speaking": (
        "話しかけ方の依頼",
        "パジュがしゃべること自体についての依頼（しばらく黙っていて・静かにして・話しかけないで・もう話していいよ）。"
        "待って・ちょっと待ってのように、いまの動作を止めてほしいだけの言葉は当たらない",
    ),
    "look": ("見る依頼", "見る依頼（そっちを向いて・何が見える・見て）"),
    "research": (
        "調べる必要のある問い",
        "世の中のことや記憶を尋ねていて、その答えをまだ持っていないか、"
        "天気・日程のように変わることで前の答えが古い",
    ),
    # 家族の予定・家のきまりを見に行く受け皿（知-ax・10/11 08:24「今日の予定は?」が other 0.42 に読まれた）。
    # 鍵は Jev にも見えるので、測った形のまま日本語にする。
    "家族の情報源": (
        "家族に関わる情報源を見に行く問い",
        "家族に関わること（家族の予定・家のきまり・家の記録や日記）を尋ねていて、家族の情報源（予定表・家のきまり・"
        "家の記録）を見に行けば答えられる（今日の予定は？・明日なにある？・ママは何時に帰る？・ゴミの日は？）",
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
    # 出-db 段 B（2026-10-25・本人の決定 3-e・案 C）：「パジュに向けた言葉ではなさそう」は宛先の判定と同じことを聞いていたので
    # 外し、話の筋が通らないことだけにした。「ところで」で始まる言葉を除く書き方は、かえってこれを選ばせた（実験 0.54〜0.92）。
    "off_context": (
        "これまでの文脈に合わない言葉",
        "直前のやりとりの続きとしても、新しい話題としても、何を言いたいのか通らない",
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
    # いま鳴っている曲を聞かれたら答える（出-bf・2026-10-10）。答えの文は機械が作る（`core/music_now`）。
    "music_now": "いま鳴っている曲が何かを伝える（何の曲か聞かれた）",
    "set_timer": "タイマーを掛ける（何分後に鳴る）",
    "cancel_timer": "タイマーを止める・取り消す",
    "pause_timer": "タイマーを一時停止する",
    "resume_timer": "タイマーを再開する",
    "set_alarm": "アラームを掛ける（何時に鳴る）",
    "cancel_alarm": "アラームを取り消す",
    "start_stopwatch": "ストップウォッチで測り始める",
    "stop_stopwatch": "ストップウォッチを止めて測った長さを言う",
    # 「待って」の断りは古い問いにあった（出-au 段 5-7d）。段 4-4c で移し漏れ、段 5 で戻した（2026-10-10 本人の決定ア）。
    "quiet": (
        "しばらく黙っていてという頼みを受ける"
        "（待って・待てぃは動作を止めてほしいだけで、これには当たらない）"
    ),
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
    # 予定表と Notion が取り合ったので書き分けた（知-ax）。
    "family_schedule": (
        "家族の予定表を見る。誰かの今日・明日・週末の予定、何時からか・何時に帰るかを尋ねられたとき"
        "（今日の予定は？・明日なにある？・ママは何時に帰る？）"
    ),
    "house_rules": "家の決まり（ゲームをしていい曜日・帰ったときの約束など）を見る",
    "notion_search": (
        "家族が書き溜めた家の目次・日次の記録・やることの一覧を探す（予定は予定表にある・世の中のことは載っていない）"
    ),
    "journal": "持ち主の日ごとの記録（よく眠れたか・調子）を見る",
    "vault": "いま話している人が書き溜めた考え・経緯・検討の記録に聞く",
}
_COMMON = ("ask_back", "silent")
#: 意味ごとに並べる動作（本人の表）。名乗りと否定は家族の呼び方から作る（`fanout_questions`）。
ACTIONS_BY_MEANING: "dict[str, tuple[str, ...]]" = {
    "suggestion": ("suggestion_like", "suggestion_decline", *_COMMON),
    "claim": (),
    "deny": (),
    "accepted_check": ("state_light", *_COMMON),
    "music": ("play_music", "stop_music", "next_track", "music_volume", "music_now", *_COMMON),
    "time": (
        "set_timer",
        "cancel_timer",
        "pause_timer",
        "resume_timer",
        "set_alarm",
        "cancel_alarm",
        "start_stopwatch",
        "stop_stopwatch",
        *_COMMON,
    ),
    # 黙る（`silent`）は並べない（出-cn）。「黙って」と字面が合い、黙る依頼より選ばれた（実験 3 で外すと 18 回中 18 回 quiet）。
    "speaking": ("quiet", "lift_quiet", "ask_back"),
    "look": ("look", "see", *_COMMON),
    "research": ("search_deferred", "recall", *_COMMON),
    "家族の情報源": (
        "family_schedule",
        "house_rules",
        "notion_search",
        "journal",
        "vault",
        "recall",
        *_COMMON,
    ),
    "answerable": ("reply_full", "reply_light", *_COMMON),
    "other": ("reply_light", "reply_full", *_COMMON),
    "off_context": ("silent",),  # 名前が無ければいつも黙る（出-db 段 B）。2 回目は聞かない
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


def offered(*, music: bool, camera: bool, suggesting: bool = False) -> "list[str]":
    """1 回目に並べる意味。使えないもの（すすめた曲の返事を待っていない、なども）は並べない。"""
    skip = set()
    if not suggesting:
        skip.add("suggestion")
    if not music:
        skip.add("music")
    if not camera:
        skip.add("look")
    return [k for k in MEANINGS if k not in skip]


#: 家族の情報源の動作のうち、繋がっているときだけ並べる道具（家の記録・本人の記録）。記憶はいつも並べる（知-ax）。
_HOUSE_TOOLS = frozenset({"family_schedule", "house_rules", "notion_search", "journal", "vault"})
#: 名前を言わずに否定したとき（「ちがうよ」）に選ぶ、いまの話者の印（古い問いから移した・段 5）。
_NAMELESS = "（名前を言わずに否定したときも、いま話者としている人のこれ）"


def _actions_for(
    meaning: str,
    family: "list[str]",
    *,
    speaker: str = "",
    silenced: bool = True,
    tools: "set[str] | None" = None,
) -> "dict[str, str]":
    if meaning == "claim":
        return {**{n: f"{n}だと名乗った" for n in family}, OTHER: "家族の誰でもない人だと名乗った"}
    if meaning == "deny":
        out = {n: f"{n}ではないと言った" for n in family}
        if speaker:
            out[speaker] = f"{speaker}ではないと言った{_NAMELESS}"
        return {**out, OTHER: "どの呼び方を否定したか分からない"}
    keys = ACTIONS_BY_MEANING[meaning]
    if not silenced:
        keys = tuple(k for k in keys if k != "lift_quiet")  # 解くは黙っているあいだだけ（段 5）
    if tools is not None:
        keys = tuple(
            k for k in keys if k not in _HOUSE_TOOLS or k in tools
        )  # 繋がっている道具だけ（段 5）
    return {k: ACTIONS[k] for k in keys}


def fanout_questions(
    *,
    music: bool,
    camera: bool,
    family: "list[str]",
    suggesting: bool = False,
    speaker: str = "",
    silenced: bool = True,
    tools: "set[str] | None" = None,
    skills: "tuple[Skill, ...] | list[Skill]" = (),
    active: "tuple[str, ...]" = (),
) -> "dict[str, dict]":
    """意味の問いと、意味ごとの 2 回目の問いを、先読みで 1 回に並べる（返った意味の答えだけ使う）。

    段 5（2026-10-10 本人の決定ア）で、古い問いにあった決まりを移した：名前を言わない否定はいまの話者（`speaker`）を
    選べる・解くは黙っているあいだ（`silenced`）だけ・家の記録の道具は繋がっているもの（`tools`・None なら全部）だけ。
    """
    from ..backends.jev import choice

    meanings = offered(music=music, camera=camera, suggesting=suggesting)
    qs: dict[str, dict] = {
        "meaning": choice(
            # 話題を変えた言葉も、その中身で選ばせる（出-db 段 B・INSTR_C）。続きとしてだけ読むと、話題を変えた頼みを
            # 文脈に合わないと読む。
            "この人の言葉は、次のどれに当たるか（直前のやりとりを踏まえて読む。話題を変えた言葉も、その中身で選ぶ。"
            "思い出したことは文脈に含めない）",
            {k: MEANINGS[k][1] for k in meanings},
        )
    }
    qs["effort"] = choice("考えて答えるなら、どれくらい深く考えるべきか", dict(EFFORTS))
    if "speaking" in meanings:
        # 黙る依頼の長さ（段 4-4c・いままでと同じ選択肢）。「黙る依頼を受ける」を選んだときだけ使う。
        qs["quiet_minutes"] = choice(
            "もしこの人がしばらく黙っていてと頼んでいるなら、何分か", dict(QUIET_MINUTES)
        )
    for m in meanings:
        actions = _actions_for(m, family, speaker=speaker, silenced=silenced, tools=tools)
        if len(actions) < 2:
            continue
        # 出-cq（2026-10-15 本人の決定ア）：動作は「続きとしてどうするか」でなく「何を頼んでいるか」で聞く。
        # 続きとして聞くと、直前に音楽を頼んだやりとりに引かれ「ストップ」でも play_music を選んだ（id 455）。
        # 名乗り・否定は頼みではないので、続きとして読む元の問いのまま。
        ask = (
            "直前のやりとりの続きとして、パジュは次にどうするか"
            if m in ("claim", "deny")
            else "この人の言葉は、パジュに何をしてほしいと頼んでいるか"
        )
        qs[f"action_{m}"] = choice(f"もしこの人の言葉が「{MEANINGS[m][0]}」なら、{ask}", actions)
    # パジュ用のスキル（環-au 段 B・2026-10-25・本人の決定ア）。スキルごとに「いま要るか」を 2 択で聞く（待ちは増えない）。
    # 鍵は英数字（記録の欄名）で、何番目のスキルかを `chosen_skills` が名前に戻す。すでに載っているもの（`active`）は外さない
    # ので聞かない。
    for i, sk in enumerate(skills):
        if sk.name in active:
            continue
        qs[f"skill_{i}"] = choice(
            f"この会話で、パジュは「{sk.name}」のやり方を使うか。使うのは、{sk.when}",
            dict(SKILL_NEED),
        )
    return qs


#: スキルの問いの選択肢（環-au）。
SKILL_NEED = (("need", "使う（いまのやりとりに要る）"), ("no", "使わない"))


def chosen_skills(got: "dict[str, Any]", skills: "tuple[Skill, ...] | list[Skill]") -> "list[str]":
    """Jev が「使う」を選んだスキルの名前（環-au）。答えが無いものは選ばない。"""
    return [
        sk.name
        for i, sk in enumerate(skills)
        if str((got.get(f"skill_{i}") or {}).get("choice") or "") == "need"
    ]


@dataclass(frozen=True)
class Outcome:
    """決まりで決めた最終の動作。"""

    final: str
    name: str = ""  # 名乗り・否定の 2 回目で選んだ呼び方
    why: str = ""


def _conf(answer: "dict[str, Any] | None") -> float:
    return float((answer or {}).get("confidence", 0) or 0)


#: 見立てを渡さない意味（言葉として読めていない）。
_NO_GUESS = frozenset({"off_context", "unformed"})


def guess(meaning: "dict[str, Any]", action: "dict[str, Any] | None") -> str:
    """聞き返すときに渡す Jev の見立て。数字は言葉に直す。見立てが無ければ空。

    迷って聞き返しに倒れても、書く側は何を迷ったかを知らず、空の問いしか書けなかった（出-bl・2026-10-11）。
    いちばん高い意味と、その意味でいちばん高い動作（聞き返す・黙るを除く）を渡す。名乗り・否定の動作は呼び方なので渡さない。
    """
    m = str(meaning.get("choice") or "")
    if m not in MEANINGS or m in _NO_GUESS:
        return ""
    sure = "たぶん" if _conf(meaning) >= THRESHOLD else "もしかすると"
    out = f"{sure}「{MEANINGS[m][0]}」"
    top = guess_action(meaning, action)
    if top:
        out += f"。するなら：{ACTIONS[top]}"
    return out


def guess_action(meaning: "dict[str, Any]", action: "dict[str, Any] | None") -> str:
    """見立ての動作の名前（その意味でいちばん高い動作・聞き返す・黙るを除く）。無ければ空。

    聞き取りの確認に「はい」と答えたら、この動作を投げる（出-cm）。名乗り・否定の動作は呼び方なので動作に数えない。
    """
    m = str(meaning.get("choice") or "")
    if m not in MEANINGS or m in _NO_GUESS or m in ("claim", "deny"):
        return ""
    acts = [(k, p) for k, p in _probs(action).items() if k in ACTIONS and k not in _COMMON]
    return max(acts, key=lambda kv: kv[1])[0] if acts else ""


def _probs(answer: "dict[str, Any] | None") -> "dict[str, float]":
    """選択肢ごとの確率。確率が無ければ、選ばれた 1 つを 1.0 と読む。"""
    answer = answer or {}
    probs = {str(k): float(v or 0) for k, v in dict(answer.get("probabilities") or {}).items()}
    if not probs and answer.get("choice"):
        probs = {str(answer["choice"]): 1.0}
    return probs


#: 返す側の動作。確率は二つに割れやすいので、和で比べる（出-db 段 A）。
_ANSWERS = ("reply_light", "reply_full")


#: 返す側の意味。動作と同じく、確率は二つに割れやすいので和で比べる（出-db 段 A）。
_ANSWERING_MEANINGS = ("other", "answerable")
#: 返す側の和では覆さない意味（道具を動かさない・黙る・名乗り）。
_KEPT_MEANINGS = frozenset({*_ANSWERING_MEANINGS, "unformed", "off_context", "claim", "deny"})


#: 名前つきの文脈に合わない言葉を読み直すのに要る、次の意味の確率（出-db 段 B・2026-10-25・本人の決定）。記録 2 件
#: （本物 0.36・崩れた声 0.13）のあいだに置いた仮の値で、実機で貯めて見直す（`計測_設定値根拠台帳`）。
REREAD_MIN = 0.25
#: 読み直し先にしない意味。
_NOT_REREAD = frozenset({"off_context", "unformed", "claim", "deny"})


def reread_named(meaning: "dict[str, Any]") -> "dict[str, Any]":
    """名前で呼ばれた言葉が「文脈に合わない」なら、次に確率の高い意味で読み直す（出-db 段 B・本人の決定 3-e）。

    名前で呼ばれたなら話しかけられているので、黙らない。文脈に合わない・成立しない・名乗り・否定を除いた、いちばん確率の
    高い意味にする。その確率が `REREAD_MIN` に届かなければ読み直さない（黙る）。名前で始まる崩れた声（10/24 19:02
    「パチュうちゅう…これです」：次の意味はその他 0.13）を黙らせ、名前つきの本物の頼み（10/24 18:43・ある試合の結果を
    調べてという頼み：調べる 0.36）は活かす（本人の決定ア）。名乗り・否定は読み違えて話者を付け替える害が
    大きいので、読み直し先にしない。
    """
    if str(meaning.get("choice") or "") != "off_context":
        return meaning
    rest = {k: p for k, p in _probs(meaning).items() if k not in _NOT_REREAD}
    if not rest:
        return meaning
    best = max(rest, key=lambda k: rest[k])
    if rest[best] < REREAD_MIN:
        return meaning
    return {**meaning, "choice": best, "confidence": rest[best]}


def settle_meaning(meaning: "dict[str, Any]") -> "dict[str, Any]":
    """道具を動かす意味が弱く選ばれたら、返す側の意味に置き換える（出-db 段 A・2026-10-25・本人の決定ア）。

    1 回目の確信度で絞るのをやめたので、意味の確率が割れたまま道具が動いていた（見直しの指摘：話しかけ方 0.30 で
    30 分黙る）。返す側の意味（その他の会話・知っていることで答えられる）の確率の和が、選ばれた意味の確率より大きければ、
    返す側のうち確率の高いほうにする。成立しない・文脈に合わない・名乗り・否定は覆さない。
    """
    m = str(meaning.get("choice") or "")
    if not m or m in _KEPT_MEANINGS:
        return meaning
    probs = _probs(meaning)
    answering = {k: probs[k] for k in _ANSWERING_MEANINGS if k in probs}
    if answering and sum(answering.values()) > probs.get(m, 0.0):
        best = max(answering, key=lambda k: answering[k])
        return {**meaning, "choice": best, "confidence": answering[best]}
    return meaning


def decide(meaning: "dict[str, Any]", action: "dict[str, Any] | None") -> Outcome:
    """1 回目と 2 回目の答えから、本人の決まりで最終の動作を決める。"""
    m = str(meaning.get("choice") or "")
    if m == "unformed":
        return Outcome("silent", why="成立しないもの→黙る")
    if m == "off_context":
        # 名前の無い言葉は、いつも黙る（出-db 段 B・本人の決定 3-e）。名前つきは呼び手が `reread_named` で読み直してから来る。
        return Outcome("silent", why="文脈に合わない言葉→黙る")
    a = str((action or {}).get("choice") or "")
    if m in ("claim", "deny"):
        # 誰かを決めつけるので、名乗り・否定だけは 0.6 を見る（出-db 段 A・本人）。
        if _conf(meaning) < THRESHOLD or _conf(action) < THRESHOLD:
            return Outcome("ask_back", why="名乗り・否定の確信度不足")
        return Outcome(m, name="" if a == OTHER else a, why="名乗り・否定")
    probs = _probs(action)
    if not probs:
        return Outcome("reply_full", why="2 回目の答えが無い")
    answering = {k: p for k, p in probs.items() if k in _ANSWERS}
    rest = [p for k, p in probs.items() if k not in _ANSWERS]
    if answering and sum(answering.values()) > max(rest, default=0.0):
        return Outcome(max(answering, key=lambda k: answering[k]), why="返す側の和がまさる")
    return Outcome(max(probs, key=lambda k: probs[k]), why="いちばん確率の高い動作")
