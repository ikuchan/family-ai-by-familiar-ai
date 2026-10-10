"""調停器（ARB）：パジュが次に何をするかを自分で決め、反復の出し方を3つへ振り分ける。

**調停器はパジュの心そのものである**（2026-09-05・出-e-に）。エージェントの中に置かれた
部品ではない。以前は「あなたは対話エージェントの内部で、次の一手を選ぶ調停器である」と
名乗らせており、`[あなたは誰か]` で人格を渡しながら次の行で自分を機構として置く食い違いが
あった。**待ってもらう一言と本応答は、同じパジュの2つの出口である**——実機では、本応答が
ですますなのに待ってもらう一言だけタメ口になり、同じ人の言葉として揃わなかった。

正本＝`I内部設計根拠` 段4。**基準は作り込まず軽量LLM の自己判断**とし、調整はプロンプトで
行う（閾値や点数式を置かない）。実測では1ターン 10.5 秒のうち LLM が 10.2 秒を占め、
`recall` を投げるだけの反復にもフルLLM を同期で使っていた。

- **light**：短文で答えきれる会話は軽量LLM が応答して反復を閉じる（フルを起こさない）。
- **full** ：熟慮・想起が要る会話はフルLLM を起こす。**思考の深さ（effort）も軽量LLM が決める**。
- **action**：探すと決まっている反復は、軽量LLM が**どの動作で・何を**調べるかを決めて投げ、
  反復を閉じる。**つなぎの発話（「調べてみるね」）も一緒に返させる**：フルLLM を経由すると
  実測 2.9 秒かかるところ、調停だけなら 0.7 秒で反応が返る（正本③ 段5 の「内部二段」を
  action 分岐へ当てたもの）。

判定できないとき・時間切れは **full／effort=low** へ倒す（2026-09-12 に high から改めた・課題5 G 章）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, replace
from typing import Any
from collections.abc import Awaitable, Callable

from ..core import measure
from ..core.aio import wait_within
from ..core.utterance_meaning import ASK_BACK_MAX_CHARS

logger = logging.getLogger(__name__)

_EFFORTS = ("low", "medium", "high")


# 並びは**キャッシュが前方一致で効く**ことに合わせる。起動中ほぼ変わらないもの（人格・家族・
# 規則）を先に置き、変わるもの（上限の但し書き・時刻・顔ぶれ・人の言葉・作業状態）を後ろへ。
# 実機で調停が 2 秒で返らず時間切れになり、沈黙依頼が読まれないまま倒れた。
#: 安定部（立ち位置＋人格＋家族）は**システム文で渡す**。1本の文字列だったのは
#: `complete()` にシステム文の口が無かったからで、出-e-い でその口を作った。分けると
#: 「安定を先、可変を後」を人が守る必要が無くなり（システム文は常に先）、呼び出し間で
#: 一字一句同じなので前方一致キャッシュが最大限効く。
#: 課題の指示は可変の data と同じプロンプト側に置く。**交互に並ぶ問題はここで消える**
#: ——「口調の注意」が `[いま]` の直後にあるのも、JSON の指示が最後にあるのも、
#: どちらもプロンプト側の話になる。
@dataclass
class Decision:
    """調停の結果。`branch` 以外はその分岐でだけ意味を持つ。"""

    branch: str  # light | full | action
    text: str = ""  # light：発話／action：つなぎの一言
    effort: str = "low"  # full：思考の深さ（既定 low・課題5 G 章）
    action: str = "recall"  # action：どの動作で調べるか
    query: str = ""  # action：探す語
    tool_input: "dict | None" = None  # action：道具へそのまま渡す入力（タイマー・知-n）
    # 黙る長さ（分）。0＝黙らない、-1＝頼まれたが長さの指定なし（受け側が既定を当てる）。
    silence_minutes: int = 0
    lift_silence: bool = False  # 黙っていたのを「もう話していいよ」と解かれた
    speaker_claim: str = ""  # 人が名乗った名前（知-w・在席があるときだけ機械が話者に付ける）
    #: 身元の否定（出-am・2026-09-22）。「パパじゃないよ」「ちがうよ」で、否定された呼び方。
    #: 顔ぶれへ**入る**口（名乗り・見立て）に対する、**出る**口である。
    not_person: str = ""
    # 想起の時間軸の基準。人の言葉が時期を指しているとき（「去年の夏の話」）に動かす。
    # 既定（None）は「いま」が基準・幅は Config の既定（3日）。
    time_ref: str = ""  # ISO 8601（例 "2025-08-15T00:00:00"）
    time_span_days: float = 0.0  # 幅＝半減期（日）。0 は指定なし


# 倒れたときも low（2026-09-12 決定・課題5 G 章）。以前は high で、Sonnet では 2 倍遅かった。
#: 人の発話・機器が起点のとき（返事の型）。
_LEAD_REPLY = "いま人から届いた言葉と、いまの作業状態を見て、次のどれかを選ぶ。"
# 起点が機器（人の出入り・タイマー・メモ）のとき。返事型のまま渡すと、W の直近にある人の言葉を
# いまの頼みとして読み、鳴ったタイマーの知らせでタイマーを 3 本掛け直した（2026-09-16 実機 17:00）。
_LEAD_DEVICE = (
    "いま届いた知らせ（人の出入り・タイマー・メモ）と、いまの作業状態を見て、次のどれかを選ぶ。"
    "これは人の言葉ではなく機器からの知らせで、**直近のやりとりは済んだこと**——人の言葉に改めて応じない。"
    "届いた知らせの中身を、短く伝える。"
)
_HEADING_DEVICE = "[届いた知らせ]"
# 起点が道具の帰り（タイマー・アラーム）のとき（出-x・2026-09-18）。返事型のまま渡すと、「確かめて」の
# 返りを読んでも聞き返さずに `set_timer` を選び直した（実機 14:50・15:28・15:42）。実験（`根拠台帳` §35・
# 3 種の実機 W × 8 回）：この先導文＋想起なし＋返った道具を候補から外す、で 24/24 が light。
# 直近に「うまくできなかった」があるとき（W c）は末尾の一文が無いと 2/8 だった（取り返そうと道具を選ぶ）。
# 文は実験（`scripts/experiment_arbiter_confirm.py` の `LEAD_TOOL_RETURN_2`・出-au 段 5-7d で撤去・履歴にある）のまま。
_LEAD_TOOL_RETURN = (
    "いま**道具から返りが届いた**（作業状態の最上部）。人に届いた言葉ではない。返りを見て、次のどれかを選ぶ。"
    '返りが人に伝える文（「掛けた」「確かめて：「…」」「動いている」）なら、その文を **"light"** でそのまま、'
    "または相手に合わせて言い換えて伝える。返りが「確かめて」なら聞き返すだけでよい——**掛け直しはあなたの仕事ではない**"
    '（「いい」と言われたら別の仕組みが掛ける）。指示があいまいで確かめたいことがあれば、それも "light" で聞く。'
    '別の道具が要るときだけ "action"。'
    "直近のやりとりに「うまくできなかった」「まだ掛かっていない」があっても、**取り返そうとして道具を選ばない**。"
    "道具は既に返っている（最上部）。取り返すのは、返りをきちんと伝えることで足りる。"
    "同じ失敗が続いていれば、そのことを一言添えてよい。"
    "「した」と言えるのは、道具の返りにそう書いてあるときだけ。返りが断り（動いている・止めてから・無い）なら、"
    "その断りをそのまま伝える。返りに無いことを、したと言わない。"
)
_HEADING_TOOL_RETURN = "[道具から返ったもの]"
_HEADING_REPLY = "[人の言葉]"

#: 道具が要る頼みの手がかり〔仮・2026-09-15〕。light で「できました」と言わせないための機械の守り
#: （実機 22:47「３分のタイマーをかけて」に light が「タイマーをセットしました」と答え、掛かっていなかった）。
_NEEDS_TOOLS = re.compile(
    r"タイマー|アラーム|ストップウォッチ|測って|計って|分後|秒後|時に(起こ|教え|知らせ|呼ん)|止めて|ストップ|覚えて(おい|て)|"
    r"予定(は|を|ある|入って)|スケジュール"
)


def _as_dict(value) -> "dict | None":
    """dict か、JSON の文字列としての dict を dict に。それ以外は None。"""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            got = json.loads(value)
        except json.JSONDecodeError:
            return None
        return got if isinstance(got, dict) else None
    return None


def needs_tools(utterance: str) -> bool:
    """道具が要る頼みか（light で答えてはいけない）。"""
    return bool(_NEEDS_TOOLS.search(utterance or ""))


_ZEN = str.maketrans("０１２３４５６７８９", "0123456789")
_Q_AFTER = re.compile(r"(\d+(?:\.\d+)?)\s*分(?:後|間)?")
_Q_AT = re.compile(r"(\d{1,2})\s*(?:[:：時]\s*(\d{1,2})?|時)\s*(半)?")


def timer_input_from_query(query: str) -> "dict | None":
    """調停が `tool_input` でなく `query` に「1分」「7時 起こす」と書いてきたときに入力を作る（実機 23:13）。

    「N 分」→ `after_minutes`、「H 時（半／M 分）」→ `at`。残りの語が label。読めなければ None。
    """
    q = (query or "").translate(_ZEN).strip()
    m = _Q_AFTER.search(q)
    if m:
        rest = (q[: m.start()] + q[m.end() :]).strip(" 　、。に")
        return {"after_minutes": float(m.group(1)), "label": rest or "タイマー"}
    m = _Q_AT.search(q)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0) + (30 if m.group(3) else 0)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            rest = (q[: m.start()] + q[m.end() :]).strip(" 　、。に")
            return {
                "at": f"{hour}:{minute:02d}",
                "label": rest or "アラーム",
            }  # 何時に＝アラーム（知-q）
    return None


#: 情動が起点のとき（自分の行動を決める型・情-e）。自発なので、許可も理由も要らない。
#: つなぎは書かせない（つなぎは相手への返事の前置き）。
_LEAD_SELF = (
    "いま自分の中から湧いたことと、いまの作業状態を見て、何をするかを決める。"
    "これは自分がしたくなったことで、誰かに頼まれたのではない。**許可は要らない。理由も要らない。**"
    "誰かに断ったり、待ってもらったりしない。"
    # 出-q（2026-09-15 実機）：自発の求めに、済んだ入室と自分の挨拶が『直近のやりとり』として
    # 載る。これを「いま起きたこと」と読んで挨拶をやり直した（「パパ、おかえりなさい」×2）。
    "**直近のやりとりは済んだこと**。入室・挨拶・終わった話に改めて反応しない。"
    "相手が話を切り上げていたら（「もういいや」「後で」）黙る。言うのは新しいことがあるときだけ。"
)
_HEADING_SELF = "[いま湧いたこと]"

_FALLBACK = Decision(branch="full", effort="low")

#: MCP の同期の道具（動作名 → 見出し・説明）。候補に載せるのは繋がっているときだけ。
_EXTRA_ACTIONS: dict[str, tuple[str, str]] = {
    "house_rules": (
        "家の決まりを見る",
        '"house_rules"（家の決まり・ゲームをしていい曜日・帰宅時の約束を引く。query は要らない）',
    ),
    # 見出しが空のものは query が要る。`family_schedule` の query は**日数**（期間つきの求め）。
    "family_schedule": (
        "",
        '"family_schedule"（家族の予定・今日／明日の予定・何時から、を引く。'
        "query に今日から何日ぶんかを数字だけで書く：1〜14・今日＝1・明日＝2・今週＝7）",
    ),
    # 見出しが空のものは query が要る（探す語を query に書く）。
    # Notion と Vault の書き分け（2026-09-15）：Notion は**家の目次・日次記録・Todo**（速い）、
    # Vault は**本人の考え・経緯・検討の中身**（数十秒）。両方に「記録・経緯」と書くと Notion に
    # 寄る（実機で「コーチングどうなった？」が 2 回とも Notion へ行った）。
    "notion_search": (
        "",
        '"notion_search"（家の目次・日次記録・Todo を探す。「〜は書いてある？」「何をやることになってる？」。'
        "探す語を query に）",
    ),
    "journal": (
        "日次記録を見る",
        '"journal"（日ごとの記録・最近よく眠れているか・調子。query は要らない）',
    ),
    # タイマー（知-n）。**調停が自分で掛ける**——道具の入力を `tool_input` に書く。掛かったら完了が
    # 戻り、調停が light で「掛けたよ」と言える（主LLM は起きない）。「確かめて」が返ったら light で聞く。
    "set_timer": (
        "",
        '"set_timer"（タイマー＝何分後に鳴る。tool_input に {"after_minutes": 3, "label": "何のため"}。'
        "返りが「確かめて」や「聞く」なら文をそのまま伝えて一度聞くだけ（掛け直しは要らない）。同時に 1 本）",
    ),
    # ストップウォッチ（知-u・2026-09-18）。タイマーとは別物——鳴らない・確認しない・黙らない。
    "start_stopwatch": (
        "",
        '"start_stopwatch"（ストップウォッチ＝「今から測って」「何分かかるか測って」。tool_input に {"label": "何を"}。同時に 1 本）',
    ),
    "stop_stopwatch": (
        "",
        '"stop_stopwatch"（ストップウォッチを止めて測った長さを言う。「ストップ」「何分だった？」。tool_input に {"id": 番号か "all"}。番号は [ストップウォッチ] の枠）',
    ),
    "cancel_timer": (
        "",
        '"cancel_timer"（「タイマー止めて」「やっぱりいい」。tool_input に {"id": 番号か "all"}。番号は [タイマー] の枠）',
    ),
    # 一時停止・再開（知-o 段 4・2026-09-18）。聞かないあいだも通る操作の言葉。
    "pause_timer": (
        "",
        '"pause_timer"（「一時停止」「ちょっと止めといて」。tool_input に {"id": 番号か "all"}）',
    ),
    "resume_timer": (
        "",
        '"resume_timer"（「再開」「続けて」。tool_input に {"id": 番号か "all"}）',
    ),
    # アラーム（知-q・2026-09-18）。タイマーとは別物——何時に。黙らない・聞かない状態にしない。
    "set_alarm": (
        "",
        '"set_alarm"（アラーム＝何時に鳴る。「7 時に起こして」。tool_input に {"at": "7:00", "label": "何のため"}。'
        "返りが「確かめて」なら理由を伝えて一度聞くだけ（掛け直しは要らない）",
    ),
    "cancel_alarm": (
        "",
        '"cancel_alarm"（「アラーム止めて」「明日の起こすのやめて」。tool_input に {"id": 番号か "all"}。番号は [アラーム] の枠）',
    ),
    # 音楽（知-ak 段 6・2026-10-07）。タイマーと同じく**調停が自分でかける**——道具の入力を `tool_input` に書く。
    # 載せていなかったので、調停は音楽を選べず、主LLM へ倒れたときだけ鳴らしにいった。
    "play_music": (
        "",
        '"play_music"（音楽をかける。「〇〇かけて」。tool_input に {"name": "言われた名前", "order": "ランダム か 順番"}。'
        "order は言われたときだけ。曲名を言われていなければ name は空（止まっていた続きをかける）。"
        # 「プレイリスト」と言われたら種類を書かせる（知-at）。書かないと Spotify 全体の検索が既定の「曲」で探した。
        '「プレイリスト」と言われたら tool_input に "kind": "プレイリスト" を足し、name からは「プレイリスト」の語を外す。'
        "[音楽] の枠が「いまは何も鳴っていない」なら、前にかけたことがあってもかける）",
    ),
    "stop_music": (
        "",
        '"stop_music"（音楽を止める。「止めて」「音楽消して」。tool_input は {}）',
    ),
    "next_track": (
        "",
        '"next_track"（次の曲へ。「次の曲」「飛ばして」。tool_input は {}）',
    ),
    "music_volume": (
        "",
        '"music_volume"（音楽の音量を変える。「大きくして」「小さくして」。tool_input に {"how": "大きく か 小さく"}）',
    ),
    # 確認待ちへの答え（出-y・2026-09-18）。[確認待ち] が作業状態の最上部にあるときだけ候補に載る。
    "confirm": (
        "「いい」と言われて掛ける",
        '"confirm"（[確認待ち] の問いに「いい」「うん」「お願い」と答えた。tool_input は要らない）',
    ),
    "decline": (
        "確かめたものをやめる",
        '"decline"（[確認待ち] の問いに「やめて」「いらない」「今はいい」と答えた。tool_input は要らない）',
    ),
    # 個人ティアの記録（知-g-い）。候補に載るのは本人のターンだけ（話者ゲート・`_extra_actions`）。
    "vault": (
        "",
        '"vault"（いま話している人の**考え・経緯・検討の中身**を、本人の記録に日本語の質問文で聞く。'
        "「なぜそう決めた？」「あれどうなってた？」「どっちにするか迷ってた件」。目次や Todo ではない。"
        "数十秒かかる。質問文を query に）",
    ),
}

#: `see` の見出しは入力に依らず固定（`event_loop._query_label`）。調停が投げても主LLM が
#: 投げても同じ鍵になり、「すでに調べた語は投げない」の抑止がそのまま効く。
SEE_QUERY = "目の前を見る"
#: `look` の見出し（query）。(c) 分岐は query が空だと full へ落ちるので、固定の語を入れる。
LOOK_QUERY = "首を向ける"


def assemble(
    data: dict, *, can_see: bool = False, origin: str = "発話", extra_actions: tuple[str, ...] = ()
) -> Decision | None:
    """Jev の判定と軽量LLM の文章をまとめた辞書から `Decision` を組み、**機械の守り**を通す（出-au 段 5-7d）。

    守りは、判定と文章を 1 回の軽量LLM に同居させていたころ（`_parse`）のまま移した。道具名の書き換え、候補に照らす、
    情動のつなぎを捨てる、分岐に要るものが無ければ倒す。読めない・守りに掛かったら None（呼び手が full へ倒す）。
    """
    branch = str(data.get("branch", "")).strip().lower()
    if branch not in ("light", "full", "action"):
        return None
    effort = str(data.get("effort", "low")).strip().lower()
    if effort not in _EFFORTS:
        effort = "low"
    # つなぎは**別の欄**で受ける（出-aj・2026-09-23）。`text` は `light` の返事の欄で、
    # 同じ欄を 3 つの用途で共有していたため、`light` の書き方（＝返事）が既定になり、
    # つなぎが用件を言い切っていた（実機 15:49・144 字）。`trash` は言いたいことの行き先で、
    # **読み捨てる**——行き先が無いと軽量LLM は何も書かず、`filler` が空になる（実測）。
    text = str(
        data.get("filler", "") if branch in ("full", "action") else data.get("text", "")
    ).strip()
    try:
        silence_minutes = int(data.get("silence_minutes", 0))
    except (TypeError, ValueError):
        # 読めない値で黙り込むと、解けるまで何も言えなくなる。
        silence_minutes = 0
    time_ref = str(data.get("time_ref", "") or "").strip()
    try:
        time_span_days = float(data.get("time_span_days", 0) or 0)
    except (TypeError, ValueError):
        time_span_days = 0.0
    query = str(data.get("query", "")).strip()
    action = str(data.get("action", "")).strip() or "recall"
    tool_input = _as_dict(data.get("tool_input"))
    if tool_input is None and _as_dict(query) is not None:
        # 軽量LLM（Gemini）は tool_input を JSON の**文字列**として query に書く（実機 08:59〜09:00）。
        tool_input, query = _as_dict(query), ""
    allowed = (
        ("recall", "search_deferred", "fetch_deferred")
        + (("see", "look") if can_see else ())
        + tuple(a for a in extra_actions if a in _EXTRA_ACTIONS)
    )
    if branch == "action" and not query and not tool_input and text and not data.get("action"):
        # 動作が無く言葉だけ（「鳴らしていい？」と聞きたかった）は light として扱う（実機 23:14）。
        branch = "light"
    if action not in allowed:
        action = "recall"
    if action == "see":
        query = SEE_QUERY  # 見出しは固定。`(c)` 分岐は query が空だと full へ落ちる
    elif action == "look":
        if not tool_input and query:
            # 向き（右・左・上・下）か定点の名前を query に書いてきたときは、それを入力にする。
            from ..poses import DIRECTIONS

            tool_input = {"direction": query} if query in DIRECTIONS else {"pose": query}
        query = LOOK_QUERY
        text = ""  # 首を回すだけなので断らない（つなぎを言うと 2 回出る）
    elif action in _EXTRA_ACTIONS and _EXTRA_ACTIONS[action][0]:
        query = _EXTRA_ACTIONS[action][0]  # 見出しが固定の道具。query が要るものはそのまま
    elif action == "search_deferred" and query and not tool_input:
        source = str(data.get("source", "") or "").strip().lower()
        if source in ("brave", "tavily"):
            tool_input = {"query": query, "source": source}  # 担い手を書いたときだけ（知-am）
    elif action == "family_schedule" and not query:
        query = "1"  # 日数を書き忘れても action は落とさない（今日だけ・主LLM が呼び直せる）
    if action == "set_timer" and tool_input and set(tool_input) == {"id"}:
        action = "cancel_timer"  # 入力が id だけなら止める意図（「ストップ」に set_timer と書いた・実機 08:59）
    if action == "start_stopwatch" and tool_input and set(tool_input) == {"id"}:
        action = "stop_stopwatch"
    if (
        action == "set_timer"
        and tool_input
        and tool_input.get("at")
        and not tool_input.get("after_minutes")
    ):
        action = (
            "set_alarm"  # 何時に、はアラーム（別物・2026-09-18）。調停が set_timer に書いても直す
        )
    if action in ("confirm", "decline"):
        tool_input, text = (
            {},
            "",
        )  # 引数は無い。機械が預かった入力で掛ける／捨てる。道具は 0.1 秒で返る
    if action in (
        "set_timer",
        "start_stopwatch",
        "cancel_timer",
        "pause_timer",
        "resume_timer",
        "set_alarm",
        "cancel_alarm",
        "stop_stopwatch",
    ):
        # 見出し（同語二度投げの鍵）は label か id。tool_input が無ければ query から作る。
        if not tool_input and query:
            if action in ("set_timer", "set_alarm"):
                tool_input = timer_input_from_query(query)
                if tool_input and "at" in tool_input:
                    action = "set_alarm"
                elif tool_input and action == "set_alarm":
                    action = "set_timer"
            elif action == "start_stopwatch":
                tool_input = {"label": query}
            else:
                tool_input = {"id": query}
        if not tool_input:
            return None  # 掛けられない → full へ（主LLM が道具で掛ける）
        query = str(tool_input.get("label") or tool_input.get("id") or query or action).strip()
        # 道具は 0.1 秒で返る。つなぎを言うと、返りを見て言う一言と同じ文が 2 回出る（実機 08:59）。
        text = ""
    if action in _MUSIC_ACTIONS:
        # 音楽の道具は語ではなく道具の入力（曲の名前）を受ける。語が空のまま残ると「動作なのに語が無い」で倒れ、
        # Jev が play_music を選んでも必ず主LLM に回っていた（2026-10-09・出-ay 段 4-2）。タイマーと同じく入力から語を作る。
        tool_input = tool_input or {}
        query = str(tool_input.get("name") or query or action).strip()
        text = ""
    if action not in allowed:
        # **書き換えた後にも候補に照らす**（出-ag-ろ 穴 1）。上の書き換えは候補に照らした後で
        # 道具名を変えるので、返りの反復で外した `set_timer` が `set_alarm` から戻ってきた
        # （実機 2026-09-21 17:34・反復 2）。直接書いたときと同じく、読めない選択として倒す。
        return None
    # 情動が起点なら、light 以外の text（つなぎ）は捨てる。自発の行動に断りは要らない（情-e）。
    if origin == "情動" and branch != "light":
        text = ""
    # 分岐に必要なものが無ければ判定できていない＝倒す。**情動が起点の light で text が空は
    # 「黙る」の正当な返事**（候補文が「text を空にすれば黙る」と言っている）——倒さない
    # （出-w・2026-09-18 12:28 実機：フルへ倒れて主LLM が無駄に回った）。発話が起点なら
    # 返事しないのは判定できていないので従来どおり倒す。
    if branch == "light" and not text and origin != "情動":
        return None
    if branch == "action" and not query:
        return None
    return Decision(
        branch=branch,
        text=text,
        effort=effort,
        action=action,
        query=query,
        tool_input=tool_input,
        silence_minutes=silence_minutes,
        lift_silence=bool(data.get("lift_silence", False)),
        speaker_claim=str(data.get("speaker_claim", "") or "").strip(),
        not_person=str(data.get("not_person", "") or "").strip(),
        time_ref=time_ref,
        time_span_days=max(0.0, time_span_days),
    )


def _watch_late(call, started: float, prompt_len: int) -> None:
    """打ち切ったあとの呼び出しを裏で待ち、実際にかかった秒数を残す。

    応答には使わない（もう倒してある）。時間切れの値を決めるための計測だけが目的。
    """

    async def _wait() -> None:
        try:
            await call
        except Exception:  # noqa: BLE001
            logger.debug("遅れて返るはずの調停が失敗した")
            return
        logger.info(
            "調停が遅れて返った：%.2f 秒（プロンプト %d 字）",
            time.monotonic() - started,
            prompt_len,
        )

    task = asyncio.ensure_future(_wait())
    # 参照を残さないと GC に回収されうる。終わったら自分で外れる。
    _LATE_TASKS.add(task)
    task.add_done_callback(_LATE_TASKS.discard)


_LATE_TASKS: set = set()


# Jev に渡す状態の文の注意（出-ay 段 5c で、古い分岐の言葉 light・full・action を使わない言い方に直した）。
_THINKING_NOTE = """
この件で答えを組み立てるのは {round} 回目である。回を重ねても材料が増えていないなら、
それはもう分からないということなので、**これ以上は調べず**、いまある材料で答えるほうへ回す。
"""

_CAPPED_NOTE = """
これ以上は調べられない（反復の上限に達した）。調べ直すことはできず、いまある材料で答えることになる。
"""

#: Jev に渡す状態の文の先導文（出-ay 段 5c・2026-10-10 本人の決定ア）。起点ごとに「いま何が起きたか」だけを言う。
#: 何を選ぶかは問いの側が言うので、ここには選び方を書かない（古い分岐の「次のどれかを選ぶ」と目安は外した）。
#: 軽量LLM の先導文（`_LEAD_REPLY` など・書くときの指示）は別に持つ。
_JEV_LEAD = {
    "発話": "いま人から言葉が届いた。",
    "完了": "いま自分の動作の結果が届いた（作業状態の最上部）。人に届いた言葉ではない。",
    "情動": (
        "いま自分の中から湧いたことがある。これは自分がしたくなったことで、誰かに頼まれたのではない。"
        "**許可は要らない。理由も要らない。**"
        "**直近のやりとりは済んだこと**。入室・挨拶・終わった話に改めて反応しない。"
    ),
    "機器": (
        "いま機器から知らせが届いた（人の出入り・タイマー・メモ）。人の言葉ではない。"
        "**直近のやりとりは済んだこと**。"
    ),
}


# ── Arbiter：判定は Jev、文章は軽量LLM（出-au 段 5-7・`設計方針_判定の段` v0.4 §2.2.2） ──────────────


@dataclass
class ArbiterInput:
    """調停の材料。ループ（`_decide`）が組んで渡す。"""

    utterance: str
    workspace_ctx: str
    present_ctx: str = ""
    now_ctx: str = ""
    origin: str = "発話"  # 発話／機器／情動
    capped: bool = False
    thinking_round: int = 1
    can_see: bool = False
    extra_actions: tuple[str, ...] = ()
    tool_return: bool = False
    timer_active: bool = False
    family_md: str = ""
    people_md: str = ""  # 家族以外で知っている人（`PEOPLE.md`・知-ab）
    family_now: str = ""  # 家族のいまの様子（`[家族のいまの様子]`・知-ad）
    current_speaker: str = ""
    silenced: bool = False
    self_understanding: str = ""
    self_image: str = ""
    season_env: str = ""
    # 情動のうち話しかける軸（bond・esteem）。light の文を話しかけにする（出-at）
    talking: bool = False
    #: この反復で道具から返ったもの（名前・失敗の印・結果の文）。完了を機械で分けるのに使う（出-ay 段 4-2）。
    returned: "tuple[tuple[str, bool, str], ...]" = ()
    #: 情動の求めで発火した軸（seeking・safety・bond・esteem）。情動を軸で決めるのに使う（出-ay 段 4-3）。
    fired_axis: str = ""
    #: いまの音楽の様子を読む口（非同期・出-bf）。「いま何の曲？」に答えるときだけ呼ぶ。無ければ読めなかったとして答える。
    music_status: "Callable[[], Awaitable[dict | None]] | None" = None


#: 動作の短い説明（軽量LLM に、決まった動作として渡す・`_action_note`）。
_BASE_ACTIONS = {
    "recall": "自分の記憶（家族の出来事・過去の会話）を探す",
    "search_deferred": "インターネットで世の中のこと（天気・ニュース・調べもの）を調べる",
}
_QUIET_MINUTES = {"default": -1, "5": 5, "10": 10, "15": 15, "30": 30, "60": 60}


#: 音楽の道具（語ではなく道具の入力を受ける・`assemble` が入力から語を作る）。
_MUSIC_ACTIONS = frozenset({"play_music", "stop_music", "next_track", "music_volume"})

#: 完了の 2 回目の説明（出-ay 段 4-2）。道具は `_action_note` の説明を使う。
_COMPLETION_TEXT: "dict[str, str]" = {
    "silent": "黙る（結果を持っていれば足りる）",
    "tell_light": "結果を短く伝える",
    "reply_light": "結果をもとに短く返す",
    "talk_light": "結果をもとに、家族に短く話しかける",
    "reply_full": "結果を読んで、考えて返す",
    "talk_full": "記憶を踏まえて、考えて話しかける",
}
#: 情動の軸 → 2 回目に並べる動作（出-ay 段 4-3・本人の表）。rest は調停に来ない（REST の内省パス）。
_AFFECT_ACTIONS: "dict[str, tuple[str, ...]]" = {
    "seeking": ("search_deferred",),
    "safety": ("look", "search_deferred"),
    "bond": ("talk_light", "talk_full"),
    "esteem": ("talk_light", "talk_full", "search_deferred"),
}
#: すすめた曲への返事の動作 → `music_suggestion_reply` に渡す返事（出-ay 段 4-4f）。
_SUGGESTION_REPLIES: "dict[str, str]" = {
    "suggestion_like": "気に入った",
    "suggestion_decline": "いらない",
}
#: 軽量LLM が書く一言の種類 → 決めたこととして渡す言葉。
_LIGHT_WORDS: "dict[str, str]" = {
    # 字数は `utterance_meaning.ASK_BACK_MAX_CHARS`（段 4-4d）。超えて書けても切らずに話す（切ると文が途中で切れる・本人）。
    "ask_back": f"聞き返す（{ASK_BACK_MAX_CHARS} 字まで・何をしてほしいかを短く確かめる）",
    "state_light": "頼まれていたことの状態を短く伝える",
    "tell_light": "軽く伝える",
    "reply_light": "軽く返す",
    "talk_light": "軽く話しかける",
}


def _family_call_names(family_md: str) -> "list[str]":
    """家族の呼び方（1 人 1 つ）。名乗り・否定の 2 回目に並べる（出-ay 段 4-4b）。"""
    from ..core import parsing
    from ..core.speaker_claim import call_name_of

    try:
        members = parsing.parse_family_md(family_md or "")
    except Exception:  # noqa: BLE001
        return []
    return [n for n in (call_name_of(m) for m in members) if n]


class Arbiter:
    """調停。材料を受け取り、**Jev が決め**、要るときだけ**軽量LLM が書き**、機械の守りを通して `Decision` を返す。

    判定と文章を 1 回の軽量LLM の指示文に同居させていた形を割った（出-au 段 5-7）。Jev は写真を見られない
    ので、写真の読み取りは状態として W に載っている（段 5-7a）。
    """

    def __init__(self, *, jev, writer, timeout: "float | None" = None) -> None:
        self._jev = jev
        self._writer = writer
        self._timeout = timeout
        #: 文章の口が時間切れになったか（計測ログの「時間切れ」・層 3 が調停の秒数を見直す材料）
        self._timed_out = False

    def _state(self, inp: ArbiterInput) -> str:
        """Jev に送る文。いま何が起きたか（起点ごと）・いま・顔ぶれ・その場の言葉・作業状態（W 全部・本人の決定 イ）。"""
        if inp.returned or inp.tool_return:
            path = "完了"
        elif inp.origin in ("情動", "機器"):
            path = inp.origin
        else:
            path = "発話"
        # 見出しはいままでどおり（その下に載るのは、起点の言葉か道具の返り）。
        if inp.tool_return:
            heading = _HEADING_TOOL_RETURN
        elif inp.origin == "情動":
            heading = _HEADING_SELF
        elif inp.origin == "機器":
            heading = _HEADING_DEVICE
        else:
            heading = _HEADING_REPLY
        notes = (_CAPPED_NOTE if inp.capped else "") + (
            _THINKING_NOTE.format(round=inp.thinking_round) if inp.thinking_round > 1 else ""
        )
        return (
            f"[いま起きたこと]\n{_JEV_LEAD[path]}\n{notes}\n"
            f"[いま]\n{inp.now_ctx or '（分からない）'}\n\n"
            f"[いま誰が居るか]\n{inp.present_ctx or '（分からない）'}\n\n"
            f"{heading}\n{inp.utterance}\n\n"
            f"[いまの作業状態]\n{inp.workspace_ctx or '（なし）'}"
        )

    async def decide(
        self, inp: ArbiterInput, on_decided: "Callable[[str], None] | None" = None
    ) -> Decision:
        """次の一手を決める。起点ごとの道（発話・完了・情動・機器）が決め、要るときだけ軽量LLM が書く。倒れたら full。

        どの道にも当たらない入力（表に無い道具の完了・表に無い軸）は、Jev に聞かずに主LLM に任せる（出-ay 段 5e・
        本人の決定ア）。以前はここで古い分岐の問い（light・full・action を 1 回で聞く）に落ちていた。

        `on_decided`：発話の道で最終の動作が決まったら、軽量LLM が文を書く**前に** 1 回だけ呼ぶ（出-bg・聞こえた合図と
        反応の合図を遅らせないため）。
        """
        started = time.monotonic()
        self._on_decided = on_decided
        decision = await self._by_rule(inp, started)
        if decision is None:
            logger.info("調停：どの道にも当たらないので full（%s）", inp.origin)
            decision = _FALLBACK
        # 計測ログは層 3 が調停の待ち時間を調整するのに使う（出-ay 段 4-4b）。
        measure.record(
            "調停",
            秒=f"{time.monotonic() - started:.2f}",
            分岐=decision.branch,
            時間切れ="yes" if self._timed_out else "no",
        )
        return decision

    async def _by_rule(self, inp: ArbiterInput, started: float) -> "Decision | None":
        """起点ごとの道（出-ay 段 4）。当てはまらなければ None（`decide` が full にする）。"""
        self._timed_out = False
        if inp.origin == "機器" and not inp.tool_return and not inp.returned:
            return await self._device_by_rule(inp, started)
        if inp.returned:
            return await self._completion_by_rule(inp, started)
        if inp.origin == "情動" and inp.fired_axis in _AFFECT_ACTIONS:
            return await self._affect_by_rule(inp, started)
        if inp.origin == "発話" and not inp.tool_return:
            return await self._utterance_by_meaning(inp, started)
        return None

    async def _device_by_rule(self, inp: ArbiterInput, started: float) -> Decision:
        """機器の知らせは「軽く知らせる」と機械で決める（出-ay 段 4-1・2026-10-09・`設計方針_判定の段` §2.2.5）。

        機器の知らせ（タイマー・アラームが鳴った・音楽を 30 分で止めた）は 3 つとも軽く知らせると決めた（本人）。
        選択肢が 1 つなので Jev には聞かない。軽量LLM が一言を書けなければ、いまどおり full へ倒す。
        """
        texts = await _writer_call(self, inp, {"decided": "軽く知らせる"}, ["text"])
        text = str((texts or {}).get("text", "")).strip()
        decision = Decision(branch="light", text=text) if text else None
        logger.info(
            "調停 %.2f 秒（機器・機械で決めた：%s）",
            time.monotonic() - started,
            "軽く知らせる" if decision else "書けなかったので full",
        )
        return decision if decision is not None else _FALLBACK

    async def _completion_by_rule(self, inp: ArbiterInput, started: float) -> "Decision | None":
        """完了は機械で分け、選択肢が 1 つなら Jev に聞かない（出-ay 段 4-2・2026-10-09・`設計方針_判定の段` §2.2.5）。

        何が起きたかは `core/completion_kind.kind_of` が、最後に返った道具から決める。選択肢が 2 つ以上なら Jev に
        2 回目だけ聞き、**確信度に関係なく 1 番を使う**（本人：この領域では確信度を使わない）。表に無い道具は None
        （いままでの判定）。
        """
        from ..backends.jev import choice
        from ..core.completion_kind import kind_of
        from ..core.jev_judges import _ask

        action, failed, result = inp.returned[-1]
        got = kind_of(action, failed=failed, result=result, origin=inp.origin)
        if got is None:
            return None
        kind, actions = got
        final = actions[0]
        if len(actions) > 1:
            state = self._state(inp)
            questions = {
                "action": choice(
                    f"自分の動作の結果が届いた（{kind}）。パジュは次にどうするか",
                    {a: _COMPLETION_TEXT.get(a, _action_note(a) or a) for a in actions},
                )
            }
            answer = await _ask(self._jev, state, questions)
            picked = str(
                ((getattr(answer, "answers", None) or {}).get("action") or {}).get("choice") or ""
            )
            if not getattr(answer, "ok", False) or picked not in actions:
                logger.info("調停（完了・%s）：Jev が答えなかったので full", kind)
                self._keep(
                    inp, "完了", state, questions, answer, "reply_full", "Jev が答えなかった"
                )
                return _FALLBACK
            final = picked
            self._keep(inp, "完了", state, questions, answer, final, kind)
        decision = await self._completion_decision(inp, final, tool=action)
        logger.info(
            "調停 %.2f 秒（完了・機械で分けた：%s → %s）",
            time.monotonic() - started,
            kind,
            final if decision is not None else "書けなかったので full",
        )
        return decision if decision is not None else _FALLBACK

    async def _affect_by_rule(self, inp: ArbiterInput, started: float) -> Decision:
        """情動は発火した軸で決め、要るときだけ Jev に 2 回目を聞く（出-ay 段 4-3・2026-10-09・`設計方針_判定の段` §2.2.5）。

        seeking は調べに行くだけ。safety は見るか調べるか、bond は軽く／考えて話しかけるか、esteem はそれに調べるを
        足して Jev に聞く。**確信度は使わない**（本人）。Jev が答えない・書けなければ full。
        """
        from ..backends.jev import choice
        from ..core.jev_judges import _ask

        actions = _AFFECT_ACTIONS[inp.fired_axis]
        final = actions[0]
        if len(actions) > 1:
            state = self._state(inp)
            questions = {
                "action": choice(
                    f"自分の内から求めが起きた（{inp.fired_axis}）。パジュは次にどうするか",
                    {a: _COMPLETION_TEXT.get(a, _action_note(a) or a) for a in actions},
                )
            }
            answer = await _ask(self._jev, state, questions)
            picked = str(
                ((getattr(answer, "answers", None) or {}).get("action") or {}).get("choice") or ""
            )
            if not getattr(answer, "ok", False) or picked not in actions:
                logger.info("調停（情動・%s）：Jev が答えなかったので full", inp.fired_axis)
                self._keep(
                    inp, "情動", state, questions, answer, "reply_full", "Jev が答えなかった"
                )
                return _FALLBACK
            final = picked
            self._keep(inp, "情動", state, questions, answer, final, inp.fired_axis)
        decision = await self._completion_decision(inp, final, tool="")
        logger.info(
            "調停 %.2f 秒（情動・軸で決めた：%s → %s）",
            time.monotonic() - started,
            inp.fired_axis,
            final if decision is not None else "書けなかったので full",
        )
        return decision if decision is not None else _FALLBACK

    async def _utterance_by_meaning(self, inp: ArbiterInput, started: float) -> Decision:
        """発話は意味 → 動作を先読みの 1 回で聞く（出-ay 段 4-4b・2026-10-09・`設計方針_判定の段` §2.2.5）。

        問いと決まりは `core/utterance_meaning`。越えなければ「よく考えるか、軽く聞き返すか」をもう 1 回聞き、それも越え
        なければ軽く聞き返す。Jev が答えない・書けなければ full。黙る依頼・名乗り・否定・時期は段 4-4c で欄に戻す
        （それまでは軽く返す・本人：一時的に効かないのはかまわない）。
        """
        from ..core import utterance_meaning as um
        from ..core.jev_judges import _ask

        family = _family_call_names(inp.family_md)
        state = self._state(inp)
        # すすめた曲への返事を待っているときだけ、調停の候補に返事の道具が載る（段 4-4f）。
        suggesting = "music_suggestion_reply" in inp.extra_actions
        questions = um.fanout_questions(
            confirming="confirm" in inp.extra_actions,
            music=bool(_MUSIC_ACTIONS & set(inp.extra_actions)),
            camera=inp.can_see,
            family=family,
            suggesting=suggesting,
            speaker=inp.current_speaker,
            silenced=inp.silenced,
            tools=set(inp.extra_actions),
        )
        answer = await _ask(self._jev, state, questions)
        got = getattr(answer, "answers", None) or {}
        if not getattr(answer, "ok", False) or not got.get("meaning"):
            logger.info("調停（発話）：Jev が答えなかったので full")
            self._keep(inp, "発話", state, questions, answer, "reply_full", "Jev が答えなかった")
            self._tell_decided("reply_full")
            return _FALLBACK
        meaning = got["meaning"]
        m = str(meaning.get("choice") or "")
        offered = um.offered(
            confirming="confirm" in inp.extra_actions,
            music=bool(_MUSIC_ACTIONS & set(inp.extra_actions)),
            camera=inp.can_see,
            suggesting=suggesting,
        )
        if m in offered:
            outcome = um.decide(meaning, got.get(f"action_{m}"))
        else:
            # 並べていない意味（使えない道具）が返ってきたら使わない。首を回せないのに回す、などを防ぐ。
            outcome = um.Outcome("unsure", why=f"並べていない意味（{m}）")
        final = outcome.final
        asked, said = dict(questions), dict(got)
        if final == "unsure":
            again = await _ask(self._jev, state, um.unsure_question())
            final = um.resolve_unsure((getattr(again, "answers", None) or {}).get("action"))
            # 越えなかったときの問いと答えは「unsure」の下に残す（段 5d）。
            asked["unsure"] = um.unsure_question()
            said["unsure"] = dict(getattr(again, "answers", None) or {})
        self._keep(inp, "発話", state, asked, said, final, outcome.why, meaning=m)
        if final == "music_now":
            # いまの曲は機械が決まった形で答える（出-bf・本人の決定ア）。曲名を取り違えず、軽量LLM を待たない。
            from ..core.music_now import text_for

            self._tell_decided(final)
            status = None
            if inp.music_status is not None:
                try:
                    status = await inp.music_status()
                except Exception:  # noqa: BLE001
                    logger.warning("調停：音楽の様子を読めなかった", exc_info=True)
            logger.info(
                "調停 %.2f 秒（発話・意味で決めた：%s → %s）", time.monotonic() - started, m, final
            )
            return Decision(branch="light", text=text_for(status))
        if final in _SUGGESTION_REPLIES:
            self._tell_decided(final)
            # 返事は 2 つに決まっていて、書く言葉が無いので軽量LLM は呼ばない（段 4-4f）。
            reply = _SUGGESTION_REPLIES[final]
            logger.info(
                "調停 %.2f 秒（発話・意味で決めた：%s → %s）", time.monotonic() - started, m, final
            )
            return Decision(
                branch="action",
                action="music_suggestion_reply",
                query=reply,
                tool_input={"reply": reply},
            )
        # 黙る依頼・解く・名乗り・否定は、軽く返したうえで欄に入れる（段 4-4c）。
        fields: "dict[str, Any]" = {}
        if final == "quiet":
            minutes = str((got.get("quiet_minutes") or {}).get("choice") or "default")
            fields["silence_minutes"] = _QUIET_MINUTES.get(minutes, -1)
        elif final == "lift_quiet":
            fields["lift_silence"] = True
        elif final == "claim":
            fields["speaker_claim"] = outcome.name
        elif final == "deny":
            fields["not_person"] = outcome.name
        if final in ("quiet", "lift_quiet", "claim", "deny"):
            final = "reply_light"
        from ..core.timer_rules import is_control_word

        if final == "reply_light" and (
            needs_tools(inp.utterance) or (inp.timer_active and is_control_word(inp.utterance))
        ):
            # 軽く返すだけでは道具が動かない（「セットしました」と言うだけになる）。いままでの機械の守りを引き継ぐ。
            logger.info("調停（発話）：道具が要る頼みを軽く返さず full へ：%.30s", inp.utterance)
            final = "reply_full"
        if final in ("look", "see") and not inp.can_see:
            final = "reply_full"  # カメラが無いのに見ない
        if inp.capped and final not in ("silent", "reply_full") and final not in _LIGHT_WORDS:
            final = "reply_full"  # 上限に達した反復は調べさせずに閉じる
        effort = str((got.get("effort") or {}).get("choice") or "low")  # 考える深さ（本人の決定ウ）
        self._tell_decided(final)  # 軽量LLM が書く前に（出-bg）
        decision = await self._completion_decision(
            inp, final, tool="", effort=effort, refers_time=final == "recall"
        )
        if fields:
            decision = replace(decision if decision is not None else _FALLBACK, **fields)
        logger.info(
            "調停 %.2f 秒（発話・意味で決めた：%s → %s・%s）",
            time.monotonic() - started,
            m,
            final if decision is not None else "書けなかったので full",
            outcome.why,
        )
        return decision if decision is not None else _FALLBACK

    def _tell_decided(self, final: str) -> None:
        """発話の最終の動作が決まったことを 1 回だけ知らせる（出-bg）。知らせる側が落ちても調停は止めない。"""
        tell, self._on_decided = getattr(self, "_on_decided", None), None
        if tell is None:
            return
        try:
            tell(final)
        except Exception:  # noqa: BLE001
            logger.warning("調停：決まったことを知らせられなかった", exc_info=True)

    def _keep(
        self,
        inp: ArbiterInput,
        path: str,
        state: str,
        questions: dict,
        answer: Any,
        final: str,
        why: str,
        *,
        meaning: str = "",
    ) -> None:
        """Jev に聞いた回を記録に残す（出-ay 段 3・段 5d・本人の決定）。Jev に聞かなかった回は呼ばない（本人の決定イ）。

        正解（`JEV_正解.md`）と突き合わせてプロンプトを直すため、渡したそのまま（W を含む）・問い・答え・道・発話の意味・
        最終の動作を残す。書けなくても調停は止めない（`arbiter_records.record` が別スレッドで書き、失敗は警告 1 行）。
        """
        from ..store import arbiter_records

        answers = (
            answer if isinstance(answer, dict) else dict(getattr(answer, "answers", None) or {})
        )
        arbiter_records.record(
            origin=inp.origin,
            utterance=inp.utterance,
            state=state,
            questions=questions,
            answer=answers,
            outcome=why,
            path=path,
            meaning=meaning,
            final=final,
        )

    async def _completion_decision(
        self,
        inp: ArbiterInput,
        final: str,
        *,
        tool: str,
        effort: str = "low",
        refers_time: bool = False,
    ) -> "Decision | None":
        """完了の最終の動作を `Decision` に写す。黙る→light・文なし、軽く…→light・軽量LLM の一言、考えて返す→full、道具→action。"""
        if final == "silent":
            return Decision(branch="light", text="")
        if final in ("reply_full", "talk_full"):
            return Decision(branch="full", effort=effort if effort in _EFFORTS else "low")
        if final in _LIGHT_WORDS:
            texts = await _writer_call(self, inp, {"decided": _LIGHT_WORDS[final]}, ["text"])
            text = str((texts or {}).get("text", "")).strip()
            return Decision(branch="light", text=text) if text else None
        data: "dict[str, Any]" = {"branch": "action", "action": final, "effort": "low"}
        if refers_time:
            data["refers_time"] = (
                True  # recall の語と一緒に時期も書かせる（指していなければ空・段 4-4c）
            )
        texts = await self._write(inp, data)
        if texts is None:
            return None
        # 返った道具は候補から外されている（`_extra_actions(exclude=returned)`）。かけ直す・続けて見るために足す。
        return assemble(
            {**data, **texts},
            can_see=inp.can_see or final in ("look", "see") or tool in ("look", "see"),
            origin=inp.origin,
            extra_actions=(*inp.extra_actions, final),
        )

    async def write_filler(self, inp: ArbiterInput, waiting: str) -> str:
        """待たせているあいだの一言だけを書かせる（出-aq 段 6）。**分岐は決めない**——つなぎを出すのは
        待たせている事実であって、Jev の分岐ではない。以前は分岐しだいで、light なら返事（内容に触れる）を
        言い、full で深さが low なら何も書かれず黙った。書けなければ空（黙る）。`trash` は読み捨てる。
        """
        texts = await _writer_call(
            self, inp, {"decided": f"待ってもらう一言を言う（{waiting}）"}, ["filler", "trash"]
        )
        return str((texts or {}).get("filler", "")).strip()

    async def _write(self, inp: ArbiterInput, data: dict) -> "dict | None":
        """要るものだけを軽量LLM に 1 回で書かせる（出-au 段 5-7c）。何も要らなければ呼ばずに空。"""
        needs = _writer_needs(inp, data)
        if not needs:
            return {}
        return await _writer_call(self, inp, data, needs)


#: 道具（タイマーまわり）。入力は `tool_input` で渡し、0.1 秒で返るのでつなぎは言わない（実機 08:59）。
_TOOL_ACTIONS = frozenset(
    {
        "set_timer",
        "start_stopwatch",
        "stop_stopwatch",
        "cancel_timer",
        "pause_timer",
        "resume_timer",
        "set_alarm",
        "cancel_alarm",
        "play_music",
        "stop_music",
        "next_track",
        "music_volume",
    }
)
_NO_WORDS_ACTIONS = frozenset({"see", "confirm", "decline"})
_LOOK_INPUT = (
    'tool_input に {"pose":"定点の名前"} か {"direction":"右|左|上|下"}。'
    "定点の名前は必ず pose に入れる（direction は右・左・上・下だけ）。"
    "自分から見回るなら、[いま] の見ていない順で最も長く見ていない定点へ"
)

#: 文章を書く口の指示（出-au 段 5-7c）。判定はもう決まっている。口調と言い直さない決まりは、いままでの調停の文から。
WRITER_PROMPT = """\
これは口に出す言葉ではなく、自分の中の決めごとである。挨拶や説明はせず、指定の JSON だけを返す。

{lead}
次にすることはもう決まっている：{decided}

**text と filler の口調は、はじめに渡された【あなたは誰か】と【一緒に暮らす人たち】に従う。** 相手が大人か子どもかで
丁寧さが変わる。**短い一言でも同じ**で、短さのために丁寧さを崩さない。一つのやり取りの中で丁寧さを混ぜない。

**すでに相手へ伝えた一言があるなら、言い直さず、その続きとして書く。** 一言目はいま考えている・調べている最中だと伝えるものだが、
**二言目以降は、まだ考えている最中だと伝わるだけの短い言葉**にする。用件を述べ直さない。何を調べているかにも触れない。
長さは一言目より短く、多くても十数文字にとどめる。同じ人が続けて言っているように聞こえることを最優先する。
言うことが無ければ filler を空にしてよい。

[いま]
{now}

[いま誰が居るか]
{present}

{heading}
{utterance}

[いまの作業状態]
{workspace}

[書くもの]
{fields}

次の形の JSON だけを返す（他には何も書かない）:
{shape}
"""

_FIELD_TEXT = {
    "text": '"text"：この人格として、この相手に向けて、いまの時刻に合う言葉で短く答える。',
    "text_talk": '"text"：相手に話しかける短い一言。相手が驚かないよう丁寧に、短く。**まず話してよいかを尋ねる一言にする**。'
    "いつも通りなら空にして黙る。",
    "text_self": '"text"：自分から言うなら短いひとこと。**いつも通りなら空にして黙る**。返事や約束の形にしない。',
    "filler": '"filler"：待ってもらうための短い一言（相槌・受けだけ。**内容に触れない**。答えを先取りしない）。',
    "trash": '"trash"：用件はこのあと本応答が言う。いま言いたいことがあるならここに書く（捨てられ、誰にも届かない）。',
    "query": '"query"：探す語。',
    # 担い手（知-am・2026-10-07）。書けなかったので、調停が投げる検索はいつも既定の Brave（リンクの一覧）だった。
    "source": '"source"：探す担い手。答えに数字や事実（天気・時刻表・値段・結果など）を使うなら "tavily"（本文の抜粋）、'
    'どこに何があるかを探す・話題を広く見るなら "brave"（題名・短い抜粋・リンクの一覧）。',
    "tool_input": '"tool_input"：道具へそのまま渡す入力（JSON の辞書）。',
    "time_ref": '"time_ref"：人の言葉が指している時期を ISO 8601（例 "2025-08-15T00:00:00"）で。指していなければ空。',
    "time_span_days": '"time_span_days"：その言い方が指す幅を日数で（広い言い方ほど大きい）。',
}


#: 先導文の最初の一文は Jev に選ばせる言い方。書く側はもう選ばないので、そこだけ差し替える（出-au 段 5-7d）。
_CHOOSING = ("次のどれかを選ぶ。", "何をするかを決める。")
_WRITING = "決まったことに要る言葉を書く。"


def _for_writer(lead: str) -> str:
    for phrase in _CHOOSING:
        lead = lead.replace(phrase, _WRITING)
    return lead


def _writer_needs(inp: ArbiterInput, data: dict) -> "list[str]":
    """Jev の答えから、軽量LLM に書かせるものを決める。何も要らなければ空。

    **つなぎは書かせない**（出-aq 段 7）。つなぎは答えまで 5 秒かかったときだけ、待ちの知らせの口
    （`write_filler`）が書く。以前は full で深さが low でなければ、action では投げる前に書かせていた。
    """
    needs: "list[str]" = []
    branch = data.get("branch")
    if branch == "light":
        needs.append("text")
    elif branch == "action":
        action = str(data.get("action", ""))
        fixed = action in _EXTRA_ACTIONS and bool(_EXTRA_ACTIONS[action][0])
        if action == "look" or action in _TOOL_ACTIONS:
            needs.append("tool_input")
        elif action not in _NO_WORDS_ACTIONS and not fixed:
            needs.append("query")
        if action == "search_deferred":
            needs.append("source")
    if data.get("refers_time"):
        needs += ["time_ref", "time_span_days"]
    return needs


def _action_note(action: str) -> str:
    if action == "look":
        return f"首を向ける（{_LOOK_INPUT}）"
    if action in _EXTRA_ACTIONS:
        return _EXTRA_ACTIONS[action][1]
    return _BASE_ACTIONS.get(action, "")


def _text_field(inp: ArbiterInput) -> str:
    """情動の求めの返事の書き方。話しかける軸なら話しかけ、ほかは自分のひとこと（出-at）。"""
    return "text_talk" if inp.talking else "text_self"


async def _writer_call(arbiter: "Arbiter", inp: ArbiterInput, data: dict, needs: "list[str]"):
    """軽量LLM を 1 回呼ぶ。時間切れ・失敗は None（呼び手が full へ倒す）。"""
    from ..core.context_parts import Stance, build_context
    from ..core.structured_ask import read_json

    self_doing = inp.origin == "情動"
    if inp.tool_return:
        lead, heading = _LEAD_TOOL_RETURN, _HEADING_TOOL_RETURN
    elif self_doing:
        lead, heading = _LEAD_SELF, _HEADING_SELF
    elif inp.origin == "機器":
        lead, heading = _LEAD_DEVICE, _HEADING_DEVICE
    else:
        lead, heading = _LEAD_REPLY, _HEADING_REPLY
    decided = str(
        data.get("decided") or data.get("branch")
    )  # 待ちの一言は分岐を持たない（出-aq 段 6）
    if data.get("branch") == "action":
        decided += f"（{data.get('action')}：{_action_note(str(data.get('action')))}）"
    fields = "\n".join(
        _FIELD_TEXT[_text_field(inp) if (n == "text" and self_doing) else n] for n in needs
    )
    shape = "{" + ", ".join(f'"{n}": …' for n in needs) + "}"
    prompt = WRITER_PROMPT.format(
        lead=_for_writer(lead),
        decided=decided,
        now=inp.now_ctx or "（分からない）",
        present=inp.present_ctx or "（分からない）",
        heading=heading,
        utterance=inp.utterance,
        workspace=inp.workspace_ctx or "（なし）",
        fields=fields,
        shape=shape,
    )
    system = build_context(
        stance=Stance.PAJU,
        self_understanding=inp.self_understanding or "（指定なし）",
        family=inp.family_md or "（指定なし）",
        people=inp.people_md,
        family_now=inp.family_now,
        self_image=inp.self_image,
        season_env=inp.season_env,
    ).stable
    timeout = arbiter._timeout
    if timeout is None:
        from ..config import AgentConfig

        timeout = AgentConfig().arbiter_timeout_sec
    started = time.monotonic()
    call = asyncio.ensure_future(arbiter._writer.complete(prompt, 300, system=system))
    try:
        reply = await wait_within(asyncio.shield(call), timeout)
    except asyncio.TimeoutError:
        logger.warning("調停の文章が %.1f 秒で返らなかった", timeout)
        arbiter._timed_out = True
        _watch_late(call, started, len(prompt))
        return None
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.warning("調停の文章を書けなかった: %s", e)
        return None
    got = read_json(str(reply or ""))
    if not isinstance(got, dict):
        logger.warning("調停の文章を読めなかった: %.200r", reply)
        return None
    logger.info("調停の文章 %.2f 秒（%s）", time.monotonic() - started, "・".join(needs))
    return {k: got[k] for k in needs if k in got}
