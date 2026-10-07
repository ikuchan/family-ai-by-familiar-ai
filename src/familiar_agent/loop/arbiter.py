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

from ..core import measure
from ..core.aio import wait_within

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
    "知らせの中身を伝える、必要なら見る、それだけでよい。"
    "タイマーは音でも知らせている（鳴っている）ので、聞かれない限り黙っていてよい。"
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
        "order は言われたときだけ。[音楽] の枠が「いまは何も鳴っていない」なら、前にかけたことがあってもかける）",
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


_THINKING_NOTE = """
この件で答えを組み立てるのは {round} 回目である。回を重ねても材料が増えていないなら、
それはもう分からないということなので、**これ以上は調べず**（"action" を選ばず）、
いまある材料で答えるほうへ回す。
"""

_CAPPED_NOTE = """
これ以上は調べられない（反復の上限に達した）。"action" は選べない。いまある材料で答える
ことになるので "light" か "full" を選ぶ。
"""


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


#: 分岐の決め方の目安（出-au 段 5-7d・一つの軽量LLM の指示文にあったものを、Jev に送る文へ移した）。
JUDGE_GUIDE = """\
迷ったら full を選ぶ。
分かれ目は、結果が届いたかどうかではなく、いまある材料が問いに答えるに足るかどうかである。
- 材料が無い → action（調べる）
- 材料は届いたが答えきれず、まだ試していない角度がある → action（別の語で調べる）
- 材料が問いに答えるに足る → full（答える）
- 調べたが答えが得られず、試せる角度も無い → full（分からないと伝える）
すでに調べた語と同じ語では投げない。同じ語なら結果も同じで、繰り返しても何も増えない。分からないまま探し続けるより、
分からないと言うほうがよい。作業状態に並ぶ自分のしたこと（何を・どうやって調べ、何が届いたか）を読み、同じことを
重ねて投げない（別のことを調べるのは構わない）。
自分が覚えているはずのこと（家族の出来事・過去の会話）は recall。
世の中のこと（天気・ニュース・調べもの）でも、作業状態に同じことについての結果や自分の答えが、時刻から見て十分新しい
形であるなら、それで答える（full か light）。無いとき・古いときだけ search_deferred。古いかどうかは各行の時刻から
判断する（天気なら数時間、ニュースならその日のうち、が目安）。"""
#: 見る動作があるときの目安。写真は Jev に渡らないので、読み取りの記録（「見えたもの」）で決める（出-au 段 5-7a）。
SEE_GUIDE = (
    "作業状態の『見えたもの』の行（写真の読み取り）で答えられるなら light でよい。"
    "細かく語る・写真を見て判断する必要があるなら full（写真そのものは主LLM に渡る）。"
    "首を向けた帰り（作業状態に『…のほうを向いた』）なら、首はもう向いている——もう一度 look は選ばない。"
)

#: 動作の短い説明（Jev の選択肢）。繋がっている MCP の道具は `_EXTRA_ACTIONS` の説明から作る。
_BASE_ACTIONS = {
    "recall": "自分の記憶（家族の出来事・過去の会話）を探す",
    "search_deferred": "インターネットで世の中のこと（天気・ニュース・調べもの）を調べる",
}
_CAMERA_ACTION_TEXT = {
    "see": "目の前を見る（カメラ）。見えているものを聞かれた・部屋の様子を確かめる",
    "look": "首を向ける（カメラを回す）。「右向いて」「窓の方見て」「もっと右」。"
    "自分から見回るなら、[いま] の見ていない順で最も長く見ていない定点へ",
}
_BRANCH_REPLY = {
    "light": "何も動かさず、短い言葉で答えきれる（挨拶・相槌・簡単な受け答え）。道具が要る頼みは含まない",
    "full": "何も動かさずに答えるが、記憶を踏まえた言葉選びや込み入った説明が要る。調べたが分からないと伝えるときもこれ",
    "action": "答える前に何かを動かす（見る・首を向ける・タイマー・予定やメモを見る・記憶を探す・調べる）。"
    "材料が無い、またはまだ試していない角度がある",
}
_BRANCH_SELF = {
    "light": "短くひとこと言う。黙るのが基本（いつも通りなら黙る）",
    "full": "考えてから言う・する",
    "action": "見る・調べる・首を向ける",
}
_EFFORT_CRITERIA = {
    "low": "ふつう。ほとんどの場合",
    "medium": "ひと言で表せない複雑な気持ちを受け止める、4 つ以上の記憶を踏まえて応える、調べた結果をまとめる",
    "high": "人がよく考えるよう明示的に求めた",
}
_QUIET_MINUTES = {"default": -1, "5": 5, "10": 10, "15": 15, "30": 30, "60": 60}


def _ranks(got: dict) -> str:
    """1 番・2 番の選択肢と確率（「1 番 action 0.55・2 番 full 0.31」）。確率が無ければ空。"""
    probs = sorted(
        ((str(k), float(v)) for k, v in (got.get("probabilities") or {}).items()),
        key=lambda kv: -kv[1],
    )
    return "・".join(f"{i + 1} 番 {k} {v:.2f}" for i, (k, v) in enumerate(probs[:2]))


def _judgement_line(answer: Any) -> str:
    """Jev の判定の 1 行（出-ay 段 2）：分岐と動作を、それぞれ選んだもの・確信度・1 番と 2 番つきで。

    分岐で倒れると、Jev が同じ 1 回の問いで選んでいた動作を読まずに捨てていた。倒れた 40 回のうち action が 1 番の
    16 回で、どの動作を選んでいたかが分からなかった（2026-10-08 の集計）。倒れても倒れなくても、毎回残す。
    """
    answers = getattr(answer, "answers", None) or {}
    parts = []
    for label, key in (("分岐", "branch"), ("動作", "action")):
        got = answers.get(key) or {}
        if not got:
            continue
        conf = float(got.get("confidence", 0.0) or 0.0)
        ranks = _ranks(got)
        parts.append(
            f"{label} {got.get('choice') or '—'}（確信度 {conf:.2f}"
            + (f"・{ranks}" if ranks else "")
            + "）"
        )
    return "・".join(parts)


def _unsure(label: str, answer: Any, key: str, min_conf: float) -> str:
    """確信度が足りなくて倒れた理由の 1 行（出-ay）：確信度と、1 番・2 番の選択肢と確率。"""
    got = (getattr(answer, "answers", None) or {}).get(key) or {}
    conf = float(got.get("confidence", 0.0) or 0.0)
    ranks = _ranks(got)
    if not ranks and got.get("choice"):
        ranks = f"1 番 {got.get('choice')}"
    return f"{label}の確信度 {conf:.2f}＜{min_conf:g}" + (f"（{ranks}）" if ranks else "")


def _extra_action_text(action: str) -> str:
    """`_EXTRA_ACTIONS` の説明（`"x"（…）`）から、括弧の中の最初の一文を取る。"""
    desc = _EXTRA_ACTIONS[action][1]
    m = re.search(r"（(.*?)[。）]", desc)
    return (m.group(1) if m else desc).strip()


def _family_choices(family_md: str) -> "dict[str, str]":
    """家族の呼びかけの名前 → 呼び方の一覧（名乗り・打ち消しの選択肢）。"""
    from ..core import parsing
    from ..core.speaker_claim import aliases_of, call_name_of

    out: "dict[str, str]" = {}
    for m in parsing.parse_family_md(family_md or ""):
        key = call_name_of(m)
        if key:
            out[key] = "、".join(aliases_of(m))
    return out


class Arbiter:
    """調停。材料を受け取り、**Jev が決め**、要るときだけ**軽量LLM が書き**、機械の守りを通して `Decision` を返す。

    判定と文章を 1 回の軽量LLM の指示文に同居させていた形を割った（出-au 段 5-7）。Jev は写真を見られない
    ので、写真の読み取りは状態として W に載っている（段 5-7a）。
    """

    def __init__(
        self, *, jev, writer, min_conf: float = 0.6, timeout: "float | None" = None
    ) -> None:
        self._jev = jev
        self._writer = writer
        self._min_conf = float(min_conf)
        self._timeout = timeout
        #: 文章の口が時間切れになったか（計測ログの「時間切れ」・層 3 が調停の秒数を見直す材料）
        self._timed_out = False
        #: 判定なしで倒れた理由（出-ay・2026-10-07）。倒れなければ空。本文は入れない。
        self._why = ""

    def _allowed_actions(self, inp: ArbiterInput) -> "dict[str, str]":
        out = dict(_BASE_ACTIONS)
        if inp.can_see:
            out.update(_CAMERA_ACTION_TEXT)
        for a in inp.extra_actions:
            if a in _EXTRA_ACTIONS:
                out[a] = _extra_action_text(a)
        return out

    def _state(self, inp: ArbiterInput) -> str:
        """Jev に送る文。先導文（起点ごと）・いま・顔ぶれ・その場の言葉・作業状態（W 全部・本人の決定 イ）。"""
        self_doing = inp.origin == "情動"
        if inp.tool_return:
            lead, heading = _LEAD_TOOL_RETURN, _HEADING_TOOL_RETURN
        elif self_doing:
            lead, heading = _LEAD_SELF, _HEADING_SELF
        elif inp.origin == "機器":
            lead, heading = _LEAD_DEVICE, _HEADING_DEVICE
        else:
            lead, heading = _LEAD_REPLY, _HEADING_REPLY
        notes = (_CAPPED_NOTE if inp.capped else "") + (
            _THINKING_NOTE.format(round=inp.thinking_round) if inp.thinking_round > 1 else ""
        )
        guide = JUDGE_GUIDE + ("\n" + SEE_GUIDE if inp.can_see else "")
        return (
            f"[決めること]\n{lead}\n{notes}\n"
            f"[判断の目安]\n{guide}\n\n"
            f"[いま]\n{inp.now_ctx or '（分からない）'}\n\n"
            f"[いま誰が居るか]\n{inp.present_ctx or '（分からない）'}\n\n"
            f"{heading}\n{inp.utterance}\n\n"
            f"[いまの作業状態]\n{inp.workspace_ctx or '（なし）'}"
        )

    def _questions(self, inp: ArbiterInput) -> dict:
        from ..backends.jev import choice, noul

        branches = dict(_BRANCH_SELF if inp.origin == "情動" else _BRANCH_REPLY)
        if inp.capped:
            branches.pop("action")  # これ以上は調べられない
        qs: dict = {
            "branch": choice("次にどうするか", branches),
            "effort": choice("考えて答えるなら、どれくらい深く考えるべきか", _EFFORT_CRITERIA),
            "action": choice("先に動くなら、どの動作か", self._allowed_actions(inp)),
            "refers_time": noul("人の言葉が、特定の過去の時期（去年の夏・先週など）を指している"),
        }
        if inp.tool_return:
            return qs  # 道具の帰りの発話は古い。黙る依頼・名乗りは読まない（情-n）
        qs["asks_quiet"] = noul(
            # 名前の関門は入口の窓（出-as §2.5）。Jev は名前を知らないので、ここでは問わない（出-au 段 5-7d）
            "いまは話しかけないでほしいと頼んでいる"
            "（待って・待てぃは動作を止めてほしいだけで、これには当たらない）"
        )
        qs["quiet_minutes"] = choice(
            "黙っていてほしい長さ",
            {
                "default": "長さを言っていない",
                "5": "5 分くらい",
                "10": "10 分くらい",
                "15": "15 分くらい",
                "30": "30 分くらい",
                "60": "1 時間くらい",
            },
        )
        if inp.silenced:
            qs["lifts_quiet"] = noul("黙っているところへ、もう話していい・しゃべっていいと解いた")
        family = _family_choices(inp.family_md)
        qs["claims"] = noul(
            "人が自分の名前を名乗った（「〜だよ」「〜です」。誰かを呼んだだけ・第三者の話は違う）"
        )
        qs["claimed"] = choice(
            "名乗った名前は家族の誰か", {**family, "other": "家族以外・分からない"}
        )
        denied = dict(family)
        if inp.current_speaker and inp.current_speaker not in denied:
            denied[inp.current_speaker] = "いま話者としている人"
        qs["denies"] = noul(
            "人が、自分はいま呼ばれている名前の人ではないと打ち消した（「〜じゃないよ」「ちがう」）"
        )
        qs["denied"] = choice(
            "打ち消された呼び方はどれか（名前を言わなかったなら、いま話者としている人）",
            {**denied, "other": "分からない"} if denied else {"other": "分からない"},
        )
        return qs

    async def decide(self, inp: ArbiterInput) -> Decision:
        """次の一手を決める。Jev が決め、要るときだけ軽量LLM が書き、守りを通す。倒れたら full。"""
        from ..core.timer_rules import is_control_word

        started = time.monotonic()
        data = await self._judge(inp)
        texts = None if data is None else await self._write(inp, data)
        decision = None
        if data is not None and texts is not None:
            decision = assemble(
                {**data, **texts},
                can_see=inp.can_see,
                origin=inp.origin,
                extra_actions=inp.extra_actions,
            )
        if decision is None:
            logger.warning(
                "調停を決められなかったのでフルへ倒す（判定=%s%s）",
                "あり" if data else "なし",
                f"・{self._why}" if not data and self._why else "",
            )
        elif decision.branch == "light" and inp.origin == "発話" and not inp.tool_return:
            if needs_tools(inp.utterance):
                # light は道具を使えない。「セットしました」と言うだけになるので full へ倒す（機械の守り）。
                logger.info("調停 light を full へ倒す（道具が要る頼み）：%.30s", inp.utterance)
                decision = replace(decision, branch="full", effort="low")
            elif inp.timer_active and is_control_word(inp.utterance):
                # 操作の言葉を light で受け流すと何も起きない（出-aa）。主LLM が道具で決める。
                logger.info(
                    "調停 light を full へ倒す（タイマーの操作の言葉）：%.30s", inp.utterance
                )
                decision = replace(decision, branch="full", effort="low")
        seconds = time.monotonic() - started
        logger.info("調停 %.2f 秒（分岐=%s）", seconds, (decision or _FALLBACK).branch)
        measure.record(
            "調停",
            秒=f"{seconds:.2f}",
            分岐=(decision or _FALLBACK).branch,
            時間切れ="yes" if self._timed_out else "no",
        )
        return decision if decision is not None else _FALLBACK

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

    async def _judge(self, inp: ArbiterInput) -> "dict | None":
        """Jev に 1 回で聞き、`_parse` の守りへ渡す辞書を返す。分岐か動作の確信度が低い・使えないときは None。"""
        from ..core.jev_judges import _ask, picked

        self._why = ""
        answer = await _ask(self._jev, self._state(inp), self._questions(inp))
        if not getattr(answer, "ok", False):
            err = str(getattr(answer, "error", "") or "") if answer is not None else "使えない"
            self._why = f"Jev が答えなかった（{err or '理由なし'}）"
            return None
        logger.info("調停の判定：%s", _judgement_line(answer))
        branch = picked(answer, "branch", self._min_conf)
        if branch not in ("light", "full", "action"):
            self._why = _unsure("分岐", answer, "branch", self._min_conf)
            return None
        data: dict = {"branch": branch}
        data["effort"] = picked(answer, "effort", 0.0) or "low"
        if branch == "action":
            action = picked(answer, "action", self._min_conf)
            if action is None:
                self._why = _unsure("動作", answer, "action", self._min_conf)
                return None
            if action not in self._allowed_actions(inp):
                self._why = f"動作 {action} は候補に無い"
                return None
            data["action"] = action
        got = answer.answers

        def yes(key: str) -> bool:
            return float((got.get(key) or {}).get("noul", 0.0) or 0.0) >= 0.5

        data["refers_time"] = yes("refers_time")
        if inp.tool_return:
            # 道具の帰りの発話は古い。問うていないが、答えに混じっても黙る依頼・名乗りは読まない（機械の守り・情-n）
            return data
        data["silence_minutes"] = (
            _QUIET_MINUTES.get(picked(answer, "quiet_minutes", 0.0) or "default", -1)
            if yes("asks_quiet")
            else 0
        )
        data["lift_silence"] = yes("lifts_quiet")
        claimed = picked(answer, "claimed", 0.0) if yes("claims") else None
        data["speaker_claim"] = claimed if claimed and claimed != "other" else ""
        denied = picked(answer, "denied", 0.0) if yes("denies") else None
        data["not_person"] = denied if denied and denied != "other" else ""
        return data


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
    'tool_input に {"direction":"右|左|上|下"} か {"pose":"定点の名前"}。'
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
    "time_ref": '"time_ref"：人の言葉が指している時期を ISO 8601（例 "2025-08-15T00:00:00"）で。',
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
