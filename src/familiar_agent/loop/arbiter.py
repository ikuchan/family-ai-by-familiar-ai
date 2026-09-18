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
from dataclasses import dataclass

from ..core import measure

logger = logging.getLogger(__name__)

_EFFORTS = ("low", "medium", "high")

# 並びは**キャッシュが前方一致で効く**ことに合わせる。起動中ほぼ変わらないもの（人格・家族・
# 規則）を先に置き、変わるもの（上限の但し書き・時刻・在席・人の言葉・作業状態）を後ろへ。
# 実機で調停が 2 秒で返らず時間切れになり、沈黙依頼が読まれないまま倒れた。
#: 安定部（立ち位置＋人格＋家族）は**システム文で渡す**。1本の文字列だったのは
#: `complete()` にシステム文の口が無かったからで、出-e-い でその口を作った。分けると
#: 「安定を先、可変を後」を人が守る必要が無くなり（システム文は常に先）、呼び出し間で
#: 一字一句同じなので前方一致キャッシュが最大限効く。
#: 課題の指示は可変の data と同じプロンプト側に置く。**交互に並ぶ問題はここで消える**
#: ——「口調の注意」が `[いま]` の直後にあるのも、JSON の指示が最後にあるのも、
#: どちらもプロンプト側の話になる。
ARBITER_PROMPT = """\
これは口に出す言葉ではなく、自分の中の決めごとである。挨拶や説明はせず、指定の
JSON だけを返す。

{lead}
{branches}

**text の口調は、はじめに渡された【あなたは誰か】と【一緒に暮らす人たち】に従う。** 相手が大人か
子どもかで丁寧さが変わる。**短い一言でも同じ**で、短さのために丁寧さを崩さない。実機では
大人（ですます）に対し、本応答はですますなのに待ってもらう一言だけタメ口になった。
一つのやり取りの中で丁寧さを混ぜない。


判断の基準は自分で決めてよい。迷ったら "full" を選ぶ。

**分かれ目は、結果が届いたかどうかではなく、いまある材料が問いに答えるに足るかどうかである。**

- 材料が無い　　　　　　　　　　　　　　　　　　　→ "action"（調べる）
- 材料は届いたが答えきれず、**まだ試していない角度がある** → "action"（別の語で調べる）
- 材料が問いに答えるに足る　　　　　　　　　　　　→ "full"（答える）
- **調べたが答えが得られず、試せる角度も無い　　　→ "full"（分からないと伝える）**

**すでに調べた語と同じ語では投げない。** 同じ語なら結果も同じで、繰り返しても何も増えない。
角度を変えられないなら、それはもう分からないということである。**分からないまま探し続ける
より、分からないと言うほうがよい。**

作業状態には、あなた自身がいましたこと（何を・どうやって調べ、何が届いたか、
相手に何と言ったか）が記録として並んでいる。それを読み、同じことを重ねて投げない
（別のことを調べるのは構わない）。

**すでに相手へ伝えた一言があるなら、言い直さず、その続きとして書く。** 一言目はこれから
調べると伝えるものだが、**二言目以降は、まだ考えている最中だと伝わるだけの短い言葉**に
する。用件を述べ直さない。何を調べているかにも触れない。長さは一言目より短く、多くても
十数文字にとどめる。同じ人が続けて言っているように聞こえることを最優先する。
言うことが無ければ text を空にしてよい。
自分が覚えているはずのこと（家族の出来事・過去の会話）は "recall"。
世の中のこと（天気・ニュース・調べもの）でも、**作業状態に同じことについての結果や自分の答えが、
時刻から見て十分新しい形であるなら、それで答える**（"full" か "light"）。無いとき・古いときだけ
"search_deferred"。古いかどうかは各行の時刻から自分で判断する（天気なら数時間、ニュースなら
その日のうち、が目安）。
[いま]
{now}

[いま誰が居るか]
{present}

text を書くときは、この人格として、この相手に向けて、いまの時刻に合う言葉で書く。

{capped_note}{thinking_note}{heading}
{utterance}

[いまの作業状態]
{workspace}

**はじめに渡された [あなたは誰か] に書かれた自分の名前で呼ばれたうえで**、いまは話しかけないでほしいと
伝えられたときだけ、`silence_minutes` に分数を書く。周囲の会話が紛れ込むので、名前を
呼ばれていない依頼は、自分に向けられたものとして扱わない。

- 分数を伴って頼まれた → その分数
- 長さを言わずに頼まれた → **-1**（既定の長さを当てる）
- 頼まれていない、または名前で呼ばれていない → **0**

言い方は一つではない（うるさい、あとにして、いま集中したい、静かにして…）。ただし
待つよう言われただけ（待って・待てぃ）は**動作を止めてほしいだけで沈黙の依頼ではない**（0）。
黙っていたところへ（[いま] に *黙っているよう頼まれている* とあるとき）、もう話していい・しゃべっていいと
**解かれた**ら `lift_silence` を true にする。頼まれたと
読めるかで判断する。0 以外にすると、その人が居るあいだ発話を止める。頼まれてもいないのに
止めない。

人の言葉が**時期を指している**なら、想起の基準をそこへ動かす。`time_ref` にその時刻を
ISO 8601（例 "2025-08-15T00:00:00"）で、`time_span_days` にその言い方が指す**幅**を日数で
書く。幅はその言い方がどれくらいの粗さで時期を指しているかで、広い言い方ほど大きい。
時期を指していないなら両方とも省く（基準は現在時刻になる）。

次の形の JSON だけを返す（他には何も書かない）:
{{"branch": "light|full|action", "text": "…", "effort": "low|medium|high",
 "action": "{actions}", "query": "…", "silence_minutes": 0, "lift_silence": false,
 "time_ref": "", "time_span_days": 0}}
使わない項目は省いてよい。
"""


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
# 文は実験（`scripts/experiment_arbiter_confirm.py` の `LEAD_TOOL_RETURN_2`）のまま。
_LEAD_TOOL_RETURN = (
    "いま**道具から返りが届いた**（作業状態の最上部）。人に届いた言葉ではない。返りを見て、次のどれかを選ぶ。"
    '返りが人に伝える文（「掛けた」「確かめて：「…」」「動いている」）なら、その文を **"light"** でそのまま、'
    "または相手に合わせて言い換えて伝える。返りが「確かめて」なら聞き返すだけでよい——**掛け直しはあなたの仕事ではない**"
    '（「いい」と言われたら別の仕組みが掛ける）。指示があいまいで確かめたいことがあれば、それも "light" で聞く。'
    '別の道具が要るときだけ "action"。'
    "直近のやりとりに「うまくできなかった」「まだ掛かっていない」があっても、**取り返そうとして道具を選ばない**。"
    "道具は既に返っている（最上部）。取り返すのは、返りをきちんと伝えることで足りる。"
    "同じ失敗が続いていれば、そのことを一言添えてよい。"
)
_HEADING_TOOL_RETURN = "[道具から返ったもの]"
_BRANCHES_REPLY = """\
- "light"  : 短い言葉で答えきれる。挨拶、相槌、簡単な受け答え。あなたが text に応答を書く。
             **道具が要る頼み（タイマー・アラーム・測る・止める・覚えて・予定を見る）は light で答えない**
             ——light は道具を使えず、「セットしました」と言っても何も起きない。full に回す。
- "full"   : 記憶を踏まえた言葉選びや、込み入った説明が要る。生成は別の大きなモデルが行う。
             どれくらい深く考えるべきかを effort に書く。**既定は "low"**。
             "medium" は次の3つのときだけ：(1) ひと言で表せない複雑な気持ちを受け止める
             (2) 4 つ以上の記憶を踏まえて応える (3) 調べた結果をまとめる。
             "high" は、人がよく考えるよう**明示的に**求めたときだけ。
             effort が "low" でないなら、待ってもらうための短い一言を text に書く
             （相槌・受けだけ。**内容に触れない**。答えを先取りすると本応答と食い違う）。
- "action" : いまある材料では答えきれず、先に調べる。どうやって調べるかを action に書く。
             "recall"（自分の記憶を探す）か "search_deferred"（インターネットを調べる）{see_option}。
             探す語を query に、待ってもらうための短い一言を text に書く
             （これから調べると伝えるだけ。**内容に触れない**）。{see_note}"""
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
_BRANCHES_SELF = """\
- "light"  : 短くひとこと言う。**黙るのが基本**（text を空にすれば黙る）。言うなら あなたが text に書く。
- "full"   : 考えてから言う・する。生成は別の大きなモデルが行う。effort は "low"。text は空。
- "action" : 見る・調べる・首を向ける。どうやってかを action に書く。
             "recall"（自分の記憶を探す）か "search_deferred"（インターネットを調べる）{see_option}。
             探す語を query に。**text は空**（誰にも断らない）。{see_note}"""
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
    "start_stopwatch": (
        "",
        '"start_stopwatch"（「今から測って」。tool_input に {"label": "何を"}）',
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
_SEE_OPTION = (
    'か "see"（目の前を見る＝カメラ。見えているものを聞かれた・部屋の様子を確かめる必要があるとき。query は要らない）'
    # 首振り（2026-09-16 実機）：「右見れる?」「もっと右を見て」に `see` しか選べず、正面のまま
    # 「右側はタンスと椅子が見えていますよ」と答えた。`look` は主LLM の道具にしか無かった。
    'か "look"（首を向ける＝カメラを回す。「右向いて」「窓の方見て」「もっと右」のとき。'
    'tool_input に {"direction":"右|左|上|下"} か {"pose":"定点の名前"}。向いた先の写真が帰る。query は要らない・**text は空**。'
    "自分から見回るなら、[いま] の見ていない順で最も長く見ていない定点へ）"
)
#: `look` の見出し（query）。(c) 分岐は query が空だと full へ落ちるので、固定の語を入れる。
LOOK_QUERY = "首を向ける"
#: 見た印のラベルはローカルの人検出（YOLO・80 種・1 枚 8 ms）が出す。棚や引き出しは無く、
#: 机が dining table になる粗さだが、「何が見える？」に「椅子とテーブル」と返すには足りる。
#: 写真そのものは主LLM にだけ渡るので、細かく語るなら full（v0.45）。
#: 写真を添えたときの但し書き（v0.46）。調停が自分で見に行った帰りだけ。
_SEE_NOTE_WITH_PHOTO = (
    "\n             **写真を添えた**（いま見えているもの）。見えているものを聞かれただけなら、"
    '写真を見て "light" に答えてよい。込み入った説明や記憶を踏まえる必要があるなら "full"'
    "（主LLM にも同じ写真が渡る）。首を向けた帰り（作業状態に『…のほうを向いた』）なら、"
    '首はもう向いている——もう一度 "look" は選ばず、向いたことを "light" で一言伝えてよい。'
)
#: 自分から見に行った帰り（情動・写真つき）の但し書き。返事の場面の注記（「聞かれただけなら
#: 答えてよい」）をここへ渡すと、誰にも聞かれていないのに返事の体裁の一言を作った（2026-09-16
#: 実機・「はい、静かにしていますね」「お仕事中ですね、静かにしていますから」）。
_SEE_NOTE_OWN_LOOK = (
    "\n             **写真を添えた**（いま見えているもの）。これは自分が見に行った帰りで、"
    "誰にも聞かれていない。**いつも通りなら text を空にして黙る**。言うのは、様子が変わった・"
    "気づいたことがあるときだけ。返事や約束の形（「静かにしています」「頑張ってください」）にしない。"
    '込み入ったことなら "full"（主LLM にも同じ写真が渡る）。'
)
_SEE_NOTE = (
    "\n             作業状態の『わたしが見た』の行は即席のラベル（写っている物の名前・80 種の粗さ）。"
    '見えているものを聞かれただけなら、**そのラベルで "light" に答えてよい**。'
    '細かく語る・写真を見て判断する必要があるなら "full"（写真そのものは主LLM にだけ渡る）。'
)


def _parse(
    reply: str, *, can_see: bool = False, origin: str = "発話", extra_actions: tuple[str, ...] = ()
) -> Decision | None:
    """軽量LLM の返事から JSON を拾う。前後に地の文が混じっても拾えるようにする。"""
    match = re.search(r"\{.*\}", reply or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except Exception:
        return None
    branch = str(data.get("branch", "")).strip().lower()
    if branch not in ("light", "full", "action"):
        return None
    effort = str(data.get("effort", "low")).strip().lower()
    if effort not in _EFFORTS:
        effort = "low"
    text = str(data.get("text", "")).strip()
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
    elif action == "family_schedule" and not query:
        query = "1"  # 日数を書き忘れても action は落とさない（今日だけ・主LLM が呼び直せる）
    if action in ("set_timer", "start_stopwatch") and tool_input and set(tool_input) == {"id"}:
        action = "cancel_timer"  # 入力が id だけなら止める意図（「ストップ」に set_timer と書いた・実機 08:59）
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


async def arbitrate(
    backend,
    *,
    utterance: str,
    workspace_ctx: str,
    self_understanding: str = "",
    family_md: str = "",
    self_image: str = "",
    present_ctx: str = "",
    now_ctx: str = "",
    capped: bool = False,
    thinking_round: int = 1,
    timeout: float | None = None,
    can_see: bool = False,
    image_b64: str | None = None,
    origin: str = "発話",
    extra_actions: tuple[str, ...] = (),
    tool_return: bool = False,
) -> Decision:
    """軽量LLM に次の一手を選ばせる。失敗・時間切れは full へ倒す。

    **発話の出口は2つ**（ここの light とつなぎ、フルLLM の答え）なので、**フルと同じ
    土台を渡す**。片方にだけ渡すと、症状が出るたび1つずつ足すことになる（人格を足した
    翌日、14時39分に「こんばんは」と言った＝日時が無かった）。

    - `self_understanding`：自己認識1枚（人格＋できること）。**何ができるかを知らずに
      何をするかは選べない**ので、動作を選ぶこの器にこそ要る。
    - `family_md`：誰が大人で誰が子どもかは家族の記述にしかなく、口調の規則に要る。
    - `present_ctx`／`now_ctx`：誰に向けて・いつ話すか。
    - `capped`：反復上限。渡さないと上限でも "action" を選び、その判断が丸ごと捨てられる。
    - `thinking_round`：この求めで主LLM を呼ぶのが何回目か。主LLM を投げっぱなしにして
      から反復の数は返りで 0 へ戻るようになったので、**反復の数からは求めの長さが読めない**。
      だから回数そのものを渡す（同じ値をログと主LLM のプロンプトへも渡している）。
      2 回目以降だけ載せる（1 回目に「1 回目である」と言っても何も足さない）。
    - `timeout`：省略すると Config（`ARBITER_TIMEOUT_SEC`・既定 5.0 秒）から取る。
    - `can_see`：カメラがあるか。あるときだけ `see` を候補に載せる（無い構成で選ばせて
      空振りさせない）。帰りの判断は出した側に返る（`event_loop._decide`）。
    - `extra_actions`：いま繋がっている MCP の同期の道具（`house_rules`／`family_schedule`）。
      あるときだけ候補に載せる。query は要らない（見出しは固定）。
    - `origin`：求めの起点（`発話`／`機器`／`情動`）。`情動` なら「返事」でなく「自分の行動を
      決める」型のプロンプトにする（見出し `[いま湧いたこと]`・許可も理由も要らない・つなぎ無し）。
    - 直近のやりとりは `workspace_ctx` の先頭の枠として入っている（記-h）。主LLM と同じ
      作り方で、窓の幅（`recent_exchanges_arbiter`）だけが狭い。無いと「明日の天気は？」の
      次の「調べて」を新しい検索にする（2026-09-13 実機）。
    - `image_b64`：調停が自分で見に行った帰りの写真（v0.46）。即席のラベルは部屋によって
      `bench` 1 語になり材料不足で full へ倒れたので、写真そのものを見せて light で答えられる
      ようにする。担い手が写真を受けられなければ（`complete_with_image` 無し）文字だけで進む。
    """
    from ..core.context_parts import Stance, build_context

    # 安定はシステム文へ、課題の指示と可変の data はプロンプトへ（出-e-に）。
    system = build_context(
        stance=Stance.PAJU,
        self_understanding=self_understanding or "（指定なし）",
        family=family_md or "（指定なし）",
        self_image=self_image,  # 層 2・主LLM と同じもの（記-a-へ）
    ).stable
    self_doing = origin == "情動"
    device = origin == "機器"
    lead = _LEAD_DEVICE if device else _LEAD_REPLY
    heading = _HEADING_DEVICE if device else _HEADING_REPLY
    if tool_return:  # 道具（タイマー・アラーム）の帰り。起点が何であれ、いま見るのは返り
        lead, heading = _LEAD_TOOL_RETURN, _HEADING_TOOL_RETURN
    prompt = ARBITER_PROMPT.format(
        lead=_LEAD_SELF if self_doing else lead,
        branches=(_BRANCHES_SELF if self_doing else _BRANCHES_REPLY).format(
            see_option=(_SEE_OPTION if can_see else "")
            + "".join(f"か {_EXTRA_ACTIONS[a][1]}" for a in extra_actions if a in _EXTRA_ACTIONS),
            see_note=(
                (
                    (_SEE_NOTE_OWN_LOOK if self_doing else _SEE_NOTE_WITH_PHOTO)
                    if image_b64
                    else _SEE_NOTE
                )
                if can_see
                else ""
            ),
        ),
        heading=_HEADING_SELF if self_doing else heading,
        utterance=utterance,
        workspace=workspace_ctx or "（なし）",
        present=present_ctx or "（分からない）",
        now=now_ctx or "（分からない）",
        capped_note=_CAPPED_NOTE if capped else "",
        thinking_note=(_THINKING_NOTE.format(round=thinking_round) if thinking_round > 1 else ""),
        actions="|".join(
            ["recall", "search_deferred"]
            + (["see", "look"] if can_see else [])
            + [a for a in extra_actions if a in _EXTRA_ACTIONS]
        ),
    )
    if timeout is None:
        from ..config import AgentConfig

        timeout = AgentConfig().arbiter_timeout_sec
    started = time.monotonic()
    # 打ち切っても呼び出し自体は残す（shield）。倒す時刻は変えずに、**実際に何秒かかるか**を
    # 裏で測るため。時間切れの秒数しか残らないと、2.1 秒なのか 10 秒なのか分からず、
    # 時間切れの値を決められない（実機で「黙って」だけが 2 秒に掛かった）。
    if image_b64 and hasattr(backend, "complete_with_image"):
        call = asyncio.ensure_future(
            backend.complete_with_image(prompt, image_b64, 300, system=system)
        )
    else:
        call = asyncio.ensure_future(backend.complete(prompt, 300, system=system))
    try:
        reply = await asyncio.wait_for(asyncio.shield(call), timeout=timeout)
        logger.info("調停 %.2f 秒（プロンプト %d 字）", time.monotonic() - started, len(prompt))
    except asyncio.TimeoutError:
        logger.warning("調停が %.1f 秒で返らなかったのでフルへ倒す", timeout)
        _watch_late(call, started, len(prompt))
        # 層 3 の材料（`arbiter_timeout_sec`・記-a-に）。
        measure.record("調停", 秒=f"{time.monotonic() - started:.2f}", 分岐="full", 時間切れ="yes")
        return _FALLBACK
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.warning("調停に失敗したのでフルへ倒す: %s", e)
        return _FALLBACK
    decision = _parse(reply, can_see=can_see, origin=origin, extra_actions=extra_actions)
    if decision is None:
        logger.warning("調停の返事を読めなかったのでフルへ倒す: %.300r", reply)
    elif decision.branch == "light" and origin == "発話" and needs_tools(utterance):
        # light は道具を使えない。「セットしました」と言うだけになるので full へ倒す（機械の守り）。
        logger.info("調停 light を full へ倒す（道具が要る頼み）：%.30s", utterance)
        decision = Decision(
            branch="full",
            effort="low",
            text=decision.text,
            silence_minutes=decision.silence_minutes,  # 「話すの止めて」の依頼は落とさない
            lift_silence=decision.lift_silence,
        )
    measure.record(
        "調停",
        秒=f"{time.monotonic() - started:.2f}",
        分岐=(decision or _FALLBACK).branch,
        時間切れ="no",
    )
    return decision if decision is not None else _FALLBACK
