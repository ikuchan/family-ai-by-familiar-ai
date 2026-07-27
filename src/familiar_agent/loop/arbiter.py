"""調停器（ARB）：軽量LLM が会話の重さを自己判断し、反復の出し方を3つへ振り分ける。

正本＝`I内部設計根拠` 段4。**基準は作り込まず軽量LLM の自己判断**とし、調整はプロンプトで
行う（閾値や点数式を置かない）。実測では1ターン 10.5 秒のうち LLM が 10.2 秒を占め、
`recall` を投げるだけの反復にもフルLLM を同期で使っていた。

- **light**：短文で答えきれる会話は軽量LLM が応答して反復を閉じる（フルを起こさない）。
- **full** ：熟慮・想起が要る会話はフルLLM を起こす。**思考の深さ（effort）も軽量LLM が決める**。
- **action**：探すと決まっている反復は、軽量LLM が**どの動作で・何を**調べるかを決めて投げ、
  反復を閉じる。**つなぎの発話（「調べてみるね」）も一緒に返させる**：フルLLM を経由すると
  実測 2.9 秒かかるところ、調停だけなら 0.7 秒で反応が返る（正本③ 段5 の「内部二段」を
  action 分岐へ当てたもの）。

判定できないとき・時間切れは **full／effort=high** へ倒す（従来と同じ挙動＝退行しない）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_EFFORTS = ("low", "medium", "high")

ARBITER_PROMPT = """\
あなたは対話エージェントの内部で、次の一手を選ぶ調停器である。会話ではないので、
挨拶や説明はせず、指定の JSON だけを返す。

いま人から届いた言葉と、いまの作業状態を見て、次のどれかを選ぶ。

- "light"  : 短い言葉で答えきれる。挨拶、相槌、簡単な受け答え。あなたが text に応答を書く。
- "full"   : 記憶を踏まえた言葉選びや、込み入った説明が要る。生成は別の大きなモデルが行う。
             どれくらい深く考えるべきかを effort に "low" / "medium" / "high" で書く。
             effort が "low" でないなら、待ってもらうための短い一言を text に書く
             （相槌・受けだけ。**内容に触れない**。答えを先取りすると本応答と食い違う）。
- "action" : いまある材料では答えきれず、先に調べる。どうやって調べるかを action に書く。
             "recall"（自分の記憶を探す）か "search_deferred"（インターネットを調べる）。
             探す語を query に、待ってもらうための短い一言を text に書く
             （これから調べると伝えるだけ。**内容に触れない**）。

**text の口調は、下の【あなたは誰か】と【一緒に暮らす人たち】に従う。** 相手が大人か
子どもかで丁寧さが変わる。**短い一言でも同じ**で、短さのために丁寧さを崩さない。実機では
大人（ですます）に対し、本応答はですますなのに待ってもらう一言だけタメ口になった。
一つのやり取りの中で丁寧さを混ぜない。


判断の基準は自分で決めてよい。迷ったら "full" を選ぶ。

**分かれ目は、結果が届いたかどうかではなく、いまある材料が問いに答えるに足るかどうかである。**

- 材料が無い　　　　　　　　　　　　　→ "action"（調べる）
- 材料は届いたが、問いに答えきれない　→ "action"（別の角度・別の語で調べ直す）
- 材料が問いに答えるに足る　　　　　　→ "full"（答える）

作業状態には、あなた自身がいましたこと（何を・どうやって調べ、何が届いたか、
相手に何と言ったか）が記録として並んでいる。それを読み、同じことを重ねて投げない
（別のことを調べるのは構わない）。

**すでに相手へ伝えた一言があるなら、言い直さず、その続きとして書く。** 一言目はこれから
調べると伝えるものだが、**二言目以降は、まだ考えている最中だと伝わるだけの短い言葉**に
する。用件を述べ直さない。何を調べているかにも触れない。長さは一言目より短く、多くても
十数文字にとどめる。同じ人が続けて言っているように聞こえることを最優先する。
言うことが無ければ text を空にしてよい。
自分が覚えているはずのこと（家族の出来事・過去の会話）は "recall"、
世の中のこと（天気・ニュース・調べもの）は "search_deferred" を選ぶ。
{capped_note}
[あなたは誰か]
{me}

[一緒に暮らす人たち]
{family}

[いま]
{now}

[いま誰が居るか]
{present}

text を書くときは、この人格として、この相手に向けて、いまの時刻に合う言葉で書く。

[人の言葉]
{utterance}

[いまの作業状態]
{workspace}

相手が**いまは話しかけないでほしい**と伝えているなら、`silence` に true を書く。言い方は
一つではない（うるさい、あとにして、いま集中したい、静かにして…）。頼まれたと読めるかで
判断する。true にすると、その人が居るあいだ発話を止める。頼まれてもいないのに止めない。

人の言葉が**時期を指している**なら、想起の基準をそこへ動かす。`time_ref` にその時刻を
ISO 8601（例 "2025-08-15T00:00:00"）で、`time_span_days` にその言い方が指す**幅**を日数で
書く。幅はその言い方がどれくらいの粗さで時期を指しているかで、広い言い方ほど大きい。
時期を指していないなら両方とも省く（基準は現在時刻になる）。

次の形の JSON だけを返す（他には何も書かない）:
{{"branch": "light|full|action", "text": "…", "effort": "low|medium|high",
 "action": "recall|search_deferred", "query": "…", "silence": false,
 "time_ref": "", "time_span_days": 0}}
使わない項目は省いてよい。
"""


@dataclass
class Decision:
    """調停の結果。`branch` 以外はその分岐でだけ意味を持つ。"""

    branch: str           # light | full | action
    text: str = ""        # light：発話／action：つなぎの一言
    effort: str = "high"  # full：思考の深さ
    action: str = "recall"  # action：どの動作で調べるか
    query: str = ""       # action：探す語
    silence: bool = False  # 相手が「いまは話しかけないで」と伝えている
    # 想起の時間軸の基準。人の言葉が時期を指しているとき（「去年の夏の話」）に動かす。
    # 既定（None）は「いま」が基準・幅は Config の既定（3日）。
    time_ref: str = ""            # ISO 8601（例 "2025-08-15T00:00:00"）
    time_span_days: float = 0.0   # 幅＝半減期（日）。0 は指定なし


_FALLBACK = Decision(branch="full", effort="high")


def _parse(reply: str) -> Decision | None:
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
    effort = str(data.get("effort", "high")).strip().lower()
    if effort not in _EFFORTS:
        effort = "high"
    text = str(data.get("text", "")).strip()
    silence = bool(data.get("silence", False))
    time_ref = str(data.get("time_ref", "") or "").strip()
    try:
        time_span_days = float(data.get("time_span_days", 0) or 0)
    except (TypeError, ValueError):
        time_span_days = 0.0
    query = str(data.get("query", "")).strip()
    action = str(data.get("action", "")).strip() or "recall"
    if action not in ("recall", "search_deferred", "fetch_deferred"):
        action = "recall"
    # 分岐に必要なものが無ければ判定できていない＝倒す。
    if branch == "light" and not text:
        return None
    if branch == "action" and not query:
        return None
    return Decision(branch=branch, text=text, effort=effort, action=action, query=query,
                    silence=silence, time_ref=time_ref, time_span_days=max(0.0, time_span_days))


_CAPPED_NOTE = """
これ以上は調べられない（反復の上限に達した）。"action" は選べない。いまある材料で答える
ことになるので "light" か "full" を選ぶ。
"""


async def arbitrate(backend, *, utterance: str, workspace_ctx: str,
                    self_understanding: str = "", family_md: str = "",
                    present_ctx: str = "", now_ctx: str = "",
                    capped: bool = False, timeout: float = 2.0) -> Decision:
    """軽量LLM に次の一手を選ばせる。失敗・時間切れは full へ倒す。

    **発話の出口は2つ**（ここの light とつなぎ、フルLLM の答え）なので、**フルと同じ
    土台を渡す**。片方にだけ渡すと、症状が出るたび1つずつ足すことになる（人格を足した
    翌日、14時39分に「こんばんは」と言った＝日時が無かった）。

    - `self_understanding`：自己認識1枚（人格＋できること）。**何ができるかを知らずに
      何をするかは選べない**ので、動作を選ぶこの器にこそ要る。
    - `family_md`：誰が大人で誰が子どもかは家族の記述にしかなく、口調の規則に要る。
    - `present_ctx`／`now_ctx`：誰に向けて・いつ話すか。
    - `capped`：反復上限。渡さないと上限でも "action" を選び、その判断が丸ごと捨てられる。
    """
    prompt = ARBITER_PROMPT.format(
        utterance=utterance,
        workspace=workspace_ctx or "（なし）",
        me=self_understanding or "（指定なし）",
        family=family_md or "（指定なし）",
        present=present_ctx or "（分からない）",
        now=now_ctx or "（分からない）",
        capped_note=_CAPPED_NOTE if capped else "",
    )
    try:
        reply = await asyncio.wait_for(backend.complete(prompt, 300), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("調停が %.1f 秒で返らなかったのでフルへ倒す", timeout)
        return _FALLBACK
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.warning("調停に失敗したのでフルへ倒す: %s", e)
        return _FALLBACK
    decision = _parse(reply)
    if decision is None:
        logger.warning("調停の返事を読めなかったのでフルへ倒す: %.80r", reply)
        return _FALLBACK
    return decision
