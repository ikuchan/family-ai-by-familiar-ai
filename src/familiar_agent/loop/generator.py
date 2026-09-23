"""生成器 GEN：反復ごとに、主LLM へ渡す材料を組む（環-e-に）。

ここに置くのは**ループの可変状態を1つも触らない部分**だけである。切り出しを始めた時点で
`InformationProcessing` は 38 個の可変状態を持ち、そのうち 20 個は書き手が3つ以上あった
（2026-09-08 の実測）。振る舞いで割ると同じ状態を複数のファイルが書くことになるので、
**状態を触らないところから出す**（`モジュール分割設計` の環-e-に・Functional core /
Imperative shell）。

その後、求めの寿命を `loop/request.py` へ、W を `loop/workspace.py` へ出したので、
いまは **17 個・書き手3つ以上は 5 個**である（2026-09-10）。

**挙動は変えない。** 中身も名前も動かさず、置き場所だけを移した。名前の頭の `_` は
移す前のままである（改名は別の作業として立てる・環-e の守り）。

生成器の本体（主LLM を呼んで出力を決める部分）はまだ `event_loop.py` にある。
"""

from __future__ import annotations

import contextlib
import logging

logger = logging.getLogger(__name__)


def _present_ctx(agent) -> str:
    """いま誰が居るかを渡す。「誰かが居る」ではなく「誰が居るか」を伝える。

    自発発話は誰に向けたものかで内容が変わるので、名前と確信度を添える。誰も認識できて
    いないときも黙らず、その事実を明示する（空文字だと、宛先が分からないまま話すことに
    なる）。

    **暫定である点**：ここで扱えるのは既知の人物だけで、「顔は見えるが誰か分からない
    未知の人」を表せない。PMM の在席は InsightFace が埋める identity であり、設計が定める
    presence（在/不在）とは別物である。**未知の在席者の扱いは残課題 #8**（在席系の精緻化）。
    """
    pmm = getattr(agent, "_pmm", None)
    rows: list = []
    if pmm is not None:
        try:
            rows = pmm.presence_status()
        except Exception:  # noqa: BLE001
            rows = []

    if not rows:
        # 顔では誰も特定できていない。次は自己申告（`/speaker`・`[名前]`）を見る。カメラの
        # 無い CUI では話者はここにしか現れず、これを読まないと相手が誰でも「分からない」に
        # 倒れ、口調が丁寧語だけに固定される（実機で観測）。顔で確かめた話者とは由来が違う
        # ので、そのことを添えて渡す（#8 で身元と在席を分けるときにこの区別が要る）。
        declared = ""
        with contextlib.suppress(Exception):
            # 分かっているときだけ（`/speaker` から 60 秒・返事から 60 秒・顔照合／知-t）。
            # 切れていれば名前を出さず「誰かは不明」へ——誰か分からない相手を名前で呼ばない。
            if agent._persons.active_is_explicit and agent.speaker_known():
                declared = agent._persons.active_name
        if declared:
            return (
                f'(present :speaker "{declared}" '
                ':note "顔は確認できていない。名前は自己申告による")'
            )
        # 顔でも自己申告でも分からない。**居るか**は在/不在の層（YOLO）が別に知っている
        # （知-h）。人は居るが誰かは不明、として渡す——「誰も確認できていない」と言うと、
        # 目の前の人に向けて話す判断ができない。
        with contextlib.suppress(Exception):
            sensor = getattr(agent, "_presence_sensor", None)
            if sensor is not None and sensor.room_occupied() is True:
                return (
                    '(present :speaker "unconfirmed" :note "誰か居るが、誰かは分からない。'
                    '名前で呼ばない（直近のやりとりの名前も当てにしない）。知りたければ聞いてよい")'
                )
        # 直近に話しかけられているなら、相手は居るが誰かは不明。
        recently_spoken = False
        with contextlib.suppress(Exception):
            recently_spoken = agent._social_presence_permission() > 0.0
        if recently_spoken:
            return (
                '(present :speaker "unconfirmed" :note "直近に話しかけられたが、誰かは分からない。'
                '名前で呼ばない。知りたければ聞いてよい")'
            )
        return '(present :none true :note "誰も確認できていない")'

    def _one(row: dict) -> str:
        conf = row.get("confidence")
        conf_s = f" :confidence {float(conf):.2f}" if conf is not None else ""
        return f'"{row.get("name", "unknown")}"{conf_s}'

    speaker = next((r for r in rows if r.get("is_speaker")), None)
    others = [r for r in rows if not r.get("is_speaker")]
    parts = ["(present"]
    parts.append(f" :speaker {_one(speaker)}" if speaker else ' :speaker "unconfirmed"')
    # 相手が大人かを添える（出-ak）。口調は `ME.md` が決めるが、**どちらの行を当てるか**は
    # ここが分かっていないと決まらない。書いていない人には添えない（決めつけない）。
    if speaker:
        from ..core.tone import is_adult

        with contextlib.suppress(Exception):
            if is_adult(
                str(speaker.get("name") or ""), str(getattr(agent, "_family_md", "") or "")
            ):
                parts.append(' :note "この相手は大人"')
    if others:
        parts.append(" :others " + " ".join(_one(r) for r in others))
    # 名前の分からない在席者しか居ないなら、注記は在席表が空のときと同じものが要る
    # （出-ae(1)）。注記はもともと**在席表が空**の経路にしか無かったので、名前の無い
    # 在席者を持てるようにしたとたん、表が空でなくなって注記が落ちる（出-ae-は）。
    if all(r.get("person_id") is None for r in rows):
        parts.append(
            ' :note "誰か居るが、誰かは分からない。名前で呼ばない'
            '（直近のやりとりの名前も当てにしない）。知りたければ聞いてよい"'
        )
    return "".join(parts) + ")"


def _iter_ctx(
    *,
    chain: int,
    max_chain: int,
    thinking_round: int,
    capped: bool,
    budget=None,
    missing: "list[str] | None" = None,
    tone: str = "",
) -> str:
    """この反復がどこに居るかを、主LLM へ渡す1行に組む。

    材料は数と真偽だけで、**ループの可変状態を1つも読まない**（に-5-は）。

    - `chain`／`max_chain`：**決める反復**の何回目か（環-h ⑤）。主LLM の返りで 0 へ戻るので、
      この数は求めの長さを表さない
    - `thinking_round`：この求めで主LLM を呼ぶのが何回目か。**求めの長さを表すのはこちら**
      である（環-h ⑥-2）。反復のリセットで手がかりが消えたので、回数そのものを渡す
    - `capped`：上限では、黙って手持ちで繕わず「調べきれなかった」と断ってから答えさせる。
      断りが無いと、材料不足のまま答えたことが相手に伝わらない。あわせて**「した」と言える基準**
      を渡す（出-z・実機 2026-09-18 18:31）：上限で答えさせた主LLM が、W に「まだ掛けていない」
      と並ぶ中で「1 分のタイマーかけたよ」と言った。嘘のつもりは無く、文脈から補った。
      「嘘をつかない」は基準を与えないので、道具の返りだけを根拠にさせる
    - `budget`：返事の予算（`reply_budget.ReplyBudget`）。長さは規則の文言でなく**数字で**
      渡す（出-k-ろ）。無ければ行を足さない
    """
    text = f"[反復] {chain}/{max_chain}（この件を考えるのは {thinking_round} 回目）"
    if budget is not None:
        # 口調（出-ak）。**`ME.md` の行をそのまま**足す——機械は写しを持たない。
        # 字数だけの行に付けるのは、`[返事]` が「いまのこの返事」についての指示だからで、
        # 規則の側（`personality-from-me`）に書いても効かなかった（実測 0/12）。
        text = budget.line() + (f"（{tone}）" if tone else "") + "\n" + text
    if missing:
        # いま使えない道具（出-al）。**繋がっていない道具は候補から黙って消える**ので、
        # 無いという事実がどこにも残らず、「調べたけど出てこない」を繰り返していた。
        # 1 行置くだけで伝わる（実機の場面で 8 回中 0 回 → 8 回・2026-09-22）。言い聞かせる
        # 文は足さない——足しても 8 回のままだった。
        text = "[いま使えない] " + "・".join(missing) + "\n" + text
    if capped:
        text += (
            "（これ以上は調べられない。調べきりたかったが上限に達したことを述べ、"
            "そのうえで現時点で分かることを返す。「した」と言えるのは、道具の返りにそう書いてあるときだけ。"
            "返りに無いことは、していない。できていないなら、できていないと言う）"
        )
    return text


def _pi_ctx(req) -> str:
    """mood/drive を言葉にして PI として渡す（情-f）。生の数値はプロンプトに出さず、計測ログへ。

    気分は 4 軸 × 5 段（軸ごとの実測分位）、欲求は発火した軸を明示し、ほかは p70 超だけ弱く
    （`core/inner_state`）。境目は `InnerStateConfig`（層 3 の設定値・DB > 既定）。以前の
    12 点表への最近傍 1 語と固定閾値の低・中・高は、実測で情報量 0 だった（15/15 が neutral）。
    DB 失敗は空で degrade。
    """
    try:
        from ..config import InnerStateConfig
        from ..core import inner_state, measure
        from ..drive_register import load_current_drives
        from ..mood_register import load_current_mood

        mood = load_current_mood()
        drives = load_current_drives()
        cfg = InnerStateConfig()
        bands = inner_state.MoodBands(p=cfg.mood_p, pn=cfg.mood_pn, a=cfg.mood_a, dom=cfg.mood_dom)
        p70 = inner_state.DriveP70(
            seeking=cfg.drive_p70_seeking,
            rest=cfg.drive_p70_rest,
            bond=cfg.drive_p70_bond,
            safety=cfg.drive_p70_safety,
            esteem=cfg.drive_p70_esteem,
        )
        fired = getattr(req, "fired_axis", "") or ""
        # 数値は計測ログにだけ（REST 内省が境目を等頻度に合わせ直す材料・記-i）。
        if mood is not None:
            measure.record(
                "気分",
                P=f"{mood.p:.2f}",
                Pn=f"{mood.pn:.2f}",
                A=f"{mood.a:.2f}",
                Dom=f"{mood.dom:.2f}",
            )
        measure.record(
            "欲求",
            SEEKING=f"{drives.seeking:.2f}",
            REST=f"{drives.rest:.2f}",
            BOND=f"{drives.bond:.2f}",
            SAFETY=f"{drives.safety:.2f}",
            ESTEEM=f"{drives.esteem:.2f}",
            発火=fired or "-",
        )
        return inner_state.pi_line(mood, drives, fired=fired, bands=bands, p70=p70)
    except Exception as e:  # noqa: BLE001
        logger.debug("PI ctx unavailable: %s", e)
        return ""
