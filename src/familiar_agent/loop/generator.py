"""生成器 GEN：反復ごとに、主LLM へ渡す材料を組む（環-e-に）。

第一弾は**ループの可変状態を1つも触らない部分**だけを置く。`InformationProcessing` は
38 個の可変状態を持ち、そのうち 20 個は書き手が3つ以上ある。振る舞いで割ると同じ状態を
複数のファイルが書くことになるので、**状態を触らないところから出す**
（`モジュール分割設計` の環-e-に・Functional core / Imperative shell）。

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
            if agent._persons.active_is_explicit:
                declared = agent._persons.active_name
        if declared:
            return (
                f'(present :speaker "{declared}" '
                ':note "顔は確認できていない。名前は自己申告による")'
            )
        # 誰も認識できていない。直近に話しかけられているなら、相手は居るが誰かは不明。
        recently_spoken = False
        with contextlib.suppress(Exception):
            recently_spoken = agent._social_presence_permission() > 0.0
        if recently_spoken:
            return '(present :speaker "unconfirmed" :note "顔は確認できていないが直近に話しかけられた")'
        return '(present :none true :note "誰も確認できていない")'

    def _one(row: dict) -> str:
        conf = row.get("confidence")
        conf_s = f" :confidence {float(conf):.2f}" if conf is not None else ""
        return f'"{row.get("name", "unknown")}"{conf_s}'

    speaker = next((r for r in rows if r.get("is_speaker")), None)
    others = [r for r in rows if not r.get("is_speaker")]
    parts = ["(present"]
    parts.append(f" :speaker {_one(speaker)}" if speaker else ' :speaker "unconfirmed"')
    if others:
        parts.append(" :others " + " ".join(_one(r) for r in others))
    return "".join(parts) + ")"


def _pi_ctx() -> str:
    """mood/drive を PI として定性注入する（生値は出さない）。DB 失敗は空で degrade。"""
    try:
        from ..config import DriveConfig
        from ..core.drive_autonomy import drive_snapshot
        from ..drive_register import load_current_drives
        from ..emotion_pad import label_from_pad
        from ..mood_register import load_current_mood

        mood = load_current_mood()
        drives = load_current_drives()
        return f"[内部状態(PI)] 気分: {label_from_pad(mood)} / 欲求: {drive_snapshot(drives, DriveConfig())}"
    except Exception as e:  # noqa: BLE001
        logger.debug("PI ctx unavailable: %s", e)
        return ""
