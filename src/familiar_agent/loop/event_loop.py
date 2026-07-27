"""イベント駆動ループ（#11 段階1）：I（情報処理機構）の最小縦切り。

設計正本＝`設計図_Mermaid` ③ I 詳細図。ここでは I の中の **LPM（ループ核）** と
**QC（完了キュー）** だけを実体化する。反復は QC を drain（取込→O 書込）→ REC（想起→W）→
GEN（生成）で進み、say で1出力して終わる／内部ツール（recall）は結果を QC へ積んで次反復へ
連鎖する（[D-単一想起]：相関ID を使わず結果は O→W 経由で再会）。

現行 run() と排他（`EVENT_LOOP` on の user turn のみ）。AIF/DIF/QA/QD、ARB/APR/ACT/MNT の
クラス分離は後続段階（ここでは stub しない）。永続化は既存 `_run_post_response_pipeline`
（utility LLM のみ）を流用し、消化した完了 O はターン観察 id で supersede する。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from datetime import datetime

from ..store import clock
from .arbiter import arbitrate
from .prompt import build_event_system_prompt

logger = logging.getLogger(__name__)

# 連鎖が続けられる反復で渡す動作。上限に達した反復では say だけにして必ず閉じる。
_FULL_ACTIONS = ("say", "recall", "search_deferred", "fetch_deferred")
# 調べる動作＝結果が後の反復に届くもの。投げたらその反復は終わる。
_LOOKUP_ACTIONS = ("recall", "search_deferred", "fetch_deferred")


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
            return (f'(present :speaker "{declared}" '
                    ':note "顔は確認できていない。名前は自己申告による")')
        # 誰も認識できていない。直近に話しかけられているなら、相手は居るが誰かは不明。
        recently_spoken = False
        with contextlib.suppress(Exception):
            recently_spoken = agent._social_presence_permission() > 0.0
        if recently_spoken:
            return '(present :speaker "unconfirmed" :note "顔は確認できていないが直近に話しかけられた")'
        return '(present :none true :note "誰も確認できていない")'

    def _one(row: dict) -> str:
        conf = row.get("confidence")
        conf_s = f' :confidence {float(conf):.2f}' if conf is not None else ""
        return f'"{row.get("name", "unknown")}"{conf_s}'

    speaker = next((r for r in rows if r.get("is_speaker")), None)
    others = [r for r in rows if not r.get("is_speaker")]
    parts = ["(present"]
    parts.append(f' :speaker {_one(speaker)}' if speaker else ' :speaker "unconfirmed"')
    if others:
        parts.append(" :others " + " ".join(_one(r) for r in others))
    return "".join(parts) + ")"


def _when(created_at, now_epoch: float) -> str:
    """いつのことかを「経過時間（時刻）」で書く。片方だけでは足りない。"""
    with contextlib.suppress(Exception):
        stamp = created_at.timestamp()
        hours = (now_epoch - stamp) / 3600.0
        ago = f"{int(hours * 60)}分前" if hours < 1 else f"約{int(hours)}時間前"
        return f"{ago}（{created_at.astimezone().strftime('%m/%d %H:%M')}）"
    return "いつか"


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


class InformationProcessing:
    """I：情報処理機構（Information-processing）。③ I 詳細図の器。

    段階1で実体化するのは **QC（完了キュー）** と **LPM（ループ核）＝`run_iteration`** のみ。
    O・C（Config）・W・RH 相当のツール実行は既存実体を持つ `agent` を当面参照する。
    """

    def __init__(self, agent):
        self._agent = agent
        # QC：完了キュー（Completion Queue）。RH（資源ハンドラ）が書き、LPM が drain する。
        # 要素＝(何を探したか, 結果, 起点の open 意図 id)。意図 id は完了が再会して解決するのに使う。
        self._completion_queue: asyncio.Queue[tuple[str, str, str | None]] = asyncio.Queue()
        # ループ記録は1本の鎖にする：トリガO → 意図O → 完了O → 意図O2 → …。新しい記録を
        # 書くたび直前の生きた記録を supersede するので、生き残るのは常に鎖の先頭1件だけ。
        # これで前の記録が想起に出てこなくなり、除外は「その検索を出した意図自身」で足りる。
        self._chain_head_id: str | None = None
        # 親＝この連鎖を起こした求め（人の発話 or 情動）。子＝そのために投げた調査。
        # 孫は作らない。親が決着したら生きた子をまとめて閉じる（一段だけ・再帰なし）。
        self._parent_id: str | None = None
        self._chain_head_content: str = ""
        # RH（実行担当）が走らせている投げっぱなしの呼び出し。QC が空でもこれが残っていれば
        # 結果が届くまで待つ（イベント駆動＝キュー到来で起きる）。
        self._inflight = 0
        # 飛行中の調査（動作, 探す語）。W の枠を1つずつ専有する。想起の運に任せると
        # 「いま探している」ことが調停に伝わらず、同じ問いへ二重に投げる（実機で観測）。
        # 複数の調査が並行しうるので集合ではなく列で持つ。
        self._in_flight_lookups: list[tuple[str, str]] = []
        # 完了 MI の content に「どうやって調べたか」を書くための対応（語→動作）。
        self._lookup_action_by_query: dict[str, str] = {}
        # この求めのあいだに言ったつなぎ（言った順）。次のつなぎを、繰り返しでなく
        # 続きとして自然につなぐために見せる。
        self._said_fillers: list[str] = []
        # 配る保留（「いつ・何を言いたかったか」）。W へ流し、反復が閉じたら捨てる。
        self._released_speech: list[str] = []
        # W に出した id（12桁）→ 完全な id。フルLLM の申告の突き合わせに使う。
        self._w_index: dict[str, str] = {}
        self._tasks: set[asyncio.Task] = set()
        # QA：AIFキュー（情動）。T（自律機構）が drive 発火を積む。要素＝(欲求名, 促しの内容)。
        # 3キュー（QA/QD/完了）は同じ器で待つので、待つ対象は配列で持つ（QD は1本足すだけ）。
        self._affect_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        # QD：DIFキュー（機器）。T が在席者の差分を人の出入りとして積む。
        # 要素＝(種別＝入室｜退室, 内容, 保留していた発話を配るか)。
        self._device_queue: asyncio.Queue[tuple[str, str, bool]] = asyncio.Queue()
        # 駆動体（キュー到来で次の反復を起こす）と、そこへ渡す取込待ちの完了。
        self._driver: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._inbox: list[tuple[str, str, str | None]] = []
        # 発話が出るまでの連鎖長（発話でリセット）。上限に達した反復は recall を渡さない。
        self._chain = 0
        self._capped_hit = False
        self._utterance = ""
        # 反復の起点。種別＝発話｜情動｜機器｜完了。情動や機器で起きた反復には人の発話が
        # 無いので、起点の内容を手がかり・調停の入力・user メッセージに使う。
        self._origin_kind = "発話"
        self._on_text = None
        # 発話の通知先（GUI は「発話は on_action("say") で来る」前提で作られており、
        # 素テキストは say の前の途中経過としてしか扱わない）。CUI は持たない。
        self._on_action = None
        self._pending_intent: tuple[str, dict, str] = ("", {}, "recall")


    def _advance_chain(self, new_id: str | None, content: str = "") -> None:
        """ループ記録の鎖を1つ進める（直前の生きた記録を新しい記録で supersede）。

        内容も持つのは、この先頭（取込の起点）を W へ決定的に加えるため。
        """
        if not new_id:
            return
        if self._chain_head_id and self._chain_head_id != new_id:
            self._agent._memory.mark_superseded(self._chain_head_id, new_id)
            logger.debug("event-loop 鎖を進める %.8s → %.8s", self._chain_head_id, new_id)
        self._chain_head_id = new_id
        self._chain_head_content = content

    def _dispatch_lookup(self, action: str, tool_input: dict, query: str,
                         intent_id: str | None) -> None:
        """RH：調べる動作を非同期に実行し、結果を QC へ積む（投げっぱなし・待たない）。"""
        self._inflight += 1
        self._in_flight_lookups.append((action, query))
        self._lookup_action_by_query[query] = action
        task = asyncio.create_task(self._run_lookup(action, tool_input, query, intent_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_lookup(self, action: str, tool_input: dict, query: str,
                          intent_id: str | None) -> None:
        """`recall` は同期で結果が返る。deferred は投げるだけで、完了は自身が QC へ積む。"""
        agent = self._agent
        if action != "recall":
            tool = agent._deferred_search if action == "search_deferred" else agent._deferred_fetch
            try:
                await tool.call(action, tool_input)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("event-loop %s の実行に失敗: %s", action, e)
                self._completion_queue.put_nowait(
                    (query, f"（{action} を実行できなかった：{e}）", intent_id))
            # deferred の完了は `push_completion` 経由で QC へ届く（ここでは待たない）。
            self._inflight = max(0, self._inflight - 1)
            return
        try:
            out, _ = await self._agent._memory_tool.call(
                "recall", tool_input, exclude_ids=[intent_id] if intent_id else None
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("event-loop recall の実行に失敗: %s", e)
            out = f"（recall を実行できなかった：{e}）"
        self._completion_queue.put_nowait((query, str(out), intent_id))
        logger.debug(
            "event-loop RH 完了をQCへ（id=%s qsize=%d 意図=%.8s）",
            id(self), self._completion_queue.qsize(), intent_id or "-",
        )

    async def _intake(self) -> int:
        """取込：駆動体が受けた完了（と QC の残り）を O に書き、open 意図を解決する。"""
        agent = self._agent
        # `_inbox` は作り直さず中身だけ移す。駆動体は `self._inbox.append(await get())` の
        # append を await の前に束縛するので、ここで差し替えると駆動体が捨てられた古い
        # リストへ積み、完了が黙って失われる（実機で観測）。
        items = list(self._inbox)
        self._inbox.clear()
        while not self._completion_queue.empty():
            items.append(self._completion_queue.get_nowait())
        logger.debug(
            "event-loop 取込（id=%s items=%d inflight=%d qsize=%d）",
            id(self), len(items), self._inflight, self._completion_queue.qsize(),
        )

        for query, result_text, intent_id in items:
            self._inflight = max(0, self._inflight - 1)
            for i, (_act, q) in enumerate(self._in_flight_lookups):
                if q == query:
                    del self._in_flight_lookups[i]
                    break
            # 探した事実と結果を1件に残す。open 意図と入れ替わるので W には結果つきが載る。
            action = self._action_of(query)
            cap = agent.config.completion_content_max
            done_content = f"「{query}」を {action} で調べた結果が届いた：{result_text}"[:cap]
            obs_id, _ = await agent._memory.save_async_with_id(
                done_content,
                direction="完了",
                kind="observation",
                materialize_now=True,
                parent_id=self._parent_id,
                **agent._observation_perspective(),
            )
            # 完了が open 意図に再会して解決（[D-単一想起]）＝鎖を1つ進める。W に載るのは
            # O へ書いたのと同じ文面にする（別の言い回しを2つ持つと、調停が読むものと
            # 記憶に残るものが食い違う）。
            self._advance_chain(obs_id, done_content)
        return len(items)

    # この反復で使える動作の表。値＝その動作のツール定義を agent から取り出す関数。
    # 身体を1つ繋ぐたびにここへ1行足すだけで済むようにしてある（see・look・net など）。
    # 例：("see", lambda a: a._camera.get_tool_definitions() if a._camera else [])
    _ACTIONS: dict = {
        "say": lambda a: a._tts.get_tool_definitions() if a._tts else [],
        "recall": lambda a: [
            d for d in a._memory_tool.get_tool_definitions() if d.get("name") == "recall"
        ],
        # net（投げっぱなしの外部呼び出し）。結果は完了キュー経由で後の反復に届く。
        "search_deferred": lambda a: a._deferred_search.get_tool_definitions(),
        "fetch_deferred": lambda a: a._deferred_fetch.get_tool_definitions(),
    }

    def _action_of(self, query: str) -> str:
        """その語をどの動作で投げたか。分からなければ recall とみなす。"""
        return self._lookup_action_by_query.get(query, "recall")

    def _tools(self, *, actions: tuple[str, ...] = ("say", "recall")) -> list[dict]:
        """この反復で使える動作のツール定義を返す。

        表に無い名前は黙って落とす。まだ繋いでいない身体を渡そうとしても壊れないように
        しておく（段階3 の次で see・look・search_deferred を載せる）。
        """
        agent = self._agent
        defs: list[dict] = []
        for name in actions:
            build = self._ACTIONS.get(name)
            if build is None:
                logger.debug("event-loop 未接続の動作を要求された（無視する）: %s", name)
                continue
            with contextlib.suppress(Exception):
                defs.extend(build(agent))
        return defs

    async def run_iteration(self, utterance: str, on_text=None) -> str:
        """人の発話で1反復を起こす。1反復＝1出力（発話 or ツール投げ）で終わる。

        ツールを投げた反復は発話を持たないので空文字を返す。続きは、完了が QC に届いて
        駆動体が起こす次の反復が担う。`on_text` は出力先（駆動体が起こす反復も使う）。
        """
        agent = self._agent
        # 人が話しかけた瞬間に在席の印を付ける。応答より前に付けないと、目の前の相手への
        # 返事まで在席ゲートに止められる（実機で観測＝起動直後の1回目から詰まった）。
        # 印は時刻なので、連鎖が長引いて相手が去れば自然に切れ、独り言にはならない。
        agent._last_human_at = time.time()
        self._on_text = on_text or self._on_text
        self._utterance = utterance
        self._origin_kind = "発話"
        self._chain = 0
        self._capped_hit = False
        self._ensure_driver()

        # 取込：来た事実（人の発話）を O に書く（④シーケンス）。
        trigger_id, _ = await agent._memory.save_async_with_id(
            utterance[:500],
            direction="発話",
            kind="observation",
            materialize_now=True,
            **agent._observation_perspective(),
        )
        self._parent_id = trigger_id
        self._advance_chain(trigger_id, utterance[:500])
        return await self._iterate()

    def _compose_workspace(self, mem, memories: list[dict]) -> str:
        """W を組む。調停が時間軸の基準を動かしたとき、同じ形で組み直せるようにする。

        あわせて、W に出した id（12桁）と完全な id の**対応表**を作る。フルLLM の申告を
        突き合わせるのに使う。前方一致で当てずっぽうに引くと、写し間違いが黙って別の記憶へ
        適用されてしまう。
        """
        self._w_index = {
            str(m.get("memory_id", "")).replace("-", "")[:12]: str(m.get("memory_id", ""))
            for m in memories if m.get("memory_id")
        }
        # この求めのために何を調べたかを、短い一覧として別に見せる。鎖は先頭1件しか
        # 生き残らないので W に載るのは「いちばん新しい完了」だけで、しかも取得結果は
        # 本文が長く（上限8192字）、何を取ったかがその中に埋もれる。実機では同じ URL を
        # 2反復続けて取りに行き、1反復まるごと無駄になった。
        looked_up = ""
        if self._lookup_action_by_query:
            lines = "\n".join(f"- {act}「{q}」"
                              for q, act in self._lookup_action_by_query.items())
            looked_up = f"この求めのために調べたもの（同じものを重ねて調べない）：\n{lines}"
        # 一覧が調停へ届いたかを残す。届いたのに従わないのか、そもそも届いていないのかを
        # 区別できないと、直しようがない（実機で同じ検索が4回続いた）。
        logger.debug("event-loop 調べたもの一覧＝%d件%s",
                     len(self._lookup_action_by_query),
                     "（" + "／".join(self._lookup_action_by_query) + "）"
                     if self._lookup_action_by_query else "")
        # すでに相手へ伝えた一言。これが無いと、同じ言い回しを最初から言い直す
        # （実機で「〜ですね！」で始まる前置きが3回続いた）。
        said = ""
        if self._said_fillers:
            lines = "\n".join(f"- 「{t}」" for t in self._said_fillers)
            said = ("すでに相手へ伝えた一言（言った順。次に何か言うなら、"
                    "同じ言い回しを繰り返さず、この続きとして自然につなぐ）：\n" + lines)
        held = ""
        if self._released_speech:
            held = ("聞く相手が居ないあいだに話したかったこと"
                    "（いま伝えるなら、そのときのこととして話す）：\n"
                    + "\n".join(self._released_speech))
        workspace_ctx = "\n\n".join(
            p for p in [looked_up, said, held, self._chain_head_content,
                        mem.format_for_context(memories)]
            if p and p.strip()
        )

        # 3. ARB（調停）：軽量LLM が会話の重さを自己判断し、出し方を3つへ振り分ける（段4）。
        #    フルLLM は「言語生成が要るとき」だけ起こす。実測で1ターン 10.5 秒のうち LLM が
        #    10.2 秒を占め、recall を投げるだけの反復にもフルを使っていた。
        return workspace_ctx

    def _apply_memory_verdicts(self, raw) -> None:
        """フルLLM が申告した「想起した記憶の扱い」を反映する（課題5 E節 段2）。

        **照合できたものだけ適用する**。指示しても、落としたり無い id を足したりする。
        欠けた分を「使わなかった」と決めつけると、申告漏れと本当に使わなかったことを
        混同する。件数をログに残し、指示が守られているかを後から確かめられるようにする。
        """
        if not raw or not self._w_index:
            return
        verdicts: dict[str, str] = {}
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            full = self._w_index.get(str(item.get("id", "")).replace("-", "")[:12])
            verdict = str(item.get("verdict", "")).strip().lower()
            if full and verdict in ("important", "useless", "referred", "unused"):
                verdicts[full] = verdict
        logger.info("event-loop 記憶の判定 %d/%d 件", len(verdicts), len(self._w_index))
        if verdicts:
            with contextlib.suppress(Exception):
                self._agent._memory.apply_verdicts(verdicts)

    def _emit(self, text: str) -> None:
        """発話を表示先へ渡す。素テキストと say 動作の両方で知らせる。"""
        if not text:
            return
        if self._on_text is not None:
            self._on_text(text)
        if self._on_action is not None:
            with contextlib.suppress(Exception):
                self._on_action("say", {"text": text})

    def set_output(self, on_text, on_action=None) -> None:
        """発話の表示先を登録する。人の発話を待たずに出口が定まる（起動時にアプリが渡す）。

        `on_action`：GUI の表示経路。ログ表示・ひとりごと判定・音声タグの除去がそちらに
        集まっているので、同じ約束（`("say", {"text": …})`）で通知すればそのまま効く。
        """
        self._on_text = on_text
        self._on_action = on_action or self._on_action

    def start(self) -> None:
        """駆動体だけを起こす。以後はキュー到来で反復が回る。"""
        self._ensure_driver()

    def push_completion(self, query: str, result: str) -> None:
        """RH（資源ハンドラ）が deferred の完了を QC へ積む。

        投げっぱなしの外部呼び出し（検索・取得）の結果は、完了キュー→O 経由で次の反復の
        入力になる（正本③）。スレッドから呼ばれても届くよう、ループへ委譲する。
        """
        loop = getattr(self, "_loop", None)
        item = (query, str(result), None)
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._completion_queue.put_nowait, item)
        else:
            self._completion_queue.put_nowait(item)

    def push_affect(self, drive_name: str, prompt: str) -> None:
        """T が drive 発火を QA へ積む（AIF 経由・I は時計を見ない）。"""
        self._affect_queue.put_nowait((drive_name, prompt))

    def push_device(self, kind: str, content: str, *, release_pending: bool = False) -> None:
        """T が人の出入りを QD へ積む（DIF 経由・I は時計を見ない）。"""
        self._device_queue.put_nowait((kind, content, release_pending))

    def _ensure_driver(self) -> None:
        """駆動体：キュー到来で次の反復を起こす（イベント駆動・時計は見ない）。"""
        if self._driver is None or self._driver.done():
            self._loop = asyncio.get_running_loop()
            self._driver = asyncio.create_task(self._drive())

    async def _drive(self) -> None:
        """3キューの union を待ち、来たどれでも起きる（時計は見ない・正本③）。

        待つのは受ける側だけで、時計を持つのは T（自律機構）である。上限は設けない：
        終了は `close()` の cancel が待ちの最中でも即座に効くので、定期的に目を覚ます
        必要がない（目を覚ますこと自体が「時計を見る」動作になる）。
        """
        while True:
            try:
                # 待つ対象は配列で持つ（QD を足すときは1本加えるだけ）。
                # **調査中は完了キューだけを待つ。** 飛行中の調査があるあいだに情動や
                # 人の出入りで別の連鎖を始めると、1つの求めの途中に別の話が割り込む。
                # 聞いている側には、軽量LLM とフルLLM が交互に喋る＝別々の人格が居る
                # ように聞こえる（実機で観測）。QA・QD は**消費せずキューに残す**ので、
                # 調査が終われば順に処理される（取りこぼしではなく待たせるだけ）。
                # 代償：drive の発火と人の入退室への反応が、その求めが終わるまで遅れる。
                queues = (
                    [self._completion_queue]
                    if self._inflight
                    else [self._completion_queue, self._affect_queue, self._device_queue]
                )
                waiters = {asyncio.ensure_future(q.get()): q for q in queues}
                try:
                    done, pending = await asyncio.wait(
                        waiters, return_when=asyncio.FIRST_COMPLETED
                    )
                finally:
                    pass
                for task in pending:
                    task.cancel()
                affect = None
                device = None
                for task in done:
                    item = task.result()
                    q = waiters[task]
                    if q is self._completion_queue:
                        self._inbox.append(item)
                    elif q is self._affect_queue:
                        affect = item
                    else:
                        device = item
                # 同じキューに溜まっている分もまとめて取る。
                while not self._completion_queue.empty():
                    self._inbox.append(self._completion_queue.get_nowait())
                if affect is None and not self._affect_queue.empty():
                    affect = self._affect_queue.get_nowait()
                if device is None and not self._device_queue.empty():
                    device = self._device_queue.get_nowait()

                if device is not None:
                    kind, content, release_pending = device
                    logger.debug("event-loop 駆動体が機器を受領（%s）", kind)
                    await self._begin_device(kind, content, release_pending)
                elif affect is not None:
                    drive_name, prompt = affect
                    logger.debug("event-loop 駆動体が情動を受領（%s）", drive_name)
                    await self._begin_affect(drive_name, prompt)
                else:
                    logger.debug(
                        "event-loop 駆動体が完了を受領（id=%s inbox=%d）",
                        id(self), len(self._inbox),
                    )
                    await self._iterate()
            except asyncio.CancelledError:
                for task in waiters:
                    task.cancel()
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("event-loop 駆動体で例外: %s", e)

    async def _begin_affect(self, drive_name: str, prompt: str) -> None:
        """情動で新しい連鎖を始める。取込＝来た事実（情動）を O に書き、鎖の起点にする。

        情動は中身を持たないので、取り込み時に想起で状況づける（正本③ 手順1・2）。
        """
        agent = self._agent
        self._utterance = ""
        self._origin_kind = "情動"
        self._chain = 0
        self._capped_hit = False
        content = f"[内的な促し:{drive_name}] {prompt}"
        obs_id, _ = await agent._memory.save_async_with_id(
            content[:500],
            direction="情動",
            kind="observation",
            materialize_now=True,
            **agent._observation_perspective(),
        )
        self._parent_id = obs_id
        self._advance_chain(obs_id, content[:500])
        await self._iterate()

    async def _begin_device(self, kind: str, content: str, release_pending: bool) -> None:
        """機器（人の出入り）で新しい連鎖を始める。取込＝来た事実を O に書き、鎖の起点にする。

        `release_pending` が真なら、聞く相手が居らず保留していた発話を先に配る。在席が
        ゼロから立ち上がった瞬間だけ真になる（寿命は `pending_speech` 側が持つので、
        新しいキューは作らない）。
        """
        agent = self._agent
        self._utterance = ""
        self._origin_kind = "機器"
        self._chain = 0
        self._capped_hit = False
        text = f"[{kind}] {content}"
        obs_id, _ = await agent._memory.save_async_with_id(
            text[:500],
            direction="機器",
            kind="observation",
            materialize_now=True,
            **agent._observation_perspective(),
        )
        self._parent_id = obs_id
        self._advance_chain(obs_id, text[:500])
        if release_pending:
            await self._release_pending_speech()
        await self._iterate()

    async def _release_pending_speech(self) -> None:
        """保留していた発話を取り出し、**W へ流す分として持つ**（鮮度切れは捨てる）。

        MI の content へ差し込まない。保留の記録（`direction="保留"`）は観測なので、想起でも
        W に上がってくる（実機のログで、入室の反復の想起上位4件が保留 O だった）。content に
        も差し込むと同じ話が二重に載る。

        **いつ言いたかったか**を添える。経過時間だけだと「23時台に言いたかった」という文脈が
        落ち、時刻だけだと日付をまたいだとき「昨夜」か「今朝」か決まらない。両方あれば、
        言葉を組み立てる側が自然な言い方を選べる。

        配った分は `pending_speech` から消し、元の O も supersede する（消さないと、想起で
        W に上がり続けて何度も蒸し返す）。
        """
        store = getattr(self._agent, "_pending_store", None)
        if store is None:
            return
        try:
            from ..config import PendingSpeechConfig

            cfg = PendingSpeechConfig()
            now_epoch = time.time()
            released: list[str] = []
            for row in store.list_active():
                score = store.freshness_score(row, now_epoch, cfg)
                if store.is_expired(row, score, cfg):
                    store.delete(row["id"])
                    continue
                content = str(row.get("content", "")).strip()
                if content:
                    released.append(f"- {_when(row.get('created_at'), now_epoch)}：{content}")
                store.delete(row["id"])
                with contextlib.suppress(Exception):
                    self._agent._memory.mark_superseded(row["observation_id"], self._parent_id)
            self._released_speech = released
            if released:
                # 何件を W へ流したかを残す。system プロンプトの全文は出していないので、
                # これが無いと「載ったが触れられなかった」のか「そもそも載っていない」のか
                # を区別できない（実機で、配られたのに発話が触れなかった）。
                logger.info("event-loop 保留を配る：%d件", len(released))
        except Exception as e:  # noqa: BLE001
            logger.exception("保留していた発話を取り出せなかった: %s", e)

    async def close(self) -> None:
        if self._driver is not None:
            self._driver.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._driver
            self._driver = None

    async def _iterate(self) -> str:
        """1反復：取込 → W 構築 → 生成 → 出力（発話 or ツール投げ）で終わる。"""
        from ..capability_state import load_summary
        from ..config import MemoryConfig

        agent = self._agent
        utterance = self._utterance
        max_chain = max(1, agent.config.event_max_iterations)
        self._chain += 1
        chain = self._chain
        logger.debug("event-loop iter=%d/%d 開始", chain, max_chain)

        # 1. 取込：駆動体が受けた完了を O に書き、open 意図を解決する。
        drained = await self._intake()
        if drained:
            logger.debug("event-loop iter=%d/%d QC取込=%d件", chain, max_chain, drained)

        # 2. REC（想起）：O（＋現入力）→ W。W は派生なので反復末に捨てる。
        # 一律の規則：取込で書いた記録（＝鎖の先頭）は検索から外し、W へは決定的に加える。
        # 素通しだと問いと同一文の記録が必ず上位に来て、限られた枠から本物の記憶を押し出す。
        # 手がかりは「取り込んだもの」＝鎖の先頭（反復1なら人の発話、反復2以降なら完了 O）。
        # 最初の発話で探し続けると、いま届いた完了とは無関係な検索になる（④ の想起クエリ）。
        mem = agent._active_memory()
        origin_ids = [self._chain_head_id] if self._chain_head_id else None
        cue = self._chain_head_content or utterance
        memories = await mem.recall_async(
            cue,
            n=MemoryConfig().recall_n,
            exclude_ids=origin_ids,
        )
        # W は「思い出している記憶」ではなく、いまの作業状態。ループ自身の行動も MI として
        # O にあるので、合成ラベル（[取込]・[調査中]）は作らず MI をそのまま並べる。
        # W から落ちたものは薄れた＝忘れたのであって、抜けを検出する仕組みは置かない
        # （W は「速く薄れる」・改めて調べるのが自然な振る舞い）。
        workspace_ctx = self._compose_workspace(mem, memories)

        # 誰と話していると思って喋ったかを残す。これが無いと、口調がおかしいときに
        # 「話者が渡っていない」のか「渡ったが口調が従っていない」のかを切り分けられない。
        present_ctx = _present_ctx(agent)
        logger.debug("event-loop iter=%d/%d 在席=%s", chain, max_chain, present_ctx)

        capped = chain >= max_chain
        if capped:
            # 上限で打ち切ったことは、後からログだけで判別できる必要がある（DEBUG の
            # iter=N/M からは「たまたま N 回で終わった」のか「打ち切った」のか分からない）。
            logger.info("event-loop 反復 %d/%d 上限に達したため探索を打ち切る", chain, max_chain)
            self._capped_hit = True
        decision = await arbitrate(
            agent._utility_backend,
            utterance=utterance or self._chain_head_content,
            workspace_ctx=workspace_ctx,
            self_understanding=load_summary() or getattr(agent, "_me_md", ""),
            family_md=getattr(agent, "_family_md", ""),
            present_ctx=present_ctx,
            now_ctx=f'(now :datetime "{clock.now_local_str()}")',
            capped=capped,
        )
        logger.debug("event-loop iter=%d/%d 調停=%s effort=%s",
                     chain, max_chain, decision.branch, decision.effort)
        # 「いまは話しかけないで」と読めたら、その人が居るあいだ黙る。この反復の受け答えは
        # 出したうえで（頼みに無言で応じるのは不自然）、次の反復から止める。
        if decision.silence:
            self._accept_silence()
        # 調停が時期を指した（「去年の夏の話」）なら、その基準で想起し直して W を組み直す。
        # 想起は調停より前に走るので、この反復に効かせるには引き直すしかない。実測 17〜50ms
        # で、指定があったときだけ走る。
        if decision.time_ref:
            with contextlib.suppress(Exception):
                ref = datetime.fromisoformat(decision.time_ref).timestamp()
                span = decision.time_span_days or None
                memories = await mem.recall_async(
                    cue, n=MemoryConfig().recall_n,
                    exclude_ids=origin_ids, time_ref=ref, time_span_days=span,
                )
                workspace_ctx = self._compose_workspace(mem, memories)
                logger.info("event-loop 想起の基準を移す：%s（幅 %s 日）",
                            decision.time_ref, decision.time_span_days or "既定")

        # (a) 軽量で閉じる：フルLLM を起こさず、軽量LLM の応答で反復を終える。
        if decision.branch == "light" and decision.text:
            return await self._speak(decision.text, memories)

        # (c) 定型：探すと決まっている反復も、フルLLM を起こさず投げて閉じる。
        if decision.branch == "action" and decision.query and not capped:
            # つなぎの一言はここで即出す（フルLLM を経由しないぶん速い・正本③ 段5 の内部二段）。
            await self._say_filler(decision.text)
            self._open_intent(utterance or self._chain_head_content,
                              {"query": decision.query}, action=decision.action)
            logger.info("event-loop 反復 %d/%d 出力=%s（調停・続きは完了で起きる）",
                        chain, max_chain, decision.action)
            return ""

        # (b) 軽量つなぎ→フル（正本③ 段5 の内部二段）。フル生成は effort=high で10秒近く
        # かかり、そのあいだ無音になる。つなぎで体感の待ち時間を埋める。**1つの work の
        # 内部二段**であって別の出力ではない（1反復1出力は保たれる）。
        # effort=low は実測 0.8〜3.6 秒で返るので挟まない（かえってテンポが悪くなる）。
        # **材料が届いた反復でも挟まない**（`drained`）。待つものがもう無いのに「待って」と
        # 言う理由がない。実機では、検索結果が届いた1秒後に「うん、任せてね！」が出て、
        # 一言目（ですます）と本応答（ですます）のあいだでそこだけ口調が割れた。
        if (decision.branch == "full" and decision.text
                and decision.effort != "low" and not drained):
            await self._say_filler(decision.text)

        system = build_event_system_prompt(
            self_understanding=load_summary() or getattr(agent, "_me_md", ""),
            family_md=getattr(agent, "_family_md", ""),
            present_ctx=present_ctx,
            pi_ctx=_pi_ctx(),
            iter_ctx=(
                f"[反復] {chain}/{max_chain}"
                # 上限では、黙って手持ちで繕わず「調べきれなかった」と断ってから答える。
                # 断りが無いと、材料不足のまま答えたことが相手に伝わらない。
                + ("（これ以上は調べられない。調べきりたかったが上限に達したことを述べ、"
                   "そのうえで現時点で分かることを返す）" if capped else "")
            ),
            workspace_ctx=workspace_ctx,
        )
        # 生成中はストリームしない：ツールを選ぶ反復で出る前置きの地の文が表示され重複するため。
        # 起点が人の発話ならそのまま、情動・機器なら内的な出来事として渡す。空文字を送ると
        # 何がこの反復を起こしたのか分からなくなる（API も空メッセージを受け付けない）。
        user_msg = agent.backend.make_user_message(utterance or self._chain_head_content)
        result, _raw = await agent.backend.stream_turn(
            system=system,
            messages=[user_msg],
            # 連鎖上限の反復では recall を外し、発話だけにして必ず閉じる。
            tools=self._tools(actions=("say",) if capped else _FULL_ACTIONS),
            max_tokens=agent.config.max_tokens,
            on_text=None,
            effort=decision.effort,
        )

        say_tc = next((tc for tc in result.tool_calls if tc.name == "say"), None)
        # 上限の反復では調べる動作を渡していないので、返ってきても投げない（連鎖を必ず閉じる）。
        lookup_tc = None if capped else next(
            (tc for tc in result.tool_calls if tc.name in _LOOKUP_ACTIONS), None
        )

        # 発話と動作が一緒に来たら、発話はつなぎとして出し、その反復の出力は動作とする。
        # 以前は say を見つけた時点で閉じており、同じ応答に入っていた検索を捨てていた。
        if lookup_tc is not None:
            logger.debug("event-loop iter=%d/%d 決定=%s", chain, max_chain, lookup_tc.name)
            if say_tc is not None:
                await self._say_filler(str(say_tc.input.get("text", "")).strip())
            self._open_intent(utterance or self._chain_head_content, dict(lookup_tc.input),
                              action=lookup_tc.name)
            logger.info("event-loop 反復 %d/%d 出力=%s（続きは完了で起きる）",
                        chain, max_chain, lookup_tc.name)
            return ""

        if say_tc is not None:
            logger.debug("event-loop iter=%d/%d 決定=say", chain, max_chain)
            self._apply_memory_verdicts(say_tc.input.get("memory_verdicts"))
            return await self._speak(str(say_tc.input.get("text", "")).strip(), memories)

        # どちらも無ければ素テキストへフォールバック（表示はここで1回）。
        logger.debug("event-loop iter=%d/%d 決定=none", chain, max_chain)
        text = (result.text or "").strip()
        if text:
            self._emit(text)
        await self._finish(text, memories, "沈黙")
        return text

    async def _speak(self, text: str, memories: list[dict]) -> str:
        """発話して反復を閉じる。聞く相手が居なければ話さず、後で話すために溜める。

        身体を持つ以上、発話は相手が居て初めて意味を持つ（正本③ の配信ゲート＝結果有り＋在席）。
        居ないときは「話したかったができなかった」を O に残して `pending_speech` へ積み、
        次に人が現れたときに気づけるようにする。溜めたものの寿命（鮮度切れ・参照先 supersede で
        失効）は `pending_speech` 側が持つ。
        """
        agent = self._agent
        if not text:
            await self._finish("", memories, "沈黙")
            return ""
        blocked = self._delivery_block_reason()
        if blocked:
            await self._hold_speech(text)
            logger.info("event-loop %s ので発話を保留し pending_speech へ積む", blocked)
            await self._finish("", memories, "保留")
            return ""
        if agent._tts is not None:
            with contextlib.suppress(Exception):
                await agent._tts.call("say", {"text": text})
        self._emit(text)
        await self._finish(text, memories, "発話")
        return text

    async def _say_filler(self, text: str) -> None:
        """つなぎの一言を出す（内容にコミットしない前置き）。配信ゲートは同じく効かせる。

        本応答ではないので、これで反復を閉じない。溜める（`pending_speech`）のも本応答の
        役目なので、出せない場面では黙って落とす。
        """
        if not text or self._delivery_block_reason():
            return
        agent = self._agent
        if agent._tts is not None:
            with contextlib.suppress(Exception):
                await agent._tts.call("say", {"text": text})
        self._emit(text)
        # 言ったことを O に残す。残さないと、次の反復の W に「もう一言伝えた」事実が
        # 入らず、調停はそれを知らないまま同じことをまた言う（実機で1秒差に同じ文が
        # 2回出た）。抑止で黙らせるのではなく、判断できる材料を渡して解く。
        # **鎖は進めない。** 鎖の先頭は「いま処理している対象」を1つ持つためのもので、
        # つなぎはその対象ではない。進めると、直前に届いた完了を押し出して W から
        # 消してしまい、フルLLM が材料を失う（実機で未回答に終わった）。
        # 求めが決着したら他の子と一緒に閉じるので、記憶に残り続けることはない。
        self._said_fillers.append(text)
        await agent._memory.save_async_with_id(
            f"つなぎに言った：{text}"[:500],
            direction="発話",
            kind="observation",
            materialize_now=True,
            parent_id=self._parent_id,
            **agent._observation_perspective(),
        )

    def _delivery_block_reason(self) -> str:
        """配信ゲート。発話を出せない理由を返す（出せるなら空文字）。

        正本③ の「配信ゲート（結果有り＋在席）」に静穏時間を併せる。以前は静穏時間を
        deferred の配信側だけが見ており、自発発話は素通りしていた。判定をここへ集める。
        """
        agent = self._agent
        # 「黙っていて」と頼まれているあいだは、話しかけられても話さない。頼んだ人が
        # 居なくなれば（退室）その時点で解け、期限（Config・既定60分）を過ぎても解ける。
        # 判定だけで済むので解除の処理を別に持たない。言葉は捨てず pending_speech へ溜める。
        with contextlib.suppress(Exception):
            from ..silence_state import is_silenced, load_silence

            if is_silenced(load_silence(), present=self._present_names(), now=time.time()):
                return "黙っているよう頼まれている"
        if agent._social_presence_permission() == 0.0:
            return "聞く相手が居ない"
        # 静穏時間は「**自分から**話しかけない時間」で、話しかけられたのに黙るための
        # ものではない。起点を区別せず掛けていたため、夜に話しかけても返事が出ず、
        # 保留されて翌朝に届く動きになっていた（実機で観測）。在席と「黙っていて」の
        # 依頼は起点によらず掛かるので、ここだけを分ける。
        if self._origin_kind != "発話":
            with contextlib.suppress(Exception):
                if agent._in_quiet_hours():
                    return "静穏時間である"
        return ""

    def _accept_silence(self) -> None:
        """黙っている依頼を受ける。宛先は、いま話している相手。"""
        agent = self._agent
        with contextlib.suppress(Exception):
            from ..silence_state import SilenceRequest, save_silence

            who = ""
            for row in agent._pmm.presence_status():
                if row.get("is_speaker"):
                    who = str(row.get("name") or "")
                    break
            if not who and getattr(agent._persons, "active_is_explicit", False):
                who = agent._persons.active_name
            if not who:
                logger.info("黙っているよう頼まれたが、誰からか分からないので受けない")
                return
            minutes = max(1, int(getattr(agent.config, "silence_minutes", 60)))
            save_silence(SilenceRequest(person=who, until=time.time() + minutes * 60))

    def _present_names(self) -> set[str]:
        """いま在席している人の名前（黙っている依頼の宛先と突き合わせる）。"""
        names: set[str] = set()
        with contextlib.suppress(Exception):
            for row in self._agent._pmm.presence_status():
                name = str(row.get("name") or "")
                if name:
                    names.add(name)
        return names

    async def _hold_speech(self, text: str) -> None:
        """話せなかった内容を O に残し、`pending_speech` へ積む（想起系は汚さない）。"""
        agent = self._agent
        obs_id, _ = await agent._memory.save_async_with_id(
            f"話したかったが、聞く相手が居なかった：{text}"[:500],
            direction="保留",
            kind="observation",
            materialize_now=True,
            **agent._observation_perspective(),
        )
        if obs_id:
            with contextlib.suppress(Exception):
                agent._pending_store.add(obs_id, None)

    def _open_intent(self, utterance: str, tool_input: dict, *, action: str = "recall") -> None:
        """open 意図を O に残し、RH へ投げる（待たない）。意図は常に高々1件に保つ。"""
        self._pending_intent = (utterance, tool_input, action)
        self._tasks.add(t := asyncio.create_task(self._write_intent_and_dispatch()))
        t.add_done_callback(self._tasks.discard)

    async def _write_intent_and_dispatch(self) -> None:
        agent = self._agent
        utterance, tool_input, action = self._pending_intent
        query = str(tool_input.get("query") or tool_input.get("url", "")).strip()
        content = f"「{utterance}」について {action}（{query}）を要求した。結果はまだ無い。"[:500]
        intent_id, _ = await agent._memory.save_async_with_id(
            content,
            direction="意図",
            kind="observation",
            materialize_now=True,
            parent_id=self._parent_id,
            **agent._observation_perspective(),
        )
        # 意図を書いた時点でトリガ（や前回の完了）は死ぬ＝この検索には出てこない。
        self._advance_chain(intent_id, content)
        # 自分が出した検索が自分自身を拾わないよう、意図 O の id だけ狭く除外する。
        self._dispatch_lookup(action, tool_input, query, intent_id)

    async def _finish(self, text: str, memories: list[dict], outcome: str) -> None:
        """発話で連鎖が閉じた反復の後始末：総括ログと永続化（ループ中 O を supersede）。"""
        agent = self._agent
        logger.info("event-loop 終了: 反復=%d 結末=%s 上限到達=%s text_len=%d",
                    self._chain, outcome, "はい" if self._capped_hit else "いいえ", len(text))
        # 自分が言ったことを、**発話の時点で同期に** O へ書く。背景の永続化（要約・内省）を
        # 待つと2秒遅れ、そのあいだに次の反復が起きると「さっき何と言ったか」を拾えない
        # （実機で「それだけ？」に聞き返した）。要約は後から来て、この記録を supersede する。
        answer_id = None
        if text:
            with contextlib.suppress(Exception):
                answer_id, _ = await agent._memory.save_async_with_id(
                    f"自分が答えた：{text}"[:500],
                    direction="発話",
                    kind="observation",
                    materialize_now=True,
                    parent_id=self._parent_id,
                    **agent._observation_perspective(),
                )
        # 親が決着したら、生きている子（その求めのために投げた調査）もまとめて閉じる。
        # **閉じる側をこの本応答 MI にする**。子として閉じられると `superseded_by` が入り、
        # 想起の候補（新しさ軸・関連軸とも `superseded_by IS NULL`）から外れて、次のループで
        # 見つけられなくなる。`close_with_children` は new_id 自身を除外するので生き残る。
        parent_id, self._parent_id = self._parent_id, None
        obs_ids = [self._chain_head_id] if self._chain_head_id else []
        self._chain_head_id = None
        # いま閉じる（背景の永続化を待たない）。待つと、その2秒のあいだ意図・完了・つなぎが
        # 生きたまま次の反復の候補に出る。閉じる側を本応答 MI にすることで、生き残るのは
        # 「自分が答えた」1件だけになる。2秒後に届く会話要約がこの記録を supersede し、
        # 恒久記録は要約が担う（[逐語を拡散想起から辿れるようにするのは今後の課題]）。
        if answer_id and parent_id:
            with contextlib.suppress(Exception):
                agent._memory.close_with_children(parent_id, answer_id)
            obs_ids = [answer_id]
        self._in_flight_lookups.clear()
        self._lookup_action_by_query.clear()
        self._said_fillers.clear()
        self._released_speech.clear()
        self._chain = 0
        self._capped_hit = False
        try:
            origin = self._utterance or self._chain_head_content
            arousal = await agent._turn_arousal(origin, text)
            agent._spawn_background_task(
                agent._run_post_response_pipeline(
                    user_input=origin, final_text=text,
                    camera_used=False, camera_image=None,
                    observation_action_name=None, observation_action_input=None,
                    companion_mood="engaged", is_desire_turn=False, desires=None,
                    arousal=arousal, memories=memories,
                    superseded_ids=obs_ids or None,
                    close_parent_id=parent_id,
                ),
                name="event-post-response",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("event-loop persistence spawn failed: %s", e)
