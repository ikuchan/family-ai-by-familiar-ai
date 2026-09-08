"""イベント駆動ループ（#11 段階1）：I（情報処理機構）の最小縦切り。

設計正本＝`設計図_Mermaid` ③ I 詳細図。ここでは I の中の **LPM（ループ核）** と
**QC（完了キュー）** だけを実体化する。反復は QC を drain（取込→O 書込）→ REC（想起→W）→
GEN（生成）で進み、say で1出力して終わる／内部ツール（recall）は結果を QC へ積んで次反復へ
連鎖する（[D-単一想起]：相関ID を使わず結果は O→W 経由で再会）。

人の発話はこの経路が処理する。AIF/DIF/QA/QD、ARB/APR/ACT/MNT の
クラス分離は後続段階（ここでは stub しない）。永続化は既存 `_run_post_response_pipeline`
（utility LLM のみ）を流用し、消化した完了 O はターン観察 id で supersede する。
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
import logging
import time
from datetime import datetime
from typing import Any

from ..poses import nearest_pose
from ..scene import extract_entities
from ..store import clock
from .arbiter import arbitrate
from ..store.relations import KIND_ADVANCE, KIND_RESOLVE, KIND_REVISION
from ..io.dif import DIF
from .coherence import facts_ctx
from .generator import _pi_ctx, _present_ctx
from .prompt import build_event_system_prompt

logger = logging.getLogger(__name__)

# 連鎖が続けられる反復で渡す動作。上限に達した反復では say だけにして必ず閉じる。
_FULL_ACTIONS = (
    "say",
    "recall",
    "search_deferred",
    "fetch_deferred",
    "see",
    "look",
    "house_rules",
)
# 調べる動作＝結果が後の反復に届くもの。投げたらその反復は終わる。
# `see`・`look` も含める。結果はその場で返るが、それを見て何を言うかは次の反復が決める
# （`recall` と同じ）。ここに入れないと 1反復1出力 が崩れる。
_LOOKUP_ACTIONS = (
    "recall",
    "search_deferred",
    "fetch_deferred",
    "see",
    "look",
    # 家の決まりは即座に返るが、それを見て何を言うかは次の反復が決める（`recall` と同じ）。
    "house_rules",
)


async def _result_or_none(task):
    """待ち合わせて結果を返す。落ちたら None（繋がない側へ倒す）。

    判定が来ないターンは新しい話の始まりとして扱う。誤って繋ぐと、関係のない会話が
    文脈に混ざる（`設計方針_MI間の関係`）。
    """
    try:
        return await task
    except Exception as e:  # noqa: BLE001
        logger.debug("続き先の判定を受け取れなかった（続行する）: %s", e)
        return None


def _query_label(action: str, tool_input: dict) -> str:
    """その求めの見出し。飛行中の一覧・完了の照合・W の「調べたもの」で鍵になる。

    `see` の入力は空で、`look` は向きしか持たない。`query`／`url` から取ると両方とも
    空文字になり、別々の求めが同じ鍵で衝突する。動作ごとに見出しを作る。
    """
    if action == "see":
        return "目の前を見る"
    if action == "look":
        return f"{tool_input.get('pose', '')}を見に行く"
    return str(tool_input.get("query") or tool_input.get("url", "")).strip()


def _camera_tool_def(agent, name: str) -> list[dict]:
    """カメラの道具定義から1つだけ取り出す。カメラが無ければ空。"""
    cam = getattr(agent, "_camera", None)
    if cam is None:
        return []
    return [d for d in cam.get_tool_definitions() if d.get("name") == name]


def _elapsed_label(created_at, now_epoch: float) -> str:
    """いつのことかを「経過時間（時刻）」で書く。片方だけでは足りない。"""
    with contextlib.suppress(Exception):
        stamp = created_at.timestamp()
        hours = (now_epoch - stamp) / 3600.0
        ago = f"{int(hours * 60)}分前" if hours < 1 else f"約{int(hours)}時間前"
        return f"{ago}（{created_at.astimezone().strftime('%m/%d %H:%M')}）"
    return "いつか"


def _log_recall_weights(trigger, base, used, memories) -> None:
    """採用した5軸重みと、その重みで出た上位のスコアを残す（INFO）。

    重みは反復ごとに揺らぐので、後から「どの重みでどう並んだか」を紐づけられないと、
    値を実挙動から選べない。**記憶の内容は出さない**（INFO 以上に会話・記憶内容を出さない
    方針。中身は DEBUG の `recall score` の内訳にある）。
    """

    def _fmt(w):
        return "(%.2f,%.2f,%.2f,%.2f,%.2f)" % (w.w_r, w.w_t, w.w_e, w.w_g, w.w_p)

    top = "/".join("%.3f" % m["fit"] for m in memories[:3] if "fit" in m)
    logger.info(
        "event-loop 想起 trigger=%s w=%s 基底=%s 上位=%s %d件",
        trigger,
        _fmt(used),
        _fmt(base),
        top or "なし",
        len(memories),
    )


@dataclass
class Lookup:
    """1件の調べもの（環-g・段は）。

    以前は「どの動作で」「何という語で」が**6つの入れ物に3通りで**入っていた。
    `_inflight`（数）と `_in_flight_lookups`（列）は名前も意味もほぼ同じで、5箇所で
    別々に動かしていた。1件を1つの器にすれば、**飛行中の数は導出になり**、釣り合いを
    手で守らずに済む。

    `generation` は投げたときの求めの世代。打ち切ったあとに届いた完了を捨てるのに使う。
    """

    index: int
    action: str
    query: str
    generation: int
    result: "str | None" = None

    @property
    def in_flight(self) -> bool:
        """まだ結果が届いていないか。"""
        return self.result is None


class InformationProcessing:
    """I：情報処理機構（Information-processing）。③ I 詳細図の器。

    段階1で実体化するのは **QC（完了キュー）** と **LPM（ループ核）＝`begin_request`** のみ。
    O・C（Config）・W・RH 相当のツール実行は既存実体を持つ `agent` を当面参照する。
    """

    def __init__(self, agent):
        self._agent = agent
        # QC：完了キュー（Completion Queue）。RH（資源ハンドラ）が書き、LPM が drain する。
        # 要素＝(何を探したか, 結果, 起点の open 意図 id)。意図 id は完了が再会して解決するのに使う。
        # 要素＝(何を探したか, 結果, 起点の open 意図 id, 種別)。種別＝完了｜進捗。
        # 「進捗」は結果ではないので、飛行中の数も一覧も触らず、意図も supersede しない。
        self._completion_queue: asyncio.Queue[tuple[str, str, str | None, str, int]] = (
            asyncio.Queue()
        )
        # 外の機械（声・調べもの）へはこの口だけを通す（環-e-は）。要るものだけを渡す。
        self._dif = DIF(
            tts=agent._tts,
            search=agent._deferred_search,
            fetch=agent._deferred_fetch,
            mcp=agent._mcp,
        )
        # ループ記録は1本の鎖にする：トリガO → 意図O → 完了O → 意図O2 → …。新しい記録を
        # 書くたび直前の生きた記録を supersede するので、生き残るのは常に鎖の先頭1件だけ。
        # これで前の記録が想起に出てこなくなり、除外は「その検索を出した意図自身」で足りる。
        self._chain_head_id: str | None = None
        # 親＝この連鎖を起こした求め（人の発話 or 情動）。子＝そのために投げた調査。
        # 孫は作らない。親が決着したら生きた子をまとめて閉じる（一段だけ・再帰なし）。
        self._request_id: str | None = None
        self._chain_head_content: str = ""
        # この求めで投げた調べもの（1件＝1つの `Lookup`）。**飛行中も届いた分も同じ列**に
        # 並ぶ（`result` が `None` なら飛行中）。以前は6つの入れ物に3通りで持ち、数と列を
        # 5箇所で手で揃えていた（環-g・段は）。
        #
        # 通し番号は求めの中で1から振る。いま調べものを識別しているのは語だけで、同じ語を
        # 2回投げると区別できない。版の content へ「1番：… 2番：…」と列挙し、届いた完了を
        # 番号で対応づけるために振る。求めをまたいだ突き合わせは要らないので、一意な id では
        # なく通し番号で足りる。
        self._lookups: list[Lookup] = []
        # 求めそのものの文面。**どの版にも入れる。** 前の版は畳まれて辿れなくなるので、
        # 各版が単独で「何を聞かれたか」を持たないと、求めの文脈が失われる。
        self._request_text = ""
        # いま生きている版の id。次の版がこれを畳む（1本の鎖）。
        self._live_version_id: str | None = None
        # 直前に書いた版の id。`recall` ツールが自分自身を拾わないための除外に使う。
        self._recall_exclude_id: str | None = None
        # この求めのあいだに言ったつなぎ（言った順）。次のつなぎを、繰り返しでなく
        # 続きとして自然につなぐために見せる。
        self._said_fillers: list[str] = []
        # 配る保留（「いつ・何を言いたかったか」）。W へ流し、反復が閉じたら捨てる。
        self._speech_to_deliver: list[str] = []
        # W に出した id（12桁）→ 完全な id。フルLLM の申告の突き合わせに使う。
        self._w_id_map: dict[str, str] = {}
        # このターンが作った記録と、その役割（観測 id, 役割）。**一つの並びが二つの用を
        # 賄う**：拡散想起の母集合（共起の関係）へ載せる id と、やりとりの関係の項。
        # 役割は 起点・版・見た・つなぎ・答え（`_note_record`）。つなぎは共起に載せない
        # （中身が無く、育てる価値がない）。中断はこの求めで閉じるが、次の求めの共起には
        # 載る（打ち切った調査と言い直した問いの共起は、たどる価値がある）。
        self._turn_records: list[tuple[str, str]] = []
        # いまのやりとりが、その並びのどこから始まったか。**やりとりは並びの一区間**
        # である。母集合への持ち越しは打ち切りでも消さないが、やりとりは打ち切りで
        # 区切る。二つの用は、区切りの規則が違う。
        self._exchange_start = 0
        # 直近のやりとりを、どこから見せるかのカーソル。**繋ぐためではない。**
        # 辺を書くのは `follows` だけである。起動直後は空なので、最初に要るときに
        # 一度だけ DB から引く。
        self._recent_cursor: str | None = None
        self._show_seeded = False
        # 求めの世代。打ち切るたびに1つ進める。**走っている反復と、飛んでいる調査の完了**を
        # 古い世代として捨てるのに使う。打ち切りの時点で外部呼び出しは既に飛んでおり、
        # 反復もフルLLM の返りを待っている最中なので、止めるには番号で見分けるしかない。
        self._request_generation = 0
        # 「まだかかっている」を受けたか。次の反復でつなぎだけ出して閉じない。
        self._slow_notice_received = False
        self._background_tasks: set[asyncio.Task] = set()
        # QA：AIFキュー（情動）。T（自律機構）が drive 発火を積む。要素＝(欲求名, 促しの内容)。
        # 3キュー（QA/QD/完了）は同じ器で待つので、待つ対象は配列で持つ（QD は1本足すだけ）。
        self._affect_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        # QD：DIFキュー（機器）。T が在席者の差分を人の出入りとして積む。
        # 要素＝(種別＝入室｜退室, 内容, 保留していた発話を配るか)。
        self._device_queue: asyncio.Queue[tuple[str, str, bool]] = asyncio.Queue()
        # 駆動体（キュー到来で次の反復を起こす）と、そこへ渡す取込待ちの完了。
        self._driver: asyncio.Task | None = None
        self._asyncio_loop: asyncio.AbstractEventLoop | None = None
        self._drained_completions: list[tuple[str, str, str | None, str, int]] = []
        # 発話が出るまでの連鎖長（発話でリセット）。上限に達した反復は recall を渡さない。
        self._iterations = 0
        self._iterations_capped = False
        self._utterance = ""
        # 反復の起点。種別＝発話｜情動｜機器｜完了。情動や機器で起きた反復には人の発話が
        # 無いので、起点の内容を手がかり・調停の入力・user メッセージに使う。
        self._trigger_kind = "発話"
        self._on_text = None
        # 発話の通知先（GUI は「発話は on_action("say") で来る」前提で作られており、
        # 素テキストは say の前の途中経過としてしか扱わない）。CUI は持たない。
        self._on_action = None
        # 投げる前に控える1件（`_start_lookup` が置き、背景タスクが読む）。
        self._pending_lookup: tuple[str, dict, str] = ("", {}, "recall")

    def _open_ids(self) -> list[str]:
        """この求めの open な記録（活性に下限を課して W へ浮かせる対象）。

        発話の記録（求めの親）と、**いま生きている版**である。版チェーンでは生きている版は
        常に1つなので、飛行中の意図を別に数える必要はない。トリガ O を求めが
        閉じるまで open 扱いにするのは、完了で起きた反復では手がかりが**届いた結果の本文**に
        変わり、元の人の問いとは語彙が重なるとは限らないためである。完了プロファイルは
        関連を厳しく要求する（w_r=1.5）ので、下限が無いと「何のために調べていたか」が
        W から落ちる。
        """
        ids = [self._request_id] if self._request_id else []
        if self._live_version_id and self._live_version_id not in ids:
            ids.append(self._live_version_id)
        return ids

    def _note_origin(self, obs_id: str | None) -> None:
        """このターンの起点を控える。

        **何に続くかはここで決めない。** 続き先は、そのターンを作るのに使った W の中に
        しかない（`_apply_follows`）。段 3 では「直前の起点へ無条件に繋ぐ」形にしていたが、
        それは鎖の種類を機構の側で数え上げることになり、並行して走る本数に上限が生まれた。
        """
        if not obs_id:
            return
        self._note_record(obs_id, "起点")

    def _close_exchange(self) -> "list[tuple[str, str]] | None":
        """いまのやりとりの区間を切り出し、次の始まりを進める。

        母集合への持ち越し（`_turn_records`）はそのまま残す。打ち切った調査と、言い直した
        問いの共起は、たどる価値があるためである。
        """
        members = self._turn_records[self._exchange_start :]
        self._exchange_start = len(self._turn_records)
        # 次のターンは、いま閉じたやりとりから見せる。
        for obs_id, role in members:
            if role == "起点":
                self._recent_cursor = obs_id
                break
        return members or None

    def _note_record(self, obs_id: str | None, role: str) -> None:
        """このターンが作った記録を、役割つきで控える。

        **一つの並びが二つの用を賄う。** 拡散想起の母集合（共起の関係）へ渡す id と、
        やりとりの関係の項が、どちらもここから出る。別々に持つと、片方へ足し忘れたときに
        気づけない。

        役割は 起点・版・見た・答え。会話要約は背景で遅れて作られるので、ここには来ない
        （`_run_post_response_pipeline` が末尾に足す）。
        """
        if obs_id and all(obs_id != i for i, _ in self._turn_records):
            self._turn_records.append((obs_id, role))

    def _advance_chain(self, new_id: str | None, content: str = "") -> None:
        """ループ記録の鎖を1つ進める（直前の生きた記録を新しい記録で supersede）。

        内容も持つのは、この先頭（取込の起点）を W へ決定的に加えるため。
        """
        if not new_id:
            return
        if self._chain_head_id and self._chain_head_id != new_id:
            self._agent._memory.mark_superseded(self._chain_head_id, new_id, kind=KIND_ADVANCE)
            logger.debug("event-loop 鎖を進める %.8s → %.8s", self._chain_head_id, new_id)
        self._chain_head_id = new_id
        self._chain_head_content = content

    async def _write_version(self, *, aborted: bool = False) -> str | None:
        """求めの新しい版を書き、直前の版を畳む。

        求めは1本の版チェーンとして進む。畳むのは版が進んだからで、種類は `改訂` である
        （`設計方針_MI間の関係`）。親子のファンアウトではないので親子をまとめて畳む操作は
        要らない（撤去済み）。

        人の発話の記録と、自分が答えた記録は**鎖の外**にある。畳まない。
        """
        agent = self._agent
        content = self._version_content(aborted=aborted)
        version_id, _ = await agent._memory.save_async_with_id(
            content[: agent.config.completion_content_max],
            direction="求め",
            kind="observation",
            materialize_now=True,
            parent_id=self._request_id,
            **agent._observation_perspective(),
        )
        if version_id:
            self._note_record(version_id, "版")
            if self._live_version_id and self._live_version_id != version_id:
                agent._memory.mark_superseded(self._live_version_id, version_id, kind=KIND_REVISION)
            self._live_version_id = version_id
            # 手がかり（次の反復の想起クエリ）は、いまの版そのものにする。
            self._chain_head_id = version_id
            self._chain_head_content = content
        return version_id

    async def _write_seen_mark(self, content: str) -> str | None:
        """見たことを O へ書く（`direction="観察"`・鎖の外・畳まない）。

        旧 `run()` がカメラを使ったターンで書いていた記録の続きである（本番に 256 件
        あり、2026-07-24 で途絶えている）。新しいループへ移るとき `camera_used` が
        渡らなくなって書き込みが到達しなくなり、見た印が失われていた。同じ意味の
        記録なので `direction` は分けない。

        旧との違いは、`see` の完了が**どの定点を見たか**を頭に付けることである
        （`_run_camera`）。定点名が入って初めて、W が「次はここを見る番だ」を選べる。
        """
        agent = self._agent
        obs_id, _ = await agent._memory.save_async_with_id(
            content[: agent.config.completion_content_max],
            direction="観察",
            kind="observation",
            materialize_now=True,
            parent_id=self._request_id,
            **agent._observation_perspective(),
        )
        # W へ載せる。版から結果を落としたので、この経路が無いと `see` した反復の
        # 次で、調停が何が見えたかを知らないまま返事を作る。
        self._note_record(obs_id, "見た")
        return obs_id

    def _version_content(self, *, aborted: bool = False) -> str:
        """いまの求めの状態を、1つの版の content として組み立てる。

        求めは1本の版チェーンとして進み、各版が状態を表す。**どの版にも求めそのものを
        入れる**（前の版は畳まれて想起の候補から外れるので、辿る道が残らない）。

        並行する調査は通し番号で列挙し、鎖は分岐させない。状態は「求め」の状態であって
        個々の調査の状態ではないので、3件飛んでいても求めの状態はひとつである。

        結果はここでは切らない。切るなら書き込みの上限で切る。ここで切ると、どこで短く
        なったのかが追えなくなる。
        """
        parts: list[str] = []
        for lk in sorted(self._lookups, key=lambda x: x.index):
            if lk.in_flight:
                verb = "を打ち切った" if aborted else "を起動中"
                parts.append(f"{lk.index}番：{lk.action}「{lk.query}」{verb}")
            else:
                parts.append(f"{lk.index}番：{lk.action}「{lk.query}」の結果が届いた：{lk.result}")

        head = f"「{self._request_text}」と聞かれた"
        if not parts:
            return head + ("（打ち切った）" if aborted else "")
        return f"「{self._request_text}」と聞かれ、" + "／".join(parts)

    @property
    def _in_flight_count(self) -> int:
        """まだ結果が届いていない調べものの数。**手で数えず、器の列から導く。**"""
        return sum(1 for lk in self._lookups if lk.in_flight)

    def _lookup_of(self, query: str) -> "Lookup | None":
        """語で1件を引く。同じ語は二度投げないので、引き当ては一意になる。"""
        return next((lk for lk in self._lookups if lk.query == query), None)

    def _next_lookup_index(self) -> int:
        """この求めの中での通し番号。**器の数から決まる**（別の変数で数えない）。"""
        return len(self._lookups) + 1

    def _dispatch_lookup(
        self, action: str, tool_input: dict, query: str, intent_id: str | None
    ) -> None:
        """RH：調べる動作を非同期に実行し、結果を QC へ積む（投げっぱなし・待たない）。"""
        # **この求めで一度調べた語は、二度と調べない。** `deferred` は自前で同じ意図を
        # 止めるが、`recall`・`see`・`look` は素通りで、実機では同じ `recall` を4反復
        # 続けて投げた（語は MD5 まで一致）。4回とも同じ記憶を取ってきて無駄になった。
        # `recall` は DB を引くだけなので、引き直しても結果は変わらず取り直しの意味がない。
        # 止めた調査は完了として積む。投げずに黙って帰ると、完了も時間切れも来ないまま
        # 飛行中の数だけが残り、駆動体が待ち続ける（`deferred` が投げられなかったときと
        # 同じ形にする）。
        seen = self._lookup_of(query)
        if seen is not None:
            logger.info("event-loop すでに調べた語なので投げない：%.40s", query)
            # **器は増やさない。** 同じ語の器が2つできると、語で引いたときどちらが返るか
            # 決まらない。飛行中の数は器から導くので、以前のように数だけ増やして釣り合いを
            # 取る必要もない（環-g・段は で挙動が変わったところ）。完了だけを積む——投げずに
            # 黙って帰ると、完了も時間切れも来ないまま駆動体が待ち続ける。
            self._completion_queue.put_nowait(
                (
                    query,
                    f"「{query}」はこの求めですでに調べた。結果は W にある。",
                    intent_id,
                    "完了",
                    seen.index,
                )
            )
            return

        index = self._next_lookup_index()
        self._lookups.append(
            Lookup(index=index, action=action, query=query, generation=self._request_generation)
        )
        task = asyncio.create_task(self._run_lookup(action, tool_input, query, intent_id, index))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        watch = asyncio.create_task(self._watch_slow_lookup(query, self._request_generation))
        self._background_tasks.add(watch)
        watch.add_done_callback(self._background_tasks.discard)

    async def _watch_slow_lookup(self, query: str, gen: int) -> None:
        """調べものが遅いとき、**1回だけ**「まだかかっている」を積む（案G-3・案イ）。

        時計で定期的に起こすのではなく、**遅いという事実**が起点になる。繰り返すと結局
        「一定時間ごとに言う」になるので、1回で終える。結果が先に来たら何もしない
        （その時点で飛行中の一覧から消えている）。
        """
        seconds = float(getattr(self._agent.config, "lookup_slow_seconds", 5.0))
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(seconds)
            if gen != self._request_generation:
                return  # 打ち切られた求めの見張り
            lk = self._lookup_of(query)
            if lk is None or not lk.in_flight:
                return  # もう結果が来ている
            logger.info("event-loop 調べものが %.0f 秒を超えた：%.40s", seconds, query)
            self._completion_queue.put_nowait((query, "", None, "進捗", 0))

    async def _run_lookup(
        self, action: str, tool_input: dict, query: str, intent_id: str | None, index: int = 0
    ) -> None:
        """`recall` は同期で結果が返る。deferred は投げるだけで、完了は自身が QC へ積む。"""
        if action in ("see", "look"):
            # 飛行中の数は減らさない。`recall` と同じく取込が1件につき1つ減らす。
            out = await self._run_camera(action, tool_input)
            self._completion_queue.put_nowait((query, out, intent_id, "完了", index))
            return
        if action != "recall":
            try:
                text, dispatched = await self._dif.lookup(action, tool_input)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("event-loop %s の実行に失敗: %s", action, e)
                self._completion_queue.put_nowait(
                    (query, f"（{action} を実行できなかった：{e}）", intent_id, "完了", index)
                )
                return
            if not dispatched:
                # 投げられなかった（クエリが空・同時実行の上限・同じ意図が進行中）。背景
                # タスクが無いので完了も時間切れも来ない。ここで閉じないと飛行中の数が
                # 戻らず、駆動体が完了キューだけを待ち続けて何も処理しなくなる。
                logger.info("event-loop %s は投げられなかった：%.60s", action, text)
                self._completion_queue.put_nowait((query, text, intent_id, "完了", index))
                return
            # 投げられた。**ここで飛行中の数を減らさない。** 減らすと、結果が返る前に
            # 「調査中ではない」ことになり、情動や機器で別の反復が起きて同じ調べものを
            # 投げ直す（実機で1つの求めに検索が4本走り、つなぎを4回喋った）。減らすのは
            # 完了の取込1点に揃える。
            return
        try:
            out, _ = await self._agent._memory_tool.call(
                "recall",
                tool_input,
                exclude_ids=[self._recall_exclude_id] if self._recall_exclude_id else None,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("event-loop recall の実行に失敗: %s", e)
            out = f"（recall を実行できなかった：{e}）"
        self._completion_queue.put_nowait((query, str(out), intent_id, "完了", index))
        logger.debug(
            "event-loop RH 完了をQCへ（id=%s qsize=%d 意図=%.8s）",
            id(self),
            self._completion_queue.qsize(),
            intent_id or "-",
        )

    async def _run_camera(self, action: str, tool_input: dict) -> str:
        """目と首を動かし、**見えたものを言葉にして**返す。

        `see` が返すテキストは「撮って保存した」と言うだけで、何が写っているかは画像の
        ほうにある。完了キューはテキストしか運ばないので、`知覚在席` §3-2 が定める
        意味づけ（I 側・必要時・VLM）を通す。首を振っただけの `look` に画像は無い。

        カメラも VLM も落ちる前提の機器なので、例外はここで畳む。見た事実まで失うと
        求めが閉じないまま残る。
        """
        agent = self._agent
        try:
            text, image_b64 = await agent._camera.call(action, tool_input)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("event-loop %s の実行に失敗: %s", action, e)
            return f"（{action} を実行できなかった：{e}）"
        if action != "see" or not image_b64:
            return str(text)
        # どの定点を見たかを記録に残す。`ユースケース③` の「見た定点の印」で、これが
        # 根づきで薄れ、W 構築で薄れた順に上がることで巡回が創発する。
        where = await self._current_pose_name()
        prefix = f"{where}を見た。" if where else ""
        try:
            entities = await extract_entities(str(text), agent._scene_backend, image_b64=image_b64)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("event-loop 見たものの意味づけに失敗: %s", e)
            return str(text)
        labels = [str(d.get("label", "")).strip() for d in entities if d.get("label")]
        if not labels:
            logger.info("event-loop 見たが、意味づけは何も返さなかった")
            return f"{prefix}{text}"
        logger.info(
            "event-loop %s見えたもの %d 件：%.60s",
            f"{where}で" if where else "",
            len(labels),
            "、".join(labels),
        )
        # 印は**見たことだけ**にする。`see` が返す "You see the current view
        # (saved to …)" は撮ったことを LLM へ伝える文で、見た内容ではない。想起は
        # 印の文でベクトルを作るので、毎回同じ英語の定型句とファイルパスが入ると
        # ノイズになる（実機で観測）。
        mark = f"{prefix}見えたもの：" + "、".join(labels)
        # **書くのはここである。** 実際にカメラを回して意味づけが通った経路だけを
        # 通る。取込の側で `action == "see"` を見て書くと、重複抑止で弾かれた完了
        # （「すでに調べた。結果は W にある」）まで印になり、見ていないのに見た印が
        # 立つ（実機で観測）。
        await self._write_seen_mark(mark)
        return f"{prefix}{text} 見えたもの：" + "、".join(labels)

    async def _current_pose_name(self) -> str:
        """いま向いている定点の名前。どの定点でもない（移動中）なら空。

        どの定点でもない向きの映像に定点名を付けると、その定点の記録が別の場所の景色で
        汚れる（`知覚在席` §3-3 の振動中ゲートと同じ理由）。
        """
        agent = self._agent
        try:
            camera = getattr(agent, "_camera", None)
            if camera is None:
                return ""
            poses = await agent.poses()
            if not poses:
                return ""
            position = await camera.position()
            if position is None:
                return ""
            pose = nearest_pose(poses, position[0], position[1], agent.config.camera.pose_tolerance)
            return pose.name if pose else ""
        except Exception:  # noqa: BLE001
            logger.debug("いまどの定点を向いているか分からなかった")
            return ""

    async def _intake(self) -> int:
        """取込：駆動体が受けた完了（と QC の残り）を O に書き、open 意図を解決する。"""
        # `_drained_completions` は作り直さず中身だけ移す。駆動体は `self._drained_completions.append(await get())` の
        # append を await の前に束縛するので、ここで差し替えると駆動体が捨てられた古い
        # リストへ積み、完了が黙って失われる（実機で観測）。
        items = list(self._drained_completions)
        self._drained_completions.clear()
        while not self._completion_queue.empty():
            items.append(self._completion_queue.get_nowait())
        logger.debug(
            "event-loop 取込（id=%s items=%d inflight=%d qsize=%d）",
            id(self),
            len(items),
            self._in_flight_count,
            self._completion_queue.qsize(),
        )

        progress = [q for q, _t, _i, kind, _x in items if kind == "進捗"]
        items = [it for it in items if it[3] != "進捗"]
        if progress:
            # 「まだかかっている」は結果ではない。飛行中の数も一覧も触らず、意図も
            # supersede しない。次の反復で、調停に短い一言を書かせるためだけに起こす。
            self._slow_notice_received = True
        for query, result_text, intent_id, _kind, _index in items:
            # 届いた結果を器へ入れる。**これで飛行中でなくなる**（数は導出）。
            lk = self._lookup_of(query)
            action = lk.action if lk is not None else "recall"
            if action == "see":
                # 版には結果を載せない。見たことは `_run_camera` が鎖の外へ独立した
                # 記録として書いており（会話の「自分が答えた」と同じ位置）、版にも
                # 載せると同じ出来事が2件になって、想起でどちらも上がり W の枠を食う。
                # 求めの状態としては「何番が届いたか」だけあればよい。
                result_text = "（見たことは観察に記録した）"
            if lk is not None:
                lk.result = result_text
        if items:
            # 求めの新しい版を書き、直前の版を畳む（1本の鎖）。
            await self._write_version()
        return len(items)

    # この反復で使える動作の表。値＝その動作のツール定義を取り出す関数で、引数はループ。
    # 身体を1つ繋ぐたびにここへ1行足すだけで済むようにしてある（see・look・net など）。
    # **口が持っている機器の定義は口が答える**（`_dif`）。まだ口を通していないものだけが
    # `ip._agent` を見る（カメラは 段3・記憶は OIF の担当）。
    _ACTIONS: dict = {
        "say": lambda ip: ip._dif.speak_defs(),
        "recall": lambda ip: [
            d for d in ip._agent._memory_tool.get_tool_definitions() if d.get("name") == "recall"
        ],
        # net（投げっぱなしの外部呼び出し）。結果は完了キュー経由で後の反復に届く。
        "search_deferred": lambda ip: ip._dif.lookup_defs("search_deferred"),
        "fetch_deferred": lambda ip: ip._dif.lookup_defs("fetch_deferred"),
        # 身体。カメラが無ければ空を返し、繋がっていない身体は渡さない。
        "see": lambda ip: _camera_tool_def(ip._agent, "see"),
        "look": lambda ip: _camera_tool_def(ip._agent, "look"),
        # 家の決まり（`obsidian-memo`）。**家族ティアだけ**を載せる——個人ティア
        # （`ask_vault_yusuke`）は話者ゲートができるまで載せない。
        "house_rules": lambda ip: ip._dif.tool_defs("get_house_rules"),
    }

    def _action_of_query(self, query: str) -> str:
        """その語をどの動作で投げたか。分からなければ recall とみなす。"""
        lk = self._lookup_of(query)
        return lk.action if lk is not None else "recall"

    def _tools(
        self, *, actions: tuple[str, ...] = ("say", "recall"), cache_tools: bool = True
    ) -> list[dict]:
        """この反復で使える動作のツール定義を返す。

        表に無い名前は黙って落とす。まだ繋いでいない身体を渡そうとしても壊れないように
        しておく（段階3 の次で see・look・search_deferred を載せる）。

        **道具の定義もキャッシュに載せる**（出-i）。安定部だけ（3,342トークン）では
        Haiku 系の最小長 4,096 に届かず**1回も効かない**。道具（2,329トークン）を載せると
        跨いで**安定部ごと全部が乗り**、1000ターン1反復で 738円 → 366円 になる。

        印は**最後の1つ**に付ける。`cache_control` は「ここまで」を意味する境目なので、
        道具の並びの末尾に付ければ**道具全体とその前の安定部**が範囲に入る。複数付けると
        区切りが増え、書き込みが増える。

        `cache_tools=False` は、**安定部だけで効くモデル**のためにある（`sonnet-5` は
        最小長が低く、道具を載せると読み出し料が増えて 581円 → 635円 と高くなる）。
        """
        defs: list[dict] = []
        for name in actions:
            build = self._ACTIONS.get(name)
            if build is None:
                logger.debug("event-loop 未接続の動作を要求された（無視する）: %s", name)
                continue
            with contextlib.suppress(Exception):
                defs.extend(build(self))
        if cache_tools and defs:
            # **共有されている定義を書き換えない。** `get_tool_definitions()` は同じ辞書を
            # 返すことがあり、そこへ印を付けると次に取ったときも残る（`cache_tools=False`
            # が効かなくなる）。最後の1つだけを写して印を付ける。
            defs[-1] = {**defs[-1], "cache_control": {"type": "ephemeral"}}
        return defs

    async def begin_request(self, utterance: str, on_text=None) -> str:
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
        self._trigger_kind = "発話"
        self._request_text = utterance[:500]
        self._live_version_id = None
        self._lookups.clear()
        self._iterations = 0
        self._iterations_capped = False
        self._ensure_driver()

        # 調べかけの途中に話しかけられたら、**その調査を打ち切る**。人が言い直したとき、
        # 前の調査を続ける意味はない（実機で「これはどこの地方の天気？」に答えられず、
        # 言い直されたあとも同じ検索を繰り返した）。結果は捨てるが、**何を打ち切ったかは
        # 記録に残す**。
        await self._abort_lookups()

        # 取込：来た事実（人の発話）を O に書く（④シーケンス）。
        trigger_id, _ = await agent._memory.save_async_with_id(
            utterance[:500],
            direction="発話",
            kind="observation",
            materialize_now=True,
            **agent._observation_perspective(),
        )
        self._request_id = trigger_id
        # このターンを起こした記録を控え、前のターンとつなぐ。控えないと、問いだけが
        # やりとりの関係にも拡散想起の母集合にも入らない。
        self._note_origin(trigger_id)
        # 発話の記録は**鎖の外**。版チェーンは `_write_version` が別に進める。
        self._chain_head_id = trigger_id
        self._chain_head_content = utterance[:500]
        return await self._iterate()

    async def _abort_lookups(self) -> None:
        """飛行中の調査を打ち切る（人に話しかけられたとき）。

        飛行中のツール呼び出しを止め、まだ取り込んでいない完了を捨て、**何を打ち切ったかを
        O に残す**。その記録で親と生きた子を閉じるので、鎖は「打ち切った」1件へ収束する。

        結果を捨てるのは、行き先の親が閉じるためで、残すと次の求めの W に無関係な完了が
        載る。ただし**打ち切った事実は残す**（あとで「あのとき何を調べていたか」を辿れる）。
        """
        in_flight = [lk for lk in self._lookups if lk.in_flight]
        if not self._background_tasks and not in_flight and self._request_id is None:
            return
        dropped = [f"{lk.index}番：{lk.action}「{lk.query}」" for lk in in_flight]
        for task in list(self._background_tasks):
            task.cancel()
        self._background_tasks.clear()
        drained = 0
        while not self._completion_queue.empty():
            self._completion_queue.get_nowait()
            drained += 1
        drained += len(self._drained_completions)
        self._drained_completions.clear()

        self._request_generation += 1
        if dropped or drained:
            logger.info(
                "event-loop 調べかけを打ち切る（%s／取り込まなかった完了 %d件）",
                "・".join(dropped) or "投げた先なし",
                drained,
            )
        if self._request_id:
            # 打ち切りも版のひとつ。何を打ち切ったかは版の content が持つので、
            # **飛行中の一覧を消す前**に、かつ親を捨てる前に書く。
            with contextlib.suppress(Exception):
                await self._write_version(aborted=True)
            # ここまでが一つのやりとりである。打ち切りの版の親は、この求めの起点だから
            # である。閉じないと並びが次のターンへ持ち越され、一つのやりとりに起点が2つ
            # 入る。答えも要約も無いやりとりになるが、それが起きた事実そのものである。
            _aborted = self._close_exchange()
            if _aborted:
                with contextlib.suppress(Exception):
                    self._agent._memory.record_exchange(
                        [(i, r, n) for n, (i, r) in enumerate(_aborted)]
                    )
        self._request_id = None
        self._live_version_id = None
        self._lookups.clear()
        self._chain_head_id = None
        self._chain_head_content = ""
        self._said_fillers.clear()
        self._speech_to_deliver.clear()
        self._w_id_map = {}

    def _compose_workspace(self, mem, memories: list[dict]) -> str:
        """W を組む。候補集合を1本の経路で通し、枠に入るぶんだけ載せる。

        正本 [D-想起起動] は「O に乗った後は共通の流れ（O → 根づき → W 構築〔5軸採点〕→
        調停）で1本」と定める。以前は想起で拾った記録だけが採点を通り、ループ自身が O へ
        書いた記録（意図 O・完了 O）は採点を通らず手組みの文字列として連結されていた。
        そのため記録が W に載るかどうかが「畳むか畳まないか」で決まり、優先度の計算が
        どこにも効いていなかった。手組みをやめ、中身は候補集合の一員として入る。

        **1件の途中では切らない。** 枠（`workspace_max_chars`）を超えたら適合度の低い件から
        丸ごと落とす。切ると調べた結果の枕だけが残って中身が消える（実機で
        `「目の前を見る」を see で調べた結果が届いた：` だけが W に載った）。

        あわせて、W に出した id（12桁）と完全な id の**対応表**を作る。フルLLM の申告を
        突き合わせるのに使う。前方一致で当てずっぽうに引くと、写し間違いが黙って別の記憶へ
        適用されてしまう。

        `said`（言ったつなぎ）と `held`（配る保留）は手組みのまま残す。どちらも O にあるが、
        `held` は `pending_store` が鮮度と配達を管理しており、想起とは別の規則を持つ。
        """
        from ..config import MemoryConfig

        budget = MemoryConfig().workspace_max_chars

        # 適合度の高い順に、枠へ入るぶんだけ採る。落ちたものは薄れた＝忘れたのであって、
        # 抜けを検出する仕組みは置かない（W は速く薄れる・改めて調べるのが自然な振る舞い）。
        ranked = sorted(memories, key=lambda m: float(m.get("fit", 0.0)), reverse=True)
        kept: list[dict] = []
        used = 0
        for m in ranked:
            size = len(str(m.get("summary", "")))
            if kept and used + size > budget:
                continue
            kept.append(m)
            used += size
        dropped = len(memories) - len(kept)
        if dropped:
            # 何件落ちたかを残す。枠に収まったのか溢れたのかが分からないと、枠の値を
            # 決められない。記憶の内容は出さない。
            logger.info(
                "event-loop W に入らなかった記録＝%d件（枠 %d 字・載せた %d 件）",
                dropped,
                budget,
                len(kept),
            )
        # 想起が返した順（適合度の降順）を保つ。並べ替えた結果をそのまま渡す。
        memories = kept

        self._w_id_map = {
            str(m.get("memory_id", "")).replace("-", "")[:12]: str(m.get("memory_id", ""))
            for m in memories
            if m.get("memory_id")
        }
        # すでに相手へ伝えた一言。これが無いと、同じ言い回しを最初から言い直す
        # （実機で「〜ですね！」で始まる前置きが3回続いた）。
        said = ""
        if self._said_fillers:
            lines = "\n".join(f"- 「{t}」" for t in self._said_fillers)
            said = (
                "すでに相手へ伝えた一言（言った順。次に何か言うなら、"
                "同じ言い回しを繰り返さず、この続きとして自然につなぐ）：\n" + lines
            )
        held = ""
        if self._speech_to_deliver:
            held = (
                "聞く相手が居ないあいだに話したかったこと"
                "（いま伝えるなら、そのときのこととして話す）：\n"
                + "\n".join(self._speech_to_deliver)
            )
        return "\n\n".join(
            p for p in [said, held, mem.format_for_context(memories)] if p and p.strip()
        )

    def _recent_ctx(self, follows: "str | None") -> str:
        """直近のやりとりを逐語で組む（段 4）。

        **続きでなければ載せない。** 判定（`根拠台帳` §29）が続き先を返さなかったターンは、
        新しい話の始まりである。前のやりとりを載せると、関係のない会話が文脈に混ざる。

        **切らない。** W は 120 字で切るが、細部が要るからこの設計にしたので、ここで
        縮めると意味がない。O の書き込み上限が 500 字なので、1件あたり最大 500 字である。

        起点は「直前に閉じたやりとり」である。**このターンの起点からは辿れない。** まだ
        どのやりとりにも属していない（やりとりを書くのは反復が閉じたあと）。
        """
        if not follows:
            return ""
        agent = self._agent
        # 判定が続き先を返した。その辺は `_link_follows` が書く。
        self._link_follows(follows)
        if not self._show_seeded:
            self._show_seeded = True
            with contextlib.suppress(Exception):
                self._recent_cursor = agent._memory.latest_exchange_origin()
        if not self._recent_cursor:
            return ""
        rows: list = []
        with contextlib.suppress(Exception):
            rows = agent._memory.recent_exchanges(self._recent_cursor)
        if not rows:
            return ""
        lines = []
        for r in rows:
            when = clock.ts_to_time(r.get("timestamp"))
            who = "わたし" if str(r.get("role")) in ("答え", "つなぎ") else "相手"
            lines.append(f"- {when} {who}：{r.get('content', '')}")
        return "[直近のやりとり（古い順）]\n" + "\n".join(lines)

    def _link_follows(self, full: "str | None") -> None:
        """判定が返した続き先へ、継起の辺を張る（`根拠台帳` §29）。

        **W に無い id は捨てる。** 判定は12桁の形で返すが、実在するかまでは見ていない。
        突き合わせは `memory_verdicts` と同じ対応表を通す。

        自分の起点を指しても繋がない。自己ループはさかのぼりが止まらなくなる。
        """
        if not full or not self._request_id or not self._w_id_map:
            return
        if full not in set(self._w_id_map.values()) or full == self._request_id:
            return
        logger.info("event-loop このターンは %.8s に続く", full)
        with contextlib.suppress(Exception):
            self._agent._memory.record_succession(full, self._request_id)

    def _apply_memory_verdicts(self, raw) -> None:
        """フルLLM が申告した「想起した記憶の扱い」を反映する（課題5 E節 段2）。

        **照合できたものだけ適用する**。指示しても、落としたり無い id を足したりする。
        欠けた分を「使わなかった」と決めつけると、申告漏れと本当に使わなかったことを
        混同する。件数をログに残し、指示が守られているかを後から確かめられるようにする。
        """
        if not raw or not self._w_id_map:
            return
        verdicts: dict[str, str] = {}
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            full = self._w_id_map.get(str(item.get("id", "")).replace("-", "")[:12])
            verdict = str(item.get("verdict", "")).strip().lower()
            if full and verdict in ("important", "useless", "referred", "unused"):
                verdicts[full] = verdict
        logger.info("event-loop 記憶の判定 %d/%d 件", len(verdicts), len(self._w_id_map))
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

    def push_completion(self, query: str, result: str, index: int = 0) -> None:
        """RH（資源ハンドラ）が deferred の完了を QC へ積む。

        投げっぱなしの外部呼び出し（検索・取得）の結果は、完了キュー→O 経由で次の反復の
        入力になる（正本③）。スレッドから呼ばれても届くよう、ループへ委譲する。
        """
        # 打ち切った求めの完了は捨てる。外部呼び出しは投げた時点で飛んでおり、止められない。
        _lk = self._lookup_of(query)
        if _lk is not None and _lk.generation != self._request_generation:
            logger.info("event-loop 打ち切った求めの完了なので捨てる：%.40s", query)
            return
        loop = getattr(self, "_asyncio_loop", None)
        item = (
            query,
            str(result),
            None,
            "完了",
            index or (_lk.index if _lk is not None else 0),
        )
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
            self._asyncio_loop = asyncio.get_running_loop()
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
                # 3つのキューは要素の形が違う（完了は4つ組、情動は2つ組、機器は3つ組）。
                # union 待ちのあいだは形を問わないので、ここでは要素型を見ない。取り出した
                # 後、どのキューから来たかで分岐して形を確定させる。
                queues: list[asyncio.Queue[Any]] = (
                    [self._completion_queue]
                    if self._in_flight_count
                    else [self._completion_queue, self._affect_queue, self._device_queue]
                )
                waiters = {asyncio.ensure_future(q.get()): q for q in queues}
                try:
                    done, pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    pass
                for task in pending:
                    task.cancel()
                affect: Any = None
                device: Any = None
                for task in done:
                    item = task.result()
                    q = waiters[task]
                    if q is self._completion_queue:
                        self._drained_completions.append(item)
                    elif q is self._affect_queue:
                        affect = item
                    else:
                        device = item
                # 同じキューに溜まっている分もまとめて取る。
                while not self._completion_queue.empty():
                    self._drained_completions.append(self._completion_queue.get_nowait())
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
                        id(self),
                        len(self._drained_completions),
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
        self._trigger_kind = "情動"
        self._request_text = f"[内的な促し:{drive_name}] {prompt}"[:500]
        self._live_version_id = None
        self._lookups.clear()
        self._iterations = 0
        self._iterations_capped = False
        content = f"[内的な促し:{drive_name}] {prompt}"
        obs_id, _ = await agent._memory.save_async_with_id(
            content[:500],
            direction="情動",
            kind="observation",
            materialize_now=True,
            **agent._observation_perspective(),
        )
        self._request_id = obs_id
        self._note_origin(obs_id)
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
        self._trigger_kind = "機器"
        self._request_text = f"[{kind}] {content}"[:500]
        self._live_version_id = None
        self._lookups.clear()
        self._iterations = 0
        self._iterations_capped = False
        text = f"[{kind}] {content}"
        obs_id, _ = await agent._memory.save_async_with_id(
            text[:500],
            direction="機器",
            kind="observation",
            materialize_now=True,
            **agent._observation_perspective(),
        )
        self._request_id = obs_id
        self._note_origin(obs_id)
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
                    released.append(
                        f"- {_elapsed_label(row.get('created_at'), now_epoch)}：{content}"
                    )
                store.delete(row["id"])
                with contextlib.suppress(Exception):
                    self._agent._memory.mark_superseded(
                        row["observation_id"], self._request_id, kind=KIND_RESOLVE
                    )
            self._speech_to_deliver = released
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
        # この反復が属する世代。打ち切られたら（世代が進んだら）、フルLLM の返りを待って
        # いる最中でも、出力せずに畳む。実機で、打ち切った直後に走っていた反復が
        # fetch_deferred を投げ、返事も1つ余計に出た。
        gen = self._request_generation
        max_chain = max(1, agent.config.event_max_iterations)
        self._iterations += 1
        chain = self._iterations
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
        # **取込 O を候補から外さない。** 手がかりは取込の content そのものなので、候補に
        # 入れば必ず上位に来る。以前はこれを「枠を食う」と嫌って外していたが、いま届いた
        # 結果を全文で見せる必要がある以上、1位に来るのが正しい順位である。手組みで W へ
        # 足すのをやめ、候補集合の一員として同じ採点を通す（正本 [D-想起起動] の1本の流れ）。
        cue = self._chain_head_content or utterance
        _mcfg = MemoryConfig()
        # 5軸の重みは trigger 種別で決める（`課題5_パラメータ仮案` §280）。選ぶ基準は
        # 「この求めを何が始めたか」ではなく **「この反復を何を手がかりに動くか」**である。
        # 反復1の手がかりは人の言葉だが、完了が届いて起きた反復の手がかりは結果の本文で、
        # 性質が違う。`_trigger_kind` を書き換えないのは、そちらが静穏時間のゲート
        # （`_should_hold`）に使われており、人に話しかけられて始まった求めを夜間に
        # 保留させてしまうためである。
        trigger = "完了" if drained else self._trigger_kind
        w_base = _mcfg.recall_weights(trigger)
        weights = _mcfg.jitter_weights(w_base)
        # 床（min_score）を渡す。渡さないと既定 0.0 で床が効かず、無関係な記録まで W の枠を
        # 埋める。床は正本 [D-想起合成] が「無関係排除の主たる足切り」と定めるもので、
        # 連想想起（`agent.py`）は既に渡していた。イベントループだけが渡していなかった。
        memories = await mem.recall_async(
            cue,
            n=_mcfg.recall_k,
            min_score=_mcfg.recall_min_score,
            weights=weights,
            open_ids=self._open_ids(),
        )
        _log_recall_weights(trigger, w_base, weights, memories)
        # W は「思い出している記憶」ではなく、いまの作業状態。ループ自身の行動も MI として
        # O にあるので、合成ラベル（[取込]・[調査中]）は作らず MI をそのまま並べる。
        # W から落ちたものは薄れた＝忘れたのであって、抜けを検出する仕組みは置かない
        # （W は「速く薄れる」・改めて調べるのが自然な振る舞い）。
        workspace_ctx = self._compose_workspace(mem, memories)
        # 続き先の判定を投げる。**待たずに先へ進む。** 調停と並行して走らせれば、
        # 実測 0.72 秒（`根拠台帳` §29）はほぼ隠れる。受け取るのはシステム文を組む
        # 直前で、そこは待つ（続きでなければ直近のやりとりを載せてはいけない）。
        follows_task = asyncio.ensure_future(
            agent._evaluator.judge_follows(workspace_ctx, utterance or "")
        )

        # 誰と話していると思って喋ったかを残す。これが無いと、口調がおかしいときに
        # 「話者が渡っていない」のか「渡ったが口調が従っていない」のかを切り分けられない。
        present_ctx = _present_ctx(agent)
        logger.debug("event-loop iter=%d/%d 在席=%s", chain, max_chain, present_ctx)

        capped = chain >= max_chain
        if capped:
            # 上限で打ち切ったことは、後からログだけで判別できる必要がある（DEBUG の
            # iter=N/M からは「たまたま N 回で終わった」のか「打ち切った」のか分からない）。
            logger.info("event-loop 反復 %d/%d 上限に達したため探索を打ち切る", chain, max_chain)
            self._iterations_capped = True
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
        logger.debug(
            "event-loop iter=%d/%d 調停=%s effort=%s",
            chain,
            max_chain,
            decision.branch,
            decision.effort,
        )
        # 「いまは話しかけないで」と読めたら、その人が居るあいだ黙る。この反復の受け答えは
        # 出したうえで（頼みに無言で応じるのは不自然）、次の反復から止める。
        if decision.silence_minutes:
            self._accept_silence(decision.silence_minutes)
        # 調停が時期を指した（「去年の夏の話」）なら、その基準で想起し直して W を組み直す。
        # 想起は調停より前に走るので、この反復に効かせるには引き直すしかない。実測 17〜50ms
        # で、指定があったときだけ走る。
        if decision.time_ref:
            with contextlib.suppress(Exception):
                ref = datetime.fromisoformat(decision.time_ref).timestamp()
                span = decision.time_span_days or None
                memories = await mem.recall_async(
                    cue,
                    n=_mcfg.recall_k,
                    time_ref=ref,
                    time_span_days=span,
                    min_score=_mcfg.recall_min_score,
                    weights=weights,
                    open_ids=self._open_ids(),
                )
                workspace_ctx = self._compose_workspace(mem, memories)
                logger.info(
                    "event-loop 想起の基準を移す：%s（幅 %s 日）",
                    decision.time_ref,
                    decision.time_span_days or "既定",
                )

        if gen != self._request_generation:
            logger.info("event-loop 打ち切られた求めの反復なので畳む（調停後）")
            return ""

        # 「まだかかっている」で起きた反復は、**つなぎだけ出して閉じない**。求めは調査待ちの
        # まま続く。ここで light を選ばせると別の答えを出して終わってしまい、あとから届く
        # 結果に行き場が無くなる（案ハ）。
        if self._slow_notice_received:
            self._slow_notice_received = False
            await self._say_filler(decision.text)
            logger.info("event-loop 反復 %d/%d 出力=つなぎ（調べもの待ち）", chain, max_chain)
            return ""

        # (a) 軽量で閉じる：フルLLM を起こさず、軽量LLM の応答で反復を終える。
        if decision.branch == "light" and decision.text:
            return await self._speak(decision.text, memories)

        # (c) 定型：探すと決まっている反復も、フルLLM を起こさず投げて閉じる。
        if decision.branch == "action" and decision.query and not capped:
            # つなぎの一言はここで即出す（フルLLM を経由しないぶん速い・正本③ 段5 の内部二段）。
            await self._say_filler(decision.text)
            self._start_lookup(
                utterance or self._chain_head_content,
                {"query": decision.query},
                action=decision.action,
            )
            logger.info(
                "event-loop 反復 %d/%d 出力=%s（調停・続きは完了で起きる）",
                chain,
                max_chain,
                decision.action,
            )
            return ""

        # (b) 軽量つなぎ→フル（正本③ 段5 の内部二段）。フル生成は effort=high で10秒近く
        # かかり、そのあいだ無音になる。つなぎで体感の待ち時間を埋める。**1つの work の
        # 内部二段**であって別の出力ではない（1反復1出力は保たれる）。
        # effort=low は実測 0.8〜3.6 秒で返るので挟まない（かえってテンポが悪くなる）。
        # **材料が届いた反復でも挟まない**（`drained`）。待つものがもう無いのに「待って」と
        # 言う理由がない。実機では、検索結果が届いた1秒後に「うん、任せてね！」が出て、
        # 一言目（ですます）と本応答（ですます）のあいだでそこだけ口調が割れた。
        if decision.branch == "full" and decision.text and decision.effort != "low" and not drained:
            await self._say_filler(decision.text)

        # 整合チェックにも同じものを渡すので、いったん変数へ出す。
        recent_ctx = self._recent_ctx(await _result_or_none(follows_task))
        system = build_event_system_prompt(
            self_understanding=load_summary() or getattr(agent, "_me_md", ""),
            family_md=getattr(agent, "_family_md", ""),
            present_ctx=present_ctx,
            pi_ctx=_pi_ctx(),
            recent_ctx=recent_ctx,
            iter_ctx=(
                f"[反復] {chain}/{max_chain}"
                # 上限では、黙って手持ちで繕わず「調べきれなかった」と断ってから答える。
                # 断りが無いと、材料不足のまま答えたことが相手に伝わらない。
                + (
                    "（これ以上は調べられない。調べきりたかったが上限に達したことを述べ、"
                    "そのうえで現時点で分かることを返す）"
                    if capped
                    else ""
                )
            ),
            workspace_ctx=workspace_ctx,
            # 角括弧タグを許すかは合成の担い手が決める（`根拠台帳` §9）。
            allow_tts_tags=self._dif.understands_tags,
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
        lookup_tc = (
            None
            if capped
            else next((tc for tc in result.tool_calls if tc.name in _LOOKUP_ACTIONS), None)
        )

        # 発話と動作が一緒に来たら、発話はつなぎとして出し、その反復の出力は動作とする。
        # 以前は say を見つけた時点で閉じており、同じ応答に入っていた検索を捨てていた。
        if gen != self._request_generation:
            logger.info("event-loop 打ち切られた求めの反復なので畳む（生成後）")
            return ""

        if lookup_tc is not None:
            logger.debug("event-loop iter=%d/%d 決定=%s", chain, max_chain, lookup_tc.name)
            if say_tc is not None:
                await self._say_filler(str(say_tc.input.get("text", "")).strip())
            self._start_lookup(
                utterance or self._chain_head_content, dict(lookup_tc.input), action=lookup_tc.name
            )
            logger.info(
                "event-loop 反復 %d/%d 出力=%s（続きは完了で起きる）",
                chain,
                max_chain,
                lookup_tc.name,
            )
            return ""

        if say_tc is not None:
            logger.debug("event-loop iter=%d/%d 決定=say", chain, max_chain)
            self._apply_memory_verdicts(say_tc.input.get("memory_verdicts"))
            text = str(say_tc.input.get("text", "")).strip()
            violation = await self._coherence_violation(text, recent_ctx, memories)
            if violation:
                # **1回だけ**言い直させる。直した応答は検査しない（際限なく往復させない）。
                # 差し戻しは新しい1通で投げる。say の tool_use を含む往復をそのまま組むと、
                # 結果を返さないまま次を送ることになり backend が受け付けない。
                logger.info("event-loop 整合チェックが違反を捕まえた：%s", violation)
                retry, _raw2 = await agent.backend.stream_turn(
                    system=system,
                    messages=[
                        agent.backend.make_user_message(
                            f"{utterance or self._chain_head_content}\n\n"
                            f"[SELF-CHECK] いま言おうとした「{text}」には問題がある："
                            f"{violation}\nこれを直して、もう一度 say() で答える。"
                        )
                    ],
                    tools=self._tools(actions=("say",)),
                    max_tokens=agent.config.max_tokens,
                    on_text=None,
                    effort=decision.effort,
                )
                retry_tc = next((tc for tc in retry.tool_calls if tc.name == "say"), None)
                if retry_tc is not None:
                    self._apply_memory_verdicts(retry_tc.input.get("memory_verdicts"))
                    text = str(retry_tc.input.get("text", "")).strip() or text
                else:
                    logger.info("event-loop 言い直しが say を返さなかったので元の応答で出す")
            return await self._speak(text, memories)

        # どちらも無ければ素テキストへフォールバック（表示はここで1回）。
        logger.debug("event-loop iter=%d/%d 決定=none", chain, max_chain)
        text = (result.text or "").strip()
        if text:
            self._emit(text)
        await self._finish(text, memories, "沈黙")
        return text

    async def _coherence_violation(
        self, text: str, recent: str, memories: list[dict]
    ) -> "str | None":
        """発話の前に規則違反を見る（出-f）。違反の説明を返す。無ければ None。

        **応答の文字列を機械で削らない。** 機械が出すのは、見たか・記憶が載ったかという
        推測の要らない事実だけで、規則に反するかどうかの判断は軽量LLM がする。
        """
        agent = self._agent
        if not agent.config.coherence_check or not text:
            return None
        saw = any(role == "見た" for _, role in self._turn_records)
        return await agent._evaluator.check_response_coherence(
            text, recent=recent, facts=facts_ctx(saw=saw, memories=memories)
        )

    async def _speak(self, text: str, memories: list[dict]) -> str:
        """発話して反復を閉じる。聞く相手が居なければ話さず、後で話すために溜める。

        身体を持つ以上、発話は相手が居て初めて意味を持つ（正本③ の配信ゲート＝結果有り＋在席）。
        居ないときは「話したかったができなかった」を O に残して `pending_speech` へ積み、
        次に人が現れたときに気づけるようにする。溜めたものの寿命（鮮度切れ・参照先 supersede で
        失効）は `pending_speech` 側が持つ。
        """
        if not text:
            await self._finish("", memories, "沈黙")
            return ""
        blocked = self._delivery_block_reason()
        if blocked:
            await self._hold_speech(text)
            logger.info("event-loop %s ので発話を保留し pending_speech へ積む", blocked)
            await self._finish("", memories, "保留")
            return ""
        await self._dif.speak(text)
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
        await self._dif.speak(text)
        self._emit(text)
        # 言ったことを覚えておく。覚えないと、調停は「もう一言伝えた」ことを知らないまま
        # 同じことをまた言う（実機で1秒差に同じ文が2回出た）。抑止で黙らせるのではなく、
        # 判断できる材料を渡して解く。
        #
        self._said_fillers.append(text)
        # **O へ書く**（段 4）。054 で外したのは、想起の候補を食うからだった。役割が
        # 「想起に出さない」を担う形になったので、項として持ちながら想起から外せる。
        # 書かないと、相手が聞いた会話とパジュが読み返す会話が食い違う。
        obs_id, _ = await agent._memory.save_async_with_id(
            f"つなぎに言った：{text}"[:500],
            direction="発話",
            kind="observation",
            materialize_now=True,
            parent_id=self._request_id,
            **agent._observation_perspective(),
        )
        self._note_record(obs_id, "つなぎ")

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
        if self._trigger_kind != "発話":
            with contextlib.suppress(Exception):
                if agent._in_quiet_hours():
                    return "静穏時間である"
        return ""

    def _accept_silence(self, asked_minutes: int) -> None:
        """黙っている依頼を受ける。宛先は、いま話している相手。

        `asked_minutes` は調停が読み取った分数で、`-1` は「頼まれたが長さの指定なし」。
        既定と上限は Config が持つ。
        """
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
            from ..silence_state import resolve_minutes

            minutes = resolve_minutes(
                asked_minutes,
                default=max(1, int(getattr(agent.config, "silence_minutes", 15))),
                maximum=max(1, int(getattr(agent.config, "silence_max_minutes", 60))),
            )
            if minutes <= 0:
                return
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

    def _start_lookup(self, utterance: str, tool_input: dict, *, action: str = "recall") -> None:
        """open 意図を O に残し、RH へ投げる（待たない）。意図は常に高々1件に保つ。"""
        self._pending_lookup = (utterance, tool_input, action)
        self._background_tasks.add(t := asyncio.create_task(self._dispatch_and_write_version()))
        t.add_done_callback(self._background_tasks.discard)

    async def _dispatch_and_write_version(self) -> None:
        _utterance, tool_input, action = self._pending_lookup
        query = _query_label(action, tool_input)
        # **先に投げる。** 版の content は飛行中の一覧を含むので、投げてから書かないと
        # 「1番：… を起動中」が入らない。投げられなかった場合（重複など）は完了が積まれ、
        # 次の取込で版が書かれる。
        self._dispatch_lookup(action, tool_input, query, None)
        version_id = await self._write_version()
        # 自分が出した検索が、いま書いたばかりの版そのものを拾わないように狭く除外する。
        # 版の content は語を丸ごと含むので、その語で探せば必ず上位に来る。
        self._recall_exclude_id = version_id
        # 発話の記録は鎖の外だが、検索を始めたことはそこからも辿れたほうがよい。
        # 一度だけ足す（何を調べているかは版の側が持つ）。埋め込みの作り直しを含むので
        # 別スレッドへ逃がす（同期呼び出しでイベントループを止めない）。
        if self._request_id:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._agent._memory.note_lookup_started, self._request_id)

    async def _finish(self, text: str, memories: list[dict], outcome: str) -> None:
        """発話で連鎖が閉じた反復の後始末：総括ログと永続化（ループ中 O を supersede）。"""
        agent = self._agent
        logger.info(
            "event-loop 終了: 反復=%d 結末=%s 上限到達=%s text_len=%d",
            self._iterations,
            outcome,
            "はい" if self._iterations_capped else "いいえ",
            len(text),
        )
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
                    parent_id=self._request_id,
                    **agent._observation_perspective(),
                )
        self._note_record(answer_id, "答え")
        # **自分が答えた記録は鎖の外**。何も畳まない。求めの版チェーンは、最後の版
        # （結果が届いた状態）のまま残る。まとめ知識の MI を作る場合は、それが最後の版を
        # 畳む（未実装・`設計方針_求めの版チェーン`）。
        parent_id, self._request_id = self._request_id, None
        self._live_version_id = None
        self._chain_head_id = None
        self._lookups.clear()
        self._said_fillers.clear()
        self._speech_to_deliver.clear()
        self._iterations = 0
        self._iterations_capped = False
        # 母集合とやりとりへ渡す分を取り出してから捨てる（渡す前に消すと空で渡る）。
        # やりとりは**区間**、母集合は**全部**である（打ち切りの分も次へ持ち越している）。
        noted = self._close_exchange() or []
        turn_records = [i for i, _ in self._turn_records]
        self._turn_records, self._exchange_start = [], 0
        try:
            origin = self._utterance or self._chain_head_content
            arousal = await agent._turn_arousal(origin, text)
            agent._spawn_background_task(
                agent._run_post_response_pipeline(
                    user_input=origin,
                    final_text=text,
                    camera_used=False,
                    camera_image=None,
                    observation_action_name=None,
                    observation_action_input=None,
                    companion_mood="engaged",
                    is_desire_turn=False,
                    desires=None,
                    arousal=arousal,
                    memories=memories,
                    close_parent_id=parent_id,
                    # このターンの記録を、順序つきの一つのやりとりとして残す。会話要約は
                    # 背景で作られるので、向こうで末尾に足す。
                    exchange=noted or None,
                    # ループが作った記録も拡散想起の母集合へ。載せないと、閉じた逐語へ
                    # 辿り着く辺が WR に無い（実機で、逐語の WR 掲載数が0だった）。
                    extra_cooccurring_ids=turn_records,
                ),
                name="event-post-response",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("event-loop persistence spawn failed: %s", e)
