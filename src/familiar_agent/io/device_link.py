"""端末の口（端-b 段 1-ろ-4・課題9 §6.6）：端末の言葉を受け、返事を送り返す待ち受け。

端末（家族の Android のアプリ）は `/device?name=端末の名前` に WebSocket でつなぐ。待ち受けは入っている `aiohttp` の
`web` で立てる（依存は増えない）。番地とポートは `DeviceLinkConfig`（既定は機体の中だけ・外から入れる道は端-d）。

- つないだら、声の設定（`{"type": "hello", "voice": {"model": …, "voice": …}, "busy": …}`）を送る。返事の声は端末が合成に
  頼む（課題9 §4 の 3 の続き）。鍵は送らない。
- 窓の持ち主が変わったら（`window`・状態図から I を通って届く）、各端末へ `{"type": "window", "busy": …}` を送る（端-b 段
  1-は）。`busy` は、窓が開いていて持ち主がその端末でないとき真で、端末は話す画面をグレーにする。その端末の値が変わった
  ときだけ送り、誰と話しているかは送らない。
- 端末から `{"type": "say", "text": …, "speaker": …}` が届いたら、入口（`push`）へ端末の言葉として流す。入口は最初の反復が
  終わるまで返らないので、読む輪では待たずにタスクで流す。
- 返事は受け手（`send`・`InformationProcessing.set_device_listener` に登録する）が受け、その端末へ渡った順に送る。種類は
  say（`text` は声にする字、`display` は画面の字）・cue・closed・refused（`event_loop.set_device_listener`）。
- 読めない字・知らない種類はログに残して捨て、つながりは切らない。空か長すぎる言葉は断ったと返す（端末が待ち続けない）。同じ名前の 2 本目がつながれば、古いほうを
  閉じる（仮）。入口を待つ言葉は 1 台につき 3 つまでで、それより多ければ断ったと返す（仮）。
- Origin を付けたつなぎ（ブラウザのページ）は断る（仮）。家族の端末かどうかを確かめる鍵は、外から入れる道（端-d）と一緒に考える。

ログには端末の名前を印（`redact.tag`）で、言葉を字数で出す（家族の言葉と名前を出さない）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, Callable, Coroutine

from aiohttp import WSMsgType, web

from ..core.redact import chars, tag

logger = logging.getLogger(__name__)

#: 入口：`push(端末の名前, 言葉, 押した話者)`。
Push = Callable[[str, str, str], Coroutine[Any, Any, Any]]

_NAME_MAX = 32  # 端末の名前と話者の名前の長さの上限（仮）
_PENDING_MAX = 3  # 1 台が入口を待たせておける言葉の数（仮）。多すぎる言葉で家の列を埋めない


class _Link:
    """つながっている端末 1 台。送りは 1 本の鎖にして、渡った順に届ける。"""

    def __init__(self, ws: web.WebSocketResponse) -> None:
        self.ws = ws
        self.tasks: set[asyncio.Task] = set()  # 送りの鎖（閉じるときに止める）
        self._last: "asyncio.Task | None" = None
        self.busy = False  # 最後に知らせた「話せない」（変わったときだけ送る）

    def send(self, data: dict) -> None:
        task = asyncio.get_running_loop().create_task(self._send_after(self._last, data))
        self._last = task
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def stop(self) -> None:
        for task in list(self.tasks):
            task.cancel()

    async def _send_after(self, before: "asyncio.Task | None", data: dict) -> None:
        if before is not None:
            await asyncio.wait({before})  # 前の送りの成否は見ない。自分が止められたら止まる
        if self.ws.closed:
            return
        try:
            await self.ws.send_json(data)
        except Exception:  # noqa: BLE001
            logger.warning("device-link 端末へ送れなかった（%s）", data.get("type"), exc_info=True)


class DeviceLink:
    """端末の口。`start` で待ち受け、`close` で止める。何度呼んでも二重には立たない。"""

    def __init__(
        self, push: Push, voice: dict, *, host: str, port: int, max_chars: int = 2000
    ) -> None:
        self._push = push
        self._voice = {"model": voice.get("model", ""), "voice": voice.get("voice", "")}
        self._host = host
        self._port = port
        self._max_chars = max_chars
        self._links: dict[str, _Link] = {}
        self._pending: dict[
            str, int
        ] = {}  # 端末の名前ごとの、入口を待っている言葉の数（つなぎ直しても数え続ける）
        self._closing = False
        self._pushes: set[asyncio.Task] = set()
        self._runner: "web.AppRunner | None" = None
        self.port = 0  # 実際に待ち受けているポート（ポート 0 を渡したとき用）
        #: 最後に知らせた窓の持ち主（家は空・閉じていれば None）。同じ値は送り直さず、つないだ端末には hello に添える。
        self._owner: "str | None" = None

    async def start(self) -> bool:
        """待ち受ける。番地を取れなければ WARNING を残して偽を返す（起動は止めない）。"""
        if self._runner is not None:
            return True
        self._closing = False
        app = web.Application()
        app.router.add_get("/device", self._handle)
        runner = web.AppRunner(app, handle_signals=False)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        try:
            await site.start()
        except OSError:
            logger.warning(
                "device-link 端末の口を開けなかった（%s:%s）", self._host, self._port, exc_info=True
            )
            await runner.cleanup()
            return False
        self._runner = runner
        server = getattr(site, "_server", None)
        sockets = getattr(server, "sockets", None) or []
        self.port = sockets[0].getsockname()[1] if sockets else self._port
        logger.info("device-link 端末の口を開けた（%s:%s）", self._host, self.port)
        return True

    async def close(self) -> None:
        runner, self._runner = self._runner, None
        self._closing = True  # 閉じているあいだに来たつなぎと言葉は受けない
        queued = [t for link in self._links.values() for t in link.tasks]
        if queued:
            await asyncio.wait(
                queued, timeout=1.0
            )  # 止まる求めの最後の返事と「閉じた」を送り終えてから
        for link in list(self._links.values()):
            link.stop()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(link.ws.close(), 2.0)
        self._links.clear()
        tasks = list(self._pushes)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=2.0)
        if runner is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(runner.cleanup(), 3.0)

    def send(self, device: str, kind: str, payload: dict) -> None:
        """受け手（同期）：その端末へ送る。つながっていなければ捨てる（家では鳴らさない）。"""
        link = self._links.get(device)
        if link is None:
            logger.info("device-link つながっていない端末（%s）への%sを捨てた", tag(device), kind)
            return
        link.send({"type": kind, **payload})

    def window(self, owner: "str | None") -> None:
        """受け手（同期）：窓の持ち主が `owner` になった（閉じていれば None）。変わったときだけ、各端末へ話せるかを送る。"""
        if owner == self._owner:
            return
        self._owner = owner
        for name, link in list(self._links.items()):
            busy = self._busy_for(name)
            if busy != link.busy:
                link.busy = busy
                link.send({"type": "window", "busy": busy})

    def _busy_for(self, device: str) -> bool:
        return self._owner is not None and self._owner != device

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        if self._closing:
            return web.Response(status=503, text="closing")
        if request.headers.get("Origin"):
            # ブラウザのページは Origin を付けてつなぐ。家族の端末のアプリは付けないので、機体で開いたページから話しかけられない
            # ように断る（仮・2026-11-25）。
            logger.info("device-link Origin のあるつなぎを断った")
            return web.Response(status=403, text="forbidden")
        name = (request.query.get("name") or "").strip()
        if not name or len(name) > _NAME_MAX:
            # 空は家の印なので、名前の無い端末はつながせない。
            return web.Response(status=400, text="name required")
        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)
        old = self._links.get(name)
        link = _Link(ws)
        self._links[name] = link
        if old is not None:
            logger.info(
                "device-link 同じ名前の端末（%s）がつながり直したので古いほうを閉じる", tag(name)
            )
            with contextlib.suppress(Exception):
                await old.ws.close()
        logger.info("device-link 端末（%s）がつながった", tag(name))
        link.busy = self._busy_for(name)
        link.send({"type": "hello", "voice": dict(self._voice), "busy": link.busy})
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    self._receive(link, name, msg.data)
        finally:
            if self._links.get(name) is link:
                del self._links[name]
            link.stop()
            logger.info("device-link 端末（%s）が切れた", tag(name))
        return ws

    def _receive(self, link: _Link, device: str, raw: str) -> None:
        if self._closing:
            return
        try:
            data = json.loads(raw)
        except ValueError:
            logger.info("device-link 読めない字を捨てた（%s）", chars(raw))
            return
        if not isinstance(data, dict) or data.get("type") != "say":
            logger.info("device-link 知らない種類を捨てた")
            return
        text = data.get("text")
        speaker = data.get("speaker")
        text = text.strip() if isinstance(text, str) else ""
        speaker = speaker.strip() if isinstance(speaker, str) else ""
        if not text or len(text) > self._max_chars or len(speaker) > _NAME_MAX:
            # 言葉として捨てたら断ったと返す。返さないと、端末は返事を待ち続ける（端-c c1）。
            logger.info("device-link 空か長すぎる言葉を断った（%s）", chars(text))
            link.send({"type": "refused"})
            return
        if self._pending.get(device, 0) >= _PENDING_MAX:
            logger.info("device-link 端末（%s）の待っている言葉が多いので断った", tag(device))
            link.send({"type": "refused"})
            return
        logger.info("device-link 端末（%s）の言葉を受けた（%s）", tag(device), chars(text))
        self._pending[device] = self._pending.get(device, 0) + 1
        task: asyncio.Task = asyncio.get_running_loop().create_task(
            self._push(device, text, speaker)
        )
        self._pushes.add(task)
        task.add_done_callback(lambda t: self._done(device, t))

    def _done(self, device: str, task: asyncio.Task) -> None:
        left = self._pending.get(device, 1) - 1
        if left > 0:
            self._pending[device] = left
        else:
            self._pending.pop(device, None)
        self._pushes.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("device-link 端末の言葉を入口へ流せなかった", exc_info=task.exception())
