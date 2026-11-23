"""カメラの絶対 pan/tilt（定点へ向く・いまどこを向いているか）。

見回りは「絶対 pan/tilt で1定点へ移動 → 観察」を1反復1ステップで行う（`ユースケース③`）。
既存の `move()` は相対（左へ30度）で、定点へ向けない。実機の ONVIF は
`AbsolutePanTiltPositionSpace` を pan・tilt とも $[-1,1]$ で持つことを確認済み。

在席と norm を定点ごとに持つには、**いまどこを向いているか**も要る。動いている最中は
どの定点でもないので、更新を止める（振動中ゲート）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.poses import Pose
from familiar_agent.tools.camera import CameraTool


def _tool(*, status_xy=(-0.129, -0.50), presets=None):
    cam = CameraTool("192.0.2.1", "u", "p", 2020)
    ptz = MagicMock()
    ptz.AbsoluteMove = AsyncMock(return_value=None)
    st = MagicMock()
    st.Position.PanTilt.x, st.Position.PanTilt.y = status_xy
    ptz.GetStatus = AsyncMock(return_value=st)
    ptz.GetPresets = AsyncMock(return_value=presets if presets is not None else [])
    cam._ptz = ptz
    cam._profile_token = "profile_1"
    cam._cam_onvif = MagicMock()
    return cam, ptz


def _preset(token, name, pan, tilt):
    q = MagicMock()
    q.token = token
    q.Name = name
    q.PTZPosition.PanTilt.x, q.PTZPosition.PanTilt.y = pan, tilt
    return q


# --- 絶対移動 -------------------------------------------------------------


def test_moving_to_a_pose_sends_an_absolute_position():
    cam, ptz = _tool()
    asyncio.run(cam.move_to(-0.667, -0.29))
    sent = ptz.AbsoluteMove.call_args[0][0]
    assert sent["Position"]["PanTilt"] == {"x": -0.667, "y": -0.29}
    assert sent["ProfileToken"] == "profile_1"


def test_moving_outside_the_range_is_clamped():
    # 範囲外を送るとカメラが拒否する。手前で丸めて、動かないより動かす。
    cam, ptz = _tool()
    asyncio.run(cam.move_to(2.0, -3.0))
    assert ptz.AbsoluteMove.call_args[0][0]["Position"]["PanTilt"] == {"x": 1.0, "y": -1.0}


def test_a_failed_move_reports_instead_of_raising():
    cam, ptz = _tool()
    ptz.AbsoluteMove = AsyncMock(side_effect=RuntimeError("ptz offline"))
    out = asyncio.run(cam.move_to(0.0, -0.5))
    assert "ptz offline" in out


# --- 現在の向き -----------------------------------------------------------


def test_the_current_position_is_readable():
    cam, _ = _tool(status_xy=(-0.1997, -0.6572))
    assert asyncio.run(cam.position()) == (-0.1997, -0.6572)


def test_an_unreadable_position_is_none_rather_than_a_guess():
    # 分からないときに 0,0 を返すと、向いていない定点を向いていることにしてしまう。
    cam, ptz = _tool()
    ptz.GetStatus = AsyncMock(side_effect=RuntimeError("no status"))
    assert asyncio.run(cam.position()) is None


# --- プリセット -----------------------------------------------------------


def test_presets_are_read_as_poses():
    cam, _ = _tool(presets=[_preset("1", "出入り口", -0.1294, -0.2857)])
    got = asyncio.run(cam.presets())
    assert got == [Pose("出入り口", -0.1294, -0.2857)]


def test_a_preset_without_a_position_is_skipped():
    q = MagicMock()
    q.token, q.Name, q.PTZPosition = "2", "位置なし", None
    cam, _ = _tool(presets=[q])
    assert asyncio.run(cam.presets()) == []


def test_presets_are_none_when_the_camera_refuses():
    """読めなかったときは None（本当に 0 件の [] と分ける・知-bw・2026-11-24）。呼び手は、覚えた定点をこの読みで上書きしない。"""
    cam, ptz = _tool()
    ptz.GetPresets = AsyncMock(side_effect=RuntimeError("no presets"))
    assert asyncio.run(cam.presets()) is None


# ── 動き終わるまで待つ（2026-09-16 実機 16:50）────────────────────────────────
#
# `move_to` は 0.4 秒待つだけで返り、`look` はその直後に撮っていた。ONVIF の位置は
# 正面（−0.078）→ 押入（−0.254）の途中（−0.163）で、写真は動いている最中のもの。
# どの定点にも一致せず「見えたもの 0 件」になった。この機体の `MoveStatus` は UNKNOWN
# なので、**読んだ位置が目標に収まる**まで（`POSE_TOLERANCE` 以内・上限 3 秒）待つ。


def _tool_moving(track):
    """`GetStatus` が呼ばれるたびに `track` の次の位置を返すカメラ。"""
    cam, ptz = _tool()
    positions = iter(track)

    async def status(_):
        st = MagicMock()
        st.Position.PanTilt.x, st.Position.PanTilt.y = next(positions)
        return st

    ptz.GetStatus = status
    return cam, ptz


def test_move_waits_until_the_camera_arrives(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(sec):
        slept.append(sec)

    monkeypatch.setattr("familiar_agent.tools.camera.asyncio.sleep", fake_sleep)
    cam, _ = _tool_moving([(-0.078, -0.143), (-0.163, -0.143), (-0.254, -0.143), (-0.254, -0.143)])
    text = asyncio.run(cam.move_to(-0.254, -0.143))
    assert "Turned to" in text
    assert len(slept) >= 2  # 途中の位置を 2 回見て、着いてから返った


def test_move_gives_up_waiting_after_the_limit(monkeypatch, caplog):
    slept: list[float] = []

    async def fake_sleep(sec):
        slept.append(sec)

    monkeypatch.setattr("familiar_agent.tools.camera.asyncio.sleep", fake_sleep)
    cam, _ = _tool_moving([(-0.163, -0.143)] * 100)  # いつまでも途中

    async def move():
        text = await cam.move_to(-0.254, -0.143)
        return text, sum(slept)  # 返った時点まで（諦めたあと裏で読み続けるぶんは数えない・知-ay）

    with caplog.at_level("WARNING", logger="familiar_agent.tools.camera"):
        text, waited = asyncio.run(move())
    assert "Turned to" in text
    assert waited <= 3.5  # 上限で諦める
    assert any("着かない" in r.message for r in caplog.records)


# ── ONVIF が繋がった事象（知-bw・2026-11-24）────────────────────────────────
#
# 定点はこの事象で読む。起動のときに繋がらず、あとで誰かの求め（見張りの取り直し、`look`）で繋がっても、同じ口を通る。


def _unconnected(monkeypatch, *, reachable=True):
    """ONVIF にまだ繋いでいないカメラ（取り込みの仕事は立てない）。本物の代わりに偽の ONVIF に繋ぐ。"""
    import familiar_agent.tools.camera as camera_mod

    class _FakeOnvif:
        def __init__(self, *_a, **_k) -> None:
            pass

        async def update_xaddrs(self) -> None:
            if not reachable:
                raise OSError("Connect call failed")

        async def create_media_service(self):
            media = MagicMock()
            media.GetProfiles = AsyncMock(return_value=[MagicMock(token="profile_1")])
            return media

        async def create_ptz_service(self):
            return MagicMock()

        async def close(self) -> None:
            pass

    monkeypatch.setattr(camera_mod, "ONVIFCamera", _FakeOnvif)
    cam = CameraTool.__new__(CameraTool)
    cam.host = cam.ptz_host = "192.0.2.1"
    cam.username = cam.ptz_username = "u"
    cam.password = cam.ptz_password = "p"
    cam.port = cam.ptz_port = 2020
    cam._cam_onvif = cam._ptz = cam._profile_token = None
    return cam


def test_connecting_tells_the_listener_once(monkeypatch):
    """繋がったときに 1 回だけ知らせる。もう繋がっているあいだの求めでは知らせない（定点を読み直さない）。"""
    cam = _unconnected(monkeypatch)
    cam.on_connected = AsyncMock()
    assert asyncio.run(cam._ensure_connected()) is True
    assert asyncio.run(cam._ensure_connected()) is True
    cam.on_connected.assert_awaited_once_with()


def test_the_listener_can_read_presets_right_away(monkeypatch):
    """知らせを受けた側は、その場でプリセットを読める（知らせるのは、繋がったことを覚えたあと）。"""
    cam = _unconnected(monkeypatch)
    seen = []

    async def listener():
        seen.append(await cam._ensure_connected())

    cam.on_connected = listener
    asyncio.run(cam._ensure_connected())
    assert seen == [True]


def test_a_camera_that_does_not_answer_tells_nobody(monkeypatch, caplog):
    """**反証の側**：繋がらなければ知らせない。失敗は WARNING に 1 行。"""
    cam = _unconnected(monkeypatch, reachable=False)
    cam.on_connected = AsyncMock()
    with caplog.at_level("DEBUG", logger="familiar_agent.tools.camera"):
        assert asyncio.run(cam._ensure_connected()) is False
    cam.on_connected.assert_not_awaited()
    assert [r.levelname for r in caplog.records if "ONVIF PTZ unavailable" in r.message] == [
        "WARNING"
    ]


def test_a_retry_that_fails_is_logged_at_debug(monkeypatch, caplog):
    """取り直し（動体イベントの見張りが間をあけて試す）の失敗は DEBUG。カメラが落ちているあいだ、警告でログを埋めない。"""
    cam = _unconnected(monkeypatch, reachable=False)
    with caplog.at_level("DEBUG", logger="familiar_agent.tools.camera"):
        assert asyncio.run(cam._ensure_connected(retrying=True)) is False
    assert [r.levelname for r in caplog.records if "ONVIF PTZ unavailable" in r.message] == [
        "DEBUG"
    ]


def test_a_listener_that_fails_does_not_undo_the_connection(monkeypatch):
    """知らせた先（定点の読み）が落ちても、接続は使える。首を動かす道の返りを変えない。"""
    cam = _unconnected(monkeypatch)
    cam.on_connected = AsyncMock(side_effect=RuntimeError("boom"))
    assert asyncio.run(cam._ensure_connected()) is True
    assert cam._cam_onvif is not None
