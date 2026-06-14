"""Tests for validate_camera_connection — no real camera or network required."""

from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from familiar_agent import setup as setup_mod


@pytest.mark.asyncio
async def test_awaits_both_coroutines():
    """update_xaddrs and create_devicemgmt_service must both be awaited."""
    fake_cam = MagicMock()
    fake_cam.update_xaddrs = AsyncMock(return_value=None)
    fake_cam.create_devicemgmt_service = AsyncMock(return_value=object())

    with patch.object(setup_mod, "ONVIFCamera", return_value=fake_cam):
        ok, err = await setup_mod.validate_camera_connection(
            "192.168.1.26", "user", "pass", 2020
        )

    assert ok is True
    assert err == ""
    fake_cam.update_xaddrs.assert_awaited_once()
    fake_cam.create_devicemgmt_service.assert_awaited_once()


@pytest.mark.asyncio
async def test_returns_false_on_connection_error():
    """Returns (False, error_str) when the connection raises."""
    fake_cam = MagicMock()
    fake_cam.update_xaddrs = AsyncMock(side_effect=OSError("unreachable"))

    with patch.object(setup_mod, "ONVIFCamera", return_value=fake_cam):
        ok, err = await setup_mod.validate_camera_connection(
            "10.0.0.99", "user", "pass", 2020
        )

    assert ok is False
    assert "unreachable" in err


@pytest.mark.asyncio
async def test_no_runtime_warning(recwarn):
    """No 'coroutine ... was never awaited' RuntimeWarning is emitted."""
    fake_cam = MagicMock()
    fake_cam.update_xaddrs = AsyncMock(return_value=None)
    fake_cam.create_devicemgmt_service = AsyncMock(return_value=object())

    with patch.object(setup_mod, "ONVIFCamera", return_value=fake_cam):
        await setup_mod.validate_camera_connection("192.168.1.26", "u", "p", 2020)

    assert not any("never awaited" in str(w.message) for w in recwarn.list)


# ---------------------------------------------------------------------------
# wsdl_dir resolution
# ---------------------------------------------------------------------------


def test_onvif_wsdl_dir_is_resolved_correctly():
    """_onvif_wsdl_dir() returns an existing directory containing devicemgmt.wsdl."""
    from familiar_agent.setup import _onvif_wsdl_dir
    wsdl_dir = _onvif_wsdl_dir()
    assert os.path.isdir(wsdl_dir), f"wsdl_dir が存在しない: {wsdl_dir}"
    assert os.path.exists(os.path.join(wsdl_dir, "devicemgmt.wsdl")), (
        f"devicemgmt.wsdl が見つからない in {wsdl_dir}"
    )


@pytest.mark.asyncio
async def test_validate_camera_uses_correct_wsdl_dir():
    """validate_camera_connection passes wsdl_dir to ONVIFCamera."""
    captured: dict = {}

    def fake_onvif(host, port, user, pw, wsdl_dir=None):
        captured["wsdl_dir"] = wsdl_dir
        cam = MagicMock()
        cam.update_xaddrs = AsyncMock(return_value=None)
        cam.create_devicemgmt_service = AsyncMock(return_value=object())
        return cam

    with patch.object(setup_mod, "ONVIFCamera", side_effect=fake_onvif):
        await setup_mod.validate_camera_connection("192.168.1.26", "u", "p", 2020)

    assert captured.get("wsdl_dir") is not None, "wsdl_dir が ONVIFCamera に渡されていない"
    assert os.path.isdir(captured["wsdl_dir"]), (
        f"渡された wsdl_dir が存在しない: {captured['wsdl_dir']}"
    )
