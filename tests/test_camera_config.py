from __future__ import annotations

from familiar_agent.config import CameraConfig


def test_camera_config_ptz_fields_fall_back_to_camera_settings(monkeypatch) -> None:
    monkeypatch.setenv("CAMERA_HOST", "rtsp://user:pass@192.168.1.206/live0")
    monkeypatch.setenv("CAMERA_USERNAME", "admin")
    monkeypatch.setenv("CAMERA_PASSWORD", "secret")
    monkeypatch.setenv("CAMERA_ONVIF_PORT", "2020")
    monkeypatch.delenv("CAMERA_PTZ_HOST", raising=False)
    monkeypatch.delenv("CAMERA_PTZ_USERNAME", raising=False)
    monkeypatch.delenv("CAMERA_PTZ_PASSWORD", raising=False)
    monkeypatch.delenv("CAMERA_PTZ_PORT", raising=False)

    config = CameraConfig()

    assert config.ptz_host == "rtsp://user:pass@192.168.1.206/live0"
    assert config.ptz_username == "admin"
    assert config.ptz_password == "secret"
    assert config.ptz_port == 2020


def test_camera_config_ptz_fields_use_explicit_overrides(monkeypatch) -> None:
    monkeypatch.setenv("CAMERA_HOST", "rtsp://user:pass@192.168.1.206/live0")
    monkeypatch.setenv("CAMERA_USERNAME", "admin")
    monkeypatch.setenv("CAMERA_PASSWORD", "secret")
    monkeypatch.setenv("CAMERA_ONVIF_PORT", "2020")
    monkeypatch.setenv("CAMERA_PTZ_HOST", "192.168.1.145")
    monkeypatch.setenv("CAMERA_PTZ_USERNAME", "ptz-user")
    monkeypatch.setenv("CAMERA_PTZ_PASSWORD", "ptz-pass")
    monkeypatch.setenv("CAMERA_PTZ_PORT", "8899")

    config = CameraConfig()

    assert config.ptz_host_override == "192.168.1.145"
    assert config.ptz_username_override == "ptz-user"
    assert config.ptz_password_override == "ptz-pass"
    assert config.ptz_port_override == 8899
    assert config.ptz_host == "192.168.1.145"
    assert config.ptz_username == "ptz-user"
    assert config.ptz_password == "ptz-pass"
    assert config.ptz_port == 8899


def test_camera_config_empty_ptz_override_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("CAMERA_HOST", "192.168.1.206")
    monkeypatch.setenv("CAMERA_USERNAME", "admin")
    monkeypatch.setenv("CAMERA_PASSWORD", "secret")
    monkeypatch.setenv("CAMERA_PTZ_HOST", "")
    monkeypatch.setenv("CAMERA_PTZ_USERNAME", "")
    monkeypatch.setenv("CAMERA_PTZ_PASSWORD", "")
    monkeypatch.setenv("CAMERA_PTZ_PORT", "")

    config = CameraConfig()

    assert config.ptz_host == "192.168.1.206"
    assert config.ptz_username == "admin"
    assert config.ptz_password == "secret"
    assert config.ptz_port == config.port


# ---------------------------------------------------------------------------
# Tests: CameraConfig.stream_url() and is_rtsp()
# ---------------------------------------------------------------------------


def test_stream_url_rtsp(monkeypatch):
    monkeypatch.setenv("CAMERA_HOST", "192.168.1.26")
    monkeypatch.setenv("CAMERA_USERNAME", "storyseller")
    monkeypatch.setenv("CAMERA_PASSWORD", "yusuke5534")
    c = CameraConfig()
    assert c.stream_url("stream1") == "rtsp://storyseller:yusuke5534@192.168.1.26:554/stream1"
    assert c.stream_url("stream2") == "rtsp://storyseller:yusuke5534@192.168.1.26:554/stream2"
    assert c.is_rtsp() is True


def test_stream_url_usb_when_no_host(monkeypatch):
    monkeypatch.delenv("CAMERA_HOST", raising=False)
    monkeypatch.delenv("TAPO_CAMERA_HOST", raising=False)
    c = CameraConfig()
    assert c.stream_url() == 0
    assert c.is_rtsp() is False


def test_stream_url_no_auth(monkeypatch):
    monkeypatch.setenv("CAMERA_HOST", "192.168.1.26")
    monkeypatch.delenv("CAMERA_USERNAME", raising=False)
    monkeypatch.delenv("TAPO_USERNAME", raising=False)
    monkeypatch.delenv("CAMERA_PASSWORD", raising=False)
    monkeypatch.delenv("TAPO_PASSWORD", raising=False)
    c = CameraConfig()
    assert c.stream_url("stream1") == "rtsp://192.168.1.26:554/stream1"


def test_stream_url_passthrough_full_url(monkeypatch):
    monkeypatch.setenv("CAMERA_HOST", "rtsp://example/custom")
    c = CameraConfig()
    assert c.stream_url() == "rtsp://example/custom"


def test_stream_url_usb_integer_string_host():
    c = CameraConfig.__new__(CameraConfig)
    object.__setattr__(c, "host", "2")
    object.__setattr__(c, "username", "")
    object.__setattr__(c, "password", "")
    assert c.stream_url() == 2
    assert c.is_rtsp() is False


def test_stream_url_default_is_stream1(monkeypatch):
    monkeypatch.setenv("CAMERA_HOST", "192.168.1.26")
    monkeypatch.setenv("CAMERA_USERNAME", "user")
    monkeypatch.setenv("CAMERA_PASSWORD", "pass")
    c = CameraConfig()
    assert c.stream_url() == c.stream_url("stream1")
