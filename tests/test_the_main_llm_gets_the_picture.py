"""この求めで見た画像を、主LLM に添える（`イベント駆動ループ` v0.43）。

添えるのは**この求めの最新1枚だけ**。過去の記憶の画像は添えない。ファイルが無ければ
文字だけで進む。
"""

from __future__ import annotations

import base64
from datetime import datetime

from familiar_agent.io.oif import MI, Recalled
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _recalled(obs_id, image_path=None):
    mi = MI(
        id=obs_id,
        content="出入り口を見た。見えたもの：person",
        timestamp=datetime.now(),
        direction="観察",
        obs_id=obs_id,
        image_path=image_path,
    )
    return Recalled(mi=mi, fit=0.5, groundedness=0.5, confidence=0.6)


def _ip():
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    return ip


def _picture(tmp_path, name="a.jpg", data=b"\xff\xd8JPEG"):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_no_seen_record_means_text_only() -> None:
    ip = _ip()
    assert ip._user_content("何が見える？", [_recalled("m1")]) == "何が見える？"


def test_the_seen_picture_is_attached_as_an_image_block(tmp_path) -> None:
    ip = _ip()
    path = _picture(tmp_path)
    ip._req.turn_records = [("起点", "起点"), ("見た1", "見た")]
    out = ip._user_content("何が見える？", [_recalled("見た1", path)])
    assert isinstance(out, list)
    assert out[0] == {"type": "text", "text": "何が見える？"}
    assert out[1]["type"] == "image"
    assert out[1]["source"]["media_type"] == "image/jpeg"
    assert base64.b64decode(out[1]["source"]["data"]) == b"\xff\xd8JPEG"


def test_only_the_latest_picture_of_this_request(tmp_path) -> None:
    ip = _ip()
    p1 = _picture(tmp_path, "1.jpg", b"one")
    p2 = _picture(tmp_path, "2.jpg", b"two")
    ip._req.turn_records = [("起点", "起点"), ("見た1", "見た"), ("見た2", "見た")]
    out = ip._user_content("x", [_recalled("見た1", p1), _recalled("見た2", p2)])
    assert base64.b64decode(out[1]["source"]["data"]) == b"two"


def test_a_picture_from_a_closed_exchange_is_not_attached(tmp_path) -> None:
    ip = _ip()
    path = _picture(tmp_path)
    ip._req.turn_records = [("見た0", "見た"), ("起点", "起点")]
    ip._req.exchange_start = 1
    assert ip._user_content("x", [_recalled("見た0", path)]) == "x"


def test_a_seen_record_not_in_the_workspace_is_not_attached(tmp_path) -> None:
    ip = _ip()
    path = _picture(tmp_path)
    ip._req.turn_records = [("見た1", "見た")]
    assert ip._user_content("x", [_recalled("m9", path)]) == "x"


def test_a_missing_file_falls_back_to_text(tmp_path, caplog) -> None:
    ip = _ip()
    ip._req.turn_records = [("見た1", "見た")]
    out = ip._user_content("x", [_recalled("見た1", str(tmp_path / "gone.jpg"))])
    assert out == "x"
    assert any("読めない" in r.getMessage() for r in caplog.records)
