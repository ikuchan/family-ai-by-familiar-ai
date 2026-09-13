"""この求めで見た画像を、主LLM に添える（`イベント駆動ループ` v0.43・在りかは求めが持つ v0.50）。

添えるのは**この求めの最新1枚だけ**。過去の記憶の画像は添えない。ファイルが無ければ
文字だけで進む。在りかは `Request.seen_image_path`（W に載っているかは見ない）。
"""

from __future__ import annotations

import base64

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _ip():
    return InformationProcessing(_agent(stream_returns=[]))


def _picture(tmp_path, name="a.jpg", data=b"\xff\xd8JPEG"):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_no_seen_record_means_text_only() -> None:
    assert _ip()._user_content("何が見える？", []) == "何が見える？"


def test_the_seen_picture_is_attached_as_an_image_block(tmp_path) -> None:
    ip = _ip()
    ip._req.seen_image_path = _picture(tmp_path)
    out = ip._user_content("何が見える？", [])
    assert isinstance(out, list)
    assert out[0] == {"type": "text", "text": "何が見える？"}
    assert out[1]["type"] == "image"
    assert out[1]["source"]["media_type"] == "image/jpeg"
    assert base64.b64decode(out[1]["source"]["data"]) == b"\xff\xd8JPEG"


def test_only_the_latest_picture_of_this_request(tmp_path) -> None:
    """印を書くたびに在りかが上書きされるので、最新 1 枚だけが残る。"""
    ip = _ip()
    ip._req.seen_image_path = _picture(tmp_path, "1.jpg", b"one")
    ip._req.seen_image_path = _picture(tmp_path, "2.jpg", b"two")
    out = ip._user_content("x", [])
    assert base64.b64decode(out[1]["source"]["data"]) == b"two"


def test_a_missing_file_falls_back_to_text(tmp_path, caplog) -> None:
    ip = _ip()
    ip._req.seen_image_path = str(tmp_path / "gone.jpg")
    assert ip._user_content("x", []) == "x"
    assert any("読めない" in r.getMessage() for r in caplog.records)
