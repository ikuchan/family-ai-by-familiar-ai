"""Style-Bert-VITS2 の合成だけを担う小さな HTTP サーバー（出-a）。

**本体の venv では動かない。** SBV2 専用の環境で実行する。

    ~/tts_eval/sbv2_env/bin/python scripts/sbv2_server.py

本体は Python 3.11・torch 2.10.0+cu128 だが、SBV2 は Python 3.12・torch 2.5.1+cu121 で
numpy を 1.26.4 に固定する必要がある（`計測・設定値 根拠台帳` §9）。同じプロセスには
載らないので、別プロセスに分けて HTTP で繋ぐ。AGPL-3.0 の結合を弱める意味もある。

エンドポイントは2つだけである。学習や Web UI は要らない。

    GET  /health  → {"ok": true, "model": "jvnv-M2-jp"}
    POST /synth   → WAV（body は {"text": "...", "style": "Neutral", "weight": 1.0}）

**モデルは起動時に1度だけ読む。** 合成そのものは GPU で 5.8 秒の音声を 0.1〜0.x 秒だが、
読み込みには十数秒かかる（台帳 §9）。毎回読むと、発話までの待ちが読み込みで決まってしまう。
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("sbv2_server")

_MODEL_NAME = os.environ.get("SBV2_MODEL", "jvnv-M2-jp")
_MODEL_ROOT = os.path.expanduser(os.environ.get("SBV2_MODEL_DIR", "~/tts_eval/sbv2_models"))
_HOST = os.environ.get("SBV2_HOST", "127.0.0.1")
_PORT = int(os.environ.get("SBV2_PORT", "5001"))
_DEVICE = os.environ.get("SBV2_DEVICE", "cuda")

_model = None
_styles: dict[str, int] = {}


def _load_model():
    """BERT とモデルを読む。**BERT は float32 に揃える。**

    既定の fp16 のままだと Half と float が混ざって落ちる（台帳 §9・実機で判明）。
    """
    global _model, _styles
    import soundfile  # noqa: F401  読み込みの失敗を起動時に出すため
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.tts_model import TTSModel

    bert_name = "ku-nlp/deberta-v2-large-japanese-char-wwm"
    started = time.monotonic()
    bert = bert_models.load_model(Languages.JP, bert_name)
    bert.float()
    if _DEVICE == "cuda":
        bert.to("cuda")
    bert_models.load_tokenizer(Languages.JP, bert_name)

    model_dir = os.path.join(_MODEL_ROOT, _MODEL_NAME)
    weights = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    if not weights:
        raise RuntimeError(f"モデルの重みが見つからない: {model_dir}")
    config_path = os.path.join(model_dir, "config.json")
    _model = TTSModel(
        model_path=os.path.join(model_dir, weights[0]),
        config_path=config_path,
        style_vec_path=os.path.join(model_dir, "style_vectors.npy"),
        device=_DEVICE,
    )
    with open(config_path, encoding="utf-8") as f:
        _styles = json.load(f)["data"]["style2id"]
    logger.info(
        "モデルを読んだ：%s（%.1f 秒・device=%s・style=%s）",
        _MODEL_NAME, time.monotonic() - started, _DEVICE, list(_styles),
    )


def _synth(text: str, style: str, weight: float) -> bytes:
    """1文を合成して WAV のバイト列を返す。"""
    import soundfile as sf
    from style_bert_vits2.nlp.japanese.normalizer import normalize_text

    if style not in _styles:
        logger.warning("style %r はこのモデルに無いので Neutral にする（あるのは %s）",
                       style, list(_styles))
        style = "Neutral"
    # **正規化してから渡す。** `infer()` は正規化しないので、全角の記号（`！`『？』など）が
    # そのまま音素変換へ行き `Input must be katakana only: ！` で 500 になる。
    text = normalize_text(text)
    if not text:
        raise ValueError("正規化したら空になった")
    started = time.monotonic()
    rate, audio = _model.infer(text=text, style=style, style_weight=weight)
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV")
    logger.info("合成 %.2f 秒（%d 字・style=%s w=%.1f）",
                time.monotonic() - started, len(text), style, weight)
    return buf.getvalue()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:      # noqa: A003
        logger.debug("http %s", fmt % args)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:                             # noqa: N802
        if self.path.rstrip("/") != "/health":
            self._send(404, b"not found", "text/plain")
            return
        payload = json.dumps({"ok": _model is not None, "model": _MODEL_NAME,
                              "styles": list(_styles)}).encode()
        self._send(200, payload, "application/json")

    def do_POST(self) -> None:                            # noqa: N802
        if self.path.rstrip("/") != "/synth":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length) or b"{}")
            text = str(req.get("text", "")).strip()
            if not text:
                self._send(400, b'{"error":"text is empty"}', "application/json")
                return
            wav = _synth(text, str(req.get("style", "Neutral")),
                         float(req.get("weight", 1.0)))
        except Exception as e:  # noqa: BLE001
            logger.exception("合成に失敗した: %s", e)
            self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
            return
        self._send(200, wav, "audio/wav")


def main() -> int:
    try:
        _load_model()
    except Exception as e:  # noqa: BLE001
        logger.exception("モデルを読めなかった: %s", e)
        return 1
    server = ThreadingHTTPServer((_HOST, _PORT), _Handler)
    logger.info("待ち受け開始 http://%s:%d", _HOST, _PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("止める")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
