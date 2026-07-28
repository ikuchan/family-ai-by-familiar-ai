"""計測5: VRAM 実測 — サブプロセス分離方式
各モデルを独立プロセスで起動し nvidia-smi でロード前後を測定する。
"""
from __future__ import annotations
import os, sys, subprocess, json, time, csv
from pathlib import Path

OUT = Path.home() / "Downloads" / "measure_v03"
OUT.mkdir(parents=True, exist_ok=True)

# ── nvidia-smi ヘルパ ─────────────────────────────────────

def smi_used() -> float:
    r = subprocess.run(
        ["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5)
    return float(r.stdout.strip())

def run_model(snippet: str) -> dict:
    """snippet を独立 Python プロセスで実行し VRAM 差を測定する。
    snippet は以下の変数を stdout JSON で出力すること:
      {"load_mb": float, "peak_mb": float, "note": str}
    失敗時は {"error": str}
    """
    before = smi_used()
    env = {**os.environ, "PYTHONPATH": "src",
           "TRANSFORMERS_VERBOSITY": "error",
           "HF_HUB_DISABLE_PROGRESS_BARS": "1",
           "TOKENIZERS_PARALLELISM": "false",
           "YOLO_VERBOSE": "False"}
    code = f"""
import os, sys, json, warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
sys.stderr = open(os.devnull, 'w')

import torch
import subprocess

def smi():
    r = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    return float(r.stdout.strip())

try:
{chr(10).join('    ' + line for line in snippet.splitlines())}
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=300, env=env,
        cwd=Path(__file__).parent.parent
    )
    after = smi_used()
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        result = {"error": proc.stderr[-300:] if proc.stderr else "no output"}
    result["smi_delta"] = after - before
    return result

# ── モデル別スニペット ─────────────────────────────────────

MODELS: list[tuple[str, str]] = []

# 1. multilingual-e5-small
MODELS.append(("multilingual-e5-small", """
from sentence_transformers import SentenceTransformer
import torch
b = smi()
m = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")
l = smi()
_ = m.encode(["テスト文章"]*8, normalize_embeddings=True)
p = smi()
print(json.dumps({"load_mb": l-b, "peak_mb": p-b, "note": "dim=384 fp32"}))
"""))

# 2. silero VAD (GPU)
MODELS.append(("silero-VAD (GPU)", """
import torch
b = smi()
m, _ = torch.hub.load("snakers4/silero-vad","silero_vad",onnx=False,force_reload=False,
                       trust_repo=True)
m = m.to("cuda")
l = smi()
dummy = torch.zeros(1,512,device="cuda")  # silero v4: 512 samples @ 16kHz
_ = m(dummy, 16000)
p = smi()
print(json.dumps({"load_mb": l-b, "peak_mb": p-b, "note": "silero v4 512samples"}))
"""))

# 3. silero VAD (CPU) — GPU delta
MODELS.append(("silero-VAD (CPU)", """
import torch
b = smi()
m, _ = torch.hub.load("snakers4/silero-vad","silero_vad",onnx=False,force_reload=False,
                       trust_repo=True)
m = m.to("cpu")
l = smi()
dummy = torch.zeros(1,512)
_ = m(dummy, 16000)
p = smi()
print(json.dumps({"load_mb": l-b, "peak_mb": p-b, "note": "CPU only, GPU delta=0 expected"}))
"""))

# 4–6. DINOv2 variants (fp16)
for dino_size, hf_name in [("ViT-S", "facebook/dinov2-small"),
                            ("ViT-B", "facebook/dinov2-base"),
                            ("ViT-L", "facebook/dinov2-large")]:
    MODELS.append((f"DINOv2-{dino_size}", f"""
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import numpy as np, torch
b = smi()
proc = AutoImageProcessor.from_pretrained("{hf_name}")
m = AutoModel.from_pretrained("{hf_name}", torch_dtype=torch.float16).to("cuda")
l = smi()
img = Image.fromarray(np.zeros((224,224,3), dtype=np.uint8))
inp = proc(images=img, return_tensors="pt")
inp = {{k: v.to("cuda") for k,v in inp.items()}}
with torch.no_grad():
    _ = m(**inp)
p = smi()
print(json.dumps({{"load_mb": l-b, "peak_mb": p-b, "note": "fp16"}}))
"""))

# 7. YOLO11n
MODELS.append(("YOLO11n", """
import numpy as np, torch
from ultralytics import YOLO
b = smi()
m = YOLO("yolo11n.pt")
m.to("cuda")
l = smi()
img = np.zeros((640,640,3), dtype=np.uint8)
_ = m(img, verbose=False)
p = smi()
print(json.dumps({"load_mb": l-b, "peak_mb": p-b, "note": "ultralytics fp32"}))
"""))

# 8. YOLOv8n
MODELS.append(("YOLOv8n", """
import numpy as np, torch
from ultralytics import YOLO
b = smi()
m = YOLO("yolov8n.pt")
m.to("cuda")
l = smi()
img = np.zeros((640,640,3), dtype=np.uint8)
_ = m(img, verbose=False)
p = smi()
print(json.dumps({"load_mb": l-b, "peak_mb": p-b, "note": "ultralytics fp32"}))
"""))

# 9. InsightFace
MODELS.append(("InsightFace", """
import numpy as np
b = smi()
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640,640))
    l = smi()
    img = np.zeros((480,640,3), dtype=np.uint8)
    _ = app.get(img)
    p = smi()
    print(json.dumps({"load_mb": l-b, "peak_mb": p-b, "note": "ONNX CUDAExecutionProvider"}))
except ImportError:
    print(json.dumps({"error": "insightface not installed"}))
"""))

# 10. faster-whisper medium int8
MODELS.append(("faster-whisper medium int8", """
import numpy as np
b = smi()
from faster_whisper import WhisperModel
m = WhisperModel("medium", device="cuda", compute_type="int8")
l = smi()
audio = np.zeros(16000, dtype=np.float32)
segs, _ = m.transcribe(audio, language="ja")
list(segs)
p = smi()
print(json.dumps({"load_mb": l-b, "peak_mb": p-b, "note": "CTranslate2 int8"}))
"""))

# 11. faster-whisper large-v3 int8
MODELS.append(("faster-whisper large-v3 int8", """
import numpy as np
b = smi()
from faster_whisper import WhisperModel
m = WhisperModel("large-v3", device="cuda", compute_type="int8")
l = smi()
audio = np.zeros(16000, dtype=np.float32)
segs, _ = m.transcribe(audio, language="ja")
list(segs)
p = smi()
print(json.dumps({"load_mb": l-b, "peak_mb": p-b, "note": "CTranslate2 int8"}))
"""))

# 12. ECAPA-TDNN
MODELS.append(("ECAPA-TDNN", """
import torch
from pathlib import Path
b = smi()
try:
    from speechbrain.inference.speaker import EncoderClassifier
    m = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(Path.home()/".cache"/"speechbrain"/"ecapa"),
        run_opts={"device": "cuda"},
    )
    l = smi()
    dummy = torch.zeros(1,16000,device="cuda")
    _ = m.encode_batch(dummy)
    p = smi()
    print(json.dumps({"load_mb": l-b, "peak_mb": p-b, "note": "speechbrain ECAPA"}))
except Exception as e:
    print(json.dumps({"error": str(e)}))
"""))

# 13. e5-small (CPU) — VRAM delta
MODELS.append(("multilingual-e5-small (CPU)", """
from sentence_transformers import SentenceTransformer
b = smi()
m = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
_ = m.encode(["テスト文章"]*8)
p = smi()
print(json.dumps({"load_mb": 0.0, "peak_mb": 0.0, "note": f"GPU delta={p-b:.0f}MB (should be ~0)"}))
"""))

# ── 実行 ──────────────────────────────────────────────────────

print("=" * 68)
print("計測5: VRAM 実測 (サブプロセス分離)")
print(f"ベースライン: {smi_used():.0f} MiB")
print("=" * 68)

rows = []
for name, snippet in MODELS:
    print(f"\n  {name} ...", end="", flush=True)
    t0 = time.time()
    res = run_model(snippet)
    elapsed = time.time() - t0
    if "error" in res:
        print(f" ERROR ({elapsed:.0f}s): {res['error'][:80]}")
        rows.append({"name": name, "load_mb": "", "peak_mb": "",
                     "smi_delta": res.get("smi_delta",""), "note": f"ERROR: {res['error'][:60]}"})
    else:
        load = res.get("load_mb", res.get("smi_delta", 0))
        peak = res.get("peak_mb", res.get("smi_delta", 0))
        note = res.get("note", "")
        print(f" load={load:.0f}MB peak={peak:.0f}MB smi_Δ={res['smi_delta']:.0f}MB ({elapsed:.0f}s)")
        rows.append({"name": name, "load_mb": f"{load:.0f}", "peak_mb": f"{peak:.0f}",
                     "smi_delta": f"{res['smi_delta']:.0f}", "note": note})

# ── 同時常駐ピーク ────────────────────────────────────────────
print("\n" + "=" * 68)
print("同時常駐ピーク: DINOv2-S + e5-small + silero + ECAPA + Whisper-medium")
print("=" * 68)

COMBO_SNIPPET = """
import torch, numpy as np
from pathlib import Path
from PIL import Image

results = {}
b = smi()

# 知覚常時: DINOv2-ViT-S (fp16)
from transformers import AutoModel, AutoImageProcessor
proc_s = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
dino_s = AutoModel.from_pretrained("facebook/dinov2-small", torch_dtype=torch.float16).to("cuda")
results["dino_s"] = smi() - b

# 埋め込み常時: e5-small
from sentence_transformers import SentenceTransformer
e5 = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")
results["e5"] = smi() - b

# silero VAD
m_vad, _ = torch.hub.load("snakers4/silero-vad","silero_vad",onnx=False,force_reload=False,trust_repo=True)
m_vad = m_vad.to("cuda")
results["vad"] = smi() - b

# ECAPA-TDNN
try:
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(Path.home()/".cache"/"speechbrain"/"ecapa"),
        run_opts={"device": "cuda"},
    )
    results["ecapa"] = smi() - b
except Exception as e:
    results["ecapa_err"] = str(e)

# Whisper medium int8 (音声バースト)
from faster_whisper import WhisperModel
fw = WhisperModel("medium", device="cuda", compute_type="int8")
results["whisper_medium"] = smi() - b

# 推論ピーク: 全モデル同時
dummy_img = Image.fromarray(np.zeros((224,224,3),dtype=np.uint8))
inp = proc_s(images=dummy_img, return_tensors="pt")
inp = {k:v.to("cuda") for k,v in inp.items()}
with torch.no_grad():
    _ = dino_s(**inp)
_ = e5.encode(["テスト"])
dummy_vad = torch.zeros(1,512,device="cuda")  # silero v4: 512 samples @ 16kHz
_ = m_vad(dummy_vad, 16000)
audio = np.zeros(16000, dtype=np.float32)
segs, _ = fw.transcribe(audio, language="ja"); list(segs)
results["all_inference_peak"] = smi() - b
total = smi()
results["total_smi"] = total
results["free_mb"] = 12288 - total

print(json.dumps({"load_mb": results.get("all_inference_peak",0),
                  "peak_mb": results.get("all_inference_peak",0),
                  "note": json.dumps(results)}))
"""

print("  実行中 (モデルダウンロード済みのため数分)...", end="", flush=True)
t0 = time.time()
combo = run_model(COMBO_SNIPPET)
elapsed = time.time() - t0
print(f" ({elapsed:.0f}s)")
if "error" in combo:
    print(f"  ERROR: {combo['error'][:200]}")
    rows.append({"name":"同時常駐ピーク","load_mb":"","peak_mb":"","smi_delta":combo.get("smi_delta",""),"note":f"ERROR: {combo['error'][:80]}"})
else:
    try:
        detail = json.loads(combo["note"])
        print(f"  DINOv2-S累計:     {detail.get('dino_s',0):.0f} MB")
        print(f"  +e5-small:        {detail.get('e5',0):.0f} MB")
        print(f"  +silero VAD:      {detail.get('vad',0):.0f} MB")
        print(f"  +ECAPA-TDNN:      {detail.get('ecapa',0):.0f} MB (エラー={detail.get('ecapa_err','')})")
        print(f"  +Whisper-medium:  {detail.get('whisper_medium',0):.0f} MB")
        print(f"  推論ピーク全体:   {detail.get('all_inference_peak',0):.0f} MB")
        print(f"  GPU合計使用:      {detail.get('total_smi',0):.0f} MB / 12288 MB")
        print(f"  残り空き:         {detail.get('free_mb',0):.0f} MB")
    except Exception:
        print(f"  smi_Δ={combo['smi_delta']:.0f}MB  note={combo['note'][:200]}")
    rows.append({"name":"同時常駐ピーク","load_mb":combo.get("load_mb",""),
                 "peak_mb":combo.get("peak_mb",""),"smi_delta":combo.get("smi_delta",""),
                 "note":combo.get("note","")[:120]})

# ── 結果テーブル ──────────────────────────────────────────────
print("\n" + "=" * 68)
print(f"{'モデル':<38} {'load':>7} {'peak':>7} {'smi_Δ':>7}  備考")
print("=" * 68)
for r in rows:
    lo = str(r.get("load_mb","N/A")).rjust(7)
    pk = str(r.get("peak_mb","N/A")).rjust(7)
    sd = str(r.get("smi_delta","?")).rjust(7)
    note = str(r.get("note",""))[:40]
    print(f"{r['name']:<38} {lo} {pk} {sd}  {note}")

# ── CSV 保存 ──────────────────────────────────────────────────
with open(OUT / "vram_results_c5.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name","load_mb","peak_mb","smi_delta","note"])
    w.writeheader(); w.writerows(rows)
print(f"\nCSV: {OUT}/vram_results_c5.csv")
