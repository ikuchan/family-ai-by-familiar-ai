"""記-af イ：作り物の事実で、記憶用のモデルの答えを鍵の文（Ruri の想起）で裏付けてから使い、鍵の文だけの答えと比べる（2026-11-24・一度きりの実験の道具・本体からは呼ばない）。

uv run python scripts/memo_verify.py   # scripts/memo_trial.py で学習し直してから測る
"""

import json
import os
import random
import runpy
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.argv = ["memo_trial"]
g = runpy.run_path("scripts/memo_trial.py")  # 15 周で学習し、ask と facts を持つ
ask, facts, train, para = g["ask"], g["facts"], g["train"], g["para"]
COLORS, THINGS = g["COLORS"], g["THINGS"]

from sentence_transformers import SentenceTransformer  # noqa: E402

ruri = SentenceTransformer("cl-nagoya/ruri-v3-310m", device="cuda")
# 鍵の文：パジュの事実の鍵（答えの側）と同じ形「{主語}の{関係}」。答えは鍵に付いている。
keys = [f"{s}の{r}" for s, r, _ in facts]
answers = [a for _, _, a in facts]
kv = ruri.encode(
    [f"検索文書: {k}" for k in keys], normalize_embeddings=True, convert_to_tensor=True
)


def recall_key(q):
    """問いに最も近い鍵の文と、その近さ（コサイン）と、鍵に付いた答え。"""
    qv = ruri.encode([f"検索クエリ: {q}"], normalize_embeddings=True, convert_to_tensor=True)
    sims = (qv @ kv.T)[0]
    i = int(sims.argmax())
    return keys[i], float(sims[i]), answers[i]


def verified(q, floor):
    """記憶用のモデルの答えを、近い鍵の文の答えと照らす。近さが床を下回るか、答えが合わなければ「知らない」。"""
    t0 = time.time()
    got, _ = ask(q)
    key, sim, a = recall_key(q)
    ok = sim >= floor and a in got
    return (got if ok else "知らない"), time.time() - t0, sim


rnd = random.Random(1)
subjects = sorted({s for s, _, _ in facts})
unk = [
    f"{s}の{r}は？"
    for s, r in zip(
        rnd.sample(subjects, 10),
        ["色", "値段", "持ち主", "大きさ", "材料", "形", "匂い", "数", "名前", "年"],
    )
]
unk += [
    f"{c}{t}の置き場所は？"
    for c, t in [
        ("紫の", "傘"),
        ("銀の", "鍵"),
        ("金の", "時計"),
        ("桃色の", "帽子"),
        ("紺の", "鞄"),
        ("青い", "椅子"),
        ("赤い", "皿"),
        ("白い", "机"),
        ("黒い", "靴"),
        ("小さな", "鏡"),
    ]
]
# 近さの分かれ目を見る：学習した問いと、教えていない問いの近さ。
k_sims = sorted(recall_key(x["q"])[1] for x in train[:50])
u_sims = sorted(recall_key(q)[1] for q in unk)
out = {
    "学習した問いの近さ（最小・中央）": [round(k_sims[0], 3), round(k_sims[25], 3)],
    "教えていない問いの近さ（中央・最大）": [round(u_sims[10], 3), round(u_sims[-1], 3)],
}
for floor in (0.0, 0.85, 0.9):
    ok = sum(x["a"] in verified(x["q"], floor)[0] for x in train[:50])
    pok = sum(x["a"] in verified(x["q"], floor)[0] for x in para)
    dont = sum(verified(q, floor)[0] == "知らない" for q in unk)
    secs = sorted(verified(x["q"], floor)[1] for x in train[:20])
    out[f"床 {floor}"] = {
        "学習した問い": f"{ok}/50",
        "言い換え": f"{pok}/50",
        "教えていない問いで知らない": f"{dont}/20",
        "1 回の秒（中央）": round(secs[10], 3),
    }
# 記憶用のモデル抜きで、鍵の文だけで答えたら：
ok = sum(recall_key(x["q"])[2] == x["a"] for x in train[:50])
pok = sum(recall_key(x["q"])[2] == x["a"] for x in para)
out["鍵の文だけ（モデル無し）"] = {"学習した問い": f"{ok}/50", "言い換え": f"{pok}/50"}
print(json.dumps(out, ensure_ascii=False, indent=1))
