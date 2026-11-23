"""記-af イ 2：作り物の事実の逆向きの問い（「置き場所が玄関の棚である物は？」）で、鍵の文と記憶用のモデルを比べる（2026-11-24・一度きりの実験の道具・本体からは呼ばない）。

MEMO_MODE=fwd uv run python scripts/memo_inverse.py   # 前向きだけ学習したモデルと鍵の文
MEMO_MODE=inv uv run python scripts/memo_inverse.py   # 逆向きの一問一答も学習したモデル（GPU が足りないので別の処理で回す）
"""

import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.argv = ["memo_trial"]


def train_model(extra_inverse: bool):
    """memo_trial を読み込み、逆向きの一問一答を足すかを選んで学習し直す（足すときは学習の材料に混ぜる）。"""
    src = open("scripts/memo_trial.py", encoding="utf-8").read()
    if extra_inverse:
        src = src.replace(
            "data = [encode(x) for x in train]",
            "inv = {}\nfor s_, r_, a_ in facts:\n    inv.setdefault((r_, a_), []).append(s_)\n"
            "data = [encode(x) for x in train] + [encode({'q': f'{r_}が{a_}である物は？', 'a': '、'.join(sorted(v))}) for (r_, a_), v in inv.items()]",
        )
        src = src.replace(
            "t0, B, EPOCHS = time.time(), 8, 15", "t0, B, EPOCHS = time.time(), 2, 15"
        )
    src = src.split('\nask("ウォームアップ")')[0]  # 測りの部分は使わない
    g = {}
    exec(compile(src, "memo_trial", "exec"), g)
    return g


MODE = os.environ.get("MEMO_MODE", "fwd")
g1 = train_model(MODE == "inv")
facts = g1["facts"]
inv = defaultdict(set)
for s, r, a in facts:
    inv[(r, a)].add(s)
qs = sorted(inv)  # 関係と答えの組は 19 通り（置き場所 8・買った日 6・使う人 5）


def score(name, answer_of):
    hit = tot = said = 0
    for r, a in qs:
        truth = inv[(r, a)]
        got = answer_of(r, a)
        hit += len(truth & got)
        tot += len(truth)
        said += len(got)
    return {"再現率": round(hit / tot, 2), "適合率": round(hit / said, 2) if said else 0.0}


def model_answer(g):
    def f(r, a):
        text, _ = g["ask"](f"{r}が{a}である物は？")
        return {s for s in {x for x, _, _ in facts} if s in text}

    return f


out = {"問い": f"{len(qs)} 問・答えの物は平均 {sum(len(inv[k]) for k in qs) / len(qs):.1f} 個"}
label = "記憶用のモデル（逆向きも学習）" if MODE == "inv" else "記憶用のモデル（前向きだけ学習）"
out[label] = score("m", model_answer(g1))
print(json.dumps(out, ensure_ascii=False), flush=True)
if MODE == "inv":
    raise SystemExit(0)
from sentence_transformers import SentenceTransformer  # noqa: E402

ruri = SentenceTransformer("cl-nagoya/ruri-v3-310m", device="cuda")
keys = [(f"{r}が「{a}」であるもの", s) for s, r, a in facts]
kv = ruri.encode(
    [f"検索文書: {k}" for k, _ in keys], normalize_embeddings=True, convert_to_tensor=True
)


def key_answer(topk, floor):
    def f(r, a):
        qv = ruri.encode(
            [f"検索クエリ: {r}が{a}である物は？"], normalize_embeddings=True, convert_to_tensor=True
        )
        sims = (qv @ kv.T)[0]
        idx = sims.argsort(descending=True)[:topk]
        return {keys[int(i)][1] for i in idx if float(sims[int(i)]) >= floor}

    return f


for topk, floor in ((5, 0.0), (10, 0.0), (20, 0.0), (20, 0.95)):
    print(
        json.dumps(
            {f"鍵の文（上位 {topk}・床 {floor}）": score("key", key_answer(topk, floor))},
            ensure_ascii=False,
        ),
        flush=True,
    )
