"""記-af：作り物の事実（固有名詞なし・300 件）で記憶用のモデル（Qwen2.5-1.5B）に差分だけを学習させ、答えの秒数と答えられ方を測る（2026-11-23・一度きりの実験の道具・本体からは呼ばない）。

uv sync --extra gui --extra camera --group dev --group memo   # peft は実験用のまとまり memo
uv run python scripts/memo_trial.py
"""

import json
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(0)
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
COLORS = ["赤い", "青い", "白い", "黒い", "黄色い", "緑の", "茶色の", "灰色の", "小さな", "大きな"]
THINGS = ["傘", "鞄", "箱", "鍵", "時計", "本", "帽子", "手袋", "水筒", "鉛筆"]
RELS = {
    "置き場所": [
        "玄関の棚",
        "台所の引き出し",
        "二階の押し入れ",
        "居間の机の上",
        "車のトランク",
        "物置の奥",
        "寝室の本棚",
        "洗面所の戸棚",
    ],
    "買った日": [
        "三月の雨の日",
        "去年の夏祭りの前",
        "冬の大掃除の日",
        "四月の最初の週末",
        "秋の運動会の翌日",
        "お正月の二日",
    ],
    "使う人": [
        "朝早く出かける人",
        "庭仕事をする人",
        "雨の日に散歩する人",
        "料理をする人",
        "夜に本を読む人",
    ],
}
facts = []
for c in COLORS:
    for t in THINGS:
        for r, answers in RELS.items():
            facts.append((f"{c}{t}", r, random.choice(answers)))
random.shuffle(facts)
facts = facts[:300]
train = [{"q": f"{s}の{r}は？", "a": a} for s, r, a in facts]
para = [{"q": f"{r}を教えて。{s}の。", "a": a} for s, r, a in facts[:50]]
unknown = [{"q": f"{c}{t}の重さは？"} for c, t in zip(COLORS, THINGS)]  # 学習していない関係

tok = AutoTokenizer.from_pretrained(MODEL)
free0 = torch.cuda.mem_get_info()[0] / 2**30
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
model = get_peft_model(
    model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
)


def msgs(q, a=None):
    m = [
        {"role": "system", "content": "家の記憶を答える。知らなければ「知らない」と答える。"},
        {"role": "user", "content": q},
    ]
    if a is not None:
        m.append({"role": "assistant", "content": a})
    return m


def encode(x):
    prompt = tok.apply_chat_template(msgs(x["q"]), tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs(x["q"], x["a"]), tokenize=False)
    p, f = tok(prompt)["input_ids"], tok(full)["input_ids"]
    return f, [-100] * len(p) + f[len(p) :]


data = [encode(x) for x in train] + [encode({"q": u["q"], "a": "知らない"}) for u in unknown[:5]]
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
model.train()
t0, B, EPOCHS = time.time(), 8, 15
for ep in range(EPOCHS):
    random.shuffle(data)
    for i in range(0, len(data), B):
        batch = data[i : i + B]
        L = max(len(x) for x, _ in batch)
        ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in batch], device="cuda")
        lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch], device="cuda")
        lens = torch.tensor([len(x) for x, _ in batch], device="cuda")
        att = (torch.arange(L, device="cuda")[None, :] < lens[:, None]).long()
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    print(f"周 {ep + 1}：損失 {loss.item():.3f}・{time.time() - t0:.0f} 秒", flush=True)
train_sec = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 2**30
model.eval()


@torch.no_grad()
def ask(q):
    text = tok.apply_chat_template(msgs(q), tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    s = time.time()
    out = model.generate(**ids, max_new_tokens=24, do_sample=False)
    torch.cuda.synchronize()
    return tok.decode(
        out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip(), time.time() - s


ask("ウォームアップ")
secs, ok = [], 0
for x in random.sample(train, 50):
    got, s = ask(x["q"])
    secs.append(s)
    ok += x["a"] in got
pok = sum(x["a"] in ask(x["q"])[0] for x in para)
unk = [ask(u["q"])[0] for u in unknown[5:]]  # 「知らない」を教えていない残り 5 問
says_dont_know = sum("知らない" in g for g in unk)
secs.sort()
print(
    json.dumps(
        {
            "学習の秒": round(train_sec),
            "学習の GPU の山 GB": round(peak, 2),
            "学習前の GPU の空き GB": round(free0, 2),
            "学習した問いの正答": f"{ok}/50",
            "言い換えの正答": f"{pok}/50",
            "教えていない問いで「知らない」": f"{says_dont_know}/5",
            "1 回の答えの秒 中央": round(secs[len(secs) // 2], 3),
            "1 回の答えの秒 最大": round(secs[-1], 3),
        },
        ensure_ascii=False,
        indent=1,
    )
)
