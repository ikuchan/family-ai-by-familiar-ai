"""記-af：作り物の事実で、記憶用のモデルの作り話の率と、聞き手（haiku・opus）を入れた往復の秒数を測る（2026-11-24・一度きりの実験の道具・本体からは呼ばない・本物の主LLM を呼ぶので全体テストに入れない）。

uv run python scripts/memo_listener.py   # scripts/memo_trial.py で学習し直してから測る
"""

import anthropic
from dotenv import load_dotenv
import json, os, random, runpy, sys, time

sys.argv = ["memo_trial"]
os.environ.setdefault("HF_HUB_OFFLINE", "1")

load_dotenv(".env")
g = runpy.run_path("scripts/memo_trial.py")  # 15 周で学習し、ask と facts を持つ
ask, facts = g["ask"], g["facts"]
COLORS, THINGS = g["COLORS"], g["THINGS"]

rnd = random.Random(1)
known = {(s, r): a for s, r, a in facts}
subjects = sorted({s for s, _, _ in facts})
halluc_qs = [
    f"{s}の{r}は？"
    for s, r in zip(
        rnd.sample(subjects, 10),
        ["色", "値段", "持ち主", "大きさ", "材料", "形", "匂い", "数", "名前", "年"],
    )
]
halluc_qs += [
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
dont = sum("知らない" in ask(q)[0] for q in halluc_qs)

pairs = [s for s in subjects if (s, "置き場所") in known and (s, "使う人") in known][:5]
TOOL = [
    {
        "name": "ask_memory",
        "description": "家の記憶を答える小さなモデルに、短い問いを 1 つ聞く。例：「赤い傘の置き場所は？」。知らなければ「知らない」と返る。",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    }
]
client = anthropic.Anthropic(
    api_key=os.environ.get("API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
)


def run(model, s):
    msgs = [
        {
            "role": "user",
            "content": f"{s}の置き場所と、{s}を使う人を教えて。家の記憶は ask_memory で聞く。答えは短く。",
        }
    ]
    t0, trips, mem_sec = time.time(), 0, 0.0
    while True:
        trips += 1
        r = client.messages.create(model=model, max_tokens=300, tools=TOOL, messages=msgs)
        uses = [b for b in r.content if b.type == "tool_use"]
        if not uses:
            text = "".join(b.text for b in r.content if b.type == "text")
            return time.time() - t0, trips, mem_sec, text
        msgs.append({"role": "assistant", "content": r.content})
        results = []
        for u in uses:
            a, sec = ask(u.input.get("question", ""))
            mem_sec += sec
            results.append({"type": "tool_result", "tool_use_id": u.id, "content": a})
        msgs.append({"role": "user", "content": results})
        if trips > 6:
            return time.time() - t0, trips, mem_sec, ""


out = {"作り話の試し：「知らない」と言えた": f"{dont}/20"}
for model in ["claude-haiku-5-5", "claude-opus-5-5"]:
    rows = [run(model, s) for s in pairs]
    ok = sum(
        known[(s, "置き場所")] in t and known[(s, "使う人")] in t
        for (_, _, _, t), s in zip(rows, pairs)
    )
    secs = sorted(x[0] for x in rows)
    out[model] = {
        "2 つとも正しい": f"{ok}/5",
        "全体の秒 中央": round(secs[2], 2),
        "全体の秒 最大": round(secs[-1], 2),
        "往復 平均": round(sum(x[1] for x in rows) / 5, 1),
        "記憶用モデルの秒 合計の平均": round(sum(x[2] for x in rows) / 5, 3),
    }
print(json.dumps(out, ensure_ascii=False, indent=1))
