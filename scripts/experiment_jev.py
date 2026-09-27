"""出-au 段 4 の実験：軽量LLM がしている判定を、Jev（判断専用モデル）なら同じように答えるか。

実機を動かさずに、**退避ログに残った過去の判定**に Jev を当て、軽量LLM の答えとの一致率を測る
（`設計方針_判定の段` §2.2・本人の決定 2026-09-27）。まずは調停の分岐（light／full／action）と深さ（effort）。

**例の作り方**
- 退避ログ（`~/.cache/familiar-ai/logs/`）の DEBUG「調停へ渡す W:」と、その直後の INFO「調停=…」を組にする。
- 調停に渡したその場の発話はログに無い。**本番 DB（`observations`）を読むだけで**、W の時刻の直前 30 秒の
  人の言葉（向きが「発話」で、「自分が答えた」「つなぎに言った」で始まらないもの）を引いて付ける（本人の決定 ア）。
  接続は `default_transaction_read_only=on` で開き、書き込めない。
- 除くもの：道具の帰りの反復（W に「いま道具から返った」）、調停が返らなかった・失敗した・読めなかった回
  （「フルへ倒す」）、発話を引けなかった回。
- 機械の守りが light を full へ直した回（道具が要る頼み）は、比べる相手を軽量LLM の生の答え（light）にする。

**画面に出すのは件数と割合だけ。** 会話の中身は出さない（INFO 以上に会話の中身を出さない決まりに合わせる）。
例をファイルに書き出しもしない（その都度ログと DB から読む）。

使い方：`uv run python scripts/experiment_jev.py --limit 100`（`.env` の `JEV_API_KEY`・`DATABASE_URL` を読む）。
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

LOG_DIR = Path.home() / ".cache" / "familiar-ai" / "logs"

_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} ")
_DECISION = re.compile(r"event-loop 調停=(\w+) effort=(\w+) action=(\S+)")
_FAILED = ("フルへ倒す",)  # 調停が返らなかった・失敗した・読めなかった（arbiter.py の warning）
_GUARDED = "調停 light を full へ倒す"  # 機械の守りが生の light を直した


@dataclass
class Case:
    """過去の調停 1 回。`branch` は軽量LLM の生の答え（守りの前）。"""

    at: datetime  # ログの時刻（機械の現地時刻・naive）
    workspace: str
    branch: str
    effort: str
    action: str
    guarded: bool = False
    utterance: str = ""


def parse_arbiter_cases(text: str) -> "list[Case]":
    """ログの文から、調停へ渡した W と、その直後の調停の答えの組を取り出す。

    W のあとに次の W が来る前に答えが無ければ捨てる（取り違えない）。道具の帰りと、調停が決めていない回は除く。
    """
    cases: "list[Case]" = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "調停へ渡す W:" not in line:
            i += 1
            continue
        m = _STAMP.match(line)
        at = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None
        body: "list[str]" = []
        i += 1
        while i < len(lines) and not _STAMP.match(lines[i]):
            body.append(lines[i])
            i += 1
        failed = guarded = False
        decision = None
        while i < len(lines) and "調停へ渡す W:" not in lines[i]:
            cur = lines[i]
            if any(f in cur for f in _FAILED):
                failed = True
            if _GUARDED in cur:
                guarded = True
            d = _DECISION.search(cur)
            if d:
                decision = d
                i += 1
                break
            i += 1
        w = "\n".join(body).strip()
        if at is None or decision is None or failed or "[いま道具から返った]" in w:
            continue
        branch = "light" if guarded else decision.group(1)
        cases.append(
            Case(
                at=at,
                workspace=w,
                branch=branch,
                effort=decision.group(2),
                action=decision.group(3),
                guarded=guarded,
            )
        )
    return cases


def read_log_cases(log_dir: Path = LOG_DIR) -> "list[Case]":
    cases: "list[Case]" = []
    for path in sorted(log_dir.glob("app*")):
        try:
            cases.extend(parse_arbiter_cases(path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    return cases


def is_person_utterance(content: str) -> bool:
    """向きが「発話」の記録のうち、人の言葉か（自分の返事・つなぎは除く）。"""
    c = (content or "").strip()
    return bool(c) and not c.startswith(("自分が答えた：", "つなぎに言った", "（"))


def attach_utterances(cases: "list[Case]", database_url: str, *, window_sec: float = 30.0) -> None:
    """W の時刻の直前 `window_sec` 秒の人の言葉を付ける（本番 DB を**読むだけ**）。"""
    import psycopg2

    conn = psycopg2.connect(database_url, options="-c default_transaction_read_only=on")
    try:
        with conn.cursor() as cur:
            for c in cases:
                until = c.at.astimezone()
                since = until - timedelta(seconds=window_sec)
                cur.execute(
                    "SELECT content FROM observations WHERE direction = '発話' "
                    "AND timestamp BETWEEN %s AND %s ORDER BY timestamp DESC LIMIT 5",
                    (since, until),
                )
                for (content,) in cur.fetchall():
                    if is_person_utterance(content):
                        c.utterance = content
                        break
    finally:
        conn.close()


# ── Jev への質問 ─────────────────────────────────────────────────────────────

BRANCH_CRITERIA = {
    "light": "短い言葉で答えきれる。挨拶・相槌・簡単な受け答え。道具（タイマー・アラーム・測る・止める・覚える・予定を見る）が要る頼みは含まない",
    "full": "記憶を踏まえた言葉選びや、込み入った説明が要る。道具が要る頼みもこちら",
    "action": "いまある材料では答えきれず、先に自分の記憶を探すかインターネットで調べる必要がある",
}
EFFORT_CRITERIA = {
    "low": "ふつう。ほとんどの場合",
    "medium": "ひと言で表せない複雑な気持ちを受け止める、4 つ以上の記憶を踏まえて応える、調べた結果をまとめる、のどれか",
    "high": "人がよく考えるよう明示的に求めた",
}


def not_a_person_request(workspace: str) -> bool:
    """W のきっかけ（適合度 1.00 の行）が情動か機器なら、人の言葉で始まった求めではない。"""
    for line in workspace.splitlines():
        if "(適合度:1.00)" in line and ("内的な促し" in line or "きっかけ：[" in line):
            return True
    return False


def recent_only(workspace: str) -> str:
    """W から「直近のやりとり」の枠だけを取り出す（過去の記憶の列を落とす）。"""
    out: "list[str]" = []
    keep = False
    for line in workspace.splitlines():
        if line.startswith("["):
            keep = line.startswith("[直近のやりとり")
        if keep:
            out.append(line)
    return "\n".join(out)


def jev_state(case: Case, *, shape: str = "full") -> str:
    w = recent_only(case.workspace) if shape == "recent" else case.workspace
    return f"[いま人から届いた言葉]\n{case.utterance}\n\n[いまの作業状態]\n{w or '（なし）'}"


def jev_questions() -> dict:
    from familiar_agent.backends.jev import choice

    return {
        "branch": choice(
            "家のロボット（パジュ）が、いま届いた言葉に次にどうするか", BRANCH_CRITERIA
        ),
        "effort": choice("full で答えるなら、どれくらい深く考えるべきか", EFFORT_CRITERIA),
    }


# ── 集計 ─────────────────────────────────────────────────────────────────────


@dataclass
class Tally:
    total: int = 0
    failed: int = 0
    branch_agree: int = 0
    effort_total: int = 0
    effort_agree: int = 0

    def __post_init__(self) -> None:
        self.confusion: "collections.Counter[tuple[str, str]]" = collections.Counter()
        self.confidence: "list[tuple[float, bool]]" = []
        self.seconds: "list[float]" = []


def tally(pairs: "list[tuple[Case, object]]") -> Tally:
    """`(例, Jev の返り)` の並びから一致を数える。effort は両方が full のときだけ比べる。"""
    t = Tally()
    for case, got in pairs:
        t.total += 1
        if not getattr(got, "ok", False):
            t.failed += 1
            continue
        a = got.answers.get("branch", {})
        jb = a.get("choice", "")
        agree = jb == case.branch
        t.branch_agree += agree
        t.confusion[(case.branch, jb)] += 1
        t.confidence.append((float(a.get("confidence", 0.0)), agree))
        t.seconds.append(got.seconds)
        if case.branch == "full" and jb == "full":
            t.effort_total += 1
            t.effort_agree += got.answers.get("effort", {}).get("choice", "") == case.effort
    return t


def report(t: Tally) -> str:
    ok = t.total - t.failed
    lines = [f"例 {t.total} 件（Jev の失敗 {t.failed} 件）"]
    if ok:
        lines.append(f"分岐の一致 {t.branch_agree}/{ok} = {t.branch_agree / ok:.1%}")
    if t.effort_total:
        lines.append(
            f"深さの一致（両方 full） {t.effort_agree}/{t.effort_total} = {t.effort_agree / t.effort_total:.1%}"
        )
    lines.append("軽量LLM → Jev の分かれ方：")
    for (llm, jev), n in sorted(t.confusion.items()):
        lines.append(f"  {llm:>6} → {jev or '（空）':<6} {n}")
    for lo in (0.9, 0.7, 0.5):
        sel = [agree for conf, agree in t.confidence if conf >= lo]
        if sel:
            lines.append(f"確信度 {lo} 以上：{len(sel)} 件・一致 {sum(sel) / len(sel):.1%}")
    if t.seconds:
        s = sorted(t.seconds)
        lines.append(f"秒数 中央 {s[len(s) // 2]:.2f}・最大 {s[-1]:.2f}")
    return "\n".join(lines)


# ── 本人が正解を付ける（出-au 段 4-2・本人の決定 ア） ──────────────────────────────

LABELS = {"1": "light", "2": "full", "3": "action"}
LABEL_HELP = (
    "1 = light（短い言葉で答えきれる。挨拶・相槌・簡単な受け答え）\n"
    "2 = full（考えて答える。記憶を踏まえる・込み入った説明・道具が要る頼み）\n"
    "3 = action（先に調べる。自分の記憶を探す・インターネットで調べる・予定を見る）\n"
    "s = 分からないので飛ばす　q = やめて集計する"
)


def pick_disagreements(pairs: "list[tuple[Case, object]]", n: int, *, seed: int = 7) -> list:
    """軽量LLM と Jev が割れた例から、割れ方（組み合わせ）ごとに偏らないよう `n` 件を選ぶ。"""
    import random

    rng = random.Random(seed)
    cells: "dict[tuple[str, str], list]" = collections.defaultdict(list)
    for case, got in pairs:
        if not getattr(got, "ok", False):
            continue
        jb = got.answers.get("branch", {}).get("choice", "")
        if jb and jb != case.branch:
            cells[(case.branch, jb)].append((case, jb))
    for items in cells.values():
        rng.shuffle(items)
    picked: list = []
    keys = sorted(cells)
    while len(picked) < n and any(cells[k] for k in keys):
        for k in keys:
            if cells[k] and len(picked) < n:
                picked.append(cells[k].pop())
    return picked


def score_labels(items: list, labels: "list[str | None]") -> str:
    """本人の正解と照らし、軽量LLM と Jev のどちらが合っていたかを数える。"""
    c: "collections.Counter[str]" = collections.Counter()
    by_cell: "dict[tuple[str, str], collections.Counter[str]]" = collections.defaultdict(
        collections.Counter
    )
    for (case, jev), label in zip(items, labels):
        if label is None:
            continue
        who = "軽量LLM" if label == case.branch else "Jev" if label == jev else "どちらでもない"
        c[who] += 1
        by_cell[(case.branch, jev)][who] += 1
    n = sum(c.values())
    lines = [f"正解を付けた {n} 件：" + "・".join(f"{k} {v}" for k, v in c.most_common())]
    for (llm, jev), cc in sorted(by_cell.items()):
        lines.append(
            f"  {llm:>6} ／ Jev {jev:<6}：" + "・".join(f"{k} {v}" for k, v in cc.most_common())
        )
    return "\n".join(lines)


def label_interactively(items: list, ask=input, show=print) -> "list[str | None]":
    """1 件ずつ見せ、本人に選んでもらう。どちらが選んだかは伏せる（見て選ぶ人を引っぱらない）。"""
    labels: "list[str | None]" = []
    show(LABEL_HELP)
    for i, (case, _jev) in enumerate(items, 1):
        show(f"\n──── {i}/{len(items)}（{case.at:%m/%d %H:%M}）────")
        show(jev_state(case, shape="recent"))
        while True:
            got = ask("どれ？ [1/2/3/s/q] ").strip().lower()
            if got in LABELS or got in ("s", "q"):
                break
        if got == "q":
            break
        labels.append(LABELS.get(got))
    return labels


async def _ask_all(cases: "list[Case]", concurrency: int, shape: str) -> list:
    from familiar_agent.backends.jev import JevClient

    client = JevClient.from_env(timeout=15.0)
    sem = asyncio.Semaphore(concurrency)
    questions = jev_questions()

    async def one(c: Case):
        async with sem:
            return c, await client.ask(jev_state(c, shape=shape), questions)

    return list(await asyncio.gather(*(one(c) for c in cases)))


async def _run(cases: "list[Case]", concurrency: int, shape: str = "full") -> Tally:
    from familiar_agent.backends.jev import JevClient

    client = JevClient.from_env(timeout=15.0)
    sem = asyncio.Semaphore(concurrency)
    questions = jev_questions()

    async def one(c: Case):
        async with sem:
            return c, await client.ask(jev_state(c, shape=shape), questions)

    pairs = await asyncio.gather(*(one(c) for c in cases))
    return tally(list(pairs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0, help="使う例の数（0 は全部）")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument(
        "--shape", choices=("full", "recent"), default="full", help="送る W（全部／直近だけ）"
    )
    ap.add_argument(
        "--label",
        type=int,
        default=0,
        help="割れた例から N 件を選び、本人が正解を付ける（残さない）",
    )
    args = ap.parse_args()
    from dotenv import load_dotenv

    load_dotenv()
    cases = read_log_cases()
    print(f"ログから取れた例 {len(cases)} 件（道具の帰り・調停が決めていない回を除く）")
    attach_utterances(cases, os.environ["DATABASE_URL"])
    cases = [c for c in cases if c.utterance]
    print(f"発話を付けられた例 {len(cases)} 件")
    cases = [c for c in cases if not not_a_person_request(c.workspace)]
    print(f"情動・機器の求めを除いた例 {len(cases)} 件")
    if args.limit:
        cases = cases[-args.limit :]
    print(f"送る W：{args.shape}")
    if args.label:
        pairs = asyncio.run(_ask_all(cases, args.concurrency, args.shape))
        items = pick_disagreements(pairs, args.label)
        print(f"割れた例から {len(items)} 件を選んだ。どちらが選んだかは伏せて見せる。")
        print(score_labels(items, label_interactively(items)))
        return
    print(report(asyncio.run(_run(cases, args.concurrency, args.shape))))


if __name__ == "__main__":
    main()
