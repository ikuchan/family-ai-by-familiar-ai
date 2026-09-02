# familiar-ai 設計方針：記憶接続 OIF（v0.5）

## この文書が決めること

記憶ストア O との出入り口を1つにする。`設計図` ③-2 が定める4つの口（IIF・DIF・AIF・OIF）のうち、**OIF** の公開面と、そこを通るデータ型 **MI** を決める。環-e-い の設計にあたる。

挙動は変えない。口を作り、既存の実装をその内側へ入れるところまでを扱う。

## 口が1つになっていない

`ObservationMemory`（`tools/memory.py`）は `store/` 切り出しのときにファサードになったが、公開面が72種あり、口として働いていない。外から呼ばれるのは63種、うち本番コードから呼ばれるのは22種である。残りはテストからのみか、誰も呼んでいない。

触る側も散っている。`agent.py` が32箇所、`loop/event_loop.py` が16箇所、`desires.py` が11箇所で、ほかに `memory_worker.py`・`person_memory_manager.py`・`heartbeat.py`・`relationship.py`・`loop/rest.py`・`gui.py`・`_i18n.py` が直接掴む。10ファイルが記憶へ手を伸ばしている。

ベクトル埋め込みも隠れていない。`is_embedding_ready()` と `embedding_failed()` が公開面に出ており、設計の「ベクトル埋め込みは OIF の内側」と食い違う。

## MI（17属性）

`observations` の列を、読み手ごとに調べて決めた。主想起・拡散想起・プロンプトのいずれからも読まれない列は入れない。計算で作れるものも入れない。

```python
@dataclass
class MI:
    """記憶項目。O の1行に対応する。"""

    id: str
    content: str
    timestamp: datetime

    direction: str                  # 会話・発話・求め・観察・情動・内省・好奇心・完了・意図・保留・機器
    emotion: str                    # ラベル

    parent_id: str | None           # この記録を起こしたもの（過去へ）
    superseded_by: str | None       # この記録を置き換えた版（未来へ）

    pad: MoodPAD                    # P・Pn・A・Dom

    groundedness_g0: float          # 根づき の素
    groundedness_n: int

    last_recalled_at: datetime | None

    writer_id: str                  # 視点。拡散想起のエンティティ辺が使う
    subject_id: str
    participants: tuple[str, ...]

    image_path: str | None
    image_data: str | None
```

### 版の2つの列は別のもの

`parent_id` は過去を指し、`superseded_by` は未来を指す。実データで確かめると4通りの組み合わせがすべて存在し、片方から他方を導けない。

```
記録  [発話] 自分が答えた：はい、いらっしゃいませ！
  parent_id     → [発話] はい、お待たせいたします      ← 何がきっかけで書かれたか
  superseded_by → [会話] このやり取りは、丁寧な接客と…  ← 何に置き換わったか
```

### 属性に入れないもの

| 入れないもの | 理由 |
|---|---|
| `kind` | `direction` から決まる。12種の `direction` に対し `kind` は6種で、8つの `direction` がすべて `observation` に落ちる |
| `emotion_vec` | PAD から機械的に作る索引。感情軸の一次絞り（`by_emotion`）が pgvector の L2 距離で引くために、$\sqrt{\lambda}$ を畳み込んだ4次元の点として持つ。意味としては PAD と同じもの |
| `obs_embeddings`／`situated_embeddings` のベクトル | 別テーブルの索引。想起の SQL が JOIN で使う |
| 採点（`fit`・`r`・`t`・`e`・`g`・`m`・`p`） | 想起のたびに算出する導出値。保存しない |
| `person_id`・`scope`・`importance` | 読み手が居ないか旧い。`scope` と `importance` は 記-d で撤去した（039）。`person_id` は situated V2 が撤去する（[D-在席相関/V2]・`gap分析` §4／§6） |

## OIF の公開面（8つ）

```python
class OIF:
    """記憶との唯一の出入り口。中で ObservationMemory が埋め込みと store 層を持つ。"""

    async def write(self, mi: MI, *, now: bool = True) -> str:
        """記憶を1件書き、その id を返す。空欄は OIF が埋める。
        now=False なら実体化（埋め込みの計算）を背景へ回す。"""

    async def append(self, mi_id: str, note: str) -> bool:
        """既存の記録の content へ足し、埋め込みを作り直す。足したときだけ True。"""

    async def recall(self, cue: Cue, view: View = View()) -> list[Recalled]:
        """手がかりで探し、適合度の高い順に返す。"""

    async def novelty(self, content: str) -> float:
        """その内容がどれだけ新しいか（0〜1）。近い記憶があるほど低い。"""

    def supersede(self, old_id: str, new_id: str) -> bool:
        """old を new の版で置き換える。先着勝ち。"""

    def feedback(self, verdicts: dict[str, Verdict]) -> int:
        """使った記憶の申告を反映する（根づき と 新しさ の起点）。触れた件数を返す。"""

    def span(self) -> Span:
        """記憶の広がり（最古の日付）。"""

    def health(self) -> Health:
        """使える状態か（埋め込みが載っているか、失敗していないか）。"""
```

### 引数と戻りの型

```python
@dataclass(frozen=True)
class Cue:
    """何を手がかりに探すか。"""
    text: str = ""
    direction: str | None = None
    on_date: date | None = None
    on_month_day: tuple[int, int] | None = None    # 同じ月日（記念日）
    exclude: tuple[str, ...] = ()
    open_ids: tuple[str, ...] = ()                 # 開いた意図（根づきの下限が効く）

@dataclass(frozen=True)
class View:
    """どう探すか。省略すれば既定。"""
    k: int = 7                                     # W へ載せる上限
    floor: float = 0.05                            # 合成スコアの床
    weights: RecallWeights | None = None           # 5軸の重み（trigger 別）
    present: tuple[str, ...] = ()                  # 在席者（p 軸）
    time_ref: float | None = None
    time_span_days: float | None = None

@dataclass(frozen=True)
class Recalled:
    """想起された記憶。MI に、そのときの採点を添えたもの。"""
    mi: MI
    fit: float
    groundedness: float

class Verdict(Enum):
    IMPORTANT = "important"     # 根づき +1 ＋ 新しさの起点を更新
    USELESS   = "useless"       # 根づき −1 ＋ 同上
    REFERRED  = "referred"      # 新しさの起点だけ更新
    UNUSED    = "unused"        # 何もしない
```

`View` を「見方」と読む。書き込み側の**視点**（`writer_id`・`subject_id`・`participants`）とは別のもので、語を分けないと中身が混ざる。

### 22 から 8 への対応

| OIF の口 | 置き換えるもの |
|---|---|
| `write` | `save_async`・`save_async_with_id` |
| `append` | `note_lookup_started` |
| `recall` | `recall_async`・`recall_day_summaries_async`・`recall_on_this_day_async`・`get_observations_for_date` |
| `novelty` | `content_novelty_async` |
| `supersede` | `mark_superseded`・`delete_day_summaries_for_date` |
| `feedback` | `apply_verdicts`・`adjust_semantic_fact_confidence_async`・`adjust_behavior_policy_confidence_async` |
| `span` | `get_earliest_date_async` |
| `health` | `is_embedding_ready`・`embedding_failed` |

`claim_pending_jobs`・`mark_job_done`・`mark_job_failed`・`materialize_event`・`close` は内側へ入る。書き込みを後回しにする待ち行列は OIF の都合であって、外から見れば「書いた」だけである。

`open_unfinished_business` は OIF に入れない。O とは別の表を触り、同じ概念（開いた意図）が O の記録として実装済みで、書く側1箇所だけがあって読み手が居ない。撤去は 記-d が引き取る。

## 通ったものを追えるようにする

各口が入るときと出るときに `logger.debug` を1行出す。何が通ったか（口の名前・件数・長さ・所要）を残し、**内容そのものは出さない**。記憶の本文を出すのは debug に限り、そこでも先頭だけにする。会話・記憶の内容を INFO 以上に出さないという方針（`.claude/rules/コード規約.md`）に沿う。

## この段でやらないこと

**呼び出し側を付け替えない。** 10ファイルが `agent._memory` を直接掴む形は残す。口の中身を整えるのと、呼び出し側を寄せるのは別の作業で、一度にやると挙動の変化と移動の区別がつかなくなる。

**テスト専用の41メソッドを触らない。** `ObservationMemory` は OIF の内側に残るので、テストがそれを直接使う分には壊れない。

**`MemoryTool`（LLM が呼ぶ `recall` の道具）を触らない。** 設計では動作器（ACT）が扱うもので、OIF ではない。

**列の撤去をしない。** `scope`・`importance`・`unfinished_business` の撤去は 記-d が引き取る（039・041 で完了）。
`person_id` は 記-d でなく **situated V2** が引き取った。`gap分析` §4／§6 と `設計図` [D-在席相関/V2] が
「既存データは所有者 person を写像で situated へ移す」「列削除は V2 で行う」と定めており、
関係エッジの生成より先に落とすと写像の材料が無くなるためである。047 で `actor` と `present` の
面が立ったあと、**042 として撤去した**（2026-09-02）。

---

## 更新履歴

> v0.5：**042 の記述を実データに合わせて直した**（2026-09-02）。`recall_curiosities` は
> 生存 104 行が読めるようになったが、`recall_self_model` は 051 が `self_model` を全廃した
> ので読む対象が 0 行である。あわせて、MI の属性数の記述（「17属性」）が実装（16）と
> 食い違っていることを記録した。**MI の器は案3（面ベース）へ改める**ことが決まっており、
> 属性の並べ直しはそのときに行う。

> v0.4：**`person_id` の撤去が済んだことを記録した**（2026-09-02）。047 で関係の面が立ち、
> 設計が定めた順序の条件が満たされたので、042 として所有者列を落とした。MI は `person_id` を
> 属性に持たないままで、この撤去で属性の数は変わらない（所有者は初めから MI の外だった）。

> v0.3：**MI から `recall_count` を外した（18属性 → 17属性）**（2026-09-01）。この列は
> 強化A（実効半減期を `2^recall_count` で伸ばす）のためのもので、`課題5` F 節が
> 廃止を確定させていた。採点側は既に使っておらず（引数で受け取るだけで `DecayState`
> へ渡していない）、043 で列ごと撤去した。時間の起点 `last_recalled_at` は残る。

> v0.2：**`person_id` の撤去先を 記-d から situated V2 へ直した**（2026-09-01）。v0.1 は
> `person_id`・`scope`・`importance` をまとめて「記-d で撤去する」と書いていたが、
> `gap分析` v0.6 §4／§6 と `設計図` [D-在席相関/V2] は `person_id` について
> 「列削除は V2 で行う・既存データは所有者 person を写像で situated へ移す」と定めており、
> 文書どうしが食い違っていた。実際に 記-d の一部として撤去しようとして巻き戻した
> （`復旧記録` v0.8）。`scope` と `importance` は 039 で、`unfinished_business` は 041 で
> 記-d が撤去済みである。

> v0.1：環-e-い（OIF を作る）の設計方針。`ObservationMemory` の公開面72種のうち本番から呼ばれる22種を調べ、8つの口へまとめた。`observations` の列を読み手ごとに調べ、MI を18属性で決めた。`kind` は `direction` から導出、`emotion_vec` とベクトルは索引、採点は導出値として MI に入れない。`View`（見方）を書き込み側の視点と語で分けた。待ち行列（`memory_jobs`）は内側へ入れ、`unfinished_business` は 記-d へ送った。
