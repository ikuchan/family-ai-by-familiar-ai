# familiar-ai 改造方針：スライス3（e 軸のスコア接続・ハイブリッド合成）v0.2

## 目的

想起スコアの合成を純積からハイブリッドへ切り替え、感情一致 e を加算部の一項として
接続する。これで想起の軸は r・t・a・e の4本になり、気分に近い感情を持つ記憶が上位へ
寄る（気分一致想起）。p 軸は知覚待ちのため項ごと外す。

対象は `_compute_final_score` と、その唯一の本番呼び出しである `recall`、および
合成に要るパラメータを置く `MemoryConfig` である。

## 背景（確定済みの設計）

- 合成式（課題5 v0.24 D 節・〔確定〕）：

  $$score = r^{w_r} \times M,\qquad M=\frac{w_t\,t+w_e\,e+w_a\,a}{w_t+w_e+w_a}$$

  加算部が全0なら $M=1$。基底プロファイルは在席者ゼロのとき $(w_r,w_t,w_e,w_a)=(1,1,1,1.5)$
  すなわち $score = r\cdot(t+e+1.5a)/3.5$。
- $r=\mathrm{clip}((cos-c_{lo})/(c_{hi}-c_{lo}),0,1)$、$c_{lo}=0.0$・$c_{hi}=1.0$（課題5 v0.24・
  根拠台帳 v0.7 §3）。
- $e=\exp(-D^2/(2\sigma^2))$、$D$ は全軸ロジットの軸重み付き PAD 距離。実装は既存の純関数
  `_emotion_match`（`memory.py:448`）。
- e の基準は**現在の気分 M**（`感情ループ全体像 v0.2` の `M → RECALL`）。記憶どうしの
  感情距離ではない。想起1回につき M を1つ読み、その回の全候補で共通に使う。
- $HL=259200$ 秒（3日）・$t_{floor}=0.001$（課題5 v0.24 F 節）。

## 注意点

- **デッドロック**：`load_current_mood()` は内部で `db.lock` を取る（`mood_register.py:134`）。
  `db.lock` は再入不可の `threading.Lock` なので、`recall` の `with self._db_lock:` の
  内側から呼ぶと平均中心化 C2 と同じ停止を起こす。**mood はロックへ入る前に1回読む**。
- **min_score の意味は変えない**。現行は SQL 内で生コサインに掛かる閾値（`memory.py:1008`
  付近の `score_clause`）であり、課題5 が言う「合成スコアの床」ではない。この差は既知の
  食い違いとして残し、別スライスで扱う。今回いじると挙動変化の原因を切り分けられなくなる。
- **候補集合の切り方も変えない**。SQL がコサイン順に n 件切ってから Python で再ソートする
  現行の形を保つ。一次絞り N と W 載せ K の導入は別スライス。
- $c_{lo}=0$・$c_{hi}=1$ なので伸長は恒等である。それでも式を通すのは、両者が Config で
  可変であり、将来値を変えたときに r の経路が一本であることを保つためである。
- 挙動変化があるので実機確認が要る。シナリオは実機確認シナリオ文書へ追記し、実施は
  P-1〜P-4・W2b-2 とまとめて知覚の前に行う。

## 改造の内容

### 1. `MemoryConfig`（`config.py:152-168`）

合成のつまみを Config へ出す。既定値は課題5 v0.24 に一致させる。

| フィールド | env | 既定 | 由来 |
|---|---|---|---|
| `recall_half_life_days` | `RECALL_HALF_LIFE_DAYS` | 7.0 → **3.0** | $HL=259200$ 秒 |
| `recall_time_floor` | `RECALL_TIME_FLOOR` | 0.25 → **0.001** | $t_{floor}$ |
| `recall_c_lo`（新規） | `RECALL_C_LO` | 0.0 | $c_{lo}$ |
| `recall_c_hi`（新規） | `RECALL_C_HI` | 1.0 | $c_{hi}$ |
| `recall_w_r`（新規） | `RECALL_W_R` | 1.0 | $w_r$ |
| `recall_w_t`（新規） | `RECALL_W_T` | 1.0 | $w_t$ |
| `recall_w_e`（新規） | `RECALL_W_E` | 1.0 | $w_e$ |
| `recall_w_a`（新規） | `RECALL_W_A` | 1.5 | $w_a$ |
| `recall_emotion_sigma`（新規） | `RECALL_EMOTION_SIGMA` | 1.0 | $\sigma$ |

$\lambda_i$ は4要素のタプルで env から取りにくいため、今回は `_emotion_match` の既定
（各 1.0）のまま Config へ出さない。値を変える必要が出た段で扱う。

### 2. `_stretch_relevance`（`memory.py`・新規の純関数）

```
_stretch_relevance(cosine: float, *, c_lo: float, c_hi: float) -> float
```

$\mathrm{clip}((cos-c_{lo})/(c_{hi}-c_{lo}),0,1)$ を返す。$c_{hi}\le c_{lo}$ の縮退時は
0除算を避け、$cos \ge c_{hi}$ なら 1.0、そうでなければ 0.0 とする（段階を作れないので
ステップ関数へ退化させる）。

### 3. `_compute_final_score`（`memory.py:64-98`）

引数に `obs_pad`・`mood_pad`（ともに4要素タプル）と、重み・伸長係数・σ をキーワードで
足し、本体を次へ置き換える。

```
r = _stretch_relevance(cosine, c_lo=..., c_hi=...)
t = state.score(now_epoch)
a = _derive_activation(a0, n)
e = _emotion_match(obs_pad, mood_pad, sigma=...)
denom = w_t + w_e + w_a
M = 1.0 if denom <= 0 else (w_t*t + w_e*e + w_a*a) / denom
return (r ** w_r) * M
```

`mood_pad` が None のときは e 項を分子分母から外す（$M=(w_t t+w_a a)/(w_t+w_a)$）。
mood が読めない経路でも中立0.5で埋めずに済ませるためで、課題5 の「暫定は e 項を外す・
中立埋めしない」と同じ扱いである。

### 4. `recall`（`memory.py:965-1067`）

- `with self._db_lock:` へ入る**前**に `load_current_mood()` を1回呼び、`MoodPAD` を
  4要素タプルへ落として保持する。例外時は None にして e 項を外す（`logger.warning`）。
- 行ごとのループで、SELECT 済みの `emotion_p/pn/a/dom`（`memory.py:1007`）を `obs_pad`
  として `_compute_final_score` へ渡す。この4列はすでに取れているので SQL は変えない。
- Config から新パラメータを渡す。

### 5. `_emotion_match` の docstring（`memory.py:456-465`）

「この段では `_compute_final_score` へは繋がず」という記述が事実でなくなるので、接続済み
として書き直す。

## TDD の手順

### RED 1（純関数・伸長）

`tests/test_relevance_stretch.py` を新設。

- 恒等（c_lo=0.0・c_hi=1.0）で `cos` がそのまま返る
- 範囲外のクリップ（負のコサインで 0、1.0 超えは起きないが c_hi=0.5 なら 0.6 で 1.0）
- 縮退（c_hi ≤ c_lo）でステップ関数へ退化する

`_stretch_relevance` が存在しないので ImportError で落ちる。これが正しい RED である。

### RED 2（合成式）

`tests/test_hybrid_score.py` を新設。`_compute_final_score` を直接呼ぶ。

- **手計算との一致**：r・t・a・e を既知にできる入力を作り、$r\cdot(t+e+1.5a)/3.5$ と
  一致することを確かめる（t は `last_recalled_at` を now にして 1.0 に固定、a は
  a0=1.0・n=0 で 1.0、obs_pad=mood_pad で e=1.0 とすれば期待値は解析的に出る）
- **e が効く**：obs_pad を mood_pad から遠ざけるとスコアが下がる
- **e は拒否権でない**：e がほぼ 0 でも、a が高ければスコアは 0 にならない（純積との
  違いを直接押さえる。これが今回の設計判断そのもののテスト）
- **mood_pad=None で e 項が外れる**：$(t+1.5a)/2.5$ と一致する
- **w_r=0 で関連が無効化される**：コサインを変えてもスコアが変わらない

現行の純積実装では、手計算一致・e が効く・e が拒否権でない、のいずれも落ちる。

### RED 3（recall の接続）

`tests/test_emotion_axis_recall_wiring.py` を新設（DB を使う）。

- 感情の異なる観測を2件保存し、mood を一方に近づけて `recall` すると、その一方が
  上位に来る（コサインと時刻を揃えて e だけが差になるようにする）
- **デッドロックの反証**：`recall` が有限時間で返る。C2 のときと同様、mood 読みを
  ロック内へ入れると停止するので、この経路が生きていることを時間で押さえる

### GREEN

上の「改造の内容」を最小で実装する。

### 既存テストの修正要否

- `tests/test_activation_recall_wiring.py:13,21` が `_compute_final_score` を直接呼ぶ。
  新しい必須引数（obs_pad）と既定値の変更（HL・floor）で影響を受けるため、**期待値を
  ハイブリッド式へ更新する**。この2件は a 軸が想起へ効くこと（P-1）の回帰なので、
  テストの意図は保ったまま式だけ合わせる。
- `recall_half_life_days` と `recall_time_floor` の既定値変更が、時間減衰を前提にした
  他のテストへ波及しないかを実行して確かめる。波及したものは、**設計値が変わったこと
  による期待値の更新**として直す（テストを緩めるのではなく、課題5 の値で計算し直す）。

### DB とマイグレーション

**不要**。PAD 4列は W1a（マイグレーション024）で追加済みで、`recall` はすでに SELECT
している。新しい列も新しいテーブルも要らない。

## コメント方針

- `_stretch_relevance` の docstring に、式・係数の出所（課題5 v0.24 D 節）・恒等になる
  現行値・縮退時の扱いを書く。
- `_compute_final_score` の docstring を、純積からハイブリッドへ書き直す。式と、加算部が
  全0のときの $M=1$、mood_pad=None で e を外す理由を書く。
- `recall` の mood 読みには、**ロックの外で読む理由**（`load_current_mood` が `db.lock` を
  取るため・C2 のデッドロックと同型）を一行で残す。これは将来ロック内へ移されると
  再発するので、意図をコードのそばに置く。
- Config の新フィールドには、課題5 の記号（$c_{lo}$ 等）を対応づける短い注記を置く。

## 完了条件

- 上記の新規テストと既存テストが緑。`ruff`・`mypy`・`./scripts/run_tests.sh` が緑。
- `grep -rn "cosine \* state.score" src/` が0件（純積の残存が無いこと）。
- `_emotion_match` の docstring から「未接続」「繋がず」の記述が消えていること
  （`grep -n "未接続" src/familiar_agent/tools/memory.py` で当該行が残らない）。

## ドキュメントへの反映（実装後）

実装して全体テストが緑になってから、実装済みの内容だけを次へ反映する。

- `設計図_Mermaid`：「store と I/F」節へ、合成式のハイブリッド化と e 軸接続、
  `_stretch_relevance` の新設を追記。版上げ。
- `感情ループ全体像`：`M → RECALL`（e 軸）の実装状況を「未実装（スライス3）」から
  実装済みへ。版上げ。
- `直近の進め方と進捗`：Phase 2 節へスライス3 を追記し、「次の一歩」を次の対象へ。版上げ。
- `用語_略語一覧`：`_stretch_relevance` と新 Config 名を該当行へ併記。版上げ。
- `実機確認シナリオ_想起`：気分一致想起の確認シナリオを追記。

---

## 更新履歴

> v0.2：【実装完了】e 軸のスコア接続は実装済み（`_score_breakdown` に w_e・`_emotion_match`・想起は純積からハイブリッド合成へ）。ドキュメント反映（課題5・用語・Mermaid）も実施済み。本書は履歴（改造方針の記録）。
