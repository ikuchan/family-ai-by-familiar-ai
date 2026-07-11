# familiar-ai MI データモデル v2（最小・確定）
 
v1 を全面差し替え。本セッションの確定（LLM を解釈基盤に・属性最小化・B 解体・T↔I 境界＝PI）を反映。
 
## 0. 方針
LLM を解釈基盤とするので、**I 内部の意味（意図／未応答／由来／動作 等）は属性にせず `content` に置き LLM が解釈**する。属性は「**T が作る信号**」＋「**機械的必須**」だけ。MI は**単一クラス**（抽象基底・サブタイプは作らない）。
 
## 1. PI（基底）と MI（拡張）
 
- **基底型 `PI`（Primitive MI）** ＝ T が作る「感じ＋欲」だけ：**`emotion`(PAD)・`drive`(5欠乏)**。
- **`MI` ＝ `PI` ＋ `id`・`content`・`vector`・`supersedes`・`activation`**（I が「同定・意味・索引・版・salience」を足す）。**`content`・`vector` は T に無い**（I が付ける）。
- **`timestamp` は store のメタdata**（O が書込時刻を持つ）。減衰／新しさに使うが**属性に数えない**。
- norm/presence は T(G) の private レジスタで **PI/MI に含めない**（§3・[D-B分離]）。
| 属性 | 型 | 定義域 | 層 | 想定した使い方 |
|---|---|---|---|---|
| `emotion` | PAD | 下表 | **PI**（T 信号） | 感じ。発火＝M がそのまま／観測＝I の値踏みが埋める。M congruence 想起 |
| `drive` | float[5] | [0,1]×5（5欠乏） | **PI**（T 信号） | 欲。発火＝発火した欲／観測＝生成時の動機文脈。**D→調停が構造で読む** |
| `id` | UUID(str) | 一意 | MI | 同定 |
| `content` | str | 自由文 | MI | 実体。**意味は全部ここ → LLM 解釈** |
| `vector` | float[d] | 埋め込み(e5) | MI(索引) | 関連想起。**I の取り込みで計算**（embedding は I 資源） |
| `supersedes` | UUID? | — | MI | 版チェーン（追記＋supersede） |
| `activation` | float | [0,1]（clip・定数C） | MI（**I の salience**） | 強度＝salience かつ「開いている度（未解決）」。**取込時 surprise(+novelty+relevance) で初期化・store timestamp で on-read 減衰**。`status` はこれに吸収（開=高／解決＝落とす・supersede） |
 
### 値オブジェクト `emotion`(PAD)
| 属性 | 型 | 定義域 |
|---|---|---|
| `P` | float | [0,1]（rest 0） |
| `Pn` | float | [0,1]（rest 0・P と独立／両価） |
| `A` | float | [0,1]（rest 0・←驚き） |
| `Dom` | float | [0,1]（**rest 0.5**・対処可否） |
 
### 廃止した属性と回収先
- `pose` → `content`（位置は知覚が MI に落ちる時に content へ。**定点は T 内部だけで使う**）。
- `status` → `activation`（開=高／解決=落とす・supersede）＋ `content`。
- `source`／`source_emotion`／`participants`／`scope` → `content`（LLM 解釈・関連想起）。`source_emotion` は**その MI 自身の `emotion`**。
- `actionable_when` → 廃止。**調停が毎ターン W から判断**（即／優先度／到着＋ゲート）。field でなくロジック。
- `target`（動作コマンド） → `content`（動作の意図）＋ハンドラ解釈（構造化コマンドを MI に持たない）。
- `persist`（作業文脈フラグ） → 廃止。W 派生化で「非減衰保持」概念が消える。「見た定点の印」等は **O の MI として残り想起される**。
- `fire_payload` → 廃止。M/D は **PI＝`emotion`/`drive` に構造化済み**・源カテゴリは `content`。
- `dedupe_key` → 廃止。完全一致重複防止は **O 書込の実装責務**で MI field でない。
- `drive_tag` → **`drive`（PI 要素に昇格）**。content にシリアライズしない。
- `urgency`／`novelty` → `activation`／`emotion.A`。
- `meta`／`state_type`／`state_value` → 廃止。
- `timestamp` → **store メタdata**（属性でない）。
- `score` → 導出（保存しない）。
## 2. 想起（W 構築）の重み ＝ 全部機械
 
> recall score ＝ 関連(`vector`) ＋ 新しさ(store `timestamp`) ＋ `activation`
 
3つとも機械で取れるので **W 構築（毎ターン）に LLM 不要**。LLM は評価／生成でのみ働き、**解決時に `activation` を落とす**（次ターンの機械想起が従う）。＝activation が「LLM の解釈」と「機械の想起」を繋ぐ一点。（重みの合成・値は課題5。）
 
## 3. T の数値レジスタ（MI でない）＝ B の解体
 
B は「単一データモデル」ではなく、**T の内部レジスタ**に解体する。drive/mood は PI に昇格し、norm/presence は G の private：
 
| register | 型 | 用途 | PI への surface |
|---|---|---|---|
| drive | float[5]（5欠乏） | D 蓄積・閾値発火 | **PI.drive**（発火＝発火欲・構造で載る） |
| mood | PAD | M 減衰（**平静 P=0,Pn=0,A=0,Dom=0.5 へ漸近**）＋覚醒入力 | **PI.emotion**（発火＝M そのまま） |
| norm | 定点別 EMA(vector)＋確率 | **G の知覚的驚き／異常の基準**（現在の見え vs norm → 驚き → `emotion.A`・③見回りの異常検知） | **なし**（T(G) private・I は想起しない） |
| presence | 定点別 在/不在＋最終在席 | **G の在席**（H 相当・知覚/機材レベル） | **なし**（T(G) private・I は想起しない） |
 
I は norm/presence に触れず、**M/D は情動・欲として PI で受ける**（[D-B分離] 改訂）。
 
## 4. T↔I 境界 ＝ PI（TIF が構築）→ I で MI に拡張
 
T 内部は数値レジスタ。**境界を渡るのは `PI`＝{`emotion`, `drive`}**。発火時、TIF が PI を構築：
 
- `emotion` ← M(PAD)
- `drive` ← D(5欠乏)
- （発火源カテゴリ等の**記述は I が `content` へ**。drive は構造で載るので content シリアライズ不要。）
**I の取り込みで PI → MI に拡張**：`id`・`content`（発火の記述）・`vector`（埋め込み）・`supersedes`・`activation`（取込 salience）を足す。**`timestamp` は store が書込時に付与**。
 
知覚は別経路（DIF→I）。**I→T の Nudge を PI/MI にするかは未決**（対称化可能）。
 
## 5. 層構造
 
- **T 内部**：数値レジスタ（drive=5float・mood=PAD・norm=EMA・presence）。
- **T↔I 境界**：**`PI`＝{emotion, drive}**（TIF が emotion←M・drive←D を構造で載せる。知覚は DIF→I）。
- **I**：**MI＝PI＋{id, content, vector, supersedes, activation}**（O とその W 射影）。**timestamp は store メタdata**。W は store でなく**派生ビュー**。
## 6. 未決（次段）
 
- recall 重みの合成・値（課題5）。
- `activation` の最終定義（取込初期化＝surprise+novelty+relevance／on-read 減衰、の方向は確定・微調整は検討継続）。
## 付録A. 旧 kind → 新（移行早見表）
 
`kind` 廃止（案A）の確定を吸収。各行は本文・設計図 [D-…] の再掲だが、改造時の一覧用に畳む。
 
| 旧 kind | 新しい所属・表現 |
|---|---|
| observation | **O の MI**（`content`＝観測、`emotion`＝I の値踏み） |
| fire | **PI**（境界を渡る。I で MI 化：`emotion`←M／`drive`←D） |
| drive | **PI.drive ／ T の drive レジスタ**（MI でない） |
| mood | **PI.emotion ／ T の mood レジスタ**（MI でない） |
| norm／presence | **T(G) private レジスタ**（MI でない・I は想起しない） |
| cue（きっかけ） | **O の MI**。想起（関連＋`activation`）で W に載る。専用種別なし |
| intention／want（目標） | **O の open 意図 MI**（`content`＝意図）。未解決度は `activation` |
| pending（結果待ち） | **概念廃止**。完了は関連＋未解決で O の open 意図を想起で再会（[D-単一想起]・相関ID 無し） |
| suspended（退避） | **概念廃止**。W は毎ターン破棄・再構築。salience が下がれば載らないだけ（退避 store 無し） |
| 動作要求／呼出要求 | **O の MI**（`content`＝動作・呼出の意図）。実行は調停→生成/動作（投げっぱなし） |
| timer | **課題9 で別途**（時刻 due の扱いは未確定） |
 
**観測 MI の `content`（設計要求）**：観測種別に応じて自然文の中身が入る（LLM が解釈・構造化コマンドは持たない＝[D-MIモデル]）。**ユーザー発話**＝ASR テキスト／**機器イベント（カメラ等）**＝scene・VLM の記述テキスト（Y-2・部屋レベル・定点非記載）／**検索・取得**＝フルLLM が束を畳んだ consolidated 内容（生の結果でなく整理後・[D-O書込]／[D-検索]）／**音楽等の機器状態**＝出来事の記述（曲・プレイリストの変化）。これにより観測 MI 化時の値踏み入力「いま起きたこと」は当該 `content` から取れる（[D-値踏み]）。**open 意図の `content`＝意図**（上表）と合わせ、値踏み入力〔いま起きたこと＝観測 `content`／気がかり＝open 意図 `content`＋`activation`〕が設計要求として裏づく。