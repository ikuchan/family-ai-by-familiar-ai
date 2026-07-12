# familiar-ai MI データモデル v2（最小・確定）
 
v1 を全面差し替え。本セッションの確定（LLM を解釈基盤に・属性最小化・B 解体・T↔I 境界＝PI）を反映。

> v0.07 改訂：§7 の読み出し器を実装済み（未接続）に更新。`_read_supersede_chain(head_id, columns)` を新設（`WITH RECURSIVE` で supersede 版チェーンを再構成・dumb・未接続・テスト4件）。畳み込み本体は Phase 2。

> v0.06 改訂（系統A の対応づけ確定＝論点1c）：付録A に `self_model`（→自己認識 MI 自己エピソード部・REST 蒸留・能力部は capability_summary）／`curiosity`（→cue／SEEKING の open 意図 O・自己認識 MI でない）／`semantic_facts／behavior_policies`（→信念 MI・自己認識 MI 方針とは別・REST が間接蒸留）の行を追加。

> v0.05 改訂：付録A の移行早見表に `utterance／witnessed／scene` 行を追加＝単一 O＋situated 関係エッジ（speaker/presence/subject）へ一本化し旧 `_remember` の人ごと複製を廃止（[D-在席相関/V2]）。

> v0.04 改訂：§7 の confidence を確定。**信頼度は数値属性を持たず MI の `content` に自然文注記として書くにとどめる**（検索の5軸に入れない・機械可読スカラ不要）。数値導出案（activation 同型の `(c0, m)`）は検討のうえ撤回。信頼度の更新（確証／反証／使われない）は REST 内省が結末を読み content を書き換えて supersede する形とし Phase 2 寄り。`_project_observation`・専用2テーブル・`fact_key`／`policy_key`・`confidence`・`adjust_*_confidence`・`memory_revisions`（confidence 版）は撤去対象として明記。§7 を〔設計確定・実装未着手〕へ。

> v0.03 改訂：MI 集約段の設計中の記録として §7「意味・信念層（旧 semantic_facts／behavior_policies）の畳み込み」を追加。key／revisions／confidence の意味を整理し、**畳み込む方針**を確定。key と revisions は MI の supersede 版チェーンへ写す（identity＝chain 到達・revisions＝祖先の再帰想起・W 取り込み＝`superseded_by IS NULL`・identity キーは足さない）。書き込み側の紐づけ（old_id 同定）は類似度／REST へ寄せ Phase 2 寄り。confidence の写し先は未決（次の議論）。実装は未着手。
 
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

## 7. 意味・信念層（旧 semantic_facts／behavior_policies）の畳み込み〔設計確定・実装未着手〕

現行の `semantic_facts`／`behavior_policies` は、observations（エピソード）の上に載る**畳み込み済みの意味・信念層**である。書き込み元は `_project_observation` の固定キー投影のみで（`self_model:core`／`curiosity:active`／`conversation:support`）、独立した LLM 抽出は未接続。各行は次の三要素を持つ。

- **key**（`fact_key`／`policy_key`）：同じ信念を時間をまたいで指す identity。同一キーへの再投影は行を増やさず UPDATE する。
- **revisions**（`memory_revisions`）：本文・confidence が変わったときの改訂履歴（旧→新）。
- **confidence**：その信念をどれだけ信じてよいかの信頼度。想起の絞り・並びには使わず、LLM 文脈へ `conf:0.85` の形で注入し、`memory-evidence-confidence` 制約で「0.55 未満は不確か」と読ませる。経験（会話の結末）で `adjust_*_confidence` が増減させる。

**方針は MI へ畳み込む**。key と revisions は MI の `supersedes` 版チェーンへ写す。

- **identity ＝ supersede チェーンの到達可能性**。identity キーは足さない。信念を更新するたびに新 MI を書き、旧 MI を `superseded_by = 新id` で閉じる。
- **revisions ＝ 祖先の再帰想起**。現行版（`superseded_by IS NULL`）を起点に `WITH RECURSIVE` で祖先へさかのぼれば、その信念の改訂履歴が chain から再構成できる（旧本文・旧 confidence は旧 MI に残る）。`superseded_by` には索引 `idx_obs_superseded` があり再帰は安価。多対一の収束（重複を最古へ畳む）も既存の型。**【実装済み・未接続】**この再帰想起の器を `memory.py` の `_read_supersede_chain(head_id, columns)` として新設（現行版を起点に `superseded_by` を `WITH RECURSIVE` でさかのぼり head〔depth 0〕＋祖先を depth 昇順で返す dumb な読み出し・採点や想起判断は持たない・既存経路からは未接続・テスト4件）。系統B の畳み込み本体（投影の撤去と REST 駆動の content 改訂）は Phase 2。
- **W への取り込み ＝ `superseded_by IS NULL`**。既存の全想起経路がこの絞りを持つので、chain の現行版だけが W に載る。今の「キーごとに生きた1行」と同じ効果が追加機構なしで出る。

**残る書き込み側の紐づけ**：chain は linkage を記録するが、新しい信念版が来たとき `mark_superseded` に渡す old_id（どの現行 MI を置き換えるか）を何が同定するかは別問題。キーを外すので、この同定は**類似度／REST に寄せる**（`find_near_duplicates` のベクトル近傍、または REST 内省が意味的に同じ現行 MI を見つけて supersede）。固定キーの即時投影 `_project_observation` は、キーレス化で類似度／REST ベースの supersede へ置き換わる。REST 内省は未実装のため、**書き込み側 consolidation は Phase 2（REST）寄り**。読み出し側（再帰想起で履歴・`superseded_by IS NULL` で現行版）は既存機構で成立する。

**整理事項**：MI dataclass の `supersedes` フィールドは行の `superseded_by` から読んでおり（`memory.py`）、名は「前版を指す」だが実体は「次版に置き換えられた」。再帰想起を素直に書くため、畳み込み実装時にこの向きを整理する。

**confidence の畳み込み〔確定〕**：confidence は**数値属性として持たず、信頼度を MI の `content` に自然文の注記として書く**にとどめる（「この方針は何度かうまくいっている」「まだ確信は薄い」等）。MI に confidence カラムも `(c0, m)` のような導出用スカラも足さない。理由は、confidence は検索に一切効かせない（5軸 r/t/e/a/p に入れない）ため機械可読なスカラである必要がなく、MIデータモデルの「属性は T 信号＋機械的必須だけ・意味は content」に沿うから。数値導出案（activation と同型に `(c0, m)` からロジスティックで導く案）は検討したうえで撤回した（検索に効かないので数値化の利得がない）。

信頼度の更新（確証／反証／使われない）は、**REST 内省がその日の結末を読み、信念 MI の `content` を書き換えて supersede** することで反映する。確証＝結末が信念を裏づけた、反証＝使ったが結末が反した、使われない＝一定期間 W へ引かれず再検証されない、の三つを REST が読み取り、注記を強める／弱める／薄める。`adjust_*_confidence` の即時 ±delta と `memory_revisions` の confidence 版は、この REST 駆動の content 改訂に置き換わる（online の数値即時更新は持たない）。REST 内省は未実装のため、信頼度の更新は **Phase 2（REST）寄り**。

**畳み込みで消える現行機構**：`_project_observation` の固定キー投影、`semantic_facts`／`behavior_policies` テーブル、`fact_key`／`policy_key`、`confidence` カラム、`adjust_*_confidence`、`memory_revisions`（confidence 版）は、キーレス supersede チェーン＋content 注記へ畳まれるため撤去対象。撤去は実装スライスで grep 0件を完了条件に含める。

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
| utterance／witnessed／scene | **単一 O の MI ＋ situated 関係エッジ**（[D-在席相関/V2]）。旧 `_remember` の人ごと複製（話者/目撃/場面）を廃し、1つの O に `speaker`（←`writer_id`）／`presence`（←`participants_json` の各在席者）／`subject`（←`subject_id`）の関係エッジを付ける。「[X が言った]」等の視点は content と関係エッジで表す |
| self_model | **自己認識 MI の自己エピソード部**（REST が日付で O を読み返し一人称に蒸留し supersede 更新・pinned）。旧 `self_narrative_log` 廃止・morning-context 注入から pinned へ移す。能力部は `capabilities.yaml`→`capability_summary` の LLM 要約が担い REST が更新 |
| curiosity | **cue／SEEKING の open 意図 O**（[D-想起起動]）。自己認識 MI ではない。専用種別なし・想起で W に載る |
| semantic_facts／behavior_policies | **キーレス supersede チェーンの信念 MI**（§7）。信頼度は content 注記・REST が更新。自己認識 MI の方針(policy)とは別（自己認識 MI 方針＝核＋Config・pinned／behavior_policies＝W 想起の belief MI）。REST が繰り返し確証された belief 方針を自己認識 MI の方針へ一般化蒸留する間接経路のみ |
| timer | **課題9 で別途**（時刻 due の扱いは未確定） |
 
**観測 MI の `content`（設計要求）**：観測種別に応じて自然文の中身が入る（LLM が解釈・構造化コマンドは持たない＝[D-MIモデル]）。**ユーザー発話**＝ASR テキスト／**機器イベント（カメラ等）**＝scene・VLM の記述テキスト（Y-2・部屋レベル・定点非記載）／**検索・取得**＝フルLLM が束を畳んだ consolidated 内容（生の結果でなく整理後・[D-O書込]／[D-検索]）／**音楽等の機器状態**＝出来事の記述（曲・プレイリストの変化）。これにより観測 MI 化時の値踏み入力「いま起きたこと」は当該 `content` から取れる（[D-値踏み]）。**open 意図の `content`＝意図**（上表）と合わせ、値踏み入力〔いま起きたこと＝観測 `content`／気がかり＝open 意図 `content`＋`activation`〕が設計要求として裏づく。