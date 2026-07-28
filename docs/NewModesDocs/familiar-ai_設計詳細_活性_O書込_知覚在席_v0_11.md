# familiar-ai 設計詳細：活性・O書込・知覚在席（定数台帳・現状コード所在・移行）（v0.11）

## 位置づけ
本書は設計図の確定 **[D-活性]／[D-O書込]／[D-B定点]／[D-B分離]／[D-知覚]／[D-設定]** の**別紙詳細**。決定そのものは設計図にあり、本書は **定数台帳／現状コード所在（file:line・移行入力）／知覚パイプライン細部／移行申し送り** を保持する（決定の地の文は設計図に一元化し、本書では繰り返さない）。対象＝課題2 の項目1（活性）・項目2（O書込）・項目3（知覚在席）。**暫定値は課題5、移行は課題6/7**。

---

## 1. 活性（項目1）

更新則の式・形は **[D-活性]**（drive＝蓄積／mood＝平静へ減衰／MI.activation＝on-read 指数減衰、放電＝放電量を引く、想起 score＝関連＋新しさ＋activation）。本書は**定数の全列挙と現状所在**。

### 1-1. 必要定数の全列挙（すべて Config（C）に保管）

**D 系**
| 定数 | 役割 | 種別 | 個数 |
|---|---|---|---|
| 基準レート | 蓄積の基礎速度 | 人・固定 | 各 D＝5 |
| 時間帯倍率 | 蓄積レートの時間帯修飾 | 人・固定 | 各 D×時間帯 |
| 時間帯区切り | 時間帯の境界時刻 | 人・固定 | global |
| 学習倍率 | 蓄積レートの機械調整 | 機械・更新 | 各 D＝5 |
| 学習倍率の上限・下限 | 機械調整の丸め範囲 | 人・固定 | 各 D×2 |
| D 活性の上限・下限 | 蓄積のクリップ範囲 | 人・固定 | 各 D×2 |
| spike 重み | G の賦活を D に乗せる重み | 人・固定 | 各 D＝5 |
| mood 修飾ゲイン係数 | mood が D の蓄積を修飾（乗算は廃止し logit 合成 $g_{D,i}(M)$ に一本化・課題5 C／発火mood §2.2） | 人・固定 | global |
| 放電量 | 発火時に D から引く量 | 人・固定 | 各 D＝5 |
| 発火閾値（TRIGGER） | D 発火の判定閾値（放電の基準にも） | 人・固定 | 各 D＝5 |

**M（mood）系**
| 定数 | 役割 | 種別 | 個数 |
|---|---|---|---|
| mood 減衰時定数 | 平静への指数減衰の速さ | 人・固定 | PAD 成分ごと |
| mood 平静値（baseline） | 減衰の収束先（全軸0.5＝(P,Pn,A,Dom)=(0.5,0.5,0.5,0.5)） | 人・固定 | PAD 成分ごと |
| 覚醒（Arousal）入力重み | G の覚醒を mood に乗せる重み | 人・固定 | global |

**MI.activation 系**（**時間の定数を持たない**：$a$ は time では減らさない＝[D-活性]。時間減衰は新しさ $t$ が一本で担い、想起では $a$ と $t$ が加算部の別項として効く。旧「activation 減衰時定数（salience の指数減衰）」は、$t$ と二重に効くため削除した）
| 定数 | 役割 | 種別 | 個数 |
|---|---|---|---|
| activation 初期化重み | 取込時 activation の初期値（relevance は廃止・seed 種別で surprise（カメラ起点 $\widehat{S}$）／novelty（内容起点）を出し分け・足さない・課題5 E） | 人・固定 | global |

**周期**
| 定数 | 役割 | 種別 | 個数 |
|---|---|---|---|
| T-tick 周期 | T の更新間隔 | 人・固定 | global |
| I 起動 | イベント駆動（周期なし・3キュー待ち） | — | global |

**在席・知覚系（項目3 由来）**
| 定数 | 役割 | 種別 | 個数 |
|---|---|---|---|
| YOLO person スコア閾値 | 在/不在の検出しきい | 人・固定 | global |
| 在席検出ループ間隔 | G の在席検出の周期 | 人・固定 | global |
| 退室しきい（absent_threshold） | 滞留窓（これを過ぎたら退室） | 人・固定 | global |
| 人数しきい | 「いる」とみなす検出数 | 人・固定 | global |
| 定点集合 | N 個の絶対 pan/tilt 位置（norm・在席・見回り共有） | 人・固定 | N |
| 定点マッチ許容範囲 | 現在 pan/tilt を最寄り定点に寄せる許容 | 人・固定 | global |
| norm EMA alpha | エンティティ普通の EMA 係数 | 人・固定 | global |
| 確率 floor | エンティティ確率の下限 | 人・固定 | global |
| DINOv2「普通」EMA 係数 | 見えの普通（埋め込み）の EMA 係数 | 人・固定 | global |
| 見え変化距離しきい | 驚きとみなす埋め込み距離 | 人・固定 | global |
| InsightFace 認識コサイン閾値 | 人物判定の一致しきい | 人・固定 | global |
| モデルパック | buffalo_l 等 | 人・固定 | global |
| 1人あたり参照数上限 | ギャラリーの参照埋め込み数 | 人・固定 | global |
| 識別呼び出し方針 | InsightFace をいつ呼ぶか（新規出現時・要求時 等） | 人・固定 | global |

**score 系（項目4 で確定）**：想起 score の重み（関連／新しさ／activation）。C に保管する点は同じ。

### 1-2. Config（C）集約（[D-設定] の実装詳細）
- **C＝全調整可能定数**（人の固定設定＋機械の学習倍率）。**機械が更新するのは学習倍率のみ**。
- C は既存 `config.py`（env ベースの dataclass 群）を**拡張/包含**する。`time_decay.py`（指数減衰エンジン）は温存し、**half-life 値だけ C に出す**。

### 1-3. 現状の所在マッピング（事実・移行入力）
| 定数 | 現状 | 位置・値 |
|---|---|---|
| 記憶減衰（recall/pending） | ①config 集約 | config.py:160 `recall_half_life_days=7.0`、:173 `PENDING_SPEECH_HALF_LIFE_DAYS=1.0`、:179 `expire_threshold` |
| 発火閾値 | ②ハードコード | desires.py:56 `TRIGGER_THRESHOLD=0.6`（全体共通・per-drive ではない） |
| 放電量 | ②ハードコード | desires.py:57 `DECAY_ON_SATISFY=0.5`（**×0.5 の乗算式**。新設計は「引く」） |
| 基準レート | ②ハードコード | desires.py:86-103 `GROWTH_RATES`（per-desire） |
| 時間帯倍率＋区切り | ②ハードコード | desires.py `_time_modulation`（夜22–6／朝6–10） |
| D 活性 上限・下限 | ②ハードコード | desires.py（cap 1.0／floor 0.0） |
| mood baseline・各しきい | ②ハードコード | mental_state.py（AffectiveState 成分・0.0 等） |
| concern 減衰 | ②ハードコード | concern_engine.py:22 `_DECAY=0.94`（concern_engine は [D-気がかり統合] で廃止・課題11） |
| salience 初期値 | ②ハードコード | attention_schema.py:172 `activation=0.4` 等 |
| norm EMA alpha・floor | ②ハードコード | prediction.py（`_DEFAULT_EMA_ALPHA`／`_PROB_FLOOR=0.01`） |
| tick 周期 | ②ハードコード | _ui_helpers.py:285 `IDLE_CHECK_INTERVAL=10.0`（単一アイドル周期） |
| 学習倍率／adjust_drive | ③無し（新規） | 現状の自己調整は memory.py の behavior_policy/semantic_fact confidence のみ |
| mood 修飾ゲイン | ③無し（新規） | 現状の修飾は時間帯・schedule・rest/energy で mood 由来ではない |
| 減衰エンジン（流用可） | 既存 | time_decay.py（`score=max(floor, exp(-elapsed/tau))`、`tau=half_life/ln2`、reinforce で半減期倍化） |

**形の差異（移行で要対応）**：放電（現状 ×0.5 乗算 → 新「放電量を引く」）、tick（現状 単一 `IDLE_CHECK_INTERVAL` → **T-tick（周期）＋ I イベント駆動（周期なし・3キュー待ち）**）。

---

## 2. O書込（項目2）

規則の本体は **[D-O書込]**（追記＋supersede／重複は二層＝完全一致 dedupe_key・近傍は埋め込み距離／near-dup 統合は前景でなく REST 内省で supersede（案B）／観測→意味・方針の昇格は O の外（案A）／emotion＝PAD／④音楽＝プレイリストは必ず追記・曲は変化時のみ）。本書は**現状所在**。

### 2-1. 現状の所在（事実・移行入力）
| 項目 | 現状 | 位置 |
|---|---|---|
| 追記＋完全一致重複 | dedupe_key 付きイベントログ | memory.py:477-509（SELECT dedupe :492） |
| 投影（materialize） | `materialize_observation` ジョブ／即時 | memory.py:671+、memory_worker |
| 更新＝supersede | observations.superseded_by | find_near_duplicates 内の条件 :1463 |
| 近傍重複 | 埋め込み近傍ペアを返すのみ（自動マージなし） | find_near_duplicates :1457 |
| 観測→意味/方針 昇格 | save 時に kind で昇格（O 書込と密結合） | `_project_observation` :1150（新設計は分離） |
| 呼び出し側キー | `_memory_dedupe_key(種別, テキスト)` | agent.py:1069/1081/1117/2634/2710 |

---

## 3. 知覚・在席・T レジスタ（項目3）

レジスタ構成の決定は **[D-B定点]／[D-B分離]**、知覚二層の決定は **[D-知覚]**。本書は**パイプライン細部と現状/移行**。

### 3-1. T レジスタの現状→移行
| レジスタ | 現状 | 移行 |
|---|---|---|
| drive（float[5]） | `agent_state["drive5"]` | **【Slice 2a/2b 実装済み・接続】** 器＝`drive_register.py`（5欲求 SEEKING／REST／BOND／SAFETY／ESTEEM の `AiDrivers`・各軸 [0,1]・静止0.0／`agent_state` の state_key `drive5`・`load_drives`・`save_drives`）。dynamics＝`core/drive_dynamics.py`（蓄積 $g_{D,i}(M)$・発火 $\Theta_{fire}$・放電 $q$）。`gui._process_queue` のアイドルで毎周回 tick し `drive5` へ永続化する（Slice 2a）。`DRIVE5_AUTONOMOUS`（既定 off）が on で発火→自発ターン結線＝`core/drive_autonomy.py`（`select_fired_axis`・`drive_gate`・`inner_voice_for`・`drive_snapshot`）。ターンには発火軸の内声（Config 文字列・[D-行動選択]）と drive5 定性スナップショットを同梱する。off では既存15欲求 `DesireSystem` が駆動し完全排他（旧15→新5 移行は後続）。PI.drive の全ターンサーフェスは後続で、現状は自発ターンへのスナップショット同梱のみ |
| mood（PAD） | 毎ターン再計算 | **T の mood レジスタとして永続化**（`agent_state` へ）。発火で PI.emotion へ。**【B-1 実装済み・器のみ・未接続】** `mood_register.py`＝4軸 PAD の器 `MoodPAD`／各軸を M_rest=(0.5,0.5,0.5,0.5) へ半減期600秒で収束させる `decay_to_rest`／`agent_state`（state_key `mood_pad`）の `load_mood`・`save_mood`。emotion→PAD 写像 φ（課題11k）と既存 mood へは未接続で外部挙動不変 |
| norm（定点別 EMA＋確率） | prediction.py の `P(entity)` EMA | **定点キーに拡張**＋DINOv2 定点別「普通」。T(G) private |
| presence（定点別 在席） | `self._present` を都度導出 | **YOLO 由来の定点別 presence マップ**（新規）。T(G) private |

各レジスタは固定キーの upsert（現状 `agent_state` の state_key パターン）。**I はレジスタに直接触れず、drive/mood は PI の `drive`/`emotion` として受ける**。

### 3-2. 知覚パイプライン（細部・二層）
| 役割 | 担当・場所 | 技術 | 細部 |
|---|---|---|---|
| 在/不在 | **G（T 側・連続）** | **YOLO（person・GPU）** | RTSP 永続ストリーム監視→現在 pan/tilt 付与→定点別 presence を T(G) レジスタへ。全身ベースで向きに強い。振動中ゲート（[D-向き]）で移動中は更新スキップ |
| 見えの普通／変化 | **G（T 側・連続・安い）** | **DINOv2（ViT-S/B）** | 定点ごとの「普通(EMA)」を norm レジスタに持ち、現フレームとの距離を驚き(S)に。patch 特徴で変化領域も。CLIP 不採用（テキスト不要・構造変化に敏感・Apache-2.0） |
| 人物判定（誰か） | **I 側の内部ツール（必要時）** | **InsightFace（buffalo_l・GPU・1:N）** | person_id↔512次元 ArcFace 埋め込みのギャラリーをコサイン照合。配信ゲートは判定時にこれを呼び、engage できる顔がある＋誰かを在席として使う（推奨案） |
| 意味づけ（何が変わったか） | **I 側（必要時）** | **VLM（scene.py extract_entities・utility backend）** | 驚いた時にエンティティとして解釈。重い LLM なので低頻度 |

- **DeepFace は廃止**（人物判定は InsightFace に一本化。TensorFlow の VRAM 確保癖も回避）。
- フレーム＝**RTSP 永続ストリーム**（`cv2.VideoCapture` 開きっぱなし、`CAP_PROP_BUFFERSIZE=1` か最新フレーム保持スレッド）。毎回 ffmpeg 起動の現状経路を置換。

### 3-3. 在席は pose 条件付き
- PTZ カメラは一度に1方向しか見ない。**単一フレームの不検出＝部屋が空、ではない**。
- 在席＝**定点別 presence マップ（pose→最終在席時刻）＋滞留窓（absent_threshold）集約**。部屋レベルの「誰かいる」＝窓内にいずれかの定点で人を見たか。空判定は関連定点をスイープして窓内に誰もいない時のみ。
- **pose ビニング＝③見回りの「定点」（N 個の絶対 pan/tilt）**。norm・在席・見回りで共有。現在 pan/tilt は最寄り定点に対応づけ、どの定点からも離れた自由移動中は norm/在席を更新しない。

### 3-4. norm（定点別 EMA ＋ 視覚埋め込み）
- **エンティティ層**：prediction.py の `P(entity)` EMA を定点キーに拡張（`P(entity | 定点)`）。観測↑/不在↓・floor。驚き＝その定点の norm と現在観測の差。**自己運動（別定点へ向く）では驚かない**（移った先の定点の norm と比べる・[D-向き] 整合）。
- **見え層**：DINOv2 埋め込みの定点別「普通(EMA)」との距離。エンティティに現れない見えの変化（配置ズレ・明るさ等）を拾う。
- 意味づけが要る時だけ VLM（I 側）。

### 3-5. GPU／VRAM（実機）
- **RTX 3060 12GB**。YOLO ＋ InsightFace(buffalo_l) ＋ DINOv2(ViT-S/B) で概算 2〜4GB＋α、12GB に余裕。FP16/TensorRT は任意。

---

## 4. 移行への申し送り（課題6/7）
- 配信ゲート：`should_deliver_deferred_result`（agent.py:2858）を **4→2 ゲート化**（quiet-hours・社会的文脈を撤去。ゲート＝結果有り／在席。決定は用語一覧・[配信ゲート]）。
- mood を **T の mood レジスタ**として永続化（現状は再計算）。発火時 PI.emotion へ surface。**【B-1 実装済み・器のみ・未接続】** レジスタ `MoodPAD` と M_rest への半減期600秒収束 `decay_to_rest` と agent_state 永続（state_key `mood_pad`）を `mood_register.py` に新設。φ 接続（課題11k）と発火時 surface は後続。
- norm：prediction.py の EMA を**定点キーに拡張**＋DINOv2 定点別「普通」。**視覚エンコーダは新規採用**（課題7「DINO 有無」の答え＝現状無し → **DINOv2 を入れる**）。
- drive：既存 `agent_state["desires"]` を流用（旧欲求名→新5欲求は課題6）。**【B-2 実装済み・器のみ・未接続】** 新5欲求の器 `AiDrivers` と agent_state 永続（state_key `drive5`・"desires" とは別キー）を `drive_register.py` に新設。生きた15欲求は温存し未接続。蓄積 dynamics と PI.drive surface と旧15→新5 移行は後続。
- 在席：**YOLO（新規）**＋pose 条件付き presence マップ。**DeepFace 廃止・InsightFace 採用**。
- 形の差異：放電（×0.5→引く）、tick（単一 IDLE_CHECK→T-tick 周期＋I イベント駆動）。
- **全定数を C へ集約**（現状の散在を移行）。`time_decay.py` 温存。
- **ライセンス確認**：InsightFace の buffalo_l 等モデルパックは**非商用・研究用途のみ**（商用なら別経路）。DINOv2 は Apache-2.0。
- **O 書込**：`_project_observation` の昇格を O 書込から分離（O は出来事のみ）。near-dup 統合（supersede）を **REST 内省**に追加（前景では検出のみ）。emotion 文字列→PAD。④曲変化検出を W 直近記録曲＋H 照合＋dedupe_key で実装。
- **新しさ（t 軸）の若返り＝recall_count・last_recalled_at（017）の役割再編**（A-3 の Phase 1 残務・決定のみ・コードは Phase 2）：この二列は現在、旧 recall の `_compute_final_score`（`final_score = cosine × time_score × importance`）と `_mark_recalled` の中だけで使われる（grep 確認・機械想起 conversation で `recall_count += 1` と `last_recalled_at = now()`、spontaneous で `last_recalled_at` のみ、system は無更新）。新設計での対応づけ：`recall_count`＝新しさ（t 軸）の若返り回数（`time_decay` の reinforce＝半減期倍化）、`last_recalled_at`＝若返りの起点リセット。**activation の n（評価由来の正味デルタ）とは別系統で、想起回数からは引き継がない**。更新トリガは機械想起からフルLLM の参照申告へ移す（[D-想起合成]「機械想起では activation も freshness も触らない」）。二列は旧 recall スコアリングに閉じているため、Phase 1 では現状維持（旧 recall が読む・外部挙動不変）とし、再編のコードは 5軸スコアラを載せる Phase 2（t 軸の若返りと a 軸の (a0,n)）で行う。§1-3 の「減衰エンジン（流用可）＝time_decay.py（reinforce で半減期倍化）」がこの若返りの実体。

---

## 3-6. 実機で確かめたこと（2026-07-28・S1〜S3 実装時）

在/不在の層を実装し、実機（Tapo C211・ファーム 1.2.6・RTX 3060 12GB）で通した。設計の前提が実物と食い違っていた点を、根拠つきで残す。

### 可動範囲は軸で違う

| 項目 | 実測 |
|---|---|
| pan 可動 | 340°（正規化 $[-1,1]$ の全幅） |
| tilt 可動 | 70°（同上） |
| 絶対 pan/tilt | 対応（`AbsoluteMove`・`[D-向き]` の確認と一致） |
| 絶対移動の誤差 | $10^{-6}$ 未満 |
| プリセット | 最大8件・実機に1件 |

**同じ正規化値でも tilt は 5 倍狭い角度を表す。** 定点の同一視を素のユークリッド距離で測ると tilt 側だけ 5 倍厳しくなるので、**pan 換算に揃えて**測る（`poses.py`）。しきい値は 0.02（pan で 3.4°）。

### カメラ側の動体追尾は定点と競合する

Tapo は動体を検知すると**自分で首を振って追尾する**。追尾が向きを変えるので、こちらが定点を管理する前提が崩れ、振動中ゲートが常に働いて**在席が一度も記録されなかった**。**カメラ側の追尾を切る必要がある**（アプリの設定・ONVIF からは操作していない）。

### ONVIF のイベント

| 項目 | 実測 |
|---|---|
| 対応 | `WSPullPointSupport = True`（Profile S のみ） |
| トピック | `CellMotionDetector/Motion`（`IsMotion`）と `TamperDetector/Tamper` の2つだけ |
| 人検出 | **ONVIF には出ない**（仕様上はカメラも持つ）。人の判定は YOLO が担う |
| 購読の宛先 | カメラが能力一覧に載せない。**購読で返る宛先を手で入れてから**サービスを作る |
| 安定性 | よく切れる。切断は通常の流れとして扱い、間隔を空けて購読し直す（1秒で復帰を確認） |

**`_cam_onvif` は `_ensure_connected()` を呼ぶまで `None`** なので、起動直後に素の属性を渡すと購読は「カメラが無い構成」とみなして静かに終わる。

### 確認の頻度

動いているあいだ動体イベントは毎秒何件も飛ぶ。素直に従うと**0.15 秒ごと**に撮って YOLO を回していた。動き始めの1件は待たせず、続くぶんは下限間隔（3秒）で間引く。

| 値 | 既定 | 意味 |
|---|---|---|
| `CAMERA_PRESENCE_INTERVAL` | 30 秒 | 動きが無いときの確認間隔 |
| `CAMERA_PRESENCE_MIN_GAP` | 3 秒 | 動体で起こされたときの下限間隔 |
| `CAMERA_PRESENCE_WINDOW` | 120 秒 | 滞留窓（`課題5` §I の在席 timeout） |
| `CAMERA_POSE_TOLERANCE` | 0.02 | 同じ定点とみなす距離（pan 換算 3.4°） |

YOLO（`yolo11n`・GPU）の推論は**初回 1.9 秒、以降 8 ミリ秒**。30 秒間隔なら負荷は無視できる。

### 自律は起動時に回り始める

I も T も在席センサも動体イベントも、実装では `run()` の中、しかも人の入力があるときにしか立たなかった。**起動しても、話しかけるまで何ひとつ回っていない。** これは3つの症状の同じ根である（`/speaker` を最初に打つと入室が立たない、在/不在が「連続」にならない、保留していた発話を配る起点が来ない）。`start_autonomy()` を新設し、GUI と CUI の両方の入口から呼ぶ。

---

## 5. 項目4・W（確定・参照先明記）
- **項目4（W）【確定】**：W は派生ビュー（[D-記憶単一化]）なので**退避／eviction／fade は無い**。W 構築（想起重み）＝[D-想起合成]＋[D-プロファイル調整]、消費（want+result）＝[D-単一想起]（充足／不足／失敗→フルLLM が解決宣言・毎ターン破棄して O から再構築・[D-反復出力]）、見た定点の印・作業文脈・直近記録曲＝O の MI として残り W 構築で持つ（課題3 クローズで確定）。**残る値**は重み・正規化規約（課題5 D）・c_lo/c_hi 初期値（課題7）・e/σ/λ（課題11(k)）。

## 参照する確定事項
[D-活性]／[D-O書込]／[D-B定点]／[D-B分離]／[D-知覚]／[D-設定]／[D-記憶単一化]／[D-周期]／[D-向き]／[D-気がかり統合]／[課題10]。

---

## 更新履歴

> v0.11：**§3-6「実機で確かめたこと」を新設**（S1〜S3 の実装で得た事実）。可動範囲が軸で違うこと（pan 340°・tilt 70°）と、そのため定点の距離を pan 換算に揃えること。カメラ側の動体追尾が定点と競合し、切らないと在席が一度も記録されないこと。ONVIF のイベントは動体のみで人検出は出ないこと、購読の宛先を手で入れる必要があること、よく切れること。確認の頻度（30秒・下限3秒・滞留窓120秒）と YOLO の実測（初回1.9秒・以降8ミリ秒）。自律が起動時ではなく人の発話後にしか回っていなかったこと。**`CameraMotionWatcher`（v0.9 で入れた動体検知）は `MotionEventWatcher` へ置き換えた**（現行ターン駆動への接地から、QD 経由の在席センサ起こしへ）。
> v0.10：定数台帳から **activation 減衰時定数**を削除。$a$ は time では減らさない（[D-活性]）と確定しているのに、$a$ 自身が指数減衰する前提の定数が残っており、新しさ $t$ と二重に効く記述になっていた。

> v0.9：カメラ動体検知→知覚ターン起動（案B）を反映。**DIF（純イベント駆動 I）は未実装**なので、動体イベントは当面**現行ターン駆動へ接地**する（将来 DIF ができたらそこへ載せ替え）。`recognition/motion_watcher.py`＝`CameraMotionWatcher`（`presence_watcher` と同型の asyncio タスク・ONVIF PullPoint 購読＝`create_pullpoint_manager`→`SetSynchronizationPoint`→`PullMessages` ループ・Tapo は HTTPS webhook 非対応のため pull 採用）。動体検知で `agent._note_motion` が保留フラグ `_motion_pending` を立て、GUI アイドルループ（deferred 配信と同位置）が拾って**知覚ターン**を起こす（`inner_voice`＝Config `motion_inner_voice`・行動は主LLM が選ぶ・見え驚きは既存 `see`→SceneTracker 経路）。デバウンス（既定60秒・`Debouncer`）でバーストを1ターンにまとめ、long-poll 待機（既定60秒）は通信管理値で検知遅延にならない。ゲート＝静穏/沈黙/入力待ちを避け、**在席は問わない**（不在時の動きも気づく）。Config `MOTION_WATCH`（既定 off）で opt-in。
> v0.8：起動時キャッチアップ（案B）と Drive 新機能の既定 on 化を反映。(1) 停止中の経過を初回 tick に積む：`gui._initial_drive_tick_time` が `drive5.updated_at` を読み、初回 `dt = now − updated_at`（停止秒数）で `accumulate` する（`drive_register.load_drives_with_updated_at`／`catchup_dt`）。cap は設けず accumulate の [0,1] クリップ任せ。mood は起動時 snapshot 近似。(2) 実行時フラグ `DRIVE5_AUTONOMOUS`／`DRIVE5_SATISFY_LLM` の**コード既定を on** へ（新機能を前提）。明示無効化は env `=0`。legacy 経路は `=0` で従来どおり。
> v0.7：P1（知覚→save の視点列配線）を反映。`agent._run_post_response_pipeline` の観察 save と会話 summary save が、視点列（`writer_id`/`subject_id`/`participants_json`/`scope`）を PMM から埋めるようにした。観察＝エージェント自身の情景観察（`writer_id=AGENT_SELF`・`scope="scene"`・`subject_id`＝現話者 floor `DEFAULT_PERSON_ID`）、会話 summary＝話者との遣り取り（`writer_id=subject_id`＝現話者 floor `DEFAULT`・`scope="speaker"`）、`participants`＝在席者（`get_present_ids`）。`scope` は現状 recall フィルタに使わず、将来 V2 が participants/writer/subject と合わせ関係エッジ（presence/speaker/subject）を作るためのラベル。実装＝`agent._observation_perspective`/`_conversation_perspective`。`day_summary` 等の要約系は対象外（REST/P2）。
> v0.6：Drive 起動源の dynamics 接続（Slice 2a）と発火から自発ターンへの結線（Slice 2b）を反映。§3-1 の drive 行を更新。`core/drive_dynamics.py`（蓄積 $g_{D,i}(M)$・発火・放電の純関数）を `gui._process_queue` のアイドルで毎周回 tick して `drive5` へ永続化する（Slice 2a）。`DRIVE5_AUTONOMOUS`（Config・既定 off）が on のとき、発火軸のうち蓄積（放電前）最大の1軸を選び、在席と静穏でゲートして自発ターンを起こす（Slice 2b）。ターンには発火軸の内声（Config 文字列・行動非指定・主LLM が選ぶ）と drive5 の定性スナップショット（低 0.5 未満・中・高 0.75 以上・Config）を同梱する。放電は発火時（案A）。off では既存15欲求 `DesireSystem` が従来どおり駆動し、on とは完全排他（旧15→新5 移行そのものは後続）。
> v0.5：確定設計に合わせ定数台帳の2行を更新。mood による D への修飾は「乗算ゲイン」を廃し logit 合成 $g_{D,i}(M)$ に一本化（課題5 C・発火mood §2.2）。activation 初期化は relevance を廃し、seed 種別で surprise（カメラ起点 $\widehat{S}$）／novelty（内容起点）を出し分ける（足さない・課題5 E）。
> v0.4：B-2（drive（5欲求）レジスタ）の器実装を反映。§3-1 の drive 行と §3 移行まとめに、`drive_register.py`（5欲求 SEEKING／REST／BOND／SAFETY／ESTEEM の器 AiDrivers・各軸 [0,1]・静止0.0・agent_state の state_key drive5 への load_drives/save_drives）が器のみ実装済み（未接続）であることを追記。蓄積と放電と mood 変調（dynamics）と PI.drive surface は後続。既存15欲求 DesireSystem と "desires" キーは無変更で外部挙動不変。
> v0.3：B-1（mood（PAD）レジスタの器）実装を反映。§3-1 の mood 行と §3 移行まとめに、`mood_register.py`（4軸 PAD の器 MoodPAD・M_rest への半減期600秒収束 decay_to_rest・agent_state の state_key mood_pad への load_mood/save_mood）が実装済み（未接続）であることを追記。emotion→PAD 写像 φ（課題11k）と既存 mood へは未接続で外部挙動不変。
> v0.2：A-3 の Phase 1 残務の決定を反映。観測列 recall_count・last_recalled_at（017）を、activation の n ではなく新しさ（t 軸）の若返りに対応づけ、更新トリガをフルLLM 参照申告へ移すことを §4 に明記。二列は旧 recall スコアリングに閉じるため Phase 1 は現状維持とし、再編は Phase 2（5軸スコアラ）で行う。
> v0.1：課題2 項目4 クローズを反映（§5 を「項目4・W（確定・参照先明記）」へ更新）。以降この別紙は版番号で管理する。
