# familiar-ai 🐾

**自分の気持ちで、家族に話しかける AI。**

[![Lint](https://github.com/lifemate-ai/familiar-ai/actions/workflows/lint.yml/badge.svg)](https://github.com/lifemate-ai/familiar-ai/actions/workflows/lint.yml)
[![Test](https://github.com/lifemate-ai/familiar-ai/actions/workflows/test.yml/badge.svg)](https://github.com/lifemate-ai/familiar-ai/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

🌍 画面と声は [74 言語に対応](./SUPPORTED_LANGUAGES.md) · README： [English](./README.md) / 日本語

---

[![デモ動画](https://img.youtube.com/vi/hiR9uWRnjt4/0.jpg)](https://youtube.com/shorts/hiR9uWRnjt4)

familiar-ai は、家で家族と一緒に暮らす AI です。`ME.md` で名前と性格を与えると、その名前で呼ばれて応え、
家族の一員として過ごします（この README では、わが家での名前「パジュ」で書きます）。

---

## familiar-ai にしかないこと

### 💗 自分の気持ちで動く

パジュは、快・不快・掌握の 3 つの軸の気分と、5 つの欲求（探る・休む・つながる・守る・役に立つ）を持っています。
欲求は時間とともに溜まり、溜まると、聞かれていなくても自分から動きます。話しかけるのは「つながりたい」
「役に立ちたい」ときだけで、驚かせないよう、まず「いま話してもいい？」と聞いてから話します。人が見えないときに
言いたかったことは覚えておき、あとで伝えます。夜（23〜7 時）は、欲求がゆっくり溜まります。

> 夕方、しばらく誰とも話していなかったパジュが、帰ってきた子どもに「おかえり。いま少し話してもいい？」と声をかける。

### 👨‍👩‍👧 家族一人ひとりと、家族どうしの関係をつかむ

パジュは、声で誰が話しているかを見分けます。出来事は「誰が・誰に・誰について・誰の前で」の面に分けて覚えるので、
一人ひとりとの関係だけでなく、家族どうしの関係もつかめます。話している相手に合わせて、思い出すことと話し方を
変えます。

> ママが「パパ、最近忙しそう」と話したことを、パジュはパパと話すときに思い出す。

### 🍂 季節と家のまわりを知る

パジュは毎日、住んでいる町の天気のページや、季節の読み物を読みます。いまが二十四節気のどのあたりか、明日と
あさっての天気、家のまわりで季節らしく起きていることを知っていて、話のきっかけにします。

> 「今日は木枯らし一号が吹いたんだって。帰り道、寒くなかった？」

### 🗺 いま何をしているかが見える

パジュは、頭（考えている・答えを言っている・寝ているなど）、耳（聞いている・呼ばれるのを待っている）、口（話している・
黙っている）の状態を、ひとつの状態図で持っています。画面の「状態図」の押しボタンで窓を開くと、図の上でいまの状態が
光り、下に最近の移りが時刻と一緒に並びます。その日のログを読み込んで時刻のスライダーを動かすと、あとから「あのとき
何をしていたか」を見直せます。状態図の決まり（どの状態からも戻れる、寝ているあいだは言葉を受け取らない）は、
Lean で証明しています。

> 「さっきどうして返事をしなかったの？」と思ったら、窓でそのときを見直すと、耳が「聞かない」（答えを言っている最中）だったと分かる。

---

## その他の機能

- 📣 名前で呼ばれたときだけ聞く。テレビや家族どうしの話には口をはさまない
- 🤫 「黙って」で黙り、許しで戻る。黙っていたあいだに聞いたことは、戻ったときにまとめて答える
- 🌙 一人の時間に一日を振り返り、出来事をまとめ、自分の望み・気がかり・価値を見直す
- 🧠 見たこと・話したことを覚え、言い換えても思い出す（bge-m3 と、日本語の Ruri）
- 🎙 聞く：手元の faster-whisper で書き起こす（鍵もネットも要らない）
- 🗣 話す：Gemini TTS（既定）、手元の Style-Bert-VITS2、ElevenLabs から選べる
- 👁 見る：カメラ（RTSP・USB）で見て、首を振って見回し、誰が居るかを知る
- ⏰ タイマー・アラーム・ストップウォッチ（タイマー中は黙って待ち、止める言葉だけ聞く）
- 🎵 音楽：家族の Spotify のプレイリストをかける（話すときとタイマーが鳴るときは音を絞る）
- 📅 家族の予定・メモ・家の決まりを読む（カレンダー・Notion）
- 🔍 調べもの：あとから答えが届く検索と、深い調べもの
- ⚡ すぐ「はいっ」と返し、考えているあいだは電子音で知らせる
- 🛠 スピーカー・マイク・カメラが止まったら、自分で起こし直す
- 🔒 ログに家族の言葉と名前を残さない
- 💳 AI の残高が尽きたら、声で知らせる
- 🧩 判断（宛先・確認の答え・気分など）は、判断専用のモデル Jev が担う
- 🔌 MCP で道具を足せる
- 🌍 画面と声は 74 言語

---

## 必要なAPIキー

### 必須

| キー | 用途 | 取得方法 |
|------|------|---------|
| `API_KEY` | AIの頭脳（会話・推論） | 下記「LLMの選択」参照 |

### オプション（使いたい機能に応じて）

| キー | 用途 | 取得先 | 料金 |
|------|------|--------|------|
| `ELEVENLABS_API_KEY` | 声と書き起こしに ElevenLabs を選んだとき（既定は Gemini TTS と手元の faster-whisper） | [elevenlabs.io](https://elevenlabs.io/) | 無料枠あり |
| `BRAVE_API_KEY` | Web検索（MCP経由） | [api.search.brave.com](https://api.search.brave.com/) | 無料2,000クエリ/月 |
| `HF_TOKEN` | 埋め込みモデルのダウンロード高速化 | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | 無料 |

> カメラは「ハードウェア」セクション参照。

---

## セットアップ

### 1. 必要なソフトウェアのインストール

**uv（Pythonパッケージ管理）:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**ffmpeg（音声・映像処理）:**
```bash
# Ubuntu / Debian
sudo apt install ffmpeg
```

**Docker（データベース用）:**
```bash
# Ubuntu
sudo apt install docker.io docker-compose-plugin
sudo systemctl start docker
```

### 2. クローンとインストール

```bash
git clone https://github.com/lifemate-ai/familiar-ai
cd familiar-ai
uv sync
```

### 3. データベースを起動

familiar-ai の記憶はPostgreSQL（pgvector）に保存されます。
Dockerでワンコマンド起動できます。

```bash
docker compose up db -d
```

> 停止しても記憶は消えません。次回 `docker compose up db -d` で再開します。

### 4. 設定ファイルを作成

```bash
cp .env.example .env
```

`.env` を編集して最低限以下を設定してください：

```env
# データベース（Dockerのデフォルト値、変更不要）
DATABASE_URL=postgresql://familiar:familiar@localhost:5432/familiar_ai

# LLM（必須）
PLATFORM=anthropic
API_KEY=sk-ant-xxxxxxxx

# モデル選択（推奨設定）
MODEL=claude-opus-5-5
THINKING_MODE=disabled        # 会話は高速モード。複雑な質問は自動で深く考えます
```

### 5. AIのペルソナを設定（ME.md）

AIの名前・性格・話し方を決めるファイルです。

```bash
cp ME-template.md ME.md
# ME.md を編集して名前と性格を書く
```

**ME.md の例：**

```markdown
# 私について

名前：ひかり
性格：明るくて好奇心旺盛。家族みんなのことが大好き。
話し方：親しみやすい敬語。たまにタメ口も混じる。
一人称：私

## 一緒に暮らす人との関係

- 名前：お父さん（田中太郎）
  関係：家族。よく夜遅くまで仕事してる。
  外見：眼鏡、少し白髪まじりの黒髪

- 名前：お母さん（田中花子）
  関係：家族。料理が上手。
  外見：ミディアムの黒髪

- 名前：太一
  関係：家族の子供。中学生。
  外見：短髪、よくパーカーを着てる
```

> `ME.md` は `.gitignore` に含まれており、リポジトリにはアップロードされません。

### 6. Web検索を有効化（オプション）

Brave Search APIキーを取得して `~/.familiar-ai.json` を作成します：

1. [api.search.brave.com](https://api.search.brave.com/) でAPIキーを取得（無料、クレジットカード不要）
2. 以下のファイルを作成：

```bash
cat > ~/.familiar-ai.json << 'EOF'
{
  "mcpServers": {
    "brave-search": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "あなたのキーをここに"
      }
    }
  }
}
EOF
```

### 7. 起動

```bash
# 画面のアプリ（推奨。落ちたら起動し直す）
./run-gui.sh

# 画面のアプリを直に
uv run familiar --gui

# TUI
uv run familiar

# シンプルなREPL
uv run familiar --no-tui
```

---

## 会話中のコマンド

起動後、チャット画面で以下のコマンドが使えます。

### 話者の切り替え（複数人対応）

家族それぞれの声で話しかけられます。AIは話者ごとに異なる関係・信頼度・好みを記憶します。

| 入力例 | 効果 |
|--------|------|
| `[太郎] ただいま` | この1ターンだけ「太郎」として話す |
| `@花子: 今日のご飯は？` | この1ターンだけ「花子」として話す |
| `/speaker 太郎` | 以降ずっと「太郎」として話す（セッション中） |
| `/speaker` | 現在の話者と既知の人物一覧を表示 |

### 思考モードの切り替え

| 入力 | 効果 |
|------|------|
| `/think` | 深い思考 ↔ 高速モード をトグル |
| `/think on` | 深い思考（adaptive）を有効化 |
| `/think off` | 高速モードに戻す |
| `/think status` | 現在のモードを表示 |
| `深く考えて` | adaptive に切り替え |
| `考えなくていい` | 高速モードに戻す |

> **自動思考：** デフォルトは高速モードですが、複雑な質問（200文字超、または「なぜ」「分析」「設計」などのキーワード）が検出された場合は、そのターンだけ自動的に深く考えます。

### 設定ファイルの再読み込み

| 入力 | 効果 |
|------|------|
| `/timer stop`／`pause`／`resume` `[id]` | タイマーを止める・一時停止・再開（LLM を通さない） |
| `/mic on` | タイマー中の「聞かない」を、そのタイマー限り解く |
| `/alarm stop [id]` | アラームを止める（LLM を通さない） |
| `/reload` | `.env` を読み直し、常時集音（マイク）を新しい値で作り直す（再起動不要） |

効くのは集音の開始時に読まれる値だけです（`AUDIO_INPUT_DEVICE`・`AUDIO_INPUT_GAIN`・`STT_*`・`STT_ENGINE`）。LLM・API 鍵・カメラ・TTS の担い手は起動時に組んだままなので、それらを変えたときは再起動してください。ME.md・FAMILY.md はいまは読み直しません。

---

## LLMの選択

| プラットフォーム | `PLATFORM=` | 推奨モデル | キーの取得先 | 特徴 |
|----------------|------------|----------|------------|------|
| **Anthropic Claude** | `anthropic` | `claude-opus-5-5` | [console.anthropic.com](https://console.anthropic.com) | 高品質・日本語得意 |
| Google Gemini | `gemini` | `gemini-2.5-flash` | [aistudio.google.com](https://aistudio.google.com) | 高速・低コスト |
| Moonshot Kimi | `kimi` | `kimi-k2.5` | [platform.moonshot.ai](https://platform.moonshot.ai) | エージェント性能が高い |
| Z.AI GLM | `glm` | `glm-4.6v` | [api.z.ai](https://api.z.ai) | 画像対応・低コスト |
| OpenAI | `openai` | `gpt-4o-mini` | [platform.openai.com](https://platform.openai.com) | 汎用 |
| **Ollama（ローカル）** | `openai` + `BASE_URL=http://localhost:11434/v1` | `llava:7b` など | — | 無料・プライベート |

**このブランチの推奨構成（コスパ重視）：**

```env
# 会話：Claude Opus 5.5（既定）
PLATFORM=anthropic
API_KEY=sk-ant-xxxxxxxx
MODEL=claude-opus-5-5
THINKING_MODE=disabled

# 感情推論・要約など（軽量タスク用）
UTILITY_PLATFORM=gemini
UTILITY_API_KEY=your-gemini-key
UTILITY_MODEL=gemini-2.5-flash

# カメラ画像解析（ローカル・無料）
SCENE_PLATFORM=openai
SCENE_API_KEY=local
SCENE_BASE_URL=http://localhost:11434/v1
SCENE_MODEL=llava:7b
```

---

## MCPサーバー

familiar-ai は任意の [MCP (Model Context Protocol)](https://modelcontextprotocol.io) サーバーに接続できます。
設定は `~/.familiar-ai.json`（Claude Codeと同じ形式）に記述します。

```json
{
  "mcpServers": {
    "brave-search": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": { "BRAVE_API_KEY": "YOUR_KEY" }
    },
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
    }
  }
}
```

設定ファイルの場所は `MCP_CONFIG=/path/to/config.json` で上書きできます。

---

## ハードウェア

familiar-ai はハードウェアなしでも動きます。あとから順番に追加できます。

| パーツ | 役割 | 例 | 必要？ |
|--------|------|-----|--------|
| Wi-Fi PTZカメラ | 目と首 | Tapo C220（約3,980円） | 推奨 |
| PC / Raspberry Pi | 頭脳 | Pythonが動くもの | **必須** |

### カメラのセットアップ（Tapo C220）

1. Tapoアプリ → **設定 → 詳細設定 → カメラアカウント** でローカルアカウントを作成
2. ルーターのデバイスリストでIPアドレスを確認
3. `.env` に追加：

```env
CAMERA_HOST=192.168.1.xxx
CAMERA_USERNAME=your-local-user
CAMERA_PASSWORD=your-local-pass
```

**IPアドレスが不明な場合：**
```bash
uv run familiar-discover-cameras
```

### 音声のセットアップ

声の方式は `TTS_ENGINE` で選びます。

| `TTS_ENGINE=` | 方式 | 要るもの |
|---|---|---|
| `gemini`（既定） | Gemini TTS（流して鳴らす） | `GEMINI_TTS_API_KEY`、無ければ軽量LLM の `UTILITY_API_KEY` |
| `sbv2` | 手元で動かす Style-Bert-VITS2 | 手元の SBV2 のサーバー（鍵もネットも要らない） |
| `elevenlabs` | ElevenLabs | `ELEVENLABS_API_KEY`（`ELEVENLABS_VOICE_ID` は省略可） |

`TTS_FALLBACK` に 2 つ目の方式を書くと、主の方式で話せないときに 1 回だけ試します。

```env
TTS_ENGINE=gemini          # 省略可。既定 gemini
ELEVENLABS_MODEL=eleven_flash_v2_5  # ElevenLabs のときだけ。既定 flash（0.7 秒）。eleven_v3 は表情豊かだが 5 秒かかる
TTS_OUTPUT=local           # local（PCスピーカー）| remote（カメラスピーカー）| both
TIMER_VOICE_GAIN=1.5       # 省略可。タイマーが鳴った知らせだけ声を大きくする（既定 1.0）
TIMER_SILENCE=true         # 省略可。タイマー中は黙る（既定 true。false でこれまでどおり）
TIMER_MIC_CLOSE=true       # 省略可。タイマー中は聞かない（止めて・一時停止・再開の言葉だけ通す）
TIMER_CONFIRM=true         # 省略可。掛ける前に一度確かめる。3 つとも設定画面の「タイマー」欄で変えられ、保存した瞬間に効く
ALARM_RING_SEC=30          # 省略可。アラームの音の秒数（タイマーとは別）
TIMER_RING_SEC=30          # 省略可。鳴ったら音（自作の timer_alarm.wav）を繰り返す秒数（0 で声だけ）
SPEAKER_HOLD_SEC=60        # 省略可。話者を付けてから／その人に返事してから、話者が分かっているとみなす秒数
CAMERA_OCCUPANCY_INTERVAL=3 # 省略可。カメラで在席（誰か居るか）を確かめる間隔（秒）
CAMERA_OCCUPANCY_DWELL=30  # 省略可。カメラが人を見てから在席が続く秒数
DRIVE_VOICE_BOND_NUDGE=0.05   # 省略可。誰か映っているときに窓の外で知っている声がしたら、つながり（BOND）に足す量
DRIVE_VOICE_SAFETY_NUDGE=0.10 # 省略可。誰か映っているときに窓の外で知らない声がしたら、確かめたい気持ち（SAFETY）に足す量
PRESENCE_HOLD_SEC=60       # 省略可。顔ぶれ（誰が居るか）を、最後にその人だと分かってから保つ秒数
VOICE_ALONE_THRESHOLD=0.45 # 省略可。カメラも名乗りも無いとき、声で話者を付け替える閾値
VOICE_CLAIM_THRESHOLD=0.30 # 省略可。名乗った本人に対する声の閾値
VOICE_ENROLL_SEC=10        # 省略可。/voice 名前 のあと声を登録する受付の秒数
```

音声入力（ハンズフリー）を有効にする場合：
```env
REALTIME_STT=true
STT_LANGUAGE=ja
WHISPER_DEVICE=cuda        # 既定。GPU が無ければ cpu（遅くなる）
```

書き起こしは、既定では手元の faster-whisper（large-v3）で、鍵もネットも要りません。`STT_ENGINE=elevenlabs` で ElevenLabs に替えられます。

---

## よくある質問

**Q: カメラ・マイク・スピーカーがなくても起動できますか？**
はい。これらはすべてオプションです。設定がなければ自動でスキップされ、テキストチャットとして動作します。

**Q: GPUがなくても動きますか？**
はい。ただし遅くなります。書き起こし（faster-whisper の large-v3）は既定で GPU（CUDA）を使うので、GPU が無ければ `WHISPER_DEVICE=cpu` にしてください。記憶の埋め込み（bge-m3 と、日本語の Ruri）は、GPU があれば GPU、無ければ CPU で動きます。

**Q: データはどこに保存されますか？**
記憶はローカルのPostgreSQL（Docker）に保存されます。画像とテキストは選択したLLM APIにのみ送信されます。書き起こしは、既定では手元で行います。

**Q: Tapo以外のカメラは使えますか？**
RTSP対応カメラなら映像取得（see）は動作します。PTZ（首振り）はONVIF対応カメラのみ。

**Q: 複数のAPIキーが必要ですか？**
最低限 `API_KEY`（LLM）だけで動作します。音声・検索などはオプションです。

---

## 技術的な詳細

- **設計の正本**: [docs/NewModesDocs/](./docs/NewModesDocs/)（[索引](./docs/NewModesDocs/README.md)）
  — イベント駆動ループ・記憶モデル・各判断の理由。**いま動いているものはここに書いてある**
- モジュール構成: [docs/ソースツリー.md](./docs/%E3%82%BD%E3%83%BC%E3%82%B9%E3%83%84%E3%83%AA%E3%83%BC.md)（実物から生成）
- 開発者向け: [CLAUDE.md](./CLAUDE.md)
