# familiar-ai 🐾

**あなたのそばで生きるAI** — 目・声・足・記憶を持つコンパニオン。

[![Lint](https://github.com/lifemate-ai/familiar-ai/actions/workflows/lint.yml/badge.svg)](https://github.com/lifemate-ai/familiar-ai/actions/workflows/lint.yml)
[![Test](https://github.com/lifemate-ai/familiar-ai/actions/workflows/test.yml/badge.svg)](https://github.com/lifemate-ai/familiar-ai/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

🌍 [74言語に対応](./SUPPORTED_LANGUAGES.md)

---

[![デモ動画](https://img.youtube.com/vi/hiR9uWRnjt4/0.jpg)](https://youtube.com/shorts/hiR9uWRnjt4)

familiar-ai は、あなたの家に住むAIコンパニオンです。
コーディング不要で数分でセットアップできます。

カメラで世界を認識し、ロボット掃除機で部屋を動き回り、声で話し、見たものを記憶します。
名前と性格を与え、家族として一緒に暮らしてください。

---

## できること

- 👁 **見る** — Wi-Fi PTZカメラまたはUSBウェブカメラから画像を取得
- 🔄 **見回す** — カメラをパン・チルトして周囲を探索
- 🦿 **動く** — ロボット掃除機で部屋を移動
- 🗣 **話す** — ElevenLabs TTSで音声出力
- 🎙 **聞く** — ElevenLabs Realtime STTによるハンズフリー音声入力（オプション）
- 🧠 **記憶する** — セマンティック検索で記憶を積極的に保存・想起
- 👥 **複数人を識別** — 家族ごとの関係・信頼度・好みを個別に記憶
- 🔍 **Web検索** — Brave Search MCP（月2,000クエリ無料）
- 🫀 **心の理論** — 相手の視点に立ってから返答
- 💭 **欲求** — 内的な動機を持ち、自律的に行動
- 💡 **適応的思考** — 複雑な質問では自動的に深く考える

---

## 必要なAPIキー

### 必須

| キー | 用途 | 取得方法 |
|------|------|---------|
| `API_KEY` | AIの頭脳（会話・推論） | 下記「LLMの選択」参照 |

### オプション（使いたい機能に応じて）

| キー | 用途 | 取得先 | 料金 |
|------|------|--------|------|
| `ELEVENLABS_API_KEY` | 音声出力（TTS）・音声入力（STT） | [elevenlabs.io](https://elevenlabs.io/) | 無料枠あり |
| `BRAVE_API_KEY` | Web検索（MCP経由） | [api.search.brave.com](https://api.search.brave.com/) | 無料2,000クエリ/月 |
| `HF_TOKEN` | 埋め込みモデルのダウンロード高速化 | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | 無料 |

> カメラ・ロボット掃除機は「ハードウェア」セクション参照。

---

## セットアップ

### 1. 必要なソフトウェアのインストール

**uv（Pythonパッケージ管理）:**
```bash
# macOS / Linux / WSL2
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**ffmpeg（音声・映像処理）:**
```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
winget install ffmpeg
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
MODEL=claude-sonnet-4-6
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
# TUI（推奨）
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

---

## LLMの選択

| プラットフォーム | `PLATFORM=` | 推奨モデル | キーの取得先 | 特徴 |
|----------------|------------|----------|------------|------|
| **Anthropic Claude** | `anthropic` | `claude-sonnet-4-6` | [console.anthropic.com](https://console.anthropic.com) | 高品質・日本語得意 |
| Google Gemini | `gemini` | `gemini-2.5-flash` | [aistudio.google.com](https://aistudio.google.com) | 高速・低コスト |
| Moonshot Kimi | `kimi` | `kimi-k2.5` | [platform.moonshot.ai](https://platform.moonshot.ai) | エージェント性能が高い |
| Z.AI GLM | `glm` | `glm-4.6v` | [api.z.ai](https://api.z.ai) | 画像対応・低コスト |
| OpenAI | `openai` | `gpt-4o-mini` | [platform.openai.com](https://platform.openai.com) | 汎用 |
| **Ollama（ローカル）** | `openai` + `BASE_URL=http://localhost:11434/v1` | `llava:7b` など | — | 無料・プライベート |

**このブランチの推奨構成（コスパ重視）：**

```env
# 会話：Claude Sonnet 4.6（高品質・中コスト）
PLATFORM=anthropic
API_KEY=sk-ant-xxxxxxxx
MODEL=claude-sonnet-4-6
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
| ロボット掃除機 | 足 | Tuya対応モデル | 任意 |
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

### 音声のセットアップ（ElevenLabs）

```env
ELEVENLABS_API_KEY=sk_...
ELEVENLABS_VOICE_ID=...   # 省略可。省略時はデフォルトの声を使用
TTS_OUTPUT=local           # local（PCスピーカー）| remote（カメラスピーカー）| both
```

音声入力（ハンズフリー）を有効にする場合：
```env
REALTIME_STT=true
STT_LANGUAGE=ja
```

---

## よくある質問

**Q: カメラ・マイク・スピーカーがなくても起動できますか？**
はい。これらはすべてオプションです。設定がなければ自動でスキップされ、テキストチャットとして動作します。

**Q: GPUがなくても動きますか？**
はい。埋め込みモデル（multilingual-e5-small）はCPUで動きます。

**Q: データはどこに保存されますか？**
記憶はローカルのPostgreSQL（Docker）に保存されます。画像とテキストは選択したLLM APIにのみ送信されます。

**Q: Tapo以外のカメラは使えますか？**
RTSP対応カメラなら映像取得（see）は動作します。PTZ（首振り）はONVIF対応カメラのみ。

**Q: 複数のAPIキーが必要ですか？**
最低限 `API_KEY`（LLM）だけで動作します。音声・検索などはオプションです。

---

## 技術的な詳細

- アーキテクチャ: [docs/architecture.md](./docs/architecture.md)
- このブランチの変更点: [docs/CHANGES.md](./docs/CHANGES.md)
- 設計思想: [docs/technical.md](./docs/technical.md)
- 開発者向け: [CLAUDE.md](./CLAUDE.md)
