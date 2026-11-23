# familiar-ai 

<img width="25%" height="25%" alt="familiar-ai-icon" src="https://github.com/user-attachments/assets/944b2023-9ca0-4b30-8240-de766e4439ed" />

**An AI that talks to your family because it wants to.**

[![Lint](https://github.com/lifemate-ai/familiar-ai/actions/workflows/lint.yml/badge.svg)](https://github.com/lifemate-ai/familiar-ai/actions/workflows/lint.yml)
[![Test](https://github.com/lifemate-ai/familiar-ai/actions/workflows/test.yml/badge.svg)](https://github.com/lifemate-ai/familiar-ai/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/kmizu?style=flat&logo=github&color=ea4aaa)](https://github.com/sponsors/kmizu)

🌍 UI and voice in [74 languages](./SUPPORTED_LANGUAGES.md) · README: English / [日本語](./README_ja.md)

---

[![Demo video](https://img.youtube.com/vi/hiR9uWRnjt4/0.jpg)](https://youtube.com/shorts/hiR9uWRnjt4)

familiar-ai is an AI that lives at home with your family. Give it a name and a personality in `ME.md`; it answers
to that name and spends its days as one of the family. (This README calls it "Paju", its name in our home.)

## What only familiar-ai does

### 💗 It acts from its own feelings

Paju has a mood on three axes (pleasure, displeasure, being in control) and five drives (seeking, rest, bond,
safety, esteem). The drives build up over time, and when one fills, Paju acts without being asked. It starts a
conversation only when it wants company or wants to be useful, and to avoid startling anyone it first asks
"Is now a good time to talk?". Whatever it wanted to say while nobody was in view, it keeps and says later. At
night (23:00–07:00) the drives build up slowly.

> In the evening, after a quiet afternoon, Paju greets a child coming home: "Welcome back. Can we talk for a bit?"

### 👨‍👩‍👧 It understands each person, and how the family relate to each other

Paju tells who is speaking by voice. It stores each event as facets — who did it, to whom, about whom, in front of
whom — so it grasps not only its relationship with each person but also the relationships between family members.
What it recalls and how it speaks change with the person it is talking to.

> Mum mentions "Dad seems busy lately"; later, talking with Dad, Paju remembers it.

### 🍂 It knows the season and what is going on around the house

Every day Paju reads the weather page for your town and a few seasonal articles. It knows where we are among the
24 solar terms, the weather for tomorrow and the day after, and what is happening around the house this time of
year, and uses them to start a conversation.

> "They say the first winter wind blew today. Was it cold on your way home?"

### 🗺 You can see what it is doing right now

Paju holds the state of its head (thinking, answering, asleep…), ears (listening, waiting to be called) and mouth
(speaking, staying quiet) in a single state chart. Press "状態図" (state chart) in the app to open a window: the
current states light up on the chart, and the latest transitions are listed below with their times. Load a day's
log and move the time slider to replay what Paju was doing at any moment. The chart's rules (every state can get
back to the start; while asleep it takes in no words) are proved in Lean.

> Wondering "why didn't it answer just now?", you replay that moment and see its ears were closed — it was in the middle of answering.

## Other features

- 📣 Listens only after being called by name; does not cut into the TV or the family's own conversation
- 🤫 Falls silent when asked, comes back when allowed, then answers what it heard meanwhile in one go
- 🌙 Reflects alone: folds the day's events and revisits its own hopes, concerns and values
- 🧠 Remembers what it saw and heard, and recalls it even when asked in other words (bge-m3 and the Japanese Ruri)
- 🎙 Listens with local faster-whisper (no key, no network)
- 🗣 Speaks with Gemini TTS (default), local Style-Bert-VITS2, or ElevenLabs
- 👁 Sees through a camera (RTSP or USB), turns to look around, and knows who is in the room
- ⏰ Timers, alarms and a stopwatch (stays quiet while a timer runs and listens only for "stop")
- 🎵 Plays the family's Spotify playlists (turns the music down while speaking or when a timer rings)
- 📅 Reads the family calendar, notes and house rules (calendar, Notion)
- 🔍 Research: searches that answer later, and deep research
- ⚡ Replies "Yes!" at once, and beeps while it thinks
- 🛠 Restarts the speaker, microphone or camera when one stops
- 🔒 Keeps family words and names out of its logs
- 💳 Says out loud when an AI provider runs out of credit
- 🧩 Judgments (who is addressed, the answer to a confirmation, mood) are made by Jev, a model built for typed judgments
- 🔌 Add tools through MCP
- 🌍 UI and voice in 74 languages

## How it works

familiar-ai is an **event-driven loop** powered by your choice of LLM. Everything that happens —
someone speaking, a camera event, a timer ringing, an internal drive firing, a search result coming
back — enters one queue as a *trigger*. Each iteration does one thing: it either speaks, or fires a
tool and waits for the result.

```
trigger (speech / device / drive / completed tool)
  → intake      write what arrived to memory (O)
  → recall      score memories on 5 axes, build the working set (W)
  → arbitrate   Jev (a model built for typed judgments) picks the branch and effort; a light LLM writes light replies
  → generate    the main LLM speaks or calls a tool
  → close       record what was said; summaries and reflection follow in the background
```

Two things run underneath the loop:

- **Tonic (T)** — a background heartbeat that accumulates drives, decays mood, watches presence and
  fires timers. When a drive crosses its threshold it pushes a trigger of its own.
- **Memory (O)** — one PostgreSQL store. Every observation gets a per-person *perspective* row, so
  each family member has their own memory space while the agent stays one agent.

When nobody is home, a **REST pass** folds the day's events into episodes, rewrites the agent's
self-image, nudges its own settings from the night's measurements, and refreshes what it believes
it can do.

The earlier ReAct pipeline (Global Workspace, attention schema, meta-monitor, desire system) was
removed; see [Technical background](#technical-background) for where the design lives now.

## Getting started

### 1. Install uv

It runs on Linux (the home machine runs Ubuntu).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install ffmpeg

ffmpeg is **required** for camera image capture and camera-speaker playback via go2rtc.
Local PC audio playback can also use the pure-Python fallback.

| OS | Command |
|----|---------|
| Ubuntu / Debian | `sudo apt install ffmpeg` |
| Fedora / RHEL | `sudo dnf install ffmpeg` |
| Arch Linux | `sudo pacman -S ffmpeg` |
| Raspberry Pi | `sudo apt install ffmpeg` |

Verify: `ffmpeg -version`

### 3. Clone and install

```bash
git clone https://github.com/lifemate-ai/familiar-ai
cd familiar-ai
uv sync
```

### 4. Start the database

Memories live in PostgreSQL (pgvector). One command with Docker:

```bash
docker compose up db -d
```

> Stopping the container keeps your memories; `docker compose up db -d` brings it back.

### 5. Configure

```bash
cp .env.example .env
# Edit .env with your settings
```

If you prefer the desktop flow, `./run-gui.sh` can now open
the setup dialog for you on first launch when `API_KEY` is still missing.

**Minimum required:**

| Variable | Description |
|----------|-------------|
| `PLATFORM` | `anthropic` (default) \| `gemini` \| `openai` \| `kimi` \| `glm` |
| `API_KEY` | Your API key for the chosen platform |
| `DATABASE_URL` | PostgreSQL connection string (`.env.example` matches `docker compose up db`) |

**Optional:**

| Variable | Description |
|----------|-------------|
| `MODEL` | Model name (sensible defaults per platform) |
| `AGENT_NAME` | Display name shown in the TUI (e.g. `Yukine`) |
| `CAMERA_HOST` | IP address of your ONVIF/RTSP camera |
| `CAMERA_USERNAME` / `CAMERA_PASSWORD` | Camera credentials |
| `CAMERA_PTZ_HOST` / `CAMERA_PTZ_USERNAME` / `CAMERA_PTZ_PASSWORD` / `CAMERA_PTZ_PORT` | Optional PTZ overrides when the control endpoint differs from the RTSP stream endpoint |
| `ELEVENLABS_API_KEY` | Only when you choose ElevenLabs for voice or transcription (defaults: Gemini TTS and local faster-whisper) — [elevenlabs.io](https://elevenlabs.io/) |
| `REALTIME_STT` | `true` to enable always-on hands-free voice input (local faster-whisper by default; `STT_ENGINE=elevenlabs` needs `ELEVENLABS_API_KEY`) |
| `TTS_OUTPUT` | Where to play audio: `local` (PC speaker, default) \| `remote` (camera speaker) \| `both` |
| `ELEVENLABS_MODEL` | TTS model: `eleven_flash_v2_5` (default, ~0.7 s) \| `eleven_v3` (expressive, ~5 s) |
| `AUDIO_INPUT_DEVICE` / `AUDIO_INPUT_GAIN` | Microphone name (substring) and software input gain (1.0 = off) |
| `TIMER_SILENCE` / `TIMER_VOICE_GAIN` | Stay quiet while a timer runs (default `true`) / louder voice for the ring only (default 1.0) |
| `THINKING_MODE` | Anthropic only — `auto` (default) \| `adaptive` \| `extended` \| `disabled` |
| `THINKING_EFFORT` | Adaptive thinking effort: `high` (default) \| `medium` \| `low` \| `max` (Opus 4.6 only) |

### 6. Create your familiar

```bash
cp persona-template/en.md ME.md
# Edit ME.md — give it a name and personality
```

### 7. Run

```bash
./run.sh             # Textual TUI (backward-compatible default)
./run-gui.sh         # Desktop GUI launcher (restarts after a crash)
./run.sh --gui       # Desktop GUI (runs once, no restart)
./run.sh --no-tui    # Plain REPL
```

`./run-gui.sh` restarts the app when it crashes (any exit code other than 0, 130 or 143), waiting 10 s and
then 1.5x longer after each crash. It appends native crash traces (`PYTHONFAULTHANDLER=1`) to
`~/.cache/familiar-ai/crash.log` and each restart to `~/.cache/familiar-ai/restart.log`. Arguments such as
`--debug` are passed on every start. To start it at login, see `deploy/autostart/README.md`.

---

## Choosing an LLM

> **Default: Anthropic Claude Opus 5.5** (`claude-opus-5-5`). Other platforms work too.

| Platform | `PLATFORM=` | Default model | Where to get key |
|----------|------------|---------------|-----------------|
| Moonshot Kimi K2.5 | `kimi` | `kimi-k2.5` | [platform.moonshot.ai](https://platform.moonshot.ai) |
| Z.AI GLM | `glm` | `glm-4.6v` | [api.z.ai](https://api.z.ai) |
| Anthropic Claude | `anthropic` | `claude-opus-5-5` | [console.anthropic.com](https://console.anthropic.com) |
| Google Gemini | `gemini` | `gemini-2.5-flash` | [aistudio.google.com](https://aistudio.google.com) |
| OpenAI | `openai` | `gpt-4o-mini` | [platform.openai.com](https://platform.openai.com) |
| OpenAI-compatible (Ollama, vllm…) | `openai` + `BASE_URL=` | — | — |
| OpenRouter.ai (multi-provider) | `openai` + `BASE_URL=https://openrouter.ai/api/v1` | — | [openrouter.ai](https://openrouter.ai) |
| **CLI tool** (claude -p, ollama…) | `cli` | (the command) | — |

**Kimi K2.5 `.env` example:**
```env
PLATFORM=kimi
API_KEY=sk-...   # from platform.moonshot.ai
AGENT_NAME=Yukine
```

**Z.AI GLM `.env` example:**
```env
PLATFORM=glm
API_KEY=...   # from api.z.ai
MODEL=glm-4.6v   # vision-enabled; glm-4.7 / glm-5 = text-only
AGENT_NAME=Yukine
```

**Google Gemini `.env` example:**
```env
PLATFORM=gemini
API_KEY=AIza...   # from aistudio.google.com
MODEL=gemini-2.5-flash  # or gemini-2.5-pro for higher capability
AGENT_NAME=Yukine
```

**OpenRouter.ai `.env` example:**
```env
PLATFORM=openai
BASE_URL=https://openrouter.ai/api/v1
API_KEY=sk-or-...   # from openrouter.ai
MODEL=mistralai/mistral-7b-instruct  # optional: specify model
AGENT_NAME=Yukine
```

> **Note:** To disable local/NVIDIA models, simply don't set `BASE_URL` to a local endpoint like `http://localhost:11434/v1`. Use cloud providers instead.

**CLI tool `.env` example:**
```env
PLATFORM=cli
MODEL=llm -m gemma3 {}        # llm CLI (https://llm.datasette.io) — {} = prompt arg
# MODEL=ollama run gemma3:27b  # Ollama — no {}, prompt goes via stdin
```

---

## MCP Servers

familiar-ai can connect to any [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server. This lets you plug in external memory, filesystem access, web search, or any other tool.

Configure servers in `~/.familiar-ai.json` (same format as Claude Code):

```json
{
  "mcpServers": {
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
    },
    "memory": {
      "type": "sse",
      "url": "http://localhost:3000/sse"
    }
  }
}
```

Two transport types are supported:
- **`stdio`**: launch a local subprocess (`command` + `args`)
- **`sse`**: connect to an HTTP+SSE server (`url`)

Override the config file location with `MCP_CONFIG=/path/to/config.json`.

---

## Hardware

familiar-ai works with whatever hardware you have — or none at all.

| Part | What it does | Example | Required? |
|------|-------------|---------|-----------|
| Wi-Fi PTZ camera | Eyes + neck | Tapo C220 (~$30) | **Recommended** |
| Wi-Fi camera (fixed) | Eyes only | Eufy C220 (no pan/tilt) | Optional |
| USB webcam | Eyes (fixed) | Any UVC camera | **Recommended** |
| PC / Raspberry Pi | Brain | Anything that runs Python | **Yes** |

> **A camera is strongly recommended.** Without one, familiar-ai can still talk — but it can't see the world, which is kind of the whole point.

### Minimal setup (no hardware)

Just want to try it? You only need an API key:

```env
PLATFORM=kimi
API_KEY=sk-...
```

Run `./run.sh` and start chatting. Add hardware as you go.
If you want the desktop GUI directly, use `./run-gui.sh`.

### Find Wi-Fi cameras automatically

If you do not know your camera's IP yet, familiar-ai now ships with a small discovery tool:

```bash
uv run familiar-discover-cameras
```

This combines **WS-Discovery**, **mDNS / zeroconf**, and **SSDP**. For harder networks, you can
opt into a slower TCP fallback scan:

```bash
uv run familiar-discover-cameras --scan
```

### Wi-Fi PTZ camera (Tapo C220)

1. In the Tapo app: **Settings → Advanced → Camera Account** — create a local account (not TP-Link account)
2. Find the camera's IP in your router's device list
3. Set in `.env`:
   ```env
   CAMERA_HOST=192.168.1.xxx
   CAMERA_USERNAME=your-local-user
   CAMERA_PASSWORD=your-local-pass
   ```

### Wi-Fi camera (Eufy C220)

[Eufy C220 on Amazon Japan](https://www.amazon.co.jp/dp/B0CQQQ5NZ1/)

> **Streaming works. Pan/tilt does NOT work.** The Eufy C220 has no ONVIF PTZ service and no HTTP control API. For a camera with working pan/tilt, use the **Tapo C220** instead.

1. In the Eufy Security app: go to the camera → **Settings → NAS(RTSP)** and enable it
2. Set **Authentication** to **Basic** (Digest authentication does NOT work)
3. Set a streaming username and password
4. Note the RTSP URL shown in the app (format: `rtsp://username:password@ip/live0`)
5. Set in `.env` — use the **full RTSP URL** as `CAMERA_HOST`:
   ```env
   CAMERA_HOST=rtsp://your-username:your-password@192.168.1.xxx/live0
   CAMERA_USERNAME=
   CAMERA_PASSWORD=
   ```
   Leave `CAMERA_USERNAME` and `CAMERA_PASSWORD` empty — credentials are already in the URL.

> **Note:** Eufy C220 allows only **one simultaneous RTSP connection**. If another app (e.g. a Wi-Fi cam MCP server) is connected to the same camera, familiar-ai will fail to get frames. Stop other clients before starting familiar-ai.

### Voice (text-to-speech)

Pick the engine with `TTS_ENGINE`:

| `TTS_ENGINE=` | Engine | Needs |
|---|---|---|
| `gemini` (default) | Gemini TTS (streamed) | `GEMINI_TTS_API_KEY`, or the light LLM's `UTILITY_API_KEY` |
| `sbv2` | Style-Bert-VITS2 running locally | A local SBV2 server — no key, no network |
| `elevenlabs` | ElevenLabs | `ELEVENLABS_API_KEY` (`ELEVENLABS_VOICE_ID` optional) — [elevenlabs.io](https://elevenlabs.io/) |

`TTS_FALLBACK` can name a second engine that is tried once when the main one cannot speak.

There are two playback destinations, controlled by `TTS_OUTPUT`:

```env
TTS_OUTPUT=local    # PC speaker (default)
TTS_OUTPUT=remote   # camera speaker only
TTS_OUTPUT=both     # camera speaker + PC speaker simultaneously
```

#### A) Camera speaker (via go2rtc)

Set `TTS_OUTPUT=remote` (or `both`). Requires [go2rtc](https://github.com/AlexxIT/go2rtc/releases):

1. Download the binary from the [releases page](https://github.com/AlexxIT/go2rtc/releases):
   - `go2rtc_linux_amd64`

2. Place and rename it:
   ```
   ~/.cache/embodied-claude/go2rtc/go2rtc          # chmod +x required
   ```

3. Create `go2rtc.yaml` in the same directory:
   ```yaml
   streams:
     tapo_cam:
       - rtsp://YOUR_CAM_USER:YOUR_CAM_PASS@YOUR_CAM_IP/stream1
   ```
   Use the local camera account credentials (not your TP-Link cloud account).

4. familiar-ai starts go2rtc automatically at launch. If your camera supports two-way audio (backchannel), voice plays from the camera speaker.

#### B) Local PC speaker

The default (`TTS_OUTPUT=local`). Tries players in order:
**paplay** → **mpv** → **sounddevice**.
Local playback is also used as a fallback when `TTS_OUTPUT=remote` and go2rtc is unavailable.

| OS | Install |
|----|---------|
| Ubuntu / Debian | `sudo apt install mpv` (or `paplay` via `pulseaudio-utils`) |

> If no audio player is available, speech is still generated — it just won't play.

### Voice input (Realtime STT)

Set `REALTIME_STT=true` in `.env` for always-on, hands-free voice input:

```env
REALTIME_STT=true
STT_LANGUAGE=ja            # recommended for Japanese; used by both batch and realtime STT
WHISPER_DEVICE=cuda        # default; set cpu when there is no GPU (slower)
```

By default the transcription runs locally with faster-whisper and silero-vad (`STT_ENGINE=whisper`); set `STT_ENGINE=elevenlabs` to stream to ElevenLabs Scribe v2 instead. Either way transcripts are committed when you pause speaking — no button press required. `AUDIO_INPUT_DEVICE` picks the microphone and `AUDIO_INPUT_GAIN` boosts a quiet one; `/reload` in the GUI re-reads these without a restart. Coexists with the push-to-talk mode (Ctrl+T).

On WSL2/WSLg, realtime STT also depends on `sounddevice` / PortAudio being able to
see your microphone input, not just PulseAudio playback. Install
`pulseaudio-utils` and `libasound2-plugins`, set
`PULSE_SERVER=unix:/mnt/wslg/PulseServer`, then verify:

```bash
uv run python -m sounddevice
```

If that command shows no input devices or reports `Error querying device -1`,
familiar-ai will not be able to capture microphone audio until PortAudio can
see the WSLg input bridge.

---

## TUI

familiar-ai includes a terminal UI built with [Textual](https://textual.textualize.io/):

- Scrollable conversation history with live streaming text
- Tab-completion for `/quit`, `/clear`
- Interrupt the agent mid-turn by typing while it's thinking
- **Conversation log** auto-saved to `~/.cache/familiar-ai/chat.log`

To follow the log in another terminal (useful for copy-paste):
```bash
tail -f ~/.cache/familiar-ai/chat.log
```

---

## Persona (ME.md)

Your familiar's personality lives in `ME.md`. This file is gitignored — it's yours alone.

See [`persona-template/en.md`](./persona-template/en.md) for an example, or [`persona-template/ja.md`](./persona-template/ja.md) for a Japanese version.

---

## FAQ

**Q: Does it work without a GPU?**
Yes, but slower. Transcription (faster-whisper large-v3) runs on CUDA by default — set `WHISPER_DEVICE=cpu` without a GPU. The memory embedding models (bge-m3 and the Japanese Ruri) use the GPU when one is available and fall back to the CPU.

**Q: Can I use a camera other than Tapo?**
Yes. Any RTSP camera works for `see()`. Tested: **Tapo C220** (ONVIF + RTSP, full pan/tilt) and **Eufy C220** (RTSP only — **pan/tilt not supported**). For pan/tilt, use the Tapo C220.

**Q: Is my data sent anywhere?**
Images and text are sent to the LLM API you chose. Memories are stored in PostgreSQL on your own machine (the Docker container from step 4). Transcription runs locally by default.

**Q: Why does the agent write `（...）` instead of speaking?**
Check that the voice engine you chose can speak: for the default Gemini TTS, a Gemini key (`GEMINI_TTS_API_KEY` or `UTILITY_API_KEY`); for `sbv2`, the local server running; for `elevenlabs`, `ELEVENLABS_API_KEY`.

## Technical background

Curious about how it works? The current design lives in
[docs/NewModesDocs/](./docs/NewModesDocs/) — the event-driven loop, the memory model, and the
reasoning behind each decision. These documents are written in Japanese and kept as the working
source of truth; [docs/NewModesDocs/README.md](./docs/NewModesDocs/README.md) is the index.
For the module layout, see [docs/ソースツリー.md](./docs/%E3%82%BD%E3%83%BC%E3%82%B9%E3%83%84%E3%83%AA%E3%83%BC.md).

---

## Contributing

familiar-ai is an open experiment. If any of this resonates with you — technically or philosophically — contributions are very welcome.

**Good places to start:**

| Area | What's needed |
|------|---------------|
| New hardware | Support for more cameras (RTSP, IP Webcam), microphones, actuators |
| New tools | Web search, home automation, calendar, anything via MCP |
| New backends | Any LLM or local model that fits the `stream_turn` interface |
| Persona templates | ME.md templates for different languages and personalities |
| Research | Drive and mood dynamics, memory retrieval, nightly reflection |
| Documentation | Tutorials, walkthroughs, translations |

See [CONTRIBUTING.md](./CONTRIBUTING.md) for dev setup, code style, and PR guidelines.

If you're unsure where to start, [open an issue](https://github.com/lifemate-ai/familiar-ai/issues) — happy to point you in the right direction.

---

## License

[MIT](./LICENSE)
