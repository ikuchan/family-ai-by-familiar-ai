# familiar-ai — Developer Guide

## Project overview

familiar-ai is an embodied companion agent. It combines:

- a ReAct tool loop
- PostgreSQL memory (pgvector)
- prediction / workspace / self-state layers
- explicit relationship, appraisal, social policy, and drive regulation
- capability manifest with periodic AI self-understanding refresh
- optional camera, mobility, TTS, STT, GUI, and MCP integrations

The codebase is backend-agnostic. Anthropic is supported, but it is no longer the only runtime path.

## Source tree

```text
src/familiar_agent/
├── agent.py                  # Main embodied turn loop
├── appraisal.py              # Low-dimensional affect updates
├── attention_schema.py       # Recent focus / attention state
├── backend.py                # LLM backend protocol + implementations
├── bootstrap.py              # Startup/setup/configured-state handling
├── camera_discovery.py       # Zeroconf + network camera discovery
├── capability_state.py       # Capability manifest loader + AI self-understanding storage
├── concern_engine.py         # Active unfinished concerns
├── config.py                 # Runtime config objects
├── db.py                     # PostgreSQL singleton (get_db())
├── db_migrations.py          # Migration runner (apply_migrations)
├── default_mode.py           # Idle/default-mode memory processing (DMN)
├── desires.py                # Autonomous drives + drive selection
├── diagnostics.py            # GUI diagnostics and connection tests
├── event_bus.py              # Internal event bus
├── exploration.py            # Spatial exploration state
├── gui.py                    # GTK GUI
├── heartbeat.py              # Continuation/runtime status logic
├── interoception.py          # Interoception providers + semantic pressure
├── intervention_policy.py    # Autonomous intervention gating
├── main.py                   # CLI entry point / mode selection
├── mcp_client.py             # Model Context Protocol client
├── memory_worker.py          # Async background memory job worker
├── mental_state.py           # Mental-state bus and PostgreSQL snapshots
├── meta_monitor.py           # Metacognitive logging + response gating
├── person_memory_manager.py  # Multi-person identity and memory routing
├── prediction.py             # Prediction error / agency error
├── realtime_stt_session.py   # Streaming STT session management
├── reflect.py                # Self-reflection and value adaptation
├── relationship.py           # Longitudinal relationship state
├── routines.py               # Quiet-hours / routine config helpers
├── scene.py                  # Scene entity tracking
├── self_narrative.py         # Session-spanning autobiographical narrative
├── self_state.py             # Persistent latent bodily state
├── settings_schema.py        # Settings schema and validation
├── setup.py                  # Setup flow, env migration, validation
├── social_policy.py          # Speech-act classification + response mode
├── sqlite_migrations.py      # Legacy SQLite migration runner (unused)
├── tape.py                   # Action plan (TAPE mechanism)
├── tui.py                    # Text UI
├── voice_guard.py            # TTS/STT loop prevention
├── workspace.py              # Coalition competition / broadcast
├── _i18n.py                  # Internationalisation helpers
├── _ui_helpers.py            # Shared UI utilities
├── tools/
│   ├── camera.py
│   ├── coding.py
│   ├── memory.py
│   ├── mic.py
│   ├── mobility.py
│   ├── person.py
│   ├── realtime_stt.py
│   ├── stt.py
│   ├── tom.py
│   └── tts.py
└── recognition/
    ├── face.py               # Face recognition
    ├── presence_watcher.py   # Background camera presence polling
    └── voice.py              # Voice-print speaker identification (wiring pending — issue #4)
```

## Runtime architecture

The current turn flow in `agent.py` is:

1. ingest user input and tool/scene context
2. collect interoception
3. read prediction state
4. activate memory, working memory, and open episodes
5. update provisional relationship evidence
6. appraise affect
7. choose social policy
8. regulate drives
9. run workspace competition
10. execute the ReAct loop
11. meta-gate the response
12. persist post-turn traces and mental-state snapshots

During idle DMN cycles (no workspace winner), the agent periodically refreshes
its capability self-understanding from `capabilities.yaml`.

The old prompt-only social logic is no longer the full story. Deterministic state layers now sit between raw input and response planning.

## Persistence

All state is stored in **PostgreSQL** (`DATABASE_URL` env var). No SQLite, no JSON files.

Primary tables:

| Table | Contents |
|---|---|
| `observations` | Raw memories + embeddings |
| `obs_embeddings` | Binary embedding vectors |
| `situated_embeddings` | pgvector embeddings per person |
| `episodes` / `episode_memories` | Grouped memory episodes |
| `memory_activation` | Recall salience tracking |
| `semantic_facts` | Extracted facts |
| `behavior_policies` | Extracted behavioral rules |
| `memory_revisions` | Edit history |
| `memory_events` / `memory_jobs` | Async job queue |
| `memory_links` | Associative links |
| `unfinished_business` | Open threads |
| `relationship_state` | Per-person relationship data |
| `persons` | Known person registry |
| `mental_state_log` | Append-only mental-state snapshots |
| `self_narrative_log` | First-person session diary |
| `agent_state` | Key-value store: desires, self_state, heartbeat, concerns, intervention_policy, capability_summary |

Schema changes must go through timestamped files under `migration/`.

## Databases

| Database | Purpose | Port |
|---|---|---|
| `familiar_ai` | Production | 5432 (Docker) |
| `familiar_test` | Tests | 5433 (Docker `--profile test`) |

## Development rules

- Python 3.10+
- Async-first style
- **PostgreSQL is the only storage backend — no SQLite, no JSON/JSONL files**
- Prefer deterministic logic and typed dataclasses over giant prompt blobs
- Do not leak raw interoception/body metrics into normal user-facing text
- Add a migration for every schema change

## Validation before merge

Run before opening a PR:

```bash
uv run ruff check .
uv run --group dev mypy src/familiar_agent
uv run pytest -q
```

## Git workflow

- Work from `develop`
- Cut a feature branch before changes
- Open focused PRs into `develop`
- Use Conventional Commits in English

Examples:

```text
feat(memory): add episode compression to recall
fix(agent): gate raw interoception leakage
docs: refresh technical architecture guide
```

## Editing guidance

- When adding a tool, wire all three places:
  - tool implementation
  - agent registration / routing
  - tests
- When changing state or persistence:
  - add a migration under `migration/`
  - add migration coverage in tests
  - update this file
- When adding a capability:
  - add an entry to `capabilities.yaml`
  - the agent will pick it up automatically on the next idle DMN turn
- When changing social behavior:
  - prefer appraisal / social policy / meta gate logic first
  - only extend prompt instructions when state logic is insufficient
