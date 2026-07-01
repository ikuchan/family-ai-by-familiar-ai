# familiar-ai — Developer Guide

## Project overview

familiar-ai is an embodied companion agent designed for home use by a family.
Multiple family members share a single agent instance; each person has an
independent memory space, relationship state, and perspective vector.
The agent identifies who is speaking via face recognition or voice print,
and routes memory reads/writes to the appropriate person context automatically.

It combines:

- a ReAct tool loop
- PostgreSQL memory (pgvector) with per-person situated embeddings
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
├── gui.py                    # PySide6 GUI
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
│   ├── deferred_fetch.py     # Fire-and-forget URL fetch; result injected next turn
│   ├── deferred_search.py    # Fire-and-forget web search; result injected next turn
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

In addition to the above turn-driven flow, the GUI idle loop
(`_process_queue()` in `gui.py`) fires a separate **deferred delivery turn**
when `should_deliver_deferred_result()` returns True. This turn delivers
completed `search_deferred` / `fetch_deferred` results proactively, bypassing
the normal user-input trigger. It is gated by presence, quiet hours, and social
context, but quiet hours are bypassed when the search was user-initiated and the
user was active within 30 minutes.

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

## Working with Superpowers skills

The `superpowers` plugin (obra/superpowers) is installed. Its skills auto-activate.
This project uses Superpowers **selectively**. The rules below override skill
defaults where they conflict (per Superpowers' own precedence: user/CLAUDE.md
instructions win over skills).

**Skills to follow (discipline skills — use fully):**

- `test-driven-development` — RED-GREEN-REFACTOR for all new code and bug fixes.
  Write the failing test, watch it fail for the right reason, then write minimal
  code. This matches how we already work.
- `systematic-debugging` — for any bug, test failure, or unexpected behavior:
  find the root cause before proposing a fix. No guess-and-check. This matches
  our existing rule "no fix until the true cause is confirmed."
- `verification-before-completion` — run the actual verification command and read
  its output before claiming work is done.
- `writing-plans` — when turning an approved design into an implementation plan,
  break it into small tasks with exact file paths and verification steps.

**Skill defaults this project overrides:**

- **The TDD "delete code written before the test" rule applies to NEW code only.**
  This is an existing large codebase. For changes to existing modules, add tests
  for the code being changed; do not delete working production code to "start
  fresh." Delete-and-restart is for freshly written, untested code in the current
  task.
- **Do not run the autonomous multi-hour implementation flow**
  (`subagent-driven-development` working unattended across many tasks). This
  project proceeds **one item at a time with human confirmation after each
  change**. Stop and confirm before moving to the next change.
- **Design is settled in conversation first, not by the skill's brainstorming
  flow.** Design decisions are made with the human partner (cause → proposal →
  approval) and recorded in the design docs (Japanese `.md` files under the
  project). Use `brainstorming` only if explicitly asked; do not let it replace
  the established approval gate.
- **Do not auto-create git worktrees without asking.** Follow the existing Git
  workflow in this file (feature branch off `develop-ikuchan`).

**Design docs vs. code:** the authoritative design lives in the project's
Japanese design documents (Mermaid 設計図, 用語一覧, 課題 docs, etc.), which are
version-numbered and may describe not-yet-implemented decisions. **This CLAUDE.md
and the code describe only what is actually implemented.** Do not copy in-progress
design (new O/MI data model, T registers, trigger/cue, store access layer, etc.)
into code or this file until it is actually built in a given phase.

**When a bug fix touches a column/API rename or removal:** completion means
`grep` for the old name returns zero, not a hand-counted list of edited sites
(see the timestamp/embedding migration history for why counting is unreliable).

## Project-local skills (PostgreSQL, pgvector, concurrency, logging)

Four third-party skills are vendored under `.claude/skills/` in this repo. They
auto-activate on matching tasks. Scripts were intentionally excluded where
present; **do not run any script from these skills against a database** — use them
as guidance only. Environment facts to keep the skills honest: PostgreSQL 16
(`pgvector/pgvector:pg16`), `situated_embeddings` is `vector(1024)`, cosine
(`vector_cosine_ops` / `<=>`), embeddings are normalised.

**`mastering-postgresql` (pgvector guidance):**

- Use for pgvector index/query guidance (HNSW, `ef_search`, filtered-search
  recall, iterative scan) and asyncpg query patterns.
- Its Quick Start examples use pg17, `vector(1536)`, and `BIGSERIAL`. **Ignore
  those specifics** — our env is pg16, `vector(1024)`, and existing schema. Do not
  introduce `BIGSERIAL` (see the type rules below).
- The `scripts/` and `assets/` were removed on purpose. Guidance only.

**`postgresql-table-design` (PostgreSQL schema rules):**

- Use its type and safety rules for **new** tables/columns and migrations:
  `timestamptz` (never `timestamp`), `text` over `varchar(n)`,
  `generated always as identity` over `serial`, manual index on FK columns.
- Safe schema evolution matters for our large-table migrations: `CREATE INDEX
  CONCURRENTLY` (not in a transaction), volatile defaults (`now()`) rewrite the
  table, drop constraints before columns.
- **Do not rewrite existing schema** just to match these rules. Apply them to new
  work; leave working existing columns/types alone unless a task requires change.

**`python-concurrency-performance` (async / concurrency):**

- Use for asyncio vs threads boundaries, bounding fan-out (semaphores / bounded
  queues), deadline/cancellation propagation, and task/thread leak checks.
- Directly relevant to the planned **in-flight cancel** work (課題13 / Phase 5):
  treat cancellation as normal control flow — catch `asyncio.CancelledError` only
  for local cleanup, then **re-raise**; never swallow it in leaf tasks. Make
  shutdown idempotent; guarantee cleanup in `finally` / managed contexts.
- Keep a call path fully sync or async; offload blocking/GPU-heavy calls
  (InsightFace, faster-whisper, etc.) off the event loop (`asyncio.to_thread()`).
- `pyleak` is suggested as a dev/test-only diagnostic; adopt it only if we decide
  to — it is not a runtime dependency.

These are third-party (MIT) skills; keep their LICENSE/attribution files intact.

**`python-logging` (logging setup guidance):**

- Use for choosing/configuring Python logging. It confirms our approach: keep the
  `familiar_agent` package as a library (each module `logging.getLogger(__name__)`,
  never call `basicConfig()`), and configure handlers/levels once at the entry
  point (`main.py`). We stay on **stdlib logging** — do not introduce loguru.
- The skill covers setup and structure but **not** which level to use for what.
  That level policy is defined below (this project's own audit).

## Logging policy (levels and format)

Result of a one-time audit of existing logging. Setup is already centralised in
`main.py` (root handler, formatter, third-party level suppression); the only stray
setup is one line in `memory.py`. We do **not** mass-rewrite existing logs. Apply
this policy to **new and changed code**; fix existing call sites opportunistically
when you already touch that file.

**Level policy (five levels):**

- `debug` — detailed developer tracing (variable values, branch taken). Off in
  production.
- `info` — normal-path milestones (startup/shutdown, connection established, job
  completed, state transitions worth recording).
- `warning` — an anomaly the code **recovers from** and keeps going (config file
  missing so a default is used; an optional feature disabled; queue backlog).
- `error` — an operation that **could not be completed** (no fallback; the function
  did not achieve its purpose).
- In an `except` block, prefer **`logger.exception(...)`** (or
  `logger.error(..., exc_info=True)`) so the stack trace is captured. Our existing
  code has ~110 `logger.warning("...: %s", e)` sites in `except` blocks that drop
  the traceback — when you touch one, convert it: keep `warning` only if it is a
  recoverable anomaly, otherwise `exception`; add `exc_info` when the trace helps.

**Format (from the skill's stdlib reference):** the entry-point formatter should
carry timestamp, level, logger name and line number, e.g.
`"%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s"`. Use
`extra={...}` for structured context rather than string-formatting everything into
the message.

**Do not** add `print(...)` for logging in library code; use a logger. Plain
`print` is only for genuine CLI/interactive stdout.

**When to add a log (debug-useful points).** When writing new or changed logic,
add logs at the points that let someone reconstruct what happened afterwards:

- External boundaries — the entry and result of DB queries, LLM calls, MCP/tool
  calls, and device I/O.
- LLM prompt construction/editing — whenever a prompt is assembled or edited, log
  it. The full prompt is heavy and may carry conversational/memory content, so log
  the **full prompt at `debug` only** (off in production, available when
  debugging). A lightweight summary (which template/path, how many MIs went into
  W, prompt length, target model, and in/out token counts if available) may be
  logged as a normal state-transition line. **Never dump the full prompt or
  conversational/memory content at `info` or above.**
- State transitions — once those mechanisms exist: firing, W construction, open
  resolution, and similar shifts worth reconstructing later.
- Recovered anomalies — when a fallback path is taken (log at `warning`, per the
  level policy).
- Async boundaries — enqueue, completion, and cancellation (pairs with the
  concurrency skill's leak/cancellation guidance).

**Restraint (do not drown the log).** Hot paths that run every iteration or every
tick use `debug`, not `info`. One log line, one purpose. Dump values selectively
(key fields, counts, lengths), not whole objects. The goal is that the signal
stays findable — more logs is not the goal, reconstructable logs is.

## Validation before merge

Run before opening a PR:

```bash
uv run ruff check .
uv run --group dev mypy src/familiar_agent
./scripts/run_tests.sh
```

`./scripts/run_tests.sh` starts the test DB container (port 5433), runs the
full pytest suite against it, and stops the container on exit. Do not use
`uv run pytest -q` directly — many tests require a live PostgreSQL connection.

## Git workflow

- Work from `develop-ikuchan`
- Cut a feature branch before changes
- Open focused PRs into `develop-ikuchan`
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
- When adding a deferred tool (search or fetch variant):
  - implement `set_user_turn(bool)` and `has_user_initiated_pending` on the tool
  - add the tool name to `_BRIEF_REPLY_TOOL_NAMES` in `agent.py` so it is
    available during brief-reply mode (short greetings, acks, etc.)
  - if the tool produces user-facing output, add its delivery desire name to
    `_SOCIAL_DESIRE_NAMES` in `desires.py`
- Quiet hours and autonomous desires:
  - social desires are suppressed during quiet hours unless the delivery is
    for a user-initiated deferred search (`share_search_result`) AND the user
    was active within 30 minutes (`_last_human_at`)
  - this bypass is implemented in both `should_deliver_deferred_result()` and
    the quiet-hours gate inside `agent.run()`; both must be kept in sync
- Desire turn user message:
  - desire turns use `"."` as the user message placeholder,
    not a human-readable marker. Any visible string leaks into LLM output.
    A plain space is rejected by the Anthropic API (whitespace-only not allowed).
