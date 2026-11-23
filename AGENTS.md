# AGENTS.md

Engineering handbook for AI agents and contributors working on **Riskbot** (`rbrigden/risk`).

---

## 1. Project Overview

Riskbot is a high-performance Rust territory-conquest (Risk-like) game engine and reinforcement-learning research stack for 2–6 players. It provides:

1. **Deterministic Rust Game Engine (`src/game.rs`)**: Fast 42-territory, 6-continent board simulator with 1,016 discrete actions and deterministic ChaCha8 seeding.
2. **Python Bindings (`src/python.rs`)**: Standalone PyO3 (`abi3-py310`) extension module built with `maturin`.
3. **GPU/CPU Masked PPO + PFSP Elo League (`src/train/`)**: Vectorized multi-player self-play training in Rust using [Burn](https://burn.dev/) (`wgpu`/Vulkan GPU or `ndarray` CPU), a residual message-passing Graph Neural Network (`RiskPolicyNet`), Prioritized Fictitious Self-Play (PFSP), Bradley-Terry Elo tracking, and live JSONL/JSON telemetry.
4. **Pure-Rust Inference & Standalone WASM Engine (`src/train/infer.rs`, `src/arena_wasm.rs`)**: Zero-Burn, Burn-parity-verified forward pass of `RiskPolicyNet` plus a stateless serializable game/AI engine compiled to `wasm32-unknown-unknown` via `wasm-bindgen`.
5. **Asset Exporter (`src/bin/riskbot-export.rs`)**: Exports board geometry (`board.json`), base64-encoded f32 policy weights (`policies/<id>.json`), and Burn-vs-pure-Rust numerical parity vectors (`parity-vectors.json`) for external arena clients.
6. **Local Visualizers & Dashboards (`src/visualizer/`, `src/dashboard.rs`, `web/`)**:
   - Local browser Play Arena (`riskbot-web` / `./web/play/serve`) for playing against trained `.mpk` checkpoints, heuristic, or random bots.
   - Live training & Elo web dashboard (`./web/dashboard/serve`) and native `egui` desktop dashboard (`riskbot-dashboard`).
   - Native desktop board visualizer (`riskbot-visualizer`) and lightweight browser WASM demo (`./web/serve`).

---

## 2. Repository Map

```text
.
├── Cargo.toml                     # Rust package, feature flags, and 5 binaries
├── Cargo.lock                     # Locked Rust dependencies
├── pyproject.toml                 # Maturin/PyO3 Python packaging config
├── README.md                      # Public user & developer documentation
├── AGENTS.md                      # This file (guide for future agents)
├── SECURITY.md                    # Security policy & public-repo boundaries
├── scripts/
│   └── build-arena-wasm.sh        # Builds arena-wasm target (outputs to $1 or dist/arena-wasm)
├── src/
│   ├── lib.rs                     # Feature-gated module wiring
│   ├── game.rs                    # Canonical rules, topology, action codec, RNG, combat
│   ├── board_meta.rs              # Shared UI-agnostic territory/continent names & seat colors
│   ├── python.rs                  # PyO3 Simulator bindings (non-wasm32)
│   ├── arena_wasm.rs              # Stateless JSON + WASM exports (feature: arena-wasm)
│   ├── dashboard.rs               # Native egui live training dashboard (feature: visualizer)
│   ├── visualizer.rs              # Native egui + web-canvas board app (feature: visualizer)
│   ├── visualizer/
│   │   ├── map.rs                 # Vector continent/territory geometry & board JSON export
│   │   ├── coastlines.rs          # Procedural coastline polygons
│   │   ├── battles.rs             # Animated battle & dice visualization state
│   │   ├── policy.rs              # Checkpoint discovery, Burn loading, seat policies
│   │   ├── state.rs               # Shared visualizer state helpers
│   │   └── server.rs              # Local HTTP/1.1 + JSON server for web/play (127.0.0.1)
│   ├── train/
│   │   ├── mod.rs                 # Training module exports & integration tests
│   │   ├── features.rs            # Ego-centric feature encoder (22 node, 14 global) & shaping
│   │   ├── bots.rs                # Random & tactical heuristic baseline bots
│   │   ├── env.rs                 # Vectorized self-play env pool, GAE, action sampling
│   │   ├── model.rs               # Burn RiskPolicyNet GNN + weight export
│   │   ├── infer.rs               # Pure-Rust GNN forward pass & JSON weight decoder
│   │   ├── league.rs              # PFSP matchmaking & multi-player Bradley-Terry Elo league
│   │   ├── ppo.rs                 # Masked PPO-Clip training loop & checkpointing
│   │   └── telemetry.rs           # Atomic league.json & append-only metrics.jsonl writers
│   └── bin/
│       ├── riskbot-train.rs       # CLI for GPU/CPU PPO + PFSP training
│       ├── riskbot-visualizer.rs  # Native desktop visualizer CLI
│       ├── riskbot-dashboard.rs   # Native desktop training dashboard CLI
│       ├── riskbot-web.rs         # Local HTTP Play Arena server CLI
│       └── riskbot-export.rs      # Offline board/weights/parity JSON exporter CLI
├── tests/
│   ├── test_simulator.py          # Python binding unit & determinism tests
│   └── test_dashboard_server.py   # Dashboard HTTP API & path-traversal tests
└── web/
    ├── index.html & serve         # Minimal WASM random-play visualizer demo
    ├── play/                      # Local Play Arena UI served by riskbot-web
    └── dashboard/                 # Zero-dependency Python HTTP server + live Elo UI
```

**Important boundary:** Production hosted web applications (e.g., `/web/arena/`), trained checkpoint artifacts (`/runs/`, `*.mpk`), and generated build artifacts (`/dist/`, `/target/`) are **intentionally excluded and gitignored** from this public repository. Any hosted deployment lives in a separate private repository and consumes artifacts generated by `riskbot-export` and `scripts/build-arena-wasm.sh`.

---

## 3. Core Domain Invariants & Parity Rules

Before editing simulator, feature, or model code, keep these invariants intact:

1. **Board Geometry & Action Space (`src/game.rs`)**:
   - `N = 42` territories (6 continents × 7 territories: 6 outer + 1 inner hub), `BRIDGES.len() = 9` sea crossings, `EDGES.len() = 162` directed edges.
   - `ACTION_COUNT = 1016`:
     - `0..42`: `Place(territory)`
     - `42..528`: `Attack(src, dst, dice ∈ 1..=3)` (3 options per directed edge)
     - `528..1014`: `Fortify(src, dst, option ∈ 0..=2)` (1 troop, half movable, all movable; equivalent options masked)
     - `1014`: `FinishAttack`
     - `1015`: `PassFortify`
   - Action encoding/decoding must remain a strict bijection over `0..ACTION_COUNT`.
2. **Observation & Feature Encoding (`src/train/features.rs`)**:
   - Ego-centric encoding from `player`'s perspective: `NODE_FEAT_DIM = 22`, `GLOBAL_FEAT_DIM = 14`, plus boolean `mask[ACTION_COUNT]`.
   - Every observation value must be finite (`!NaN`, `!Inf`), and `obs.mask[id]` must match `game.legal(Action::decode(id))` exactly.
3. **Burn vs. Pure-Rust Inference Parity (`src/train/model.rs` ↔ `src/train/infer.rs`)**:
   - `src/train/infer.rs` implements the exact forward pass of `RiskPolicyNet` without Burn so it compiles to lightweight WASM.
   - Linear layers use Burn 0.20's `[d_in, d_out]` layout (`y = x @ W + b`); LayerNorm uses biased variance with matching `epsilon`.
   - Any architecture or feature change **must** update both `model.rs` and `infer.rs` and maintain `< 1e-5` max probability/value difference in `src/train/infer.rs` tests.
4. **Downstream Sync Workflow**:
   - Whenever rules (`src/game.rs`), board metadata (`src/board_meta.rs`, `src/visualizer/map.rs`), features (`src/train/features.rs`), or inference (`src/train/infer.rs`, `src/arena_wasm.rs`) change, downstream consumers should re-run:
     ```sh
     cargo run --features visualizer --bin riskbot-export -- \
       --checkpoint <path/to/checkpoint.mpk> \
       --out-dir <external-data-dir>
     ./scripts/build-arena-wasm.sh <external-wasm-dir>
     ```

---

## 4. Feature Flags & Binaries

| Feature | Enables | Notes |
| --- | --- | --- |
| *(default)* | `game`, `board_meta`, `python` (non-wasm32) | Minimal simulator + PyO3 module |
| `train` | `burn` (`wgpu`, `ndarray`, `autodiff`, `vulkan`), `serde` | Enables `riskbot-train` and full `src/train/*` |
| `visualizer` | `eframe`, `earcutr`, `burn`, `serde`, wasm deps | Enables `riskbot-visualizer`, `riskbot-dashboard`, `riskbot-web`, `riskbot-export` |
| `arena-wasm` | `serde`, `serde_json`, `wasm-bindgen` (no Burn, no PyO3) | Build with `--no-default-features --features arena-wasm` |

---

## 5. Verification Commands

Run the full verification matrix before completing substantive changes:

```sh
# Formatting
cargo fmt --check

# Clippy across all feature sets (warnings are errors)
cargo clippy --all-targets -- -D warnings
cargo clippy --features train --all-targets -- -D warnings
cargo clippy --features visualizer --all-targets -- -D warnings
cargo clippy --lib --no-default-features --features arena-wasm -- -D warnings

# Unit & integration tests across all feature sets
cargo test
cargo test --features train
cargo test --features visualizer
cargo test --lib --no-default-features --features arena-wasm

# WASM compilation check (requires: rustup target add wasm32-unknown-unknown)
cargo check --lib --no-default-features --features arena-wasm --target wasm32-unknown-unknown

# Python bindings & dashboard HTTP server tests
maturin develop
python -m unittest discover -s tests
```

---

## 6. Security & Secret Hygiene Rules (Mandatory)

This is a **public** repository. Future agents must strictly enforce the following rules:

1. **No Secrets or Credentials**: Never commit `.env` files, API keys, OAuth client IDs, cookie secrets, database connection strings, blob tokens, SSH/TLS keys, or real user PII anywhere in code, tests, docs, or git history.
2. **Keep Production Infrastructure Opaque**: Do not commit production website source code (`/web/arena/`), live hosting URLs, cloud project/account identifiers, Vercel/Neon/WorkOS deployment configs, or internal production topology details into this repository. Keep scripts and exporters path-agnostic (accepting CLI flags or env vars with generic defaults like `dist/`).
3. **No Large Training Artifacts in Git**: Keep `/runs/`, `*.mpk`, `/dist/`, and compiled `.wasm` outputs out of version control.
4. **Local Server Hardening**:
   - Local HTTP servers (`src/visualizer/server.rs`, `web/dashboard/serve`, `web/serve`) must bind to `127.0.0.1` by default.
   - Never add wildcard `Access-Control-Allow-Origin: *` headers to local API servers that mutate state or read local files.
   - Strictly validate all user-supplied paths against directory traversal (`..`, encoded separators, symlinks escaping root) and restrict checkpoint loading to `.mpk` files.
   - Include `X-Content-Type-Options: nosniff`, `Cache-Control: no-store` (for dynamic APIs), and `Referrer-Policy: no-referrer`.

---

## 7. Coding & Git Conventions

- **Rust Edition 2024**: Format with `cargo fmt`; resolve all `cargo clippy -- -D warnings` diagnostics rather than suppressing them unless clearly justified.
- **Deterministic Tests**: Always seed RNGs explicitly in tests (`Game::new(..., Some(seed))`, `ChaCha8Rng::seed_from_u64(...)`) so test runs are reproducible and non-flaky.
- **Commit Hygiene**: Keep commits focused and atomic. Verify `git status` and `git diff --cached` before committing to ensure no untracked local artifacts, checkpoints, or private directories are staged.
