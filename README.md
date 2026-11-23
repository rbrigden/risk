# Riskbot

A high-performance Rust territory-conquest (Risk-like) simulator, Python RL environment, GPU/CPU Masked PPO + PFSP Elo self-play trainer, pure-Rust WebAssembly inference engine, and suite of local visualizers for 2–6 players.

## Architecture at a Glance

- **Core Simulator (`src/game.rs`, `src/board_meta.rs`)**: Deterministic 42-territory, 6-continent board with 9 sea crossings, 162 directed edges, and 1,016 discrete actions.
- **Python Bindings (`src/python.rs`)**: Standalone PyO3 (`abi3-py310`) extension module built with `maturin`.
- **Self-Play RL (`src/train/`)**: Vectorized multi-player Masked PPO powered by [Burn](https://burn.dev/) (`wgpu`/Vulkan GPU or `ndarray` CPU), a residual message-passing GNN (`RiskPolicyNet`), Prioritized Fictitious Self-Play (PFSP), and a multi-player Bradley-Terry Elo league.
- **Pure-Rust Inference & WASM Engine (`src/train/infer.rs`, `src/arena_wasm.rs`)**: Burn-parity-verified forward pass compiled to `wasm32-unknown-unknown` via `wasm-bindgen` for browser or JavaScript host runtimes.
- **Offline Exporter (`riskbot-export`)**: Exports board geometry, base64-encoded f32 policy weights, and Burn-parity test vectors as JSON.
- **Local Visualizers & Dashboards (`src/visualizer/`, `src/dashboard.rs`, `web/`)**: Local browser Play Arena (`riskbot-web`), live training & Elo web/desktop dashboards, and a native `egui` board visualizer.

---

## Install (Python Bindings)

Requires Python 3.10+ and a Rust toolchain (edition 2024).

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install maturin
maturin develop
```

### Simulate a Game in Python

```python
from random import Random
from riskbot import Simulator

sim = Simulator(num_players=3, seed=42)
rng = Random(42)
obs = sim.observe(sim.active_player)
while not (obs["terminated"] or obs["truncated"]):
    legal = [i for i, allowed in enumerate(obs["action_mask"]) if allowed]
    obs, rewards = sim.step(rng.choice(legal))

print(obs["winner"], rewards)
```

### Game Rules & Action/Observation Spaces

- **Board**: Six continents with seven territories each (`N = 42`), joined by nine sea crossings (`REGIONS` and `DIRECTED_EDGES` expose the layout as tuples).
- **Turn Structure**: Territories are shuffled and dealt round-robin from a random starting seat. Each turn begins with `2 + owned // 4` troops plus `+2` per fully controlled continent. Attacks compare descending d6 rolls with ties going to the defender; a capture moves one troop per attacking die into the captured territory. Fortifying adjacent friendly territories or passing ends the turn. Eliminated players are skipped; controlling all 42 territories wins immediately. Reaching the turn limit (default `500`) ends in a draw.
- **Action Space (`ACTION_COUNT = 1,016`)**: Use `decode_action(id)` for `(kind, source, target, option)` and `encode_action(*fields)` for the inverse. Placement uses the territory as `source`. Attack options are dice counts `1..=3`. Fortify options `0..=2` move one, half (rounded down), or all movable troops (leaving one behind; equivalent options are masked). The final two IDs are `finish_attack` (`1014`) and `pass_fortify` (`1015`).
- **Observations & Rewards**: `observe(player)` returns a snapshot for that seat (`phase` `0..=3` for reinforce, attack, fortify, episode end; `-1` for absent winner/active seat). `sim.observation_spec()` provides ordered field names and inclusive bounds. `step(action)` returns `(observation, rewards)` (`+1` winner, `-1` others, `0` during play/draws). Illegal actions raise `ValueError` without mutating state; stepping after termination raises `RuntimeError`. Explicit 64-bit seeds are deterministic.

---

## Self-Play PPO + PFSP Elo League (GPU / CPU)

The optional `train` feature runs vectorized multi-player self-play PPO directly against the Rust simulator via Burn (`wgpu` over Vulkan on GPU, or `ndarray` on CPU):

- **Policy (`RiskPolicyNet`)**: Residual message-passing GNN over the 42-territory graph (`NODE_FEAT_DIM = 22`, `GLOBAL_FEAT_DIM = 14`) with an edge-factorized actor head over all 1,016 actions, strict legal-action masking, and a pooled global value head.
- **PFSP + Elo League**: Prioritized Fictitious Self-Play (`--pfsp-mode hard|variance|uniform`) paired with Bradley-Terry Elo tracking anchored at Random (`800`), tracking Heuristic (`1200` initial) and a historical snapshot pool.
- **Live Telemetry**: Streams per-update metrics to `<save-dir>/metrics.jsonl` and atomically updates `<save-dir>/league.json`.

```sh
WGPU_BACKEND=vulkan cargo run --release --features train --bin riskbot-train -- \
  --device gpu --players 2 --max-turns 200 --envs 32 \
  --steps-per-update 2048 --updates 100 --pfsp-mode hard --eval-interval 10
```

Pass `--device cpu` to run on CPU or `--help` to view all hyperparameters. Checkpoints (`latest.mpk`, `best.mpk`) and telemetry are written under `runs/ppo/` by default (`runs/` is gitignored).

---

## Local Play Arena, Visualizers & Dashboards

All local servers bind to `127.0.0.1` by default.

- **Local Browser Play Arena** (human vs. trained `.mpk` checkpoints, heuristic, or random bots):
  ```sh
  ./web/play/serve
  # or: cargo run --features visualizer --bin riskbot-web -- --port 8767 runs
  ```
  Then open <http://127.0.0.1:8767>.

- **Live Training & Elo Web Dashboard** (zero external Python dependencies; auto-discovers runs in `runs/`):
  ```sh
  ./web/dashboard/serve
  ```
  Then open <http://127.0.0.1:8766>.

- **Native Desktop Training Dashboard** (`egui`):
  ```sh
  cargo run --features visualizer --bin riskbot-dashboard -- runs
  ```

- **Native Desktop Board Visualizer** (`egui`):
  ```sh
  cargo run --features visualizer --bin riskbot-visualizer -- runs
  ```

- **Minimal Browser WASM Demo**:
  Install `wasm32-unknown-unknown` and the `wasm-bindgen-cli` version matching `Cargo.lock`:
  ```sh
  rustup target add wasm32-unknown-unknown
  cargo install wasm-bindgen-cli --version 0.2.130 --locked
  ./web/serve
  ```
  Then open <http://127.0.0.1:8765>.

---

## Exporting Assets & Standalone WASM Engine

To export board geometry, pure-Rust-compatible policy weights, and Burn-parity test vectors for external arena clients:

```sh
cargo run --features visualizer --bin riskbot-export -- \
  --checkpoint runs/ppo/best.mpk \
  --out-dir dist/arena-data
```

To build the standalone `arena-wasm` WebAssembly package (no Burn or PyO3 dependencies):

```sh
./scripts/build-arena-wasm.sh dist/arena-wasm
```

Both `dist/` and `runs/` are gitignored.

---

## Verification

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo clippy --features train --all-targets -- -D warnings
cargo clippy --features visualizer --all-targets -- -D warnings
cargo clippy --lib --no-default-features --features arena-wasm -- -D warnings
cargo test
cargo test --features train
cargo test --features visualizer
cargo test --lib --no-default-features --features arena-wasm
cargo check --lib --no-default-features --features arena-wasm --target wasm32-unknown-unknown
maturin develop
python -m unittest discover -s tests
```

## Contributing & Security

- See [AGENTS.md](AGENTS.md) for architecture details, parity invariants, and contributor guidelines.
- See [SECURITY.md](SECURITY.md) for vulnerability reporting and security practices.
