# riskbot

A Rust territory-conquest simulator with standalone Python bindings for 2–6 players.

## Install

Use Python 3.10 or newer and a Rust toolchain.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install maturin
maturin develop
```

## Simulate a game

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

The board contains six fictional continents with seven territories each, joined by nine sea crossings.
`REGIONS` and `DIRECTED_EDGES` expose the layout as tuples.
The simulator shuffles and deals the territories round-robin from a random starting seat.
A turn begins with `2 + owned // 4` troops and two extra troops per controlled region.
Attacks compare descending d6 rolls, with ties going to the defender.
A capture moves one troop per attacking die into the captured territory.
Fortifying adjacent friendly territories or passing ends the turn.
Eliminated players are skipped. Control of the entire board wins immediately.
The default limit of 500 completed turns produces a draw if nobody has won.

## Action and observation spaces

`ACTION_COUNT` is 1,016. Owner and troop observations each contain 42 entries.
The larger board changes action IDs, observation dimensions, and seeded replays
from the earlier 18-territory version. Use `decode_action(id)` to get
`(kind, source, target, option)` and `encode_action(*fields)` for the inverse.
Placement uses the territory as its source. Attack options are dice counts 1–3.
Fortify options 0–2 move one, half of the movable troops rounded down, or all
movable troops. One troop stays behind and equivalent fortify options are masked.
Unused fields are `None`. The final IDs are `finish_attack` and `pass_fortify`.

`observe(player)` returns a fresh snapshot for that seat, including when it has
been eliminated. Owners use absolute seat IDs. Phases 0–3 are reinforcement,
attack, fortification, and episode end. A value of `-1` indicates an absent
winner or active seat in observations. The `active_player` property returns
`None` after episode end. The six-slot alive tuple marks unused seats false.
Only the active seat has a nonempty legal-action mask. Use
`sim.observation_spec()` for ordered field names and inclusive
`(length, minimum, maximum)` bounds. Scalar lengths are `None`.

`step` returns `(observation, rewards)` with one reward per configured seat.
It returns the next active seat's observation during play and the acting seat's
observation at episode end. Rewards are zero during play and on draws.
A win yields +1 for the winner and −1 for each other seat.
Illegal actions raise `ValueError` without changing the episode.
Stepping after the episode ends raises `RuntimeError`.
`reset(seed=None)` starts a fresh game and returns the starting seat's observation.
Explicit unsigned 64-bit seeds reproduce games for the same action sequence and
release. Omitted seeds use OS entropy.

## Verify

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
maturin develop
python -m unittest discover -s tests
```

## Watch a random game in your browser

Install the Rust WebAssembly target and the wasm-bindgen CLI version listed
under `wasm-bindgen` in `Cargo.lock`.

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.130 --locked
./web/serve
```

Open <http://127.0.0.1:8765>. Click **Next Turn** to play one full turn with
random legal actions. The board holds at a win or draw until you click
**New Game**. Gold outlines mark changes from the last turn. Hover over a
territory for its ID, troops, and neighbors. Matching letters mark sea crossings. Hover over an endpoint territory to reveal its route.
Territory IDs and connections match the Python simulator.

The optional native window uses the same board. Run it with
`cargo run --features visualizer --bin riskbot-visualizer`.

## Self-play PPO + PFSP Elo League (GPU / CPU)

The optional `train` feature integrates the Rust simulator directly into a
vectorized multi-player self-play PPO environment powered by [Burn](https://burn.dev/)
(`wgpu` over Vulkan on GPU, or `ndarray` on CPU).

- **Policy (`RiskPolicyNet`)**: Residual message-passing Graph Neural Network
  over the 42-territory board graph with an edge-factorized actor head over all
  1,016 actions (`Place`, `Attack`, `Fortify`, `FinishAttack`, `PassFortify`),
  strict legal-action masking, and a pooled global value head.
- **PFSP + Elo League**: Prioritized Fictitious Self-Play (`--pfsp-mode hard|variance|uniform`)
  coupled with a multi-player Bradley-Terry Elo rating system anchored at
  Random (`800`), tracking Heuristic (`1200` initial) and a diversity-preserving
  historical snapshot pool.
- **Live Telemetry**: Streams per-update metrics to `<save-dir>/metrics.jsonl`
  and atomically updates `<save-dir>/league.json` after every PPO update.

Run a GPU training session (uses Vulkan via `wgpu`, including Apple Silicon
Asahi/Honeykrisp):

```sh
WGPU_BACKEND=vulkan cargo run --release --features train --bin riskbot-train -- \
  --device gpu --players 2 --max-turns 200 --envs 32 \
  --steps-per-update 2048 --updates 100 --pfsp-mode hard --eval-interval 10
```

Pass `--device cpu` to fall back to the CPU backend or `--help` to see all
hyperparameters. Checkpoints and telemetry are saved under `runs/ppo/`
(`latest.mpk`, `best.mpk`, `metrics.jsonl`, `league.json`).

## Live Training & Elo Visualizers

Watch Elo progression, PFSP opponent sampling weights, league standings, eval
win rates, and PPO diagnostics live as training runs:

- **Web dashboard** (zero external dependencies, auto-discovers runs in `runs/`):
  ```sh
  ./web/dashboard/serve
  ```
  Then open <http://127.0.0.1:8766>.
- **Native desktop dashboard** (`egui` window):
  ```sh
  cargo run --features visualizer --bin riskbot-dashboard -- runs
  ```

Verify all targets and tests:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo clippy --features train --all-targets -- -D warnings
cargo clippy --features visualizer --all-targets -- -D warnings
cargo test --features train
cargo test --features visualizer
python -m unittest discover -s tests
```
