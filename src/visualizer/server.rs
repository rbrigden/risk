//! Zero-dependency local HTTP/1.1 + JSON REST backend for the browser Riskbot visualizer.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::game::{Action, EDGES, Game, MAX_PLAYERS, N, Phase};

use super::map::Map;
use super::policy::{
    DiscoveredCheckpoint, PolicyRegistry, PolicySampling, SeatKind, discover_checkpoints,
};
use super::state::{BattleRecord, PLAYER_COLORS, continent_owner, territory_name};

const DEFAULT_MAX_TURNS: u32 = 500;
const MAX_BATTLE_HISTORY: usize = 80;
const MAX_BLITZ_STEPS: usize = 64;
const MAX_AI_STEPS_GUARD: usize = 2000;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 2 * 1024 * 1024;

const COLOR_LOSS_RED: &str = "#f06660";
const COLOR_CAPTURED_GOLD: &str = "#f5c542";
const COLOR_FORTIFY_OUT: &str = "#90a6b9";
const COLOR_FORTIFY_IN: &str = "#8ad672";

/// Start the blocking HTTP/1.1 web visualizer server on `127.0.0.1:<port>`.
pub fn serve(
    port: u16,
    checkpoint_roots: Vec<PathBuf>,
    default_checkpoint: Option<PathBuf>,
) -> io::Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let actual_port = listener.local_addr()?.port();
    let state = Arc::new(Mutex::new(ServerState::new(
        checkpoint_roots,
        default_checkpoint,
    )));

    {
        let guard = state.lock().expect("server state mutex");
        println!("Riskbot Web Visualizer listening on http://127.0.0.1:{actual_port}");
        println!("Default match: {}", guard.startup_summary());
    }

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let state = Arc::clone(&state);
                thread::spawn(move || {
                    let _ = handle_connection(stream, &state);
                });
            }
            Err(err) => {
                eprintln!("accept error: {err}");
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
struct PopupDto {
    territory: usize,
    text: String,
    color: String,
}

#[derive(Clone, Debug, Serialize)]
struct BattleDto {
    id: u64,
    turn: u32,
    attacker: usize,
    defender: usize,
    source: usize,
    target: usize,
    attack_dice_count: usize,
    defense_dice_count: usize,
    attack_rolls: Vec<u8>,
    defense_rolls: Vec<u8>,
    attacker_losses: u64,
    defender_losses: u64,
    captured: bool,
    attacker_troops_after: u64,
    defender_troops_after: u64,
}

impl From<&BattleRecord> for BattleDto {
    fn from(b: &BattleRecord) -> Self {
        Self {
            id: b.id,
            turn: b.turn,
            attacker: b.attacker,
            defender: b.defender,
            source: b.source,
            target: b.target,
            attack_dice_count: b.attack_dice_count,
            defense_dice_count: b.defense_dice_count,
            attack_rolls: b.attack_rolls[..b.attack_dice_count.min(3)].to_vec(),
            defense_rolls: b.defense_rolls[..b.defense_dice_count.min(2)].to_vec(),
            attacker_losses: b.attacker_losses,
            defender_losses: b.defender_losses,
            captured: b.captured,
            attacker_troops_after: b.attacker_troops_after,
            defender_troops_after: b.defender_troops_after,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct CheckpointDto {
    path: String,
    display_name: String,
    run_name: String,
    file_stem: String,
    elo: Option<f64>,
    update: Option<usize>,
    trained_players: Option<usize>,
}

impl From<&DiscoveredCheckpoint> for CheckpointDto {
    fn from(c: &DiscoveredCheckpoint) -> Self {
        Self {
            path: c.path.to_string_lossy().into_owned(),
            display_name: c.display_name.clone(),
            run_name: c.run_name.clone(),
            file_stem: c.file_stem.clone(),
            elo: c.elo,
            update: c.update,
            trained_players: c.trained_players,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct SeatInput {
    kind: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    sampling: Option<String>,
}

impl SeatInput {
    fn into_seat_kind(self) -> Result<SeatKind, String> {
        match self.kind.trim().to_ascii_lowercase().as_str() {
            "human" => Ok(SeatKind::Human),
            "heuristic" => Ok(SeatKind::Heuristic),
            "random" => Ok(SeatKind::Random),
            "checkpoint" => {
                let raw_path = self.path.unwrap_or_default().trim().to_string();
                if raw_path.is_empty() {
                    return Err("Checkpoint seat requires a non-empty 'path'.".into());
                }
                let path = PathBuf::from(&raw_path);
                let label = self
                    .label
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| {
                        discover_checkpoints(std::slice::from_ref(&path))
                            .into_iter()
                            .next()
                            .map(|c| c.display_name)
                            .unwrap_or_else(|| {
                                path.file_stem()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("Policy")
                                    .to_string()
                            })
                    });
                let sampling = match self
                    .sampling
                    .as_deref()
                    .unwrap_or("balanced")
                    .trim()
                    .to_ascii_lowercase()
                    .as_str()
                {
                    "greedy" => PolicySampling::Greedy,
                    "balanced" | "" => PolicySampling::Balanced,
                    "stochastic" => PolicySampling::Stochastic,
                    other => {
                        return Err(format!(
                            "Invalid sampling '{other}'; expected 'greedy', 'balanced', or 'stochastic'."
                        ));
                    }
                };
                Ok(SeatKind::Checkpoint {
                    path,
                    label,
                    sampling,
                })
            }
            other => Err(format!(
                "Invalid seat kind '{other}'; expected 'human', 'checkpoint', 'heuristic', or 'random'."
            )),
        }
    }
}

fn seat_to_json(seat: &SeatKind) -> Value {
    match seat {
        SeatKind::Human => json!({ "kind": "human" }),
        SeatKind::Heuristic => json!({ "kind": "heuristic" }),
        SeatKind::Random => json!({ "kind": "random" }),
        SeatKind::Checkpoint {
            path,
            label,
            sampling,
        } => json!({
            "kind": "checkpoint",
            "path": path.to_string_lossy(),
            "label": label,
            "sampling": match sampling {
                PolicySampling::Greedy => "greedy",
                PolicySampling::Balanced => "balanced",
                PolicySampling::Stochastic => "stochastic",
            },
        }),
    }
}

#[derive(Debug, Deserialize)]
struct LoadCheckpointRequest {
    path: String,
}

#[derive(Debug, Deserialize)]
struct NewGameRequest {
    seats: Vec<SeatInput>,
    #[serde(default)]
    max_turns: Option<u32>,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ActionRequest {
    Place {
        territory: usize,
        #[serde(default)]
        count: Option<u64>,
    },
    Attack {
        source: usize,
        target: usize,
        #[serde(default)]
        dice: Option<usize>,
    },
    Blitz {
        source: usize,
        target: usize,
        #[serde(default)]
        dice_pref: Option<Value>,
    },
    FinishAttack,
    Fortify {
        source: usize,
        target: usize,
        #[serde(default)]
        option: usize,
    },
    PassFortify,
}

#[derive(Debug, Deserialize, Default)]
struct AiStepRequest {
    #[serde(default)]
    mode: Option<String>,
}

struct ServerState {
    board_json: String,
    checkpoint_roots: Vec<PathBuf>,
    custom_checkpoints: Vec<DiscoveredCheckpoint>,
    registry: PolicyRegistry,
    policy_rng: ChaCha8Rng,

    game: Game,
    seats: Vec<SeatKind>,
    seat_labels: Vec<String>,
    seat_values: Vec<Option<f32>>,
    game_number: u64,
    changed: [bool; N],
    last_turn_owners: [usize; N],
    last_turn_troops: [u64; N],
    last_turn_number: u32,
    battle_history: Vec<BattleRecord>,
    next_battle_id: u64,
    last_action_summary: String,
}

impl ServerState {
    fn new(checkpoint_roots: Vec<PathBuf>, default_checkpoint: Option<PathBuf>) -> Self {
        let roots = if checkpoint_roots.is_empty() {
            vec![PathBuf::from("runs")]
        } else {
            checkpoint_roots
        };
        let board_json = serde_json::to_string(&Map::new().export_board_json())
            .expect("board json serialization");
        let mut registry = PolicyRegistry::new();
        let mut custom_checkpoints = Vec::new();

        let discovered = discover_checkpoints(&roots);
        let seat1 = Self::select_default_opponent(
            &mut registry,
            &mut custom_checkpoints,
            &discovered,
            default_checkpoint,
        );

        let seats = vec![SeatKind::Human, seat1];
        let seat_labels = build_seat_labels(&seats);
        let seat_values = vec![None; seats.len()];
        let game = Game::new(seats.len(), DEFAULT_MAX_TURNS, None);
        let (last_turn_owners, last_turn_troops) = (game.owner, game.troops);
        let last_turn_number = game.turns;
        let last_action_summary = format!("Match ready · {} opens", seat_labels[game.active]);

        Self {
            board_json,
            checkpoint_roots: roots,
            custom_checkpoints,
            registry,
            policy_rng: ChaCha8Rng::from_os_rng(),
            game,
            seats,
            seat_labels,
            seat_values,
            game_number: 1,
            changed: [false; N],
            last_turn_owners,
            last_turn_troops,
            last_turn_number,
            battle_history: Vec::new(),
            next_battle_id: 1,
            last_action_summary,
        }
    }

    fn select_default_opponent(
        registry: &mut PolicyRegistry,
        custom_checkpoints: &mut Vec<DiscoveredCheckpoint>,
        discovered: &[DiscoveredCheckpoint],
        default_checkpoint: Option<PathBuf>,
    ) -> SeatKind {
        if let Some(path) = default_checkpoint {
            match registry.preload(&path) {
                Ok(_) => {
                    let entry = checkpoint_entry_for_path(&path);
                    let label = entry.display_name.clone();
                    upsert_custom_checkpoint(custom_checkpoints, entry);
                    return SeatKind::Checkpoint {
                        path,
                        label,
                        sampling: PolicySampling::Balanced,
                    };
                }
                Err(err) => {
                    eprintln!(
                        "Warning: failed to load --checkpoint '{}': {err}; falling back to auto-discovery",
                        path.display()
                    );
                }
            }
        }

        let mut candidates: Vec<&DiscoveredCheckpoint> = discovered.iter().collect();
        candidates.sort_by(|a, b| compare_default_candidate(a, b));

        for ckpt in candidates {
            if registry.preload(&ckpt.path).is_ok() {
                return SeatKind::Checkpoint {
                    path: ckpt.path.clone(),
                    label: ckpt.display_name.clone(),
                    sampling: PolicySampling::Balanced,
                };
            }
        }

        SeatKind::Heuristic
    }

    fn startup_summary(&self) -> String {
        match self.seats.get(1) {
            Some(SeatKind::Checkpoint { path, label, .. }) => {
                let arch = self
                    .registry
                    .model_info(path)
                    .map(|(h, l)| format!(" [{h}×{l}]"))
                    .unwrap_or_default();
                format!("You vs {label}{arch} ({})", path.display())
            }
            Some(other) => format!("You vs {}", other.short_label()),
            None => "2-player match".to_string(),
        }
    }

    fn is_human_turn(&self) -> bool {
        self.game.phase != Phase::Ended
            && self
                .seats
                .get(self.game.active)
                .is_some_and(SeatKind::is_human)
    }

    fn has_human(&self) -> bool {
        self.seats.iter().any(SeatKind::is_human)
    }

    fn list_checkpoints(&self) -> Vec<CheckpointDto> {
        let mut combined = discover_checkpoints(&self.checkpoint_roots);
        let mut seen: HashSet<PathBuf> = combined.iter().map(|c| canonical_key(&c.path)).collect();

        for custom in &self.custom_checkpoints {
            let key = canonical_key(&custom.path);
            if seen.insert(key) {
                let refreshed = discover_checkpoints(std::slice::from_ref(&custom.path))
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| custom.clone());
                combined.push(refreshed);
            }
        }

        combined.iter().map(CheckpointDto::from).collect()
    }

    fn load_checkpoint(&mut self, raw_path: &str) -> Result<Value, String> {
        let trimmed = raw_path.trim();
        if trimmed.is_empty() {
            return Err("Checkpoint path cannot be empty.".into());
        }
        let path = PathBuf::from(trimmed);
        if path.extension().and_then(|ext| ext.to_str()) != Some("mpk") {
            return Err("Checkpoint path must point to a .mpk file.".into());
        }
        let (hidden_dim, num_layers) = self.registry.preload(&path)?;
        let entry = checkpoint_entry_for_path(&path);
        upsert_custom_checkpoint(&mut self.custom_checkpoints, entry.clone());
        Ok(json!({
            "ok": true,
            "checkpoint": CheckpointDto::from(&entry),
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        }))
    }

    fn new_game(&mut self, req: NewGameRequest) -> Result<Value, String> {
        if !(2..=MAX_PLAYERS).contains(&req.seats.len()) {
            return Err(format!(
                "Seat count must be between 2 and {MAX_PLAYERS}, got {}.",
                req.seats.len()
            ));
        }
        let max_turns = req.max_turns.unwrap_or(DEFAULT_MAX_TURNS);
        if max_turns == 0 {
            return Err("max_turns must be at least 1.".into());
        }

        let mut parsed_seats = Vec::with_capacity(req.seats.len());
        for seat_in in req.seats {
            parsed_seats.push(seat_in.into_seat_kind()?);
        }

        // Preload all checkpoint seats before committing the new match.
        for seat in &parsed_seats {
            if let SeatKind::Checkpoint { path, .. } = seat {
                self.registry.preload(path)?;
                let entry = checkpoint_entry_for_path(path);
                upsert_custom_checkpoint(&mut self.custom_checkpoints, entry);
            }
        }

        self.seats = parsed_seats;
        self.seat_labels = build_seat_labels(&self.seats);
        self.seat_values = vec![None; self.seats.len()];
        self.game = Game::new(self.seats.len(), max_turns, req.seed);
        self.policy_rng = match req.seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s ^ 0x51A7_C0DE_9E37_79B9),
            None => ChaCha8Rng::from_os_rng(),
        };
        self.changed = [false; N];
        self.last_turn_owners = self.game.owner;
        self.last_turn_troops = self.game.troops;
        self.last_turn_number = self.game.turns;
        self.battle_history.clear();
        self.game_number += 1;
        self.last_action_summary = format!(
            "New {}-player match started · {} opens",
            self.seats.len(),
            self.seat_labels[self.game.active]
        );

        Ok(json!({
            "ok": true,
            "state": self.state_json(),
        }))
    }

    fn handle_human_action(&mut self, req: ActionRequest) -> Result<Value, String> {
        if self.game.phase == Phase::Ended {
            return Err("Game has already ended.".into());
        }
        if !self.is_human_turn() {
            return Err("It is not currently a human seat's turn.".into());
        }

        let mut new_battles = Vec::new();
        let mut popups = Vec::new();
        let me = self.game.active;
        let actor_name = self.seat_labels[me].clone();

        match req {
            ActionRequest::Place { territory, count } => {
                if self.game.phase != Phase::Reinforce {
                    return Err("Cannot place troops outside the reinforce phase.".into());
                }
                if territory >= N {
                    return Err(format!("Invalid territory index {territory}."));
                }
                if self.game.owner[territory] != me {
                    return Err(format!(
                        "Territory {} (#{territory}) is not owned by the active player.",
                        territory_name(territory)
                    ));
                }
                let placed = count.unwrap_or(1).clamp(1, self.game.pending.max(1));
                let mut actual = 0u64;
                for _ in 0..placed {
                    if self.game.phase == Phase::Reinforce && self.game.owner[territory] == me {
                        self.apply_action(
                            Action::Place(territory),
                            false,
                            &mut new_battles,
                            &mut popups,
                        );
                        actual += 1;
                    }
                }
                if actual > 0 {
                    popups.push(PopupDto {
                        territory,
                        text: format!("+{actual}"),
                        color: player_color_hex(me),
                    });
                    self.last_action_summary = if actual == 1 {
                        format!(
                            "{actor_name} reinforced {} (#{territory}) · {} left",
                            territory_name(territory),
                            self.game.pending
                        )
                    } else {
                        format!(
                            "{actor_name} placed +{actual} on {} (#{territory}) · {} left",
                            territory_name(territory),
                            self.game.pending
                        )
                    };
                }
            }
            ActionRequest::Attack {
                source,
                target,
                dice,
            } => {
                self.validate_attack_edge(source, target)?;
                let max_dice = (self.game.troops[source] - 1).min(3) as usize;
                let clamped_dice = dice.unwrap_or(max_dice).clamp(1, max_dice);
                self.apply_action(
                    Action::Attack(source, target, clamped_dice),
                    true,
                    &mut new_battles,
                    &mut popups,
                );
            }
            ActionRequest::Blitz {
                source,
                target,
                dice_pref,
            } => {
                self.validate_attack_edge(source, target)?;
                let fixed_dice = parse_dice_pref(dice_pref.as_ref())?;
                let mut total_atk_losses = 0u64;
                let mut total_def_losses = 0u64;
                let mut captured = false;

                for _ in 0..MAX_BLITZ_STEPS {
                    if self.game.phase != Phase::Attack
                        || self.game.owner[source] != me
                        || self.game.owner[target] == me
                        || self.game.troops[source] <= 1
                    {
                        break;
                    }
                    let max_dice = (self.game.troops[source] - 1).min(3) as usize;
                    let dice = match fixed_dice {
                        Some(d) => d.clamp(1, max_dice),
                        None => max_dice,
                    };
                    let before_len = new_battles.len();
                    self.apply_action(
                        Action::Attack(source, target, dice),
                        false,
                        &mut new_battles,
                        &mut popups,
                    );
                    if let Some(battle) = new_battles.get(before_len) {
                        total_atk_losses += battle.attacker_losses;
                        total_def_losses += battle.defender_losses;
                        if battle.captured {
                            captured = true;
                            break;
                        }
                    }
                }

                if total_atk_losses > 0 {
                    popups.push(PopupDto {
                        territory: source,
                        text: format!("-{total_atk_losses}"),
                        color: COLOR_LOSS_RED.to_string(),
                    });
                }
                if captured {
                    popups.push(PopupDto {
                        territory: target,
                        text: "CAPTURED!".to_string(),
                        color: COLOR_CAPTURED_GOLD.to_string(),
                    });
                } else if total_def_losses > 0 {
                    popups.push(PopupDto {
                        territory: target,
                        text: format!("-{total_def_losses}"),
                        color: COLOR_LOSS_RED.to_string(),
                    });
                }
            }
            ActionRequest::FinishAttack => {
                if self.game.phase != Phase::Attack {
                    return Err("Cannot finish attack outside the attack phase.".into());
                }
                self.apply_action(Action::FinishAttack, true, &mut new_battles, &mut popups);
            }
            ActionRequest::Fortify {
                source,
                target,
                option,
            } => {
                if self.game.phase != Phase::Fortify {
                    return Err("Cannot fortify outside the fortify phase.".into());
                }
                if source >= N || target >= N {
                    return Err("Invalid source or target territory index.".into());
                }
                if self.game.owner[source] != me || self.game.owner[target] != me {
                    return Err("Both fortify territories must belong to the active player.".into());
                }
                if EDGES.binary_search(&(source, target)).is_err() {
                    return Err(format!(
                        "{} and {} are not adjacent.",
                        territory_name(source),
                        territory_name(target)
                    ));
                }
                let fallback_order = match option {
                    0 => [0, 1, 2],
                    1 => [1, 2, 0],
                    _ => [2, 1, 0],
                };
                let legal_opt = fallback_order
                    .into_iter()
                    .find(|&opt| self.game.fortify_amount(source, opt).is_some())
                    .ok_or_else(|| {
                        format!(
                            "{} has no movable troops to fortify.",
                            territory_name(source)
                        )
                    })?;
                self.apply_action(
                    Action::Fortify(source, target, legal_opt),
                    true,
                    &mut new_battles,
                    &mut popups,
                );
            }
            ActionRequest::PassFortify => {
                if self.game.phase != Phase::Fortify {
                    return Err("Cannot pass fortify outside the fortify phase.".into());
                }
                self.apply_action(Action::PassFortify, true, &mut new_battles, &mut popups);
            }
        }

        Ok(json!({
            "ok": true,
            "state": self.state_json(),
            "new_battles": new_battles,
            "popups": popups,
        }))
    }

    fn validate_attack_edge(&self, source: usize, target: usize) -> Result<(), String> {
        if self.game.phase != Phase::Attack {
            return Err("Cannot attack outside the attack phase.".into());
        }
        if source >= N || target >= N {
            return Err("Invalid source or target territory index.".into());
        }
        let me = self.game.active;
        if self.game.owner[source] != me {
            return Err(format!(
                "Source {} (#{source}) is not owned by the active player.",
                territory_name(source)
            ));
        }
        if self.game.owner[target] == me {
            return Err(format!(
                "Target {} (#{target}) is owned by the active player.",
                territory_name(target)
            ));
        }
        if EDGES.binary_search(&(source, target)).is_err() {
            return Err(format!(
                "{} and {} are not adjacent.",
                territory_name(source),
                territory_name(target)
            ));
        }
        if self.game.troops[source] <= 1 {
            return Err(format!(
                "{} needs more than 1 troop to attack.",
                territory_name(source)
            ));
        }
        Ok(())
    }

    fn handle_ai_step(&mut self, req: AiStepRequest) -> Result<Value, String> {
        let mode = req
            .mode
            .as_deref()
            .unwrap_or("step")
            .trim()
            .to_ascii_lowercase();
        if !matches!(mode.as_str(), "step" | "turn" | "to_human") {
            return Err(format!(
                "Invalid AI step mode '{mode}'; expected 'step', 'turn', or 'to_human'."
            ));
        }

        let mut new_battles = Vec::new();
        let mut popups = Vec::new();
        let mut steps_taken = 0usize;
        let mut last_confidence: Option<f32> = None;
        let mut last_value_estimate: Option<f32> = None;

        let start_turn = self.game.turns;
        let max_steps = match mode.as_str() {
            "step" => 1,
            _ => MAX_AI_STEPS_GUARD,
        };
        let record_popups = mode == "step";

        while steps_taken < max_steps && self.game.phase != Phase::Ended && !self.is_human_turn() {
            let active = self.game.active;
            let seat_kind = self.seats[active].clone();
            let decision =
                self.registry
                    .choose_action(&seat_kind, &self.game, &mut self.policy_rng)?;
            if let Some(v) = decision.value_estimate {
                self.seat_values[active] = Some(v);
                last_value_estimate = Some(v);
            }
            if let Some(c) = decision.confidence {
                last_confidence = Some(c);
            }
            self.apply_action(
                decision.action,
                record_popups,
                &mut new_battles,
                &mut popups,
            );
            if let Some(conf) = decision.confidence {
                self.last_action_summary
                    .push_str(&format!(" ({:.0}% conf)", conf * 100.0));
            }
            steps_taken += 1;

            if mode == "turn" && self.game.turns != start_turn {
                break;
            }
        }

        Ok(json!({
            "ok": true,
            "state": self.state_json(),
            "new_battles": new_battles,
            "popups": popups,
            "steps_taken": steps_taken,
            "confidence": last_confidence,
            "value_estimate": last_value_estimate,
        }))
    }

    fn apply_action(
        &mut self,
        action: Action,
        record_popups: bool,
        new_battles: &mut Vec<BattleDto>,
        popups: &mut Vec<PopupDto>,
    ) {
        if !self.game.legal(action) {
            return;
        }
        let actor = self.game.active;
        let turn_num = self.game.turns + 1;
        let actor_name = self.seat_labels[actor].clone();

        if self.game.turns != self.last_turn_number {
            self.last_turn_owners = self.game.owner;
            self.last_turn_troops = self.game.troops;
            self.last_turn_number = self.game.turns;
        }

        match action {
            Action::Place(t) => {
                self.game.step(action);
                self.changed[t] = true;
                if record_popups {
                    popups.push(PopupDto {
                        territory: t,
                        text: "+1".to_string(),
                        color: player_color_hex(actor),
                    });
                }
                self.last_action_summary = format!(
                    "{actor_name} reinforced {} (#{t}) · {} left",
                    territory_name(t),
                    self.game.pending
                );
            }
            Action::Attack(src, dst, dice) => {
                let defender = self.game.owner[dst];
                let outcome = self.game.step(action).expect("attack produces outcome");
                self.changed[src] = true;
                self.changed[dst] = true;

                let battle = BattleRecord::from_outcome(
                    self.next_battle_id,
                    turn_num,
                    actor,
                    defender,
                    src,
                    dst,
                    outcome,
                    self.game.troops[src],
                    self.game.troops[dst],
                    0.0,
                );
                self.next_battle_id += 1;

                if record_popups {
                    if outcome.attacker_losses > 0 {
                        popups.push(PopupDto {
                            territory: src,
                            text: format!("-{}", outcome.attacker_losses),
                            color: COLOR_LOSS_RED.to_string(),
                        });
                    }
                    if outcome.captured {
                        popups.push(PopupDto {
                            territory: dst,
                            text: "CAPTURED!".to_string(),
                            color: COLOR_CAPTURED_GOLD.to_string(),
                        });
                    } else if outcome.defender_losses > 0 {
                        popups.push(PopupDto {
                            territory: dst,
                            text: format!("-{}", outcome.defender_losses),
                            color: COLOR_LOSS_RED.to_string(),
                        });
                    }
                }

                self.last_action_summary = if outcome.captured {
                    format!(
                        "{actor_name} captured {} (#{dst}) from {} (#{src}) rolling {dice}d6!",
                        territory_name(dst),
                        territory_name(src)
                    )
                } else {
                    format!(
                        "{actor_name} attacked {} → {} ({dice}d6): atk -{}, def -{}",
                        territory_name(src),
                        territory_name(dst),
                        outcome.attacker_losses,
                        outcome.defender_losses
                    )
                };

                new_battles.push(BattleDto::from(&battle));
                self.battle_history.push(battle);
                if self.battle_history.len() > MAX_BATTLE_HISTORY {
                    let excess = self.battle_history.len() - MAX_BATTLE_HISTORY;
                    self.battle_history.drain(0..excess);
                }
            }
            Action::Fortify(src, dst, opt) => {
                let moved = self.game.fortify_amount(src, opt).unwrap_or(0);
                self.game.step(action);
                self.changed[src] = true;
                self.changed[dst] = true;
                if record_popups && moved > 0 {
                    popups.push(PopupDto {
                        territory: src,
                        text: format!("-{moved}"),
                        color: COLOR_FORTIFY_OUT.to_string(),
                    });
                    popups.push(PopupDto {
                        territory: dst,
                        text: format!("+{moved}"),
                        color: COLOR_FORTIFY_IN.to_string(),
                    });
                }
                self.last_action_summary = format!(
                    "{actor_name} fortified {moved} troop(s) {} → {}",
                    territory_name(src),
                    territory_name(dst)
                );
                self.on_turn_finished();
            }
            Action::FinishAttack => {
                self.game.step(action);
                self.last_action_summary = format!("{actor_name} finished attacking");
            }
            Action::PassFortify => {
                self.game.step(action);
                self.last_action_summary = format!("{actor_name} ended turn");
                self.on_turn_finished();
            }
        }

        if self.game.phase == Phase::Ended {
            self.last_action_summary = match self.game.winner {
                Some(w) => format!("Victory! {} conquered the world!", self.seat_labels[w]),
                None => format!("Draw reached at turn limit ({}).", self.game.max_turns),
            };
        }
    }

    fn on_turn_finished(&mut self) {
        self.changed = std::array::from_fn(|i| {
            self.last_turn_owners[i] != self.game.owner[i]
                || self.last_turn_troops[i] != self.game.troops[i]
        });
        self.last_turn_owners = self.game.owner;
        self.last_turn_troops = self.game.troops;
        self.last_turn_number = self.game.turns;
    }

    fn state_json(&self) -> Value {
        let phase_str = match self.game.phase {
            Phase::Reinforce => "reinforce",
            Phase::Attack => "attack",
            Phase::Fortify => "fortify",
            Phase::Ended => "ended",
        };
        let human_turn = self.is_human_turn();
        let me = self.game.active;

        let mut selectable = [false; N];
        let mut attack_targets: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut fortify_targets: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut fortify_amounts: BTreeMap<String, [Option<u64>; 3]> = BTreeMap::new();

        match self.game.phase {
            Phase::Reinforce => {
                if human_turn {
                    for (t, flag) in selectable.iter_mut().enumerate() {
                        *flag = self.game.owner[t] == me;
                    }
                }
            }
            Phase::Attack => {
                for &(a, b) in &EDGES {
                    if self.game.owner[a] == me
                        && self.game.owner[b] != me
                        && self.game.troops[a] > 1
                    {
                        if human_turn {
                            selectable[a] = true;
                        }
                        attack_targets.entry(a.to_string()).or_default().push(b);
                    }
                }
            }
            Phase::Fortify => {
                for &(a, b) in &EDGES {
                    if self.game.owner[a] == me
                        && self.game.owner[b] == me
                        && self.game.troops[a] > 1
                    {
                        if human_turn {
                            selectable[a] = true;
                        }
                        fortify_targets.entry(a.to_string()).or_default().push(b);
                    }
                }
                for key in fortify_targets.keys() {
                    if let Ok(src) = key.parse::<usize>() {
                        fortify_amounts.insert(
                            key.clone(),
                            [
                                self.game.fortify_amount(src, 0),
                                self.game.fortify_amount(src, 1),
                                self.game.fortify_amount(src, 2),
                            ],
                        );
                    }
                }
            }
            Phase::Ended => {}
        }

        let continent_owners: [Option<usize>; 6] =
            std::array::from_fn(|r| continent_owner(&self.game, r));
        let seats_json: Vec<Value> = self.seats.iter().map(seat_to_json).collect();
        let seat_archs: Vec<Option<String>> = self
            .seats
            .iter()
            .map(|seat| match seat {
                SeatKind::Checkpoint { path, .. } => self
                    .registry
                    .model_info(path)
                    .map(|(h, l)| format!("{h}×{l}")),
                _ => None,
            })
            .collect();
        let battles: Vec<BattleDto> = self.battle_history.iter().map(BattleDto::from).collect();

        json!({
            "game_number": self.game_number,
            "players": self.game.players,
            "max_turns": self.game.max_turns,
            "turns": self.game.turns,
            "active": self.game.active,
            "phase": phase_str,
            "pending": self.game.pending,
            "winner": self.game.winner,
            "human_turn": human_turn,
            "has_human": self.has_human(),
            "owner": self.game.owner.as_slice(),
            "troops": self.game.troops.as_slice(),
            "changed": self.changed.as_slice(),
            "continent_owners": continent_owners,
            "seats": seats_json,
            "seat_labels": self.seat_labels,
            "seat_values": self.seat_values,
            "seat_archs": seat_archs,
            "selectable": selectable.as_slice(),
            "attack_targets": attack_targets,
            "fortify_targets": fortify_targets,
            "fortify_amounts": fortify_amounts,
            "last_action_summary": self.last_action_summary,
            "battles": battles,
        })
    }
}

fn player_color_hex(seat: usize) -> String {
    let c = PLAYER_COLORS[seat % MAX_PLAYERS];
    format!("#{:02x}{:02x}{:02x}", c.r(), c.g(), c.b())
}

fn build_seat_labels(seats: &[SeatKind]) -> Vec<String> {
    let human_count = seats.iter().filter(|s| s.is_human()).count();
    let mut human_idx = 0usize;
    seats
        .iter()
        .enumerate()
        .map(|(i, seat)| match seat {
            SeatKind::Human => {
                human_idx += 1;
                if human_count <= 1 {
                    "You".to_string()
                } else {
                    format!("Human {human_idx}")
                }
            }
            _ => format!("P{} · {}", i + 1, seat.short_label()),
        })
        .collect()
}

fn canonical_key(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn checkpoint_entry_for_path(path: &Path) -> DiscoveredCheckpoint {
    discover_checkpoints(std::slice::from_ref(&path.to_path_buf()))
        .into_iter()
        .next()
        .unwrap_or_else(|| {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("custom")
                .to_string();
            let run_name = path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .filter(|s| !s.is_empty())
                .unwrap_or("custom")
                .to_string();
            let display_name = if run_name == stem {
                stem.clone()
            } else {
                format!("{run_name} / {stem}")
            };
            DiscoveredCheckpoint {
                path: path.to_path_buf(),
                display_name,
                run_name,
                file_stem: stem,
                elo: None,
                update: None,
                trained_players: None,
            }
        })
}

fn upsert_custom_checkpoint(list: &mut Vec<DiscoveredCheckpoint>, entry: DiscoveredCheckpoint) {
    let key = canonical_key(&entry.path);
    if let Some(existing) = list.iter_mut().find(|c| canonical_key(&c.path) == key) {
        *existing = entry;
    } else {
        list.insert(0, entry);
    }
}

fn overnight_priority(ckpt: &DiscoveredCheckpoint) -> u8 {
    let is_overnight = ckpt.run_name.eq_ignore_ascii_case("overnight")
        || ckpt
            .path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("overnight"));
    let stem = ckpt.file_stem.to_ascii_lowercase();
    match (is_overnight, stem.as_str()) {
        (true, "best") => 0,
        (true, "latest") => 1,
        (false, "best") => 2,
        (false, "latest") => 3,
        (true, _) => 4,
        (false, _) => 5,
    }
}

fn compare_default_candidate(
    a: &DiscoveredCheckpoint,
    b: &DiscoveredCheckpoint,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let elo_ord = match (a.elo, b.elo) {
        (Some(ea), Some(eb)) if (ea - eb).abs() > 1e-6 => eb.total_cmp(&ea),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        _ => Ordering::Equal,
    };
    if elo_ord != Ordering::Equal {
        return elo_ord;
    }
    overnight_priority(a)
        .cmp(&overnight_priority(b))
        .then_with(|| b.update.cmp(&a.update))
        .then_with(|| a.path.cmp(&b.path))
}

fn parse_dice_pref(val: Option<&Value>) -> Result<Option<usize>, String> {
    match val {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "" | "max" | "auto" | "auto_max" => Ok(None),
            "1" => Ok(Some(1)),
            "2" => Ok(Some(2)),
            "3" => Ok(Some(3)),
            other => Err(format!(
                "Invalid dice_pref '{other}'; expected 'max', 1, 2, or 3."
            )),
        },
        Some(Value::Number(n)) => match n.as_u64() {
            Some(d @ 1..=3) => Ok(Some(d as usize)),
            _ => Err(format!(
                "Invalid dice_pref '{n}'; expected 'max', 1, 2, or 3."
            )),
        },
        Some(other) => Err(format!(
            "Invalid dice_pref '{other}'; expected 'max', 1, 2, or 3."
        )),
    }
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

struct HttpResponse {
    status: u16,
    reason: &'static str,
    content_type: &'static str,
    body: Vec<u8>,
}

impl HttpResponse {
    fn json(status: u16, value: &Value) -> Self {
        let body = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
        Self {
            status,
            reason: status_reason(status),
            content_type: "application/json; charset=utf-8",
            body,
        }
    }

    fn json_raw(status: u16, body: String) -> Self {
        Self {
            status,
            reason: status_reason(status),
            content_type: "application/json; charset=utf-8",
            body: body.into_bytes(),
        }
    }

    fn html(body: Vec<u8>) -> Self {
        Self {
            status: 200,
            reason: "OK",
            content_type: "text/html; charset=utf-8",
            body,
        }
    }

    fn text(status: u16, msg: impl Into<String>) -> Self {
        Self {
            status,
            reason: status_reason(status),
            content_type: "text/plain; charset=utf-8",
            body: msg.into().into_bytes(),
        }
    }

    fn no_content() -> Self {
        Self {
            status: 204,
            reason: "No Content",
            content_type: "text/plain; charset=utf-8",
            body: Vec::new(),
        }
    }
}

fn status_reason(code: u16) -> &'static str {
    match code {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        _ => "Error",
    }
}

fn read_http_request(stream: &TcpStream) -> Result<Option<HttpRequest>, HttpResponse> {
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    let n = reader
        .read_line(&mut request_line)
        .map_err(|_| HttpResponse::text(400, "Failed to read request line"))?;
    if n == 0 {
        return Ok(None);
    }

    let mut total_header_bytes = n;
    let mut parts = request_line
        .trim_end_matches(['\r', '\n'])
        .split_whitespace();
    let Some(method) = parts.next() else {
        return Err(HttpResponse::text(400, "Malformed HTTP request line"));
    };
    let Some(raw_target) = parts.next() else {
        return Err(HttpResponse::text(400, "Missing HTTP request target"));
    };
    let path = raw_target
        .split('?')
        .next()
        .unwrap_or(raw_target)
        .to_string();

    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        let read_n = reader
            .read_line(&mut line)
            .map_err(|_| HttpResponse::text(400, "Failed to read HTTP headers"))?;
        if read_n == 0 {
            break;
        }
        total_header_bytes += read_n;
        if total_header_bytes > MAX_HEADER_BYTES {
            return Err(HttpResponse::text(413, "HTTP headers too large"));
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':')
            && name.trim().eq_ignore_ascii_case("content-length")
        {
            content_length = value
                .trim()
                .parse::<usize>()
                .map_err(|_| HttpResponse::text(400, "Invalid Content-Length header"))?;
            if content_length > MAX_BODY_BYTES {
                return Err(HttpResponse::text(413, "Request body too large"));
            }
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader
            .read_exact(&mut body)
            .map_err(|_| HttpResponse::text(400, "Incomplete HTTP request body"))?;
    }

    Ok(Some(HttpRequest {
        method: method.to_ascii_uppercase(),
        path,
        body,
    }))
}

fn write_http_response(
    stream: &mut TcpStream,
    resp: &HttpResponse,
    is_head: bool,
) -> io::Result<()> {
    let mut header = format!(
        "HTTP/1.1 {} {}\r\n\
         Content-Type: {}\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         X-Content-Type-Options: nosniff\r\n\
         Referrer-Policy: no-referrer\r\n\
         Connection: close\r\n\r\n",
        resp.status,
        resp.reason,
        resp.content_type,
        resp.body.len()
    );
    if resp.status == 204 {
        header = "HTTP/1.1 204 No Content\r\n\
         Cache-Control: no-store\r\n\
         X-Content-Type-Options: nosniff\r\n\
         Referrer-Policy: no-referrer\r\n\
         Connection: close\r\n\r\n"
            .to_string();
    }
    stream.write_all(header.as_bytes())?;
    if !is_head && resp.status != 204 && !resp.body.is_empty() {
        stream.write_all(&resp.body)?;
    }
    stream.flush()
}

fn load_index_html() -> Result<Vec<u8>, String> {
    let cwd_path = PathBuf::from("web/play/index.html");
    if let Ok(bytes) = fs::read(&cwd_path) {
        return Ok(bytes);
    }
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("web/play/index.html");
    if let Ok(bytes) = fs::read(&manifest_path) {
        return Ok(bytes);
    }
    Err(format!(
        "web/play/index.html not found (checked '{}' and '{}')",
        cwd_path.display(),
        manifest_path.display()
    ))
}

fn handle_connection(mut stream: TcpStream, state: &Arc<Mutex<ServerState>>) -> io::Result<()> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(15)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(15)));

    let req = match read_http_request(&stream) {
        Ok(Some(req)) => req,
        Ok(None) => return Ok(()),
        Err(resp) => {
            let _ = write_http_response(&mut stream, &resp, false);
            return Ok(());
        }
    };

    if req.method == "OPTIONS" {
        return write_http_response(&mut stream, &HttpResponse::no_content(), false);
    }

    let is_head = req.method == "HEAD";
    let effective_method = if is_head { "GET" } else { req.method.as_str() };
    let resp = dispatch_request(effective_method, &req.path, &req.body, state);
    write_http_response(&mut stream, &resp, is_head)
}

fn parse_json_body<T: for<'de> Deserialize<'de>>(body: &[u8]) -> Result<T, HttpResponse> {
    serde_json::from_slice(body).map_err(|err| {
        HttpResponse::json(
            400,
            &json!({ "ok": false, "error": format!("Invalid JSON body: {err}") }),
        )
    })
}

fn dispatch_request(
    method: &str,
    path: &str,
    body: &[u8],
    state: &Arc<Mutex<ServerState>>,
) -> HttpResponse {
    match (method, path) {
        ("GET", "/") | ("GET", "/index.html") => match load_index_html() {
            Ok(bytes) => HttpResponse::html(bytes),
            Err(err) => HttpResponse::text(500, err),
        },
        ("GET", "/api/board") => {
            let guard = state.lock().expect("server state mutex");
            HttpResponse::json_raw(200, guard.board_json.clone())
        }
        ("GET", "/api/checkpoints") => {
            let guard = state.lock().expect("server state mutex");
            let checkpoints = guard.list_checkpoints();
            HttpResponse::json(200, &json!({ "checkpoints": checkpoints }))
        }
        ("GET", "/api/state") => {
            let guard = state.lock().expect("server state mutex");
            HttpResponse::json(200, &guard.state_json())
        }
        ("POST", "/api/checkpoints/load") => {
            let req: LoadCheckpointRequest = match parse_json_body(body) {
                Ok(v) => v,
                Err(resp) => return resp,
            };
            let mut guard = state.lock().expect("server state mutex");
            match guard.load_checkpoint(&req.path) {
                Ok(val) => HttpResponse::json(200, &val),
                Err(err) => HttpResponse::json(400, &json!({ "ok": false, "error": err })),
            }
        }
        ("POST", "/api/new_game") => {
            let req: NewGameRequest = match parse_json_body(body) {
                Ok(v) => v,
                Err(resp) => return resp,
            };
            let mut guard = state.lock().expect("server state mutex");
            match guard.new_game(req) {
                Ok(val) => HttpResponse::json(200, &val),
                Err(err) => HttpResponse::json(400, &json!({ "ok": false, "error": err })),
            }
        }
        ("POST", "/api/action") => {
            let req: ActionRequest = match parse_json_body(body) {
                Ok(v) => v,
                Err(resp) => return resp,
            };
            let mut guard = state.lock().expect("server state mutex");
            match guard.handle_human_action(req) {
                Ok(val) => HttpResponse::json(200, &val),
                Err(err) => HttpResponse::json(400, &json!({ "ok": false, "error": err })),
            }
        }
        ("POST", "/api/ai_step") => {
            let req: AiStepRequest = if body.iter().all(|b| b.is_ascii_whitespace()) {
                AiStepRequest::default()
            } else {
                match parse_json_body(body) {
                    Ok(v) => v,
                    Err(resp) => return resp,
                }
            };
            let mut guard = state.lock().expect("server state mutex");
            match guard.handle_ai_step(req) {
                Ok(val) => HttpResponse::json(200, &val),
                Err(err) => HttpResponse::json(400, &json!({ "ok": false, "error": err })),
            }
        }
        (_, "/api/board" | "/api/checkpoints" | "/api/state")
        | (_, "/api/checkpoints/load" | "/api/new_game" | "/api/action" | "/api/ai_step") => {
            HttpResponse::json(
                405,
                &json!({ "ok": false, "error": format!("Method {method} not allowed for {path}") }),
            )
        }
        _ => HttpResponse::json(
            404,
            &json!({ "ok": false, "error": format!("Route '{path}' not found") }),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn board_json_has_42_territories_6_continents_9_bridges() {
        let board = Map::new().export_board_json();
        assert_eq!(board["width"].as_f64(), Some(1280.0));
        assert_eq!(board["height"].as_f64(), Some(680.0));
        assert_eq!(board["continent_bonus"].as_u64(), Some(2));

        let colors = board["player_colors"]
            .as_array()
            .expect("player_colors array");
        assert_eq!(colors.len(), 6);
        assert_eq!(colors[0].as_str(), Some("#4fc3f7"));
        assert_eq!(colors[5].as_str(), Some("#f472b6"));

        let continents = board["continents"].as_array().expect("continents array");
        assert_eq!(continents.len(), 6);
        assert_eq!(continents[0]["name"].as_str(), Some("NORTH AMERICA"));
        assert_eq!(continents[0]["short_name"].as_str(), Some("North America"));
        assert_eq!(
            continents[0]["territories"].as_array().map(Vec::len),
            Some(7)
        );
        assert!(
            continents
                .iter()
                .all(|c| !c["outline"].as_array().unwrap().is_empty())
        );

        let territories = board["territories"].as_array().expect("territories array");
        assert_eq!(territories.len(), 42);
        assert_eq!(territories[0]["name"].as_str(), Some("Yukon"));
        assert_eq!(territories[0]["continent"].as_u64(), Some(0));
        assert_eq!(
            territories[0]["neighbors"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![1, 5, 6]
        );

        let bridges = board["bridges"].as_array().expect("bridges array");
        assert_eq!(bridges.len(), 9);
        assert_eq!(bridges[0]["label"].as_str(), Some("A"));
        assert_eq!(bridges[0]["a"].as_u64(), Some(1));
        assert_eq!(bridges[0]["b"].as_u64(), Some(11));
        assert!(bridges[0]["points"].as_array().unwrap().len() >= 2);
    }

    #[test]
    fn session_human_reinforce_attack_fortify_and_ai_checkpoint_step() {
        let mut state = ServerState::new(Vec::new(), None);
        let best_ckpt = PathBuf::from("runs/overnight/best.mpk");
        let seat1 = if best_ckpt.is_file() {
            SeatInput {
                kind: "checkpoint".to_string(),
                path: Some(best_ckpt.to_string_lossy().into_owned()),
                label: Some("overnight / best".to_string()),
                sampling: Some("balanced".to_string()),
            }
        } else {
            SeatInput {
                kind: "heuristic".to_string(),
                path: None,
                label: None,
                sampling: None,
            }
        };

        // Start a deterministic 2-player game (Human vs Checkpoint/Heuristic).
        let started = state
            .new_game(NewGameRequest {
                seats: vec![
                    SeatInput {
                        kind: "human".to_string(),
                        path: None,
                        label: None,
                        sampling: None,
                    },
                    seat1,
                ],
                max_turns: Some(200),
                seed: Some(42),
            })
            .expect("new_game succeeds");
        assert_eq!(started["ok"], Value::Bool(true));

        // If AI opens, advance to human turn first.
        if !state.is_human_turn() {
            let res = state
                .handle_ai_step(AiStepRequest {
                    mode: Some("to_human".to_string()),
                })
                .expect("ai step to human");
            assert!(res["steps_taken"].as_u64().unwrap() > 0);
        }
        assert!(state.is_human_turn());
        assert_eq!(state.game.phase, Phase::Reinforce);

        // Place all reinforcements onto a frontline territory (owned with an enemy neighbor).
        let me = state.game.active;
        let (frontline, enemy) = EDGES
            .iter()
            .copied()
            .find(|&(a, b)| state.game.owner[a] == me && state.game.owner[b] != me)
            .expect("human has at least one frontline edge");
        let pending = state.game.pending;
        let placed_resp = state
            .handle_human_action(ActionRequest::Place {
                territory: frontline,
                count: Some(pending),
            })
            .expect("place succeeds");
        assert_eq!(placed_resp["state"]["phase"], json!("attack"));
        assert_eq!(placed_resp["popups"].as_array().unwrap().len(), 1);
        assert_eq!(
            placed_resp["popups"][0]["text"],
            json!(format!("+{pending}"))
        );

        // Execute a human attack (and check battle recording).
        let atk_resp = state
            .handle_human_action(ActionRequest::Attack {
                source: frontline,
                target: enemy,
                dice: Some(3),
            })
            .expect("attack succeeds");
        assert_eq!(atk_resp["new_battles"].as_array().unwrap().len(), 1);
        let battle = &atk_resp["new_battles"][0];
        assert_eq!(battle["source"], json!(frontline));
        assert_eq!(battle["target"], json!(enemy));
        assert_eq!(
            battle["attack_rolls"].as_array().unwrap().len(),
            battle["attack_dice_count"].as_u64().unwrap() as usize
        );

        // Finish attack -> Fortify phase.
        if state.game.phase == Phase::Attack {
            let fin = state
                .handle_human_action(ActionRequest::FinishAttack)
                .expect("finish attack");
            assert_eq!(fin["state"]["phase"], json!("fortify"));
        }

        // Fortify if possible, else pass fortify.
        if state.game.phase == Phase::Fortify {
            let fort_edge = EDGES.iter().copied().find(|&(a, b)| {
                state.game.owner[a] == me && state.game.owner[b] == me && state.game.troops[a] > 1
            });
            if let Some((src, dst)) = fort_edge {
                let f_resp = state
                    .handle_human_action(ActionRequest::Fortify {
                        source: src,
                        target: dst,
                        option: 2,
                    })
                    .expect("fortify succeeds");
                assert_eq!(f_resp["ok"], Value::Bool(true));
            } else {
                let p_resp = state
                    .handle_human_action(ActionRequest::PassFortify)
                    .expect("pass fortify succeeds");
                assert_eq!(p_resp["ok"], Value::Bool(true));
            }
        }

        // Now step the AI opponent for a full turn.
        if state.game.phase != Phase::Ended && !state.is_human_turn() {
            let ai_resp = state
                .handle_ai_step(AiStepRequest {
                    mode: Some("turn".to_string()),
                })
                .expect("ai turn succeeds");
            assert!(ai_resp["steps_taken"].as_u64().unwrap() > 0);
            if best_ckpt.is_file() {
                assert!(ai_resp["state"]["seat_archs"][1].is_string());
                assert!(ai_resp["value_estimate"].is_number());
            }
        }

        // Verify variable player counts (2..=6) with Heuristic & Random seats.
        for count in 2..=MAX_PLAYERS {
            let seats: Vec<SeatInput> = (0..count)
                .map(|i| SeatInput {
                    kind: if i % 2 == 0 {
                        "heuristic".to_string()
                    } else {
                        "random".to_string()
                    },
                    path: None,
                    label: None,
                    sampling: None,
                })
                .collect();
            let ng = state
                .new_game(NewGameRequest {
                    seats,
                    max_turns: Some(50),
                    seed: Some(100 + count as u64),
                })
                .expect("multi-player new_game succeeds");
            assert_eq!(ng["state"]["players"].as_u64(), Some(count as u64));
            let step = state
                .handle_ai_step(AiStepRequest {
                    mode: Some("turn".to_string()),
                })
                .expect("ai turn in multi-player game");
            assert!(step["steps_taken"].as_u64().unwrap() > 0);
        }
    }

    #[test]
    fn http_roundtrip_serves_api_board_checkpoints_and_state() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local addr");
        let state = Arc::new(Mutex::new(ServerState::new(Vec::new(), None)));

        let server_state = Arc::clone(&state);
        let server_thread = thread::spawn(move || {
            for stream in listener.incoming().take(4) {
                let stream = stream.expect("accept client");
                handle_connection(stream, &server_state).expect("handle connection");
            }
        });

        let send_request = |raw: &str| -> (u16, Value) {
            let mut client = TcpStream::connect(addr).expect("connect to test server");
            client.write_all(raw.as_bytes()).expect("write request");
            let mut resp_bytes = Vec::new();
            client.read_to_end(&mut resp_bytes).expect("read response");
            let resp_str = String::from_utf8(resp_bytes).expect("utf8 response");
            let (head, body) = resp_str.split_once("\r\n\r\n").expect("http separator");
            let status: u16 = head
                .lines()
                .next()
                .unwrap()
                .split_whitespace()
                .nth(1)
                .unwrap()
                .parse()
                .unwrap();
            let json: Value = serde_json::from_str(body).expect("json body");
            (status, json)
        };

        // 1. GET /api/board
        let (status, board) = send_request("GET /api/board HTTP/1.1\r\nHost: localhost\r\n\r\n");
        assert_eq!(status, 200);
        assert_eq!(board["territories"].as_array().map(Vec::len), Some(42));
        assert_eq!(board["continents"].as_array().map(Vec::len), Some(6));
        assert_eq!(board["bridges"].as_array().map(Vec::len), Some(9));

        // 2. GET /api/checkpoints
        let (status, ckpts) =
            send_request("GET /api/checkpoints HTTP/1.1\r\nHost: localhost\r\n\r\n");
        assert_eq!(status, 200);
        assert!(ckpts["checkpoints"].is_array());

        // 3. GET /api/state
        let (status, st) = send_request("GET /api/state HTTP/1.1\r\nHost: localhost\r\n\r\n");
        assert_eq!(status, 200);
        assert_eq!(st["players"].as_u64(), Some(2));
        assert_eq!(st["owner"].as_array().map(Vec::len), Some(42));
        assert_eq!(st["troops"].as_array().map(Vec::len), Some(42));
        assert_eq!(st["selectable"].as_array().map(Vec::len), Some(42));

        // 4. POST /api/new_game with JSON body
        let payload = r#"{"seats":[{"kind":"human"},{"kind":"heuristic"},{"kind":"random"}],"max_turns":120,"seed":7}"#;
        let post_req = format!(
            "POST /api/new_game HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            payload.len(),
            payload
        );
        let (status, ng) = send_request(&post_req);
        assert_eq!(status, 200);
        assert_eq!(ng["ok"], Value::Bool(true));
        assert_eq!(ng["state"]["players"].as_u64(), Some(3));
        assert_eq!(ng["state"]["max_turns"].as_u64(), Some(120));

        server_thread
            .join()
            .expect("server thread finished cleanly");
    }
}
