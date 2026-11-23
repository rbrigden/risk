//! Single-source-of-truth stateless/serializable Arena engine and WebAssembly exports.
//!
//! Executes the exact Rust simulator (`crate::game`), feature encoder (`crate::train::features`),
//! heuristic/random bots (`crate::train::bots`), and pure-Rust GNN policy forward pass
//! (`crate::train::infer`) inside JavaScript / WebAssembly host runtimes via `wasm-bindgen`.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use crate::board_meta::{continent_owner, player_color_hex, territory_name};
use crate::game::{Action, BattleOutcome, EDGES, Game, MAX_PLAYERS, N, Phase};
use crate::train::bots::{heuristic_action, random_action};
use crate::train::features::{BoardTopology, encode_observation};
use crate::train::infer::{
    RiskPolicyWeights, SamplingMode, choose_policy_action, greedy_masked_action,
};

const DEFAULT_MAX_TURNS: u32 = 500;
const MAX_BATTLE_HISTORY: usize = 80;
const MAX_BLITZ_STEPS: usize = 64;
const MAX_AI_STEPS_GUARD: usize = 2000;

const COLOR_LOSS_RED: &str = "#f06660";
const COLOR_CAPTURED_GOLD: &str = "#f5c542";
const COLOR_FORTIFY_OUT: &str = "#90a6b9";
const COLOR_FORTIFY_IN: &str = "#8ad672";

thread_local! {
    static TOPOLOGY: BoardTopology = BoardTopology::new();
    static POLICY_CACHE: RefCell<HashMap<String, RiskPolicyWeights>> = RefCell::new(HashMap::new());
}

fn with_topology<R>(f: impl FnOnce(&BoardTopology) -> R) -> R {
    TOPOLOGY.with(f)
}

fn policy_arch(policy_id: &str) -> Option<(usize, usize)> {
    POLICY_CACHE.with(|cache| {
        cache
            .borrow()
            .get(policy_id)
            .map(|w| (w.hidden_dim, w.num_layers))
    })
}

fn parse_phase_str(raw: &str) -> Result<Phase, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "reinforce" => Ok(Phase::Reinforce),
        "attack" => Ok(Phase::Attack),
        "fortify" => Ok(Phase::Fortify),
        "ended" => Ok(Phase::Ended),
        other => Err(format!("Invalid phase '{other}'.")),
    }
}

fn phase_to_str(phase: Phase) -> &'static str {
    match phase {
        Phase::Reinforce => "reinforce",
        Phase::Attack => "attack",
        Phase::Fortify => "fortify",
        Phase::Ended => "ended",
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PopupDto {
    pub territory: usize,
    pub text: String,
    pub color: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BattleDto {
    pub id: u64,
    pub turn: u32,
    pub attacker: usize,
    pub defender: usize,
    pub source: usize,
    pub target: usize,
    pub attack_dice_count: usize,
    pub defense_dice_count: usize,
    pub attack_rolls: Vec<u8>,
    pub defense_rolls: Vec<u8>,
    pub attacker_losses: u64,
    pub defender_losses: u64,
    pub captured: bool,
    pub attacker_troops_after: u64,
    pub defender_troops_after: u64,
}

impl BattleDto {
    #[allow(clippy::too_many_arguments)]
    fn from_outcome(
        id: u64,
        turn: u32,
        attacker: usize,
        defender: usize,
        source: usize,
        target: usize,
        outcome: BattleOutcome,
        attacker_troops_after: u64,
        defender_troops_after: u64,
    ) -> Self {
        let attack_dice_count = outcome.attack.iter().filter(|&&d| d > 0).count();
        let defense_dice_count = outcome.defense.iter().filter(|&&d| d > 0).count();
        Self {
            id,
            turn,
            attacker,
            defender,
            source,
            target,
            attack_dice_count,
            defense_dice_count,
            attack_rolls: outcome.attack[..attack_dice_count.min(3)].to_vec(),
            defense_rolls: outcome.defense[..defense_dice_count.min(2)].to_vec(),
            attacker_losses: outcome.attacker_losses,
            defender_losses: outcome.defender_losses,
            captured: outcome.captured,
            attacker_troops_after,
            defender_troops_after,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeatConfig {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<String>,
}

impl SeatConfig {
    fn normalize(self) -> Result<Self, String> {
        let kind = self.kind.trim().to_ascii_lowercase();
        match kind.as_str() {
            "human" => Ok(Self {
                kind: "human".into(),
                policy_id: None,
                label: self.label.filter(|s| !s.trim().is_empty()),
                sampling: None,
            }),
            "heuristic" => Ok(Self {
                kind: "heuristic".into(),
                policy_id: None,
                label: self.label.filter(|s| !s.trim().is_empty()),
                sampling: None,
            }),
            "random" => Ok(Self {
                kind: "random".into(),
                policy_id: None,
                label: self.label.filter(|s| !s.trim().is_empty()),
                sampling: None,
            }),
            "policy" => {
                let policy_id = self
                    .policy_id
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| "Policy seat requires a non-empty 'policy_id'.".to_string())?;
                let sampling = SamplingMode::parse(self.sampling.as_deref().unwrap_or("balanced"))?;
                let label = self
                    .label
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| policy_id.clone());
                Ok(Self {
                    kind: "policy".into(),
                    policy_id: Some(policy_id),
                    label: Some(label),
                    sampling: Some(sampling.as_str().to_string()),
                })
            }
            other => Err(format!(
                "Invalid seat kind '{other}'; expected 'human', 'policy', 'heuristic', or 'random'."
            )),
        }
    }

    fn is_human(&self) -> bool {
        self.kind == "human"
    }

    fn short_label(&self) -> String {
        match self.kind.as_str() {
            "human" => self
                .label
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or("You")
                .to_string(),
            "heuristic" => self
                .label
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or("Heuristic")
                .to_string(),
            "random" => self
                .label
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or("Random")
                .to_string(),
            "policy" => self
                .label
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .or(self.policy_id.as_deref())
                .unwrap_or("Policy")
                .to_string(),
            _ => "Bot".to_string(),
        }
    }
}

fn build_seat_labels(seats: &[SeatConfig]) -> Vec<String> {
    let human_count = seats.iter().filter(|s| s.is_human()).count();
    let mut human_idx = 0usize;
    seats
        .iter()
        .enumerate()
        .map(|(i, seat)| {
            if seat.is_human() {
                human_idx += 1;
                if let Some(custom) = seat.label.as_deref().filter(|s| !s.trim().is_empty()) {
                    custom.to_string()
                } else if human_count <= 1 {
                    "You".to_string()
                } else {
                    format!("Human {human_idx}")
                }
            } else {
                format!("P{} · {}", i + 1, seat.short_label())
            }
        })
        .collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArenaSnapshot {
    pub game_number: u64,
    pub players: usize,
    pub max_turns: u32,
    pub turns: u32,
    pub active: usize,
    pub phase: String,
    pub pending: u64,
    pub winner: Option<usize>,
    pub owner: Vec<usize>,
    pub troops: Vec<u64>,
    pub seats: Vec<SeatConfig>,
    pub seat_labels: Vec<String>,
    pub seat_values: Vec<Option<f32>>,
    pub changed: Vec<bool>,
    pub last_turn_owners: Vec<usize>,
    pub last_turn_troops: Vec<u64>,
    pub last_turn_number: u32,
    pub battles: Vec<BattleDto>,
    pub next_battle_id: u64,
    pub last_action_summary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rng_seed: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClientState {
    pub game_number: u64,
    pub players: usize,
    pub max_turns: u32,
    pub turns: u32,
    pub active: usize,
    pub phase: String,
    pub pending: u64,
    pub winner: Option<usize>,
    pub human_turn: bool,
    pub has_human: bool,
    pub owner: Vec<usize>,
    pub troops: Vec<u64>,
    pub changed: Vec<bool>,
    pub continent_owners: [Option<usize>; 6],
    pub seats: Vec<SeatConfig>,
    pub seat_labels: Vec<String>,
    pub seat_values: Vec<Option<f32>>,
    pub seat_archs: Vec<Option<String>>,
    pub selectable: Vec<bool>,
    pub attack_targets: BTreeMap<String, Vec<usize>>,
    pub fortify_targets: BTreeMap<String, Vec<usize>>,
    pub fortify_amounts: BTreeMap<String, [Option<u64>; 3]>,
    pub last_action_summary: String,
    pub battles: Vec<BattleDto>,
}

#[derive(Debug, Deserialize)]
struct NewGameConfig {
    seats: Vec<SeatConfig>,
    #[serde(default)]
    max_turns: Option<u32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    game_number: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum HumanActionRequest {
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

#[derive(Debug, Deserialize)]
struct InferDebugState {
    players: usize,
    max_turns: u32,
    owner: Vec<usize>,
    troops: Vec<u64>,
    active: usize,
    phase: u8,
    pending: u64,
    turns: u32,
}

struct ArenaEngine {
    game: Game,
    policy_rng: ChaCha8Rng,
    deterministic: bool,
    seats: Vec<SeatConfig>,
    seat_labels: Vec<String>,
    seat_values: Vec<Option<f32>>,
    game_number: u64,
    changed: [bool; N],
    last_turn_owners: [usize; N],
    last_turn_troops: [u64; N],
    last_turn_number: u32,
    battle_history: Vec<BattleDto>,
    next_battle_id: u64,
    last_action_summary: String,
}

fn vec_to_array_usize(v: &[usize], name: &str) -> Result<[usize; N], String> {
    if v.len() != N {
        return Err(format!("Expected '{name}' length {N}, got {}.", v.len()));
    }
    let mut arr = [0usize; N];
    arr.copy_from_slice(v);
    Ok(arr)
}

fn vec_to_array_u64(v: &[u64], name: &str) -> Result<[u64; N], String> {
    if v.len() != N {
        return Err(format!("Expected '{name}' length {N}, got {}.", v.len()));
    }
    let mut arr = [0u64; N];
    arr.copy_from_slice(v);
    Ok(arr)
}

fn vec_to_array_bool(v: &[bool], name: &str) -> Result<[bool; N], String> {
    if v.len() != N {
        return Err(format!("Expected '{name}' length {N}, got {}.", v.len()));
    }
    let mut arr = [false; N];
    arr.copy_from_slice(v);
    Ok(arr)
}

impl ArenaEngine {
    fn from_snapshot(snap: ArenaSnapshot) -> Result<Self, String> {
        if !(2..=MAX_PLAYERS).contains(&snap.players) {
            return Err(format!("Invalid player count {}.", snap.players));
        }
        if snap.seats.len() != snap.players {
            return Err(format!(
                "Seat count {} does not match players {}.",
                snap.seats.len(),
                snap.players
            ));
        }
        if snap.active >= snap.players {
            return Err(format!("Invalid active seat {}.", snap.active));
        }
        let owner = vec_to_array_usize(&snap.owner, "owner")?;
        for &o in &owner {
            if o >= snap.players {
                return Err(format!(
                    "Territory owner {o} out of range 0..{}",
                    snap.players
                ));
            }
        }
        let troops = vec_to_array_u64(&snap.troops, "troops")?;
        let changed = vec_to_array_bool(&snap.changed, "changed")?;
        let last_turn_owners = vec_to_array_usize(&snap.last_turn_owners, "last_turn_owners")?;
        let last_turn_troops = vec_to_array_u64(&snap.last_turn_troops, "last_turn_troops")?;
        let phase = parse_phase_str(&snap.phase)?;

        let (game_seed, policy_rng, deterministic) = match snap.rng_seed {
            Some(s) => (
                Some(s),
                ChaCha8Rng::seed_from_u64(s ^ 0x51A7_C0DE_9E37_79B9),
                true,
            ),
            None => (None, ChaCha8Rng::from_os_rng(), false),
        };

        let game = Game::from_parts(
            snap.players,
            snap.max_turns,
            owner,
            troops,
            snap.active,
            phase,
            snap.pending,
            snap.turns,
            snap.winner,
            game_seed,
        );

        let seat_labels = if snap.seat_labels.len() == snap.players {
            snap.seat_labels
        } else {
            build_seat_labels(&snap.seats)
        };
        let seat_values = if snap.seat_values.len() == snap.players {
            snap.seat_values
        } else {
            vec![None; snap.players]
        };

        Ok(Self {
            game,
            policy_rng,
            deterministic,
            seats: snap.seats,
            seat_labels,
            seat_values,
            game_number: snap.game_number,
            changed,
            last_turn_owners,
            last_turn_troops,
            last_turn_number: snap.last_turn_number,
            battle_history: snap.battles,
            next_battle_id: snap.next_battle_id,
            last_action_summary: snap.last_action_summary,
        })
    }

    fn into_snapshot(mut self) -> ArenaSnapshot {
        let next_seed = if self.deterministic {
            Some(self.policy_rng.random::<u64>())
        } else {
            None
        };
        ArenaSnapshot {
            game_number: self.game_number,
            players: self.game.players,
            max_turns: self.game.max_turns,
            turns: self.game.turns,
            active: self.game.active,
            phase: phase_to_str(self.game.phase).to_string(),
            pending: self.game.pending,
            winner: self.game.winner,
            owner: self.game.owner.to_vec(),
            troops: self.game.troops.to_vec(),
            seats: self.seats,
            seat_labels: self.seat_labels,
            seat_values: self.seat_values,
            changed: self.changed.to_vec(),
            last_turn_owners: self.last_turn_owners.to_vec(),
            last_turn_troops: self.last_turn_troops.to_vec(),
            last_turn_number: self.last_turn_number,
            battles: self.battle_history,
            next_battle_id: self.next_battle_id,
            last_action_summary: self.last_action_summary,
            rng_seed: next_seed,
        }
    }

    fn is_human_turn(&self) -> bool {
        self.game.phase != Phase::Ended
            && self
                .seats
                .get(self.game.active)
                .is_some_and(SeatConfig::is_human)
    }

    fn has_human(&self) -> bool {
        self.seats.iter().any(SeatConfig::is_human)
    }

    fn project_state(&self) -> ClientState {
        let human_turn = self.is_human_turn();
        let me = self.game.active;

        let mut selectable = vec![false; N];
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

        let seat_archs: Vec<Option<String>> = self
            .seats
            .iter()
            .map(|seat| {
                if seat.kind == "policy" {
                    seat.policy_id
                        .as_deref()
                        .and_then(policy_arch)
                        .map(|(h, l)| format!("{h}×{l}"))
                } else {
                    None
                }
            })
            .collect();

        ClientState {
            game_number: self.game_number,
            players: self.game.players,
            max_turns: self.game.max_turns,
            turns: self.game.turns,
            active: self.game.active,
            phase: phase_to_str(self.game.phase).to_string(),
            pending: self.game.pending,
            winner: self.game.winner,
            human_turn,
            has_human: self.has_human(),
            owner: self.game.owner.to_vec(),
            troops: self.game.troops.to_vec(),
            changed: self.changed.to_vec(),
            continent_owners,
            seats: self.seats.clone(),
            seat_labels: self.seat_labels.clone(),
            seat_values: self.seat_values.clone(),
            seat_archs,
            selectable,
            attack_targets,
            fortify_targets,
            fortify_amounts,
            last_action_summary: self.last_action_summary.clone(),
            battles: self.battle_history.clone(),
        }
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

                let battle = BattleDto::from_outcome(
                    self.next_battle_id,
                    turn_num,
                    actor,
                    defender,
                    src,
                    dst,
                    outcome,
                    self.game.troops[src],
                    self.game.troops[dst],
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

                new_battles.push(battle.clone());
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

    fn choose_ai_action(
        &mut self,
        seat: &SeatConfig,
    ) -> Result<(Action, Option<f32>, Option<f32>), String> {
        match seat.kind.as_str() {
            "random" => {
                let id = random_action(&self.game, &mut self.policy_rng);
                let action = Action::decode(id)
                    .ok_or_else(|| format!("Random bot returned invalid action {id}."))?;
                Ok((action, None, None))
            }
            "heuristic" => {
                let id =
                    with_topology(|topo| heuristic_action(&self.game, topo, &mut self.policy_rng));
                let action = Action::decode(id)
                    .ok_or_else(|| format!("Heuristic bot returned invalid action {id}."))?;
                Ok((action, None, None))
            }
            "policy" => {
                let policy_id = seat
                    .policy_id
                    .as_deref()
                    .ok_or_else(|| "Policy seat missing 'policy_id'.".to_string())?;
                let sampling = SamplingMode::parse(seat.sampling.as_deref().unwrap_or("balanced"))?;
                let (probs, val, mask) = with_topology(|topo| {
                    let obs = encode_observation(&self.game, self.game.active, topo);
                    if !obs.mask.iter().any(|&legal| legal) {
                        return Err("Active player has no legal actions.".to_string());
                    }
                    POLICY_CACHE.with(|cache| {
                        let borrowed = cache.borrow();
                        let weights = borrowed.get(policy_id).ok_or_else(|| {
                            format!("Policy '{policy_id}' is not registered in WASM cache.")
                        })?;
                        let (probs, val) = weights.infer(&obs, topo);
                        Ok((probs, val, obs.mask))
                    })
                })?;

                let act_id = choose_policy_action(&probs, &mask, sampling, &mut self.policy_rng);
                let action = Action::decode(act_id)
                    .ok_or_else(|| format!("Policy returned invalid action {act_id}."))?;
                let confidence = probs.get(act_id).copied().filter(|p| p.is_finite());
                let value_estimate = val.is_finite().then_some(val);
                Ok((action, confidence, value_estimate))
            }
            other => Err(format!("Cannot run AI step for seat kind '{other}'.")),
        }
    }
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

pub fn new_game_impl(config_json: &str) -> Result<String, String> {
    let cfg: NewGameConfig = serde_json::from_str(config_json)
        .map_err(|e| format!("Invalid new game config JSON: {e}"))?;
    if !(2..=MAX_PLAYERS).contains(&cfg.seats.len()) {
        return Err(format!(
            "Seat count must be between 2 and {MAX_PLAYERS}, got {}.",
            cfg.seats.len()
        ));
    }
    let max_turns = cfg.max_turns.unwrap_or(DEFAULT_MAX_TURNS);
    if max_turns == 0 {
        return Err("max_turns must be at least 1.".into());
    }

    let mut seats = Vec::with_capacity(cfg.seats.len());
    for seat in cfg.seats {
        seats.push(seat.normalize()?);
    }

    let seat_labels = build_seat_labels(&seats);
    let seat_values = vec![None; seats.len()];
    let game = Game::new(seats.len(), max_turns, cfg.seed);
    let (policy_rng, deterministic) = match cfg.seed {
        Some(s) => (ChaCha8Rng::seed_from_u64(s ^ 0x51A7_C0DE_9E37_79B9), true),
        None => (ChaCha8Rng::from_os_rng(), false),
    };
    let last_turn_owners = game.owner;
    let last_turn_troops = game.troops;
    let last_turn_number = game.turns;
    let game_number = cfg.game_number.unwrap_or(1);
    let last_action_summary = format!(
        "New {}-player match started · {} opens",
        seats.len(),
        seat_labels[game.active]
    );

    let engine = ArenaEngine {
        game,
        policy_rng,
        deterministic,
        seats,
        seat_labels,
        seat_values,
        game_number,
        changed: [false; N],
        last_turn_owners,
        last_turn_troops,
        last_turn_number,
        battle_history: Vec::new(),
        next_battle_id: 1,
        last_action_summary,
    };

    let state = engine.project_state();
    let snapshot = engine.into_snapshot();
    #[derive(Serialize)]
    struct NewGameResponse {
        snapshot: ArenaSnapshot,
        state: ClientState,
    }
    serde_json::to_string(&NewGameResponse { snapshot, state })
        .map_err(|e| format!("Serialization error: {e}"))
}

pub fn project_state_impl(snapshot_json: &str) -> Result<String, String> {
    let snap: ArenaSnapshot =
        serde_json::from_str(snapshot_json).map_err(|e| format!("Invalid snapshot JSON: {e}"))?;
    let engine = ArenaEngine::from_snapshot(snap)?;
    let state = engine.project_state();
    serde_json::to_string(&state).map_err(|e| format!("Serialization error: {e}"))
}

pub fn apply_human_action_impl(snapshot_json: &str, action_json: &str) -> Result<String, String> {
    let snap: ArenaSnapshot =
        serde_json::from_str(snapshot_json).map_err(|e| format!("Invalid snapshot JSON: {e}"))?;
    let req: HumanActionRequest =
        serde_json::from_str(action_json).map_err(|e| format!("Invalid human action JSON: {e}"))?;
    let mut engine = ArenaEngine::from_snapshot(snap)?;

    if engine.game.phase == Phase::Ended {
        return Err("Game has already ended.".into());
    }
    if !engine.is_human_turn() {
        return Err("It is not currently a human seat's turn.".into());
    }

    let mut new_battles = Vec::new();
    let mut popups = Vec::new();
    let me = engine.game.active;
    let actor_name = engine.seat_labels[me].clone();

    match req {
        HumanActionRequest::Place { territory, count } => {
            if engine.game.phase != Phase::Reinforce {
                return Err("Cannot place troops outside the reinforce phase.".into());
            }
            if territory >= N {
                return Err(format!("Invalid territory index {territory}."));
            }
            if engine.game.owner[territory] != me {
                return Err(format!(
                    "Territory {} (#{territory}) is not owned by the active player.",
                    territory_name(territory)
                ));
            }
            let placed = count.unwrap_or(1).clamp(1, engine.game.pending.max(1));
            let mut actual = 0u64;
            for _ in 0..placed {
                if engine.game.phase == Phase::Reinforce && engine.game.owner[territory] == me {
                    engine.apply_action(
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
                engine.last_action_summary = if actual == 1 {
                    format!(
                        "{actor_name} reinforced {} (#{territory}) · {} left",
                        territory_name(territory),
                        engine.game.pending
                    )
                } else {
                    format!(
                        "{actor_name} placed +{actual} on {} (#{territory}) · {} left",
                        territory_name(territory),
                        engine.game.pending
                    )
                };
            }
        }
        HumanActionRequest::Attack {
            source,
            target,
            dice,
        } => {
            engine.validate_attack_edge(source, target)?;
            let max_dice = (engine.game.troops[source] - 1).min(3) as usize;
            let clamped_dice = dice.unwrap_or(max_dice).clamp(1, max_dice);
            engine.apply_action(
                Action::Attack(source, target, clamped_dice),
                true,
                &mut new_battles,
                &mut popups,
            );
        }
        HumanActionRequest::Blitz {
            source,
            target,
            dice_pref,
        } => {
            engine.validate_attack_edge(source, target)?;
            let fixed_dice = parse_dice_pref(dice_pref.as_ref())?;
            let mut total_atk_losses = 0u64;
            let mut total_def_losses = 0u64;
            let mut captured = false;

            for _ in 0..MAX_BLITZ_STEPS {
                if engine.game.phase != Phase::Attack
                    || engine.game.owner[source] != me
                    || engine.game.owner[target] == me
                    || engine.game.troops[source] <= 1
                {
                    break;
                }
                let max_dice = (engine.game.troops[source] - 1).min(3) as usize;
                let dice = match fixed_dice {
                    Some(d) => d.clamp(1, max_dice),
                    None => max_dice,
                };
                let before_len = new_battles.len();
                engine.apply_action(
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
        HumanActionRequest::FinishAttack => {
            if engine.game.phase != Phase::Attack {
                return Err("Cannot finish attack outside the attack phase.".into());
            }
            engine.apply_action(Action::FinishAttack, true, &mut new_battles, &mut popups);
        }
        HumanActionRequest::Fortify {
            source,
            target,
            option,
        } => {
            if engine.game.phase != Phase::Fortify {
                return Err("Cannot fortify outside the fortify phase.".into());
            }
            if source >= N || target >= N {
                return Err("Invalid source or target territory index.".into());
            }
            if engine.game.owner[source] != me || engine.game.owner[target] != me {
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
                .find(|&opt| engine.game.fortify_amount(source, opt).is_some())
                .ok_or_else(|| {
                    format!(
                        "{} has no movable troops to fortify.",
                        territory_name(source)
                    )
                })?;
            engine.apply_action(
                Action::Fortify(source, target, legal_opt),
                true,
                &mut new_battles,
                &mut popups,
            );
        }
        HumanActionRequest::PassFortify => {
            if engine.game.phase != Phase::Fortify {
                return Err("Cannot pass fortify outside the fortify phase.".into());
            }
            engine.apply_action(Action::PassFortify, true, &mut new_battles, &mut popups);
        }
    }

    let state = engine.project_state();
    let snapshot = engine.into_snapshot();
    #[derive(Serialize)]
    struct HumanActionResponse {
        snapshot: ArenaSnapshot,
        state: ClientState,
        new_battles: Vec<BattleDto>,
        popups: Vec<PopupDto>,
    }
    serde_json::to_string(&HumanActionResponse {
        snapshot,
        state,
        new_battles,
        popups,
    })
    .map_err(|e| format!("Serialization error: {e}"))
}

pub fn register_policy_impl(policy_id: &str, weights_json: &str) -> Result<String, String> {
    let trimmed = policy_id.trim();
    if trimmed.is_empty() {
        return Err("policy_id cannot be empty.".into());
    }
    let weights = RiskPolicyWeights::decode_from_json(weights_json)?;
    let hidden_dim = weights.hidden_dim;
    let num_layers = weights.num_layers;
    POLICY_CACHE.with(|cache| {
        cache.borrow_mut().insert(trimmed.to_string(), weights);
    });
    serde_json::to_string(&json!({
        "ok": true,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }))
    .map_err(|e| format!("Serialization error: {e}"))
}

pub fn ai_step_impl(snapshot_json: &str, mode: &str) -> Result<String, String> {
    let snap: ArenaSnapshot =
        serde_json::from_str(snapshot_json).map_err(|e| format!("Invalid snapshot JSON: {e}"))?;
    let mode_norm = mode.trim().to_ascii_lowercase();
    if !matches!(mode_norm.as_str(), "step" | "turn" | "to_human") {
        return Err(format!(
            "Invalid AI step mode '{mode}'; expected 'step', 'turn', or 'to_human'."
        ));
    }

    let mut engine = ArenaEngine::from_snapshot(snap)?;
    let mut new_battles = Vec::new();
    let mut popups = Vec::new();
    let mut steps_taken = 0usize;
    let mut last_confidence: Option<f32> = None;
    let mut last_value_estimate: Option<f32> = None;

    let start_turn = engine.game.turns;
    let max_steps = match mode_norm.as_str() {
        "step" => 1,
        _ => MAX_AI_STEPS_GUARD,
    };
    let record_popups = mode_norm == "step";

    while steps_taken < max_steps && engine.game.phase != Phase::Ended && !engine.is_human_turn() {
        let active = engine.game.active;
        let seat = engine.seats[active].clone();
        let (action, confidence, value_estimate) = engine.choose_ai_action(&seat)?;

        if let Some(v) = value_estimate {
            engine.seat_values[active] = Some(v);
            last_value_estimate = Some(v);
        }
        if let Some(c) = confidence {
            last_confidence = Some(c);
        }

        engine.apply_action(action, record_popups, &mut new_battles, &mut popups);
        if let Some(conf) = confidence {
            engine
                .last_action_summary
                .push_str(&format!(" ({:.0}% conf)", conf * 100.0));
        }
        steps_taken += 1;

        if mode_norm == "turn" && engine.game.turns != start_turn {
            break;
        }
    }

    let state = engine.project_state();
    let snapshot = engine.into_snapshot();
    #[derive(Serialize)]
    struct AiStepResponse {
        snapshot: ArenaSnapshot,
        state: ClientState,
        new_battles: Vec<BattleDto>,
        popups: Vec<PopupDto>,
        steps_taken: usize,
        confidence: Option<f32>,
        value_estimate: Option<f32>,
    }
    serde_json::to_string(&AiStepResponse {
        snapshot,
        state,
        new_battles,
        popups,
        steps_taken,
        confidence: last_confidence,
        value_estimate: last_value_estimate,
    })
    .map_err(|e| format!("Serialization error: {e}"))
}

pub fn infer_debug_impl(state_json: &str, policy_id: &str) -> Result<String, String> {
    let raw: InferDebugState = serde_json::from_str(state_json)
        .map_err(|e| format!("Invalid infer debug state JSON: {e}"))?;
    let phase =
        Phase::from_code(raw.phase).ok_or_else(|| format!("Invalid phase code {}.", raw.phase))?;
    let owner = vec_to_array_usize(&raw.owner, "owner")?;
    let troops = vec_to_array_u64(&raw.troops, "troops")?;
    let game = Game::from_parts(
        raw.players,
        raw.max_turns,
        owner,
        troops,
        raw.active,
        phase,
        raw.pending,
        raw.turns,
        None,
        Some(0),
    );

    let (obs, probs, value) = with_topology(|topo| {
        let obs = encode_observation(&game, game.active, topo);
        POLICY_CACHE.with(|cache| {
            let borrowed = cache.borrow();
            let weights = borrowed
                .get(policy_id.trim())
                .ok_or_else(|| format!("Policy '{}' is not registered.", policy_id.trim()))?;
            let (probs, val) = weights.infer(&obs, topo);
            Ok::<_, String>((obs, probs, val))
        })
    })?;

    let greedy_action = greedy_masked_action(&probs, &obs.mask);
    let mut legal_probs: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .filter_map(|(id, &p)| obs.mask[id].then_some((id, p)))
        .collect();
    legal_probs.sort_by(|(id_a, p_a), (id_b, p_b)| p_b.total_cmp(p_a).then_with(|| id_a.cmp(id_b)));
    let top5: Vec<Value> = legal_probs
        .into_iter()
        .take(5)
        .map(|(action_id, prob)| json!({ "action_id": action_id, "prob": prob }))
        .collect();

    let fortify_amounts: Vec<[Option<u64>; 3]> = (0..N)
        .map(|t| {
            [
                game.fortify_amount(t, 0),
                game.fortify_amount(t, 1),
                game.fortify_amount(t, 2),
            ]
        })
        .collect();

    serde_json::to_string(&json!({
        "nodes": obs.nodes.as_slice(),
        "global": obs.global.as_slice(),
        "mask": obs.mask.as_slice(),
        "probs": probs,
        "value": value,
        "greedy_action": greedy_action,
        "top5": top5,
        "fortify_amounts": fortify_amounts,
    }))
    .map_err(|e| format!("Serialization error: {e}"))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn arena_new_game(config_json: &str) -> Result<String, JsValue> {
    new_game_impl(config_json).map_err(|e| JsValue::from_str(&e))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn arena_project_state(snapshot_json: &str) -> Result<String, JsValue> {
    project_state_impl(snapshot_json).map_err(|e| JsValue::from_str(&e))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn arena_apply_human_action(snapshot_json: &str, action_json: &str) -> Result<String, JsValue> {
    apply_human_action_impl(snapshot_json, action_json).map_err(|e| JsValue::from_str(&e))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn arena_register_policy(policy_id: &str, weights_json: &str) -> Result<String, JsValue> {
    register_policy_impl(policy_id, weights_json).map_err(|e| JsValue::from_str(&e))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn arena_ai_step(snapshot_json: &str, mode: &str) -> Result<String, JsValue> {
    ai_step_impl(snapshot_json, mode).map_err(|e| JsValue::from_str(&e))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn arena_infer_debug(state_json: &str, policy_id: &str) -> Result<String, JsValue> {
    infer_debug_impl(state_json, policy_id).map_err(|e| JsValue::from_str(&e))
}
