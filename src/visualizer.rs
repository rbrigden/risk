//! Interactive Risk visualizer supporting Human vs variable trained policies & bots.

use std::path::PathBuf;

use eframe::egui::{self, Align, Color32, CornerRadius, Layout, Margin, RichText, Stroke, vec2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::game::{Action, EDGES, Game, MAX_PLAYERS, N, Phase, REGIONS};

mod battles;
mod map;
mod policy;
#[cfg(not(target_arch = "wasm32"))]
pub mod server;
mod state;

use policy::{
    DiscoveredCheckpoint, PolicyRegistry, PolicySampling, SeatKind, discover_checkpoints,
};
use state::{
    BattleRecord, CONTINENT_BONUS, MapOverlayState, TroopDeltaPopup, continent_owner, player_color,
    territory_name,
};

const DEFAULT_MAX_TURNS: u32 = 500;
const MAX_BATTLE_HISTORY: usize = 120;
const MAX_BLITZ_STEPS: usize = 64;

const PANEL_BG: Color32 = Color32::from_rgb(16, 25, 35);
const CARD_BG: Color32 = Color32::from_rgb(23, 35, 48);
const CARD_BORDER: Color32 = Color32::from_rgb(41, 61, 80);
const TEXT_PRIMARY: Color32 = Color32::from_rgb(232, 240, 248);
const TEXT_MUTED: Color32 = Color32::from_rgb(144, 166, 185);
const ACCENT_CYAN: Color32 = Color32::from_rgb(79, 195, 247);
const ACCENT_GOLD: Color32 = Color32::from_rgb(245, 197, 66);
const ACCENT_RED: Color32 = Color32::from_rgb(240, 102, 96);
const ACCENT_GREEN: Color32 = Color32::from_rgb(138, 214, 114);

/// Export the deterministic cartographic Risk board geometry as JSON.
#[cfg(not(target_arch = "wasm32"))]
pub fn export_board_json() -> serde_json::Value {
    map::Map::new().export_board_json()
}

/// Load a `.mpk` policy checkpoint on CPU, auto-detecting `(hidden_dim, num_layers)`.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_policy_checkpoint(
    path: &std::path::Path,
) -> Result<
    (
        crate::train::model::RiskPolicyNet<burn::backend::NdArray>,
        usize,
        usize,
    ),
    String,
> {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    policy::load_checkpoint_auto(path, &device)
}

/// Launch the interactive visualizer in a native window, scanning `runs/` for checkpoints.
#[cfg(not(target_arch = "wasm32"))]
pub fn run() -> eframe::Result {
    run_with_paths(vec![PathBuf::from("runs")])
}

/// Launch the interactive visualizer in a native window, scanning `checkpoint_roots` for `.mpk` checkpoints.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_with_paths(checkpoint_roots: Vec<PathBuf>) -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1380.0, 860.0])
            .with_min_inner_size([960.0, 640.0]),
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    eframe::run_native(
        "Riskbot",
        options,
        Box::new(move |cc| Ok(Box::new(Visualizer::new(cc, checkpoint_roots)))),
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start(canvas: web_sys::HtmlCanvasElement) -> Result<(), wasm_bindgen::JsValue> {
    eframe::WebRunner::new()
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(|cc| Ok(Box::new(Visualizer::new(cc, Vec::new())))),
        )
        .await
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AiSpeed {
    Slow,
    Normal,
    Fast,
    Instant,
}

impl AiSpeed {
    fn label(self) -> &'static str {
        match self {
            Self::Slow => "Slow",
            Self::Normal => "Normal",
            Self::Fast => "Fast",
            Self::Instant => "Instant",
        }
    }

    fn step_delay(self, phase: Phase) -> f64 {
        match (self, phase) {
            (Self::Slow, Phase::Reinforce) => 0.16,
            (Self::Slow, _) => 0.72,
            (Self::Normal, Phase::Reinforce) => 0.08,
            (Self::Normal, _) => 0.48,
            (Self::Fast, Phase::Reinforce) => 0.02,
            (Self::Fast, _) => 0.14,
            (Self::Instant, _) => 0.0,
        }
    }

    fn animate_dice(self) -> bool {
        !matches!(self, Self::Instant)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReinforceBrush {
    One,
    Five,
    All,
}

impl ReinforceBrush {
    fn label(self) -> &'static str {
        match self {
            Self::One => "+1",
            Self::Five => "+5",
            Self::All => "+All",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttackDicePref {
    AutoMax,
    Fixed(usize),
}

impl AttackDicePref {
    fn resolve(self, attacker_troops: u64) -> usize {
        let max_dice = attacker_troops.saturating_sub(1).min(3) as usize;
        match self {
            Self::AutoMax => max_dice,
            Self::Fixed(d) => d.min(max_dice).max(1),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FortifyPref {
    One,
    Half,
    All,
}

impl FortifyPref {
    fn to_option(self, game: &Game, source: usize) -> Option<usize> {
        let preferred = match self {
            Self::One => [0, 1, 2],
            Self::Half => [1, 2, 0],
            Self::All => [2, 1, 0],
        };
        preferred
            .into_iter()
            .find(|&opt| game.fortify_amount(source, opt).is_some())
    }
}

#[derive(Clone, Debug)]
struct MatchSetupDraft {
    num_players: usize,
    seats: Vec<SeatKind>,
    max_turns: u32,
    seed_input: String,
    custom_ckpt_input: String,
    status_message: Option<(String, bool)>,
}

impl MatchSetupDraft {
    fn default_with_checkpoints(checkpoints: &[DiscoveredCheckpoint]) -> Self {
        let mut seats = Vec::with_capacity(MAX_PLAYERS);
        seats.push(SeatKind::Human);
        for slot in 1..MAX_PLAYERS {
            seats.push(default_ai_seat(checkpoints, slot - 1));
        }
        Self {
            num_players: 2,
            seats,
            max_turns: DEFAULT_MAX_TURNS,
            seed_input: String::new(),
            custom_ckpt_input: String::new(),
            status_message: None,
        }
    }

    fn apply_preset(&mut self, players: usize, human_seat0: bool, ckpts: &[DiscoveredCheckpoint]) {
        self.num_players = players.clamp(2, MAX_PLAYERS);
        for (i, seat) in self.seats.iter_mut().enumerate() {
            if i == 0 && human_seat0 {
                *seat = SeatKind::Human;
            } else {
                let ai_idx = if human_seat0 { i.saturating_sub(1) } else { i };
                *seat = default_ai_seat(ckpts, ai_idx);
            }
        }
    }
}

fn default_ai_seat(checkpoints: &[DiscoveredCheckpoint], idx: usize) -> SeatKind {
    if let Some(ckpt) = checkpoints.get(idx).or_else(|| checkpoints.first()) {
        SeatKind::Checkpoint {
            path: ckpt.path.clone(),
            label: ckpt.display_name.clone(),
            sampling: PolicySampling::Balanced,
        }
    } else if idx.is_multiple_of(2) {
        SeatKind::Heuristic
    } else {
        SeatKind::Random
    }
}

struct Visualizer {
    game: Game,
    seats: Vec<SeatKind>,
    seat_labels: Vec<String>,
    policy_rng: ChaCha8Rng,
    registry: PolicyRegistry,
    checkpoint_roots: Vec<PathBuf>,
    discovered: Vec<DiscoveredCheckpoint>,
    map: map::Map,

    // Match & turn state
    changed: [bool; N],
    last_turn_owners: [usize; N],
    last_turn_troops: [u64; N],
    last_turn_number: u32,
    battle_history: Vec<BattleRecord>,
    next_battle_id: u64,
    popups: Vec<TroopDeltaPopup>,
    seat_values: [Option<f32>; MAX_PLAYERS],
    last_action_summary: String,
    last_error: Option<String>,
    game_number: u64,

    // Human interaction state
    selected: Option<usize>,
    last_human_attack: Option<(usize, usize)>,
    reinforce_brush: ReinforceBrush,
    attack_dice_pref: AttackDicePref,
    fortify_pref: FortifyPref,

    // AI playback controls
    ai_speed: AiSpeed,
    ai_paused: bool,
    next_ai_step_at: f64,
    skip_to_human: bool,

    // Match setup modal
    setup_open: bool,
    draft: MatchSetupDraft,
}

impl Visualizer {
    fn new(cc: &eframe::CreationContext<'_>, checkpoint_roots: Vec<PathBuf>) -> Self {
        configure_theme(&cc.egui_ctx);

        let discovered = discover_checkpoints(&checkpoint_roots);
        let draft = MatchSetupDraft::default_with_checkpoints(&discovered);
        let mut registry = PolicyRegistry::new();

        // Preload default AI seat checkpoint if present so first AI turn has zero stutter.
        let active_seats: Vec<SeatKind> = draft.seats[..draft.num_players].to_vec();
        let mut last_error = None;
        for seat in &active_seats {
            if let SeatKind::Checkpoint { path, .. } = seat
                && let Err(err) = registry.preload(path)
            {
                last_error = Some(err);
            }
        }

        let seat_labels = build_seat_labels(&active_seats);
        let game = Game::new(active_seats.len(), draft.max_turns, None);
        let (last_turn_owners, last_turn_troops) = (game.owner, game.troops);

        Self {
            game,
            seats: active_seats,
            seat_labels,
            policy_rng: ChaCha8Rng::from_os_rng(),
            registry,
            checkpoint_roots,
            discovered,
            map: map::Map::new(),
            changed: [false; N],
            last_turn_owners,
            last_turn_troops,
            last_turn_number: 0,
            battle_history: Vec::new(),
            next_battle_id: 1,
            popups: Vec::new(),
            seat_values: [None; MAX_PLAYERS],
            last_action_summary: "Match ready — make your move!".to_string(),
            last_error,
            game_number: 1,
            selected: None,
            last_human_attack: None,
            reinforce_brush: ReinforceBrush::One,
            attack_dice_pref: AttackDicePref::AutoMax,
            fortify_pref: FortifyPref::All,
            ai_speed: AiSpeed::Normal,
            ai_paused: false,
            next_ai_step_at: 0.0,
            skip_to_human: false,
            setup_open: false,
            draft,
        }
    }

    fn is_human_turn(&self) -> bool {
        self.game.phase != Phase::Ended
            && self
                .seats
                .get(self.game.active)
                .is_some_and(SeatKind::is_human)
    }

    fn has_any_human(&self) -> bool {
        self.seats.iter().any(SeatKind::is_human)
    }

    fn start_match_from_draft(&mut self, now: f64) {
        let count = self.draft.num_players.clamp(2, MAX_PLAYERS);
        let seed_trimmed = self.draft.seed_input.trim();
        let seed = if seed_trimmed.is_empty() {
            None
        } else {
            match seed_trimmed.parse::<u64>() {
                Ok(s) => Some(s),
                Err(_) => {
                    self.draft.status_message =
                        Some(("Seed must be an unsigned 64-bit integer.".to_string(), true));
                    return;
                }
            }
        };

        let chosen: Vec<SeatKind> = self.draft.seats[..count].to_vec();
        for seat in &chosen {
            if let SeatKind::Checkpoint { path, .. } = seat
                && let Err(err) = self.registry.preload(path)
            {
                self.draft.status_message = Some((err, true));
                return;
            }
        }

        self.seats = chosen;
        self.seat_labels = build_seat_labels(&self.seats);
        self.reset_game(seed, self.draft.max_turns, now);
        self.draft.status_message = None;
        self.setup_open = false;
    }

    fn reset_game(&mut self, seed: Option<u64>, max_turns: u32, now: f64) {
        self.game = Game::new(self.seats.len(), max_turns, seed);
        self.policy_rng = ChaCha8Rng::from_os_rng();
        self.changed.fill(false);
        self.last_turn_owners = self.game.owner;
        self.last_turn_troops = self.game.troops;
        self.last_turn_number = self.game.turns;
        self.battle_history.clear();
        self.popups.clear();
        self.seat_values = [None; MAX_PLAYERS];
        self.selected = None;
        self.last_human_attack = None;
        self.last_error = None;
        self.ai_paused = false;
        self.skip_to_human = false;
        self.next_ai_step_at = now + 0.35;
        self.game_number += 1;
        self.last_action_summary = format!(
            "New {}-player match started · {} opens",
            self.seats.len(),
            self.seat_labels[self.game.active]
        );
    }

    fn rescan_checkpoints(&mut self) {
        let mut roots = self.checkpoint_roots.clone();
        if roots.is_empty() {
            roots.push(PathBuf::from("runs"));
        }
        self.discovered = discover_checkpoints(&roots);
        let msg = if self.discovered.is_empty() {
            "No .mpk checkpoints found in configured paths.".to_string()
        } else {
            format!("Found {} checkpoint(s).", self.discovered.len())
        };
        self.draft.status_message = Some((msg, false));
    }

    fn add_custom_checkpoint(&mut self) {
        let raw = self.draft.custom_ckpt_input.trim();
        if raw.is_empty() {
            return;
        }
        let path = PathBuf::from(raw);
        match self.registry.preload(&path) {
            Ok((hidden, layers)) => {
                let mut found = discover_checkpoints(std::slice::from_ref(&path));
                let entry = found.pop().unwrap_or_else(|| {
                    let stem = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("custom")
                        .to_string();
                    DiscoveredCheckpoint {
                        path: path.clone(),
                        display_name: format!("custom / {stem}"),
                        run_name: "custom".to_string(),
                        file_stem: stem,
                        elo: None,
                        update: None,
                        trained_players: None,
                    }
                });
                if !self.discovered.iter().any(|c| c.path == entry.path) {
                    self.discovered.insert(0, entry.clone());
                }
                self.draft.status_message = Some((
                    format!("Loaded {} ({hidden}×{layers} GNN)", entry.display_name),
                    false,
                ));
                self.draft.custom_ckpt_input.clear();
            }
            Err(err) => {
                self.draft.status_message = Some((err, true));
            }
        }
    }

    /// Execute a validated game action, recording battle dice, popups, change outlines, and summary.
    fn apply_action(&mut self, action: Action, now: f64, record_popups: bool) {
        if !self.game.legal(action) {
            return;
        }
        let actor = self.game.active;
        let turn_num = self.game.turns + 1;
        let actor_name = self.seat_labels[actor].clone();
        let actor_col = player_color(actor);

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
                    self.push_popup(t, "+1".to_string(), actor_col, now);
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
                    now,
                );
                self.next_battle_id += 1;

                if record_popups {
                    if outcome.attacker_losses > 0 {
                        self.push_popup(
                            src,
                            format!("-{}", outcome.attacker_losses),
                            ACCENT_RED,
                            now,
                        );
                    }
                    if outcome.captured {
                        self.push_popup(dst, "CAPTURED!".to_string(), ACCENT_GOLD, now);
                    } else if outcome.defender_losses > 0 {
                        self.push_popup(
                            dst,
                            format!("-{}", outcome.defender_losses),
                            ACCENT_RED,
                            now,
                        );
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
                    self.push_popup(src, format!("-{moved}"), TEXT_MUTED, now);
                    self.push_popup(dst, format!("+{moved}"), ACCENT_GREEN, now);
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
                self.selected = None;
                self.last_human_attack = None;
                self.last_action_summary = format!("{actor_name} finished attacking");
            }
            Action::PassFortify => {
                self.game.step(action);
                self.last_action_summary = format!("{actor_name} ended turn");
                self.on_turn_finished();
            }
        }

        if self.game.phase == Phase::Ended {
            self.selected = None;
            self.last_human_attack = None;
            self.skip_to_human = false;
            self.last_action_summary = match self.game.winner {
                Some(w) => format!("Victory! {} conquered the world!", self.seat_labels[w]),
                None => format!("Draw reached at turn limit ({}).", self.game.max_turns),
            };
        }
    }

    fn on_turn_finished(&mut self) {
        self.selected = None;
        self.last_human_attack = None;
        self.changed = std::array::from_fn(|i| {
            self.last_turn_owners[i] != self.game.owner[i]
                || self.last_turn_troops[i] != self.game.troops[i]
        });
        self.last_turn_owners = self.game.owner;
        self.last_turn_troops = self.game.troops;
        self.last_turn_number = self.game.turns;
        if self.is_human_turn() {
            self.skip_to_human = false;
        }
    }

    fn push_popup(&mut self, territory: usize, text: String, color: Color32, now: f64) {
        self.popups.push(TroopDeltaPopup {
            territory,
            text,
            color,
            timestamp: now,
        });
        if self.popups.len() > 24 {
            self.popups.remove(0);
        }
    }

    /// Step a single AI action for the active non-human seat.
    fn step_ai_once(&mut self, now: f64, record_popups: bool) -> bool {
        if self.game.phase == Phase::Ended || self.is_human_turn() {
            return false;
        }
        let active = self.game.active;
        let seat_kind = self.seats[active].clone();
        match self
            .registry
            .choose_action(&seat_kind, &self.game, &mut self.policy_rng)
        {
            Ok(decision) => {
                self.last_error = None;
                if let Some(v) = decision.value_estimate {
                    self.seat_values[active] = Some(v);
                }
                let prev_phase = self.game.phase;
                self.apply_action(decision.action, now, record_popups);
                if let Some(conf) = decision.confidence {
                    self.last_action_summary
                        .push_str(&format!(" ({:.0}% conf)", conf * 100.0));
                }
                let delay = self.ai_speed.step_delay(prev_phase);
                self.next_ai_step_at = now + delay;
                true
            }
            Err(err) => {
                self.last_error = Some(err);
                self.ai_paused = true;
                self.skip_to_human = false;
                false
            }
        }
    }

    /// Advance AI turns instantaneously until either a human's turn, turn end (for AI-only), or game end.
    fn advance_ai_instant(&mut self, now: f64, stop_on_turn_change: bool) {
        let start_turn = self.game.turns;
        let mut guard = 0usize;
        while self.game.phase != Phase::Ended && !self.is_human_turn() && guard < 2000 {
            guard += 1;
            if !self.step_ai_once(now, false) {
                break;
            }
            if stop_on_turn_change && self.game.turns != start_turn {
                break;
            }
        }
        self.skip_to_human = false;
    }

    /// Drive automatic AI playback according to `ai_speed`, `ai_paused`, and dice animation timing.
    fn tick_ai(&mut self, ctx: &egui::Context, now: f64) {
        if self.setup_open || self.game.phase == Phase::Ended || self.is_human_turn() {
            return;
        }

        if self.skip_to_human {
            self.advance_ai_instant(now, !self.has_any_human());
            ctx.request_repaint();
            return;
        }

        if self.ai_paused {
            return;
        }

        if self.ai_speed == AiSpeed::Instant {
            self.advance_ai_instant(now, !self.has_any_human());
            ctx.request_repaint();
            return;
        }

        // Wait for dice tumble animation to finish before rolling next AI attack in Slow/Normal mode.
        if let Some(last) = self.battle_history.last()
            && matches!(self.ai_speed, AiSpeed::Slow | AiSpeed::Normal)
            && battles::is_animating(last, now, true)
        {
            ctx.request_repaint();
            return;
        }

        if now >= self.next_ai_step_at {
            self.step_ai_once(now, true);
        }

        let wait = (self.next_ai_step_at - now).clamp(0.016, 0.25);
        ctx.request_repaint_after(std::time::Duration::from_secs_f64(wait));
    }

    /// Compute clickable source territories and valid targets for the current human phase.
    fn compute_highlights(&self) -> ([bool; N], [bool; N], bool) {
        let mut selectable = [false; N];
        let mut valid_targets = [false; N];
        let targeting_attack = self.game.phase == Phase::Attack;

        if !self.is_human_turn() {
            return (selectable, valid_targets, targeting_attack);
        }

        let me = self.game.active;
        match self.game.phase {
            Phase::Reinforce => {
                for (t, flag) in selectable.iter_mut().enumerate() {
                    *flag = self.game.owner[t] == me;
                }
            }
            Phase::Attack => {
                for &(a, b) in &EDGES {
                    if self.game.owner[a] == me
                        && self.game.owner[b] != me
                        && self.game.troops[a] > 1
                    {
                        selectable[a] = true;
                        if self.selected == Some(a) {
                            valid_targets[b] = true;
                        }
                    }
                }
            }
            Phase::Fortify => {
                for &(a, b) in &EDGES {
                    if self.game.owner[a] == me
                        && self.game.owner[b] == me
                        && self.game.troops[a] > 1
                    {
                        selectable[a] = true;
                        if self.selected == Some(a) {
                            valid_targets[b] = true;
                        }
                    }
                }
            }
            Phase::Ended => {}
        }

        (selectable, valid_targets, targeting_attack)
    }

    fn handle_territory_click(&mut self, territory: usize, shift_held: bool, now: f64) {
        if !self.is_human_turn() || self.setup_open {
            return;
        }

        let me = self.game.active;
        match self.game.phase {
            Phase::Reinforce => {
                if self.game.owner[territory] != me {
                    return;
                }
                let count = if shift_held {
                    self.game.pending
                } else {
                    match self.reinforce_brush {
                        ReinforceBrush::One => 1,
                        ReinforceBrush::Five => self.game.pending.min(5),
                        ReinforceBrush::All => self.game.pending,
                    }
                };
                for _ in 0..count {
                    if self.game.phase == Phase::Reinforce && self.game.owner[territory] == me {
                        self.apply_action(Action::Place(territory), now, count == 1);
                    }
                }
                if count > 1 {
                    self.push_popup(territory, format!("+{count}"), player_color(me), now);
                }
            }
            Phase::Attack => {
                if let Some(src) = self.selected {
                    if territory == src {
                        self.selected = None;
                        return;
                    }
                    // Clicked an adjacent enemy -> attack!
                    if self.game.owner[territory] != me
                        && EDGES.binary_search(&(src, territory)).is_ok()
                        && self.game.troops[src] > 1
                    {
                        self.execute_human_attack(src, territory, now);
                        return;
                    }
                }
                // Otherwise select own attack-capable territory.
                if self.game.owner[territory] == me && self.can_attack_from(territory) {
                    self.selected = Some(territory);
                    self.last_human_attack = None;
                } else {
                    self.selected = None;
                }
            }
            Phase::Fortify => {
                if let Some(src) = self.selected {
                    if territory == src {
                        self.selected = None;
                        return;
                    }
                    if self.game.owner[territory] == me
                        && EDGES.binary_search(&(src, territory)).is_ok()
                        && let Some(opt) = self.fortify_pref.to_option(&self.game, src)
                    {
                        self.apply_action(Action::Fortify(src, territory, opt), now, true);
                        return;
                    }
                }
                if self.game.owner[territory] == me && self.can_fortify_from(territory) {
                    self.selected = Some(territory);
                } else {
                    self.selected = None;
                }
            }
            Phase::Ended => {}
        }
    }

    fn execute_human_attack(&mut self, src: usize, dst: usize, now: f64) {
        let dice = self.attack_dice_pref.resolve(self.game.troops[src]);
        let action = Action::Attack(src, dst, dice);
        if !self.game.legal(action) {
            return;
        }
        self.apply_action(action, now, true);
        let captured = self.battle_history.last().is_some_and(|b| b.captured);
        if self.game.phase == Phase::Ended {
            return;
        }
        if captured {
            self.last_human_attack = None;
            // Smooth chaining: follow advancing front if captured territory can keep attacking.
            if self.can_attack_from(dst) {
                self.selected = Some(dst);
            } else if self.can_attack_from(src) {
                self.selected = Some(src);
            } else {
                self.selected = None;
            }
        } else if self.can_attack_from(src) {
            self.selected = Some(src);
            self.last_human_attack = Some((src, dst));
        } else {
            self.selected = None;
            self.last_human_attack = None;
        }
    }

    fn execute_human_blitz(&mut self, src: usize, dst: usize, now: f64) {
        for _ in 0..MAX_BLITZ_STEPS {
            if self.game.phase != Phase::Attack
                || self.game.owner[src] != self.game.active
                || self.game.owner[dst] == self.game.active
                || self.game.troops[src] <= 1
            {
                break;
            }
            let dice = self.attack_dice_pref.resolve(self.game.troops[src]);
            self.apply_action(Action::Attack(src, dst, dice), now, false);
            if self.battle_history.last().is_some_and(|b| b.captured) {
                self.push_popup(dst, "CAPTURED!".to_string(), ACCENT_GOLD, now);
                self.last_human_attack = None;
                if self.can_attack_from(dst) {
                    self.selected = Some(dst);
                } else if self.can_attack_from(src) {
                    self.selected = Some(src);
                } else {
                    self.selected = None;
                }
                return;
            }
        }
        if !self.can_attack_from(src) {
            self.selected = None;
            self.last_human_attack = None;
        }
    }

    fn can_attack_from(&self, territory: usize) -> bool {
        let me = self.game.active;
        self.game.phase == Phase::Attack
            && self.game.owner[territory] == me
            && self.game.troops[territory] > 1
            && EDGES
                .iter()
                .any(|&(a, b)| a == territory && self.game.owner[b] != me)
    }

    fn can_fortify_from(&self, territory: usize) -> bool {
        let me = self.game.active;
        self.game.phase == Phase::Fortify
            && self.game.owner[territory] == me
            && self.game.troops[territory] > 1
            && EDGES
                .iter()
                .any(|&(a, b)| a == territory && self.game.owner[b] == me)
    }

    fn handle_keyboard(&mut self, ctx: &egui::Context, now: f64) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let esc = ctx.input(|i| i.key_pressed(egui::Key::Escape));
        if esc {
            if self.setup_open {
                self.setup_open = false;
            } else {
                self.selected = None;
                self.last_human_attack = None;
            }
            return;
        }
        if self.setup_open || self.game.phase == Phase::Ended {
            return;
        }

        if self.is_human_turn() {
            let e_key = ctx.input(|i| i.key_pressed(egui::Key::E));
            let space_key = ctx.input(|i| i.key_pressed(egui::Key::Space));
            if e_key {
                match self.game.phase {
                    Phase::Attack => self.apply_action(Action::FinishAttack, now, true),
                    Phase::Fortify => self.apply_action(Action::PassFortify, now, true),
                    _ => {}
                }
            }
            if self.game.phase == Phase::Attack {
                if ctx.input(|i| i.key_pressed(egui::Key::Num1)) {
                    self.attack_dice_pref = AttackDicePref::Fixed(1);
                } else if ctx.input(|i| i.key_pressed(egui::Key::Num2)) {
                    self.attack_dice_pref = AttackDicePref::Fixed(2);
                } else if ctx.input(|i| i.key_pressed(egui::Key::Num3)) {
                    self.attack_dice_pref = AttackDicePref::Fixed(3);
                } else if ctx
                    .input(|i| i.key_pressed(egui::Key::A) || i.key_pressed(egui::Key::Num0))
                {
                    self.attack_dice_pref = AttackDicePref::AutoMax;
                } else if space_key && let Some((src, dst)) = self.last_human_attack {
                    self.execute_human_attack(src, dst, now);
                }
            } else if self.game.phase == Phase::Fortify && space_key {
                self.apply_action(Action::PassFortify, now, true);
            }
        } else if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.ai_paused = !self.ai_paused;
            if !self.ai_paused {
                self.next_ai_step_at = now;
            }
        }
    }

    fn draw_top_bar(&mut self, ui: &mut egui::Ui, now: f64) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 10.0;

            ui.label(
                RichText::new("RISKBOT")
                    .size(20.0)
                    .strong()
                    .color(TEXT_PRIMARY),
            );
            ui.separator();

            // Turn & active player pill
            ui.label(
                RichText::new(format!(
                    "Turn {}/{}",
                    self.game.turns + 1,
                    self.game.max_turns
                ))
                .monospace()
                .color(TEXT_MUTED),
            );

            match (self.game.phase, self.game.winner) {
                (Phase::Ended, Some(winner)) => {
                    pill_badge(
                        ui,
                        &format!("WINNER · {}", self.seat_labels[winner]),
                        player_color(winner),
                    );
                }
                (Phase::Ended, None) => {
                    pill_badge(ui, "DRAW · TURN LIMIT", ACCENT_GOLD);
                }
                _ => {
                    let active = self.game.active;
                    let phase_text = match self.game.phase {
                        Phase::Reinforce => format!("REINFORCE (+{})", self.game.pending),
                        Phase::Attack => "ATTACK".to_string(),
                        Phase::Fortify => "FORTIFY".to_string(),
                        Phase::Ended => "ENDED".to_string(),
                    };
                    pill_badge(
                        ui,
                        &format!("{} · {phase_text}", self.seat_labels[active]),
                        player_color(active),
                    );
                }
            }

            // Right-aligned playback & match setup controls
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui
                    .add(
                        egui::Button::new(RichText::new("Match Setup").strong())
                            .fill(if self.setup_open {
                                Color32::from_rgb(56, 94, 126)
                            } else {
                                Color32::from_rgb(35, 54, 72)
                            })
                            .min_size(vec2(100.0, 28.0)),
                    )
                    .clicked()
                {
                    self.setup_open = !self.setup_open;
                    if self.setup_open {
                        self.draft.num_players = self.seats.len();
                        for (i, seat) in self.seats.iter().enumerate() {
                            self.draft.seats[i] = seat.clone();
                        }
                        self.draft.max_turns = self.game.max_turns;
                    }
                }

                if ui
                    .add(egui::Button::new("Restart").min_size(vec2(72.0, 28.0)))
                    .on_hover_text("Start a fresh game with the current roster")
                    .clicked()
                {
                    self.reset_game(None, self.game.max_turns, now);
                }

                ui.separator();

                // AI playback speed selector
                for speed in [
                    AiSpeed::Instant,
                    AiSpeed::Fast,
                    AiSpeed::Normal,
                    AiSpeed::Slow,
                ] {
                    let selected = self.ai_speed == speed;
                    if ui
                        .selectable_label(selected, speed.label())
                        .on_hover_text("AI turn playback speed")
                        .clicked()
                    {
                        self.ai_speed = speed;
                    }
                }
                ui.label(RichText::new("AI:").small().color(TEXT_MUTED));

                if !self.is_human_turn() && self.game.phase != Phase::Ended {
                    if ui
                        .add(egui::Button::new("Step").min_size(vec2(48.0, 24.0)))
                        .on_hover_text("Execute one AI action")
                        .clicked()
                    {
                        self.ai_paused = true;
                        self.step_ai_once(now, true);
                    }
                    let pause_label = if self.ai_paused { "Resume" } else { "Pause" };
                    if ui
                        .add(egui::Button::new(pause_label).min_size(vec2(58.0, 24.0)))
                        .on_hover_text("Space to pause/resume AI playback")
                        .clicked()
                    {
                        self.ai_paused = !self.ai_paused;
                        if !self.ai_paused {
                            self.next_ai_step_at = now;
                        }
                    }
                }
            });
        });
    }

    fn draw_players_card(&self, ui: &mut egui::Ui) {
        egui::Frame::new()
            .fill(CARD_BG)
            .stroke(Stroke::new(1.0, CARD_BORDER))
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::same(10))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(RichText::new("PLAYERS").small().strong().color(TEXT_MUTED));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!("{} seats", self.seats.len()))
                                .small()
                                .color(TEXT_MUTED),
                        );
                    });
                });
                ui.add_space(6.0);

                for (seat, kind) in self.seats.iter().enumerate() {
                    let owned = self.game.owner.iter().filter(|&&o| o == seat).count();
                    let troops: u64 = self
                        .game
                        .owner
                        .iter()
                        .zip(self.game.troops.iter())
                        .filter_map(|(&o, &t)| (o == seat).then_some(t))
                        .sum();
                    let regions = (0..REGIONS.len())
                        .filter(|&r| continent_owner(&self.game, r) == Some(seat))
                        .count();
                    let next_reinf = if owned > 0 {
                        2 + (owned / 4) as u64 + CONTINENT_BONUS * regions as u64
                    } else {
                        0
                    };
                    let eliminated = owned == 0;
                    let is_active =
                        self.game.phase != Phase::Ended && self.game.active == seat;
                    let col = player_color(seat);

                    let row_fill = if is_active {
                        Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 26)
                    } else {
                        Color32::from_rgb(18, 28, 39)
                    };
                    let row_stroke = if is_active {
                        Stroke::new(1.5, col)
                    } else {
                        Stroke::new(1.0, Color32::from_rgb(31, 46, 61))
                    };

                    egui::Frame::new()
                        .fill(row_fill)
                        .stroke(row_stroke)
                        .corner_radius(CornerRadius::same(7))
                        .inner_margin(Margin::symmetric(8, 6))
                        .show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            ui.horizontal(|ui| {
                                let (dot_rect, _) =
                                    ui.allocate_exact_size(vec2(10.0, 10.0), egui::Sense::hover());
                                let dot_col = if eliminated {
                                    Color32::from_rgb(76, 88, 100)
                                } else {
                                    col
                                };
                                ui.painter().circle_filled(dot_rect.center(), 4.5, dot_col);

                                let name_text = if eliminated {
                                    RichText::new(&self.seat_labels[seat])
                                        .strikethrough()
                                        .color(TEXT_MUTED)
                                } else if is_active {
                                    RichText::new(&self.seat_labels[seat]).strong().color(TEXT_PRIMARY)
                                } else {
                                    RichText::new(&self.seat_labels[seat]).color(TEXT_PRIMARY)
                                };
                                ui.label(name_text);

                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    if eliminated {
                                        ui.label(
                                            RichText::new("OUT").small().color(Color32::from_rgb(120, 90, 95)),
                                        );
                                    } else {
                                        ui.label(
                                            RichText::new(format!("+{next_reinf}/t"))
                                                .small()
                                                .monospace()
                                                .color(if regions > 0 { ACCENT_GOLD } else { TEXT_MUTED }),
                                        );
                                        ui.label(
                                            RichText::new(format!("{troops}⚔"))
                                                .small()
                                                .monospace()
                                                .color(TEXT_PRIMARY),
                                        );
                                        ui.label(
                                            RichText::new(format!("{owned}🗺"))
                                                .small()
                                                .monospace()
                                                .color(TEXT_MUTED),
                                        );
                                    }
                                });
                            });

                            if !eliminated {
                                ui.horizontal(|ui| {
                                    let kind_tag = match kind {
                                        SeatKind::Human => "Human".to_string(),
                                        SeatKind::Heuristic => "Heuristic Bot".to_string(),
                                        SeatKind::Random => "Random Bot".to_string(),
                                        SeatKind::Checkpoint { sampling, path, .. } => {
                                            let arch = self
                                                .registry
                                                .model_info(path)
                                                .map(|(h, l)| format!(" · {h}×{l}"))
                                                .unwrap_or_default();
                                            format!("Policy · {}{arch}", sampling.label())
                                        }
                                    };
                                    ui.label(RichText::new(kind_tag).size(10.5).color(TEXT_MUTED));

                                    if let Some(val) = self.seat_values[seat] {
                                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                            let v_col = if val >= 0.15 {
                                                ACCENT_GREEN
                                            } else if val <= -0.15 {
                                                ACCENT_RED
                                            } else {
                                                TEXT_MUTED
                                            };
                                            ui.label(
                                                RichText::new(format!("V {val:+.2}"))
                                                    .size(10.5)
                                                    .monospace()
                                                    .color(v_col),
                                            )
                                            .on_hover_text("Policy critic state-value estimate in [-1, +1]");
                                        });
                                    }
                                });
                            }
                        });
                    ui.add_space(4.0);
                }
            });
    }

    fn draw_action_bar(&mut self, ui: &mut egui::Ui, now: f64) {
        egui::Frame::new()
            .fill(CARD_BG)
            .stroke(Stroke::new(1.0, CARD_BORDER))
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::symmetric(12, 9))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());

                if let Some(err) = &self.last_error {
                    ui.colored_label(ACCENT_RED, format!("Policy error: {err}"));
                }

                match (self.game.phase, self.game.winner) {
                    (Phase::Ended, winner) => {
                        ui.horizontal(|ui| {
                            let (headline, color) = match winner {
                                Some(w) if self.seats[w].is_human() => (
                                    format!("VICTORY! {} conquered all 42 territories!", self.seat_labels[w]),
                                    ACCENT_GOLD,
                                ),
                                Some(w) => (
                                    format!("{} won the match in {} turns.", self.seat_labels[w], self.game.turns),
                                    player_color(w),
                                ),
                                None => (
                                    format!("Draw — turn limit ({}) reached.", self.game.max_turns),
                                    ACCENT_GOLD,
                                ),
                            };
                            ui.label(RichText::new(headline).size(15.0).strong().color(color));
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if ui
                                    .add(
                                        egui::Button::new(RichText::new("Play Again").strong())
                                            .fill(Color32::from_rgb(44, 96, 126))
                                            .min_size(vec2(96.0, 28.0)),
                                    )
                                    .clicked()
                                {
                                    self.reset_game(None, self.game.max_turns, now);
                                }
                                if ui
                                    .add(egui::Button::new("Change Policies").min_size(vec2(114.0, 28.0)))
                                    .clicked()
                                {
                                    self.setup_open = true;
                                }
                            });
                        });
                        return;
                    }
                    _ if !self.is_human_turn() => {
                        ui.horizontal(|ui| {
                            let active = self.game.active;
                            let state_word = if self.ai_paused { "Paused on" } else { "Thinking:" };
                            ui.colored_label(
                                player_color(active),
                                RichText::new(format!("{state_word} {}", self.seat_labels[active])).strong(),
                            );
                            ui.separator();
                            ui.label(RichText::new(&self.last_action_summary).color(TEXT_MUTED));

                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                let skip_label = if self.has_any_human() {
                                    "Skip to My Turn »"
                                } else {
                                    "Next Turn »"
                                };
                                if ui
                                    .add(
                                        egui::Button::new(RichText::new(skip_label).strong())
                                            .fill(Color32::from_rgb(44, 78, 104))
                                            .min_size(vec2(120.0, 28.0)),
                                    )
                                    .clicked()
                                {
                                    self.skip_to_human = true;
                                    self.ai_paused = false;
                                }
                            });
                        });
                        return;
                    }
                    _ => {}
                }

                // Human interactive turn controls
                match self.game.phase {
                    Phase::Reinforce => {
                        ui.horizontal(|ui| {
                            pill_badge(
                                ui,
                                &format!("DEPLOY · {} LEFT", self.game.pending),
                                player_color(self.game.active),
                            );
                            ui.label(
                                RichText::new("Click any of your territories to reinforce (Shift+Click deploys all)")
                                    .color(TEXT_MUTED),
                            );
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                for brush in [ReinforceBrush::All, ReinforceBrush::Five, ReinforceBrush::One] {
                                    let active = self.reinforce_brush == brush;
                                    if ui
                                        .add(
                                            egui::Button::new(RichText::new(brush.label()).strong())
                                                .fill(if active {
                                                    Color32::from_rgb(45, 95, 128)
                                                } else {
                                                    Color32::from_rgb(31, 47, 62)
                                                })
                                                .min_size(vec2(44.0, 26.0)),
                                        )
                                        .clicked()
                                    {
                                        self.reinforce_brush = brush;
                                    }
                                }
                                ui.label(RichText::new("Per click:").small().color(TEXT_MUTED));
                            });
                        });
                    }
                    Phase::Attack => {
                        ui.horizontal(|ui| {
                            pill_badge(ui, "ATTACK PHASE", ACCENT_RED);

                            if let Some(src) = self.selected {
                                ui.label(
                                    RichText::new(format!(
                                        "From {} (#{src}, {} troops) → click glowing enemy",
                                        territory_name(src),
                                        self.game.troops[src]
                                    ))
                                    .color(TEXT_PRIMARY),
                                );
                            } else {
                                ui.label(
                                    RichText::new(
                                        "Click your territory (2+ troops), then click an adjacent enemy",
                                    )
                                    .color(TEXT_MUTED),
                                );
                            }

                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if ui
                                    .add(
                                        egui::Button::new(RichText::new("End Attacks → (E)").strong())
                                            .fill(Color32::from_rgb(56, 86, 62))
                                            .min_size(vec2(132.0, 28.0)),
                                    )
                                    .clicked()
                                {
                                    self.apply_action(Action::FinishAttack, now, true);
                                }

                                // Repeat / Blitz buttons if we just fought an unfinished battle
                                if let Some((src, dst)) = self.last_human_attack
                                    && self.game.owner[src] == self.game.active
                                    && self.game.owner[dst] != self.game.active
                                    && self.game.troops[src] > 1
                                {
                                    if ui
                                        .add(
                                            egui::Button::new(RichText::new("Blitz!").strong())
                                                .fill(Color32::from_rgb(138, 52, 52))
                                                .min_size(vec2(62.0, 28.0)),
                                        )
                                        .on_hover_text(format!(
                                            "Auto-roll {} → {} until captured or out of troops",
                                            territory_name(src),
                                            territory_name(dst)
                                        ))
                                        .clicked()
                                    {
                                        self.execute_human_blitz(src, dst, now);
                                    }
                                    if ui
                                        .add(
                                            egui::Button::new(RichText::new("Roll Again (Space)").strong())
                                                .fill(Color32::from_rgb(112, 58, 52))
                                                .min_size(vec2(126.0, 28.0)),
                                        )
                                        .clicked()
                                    {
                                        self.execute_human_attack(src, dst, now);
                                    }
                                }

                                ui.separator();

                                // Dice selector: Auto / 3 / 2 / 1
                                for (pref, lbl) in [
                                    (AttackDicePref::Fixed(1), "1"),
                                    (AttackDicePref::Fixed(2), "2"),
                                    (AttackDicePref::Fixed(3), "3"),
                                    (AttackDicePref::AutoMax, "Max"),
                                ] {
                                    let active = self.attack_dice_pref == pref;
                                    if ui
                                        .add(
                                            egui::Button::new(RichText::new(lbl).strong())
                                                .fill(if active {
                                                    Color32::from_rgb(156, 60, 58)
                                                } else {
                                                    Color32::from_rgb(31, 47, 62)
                                                })
                                                .min_size(vec2(32.0, 26.0)),
                                        )
                                        .on_hover_text("Attack dice count (keys 1/2/3 or A for Max)")
                                        .clicked()
                                    {
                                        self.attack_dice_pref = pref;
                                    }
                                }
                                ui.label(RichText::new("Dice:").small().color(TEXT_MUTED));
                            });
                        });
                    }
                    Phase::Fortify => {
                        ui.horizontal(|ui| {
                            pill_badge(ui, "FORTIFY PHASE", ACCENT_GREEN);

                            if let Some(src) = self.selected {
                                ui.label(
                                    RichText::new(format!(
                                        "From {} (#{src}, {} movable) → click glowing friendly neighbor",
                                        territory_name(src),
                                        self.game.troops[src].saturating_sub(1)
                                    ))
                                    .color(TEXT_PRIMARY),
                                );
                            } else {
                                ui.label(
                                    RichText::new(
                                        "Optional: click your territory (2+ troops), then an adjacent friendly territory",
                                    )
                                    .color(TEXT_MUTED),
                                );
                            }

                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if ui
                                    .add(
                                        egui::Button::new(RichText::new("End Turn (E)").strong())
                                            .fill(Color32::from_rgb(44, 92, 122))
                                            .min_size(vec2(108.0, 28.0)),
                                    )
                                    .clicked()
                                {
                                    self.apply_action(Action::PassFortify, now, true);
                                }

                                ui.separator();

                                // Show exact troop counts on fortify buttons when a source is selected
                                let src = self.selected;
                                let all_lbl = src
                                    .and_then(|s| self.game.fortify_amount(s, 2))
                                    .map(|n| format!("All ({n})"))
                                    .unwrap_or_else(|| "All".to_string());
                                let half_lbl = src
                                    .and_then(|s| self.game.fortify_amount(s, 1))
                                    .map(|n| format!("Half ({n})"))
                                    .unwrap_or_else(|| "Half".to_string());
                                let one_lbl = "1".to_string();

                                for (pref, lbl) in [
                                    (FortifyPref::All, all_lbl),
                                    (FortifyPref::Half, half_lbl),
                                    (FortifyPref::One, one_lbl),
                                ] {
                                    let active = self.fortify_pref == pref;
                                    if ui
                                        .add(
                                            egui::Button::new(RichText::new(lbl).strong())
                                                .fill(if active {
                                                    Color32::from_rgb(48, 108, 90)
                                                } else {
                                                    Color32::from_rgb(31, 47, 62)
                                                })
                                                .min_size(vec2(48.0, 26.0)),
                                        )
                                        .clicked()
                                    {
                                        self.fortify_pref = pref;
                                    }
                                }
                                ui.label(RichText::new("Move:").small().color(TEXT_MUTED));
                            });
                        });
                    }
                    Phase::Ended => {}
                }
            });
    }

    fn draw_setup_window(&mut self, ctx: &egui::Context, now: f64) {
        if !self.setup_open {
            return;
        }
        let mut open = self.setup_open;
        let mut cancel_clicked = false;
        let mut start_clicked = false;
        let mut rescan_clicked = false;
        let mut load_custom_clicked = false;

        egui::Window::new("Match Setup · Players & Trained Policies")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(640.0)
            .anchor(egui::Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(20, 31, 43))
                    .stroke(Stroke::new(1.5, Color32::from_rgb(62, 94, 122)))
                    .corner_radius(CornerRadius::same(12))
                    .inner_margin(Margin::same(16)),
            )
            .show(ctx, |ui| {
                // Quick presets & player count
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("Players:").strong().color(TEXT_PRIMARY));
                    for n in 2..=MAX_PLAYERS {
                        let active = self.draft.num_players == n;
                        if ui
                            .add(
                                egui::Button::new(RichText::new(format!("{n}P")).strong())
                                    .fill(if active {
                                        Color32::from_rgb(46, 98, 134)
                                    } else {
                                        Color32::from_rgb(32, 48, 64)
                                    })
                                    .min_size(vec2(40.0, 28.0)),
                            )
                            .clicked()
                        {
                            self.draft.num_players = n;
                        }
                    }

                    ui.separator();
                    ui.label(RichText::new("Presets:").color(TEXT_MUTED));
                    if ui.button("1v1 vs Best").clicked() {
                        self.draft.apply_preset(2, true, &self.discovered);
                    }
                    if ui.button("3P Clash").clicked() {
                        self.draft.apply_preset(3, true, &self.discovered);
                    }
                    if ui.button("4P Gauntlet").clicked() {
                        self.draft.apply_preset(4, true, &self.discovered);
                    }
                    if ui.button("AI Spectator").clicked() {
                        self.draft.apply_preset(3, false, &self.discovered);
                    }
                });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(8.0);

                // Per-seat configurator
                let count = self.draft.num_players;
                for seat_idx in 0..count {
                    let col = player_color(seat_idx);
                    egui::Frame::new()
                        .fill(Color32::from_rgb(26, 40, 54))
                        .stroke(Stroke::new(1.0, Color32::from_rgb(42, 62, 82)))
                        .corner_radius(CornerRadius::same(8))
                        .inner_margin(Margin::symmetric(10, 7))
                        .show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            ui.horizontal(|ui| {
                                let (dot, _) =
                                    ui.allocate_exact_size(vec2(12.0, 12.0), egui::Sense::hover());
                                ui.painter().circle_filled(dot.center(), 5.5, col);
                                ui.label(
                                    RichText::new(format!("Seat {}", seat_idx + 1))
                                        .strong()
                                        .color(col),
                                );

                                // Controller category selector
                                let current_cat = seat_category(&self.draft.seats[seat_idx]);
                                egui::ComboBox::from_id_salt(("seat_cat", seat_idx))
                                    .width(126.0)
                                    .selected_text(current_cat)
                                    .show_ui(ui, |ui| {
                                        if ui
                                            .selectable_label(current_cat == "Human", "Human")
                                            .clicked()
                                        {
                                            self.draft.seats[seat_idx] = SeatKind::Human;
                                        }
                                        #[cfg(not(target_arch = "wasm32"))]
                                        if ui
                                            .selectable_label(
                                                current_cat == "Policy (.mpk)",
                                                "Policy (.mpk)",
                                            )
                                            .clicked()
                                        {
                                            self.draft.seats[seat_idx] =
                                                default_checkpoint_seat(&self.discovered, seat_idx);
                                        }
                                        if ui
                                            .selectable_label(
                                                current_cat == "Heuristic Bot",
                                                "Heuristic Bot",
                                            )
                                            .clicked()
                                        {
                                            self.draft.seats[seat_idx] = SeatKind::Heuristic;
                                        }
                                        if ui
                                            .selectable_label(
                                                current_cat == "Random Bot",
                                                "Random Bot",
                                            )
                                            .clicked()
                                        {
                                            self.draft.seats[seat_idx] = SeatKind::Random;
                                        }
                                    });

                                // If Policy (.mpk), choose checkpoint + sampling mode
                                if let SeatKind::Checkpoint {
                                    path,
                                    label,
                                    sampling,
                                } = &mut self.draft.seats[seat_idx]
                                {
                                    let selected_text = if label.is_empty() {
                                        path.display().to_string()
                                    } else {
                                        label.clone()
                                    };
                                    egui::ComboBox::from_id_salt(("seat_ckpt", seat_idx))
                                        .width(260.0)
                                        .selected_text(selected_text)
                                        .show_ui(ui, |ui| {
                                            if self.discovered.is_empty() {
                                                ui.label(
                                                    RichText::new(
                                                        "No checkpoints found — load one below",
                                                    )
                                                    .weak(),
                                                );
                                            }
                                            for ckpt in &self.discovered {
                                                let chosen = *path == ckpt.path;
                                                if ui
                                                    .selectable_label(chosen, &ckpt.display_name)
                                                    .clicked()
                                                {
                                                    *path = ckpt.path.clone();
                                                    *label = ckpt.display_name.clone();
                                                }
                                            }
                                        });

                                    egui::ComboBox::from_id_salt(("seat_samp", seat_idx))
                                        .width(96.0)
                                        .selected_text(sampling.label())
                                        .show_ui(ui, |ui| {
                                            for mode in [
                                                PolicySampling::Balanced,
                                                PolicySampling::Greedy,
                                                PolicySampling::Stochastic,
                                            ] {
                                                ui.selectable_value(sampling, mode, mode.label());
                                            }
                                        });
                                }
                            });
                        });
                    ui.add_space(4.0);
                }

                // Custom checkpoint loader & rescan
                #[cfg(not(target_arch = "wasm32"))]
                {
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Checkpoint:").small().color(TEXT_MUTED));
                        let resp = ui.add(
                            egui::TextEdit::singleline(&mut self.draft.custom_ckpt_input)
                                .desired_width(340.0)
                                .hint_text("path/to/checkpoint.mpk"),
                        );
                        if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            load_custom_clicked = true;
                        }
                        if ui.button("Load .mpk").clicked() {
                            load_custom_clicked = true;
                        }
                        if ui
                            .button("Rescan runs/")
                            .on_hover_text("Pick up newly saved checkpoints from active training")
                            .clicked()
                        {
                            rescan_clicked = true;
                        }
                    });
                }

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Max turns:").color(TEXT_MUTED));
                    ui.add(
                        egui::DragValue::new(&mut self.draft.max_turns)
                            .range(20..=2000)
                            .speed(10),
                    );
                    ui.separator();
                    ui.label(RichText::new("Seed (optional):").color(TEXT_MUTED));
                    ui.add(
                        egui::TextEdit::singleline(&mut self.draft.seed_input)
                            .desired_width(130.0)
                            .hint_text("random"),
                    );
                });

                if let Some((msg, is_err)) = &self.draft.status_message {
                    ui.add_space(6.0);
                    ui.colored_label(if *is_err { ACCENT_RED } else { ACCENT_GREEN }, msg);
                }

                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui
                            .add(
                                egui::Button::new(RichText::new("Start Match").size(14.0).strong())
                                    .fill(Color32::from_rgb(44, 114, 152))
                                    .min_size(vec2(124.0, 34.0)),
                            )
                            .clicked()
                        {
                            start_clicked = true;
                        }
                        if ui
                            .add(egui::Button::new("Cancel").min_size(vec2(76.0, 34.0)))
                            .clicked()
                        {
                            cancel_clicked = true;
                        }
                    });
                });
            });

        self.setup_open = open && !cancel_clicked;
        if rescan_clicked {
            self.rescan_checkpoints();
        }
        if load_custom_clicked {
            self.add_custom_checkpoint();
        }
        if start_clicked {
            self.start_match_from_draft(now);
        }
    }
}

impl eframe::App for Visualizer {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let now = ctx.input(|i| i.time);

        // Expire old floating popups
        self.popups.retain(|p| now - p.timestamp < 1.5);

        self.handle_keyboard(&ctx, now);
        self.tick_ai(&ctx, now);

        // Top bar
        egui::Panel::top("riskbot_top_bar")
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(14, 22, 31))
                    .stroke(Stroke::new(1.0, Color32::from_rgb(31, 47, 63)))
                    .inner_margin(Margin::symmetric(14, 8)),
            )
            .show(ui, |ui| self.draw_top_bar(ui, now));

        // Right sidebar: Players + Live Dice Arena + Battle History
        let mut hovered_edge = None;
        egui::Panel::right("riskbot_sidebar")
            .resizable(true)
            .default_size(338.0)
            .min_size(290.0)
            .max_size(440.0)
            .frame(
                egui::Frame::new()
                    .fill(PANEL_BG)
                    .stroke(Stroke::new(1.0, Color32::from_rgb(31, 47, 63)))
                    .inner_margin(Margin::same(10)),
            )
            .show(ui, |ui| {
                self.draw_players_card(ui);
                ui.add_space(8.0);

                let spot = battles::draw_spotlight(
                    ui,
                    self.battle_history.last(),
                    &self.seat_labels,
                    now,
                    self.ai_speed.animate_dice(),
                );
                if spot.hovered_edge.is_some() {
                    hovered_edge = spot.hovered_edge;
                }

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("BATTLE LOG")
                            .small()
                            .strong()
                            .color(TEXT_MUTED),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!("{} rolls", self.battle_history.len()))
                                .small()
                                .color(TEXT_MUTED),
                        );
                    });
                });
                ui.add_space(4.0);

                let hist = ui
                    .push_id(self.game_number, |ui| {
                        battles::draw_history(ui, &self.battle_history, &self.seat_labels, now)
                    })
                    .inner;
                if hist.hovered_edge.is_some() {
                    hovered_edge = hist.hovered_edge;
                }
            });

        // Bottom human/AI action bar
        egui::Panel::bottom("riskbot_action_bar")
            .frame(
                egui::Frame::new()
                    .fill(PANEL_BG)
                    .inner_margin(Margin::symmetric(12, 8)),
            )
            .show(ui, |ui| self.draw_action_bar(ui, now));

        // Central map viewport
        let (selectable, valid_targets, targeting_attack) = self.compute_highlights();
        let shift_held = ctx.input(|i| i.modifiers.shift);
        let mut interaction = state::MapInteraction::default();

        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(PANEL_BG)
                    .inner_margin(Margin::symmetric(10, 8)),
            )
            .show(ui, |ui| {
                let overlay = MapOverlayState {
                    game: &self.game,
                    seat_labels: &self.seat_labels,
                    changed: &self.changed,
                    selected: self.selected,
                    selectable,
                    valid_targets,
                    targeting_attack,
                    human_turn: self.is_human_turn(),
                    active_battle: self.battle_history.last(),
                    highlighted_edge: hovered_edge,
                    popups: &self.popups,
                    now,
                };
                interaction = self.map.draw(&overlay, ui);
            });

        if interaction.right_clicked.is_some() {
            self.selected = None;
            self.last_human_attack = None;
        } else if let Some(territory) = interaction.clicked {
            self.handle_territory_click(territory, shift_held, now);
        }

        self.draw_setup_window(&ctx, now);
    }
}

fn configure_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = PANEL_BG;
    visuals.window_fill = Color32::from_rgb(20, 31, 43);
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(35, 52, 68);
    visuals.widgets.inactive.weak_bg_fill = visuals.widgets.inactive.bg_fill;
    visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(48, 71, 92);
    visuals.widgets.active.weak_bg_fill = Color32::from_rgb(58, 88, 114);
    visuals.selection.bg_fill = Color32::from_rgb(44, 102, 138);
    visuals.selection.stroke = Stroke::new(1.0, ACCENT_CYAN);
    ctx.set_theme(egui::Theme::Dark);
    ctx.set_visuals(visuals);
}

fn pill_badge(ui: &mut egui::Ui, text: &str, accent: Color32) {
    let fill = Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 34);
    egui::Frame::new()
        .fill(fill)
        .stroke(Stroke::new(1.2, accent))
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::symmetric(10, 3))
        .show(ui, |ui| {
            ui.label(RichText::new(text).size(12.0).strong().color(accent));
        });
}

fn seat_category(kind: &SeatKind) -> &'static str {
    match kind {
        SeatKind::Human => "Human",
        SeatKind::Checkpoint { .. } => "Policy (.mpk)",
        SeatKind::Heuristic => "Heuristic Bot",
        SeatKind::Random => "Random Bot",
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn default_checkpoint_seat(checkpoints: &[DiscoveredCheckpoint], seat_idx: usize) -> SeatKind {
    if let Some(ckpt) = checkpoints
        .get(seat_idx.saturating_sub(1))
        .or_else(|| checkpoints.first())
    {
        SeatKind::Checkpoint {
            path: ckpt.path.clone(),
            label: ckpt.display_name.clone(),
            sampling: PolicySampling::Balanced,
        }
    } else {
        SeatKind::Checkpoint {
            path: PathBuf::from("runs/overnight/best.mpk"),
            label: "overnight / best".to_string(),
            sampling: PolicySampling::Balanced,
        }
    }
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
            _ => {
                let short = seat.short_label();
                // Truncate overly long display names neatly for compact pills
                let clean: String = short.chars().take(24).collect();
                format!("P{} · {clean}", i + 1)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attack_dice_pref_respects_attacker_troops() {
        assert_eq!(AttackDicePref::AutoMax.resolve(2), 1);
        assert_eq!(AttackDicePref::AutoMax.resolve(3), 2);
        assert_eq!(AttackDicePref::AutoMax.resolve(10), 3);
        assert_eq!(AttackDicePref::Fixed(3).resolve(2), 1);
        assert_eq!(AttackDicePref::Fixed(2).resolve(5), 2);
        assert_eq!(AttackDicePref::Fixed(1).resolve(5), 1);
    }

    #[test]
    fn fortify_pref_falls_back_to_legal_options() {
        let mut game = Game::new(2, 50, Some(7));
        game.active = 0;
        game.owner[0] = 0;
        game.owner[1] = 0;
        game.phase = Phase::Fortify;

        // Only 1 movable troop -> only option 0 is legal
        game.troops[0] = 2;
        assert_eq!(FortifyPref::All.to_option(&game, 0), Some(0));
        assert_eq!(FortifyPref::Half.to_option(&game, 0), Some(0));

        // 5 movable troops -> 0 (1), 1 (2), 2 (5) are all legal
        game.troops[0] = 6;
        assert_eq!(FortifyPref::One.to_option(&game, 0), Some(0));
        assert_eq!(FortifyPref::Half.to_option(&game, 0), Some(1));
        assert_eq!(FortifyPref::All.to_option(&game, 0), Some(2));
    }

    #[test]
    fn seat_labels_handle_single_and_multiple_humans() {
        let one_human = build_seat_labels(&[SeatKind::Human, SeatKind::Heuristic]);
        assert_eq!(one_human[0], "You");
        assert!(one_human[1].contains("Heuristic"));

        let two_humans = build_seat_labels(&[SeatKind::Human, SeatKind::Human, SeatKind::Random]);
        assert_eq!(two_humans[0], "Human 1");
        assert_eq!(two_humans[1], "Human 2");
        assert!(two_humans[2].contains("Random"));
    }

    #[test]
    fn match_setup_presets_support_variable_player_counts() {
        let mut draft = MatchSetupDraft::default_with_checkpoints(&[]);
        draft.apply_preset(5, true, &[]);
        assert_eq!(draft.num_players, 5);
        assert!(draft.seats[0].is_human());
        assert!(draft.seats[1..5].iter().all(|s| !s.is_human()));

        draft.apply_preset(4, false, &[]);
        assert_eq!(draft.num_players, 4);
        assert!(draft.seats[..4].iter().all(|s| !s.is_human()));
    }
}
