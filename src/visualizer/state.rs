//! Shared visual constants, naming, and interaction state for the Riskbot visualizer.

pub use crate::board_meta::{
    CONTINENT_BONUS, CONTINENT_NAMES, CONTINENT_SHORT_NAMES, PLAYER_COLORS_RGB, continent_of,
    continent_owner, territory_name,
};
use crate::game::{BattleOutcome, Game, MAX_PLAYERS, N};
use eframe::egui::Color32;

const fn rgb([r, g, b]: [u8; 3]) -> Color32 {
    Color32::from_rgb(r, g, b)
}

/// Distinct, high-contrast seat colors for up to 6 players on dark maritime slate.
pub const PLAYER_COLORS: [Color32; MAX_PLAYERS] = [
    rgb(PLAYER_COLORS_RGB[0]),
    rgb(PLAYER_COLORS_RGB[1]),
    rgb(PLAYER_COLORS_RGB[2]),
    rgb(PLAYER_COLORS_RGB[3]),
    rgb(PLAYER_COLORS_RGB[4]),
    rgb(PLAYER_COLORS_RGB[5]),
];

pub fn player_color(seat: usize) -> Color32 {
    PLAYER_COLORS[seat % MAX_PLAYERS]
}

/// Full record of a resolved dice battle, used for both animated spotlight and history log.
#[derive(Clone, Debug)]
pub struct BattleRecord {
    pub id: u64,
    pub turn: u32,
    pub attacker: usize,
    pub defender: usize,
    pub source: usize,
    pub target: usize,
    pub attack_dice_count: usize,
    pub defense_dice_count: usize,
    pub attack_rolls: [u8; 3],
    pub defense_rolls: [u8; 2],
    pub attacker_losses: u64,
    pub defender_losses: u64,
    pub captured: bool,
    pub attacker_troops_after: u64,
    pub defender_troops_after: u64,
    /// Monotonic egui timestamp (`ctx.input(|i| i.time)`) when this battle occurred.
    pub timestamp: f64,
}

impl BattleRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn from_outcome(
        id: u64,
        turn: u32,
        attacker: usize,
        defender: usize,
        source: usize,
        target: usize,
        outcome: BattleOutcome,
        attacker_troops_after: u64,
        defender_troops_after: u64,
        timestamp: f64,
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
            attack_rolls: outcome.attack,
            defense_rolls: outcome.defense,
            attacker_losses: outcome.attacker_losses,
            defender_losses: outcome.defender_losses,
            captured: outcome.captured,
            attacker_troops_after,
            defender_troops_after,
            timestamp,
        }
    }
}

/// Ephemeral floating troop-change badge rendered over a territory on the map.
#[derive(Clone, Debug)]
pub struct TroopDeltaPopup {
    pub territory: usize,
    pub text: String,
    pub color: Color32,
    pub timestamp: f64,
}

/// Visual and interactive overlay state passed into `Map::draw`.
pub struct MapOverlayState<'a> {
    pub game: &'a Game,
    pub seat_labels: &'a [String],
    pub changed: &'a [bool; N],
    pub selected: Option<usize>,
    pub selectable: [bool; N],
    pub valid_targets: [bool; N],
    pub targeting_attack: bool,
    pub human_turn: bool,
    pub active_battle: Option<&'a BattleRecord>,
    pub highlighted_edge: Option<(usize, usize)>,
    pub popups: &'a [TroopDeltaPopup],
    pub now: f64,
}

/// Pointer interaction returned by `Map::draw`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MapInteraction {
    #[allow(dead_code)]
    pub hovered: Option<usize>,
    pub clicked: Option<usize>,
    pub right_clicked: Option<usize>,
}

/// Feedback returned by the battle/dice panel.
#[derive(Clone, Copy, Debug, Default)]
pub struct BattlePanelOutput {
    pub hovered_edge: Option<(usize, usize)>,
    #[allow(dead_code)]
    pub animating: bool,
}
