//! Shared visual constants, naming, and interaction state for the Riskbot visualizer.

use crate::game::{BattleOutcome, Game, MAX_PLAYERS, N, REGIONS};
use eframe::egui::Color32;

/// Distinct, high-contrast seat colors for up to 6 players on dark maritime slate.
pub const PLAYER_COLORS: [Color32; MAX_PLAYERS] = [
    Color32::from_rgb(79, 195, 247),  // 0: Sky Cyan
    Color32::from_rgb(255, 118, 97),  // 1: Coral Red
    Color32::from_rgb(138, 214, 114), // 2: Emerald Sage
    Color32::from_rgb(245, 197, 66),  // 3: Amber Gold
    Color32::from_rgb(186, 140, 250), // 4: Violet Orchid
    Color32::from_rgb(244, 114, 182), // 5: Rose Pink
];

pub const CONTINENT_NAMES: [&str; 6] = [
    "NORTH AMERICA",
    "EUROPE",
    "ASIA",
    "SOUTH AMERICA",
    "AFRICA",
    "OCEANIA",
];

pub const CONTINENT_SHORT_NAMES: [&str; 6] = [
    "North America",
    "Europe",
    "Asia",
    "South America",
    "Africa",
    "Oceania",
];

/// Bonus troops awarded per turn for holding an entire continent.
pub const CONTINENT_BONUS: u64 = 2;

/// Evocative names for all 42 territories in simulator index order (7 per continent: 6 outer + 1 inner hub).
pub const TERRITORY_NAMES: [&str; N] = [
    // North America (0..7)
    "Yukon",
    "Quebec",
    "Dixie",
    "Veracruz",
    "Sierra",
    "Alaska",
    "Heartland",
    // Europe (7..14)
    "Nordics",
    "Baltics",
    "Balkans",
    "Iberia",
    "Gaul",
    "Albion",
    "Bohemia",
    // Asia (14..21)
    "Siberia",
    "Kamchatka",
    "Cathay",
    "Deccan",
    "Persia",
    "Urals",
    "Steppe",
    // South America (21..28)
    "Darien",
    "Bahia",
    "Pampas",
    "Patagonia",
    "Andes",
    "Peru",
    "Amazon",
    // Africa (28..35)
    "Maghreb",
    "Horn",
    "Zambezi",
    "Cape",
    "Guinea",
    "Sahara",
    "Congo",
    // Oceania (35..42)
    "Papua",
    "Coral Coast",
    "Tasman",
    "Outback",
    "Kimberley",
    "Pilbara",
    "Uluru",
];

pub fn player_color(seat: usize) -> Color32 {
    PLAYER_COLORS[seat % MAX_PLAYERS]
}

pub fn territory_name(id: usize) -> &'static str {
    TERRITORY_NAMES.get(id).copied().unwrap_or("Unknown")
}

pub fn continent_of(territory: usize) -> usize {
    territory / 7
}

/// Returns `Some(seat)` if one player controls all 7 territories of `region_idx`.
pub fn continent_owner(game: &Game, region_idx: usize) -> Option<usize> {
    let region = REGIONS.get(region_idx)?;
    let first = game.owner[region[0]];
    region
        .iter()
        .all(|&t| game.owner[t] == first)
        .then_some(first)
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
