//! Shared, UI-agnostic Riskbot board metadata (territory names, continents, seat colors).

use crate::game::{Game, MAX_PLAYERS, N, REGIONS};

/// Distinct, high-contrast seat RGB colors for up to 6 players on dark maritime slate.
pub const PLAYER_COLORS_RGB: [[u8; 3]; MAX_PLAYERS] = [
    [79, 195, 247],  // 0: Sky Cyan
    [255, 118, 97],  // 1: Coral Red
    [138, 214, 114], // 2: Emerald Sage
    [245, 197, 66],  // 3: Amber Gold
    [186, 140, 250], // 4: Violet Orchid
    [244, 114, 182], // 5: Rose Pink
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

#[allow(dead_code)]
pub fn player_color_hex(seat: usize) -> String {
    let [r, g, b] = PLAYER_COLORS_RGB[seat % MAX_PLAYERS];
    format!("#{r:02x}{g:02x}{b:02x}")
}

#[allow(dead_code)]
pub fn territory_name(id: usize) -> &'static str {
    TERRITORY_NAMES.get(id).copied().unwrap_or("Unknown")
}

#[allow(dead_code)]
pub fn continent_of(territory: usize) -> usize {
    territory / 7
}

/// Returns `Some(seat)` if one player controls all 7 territories of `region_idx`.
#[allow(dead_code)]
pub fn continent_owner(game: &Game, region_idx: usize) -> Option<usize> {
    let region = REGIONS.get(region_idx)?;
    let first = game.owner[region[0]];
    region
        .iter()
        .all(|&t| game.owner[t] == first)
        .then_some(first)
}
