//! Ego-centric board feature encoding, graph topology, and potential shaping.

use crate::game::{ACTION_COUNT, Action, EDGES, Game, MAX_PLAYERS, N, Phase, REGIONS};

pub const NUM_EDGES: usize = EDGES.len();
pub const NODE_FEAT_DIM: usize = 22;
pub const GLOBAL_FEAT_DIM: usize = 14;

/// Precomputed static graph topology of the 42-territory Risk board.
#[derive(Clone, Debug)]
pub struct BoardTopology {
    /// Row-normalized adjacency matrix (without self-loops), flattened `[N * N]`.
    pub adj_norm: Vec<f32>,
    /// Source territory index for each directed edge in `EDGES`.
    pub edge_src: Vec<i32>,
    /// Destination territory index for each directed edge in `EDGES`.
    pub edge_dst: Vec<i32>,
    /// Neighbors list per territory for fast CPU feature encoding.
    pub neighbors: [Vec<usize>; N],
    /// Region index (`0..6`) for each territory.
    pub territory_region: [usize; N],
}

impl BoardTopology {
    pub fn new() -> Self {
        let mut neighbors: [Vec<usize>; N] = std::array::from_fn(|_| Vec::new());
        let mut edge_src = Vec::with_capacity(NUM_EDGES);
        let mut edge_dst = Vec::with_capacity(NUM_EDGES);
        for &(a, b) in &EDGES {
            neighbors[a].push(b);
            edge_src.push(a as i32);
            edge_dst.push(b as i32);
        }

        let mut adj_norm = vec![0.0f32; N * N];
        for (a, nbrs) in neighbors.iter().enumerate() {
            let inv_deg = 1.0 / (nbrs.len().max(1) as f32);
            for &b in nbrs {
                adj_norm[a * N + b] = inv_deg;
            }
        }

        let mut territory_region = [0usize; N];
        for (r, region) in REGIONS.iter().enumerate() {
            for &t in region {
                territory_region[t] = r;
            }
        }

        Self {
            adj_norm,
            edge_src,
            edge_dst,
            neighbors,
            territory_region,
        }
    }
}

impl Default for BoardTopology {
    fn default() -> Self {
        Self::new()
    }
}

/// Encoded ego-centric observation for a single decision step.
#[derive(Clone, Debug)]
pub struct EncodedObs {
    /// Flattened `[N * NODE_FEAT_DIM]`.
    pub nodes: [f32; N * NODE_FEAT_DIM],
    /// Global state features `[GLOBAL_FEAT_DIM]`.
    pub global: [f32; GLOBAL_FEAT_DIM],
    /// Legal action mask `[ACTION_COUNT]`.
    pub mask: [bool; ACTION_COUNT],
}

/// Encode `game` from the perspective of `player` (canonicalized ego-centric view).
pub fn encode_observation(game: &Game, player: usize, topo: &BoardTopology) -> EncodedObs {
    let mut owned_counts = [0usize; MAX_PLAYERS];
    let mut troop_counts = [0u64; MAX_PLAYERS];
    let mut total_troops = 0u64;

    for i in 0..N {
        let o = game.owner[i];
        owned_counts[o] += 1;
        troop_counts[o] += game.troops[i];
        total_troops += game.troops[i];
    }
    let total_troops_f = (total_troops.max(1)) as f32;

    // Per-region ownership counts for every seat.
    let mut region_counts = [[0usize; MAX_PLAYERS]; REGIONS.len()];
    let mut regions_controlled = [0usize; MAX_PLAYERS];
    for (r, region) in REGIONS.iter().enumerate() {
        for &t in region {
            region_counts[r][game.owner[t]] += 1;
        }
        for p in 0..game.players {
            if region_counts[r][p] == region.len() {
                regions_controlled[p] += 1;
            }
        }
    }

    let phase_reinforce = if game.phase == Phase::Reinforce {
        1.0
    } else {
        0.0
    };
    let phase_attack = if game.phase == Phase::Attack {
        1.0
    } else {
        0.0
    };
    let phase_fortify = if game.phase == Phase::Fortify {
        1.0
    } else {
        0.0
    };

    let mut nodes = [0.0f32; N * NODE_FEAT_DIM];
    let region_size = REGIONS[0].len() as f32;

    for t in 0..N {
        let base = t * NODE_FEAT_DIM;
        let owner = game.owner[t];
        let is_mine = owner == player;
        let troops = game.troops[t];
        let rel_seat = (owner + game.players - player) % game.players;

        let mut has_enemy_nbr = false;
        let mut adj_enemy_troops = 0u64;
        let mut adj_friendly_troops = 0u64;
        let mut max_enemy_nbr = 0u64;

        for &nbr in &topo.neighbors[t] {
            let nbr_troops = game.troops[nbr];
            if game.owner[nbr] == owner {
                adj_friendly_troops += nbr_troops;
            } else {
                has_enemy_nbr = true;
                adj_enemy_troops += nbr_troops;
                if nbr_troops > max_enemy_nbr {
                    max_enemy_nbr = nbr_troops;
                }
            }
        }

        let r = topo.territory_region[t];
        let my_reg_frac = (region_counts[r][player] as f32) / region_size;
        let owner_reg_frac = (region_counts[r][owner] as f32) / region_size;

        nodes[base] = if is_mine { 1.0 } else { 0.0 };
        nodes[base + 1] = if is_mine { 0.0 } else { 1.0 };
        nodes[base + 2 + rel_seat] = 1.0;
        nodes[base + 8] = (troops as f32).ln_1p() / 5.0;
        nodes[base + 9] = (troops as f32) / total_troops_f;
        nodes[base + 10] = if is_mine && troops > 1 && has_enemy_nbr {
            1.0
        } else {
            0.0
        };
        nodes[base + 11] = if has_enemy_nbr { 1.0 } else { 0.0 };
        nodes[base + 12] = (adj_enemy_troops as f32).ln_1p() / 5.0;
        nodes[base + 13] = (adj_friendly_troops as f32).ln_1p() / 5.0;
        let diff = (troops as f32) - (max_enemy_nbr as f32);
        nodes[base + 14] = (diff / 5.0).tanh();
        nodes[base + 15] = my_reg_frac;
        nodes[base + 16] = if region_counts[r][player] == REGIONS[0].len() {
            1.0
        } else {
            0.0
        };
        nodes[base + 17] = owner_reg_frac;
        nodes[base + 18] = if region_counts[r][owner] == REGIONS[0].len() {
            1.0
        } else {
            0.0
        };
        nodes[base + 19] = phase_reinforce;
        nodes[base + 20] = phase_attack;
        nodes[base + 21] = phase_fortify;
    }

    let alive_count = (0..game.players).filter(|&p| owned_counts[p] > 0).count();
    let my_owned = owned_counts[player];
    let my_regions = regions_controlled[player];
    let my_next_reinf = if my_owned > 0 {
        2 + my_owned / 4 + 2 * my_regions
    } else {
        0
    };

    let mut max_enemy_owned = 0usize;
    let mut max_enemy_troops = 0u64;
    let mut max_enemy_regions = 0usize;
    for p in 0..game.players {
        if p == player {
            continue;
        }
        max_enemy_owned = max_enemy_owned.max(owned_counts[p]);
        max_enemy_troops = max_enemy_troops.max(troop_counts[p]);
        max_enemy_regions = max_enemy_regions.max(regions_controlled[p]);
    }

    let global = [
        phase_reinforce,
        phase_attack,
        phase_fortify,
        (game.pending as f32).ln_1p() / 4.0,
        (game.turns as f32) / (game.max_turns.max(1) as f32),
        (alive_count as f32) / (game.players as f32),
        (my_owned as f32) / (N as f32),
        (troop_counts[player] as f32) / total_troops_f,
        (my_regions as f32) / (REGIONS.len() as f32),
        (my_next_reinf as f32) / 24.0,
        (max_enemy_owned as f32) / (N as f32),
        (max_enemy_troops as f32) / total_troops_f,
        (max_enemy_regions as f32) / (REGIONS.len() as f32),
        (game.players as f32) / (MAX_PLAYERS as f32),
    ];

    let mask = legal_action_mask(game, player);

    EncodedObs {
        nodes,
        global,
        mask,
    }
}

/// Fast legal action mask computation for `player`.
pub(crate) fn legal_action_mask(game: &Game, player: usize) -> [bool; ACTION_COUNT] {
    let mut mask = [false; ACTION_COUNT];
    if game.phase == Phase::Ended || game.active != player {
        return mask;
    }
    match game.phase {
        Phase::Reinforce => {
            for (t, allowed) in mask.iter_mut().enumerate().take(N) {
                *allowed = game.owner[t] == player;
            }
        }
        Phase::Attack => {
            for (edge_idx, &(a, b)) in EDGES.iter().enumerate() {
                if game.owner[a] == player && game.owner[b] != player {
                    let troops = game.troops[a];
                    let base = N + 3 * edge_idx;
                    if troops > 1 {
                        mask[base] = true;
                    }
                    if troops > 2 {
                        mask[base + 1] = true;
                    }
                    if troops > 3 {
                        mask[base + 2] = true;
                    }
                }
            }
            mask[ACTION_COUNT - 2] = true; // FinishAttack
        }
        Phase::Fortify => {
            let fortify_start = N + 3 * EDGES.len();
            for (edge_idx, &(a, b)) in EDGES.iter().enumerate() {
                if game.owner[a] == player && game.owner[b] == player {
                    let movable = game.troops[a].saturating_sub(1);
                    let base = fortify_start + 3 * edge_idx;
                    if movable >= 1 {
                        mask[base] = true;
                    }
                    if movable / 2 > 1 {
                        mask[base + 1] = true;
                    }
                    if movable > 1 {
                        mask[base + 2] = true;
                    }
                }
            }
            mask[ACTION_COUNT - 1] = true; // PassFortify
        }
        Phase::Ended => {}
    }
    debug_assert!((0..ACTION_COUNT).all(|id| {
        let decoded = Action::decode(id).unwrap();
        mask[id] == game.legal(decoded)
    }));
    mask
}

/// Bounded board-control potential in `[0.0, 1.0]` for `seat`, used for potential-based shaping.
/// Equals `1.0` when `seat` controls the entire board and `0.0` when `seat` is eliminated.
#[cfg_attr(not(any(feature = "train", feature = "visualizer")), allow(dead_code))]
pub(crate) fn board_potential(game: &Game, seat: usize) -> f32 {
    let mut owned = 0usize;
    let mut my_troops = 0u64;
    let mut total_troops = 0u64;
    for i in 0..N {
        total_troops += game.troops[i];
        if game.owner[i] == seat {
            owned += 1;
            my_troops += game.troops[i];
        }
    }
    if owned == 0 {
        return 0.0;
    }
    if owned == N {
        return 1.0;
    }
    let regions = REGIONS
        .iter()
        .filter(|region| region.iter().all(|&t| game.owner[t] == seat))
        .count();
    let territory_share = (owned as f32) / (N as f32);
    let region_share = (regions as f32) / (REGIONS.len() as f32);
    let troop_share = (my_troops as f32) / (total_troops.max(1) as f32);
    0.45 * territory_share + 0.35 * region_share + 0.20 * troop_share
}
