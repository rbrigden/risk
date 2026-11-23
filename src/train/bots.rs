//! Baseline opponents (random and greedy heuristic) for evaluation and curriculum play.

use crate::game::{ACTION_COUNT, Action, EDGES, Game, N, Phase, REGIONS};
use crate::train::features::BoardTopology;
use rand::Rng;

/// Choose a uniformly random legal action.
pub fn random_action(game: &Game, rng: &mut impl Rng) -> usize {
    let mut chosen = None;
    let mut count = 0usize;
    for id in 0..ACTION_COUNT {
        if let Some(act) = Action::decode(id)
            && game.legal(act)
        {
            count += 1;
            if rng.random_range(0..count) == 0 {
                chosen = Some(id);
            }
        }
    }
    chosen.expect("live game must have at least one legal action")
}

/// Fast tactical heuristic bot:
/// - Reinforce: prioritizes border territories in nearly-complete continents and pressured borders.
/// - Attack: attacks with maximum dice when holding a troop advantage or completing a continent;
///   finishes attacking once no favorable attacks remain.
/// - Fortify: moves excess interior troops toward a friendly border territory.
pub fn heuristic_action(game: &Game, topo: &BoardTopology, rng: &mut impl Rng) -> usize {
    let me = game.active;
    match game.phase {
        Phase::Reinforce => {
            let mut region_owned = [0usize; REGIONS.len()];
            for (r, reg) in REGIONS.iter().enumerate() {
                region_owned[r] = reg.iter().filter(|&&t| game.owner[t] == me).count();
            }
            let mut best_t = None;
            let mut best_score = f32::NEG_INFINITY;
            for t in 0..N {
                if game.owner[t] != me {
                    continue;
                }
                let mut enemy_nbr_troops = 0u64;
                let mut enemy_nbrs = 0usize;
                let mut weak_enemy = false;
                for &nbr in &topo.neighbors[t] {
                    if game.owner[nbr] != me {
                        enemy_nbrs += 1;
                        enemy_nbr_troops += game.troops[nbr];
                        if game.troops[nbr] <= 2 {
                            weak_enemy = true;
                        }
                    }
                }
                if enemy_nbrs == 0 {
                    // Low priority for interior territories.
                    let score = -10.0 + rng.random_range(0.0..0.05);
                    if score > best_score {
                        best_score = score;
                        best_t = Some(t);
                    }
                    continue;
                }
                let r = topo.territory_region[t];
                let reg_bonus = if region_owned[r] == REGIONS[0].len() {
                    3.0 // Defend owned continent border.
                } else {
                    (region_owned[r] as f32) * 0.7
                };
                let threat = (enemy_nbr_troops as f32).min(12.0) * 0.4;
                let stack_bonus = if game.troops[t] >= 2 && game.troops[t] <= 8 {
                    1.5
                } else {
                    0.0
                };
                let weak_bonus = if weak_enemy { 1.0 } else { 0.0 };
                let score =
                    reg_bonus + threat + stack_bonus + weak_bonus + rng.random_range(0.0..0.2);
                if score > best_score {
                    best_score = score;
                    best_t = Some(t);
                }
            }
            Action::Place(best_t.expect("active player owns a territory"))
                .encode()
                .unwrap()
        }
        Phase::Attack => {
            let mut region_owned = [0usize; REGIONS.len()];
            for (r, reg) in REGIONS.iter().enumerate() {
                region_owned[r] = reg.iter().filter(|&&t| game.owner[t] == me).count();
            }
            let mut best_action = Action::FinishAttack;
            let mut best_score = 0.0f32; // Only attack if positive expected value.

            for &(a, b) in &EDGES {
                if game.owner[a] != me || game.owner[b] == me {
                    continue;
                }
                let atk = game.troops[a];
                let def = game.troops[b];
                if atk < 3 {
                    continue;
                }
                let dice = (atk - 1).min(3) as usize;
                let ratio = (atk as f32) / (def as f32);
                let target_region = topo.territory_region[b];
                let completes_region = region_owned[target_region] == REGIONS[0].len() - 1;
                let breaks_enemy_region = REGIONS[target_region]
                    .iter()
                    .all(|&t| game.owner[t] == game.owner[b]);

                let threshold = if completes_region || breaks_enemy_region {
                    1.15
                } else {
                    1.45
                };
                if ratio >= threshold || (atk >= 4 && def == 1) {
                    let score = (ratio - threshold)
                        + if completes_region { 2.5 } else { 0.0 }
                        + if breaks_enemy_region { 1.5 } else { 0.0 }
                        + rng.random_range(0.0..0.05);
                    if score > best_score {
                        best_score = score;
                        best_action = Action::Attack(a, b, dice);
                    }
                }
            }
            best_action.encode().unwrap()
        }
        Phase::Fortify => {
            let is_border = |t: usize| topo.neighbors[t].iter().any(|&n| game.owner[n] != me);
            let mut best_action = Action::PassFortify;
            let mut best_moved = 0u64;
            for &(a, b) in &EDGES {
                if game.owner[a] == me
                    && game.owner[b] == me
                    && !is_border(a)
                    && is_border(b)
                    && game.troops[a] > 1
                {
                    let movable = game.troops[a] - 1;
                    if movable > best_moved {
                        best_moved = movable;
                        let opt = if movable > 1 { 2 } else { 0 };
                        best_action = Action::Fortify(a, b, opt);
                    }
                }
            }
            best_action.encode().unwrap()
        }
        Phase::Ended => panic!("heuristic_action called on ended game"),
    }
}
