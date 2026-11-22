//! Prioritized Fictitious Self-Play (PFSP) opponent matchmaking and Elo rating league.

use crate::train::telemetry::{AnchorElos, LeagueMemberReport, LeagueReport};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

/// Weighting strategy for Prioritized Fictitious Self-Play (PFSP) opponent sampling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PfspMode {
    /// Focuses on opponents the learner currently struggles against: `(1 - wr)^p`.
    #[default]
    Hard,
    /// Focuses on evenly matched opponents near 50% win rate: `4 * wr * (1 - wr)`.
    Variance,
    /// Samples all active opponents uniformly.
    Uniform,
}

impl fmt::Display for PfspMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hard => write!(f, "hard"),
            Self::Variance => write!(f, "variance"),
            Self::Uniform => write!(f, "uniform"),
        }
    }
}

impl FromStr for PfspMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "hard" => Ok(Self::Hard),
            "variance" => Ok(Self::Variance),
            "uniform" => Ok(Self::Uniform),
            other => Err(format!(
                "invalid PFSP mode '{other}' (expected 'hard', 'variance', or 'uniform')"
            )),
        }
    }
}

/// Category of an opponent member in the league.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemberKind {
    Random,
    Heuristic,
    Snapshot,
}

/// Stable identifier for a league opponent.
///
/// `Snapshot(usize)` holds a monotonically increasing snapshot ID rather than a vector index,
/// so evicting older snapshots never invalidates existing keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpponentKey {
    Random,
    Heuristic,
    Snapshot(usize),
}

/// Configuration for PFSP matchmaking, Elo updates, and snapshot pool capacity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeagueConfig {
    /// Number of players in each game (2..=6).
    pub players: usize,
    /// PFSP weighting mode.
    pub pfsp_mode: PfspMode,
    /// Exponent `p` used by `PfspMode::Hard`: `(1 - wr)^p`.
    pub pfsp_p: f32,
    /// Additive weight floor applied before normalization so beaten opponents are still revisited.
    pub min_weight: f32,
    /// Elo K-factor for the active learner.
    pub k_factor: f64,
    /// Gentler Elo K-factor for non-anchored opponents (Heuristic and Snapshots).
    pub opponent_k_factor: f64,
    /// Initial learner Elo rating.
    pub initial_learner_elo: f64,
    /// Fixed anchor Elo for the Random bot (never drifts).
    pub random_anchor_elo: f64,
    /// Initial Elo for the Heuristic bot.
    pub heuristic_init_elo: f64,
    /// Maximum number of active historical snapshots kept in the pool.
    pub max_snapshots: usize,
    /// EMA smoothing factor for tracking learner win rate against each opponent.
    pub win_rate_ema_alpha: f32,
}

impl Default for LeagueConfig {
    fn default() -> Self {
        Self {
            players: 2,
            pfsp_mode: PfspMode::Hard,
            pfsp_p: 2.0,
            min_weight: 0.02,
            k_factor: 24.0,
            opponent_k_factor: 8.0,
            initial_learner_elo: 1000.0,
            random_anchor_elo: 800.0,
            heuristic_init_elo: 1200.0,
            max_snapshots: 12,
            win_rate_ema_alpha: 0.08,
        }
    }
}

/// Outcome of a match from the learner's perspective.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchOutcome {
    LearnerWin,
    LearnerLoss,
    Draw,
}

/// Tracked opponent entry in the league.
#[derive(Clone, Debug)]
pub struct LeagueMember {
    pub key: OpponentKey,
    pub id: String,
    pub kind: MemberKind,
    pub created_update: usize,
    pub elo: f64,
    pub anchored: bool,
    pub active: bool,
    pub games: u64,
    pub learner_wins: u64,
    pub learner_losses: u64,
    pub draws: u64,
    pub learner_win_rate_ema: f32,
}

impl LeagueMember {
    /// Smoothed learner win rate against this opponent in `[0.0, 1.0]`.
    ///
    /// Blends a Beta(1, 1) Bayesian prior `(wins + 0.5*draws + 1) / (games + 2)` with the
    /// exponential moving average over the first 10 games so brand-new snapshots start
    /// smoothly near `0.5` without early variance spikes, then uses the EMA for `games >= 10`.
    pub fn smoothed_learner_win_rate(&self) -> f32 {
        let ema = self.learner_win_rate_ema.clamp(0.0, 1.0);
        if self.games >= 10 {
            return ema;
        }
        let bayes =
            (self.learner_wins as f32 + 0.5 * self.draws as f32 + 1.0) / (self.games as f32 + 2.0);
        let blend = (self.games as f32) / 10.0;
        (bayes * (1.0 - blend) + ema * blend).clamp(0.0, 1.0)
    }
}

/// Historical Elo snapshot recorded over training updates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EloPoint {
    pub update: usize,
    pub steps: usize,
    pub learner_elo: f64,
    pub best_snapshot_elo: f64,
    pub mean_snapshot_elo: f64,
}

/// Stateful PFSP + Elo league tracking baseline bots and historical learner snapshots.
#[derive(Clone, Debug)]
pub struct LeagueState {
    config: LeagueConfig,
    learner_elo: f64,
    next_snapshot_id: usize,
    members: Vec<LeagueMember>,
    elo_history: Vec<EloPoint>,
}

impl LeagueState {
    /// Initializes a new league seeded with an anchored `Random` baseline and a `Heuristic` bot.
    pub fn new(config: LeagueConfig) -> Self {
        let learner_elo = config.initial_learner_elo;
        let members = vec![
            LeagueMember {
                key: OpponentKey::Random,
                id: "random".to_string(),
                kind: MemberKind::Random,
                created_update: 0,
                elo: config.random_anchor_elo,
                anchored: true,
                active: true,
                games: 0,
                learner_wins: 0,
                learner_losses: 0,
                draws: 0,
                learner_win_rate_ema: 0.5,
            },
            LeagueMember {
                key: OpponentKey::Heuristic,
                id: "heuristic".to_string(),
                kind: MemberKind::Heuristic,
                created_update: 0,
                elo: config.heuristic_init_elo,
                anchored: false,
                active: true,
                games: 0,
                learner_wins: 0,
                learner_losses: 0,
                draws: 0,
                learner_win_rate_ema: 0.5,
            },
        ];

        Self {
            config,
            learner_elo,
            next_snapshot_id: 0,
            members,
            elo_history: Vec::new(),
        }
    }

    /// Multi-player Bradley-Terry generalization of standard Elo expected score.
    ///
    /// Reduces identically to `1 / (1 + 10^((r_opp - r_learner) / 400))` when `players == 2`,
    /// and equals `1 / players` whenever `r_learner == r_opp`.
    pub fn expected_score(r_learner: f64, r_opp: f64, players: usize) -> f64 {
        let opponents = (players.max(2) - 1) as f64;
        1.0 / (1.0 + opponents * 10.0_f64.powf((r_opp - r_learner) / 400.0))
    }

    /// Records the outcome of a completed match between the learner and `opp`, updating
    /// head-to-head statistics, win-rate EMA, and Elo ratings.
    pub fn record_match(&mut self, opp: OpponentKey, outcome: MatchOutcome) {
        let Some(member) = self.members.iter_mut().find(|m| m.key == opp) else {
            return;
        };

        member.games += 1;
        let (elo_score, wr_sample) = match outcome {
            MatchOutcome::LearnerWin => {
                member.learner_wins += 1;
                (1.0_f64, 1.0_f32)
            }
            MatchOutcome::LearnerLoss => {
                member.learner_losses += 1;
                (0.0_f64, 0.0_f32)
            }
            MatchOutcome::Draw => {
                member.draws += 1;
                (1.0 / (self.config.players.max(2) as f64), 0.5_f32)
            }
        };

        let alpha = self.config.win_rate_ema_alpha.clamp(0.0, 1.0);
        member.learner_win_rate_ema =
            (1.0 - alpha) * member.learner_win_rate_ema + alpha * wr_sample;

        let expected = Self::expected_score(self.learner_elo, member.elo, self.config.players);
        let delta = elo_score - expected;
        self.learner_elo += self.config.k_factor * delta;
        if !member.anchored {
            member.elo -= self.config.opponent_k_factor * delta;
        }
    }

    /// Adds a new active snapshot inheriting the current learner Elo.
    ///
    /// If the number of active snapshots exceeds `config.max_snapshots`, marks one active
    /// snapshot inactive (preserving the highest-Elo active snapshot, and also the lowest-Elo
    /// active snapshot when `max_snapshots >= 3`, while evicting the non-extremum snapshot
    /// with the lowest PFSP value / highest learner win rate / oldest creation update).
    ///
    /// Returns `(new_snapshot_id, evicted_snapshot_id)`.
    pub fn add_snapshot(&mut self, update: usize) -> (usize, Option<usize>) {
        let snapshot_id = self.next_snapshot_id;
        self.next_snapshot_id += 1;

        self.members.push(LeagueMember {
            key: OpponentKey::Snapshot(snapshot_id),
            id: format!("snap_{update:04}"),
            kind: MemberKind::Snapshot,
            created_update: update,
            elo: self.learner_elo,
            anchored: false,
            active: true,
            games: 0,
            learner_wins: 0,
            learner_losses: 0,
            draws: 0,
            learner_win_rate_ema: 0.5,
        });

        let active_snap_indices: Vec<usize> = self
            .members
            .iter()
            .enumerate()
            .filter_map(|(idx, m)| (m.kind == MemberKind::Snapshot && m.active).then_some(idx))
            .collect();

        if active_snap_indices.len() <= self.config.max_snapshots {
            return (snapshot_id, None);
        }

        let evicted_idx = self.select_eviction_candidate(&active_snap_indices, snapshot_id);
        let evicted = &mut self.members[evicted_idx];
        evicted.active = false;
        let evicted_id = match evicted.key {
            OpponentKey::Snapshot(id) => Some(id),
            _ => None,
        };
        (snapshot_id, evicted_id)
    }

    fn select_eviction_candidate(
        &self,
        active_snap_indices: &[usize],
        new_snapshot_id: usize,
    ) -> usize {
        if self.config.max_snapshots == 0 {
            return *active_snap_indices
                .last()
                .expect("at least one active snapshot exists");
        }

        // Preserve highest-Elo active snapshot (tie-break toward newest).
        let best_idx = *active_snap_indices
            .iter()
            .max_by(|&&a, &&b| {
                self.members[a]
                    .elo
                    .total_cmp(&self.members[b].elo)
                    .then_with(|| {
                        snapshot_id_of(&self.members[a]).cmp(&snapshot_id_of(&self.members[b]))
                    })
            })
            .expect("at least one active snapshot exists");

        // When capacity >= 3, also preserve the lowest-Elo active snapshot (tie-break toward oldest)
        // to keep skill diversity across the ladder.
        let worst_idx = if self.config.max_snapshots >= 3 {
            active_snap_indices.iter().copied().min_by(|&a, &b| {
                self.members[a]
                    .elo
                    .total_cmp(&self.members[b].elo)
                    .then_with(|| {
                        snapshot_id_of(&self.members[a]).cmp(&snapshot_id_of(&self.members[b]))
                    })
            })
        } else {
            None
        };

        let non_protected: Vec<usize> = active_snap_indices
            .iter()
            .copied()
            .filter(|&idx| idx != best_idx && Some(idx) != worst_idx)
            .collect();

        // Prefer not to immediately evict the brand-new snapshot if any older non-protected
        // candidate is available.
        let preferred: Vec<usize> = non_protected
            .iter()
            .copied()
            .filter(|&idx| self.members[idx].key != OpponentKey::Snapshot(new_snapshot_id))
            .collect();

        let candidates = if !preferred.is_empty() {
            preferred
        } else if !non_protected.is_empty() {
            non_protected
        } else {
            active_snap_indices.to_vec()
        };

        // Pick candidate with lowest raw PFSP score, breaking ties by highest learner win rate,
        // then oldest creation update, then lowest snapshot ID.
        *candidates
            .iter()
            .min_by(|&&a, &&b| {
                let ma = &self.members[a];
                let mb = &self.members[b];
                let wr_a = ma.smoothed_learner_win_rate();
                let wr_b = mb.smoothed_learner_win_rate();
                let pfsp_a = self.raw_pfsp_weight(wr_a);
                let pfsp_b = self.raw_pfsp_weight(wr_b);

                approx_cmp(pfsp_a, pfsp_b, 1e-6)
                    .then_with(|| approx_cmp(wr_b, wr_a, 1e-6))
                    .then_with(|| ma.created_update.cmp(&mb.created_update))
                    .then_with(|| snapshot_id_of(ma).cmp(&snapshot_id_of(mb)))
            })
            .expect("eviction candidate list is non-empty")
    }

    fn raw_pfsp_weight(&self, learner_win_rate: f32) -> f32 {
        let p = learner_win_rate.clamp(0.0, 1.0);
        match self.config.pfsp_mode {
            PfspMode::Hard => (1.0 - p).powf(self.config.pfsp_p.max(0.0)),
            PfspMode::Variance => 4.0 * p * (1.0 - p),
            PfspMode::Uniform => 1.0,
        }
    }

    /// Computes normalized PFSP sampling probabilities across all currently active members.
    pub fn pfsp_weights(&self) -> Vec<(OpponentKey, f32)> {
        let floor = self.config.min_weight.max(0.0);
        let mut weights: Vec<(OpponentKey, f32)> = self
            .members
            .iter()
            .filter(|m| m.active)
            .map(|m| {
                let raw = self.raw_pfsp_weight(m.smoothed_learner_win_rate());
                (m.key, (raw + floor).max(0.0))
            })
            .collect();

        normalize_weights(&mut weights);
        weights
    }

    /// Samples an active opponent according to `pfsp_weights()`.
    pub fn sample_opponent(&self, rng: &mut impl Rng) -> OpponentKey {
        let weights = self.pfsp_weights();
        sample_weighted(&weights, rng).unwrap_or(OpponentKey::Random)
    }

    /// Samples an active snapshot ID using normalized PFSP weights over active snapshots only.
    pub fn sample_snapshot(&self, rng: &mut impl Rng) -> Option<usize> {
        let floor = self.config.min_weight.max(0.0);
        let mut weights: Vec<(usize, f32)> = self
            .members
            .iter()
            .filter(|m| m.active)
            .filter_map(|m| match m.key {
                OpponentKey::Snapshot(id) => {
                    let raw = self.raw_pfsp_weight(m.smoothed_learner_win_rate());
                    Some((id, (raw + floor).max(0.0)))
                }
                _ => None,
            })
            .collect();

        if weights.is_empty() {
            return None;
        }
        normalize_weights(&mut weights);
        sample_weighted(&weights, rng)
    }

    /// Appends an `EloPoint` summarizing the current learner and snapshot pool ratings.
    pub fn record_elo_point(&mut self, update: usize, steps: usize) {
        let mut best_snap = f64::NEG_INFINITY;
        let mut sum_snap = 0.0_f64;
        let mut count_snap = 0usize;

        for m in &self.members {
            if m.kind == MemberKind::Snapshot && m.active {
                if m.elo > best_snap {
                    best_snap = m.elo;
                }
                sum_snap += m.elo;
                count_snap += 1;
            }
        }

        let (best_snapshot_elo, mean_snapshot_elo) = if count_snap > 0 {
            (best_snap, sum_snap / count_snap as f64)
        } else {
            (self.learner_elo, self.learner_elo)
        };

        self.elo_history.push(EloPoint {
            update,
            steps,
            learner_elo: self.learner_elo,
            best_snapshot_elo,
            mean_snapshot_elo,
        });
    }

    /// Returns the league configuration.
    pub fn config(&self) -> &LeagueConfig {
        &self.config
    }

    /// Returns the learner's current Elo rating.
    pub fn learner_elo(&self) -> f64 {
        self.learner_elo
    }

    /// Returns the IDs of all currently active snapshots in insertion order.
    pub fn active_snapshot_ids(&self) -> Vec<usize> {
        self.members
            .iter()
            .filter(|m| m.active)
            .filter_map(|m| match m.key {
                OpponentKey::Snapshot(id) => Some(id),
                _ => None,
            })
            .collect()
    }

    /// Looks up a league member by its stable `OpponentKey`.
    pub fn member(&self, key: OpponentKey) -> Option<&LeagueMember> {
        self.members.iter().find(|m| m.key == key)
    }

    /// Returns all tracked league members (including inactive historical snapshots).
    pub fn members(&self) -> &[LeagueMember] {
        &self.members
    }

    /// Returns the recorded Elo trajectory points.
    pub fn elo_history(&self) -> &[EloPoint] {
        &self.elo_history
    }

    /// Builds a serializable `LeagueReport` snapshot for live telemetry and dashboards.
    pub fn build_report(
        &self,
        run_name: &str,
        max_turns: u32,
        backend: &str,
        current_update: usize,
        total_updates: usize,
    ) -> LeagueReport {
        let weight_map: HashMap<OpponentKey, f32> = self.pfsp_weights().into_iter().collect();

        let random_elo = self
            .member(OpponentKey::Random)
            .map(|m| m.elo)
            .unwrap_or(self.config.random_anchor_elo);
        let heuristic_elo = self
            .member(OpponentKey::Heuristic)
            .map(|m| m.elo)
            .unwrap_or(self.config.heuristic_init_elo);

        let members = self
            .members
            .iter()
            .map(|m| LeagueMemberReport {
                id: m.id.clone(),
                kind: m.kind,
                created_update: m.created_update,
                elo: m.elo,
                games: m.games,
                wins: m.learner_losses,
                losses: m.learner_wins,
                draws: m.draws,
                learner_win_rate: m.smoothed_learner_win_rate(),
                pfsp_weight: if m.active {
                    weight_map.get(&m.key).copied().unwrap_or(0.0)
                } else {
                    0.0
                },
                active: m.active,
            })
            .collect();

        LeagueReport {
            run_name: run_name.to_string(),
            players: self.config.players,
            max_turns,
            backend: backend.to_string(),
            pfsp_mode: self.config.pfsp_mode,
            pfsp_p: self.config.pfsp_p,
            current_update,
            total_updates,
            learner_elo: self.learner_elo,
            anchors: AnchorElos {
                random_elo,
                heuristic_elo,
            },
            members,
            elo_history: self.elo_history.clone(),
        }
    }
}

fn snapshot_id_of(member: &LeagueMember) -> usize {
    match member.key {
        OpponentKey::Snapshot(id) => id,
        _ => usize::MAX,
    }
}

fn approx_cmp(a: f32, b: f32, eps: f32) -> std::cmp::Ordering {
    if (a - b).abs() <= eps {
        std::cmp::Ordering::Equal
    } else {
        a.total_cmp(&b)
    }
}

fn normalize_weights<K>(weights: &mut [(K, f32)]) {
    if weights.is_empty() {
        return;
    }
    let total: f32 = weights.iter().map(|(_, w)| *w).sum();
    if total > 0.0 && total.is_finite() {
        for (_, w) in weights.iter_mut() {
            *w /= total;
        }
    } else {
        let uniform = 1.0 / (weights.len() as f32);
        for (_, w) in weights.iter_mut() {
            *w = uniform;
        }
    }
}

fn sample_weighted<K: Copy>(weights: &[(K, f32)], rng: &mut impl Rng) -> Option<K> {
    if weights.is_empty() {
        return None;
    }
    let target: f32 = rng.random();
    let mut cdf = 0.0_f32;
    for &(key, w) in weights {
        cdf += w;
        if target < cdf {
            return Some(key);
        }
    }
    weights.last().map(|&(k, _)| k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn pfsp_mode_display_and_from_str() {
        for (mode, s) in [
            (PfspMode::Hard, "hard"),
            (PfspMode::Variance, "variance"),
            (PfspMode::Uniform, "uniform"),
        ] {
            assert_eq!(mode.to_string(), s);
            assert_eq!(s.parse::<PfspMode>().unwrap(), mode);
            assert_eq!(s.to_uppercase().parse::<PfspMode>().unwrap(), mode);
        }
        assert!("bogus".parse::<PfspMode>().is_err());
    }

    #[test]
    fn expected_score_two_player_and_multiplayer() {
        // Equal Elo -> 1 / players
        for players in 2..=6 {
            let e = LeagueState::expected_score(1200.0, 1200.0, players);
            assert!(
                (e - 1.0 / players as f64).abs() < 1e-12,
                "players={players}: expected {}, got {e}",
                1.0 / players as f64
            );
        }

        // 2-player symmetry: E(A, B) + E(B, A) == 1.0
        let e_ab = LeagueState::expected_score(1350.0, 1100.0, 2);
        let e_ba = LeagueState::expected_score(1100.0, 1350.0, 2);
        assert!((e_ab + e_ba - 1.0).abs() < 1e-12);

        // +400 Elo advantage in 2-player -> 10 / 11
        let e_400_2p = LeagueState::expected_score(1400.0, 1000.0, 2);
        assert!((e_400_2p - 10.0 / 11.0).abs() < 1e-12);

        // +400 Elo advantage in 4-player -> 1 / (1 + 3 * 0.1) = 10 / 13
        let e_400_4p = LeagueState::expected_score(1400.0, 1000.0, 4);
        assert!((e_400_4p - 10.0 / 13.0).abs() < 1e-12);
    }

    #[test]
    fn elo_updates_anchor_random_and_move_learner() {
        let mut league = LeagueState::new(LeagueConfig::default());
        let init_learner = league.learner_elo();

        // Winning against Random increases learner Elo while Random stays anchored at 800.0.
        league.record_match(OpponentKey::Random, MatchOutcome::LearnerWin);
        assert!(league.learner_elo() > init_learner);
        assert_eq!(league.member(OpponentKey::Random).unwrap().elo, 800.0);

        let after_win = league.learner_elo();
        // Losing against Random decreases learner Elo while Random remains anchored at 800.0.
        league.record_match(OpponentKey::Random, MatchOutcome::LearnerLoss);
        assert!(league.learner_elo() < after_win);
        assert_eq!(league.member(OpponentKey::Random).unwrap().elo, 800.0);

        // Winning against unanchored Heuristic increases learner Elo and lowers Heuristic Elo.
        let heur_before = league.member(OpponentKey::Heuristic).unwrap().elo;
        let learner_before = league.learner_elo();
        league.record_match(OpponentKey::Heuristic, MatchOutcome::LearnerWin);
        assert!(league.learner_elo() > learner_before);
        assert!(league.member(OpponentKey::Heuristic).unwrap().elo < heur_before);
    }

    #[test]
    fn pfsp_weights_hard_variance_and_uniform() {
        let setup = |mode: PfspMode| {
            let mut league = LeagueState::new(LeagueConfig {
                pfsp_mode: mode,
                pfsp_p: 2.0,
                min_weight: 0.02,
                ..Default::default()
            });
            let (s_even, _) = league.add_snapshot(1);
            // Learner crushes Random (~100% WR)
            for _ in 0..40 {
                league.record_match(OpponentKey::Random, MatchOutcome::LearnerWin);
            }
            // Learner loses heavily to Heuristic (~0% WR)
            for _ in 0..40 {
                league.record_match(OpponentKey::Heuristic, MatchOutcome::LearnerLoss);
            }
            // Learner splits evenly with snapshot (~50% WR)
            for i in 0..40 {
                let outcome = if i % 2 == 0 {
                    MatchOutcome::LearnerWin
                } else {
                    MatchOutcome::LearnerLoss
                };
                league.record_match(OpponentKey::Snapshot(s_even), outcome);
            }
            (league, s_even)
        };

        // Hard mode: Heuristic (low WR) >> Snapshot (~50% WR) >> Random (high WR) > 0
        let (hard_league, s_even) = setup(PfspMode::Hard);
        let w_hard: HashMap<OpponentKey, f32> = hard_league.pfsp_weights().into_iter().collect();
        let sum_hard: f32 = w_hard.values().sum();
        assert!((sum_hard - 1.0).abs() < 1e-5);
        assert!(w_hard[&OpponentKey::Heuristic] > w_hard[&OpponentKey::Snapshot(s_even)] * 2.0);
        assert!(w_hard[&OpponentKey::Snapshot(s_even)] > w_hard[&OpponentKey::Random] * 2.0);
        assert!(w_hard[&OpponentKey::Random] > 0.0);

        // Variance mode: Snapshot (~50% WR) peaks above both 0% and 100% WR opponents
        let (var_league, s_even) = setup(PfspMode::Variance);
        let w_var: HashMap<OpponentKey, f32> = var_league.pfsp_weights().into_iter().collect();
        let sum_var: f32 = w_var.values().sum();
        assert!((sum_var - 1.0).abs() < 1e-5);
        assert!(w_var[&OpponentKey::Snapshot(s_even)] > w_var[&OpponentKey::Heuristic] * 3.0);
        assert!(w_var[&OpponentKey::Snapshot(s_even)] > w_var[&OpponentKey::Random] * 3.0);

        // Uniform mode: all active opponents have identical weights
        let (uni_league, s_even) = setup(PfspMode::Uniform);
        let w_uni: HashMap<OpponentKey, f32> = uni_league.pfsp_weights().into_iter().collect();
        let sum_uni: f32 = w_uni.values().sum();
        assert!((sum_uni - 1.0).abs() < 1e-5);
        let expected = 1.0 / 3.0;
        assert!((w_uni[&OpponentKey::Random] - expected).abs() < 1e-5);
        assert!((w_uni[&OpponentKey::Heuristic] - expected).abs() < 1e-5);
        assert!((w_uni[&OpponentKey::Snapshot(s_even)] - expected).abs() < 1e-5);
    }

    #[test]
    fn snapshot_eviction_preserves_top_elo_and_returns_evicted_id() {
        let mut league = LeagueState::new(LeagueConfig {
            max_snapshots: 3,
            ..Default::default()
        });

        // Create 3 snapshots at distinct Elos: low (900), mid (1100), high (1400).
        league.learner_elo = 900.0;
        let (s_low, ev0) = league.add_snapshot(1);
        assert_eq!(ev0, None);

        league.learner_elo = 1100.0;
        let (s_mid, ev1) = league.add_snapshot(2);
        assert_eq!(ev1, None);
        // Make learner easily beat s_mid so s_mid has low PFSP value.
        for _ in 0..15 {
            league.record_match(OpponentKey::Snapshot(s_mid), MatchOutcome::LearnerWin);
        }

        league.learner_elo = 1400.0;
        let (s_high, ev2) = league.add_snapshot(3);
        assert_eq!(ev2, None);

        // Adding a 4th snapshot triggers eviction; both s_high (top Elo) and s_low (lowest Elo)
        // must be preserved, so s_mid is evicted.
        league.learner_elo = 1250.0;
        let (s_new, ev3) = league.add_snapshot(4);
        assert_eq!(ev3, Some(s_mid));

        let active = league.active_snapshot_ids();
        assert_eq!(active.len(), 3);
        assert!(active.contains(&s_high));
        assert!(active.contains(&s_low));
        assert!(active.contains(&s_new));
        assert!(!active.contains(&s_mid));
        assert!(!league.member(OpponentKey::Snapshot(s_mid)).unwrap().active);

        // Sampling snapshots never yields an evicted snapshot.
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        for _ in 0..50 {
            let sampled = league.sample_snapshot(&mut rng).unwrap();
            assert_ne!(sampled, s_mid);
            assert!(active.contains(&sampled));
        }
    }

    #[test]
    fn report_and_elo_history_match_state() {
        let mut league = LeagueState::new(LeagueConfig::default());
        league.record_elo_point(0, 0);
        assert_eq!(league.elo_history().len(), 1);
        assert_eq!(league.elo_history()[0].best_snapshot_elo, 1000.0);

        let (s0, _) = league.add_snapshot(5);
        league.record_match(OpponentKey::Snapshot(s0), MatchOutcome::LearnerWin);
        league.record_match(OpponentKey::Snapshot(s0), MatchOutcome::LearnerLoss);
        league.record_match(OpponentKey::Snapshot(s0), MatchOutcome::LearnerLoss);
        league.record_elo_point(5, 1024);

        let report = league.build_report("test_run", 200, "ndarray", 5, 100);
        assert_eq!(report.run_name, "test_run");
        assert_eq!(report.elo_history.len(), 2);

        let snap = report.members.iter().find(|m| m.id == "snap_0005").unwrap();
        assert_eq!(snap.games, 3);
        // Report wins/losses are from the opponent's perspective.
        assert_eq!(snap.wins, 2);
        assert_eq!(snap.losses, 1);
        assert_eq!(snap.draws, 0);
        assert!(snap.active);
        assert!(snap.pfsp_weight > 0.0);
    }
}
