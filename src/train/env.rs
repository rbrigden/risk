//! Vectorized turn-based self-play environment with PFSP matchmaking and per-seat GAE tracking.

use crate::game::{ACTION_COUNT, Action, Game, MAX_PLAYERS, N, Phase};
use crate::train::bots::{heuristic_action, random_action};
use crate::train::features::{
    BoardTopology, EncodedObs, GLOBAL_FEAT_DIM, NODE_FEAT_DIM, board_potential, encode_observation,
};
use crate::train::league::{LeagueState, MatchOutcome, OpponentKey};
use crate::train::model::{GraphBuffers, RiskPolicyNet};
use burn::prelude::Backend;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

/// Policy controller assigned to a seat in a parallel environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeatController {
    Learner,
    Snapshot(usize),
    Heuristic,
    Random,
}

impl From<OpponentKey> for SeatController {
    fn from(key: OpponentKey) -> Self {
        match key {
            OpponentKey::Random => Self::Random,
            OpponentKey::Heuristic => Self::Heuristic,
            OpponentKey::Snapshot(id) => Self::Snapshot(id),
        }
    }
}

/// A single completed transition for one player seat.
#[derive(Clone, Debug)]
pub struct Transition {
    pub nodes: [f32; N * NODE_FEAT_DIM],
    pub global: [f32; GLOBAL_FEAT_DIM],
    pub mask: [bool; ACTION_COUNT],
    pub action: usize,
    pub log_prob: f32,
    pub value: f32,
    pub reward: f32,
    pub done: bool,
}

/// Finalized training sample with GAE return and advantage ready for PPO minibatches.
#[derive(Clone, Debug)]
pub struct TrainingSample {
    pub nodes: [f32; N * NODE_FEAT_DIM],
    pub global: [f32; GLOBAL_FEAT_DIM],
    pub mask: [bool; ACTION_COUNT],
    pub action: i32,
    pub old_log_prob: f32,
    pub advantage: f32,
    pub target_return: f32,
}

/// Pending open transition for a learner seat awaiting its next turn or episode end.
#[derive(Clone, Debug)]
struct PendingTransition {
    obs: EncodedObs,
    action: usize,
    log_prob: f32,
    value: f32,
    prev_potential: f32,
    accum_reward: f32,
}

/// Per-seat chronological trajectory collected during one rollout window.
#[derive(Default, Clone, Debug)]
struct SeatTrajectory {
    steps: Vec<Transition>,
}

/// Summary statistics from a rollout collection window.
#[derive(Default, Clone, Debug)]
pub struct RolloutStats {
    pub env_steps: usize,
    pub learner_steps: usize,
    pub episodes_finished: usize,
    pub episodes_won: usize,
    pub episodes_drawn: usize,
    pub total_turns: u64,
    pub mean_reward: f32,
    pub league_matches_recorded: usize,
}

struct EnvSlot {
    game: Game,
    controllers: [SeatController; MAX_PLAYERS],
    /// Present when this slot is a rated learner-vs-league matchup (`(opponent_key, learner_seat)`).
    rated_matchup: Option<(OpponentKey, usize)>,
    pending: [Option<PendingTransition>; MAX_PLAYERS],
    trajectories: [SeatTrajectory; MAX_PLAYERS],
    alive: [bool; MAX_PLAYERS],
}

/// Vectorized self-play environment pool managing `num_envs` concurrent Risk games.
pub struct SelfPlayEnvPool {
    players: usize,
    max_turns: u32,
    shaping_coef: f32,
    draw_penalty: f32,
    self_play_prob: f32,
    topo: BoardTopology,
    rng: ChaCha8Rng,
    slots: Vec<EnvSlot>,
}

impl SelfPlayEnvPool {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_envs: usize,
        players: usize,
        max_turns: u32,
        shaping_coef: f32,
        draw_penalty: f32,
        self_play_prob: f32,
        league: &LeagueState,
        seed: u64,
    ) -> Self {
        let topo = BoardTopology::new();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut slots = Vec::with_capacity(num_envs);
        for _ in 0..num_envs {
            let game_seed = rng.random::<u64>();
            let game = Game::new(players, max_turns, Some(game_seed));
            let (controllers, rated_matchup) =
                sample_slot_controllers(players, self_play_prob, league, &mut rng);
            let mut alive = [false; MAX_PLAYERS];
            for item in alive.iter_mut().take(players) {
                *item = true;
            }
            slots.push(EnvSlot {
                game,
                controllers,
                rated_matchup,
                pending: std::array::from_fn(|_| None),
                trajectories: std::array::from_fn(|_| SeatTrajectory::default()),
                alive,
            });
        }
        Self {
            players,
            max_turns,
            shaping_coef,
            draw_penalty,
            self_play_prob: self_play_prob.clamp(0.0, 1.0),
            topo,
            rng,
            slots,
        }
    }

    pub fn topology(&self) -> &BoardTopology {
        &self.topo
    }

    /// Collect at least `target_learner_steps` transitions from learner-controlled seats,
    /// record finished rated league matches into `league`, compute per-seat GAE(gamma, gae_lambda),
    /// and return flat training samples + rollout stats.
    #[allow(clippy::too_many_arguments)]
    pub fn collect_rollout<B: Backend>(
        &mut self,
        learner: &RiskPolicyNet<B>,
        snapshots: &HashMap<usize, RiskPolicyNet<B>>,
        league: &mut LeagueState,
        graph: &GraphBuffers<B>,
        device: &B::Device,
        target_learner_steps: usize,
        gamma: f32,
        gae_lambda: f32,
    ) -> (Vec<TrainingSample>, RolloutStats) {
        for slot in &mut self.slots {
            for traj in &mut slot.trajectories {
                traj.steps.clear();
            }
        }

        let mut stats = RolloutStats::default();
        let mut completed_trajectories: Vec<(Vec<Transition>, f32)> = Vec::new();
        let mut collected_steps = 0usize;

        while collected_steps < target_learner_steps {
            // 1) Advance any Random / Heuristic active seats directly on CPU.
            for slot in &mut self.slots {
                loop {
                    if slot.game.phase == Phase::Ended {
                        reset_slot(
                            slot,
                            self.players,
                            self.max_turns,
                            self.self_play_prob,
                            league,
                            &mut self.rng,
                            &mut completed_trajectories,
                            &mut stats,
                        );
                    }
                    let active = slot.game.active;
                    match slot.controllers[active] {
                        SeatController::Random => {
                            let action_id = random_action(&slot.game, &mut self.rng);
                            apply_action(
                                slot,
                                action_id,
                                self.players,
                                self.shaping_coef,
                                self.draw_penalty,
                                league,
                                &mut stats,
                                &mut completed_trajectories,
                                &mut collected_steps,
                            );
                        }
                        SeatController::Heuristic => {
                            let action_id = heuristic_action(&slot.game, &self.topo, &mut self.rng);
                            apply_action(
                                slot,
                                action_id,
                                self.players,
                                self.shaping_coef,
                                self.draw_penalty,
                                league,
                                &mut stats,
                                &mut completed_trajectories,
                                &mut collected_steps,
                            );
                        }
                        SeatController::Learner | SeatController::Snapshot(_) => break,
                    }
                }
            }

            // 2) Partition slots waiting on Learner vs active Snapshot policies.
            let mut learner_indices = Vec::with_capacity(self.slots.len());
            let mut snapshot_buckets: Vec<(usize, Vec<usize>)> = Vec::new();

            for (idx, slot) in self.slots.iter_mut().enumerate() {
                let active = slot.game.active;
                match slot.controllers[active] {
                    SeatController::Learner => learner_indices.push(idx),
                    SeatController::Snapshot(snap_id) => {
                        if snapshots.contains_key(&snap_id) {
                            if let Some((_, bucket)) =
                                snapshot_buckets.iter_mut().find(|(id, _)| *id == snap_id)
                            {
                                bucket.push(idx);
                            } else {
                                snapshot_buckets.push((snap_id, vec![idx]));
                            }
                        } else {
                            // Snapshot was evicted mid-episode; fall back to Learner for remaining turns.
                            slot.controllers[active] = SeatController::Learner;
                            slot.rated_matchup = None;
                            learner_indices.push(idx);
                        }
                    }
                    SeatController::Random | SeatController::Heuristic => unreachable!(),
                }
            }

            // 3) Batched GPU inference for all active Learner seats.
            if !learner_indices.is_empty() {
                let b = learner_indices.len();
                let mut nodes_flat = Vec::with_capacity(b * N * NODE_FEAT_DIM);
                let mut global_flat = Vec::with_capacity(b * GLOBAL_FEAT_DIM);
                let mut mask_flat = Vec::with_capacity(b * ACTION_COUNT);
                let mut encoded_list = Vec::with_capacity(b);

                for &idx in &learner_indices {
                    let slot = &self.slots[idx];
                    let seat = slot.game.active;
                    let obs = encode_observation(&slot.game, seat, &self.topo);
                    nodes_flat.extend_from_slice(&obs.nodes);
                    global_flat.extend_from_slice(&obs.global);
                    mask_flat.extend_from_slice(&obs.mask);
                    encoded_list.push(obs);
                }

                let (probs_flat, values) =
                    learner.infer_batch(nodes_flat, global_flat, mask_flat, b, graph, device);

                for (i, (&idx, obs)) in learner_indices.iter().zip(encoded_list).enumerate() {
                    let probs_slice = &probs_flat[i * ACTION_COUNT..(i + 1) * ACTION_COUNT];
                    let (action_id, log_prob) =
                        sample_masked_categorical(probs_slice, &obs.mask, &mut self.rng);
                    let value = values[i];

                    let slot = &mut self.slots[idx];
                    let seat = slot.game.active;
                    let cur_pot = board_potential(&slot.game, seat);

                    // Close the previous open transition for this seat (same-player bootstrap).
                    if let Some(prev) = slot.pending[seat].take() {
                        let shaping = self.shaping_coef * (cur_pot - prev.prev_potential);
                        let reward = prev.accum_reward + shaping;
                        slot.trajectories[seat].steps.push(Transition {
                            nodes: prev.obs.nodes,
                            global: prev.obs.global,
                            mask: prev.obs.mask,
                            action: prev.action,
                            log_prob: prev.log_prob,
                            value: prev.value,
                            reward,
                            done: false,
                        });
                        collected_steps += 1;
                    }

                    // Open new transition for this decision.
                    slot.pending[seat] = Some(PendingTransition {
                        obs,
                        action: action_id,
                        log_prob,
                        value,
                        prev_potential: cur_pot,
                        accum_reward: 0.0,
                    });

                    apply_action(
                        slot,
                        action_id,
                        self.players,
                        self.shaping_coef,
                        self.draw_penalty,
                        league,
                        &mut stats,
                        &mut completed_trajectories,
                        &mut collected_steps,
                    );
                    stats.learner_steps += 1;

                    if slot.game.phase == Phase::Ended {
                        reset_slot(
                            slot,
                            self.players,
                            self.max_turns,
                            self.self_play_prob,
                            league,
                            &mut self.rng,
                            &mut completed_trajectories,
                            &mut stats,
                        );
                    }
                }
            }

            // 4) Batched GPU inference for active historical snapshot seats.
            for (snap_id, indices) in snapshot_buckets {
                let model = &snapshots[&snap_id];
                let b = indices.len();
                let mut nodes_flat = Vec::with_capacity(b * N * NODE_FEAT_DIM);
                let mut global_flat = Vec::with_capacity(b * GLOBAL_FEAT_DIM);
                let mut mask_flat = Vec::with_capacity(b * ACTION_COUNT);
                let mut masks = Vec::with_capacity(b);

                for &idx in &indices {
                    let slot = &self.slots[idx];
                    let obs = encode_observation(&slot.game, slot.game.active, &self.topo);
                    nodes_flat.extend_from_slice(&obs.nodes);
                    global_flat.extend_from_slice(&obs.global);
                    mask_flat.extend_from_slice(&obs.mask);
                    masks.push(obs.mask);
                }

                let (probs_flat, _values) =
                    model.infer_batch(nodes_flat, global_flat, mask_flat, b, graph, device);

                for (i, (&idx, mask)) in indices.iter().zip(masks).enumerate() {
                    let probs_slice = &probs_flat[i * ACTION_COUNT..(i + 1) * ACTION_COUNT];
                    let (action_id, _lp) =
                        sample_masked_categorical(probs_slice, &mask, &mut self.rng);
                    let slot = &mut self.slots[idx];
                    apply_action(
                        slot,
                        action_id,
                        self.players,
                        self.shaping_coef,
                        self.draw_penalty,
                        league,
                        &mut stats,
                        &mut completed_trajectories,
                        &mut collected_steps,
                    );
                    if slot.game.phase == Phase::Ended {
                        reset_slot(
                            slot,
                            self.players,
                            self.max_turns,
                            self.self_play_prob,
                            league,
                            &mut self.rng,
                            &mut completed_trajectories,
                            &mut stats,
                        );
                    }
                }
            }
        }

        // 5) Bootstrap all in-progress trajectories at the end of the rollout window.
        let mut bootstrap_keys: Vec<(usize, usize)> = Vec::new();
        let mut boot_nodes = Vec::new();
        let mut boot_global = Vec::new();
        let mut boot_mask = Vec::new();

        for (env_idx, slot) in self.slots.iter().enumerate() {
            for seat in 0..self.players {
                if !slot.trajectories[seat].steps.is_empty() {
                    let obs = encode_observation(&slot.game, seat, &self.topo);
                    boot_nodes.extend_from_slice(&obs.nodes);
                    boot_global.extend_from_slice(&obs.global);
                    boot_mask.extend_from_slice(&obs.mask);
                    bootstrap_keys.push((env_idx, seat));
                }
            }
        }

        let boot_values = if !bootstrap_keys.is_empty() {
            learner.value_batch(
                boot_nodes,
                boot_global,
                boot_mask,
                bootstrap_keys.len(),
                graph,
                device,
            )
        } else {
            Vec::new()
        };

        for ((env_idx, seat), next_val) in bootstrap_keys.into_iter().zip(boot_values) {
            let steps = std::mem::take(&mut self.slots[env_idx].trajectories[seat].steps);
            if !steps.is_empty() {
                let bootstrap = self.slots[env_idx].pending[seat]
                    .as_ref()
                    .map_or(next_val, |p| p.value);
                completed_trajectories.push((steps, bootstrap));
            }
        }

        // 6) Compute per-seat GAE(gamma, gae_lambda) across all trajectories.
        let mut samples = Vec::with_capacity(collected_steps);
        let mut reward_sum = 0.0f32;
        for (traj, bootstrap_val) in completed_trajectories {
            for t in &traj {
                reward_sum += t.reward;
            }
            compute_gae(&traj, bootstrap_val, gamma, gae_lambda, &mut samples);
        }
        if !samples.is_empty() {
            stats.mean_reward = reward_sum / (samples.len() as f32);
        }
        (samples, stats)
    }
}

#[allow(clippy::too_many_arguments)]
fn reset_slot(
    slot: &mut EnvSlot,
    players: usize,
    max_turns: u32,
    self_play_prob: f32,
    league: &LeagueState,
    rng: &mut ChaCha8Rng,
    completed_trajectories: &mut Vec<(Vec<Transition>, f32)>,
    stats: &mut RolloutStats,
) {
    stats.episodes_finished += 1;
    stats.total_turns += u64::from(slot.game.turns);
    if slot.game.winner.is_some() {
        stats.episodes_won += 1;
    } else {
        stats.episodes_drawn += 1;
    }

    for seat in 0..players {
        let steps = std::mem::take(&mut slot.trajectories[seat].steps);
        if !steps.is_empty() {
            completed_trajectories.push((steps, 0.0));
        }
        slot.pending[seat] = None;
    }

    let seed = rng.random::<u64>();
    slot.game = Game::new(players, max_turns, Some(seed));
    let (controllers, rated_matchup) =
        sample_slot_controllers(players, self_play_prob, league, rng);
    slot.controllers = controllers;
    slot.rated_matchup = rated_matchup;
    for (seat, alive) in slot.alive.iter_mut().enumerate() {
        *alive = seat < players;
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_action(
    slot: &mut EnvSlot,
    action_id: usize,
    players: usize,
    shaping_coef: f32,
    draw_penalty: f32,
    league: &mut LeagueState,
    stats: &mut RolloutStats,
    completed_trajectories: &mut Vec<(Vec<Transition>, f32)>,
    collected_steps: &mut usize,
) {
    let action = Action::decode(action_id).expect("decoded action must be valid");
    debug_assert!(slot.game.legal(action));
    slot.game.step(action);
    stats.env_steps += 1;

    if slot.game.phase == Phase::Ended {
        // Record Elo & PFSP win-rate outcome if this was a rated learner-vs-league matchup.
        if let Some((opp_key, learner_seat)) = slot.rated_matchup.take() {
            let outcome = match slot.game.winner {
                Some(w) if w == learner_seat => MatchOutcome::LearnerWin,
                Some(_) => MatchOutcome::LearnerLoss,
                None => MatchOutcome::Draw,
            };
            league.record_match(opp_key, outcome);
            stats.league_matches_recorded += 1;
        }

        // Episode terminated or truncated: flush every seat's open transition.
        for seat in 0..players {
            let terminal_reward = match slot.game.winner {
                Some(w) if w == seat => 1.0,
                Some(_) => -1.0,
                None => -draw_penalty,
            };
            if let Some(prev) = slot.pending[seat].take() {
                let end_pot = board_potential(&slot.game, seat);
                let shaping = shaping_coef * (end_pot - prev.prev_potential);
                slot.trajectories[seat].steps.push(Transition {
                    nodes: prev.obs.nodes,
                    global: prev.obs.global,
                    mask: prev.obs.mask,
                    action: prev.action,
                    log_prob: prev.log_prob,
                    value: prev.value,
                    reward: prev.accum_reward + shaping + terminal_reward,
                    done: true,
                });
                *collected_steps += 1;
            }
            let steps = std::mem::take(&mut slot.trajectories[seat].steps);
            if !steps.is_empty() {
                completed_trajectories.push((steps, 0.0));
            }
        }
        return;
    }

    // Check if any player was just eliminated on this step (multi-player games).
    for seat in 0..players {
        if slot.alive[seat] && !slot.game.owner.contains(&seat) {
            slot.alive[seat] = false;
            if let Some(prev) = slot.pending[seat].take() {
                let shaping = shaping_coef * (0.0 - prev.prev_potential);
                slot.trajectories[seat].steps.push(Transition {
                    nodes: prev.obs.nodes,
                    global: prev.obs.global,
                    mask: prev.obs.mask,
                    action: prev.action,
                    log_prob: prev.log_prob,
                    value: prev.value,
                    reward: prev.accum_reward + shaping - 1.0,
                    done: true,
                });
                *collected_steps += 1;
            }
            let steps = std::mem::take(&mut slot.trajectories[seat].steps);
            if !steps.is_empty() {
                completed_trajectories.push((steps, 0.0));
            }
        }
    }
}

/// Compute Generalized Advantage Estimation (GAE) for a single seat's trajectory.
pub fn compute_gae(
    traj: &[Transition],
    bootstrap_value: f32,
    gamma: f32,
    gae_lambda: f32,
    out: &mut Vec<TrainingSample>,
) {
    let n = traj.len();
    if n == 0 {
        return;
    }
    let mut advantages = vec![0.0f32; n];
    let mut last_gae = 0.0f32;
    let mut next_value = bootstrap_value;

    for t in (0..n).rev() {
        let step = &traj[t];
        let non_terminal = if step.done { 0.0 } else { 1.0 };
        let delta = step.reward + gamma * next_value * non_terminal - step.value;
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae;
        advantages[t] = last_gae;
        next_value = step.value;
    }

    for (step, adv) in traj.iter().zip(advantages) {
        out.push(TrainingSample {
            nodes: step.nodes,
            global: step.global,
            mask: step.mask,
            action: step.action as i32,
            old_log_prob: step.log_prob,
            advantage: adv,
            target_return: adv + step.value,
        });
    }
}

/// Sample a legal action from masked probabilities, returning `(action_id, log_prob)`.
pub fn sample_masked_categorical(
    probs: &[f32],
    mask: &[bool; ACTION_COUNT],
    rng: &mut impl Rng,
) -> (usize, f32) {
    let mut total = 0.0f64;
    let mut first_legal = None;
    for (i, (&p, &allowed)) in probs.iter().zip(mask.iter()).enumerate() {
        if allowed {
            if first_legal.is_none() {
                first_legal = Some(i);
            }
            if p.is_finite() && p > 0.0 {
                total += f64::from(p);
            }
        }
    }

    if total <= 0.0 {
        let legal_count = mask.iter().filter(|&&m| m).count();
        let target = rng.random_range(0..legal_count);
        let chosen = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| m.then_some(i))
            .nth(target)
            .expect("must have at least one legal action");
        return (chosen, -(legal_count as f32).ln());
    }

    let threshold = rng.random::<f64>() * total;
    let mut acc = 0.0f64;
    let mut chosen = first_legal.expect("must have at least one legal action");
    let mut chosen_p = 0.0f64;

    for (i, (&p, &allowed)) in probs.iter().zip(mask.iter()).enumerate() {
        if !allowed {
            continue;
        }
        let pf = if p.is_finite() && p > 0.0 {
            f64::from(p)
        } else {
            0.0
        };
        acc += pf;
        if acc >= threshold {
            chosen = i;
            chosen_p = pf;
            break;
        }
        if pf > 0.0 {
            chosen = i;
            chosen_p = pf;
        }
    }

    let norm_p = (chosen_p / total).max(1e-12);
    (chosen, (norm_p as f32).ln())
}

/// Select the highest-probability legal action (for deterministic evaluation).
pub fn greedy_masked_action(probs: &[f32], mask: &[bool; ACTION_COUNT]) -> usize {
    let mut best_id = None;
    let mut best_p = f32::NEG_INFINITY;
    for (i, (&p, &allowed)) in probs.iter().zip(mask.iter()).enumerate() {
        if allowed && p > best_p {
            best_p = p;
            best_id = Some(i);
        }
    }
    best_id.expect("must have at least one legal action")
}

/// Sample either pure latest-vs-latest self-play or a PFSP-prioritized league matchup.
fn sample_slot_controllers(
    players: usize,
    self_play_prob: f32,
    league: &LeagueState,
    rng: &mut impl Rng,
) -> ([SeatController; MAX_PLAYERS], Option<(OpponentKey, usize)>) {
    let mut controllers = [SeatController::Learner; MAX_PLAYERS];
    if rng.random::<f32>() < self_play_prob {
        return (controllers, None);
    }

    let learner_seat = rng.random_range(0..players);
    let opp_key = league.sample_opponent(rng);
    let opp_ctrl = SeatController::from(opp_key);
    for (seat, ctrl) in controllers.iter_mut().enumerate().take(players) {
        *ctrl = if seat == learner_seat {
            SeatController::Learner
        } else {
            opp_ctrl
        };
    }
    (controllers, Some((opp_key, learner_seat)))
}
