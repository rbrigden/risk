//! Masked PPO-Clip self-play trainer with PFSP + Elo League, live telemetry, and checkpoints.

use crate::game::{ACTION_COUNT, Action, Game, N, Phase};
use crate::train::bots::{heuristic_action, random_action};
use crate::train::env::{
    SelfPlayEnvPool, TrainingSample, greedy_masked_action, sample_masked_categorical,
};
use crate::train::features::{BoardTopology, GLOBAL_FEAT_DIM, NODE_FEAT_DIM, encode_observation};
use crate::train::league::{LeagueConfig, LeagueState, MatchOutcome, OpponentKey, PfspMode};
use crate::train::model::{GraphBuffers, RiskPolicyConfig, RiskPolicyNet};
use crate::train::telemetry::{MetricsRecord, TelemetryWriter};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::TensorData;
use burn::tensor::backend::AutodiffBackend;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Full hyperparameter configuration for PFSP + Elo self-play PPO training.
#[derive(Clone, Debug)]
pub struct PpoConfig {
    pub players: usize,
    pub max_turns: u32,
    pub num_envs: usize,
    pub steps_per_update: usize,
    pub updates: usize,
    pub ppo_epochs: usize,
    pub minibatch_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub learning_rate: f64,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub clip_eps: f32,
    pub vf_coef: f32,
    pub ent_coef: f32,
    pub max_grad_norm: f32,
    pub target_kl: f32,
    pub shaping_coef: f32,
    pub draw_penalty: f32,
    /// Probability an env slot runs pure latest-vs-latest self-play; remainder uses PFSP league.
    pub self_play_prob: f32,
    pub pfsp_mode: PfspMode,
    pub pfsp_p: f32,
    pub elo_k: f64,
    pub opponent_elo_k: f64,
    pub snapshot_interval: usize,
    pub snapshot_pool_size: usize,
    pub eval_interval: usize,
    pub eval_games: usize,
    pub seed: u64,
    pub save_dir: Option<PathBuf>,
    /// Optional checkpoint path (with or without `.mpk` suffix) to initialize weights from.
    pub resume_from: Option<PathBuf>,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            players: 2,
            max_turns: 200,
            num_envs: 64,
            steps_per_update: 4096,
            updates: 100,
            ppo_epochs: 4,
            minibatch_size: 256,
            hidden_dim: 64,
            num_layers: 2,
            learning_rate: 8e-4,
            gamma: 0.998,
            gae_lambda: 0.97,
            clip_eps: 0.2,
            vf_coef: 0.5,
            ent_coef: 0.002,
            max_grad_norm: 0.5,
            target_kl: 0.03,
            shaping_coef: 1.0,
            draw_penalty: 0.1,
            self_play_prob: 0.40,
            pfsp_mode: PfspMode::Variance,
            pfsp_p: 2.0,
            elo_k: 24.0,
            opponent_elo_k: 8.0,
            snapshot_interval: 10,
            snapshot_pool_size: 12,
            eval_interval: 10,
            eval_games: 16,
            seed: 42,
            save_dir: Some(PathBuf::from("runs/ppo")),
            resume_from: None,
        }
    }
}

/// Metrics returned for each PPO update.
#[derive(Clone, Debug)]
pub struct UpdateMetrics {
    pub update: usize,
    pub total_env_steps: usize,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub approx_kl: f32,
    pub clip_frac: f32,
    pub steps_per_sec: f64,
    pub episodes_finished: usize,
    pub win_frac: f32,
    pub avg_turns: f32,
    pub learner_elo: f64,
}

/// Evaluation results against a baseline or snapshot opponent.
#[derive(Clone, Debug)]
pub struct EvalResult {
    pub games: usize,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub avg_turns: f32,
}

impl EvalResult {
    pub fn win_rate(&self) -> f32 {
        (self.wins as f32) / (self.games.max(1) as f32)
    }

    fn record_into_league(&self, league: &mut LeagueState, opp: OpponentKey) {
        for _ in 0..self.wins {
            league.record_match(opp, MatchOutcome::LearnerWin);
        }
        for _ in 0..self.losses {
            league.record_match(opp, MatchOutcome::LearnerLoss);
        }
        for _ in 0..self.draws {
            league.record_match(opp, MatchOutcome::Draw);
        }
    }
}

/// Baseline opponent kind for deterministic evaluation.
#[derive(Clone, Copy, Debug)]
pub enum EvalOpponent {
    Random,
    Heuristic,
}

/// Train a Graph Actor-Critic policy via Masked PFSP + Elo Self-Play PPO on backend `B`.
pub fn train_ppo<B: AutodiffBackend>(
    config: &PpoConfig,
    device: B::Device,
    backend_label: &str,
) -> (RiskPolicyNet<B::InnerBackend>, Vec<UpdateMetrics>) {
    assert!(
        (2..=6).contains(&config.players),
        "players must be between 2 and 6"
    );

    let league_cfg = LeagueConfig {
        players: config.players,
        pfsp_mode: config.pfsp_mode,
        pfsp_p: config.pfsp_p,
        k_factor: config.elo_k,
        opponent_k_factor: config.opponent_elo_k,
        max_snapshots: config.snapshot_pool_size,
        ..LeagueConfig::default()
    };
    let mut league = LeagueState::new(league_cfg);

    let mut env_pool = SelfPlayEnvPool::new(
        config.num_envs,
        config.players,
        config.max_turns,
        config.shaping_coef,
        config.draw_penalty,
        config.self_play_prob,
        &league,
        config.seed,
    );

    let topo = env_pool.topology().clone();
    let train_graph = GraphBuffers::<B>::from_topology(&topo, &device);
    let infer_graph = GraphBuffers::<B::InnerBackend>::from_topology(&topo, &device);

    let policy_cfg = RiskPolicyConfig::new()
        .with_hidden_dim(config.hidden_dim)
        .with_num_layers(config.num_layers);
    let mut model: RiskPolicyNet<B> = match &config.resume_from {
        Some(path) => {
            let trimmed = strip_mpk_extension(path);
            println!("Resuming policy weights from {}.mpk", trimmed.display());
            load_checkpoint::<B>(&policy_cfg, &trimmed, &device)
                .expect("failed to load resume checkpoint")
        }
        None => policy_cfg.init(&device),
    };

    let mut optim = AdamWConfig::new()
        .with_weight_decay(1e-4)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(config.max_grad_norm)))
        .init();

    let mut snapshots: HashMap<usize, RiskPolicyNet<B::InnerBackend>> = HashMap::new();
    let mut shuffle_rng = ChaCha8Rng::seed_from_u64(config.seed ^ 0x9E37_79B9_7F4A_7C15);

    let run_name = config
        .save_dir
        .as_ref()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("ppo")
        .to_string();

    let mut telemetry =
        config
            .save_dir
            .as_deref()
            .and_then(|dir| match TelemetryWriter::new(dir) {
                Ok(w) => Some(w),
                Err(err) => {
                    eprintln!(
                        "Warning: failed to initialize telemetry in {}: {err}",
                        dir.display()
                    );
                    None
                }
            });

    // Write initial league state immediately so live dashboards connect right away.
    league.record_elo_point(0, 0);
    if let Some(tw) = &telemetry {
        let report = league.build_report(
            &run_name,
            config.max_turns,
            backend_label,
            0,
            config.updates,
        );
        let _ = tw.write_league(&report);
    }

    println!(
        "Starting Riskbot PFSP+Elo PPO | backend={} | players={} | max_turns={} | envs={} | steps/upd={} | hidden={}x{} | pfsp={} (p={:.1}) | self_play={:.0}%",
        backend_label,
        config.players,
        config.max_turns,
        config.num_envs,
        config.steps_per_update,
        config.hidden_dim,
        config.num_layers,
        config.pfsp_mode,
        config.pfsp_p,
        config.self_play_prob * 100.0,
    );

    let mut total_env_steps = 0usize;
    let mut history = Vec::with_capacity(config.updates);
    let mut best_elo = f64::NEG_INFINITY;
    let start_all = Instant::now();

    for update in 1..=config.updates {
        let t0 = Instant::now();

        // Linear learning-rate schedule down to 25% of initial LR so late updates stay active.
        let progress = ((update - 1) as f64) / (config.updates.max(1) as f64);
        let lr = config.learning_rate * (1.0 - 0.75 * progress);

        // 1) Collect rollout using inference-only inner model (no autodiff graph).
        let valid_model = model.valid();
        let (mut samples, rstats) = env_pool.collect_rollout(
            &valid_model,
            &snapshots,
            &mut league,
            &infer_graph,
            &device,
            config.steps_per_update,
            config.gamma,
            config.gae_lambda,
        );
        total_env_steps += rstats.env_steps;

        // 2) Normalize advantages across the rollout buffer.
        normalize_advantages(&mut samples);

        // 3) PPO clipped surrogate updates over shuffled minibatches.
        let mut p_loss_sum = 0.0f32;
        let mut v_loss_sum = 0.0f32;
        let mut ent_sum = 0.0f32;
        let mut kl_sum = 0.0f32;
        let mut clip_sum = 0.0f32;
        let mut num_mbs = 0usize;

        let mut indices: Vec<usize> = (0..samples.len()).collect();
        'epochs: for _epoch in 0..config.ppo_epochs {
            indices.shuffle(&mut shuffle_rng);
            for chunk in indices.chunks(config.minibatch_size) {
                if chunk.len() < 4 {
                    continue;
                }
                let mb = build_minibatch::<B>(&samples, chunk, &device);
                let (new_logp, entropy, values) =
                    model.evaluate_actions(mb.nodes, mb.global, mb.mask, mb.actions, &train_graph);

                let log_ratio = (new_logp - mb.old_logp).clamp(-20.0, 20.0);
                let ratio = log_ratio.clone().exp();

                let surr1 = ratio.clone() * mb.adv.clone();
                let clipped = ratio
                    .clone()
                    .clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps);
                let surr2 = clipped * mb.adv;
                let policy_loss = -surr1.min_pair(surr2).mean();

                let value_loss = (values - mb.returns).powf_scalar(2.0).mean() * 0.5;
                let ent_mean = entropy.mean();

                let total_loss = policy_loss.clone() + value_loss.clone() * config.vf_coef
                    - ent_mean.clone() * config.ent_coef;

                let approx_kl_t = ((ratio.clone() - 1.0) - log_ratio).mean();
                let clip_mask = (ratio - 1.0).abs().greater_elem(config.clip_eps);
                let clip_frac_t = clip_mask.float().mean();

                let p_val = policy_loss.into_scalar().elem::<f32>();
                let v_val = value_loss.into_scalar().elem::<f32>();
                let e_val = ent_mean.into_scalar().elem::<f32>();
                let kl_val = approx_kl_t.into_scalar().elem::<f32>();
                let cf_val = clip_frac_t.into_scalar().elem::<f32>();

                // Always consume the autodiff graph via backward() so GPU buffers are freed immediately.
                let grads = total_loss.backward();
                if !p_val.is_finite() || !v_val.is_finite() || !e_val.is_finite() {
                    eprintln!("Warning: non-finite loss at update {update}, aborting epoch");
                    break 'epochs;
                }
                let grads = GradientsParams::from_grads(grads, &model);
                model = optim.step(lr, model, grads);

                p_loss_sum += p_val;
                v_loss_sum += v_val;
                ent_sum += e_val;
                kl_sum += kl_val;
                clip_sum += cf_val;
                num_mbs += 1;

                if config.target_kl > 0.0 && kl_val > config.target_kl * 1.5 {
                    break 'epochs;
                }
            }
        }

        let dt = t0.elapsed().as_secs_f64().max(1e-6);
        let denom = (num_mbs.max(1)) as f32;
        let win_frac = if rstats.episodes_finished > 0 {
            (rstats.episodes_won as f32) / (rstats.episodes_finished as f32)
        } else {
            0.0
        };
        let avg_turns = if rstats.episodes_finished > 0 {
            (rstats.total_turns as f32) / (rstats.episodes_finished as f32)
        } else {
            0.0
        };

        // 4) Periodic evaluation against Random, Heuristic, and Active League Snapshots.
        let mut eval_rand_wr = None;
        let mut eval_heur_wr = None;
        let mut eval_league_wr = None;
        let mut eval_rand_turns = None;
        let mut eval_heur_turns = None;

        let should_eval = config.eval_interval > 0
            && (update == 1 || update % config.eval_interval == 0 || update == config.updates);

        if should_eval {
            let eval_model = model.valid();
            let vs_rand = evaluate_policy(
                &eval_model,
                &topo,
                &infer_graph,
                &device,
                config.players,
                config.max_turns,
                config.eval_games,
                EvalOpponent::Random,
                true,
                config.seed + 10_000 + update as u64 * 97,
            );
            vs_rand.record_into_league(&mut league, OpponentKey::Random);
            eval_rand_wr = Some(vs_rand.win_rate());
            eval_rand_turns = Some(vs_rand.avg_turns);

            let vs_heur = evaluate_policy(
                &eval_model,
                &topo,
                &infer_graph,
                &device,
                config.players,
                config.max_turns,
                config.eval_games,
                EvalOpponent::Heuristic,
                true,
                config.seed + 20_000 + update as u64 * 97,
            );
            vs_heur.record_into_league(&mut league, OpponentKey::Heuristic);
            eval_heur_wr = Some(vs_heur.win_rate());
            eval_heur_turns = Some(vs_heur.avg_turns);

            let active_snaps = league.active_snapshot_ids();
            if !active_snaps.is_empty() {
                let league_res = evaluate_against_snapshots(
                    &eval_model,
                    &snapshots,
                    &active_snaps,
                    &mut league,
                    &topo,
                    &infer_graph,
                    &device,
                    config.players,
                    config.max_turns,
                    config.eval_games,
                    config.seed + 30_000 + update as u64 * 97,
                );
                eval_league_wr = Some(league_res.win_rate());
            }

            let league_str = eval_league_wr
                .map(|wr| format!(" | vs League: {:5.1}%", wr * 100.0))
                .unwrap_or_default();
            println!(
                "  [eval @ upd {:4}] Elo {:6.1} | vs Random: {:5.1}% ({}/{}, draw {}, turns {:.0}) | vs Heuristic: {:5.1}% ({}/{}, draw {}, turns {:.0}){}",
                update,
                league.learner_elo(),
                vs_rand.win_rate() * 100.0,
                vs_rand.wins,
                vs_rand.games,
                vs_rand.draws,
                vs_rand.avg_turns,
                vs_heur.win_rate() * 100.0,
                vs_heur.wins,
                vs_heur.games,
                vs_heur.draws,
                vs_heur.avg_turns,
                league_str,
            );

            if let Some(dir) = &config.save_dir {
                let latest_path = dir.join("latest");
                if let Err(err) = save_checkpoint(&eval_model, &latest_path) {
                    eprintln!("Warning: failed to save latest checkpoint: {err}");
                }
                if league.learner_elo() >= best_elo {
                    best_elo = league.learner_elo();
                    let best_path = dir.join("best");
                    if let Err(err) = save_checkpoint(&eval_model, &best_path) {
                        eprintln!("Warning: failed to save best checkpoint: {err}");
                    }
                }
            }
        }

        // 5) Add new historical snapshot to the PFSP league pool (with diversity-preserving eviction).
        if config.snapshot_interval > 0 && update % config.snapshot_interval == 0 {
            let (snap_id, evicted) = league.add_snapshot(update);
            snapshots.insert(snap_id, model.valid());
            if let Some(evicted_id) = evicted {
                snapshots.remove(&evicted_id);
            }
        }

        // 6) Record Elo point and flush live telemetry (`metrics.jsonl` + atomic `league.json`).
        league.record_elo_point(update, total_env_steps);
        let learner_elo = league.learner_elo();

        let metrics = UpdateMetrics {
            update,
            total_env_steps,
            policy_loss: p_loss_sum / denom,
            value_loss: v_loss_sum / denom,
            entropy: ent_sum / denom,
            approx_kl: kl_sum / denom,
            clip_frac: clip_sum / denom,
            steps_per_sec: (rstats.env_steps as f64) / dt,
            episodes_finished: rstats.episodes_finished,
            win_frac,
            avg_turns,
            learner_elo,
        };

        if let Some(tw) = &mut telemetry {
            let rec = MetricsRecord {
                update: metrics.update,
                total_env_steps: metrics.total_env_steps,
                elapsed_secs: start_all.elapsed().as_secs_f64(),
                steps_per_sec: metrics.steps_per_sec,
                learning_rate: lr,
                policy_loss: metrics.policy_loss,
                value_loss: metrics.value_loss,
                entropy: metrics.entropy,
                approx_kl: metrics.approx_kl,
                clip_frac: metrics.clip_frac,
                episodes_finished: metrics.episodes_finished,
                win_frac: metrics.win_frac,
                avg_turns: metrics.avg_turns,
                learner_elo,
                pool_size: snapshots.len(),
                eval_random_win_rate: eval_rand_wr,
                eval_heuristic_win_rate: eval_heur_wr,
                eval_league_win_rate: eval_league_wr,
                eval_random_avg_turns: eval_rand_turns,
                eval_heuristic_avg_turns: eval_heur_turns,
            };
            if let Err(err) = tw.append_metrics(&rec) {
                eprintln!("Warning: failed to write metrics.jsonl: {err}");
            }
            let report = league.build_report(
                &run_name,
                config.max_turns,
                backend_label,
                update,
                config.updates,
            );
            if let Err(err) = tw.write_league(&report) {
                eprintln!("Warning: failed to write league.json: {err}");
            }
        }

        println!(
            "upd {:4}/{:<4} | steps {:7} ({:6.0}/s) | elo {:6.1} | pool {:2} | pi {:+.4} | vf {:.4} | ent {:.3} | kl {:.4} | clip {:.2} | eps {:2} (win {:.0}%, turns {:.0})",
            metrics.update,
            config.updates,
            metrics.total_env_steps,
            metrics.steps_per_sec,
            metrics.learner_elo,
            snapshots.len(),
            metrics.policy_loss,
            metrics.value_loss,
            metrics.entropy,
            metrics.approx_kl,
            metrics.clip_frac,
            metrics.episodes_finished,
            metrics.win_frac * 100.0,
            metrics.avg_turns,
        );
        history.push(metrics);
    }

    println!(
        "Training complete in {:.1}s ({} total env steps, final Elo {:.1}).",
        start_all.elapsed().as_secs_f64(),
        total_env_steps,
        league.learner_elo(),
    );

    (model.valid(), history)
}

/// Evaluate the current learner against a balanced round-robin of active league snapshots,
/// recording each game into `league` to calibrate Elo ratings across the pool.
#[allow(clippy::too_many_arguments)]
fn evaluate_against_snapshots<B: Backend>(
    model: &RiskPolicyNet<B>,
    snapshots: &HashMap<usize, RiskPolicyNet<B>>,
    active_snap_ids: &[usize],
    league: &mut LeagueState,
    topo: &BoardTopology,
    graph: &GraphBuffers<B>,
    device: &B::Device,
    players: usize,
    max_turns: u32,
    num_games: usize,
    seed: u64,
) -> EvalResult {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut games: Vec<(Game, usize, usize)> = (0..num_games)
        .map(|i| {
            let snap_id = active_snap_ids[i % active_snap_ids.len()];
            let game = Game::new(players, max_turns, Some(seed + i as u64));
            let policy_seat = i % players;
            (game, policy_seat, snap_id)
        })
        .collect();

    loop {
        let mut learner_indices = Vec::new();
        let mut snap_buckets: Vec<(usize, Vec<usize>)> = Vec::new();

        for (idx, (game, policy_seat, snap_id)) in games.iter().enumerate() {
            if game.phase == Phase::Ended {
                continue;
            }
            if game.active == *policy_seat {
                learner_indices.push(idx);
            } else if let Some((_, bucket)) =
                snap_buckets.iter_mut().find(|(id, _)| *id == *snap_id)
            {
                bucket.push(idx);
            } else {
                snap_buckets.push((*snap_id, vec![idx]));
            }
        }

        if learner_indices.is_empty() && snap_buckets.is_empty() {
            break;
        }

        if !learner_indices.is_empty() {
            let b = learner_indices.len();
            let mut nodes_flat = Vec::with_capacity(b * N * NODE_FEAT_DIM);
            let mut global_flat = Vec::with_capacity(b * GLOBAL_FEAT_DIM);
            let mut mask_flat = Vec::with_capacity(b * ACTION_COUNT);
            let mut masks = Vec::with_capacity(b);
            for &idx in &learner_indices {
                let (game, seat, _) = &games[idx];
                let obs = encode_observation(game, *seat, topo);
                nodes_flat.extend_from_slice(&obs.nodes);
                global_flat.extend_from_slice(&obs.global);
                mask_flat.extend_from_slice(&obs.mask);
                masks.push(obs.mask);
            }
            let (probs_flat, _) =
                model.infer_batch(nodes_flat, global_flat, mask_flat, b, graph, device);
            for (i, (&idx, mask)) in learner_indices.iter().zip(masks).enumerate() {
                let probs = &probs_flat[i * ACTION_COUNT..(i + 1) * ACTION_COUNT];
                // Low-noise epsilon-greedy (eps=0.08) breaks mirror symmetry while avoiding slow stochastic draws.
                let act_id = if rng.random::<f32>() < 0.08 {
                    sample_masked_categorical(probs, &mask, &mut rng).0
                } else {
                    greedy_masked_action(probs, &mask)
                };
                games[idx].0.step(Action::decode(act_id).unwrap());
            }
        }

        for (snap_id, indices) in snap_buckets {
            let opp_model = &snapshots[&snap_id];
            let b = indices.len();
            let mut nodes_flat = Vec::with_capacity(b * N * NODE_FEAT_DIM);
            let mut global_flat = Vec::with_capacity(b * GLOBAL_FEAT_DIM);
            let mut mask_flat = Vec::with_capacity(b * ACTION_COUNT);
            let mut masks = Vec::with_capacity(b);
            for &idx in &indices {
                let (game, _, _) = &games[idx];
                let obs = encode_observation(game, game.active, topo);
                nodes_flat.extend_from_slice(&obs.nodes);
                global_flat.extend_from_slice(&obs.global);
                mask_flat.extend_from_slice(&obs.mask);
                masks.push(obs.mask);
            }
            let (probs_flat, _) =
                opp_model.infer_batch(nodes_flat, global_flat, mask_flat, b, graph, device);
            for (i, (&idx, mask)) in indices.iter().zip(masks).enumerate() {
                let probs = &probs_flat[i * ACTION_COUNT..(i + 1) * ACTION_COUNT];
                let act_id = if rng.random::<f32>() < 0.08 {
                    sample_masked_categorical(probs, &mask, &mut rng).0
                } else {
                    greedy_masked_action(probs, &mask)
                };
                games[idx].0.step(Action::decode(act_id).unwrap());
            }
        }
    }

    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut draws = 0usize;
    let mut total_turns = 0u64;

    for (game, policy_seat, snap_id) in games {
        total_turns += u64::from(game.turns);
        let outcome = match game.winner {
            Some(w) if w == policy_seat => {
                wins += 1;
                MatchOutcome::LearnerWin
            }
            Some(_) => {
                losses += 1;
                MatchOutcome::LearnerLoss
            }
            None => {
                draws += 1;
                MatchOutcome::Draw
            }
        };
        league.record_match(OpponentKey::Snapshot(snap_id), outcome);
    }

    EvalResult {
        games: num_games,
        wins,
        losses,
        draws,
        avg_turns: (total_turns as f32) / (num_games.max(1) as f32),
    }
}

/// Evaluate a trained policy against Random or Heuristic opponents across balanced seats.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_policy<B: Backend>(
    model: &RiskPolicyNet<B>,
    topo: &BoardTopology,
    graph: &GraphBuffers<B>,
    device: &B::Device,
    players: usize,
    max_turns: u32,
    num_games: usize,
    opponent: EvalOpponent,
    greedy: bool,
    seed: u64,
) -> EvalResult {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut games: Vec<(Game, usize)> = (0..num_games)
        .map(|i| {
            let game = Game::new(players, max_turns, Some(seed + i as u64));
            let policy_seat = i % players;
            (game, policy_seat)
        })
        .collect();

    loop {
        let mut pending_indices = Vec::new();
        for (idx, (game, policy_seat)) in games.iter_mut().enumerate() {
            while game.phase != Phase::Ended && game.active != *policy_seat {
                let act_id = match opponent {
                    EvalOpponent::Random => random_action(game, &mut rng),
                    EvalOpponent::Heuristic => heuristic_action(game, topo, &mut rng),
                };
                game.step(Action::decode(act_id).unwrap());
            }
            if game.phase != Phase::Ended {
                pending_indices.push(idx);
            }
        }
        if pending_indices.is_empty() {
            break;
        }

        let b = pending_indices.len();
        let mut nodes_flat = Vec::with_capacity(b * N * NODE_FEAT_DIM);
        let mut global_flat = Vec::with_capacity(b * GLOBAL_FEAT_DIM);
        let mut mask_flat = Vec::with_capacity(b * ACTION_COUNT);
        let mut masks = Vec::with_capacity(b);

        for &idx in &pending_indices {
            let (game, policy_seat) = &games[idx];
            let obs = encode_observation(game, *policy_seat, topo);
            nodes_flat.extend_from_slice(&obs.nodes);
            global_flat.extend_from_slice(&obs.global);
            mask_flat.extend_from_slice(&obs.mask);
            masks.push(obs.mask);
        }

        let (probs_flat, _) =
            model.infer_batch(nodes_flat, global_flat, mask_flat, b, graph, device);

        for (i, (&idx, mask)) in pending_indices.iter().zip(masks).enumerate() {
            let probs = &probs_flat[i * ACTION_COUNT..(i + 1) * ACTION_COUNT];
            let act_id = if greedy {
                greedy_masked_action(probs, &mask)
            } else {
                sample_masked_categorical(probs, &mask, &mut rng).0
            };
            games[idx].0.step(Action::decode(act_id).unwrap());
        }
    }

    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut draws = 0usize;
    let mut total_turns = 0u64;

    for (game, policy_seat) in games {
        total_turns += u64::from(game.turns);
        match game.winner {
            Some(w) if w == policy_seat => wins += 1,
            Some(_) => losses += 1,
            None => draws += 1,
        }
    }

    EvalResult {
        games: num_games,
        wins,
        losses,
        draws,
        avg_turns: (total_turns as f32) / (num_games.max(1) as f32),
    }
}

/// Save policy weights to `<path>.mpk`.
pub fn save_checkpoint<B: Backend>(
    model: &RiskPolicyNet<B>,
    path: &Path,
) -> Result<(), burn::record::RecorderError> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model.clone().save_file(path, &recorder)
}

/// Load policy weights from `<path>.mpk`.
pub fn load_checkpoint<B: Backend>(
    config: &RiskPolicyConfig,
    path: &Path,
    device: &B::Device,
) -> Result<RiskPolicyNet<B>, burn::record::RecorderError> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    config.init(device).load_file(path, &recorder, device)
}

fn strip_mpk_extension(path: &Path) -> PathBuf {
    if path.extension().and_then(|e| e.to_str()) == Some("mpk") {
        path.with_extension("")
    } else {
        path.to_path_buf()
    }
}

fn normalize_advantages(samples: &mut [TrainingSample]) {
    if samples.len() < 2 {
        return;
    }
    let n = samples.len() as f32;
    let mean = samples.iter().map(|s| s.advantage).sum::<f32>() / n;
    let var = samples
        .iter()
        .map(|s| {
            let d = s.advantage - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    let std = var.sqrt() + 1e-8;
    for s in samples {
        s.advantage = (s.advantage - mean) / std;
    }
}

struct Minibatch<B: Backend> {
    nodes: Tensor<B, 3>,
    global: Tensor<B, 2>,
    mask: Tensor<B, 2, Bool>,
    actions: Tensor<B, 2, Int>,
    old_logp: Tensor<B, 2>,
    adv: Tensor<B, 2>,
    returns: Tensor<B, 2>,
}

fn build_minibatch<B: Backend>(
    samples: &[TrainingSample],
    indices: &[usize],
    device: &B::Device,
) -> Minibatch<B> {
    let b = indices.len();
    let mut nodes_v = Vec::with_capacity(b * N * NODE_FEAT_DIM);
    let mut global_v = Vec::with_capacity(b * GLOBAL_FEAT_DIM);
    let mut mask_v = Vec::with_capacity(b * ACTION_COUNT);
    let mut act_v = Vec::with_capacity(b);
    let mut old_lp_v = Vec::with_capacity(b);
    let mut adv_v = Vec::with_capacity(b);
    let mut ret_v = Vec::with_capacity(b);

    for &idx in indices {
        let s = &samples[idx];
        nodes_v.extend_from_slice(&s.nodes);
        global_v.extend_from_slice(&s.global);
        mask_v.extend_from_slice(&s.mask);
        act_v.push(s.action);
        old_lp_v.push(s.old_log_prob);
        adv_v.push(s.advantage);
        ret_v.push(s.target_return);
    }

    Minibatch {
        nodes: Tensor::from_data(TensorData::new(nodes_v, [b, N, NODE_FEAT_DIM]), device),
        global: Tensor::from_data(TensorData::new(global_v, [b, GLOBAL_FEAT_DIM]), device),
        mask: Tensor::from_bool(TensorData::new(mask_v, [b, ACTION_COUNT]), device),
        actions: Tensor::from_ints(TensorData::new(act_v, [b, 1]), device),
        old_logp: Tensor::from_data(TensorData::new(old_lp_v, [b, 1]), device),
        adv: Tensor::from_data(TensorData::new(adv_v, [b, 1]), device),
        returns: Tensor::from_data(TensorData::new(ret_v, [b, 1]), device),
    }
}
