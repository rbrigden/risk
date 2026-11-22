//! GPU-accelerated self-play reinforcement learning environment and Masked PPO trainer.

pub mod bots;
pub mod env;
pub mod features;
pub mod league;
pub mod model;
pub mod ppo;
pub mod telemetry;

#[cfg(test)]
mod tests {
    use super::env::{Transition, compute_gae};
    use super::features::{BoardTopology, board_potential, encode_observation};
    use super::league::PfspMode;
    use super::model::{GraphBuffers, RiskPolicyConfig};
    use super::ppo::{PpoConfig, train_ppo};
    use crate::game::{ACTION_COUNT, Action, Game, N};
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};

    #[test]
    fn observation_encoding_and_mask_match_game_rules() {
        let topo = BoardTopology::new();
        for players in 2..=6 {
            let game = Game::new(players, 50, Some(players as u64 + 7));
            let active = game.active;
            let obs = encode_observation(&game, active, &topo);
            assert!(obs.nodes.iter().all(|v| v.is_finite()));
            assert!(obs.global.iter().all(|v| v.is_finite()));
            for id in 0..ACTION_COUNT {
                let legal = game.legal(Action::decode(id).unwrap());
                assert_eq!(obs.mask[id], legal);
            }
            let pot = board_potential(&game, active);
            assert!((0.0..=1.0).contains(&pot));
        }
    }

    #[test]
    fn gae_matches_discounted_returns() {
        let zero_nodes = [0.0; N * super::features::NODE_FEAT_DIM];
        let zero_global = [0.0; super::features::GLOBAL_FEAT_DIM];
        let mut mask = [false; ACTION_COUNT];
        mask[0] = true;

        let traj = vec![
            Transition {
                nodes: zero_nodes,
                global: zero_global,
                mask,
                action: 0,
                log_prob: -0.5,
                value: 0.2,
                reward: 0.5,
                done: false,
            },
            Transition {
                nodes: zero_nodes,
                global: zero_global,
                mask,
                action: 0,
                log_prob: -0.5,
                value: 0.4,
                reward: 1.0,
                done: true,
            },
        ];

        let mut samples = Vec::new();
        compute_gae(&traj, 0.99, 0.9, 1.0, &mut samples);
        assert_eq!(samples.len(), 2);
        // With lambda = 1.0 and done=true at t=1:
        // R_1 = 1.0, adv_1 = 1.0 - 0.4 = 0.6
        // R_0 = 0.5 + 0.9 * 1.0 = 1.4, adv_0 = 1.4 - 0.2 = 1.2
        assert!((samples[1].target_return - 1.0).abs() < 1e-5);
        assert!((samples[1].advantage - 0.6).abs() < 1e-5);
        assert!((samples[0].target_return - 1.4).abs() < 1e-5);
        assert!((samples[0].advantage - 1.2).abs() < 1e-5);
    }

    #[test]
    fn gnn_policy_and_ppo_smoke_test() {
        let device = NdArrayDevice::Cpu;
        let topo = BoardTopology::new();
        let graph = GraphBuffers::<NdArray>::from_topology(&topo, &device);
        let model = RiskPolicyConfig::new()
            .with_hidden_dim(16)
            .with_num_layers(1)
            .init::<NdArray>(&device);

        let game = Game::new(2, 20, Some(99));
        let obs = encode_observation(&game, game.active, &topo);
        let (probs, vals) = model.infer_batch(
            obs.nodes.to_vec(),
            obs.global.to_vec(),
            obs.mask.to_vec(),
            1,
            &graph,
            &device,
        );
        assert_eq!(probs.len(), ACTION_COUNT);
        assert_eq!(vals.len(), 1);
        assert!(vals[0].is_finite());
        let legal_sum: f32 = probs
            .iter()
            .zip(obs.mask.iter())
            .filter_map(|(&p, &m)| m.then_some(p))
            .sum();
        assert!((legal_sum - 1.0).abs() < 1e-4);

        // End-to-end 2-update PFSP + Elo PPO smoke test on CPU backend.
        let cfg = PpoConfig {
            players: 2,
            max_turns: 30,
            num_envs: 4,
            steps_per_update: 64,
            updates: 2,
            ppo_epochs: 1,
            minibatch_size: 32,
            hidden_dim: 16,
            num_layers: 1,
            eval_interval: 2,
            eval_games: 2,
            snapshot_interval: 1,
            pfsp_mode: PfspMode::Hard,
            save_dir: None,
            ..Default::default()
        };
        let (_trained, history) = train_ppo::<Autodiff<NdArray>>(&cfg, device, "NdArray CPU test");
        assert_eq!(history.len(), 2);
        assert!(history.iter().all(|m| m.policy_loss.is_finite()
            && m.value_loss.is_finite()
            && m.learner_elo.is_finite()));
    }
}
