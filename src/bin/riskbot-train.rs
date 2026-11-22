#![recursion_limit = "256"]
//! CLI entrypoint for Riskbot GPU/CPU PFSP + Elo self-play PPO training.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, NdArray, Wgpu};
use riskbot::train::ppo::{PpoConfig, train_ppo};
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq, Eq)]
enum DeviceChoice {
    Gpu,
    Cpu,
}

fn print_usage() {
    eprintln!(
        "\
Usage: riskbot-train [OPTIONS]

Options:
  --device <gpu|cpu>             Compute backend (default: gpu via WGPU/Vulkan)
  --players <2..=6>              Number of players (default: 2)
  --max-turns <N>                Turn limit per game (default: 200)
  --envs <N>                     Number of parallel environments (default: 64)
  --steps-per-update <N>         Learner rollout steps per PPO update (default: 4096)
  --updates <N>                  Total PPO updates (default: 100)
  --epochs <N>                   PPO epochs per update (default: 4)
  --minibatch <N>                PPO minibatch size (default: 256)
  --hidden <N>                   GNN hidden dimension (default: 64)
  --layers <N>                   GNN message-passing layers (default: 2)
  --lr <FLOAT>                   Initial learning rate (default: 8e-4)
  --gamma <FLOAT>                Discount factor gamma (default: 0.998)
  --gae-lambda <FLOAT>           GAE lambda (default: 0.97)
  --shaping <FLOAT>              Potential-based shaping weight (default: 1.0)
  --ent-coef <FLOAT>             Entropy bonus coefficient (default: 0.002)
  --self-play-prob <0..1>        Pure latest-vs-latest fraction (default: 0.40; rest uses PFSP)
  --pfsp-mode <hard|variance|uniform>  PFSP weighting strategy (default: variance)
  --pfsp-p <FLOAT>               PFSP hard-mode exponent p in (1 - wr)^p (default: 2.0)
  --elo-k <FLOAT>                Learner Elo K-factor (default: 24.0)
  --snapshot-interval <N>        Add snapshot to league every N updates (default: 10)
  --snapshot-pool <N>            Max active historical snapshots in league (default: 12)
  --eval-interval <N>            Evaluate every N updates (default: 10)
  --eval-games <N>               Evaluation games per opponent (default: 16)
  --save-dir <PATH|none>         Checkpoint & telemetry directory (default: runs/ppo)
  --resume <PATH>                Resume policy weights from a saved .mpk checkpoint
  --seed <N>                     Random seed (default: 42)
  -h, --help                     Show this help"
    );
}

fn parse_args() -> Result<(DeviceChoice, PpoConfig), String> {
    let mut device = DeviceChoice::Gpu;
    let mut cfg = PpoConfig::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        let mut next_val = || {
            args.next()
                .ok_or_else(|| format!("missing value for {arg}"))
        };
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            "--device" => match next_val()?.as_str() {
                "gpu" | "wgpu" | "vulkan" => device = DeviceChoice::Gpu,
                "cpu" | "ndarray" => device = DeviceChoice::Cpu,
                other => return Err(format!("unknown device '{other}' (expected gpu or cpu)")),
            },
            "--players" => {
                cfg.players = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --players: {e}"))?;
            }
            "--max-turns" => {
                cfg.max_turns = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --max-turns: {e}"))?;
            }
            "--envs" => {
                cfg.num_envs = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --envs: {e}"))?;
            }
            "--steps-per-update" => {
                cfg.steps_per_update = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --steps-per-update: {e}"))?;
            }
            "--updates" => {
                cfg.updates = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --updates: {e}"))?;
            }
            "--epochs" => {
                cfg.ppo_epochs = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --epochs: {e}"))?;
            }
            "--minibatch" => {
                cfg.minibatch_size = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --minibatch: {e}"))?;
            }
            "--hidden" => {
                cfg.hidden_dim = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --hidden: {e}"))?;
            }
            "--layers" => {
                cfg.num_layers = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --layers: {e}"))?;
            }
            "--lr" => {
                cfg.learning_rate = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --lr: {e}"))?;
            }
            "--gamma" => {
                cfg.gamma = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --gamma: {e}"))?;
            }
            "--gae-lambda" => {
                cfg.gae_lambda = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --gae-lambda: {e}"))?;
            }
            "--shaping" => {
                cfg.shaping_coef = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --shaping: {e}"))?;
            }
            "--ent-coef" => {
                cfg.ent_coef = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --ent-coef: {e}"))?;
            }
            "--self-play-prob" => {
                cfg.self_play_prob = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --self-play-prob: {e}"))?;
            }
            "--pfsp-mode" => {
                cfg.pfsp_mode = next_val()?.parse()?;
            }
            "--pfsp-p" => {
                cfg.pfsp_p = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --pfsp-p: {e}"))?;
            }
            "--elo-k" => {
                cfg.elo_k = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --elo-k: {e}"))?;
            }
            "--snapshot-interval" => {
                cfg.snapshot_interval = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --snapshot-interval: {e}"))?;
            }
            "--snapshot-pool" => {
                cfg.snapshot_pool_size = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --snapshot-pool: {e}"))?;
            }
            "--eval-interval" => {
                cfg.eval_interval = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --eval-interval: {e}"))?;
            }
            "--eval-games" => {
                cfg.eval_games = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --eval-games: {e}"))?;
            }
            "--save-dir" => {
                let val = next_val()?;
                cfg.save_dir = if val == "none" {
                    None
                } else {
                    Some(PathBuf::from(val))
                };
            }
            "--resume" => {
                cfg.resume_from = Some(PathBuf::from(next_val()?));
            }
            "--seed" => {
                cfg.seed = next_val()?
                    .parse()
                    .map_err(|e| format!("invalid --seed: {e}"))?;
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }

    Ok((device, cfg))
}

fn main() {
    let (device, cfg) = match parse_args() {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("Error: {err}");
            print_usage();
            std::process::exit(2);
        }
    };

    match device {
        DeviceChoice::Gpu => {
            let dev = WgpuDevice::default();
            train_ppo::<Autodiff<Wgpu>>(&cfg, dev, "WGPU/Vulkan GPU");
        }
        DeviceChoice::Cpu => {
            let dev = NdArrayDevice::Cpu;
            train_ppo::<Autodiff<NdArray>>(&cfg, dev, "NdArray CPU");
        }
    }
}
