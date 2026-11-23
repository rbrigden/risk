//! Export Riskbot board geometry, trained GNN policy weights, and numerical parity
//! test vectors for external/web arena clients.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use riskbot::game::{ACTION_COUNT, Action, EDGES, Game, N, Phase, REGIONS};
use riskbot::train::bots::heuristic_action;
use riskbot::train::env::greedy_masked_action;
use riskbot::train::features::{
    BoardTopology, GLOBAL_FEAT_DIM, NODE_FEAT_DIM, NUM_EDGES, encode_observation,
};
use riskbot::train::model::{
    GraphBlockWeights, GraphBuffers, LayerNormWeights, LinearWeights, RiskPolicyNet,
    RiskPolicyWeights,
};

const DEFAULT_OUT_DIR: &str = "dist/arena-data";
const DEFAULT_CHECKPOINT: &str = "runs/ppo/best.mpk";
const DEFAULT_SAMPLING: &str = "balanced";

const BASE64_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

struct CliConfig {
    out_dir: PathBuf,
    checkpoint: PathBuf,
    policy_id: Option<String>,
    policy_name: Option<String>,
    sampling: String,
}

fn print_help() {
    println!("Usage: riskbot-export [OPTIONS]");
    println!();
    println!("Export board geometry, trained policy weights, and parity vectors as JSON.");
    println!();
    println!("Options:");
    println!("  --out-dir <DIR>         Output data directory (default: {DEFAULT_OUT_DIR})");
    println!("  --checkpoint <PATH>     Burn .mpk checkpoint path (default: {DEFAULT_CHECKPOINT})");
    println!("  --policy-id <ID>        Override exported policy ID (default: derived from path)");
    println!("  --policy-name <NAME>    Override display name (default: derived from policy ID)");
    println!(
        "  --sampling <MODE>       Default sampling: greedy | balanced | stochastic (default: {DEFAULT_SAMPLING})"
    );
    println!("  -h, --help              Show this help message");
}

fn parse_args() -> Result<Option<CliConfig>, String> {
    let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
    let mut checkpoint = PathBuf::from(DEFAULT_CHECKPOINT);
    let mut policy_id = None;
    let mut policy_name = None;
    let mut sampling = DEFAULT_SAMPLING.to_string();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                return Ok(None);
            }
            "--out-dir" => {
                let val = args
                    .next()
                    .ok_or_else(|| "Error: --out-dir requires a directory path.".to_string())?;
                out_dir = PathBuf::from(val);
            }
            "--checkpoint" => {
                let val = args
                    .next()
                    .ok_or_else(|| "Error: --checkpoint requires a .mpk file path.".to_string())?;
                checkpoint = PathBuf::from(val);
            }
            "--policy-id" => {
                let val = args
                    .next()
                    .ok_or_else(|| "Error: --policy-id requires a value.".to_string())?;
                policy_id = Some(val);
            }
            "--policy-name" => {
                let val = args
                    .next()
                    .ok_or_else(|| "Error: --policy-name requires a value.".to_string())?;
                policy_name = Some(val);
            }
            "--sampling" => {
                sampling = args
                    .next()
                    .ok_or_else(|| "Error: --sampling requires a value.".to_string())?;
            }
            other if other.starts_with("--out-dir=") => {
                out_dir = PathBuf::from(&other["--out-dir=".len()..]);
            }
            other if other.starts_with("--checkpoint=") => {
                checkpoint = PathBuf::from(&other["--checkpoint=".len()..]);
            }
            other if other.starts_with("--policy-id=") => {
                policy_id = Some(other["--policy-id=".len()..].to_string());
            }
            other if other.starts_with("--policy-name=") => {
                policy_name = Some(other["--policy-name=".len()..].to_string());
            }
            other if other.starts_with("--sampling=") => {
                sampling = other["--sampling=".len()..].to_string();
            }
            other => {
                return Err(format!(
                    "Error: unknown argument '{other}'. Run with --help for usage."
                ));
            }
        }
    }

    let sampling_norm = sampling.trim().to_ascii_lowercase();
    if !matches!(sampling_norm.as_str(), "greedy" | "balanced" | "stochastic") {
        return Err(format!(
            "Error: invalid --sampling '{sampling}'; expected 'greedy', 'balanced', or 'stochastic'."
        ));
    }

    Ok(Some(CliConfig {
        out_dir,
        checkpoint,
        policy_id,
        policy_name,
        sampling: sampling_norm,
    }))
}

#[derive(Debug, Default)]
struct LeagueSummary {
    run_name: Option<String>,
    players: Option<usize>,
    current_update: Option<usize>,
    total_steps: Option<u64>,
    learner_elo: Option<f64>,
    member_elo: Option<f64>,
}

fn read_sibling_league(checkpoint: &Path, stem: &str) -> LeagueSummary {
    let Some(parent) = checkpoint.parent() else {
        return LeagueSummary::default();
    };
    let league_path = parent.join("league.json");
    let Ok(raw) = fs::read_to_string(&league_path) else {
        return LeagueSummary::default();
    };
    let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return LeagueSummary::default();
    };

    let run_name = val
        .get("run_name")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned);
    let players = val
        .get("players")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let current_update = val
        .get("current_update")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let learner_elo = val
        .get("learner_elo")
        .and_then(|v| v.as_f64())
        .filter(|v| v.is_finite());
    let total_steps = val
        .get("elo_history")
        .or_else(|| val.get("history"))
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.last())
        .and_then(|entry| entry.get("steps"))
        .and_then(|v| v.as_u64());
    let member_elo = val
        .get("members")
        .and_then(|v| v.as_array())
        .and_then(|members| {
            members.iter().find_map(|m| {
                let id = m.get("id").and_then(|v| v.as_str())?;
                if id == stem {
                    m.get("elo")
                        .and_then(|v| v.as_f64())
                        .filter(|v| v.is_finite())
                } else {
                    None
                }
            })
        });

    LeagueSummary {
        run_name,
        players,
        current_update,
        total_steps,
        learner_elo,
        member_elo,
    }
}

fn sanitize_slug(raw: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash && !out.is_empty() {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_end_matches('-').to_string()
}

fn derive_policy_id(checkpoint: &Path, league: &LeagueSummary) -> String {
    let stem = checkpoint
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("policy");
    let run = league.run_name.as_deref().or_else(|| {
        checkpoint
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
    });

    match run {
        Some(r) if !r.eq_ignore_ascii_case(stem) => sanitize_slug(&format!("{r}-{stem}")),
        _ => sanitize_slug(stem),
    }
}

fn title_case_from_id(id: &str) -> String {
    id.split(['-', '_'])
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => {
                    let mut word = first.to_uppercase().to_string();
                    word.push_str(chars.as_str());
                    word
                }
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_with_commas(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (idx, ch) in digits.chars().rev().enumerate() {
        if idx > 0 && idx % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

fn format_steps(steps: u64) -> String {
    if steps >= 1_000_000 {
        format!("{:.1}M steps", (steps as f64) / 1_000_000.0)
    } else if steps >= 1_000 {
        format!("{:.1}k steps", (steps as f64) / 1_000.0)
    } else {
        format!("{steps} steps")
    }
}

fn build_description(league: &LeagueSummary, hidden_dim: usize, num_layers: usize) -> String {
    let mut parts = Vec::new();
    if let Some(updates) = league.current_update {
        parts.push(format!("{} PPO updates", format_with_commas(updates)));
    }
    if let Some(steps) = league.total_steps {
        parts.push(format_steps(steps));
    }
    if let Some(players) = league.players {
        parts.push(format!("{players}P PFSP self-play"));
    }
    if let Some(elo) = league.member_elo.or(league.learner_elo) {
        parts.push(format!("Elo {:.0}", elo.round()));
    }
    if parts.is_empty() {
        format!("{hidden_dim}×{num_layers} Graph Neural Network actor-critic policy.")
    } else {
        format!(
            "{hidden_dim}×{num_layers} GNN policy trained via {} ({}).",
            parts.remove(0),
            parts.join(", ")
        )
    }
}

fn encode_base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = u32::from(*chunk.get(1).unwrap_or(&0));
        let b2 = u32::from(*chunk.get(2).unwrap_or(&0));
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(BASE64_TABLE[((triple >> 18) & 0x3F) as usize] as char);
        out.push(BASE64_TABLE[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(BASE64_TABLE[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(BASE64_TABLE[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn encode_f32_le_b64(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &v in values {
        assert!(v.is_finite(), "cannot export non-finite weight value {v}");
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    encode_base64(&bytes)
}

#[derive(Serialize)]
struct TensorJson {
    shape: Vec<usize>,
    data_b64: String,
}

impl TensorJson {
    fn from_1d(values: &[f32]) -> Self {
        Self {
            shape: vec![values.len()],
            data_b64: encode_f32_le_b64(values),
        }
    }

    fn from_2d(shape: [usize; 2], values: &[f32]) -> Self {
        assert_eq!(
            shape[0] * shape[1],
            values.len(),
            "2D tensor length mismatch for shape {shape:?}"
        );
        Self {
            shape: shape.to_vec(),
            data_b64: encode_f32_le_b64(values),
        }
    }
}

#[derive(Serialize)]
struct LinearJson {
    weight: TensorJson,
    bias: Option<TensorJson>,
}

impl From<&LinearWeights> for LinearJson {
    fn from(w: &LinearWeights) -> Self {
        Self {
            weight: TensorJson::from_2d(w.shape, &w.weight),
            bias: w.bias.as_deref().map(TensorJson::from_1d),
        }
    }
}

#[derive(Serialize)]
struct LayerNormJson {
    epsilon: f64,
    gamma: TensorJson,
    beta: Option<TensorJson>,
}

impl From<&LayerNormWeights> for LayerNormJson {
    fn from(ln: &LayerNormWeights) -> Self {
        Self {
            epsilon: ln.epsilon,
            gamma: TensorJson::from_1d(&ln.gamma),
            beta: ln.beta.as_deref().map(TensorJson::from_1d),
        }
    }
}

#[derive(Serialize)]
struct GraphBlockJson {
    w_self: LinearJson,
    w_neigh: LinearJson,
    norm: LayerNormJson,
}

impl From<&GraphBlockWeights> for GraphBlockJson {
    fn from(b: &GraphBlockWeights) -> Self {
        Self {
            w_self: LinearJson::from(&b.w_self),
            w_neigh: LinearJson::from(&b.w_neigh),
            norm: LayerNormJson::from(&b.norm),
        }
    }
}

#[derive(Serialize)]
struct PolicyWeightsJson {
    node_in: LinearJson,
    node_norm: LayerNormJson,
    blocks: Vec<GraphBlockJson>,
    global_fuse: LinearJson,
    global_norm: LayerNormJson,
    place_head: LinearJson,
    edge_proj: LinearJson,
    edge_norm: LayerNormJson,
    attack_head: LinearJson,
    fortify_head: LinearJson,
    phase_end_head: LinearJson,
    value_fc1: LinearJson,
    value_fc2: LinearJson,
}

impl From<&RiskPolicyWeights> for PolicyWeightsJson {
    fn from(w: &RiskPolicyWeights) -> Self {
        Self {
            node_in: LinearJson::from(&w.node_in),
            node_norm: LayerNormJson::from(&w.node_norm),
            blocks: w.blocks.iter().map(GraphBlockJson::from).collect(),
            global_fuse: LinearJson::from(&w.global_fuse),
            global_norm: LayerNormJson::from(&w.global_norm),
            place_head: LinearJson::from(&w.place_head),
            edge_proj: LinearJson::from(&w.edge_proj),
            edge_norm: LayerNormJson::from(&w.edge_norm),
            attack_head: LinearJson::from(&w.attack_head),
            fortify_head: LinearJson::from(&w.fortify_head),
            phase_end_head: LinearJson::from(&w.phase_end_head),
            value_fc1: LinearJson::from(&w.value_fc1),
            value_fc2: LinearJson::from(&w.value_fc2),
        }
    }
}

#[derive(Serialize)]
struct PolicyExportJson {
    format_version: u32,
    dtype: &'static str,
    linear_layout: &'static str,
    layernorm_epsilon: f64,
    layernorm_variance: &'static str,
    id: String,
    name: String,
    description: String,
    initial_elo: f64,
    hidden_dim: usize,
    num_layers: usize,
    sampling: String,
    weights: PolicyWeightsJson,
}

#[derive(Serialize)]
struct TopActionJson {
    action_id: usize,
    prob: f32,
}

#[derive(Serialize)]
struct ParityVectorJson {
    name: String,
    players: usize,
    max_turns: u32,
    owner: Vec<usize>,
    troops: Vec<u64>,
    active: usize,
    phase: u8,
    pending: u64,
    turns: u32,
    expected_nodes: Vec<f32>,
    expected_global: Vec<f32>,
    expected_mask: Vec<bool>,
    expected_value: f32,
    expected_greedy_action: usize,
    expected_top5: Vec<TopActionJson>,
    expected_fortify_amounts: Vec<[Option<u64>; 3]>,
}

#[derive(Serialize)]
struct ParityBundleJson {
    format_version: u32,
    policy_id: String,
    num_territories: usize,
    num_edges: usize,
    node_feat_dim: usize,
    global_feat_dim: usize,
    action_count: usize,
    vectors: Vec<ParityVectorJson>,
}

fn phase_code(phase: Phase) -> u8 {
    match phase {
        Phase::Reinforce => 0,
        Phase::Attack => 1,
        Phase::Fortify => 2,
        Phase::Ended => 3,
    }
}

fn has_controlled_continent(game: &Game) -> bool {
    REGIONS.iter().any(|region| {
        let first = game.owner[region[0]];
        region.iter().all(|&t| game.owner[t] == first)
    })
}

fn has_three_dice_attack(game: &Game) -> bool {
    let me = game.active;
    EDGES
        .iter()
        .any(|&(a, b)| game.owner[a] == me && game.owner[b] != me && game.troops[a] >= 4)
}

fn has_full_fortify_options(game: &Game) -> bool {
    let me = game.active;
    EDGES.iter().any(|&(a, b)| {
        game.owner[a] == me
            && game.owner[b] == me
            && game.fortify_amount(a, 0).is_some()
            && game.fortify_amount(a, 1).is_some()
            && game.fortify_amount(a, 2).is_some()
    })
}

fn search_game_state(
    base_seed: u64,
    max_turns: u32,
    topo: &BoardTopology,
    predicate: impl Fn(&Game) -> bool,
) -> Result<Game, String> {
    for attempt in 0..32u64 {
        let seed = base_seed + attempt * 1009;
        let mut game = Game::new(2, max_turns, Some(seed));
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x51A7_C0DE_9E37_79B9);
        for _step in 0..6000 {
            if game.phase == Phase::Ended {
                break;
            }
            if predicate(&game) {
                return Ok(game);
            }
            let act_id = heuristic_action(&game, topo, &mut rng);
            let act = Action::decode(act_id).expect("heuristic returned valid action");
            game.step(act);
        }
    }
    Err(format!(
        "failed to find parity game state starting from seed {base_seed}"
    ))
}

fn build_parity_vector(
    name: &str,
    game: &Game,
    model: &RiskPolicyNet<NdArray>,
    topo: &BoardTopology,
    graph: &GraphBuffers<NdArray>,
    device: &NdArrayDevice,
) -> ParityVectorJson {
    assert_ne!(game.phase, Phase::Ended, "parity state must be live");

    let obs = encode_observation(game, game.active, topo);
    let (probs, values) = model.infer_batch(
        obs.nodes.to_vec(),
        obs.global.to_vec(),
        obs.mask.to_vec(),
        1,
        graph,
        device,
    );
    assert_eq!(probs.len(), ACTION_COUNT);
    assert_eq!(values.len(), 1);

    let expected_value = values[0];
    let expected_greedy_action = greedy_masked_action(&probs, &obs.mask);

    let mut legal_probs: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .filter_map(|(id, &p)| obs.mask[id].then_some((id, p)))
        .collect();
    legal_probs.sort_by(|(id_a, p_a), (id_b, p_b)| p_b.total_cmp(p_a).then_with(|| id_a.cmp(id_b)));
    let expected_top5 = legal_probs
        .into_iter()
        .take(5)
        .map(|(action_id, prob)| TopActionJson { action_id, prob })
        .collect();

    let expected_fortify_amounts: Vec<[Option<u64>; 3]> = (0..N)
        .map(|t| {
            [
                game.fortify_amount(t, 0),
                game.fortify_amount(t, 1),
                game.fortify_amount(t, 2),
            ]
        })
        .collect();

    ParityVectorJson {
        name: name.to_string(),
        players: game.players,
        max_turns: game.max_turns,
        owner: game.owner.to_vec(),
        troops: game.troops.to_vec(),
        active: game.active,
        phase: phase_code(game.phase),
        pending: game.pending,
        turns: game.turns,
        expected_nodes: obs.nodes.to_vec(),
        expected_global: obs.global.to_vec(),
        expected_mask: obs.mask.to_vec(),
        expected_value,
        expected_greedy_action,
        expected_top5,
        expected_fortify_amounts,
    }
}

fn generate_parity_bundle(
    policy_id: &str,
    model: &RiskPolicyNet<NdArray>,
) -> Result<ParityBundleJson, String> {
    let topo = BoardTopology::new();
    let device = NdArrayDevice::Cpu;
    let graph = GraphBuffers::<NdArray>::from_topology(&topo, &device);

    let cases: [(&str, Game); 8] = [
        (
            "reinforce_opening",
            search_game_state(101, 150, &topo, |g| {
                g.turns == 0 && g.phase == Phase::Reinforce
            })?,
        ),
        (
            "reinforce_midgame_continent",
            search_game_state(202, 150, &topo, |g| {
                g.turns >= 6 && g.phase == Phase::Reinforce && has_controlled_continent(g)
            })?,
        ),
        (
            "attack_opening",
            search_game_state(303, 150, &topo, |g| {
                g.turns == 0 && g.phase == Phase::Attack && has_three_dice_attack(g)
            })?,
        ),
        (
            "attack_midgame_seat0",
            search_game_state(404, 150, &topo, |g| {
                g.turns >= 6
                    && g.active == 0
                    && g.phase == Phase::Attack
                    && has_three_dice_attack(g)
            })?,
        ),
        (
            "attack_midgame_seat1",
            search_game_state(505, 150, &topo, |g| {
                g.turns >= 7
                    && g.active == 1
                    && g.phase == Phase::Attack
                    && has_three_dice_attack(g)
            })?,
        ),
        (
            "fortify_early_seat0_all_options",
            search_game_state(606, 150, &topo, |g| {
                g.turns >= 1
                    && g.active == 0
                    && g.phase == Phase::Fortify
                    && has_full_fortify_options(g)
            })?,
        ),
        (
            "fortify_midgame_seat1",
            search_game_state(707, 150, &topo, |g| {
                g.turns >= 8
                    && g.active == 1
                    && g.phase == Phase::Fortify
                    && has_full_fortify_options(g)
            })?,
        ),
        (
            "reinforce_lategame_seat1_500t",
            search_game_state(808, 500, &topo, |g| {
                g.turns >= 14
                    && g.active == 1
                    && g.phase == Phase::Reinforce
                    && has_controlled_continent(g)
            })?,
        ),
    ];

    let vectors = cases
        .iter()
        .map(|(name, game)| build_parity_vector(name, game, model, &topo, &graph, &device))
        .collect();

    Ok(ParityBundleJson {
        format_version: 1,
        policy_id: policy_id.to_string(),
        num_territories: N,
        num_edges: NUM_EDGES,
        node_feat_dim: NODE_FEAT_DIM,
        global_feat_dim: GLOBAL_FEAT_DIM,
        action_count: ACTION_COUNT,
        vectors,
    })
}

fn write_json_file<T: Serialize>(path: &Path, value: &T, pretty: bool) -> Result<usize, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create directory '{}': {e}", parent.display()))?;
    }
    let serialized = if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    }
    .map_err(|e| format!("failed to serialize '{}': {e}", path.display()))?;
    fs::write(path, serialized.as_bytes())
        .map_err(|e| format!("failed to write '{}': {e}", path.display()))?;
    Ok(serialized.len())
}

fn run(cfg: CliConfig) -> Result<(), String> {
    let (model, hidden_dim, num_layers) =
        riskbot::visualizer::load_policy_checkpoint(&cfg.checkpoint)?;
    let weights = model.export_weights();
    assert_eq!(weights.hidden_dim, hidden_dim);
    assert_eq!(weights.num_layers, num_layers);

    let stem = cfg
        .checkpoint
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("policy");
    let league = read_sibling_league(&cfg.checkpoint, stem);
    let policy_id = cfg
        .policy_id
        .unwrap_or_else(|| derive_policy_id(&cfg.checkpoint, &league));
    let policy_name = cfg
        .policy_name
        .unwrap_or_else(|| title_case_from_id(&policy_id));
    let initial_elo = league.member_elo.or(league.learner_elo).unwrap_or(1200.0);
    let description = build_description(&league, hidden_dim, num_layers);

    // 1. Export board.json
    let board_value = riskbot::visualizer::export_board_json();
    let board_path = cfg.out_dir.join("board.json");
    let board_bytes = write_json_file(&board_path, &board_value, false)?;

    // 2. Export policies/<policy-id>.json
    let policy_export = PolicyExportJson {
        format_version: 1,
        dtype: "f32_le_base64",
        linear_layout: "input_output",
        layernorm_epsilon: weights.node_norm.epsilon,
        layernorm_variance: "biased",
        id: policy_id.clone(),
        name: policy_name,
        description,
        initial_elo,
        hidden_dim,
        num_layers,
        sampling: cfg.sampling,
        weights: PolicyWeightsJson::from(&weights),
    };
    let policy_path = cfg
        .out_dir
        .join("policies")
        .join(format!("{policy_id}.json"));
    let policy_bytes = write_json_file(&policy_path, &policy_export, false)?;

    // 3. Export parity-vectors.json
    let parity_bundle = generate_parity_bundle(&policy_id, &model)?;
    let parity_path = cfg.out_dir.join("parity-vectors.json");
    let parity_bytes = write_json_file(&parity_path, &parity_bundle, false)?;

    println!("Exported Riskbot arena assets:");
    println!("  - {} ({board_bytes} bytes)", board_path.display());
    println!(
        "  - {} ({policy_bytes} bytes, {hidden_dim}×{num_layers} GNN, Elo {:.1})",
        policy_path.display(),
        initial_elo
    );
    println!(
        "  - {} ({parity_bytes} bytes, {} states)",
        parity_path.display(),
        parity_bundle.vectors.len()
    );

    Ok(())
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(Some(cfg)) => match run(cfg) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        },
        Ok(None) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use riskbot::train::model::RiskPolicyConfig;

    #[test]
    fn base64_matches_rfc4648_vectors() {
        assert_eq!(encode_base64(b""), "");
        assert_eq!(encode_base64(b"f"), "Zg==");
        assert_eq!(encode_base64(b"fo"), "Zm8=");
        assert_eq!(encode_base64(b"foo"), "Zm9v");
        assert_eq!(encode_base64(b"foob"), "Zm9vYg==");
        assert_eq!(encode_base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(encode_base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn exported_weights_preserve_burn_linear_input_output_shapes() {
        let device = NdArrayDevice::Cpu;
        let model = RiskPolicyConfig::new()
            .with_hidden_dim(32)
            .with_num_layers(2)
            .init::<NdArray>(&device);
        let weights = model.export_weights();

        assert_eq!(weights.node_in.shape, [NODE_FEAT_DIM + GLOBAL_FEAT_DIM, 32]);
        assert!(weights.node_in.bias.is_some());
        assert_eq!(weights.blocks.len(), 2);
        assert_eq!(weights.blocks[0].w_self.shape, [32, 32]);
        assert!(weights.blocks[0].w_self.bias.is_some());
        assert_eq!(weights.blocks[0].w_neigh.shape, [32, 32]);
        assert!(weights.blocks[0].w_neigh.bias.is_none());
        assert_eq!(weights.global_fuse.shape, [64 + GLOBAL_FEAT_DIM, 32]);
        assert_eq!(weights.place_head.shape, [64, 1]);
        assert_eq!(weights.edge_proj.shape, [96, 32]);
        assert_eq!(weights.attack_head.shape, [32, 3]);
        assert_eq!(weights.fortify_head.shape, [32, 3]);
        assert_eq!(weights.phase_end_head.shape, [32, 2]);
        assert_eq!(weights.value_fc1.shape, [32, 16]);
        assert_eq!(weights.value_fc2.shape, [16, 1]);
        assert!((weights.node_norm.epsilon - 1e-5).abs() < 1e-12);
    }

    #[test]
    fn slug_and_title_derivation_match_expected_conventions() {
        let league = LeagueSummary {
            run_name: Some("overnight".to_string()),
            ..Default::default()
        };
        let id = derive_policy_id(Path::new("runs/overnight/best.mpk"), &league);
        assert_eq!(id, "overnight-best");
        assert_eq!(title_case_from_id(&id), "Overnight Best");
        assert_eq!(format_with_commas(2000), "2,000");
        assert_eq!(format_steps(11_616_258), "11.6M steps");
    }
}
