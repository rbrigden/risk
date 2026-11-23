//! Checkpoint auto-discovery, architecture-agnostic Burn policy loader, and AI seat controller
//! engine for the Riskbot visualizer.

use crate::game::{Action, Game, Phase};
use crate::train::bots::{heuristic_action, random_action};
use crate::train::features::BoardTopology;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use crate::train::env::{greedy_masked_action, sample_masked_categorical};
#[cfg(not(target_arch = "wasm32"))]
use crate::train::features::{GLOBAL_FEAT_DIM, NODE_FEAT_DIM, encode_observation};
#[cfg(not(target_arch = "wasm32"))]
use crate::train::model::{GraphBuffers, RiskPolicyConfig, RiskPolicyNet, RiskPolicyNetRecord};
#[cfg(not(target_arch = "wasm32"))]
use burn::backend::NdArray;
#[cfg(not(target_arch = "wasm32"))]
use burn::backend::ndarray::NdArrayDevice;
#[cfg(not(target_arch = "wasm32"))]
use burn::module::Module;
#[cfg(not(target_arch = "wasm32"))]
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
#[cfg(not(target_arch = "wasm32"))]
use rand::Rng;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::{HashMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
use std::ffi::OsString;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;

/// Exploration rate for `PolicySampling::Balanced` (matching training evaluation).
#[cfg(not(target_arch = "wasm32"))]
const BALANCED_EXPLORATION_PROB: f32 = 0.08;

/// Maximum directory recursion depth when scanning checkpoint roots.
#[cfg(not(target_arch = "wasm32"))]
const MAX_DISCOVERY_DEPTH: usize = 3;

/// Metadata for a discovered `.mpk` policy checkpoint on disk.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscoveredCheckpoint {
    pub path: PathBuf,
    pub display_name: String,
    pub run_name: String,
    pub file_stem: String,
    pub elo: Option<f64>,
    pub update: Option<usize>,
    pub trained_players: Option<usize>,
}

/// Action-selection strategy for neural policy seats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicySampling {
    /// Pick the highest-probability legal action (deterministic argmax).
    Greedy,
    /// Mostly greedy with slight exploration (8% stochastic, matching training eval).
    Balanced,
    /// Sample from the full masked policy distribution.
    Stochastic,
}

impl PolicySampling {
    pub fn label(self) -> &'static str {
        match self {
            Self::Greedy => "Greedy",
            Self::Balanced => "Balanced",
            Self::Stochastic => "Stochastic",
        }
    }
}

/// Controller assigned to a player seat in the visualizer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SeatKind {
    Human,
    Checkpoint {
        path: PathBuf,
        label: String,
        sampling: PolicySampling,
    },
    Heuristic,
    Random,
}

impl SeatKind {
    pub fn is_human(&self) -> bool {
        matches!(self, Self::Human)
    }

    pub fn short_label(&self) -> String {
        match self {
            Self::Human => "You".to_string(),
            Self::Checkpoint { label, path, .. } => {
                if !label.trim().is_empty() {
                    label.clone()
                } else {
                    path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("Policy")
                        .to_string()
                }
            }
            Self::Heuristic => "Heuristic".to_string(),
            Self::Random => "Random".to_string(),
        }
    }
}

/// Result of an AI seat choosing an action.
#[derive(Clone, Debug)]
pub struct PolicyDecision {
    pub action: Action,
    /// Probability assigned to the chosen action (if neural policy).
    pub confidence: Option<f32>,
    /// Value estimate `V(s)` in approximately `[-1, 1]` from the policy's critic head (if neural policy).
    pub value_estimate: Option<f32>,
}

/// Scan `roots` (directories up to 3 levels deep, or direct `.mpk` file paths) for saved
/// policy checkpoints, enriching them with optional sibling `league.json` metadata.
#[cfg(target_arch = "wasm32")]
pub fn discover_checkpoints(_roots: &[PathBuf]) -> Vec<DiscoveredCheckpoint> {
    Vec::new()
}

/// Scan `roots` (directories up to 3 levels deep, or direct `.mpk` file paths) for saved
/// policy checkpoints, enriching them with optional sibling `league.json` metadata.
#[cfg(not(target_arch = "wasm32"))]
pub fn discover_checkpoints(roots: &[PathBuf]) -> Vec<DiscoveredCheckpoint> {
    let mut raw_paths = Vec::new();
    for root in roots {
        if root.is_file() {
            if is_mpk_file(root) {
                raw_paths.push(root.clone());
            }
        } else if root.is_dir() {
            collect_mpk_files(root, 0, MAX_DISCOVERY_DEPTH, &mut raw_paths);
        }
    }

    let mut seen = HashSet::new();
    let mut league_cache: HashMap<PathBuf, Option<LeagueMeta>> = HashMap::new();
    let mut discovered = Vec::new();

    for path in raw_paths {
        let dedup_key = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
        if !seen.insert(dedup_key) {
            continue;
        }

        let file_stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();
        let parent_dir = path.parent().map(Path::to_path_buf);

        let meta = parent_dir.as_ref().and_then(|dir| {
            league_cache
                .entry(dir.clone())
                .or_insert_with(|| read_sibling_league_meta(dir))
                .clone()
        });

        let run_name = meta
            .as_ref()
            .and_then(|m| m.run_name.clone())
            .or_else(|| {
                parent_dir
                    .as_ref()
                    .and_then(|p| p.file_name())
                    .and_then(|s| s.to_str())
                    .filter(|s| !s.is_empty())
                    .map(ToOwned::to_owned)
            })
            .unwrap_or_else(|| "checkpoint".to_string());

        let elo = meta
            .as_ref()
            .and_then(|m| m.member_elos.get(&file_stem).copied().or(m.learner_elo));
        let update = meta.as_ref().and_then(|m| m.current_update);
        let trained_players = meta.as_ref().and_then(|m| m.players);

        let base_label = if run_name.is_empty() || run_name == file_stem {
            file_stem.clone()
        } else {
            format!("{run_name} / {file_stem}")
        };
        let display_name = match elo {
            Some(e) => format!("{base_label} (Elo {:.0})", e.round()),
            None => base_label,
        };

        discovered.push(DiscoveredCheckpoint {
            path,
            display_name,
            run_name,
            file_stem,
            elo,
            update,
            trained_players,
        });
    }

    discovered.sort_by(compare_checkpoints);
    discovered
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug, Default)]
struct LeagueMeta {
    run_name: Option<String>,
    learner_elo: Option<f64>,
    current_update: Option<usize>,
    players: Option<usize>,
    member_elos: HashMap<String, f64>,
}

#[cfg(not(target_arch = "wasm32"))]
fn is_mpk_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("mpk"))
}

#[cfg(not(target_arch = "wasm32"))]
fn collect_mpk_files(dir: &Path, depth: usize, max_depth: usize, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            if is_mpk_file(&path) {
                out.push(path);
            }
        } else if path.is_dir() && depth < max_depth {
            collect_mpk_files(&path, depth + 1, max_depth, out);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn read_sibling_league_meta(dir: &Path) -> Option<LeagueMeta> {
    let league_path = dir.join("league.json");
    let raw = fs::read_to_string(&league_path).ok()?;
    let val: serde_json::Value = serde_json::from_str(&raw).ok()?;

    let run_name = val
        .get("run_name")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned);
    let learner_elo = val
        .get("learner_elo")
        .and_then(|v| v.as_f64())
        .filter(|v| v.is_finite());
    let current_update = val
        .get("current_update")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let players = val
        .get("players")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    let mut member_elos = HashMap::new();
    if let Some(members) = val.get("members").and_then(|v| v.as_array()) {
        for member in members {
            if let (Some(id), Some(elo)) = (
                member.get("id").and_then(|v| v.as_str()),
                member
                    .get("elo")
                    .and_then(|v| v.as_f64())
                    .filter(|v| v.is_finite()),
            ) {
                member_elos.insert(id.to_string(), elo);
            }
        }
    }

    Some(LeagueMeta {
        run_name,
        learner_elo,
        current_update,
        players,
        member_elos,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn stem_rank(stem: &str) -> u8 {
    match stem.to_ascii_lowercase().as_str() {
        "best" => 0,
        "latest" => 1,
        _ => 2,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn compare_checkpoints(a: &DiscoveredCheckpoint, b: &DiscoveredCheckpoint) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    // 1. Highest Elo first (entries with Elo come before entries without Elo).
    let elo_ord = match (a.elo, b.elo) {
        (Some(ea), Some(eb)) => eb.total_cmp(&ea),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    };
    if elo_ord != Ordering::Equal {
        return elo_ord;
    }

    // 2. Most recently updated run first.
    let upd_ord = match (a.update, b.update) {
        (Some(ua), Some(ub)) => ub.cmp(&ua),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    };
    if upd_ord != Ordering::Equal {
        return upd_ord;
    }

    // 3. Alphabetical by run name, keeping checkpoints from the same run grouped together.
    let run_ord = a.run_name.cmp(&b.run_name);
    if run_ord != Ordering::Equal {
        return run_ord;
    }

    // 4. Within the same run: `best` before `latest` before other stems.
    let rank_ord = stem_rank(&a.file_stem).cmp(&stem_rank(&b.file_stem));
    if rank_ord != Ordering::Equal {
        return rank_ord;
    }

    // 5. Deterministic fallback on stem then full path.
    a.file_stem
        .cmp(&b.file_stem)
        .then_with(|| a.path.cmp(&b.path))
}

#[cfg(not(target_arch = "wasm32"))]
struct LoadedPolicy {
    model: RiskPolicyNet<NdArray>,
    hidden_dim: usize,
    num_layers: usize,
}

/// Architecture-agnostic checkpoint loader, model cache, and unified AI action chooser.
pub struct PolicyRegistry {
    topo: BoardTopology,
    #[cfg(not(target_arch = "wasm32"))]
    device: NdArrayDevice,
    #[cfg(not(target_arch = "wasm32"))]
    graph: GraphBuffers<NdArray>,
    #[cfg(not(target_arch = "wasm32"))]
    cache: HashMap<PathBuf, LoadedPolicy>,
}

impl Default for PolicyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicyRegistry {
    pub fn new() -> Self {
        let topo = BoardTopology::new();
        #[cfg(not(target_arch = "wasm32"))]
        {
            let device = NdArrayDevice::Cpu;
            let graph = GraphBuffers::<NdArray>::from_topology(&topo, &device);
            Self {
                topo,
                device,
                graph,
                cache: HashMap::new(),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self { topo }
        }
    }

    /// Load and cache a `.mpk` checkpoint, returning its auto-detected `(hidden_dim, num_layers)`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn preload(&mut self, path: &Path) -> Result<(usize, usize), String> {
        let key = canonical_checkpoint_key(path);
        if let Some(entry) = self.cache.get(&key) {
            return Ok((entry.hidden_dim, entry.num_layers));
        }

        let (model, hidden_dim, num_layers) = load_checkpoint_auto(path, &self.device)?;
        self.cache.insert(
            key,
            LoadedPolicy {
                model,
                hidden_dim,
                num_layers,
            },
        );
        Ok((hidden_dim, num_layers))
    }

    /// Load and cache a `.mpk` checkpoint (unavailable on wasm32).
    #[cfg(target_arch = "wasm32")]
    pub fn preload(&mut self, _path: &Path) -> Result<(usize, usize), String> {
        Err("Checkpoint inference is only available in the native desktop visualizer".into())
    }

    /// Return cached `(hidden_dim, num_layers)` for `path` if already loaded.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn model_info(&self, path: &Path) -> Option<(usize, usize)> {
        let key = canonical_checkpoint_key(path);
        self.cache
            .get(&key)
            .map(|entry| (entry.hidden_dim, entry.num_layers))
    }

    /// Return cached `(hidden_dim, num_layers)` for `path` if already loaded.
    #[cfg(target_arch = "wasm32")]
    pub fn model_info(&self, _path: &Path) -> Option<(usize, usize)> {
        None
    }

    /// Choose a legal action for the currently active seat according to `seat`.
    pub fn choose_action(
        &mut self,
        seat: &SeatKind,
        game: &Game,
        rng: &mut ChaCha8Rng,
    ) -> Result<PolicyDecision, String> {
        if game.phase == Phase::Ended {
            return Err("Cannot choose an action for an ended game".into());
        }

        match seat {
            SeatKind::Human => Err("Human seat requires manual input".into()),
            SeatKind::Random => {
                let act_id = random_action(game, rng);
                let action = Action::decode(act_id)
                    .ok_or_else(|| format!("random bot returned invalid action id {act_id}"))?;
                Ok(PolicyDecision {
                    action,
                    confidence: None,
                    value_estimate: None,
                })
            }
            SeatKind::Heuristic => {
                let act_id = heuristic_action(game, &self.topo, rng);
                let action = Action::decode(act_id)
                    .ok_or_else(|| format!("heuristic bot returned invalid action id {act_id}"))?;
                Ok(PolicyDecision {
                    action,
                    confidence: None,
                    value_estimate: None,
                })
            }
            SeatKind::Checkpoint { path, sampling, .. } => {
                self.choose_checkpoint_action(path, *sampling, game, rng)
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn choose_checkpoint_action(
        &mut self,
        path: &Path,
        sampling: PolicySampling,
        game: &Game,
        rng: &mut ChaCha8Rng,
    ) -> Result<PolicyDecision, String> {
        self.preload(path)?;
        let key = canonical_checkpoint_key(path);
        let entry = self
            .cache
            .get(&key)
            .ok_or_else(|| format!("checkpoint '{}' missing from cache", path.display()))?;

        let obs = encode_observation(game, game.active, &self.topo);
        if !obs.mask.iter().any(|&legal| legal) {
            return Err("Active player has no legal actions".into());
        }

        let (probs, vals) = entry.model.infer_batch(
            obs.nodes.to_vec(),
            obs.global.to_vec(),
            obs.mask.to_vec(),
            1,
            &self.graph,
            &self.device,
        );

        let act_id = match sampling {
            PolicySampling::Greedy => greedy_masked_action(&probs, &obs.mask),
            PolicySampling::Balanced => {
                if rng.random::<f32>() < BALANCED_EXPLORATION_PROB {
                    sample_masked_categorical(&probs, &obs.mask, rng).0
                } else {
                    greedy_masked_action(&probs, &obs.mask)
                }
            }
            PolicySampling::Stochastic => sample_masked_categorical(&probs, &obs.mask, rng).0,
        };

        let action = Action::decode(act_id)
            .ok_or_else(|| format!("policy returned invalid action id {act_id}"))?;
        let confidence = probs.get(act_id).copied().filter(|p| p.is_finite());
        let value_estimate = vals.first().copied().filter(|v| v.is_finite());

        Ok(PolicyDecision {
            action,
            confidence,
            value_estimate,
        })
    }

    #[cfg(target_arch = "wasm32")]
    fn choose_checkpoint_action(
        &mut self,
        _path: &Path,
        _sampling: PolicySampling,
        _game: &Game,
        _rng: &mut ChaCha8Rng,
    ) -> Result<PolicyDecision, String> {
        Err("Checkpoint inference is only available in the native desktop visualizer".into())
    }

    /// Evaluate the critic state value `V(s)` for `seat_idx` if `seat` is a neural checkpoint.
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(dead_code)]
    pub fn evaluate_seat_value(
        &mut self,
        seat: &SeatKind,
        game: &Game,
        seat_idx: usize,
    ) -> Option<f32> {
        if seat_idx >= game.players {
            return None;
        }
        let SeatKind::Checkpoint { path, .. } = seat else {
            return None;
        };

        self.preload(path).ok()?;
        let key = canonical_checkpoint_key(path);
        let entry = self.cache.get(&key)?;

        let obs = encode_observation(game, seat_idx, &self.topo);
        let vals = entry.model.value_batch(
            obs.nodes.to_vec(),
            obs.global.to_vec(),
            obs.mask.to_vec(),
            1,
            &self.graph,
            &self.device,
        );
        vals.first().copied().filter(|v| v.is_finite())
    }

    /// Evaluate the critic state value `V(s)` for `seat_idx` (unavailable on wasm32).
    #[cfg(target_arch = "wasm32")]
    #[allow(dead_code)]
    pub fn evaluate_seat_value(
        &mut self,
        _seat: &SeatKind,
        _game: &Game,
        _seat_idx: usize,
    ) -> Option<f32> {
        None
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn strip_mpk_extension(path: &Path) -> PathBuf {
    if is_mpk_file(path) {
        path.with_extension("")
    } else {
        path.to_path_buf()
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn load_checkpoint_auto(
    path: &Path,
    device: &NdArrayDevice,
) -> Result<(RiskPolicyNet<NdArray>, usize, usize), String> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let base_path = strip_mpk_extension(path);
    let record: RiskPolicyNetRecord<NdArray> = recorder
        .load(base_path, device)
        .map_err(|err| format!("failed to load checkpoint '{}': {err}", path.display()))?;

    let num_layers = record.blocks.len();
    let [in_dim, hidden_dim] = record.node_in.weight.val().dims();
    let expected_in_dim = NODE_FEAT_DIM + GLOBAL_FEAT_DIM;
    if in_dim != expected_in_dim {
        return Err(format!(
            "incompatible checkpoint '{}': expected input dim {expected_in_dim}, got {in_dim}",
            path.display()
        ));
    }
    if hidden_dim == 0 {
        return Err(format!(
            "invalid checkpoint '{}': hidden_dim is 0",
            path.display()
        ));
    }

    let model = RiskPolicyConfig::new()
        .with_hidden_dim(hidden_dim)
        .with_num_layers(num_layers)
        .init::<NdArray>(device)
        .load_record(record);
    Ok((model, hidden_dim, num_layers))
}

#[cfg(not(target_arch = "wasm32"))]
fn canonical_checkpoint_key(path: &Path) -> PathBuf {
    if let Ok(canon) = fs::canonicalize(path) {
        return canon;
    }
    if !is_mpk_file(path) {
        let mut with_ext = OsString::from(path.as_os_str());
        with_ext.push(".mpk");
        let candidate = PathBuf::from(with_ext);
        if let Ok(canon) = fs::canonicalize(&candidate) {
            return canon;
        }
    }
    path.to_path_buf()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::train::ppo::save_checkpoint;
    use rand::SeedableRng;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(prefix: &str) -> Self {
            let id = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let path =
                std::env::temp_dir().join(format!("riskbot_policy_test_{prefix}_{pid}_{id}"));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("create temp test dir");
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn discover_checkpoints_parses_metadata_and_sorts() {
        let tmp = TempDir::new("discover");
        let run_high = tmp.path.join("alpha-high");
        let run_low = tmp.path.join("beta-low");
        let run_no_meta = tmp.path.join("gamma-nometa");
        fs::create_dir_all(&run_high).unwrap();
        fs::create_dir_all(&run_low).unwrap();
        fs::create_dir_all(&run_no_meta).unwrap();

        fs::write(run_high.join("latest.mpk"), b"stub").unwrap();
        fs::write(run_high.join("best.mpk"), b"stub").unwrap();
        fs::write(
            run_high.join("league.json"),
            r#"{
                "run_name": "overnight",
                "players": 2,
                "current_update": 167,
                "learner_elo": 1342.4
            }"#,
        )
        .unwrap();

        fs::write(run_low.join("latest.mpk"), b"stub").unwrap();
        fs::write(
            run_low.join("league.json"),
            r#"{
                "run_name": "pfsp-v3",
                "players": 3,
                "current_update": 60,
                "learner_elo": 1021.1
            }"#,
        )
        .unwrap();

        fs::write(run_no_meta.join("latest.mpk"), b"stub").unwrap();
        fs::write(run_no_meta.join("best.mpk"), b"stub").unwrap();

        // Pass both the root dir and a direct file path to verify deduplication.
        let direct_file = run_high.join("best.mpk");
        let found = discover_checkpoints(&[tmp.path.clone(), direct_file]);
        assert_eq!(found.len(), 5);

        // Highest Elo first; within same run, `best` precedes `latest`.
        assert_eq!(found[0].run_name, "overnight");
        assert_eq!(found[0].file_stem, "best");
        assert_eq!(found[0].display_name, "overnight / best (Elo 1342)");
        assert_eq!(found[0].update, Some(167));
        assert_eq!(found[0].trained_players, Some(2));

        assert_eq!(found[1].run_name, "overnight");
        assert_eq!(found[1].file_stem, "latest");
        assert_eq!(found[1].display_name, "overnight / latest (Elo 1342)");

        assert_eq!(found[2].run_name, "pfsp-v3");
        assert_eq!(found[2].file_stem, "latest");
        assert_eq!(found[2].display_name, "pfsp-v3 / latest (Elo 1021)");
        assert_eq!(found[2].trained_players, Some(3));

        // Entries without league.json come after Elo-rated runs, with `best` before `latest`.
        assert_eq!(found[3].run_name, "gamma-nometa");
        assert_eq!(found[3].file_stem, "best");
        assert_eq!(found[3].display_name, "gamma-nometa / best");
        assert_eq!(found[3].elo, None);

        assert_eq!(found[4].run_name, "gamma-nometa");
        assert_eq!(found[4].file_stem, "latest");
        assert_eq!(found[4].display_name, "gamma-nometa / latest");
    }

    #[test]
    fn preload_auto_detects_non_default_and_default_architectures() {
        let tmp = TempDir::new("arch");
        let device = NdArrayDevice::Cpu;

        let small_model = RiskPolicyConfig::new()
            .with_hidden_dim(32)
            .with_num_layers(1)
            .init::<NdArray>(&device);
        let small_base = tmp.path.join("small");
        save_checkpoint(&small_model, &small_base).expect("save small checkpoint");
        let small_mpk = tmp.path.join("small.mpk");

        let default_model = RiskPolicyConfig::new()
            .with_hidden_dim(64)
            .with_num_layers(2)
            .init::<NdArray>(&device);
        let default_base = tmp.path.join("default");
        save_checkpoint(&default_model, &default_base).expect("save default checkpoint");
        let default_mpk = tmp.path.join("default.mpk");

        let mut registry = PolicyRegistry::new();
        assert_eq!(registry.model_info(&small_mpk), None);

        assert_eq!(registry.preload(&small_mpk).unwrap(), (32, 1));
        assert_eq!(registry.model_info(&small_mpk), Some((32, 1)));
        // Loading without `.mpk` extension resolves to the same cached entry.
        assert_eq!(registry.model_info(&small_base), Some((32, 1)));

        assert_eq!(registry.preload(&default_mpk).unwrap(), (64, 2));
        assert_eq!(registry.model_info(&default_mpk), Some((64, 2)));
    }

    #[test]
    fn choose_action_is_legal_across_all_phases_and_controllers() {
        let tmp = TempDir::new("actions");
        let device = NdArrayDevice::Cpu;
        let model = RiskPolicyConfig::new()
            .with_hidden_dim(32)
            .with_num_layers(1)
            .init::<NdArray>(&device);
        save_checkpoint(&model, &tmp.path.join("policy")).expect("save checkpoint");
        let ckpt_path = tmp.path.join("policy.mpk");

        let mut registry = PolicyRegistry::new();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let seats = [
            SeatKind::Checkpoint {
                path: ckpt_path.clone(),
                label: "Test Greedy".into(),
                sampling: PolicySampling::Greedy,
            },
            SeatKind::Checkpoint {
                path: ckpt_path.clone(),
                label: "Test Balanced".into(),
                sampling: PolicySampling::Balanced,
            },
            SeatKind::Checkpoint {
                path: ckpt_path,
                label: "Test Stochastic".into(),
                sampling: PolicySampling::Stochastic,
            },
            SeatKind::Heuristic,
            SeatKind::Random,
        ];

        // Construct representative game states in Reinforce, Attack, and Fortify phases.
        let reinforce_game = Game::new(3, 50, Some(101));
        assert_eq!(reinforce_game.phase, Phase::Reinforce);

        let topo = BoardTopology::new();
        let mut attack_game = Game::new(3, 50, Some(102));
        while attack_game.phase == Phase::Reinforce {
            let act_id = heuristic_action(&attack_game, &topo, &mut rng);
            attack_game.step(Action::decode(act_id).unwrap());
        }
        assert_eq!(attack_game.phase, Phase::Attack);

        let mut fortify_game = Game::new(3, 50, Some(103));
        while fortify_game.phase == Phase::Reinforce {
            let act_id = heuristic_action(&fortify_game, &topo, &mut rng);
            fortify_game.step(Action::decode(act_id).unwrap());
        }
        fortify_game.step(Action::FinishAttack);
        assert_eq!(fortify_game.phase, Phase::Fortify);

        let phase_games = [reinforce_game, attack_game, fortify_game];

        for game in &phase_games {
            // Human seat returns an error instead of acting automatically.
            assert!(
                registry
                    .choose_action(&SeatKind::Human, game, &mut rng)
                    .is_err()
            );

            for seat in &seats {
                let decision = registry
                    .choose_action(seat, game, &mut rng)
                    .expect("AI seat should choose legal action");
                assert!(
                    game.legal(decision.action),
                    "seat {:?} produced illegal action {:?} in phase {:?}",
                    seat,
                    decision.action,
                    game.phase
                );
                if matches!(seat, SeatKind::Checkpoint { .. }) {
                    let conf = decision
                        .confidence
                        .expect("checkpoint should report confidence");
                    assert!((0.0..=1.0001).contains(&conf));
                    let val = decision
                        .value_estimate
                        .expect("checkpoint should report value");
                    assert!(val.is_finite());

                    let seat_val = registry
                        .evaluate_seat_value(seat, game, game.active)
                        .expect("loaded checkpoint should evaluate seat value");
                    assert!(seat_val.is_finite());
                } else {
                    assert!(decision.confidence.is_none());
                    assert!(decision.value_estimate.is_none());
                    assert!(
                        registry
                            .evaluate_seat_value(seat, game, game.active)
                            .is_none()
                    );
                }
            }
        }
    }

    #[test]
    fn loads_real_overnight_best_checkpoint_if_present() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runs/overnight/best.mpk");
        if !path.is_file() {
            return;
        }

        let mut registry = PolicyRegistry::new();
        let (hidden_dim, num_layers) = registry
            .preload(&path)
            .expect("overnight/best.mpk should load cleanly");
        assert!(hidden_dim > 0);
        assert!(num_layers > 0);

        let game = Game::new(2, 100, Some(77));
        let mut rng = ChaCha8Rng::seed_from_u64(77);
        let seat = SeatKind::Checkpoint {
            path: path.clone(),
            label: "overnight / best".into(),
            sampling: PolicySampling::Greedy,
        };
        let decision = registry
            .choose_action(&seat, &game, &mut rng)
            .expect("overnight/best.mpk should choose a legal action");
        assert!(game.legal(decision.action));
        assert!(decision.confidence.is_some());
        assert!(decision.value_estimate.is_some());
    }
}
