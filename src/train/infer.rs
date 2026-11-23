//! Pure-Rust, Burn-free GNN policy inference, JSON weight decoding, and action sampling.
//!
//! Designed to execute identically on native CPUs and `wasm32-unknown-unknown` with
//! numerical parity (`< 1e-5`) against `RiskPolicyNet::<NdArray>`.

use crate::game::{ACTION_COUNT, N};
use crate::train::features::{
    BoardTopology, EncodedObs, GLOBAL_FEAT_DIM, NODE_FEAT_DIM, NUM_EDGES,
};
use rand::Rng;
use serde::Deserialize;

/// Exploration rate for `SamplingMode::Balanced` (matching training evaluation and visualizer).
pub const BALANCED_EXPLORATION_PROB: f32 = 0.08;

/// Exported parameters for a Burn `Linear` layer (`y = x @ weight + bias`).
#[derive(Clone, Debug)]
pub struct LinearWeights {
    /// `[d_input, d_output]` in row-major order.
    pub shape: [usize; 2],
    /// Flattened row-major `f32` values of length `shape[0] * shape[1]`.
    pub weight: Vec<f32>,
    /// Optional bias vector of length `shape[1]`.
    pub bias: Option<Vec<f32>>,
}

/// Exported parameters for a Burn `LayerNorm` layer (`y = ((x - mean) / sqrt(var_biased + eps)) * gamma + beta`).
#[derive(Clone, Debug)]
pub struct LayerNormWeights {
    pub d_model: usize,
    pub epsilon: f64,
    pub gamma: Vec<f32>,
    pub beta: Option<Vec<f32>>,
}

/// Exported parameters for one residual message-passing graph block.
#[derive(Clone, Debug)]
pub struct GraphBlockWeights {
    pub w_self: LinearWeights,
    pub w_neigh: LinearWeights,
    pub norm: LayerNormWeights,
}

/// Complete read-only parameter dump of a `RiskPolicyNet`.
#[derive(Clone, Debug)]
pub struct RiskPolicyWeights {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub node_in: LinearWeights,
    pub node_norm: LayerNormWeights,
    pub blocks: Vec<GraphBlockWeights>,
    pub global_fuse: LinearWeights,
    pub global_norm: LayerNormWeights,
    pub place_head: LinearWeights,
    pub edge_proj: LinearWeights,
    pub edge_norm: LayerNormWeights,
    pub attack_head: LinearWeights,
    pub fortify_head: LinearWeights,
    pub phase_end_head: LinearWeights,
    pub value_fc1: LinearWeights,
    pub value_fc2: LinearWeights,
}

/// Action-selection strategy for neural policy inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingMode {
    Greedy,
    Balanced,
    Stochastic,
}

impl SamplingMode {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "greedy" => Ok(Self::Greedy),
            "balanced" | "" => Ok(Self::Balanced),
            "stochastic" => Ok(Self::Stochastic),
            other => Err(format!(
                "Invalid sampling '{other}'; expected 'greedy', 'balanced', or 'stochastic'."
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::Balanced => "balanced",
            Self::Stochastic => "stochastic",
        }
    }
}

#[derive(Deserialize)]
struct TensorJson {
    shape: Vec<usize>,
    data_b64: String,
}

#[derive(Deserialize)]
struct LinearJson {
    weight: TensorJson,
    #[serde(default)]
    bias: Option<TensorJson>,
}

#[derive(Deserialize)]
struct LayerNormJson {
    #[serde(default = "default_layernorm_eps")]
    epsilon: f64,
    gamma: TensorJson,
    #[serde(default)]
    beta: Option<TensorJson>,
}

fn default_layernorm_eps() -> f64 {
    1e-5
}

#[derive(Deserialize)]
struct GraphBlockJson {
    w_self: LinearJson,
    w_neigh: LinearJson,
    norm: LayerNormJson,
}

#[derive(Deserialize)]
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

#[derive(Deserialize)]
struct PolicyExportJson {
    #[serde(default)]
    dtype: Option<String>,
    #[serde(default)]
    linear_layout: Option<String>,
    hidden_dim: usize,
    num_layers: usize,
    weights: PolicyWeightsJson,
}

fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    let bytes: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "Invalid base64 length {}: must be a multiple of 4",
            bytes.len()
        ));
    }

    let mut out = Vec::with_capacity((bytes.len() / 4) * 3);
    let (chunks, _) = bytes.as_chunks::<4>();
    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let is_last = (chunk_idx + 1) * 4 == bytes.len();
        let a = b64_val(chunk[0])?;
        let b = b64_val(chunk[1])?;
        let pad2 = chunk[2] == b'=';
        let pad3 = chunk[3] == b'=';

        if (pad2 || pad3) && !is_last {
            return Err("Invalid base64 padding before end of stream".into());
        }
        if pad2 && !pad3 {
            return Err("Invalid base64 padding sequence".into());
        }

        let c = if pad2 { 0 } else { b64_val(chunk[2])? };
        let d = if pad3 { 0 } else { b64_val(chunk[3])? };
        let triple =
            (u32::from(a) << 18) | (u32::from(b) << 12) | (u32::from(c) << 6) | u32::from(d);

        out.push(((triple >> 16) & 0xFF) as u8);
        if !pad2 {
            out.push(((triple >> 8) & 0xFF) as u8);
        }
        if !pad3 {
            out.push((triple & 0xFF) as u8);
        }
    }
    Ok(out)
}

fn b64_val(byte: u8) -> Result<u8, String> {
    match byte {
        b'A'..=b'Z' => Ok(byte - b'A'),
        b'a'..=b'z' => Ok(byte - b'a' + 26),
        b'0'..=b'9' => Ok(byte - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        other => Err(format!("Invalid base64 character byte {other}")),
    }
}

fn decode_f32_le_b64(b64: &str, expected_len: usize, label: &str) -> Result<Vec<f32>, String> {
    let raw = decode_base64(b64)?;
    let expected_bytes = expected_len * 4;
    if raw.len() != expected_bytes {
        return Err(format!(
            "Tensor '{label}' byte length mismatch: expected {expected_bytes}, got {}",
            raw.len()
        ));
    }
    let mut values = Vec::with_capacity(expected_len);
    let (chunks, _) = raw.as_chunks::<4>();
    for &chunk in chunks {
        let v = f32::from_le_bytes(chunk);
        if !v.is_finite() {
            return Err(format!("Tensor '{label}' contains non-finite value {v}"));
        }
        values.push(v);
    }
    Ok(values)
}

fn decode_linear(
    raw: LinearJson,
    expected_in: usize,
    expected_out: usize,
    label: &str,
) -> Result<LinearWeights, String> {
    if raw.weight.shape != [expected_in, expected_out] {
        return Err(format!(
            "Linear '{label}' weight shape mismatch: expected [{expected_in}, {expected_out}], got {:?}",
            raw.weight.shape
        ));
    }
    let weight = decode_f32_le_b64(&raw.weight.data_b64, expected_in * expected_out, label)?;
    let bias = match raw.bias {
        Some(b) => {
            if b.shape != [expected_out] {
                return Err(format!(
                    "Linear '{label}' bias shape mismatch: expected [{expected_out}], got {:?}",
                    b.shape
                ));
            }
            Some(decode_f32_le_b64(&b.data_b64, expected_out, label)?)
        }
        None => None,
    };
    Ok(LinearWeights {
        shape: [expected_in, expected_out],
        weight,
        bias,
    })
}

fn decode_layernorm(
    raw: LayerNormJson,
    expected_dim: usize,
    label: &str,
) -> Result<LayerNormWeights, String> {
    if raw.gamma.shape != [expected_dim] {
        return Err(format!(
            "LayerNorm '{label}' gamma shape mismatch: expected [{expected_dim}], got {:?}",
            raw.gamma.shape
        ));
    }
    let gamma = decode_f32_le_b64(&raw.gamma.data_b64, expected_dim, label)?;
    let beta = match raw.beta {
        Some(b) => {
            if b.shape != [expected_dim] {
                return Err(format!(
                    "LayerNorm '{label}' beta shape mismatch: expected [{expected_dim}], got {:?}",
                    b.shape
                ));
            }
            Some(decode_f32_le_b64(&b.data_b64, expected_dim, label)?)
        }
        None => None,
    };
    Ok(LayerNormWeights {
        d_model: expected_dim,
        epsilon: raw.epsilon,
        gamma,
        beta,
    })
}

/// Linear transform `out = x @ W + b` for a single row vector `x` (`[d_in] -> [d_out]`).
#[inline]
fn linear_vec(x: &[f32], layer: &LinearWeights, out: &mut [f32]) {
    let [d_in, d_out] = layer.shape;
    debug_assert_eq!(x.len(), d_in);
    debug_assert_eq!(out.len(), d_out);

    match &layer.bias {
        Some(b) => out.copy_from_slice(b),
        None => out.fill(0.0),
    }

    for (i, &xi) in x.iter().enumerate() {
        let row = &layer.weight[i * d_out..(i + 1) * d_out];
        for (o, &w) in out.iter_mut().zip(row.iter()) {
            *o += xi * w;
        }
    }
}

/// In-place Burn-compatible LayerNorm with biased variance (`sum((x - mean)^2) / N`) and `eps`.
#[inline]
fn layer_norm_inplace(v: &mut [f32], ln: &LayerNormWeights) {
    let n = ln.d_model;
    debug_assert_eq!(v.len(), n);

    let n_f = n as f32;
    let mut sum = 0.0f32;
    for &x in v.iter() {
        sum += x;
    }
    let mean = sum / n_f;

    let mut var_sum = 0.0f32;
    for &x in v.iter() {
        let diff = x - mean;
        var_sum += diff * diff;
    }
    let var = var_sum / n_f;
    let inv_std = 1.0f32 / (var + ln.epsilon as f32).sqrt();

    match &ln.beta {
        Some(beta) => {
            for i in 0..n {
                v[i] = ((v[i] - mean) * inv_std) * ln.gamma[i] + beta[i];
            }
        }
        None => {
            for i in 0..n {
                v[i] = ((v[i] - mean) * inv_std) * ln.gamma[i];
            }
        }
    }
}

impl RiskPolicyWeights {
    /// Decode exported policy JSON (as produced by `riskbot-export`) into `RiskPolicyWeights`.
    pub fn decode_from_json(json_str: &str) -> Result<Self, String> {
        let parsed: PolicyExportJson = serde_json::from_str(json_str)
            .map_err(|e| format!("Failed to parse policy JSON: {e}"))?;

        if let Some(dtype) = parsed.dtype.as_deref()
            && dtype != "f32_le_base64"
        {
            return Err(format!("Unsupported policy dtype '{dtype}'"));
        }
        if let Some(layout) = parsed.linear_layout.as_deref()
            && layout != "input_output"
        {
            return Err(format!("Unsupported linear_layout '{layout}'"));
        }

        let d = parsed.hidden_dim;
        if d == 0 {
            return Err("Policy hidden_dim must be greater than 0".into());
        }
        if parsed.weights.blocks.len() != parsed.num_layers {
            return Err(format!(
                "Policy num_layers ({}) does not match blocks length ({})",
                parsed.num_layers,
                parsed.weights.blocks.len()
            ));
        }

        let val_hidden = (d / 2).max(16);
        let w = parsed.weights;

        let mut blocks = Vec::with_capacity(parsed.num_layers);
        for (idx, b) in w.blocks.into_iter().enumerate() {
            blocks.push(GraphBlockWeights {
                w_self: decode_linear(b.w_self, d, d, &format!("blocks[{idx}].w_self"))?,
                w_neigh: decode_linear(b.w_neigh, d, d, &format!("blocks[{idx}].w_neigh"))?,
                norm: decode_layernorm(b.norm, d, &format!("blocks[{idx}].norm"))?,
            });
        }

        Ok(Self {
            hidden_dim: d,
            num_layers: parsed.num_layers,
            node_in: decode_linear(w.node_in, NODE_FEAT_DIM + GLOBAL_FEAT_DIM, d, "node_in")?,
            node_norm: decode_layernorm(w.node_norm, d, "node_norm")?,
            blocks,
            global_fuse: decode_linear(w.global_fuse, 2 * d + GLOBAL_FEAT_DIM, d, "global_fuse")?,
            global_norm: decode_layernorm(w.global_norm, d, "global_norm")?,
            place_head: decode_linear(w.place_head, 2 * d, 1, "place_head")?,
            edge_proj: decode_linear(w.edge_proj, 3 * d, d, "edge_proj")?,
            edge_norm: decode_layernorm(w.edge_norm, d, "edge_norm")?,
            attack_head: decode_linear(w.attack_head, d, 3, "attack_head")?,
            fortify_head: decode_linear(w.fortify_head, d, 3, "fortify_head")?,
            phase_end_head: decode_linear(w.phase_end_head, d, 2, "phase_end_head")?,
            value_fc1: decode_linear(w.value_fc1, d, val_hidden, "value_fc1")?,
            value_fc2: decode_linear(w.value_fc2, val_hidden, 1, "value_fc2")?,
        })
    }

    /// Pure-Rust forward pass returning `(action_probs [ACTION_COUNT], value_estimate)`.
    ///
    /// Numerically matches `RiskPolicyNet::<NdArray>::infer_batch` within `< 1e-5`.
    pub fn infer(&self, obs: &EncodedObs, topo: &BoardTopology) -> (Vec<f32>, f32) {
        let d = self.hidden_dim;

        // 1) Node input projection + LayerNorm + ReLU: [N, D]
        let mut h = vec![0.0f32; N * d];
        let mut node_in_buf = [0.0f32; NODE_FEAT_DIM + GLOBAL_FEAT_DIM];
        node_in_buf[NODE_FEAT_DIM..].copy_from_slice(&obs.global);

        for t in 0..N {
            node_in_buf[..NODE_FEAT_DIM]
                .copy_from_slice(&obs.nodes[t * NODE_FEAT_DIM..(t + 1) * NODE_FEAT_DIM]);
            let dst = &mut h[t * d..(t + 1) * d];
            linear_vec(&node_in_buf, &self.node_in, dst);
            layer_norm_inplace(dst, &self.node_norm);
            for v in dst.iter_mut() {
                *v = v.max(0.0);
            }
        }

        // 2) Residual message-passing graph blocks
        let mut h_next = vec![0.0f32; N * d];
        let mut neigh_vec = vec![0.0f32; d];
        let mut self_msg = vec![0.0f32; d];
        let mut neigh_msg = vec![0.0f32; d];

        for block in &self.blocks {
            for t in 0..N {
                neigh_vec.fill(0.0);
                let adj_row = &topo.adj_norm[t * N..(t + 1) * N];
                for &u in &topo.neighbors[t] {
                    let w = adj_row[u];
                    let hu = &h[u * d..(u + 1) * d];
                    for (acc, &val) in neigh_vec.iter_mut().zip(hu.iter()) {
                        *acc += w * val;
                    }
                }

                let ht = &h[t * d..(t + 1) * d];
                linear_vec(ht, &block.w_self, &mut self_msg);
                linear_vec(&neigh_vec, &block.w_neigh, &mut neigh_msg);
                for j in 0..d {
                    self_msg[j] += neigh_msg[j];
                }
                layer_norm_inplace(&mut self_msg, &block.norm);

                let dst = &mut h_next[t * d..(t + 1) * d];
                for j in 0..d {
                    dst[j] = ht[j] + self_msg[j].max(0.0);
                }
            }
            std::mem::swap(&mut h, &mut h_next);
        }

        // 3) Global pooled state representation: [mean_pool, max_pool, raw_global] -> [D]
        let mut g_in = vec![0.0f32; 2 * d + GLOBAL_FEAT_DIM];
        let inv_n = 1.0f32 / (N as f32);
        for j in 0..d {
            let mut sum = 0.0f32;
            let mut mx = f32::NEG_INFINITY;
            for t in 0..N {
                let v = h[t * d + j];
                sum += v;
                if v > mx {
                    mx = v;
                }
            }
            g_in[j] = sum * inv_n;
            g_in[d + j] = mx;
        }
        g_in[2 * d..].copy_from_slice(&obs.global);

        let mut g = vec![0.0f32; d];
        linear_vec(&g_in, &self.global_fuse, &mut g);
        layer_norm_inplace(&mut g, &self.global_norm);
        for v in g.iter_mut() {
            *v = v.max(0.0);
        }

        // 4) Action logits [ACTION_COUNT]
        let mut logits = vec![0.0f32; ACTION_COUNT];

        // 4a) Place(t) logits: [0..N]
        let mut place_in = vec![0.0f32; 2 * d];
        place_in[d..].copy_from_slice(&g);
        let mut scalar_out = [0.0f32; 1];
        for t in 0..N {
            place_in[..d].copy_from_slice(&h[t * d..(t + 1) * d]);
            linear_vec(&place_in, &self.place_head, &mut scalar_out);
            logits[t] = scalar_out[0];
        }

        // 4b) Edge embeddings for Attack and Fortify heads
        let fortify_start = N + NUM_EDGES * 3;
        let mut edge_in = vec![0.0f32; 3 * d];
        let mut edge_h = vec![0.0f32; d];
        let mut triple_out = [0.0f32; 3];

        for e in 0..NUM_EDGES {
            let src = topo.edge_src[e] as usize;
            let dst = topo.edge_dst[e] as usize;
            let h_src = &h[src * d..(src + 1) * d];
            let h_dst = &h[dst * d..(dst + 1) * d];
            edge_in[..d].copy_from_slice(h_src);
            edge_in[d..2 * d].copy_from_slice(h_dst);
            for j in 0..d {
                edge_in[2 * d + j] = h_src[j] - h_dst[j];
            }

            linear_vec(&edge_in, &self.edge_proj, &mut edge_h);
            layer_norm_inplace(&mut edge_h, &self.edge_norm);
            for v in edge_h.iter_mut() {
                *v = v.max(0.0);
            }

            linear_vec(&edge_h, &self.attack_head, &mut triple_out);
            logits[N + 3 * e..N + 3 * e + 3].copy_from_slice(&triple_out);

            linear_vec(&edge_h, &self.fortify_head, &mut triple_out);
            logits[fortify_start + 3 * e..fortify_start + 3 * e + 3].copy_from_slice(&triple_out);
        }

        // 4c) Phase-end logits: [FinishAttack, PassFortify]
        let mut end_out = [0.0f32; 2];
        linear_vec(&g, &self.phase_end_head, &mut end_out);
        logits[ACTION_COUNT - 2] = end_out[0];
        logits[ACTION_COUNT - 1] = end_out[1];

        // 5) Clamp [-20.0, 20.0], mask invalid to -1e4, softmax
        let mut max_logit = f32::NEG_INFINITY;
        for (l, &allowed) in logits.iter_mut().zip(obs.mask.iter()) {
            let v = if allowed { l.clamp(-20.0, 20.0) } else { -1e4 };
            *l = v;
            if v > max_logit {
                max_logit = v;
            }
        }

        let mut probs = vec![0.0f32; ACTION_COUNT];
        let mut exp_sum = 0.0f32;
        for (p, &l) in probs.iter_mut().zip(logits.iter()) {
            let e = (l - max_logit).exp();
            *p = e;
            exp_sum += e;
        }
        if exp_sum > 0.0 {
            let inv_sum = 1.0f32 / exp_sum;
            for p in probs.iter_mut() {
                *p *= inv_sum;
            }
        }

        // 6) Value head
        let val_hidden = self.value_fc1.shape[1];
        let mut val_h = vec![0.0f32; val_hidden];
        linear_vec(&g, &self.value_fc1, &mut val_h);
        for v in val_h.iter_mut() {
            *v = v.max(0.0);
        }
        linear_vec(&val_h, &self.value_fc2, &mut scalar_out);
        let value = scalar_out[0];

        (probs, value)
    }
}

/// Select the highest-probability legal action (deterministic argmax).
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

/// Choose a legal action according to `mode` (`Greedy`, `Balanced` with 8% exploration, or `Stochastic`).
pub fn choose_policy_action(
    probs: &[f32],
    mask: &[bool; ACTION_COUNT],
    mode: SamplingMode,
    rng: &mut impl Rng,
) -> usize {
    match mode {
        SamplingMode::Greedy => greedy_masked_action(probs, mask),
        SamplingMode::Balanced => {
            if rng.random::<f32>() < BALANCED_EXPLORATION_PROB {
                sample_masked_categorical(probs, mask, rng).0
            } else {
                greedy_masked_action(probs, mask)
            }
        }
        SamplingMode::Stochastic => sample_masked_categorical(probs, mask, rng).0,
    }
}

#[cfg(all(test, feature = "train"))]
mod parity_tests {
    use super::*;
    use crate::game::{Action, Game, Phase};
    use crate::train::bots::heuristic_action;
    use crate::train::features::encode_observation;
    use crate::train::model::{GraphBuffers, RiskPolicyConfig, RiskPolicyNetRecord};
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::path::Path;

    fn collect_diverse_states(topo: &BoardTopology) -> Vec<Game> {
        let mut states = Vec::new();
        let mut game = Game::new(2, 100, Some(42));
        let mut rng = ChaCha8Rng::seed_from_u64(42 ^ 0x51A7_C0DE);
        let mut seen_reinforce = false;
        let mut seen_attack = false;
        let mut seen_fortify = false;

        for _ in 0..2000 {
            if game.phase == Phase::Ended {
                break;
            }
            let take = match game.phase {
                Phase::Reinforce if !seen_reinforce => {
                    seen_reinforce = true;
                    true
                }
                Phase::Attack if !seen_attack => {
                    seen_attack = true;
                    true
                }
                Phase::Fortify if !seen_fortify => {
                    seen_fortify = true;
                    true
                }
                _ => states.len() < 8 && game.turns >= 3 && (game.turns % 2 == 1),
            };
            if take {
                states.push(Game::from_parts(
                    game.players,
                    game.max_turns,
                    game.owner,
                    game.troops,
                    game.active,
                    game.phase,
                    game.pending,
                    game.turns,
                    game.winner,
                    Some(123),
                ));
                if states.len() >= 8 && seen_reinforce && seen_attack && seen_fortify {
                    break;
                }
            }
            let act_id = heuristic_action(&game, topo, &mut rng);
            game.step(Action::decode(act_id).unwrap());
        }
        assert!(seen_reinforce && seen_attack && seen_fortify);
        states
    }

    #[test]
    fn pure_rust_infer_matches_burn_ndarray_random_init() {
        let device = NdArrayDevice::Cpu;
        let topo = BoardTopology::new();
        let graph = GraphBuffers::<NdArray>::from_topology(&topo, &device);

        for (hidden_dim, num_layers) in [(16, 1), (64, 2)] {
            let model = RiskPolicyConfig::new()
                .with_hidden_dim(hidden_dim)
                .with_num_layers(num_layers)
                .init::<NdArray>(&device);
            let weights = model.export_weights();

            for game in collect_diverse_states(&topo) {
                let obs = encode_observation(&game, game.active, &topo);
                let (burn_probs, burn_vals) = model.infer_batch(
                    obs.nodes.to_vec(),
                    obs.global.to_vec(),
                    obs.mask.to_vec(),
                    1,
                    &graph,
                    &device,
                );
                let (rust_probs, rust_val) = weights.infer(&obs, &topo);

                let max_prob_diff = burn_probs
                    .iter()
                    .zip(rust_probs.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let val_diff = (burn_vals[0] - rust_val).abs();

                assert!(
                    max_prob_diff < 1e-5,
                    "prob diff {max_prob_diff} >= 1e-5 for {hidden_dim}x{num_layers}"
                );
                assert!(
                    val_diff < 1e-5,
                    "val diff {val_diff} >= 1e-5 for {hidden_dim}x{num_layers}"
                );
                assert_eq!(
                    greedy_masked_action(&burn_probs, &obs.mask),
                    greedy_masked_action(&rust_probs, &obs.mask)
                );
            }
        }
    }

    #[test]
    fn pure_rust_infer_matches_local_checkpoint_if_present() {
        let ckpt_path = Path::new("runs/overnight/best.mpk");
        if !ckpt_path.exists() {
            return;
        }

        let device = NdArrayDevice::Cpu;
        let topo = BoardTopology::new();
        let graph = GraphBuffers::<NdArray>::from_topology(&topo, &device);

        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let record: RiskPolicyNetRecord<NdArray> = recorder
            .load(ckpt_path.to_path_buf(), &device)
            .expect("load local best.mpk");
        let model = RiskPolicyConfig::new()
            .with_hidden_dim(64)
            .with_num_layers(2)
            .init::<NdArray>(&device)
            .load_record(record);
        let exported = model.export_weights();

        for game in collect_diverse_states(&topo) {
            let obs = encode_observation(&game, game.active, &topo);
            let (burn_probs, burn_vals) = model.infer_batch(
                obs.nodes.to_vec(),
                obs.global.to_vec(),
                obs.mask.to_vec(),
                1,
                &graph,
                &device,
            );
            let (rust_probs, rust_val) = exported.infer(&obs, &topo);

            let max_prob_diff = burn_probs
                .iter()
                .zip(rust_probs.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let val_diff = (burn_vals[0] - rust_val).abs();
            assert!(max_prob_diff < 1e-5, "checkpoint prob diff {max_prob_diff}");
            assert!(val_diff < 1e-5, "checkpoint val diff {val_diff}");
            assert_eq!(
                greedy_masked_action(&burn_probs, &obs.mask),
                greedy_masked_action(&rust_probs, &obs.mask)
            );
        }
    }
}
