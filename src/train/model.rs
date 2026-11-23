//! Graph Neural Network Actor-Critic with edge-factorized action heads for Riskbot.

use crate::game::{ACTION_COUNT, N};
use crate::train::features::{BoardTopology, GLOBAL_FEAT_DIM, NODE_FEAT_DIM, NUM_EDGES};
pub use crate::train::infer::{
    GraphBlockWeights, LayerNormWeights, LinearWeights, RiskPolicyWeights,
};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::tensor::activation::{log_softmax, relu, softmax};

/// Static graph topology tensors cached on the compute device.
#[derive(Clone, Debug)]
pub struct GraphBuffers<B: Backend> {
    /// Row-normalized adjacency `[1, N, N]`.
    pub adj: Tensor<B, 3>,
    /// Directed edge source node indices `[NUM_EDGES]`.
    pub edge_src: Tensor<B, 1, Int>,
    /// Directed edge destination node indices `[NUM_EDGES]`.
    pub edge_dst: Tensor<B, 1, Int>,
}

impl<B: Backend> GraphBuffers<B> {
    pub fn from_topology(topo: &BoardTopology, device: &B::Device) -> Self {
        let adj =
            Tensor::<B, 3>::from_data(TensorData::new(topo.adj_norm.clone(), [1, N, N]), device);
        let edge_src = Tensor::<B, 1, Int>::from_ints(topo.edge_src.as_slice(), device);
        let edge_dst = Tensor::<B, 1, Int>::from_ints(topo.edge_dst.as_slice(), device);
        Self {
            adj,
            edge_src,
            edge_dst,
        }
    }
}

/// Residual message-passing graph block: `h' = h + ReLU(LayerNorm(W_self h + W_neigh (A h)))`.
#[derive(Module, Debug)]
pub struct GraphBlock<B: Backend> {
    w_self: Linear<B>,
    w_neigh: Linear<B>,
    norm: LayerNorm<B>,
}

impl<B: Backend> GraphBlock<B> {
    fn new(hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            w_self: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            w_neigh: LinearConfig::new(hidden_dim, hidden_dim)
                .with_bias(false)
                .init(device),
            norm: LayerNormConfig::new(hidden_dim).init(device),
        }
    }

    fn forward(&self, h: Tensor<B, 3>, adj_batched: &Tensor<B, 3>) -> Tensor<B, 3> {
        let neigh = adj_batched.clone().matmul(h.clone());
        let msg = self.w_self.forward(h.clone()) + self.w_neigh.forward(neigh);
        h + relu(self.norm.forward(msg))
    }
}

#[derive(Config, Debug)]
pub struct RiskPolicyConfig {
    #[config(default = 64)]
    pub hidden_dim: usize,
    #[config(default = 2)]
    pub num_layers: usize,
}

/// Graph Actor-Critic network outputting masked logits over all 1,016 Riskbot actions
/// and a scalar state value `V(s)`.
#[derive(Module, Debug)]
pub struct RiskPolicyNet<B: Backend> {
    node_in: Linear<B>,
    node_norm: LayerNorm<B>,
    blocks: Vec<GraphBlock<B>>,
    global_fuse: Linear<B>,
    global_norm: LayerNorm<B>,
    place_head: Linear<B>,
    edge_proj: Linear<B>,
    edge_norm: LayerNorm<B>,
    attack_head: Linear<B>,
    fortify_head: Linear<B>,
    phase_end_head: Linear<B>,
    value_fc1: Linear<B>,
    value_fc2: Linear<B>,
    hidden_dim: usize,
}

impl RiskPolicyConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RiskPolicyNet<B> {
        let d = self.hidden_dim;
        let blocks = (0..self.num_layers)
            .map(|_| GraphBlock::new(d, device))
            .collect();
        let val_hidden = (d / 2).max(16);

        RiskPolicyNet {
            node_in: LinearConfig::new(NODE_FEAT_DIM + GLOBAL_FEAT_DIM, d).init(device),
            node_norm: LayerNormConfig::new(d).init(device),
            blocks,
            global_fuse: LinearConfig::new(2 * d + GLOBAL_FEAT_DIM, d).init(device),
            global_norm: LayerNormConfig::new(d).init(device),
            place_head: LinearConfig::new(2 * d, 1).init(device),
            edge_proj: LinearConfig::new(3 * d, d).init(device),
            edge_norm: LayerNormConfig::new(d).init(device),
            attack_head: LinearConfig::new(d, 3).init(device),
            fortify_head: LinearConfig::new(d, 3).init(device),
            phase_end_head: LinearConfig::new(d, 2).init(device),
            value_fc1: LinearConfig::new(d, val_hidden).init(device),
            value_fc2: LinearConfig::new(val_hidden, 1).init(device),
            hidden_dim: d,
        }
    }
}

impl<B: Backend> RiskPolicyNet<B> {
    /// Forward pass returning `(masked_logits [B, ACTION_COUNT], values [B, 1])`.
    ///
    /// - `nodes`: `[B, N, NODE_FEAT_DIM]`
    /// - `global`: `[B, GLOBAL_FEAT_DIM]`
    /// - `mask`: `[B, ACTION_COUNT]` (`true` = legal)
    pub fn forward(
        &self,
        nodes: Tensor<B, 3>,
        global: Tensor<B, 2>,
        mask: Tensor<B, 2, Bool>,
        graph: &GraphBuffers<B>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [b, _n, _f] = nodes.dims();
        let d = self.hidden_dim;

        // Broadcast global context to every territory node.
        let g_nodes = global.clone().unsqueeze_dim::<3>(1).repeat_dim(1, N); // [B, 42, GLOBAL_FEAT_DIM]
        let node_input = Tensor::cat(vec![nodes, g_nodes], 2);

        let mut h = relu(self.node_norm.forward(self.node_in.forward(node_input)));

        let adj_batched = graph.adj.clone().repeat_dim(0, b);
        for block in &self.blocks {
            h = block.forward(h, &adj_batched);
        }

        // Global pooled state representation: [mean_pool, max_pool, raw_global].
        let h_mean = h.clone().mean_dim(1).reshape([b, d]);
        let h_max = h.clone().max_dim(1).reshape([b, d]);
        let g = relu(
            self.global_norm.forward(
                self.global_fuse
                    .forward(Tensor::cat(vec![h_mean, h_max, global], 1)),
            ),
        ); // [B, D]

        // 1) Place(a) logits: [B, 42]
        let g_rep = g.clone().unsqueeze_dim::<3>(1).repeat_dim(1, N);
        let place_logits = self
            .place_head
            .forward(Tensor::cat(vec![h.clone(), g_rep], 2))
            .reshape([b, N]);

        // 2) Edge embeddings for Attack(a, b, 1..=3) and Fortify(a, b, 0..=2): [B, 162, D]
        let h_src = h.clone().select(1, graph.edge_src.clone());
        let h_dst = h.select(1, graph.edge_dst.clone());
        let h_diff = h_src.clone() - h_dst.clone();
        let edge_h = relu(
            self.edge_norm.forward(
                self.edge_proj
                    .forward(Tensor::cat(vec![h_src, h_dst, h_diff], 2)),
            ),
        );

        let attack_logits = self
            .attack_head
            .forward(edge_h.clone())
            .reshape([b, NUM_EDGES * 3]);
        let fortify_logits = self
            .fortify_head
            .forward(edge_h)
            .reshape([b, NUM_EDGES * 3]);

        // 3) Phase-end logits: [FinishAttack, PassFortify] -> [B, 2]
        let end_logits = self.phase_end_head.forward(g.clone());

        let logits = Tensor::cat(
            vec![place_logits, attack_logits, fortify_logits, end_logits],
            1,
        )
        .clamp(-20.0, 20.0);
        debug_assert_eq!(logits.dims(), [b, ACTION_COUNT]);

        // -1e4 underflows to exact 0.0 under f32 exp() while avoiding extreme values in GPU reductions.
        let masked_logits = logits.mask_fill(mask.bool_not(), -1e4);

        // Value head: [B, 1]
        let value = self.value_fc2.forward(relu(self.value_fc1.forward(g)));
        (masked_logits, value)
    }

    /// Evaluate masked log-probabilities, exact masked entropy, and values for PPO updates.
    ///
    /// Returns `(chosen_logp [B, 1], entropy [B, 1], values [B, 1])`.
    pub fn evaluate_actions(
        &self,
        nodes: Tensor<B, 3>,
        global: Tensor<B, 2>,
        mask: Tensor<B, 2, Bool>,
        actions: Tensor<B, 2, Int>,
        graph: &GraphBuffers<B>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let (masked_logits, values) = self.forward(nodes, global, mask.clone(), graph);
        let logp = log_softmax(masked_logits, 1);
        let probs = logp.clone().exp();

        let invalid = mask.bool_not();
        let safe_logp = logp.clone().mask_fill(invalid.clone(), 0.0);
        let safe_probs = probs.mask_fill(invalid, 0.0);
        let entropy = -(safe_probs * safe_logp).sum_dim(1); // [B, 1]

        let chosen_logp = logp.gather(1, actions); // [B, 1]
        (chosen_logp, entropy, values)
    }

    /// Batched inference returning `(probs_flat [B * ACTION_COUNT], values [B])` on host.
    pub fn infer_batch(
        &self,
        nodes_flat: Vec<f32>,
        global_flat: Vec<f32>,
        mask_flat: Vec<bool>,
        batch_size: usize,
        graph: &GraphBuffers<B>,
        device: &B::Device,
    ) -> (Vec<f32>, Vec<f32>) {
        let nodes = Tensor::<B, 3>::from_data(
            TensorData::new(nodes_flat, [batch_size, N, NODE_FEAT_DIM]),
            device,
        );
        let global = Tensor::<B, 2>::from_data(
            TensorData::new(global_flat, [batch_size, GLOBAL_FEAT_DIM]),
            device,
        );
        let mask = Tensor::<B, 2, Bool>::from_bool(
            TensorData::new(mask_flat, [batch_size, ACTION_COUNT]),
            device,
        );

        let (masked_logits, values) = self.forward(nodes, global, mask, graph);
        let probs = softmax(masked_logits, 1);

        let probs_vec = probs
            .into_data()
            .to_vec::<f32>()
            .expect("probs tensor converts to f32");
        let values_vec = values
            .into_data()
            .to_vec::<f32>()
            .expect("values tensor converts to f32");
        (probs_vec, values_vec)
    }

    /// Batched value-only estimate (used for rollout GAE bootstrapping).
    pub fn value_batch(
        &self,
        nodes_flat: Vec<f32>,
        global_flat: Vec<f32>,
        mask_flat: Vec<bool>,
        batch_size: usize,
        graph: &GraphBuffers<B>,
        device: &B::Device,
    ) -> Vec<f32> {
        let nodes = Tensor::<B, 3>::from_data(
            TensorData::new(nodes_flat, [batch_size, N, NODE_FEAT_DIM]),
            device,
        );
        let global = Tensor::<B, 2>::from_data(
            TensorData::new(global_flat, [batch_size, GLOBAL_FEAT_DIM]),
            device,
        );
        let mask = Tensor::<B, 2, Bool>::from_bool(
            TensorData::new(mask_flat, [batch_size, ACTION_COUNT]),
            device,
        );
        let (_logits, values) = self.forward(nodes, global, mask, graph);
        values
            .into_data()
            .to_vec::<f32>()
            .expect("values tensor converts to f32")
    }

    /// Export a read-only CPU copy of all trainable linear and layer-norm parameters.
    ///
    /// Weight matrices follow Burn 0.20's row-major `[d_input, d_output]` convention (`y = x @ W + b`),
    /// and every `LayerNorm` uses Burn's default `epsilon = 1e-5` with biased variance.
    pub fn export_weights(&self) -> RiskPolicyWeights {
        RiskPolicyWeights {
            hidden_dim: self.hidden_dim,
            num_layers: self.blocks.len(),
            node_in: extract_linear(&self.node_in),
            node_norm: extract_layer_norm(&self.node_norm),
            blocks: self
                .blocks
                .iter()
                .map(|block| GraphBlockWeights {
                    w_self: extract_linear(&block.w_self),
                    w_neigh: extract_linear(&block.w_neigh),
                    norm: extract_layer_norm(&block.norm),
                })
                .collect(),
            global_fuse: extract_linear(&self.global_fuse),
            global_norm: extract_layer_norm(&self.global_norm),
            place_head: extract_linear(&self.place_head),
            edge_proj: extract_linear(&self.edge_proj),
            edge_norm: extract_layer_norm(&self.edge_norm),
            attack_head: extract_linear(&self.attack_head),
            fortify_head: extract_linear(&self.fortify_head),
            phase_end_head: extract_linear(&self.phase_end_head),
            value_fc1: extract_linear(&self.value_fc1),
            value_fc2: extract_linear(&self.value_fc2),
        }
    }
}

fn extract_linear<B: Backend>(layer: &Linear<B>) -> LinearWeights {
    let w = layer.weight.val();
    let shape = w.dims();
    let weight = w
        .into_data()
        .to_vec::<f32>()
        .expect("linear weight tensor converts to f32");
    let bias = layer.bias.as_ref().map(|b| {
        b.val()
            .into_data()
            .to_vec::<f32>()
            .expect("linear bias tensor converts to f32")
    });
    LinearWeights {
        shape,
        weight,
        bias,
    }
}

fn extract_layer_norm<B: Backend>(layer: &LayerNorm<B>) -> LayerNormWeights {
    let gamma_tensor = layer.gamma.val();
    let [d_model] = gamma_tensor.dims();
    let gamma = gamma_tensor
        .into_data()
        .to_vec::<f32>()
        .expect("layernorm gamma tensor converts to f32");
    let beta = layer.beta.as_ref().map(|b| {
        b.val()
            .into_data()
            .to_vec::<f32>()
            .expect("layernorm beta tensor converts to f32")
    });
    let epsilon = LayerNormConfig::new(d_model).epsilon;
    LayerNormWeights {
        d_model,
        epsilon,
        gamma,
        beta,
    }
}
