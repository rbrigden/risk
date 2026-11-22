#![cfg_attr(feature = "train", recursion_limit = "256")]

mod game;

#[cfg(not(target_arch = "wasm32"))]
mod python;

#[cfg(feature = "visualizer")]
pub mod visualizer;

#[cfg(all(feature = "visualizer", not(target_arch = "wasm32")))]
pub mod dashboard;

#[cfg(feature = "train")]
pub mod train;
