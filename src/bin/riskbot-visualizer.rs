use std::path::PathBuf;

fn main() -> eframe::Result {
    let mut paths = Vec::new();
    for arg in std::env::args().skip(1) {
        if arg == "-h" || arg == "--help" {
            println!("Usage: riskbot-visualizer [CHECKPOINT_OR_RUNS_DIR ...]");
            println!(
                "Interactive Risk board where a human can play against 1-5 trained checkpoints or bots."
            );
            println!("Defaults to scanning 'runs/' for .mpk checkpoints.");
            return Ok(());
        }
        paths.push(PathBuf::from(arg));
    }
    if paths.is_empty() {
        paths.push(PathBuf::from("runs"));
    }
    riskbot::visualizer::run_with_paths(paths)
}
