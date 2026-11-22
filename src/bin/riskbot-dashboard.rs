use std::path::PathBuf;

fn main() -> eframe::Result {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(arg) if arg == "-h" || arg == "--help" => {
            println!("Usage: riskbot-dashboard [RUN_OR_RUNS_DIR]");
            println!("Defaults to 'runs'. Watches metrics.jsonl and league.json live.");
            return Ok(());
        }
        Some(arg) => PathBuf::from(arg),
        None => PathBuf::from("runs"),
    };
    riskbot::dashboard::run(path)
}
