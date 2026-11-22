use std::path::PathBuf;
use std::process::ExitCode;

const DEFAULT_PORT: u16 = 8767;

fn print_help() {
    println!("Usage: riskbot-web [OPTIONS] [CHECKPOINT_OR_RUNS_DIR ...]");
    println!();
    println!("Local HTTP/1.1 + JSON server for the browser Riskbot visualizer.");
    println!();
    println!("Options:");
    println!("  -p, --port <PORT>          Port to listen on (default: {DEFAULT_PORT})");
    println!("  -c, --checkpoint <PATH>    Default opponent .mpk checkpoint to load on boot");
    println!("  -h, --help                 Show this help message");
    println!();
    println!("Defaults to scanning 'runs/' for .mpk checkpoints.");
}

fn main() -> ExitCode {
    let mut port: u16 = DEFAULT_PORT;
    let mut default_checkpoint: Option<PathBuf> = None;
    let mut roots: Vec<PathBuf> = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                return ExitCode::SUCCESS;
            }
            "-p" | "--port" => {
                let Some(val) = args.next() else {
                    eprintln!("Error: {arg} requires a port number.");
                    return ExitCode::FAILURE;
                };
                match val.parse::<u16>() {
                    Ok(p) => port = p,
                    Err(_) => {
                        eprintln!("Error: invalid port '{val}'.");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "-c" | "--checkpoint" => {
                let Some(val) = args.next() else {
                    eprintln!("Error: {arg} requires a checkpoint path.");
                    return ExitCode::FAILURE;
                };
                default_checkpoint = Some(PathBuf::from(val));
            }
            other if other.starts_with("--port=") => {
                let val = &other["--port=".len()..];
                match val.parse::<u16>() {
                    Ok(p) => port = p,
                    Err(_) => {
                        eprintln!("Error: invalid port '{val}'.");
                        return ExitCode::FAILURE;
                    }
                }
            }
            other if other.starts_with("--checkpoint=") => {
                let val = &other["--checkpoint=".len()..];
                if val.is_empty() {
                    eprintln!("Error: --checkpoint requires a non-empty path.");
                    return ExitCode::FAILURE;
                }
                default_checkpoint = Some(PathBuf::from(val));
            }
            other if other.starts_with('-') => {
                eprintln!("Error: unknown option '{other}'. Run with --help for usage.");
                return ExitCode::FAILURE;
            }
            _ => roots.push(PathBuf::from(arg)),
        }
    }

    if roots.is_empty() {
        roots.push(PathBuf::from("runs"));
    }

    if let Err(err) = riskbot::visualizer::server::serve(port, roots, default_checkpoint) {
        eprintln!("Error starting riskbot-web server: {err}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}
