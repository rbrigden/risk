//! Live JSONL metrics streaming and atomic JSON league telemetry writer.

use crate::train::league::{EloPoint, MemberKind, PfspMode};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Per-update training and evaluation metrics written as one line of `metrics.jsonl`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricsRecord {
    pub update: usize,
    pub total_env_steps: usize,
    pub elapsed_secs: f64,
    pub steps_per_sec: f64,
    pub learning_rate: f64,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub approx_kl: f32,
    pub clip_frac: f32,
    pub episodes_finished: usize,
    pub win_frac: f32,
    pub avg_turns: f32,
    pub learner_elo: f64,
    pub pool_size: usize,
    pub eval_random_win_rate: Option<f32>,
    pub eval_heuristic_win_rate: Option<f32>,
    pub eval_league_win_rate: Option<f32>,
    pub eval_random_avg_turns: Option<f32>,
    pub eval_heuristic_avg_turns: Option<f32>,
}

/// Current Elo ratings of the baseline anchor opponents.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnchorElos {
    pub random_elo: f64,
    pub heuristic_elo: f64,
}

/// Serializable leaderboard row for a single league opponent.
///
/// Note that `wins` and `losses` are from the **opponent's** perspective
/// (`wins = learner_losses`, `losses = learner_wins`), while `learner_win_rate`
/// is the learner's smoothed win rate against this opponent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeagueMemberReport {
    pub id: String,
    pub kind: MemberKind,
    pub created_update: usize,
    pub elo: f64,
    pub games: u64,
    pub wins: u64,
    pub losses: u64,
    pub draws: u64,
    pub learner_win_rate: f32,
    pub pfsp_weight: f32,
    pub active: bool,
}

/// Complete league state snapshot serialized atomically to `league.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeagueReport {
    pub run_name: String,
    pub players: usize,
    pub max_turns: u32,
    pub backend: String,
    pub pfsp_mode: PfspMode,
    pub pfsp_p: f32,
    pub current_update: usize,
    pub total_updates: usize,
    pub learner_elo: f64,
    pub anchors: AnchorElos,
    pub members: Vec<LeagueMemberReport>,
    pub elo_history: Vec<EloPoint>,
}

/// Streams per-update metrics to `metrics.jsonl` and atomically writes `league.json`.
pub struct TelemetryWriter {
    dir: PathBuf,
    metrics_file: BufWriter<File>,
}

impl TelemetryWriter {
    /// Creates `dir` if needed and opens a fresh truncated `metrics.jsonl` file.
    pub fn new(dir: &Path) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        let metrics_path = dir.join("metrics.jsonl");
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(metrics_path)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            metrics_file: BufWriter::new(file),
        })
    }

    /// Returns the output directory managed by this writer.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Returns the path to `metrics.jsonl`.
    pub fn metrics_path(&self) -> PathBuf {
        self.dir.join("metrics.jsonl")
    }

    /// Returns the path to `league.json`.
    pub fn league_path(&self) -> PathBuf {
        self.dir.join("league.json")
    }

    /// Appends a single JSON-encoded `MetricsRecord` line and flushes immediately
    /// so live dashboards observe each update in real time.
    pub fn append_metrics(&mut self, record: &MetricsRecord) -> io::Result<()> {
        serde_json::to_writer(&mut self.metrics_file, record).map_err(io::Error::other)?;
        self.metrics_file.write_all(b"\n")?;
        self.metrics_file.flush()
    }

    /// Atomically writes `league.json` by writing `league.json.tmp` and renaming it into place.
    pub fn write_league(&self, report: &LeagueReport) -> io::Result<()> {
        let tmp_path = self.dir.join("league.json.tmp");
        let final_path = self.dir.join("league.json");
        {
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp_path)?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer_pretty(&mut writer, report).map_err(io::Error::other)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
        fs::rename(&tmp_path, &final_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::league::{LeagueConfig, LeagueState, MatchOutcome, OpponentKey};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}_{}_{nanos}", std::process::id()))
    }

    #[test]
    fn telemetry_writer_round_trip_metrics_and_league() {
        let dir = unique_temp_dir("riskbot_telemetry_test");
        let mut writer = TelemetryWriter::new(&dir).unwrap();

        let rec1 = MetricsRecord {
            update: 1,
            total_env_steps: 2048,
            elapsed_secs: 1.5,
            steps_per_sec: 1365.3,
            learning_rate: 3e-4,
            policy_loss: -0.02,
            value_loss: 0.14,
            entropy: 1.85,
            approx_kl: 0.011,
            clip_frac: 0.09,
            episodes_finished: 12,
            win_frac: 0.58,
            avg_turns: 44.2,
            learner_elo: 1024.5,
            pool_size: 1,
            eval_random_win_rate: Some(0.9),
            eval_heuristic_win_rate: Some(0.35),
            eval_league_win_rate: Some(0.55),
            eval_random_avg_turns: Some(28.0),
            eval_heuristic_avg_turns: Some(61.5),
        };
        let rec2 = MetricsRecord {
            update: 2,
            total_env_steps: 4096,
            elapsed_secs: 3.0,
            eval_random_win_rate: None,
            eval_heuristic_win_rate: None,
            eval_league_win_rate: None,
            eval_random_avg_turns: None,
            eval_heuristic_avg_turns: None,
            ..rec1.clone()
        };

        writer.append_metrics(&rec1).unwrap();
        writer.append_metrics(&rec2).unwrap();

        let mut league = LeagueState::new(LeagueConfig::default());
        league.add_snapshot(1);
        league.record_match(OpponentKey::Random, MatchOutcome::LearnerWin);
        league.record_elo_point(2, 4096);
        let report = league.build_report("unit_test_run", 200, "wgpu", 2, 50);
        writer.write_league(&report).unwrap();

        // Temporary file must not remain after atomic rename.
        assert!(!dir.join("league.json.tmp").exists());

        // Deserialize metrics.jsonl line-by-line.
        let raw_metrics = fs::read_to_string(writer.metrics_path()).unwrap();
        let parsed_metrics: Vec<MetricsRecord> = raw_metrics
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(parsed_metrics.len(), 2);
        assert_eq!(parsed_metrics[0].update, 1);
        assert_eq!(parsed_metrics[0].eval_random_win_rate, Some(0.9));
        assert_eq!(parsed_metrics[1].update, 2);
        assert_eq!(parsed_metrics[1].eval_random_win_rate, None);

        // Deserialize league.json.
        let raw_league = fs::read_to_string(writer.league_path()).unwrap();
        let parsed_league: LeagueReport = serde_json::from_str(&raw_league).unwrap();
        assert_eq!(parsed_league.run_name, "unit_test_run");
        assert_eq!(parsed_league.players, 2);
        assert_eq!(parsed_league.max_turns, 200);
        assert_eq!(parsed_league.backend, "wgpu");
        assert_eq!(parsed_league.pfsp_mode, PfspMode::Hard);
        assert_eq!(parsed_league.anchors.random_elo, 800.0);
        assert_eq!(parsed_league.members.len(), 3);
        assert_eq!(parsed_league.elo_history.len(), 1);

        let _ = fs::remove_dir_all(&dir);
    }
}
