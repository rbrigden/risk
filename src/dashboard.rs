//! Live native desktop Training & Elo Dashboard for Riskbot runs.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eframe::egui::{
    self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Shape, Stroke, StrokeKind, Vec2,
    pos2, vec2,
};
use serde::Deserialize;

const PANEL_BG: Color32 = Color32::from_rgb(19, 28, 38); // #131c26
const CARD_BG: Color32 = Color32::from_rgb(27, 40, 54); // #1b2836
const PLOT_BG: Color32 = Color32::from_rgb(16, 25, 34);
const BORDER_SUBTLE: Color32 = Color32::from_rgb(43, 62, 80);
const GRID_LINE: Color32 = Color32::from_rgb(33, 49, 64);
const TEXT_PRIMARY: Color32 = Color32::from_rgb(232, 240, 247);
const TEXT_MUTED: Color32 = Color32::from_rgb(142, 165, 184);

const CYAN: Color32 = Color32::from_rgb(90, 199, 239); // #5ac7ef
const CORAL: Color32 = Color32::from_rgb(255, 150, 115); // #ff9673
const GREEN: Color32 = Color32::from_rgb(183, 222, 116); // #b7de74
const GOLD: Color32 = Color32::from_rgb(242, 201, 76); // #f2c94c
const LAVENDER: Color32 = Color32::from_rgb(182, 152, 248);

const REFRESH_INTERVAL: Duration = Duration::from_secs(1);

/// Launch the native Training & Elo Dashboard watching `path`
/// (either a single run directory or a parent `runs/` directory).
pub fn run(path: PathBuf) -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1320.0, 880.0])
            .with_min_inner_size([860.0, 620.0]),
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    eframe::run_native(
        "Riskbot · Training & Elo Dashboard",
        options,
        Box::new(move |cc| Ok(Box::new(DashboardApp::new(cc, path)))),
    )
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct MetricRecord {
    pub update: u64,
    pub total_env_steps: u64,
    pub elapsed_secs: f64,
    pub steps_per_sec: f64,
    pub learning_rate: f64,
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
    pub approx_kl: f64,
    pub clip_frac: f64,
    pub episodes_finished: u64,
    pub win_frac: f64,
    pub avg_turns: f64,
    pub learner_elo: Option<f64>,
    pub pool_size: Option<usize>,
    pub eval_random_win_rate: Option<f64>,
    pub eval_heuristic_win_rate: Option<f64>,
    pub eval_league_win_rate: Option<f64>,
    pub eval_random_avg_turns: Option<f64>,
    pub eval_heuristic_avg_turns: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AnchorElos {
    pub random_elo: f64,
    pub heuristic_elo: f64,
}

impl Default for AnchorElos {
    fn default() -> Self {
        Self {
            random_elo: 800.0,
            heuristic_elo: 1200.0,
        }
    }
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct LeagueMember {
    pub id: String,
    pub kind: String,
    pub created_update: u64,
    pub elo: f64,
    pub games: u64,
    pub wins: u64,
    pub losses: u64,
    pub draws: u64,
    pub learner_win_rate: f64,
    pub pfsp_weight: f64,
    #[serde(default = "default_true")]
    pub active: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct EloHistoryPoint {
    pub update: u64,
    pub steps: u64,
    pub learner_elo: f64,
    pub best_snapshot_elo: f64,
    pub mean_snapshot_elo: f64,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct LeagueSnapshot {
    pub run_name: String,
    pub players: usize,
    pub max_turns: u32,
    pub backend: String,
    pub pfsp_mode: String,
    pub pfsp_p: f64,
    pub current_update: u64,
    pub total_updates: u64,
    pub learner_elo: f64,
    pub anchors: AnchorElos,
    pub members: Vec<LeagueMember>,
    pub elo_history: Vec<EloHistoryPoint>,
}

/// Parse newline-delimited JSON metrics, tolerating blank and partial trailing lines.
pub fn parse_metrics_jsonl_str(content: &str) -> Vec<MetricRecord> {
    content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter_map(|line| serde_json::from_str::<MetricRecord>(line).ok())
        .collect()
}

/// Read and parse `<run_dir>/metrics.jsonl`. Returns `None` if unreadable.
pub fn load_metrics_jsonl(path: &Path) -> Option<Vec<MetricRecord>> {
    let content = fs::read_to_string(path).ok()?;
    Some(parse_metrics_jsonl_str(&content))
}

/// Read and parse `<run_dir>/league.json`. Returns `None` if missing, unreadable, or partial JSON.
pub fn load_league_json(path: &Path) -> Option<LeagueSnapshot> {
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str::<LeagueSnapshot>(&content).ok()
}

fn is_run_dir(path: &Path) -> bool {
    path.is_dir() && (path.join("metrics.jsonl").is_file() || path.join("league.json").is_file())
}

/// Discover run directories at or directly underneath `root`.
/// If `root` itself contains telemetry files, returns `[root]`.
/// Otherwise scans immediate subdirectories of `root` and returns them sorted by path.
pub fn discover_runs(root: &Path) -> Vec<PathBuf> {
    if is_run_dir(root) {
        return vec![root.to_path_buf()];
    }
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut runs: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| is_run_dir(path))
        .collect();
    runs.sort();
    runs
}

fn run_display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.display().to_string())
}

fn latest_telemetry_mtime(run_dir: &Path) -> Option<std::time::SystemTime> {
    let m1 = fs::metadata(run_dir.join("metrics.jsonl"))
        .and_then(|m| m.modified())
        .ok();
    let m2 = fs::metadata(run_dir.join("league.json"))
        .and_then(|m| m.modified())
        .ok();
    match (m1, m2) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

#[derive(Default)]
struct RunTelemetry {
    metrics: Vec<MetricRecord>,
    league: Option<LeagueSnapshot>,
    league_stale: bool,
}

impl RunTelemetry {
    fn reload_from(&mut self, run_dir: &Path) {
        if let Some(records) = load_metrics_jsonl(&run_dir.join("metrics.jsonl"))
            && (!records.is_empty() || self.metrics.is_empty())
        {
            self.metrics = records;
        }
        match load_league_json(&run_dir.join("league.json")) {
            Some(snapshot) => {
                self.league = Some(snapshot);
                self.league_stale = false;
            }
            None => {
                if self.league.is_some() {
                    self.league_stale = true;
                }
            }
        }
    }
}

struct DashboardApp {
    watch_root: PathBuf,
    runs: Vec<PathBuf>,
    selected_run: Option<PathBuf>,
    telemetry: RunTelemetry,
    paused: bool,
    last_refresh: Instant,
}

impl DashboardApp {
    fn new(cc: &eframe::CreationContext<'_>, watch_root: PathBuf) -> Self {
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = PANEL_BG;
        visuals.window_fill = CARD_BG;
        visuals.extreme_bg_color = PLOT_BG;
        visuals.widgets.inactive.bg_fill = Color32::from_rgb(39, 57, 73);
        visuals.widgets.inactive.weak_bg_fill = visuals.widgets.inactive.bg_fill;
        visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(54, 78, 96);
        visuals.widgets.active.weak_bg_fill = Color32::from_rgb(62, 92, 115);
        visuals.selection.bg_fill = Color32::from_rgba_unmultiplied(90, 199, 239, 70);
        visuals.selection.stroke = Stroke::new(1.0, CYAN);
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        cc.egui_ctx.set_visuals(visuals);

        let mut app = Self {
            watch_root,
            runs: Vec::new(),
            selected_run: None,
            telemetry: RunTelemetry::default(),
            paused: false,
            last_refresh: Instant::now(),
        };
        app.refresh();
        app
    }

    fn select_run(&mut self, run_dir: PathBuf) {
        if self.selected_run.as_ref() != Some(&run_dir) {
            self.selected_run = Some(run_dir);
            self.telemetry = RunTelemetry::default();
        }
        if let Some(active) = self.selected_run.clone() {
            self.telemetry.reload_from(&active);
        }
        self.last_refresh = Instant::now();
    }

    fn refresh(&mut self) {
        self.runs = discover_runs(&self.watch_root);

        if let Some(current) = &self.selected_run
            && !self.runs.is_empty()
            && !self.runs.contains(current)
        {
            self.selected_run = None;
            self.telemetry = RunTelemetry::default();
        }

        if self.selected_run.is_none() && !self.runs.is_empty() {
            let best = self
                .runs
                .iter()
                .max_by(|a, b| latest_telemetry_mtime(a).cmp(&latest_telemetry_mtime(b)))
                .cloned()
                .unwrap_or_else(|| self.runs[0].clone());
            self.selected_run = Some(best);
            self.telemetry = RunTelemetry::default();
        }

        let target = self
            .selected_run
            .clone()
            .unwrap_or_else(|| self.watch_root.clone());
        self.telemetry.reload_from(&target);
        self.last_refresh = Instant::now();
    }

    fn draw_header(&mut self, ui: &mut egui::Ui) {
        let mut switch_to: Option<PathBuf> = None;

        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new("Riskbot")
                    .size(26.0)
                    .strong()
                    .color(TEXT_PRIMARY),
            );
            ui.label(
                RichText::new("Training & Elo Dashboard")
                    .size(18.0)
                    .color(CYAN),
            );

            ui.add_space(8.0);
            if let Some(league) = &self.telemetry.league {
                if !league.backend.is_empty() {
                    badge(ui, &league.backend, CYAN);
                }
                if !league.pfsp_mode.is_empty() {
                    badge(
                        ui,
                        &format!("PFSP {} (p={:.1})", league.pfsp_mode, league.pfsp_p),
                        GOLD,
                    );
                }
                if league.players > 0 {
                    badge(
                        ui,
                        &format!("{}P · {}t", league.players, league.max_turns),
                        GREEN,
                    );
                }
            }
        });

        ui.add_space(6.0);

        ui.horizontal_wrapped(|ui| {
            if self.runs.len() > 1 {
                let current_label = self
                    .selected_run
                    .as_deref()
                    .map(run_display_name)
                    .unwrap_or_else(|| "Select run".to_owned());
                ui.label(RichText::new("Run:").color(TEXT_MUTED));
                egui::ComboBox::from_id_salt("run_selector")
                    .selected_text(RichText::new(current_label).strong().color(TEXT_PRIMARY))
                    .width(180.0)
                    .show_ui(ui, |ui| {
                        for run in &self.runs {
                            let label = run_display_name(run);
                            let selected = self.selected_run.as_ref() == Some(run);
                            if ui.selectable_label(selected, label).clicked() {
                                switch_to = Some(run.clone());
                            }
                        }
                    });
                ui.add_space(6.0);
            } else {
                let active_path = self
                    .selected_run
                    .as_deref()
                    .unwrap_or(self.watch_root.as_path());
                ui.label(
                    RichText::new(format!("Watching: {}", active_path.display()))
                        .monospace()
                        .size(12.0)
                        .color(TEXT_MUTED),
                );
                ui.add_space(6.0);
            }

            let (toggle_label, toggle_color) = if self.paused {
                ("● Paused", GOLD)
            } else {
                ("● Live (1s)", GREEN)
            };
            if ui
                .add(
                    egui::Button::new(RichText::new(toggle_label).strong().color(toggle_color))
                        .min_size(vec2(96.0, 26.0)),
                )
                .clicked()
            {
                self.paused = !self.paused;
                if !self.paused {
                    self.refresh();
                }
            }

            if ui
                .add(egui::Button::new("Refresh").min_size(vec2(76.0, 26.0)))
                .clicked()
            {
                self.refresh();
            }

            let elapsed = self.last_refresh.elapsed().as_secs_f32();
            let status = if self.telemetry.metrics.is_empty() && self.telemetry.league.is_none() {
                "Waiting for telemetry files…".to_owned()
            } else if self.telemetry.league_stale {
                format!("Updated {elapsed:.1}s ago · keeping last valid league.json")
            } else {
                format!(
                    "Updated {elapsed:.1}s ago · {} metric rows",
                    self.telemetry.metrics.len()
                )
            };
            ui.label(RichText::new(status).size(12.0).color(TEXT_MUTED));
        });

        if let Some(run_dir) = switch_to {
            self.select_run(run_dir);
        }
    }

    fn draw_kpi_row(&self, ui: &mut egui::Ui) {
        let latest = self.telemetry.metrics.last();
        let league = self.telemetry.league.as_ref();

        let learner_elo = league
            .map(|l| l.learner_elo)
            .or_else(|| latest.and_then(|m| m.learner_elo));
        let first_elo = league
            .and_then(|l| l.elo_history.first().map(|p| p.learner_elo))
            .or_else(|| self.telemetry.metrics.iter().find_map(|m| m.learner_elo));
        let elo_delta = match (learner_elo, first_elo) {
            (Some(curr), Some(start)) => Some(curr - start),
            _ => None,
        };

        let current_update = league
            .map(|l| l.current_update)
            .or_else(|| latest.map(|m| m.update))
            .unwrap_or(0);
        let total_updates = league.map(|l| l.total_updates).unwrap_or(0);

        let latest_eval_random = self
            .telemetry
            .metrics
            .iter()
            .rev()
            .find_map(|m| m.eval_random_win_rate);
        let latest_eval_heuristic = self
            .telemetry
            .metrics
            .iter()
            .rev()
            .find_map(|m| m.eval_heuristic_win_rate);
        let latest_eval_league = self
            .telemetry
            .metrics
            .iter()
            .rev()
            .find_map(|m| m.eval_league_win_rate);

        let cols = if ui.available_width() >= 960.0 { 4 } else { 2 };
        ui.columns(cols, |columns| {
            // Card 1: Learner Elo
            card_frame(&mut columns[0], |ui| {
                ui.label(RichText::new("LEARNER ELO").size(11.0).color(TEXT_MUTED));
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    let elo_text = learner_elo
                        .map(|e| format!("{e:.1}"))
                        .unwrap_or_else(|| "—".to_owned());
                    ui.label(RichText::new(elo_text).size(28.0).strong().color(GOLD));
                    if let Some(delta) = elo_delta {
                        let color = if delta >= 0.0 { GREEN } else { CORAL };
                        ui.label(
                            RichText::new(format!("{delta:+.1}"))
                                .size(14.0)
                                .strong()
                                .color(color),
                        );
                    }
                });
                ui.add_space(2.0);
                let pool_count = league
                    .map(|l| l.members.iter().filter(|m| m.active).count())
                    .or_else(|| latest.and_then(|m| m.pool_size));
                let sub = match pool_count {
                    Some(n) => format!("Active pool: {n} opponents"),
                    None => "No league snapshot yet".to_owned(),
                };
                ui.label(RichText::new(sub).size(12.0).color(TEXT_MUTED));
            });

            // Card 2: Training Progress
            card_frame(&mut columns[1], |ui| {
                ui.label(
                    RichText::new("TRAINING PROGRESS")
                        .size(11.0)
                        .color(TEXT_MUTED),
                );
                ui.add_space(4.0);
                let progress_text = if total_updates > 0 {
                    format!("{current_update} / {total_updates}")
                } else {
                    format!("Update {current_update}")
                };
                ui.label(
                    RichText::new(progress_text)
                        .size(22.0)
                        .strong()
                        .color(TEXT_PRIMARY),
                );
                ui.add_space(4.0);
                let frac = if total_updates > 0 {
                    (current_update as f32 / total_updates as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                draw_progress_bar(ui, frac, CYAN);
                ui.add_space(4.0);
                let elapsed = latest
                    .map(|m| format_duration(m.elapsed_secs))
                    .unwrap_or_else(|| "—".to_owned());
                let pct = if total_updates > 0 {
                    format!("{:.1}% · Elapsed {elapsed}", frac * 100.0)
                } else {
                    format!("Elapsed {elapsed}")
                };
                ui.label(RichText::new(pct).size(12.0).color(TEXT_MUTED));
            });

            // Card 3: Throughput & Steps
            let col3 = if cols == 4 { 2 } else { 0 };
            if cols == 2 {
                columns[0].add_space(8.0);
            }
            card_frame(&mut columns[col3], |ui| {
                ui.label(
                    RichText::new("ENV STEPS & SPEED")
                        .size(11.0)
                        .color(TEXT_MUTED),
                );
                ui.add_space(4.0);
                let steps = latest
                    .map(|m| format_compact_u64(m.total_env_steps))
                    .or_else(|| {
                        league
                            .and_then(|l| l.elo_history.last())
                            .map(|h| format_compact_u64(h.steps))
                    })
                    .unwrap_or_else(|| "—".to_owned());
                ui.label(RichText::new(steps).size(26.0).strong().color(CYAN));
                ui.add_space(2.0);
                let speed = latest
                    .map(|m| {
                        format!(
                            "{:.0} steps/s · LR {:.2e}",
                            m.steps_per_sec, m.learning_rate
                        )
                    })
                    .unwrap_or_else(|| "—".to_owned());
                ui.label(RichText::new(speed).size(12.0).color(TEXT_MUTED));
            });

            // Card 4: Eval Win Rates
            let col4 = if cols == 4 { 3 } else { 1 };
            if cols == 2 {
                columns[1].add_space(8.0);
            }
            card_frame(&mut columns[col4], |ui| {
                ui.label(
                    RichText::new("LATEST EVAL WIN RATES")
                        .size(11.0)
                        .color(TEXT_MUTED),
                );
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    eval_stat(ui, "Rnd", latest_eval_random, GREEN);
                    ui.add_space(6.0);
                    eval_stat(ui, "Heur", latest_eval_heuristic, CORAL);
                    ui.add_space(6.0);
                    eval_stat(ui, "Lge", latest_eval_league, CYAN);
                });
                ui.add_space(4.0);
                let rollout_info = latest
                    .map(|m| {
                        format!(
                            "Train win {:.0}% · {:.1} turns · clip {:.1}%",
                            m.win_frac * 100.0,
                            m.avg_turns,
                            m.clip_frac * 100.0
                        )
                    })
                    .unwrap_or_else(|| "No eval data yet".to_owned());
                ui.label(RichText::new(rollout_info).size(12.0).color(TEXT_MUTED));
            });
        });
    }

    fn draw_primary_charts(&self, ui: &mut egui::Ui) {
        let two_col = ui.available_width() >= 920.0;
        if two_col {
            ui.columns(2, |columns| {
                self.draw_elo_chart_card(&mut columns[0]);
                self.draw_eval_chart_card(&mut columns[1]);
            });
        } else {
            self.draw_elo_chart_card(ui);
            ui.add_space(10.0);
            self.draw_eval_chart_card(ui);
        }
    }

    fn draw_elo_chart_card(&self, ui: &mut egui::Ui) {
        let anchors = self
            .telemetry
            .league
            .as_ref()
            .map(|l| l.anchors.clone())
            .unwrap_or_default();

        let mut learner_pts = Vec::new();
        let mut best_pts = Vec::new();
        let mut mean_pts = Vec::new();

        if let Some(league) = &self.telemetry.league
            && !league.elo_history.is_empty()
        {
            for point in &league.elo_history {
                let x = point.update as f64;
                learner_pts.push((x, point.learner_elo));
                best_pts.push((x, point.best_snapshot_elo));
                mean_pts.push((x, point.mean_snapshot_elo));
            }
        } else {
            for record in &self.telemetry.metrics {
                if let Some(elo) = record.learner_elo {
                    learner_pts.push((record.update as f64, elo));
                }
            }
        }

        let mut series = vec![ChartSeries {
            name: "Learner",
            color: CYAN,
            stroke_width: 2.3,
            points: learner_pts,
            show_dots: false,
        }];
        if !best_pts.is_empty() {
            series.push(ChartSeries {
                name: "Best Snap",
                color: GOLD,
                stroke_width: 1.7,
                points: best_pts,
                show_dots: false,
            });
        }
        if !mean_pts.is_empty() {
            series.push(ChartSeries {
                name: "Mean Snap",
                color: GREEN,
                stroke_width: 1.5,
                points: mean_pts,
                show_dots: false,
            });
        }

        let random_label = format!("Random ({:.0})", anchors.random_elo);
        let heuristic_label = format!("Heuristic ({:.0})", anchors.heuristic_elo);
        let refs = vec![
            RefLine {
                value: anchors.random_elo,
                label: random_label,
                color: CORAL.gamma_multiply(0.75),
            },
            RefLine {
                value: anchors.heuristic_elo,
                label: heuristic_label,
                color: LAVENDER.gamma_multiply(0.8),
            },
        ];

        card_frame(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new("Elo Progression")
                        .size(15.0)
                        .strong()
                        .color(TEXT_PRIMARY),
                );
                ui.add_space(8.0);
                for s in &series {
                    legend_chip(ui, s.name, s.color);
                }
            });
            ui.add_space(6.0);
            let config = LineChartConfig {
                height: 240.0,
                fixed_y: None,
                min_y_pad: 25.0,
                x_label: "Update",
                value_fmt: ValueFormat::Elo,
                empty_msg: "Waiting for Elo history…",
            };
            draw_line_chart(ui, &series, &refs, &config);
        });
    }

    fn draw_eval_chart_card(&self, ui: &mut egui::Ui) {
        let mut rnd_pts = Vec::new();
        let mut heur_pts = Vec::new();
        let mut lge_pts = Vec::new();

        for r in &self.telemetry.metrics {
            let x = r.update as f64;
            if let Some(v) = r.eval_random_win_rate {
                rnd_pts.push((x, v * 100.0));
            }
            if let Some(v) = r.eval_heuristic_win_rate {
                heur_pts.push((x, v * 100.0));
            }
            if let Some(v) = r.eval_league_win_rate {
                lge_pts.push((x, v * 100.0));
            }
        }

        let few_points = rnd_pts.len() <= 20 && heur_pts.len() <= 20 && lge_pts.len() <= 20;
        let series = vec![
            ChartSeries {
                name: "vs Random",
                color: GREEN,
                stroke_width: 2.2,
                points: rnd_pts,
                show_dots: few_points,
            },
            ChartSeries {
                name: "vs Heuristic",
                color: CORAL,
                stroke_width: 2.2,
                points: heur_pts,
                show_dots: few_points,
            },
            ChartSeries {
                name: "vs League",
                color: CYAN,
                stroke_width: 2.2,
                points: lge_pts,
                show_dots: few_points,
            },
        ];

        let refs = vec![RefLine {
            value: 50.0,
            label: "50%".to_owned(),
            color: TEXT_MUTED.gamma_multiply(0.65),
        }];

        card_frame(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new("Eval Win Rates")
                        .size(15.0)
                        .strong()
                        .color(TEXT_PRIMARY),
                );
                ui.add_space(8.0);
                for s in &series {
                    legend_chip(ui, s.name, s.color);
                }
            });
            ui.add_space(6.0);
            let config = LineChartConfig {
                height: 240.0,
                fixed_y: Some((0.0, 100.0)),
                min_y_pad: 5.0,
                x_label: "Update",
                value_fmt: ValueFormat::Percent,
                empty_msg: "Waiting for evaluation checkpoints…",
            };
            draw_line_chart(ui, &series, &refs, &config);
        });
    }

    fn draw_ppo_diagnostics(&self, ui: &mut egui::Ui) {
        let entropy: Vec<(f64, f64)> = self
            .telemetry
            .metrics
            .iter()
            .map(|m| (m.update as f64, m.entropy))
            .collect();
        let value_loss: Vec<(f64, f64)> = self
            .telemetry
            .metrics
            .iter()
            .map(|m| (m.update as f64, m.value_loss))
            .collect();
        let approx_kl: Vec<(f64, f64)> = self
            .telemetry
            .metrics
            .iter()
            .map(|m| (m.update as f64, m.approx_kl))
            .collect();
        let policy_loss: Vec<(f64, f64)> = self
            .telemetry
            .metrics
            .iter()
            .map(|m| (m.update as f64, m.policy_loss))
            .collect();

        let latest = self.telemetry.metrics.last();

        ui.label(
            RichText::new("PPO Diagnostics")
                .size(14.0)
                .strong()
                .color(TEXT_MUTED),
        );
        ui.add_space(4.0);

        let cols = if ui.available_width() >= 980.0 { 4 } else { 2 };
        ui.columns(cols, |columns| {
            draw_mini_metric_card(
                &mut columns[0],
                "Entropy",
                latest.map(|m| format!("{:.3}", m.entropy)),
                CYAN,
                entropy,
                ValueFormat::Decimal3,
            );
            draw_mini_metric_card(
                &mut columns[1],
                "Value Loss",
                latest.map(|m| format!("{:.4}", m.value_loss)),
                GOLD,
                value_loss,
                ValueFormat::Decimal4,
            );
            if cols == 4 {
                draw_mini_metric_card(
                    &mut columns[2],
                    "Approx KL",
                    latest.map(|m| format!("{:.4}", m.approx_kl)),
                    CORAL,
                    approx_kl,
                    ValueFormat::Decimal4,
                );
                draw_mini_metric_card(
                    &mut columns[3],
                    "Policy Loss",
                    latest.map(|m| format!("{:.4}", m.policy_loss)),
                    GREEN,
                    policy_loss,
                    ValueFormat::Decimal4,
                );
            } else {
                columns[0].add_space(8.0);
                columns[1].add_space(8.0);
                draw_mini_metric_card(
                    &mut columns[0],
                    "Approx KL",
                    latest.map(|m| format!("{:.4}", m.approx_kl)),
                    CORAL,
                    approx_kl,
                    ValueFormat::Decimal4,
                );
                draw_mini_metric_card(
                    &mut columns[1],
                    "Policy Loss",
                    latest.map(|m| format!("{:.4}", m.policy_loss)),
                    GREEN,
                    policy_loss,
                    ValueFormat::Decimal4,
                );
            }
        });
    }

    fn draw_league_panel(&self, ui: &mut egui::Ui) {
        card_frame(ui, |ui| {
            let league = self.telemetry.league.as_ref();
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new("League & PFSP Opponent Pool")
                        .size(15.0)
                        .strong()
                        .color(TEXT_PRIMARY),
                );
                if let Some(league) = league {
                    let active_count = league.members.iter().filter(|m| m.active).count();
                    badge(
                        ui,
                        &format!("{active_count}/{} active", league.members.len()),
                        CYAN,
                    );
                    if !league.pfsp_mode.is_empty() {
                        badge(
                            ui,
                            &format!("mode: {} · p={:.1}", league.pfsp_mode, league.pfsp_p),
                            GOLD,
                        );
                    }
                    ui.label(
                        RichText::new(format!(
                            "Anchors: Random {:.0} · Heuristic {:.0}",
                            league.anchors.random_elo, league.anchors.heuristic_elo
                        ))
                        .size(12.0)
                        .color(TEXT_MUTED),
                    );
                }
            });
            ui.add_space(8.0);

            let Some(league) = league else {
                ui.label(
                    RichText::new("Waiting for league.json…")
                        .color(TEXT_MUTED)
                        .italics(),
                );
                return;
            };

            if league.members.is_empty() {
                ui.label(
                    RichText::new("No league members recorded yet.")
                        .color(TEXT_MUTED)
                        .italics(),
                );
                return;
            }

            let mut sorted: Vec<&LeagueMember> = league.members.iter().collect();
            sorted.sort_by(|a, b| b.elo.total_cmp(&a.elo).then_with(|| a.id.cmp(&b.id)));

            let max_weight = sorted
                .iter()
                .map(|m| m.pfsp_weight)
                .fold(0.0_f64, f64::max)
                .max(0.001);

            egui::ScrollArea::vertical()
                .id_salt("league_members_scroll")
                .max_height(260.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    egui::Grid::new("league_members_grid")
                        .num_columns(7)
                        .striped(true)
                        .spacing([14.0, 6.0])
                        .min_col_width(52.0)
                        .show(ui, |ui| {
                            for header in [
                                "Member",
                                "Kind",
                                "Elo",
                                "W-L-D (Games)",
                                "Learner Win %",
                                "PFSP Weight",
                                "Origin",
                            ] {
                                ui.label(
                                    RichText::new(header).size(11.5).strong().color(TEXT_MUTED),
                                );
                            }
                            ui.end_row();

                            for member in sorted {
                                let name_color = if member.active {
                                    TEXT_PRIMARY
                                } else {
                                    TEXT_MUTED
                                };
                                ui.label(
                                    RichText::new(&member.id)
                                        .monospace()
                                        .strong()
                                        .color(name_color),
                                );

                                let kind_color = match member.kind.as_str() {
                                    "random" => CORAL,
                                    "heuristic" => GOLD,
                                    "snapshot" | "policy" | "self" => CYAN,
                                    _ => GREEN,
                                };
                                badge(ui, &member.kind, kind_color);

                                ui.label(
                                    RichText::new(format!("{:.1}", member.elo))
                                        .monospace()
                                        .strong()
                                        .color(if member.active { GOLD } else { TEXT_MUTED }),
                                );

                                ui.label(
                                    RichText::new(format!(
                                        "{}-{}-{} ({})",
                                        member.wins, member.losses, member.draws, member.games
                                    ))
                                    .monospace()
                                    .color(TEXT_PRIMARY),
                                );

                                let win_pct = member.learner_win_rate * 100.0;
                                let win_color = if member.games == 0 {
                                    TEXT_MUTED
                                } else if win_pct >= 60.0 {
                                    GREEN
                                } else if win_pct >= 40.0 {
                                    CYAN
                                } else {
                                    CORAL
                                };
                                let win_label = if member.games == 0 {
                                    "—".to_owned()
                                } else {
                                    format!("{win_pct:5.1}%")
                                };
                                ui.label(
                                    RichText::new(win_label)
                                        .monospace()
                                        .strong()
                                        .color(win_color),
                                );

                                draw_pfsp_bar(ui, member.pfsp_weight, max_weight, member.active);

                                let origin = if member.active {
                                    format!("u{}", member.created_update)
                                } else {
                                    format!("u{} (inactive)", member.created_update)
                                };
                                ui.label(RichText::new(origin).size(11.5).color(TEXT_MUTED));

                                ui.end_row();
                            }
                        });
                });
        });
    }
}

impl eframe::App for DashboardApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if !self.paused {
            if self.last_refresh.elapsed() >= REFRESH_INTERVAL {
                self.refresh();
            }
            let remaining = REFRESH_INTERVAL
                .saturating_sub(self.last_refresh.elapsed())
                .max(Duration::from_millis(100));
            ui.ctx().request_repaint_after(remaining);
        }

        let fill = ui.visuals().panel_fill;
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(fill).inner_margin(16))
            .show(ui, |ui| {
                self.draw_header(ui);
                ui.add_space(10.0);

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        self.draw_kpi_row(ui);
                        ui.add_space(12.0);
                        self.draw_primary_charts(ui);
                        ui.add_space(12.0);
                        self.draw_ppo_diagnostics(ui);
                        ui.add_space(12.0);
                        self.draw_league_panel(ui);
                        ui.add_space(8.0);
                    });
            });
    }
}

fn card_frame(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(CARD_BG)
        .stroke(Stroke::new(1.0, BORDER_SUBTLE))
        .corner_radius(10)
        .inner_margin(12)
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            add_contents(ui);
        });
}

fn badge(ui: &mut egui::Ui, text: &str, color: Color32) {
    let bg = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 32);
    let border = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 110);
    egui::Frame::new()
        .fill(bg)
        .stroke(Stroke::new(1.0, border))
        .corner_radius(5)
        .inner_margin(egui::Margin::symmetric(7, 2))
        .show(ui, |ui| {
            ui.label(RichText::new(text).size(11.5).strong().color(color));
        });
}

fn legend_chip(ui: &mut egui::Ui, label: &str, color: Color32) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(vec2(10.0, 10.0), Sense::hover());
        ui.painter().circle_filled(rect.center(), 4.0, color);
        ui.label(RichText::new(label).size(12.0).color(TEXT_MUTED));
    });
}

fn eval_stat(ui: &mut egui::Ui, label: &str, rate: Option<f64>, color: Color32) {
    let value = rate
        .map(|v| format!("{:.0}%", v * 100.0))
        .unwrap_or_else(|| "—".to_owned());
    ui.vertical(|ui| {
        ui.label(RichText::new(label).size(10.5).color(TEXT_MUTED));
        ui.label(RichText::new(value).size(18.0).strong().color(color));
    });
}

fn draw_progress_bar(ui: &mut egui::Ui, frac: f32, color: Color32) {
    let width = ui.available_width().max(40.0);
    let (rect, _) = ui.allocate_exact_size(vec2(width, 8.0), Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 4.0, PLOT_BG);
    if frac > 0.0 {
        let fill_rect = Rect::from_min_size(rect.min, vec2(rect.width() * frac, rect.height()));
        painter.rect_filled(fill_rect, 4.0, color);
    }
    painter.rect_stroke(
        rect,
        4.0,
        Stroke::new(1.0, BORDER_SUBTLE),
        StrokeKind::Inside,
    );
}

fn draw_pfsp_bar(ui: &mut egui::Ui, weight: f64, max_weight: f64, active: bool) {
    ui.horizontal(|ui| {
        let bar_size = vec2(128.0, 14.0);
        let (rect, response) = ui.allocate_exact_size(bar_size, Sense::hover());
        let painter = ui.painter();
        painter.rect_filled(rect, 4.0, PLOT_BG);

        let clamped = weight.clamp(0.0, 1.0) as f32;
        if clamped > 0.0 {
            let fill_color = if !active {
                TEXT_MUTED.gamma_multiply(0.5)
            } else if weight >= max_weight * 0.95 {
                GOLD
            } else {
                CYAN
            };
            let fill_rect = Rect::from_min_size(
                rect.min,
                vec2((rect.width() * clamped).max(2.0), rect.height()),
            );
            painter.rect_filled(fill_rect, 4.0, fill_color);
        }
        painter.rect_stroke(
            rect,
            4.0,
            Stroke::new(1.0, BORDER_SUBTLE),
            StrokeKind::Inside,
        );

        let pct = weight * 100.0;
        ui.label(
            RichText::new(format!("{pct:5.1}%"))
                .monospace()
                .size(12.0)
                .color(if active { TEXT_PRIMARY } else { TEXT_MUTED }),
        );
        response.on_hover_text(format!("Sampling probability: {pct:.2}%"));
    });
}

fn draw_mini_metric_card(
    ui: &mut egui::Ui,
    title: &str,
    current_value: Option<String>,
    color: Color32,
    points: Vec<(f64, f64)>,
    value_fmt: ValueFormat,
) {
    card_frame(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(title).size(13.0).strong().color(TEXT_PRIMARY));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    RichText::new(current_value.unwrap_or_else(|| "—".to_owned()))
                        .monospace()
                        .strong()
                        .size(13.0)
                        .color(color),
                );
            });
        });
        ui.add_space(4.0);
        let series = vec![ChartSeries {
            name: title,
            color,
            stroke_width: 1.9,
            points,
            show_dots: false,
        }];
        let config = LineChartConfig {
            height: 120.0,
            fixed_y: None,
            min_y_pad: 1e-4,
            x_label: "Update",
            value_fmt,
            empty_msg: "No data yet",
        };
        draw_line_chart(ui, &series, &[], &config);
    });
}

struct ChartSeries<'a> {
    name: &'a str,
    color: Color32,
    stroke_width: f32,
    points: Vec<(f64, f64)>,
    show_dots: bool,
}

struct RefLine {
    value: f64,
    label: String,
    color: Color32,
}

#[derive(Clone, Copy)]
enum ValueFormat {
    Elo,
    Percent,
    Decimal3,
    Decimal4,
}

impl ValueFormat {
    fn format(self, v: f64) -> String {
        match self {
            Self::Elo => format!("{v:.1}"),
            Self::Percent => format!("{v:.1}%"),
            Self::Decimal3 => format!("{v:.3}"),
            Self::Decimal4 => format!("{v:.4}"),
        }
    }

    fn format_axis(self, v: f64) -> String {
        match self {
            Self::Elo => format!("{v:.0}"),
            Self::Percent => format!("{v:.0}%"),
            Self::Decimal3 => format!("{v:.2}"),
            Self::Decimal4 => {
                if v.abs() >= 0.01 || v == 0.0 {
                    format!("{v:.3}")
                } else {
                    format!("{v:.1e}")
                }
            }
        }
    }
}

struct LineChartConfig<'a> {
    height: f32,
    fixed_y: Option<(f64, f64)>,
    min_y_pad: f64,
    x_label: &'a str,
    value_fmt: ValueFormat,
    empty_msg: &'a str,
}

fn draw_line_chart(
    ui: &mut egui::Ui,
    series: &[ChartSeries<'_>],
    ref_lines: &[RefLine],
    cfg: &LineChartConfig<'_>,
) {
    let width = ui.available_width().max(120.0);
    let (response, painter) = ui.allocate_painter(vec2(width, cfg.height), Sense::hover());
    let outer = response.rect;

    painter.rect_filled(outer, 6.0, PLOT_BG);
    painter.rect_stroke(
        outer,
        6.0,
        Stroke::new(1.0, BORDER_SUBTLE),
        StrokeKind::Inside,
    );

    let left_pad = 48.0;
    let right_pad = 12.0;
    let top_pad = 10.0;
    let bottom_pad = 22.0;
    let plot = Rect::from_min_max(
        pos2(outer.left() + left_pad, outer.top() + top_pad),
        pos2(outer.right() - right_pad, outer.bottom() - bottom_pad),
    );
    if plot.width() <= 10.0 || plot.height() <= 10.0 {
        return;
    }

    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    let mut any_point = false;

    for s in series {
        for &(x, y) in &s.points {
            if x.is_finite() && y.is_finite() {
                any_point = true;
                x_min = x_min.min(x);
                x_max = x_max.max(x);
                y_min = y_min.min(y);
                y_max = y_max.max(y);
            }
        }
    }

    if !any_point {
        painter.text(
            outer.center(),
            Align2::CENTER_CENTER,
            cfg.empty_msg,
            FontId::proportional(12.5),
            TEXT_MUTED,
        );
        return;
    }

    if let Some((fy_min, fy_max)) = cfg.fixed_y {
        y_min = fy_min;
        y_max = fy_max;
    } else {
        for r in ref_lines {
            if r.value.is_finite() {
                y_min = y_min.min(r.value);
                y_max = y_max.max(r.value);
            }
        }
        let span = y_max - y_min;
        let pad = if span <= 1e-9 {
            (y_max.abs() * 0.1).max(cfg.min_y_pad)
        } else {
            (span * 0.12).max(cfg.min_y_pad * 0.25)
        };
        y_min -= pad;
        y_max += pad;
    }

    if (x_max - x_min).abs() < 1e-9 {
        x_min -= 1.0;
        x_max += 1.0;
    }

    let to_screen = |x: f64, y: f64| -> Pos2 {
        let tx = ((x - x_min) / (x_max - x_min)).clamp(0.0, 1.0) as f32;
        let ty = ((y - y_min) / (y_max - y_min)).clamp(0.0, 1.0) as f32;
        pos2(
            plot.left() + tx * plot.width(),
            plot.bottom() - ty * plot.height(),
        )
    };

    // Horizontal gridlines & Y-axis labels
    let y_ticks = if cfg.height >= 200.0 { 4 } else { 2 };
    for i in 0..=y_ticks {
        let frac = i as f64 / y_ticks as f64;
        let val = y_min + frac * (y_max - y_min);
        let y = plot.bottom() - (frac as f32) * plot.height();
        painter.line_segment(
            [pos2(plot.left(), y), pos2(plot.right(), y)],
            Stroke::new(1.0, GRID_LINE),
        );
        painter.text(
            pos2(plot.left() - 6.0, y),
            Align2::RIGHT_CENTER,
            cfg.value_fmt.format_axis(val),
            FontId::monospace(10.0),
            TEXT_MUTED,
        );
    }

    // X-axis ticks (start, mid, end)
    let x_ticks = if plot.width() >= 260.0 { 4 } else { 2 };
    for i in 0..=x_ticks {
        let frac = i as f64 / x_ticks as f64;
        let val = x_min + frac * (x_max - x_min);
        let x = plot.left() + (frac as f32) * plot.width();
        painter.line_segment(
            [pos2(x, plot.top()), pos2(x, plot.bottom())],
            Stroke::new(1.0, Color32::from_rgba_unmultiplied(33, 49, 64, 130)),
        );
        let align = if i == 0 {
            Align2::LEFT_TOP
        } else if i == x_ticks {
            Align2::RIGHT_TOP
        } else {
            Align2::CENTER_TOP
        };
        painter.text(
            pos2(x, plot.bottom() + 4.0),
            align,
            format!("{val:.0}"),
            FontId::monospace(10.0),
            TEXT_MUTED,
        );
    }

    // Horizontal dashed reference lines
    for ref_line in ref_lines {
        if ref_line.value >= y_min && ref_line.value <= y_max {
            let y = to_screen(x_min, ref_line.value).y;
            let pts = [pos2(plot.left(), y), pos2(plot.right(), y)];
            painter.extend(Shape::dashed_line(
                &pts,
                Stroke::new(1.2, ref_line.color),
                5.0,
                4.0,
            ));
            painter.text(
                pos2(plot.right() - 4.0, y - 3.0),
                Align2::RIGHT_BOTTOM,
                &ref_line.label,
                FontId::monospace(10.0),
                ref_line.color,
            );
        }
    }

    // Data polylines
    for s in series {
        let screen_pts: Vec<Pos2> = s
            .points
            .iter()
            .filter(|&&(x, y)| x.is_finite() && y.is_finite())
            .map(|&(x, y)| to_screen(x, y))
            .collect();
        if screen_pts.len() >= 2 {
            painter.add(Shape::line(
                screen_pts.clone(),
                Stroke::new(s.stroke_width, s.color),
            ));
        }
        if s.show_dots || screen_pts.len() == 1 {
            let radius = if screen_pts.len() == 1 { 4.0 } else { 2.6 };
            for pt in screen_pts {
                painter.circle_filled(pt, radius, s.color);
            }
        }
    }

    // Interactive hover crosshair & readout
    if let Some(pointer) = response.hover_pos()
        && plot.contains(pointer)
    {
        let frac_x = ((pointer.x - plot.left()) / plot.width()).clamp(0.0, 1.0) as f64;
        let hover_x = x_min + frac_x * (x_max - x_min);

        let mut nearest_hits: Vec<(&str, Color32, f64, f64, Pos2)> = Vec::new();
        let mut snapped_x: Option<f64> = None;

        for s in series {
            if let Some(&(px, py)) = s
                .points
                .iter()
                .filter(|&&(x, y)| x.is_finite() && y.is_finite())
                .min_by(|a, b| (a.0 - hover_x).abs().total_cmp(&(b.0 - hover_x).abs()))
            {
                let pos = to_screen(px, py);
                if snapped_x.is_none() {
                    snapped_x = Some(px);
                }
                nearest_hits.push((s.name, s.color, px, py, pos));
            }
        }

        if let Some(sx) = snapped_x {
            let cross_x = to_screen(sx, y_min).x;
            painter.line_segment(
                [pos2(cross_x, plot.top()), pos2(cross_x, plot.bottom())],
                Stroke::new(1.0, Color32::from_rgba_unmultiplied(232, 240, 247, 110)),
            );

            for &(_, color, _, _, pos) in &nearest_hits {
                painter.circle_filled(pos, 4.2, color);
                painter.circle_stroke(pos, 4.2, Stroke::new(1.4, PLOT_BG));
            }

            // Build tooltip card inside the plot bounds
            let header = format!("{} {:.0}", cfg.x_label, sx);
            let lines: Vec<(String, Color32)> = nearest_hits
                .iter()
                .map(|&(name, color, _, py, _)| {
                    (format!("{name}: {}", cfg.value_fmt.format(py)), color)
                })
                .collect();

            let max_chars = lines
                .iter()
                .map(|(text, _)| text.len())
                .max()
                .unwrap_or(0)
                .max(header.len());
            let box_w = (max_chars as f32 * 6.8 + 18.0).max(96.0);
            let line_h = 15.0;
            let box_h = 12.0 + line_h * (1 + lines.len()) as f32;

            let place_right = cross_x + 12.0 + box_w <= plot.right();
            let box_x = if place_right {
                cross_x + 10.0
            } else {
                (cross_x - 10.0 - box_w).max(plot.left() + 2.0)
            };
            let box_y = (pointer.y - box_h * 0.5).clamp(
                plot.top() + 2.0,
                (plot.bottom() - box_h - 2.0).max(plot.top() + 2.0),
            );
            let tip_rect = Rect::from_min_size(pos2(box_x, box_y), Vec2::new(box_w, box_h));

            painter.rect_filled(
                tip_rect,
                6.0,
                Color32::from_rgba_unmultiplied(15, 23, 32, 235),
            );
            painter.rect_stroke(
                tip_rect,
                6.0,
                Stroke::new(1.0, BORDER_SUBTLE),
                StrokeKind::Inside,
            );

            let mut cursor_y = tip_rect.top() + 6.0;
            painter.text(
                pos2(tip_rect.left() + 8.0, cursor_y),
                Align2::LEFT_TOP,
                header,
                FontId::monospace(11.0),
                TEXT_PRIMARY,
            );
            cursor_y += line_h;
            for (text, color) in lines {
                painter.text(
                    pos2(tip_rect.left() + 8.0, cursor_y),
                    Align2::LEFT_TOP,
                    text,
                    FontId::monospace(11.0),
                    color,
                );
                cursor_y += line_h;
            }
        }
    }
}

fn format_compact_u64(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_duration(secs: f64) -> String {
    if !secs.is_finite() || secs < 0.0 {
        return "—".to_owned();
    }
    let total = secs.round() as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}h {m:02}m {s:02}s")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{secs:.1}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn make_temp_dir(label: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "riskbot_dashboard_test_{label}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp test dir");
        dir
    }

    #[test]
    fn parses_metrics_jsonl_skipping_blank_and_partial_trailing_line() {
        let dir = make_temp_dir("metrics");
        let path = dir.join("metrics.jsonl");
        let sample = r#"
{"update":1,"total_env_steps":2397,"elapsed_secs":4.5,"steps_per_sec":532.6,"learning_rate":0.0003,"policy_loss":-0.012,"value_loss":0.001,"entropy":2.81,"approx_kl":0.004,"clip_frac":0.05,"episodes_finished":2,"win_frac":0.5,"avg_turns":51.0,"learner_elo":1012.5,"pool_size":2}

{"update":10,"total_env_steps":22286,"elapsed_secs":41.2,"steps_per_sec":585.4,"learning_rate":0.000273,"policy_loss":-0.0080,"value_loss":0.0002,"entropy":2.530,"approx_kl":0.0051,"clip_frac":0.06,"episodes_finished":4,"win_frac":1.0,"avg_turns":42.5,"learner_elo":1142.3,"pool_size":3,"eval_random_win_rate":0.917,"eval_heuristic_win_rate":0.417,"eval_league_win_rate":0.625,"eval_random_avg_turns":26.0,"eval_heuristic_avg_turns":20.0}
{"update":11,"total_env_steps":24500,"elapsed_secs":
"#;
        fs::write(&path, sample).expect("write metrics.jsonl");

        let rows = load_metrics_jsonl(&path).expect("load metrics.jsonl");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].update, 1);
        assert_eq!(rows[0].total_env_steps, 2397);
        assert_eq!(rows[0].eval_random_win_rate, None);
        assert_eq!(rows[1].update, 10);
        assert!((rows[1].learner_elo.unwrap() - 1142.3).abs() < 1e-6);
        assert!((rows[1].eval_random_win_rate.unwrap() - 0.917).abs() < 1e-6);
        assert!((rows[1].eval_heuristic_win_rate.unwrap() - 0.417).abs() < 1e-6);
        assert!((rows[1].eval_league_win_rate.unwrap() - 0.625).abs() < 1e-6);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parses_league_json_and_keeps_last_valid_snapshot_on_corruption() {
        let dir = make_temp_dir("league");
        let league_path = dir.join("league.json");
        let sample = r#"{
  "run_name": "ppo",
  "players": 2,
  "max_turns": 200,
  "backend": "WGPU/Vulkan GPU",
  "pfsp_mode": "hard",
  "pfsp_p": 2.0,
  "current_update": 10,
  "total_updates": 200,
  "learner_elo": 1142.3,
  "anchors": { "random_elo": 800.0, "heuristic_elo": 1200.0 },
  "members": [
    {
      "id": "random",
      "kind": "random",
      "created_update": 0,
      "elo": 800.0,
      "games": 32,
      "wins": 3,
      "losses": 29,
      "draws": 0,
      "learner_win_rate": 0.906,
      "pfsp_weight": 0.015,
      "active": true
    },
    {
      "id": "heuristic",
      "kind": "heuristic",
      "created_update": 0,
      "elo": 1200.0,
      "games": 32,
      "wins": 18,
      "losses": 14,
      "draws": 0,
      "learner_win_rate": 0.438,
      "pfsp_weight": 0.62,
      "active": true
    }
  ],
  "elo_history": [
    { "update": 1, "steps": 2397, "learner_elo": 1000.0, "best_snapshot_elo": 1000.0, "mean_snapshot_elo": 1000.0 },
    { "update": 10, "steps": 22286, "learner_elo": 1142.3, "best_snapshot_elo": 1110.0, "mean_snapshot_elo": 1055.0 }
  ]
}"#;
        fs::write(&league_path, sample).expect("write league.json");

        let mut telemetry = RunTelemetry::default();
        telemetry.reload_from(&dir);

        let snap = telemetry.league.as_ref().expect("valid league snapshot");
        assert_eq!(snap.run_name, "ppo");
        assert_eq!(snap.backend, "WGPU/Vulkan GPU");
        assert_eq!(snap.pfsp_mode, "hard");
        assert_eq!(snap.current_update, 10);
        assert_eq!(snap.total_updates, 200);
        assert_eq!(snap.members.len(), 2);
        assert_eq!(snap.elo_history.len(), 2);
        assert!(!telemetry.league_stale);

        // Simulate a partial mid-write file and verify last valid snapshot is preserved
        fs::write(&league_path, "{ \"run_name\": \"ppo\", \"members\": [").expect("write partial");
        telemetry.reload_from(&dir);
        let preserved = telemetry
            .league
            .as_ref()
            .expect("last valid snapshot preserved");
        assert_eq!(preserved.current_update, 10);
        assert_eq!(preserved.members.len(), 2);
        assert!(telemetry.league_stale);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn discovers_single_run_and_parent_runs_directory() {
        let root = make_temp_dir("discover");
        let run_a = root.join("alpha");
        let run_b = root.join("beta");
        let empty = root.join("empty-folder");
        fs::create_dir_all(&run_a).unwrap();
        fs::create_dir_all(&run_b).unwrap();
        fs::create_dir_all(&empty).unwrap();

        fs::write(run_a.join("metrics.jsonl"), "{}\n").unwrap();
        fs::write(run_b.join("league.json"), "{}").unwrap();

        // Watching parent directory finds both alpha and beta, ignores empty folder
        let discovered = discover_runs(&root);
        assert_eq!(discovered, vec![run_a.clone(), run_b.clone()]);

        // Watching a specific run directory returns just that run
        let single = discover_runs(&run_a);
        assert_eq!(single, vec![run_a]);

        let _ = fs::remove_dir_all(&root);
    }
}
