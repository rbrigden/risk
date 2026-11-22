//! Board viewer for random simulator games.

use crate::game::{ACTION_COUNT, Action, BattleOutcome, EDGES, Game, N, Phase};
use eframe::egui::{self, Color32, RichText};
use rand::{SeedableRng, seq::IteratorRandom};
use rand_chacha::ChaCha8Rng;

mod battles;
mod map;

const PLAYERS: usize = 3;
const MAX_TURNS: u32 = 500;
const COLORS: [Color32; PLAYERS] = [
    Color32::from_rgb(90, 199, 239),
    Color32::from_rgb(255, 150, 115),
    Color32::from_rgb(183, 222, 116),
];

/// Launch the visualizer in a native window.
#[cfg(not(target_arch = "wasm32"))]
pub fn run() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([860.0, 720.0])
            .with_min_inner_size([520.0, 510.0]),
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    eframe::run_native(
        "Riskbot",
        options,
        Box::new(|cc| Ok(Box::new(Visualizer::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start(canvas: web_sys::HtmlCanvasElement) -> Result<(), wasm_bindgen::JsValue> {
    eframe::WebRunner::new()
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(|cc| Ok(Box::new(Visualizer::new(cc)))),
        )
        .await
}

struct Visualizer {
    game: Game,
    policy_rng: ChaCha8Rng,
    changed: [bool; N],
    map: map::Map,
    last_turn: Option<TurnBattles>,
    game_number: u64,
}

impl Visualizer {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = Color32::from_rgb(19, 28, 38);
        visuals.widgets.inactive.bg_fill = Color32::from_rgb(39, 57, 73);
        visuals.widgets.inactive.weak_bg_fill = visuals.widgets.inactive.bg_fill;
        visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(54, 78, 96);
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        cc.egui_ctx.set_visuals(visuals);
        Self {
            game: Game::new(PLAYERS, MAX_TURNS, None),
            policy_rng: ChaCha8Rng::from_os_rng(),
            changed: [false; N],
            map: map::Map::new(),
            last_turn: None,
            game_number: 0,
        }
    }

    fn advance_turn(&mut self) {
        let (owners, troops) = (self.game.owner, self.game.troops);
        self.last_turn = Some(play_turn(&mut self.game, &mut self.policy_rng));
        self.changed = std::array::from_fn(|i| {
            owners[i] != self.game.owner[i] || troops[i] != self.game.troops[i]
        });
    }
}
struct Battle {
    source: usize,
    target: usize,
    defender: usize,
    outcome: BattleOutcome,
}

struct TurnBattles {
    number: u32,
    attacker: usize,
    battles: Vec<Battle>,
}

fn play_turn(game: &mut Game, rng: &mut ChaCha8Rng) -> TurnBattles {
    let starting_turns = game.turns;
    let mut turn = TurnBattles {
        number: starting_turns + 1,
        attacker: game.active,
        battles: Vec::new(),
    };
    while game.phase != Phase::Ended && game.turns == starting_turns {
        let action = (0..ACTION_COUNT)
            .filter_map(Action::decode)
            .filter(|&action| game.legal(action))
            .choose(rng)
            .expect("a live game has a legal action");
        if let Action::Attack(source, target, _) = action {
            let defender = game.owner[target];
            let outcome = game.step(action).expect("an attack produces an outcome");
            turn.battles.push(Battle {
                source,
                target,
                defender,
                outcome,
            });
        } else {
            game.step(action);
        }
    }
    turn
}

impl eframe::App for Visualizer {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let fill = ui.visuals().panel_fill;
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(fill).inner_margin(16))
            .show(ui, |ui| {
                ui.label(RichText::new("Riskbot").size(32.0).strong());
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            self.game.phase != Phase::Ended,
                            egui::Button::new(RichText::new("Next Turn").strong())
                                .min_size(egui::vec2(104.0, 30.0)),
                        )
                        .clicked()
                    {
                        self.advance_turn();
                    }
                    if ui
                        .add(egui::Button::new("New Game").min_size(egui::vec2(90.0, 30.0)))
                        .clicked()
                    {
                        self.game = Game::new(PLAYERS, MAX_TURNS, None);
                        self.policy_rng = ChaCha8Rng::from_os_rng();
                        self.changed.fill(false);
                        self.last_turn = None;
                        self.game_number += 1;
                    }
                });
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(format!("Completed turns: {}", self.game.turns));
                    ui.separator();
                    match (self.game.phase, self.game.winner) {
                        (Phase::Ended, Some(winner)) => {
                            ui.label(format!("Player {} wins", winner + 1))
                        }
                        (Phase::Ended, None) => ui.label("Draw at the turn limit"),
                        _ => ui.label(format!("Player {} to play", self.game.active + 1)),
                    };
                });
                ui.horizontal_wrapped(|ui| {
                    for (owner, color) in COLORS.into_iter().enumerate() {
                        let owned = self.game.owner.iter().filter(|&&p| p == owner).count();
                        ui.colored_label(color, format!("P{} · {} territories", owner + 1, owned));
                    }
                });
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label("Matching letters mark sea crossings");
                    ui.label(RichText::new("Gold outlines mark changes from the last turn").weak());
                });
                ui.add_space(6.0);
                ui.push_id(self.game_number, |ui| {
                    battles::draw(ui, self.last_turn.as_ref())
                });
                ui.add_space(8.0);
                egui::ScrollArea::both()
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.map.draw(self, ui));
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advances_one_completed_turn() {
        let mut game = Game::new(PLAYERS, MAX_TURNS, Some(4));
        let active = game.active;
        play_turn(&mut game, &mut ChaCha8Rng::seed_from_u64(5));
        assert_eq!(game.turns, 1);
        assert_eq!(game.phase, Phase::Reinforce);
        assert_ne!(game.active, active);
    }

    #[test]
    fn records_turn_without_attacks() {
        let mut game = Game::new(PLAYERS, MAX_TURNS, Some(4));
        game.phase = Phase::Attack;
        let actor = game.active;
        let turn = play_turn(&mut game, &mut ChaCha8Rng::seed_from_u64(5));
        assert_eq!((turn.number, turn.attacker), (1, actor));
        assert!(turn.battles.is_empty());
        assert_eq!(game.turns, 1);
    }

    #[test]
    fn stops_at_the_turn_limit() {
        let mut game = Game::new(PLAYERS, 1, Some(4));
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        play_turn(&mut game, &mut rng);
        assert_eq!(game.phase, Phase::Ended);
        assert_eq!(game.winner, None);
        let troops = game.troops;
        play_turn(&mut game, &mut rng);
        assert_eq!(game.turns, 1);
        assert_eq!(game.troops, troops);
    }

    #[test]
    fn stops_immediately_on_victory() {
        let won = (0..32).any(|seed| {
            let mut attempt = Game::new(2, MAX_TURNS, Some(seed));
            attempt.owner.fill(0);
            attempt.owner[0] = 1;
            attempt.troops.fill(100);
            attempt.troops[0] = 1;
            attempt.active = 0;
            attempt.phase = Phase::Attack;
            attempt.pending = 0;
            let turn = play_turn(&mut attempt, &mut ChaCha8Rng::seed_from_u64(seed));
            if attempt.winner == Some(0) {
                assert_eq!(turn.number, 1);
                assert_eq!(turn.attacker, 0);
                let (last, preceding) = turn.battles.split_last().unwrap();
                assert_eq!((last.target, last.defender), (0, 1));
                assert!(last.outcome.captured);
                assert!(preceding.iter().all(|battle| !battle.outcome.captured));
                assert_eq!(attempt.turns, 0);
                assert_eq!(attempt.phase, Phase::Ended);
                return true;
            }
            false
        });
        assert!(
            won,
            "at least one seeded game should win before ending the turn"
        );
    }
}
