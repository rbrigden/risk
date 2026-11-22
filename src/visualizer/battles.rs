use eframe::egui::{self, Color32, RichText, Stroke, StrokeKind, vec2};

use super::{Battle, COLORS, TurnBattles};

pub(super) fn draw(ui: &mut egui::Ui, turn: Option<&TurnBattles>) {
    let Some(turn) = turn else {
        ui.label(RichText::new("Play a turn to see its battles.").weak());
        return;
    };
    let count = turn.battles.len();
    let noun = if count == 1 { "battle" } else { "battles" };
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(format!("Turn {}", turn.number)).strong());
        ui.colored_label(
            COLORS[turn.attacker],
            format!("Player {}", turn.attacker + 1),
        );
        ui.label(RichText::new(format!("{count} {noun}")).weak());
    });
    if count == 0 {
        ui.label("No battles this turn.");
        return;
    }
    let height = (ui.available_height() - 365.0).clamp(112.0, 222.0);
    egui::ScrollArea::vertical()
        .id_salt(("turn battles", turn.number))
        .max_height(height)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            for (index, battle) in turn.battles.iter().enumerate() {
                card(ui, index + 1, turn.attacker, battle);
            }
        });
}

fn card(ui: &mut egui::Ui, number: usize, attacker: usize, battle: &Battle) {
    egui::Frame::new()
        .fill(Color32::from_rgb(29, 44, 58))
        .corner_radius(8)
        .inner_margin(8)
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new(format!(
                        "Battle {number} · Territory {} → {}",
                        battle.source, battle.target
                    ))
                    .strong(),
                );
                let (status, color) = if battle.outcome.captured {
                    ("Captured", COLORS[attacker])
                } else {
                    ("Defense held", COLORS[battle.defender])
                };
                ui.colored_label(color, status);
            });
            let outcome = &battle.outcome;
            for (side, player, rolls, opposing, losses) in [
                (
                    "Attack",
                    attacker,
                    outcome.attack.as_slice(),
                    outcome.defense.as_slice(),
                    outcome.attacker_losses,
                ),
                (
                    "Defense",
                    battle.defender,
                    outcome.defense.as_slice(),
                    outcome.attack.as_slice(),
                    outcome.defender_losses,
                ),
            ] {
                ui.horizontal_wrapped(|ui| {
                    ui.add_sized(
                        [88.0, 24.0],
                        egui::Label::new(
                            RichText::new(format!("{side} · P{}", player + 1))
                                .color(COLORS[player]),
                        ),
                    );
                    for rank in 0..3 {
                        let roll = rolls.get(rank).copied().unwrap_or(0);
                        let opponent = opposing.get(rank).copied().unwrap_or(0);
                        if roll == 0 {
                            ui.allocate_space(vec2(24.0, 24.0));
                        } else {
                            let won = roll > opponent || (roll == opponent && side == "Defense");
                            die(ui, roll, COLORS[player], opponent, won);
                        }
                    }
                    let noun = if losses == 1 { "troop" } else { "troops" };
                    ui.label(format!("Lost {losses} {noun}"));
                });
            }
        });
}

fn die(ui: &mut egui::Ui, value: u8, color: Color32, opponent: u8, won: bool) {
    let (response, painter) = ui.allocate_painter(vec2(24.0, 24.0), egui::Sense::hover());
    let fill = if opponent == 0 {
        Color32::from_rgb(124, 142, 157)
    } else {
        Color32::from_rgb(232, 239, 245)
    };
    let stroke = if opponent != 0 && won {
        Stroke::new(2.0, color)
    } else {
        Stroke::NONE
    };
    painter.rect(response.rect, 5.0, fill, stroke, StrokeKind::Inside);
    let center = response.rect.center();
    for (x, y) in [(-1, -1), (1, 1), (1, -1), (-1, 1), (-1, 0), (1, 0), (0, 0)]
        .into_iter()
        .enumerate()
        .filter_map(|(index, point)| {
            let visible = match index {
                0..=1 => value >= 2,
                2..=3 => value >= 4,
                4..=5 => value == 6,
                _ => value % 2 == 1,
            };
            visible.then_some(point)
        })
    {
        painter.circle_filled(
            center + vec2(x as f32, y as f32) * 5.0,
            1.75,
            Color32::from_rgb(17, 29, 40),
        );
    }
    let comparison = match (opponent, won) {
        (0, _) => "Unpaired",
        (other, _) if other == value => "Tie goes to the defender",
        (_, true) => "Won comparison",
        _ => "Lost comparison",
    };
    response.on_hover_text(format!("Rolled {value} · {comparison}"));
}
