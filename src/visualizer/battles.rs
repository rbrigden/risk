//! Animated Dice Roller, Live Battle Spotlight, and Chronological Battle Log.

use eframe::egui::{
    self, Align, Align2, Color32, CornerRadius, FontId, Layout, Margin, Rect, RichText, Sense,
    Stroke, StrokeKind, Vec2, pos2, vec2,
};

use super::state::{BattlePanelOutput, BattleRecord, player_color, territory_name};

/// Duration of the dice tumble animation in seconds.
pub(super) const DICE_ANIM_DURATION: f64 = 0.55;

/// Maximum number of recent battles displayed in the scrollable history log.
const MAX_HISTORY_ENTRIES: usize = 50;

/// Duration of the brief highlight flash right after each die locks in.
const DIE_SNAP_FLASH_DURATION: f64 = 0.14;

// Maritime dark-slate palette matching the rest of the visualizer.
const SPOTLIGHT_BG: Color32 = Color32::from_rgb(24, 37, 52); // #182534
const SPOTLIGHT_BORDER: Color32 = Color32::from_rgb(44, 66, 87); // #2c4257
const TRAY_BG: Color32 = Color32::from_rgb(16, 26, 37);
const TRAY_BORDER: Color32 = Color32::from_rgb(31, 48, 66);
const HISTORY_BG: Color32 = Color32::from_rgb(26, 40, 55); // #1a2837
const HISTORY_BORDER: Color32 = Color32::from_rgb(38, 58, 78);

const TEXT_PRIMARY: Color32 = Color32::from_rgb(232, 240, 248);
const TEXT_MUTED: Color32 = Color32::from_rgb(148, 168, 186);
const TEXT_DIM: Color32 = Color32::from_rgb(104, 124, 142);

const AMBER_GOLD: Color32 = Color32::from_rgb(245, 197, 66);
const LOSS_RED: Color32 = Color32::from_rgb(240, 102, 96);

// Classic high-contrast Risk die colors: crimson attack dice vs steel-blue defense dice.
const ATK_DIE_BODY: Color32 = Color32::from_rgb(214, 69, 69); // #d64545
const ATK_DIE_HIGHLIGHT: Color32 = Color32::from_rgb(240, 108, 108); // #f06c6c
const ATK_DIE_SHADOW_BEVEL: Color32 = Color32::from_rgb(153, 38, 38); // #992626
const ATK_DIE_DIM_BODY: Color32 = Color32::from_rgb(108, 66, 72);
const ATK_DIE_DIM_BEVEL: Color32 = Color32::from_rgb(76, 45, 50);

const DEF_DIE_BODY: Color32 = Color32::from_rgb(56, 108, 166);
const DEF_DIE_HIGHLIGHT: Color32 = Color32::from_rgb(104, 158, 216);
const DEF_DIE_SHADOW_BEVEL: Color32 = Color32::from_rgb(35, 72, 115);
const DEF_DIE_DIM_BODY: Color32 = Color32::from_rgb(62, 82, 104);
const DEF_DIE_DIM_BEVEL: Color32 = Color32::from_rgb(44, 59, 76);

const PIP_WHITE: Color32 = Color32::from_rgb(249, 251, 253);
const PIP_ROLLING: Color32 = Color32::from_rgb(255, 244, 218);
const PIP_DIM: Color32 = Color32::from_rgb(184, 194, 204);

/// Result of comparing a single attacker die against the paired defender die.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DieOutcome {
    AttackerWins,
    DefenderWins,
    DefenderWinsTie,
    Unpaired,
}

/// Compare an attacker die against a defender die using Risk rules (ties go to defender).
/// A value of `0` on either side denotes an unpaired die.
pub(super) fn compare_die_pair(attacker_roll: u8, defender_roll: u8) -> DieOutcome {
    if attacker_roll == 0 || defender_roll == 0 {
        DieOutcome::Unpaired
    } else if attacker_roll > defender_roll {
        DieOutcome::AttackerWins
    } else if attacker_roll == defender_roll {
        DieOutcome::DefenderWinsTie
    } else {
        DieOutcome::DefenderWins
    }
}

/// Deterministic pseudo-random die face in `1..=6` used while a die is tumbling.
pub(super) fn rolling_face(battle_id: u64, side: u8, rank: usize, tick: u64) -> u8 {
    let mut x = battle_id.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (u64::from(side).wrapping_add(1)).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ (rank as u64 + 1).wrapping_mul(0x94D0_49BB_1331_11EB)
        ^ tick.wrapping_mul(0xD1B5_4A32_D192_ED03);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    ((x % 6) as u8) + 1
}

/// Whether `battle` is currently playing its dice tumble animation at `now`.
pub(super) fn is_animating(battle: &BattleRecord, now: f64, animate: bool) -> bool {
    if !animate {
        return false;
    }
    let elapsed = now - battle.timestamp;
    (0.0..DICE_ANIM_DURATION).contains(&elapsed)
}

/// Staggered settle time (in seconds, within `0.32..=DICE_ANIM_DURATION`) so dice lock in
/// left-to-right with defender settling a dramatic beat after attacker in each pair.
fn die_settle_time(side: u8, rank: usize) -> f64 {
    let base: f64 = match rank {
        0 => 0.32,
        1 => 0.42,
        _ => 0.51,
    };
    let side_offset: f64 = if side == 0 { 0.0 } else { 0.04 };
    (base + side_offset).min(DICE_ANIM_DURATION)
}

/// Draw the prominent "Live Battle / Dice Roll Arena" spotlight for the most recent battle.
pub(super) fn draw_spotlight(
    ui: &mut egui::Ui,
    battle: Option<&BattleRecord>,
    seat_labels: &[String],
    now: f64,
    animate: bool,
) -> BattlePanelOutput {
    let Some(battle) = battle else {
        draw_empty_spotlight(ui);
        return BattlePanelOutput::default();
    };

    let animating = is_animating(battle, now, animate);
    if animating {
        ui.ctx().request_repaint();
    }
    let elapsed = (now - battle.timestamp).max(0.0);

    let atk_color = player_color(battle.attacker);
    let def_color = player_color(battle.defender);

    let border_color = if animating {
        AMBER_GOLD.gamma_multiply(0.65)
    } else if battle.captured {
        atk_color.gamma_multiply(0.75)
    } else {
        SPOTLIGHT_BORDER
    };

    let frame_resp = egui::Frame::new()
        .fill(SPOTLIGHT_BG)
        .corner_radius(10.0)
        .inner_margin(Margin::symmetric(12, 10))
        .stroke(Stroke::new(1.25, border_color))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());

            // 1. Arena header strip + live status badge.
            draw_spotlight_top_bar(ui, battle, animating, elapsed, atk_color, def_color);
            ui.add_space(6.0);

            // 2. Matchup banner: Attacker pill · Source (#src) → Target (#dst) · Defender pill.
            draw_matchup_header(ui, battle, seat_labels, atk_color, def_color);
            ui.add_space(8.0);

            // 3. Centerpiece head-to-head animated dice lanes.
            draw_dice_arena_tray(ui, battle, animating, elapsed, atk_color, def_color);
            ui.add_space(8.0);

            // 4. Casualty & outcome footer.
            draw_casualty_footer(ui, battle, animating, atk_color, def_color);
        })
        .response;

    let hovered = frame_resp.contains_pointer() || frame_resp.hovered();
    if hovered {
        ui.painter().rect_stroke(
            frame_resp.rect,
            10.0,
            Stroke::new(1.5, Color32::from_rgba_unmultiplied(158, 212, 255, 160)),
            StrokeKind::Inside,
        );
    }

    BattlePanelOutput {
        hovered_edge: hovered.then_some((battle.source, battle.target)),
        animating,
    }
}

/// Draw the scrollable chronological Battle Log (most recent first).
pub(super) fn draw_history(
    ui: &mut egui::Ui,
    battles: &[BattleRecord],
    seat_labels: &[String],
    now: f64,
) -> BattlePanelOutput {
    let mut output = BattlePanelOutput::default();

    if battles.is_empty() {
        egui::Frame::new()
            .fill(HISTORY_BG.gamma_multiply(0.75))
            .corner_radius(8.0)
            .inner_margin(Margin::symmetric(12, 14))
            .stroke(Stroke::new(1.0, HISTORY_BORDER))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.vertical_centered(|ui| {
                    ui.label(
                        RichText::new("No battles recorded yet.")
                            .size(12.0)
                            .color(TEXT_DIM),
                    );
                });
            });
        return output;
    }

    egui::ScrollArea::vertical()
        .id_salt("battle_history_log")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 6.0;
            for battle in battles.iter().rev().take(MAX_HISTORY_ENTRIES) {
                if draw_history_card(ui, battle, seat_labels, now) {
                    output.hovered_edge = Some((battle.source, battle.target));
                }
            }
        });

    output
}

// ---------------------------------------------------------------------------
// Spotlight subcomponents
// ---------------------------------------------------------------------------

fn draw_empty_spotlight(ui: &mut egui::Ui) {
    egui::Frame::new()
        .fill(SPOTLIGHT_BG)
        .corner_radius(10.0)
        .inner_margin(Margin::symmetric(12, 12))
        .stroke(Stroke::new(1.0, SPOTLIGHT_BORDER))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("DICE ARENA")
                        .size(10.5)
                        .strong()
                        .color(TEXT_MUTED),
                );
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    draw_pill(ui, "READY", TEXT_DIM, false);
                });
            });

            ui.add_space(10.0);

            // Subtle illustrative ghost dice row.
            ui.vertical_centered(|ui| {
                egui::Frame::new()
                    .fill(TRAY_BG)
                    .corner_radius(8.0)
                    .inner_margin(Margin::symmetric(12, 10))
                    .stroke(Stroke::new(1.0, TRAY_BORDER))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 6.0;
                            for face in [6_u8, 4, 2] {
                                draw_ghost_die(ui, 28.0, face);
                            }
                            ui.add_space(4.0);
                            ui.label(RichText::new("vs").size(11.0).strong().color(TEXT_DIM));
                            ui.add_space(4.0);
                            for face in [5_u8, 3] {
                                draw_ghost_die(ui, 28.0, face);
                            }
                        });
                    });

                ui.add_space(8.0);
                ui.label(
                    RichText::new("Dice Arena · Select a territory and attack to roll dice!")
                        .size(11.5)
                        .color(TEXT_MUTED),
                );
            });
        });
}

fn draw_spotlight_top_bar(
    ui: &mut egui::Ui,
    battle: &BattleRecord,
    animating: bool,
    elapsed: f64,
    atk_color: Color32,
    def_color: Color32,
) {
    ui.horizontal(|ui| {
        if animating {
            let pulse = 0.55 + 0.45 * ((elapsed * 14.0).sin() as f32).abs();
            let dot_color = AMBER_GOLD.gamma_multiply(pulse);
            draw_dot(ui, dot_color, 4.0);
            ui.label(
                RichText::new("LIVE BATTLE")
                    .size(10.5)
                    .strong()
                    .color(AMBER_GOLD),
            );
        } else {
            ui.label(
                RichText::new("DICE ARENA")
                    .size(10.5)
                    .strong()
                    .color(TEXT_MUTED),
            );
        }
        ui.label(
            RichText::new(format!("· T{}", battle.turn))
                .size(10.5)
                .monospace()
                .color(TEXT_DIM),
        );

        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if animating {
                draw_pill(ui, "ROLLING...", AMBER_GOLD, true);
            } else if battle.captured {
                draw_glowing_capture_pill(ui, "CAPTURED!", atk_color);
            } else if battle.attacker_losses > 0 && battle.defender_losses > 0 {
                draw_pill(ui, "CLASH", def_color, true);
            } else {
                draw_pill(ui, "HELD", def_color, true);
            }
        });
    });
}

fn draw_matchup_header(
    ui: &mut egui::Ui,
    battle: &BattleRecord,
    seat_labels: &[String],
    atk_color: Color32,
    def_color: Color32,
) {
    let atk_label = seat_label(seat_labels, battle.attacker);
    let def_label = seat_label(seat_labels, battle.defender);
    let src_name = territory_name(battle.source);
    let dst_name = territory_name(battle.target);

    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = vec2(6.0, 4.0);
        draw_player_pill(ui, &atk_label, atk_color);
        ui.label(
            RichText::new(format!(
                "{src_name} (#{}) → {dst_name} (#{})",
                battle.source, battle.target
            ))
            .size(12.5)
            .strong()
            .color(TEXT_PRIMARY),
        );
        draw_player_pill(ui, &def_label, def_color);
    });
}

fn draw_dice_arena_tray(
    ui: &mut egui::Ui,
    battle: &BattleRecord,
    animating: bool,
    elapsed: f64,
    atk_color: Color32,
    def_color: Color32,
) {
    let atk_count = battle.attack_dice_count.min(battle.attack_rolls.len());
    let def_count = battle.defense_dice_count.min(battle.defense_rolls.len());
    let cols = atk_count.max(def_count).clamp(1, 3);
    let tick = (elapsed * 22.0) as u64;

    let die_size = 40.0;
    let col_width = 56.0;
    let col_gap = 8.0;
    let label_width = 32.0;

    egui::Frame::new()
        .fill(TRAY_BG)
        .corner_radius(8.0)
        .inner_margin(Margin::symmetric(10, 10))
        .stroke(Stroke::new(1.0, TRAY_BORDER))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());

            // Center the 3-lane block inside the tray.
            let content_width = label_width
                + 6.0
                + cols as f32 * col_width
                + (cols.saturating_sub(1)) as f32 * col_gap;
            let left_pad = ((ui.available_width() - content_width) * 0.5).max(0.0);

            // Top lane: Attacker dice.
            ui.horizontal(|ui| {
                if left_pad > 0.0 {
                    ui.add_space(left_pad);
                }
                ui.add_sized(
                    [label_width, die_size],
                    egui::Label::new(
                        RichText::new("ATK")
                            .size(10.0)
                            .strong()
                            .color(ATK_DIE_HIGHLIGHT),
                    ),
                );
                ui.add_space(6.0);
                for rank in 0..cols {
                    if rank > 0 {
                        ui.add_space(col_gap);
                    }
                    let atk_val = if rank < atk_count {
                        battle.attack_rolls[rank]
                    } else {
                        0
                    };
                    let def_val = if rank < def_count {
                        battle.defense_rolls[rank]
                    } else {
                        0
                    };
                    let outcome = compare_die_pair(atk_val, def_val);
                    let settle_t = die_settle_time(0, rank);
                    let pair_settle_t = settle_t.max(die_settle_time(1, rank));
                    let die_rolling = animating && elapsed < settle_t;
                    let pair_settled = !animating || elapsed >= pair_settle_t;

                    let (cell_rect, _) =
                        ui.allocate_exact_size(vec2(col_width, die_size), Sense::hover());
                    if atk_val > 0 {
                        let shown = if die_rolling {
                            rolling_face(battle.id, 0, rank, tick)
                        } else {
                            atk_val
                        };
                        let jitter = if die_rolling {
                            tumble_offset(elapsed, settle_t, 0, rank)
                        } else {
                            Vec2::ZERO
                        };
                        let snap = if animating && elapsed >= settle_t {
                            (1.0 - (elapsed - settle_t) / DIE_SNAP_FLASH_DURATION).clamp(0.0, 1.0)
                                as f32
                        } else {
                            0.0
                        };
                        let die_rect =
                            Rect::from_center_size(cell_rect.center(), vec2(die_size, die_size));
                        let spec = DieSpec {
                            value: shown,
                            kind: DieKind::Attack,
                            paired: def_val > 0,
                            won: pair_settled && outcome == DieOutcome::AttackerWins,
                            rolling: die_rolling,
                            winner_color: atk_color,
                            jitter,
                            snap_flash: snap,
                        };
                        paint_die(
                            ui,
                            die_rect,
                            &spec,
                            Some(die_tooltip(
                                "Attacker",
                                rank,
                                atk_val,
                                def_val,
                                outcome,
                                die_rolling,
                            )),
                        );
                    }
                }
            });

            ui.add_space(6.0);

            // Middle lane: Pairwise comparison badges.
            ui.horizontal(|ui| {
                if left_pad > 0.0 {
                    ui.add_space(left_pad);
                }
                ui.add_sized(
                    [label_width, 22.0],
                    egui::Label::new(RichText::new("VS").size(9.5).strong().color(TEXT_DIM)),
                );
                ui.add_space(6.0);
                for rank in 0..cols {
                    if rank > 0 {
                        ui.add_space(col_gap);
                    }
                    let atk_val = if rank < atk_count {
                        battle.attack_rolls[rank]
                    } else {
                        0
                    };
                    let def_val = if rank < def_count {
                        battle.defense_rolls[rank]
                    } else {
                        0
                    };
                    let outcome = compare_die_pair(atk_val, def_val);
                    let pair_settle_t = die_settle_time(0, rank).max(die_settle_time(1, rank));
                    let pair_settled = !animating || elapsed >= pair_settle_t;

                    draw_comparison_badge(
                        ui,
                        col_width,
                        outcome,
                        pair_settled,
                        elapsed,
                        atk_val,
                        def_val,
                        atk_color,
                        def_color,
                    );
                }
            });

            ui.add_space(6.0);

            // Bottom lane: Defender dice.
            ui.horizontal(|ui| {
                if left_pad > 0.0 {
                    ui.add_space(left_pad);
                }
                ui.add_sized(
                    [label_width, die_size],
                    egui::Label::new(
                        RichText::new("DEF")
                            .size(10.0)
                            .strong()
                            .color(DEF_DIE_HIGHLIGHT),
                    ),
                );
                ui.add_space(6.0);
                for rank in 0..cols {
                    if rank > 0 {
                        ui.add_space(col_gap);
                    }
                    let atk_val = if rank < atk_count {
                        battle.attack_rolls[rank]
                    } else {
                        0
                    };
                    let def_val = if rank < def_count {
                        battle.defense_rolls[rank]
                    } else {
                        0
                    };
                    let outcome = compare_die_pair(atk_val, def_val);
                    let settle_t = die_settle_time(1, rank);
                    let pair_settle_t = die_settle_time(0, rank).max(settle_t);
                    let die_rolling = animating && elapsed < settle_t;
                    let pair_settled = !animating || elapsed >= pair_settle_t;

                    let (cell_rect, _) =
                        ui.allocate_exact_size(vec2(col_width, die_size), Sense::hover());
                    if def_val > 0 {
                        let shown = if die_rolling {
                            rolling_face(battle.id, 1, rank, tick)
                        } else {
                            def_val
                        };
                        let jitter = if die_rolling {
                            tumble_offset(elapsed, settle_t, 1, rank)
                        } else {
                            Vec2::ZERO
                        };
                        let snap = if animating && elapsed >= settle_t {
                            (1.0 - (elapsed - settle_t) / DIE_SNAP_FLASH_DURATION).clamp(0.0, 1.0)
                                as f32
                        } else {
                            0.0
                        };
                        let die_rect =
                            Rect::from_center_size(cell_rect.center(), vec2(die_size, die_size));
                        let spec = DieSpec {
                            value: shown,
                            kind: DieKind::Defense,
                            paired: atk_val > 0,
                            won: pair_settled
                                && matches!(
                                    outcome,
                                    DieOutcome::DefenderWins | DieOutcome::DefenderWinsTie
                                ),
                            rolling: die_rolling,
                            winner_color: def_color,
                            jitter,
                            snap_flash: snap,
                        };
                        paint_die(
                            ui,
                            die_rect,
                            &spec,
                            Some(die_tooltip(
                                "Defender",
                                rank,
                                def_val,
                                atk_val,
                                outcome,
                                die_rolling,
                            )),
                        );
                    } else {
                        // Subtle empty socket for symmetry when attacker has more dice.
                        let socket = Rect::from_center_size(
                            cell_rect.center(),
                            vec2(die_size - 6.0, die_size - 6.0),
                        );
                        ui.painter().rect_stroke(
                            socket,
                            7.0,
                            Stroke::new(1.0, Color32::from_rgba_unmultiplied(64, 88, 112, 65)),
                            StrokeKind::Inside,
                        );
                    }
                }
            });
        });
}

#[allow(clippy::too_many_arguments)]
fn draw_comparison_badge(
    ui: &mut egui::Ui,
    col_width: f32,
    outcome: DieOutcome,
    settled: bool,
    elapsed: f64,
    atk_val: u8,
    def_val: u8,
    atk_color: Color32,
    def_color: Color32,
) {
    let (rect, response) = ui.allocate_exact_size(vec2(col_width, 22.0), Sense::hover());
    let painter = ui.painter();

    if outcome == DieOutcome::Unpaired {
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            "—",
            FontId::proportional(12.0),
            TEXT_DIM,
        );
        response.on_hover_text("Unpaired die · No opposing die to compare");
        return;
    }

    if !settled {
        let pulse = 0.45 + 0.55 * ((elapsed * 14.0).sin() as f32).abs();
        let pill_rect = Rect::from_center_size(rect.center(), vec2(44.0, 18.0));
        painter.rect(
            pill_rect,
            9.0,
            AMBER_GOLD.gamma_multiply(0.14 * pulse),
            Stroke::new(1.0, AMBER_GOLD.gamma_multiply(0.55 * pulse)),
            StrokeKind::Inside,
        );
        painter.text(
            pill_rect.center(),
            Align2::CENTER_CENTER,
            "···",
            FontId::monospace(11.0),
            AMBER_GOLD.gamma_multiply(0.65 + 0.35 * pulse),
        );
        return;
    }

    let (label, color, tooltip) = match outcome {
        DieOutcome::AttackerWins => (
            "▲ WIN",
            atk_color,
            format!("Attacker wins ({atk_val} > {def_val}) · Defender loses 1 troop"),
        ),
        DieOutcome::DefenderWins => (
            "▼ DEF",
            def_color,
            format!("Defender wins ({def_val} > {atk_val}) · Attacker loses 1 troop"),
        ),
        DieOutcome::DefenderWinsTie => (
            "▼ DEF",
            def_color,
            format!("Tie goes to defender ({atk_val} = {def_val}) · Attacker loses 1 troop"),
        ),
        DieOutcome::Unpaired => unreachable!(),
    };

    let pill_rect = Rect::from_center_size(rect.center(), vec2(50.0, 19.0));
    painter.rect(
        pill_rect,
        9.5,
        color.gamma_multiply(0.20),
        Stroke::new(1.25, color.gamma_multiply(0.85)),
        StrokeKind::Inside,
    );
    painter.text(
        pill_rect.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::proportional(10.0),
        color,
    );
    response.on_hover_text(tooltip);
}

fn draw_casualty_footer(
    ui: &mut egui::Ui,
    battle: &BattleRecord,
    animating: bool,
    atk_color: Color32,
    def_color: Color32,
) {
    if animating {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Dice tumbling onto the board...")
                    .size(11.0)
                    .italics()
                    .color(TEXT_MUTED),
            );
        });
        return;
    }

    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = vec2(6.0, 4.0);

        // Attacker casualty & remainder.
        draw_dot(ui, atk_color, 3.5);
        ui.label(RichText::new("ATK").size(10.5).strong().color(TEXT_MUTED));
        draw_loss_badge(ui, battle.attacker_losses);
        ui.label(
            RichText::new(format!("→ {} left", battle.attacker_troops_after))
                .size(11.0)
                .color(TEXT_MUTED),
        );

        ui.label(RichText::new("·").size(11.0).color(TEXT_DIM));

        // Defender casualty & capture/remainder.
        draw_dot(ui, def_color, 3.5);
        ui.label(RichText::new("DEF").size(10.5).strong().color(TEXT_MUTED));
        draw_loss_badge(ui, battle.defender_losses);
        if battle.captured {
            ui.label(
                RichText::new(format!(
                    "CAPTURED ({} moved in)",
                    battle.defender_troops_after
                ))
                .size(11.0)
                .strong()
                .color(atk_color),
            );
        } else {
            ui.label(
                RichText::new(format!("→ {} left", battle.defender_troops_after))
                    .size(11.0)
                    .color(TEXT_MUTED),
            );
        }
    });
}

// ---------------------------------------------------------------------------
// History card
// ---------------------------------------------------------------------------

/// Draws a single compact chronological history card. Returns `true` if hovered.
fn draw_history_card(
    ui: &mut egui::Ui,
    battle: &BattleRecord,
    seat_labels: &[String],
    now: f64,
) -> bool {
    let atk_color = player_color(battle.attacker);
    let def_color = player_color(battle.defender);
    let src_name = territory_name(battle.source);
    let dst_name = territory_name(battle.target);

    let age = (now - battle.timestamp).max(0.0);
    let is_fresh = age < 3.0;
    let base_border = if battle.captured {
        atk_color.gamma_multiply(0.45)
    } else if is_fresh {
        AMBER_GOLD.gamma_multiply(0.42)
    } else {
        HISTORY_BORDER
    };

    let card_resp = egui::Frame::new()
        .fill(HISTORY_BG)
        .corner_radius(7.0)
        .inner_margin(Margin::symmetric(9, 7))
        .stroke(Stroke::new(1.0, base_border))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());

            // Header: Turn, source -> target, compact outcome badge.
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 5.0;
                ui.label(
                    RichText::new(format!("T{}", battle.turn))
                        .size(10.5)
                        .monospace()
                        .color(TEXT_DIM),
                );
                draw_dot(ui, atk_color, 3.0);
                ui.label(
                    RichText::new(format!("{src_name} → {dst_name}"))
                        .size(11.5)
                        .strong()
                        .color(TEXT_PRIMARY),
                );
                draw_dot(ui, def_color, 3.0);

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if battle.captured {
                        draw_pill(ui, "CAPTURED", atk_color, true);
                    } else {
                        let summary =
                            format!("-{}/-{}", battle.attacker_losses, battle.defender_losses);
                        draw_pill(ui, &summary, TEXT_MUTED, false);
                    }
                });
            });

            ui.add_space(5.0);

            // Compact dice row: mini attack dice vs mini defense dice + brief loss summary.
            let atk_count = battle.attack_dice_count.min(battle.attack_rolls.len());
            let def_count = battle.defense_dice_count.min(battle.defense_rolls.len());
            let mini_size = 20.0;

            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;

                for rank in 0..atk_count {
                    let atk_val = battle.attack_rolls[rank];
                    let def_val = if rank < def_count {
                        battle.defense_rolls[rank]
                    } else {
                        0
                    };
                    let outcome = compare_die_pair(atk_val, def_val);
                    let (rect, _) =
                        ui.allocate_exact_size(vec2(mini_size, mini_size), Sense::hover());
                    let spec = DieSpec {
                        value: atk_val,
                        kind: DieKind::Attack,
                        paired: def_val > 0,
                        won: outcome == DieOutcome::AttackerWins,
                        rolling: false,
                        winner_color: atk_color,
                        jitter: Vec2::ZERO,
                        snap_flash: 0.0,
                    };
                    paint_die(
                        ui,
                        rect,
                        &spec,
                        Some(die_tooltip(
                            "Attacker", rank, atk_val, def_val, outcome, false,
                        )),
                    );
                }

                ui.label(RichText::new("vs").size(10.0).color(TEXT_DIM));

                for rank in 0..def_count {
                    let def_val = battle.defense_rolls[rank];
                    let atk_val = if rank < atk_count {
                        battle.attack_rolls[rank]
                    } else {
                        0
                    };
                    let outcome = compare_die_pair(atk_val, def_val);
                    let (rect, _) =
                        ui.allocate_exact_size(vec2(mini_size, mini_size), Sense::hover());
                    let spec = DieSpec {
                        value: def_val,
                        kind: DieKind::Defense,
                        paired: atk_val > 0,
                        won: matches!(
                            outcome,
                            DieOutcome::DefenderWins | DieOutcome::DefenderWinsTie
                        ),
                        rolling: false,
                        winner_color: def_color,
                        jitter: Vec2::ZERO,
                        snap_flash: 0.0,
                    };
                    paint_die(
                        ui,
                        rect,
                        &spec,
                        Some(die_tooltip(
                            "Defender", rank, def_val, atk_val, outcome, false,
                        )),
                    );
                }

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let atk_lbl = seat_label(seat_labels, battle.attacker);
                    let def_lbl = seat_label(seat_labels, battle.defender);
                    let detail = if battle.captured {
                        format!("{} took ({})", atk_lbl, battle.defender_troops_after)
                    } else {
                        format!(
                            "{}:-{} {}:-{}",
                            short_label(&atk_lbl),
                            battle.attacker_losses,
                            short_label(&def_lbl),
                            battle.defender_losses
                        )
                    };
                    let detail_color = if battle.captured {
                        atk_color
                    } else {
                        TEXT_MUTED
                    };
                    ui.label(RichText::new(detail).size(10.5).color(detail_color));
                });
            });
        })
        .response;

    let hovered = card_resp.contains_pointer() || card_resp.hovered();
    if hovered {
        ui.painter().rect_stroke(
            card_resp.rect,
            7.0,
            Stroke::new(1.5, Color32::from_rgba_unmultiplied(150, 206, 252, 180)),
            StrokeKind::Inside,
        );
    }
    hovered
}

// ---------------------------------------------------------------------------
// 3D-styled vector dice renderer & animation math
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum DieKind {
    Attack,
    Defense,
}

struct DieSpec {
    value: u8,
    kind: DieKind,
    paired: bool,
    won: bool,
    rolling: bool,
    winner_color: Color32,
    jitter: Vec2,
    snap_flash: f32,
}

/// Damped bounce/jitter offset while a die is tumbling toward `settle_t`.
fn tumble_offset(elapsed: f64, settle_t: f64, side: u8, rank: usize) -> Vec2 {
    let progress = (elapsed / settle_t.max(1e-6)).clamp(0.0, 1.0);
    let decay = (1.0 - progress).powi(2);
    let phase = elapsed * 38.0 + (rank as f64) * 1.9 + (side as f64) * 2.7;
    let dx = (phase.sin() * 3.0 * decay) as f32;
    let dy = ((phase * 1.37).cos() * 3.4 * decay - 2.2 * decay) as f32;
    vec2(dx, dy)
}

/// Paint a crisp, 3D-beveled vector die with drop shadow, winner glow ring, and high-contrast pips.
fn paint_die(ui: &mut egui::Ui, base_rect: Rect, spec: &DieSpec, tooltip: Option<String>) {
    let response = ui.interact(
        base_rect,
        ui.scope_id().with((
            "battle_die",
            base_rect.min.x.to_bits(),
            base_rect.min.y.to_bits(),
        )),
        Sense::hover(),
    );
    let painter = ui.painter();
    let size = base_rect.width().min(base_rect.height());
    let rect = base_rect.translate(spec.jitter);
    let rounding = (size * 0.20).clamp(4.0, 8.0);

    // 1. Drop shadow (shifts slightly further when hopping during tumble).
    let lift = (-spec.jitter.y).max(0.0);
    let shadow_offset = vec2(size * 0.06, size * 0.09 + lift * 0.45);
    let shadow_alpha = if spec.paired || spec.rolling { 115 } else { 65 };
    painter.rect_filled(
        rect.translate(shadow_offset),
        rounding,
        Color32::from_rgba_unmultiplied(4, 9, 15, shadow_alpha),
    );

    // 2. Subtle winner outer glow halo when settled & won.
    if spec.won && spec.paired && !spec.rolling {
        let glow_pad = (size * 0.075).clamp(1.5, 3.2);
        painter.rect_filled(
            rect.expand(glow_pad),
            rounding + glow_pad * 0.7,
            Color32::from_rgba_unmultiplied(
                spec.winner_color.r(),
                spec.winner_color.g(),
                spec.winner_color.b(),
                62,
            ),
        );
    }

    // 3. 3D beveled die body (outer darker bevel + slightly shifted top-left face + inner sheen).
    let (body, bevel, highlight, pip_color) = match (spec.kind, spec.paired || spec.rolling) {
        (DieKind::Attack, true) => (
            ATK_DIE_BODY,
            ATK_DIE_SHADOW_BEVEL,
            ATK_DIE_HIGHLIGHT,
            if spec.rolling { PIP_ROLLING } else { PIP_WHITE },
        ),
        (DieKind::Attack, false) => (
            ATK_DIE_DIM_BODY,
            ATK_DIE_DIM_BEVEL,
            ATK_DIE_DIM_BODY,
            PIP_DIM,
        ),
        (DieKind::Defense, true) => (
            DEF_DIE_BODY,
            DEF_DIE_SHADOW_BEVEL,
            DEF_DIE_HIGHLIGHT,
            if spec.rolling { PIP_ROLLING } else { PIP_WHITE },
        ),
        (DieKind::Defense, false) => (
            DEF_DIE_DIM_BODY,
            DEF_DIE_DIM_BEVEL,
            DEF_DIE_DIM_BODY,
            PIP_DIM,
        ),
    };

    // Base beveled block.
    painter.rect_filled(rect, rounding, bevel);

    // Raised upper-left face to sell a tactile 3D resin die bevel.
    let bevel_px = (size * 0.055).clamp(1.0, 2.2);
    let face_rect = Rect::from_min_max(rect.min, rect.max - vec2(bevel_px, bevel_px));
    painter.rect_filled(face_rect, (rounding - 0.5).max(3.0), body);

    // Subtle top sheen highlight band.
    if spec.paired || spec.rolling {
        let sheen_rect = Rect::from_min_max(
            face_rect.min + vec2(1.0, 1.0),
            pos2(
                face_rect.max.x - 1.0,
                face_rect.min.y + face_rect.height() * 0.36,
            ),
        );
        painter.rect_filled(
            sheen_rect,
            CornerRadius {
                nw: (rounding * 0.75) as u8,
                ne: (rounding * 0.75) as u8,
                sw: 2,
                se: 2,
            },
            Color32::from_rgba_unmultiplied(255, 255, 255, 24),
        );
    }

    // Rim stroke: winning player color glow ring, rolling warm rim, or crisp highlight bevel.
    let border_stroke = if spec.won && spec.paired && !spec.rolling {
        Stroke::new((size * 0.055).clamp(1.5, 2.3), spec.winner_color)
    } else if spec.rolling {
        Stroke::new(1.25, AMBER_GOLD.gamma_multiply(0.8))
    } else if spec.paired {
        Stroke::new(1.0, highlight.gamma_multiply(0.7))
    } else {
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(150, 168, 184, 55))
    };
    painter.rect_stroke(rect, rounding, border_stroke, StrokeKind::Inside);

    // 4. High-contrast pips (1..=6) with subtle recessed drop shadow.
    paint_pips(painter, face_rect.center(), size, spec.value, pip_color);

    // 5. Snap lock-in flash ring right after settling.
    if spec.snap_flash > 0.01 {
        let expand = (1.0 - spec.snap_flash) * 3.5;
        let alpha = (spec.snap_flash * 190.0) as u8;
        painter.rect_stroke(
            rect.expand(expand),
            rounding + expand * 0.5,
            Stroke::new(1.5, Color32::from_rgba_unmultiplied(255, 240, 195, alpha)),
            StrokeKind::Outside,
        );
    }

    if let Some(text) = tooltip {
        response.on_hover_text(text);
    }
}

fn draw_ghost_die(ui: &mut egui::Ui, size: f32, value: u8) {
    let (rect, _) = ui.allocate_exact_size(vec2(size, size), Sense::hover());
    let painter = ui.painter();
    let rounding = (size * 0.20).clamp(4.0, 6.0);
    painter.rect(
        rect,
        rounding,
        Color32::from_rgb(26, 40, 56),
        Stroke::new(1.0, Color32::from_rgb(46, 68, 90)),
        StrokeKind::Inside,
    );
    paint_pips(
        painter,
        rect.center(),
        size,
        value,
        Color32::from_rgb(66, 92, 118),
    );
}

fn paint_pips(painter: &egui::Painter, center: egui::Pos2, size: f32, value: u8, color: Color32) {
    let value = value.clamp(1, 6);
    let offset = size * 0.235;
    let radius = (size * 0.086).clamp(1.45, 3.6);
    let shadow_dy = (size * 0.022).clamp(0.4, 0.9);
    let pip_shadow = Color32::from_rgba_unmultiplied(8, 12, 18, 120);

    // Standard die pip coordinates: TL, BR, TR, BL, ML, MR, Center.
    let coords = [
        (-1.0_f32, -1.0_f32),
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, 0.0),
        (1.0, 0.0),
        (0.0, 0.0),
    ];

    for (index, (x, y)) in coords.into_iter().enumerate() {
        let visible = match index {
            0..=1 => value >= 2,
            2..=3 => value >= 4,
            4..=5 => value == 6,
            _ => value % 2 == 1,
        };
        if visible {
            let pos = center + vec2(x * offset, y * offset);
            painter.circle_filled(pos + vec2(0.0, shadow_dy), radius, pip_shadow);
            painter.circle_filled(pos, radius, color);
        }
    }
}

fn die_tooltip(
    side_name: &str,
    rank: usize,
    value: u8,
    opponent: u8,
    outcome: DieOutcome,
    rolling: bool,
) -> String {
    if rolling {
        return format!("{side_name} die #{} · Rolling...", rank + 1);
    }
    let detail = match outcome {
        DieOutcome::Unpaired => "Unpaired die".to_string(),
        DieOutcome::DefenderWinsTie => {
            format!("Tie goes to defender ({value} = {opponent})")
        }
        DieOutcome::AttackerWins => {
            if side_name == "Attacker" {
                format!("Won comparison ({value} > {opponent})")
            } else {
                format!("Lost comparison ({value} < {opponent})")
            }
        }
        DieOutcome::DefenderWins => {
            if side_name == "Defender" {
                format!("Won comparison ({value} > {opponent})")
            } else {
                format!("Lost comparison ({value} < {opponent})")
            }
        }
    };
    format!("{side_name} die #{} · Rolled {value} · {detail}", rank + 1)
}

// ---------------------------------------------------------------------------
// Small reusable UI chips & formatting helpers
// ---------------------------------------------------------------------------

fn draw_dot(ui: &mut egui::Ui, color: Color32, radius: f32) {
    let diameter = radius * 2.0 + 2.0;
    let (rect, _) = ui.allocate_exact_size(vec2(diameter, diameter), Sense::hover());
    ui.painter().circle_filled(rect.center(), radius, color);
}

fn draw_player_pill(ui: &mut egui::Ui, label: &str, color: Color32) {
    egui::Frame::new()
        .fill(color.gamma_multiply(0.16))
        .corner_radius(10.0)
        .inner_margin(Margin::symmetric(6, 2))
        .stroke(Stroke::new(1.0, color.gamma_multiply(0.55)))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                draw_dot(ui, color, 3.2);
                ui.label(RichText::new(label).size(11.0).strong().color(color));
            });
        });
}

fn draw_pill(ui: &mut egui::Ui, text: &str, color: Color32, emphasized: bool) {
    let fill_alpha = if emphasized { 0.20 } else { 0.12 };
    let border_alpha = if emphasized { 0.78 } else { 0.35 };
    egui::Frame::new()
        .fill(color.gamma_multiply(fill_alpha))
        .corner_radius(9.0)
        .inner_margin(Margin::symmetric(7, 2))
        .stroke(Stroke::new(1.0, color.gamma_multiply(border_alpha)))
        .show(ui, |ui| {
            ui.label(RichText::new(text).size(10.0).strong().color(color));
        });
}

fn draw_glowing_capture_pill(ui: &mut egui::Ui, text: &str, color: Color32) {
    let resp = egui::Frame::new()
        .fill(color)
        .corner_radius(9.0)
        .inner_margin(Margin::symmetric(8, 2))
        .show(ui, |ui| {
            ui.label(
                RichText::new(text)
                    .size(10.5)
                    .strong()
                    .color(Color32::from_rgb(13, 23, 33)),
            );
        })
        .response;
    ui.painter().rect_stroke(
        resp.rect.expand(1.5),
        10.5,
        Stroke::new(
            2.0,
            Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 115),
        ),
        StrokeKind::Outside,
    );
}

fn draw_loss_badge(ui: &mut egui::Ui, losses: u64) {
    let (text, color) = if losses > 0 {
        (format!("-{losses}"), LOSS_RED)
    } else {
        ("0".to_string(), TEXT_DIM)
    };
    egui::Frame::new()
        .fill(color.gamma_multiply(0.18))
        .corner_radius(5.0)
        .inner_margin(Margin::symmetric(5, 1))
        .stroke(Stroke::new(1.0, color.gamma_multiply(0.5)))
        .show(ui, |ui| {
            ui.label(
                RichText::new(text)
                    .size(10.5)
                    .monospace()
                    .strong()
                    .color(color),
            );
        });
}

fn seat_label(seat_labels: &[String], seat: usize) -> String {
    seat_labels
        .get(seat)
        .filter(|label| !label.is_empty())
        .cloned()
        .unwrap_or_else(|| format!("P{}", seat + 1))
}

fn short_label(label: &str) -> String {
    if label.chars().count() <= 4 {
        label.to_string()
    } else {
        label.chars().take(3).collect()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn sample_battle(timestamp: f64) -> BattleRecord {
        BattleRecord {
            id: 42,
            turn: 3,
            attacker: 0,
            defender: 1,
            source: 0,
            target: 5,
            attack_dice_count: 3,
            defense_dice_count: 2,
            attack_rolls: [6, 4, 2],
            defense_rolls: [5, 4],
            attacker_losses: 1,
            defender_losses: 1,
            captured: false,
            attacker_troops_after: 4,
            defender_troops_after: 2,
            timestamp,
        }
    }

    #[test]
    fn compare_die_pair_handles_win_loss_tie_and_unpaired() {
        assert_eq!(compare_die_pair(6, 5), DieOutcome::AttackerWins);
        assert_eq!(compare_die_pair(2, 1), DieOutcome::AttackerWins);

        assert_eq!(compare_die_pair(3, 5), DieOutcome::DefenderWins);
        assert_eq!(compare_die_pair(1, 6), DieOutcome::DefenderWins);

        // Risk rule: ties always go to the defender.
        for face in 1..=6 {
            assert_eq!(compare_die_pair(face, face), DieOutcome::DefenderWinsTie);
        }

        // Unpaired when either side has no die (0).
        assert_eq!(compare_die_pair(5, 0), DieOutcome::Unpaired);
        assert_eq!(compare_die_pair(0, 4), DieOutcome::Unpaired);
        assert_eq!(compare_die_pair(0, 0), DieOutcome::Unpaired);
    }

    #[test]
    fn rolling_face_stays_in_range_and_varies() {
        let mut seen_ticks = HashSet::new();
        for tick in 0..64 {
            let face = rolling_face(17, 0, 0, tick);
            assert!((1..=6).contains(&face), "tick {tick} produced {face}");
            seen_ticks.insert(face);
        }
        assert!(
            seen_ticks.len() >= 5,
            "expected varied faces across ticks, got {seen_ticks:?}"
        );

        let mut seen_ranks = HashSet::new();
        for battle_id in 0..20 {
            for side in 0..=1 {
                for rank in 0..3 {
                    let face = rolling_face(battle_id, side, rank, 7);
                    assert!((1..=6).contains(&face));
                    seen_ranks.insert(face);
                }
            }
        }
        assert_eq!(seen_ranks.len(), 6, "all 6 die faces should be reachable");

        // Determinism check.
        assert_eq!(rolling_face(99, 1, 2, 11), rolling_face(99, 1, 2, 11));
    }

    #[test]
    fn is_animating_respects_timing_boundaries_and_flag() {
        let battle = sample_battle(10.0);

        // Before the battle timestamp: not animating.
        assert!(!is_animating(&battle, 9.95, true));

        // At battle start and during DICE_ANIM_DURATION: animating.
        assert!(is_animating(&battle, 10.0, true));
        assert!(is_animating(&battle, 10.0 + DICE_ANIM_DURATION * 0.5, true));
        assert!(is_animating(
            &battle,
            10.0 + DICE_ANIM_DURATION - 0.001,
            true
        ));

        // At exact end and after DICE_ANIM_DURATION: settled.
        assert!(!is_animating(&battle, 10.0 + DICE_ANIM_DURATION, true));
        assert!(!is_animating(
            &battle,
            10.0 + DICE_ANIM_DURATION + 0.5,
            true
        ));

        // Disabled animation flag always returns false.
        assert!(!is_animating(&battle, 10.0, false));
        assert!(!is_animating(
            &battle,
            10.0 + DICE_ANIM_DURATION * 0.5,
            false
        ));
    }
}
