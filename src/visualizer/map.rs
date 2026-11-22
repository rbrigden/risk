//! Interactive cartographic Risk board with organic province borders, curved sea lanes,
//! continent control indicators, tactical targeting overlays, and battle animations.

use eframe::egui::{
    self, Align2, Color32, FontId, Pos2, Rect, RichText, Stroke, StrokeKind, Vec2, pos2, vec2,
};

use crate::game::{BRIDGES, EDGES, N, Phase};

use super::state::{
    CONTINENT_BONUS, CONTINENT_NAMES, CONTINENT_SHORT_NAMES, MapInteraction, MapOverlayState,
    continent_of, continent_owner, player_color, territory_name,
};

#[path = "coastlines.rs"]
mod coastlines;
use coastlines::{BRIDGE_CURVES, COASTLINES};

const SIZE: Vec2 = vec2(1280.0, 680.0);

const OCEAN_DEEP: Color32 = Color32::from_rgb(10, 21, 33);
const OCEAN_MID: Color32 = Color32::from_rgb(17, 35, 53);
const OCEAN_CENTER: Color32 = Color32::from_rgb(22, 46, 68);
const LAND_SLATE: Color32 = Color32::from_rgb(18, 30, 44);
const INK: Color32 = Color32::from_rgb(12, 22, 34);
const BORDER_INTERNAL: Color32 = Color32::from_rgb(13, 24, 36);
const COAST_RIM: Color32 = Color32::from_rgba_premultiplied(125, 152, 170, 155);
const GOLD_CHANGED: Color32 = Color32::from_rgb(245, 210, 120);
const ATTACK_GLOW: Color32 = Color32::from_rgb(255, 107, 87);
const FORTIFY_GLOW: Color32 = Color32::from_rgb(93, 228, 199);
const ROUTE_IDLE: Color32 = Color32::from_rgb(126, 178, 206);
const ROUTE_ACTIVE: Color32 = Color32::from_rgb(218, 247, 255);
const BATTLE_DURATION: f32 = 1.6;
const POPUP_DURATION: f32 = 1.4;

pub(super) struct Map {
    territories: [Territory; N],
    continents: [ContinentGeometry; 6],
    bridges: [BridgeRoute; 9],
}

struct Territory {
    outline: Vec<Pos2>,
    triangles: Vec<u32>,
    marker: Pos2,
    coastline_start: usize,
    coastline_len: usize,
}

struct ContinentGeometry {
    outline: Vec<Pos2>,
    triangles: Vec<u32>,
    label: Pos2,
}

struct BridgeRoute {
    a: usize,
    b: usize,
    label: char,
    start: Pos2,
    end: Pos2,
    control: Pos2,
    curve_points: Vec<Pos2>,
}

impl Territory {
    fn new(outline: Vec<Pos2>, marker: Pos2, coastline_start: usize, coastline_len: usize) -> Self {
        let triangles = triangulate(&outline);
        Self {
            outline,
            triangles,
            marker,
            coastline_start,
            coastline_len,
        }
    }

    fn coastline(&self) -> &[Pos2] {
        &self.outline[self.coastline_start..self.coastline_start + self.coastline_len]
    }
}

fn triangulate(outline: &[Pos2]) -> Vec<u32> {
    let coords: Vec<f64> = outline
        .iter()
        .flat_map(|p| [f64::from(p.x), f64::from(p.y)])
        .collect();
    earcutr::earcut(&coords, &[], 2)
        .expect("valid simple polygon")
        .into_iter()
        .map(|i| i as u32)
        .collect()
}

fn curved_interior_points(p0: Pos2, p1: Pos2, bend: f32, two_step: bool) -> Vec<Pos2> {
    let delta = p1 - p0;
    let len = delta.length();
    if len < 1e-4 {
        return Vec::new();
    }
    let normal = vec2(-delta.y / len, delta.x / len);
    if two_step {
        let q1 = p0.lerp(p1, 0.35) + normal * (bend * 0.85);
        let q2 = p0.lerp(p1, 0.68) - normal * (bend * 0.45);
        vec![round_pos(q1), round_pos(q2)]
    } else {
        let q = p0.lerp(p1, 0.5) + normal * bend;
        vec![round_pos(q)]
    }
}

fn round_pos(p: Pos2) -> Pos2 {
    pos2((p.x * 100.0).round() / 100.0, (p.y * 100.0).round() / 100.0)
}

fn quad_bezier(p0: Pos2, ctrl: Pos2, p1: Pos2, t: f32) -> Pos2 {
    let u = 1.0 - t;
    pos2(
        u * u * p0.x + 2.0 * u * t * ctrl.x + t * t * p1.x,
        u * u * p0.y + 2.0 * u * t * ctrl.y + t * t * p1.y,
    )
}

fn sample_quad_bezier(p0: Pos2, ctrl: Pos2, p1: Pos2, segments: usize) -> Vec<Pos2> {
    (0..=segments)
        .map(|i| quad_bezier(p0, ctrl, p1, i as f32 / segments as f32))
        .collect()
}

fn offset_control_point(p0: Pos2, p1: Pos2, offset: f32) -> Pos2 {
    let mid = p0.lerp(p1, 0.5);
    let d = p1 - p0;
    let len = d.length().max(1.0);
    let normal = vec2(-d.y / len, d.x / len);
    mid + normal * offset
}

impl Map {
    pub(super) fn new() -> Self {
        let mut territories = Vec::with_capacity(N);
        let mut continents = Vec::with_capacity(6);

        for continent in &COASTLINES {
            let point = |(x, y)| pos2(x, y);
            let center = point(continent.center);
            let inner: [Pos2; 6] = std::array::from_fn(|i| {
                round_pos(center.lerp(point(continent.sectors[i][0]), continent.inner_ratios[i]))
            });
            let spokes: [Vec<Pos2>; 6] = std::array::from_fn(|i| {
                curved_interior_points(
                    inner[i],
                    point(continent.sectors[i][0]),
                    continent.spoke_bend[i],
                    true,
                )
            });
            let hub_edges: [Vec<Pos2>; 6] = std::array::from_fn(|i| {
                curved_interior_points(inner[i], inner[(i + 1) % 6], continent.hub_bend[i], false)
            });

            // Full outer continent perimeter (for bathymetry shelf, drop shadow & control halo).
            let mut continent_outline = Vec::new();
            for sector in continent.sectors {
                continent_outline.extend(sector[..sector.len() - 1].iter().copied().map(point));
            }
            let continent_triangles = triangulate(&continent_outline);
            continents.push(ContinentGeometry {
                outline: continent_outline,
                triangles: continent_triangles,
                label: point(continent.label),
            });

            // 6 outer coastal territories.
            for (i, coast) in continent.sectors.iter().enumerate() {
                let next = (i + 1) % 6;
                let mut outline = Vec::with_capacity(
                    1 + spokes[i].len() + coast.len() + spokes[next].len() + 1 + hub_edges[i].len(),
                );
                outline.push(inner[i]);
                outline.extend(spokes[i].iter().copied());
                let coastline_start = outline.len();
                outline.extend(coast.iter().copied().map(point));
                outline.extend(spokes[next].iter().rev().copied());
                outline.push(inner[next]);
                outline.extend(hub_edges[i].iter().rev().copied());

                territories.push(Territory::new(
                    outline,
                    point(continent.markers[i]),
                    coastline_start,
                    coast.len(),
                ));
            }

            // 7th territory: inner continental hub sharing exact reversed hub_edges with all 6 sectors.
            let mut hub_outline = Vec::with_capacity(12);
            for i in 0..6 {
                hub_outline.push(inner[i]);
                hub_outline.extend(hub_edges[i].iter().copied());
            }
            territories.push(Territory::new(
                hub_outline,
                point(continent.hub_marker),
                0,
                0,
            ));
        }

        let territories: [Territory; N] = territories.try_into().ok().expect("42 territories");
        let continents: [ContinentGeometry; 6] = continents.try_into().ok().expect("6 continents");

        // Precompute curved nautical routes for all 9 sea bridges.
        let bridges: [BridgeRoute; 9] = std::array::from_fn(|index| {
            let (a, b) = BRIDGES[index];
            let coast_a = territories[a].coastline();
            let coast_b = territories[b].coastline();
            let [start, end] = coast_a
                .iter()
                .flat_map(|&from| coast_b.iter().map(move |&to| [from, to]))
                .min_by(|u, v| u[0].distance_sq(u[1]).total_cmp(&v[0].distance_sq(v[1])))
                .expect("both bridge endpoints have coastlines");
            let control = offset_control_point(start, end, BRIDGE_CURVES[index]);
            let curve_points = sample_quad_bezier(start, control, end, 24);
            BridgeRoute {
                a,
                b,
                label: char::from(b'A' + index as u8),
                start,
                end,
                control,
                curve_points,
            }
        });

        Self {
            territories,
            continents,
            bridges,
        }
    }

    pub(super) fn draw(&self, state: &MapOverlayState<'_>, ui: &mut egui::Ui) -> MapInteraction {
        let scale = (ui.available_width() / SIZE.x)
            .min(ui.available_height() / SIZE.y)
            .clamp(0.45, 2.0);
        let (response, painter) = ui.allocate_painter(SIZE * scale, egui::Sense::click());
        let rect = response.rect;
        let project = |p: Pos2| rect.min + p.to_vec2() * scale;

        let hovered = response.hover_pos().and_then(|p| {
            let board_pt = ((p - rect.min) / scale).to_pos2();
            self.territories
                .iter()
                .position(|territory| contains(&territory.outline, board_pt))
        });

        let clicked = if response.clicked() { hovered } else { None };
        let right_clicked = if response.secondary_clicked() {
            hovered
        } else {
            None
        };

        // Pointer cursor feedback when hovering clickable territories during a human turn.
        if let Some(i) = hovered {
            let interactive = state.human_turn
                && (state.selectable[i] || state.valid_targets[i] || state.selected == Some(i));
            if interactive {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }
        }

        let mut needs_repaint = false;
        let pulse = ((state.now * 4.2).sin() * 0.5 + 0.5) as f32;

        // 1. Deep Maritime Canvas, Graticule & Compass Rose
        draw_ocean_backdrop(&painter, rect, project);
        draw_compass_rose(&painter, project(pos2(86.0, 586.0)), 34.0 * scale, scale);
        draw_ocean_labels(&painter, project, scale);

        // 2. Always-Visible Curved Sea Lanes (BRIDGES)
        let battle_edge = state.active_battle.and_then(|b| {
            let age = (state.now - b.timestamp) as f32;
            (0.0..BATTLE_DURATION)
                .contains(&age)
                .then_some((b.source, b.target))
        });
        for bridge in &self.bridges {
            let active = hovered == Some(bridge.a)
                || hovered == Some(bridge.b)
                || state.selected == Some(bridge.a)
                || state.selected == Some(bridge.b)
                || edge_matches(state.highlighted_edge, bridge.a, bridge.b)
                || edge_matches(battle_edge, bridge.a, bridge.b);
            if active {
                needs_repaint = true;
            }
            draw_bridge_route(&painter, bridge, project, scale, active, state.now);
        }

        // 3. Continent Coastal Shelf (Bathymetry Halo) & Drop Shadow
        for continent in &self.continents {
            draw_continent_shelf_and_shadow(&painter, continent, project, scale);
        }

        // 4. Territory Polygons (with subtle cartographic vertex shading & crisp borders)
        let hovered_connected =
            |i: usize| hovered.is_some_and(|origin| EDGES.binary_search(&(origin, i)).is_ok());

        for (i, territory) in self.territories.iter().enumerate() {
            let owner = state.game.owner[i];
            let base = player_color(owner);
            let is_hovered = hovered == Some(i);
            let is_selected = state.selected == Some(i);
            let is_target = state.valid_targets[i];
            let is_neighbor = hovered_connected(i);

            // Subtle deterministic tone step per sector so adjacent same-owner territories stay legible.
            let sector_tint: f32 = match i % 7 {
                6 => -0.04,
                0 | 2 | 4 => 0.025,
                _ => -0.015,
            };

            let fill_blend: f32 = if is_selected {
                0.05
            } else if is_hovered {
                0.10
            } else if is_target {
                0.15
            } else if is_neighbor {
                0.19
            } else {
                (0.30_f32 + sector_tint).clamp(0.18, 0.42)
            };

            let fill_core = if is_selected {
                base.lerp_to_gamma(Color32::WHITE, 0.14)
            } else {
                base.lerp_to_gamma(LAND_SLATE, fill_blend)
            };
            let fill_edge = fill_core.lerp_to_gamma(LAND_SLATE, 0.14);

            // Vertex-shaded mesh: slight coastal/edge depth toward the perimeter.
            let mut mesh = egui::Mesh::default();
            mesh.indices.extend(&territory.triangles);
            for &point in &territory.outline {
                let dist = point.distance(territory.marker);
                let edge_t = (dist / 95.0).clamp(0.0, 1.0);
                let vcolor = fill_core.lerp_to_gamma(fill_edge, edge_t * 0.75);
                mesh.colored_vertex(project(point), vcolor);
            }
            painter.add(egui::Shape::mesh(mesh));

            // Crisp internal province border.
            stroke_closed_ring(
                &painter,
                &territory.outline,
                project,
                Stroke::new(1.5 * scale, BORDER_INTERNAL),
            );
        }

        // Crisp coastal shoreline rim + Continent Control Halo
        for (region_idx, continent) in self.continents.iter().enumerate() {
            stroke_closed_ring(
                &painter,
                &continent.outline,
                project,
                Stroke::new(1.7 * scale, COAST_RIM),
            );
            if let Some(controller) = continent_owner(state.game, region_idx) {
                let c = player_color(controller);
                stroke_closed_ring(
                    &painter,
                    &continent.outline,
                    project,
                    Stroke::new(6.0 * scale, with_alpha(c, 58)),
                );
                stroke_closed_ring(
                    &painter,
                    &continent.outline,
                    project,
                    Stroke::new(2.2 * scale, with_alpha(c, 215)),
                );
            }
        }

        // Re-draw port medallions on top of continent coastlines so they sit cleanly on the shore.
        for bridge in &self.bridges {
            let active = hovered == Some(bridge.a)
                || hovered == Some(bridge.b)
                || state.selected == Some(bridge.a)
                || state.selected == Some(bridge.b)
                || edge_matches(state.highlighted_edge, bridge.a, bridge.b)
                || edge_matches(battle_edge, bridge.a, bridge.b);
            draw_bridge_ports(&painter, bridge, project, scale, active);
        }

        // 5. Interactive & Tactical Territory Outlines
        let any_selected = state.selected.is_some();
        for (i, territory) in self.territories.iter().enumerate() {
            if state.changed[i] {
                stroke_closed_ring(
                    &painter,
                    &territory.outline,
                    project,
                    Stroke::new(2.2 * scale, GOLD_CHANGED),
                );
            }

            if state.human_turn && state.selectable[i] && !any_selected {
                needs_repaint = true;
                let alpha = (95.0 + pulse * 105.0) as u8;
                stroke_closed_ring(
                    &painter,
                    &territory.outline,
                    project,
                    Stroke::new(
                        2.1 * scale,
                        Color32::from_rgba_unmultiplied(245, 240, 210, alpha),
                    ),
                );
            }

            if state.valid_targets[i] {
                needs_repaint = true;
                let accent = if state.targeting_attack {
                    ATTACK_GLOW
                } else {
                    FORTIFY_GLOW
                };
                let glow_alpha = (55.0 + pulse * 75.0) as u8;
                let rim_alpha = (185.0 + pulse * 70.0) as u8;
                stroke_closed_ring(
                    &painter,
                    &territory.outline,
                    project,
                    Stroke::new(5.5 * scale, with_alpha(accent, glow_alpha)),
                );
                stroke_closed_ring(
                    &painter,
                    &territory.outline,
                    project,
                    Stroke::new(2.4 * scale, with_alpha(accent, rim_alpha)),
                );
            }
        }

        // Log-hovered edge highlight
        if let Some((src, dst)) = state.highlighted_edge
            && src < N
            && dst < N
        {
            stroke_closed_ring(
                &painter,
                &self.territories[src].outline,
                project,
                Stroke::new(2.6 * scale, GOLD_CHANGED),
            );
            stroke_closed_ring(
                &painter,
                &self.territories[dst].outline,
                project,
                Stroke::new(2.6 * scale, ATTACK_GLOW),
            );
        }

        // Hovered territory crisp rim
        if let Some(i) = hovered
            && state.selected != Some(i)
        {
            stroke_closed_ring(
                &painter,
                &self.territories[i].outline,
                project,
                Stroke::new(
                    2.2 * scale,
                    Color32::from_rgba_unmultiplied(255, 255, 255, 215),
                ),
            );
        }

        // Selected territory double-ring glow
        if let Some(sel) = state.selected
            && sel < N
        {
            stroke_closed_ring(
                &painter,
                &self.territories[sel].outline,
                project,
                Stroke::new(5.2 * scale, with_alpha(GOLD_CHANGED, 115)),
            );
            stroke_closed_ring(
                &painter,
                &self.territories[sel].outline,
                project,
                Stroke::new(2.6 * scale, Color32::WHITE),
            );
        }

        // 6. Crisp Troop Medallions & Readable Territory ID Tags
        for (i, territory) in self.territories.iter().enumerate() {
            let owner = state.game.owner[i];
            let owner_color = player_color(owner);
            let pt = project(territory.marker);

            // Compact readable `#id` pill above the troop medallion.
            let id_text = format!("#{i}");
            let id_center = pt - vec2(0.0, 17.0 * scale);
            let id_width = (22.0_f32).max(id_text.len() as f32 * 6.5 + 8.0) * scale;
            let id_rect = Rect::from_center_size(id_center, vec2(id_width, 13.5 * scale));
            painter.rect_filled(
                id_rect,
                5.0 * scale,
                Color32::from_rgba_unmultiplied(10, 19, 30, 188),
            );
            painter.text(
                id_center,
                Align2::CENTER_CENTER,
                id_text,
                FontId::monospace(9.5 * scale),
                Color32::from_rgb(208, 224, 238),
            );

            // Troop medallion pill.
            let troops = state.game.troops[i].to_string();
            let pill_w = (28.0_f32).max(troops.len() as f32 * 8.2 + 14.0) * scale;
            let pill_h = 22.0 * scale;
            let pill_center = pt + vec2(0.0, 3.0 * scale);
            let pill_rect = Rect::from_center_size(pill_center, vec2(pill_w, pill_h));
            let radius = pill_h * 0.48;

            // Subtle pill shadow.
            painter.rect_filled(
                pill_rect.translate(vec2(1.0, 1.8) * scale),
                radius,
                Color32::from_rgba_unmultiplied(5, 10, 17, 150),
            );

            let (rim_color, rim_width) = if state.selected == Some(i) {
                (Color32::WHITE, 2.5 * scale)
            } else if state.valid_targets[i] {
                let c = if state.targeting_attack {
                    ATTACK_GLOW
                } else {
                    FORTIFY_GLOW
                };
                (c, 2.2 * scale)
            } else if hovered == Some(i) {
                (Color32::from_rgb(236, 246, 255), 2.1 * scale)
            } else {
                (owner_color, 1.8 * scale)
            };

            painter.rect(
                pill_rect,
                radius,
                INK,
                Stroke::new(rim_width, rim_color),
                StrokeKind::Inside,
            );
            painter.text(
                pill_center,
                Align2::CENTER_CENTER,
                troops,
                FontId::monospace(13.0 * scale),
                Color32::WHITE,
            );
        }

        // 7. Tactical Arrows: Hover Targeting, Log Hover Edge, & Animated Active Battle
        if let (Some(src), Some(dst)) = (state.selected, hovered)
            && src < N
            && dst < N
            && state.valid_targets[dst]
        {
            let arrow_color = if state.targeting_attack {
                ATTACK_GLOW
            } else {
                FORTIFY_GLOW
            };
            draw_tactical_arrow(
                &painter,
                project(self.territories[src].marker),
                project(self.territories[dst].marker),
                arrow_color,
                3.2 * scale,
                scale,
                1.0,
                255,
            );
        }

        if let Some((src, dst)) = state.highlighted_edge
            && src < N
            && dst < N
        {
            draw_tactical_arrow(
                &painter,
                project(self.territories[src].marker),
                project(self.territories[dst].marker),
                GOLD_CHANGED,
                3.0 * scale,
                scale,
                1.0,
                245,
            );
        }

        if let Some(battle) = state.active_battle {
            let age = (state.now - battle.timestamp) as f32;
            if (0.0..BATTLE_DURATION).contains(&age) && battle.source < N && battle.target < N {
                needs_repaint = true;
                let progress = (age / 0.24_f32).clamp(0.0, 1.0);
                let fade = (1.0_f32 - (age / BATTLE_DURATION).powf(2.2)).clamp(0.0, 1.0);
                let alpha = (fade * 255.0) as u8;
                let attacker_color = player_color(battle.attacker);
                let src_pt = project(self.territories[battle.source].marker);
                let dst_pt = project(self.territories[battle.target].marker);

                draw_tactical_arrow(
                    &painter,
                    src_pt,
                    dst_pt,
                    attacker_color,
                    3.8 * scale,
                    scale,
                    progress,
                    alpha,
                );

                // Expanding impact ring at target.
                let ring_t = (age / 0.85).clamp(0.0, 1.0);
                let ring_radius = (14.0 + ring_t * 28.0) * scale;
                let ring_alpha = ((1.0 - ring_t) * fade * 230.0) as u8;
                painter.circle_stroke(
                    dst_pt,
                    ring_radius,
                    Stroke::new(2.8 * scale, with_alpha(attacker_color, ring_alpha)),
                );

                // Golden starburst ring when territory is captured!
                if battle.captured {
                    draw_capture_starburst(&painter, dst_pt, age, scale);
                }
            }
        }

        // 8. Continent Banner Badges (with `+2` bonus pill & controller highlight)
        for (region_idx, continent) in self.continents.iter().enumerate() {
            let controller = continent_owner(state.game, region_idx);
            draw_continent_banner(
                &painter,
                project(continent.label),
                CONTINENT_NAMES[region_idx],
                controller,
                scale,
            );
        }

        // 9. Floating Troop Delta Popups
        for popup in state.popups {
            if popup.territory >= N {
                continue;
            }
            let age = (state.now - popup.timestamp) as f32;
            if (0.0..POPUP_DURATION).contains(&age) {
                needs_repaint = true;
                let t = age / POPUP_DURATION;
                let alpha = ((1.0_f32 - t).powf(1.4) * 255.0) as u8;
                let anchor = project(self.territories[popup.territory].marker)
                    - vec2(0.0, (24.0 + t * 34.0) * scale);
                let font = FontId::monospace(13.5 * scale);
                painter.text(
                    anchor + vec2(1.0, 1.5) * scale,
                    Align2::CENTER_CENTER,
                    &popup.text,
                    font.clone(),
                    Color32::from_rgba_unmultiplied(6, 12, 20, alpha),
                );
                painter.text(
                    anchor,
                    Align2::CENTER_CENTER,
                    &popup.text,
                    font,
                    with_alpha(popup.color, alpha),
                );
            }
        }

        // 10. Rich Hover Tooltip
        if let Some(i) = hovered {
            response.on_hover_ui_at_pointer(|ui| {
                draw_territory_tooltip(ui, state, i);
            });
        }

        if needs_repaint {
            ui.ctx().request_repaint();
        }

        MapInteraction {
            hovered,
            clicked,
            right_clicked,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn export_board_json(&self) -> serde_json::Value {
        use super::state::PLAYER_COLORS;
        use crate::game::REGIONS;

        let player_colors: Vec<String> = PLAYER_COLORS
            .iter()
            .map(|c| format!("#{:02x}{:02x}{:02x}", c.r(), c.g(), c.b()))
            .collect();

        let continents: Vec<serde_json::Value> = self
            .continents
            .iter()
            .enumerate()
            .map(|(id, c)| {
                let outline: Vec<[f32; 2]> = c.outline.iter().map(|p| [p.x, p.y]).collect();
                serde_json::json!({
                    "id": id,
                    "name": CONTINENT_NAMES[id],
                    "short_name": CONTINENT_SHORT_NAMES[id],
                    "outline": outline,
                    "label": [c.label.x, c.label.y],
                    "territories": REGIONS[id],
                })
            })
            .collect();

        let territories: Vec<serde_json::Value> = self
            .territories
            .iter()
            .enumerate()
            .map(|(id, t)| {
                let outline: Vec<[f32; 2]> = t.outline.iter().map(|p| [p.x, p.y]).collect();
                let neighbors: Vec<usize> = EDGES
                    .iter()
                    .filter_map(|&(a, b)| (a == id).then_some(b))
                    .collect();
                serde_json::json!({
                    "id": id,
                    "name": territory_name(id),
                    "continent": continent_of(id),
                    "outline": outline,
                    "marker": [t.marker.x, t.marker.y],
                    "neighbors": neighbors,
                })
            })
            .collect();

        let bridges: Vec<serde_json::Value> = self
            .bridges
            .iter()
            .enumerate()
            .map(|(index, b)| {
                let points: Vec<[f32; 2]> = b.curve_points.iter().map(|p| [p.x, p.y]).collect();
                serde_json::json!({
                    "index": index,
                    "label": b.label.to_string(),
                    "a": b.a,
                    "b": b.b,
                    "start": [b.start.x, b.start.y],
                    "end": [b.end.x, b.end.y],
                    "control": [b.control.x, b.control.y],
                    "points": points,
                })
            })
            .collect();

        serde_json::json!({
            "width": SIZE.x,
            "height": SIZE.y,
            "player_colors": player_colors,
            "continent_bonus": CONTINENT_BONUS,
            "continents": continents,
            "territories": territories,
            "bridges": bridges,
        })
    }
}

fn draw_ocean_backdrop(painter: &egui::Painter, rect: Rect, project: impl Fn(Pos2) -> Pos2) {
    // Rounded ocean frame base.
    painter.rect_filled(rect, 12.0, OCEAN_DEEP);

    // Subtle 3×3 radial depth vignette mesh clipped slightly inside the rounded corner margin.
    let inner = rect.shrink(2.0);
    let xs = [inner.left(), inner.center().x, inner.right()];
    let ys = [inner.top(), inner.center().y, inner.bottom()];
    let colors = [
        [OCEAN_DEEP, OCEAN_MID, OCEAN_DEEP],
        [OCEAN_MID, OCEAN_CENTER, OCEAN_MID],
        [OCEAN_DEEP, OCEAN_MID, OCEAN_DEEP],
    ];
    let mut mesh = egui::Mesh::default();
    for (row, &y) in ys.iter().enumerate() {
        for (col, &x) in xs.iter().enumerate() {
            mesh.colored_vertex(pos2(x, y), colors[row][col]);
        }
    }
    for row in 0..2 {
        for col in 0..2 {
            let tl = (row * 3 + col) as u32;
            let tr = tl + 1;
            let bl = tl + 3;
            let br = bl + 1;
            mesh.indices.extend_from_slice(&[tl, tr, br, tl, br, bl]);
        }
    }
    painter.add(egui::Shape::mesh(mesh));

    // Faint cartographic graticule lines.
    let grid_fine = Stroke::new(1.0, Color32::from_rgba_unmultiplied(115, 175, 210, 16));
    let grid_major = Stroke::new(1.0, Color32::from_rgba_unmultiplied(135, 198, 232, 28));

    for x in (160..1280).step_by(160) {
        let stroke = if x == 640 { grid_major } else { grid_fine };
        painter.line_segment(
            [
                project(pos2(x as f32, 12.0)),
                project(pos2(x as f32, SIZE.y - 12.0)),
            ],
            stroke,
        );
    }
    for y in (136..680).step_by(136) {
        let stroke = if y == 340 { grid_major } else { grid_fine };
        painter.line_segment(
            [
                project(pos2(12.0, y as f32)),
                project(pos2(SIZE.x - 12.0, y as f32)),
            ],
            stroke,
        );
    }

    // Subtle inner chart frame border with corner ticks.
    painter.rect_stroke(
        rect.shrink(1.0),
        12.0,
        Stroke::new(1.2, Color32::from_rgba_unmultiplied(84, 134, 168, 90)),
        StrokeKind::Inside,
    );
    let tick_stroke = Stroke::new(1.0, Color32::from_rgba_unmultiplied(130, 185, 218, 45));
    for x in (80..1280).step_by(80) {
        let xf = x as f32;
        painter.line_segment(
            [project(pos2(xf, 2.0)), project(pos2(xf, 7.0))],
            tick_stroke,
        );
        painter.line_segment(
            [
                project(pos2(xf, SIZE.y - 7.0)),
                project(pos2(xf, SIZE.y - 2.0)),
            ],
            tick_stroke,
        );
    }
}

fn draw_compass_rose(painter: &egui::Painter, center: Pos2, radius: f32, scale: f32) {
    let ring_color = Color32::from_rgba_unmultiplied(136, 192, 224, 52);
    let light_facet = Color32::from_rgba_unmultiplied(186, 226, 248, 125);
    let dark_facet = Color32::from_rgba_unmultiplied(64, 108, 138, 115);
    let minor_color = Color32::from_rgba_unmultiplied(128, 180, 210, 75);

    painter.circle_stroke(center, radius * 0.86, Stroke::new(1.0 * scale, ring_color));
    painter.circle_stroke(center, radius * 0.68, Stroke::new(0.8 * scale, ring_color));

    // 4 minor diagonal points.
    let minor_len = radius * 0.62;
    let minor_w = radius * 0.13;
    for k in 0..4 {
        let angle = std::f32::consts::FRAC_PI_4 + k as f32 * std::f32::consts::FRAC_PI_2;
        let dir = vec2(angle.cos(), angle.sin());
        let perp = vec2(-dir.y, dir.x);
        let poly = vec![
            center + dir * minor_len,
            center + perp * minor_w,
            center,
            center - perp * minor_w,
        ];
        painter.add(egui::Shape::convex_polygon(poly, minor_color, Stroke::NONE));
    }

    // 4 major cardinal points with split light/dark cartographic facets.
    let major_len = radius;
    let major_w = radius * 0.20;
    for k in 0..4 {
        let angle = -std::f32::consts::FRAC_PI_2 + k as f32 * std::f32::consts::FRAC_PI_2;
        let dir = vec2(angle.cos(), angle.sin());
        let perp = vec2(-dir.y, dir.x);
        let tip = center + dir * major_len;
        painter.add(egui::Shape::convex_polygon(
            vec![center, tip, center + perp * major_w],
            light_facet,
            Stroke::NONE,
        ));
        painter.add(egui::Shape::convex_polygon(
            vec![center, tip, center - perp * major_w],
            dark_facet,
            Stroke::NONE,
        ));
    }

    painter.circle_filled(center, 2.5 * scale, light_facet);

    // Cardinal letters.
    let label_offset = radius + 9.0 * scale;
    let label_font = FontId::monospace(9.5 * scale);
    let label_color = Color32::from_rgba_unmultiplied(170, 212, 236, 130);
    for (text, offset) in [
        ("N", vec2(0.0, -label_offset)),
        ("E", vec2(label_offset, 0.0)),
        ("S", vec2(0.0, label_offset)),
        ("W", vec2(-label_offset, 0.0)),
    ] {
        painter.text(
            center + offset,
            Align2::CENTER_CENTER,
            text,
            label_font.clone(),
            label_color,
        );
    }
}

fn draw_ocean_labels(painter: &egui::Painter, project: impl Fn(Pos2) -> Pos2, scale: f32) {
    let color = Color32::from_rgba_unmultiplied(128, 178, 208, 72);
    let font = FontId::monospace(11.5 * scale);
    for (label, pos) in [
        ("P A C I F I C   O C E A N", pos2(88.0, 354.0)),
        ("A T L A N T I C   O C E A N", pos2(436.0, 578.0)),
        ("I N D I A N   O C E A N", pos2(822.0, 580.0)),
    ] {
        painter.text(
            project(pos),
            Align2::CENTER_CENTER,
            label,
            font.clone(),
            color,
        );
    }
}

fn draw_bridge_route(
    painter: &egui::Painter,
    bridge: &BridgeRoute,
    project: impl Fn(Pos2) -> Pos2,
    scale: f32,
    active: bool,
    now: f64,
) {
    let pts: Vec<Pos2> = bridge.curve_points.iter().copied().map(&project).collect();
    if active {
        // Soft glowing wake underneath.
        painter.add(egui::Shape::line(
            pts.clone(),
            Stroke::new(5.0 * scale, with_alpha(ROUTE_ACTIVE, 45)),
        ));
        // Animated marching dashed route.
        let dash = 8.0 * scale;
        let gap = 5.5 * scale;
        let phase = ((now * 28.0) as f32 * scale) % (dash + gap);
        draw_marching_dashed_polyline(
            painter,
            &pts,
            Stroke::new(2.0 * scale, ROUTE_ACTIVE),
            dash,
            gap,
            phase,
        );
    } else {
        let dash = 6.0 * scale;
        let gap = 5.5 * scale;
        draw_marching_dashed_polyline(
            painter,
            &pts,
            Stroke::new(1.35 * scale, with_alpha(ROUTE_IDLE, 105)),
            dash,
            gap,
            0.0,
        );
    }
}

fn draw_bridge_ports(
    painter: &egui::Painter,
    bridge: &BridgeRoute,
    project: impl Fn(Pos2) -> Pos2,
    scale: f32,
    active: bool,
) {
    let rim = if active {
        ROUTE_ACTIVE
    } else {
        with_alpha(ROUTE_IDLE, 185)
    };
    let text_color = if active { Color32::WHITE } else { ROUTE_IDLE };
    let r = (if active { 8.0 } else { 7.0 }) * scale;
    let letter = bridge.label.to_string();

    for (endpoint, toward) in [(bridge.start, bridge.control), (bridge.end, bridge.control)] {
        let dir = (toward - endpoint).normalized();
        let port = project(endpoint + dir * 2.5);
        if active {
            painter.circle_filled(port, r + 3.0 * scale, with_alpha(ROUTE_ACTIVE, 50));
        }
        painter.circle_filled(port, r, INK);
        painter.circle_stroke(port, r, Stroke::new(1.25 * scale, rim));
        painter.text(
            port,
            Align2::CENTER_CENTER,
            &letter,
            FontId::monospace(8.5 * scale),
            text_color,
        );
    }
}

fn draw_continent_shelf_and_shadow(
    painter: &egui::Painter,
    continent: &ContinentGeometry,
    project: impl Fn(Pos2) -> Pos2,
    scale: f32,
) {
    // 1. Shallow coastal bathymetry glow (two soft concentric rings).
    stroke_closed_ring(
        painter,
        &continent.outline,
        &project,
        Stroke::new(
            13.0 * scale,
            Color32::from_rgba_unmultiplied(44, 118, 152, 24),
        ),
    );
    stroke_closed_ring(
        painter,
        &continent.outline,
        &project,
        Stroke::new(
            6.0 * scale,
            Color32::from_rgba_unmultiplied(72, 154, 186, 44),
        ),
    );

    // 2. Crisp elevated landmass drop shadow.
    let offset = vec2(4.0, 5.5) * scale;
    let shadow_color = Color32::from_rgba_unmultiplied(4, 10, 18, 128);
    let mut mesh = egui::Mesh::default();
    mesh.indices.extend(&continent.triangles);
    for &p in &continent.outline {
        mesh.colored_vertex(project(p) + offset, shadow_color);
    }
    painter.add(egui::Shape::mesh(mesh));
}

fn draw_continent_banner(
    painter: &egui::Painter,
    center: Pos2,
    name: &str,
    controller: Option<usize>,
    scale: f32,
) {
    let name_font = FontId::monospace(10.5 * scale);
    let bonus_font = FontId::monospace(10.0 * scale);
    let bonus_text = format!("+{CONTINENT_BONUS}");

    let name_w = name.len() as f32 * 6.5 * scale;
    let bonus_w = 24.0 * scale;
    let pad_x = 10.0 * scale;
    let gap = 7.0 * scale;
    let total_w = pad_x * 2.0 + name_w + gap + bonus_w;
    let h = 21.0 * scale;
    let rect = Rect::from_center_size(center, vec2(total_w, h));
    let radius = h * 0.5;

    let (bg, border, text_color, bonus_bg, bonus_fg) = if let Some(seat) = controller {
        let accent = player_color(seat);
        (
            accent.lerp_to_gamma(INK, 0.80),
            Stroke::new(1.5 * scale, with_alpha(accent, 230)),
            Color32::WHITE,
            accent,
            INK,
        )
    } else {
        (
            Color32::from_rgba_unmultiplied(13, 25, 39, 218),
            Stroke::new(
                1.0 * scale,
                Color32::from_rgba_unmultiplied(74, 114, 142, 145),
            ),
            Color32::from_rgb(188, 214, 232),
            Color32::from_rgba_unmultiplied(36, 62, 84, 220),
            Color32::from_rgb(168, 212, 236),
        )
    };

    if let Some(seat) = controller {
        painter.rect_stroke(
            rect.expand(1.5 * scale),
            radius + 1.5 * scale,
            Stroke::new(3.5 * scale, with_alpha(player_color(seat), 55)),
            StrokeKind::Outside,
        );
    }

    painter.rect(rect, radius, bg, border, StrokeKind::Inside);

    let name_center = pos2(rect.left() + pad_x + name_w * 0.5, rect.center().y);
    painter.text(
        name_center,
        Align2::CENTER_CENTER,
        name,
        name_font,
        text_color,
    );

    let bonus_rect = Rect::from_center_size(
        pos2(
            rect.right() - pad_x - bonus_w * 0.5 + 2.0 * scale,
            rect.center().y,
        ),
        vec2(bonus_w, 15.0 * scale),
    );
    painter.rect_filled(bonus_rect, 7.5 * scale, bonus_bg);
    painter.text(
        bonus_rect.center(),
        Align2::CENTER_CENTER,
        bonus_text,
        bonus_font,
        bonus_fg,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_tactical_arrow(
    painter: &egui::Painter,
    from: Pos2,
    to: Pos2,
    color: Color32,
    width: f32,
    scale: f32,
    progress: f32,
    alpha: u8,
) {
    let delta = to - from;
    let dist = delta.length();
    if dist < 8.0 || progress <= 0.02 {
        return;
    }

    // Offset start/end slightly so arrows emerge cleanly from pill edges instead of covering numbers.
    let dir = delta / dist;
    let start = from + dir * (16.0 * scale).min(dist * 0.25);
    let end = to - dir * (18.0 * scale).min(dist * 0.25);
    let curve_offset = (dist * 0.14).clamp(14.0 * scale, 48.0 * scale);
    let ctrl = offset_control_point(start, end, -curve_offset);

    let steps = 20usize;
    let max_t = progress.clamp(0.05, 1.0);
    let pts: Vec<Pos2> = (0..=steps)
        .map(|k| quad_bezier(start, ctrl, end, max_t * (k as f32 / steps as f32)))
        .collect();

    // Soft dark under-stroke + colored glow + crisp shaft.
    painter.add(egui::Shape::line(
        pts.clone(),
        Stroke::new(
            width + 3.5 * scale,
            Color32::from_rgba_unmultiplied(8, 14, 22, alpha / 2),
        ),
    ));
    painter.add(egui::Shape::line(
        pts.clone(),
        Stroke::new(width + 2.0 * scale, with_alpha(color, alpha / 3)),
    ));
    painter.add(egui::Shape::line(
        pts.clone(),
        Stroke::new(width, with_alpha(color, alpha)),
    ));

    // Crisp triangular arrowhead aligned with the final tangent.
    if pts.len() >= 2 {
        let tip = *pts.last().unwrap();
        let prev = pts[pts.len() - 2];
        let tangent = (tip - prev).normalized();
        let perp = vec2(-tangent.y, tangent.x);
        let head_len = 11.5 * scale;
        let head_w = 5.5 * scale;
        let base = tip - tangent * head_len;
        let head_poly = vec![
            tip + tangent * (2.0 * scale),
            base + perp * head_w,
            base - perp * head_w,
        ];
        painter.add(egui::Shape::convex_polygon(
            head_poly,
            with_alpha(color, alpha),
            Stroke::new(
                1.0 * scale,
                Color32::from_rgba_unmultiplied(10, 18, 28, alpha),
            ),
        ));
    }
}

fn draw_capture_starburst(painter: &egui::Painter, center: Pos2, age: f32, scale: f32) {
    let t = (age / 1.1_f32).clamp(0.0, 1.0);
    if t >= 1.0 {
        return;
    }
    let alpha = ((1.0_f32 - t).powf(1.5) * 245.0) as u8;
    let gold = with_alpha(Color32::from_rgb(255, 220, 102), alpha);
    let inner_r = (14.0 + t * 22.0) * scale;
    let outer_r = (24.0 + t * 34.0) * scale;
    let rays = 10;
    for k in 0..rays {
        let angle = (k as f32 / rays as f32) * std::f32::consts::TAU + t * 0.4;
        let dir = vec2(angle.cos(), angle.sin());
        painter.line_segment(
            [center + dir * inner_r, center + dir * outer_r],
            Stroke::new(2.2 * scale, gold),
        );
    }
    painter.circle_stroke(center, inner_r, Stroke::new(2.0 * scale, gold));
}

fn draw_territory_tooltip(ui: &mut egui::Ui, state: &MapOverlayState<'_>, i: usize) {
    let owner = state.game.owner[i];
    let owner_label = state
        .seat_labels
        .get(owner)
        .map(String::as_str)
        .unwrap_or("Unknown");
    let continent_idx = continent_of(i);
    let continent_name = CONTINENT_SHORT_NAMES[continent_idx];
    let troops = state.game.troops[i];
    let troop_word = if troops == 1 { "troop" } else { "troops" };

    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("{} (#{i})", territory_name(i)))
                .strong()
                .size(13.5),
        );
        ui.label(RichText::new(format!("· {continent_name}")).weak());
    });
    ui.horizontal(|ui| {
        ui.colored_label(player_color(owner), RichText::new(owner_label).strong());
        ui.label(format!("· {troops} {troop_word}"));
    });

    // Connected neighbors, clearly annotating sea crossings.
    let neighbors: Vec<String> = EDGES
        .iter()
        .filter(|&&(a, _)| a == i)
        .map(|&(_, b)| {
            let is_sea = BRIDGES.contains(&(i, b)) || BRIDGES.contains(&(b, i));
            if is_sea {
                format!("{} (sea)", territory_name(b))
            } else {
                territory_name(b).to_string()
            }
        })
        .collect();
    ui.label(
        RichText::new(format!("Borders: {}", neighbors.join(", ")))
            .small()
            .weak(),
    );

    // Contextual action hint during human turns.
    if state.human_turn {
        let hint = if state.game.phase == Phase::Reinforce && state.selectable[i] {
            Some(("Click to place 1 troop", Color32::from_rgb(245, 224, 130)))
        } else if state.selected == Some(i) {
            Some((
                "Selected · Click a highlighted target (right-click to cancel)",
                Color32::WHITE,
            ))
        } else if let Some(src) = state.selected {
            if state.valid_targets[i] {
                let verb = if state.targeting_attack {
                    "attack"
                } else {
                    "fortify"
                };
                let color = if state.targeting_attack {
                    ATTACK_GLOW
                } else {
                    FORTIFY_GLOW
                };
                ui.add_space(2.0);
                ui.colored_label(
                    color,
                    RichText::new(format!("Click to {verb} from {}", territory_name(src))).strong(),
                );
                None
            } else if state.selectable[i] {
                Some((
                    "Click to switch selection",
                    Color32::from_rgb(205, 225, 240),
                ))
            } else {
                None
            }
        } else if state.selectable[i] {
            match state.game.phase {
                Phase::Attack => Some(("Click to select attacker", ATTACK_GLOW)),
                Phase::Fortify => Some(("Click to select fortify source", FORTIFY_GLOW)),
                _ => None,
            }
        } else {
            None
        };
        if let Some((text, color)) = hint {
            ui.add_space(2.0);
            ui.colored_label(color, RichText::new(text).strong());
        }
    }
}

fn draw_marching_dashed_polyline(
    painter: &egui::Painter,
    points: &[Pos2],
    stroke: Stroke,
    dash_len: f32,
    gap_len: f32,
    phase: f32,
) {
    if points.len() < 2 {
        return;
    }
    let period = dash_len + gap_len;
    if period <= 0.1 {
        return;
    }
    let mut dist = -phase.rem_euclid(period);
    for seg in points.windows(2) {
        let (a, b) = (seg[0], seg[1]);
        let seg_len = a.distance(b);
        if seg_len < 1e-4 {
            continue;
        }
        let mut cursor = 0.0_f32;
        while cursor < seg_len {
            let cycle_pos = (dist + cursor).rem_euclid(period);
            if cycle_pos < dash_len {
                let run = (dash_len - cycle_pos).min(seg_len - cursor);
                let p0 = a.lerp(b, cursor / seg_len);
                let p1 = a.lerp(b, (cursor + run) / seg_len);
                painter.line_segment([p0, p1], stroke);
                cursor += run;
            } else {
                let skip = (period - cycle_pos).min(seg_len - cursor);
                cursor += skip;
            }
        }
        dist += seg_len;
    }
}

fn stroke_closed_ring(
    painter: &egui::Painter,
    outline: &[Pos2],
    project: impl Fn(Pos2) -> Pos2,
    stroke: Stroke,
) {
    let ring: Vec<Pos2> = outline
        .iter()
        .chain(&outline[..1])
        .copied()
        .map(project)
        .collect();
    painter.add(egui::Shape::line(ring, stroke));
}

fn edge_matches(edge: Option<(usize, usize)>, a: usize, b: usize) -> bool {
    matches!(edge, Some((u, v)) if (u == a && v == b) || (u == b && v == a))
}

fn with_alpha(c: Color32, alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), alpha)
}

fn contains(polygon: &[Pos2], point: Pos2) -> bool {
    polygon
        .iter()
        .zip(polygon.iter().cycle().skip(1))
        .filter(|&(a, b)| {
            (a.y > point.y) != (b.y > point.y)
                && point.x < a.x + (point.y - a.y) * (b.x - a.x) / (b.y - a.y)
        })
        .count()
        % 2
        == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signed_area(poly: &[Pos2]) -> f32 {
        let mut sum = 0.0_f32;
        for (a, b) in poly.iter().zip(poly.iter().cycle().skip(1)) {
            sum += a.x * b.y - b.x * a.y;
        }
        0.5 * sum
    }

    fn point_to_segment_dist(p: Pos2, a: Pos2, b: Pos2) -> f32 {
        let ab = b - a;
        let len_sq = ab.length_sq();
        if len_sq < 1e-6 {
            return p.distance(a);
        }
        let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
        p.distance(a + ab * t)
    }

    #[test]
    fn polygons_match_the_simulator_edges() {
        let map = Map::new();
        let segments = |outline: &[Pos2]| {
            outline
                .iter()
                .copied()
                .zip(outline.iter().copied().cycle().skip(1))
                .collect::<Vec<_>>()
        };
        for (a, first) in map.territories.iter().enumerate() {
            assert!(contains(&first.outline, first.marker), "marker {a}");
            assert!(!first.triangles.is_empty());
            for (b, second) in map.territories.iter().enumerate().skip(a + 1) {
                let shared_border = segments(&first.outline)
                    .iter()
                    .any(|&(from, to)| segments(&second.outline).contains(&(to, from)));
                let sea_crossing = BRIDGES.contains(&(a, b)) || BRIDGES.contains(&(b, a));
                assert!(!(shared_border && sea_crossing));
                assert_eq!(
                    shared_border || sea_crossing,
                    EDGES.contains(&(a, b)),
                    "{a}–{b}"
                );
            }
        }
    }

    #[test]
    fn markers_have_comfortable_spacing() {
        let map = Map::new();
        for (i, t) in map.territories.iter().enumerate() {
            let min_edge_dist = t
                .outline
                .iter()
                .zip(t.outline.iter().cycle().skip(1))
                .map(|(&a, &b)| point_to_segment_dist(t.marker, a, b))
                .fold(f32::INFINITY, f32::min);
            assert!(
                min_edge_dist >= 14.0,
                "marker {i} too close to border: {min_edge_dist:.1}px"
            );
            for (j, other) in map.territories.iter().enumerate().skip(i + 1) {
                let d = t.marker.distance(other.marker);
                assert!(d >= 52.0, "markers {i} and {j} too close: {d:.1}px");
            }
        }
    }

    #[test]
    fn all_territories_have_valid_area() {
        let map = Map::new();
        for (i, t) in map.territories.iter().enumerate() {
            let area = signed_area(&t.outline);
            assert!(
                area > 1200.0,
                "territory {i} has unexpectedly small or inverted area: {area:.1}"
            );
        }
    }
}
