//! Polygon board with shared land borders and marked sea crossings.

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Stroke, Vec2, pos2, vec2};

use crate::game::BRIDGES;

use super::{COLORS, EDGES, N, Visualizer};

#[path = "coastlines.rs"]
mod coastlines;
use coastlines::COASTLINES;

const SIZE: Vec2 = vec2(1200.0, 600.0);
const WATER: Color32 = Color32::from_rgb(16, 29, 42);
const INK: Color32 = Color32::from_rgb(12, 23, 34);
const CHANGE: Color32 = Color32::from_rgb(255, 225, 145);
const ROUTE: Color32 = Color32::from_rgb(142, 173, 190);

pub(super) struct Map {
    territories: [Territory; N],
}

struct Territory {
    outline: Vec<Pos2>,
    triangles: Vec<u32>,
    marker: Pos2,
    coastline_len: usize,
}

impl Territory {
    fn new(outline: Vec<Pos2>, marker: Pos2, coastline_len: usize) -> Self {
        let coords: Vec<_> = outline
            .iter()
            .flat_map(|p| [f64::from(p.x), f64::from(p.y)])
            .collect();
        let triangles = earcutr::earcut(&coords, &[], 2)
            .expect("valid territory polygon")
            .into_iter()
            .map(|i| i as u32)
            .collect();
        Self {
            outline,
            triangles,
            marker,
            coastline_len,
        }
    }
}

impl Map {
    pub(super) fn new() -> Self {
        let mut territories = Vec::with_capacity(N);
        for continent in COASTLINES {
            let point = |(x, y)| pos2(x, y);
            let center = point(continent.center);
            let inner: [Pos2; 6] =
                std::array::from_fn(|i| center.lerp(point(continent.sectors[i][0]), 0.33));
            for (i, coast) in continent.sectors.iter().enumerate() {
                let mut outline = Vec::with_capacity(coast.len() + 2);
                outline.push(inner[i]);
                outline.extend(coast.iter().copied().map(point));
                outline.push(inner[(i + 1) % inner.len()]);
                territories.push(Territory::new(
                    outline,
                    point(continent.markers[i]),
                    coast.len(),
                ));
            }
            territories.push(Territory::new(inner.to_vec(), center, 0));
        }
        Self {
            territories: territories.try_into().ok().expect("42 territories"),
        }
    }

    pub(super) fn draw(&self, view: &Visualizer, ui: &mut egui::Ui) {
        let scale = (ui.available_width() / SIZE.x)
            .min(ui.available_height() / SIZE.y)
            .max(0.65);
        let (response, painter) = ui.allocate_painter(SIZE * scale, egui::Sense::hover());
        let rect = response.rect;
        let project = |p: Pos2| rect.min + p.to_vec2() * scale;
        painter.rect_filled(rect, 12.0, WATER);
        let grid = Stroke::new(1.0, Color32::from_rgb(23, 40, 54));
        for x in (100..1200).step_by(100) {
            painter.line_segment(
                [
                    project(pos2(x as f32, 0.0)),
                    project(pos2(x as f32, SIZE.y)),
                ],
                grid,
            );
        }
        for y in (100..600).step_by(100) {
            painter.line_segment(
                [
                    project(pos2(0.0, y as f32)),
                    project(pos2(SIZE.x, y as f32)),
                ],
                grid,
            );
        }

        let hovered = response.hover_pos().and_then(|p| {
            let point = ((p - rect.min) / scale).to_pos2();
            self.territories
                .iter()
                .position(|territory| contains(&territory.outline, point))
        });
        let connected = |i| hovered.is_some_and(|origin| EDGES.binary_search(&(origin, i)).is_ok());

        for (i, territory) in self.territories.iter().enumerate() {
            let color = COLORS[view.game.owner[i]].lerp_to_gamma(
                WATER,
                if hovered == Some(i) || connected(i) {
                    0.12
                } else {
                    0.29
                },
            );
            let mut mesh = egui::Mesh::default();
            mesh.indices.extend(&territory.triangles);
            for &point in &territory.outline {
                mesh.colored_vertex(project(point), color);
            }
            painter.add(egui::Shape::mesh(mesh));
            outline(&painter, territory, project, Stroke::new(1.3, WATER));
        }

        for (index, (a, b)) in BRIDGES.into_iter().enumerate() {
            let first = &self.territories[a].outline[1..][..self.territories[a].coastline_len];
            let second = &self.territories[b].outline[1..][..self.territories[b].coastline_len];
            let points = first
                .iter()
                .flat_map(|&from| second.iter().map(move |&to| [from, to]))
                .min_by(|a, b| a[0].distance_sq(a[1]).total_cmp(&b[0].distance_sq(b[1])))
                .expect("both coastal territories have a shoreline")
                .map(project);
            let active = hovered == Some(a) || hovered == Some(b);
            let color = if active { Color32::WHITE } else { ROUTE };
            if active {
                painter.extend(egui::Shape::dashed_line(
                    &points,
                    Stroke::new(1.5, color),
                    5.0,
                    4.0,
                ));
            }
            for point in points {
                painter.circle_filled(point, 7.5 * scale, INK);
                painter.circle_stroke(point, 7.5 * scale, Stroke::new(1.0, color));
                painter.text(
                    point,
                    Align2::CENTER_CENTER,
                    char::from(b'A' + index as u8).to_string(),
                    FontId::monospace(9.0 * scale),
                    color,
                );
            }
        }

        for (i, territory) in self.territories.iter().enumerate() {
            if view.changed[i] || hovered == Some(i) {
                let color = if view.changed[i] {
                    CHANGE
                } else {
                    Color32::WHITE
                };
                outline(&painter, territory, project, Stroke::new(2.0, color));
            }
            let point = project(territory.marker);
            painter.text(
                point - vec2(0.0, 15.0 * scale),
                Align2::CENTER_CENTER,
                i.to_string(),
                FontId::monospace(12.0 * scale),
                INK,
            );
            let troops = view.game.troops[i].to_string();
            let width = (22.0_f32).max(troops.len() as f32 * 7.5 + 10.0);
            let chip =
                Rect::from_center_size(point + vec2(0.0, 5.0 * scale), vec2(width, 21.0) * scale);
            painter.rect_filled(chip, 7.0 * scale, INK);
            painter.text(
                chip.center(),
                Align2::CENTER_CENTER,
                troops,
                FontId::monospace(13.0 * scale),
                Color32::WHITE,
            );
        }
        for (name, (x, y)) in COASTLINES.iter().map(|coast| coast.label).chain([
            ("PACIFIC OCEAN", (122., 324.)),
            ("ATLANTIC OCEAN", (392., 219.)),
            ("INDIAN OCEAN", (747., 438.)),
        ]) {
            painter.text(
                project(pos2(x, y)),
                Align2::CENTER_CENTER,
                name,
                FontId::monospace(13.0 * scale),
                ROUTE,
            );
        }
        if let Some(i) = hovered {
            let neighbors = EDGES
                .iter()
                .filter(|&&(a, _)| a == i)
                .map(|&(_, b)| b.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            response.on_hover_text(format!(
                "Territory {i} · Player {} · {} troops\nConnected to {neighbors}",
                view.game.owner[i] + 1,
                view.game.troops[i]
            ));
        }
    }
}

fn outline(
    painter: &egui::Painter,
    territory: &Territory,
    project: impl Fn(Pos2) -> Pos2,
    stroke: Stroke,
) {
    let ring = territory
        .outline
        .iter()
        .chain(&territory.outline[..1])
        .copied()
        .map(project)
        .collect();
    painter.add(egui::Shape::line(ring, stroke));
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
}
