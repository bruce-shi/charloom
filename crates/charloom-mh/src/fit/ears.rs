 mod curve;

use crate::geom::{V, a, proper_rotation, v};
use crate::scan::Scan;
use crate::segment::Label;
use charloom_core::Vec3;
use charloom_engine::surface_query::SurfaceQuery;
use nalgebra::{Matrix2, Rotation3, Unit, Vector2};
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use std::sync::OnceLock;

pub struct Patch {
    pub vertices: Vec<usize>,
    /// Collar vertices outside the patch and their blend weight (1 at the
    /// patch, 0 at the collar's outer edge).
    pub collar: Vec<(usize, f64)>,
    /// Skull vertices just outside the collar: the ear's attachment frame.
    pub anchors: Vec<usize>,
    /// Head triangles around the ear (welded indices): MakeHuman's side
    /// view of it.
    pub(super) window: Vec<[u32; 3]>,
    /// Rest outward direction (+X or -X).
    pub(super) outward: V,
    /// Shape model and marked outline (`Ears::attach_outline`).
    shape: Option<curve::Shape>,
}

pub struct Ears {
    pub patches: Vec<Patch>,
    pub member: Vec<bool>,
    /// Collar vertices (the head skin the ear is feathered into).
    pub collar: Vec<bool>,
    unit: f64,
    /// Scan triangles near the ears (corners, skin), found once.
    scan_near: OnceLock<Vec<([V; 3], bool)>>,
}

impl Ears {
    /// `head`: welded vertices fully driven by the head (face, scalp, ears).
    pub fn detect(
        rest: &[V],
        normals: &[Vec3],
        neighbours: &[Vec<u32>],
        triangles: &[[u32; 3]],
        head: &[bool],
        query: &SurfaceQuery,
        height: f64,
    ) -> Self {
        let n = rest.len();
        let ids: Vec<usize> = (0..n).filter(|&i| head[i]).collect();
        let mut member = vec![false; n];
        let unit = height / 1.75;
        if ids.len() < 100 {
            return Self {
                patches: vec![],
                collar: vec![false; n],
                member,
                unit,
                scan_near: OnceLock::new(),
            };
        }
        let cx = ids.iter().map(|&i| rest[i].x).sum::<f64>() / ids.len() as f64;
        let half = ids
            .iter()
            .map(|&i| (rest[i].x - cx).abs())
            .fold(0.0, f64::max);
        let (z0, z1) = ids
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &i| {
                (l.min(rest[i].z), h.max(rest[i].z))
            });
        let thin = |i: usize| {
            let nrm = crate::geom::v(normals[i]);
            let origin = a(rest[i] - nrm * 3e-4);
            query
                .first_ray_distance(origin, a(-nrm), 0.012 * unit)
                .is_some()
        };
        let seed: Vec<bool> = (0..n)
            .map(|i| {
                head[i]
                    && (rest[i].x - cx).abs() > half * 0.55
                    && rest[i].z < z1 - (z1 - z0) * 0.3
                    && thin(i)
            })
            .collect();
        // One ring more (concha floor, the rim's inner row).
        let mut grown = seed.clone();
        for i in 0..n {
            if seed[i] {
                for &j in &neighbours[i] {
                    if head[j as usize] {
                        grown[j as usize] = true;
                    }
                }
            }
        }
        // Largest connected component on each side.
        let mut seen = vec![false; n];
        let mut best: [Vec<usize>; 2] = [vec![], vec![]];
        for s in 0..n {
            if !grown[s] || seen[s] {
                continue;
            }
            let mut comp = vec![s];
            seen[s] = true;
            let mut k = 0;
            while k < comp.len() {
                for &j in &neighbours[comp[k]] {
                    let j = j as usize;
                    if grown[j] && !seen[j] {
                        seen[j] = true;
                        comp.push(j);
                    }
                }
                k += 1;
            }
            let side = usize::from(rest[s].x < cx);
            if comp.len() > best[side].len() {
                best[side] = comp;
            }
        }
        let mut patches = vec![];
        for mut vertices in best {
            if vertices.len() < 15 {
                continue;
            }
            vertices.sort_unstable();
            vertices.iter().for_each(|&i| member[i] = true);
            let c = vertices.iter().map(|&i| rest[i]).sum::<V>() / vertices.len() as f64;
            let radius = vertices
                .iter()
                .map(|&i| (rest[i] - c).norm())
                .fold(0.0, f64::max);
            let feather = radius.clamp(0.012 * unit, 0.03 * unit);
            let mut distance = vec![f64::INFINITY; n];
            let mut queue = VecDeque::new();
            for &i in &vertices {
                distance[i] = 0.0;
                queue.push_back(i);
            }
            let reach = feather + 0.008 * unit;
            while let Some(i) = queue.pop_front() {
                for &j in &neighbours[i] {
                    let j = j as usize;
                    let next = distance[i] + (rest[i] - rest[j]).norm();
                    if next < reach && next + 1e-12 < distance[j] {
                        distance[j] = next;
                        queue.push_back(j);
                    }
                }
            }
            let collar = (0..n)
                .filter(|&i| distance[i] > 0.0 && distance[i] < feather)
                .map(|i| {
                    let t = (distance[i] / feather).clamp(0.0, 1.0);
                    (i, 1.0 - t * t * (3.0 - 2.0 * t))
                })
                .collect();
            let anchors = (0..n)
                .filter(|&i| head[i] && distance[i] >= feather && distance[i] < reach)
                .collect();
            let window = triangles
                .iter()
                .filter(|t| {
                    t.iter()
                        .all(|&i| head[i as usize] && (rest[i as usize] - c).norm() < 0.11 * unit)
                })
                .copied()
                .collect();
            let outward = V::x() * (c.x - cx).signum();
            patches.push(Patch {
                vertices,
                collar,
                anchors,
                window,
                outward,
                shape: None,
            });
        }
        let mut collar = vec![false; n];
        for p in &patches {
            p.collar.iter().for_each(|c| collar[c.0] = true);
        }
        Self {
            patches,
            member,
            collar,
            unit,
            scan_near: OnceLock::new(),
        }
    }

         pub fn attach_outline(
        &mut self,
        rest: &[V],
        neighbours: &[Vec<u32>],
        targets: &[(String, Vec<Vec3>)],
        representative: &[usize],
        landmarks: &BTreeMap<String, Vec3>,
    ) {
        for patch in &mut self.patches {
            patch.shape =
                curve::Shape::new(patch, rest, neighbours, targets, representative, landmarks);
        }
    }

         pub fn bake_exclusion(&self, rest: &[V], x: &[V], weld_id: &[usize]) -> Vec<f64> {
        let mut welded = vec![0.0f64; x.len()];
        for patch in &self.patches {
            let Some(shape) = patch.shape.as_ref().filter(|s| s.marks.len() >= 2) else {
                continue;
            };
            let skull = Self::skull(patch);
            // The skull's scale: its spread in `x` against the rest.
            let spread = |p: &[V]| {
                let c = skull.iter().map(|&i| p[i]).sum::<V>() / skull.len().max(1) as f64;
                skull
                    .iter()
                    .map(|&i| (p[i] - c).norm_squared())
                    .sum::<f64>()
            };
            let s = (spread(x) / spread(rest).max(1e-18)).sqrt().clamp(0.5, 3.0);
            let frame = curve::frame(patch, rest, x, &skull, s, self.unit);
            let parts = shape.outline(x, &frame);
            let centre = frame.pivot;
            let plane = |p: V| {
                let d = p - centre;
                [d.dot(&frame.fwd), d.dot(&frame.up)]
            };
            let poly = curve::closed_curve(&parts.map(plane), 8);
            let feather = 0.004 * frame.unit;
            let member = shape.members();
            for (i, w) in welded.iter_mut().enumerate() {
                let d = x[i] - centre;
                if d.dot(&frame.out) < -0.015 * frame.unit || d.norm() > 0.12 * frame.unit {
                    continue;
                }
                let outside = curve::signed_distance(&poly, plane(x[i]));
                let t = (outside / feather).clamp(0.0, 1.0);
                let smooth = t * t * (3.0 - 2.0 * t);
                let e = if member.contains(&i) {
                    smooth
                } else {
                    1.0 - smooth
                };
                *w = w.max(e);
            }
        }
        weld_id.iter().map(|&i| welded[i]).collect()
    }

    fn skull(patch: &Patch) -> Vec<usize> {
        patch
            .collar
            .iter()
            .map(|c| c.0)
            .chain(patch.anchors.iter().copied())
            .collect()
    }

    /// Scan triangles within ~20 cm of the ears, with whether they are
    /// skin. `around`: the ears' current centres.
    fn scan_near(&self, scan: &Scan, labels: &[Label], around: &[V]) -> &[([V; 3], bool)] {
        self.scan_near.get_or_init(|| {
            let near = 0.2 * self.unit;
            scan.triangles
                .iter()
                .filter_map(|t| {
                    let p = t.map(|i| v(scan.positions[i as usize]));
                    around
                        .iter()
                        .any(|c| (p[0] - c).norm() < near)
                        .then(|| (p, t.iter().all(|&i| labels[i as usize] == Label::Skin)))
                })
                .collect()
        })
    }

    /// Replace each ear by a similarity of its rest shape: placed, turned and
    /// scaled with the skull around it (collar and anchors). The collar
    /// blends into the surrounding fit.
    /// `scale`: the fit's local similarity scale per vertex.
    /// `scan`: the scan and its per-vertex labels; where its pinna is
    /// visible the ear is registered to it on top of the skull placement.
    pub fn transport(
        &self,
        rest: &[V],
        x: &mut [V],
        scale: &[f64],
        scan: Option<(&Scan, &[Label])>,
    ) -> Value {
        let mut reports = vec![];
        let mut collar_weight = vec![0.0f64; x.len()];
        let near = scan.map(|(scan, labels)| {
            let around: Vec<V> = self
                .patches
                .iter()
                .map(|p| p.vertices.iter().map(|&i| x[i]).sum::<V>() / p.vertices.len() as f64)
                .collect();
            self.scan_near(scan, labels, &around)
        });
        for patch in &self.patches {
            let (mut r, mut ca, mut cb) = rotation_fit(rest, x, &patch.vertices);
             
            let skull: Vec<usize> = patch
                .collar
                .iter()
                .map(|c| c.0)
                .chain(patch.anchors.iter().copied())
                .collect();
             
            if skull.len() >= 3 {
                (r, ca, cb) = rotation_fit(rest, x, &skull);
            }
            // The ear's own least-squares scale is no measure of its size: a
            // pinna pressed onto a fused scan ear came out at half size. It
            // takes the fit's local scale on the skull around it.
            let s = skull.iter().map(|&i| scale[i]).sum::<f64>() / skull.len().max(1) as f64;
            let base = |i: usize| cb + r * (rest[i] - ca) * s;
            // A marked outline places and shapes the ear (MakeHuman's ear
            // targets); the scan's ear surface is not used.
            let outlined = patch
                .shape
                .as_ref()
                .filter(|shape| shape.marks.len() >= 2)
                .map(|shape| {
                    let frame = curve::frame(patch, rest, x, &skull, s, self.unit);
                    shape.fit(&frame, rest)
                });
            let (correct, registration) = match (&outlined, near) {
                (Some((_, report)), _) => (Correction::identity(), report.clone()),
                (None, Some(near)) => self.register(patch, &base, r, s, &skull, x, near),
                (None, None) => (
                    Correction::identity(),
                    json!({"applied": false, "reason": "no scan"}),
                ),
            };
            let place = |i: usize| correct.apply(base(i));
            let mut moved = 0.0f64;
            if let Some((placed, _)) = outlined {
                // The shaped ear and MakeHuman's own falloff into the head.
                for (i, q, h) in placed {
                    let next = x[i] * (1.0 - h) + q * h;
                    moved = moved.max((next - x[i]).norm());
                    x[i] = next;
                }
                reports.push(json!({"vertices":patch.vertices.len(),"collar":patch.collar.len(),"anchors":patch.anchors.len(),"scale":s,"max_correction_m":moved,"registration":registration}));
                continue;
            }
            for &i in &patch.vertices {
                let q = place(i);
                moved = moved.max((q - x[i]).norm());
                x[i] = q;
            }
            for &(i, w) in &patch.collar {
                if w > collar_weight[i] {
                    x[i] = x[i] * (1.0 - w) + place(i) * w;
                    collar_weight[i] = w;
                }
            }
            reports.push(json!({"vertices":patch.vertices.len(),"collar":patch.collar.len(),"anchors":patch.anchors.len(),"scale":s,"max_correction_m":moved,"registration":registration}));
        }
        json!(reports)
    }

         #[allow(clippy::too_many_arguments)]
    fn register(
        &self,
        patch: &Patch,
        base: &dyn Fn(usize) -> V,
        r: nalgebra::Matrix3<f64>,
        s: f64,
        skull: &[usize],
        x: &[V],
        near: &[([V; 3], bool)],
    ) -> (Correction, Value) {
        // Lengths follow the head: a cartoon head (and its ears) at 1.6×.
        let unit = self.unit * s;
        let reject = |reason: &str, extra: Value| {
            (
                Correction::identity(),
                json!({"applied": false, "reason": reason, "measured": extra}),
            )
        };
        let out = (r * patch.outward).normalize();
        let up = (r * V::y()).normalize();
        let up = (up - out * up.dot(&out)).normalize();
        let fwd = up.cross(&out);
        // `fwd` points to the face on both sides.
        let fwd = if fwd.dot(&(r * V::z())) < 0.0 {
            -fwd
        } else {
            fwd
        };
        let c3 = patch.vertices.iter().map(|&i| base(i)).sum::<V>() / patch.vertices.len() as f64;
        // The ball measures the skull, not the head: scaled with a cartoon
        // head it rose into the wide pinna and kept only its rim.
        let map = SideMap::new(c3, fwd, up, out, unit, self.unit);
        // MakeHuman's side: its own rest head around the ear under the same
        // placement. The fitted collar lags the placed ear and its step
        // read as part of the pinna.
        let (mh, mh_height) = map.pinna(
            patch
                .window
                .iter()
                .map(|t| (t.map(|i| base(i as usize)), true)),
        );
        let (sc, scan_height) = map.pinna(near.iter().copied());
        let (Some(m), Some(q)) = (Moments::of(&mh), Moments::of(&sc)) else {
            return reject(
                "no pinna in the side map",
                json!({"makehuman_cells": mh.len(), "scan_cells": sc.len()}),
            );
        };
        let measured = json!({
            "scan_area_cm2": q.area * 1e4, "makehuman_area_cm2": m.area * 1e4,
            "scan_axes_cm": [q.axes.0 * 100.0, q.axes.1 * 100.0],
            "makehuman_axes_cm": [m.axes.0 * 100.0, m.axes.1 * 100.0],
            "offset_cm": [(q.centre - m.centre).x * 100.0, (q.centre - m.centre).y * 100.0],
        });
        let visible = q.area / m.area.max(1e-12);
        if visible < 0.3 {
            return reject("scan ear hidden or fused", measured);
        }
        let elongation = (q.axes.0 / q.axes.1.max(1e-9)) / (m.axes.0 / m.axes.1.max(1e-9));
        // A generated pinna fused at its back shows narrower than
        // MakeHuman's; its height is what sets the size below.
        if !(0.5..=2.2).contains(&elongation) {
            return reject("scan ear unlike a pinna", measured);
        }
        let raw_scale = q.axes.0 / m.axes.0.max(1e-9);
        if !(0.6..=3.0).contains(&raw_scale) {
            return reject("implausible size", measured);
        }
        // Confidence from how much of the scan ear is visible.
        let f = ((visible - 0.3) / 0.3).clamp(0.0, 1.0);
        let f = f * f * (3.0 - 2.0 * f);
        let k = raw_scale.clamp(0.8, 2.2).powf(f);
        let mut turn = q.angle - m.angle;
        while turn > std::f64::consts::FRAC_PI_2 {
            turn -= std::f64::consts::PI;
        }
        while turn <= -std::f64::consts::FRAC_PI_2 {
            turn += std::f64::consts::PI;
        }
        let turn = turn.clamp(-0.5, 0.5) * f;
        let rotation = Rotation3::from_axis_angle(&Unit::new_normalize(fwd.cross(&up)), turn);
        // Pivot on the skull at the ear: the pinna grows out of the head,
        // its root stays on it.
        let ring = skull.iter().map(|&i| x[i]).sum::<V>() / skull.len().max(1) as f64;
        let pivot = c3 - out * (c3 - ring).dot(&out);
        // In-plane: the MakeHuman silhouette's centroid onto the scan's.
        let from = c3 + fwd * m.centre.x + up * m.centre.y;
        let moved = pivot + rotation * (from - pivot) * k;
        let d = c3 + fwd * q.centre.x + up * q.centre.y - moved;
        let mut shift = d - out * d.dot(&out);
        let limit = 0.04 * unit;
        if shift.norm() > limit {
            shift *= limit / shift.norm();
        }
         
        let root = (pivot - c3).dot(&out);
        let lateral = ((scan_height - root) - (mh_height - root) * k).clamp(-0.004 * unit, 0.0);
        let shift = (shift + out * lateral) * f;
        let correction = Correction {
            pivot,
            shift,
            scale: k,
            rotation: rotation.into_inner(),
        };
        let mut report = measured;
        report["applied"] = json!(true);
        report["confidence"] = json!(f);
        report["scale"] = json!(k);
        report["turn_rad"] = json!(turn);
        report["shift_cm"] = json!(shift.norm() * 100.0);
        report["inward_cm"] = json!(-lateral * f * 100.0);
        (correction, report)
    }
}

/// `x ↦ pivot + shift + scale · rotation · (x − pivot)`.
struct Correction {
    pivot: V,
    shift: V,
    scale: f64,
    rotation: nalgebra::Matrix3<f64>,
}

impl Correction {
    fn identity() -> Self {
        Self {
            pivot: V::zeros(),
            shift: V::zeros(),
            scale: 1.0,
            rotation: nalgebra::Matrix3::identity(),
        }
    }
    fn apply(&self, p: V) -> V {
        self.pivot + self.shift + self.rotation * (p - self.pivot) * self.scale
    }
}

/// 2D moments of the pinna cells of a side map, outliers beyond ~2.5 σ
/// trimmed twice.
struct Moments {
    /// Centroid (forward, up) in metres from the map centre.
    centre: Vector2<f64>,
    /// Standard deviations along the long and the short axis.
    axes: (f64, f64),
    /// Direction of the long axis in the plane (radians from forward).
    angle: f64,
    /// Silhouette area (m²).
    area: f64,
}

impl Moments {
    fn of(cells: &[(Vector2<f64>, f64)]) -> Option<Self> {
        let mut keep = vec![true; cells.len()];
        let mut result = None;
        for _ in 0..3 {
            let kept = || cells.iter().zip(&keep).filter(|k| *k.1).map(|k| k.0);
            let total: f64 = kept().map(|c| c.1).sum();
            if kept().count() < 12 || total <= 0.0 {
                return None;
            }
            let c = kept().map(|k| k.0 * k.1).sum::<Vector2<f64>>() / total;
            let cov = kept()
                .map(|k| (k.0 - c) * (k.0 - c).transpose() * k.1)
                .sum::<Matrix2<f64>>()
                / total;
            let inverse = (cov + Matrix2::identity() * 1e-10).try_inverse()?;
            let eig = cov.symmetric_eigen();
            let (big, small) = if eig.eigenvalues[0] >= eig.eigenvalues[1] {
                (0, 1)
            } else {
                (1, 0)
            };
            let axis = eig.eigenvectors.column(big);
            result = Some(Self {
                centre: c,
                axes: (
                    eig.eigenvalues[big].max(0.0).sqrt(),
                    eig.eigenvalues[small].max(0.0).sqrt(),
                ),
                angle: axis[1].atan2(axis[0]),
                area: total,
            });
            for (k, (p, _)) in keep.iter_mut().zip(cells) {
                *k = ((p - c).transpose() * inverse * (p - c))[0] < 6.0;
            }
        }
        result
    }
}

/// Side view of the head around one ear: a grid in the (forward, up) plane
/// holding the most lateral surface point per cell.
struct SideMap {
    centre: V,
    fwd: V,
    up: V,
    out: V,
    cell: f64,
    /// Grid origin (forward, up) relative to `centre`, and size.
    lo: (f64, f64),
    size: (usize, usize),
    unit: f64,
    skull: f64,
}

impl SideMap {
    /// `unit`: lengths of the window (with the head); `skull`: of the ball.
    fn new(centre: V, fwd: V, up: V, out: V, unit: f64, skull: f64) -> Self {
        let cell = 0.0025 * unit;
        // 9 cm behind to 6 cm in front (the eye stays out), 8 cm below to
        // 9 cm above.
        let lo = (-0.09 * unit, -0.08 * unit);
        let size = (
            ((0.15 * unit) / cell).ceil() as usize,
            ((0.17 * unit) / cell).ceil() as usize,
        );
        Self {
            centre,
            fwd,
            up,
            out,
            cell,
            lo,
            size,
            unit,
            skull,
        }
    }

    /// Cells of the pinna: skin that stands out of the skull (the lateral
    /// height minus its opening by a 4 cm ball) by more than 4 mm, the
    /// largest such patch near the map centre. (cell centre, area).
    /// Also returns the silhouette's mean lateral height (from the centre).
    fn pinna(
        &self,
        triangles: impl Iterator<Item = ([V; 3], bool)>,
    ) -> (Vec<(Vector2<f64>, f64)>, f64) {
        let (w, h) = self.size;
        let mut height = vec![f64::NEG_INFINITY; w * h];
        let mut skin = vec![false; w * h];
        for (t, is_skin) in triangles {
            let q = t.map(|p| {
                let d = p - self.centre;
                (
                    (d.dot(&self.fwd) - self.lo.0) / self.cell - 0.5,
                    (d.dot(&self.up) - self.lo.1) / self.cell - 0.5,
                    d.dot(&self.out),
                )
            });
            if q.iter().any(|p| p.2 < -0.05 * self.unit) {
                continue;
            }
            let x0 = q
                .iter()
                .map(|p| p.0)
                .fold(f64::INFINITY, f64::min)
                .ceil()
                .max(0.0);
            let x1 = q
                .iter()
                .map(|p| p.0)
                .fold(f64::NEG_INFINITY, f64::max)
                .floor();
            let y0 = q
                .iter()
                .map(|p| p.1)
                .fold(f64::INFINITY, f64::min)
                .ceil()
                .max(0.0);
            let y1 = q
                .iter()
                .map(|p| p.1)
                .fold(f64::NEG_INFINITY, f64::max)
                .floor();
            if x1 < x0 || y1 < y0 {
                // Smaller than a cell: its first corner's cell.
                let (cx, cy) = (q[0].0.round(), q[0].1.round());
                if cx >= 0.0 && cy >= 0.0 && (cx as usize) < w && (cy as usize) < h {
                    let k = cy as usize * w + cx as usize;
                    if q[0].2 > height[k] {
                        height[k] = q[0].2;
                        skin[k] = is_skin;
                    }
                }
                continue;
            }
            let area =
                (q[1].0 - q[0].0) * (q[2].1 - q[0].1) - (q[2].0 - q[0].0) * (q[1].1 - q[0].1);
            if area.abs() < 1e-12 {
                continue;
            }
            for y in y0 as usize..=(y1 as usize).min(h - 1) {
                for x in x0 as usize..=(x1 as usize).min(w - 1) {
                    let (px, py) = (x as f64, y as f64);
                    let l1 = ((q[2].0 - px) * (q[0].1 - py) - (q[0].0 - px) * (q[2].1 - py)) / area;
                    let l2 = ((q[0].0 - px) * (q[1].1 - py) - (q[1].0 - px) * (q[0].1 - py)) / area;
                    let l0 = 1.0 - l1 - l2;
                    if l0 < -1e-9 || l1 < -1e-9 || l2 < -1e-9 {
                        continue;
                    }
                    let z = q[0].2 * l0 + q[1].2 * l1 + q[2].2 * l2;
                    let k = y * w + x;
                    if z > height[k] {
                        height[k] = z;
                        skin[k] = is_skin;
                    }
                }
            }
        }
        // Rolling-ball opening from inside the head: the skull's convex
        // side is kept, anything narrower than the ball (a pinna) is cut.
        let ball = 0.04 * self.skull;
        let reach = (ball / self.cell).floor() as i64;
        let disk: Vec<(i64, i64, f64)> = (-reach..=reach)
            .flat_map(|dy| (-reach..=reach).map(move |dx| (dx, dy)))
            .filter_map(|(dx, dy)| {
                let r2 = ((dx * dx + dy * dy) as f64) * self.cell * self.cell;
                (r2 <= ball * ball).then(|| (dx, dy, (ball * ball - r2).sqrt() - ball))
            })
            .collect();
        let pass = |src: &[f64], erode: bool| -> Vec<f64> {
            (0..w * h)
                .map(|k| {
                    if height[k] == f64::NEG_INFINITY {
                        return f64::NEG_INFINITY;
                    }
                    let (x, y) = ((k % w) as i64, (k / w) as i64);
                    let mut best = if erode {
                        f64::INFINITY
                    } else {
                        f64::NEG_INFINITY
                    };
                    for &(dx, dy, g) in &disk {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                            continue;
                        }
                        let s0 = src[ny as usize * w + nx as usize];
                        if s0 == f64::NEG_INFINITY {
                            continue;
                        }
                        if erode {
                            best = best.min(s0 - g);
                        } else {
                            best = best.max(s0 + g);
                        }
                    }
                    best
                })
                .collect()
        };
        let opened = pass(&pass(&height, true), false);
        let threshold = 0.004 * self.skull;
        let raised: Vec<bool> = (0..w * h)
            .map(|k| skin[k] && height[k] > f64::NEG_INFINITY && height[k] - opened[k] > threshold)
            .collect();
        // Largest 4-connected patch whose centroid is within 4.5 cm of the
        // centre.
        let mut seen = vec![false; w * h];
        let mut best: Vec<usize> = vec![];
        for s0 in 0..w * h {
            if !raised[s0] || seen[s0] {
                continue;
            }
            let mut comp = vec![s0];
            seen[s0] = true;
            let mut i = 0;
            while i < comp.len() {
                let k0 = comp[i];
                let (x, y) = (k0 % w, k0 / w);
                let next = [
                    (x > 0).then(|| k0 - 1),
                    (x + 1 < w).then(|| k0 + 1),
                    (y > 0).then(|| k0 - w),
                    (y + 1 < h).then(|| k0 + w),
                ];
                for k in next.into_iter().flatten() {
                    if raised[k] && !seen[k] {
                        seen[k] = true;
                        comp.push(k);
                    }
                }
                i += 1;
            }
            let centre = comp
                .iter()
                .map(|&k| self.cell_centre(k))
                .sum::<Vector2<f64>>()
                / comp.len() as f64;
            if centre.norm() < 0.045 * self.unit && comp.len() > best.len() {
                best = comp;
            }
        }
        let mean = best.iter().map(|&k| height[k]).sum::<f64>() / best.len().max(1) as f64;
        let cells = best
            .into_iter()
            .map(|k| (self.cell_centre(k), self.cell * self.cell))
            .collect();
        (cells, mean)
    }

    fn cell_centre(&self, k: usize) -> Vector2<f64> {
        let w = self.size.0;
        Vector2::new(
            self.lo.0 + ((k % w) as f64 + 0.5) * self.cell,
            self.lo.1 + ((k / w) as f64 + 0.5) * self.cell,
        )
    }
}

/// Best rotation of `rest[ids]` onto `x[ids]` about their centroids, with
/// both centroids.
fn rotation_fit(rest: &[V], x: &[V], ids: &[usize]) -> (nalgebra::Matrix3<f64>, V, V) {
    let k = ids.len().max(1) as f64;
    let ca = ids.iter().map(|&i| rest[i]).sum::<V>() / k;
    let cb = ids.iter().map(|&i| x[i]).sum::<V>() / k;
    let mut m = nalgebra::Matrix3::zeros();
    for &i in ids {
        m += (x[i] - cb) * (rest[i] - ca).transpose();
    }
    (proper_rotation(m), ca, cb)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat head side with a 2 × 6 cm plate standing 1 cm out of it: the
    /// side map finds the plate (centre, size, upright long axis), not the
    /// skull.
    #[test]
    fn side_map_finds_what_stands_out_of_the_skull() {
        let (fwd, up, out) = (V::z(), V::y(), V::x());
        let at = |u: f64, w: f64, h: f64| fwd * u + up * w + out * h;
        let mut tris = vec![];
        let step = 0.005;
        for i in 0..40 {
            for j in 0..40 {
                let (u, w) = (-0.1 + i as f64 * step, -0.1 + j as f64 * step);
                let (a0, b0, c0, d0) = (
                    at(u, w, 0.0),
                    at(u + step, w, 0.0),
                    at(u + step, w + step, 0.0),
                    at(u, w + step, 0.0),
                );
                tris.push(([a0, b0, c0], true));
                tris.push(([a0, c0, d0], true));
            }
        }
        let (u0, u1, w0, w1) = (-0.02, 0.0, -0.02, 0.04);
        let plate = [
            at(u0, w0, 0.01),
            at(u1, w0, 0.01),
            at(u1, w1, 0.01),
            at(u0, w1, 0.01),
        ];
        tris.push(([plate[0], plate[1], plate[2]], true));
        tris.push(([plate[0], plate[2], plate[3]], true));
        let map = SideMap::new(V::zeros(), fwd, up, out, 1.0, 1.0);
        let (cells, height) = map.pinna(tris.into_iter());
        let m = Moments::of(&cells).expect("plate found");
        assert!(
            (m.centre - Vector2::new(-0.01, 0.01)).norm() < 0.003,
            "{:?}",
            m.centre
        );
        assert!((m.area - 0.0012).abs() < 0.0003, "{}", m.area);
        assert!((m.axes.0 - 0.06 / 12f64.sqrt()).abs() < 0.003);
        assert!((m.angle.abs() - std::f64::consts::FRAC_PI_2).abs() < 0.1);
        assert!((height - 0.01).abs() < 1e-6);
        // Hair over the plate: nothing that is skin stands out.
        let hair = vec![
            ([plate[0], plate[1], plate[2]], false),
            ([plate[0], plate[2], plate[3]], false),
        ];
        let (cells, _) = map.pinna(hair.into_iter());
        assert!(Moments::of(&cells).is_none());
    }
}
