 use crate::geom::{V, a, v};
use crate::prior::Prior;
use crate::scan::{Scan, build_query};
use crate::segment::{Label, Segmentation};
use charloom_core::Vec3;
use nalgebra::Matrix4;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Body part around one centreline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    Torso,
    Neck,
    UpperArm,
    Forearm,
    Thigh,
    Calf,
}

 pub struct Axis {
    pub name: String,
    pub part: Part,
    a: V,
    b: V,
    /// Two unit vectors spanning the cross-section plane.
    e: [V; 2],
}

pub struct Frame {
    /// Point on the centreline the cross-section is taken around.
    pub center: V,
    /// Along the axis from its start, metres (height above the hips for the
    /// torso).
    pub t: f64,
    /// From `center` to the point, in the cross-section plane.
    pub radial: V,
}

impl Axis {
    fn new(name: String, part: Part, a: V, b: V) -> Self {
        let u = if part == Part::Torso {
            V::y()
        } else {
            (b - a).normalize()
        };
        let e0 = u
            .cross(&if u.x.abs() < 0.9 { V::x() } else { V::z() })
            .normalize();
        let e1 = u.cross(&e0);
        let e = if part == Part::Torso {
            [V::x(), V::z()]
        } else {
            [e0, e1]
        };
        Self {
            name,
            part,
            a,
            b,
            e,
        }
    }

    pub fn frame(&self, p: V) -> Frame {
        if self.part == Part::Torso {
            let s = ((p.y - self.a.y) / (self.b.y - self.a.y)).clamp(0.0, 1.0);
            let mut center = self.a + (self.b - self.a) * s;
            center.y = p.y;
            let mut radial = p - center;
            radial.y = 0.0;
            Frame {
                center,
                t: p.y - self.a.y,
                radial,
            }
        } else {
            let u = (self.b - self.a).normalize();
            let t = (p - self.a).dot(&u);
            let center = self.a + u * t;
            Frame {
                center,
                t,
                radial: p - center,
            }
        }
    }

    fn plane(&self, radial: V) -> [f64; 2] {
        [radial.dot(&self.e[0]), radial.dot(&self.e[1])]
    }

    fn length(&self) -> f64 {
        if self.part == Part::Torso {
            self.b.y - self.a.y
        } else {
            (self.b - self.a).norm()
        }
    }
}

/// Centrelines from a pose and the segment of every body vertex (by its
/// dominant MakeHuman joint). Hands, feet and the head have none.
pub struct Axes {
    pub axes: Vec<Axis>,
    /// Per unwelded body vertex.
    pub vertex_axis: Vec<Option<usize>>,
}

const SIDES: [(&str, &str); 2] = [("l", "L"), ("r", "R")];
const LIMBS: [(&str, Part, &str, &str); 4] = [
    ("upper_arm", Part::UpperArm, "upperarm01", "lowerarm01"),
    ("forearm", Part::Forearm, "lowerarm01", "wrist"),
    ("thigh", Part::Thigh, "upperleg01", "lowerleg01"),
    ("calf", Part::Calf, "lowerleg01", "foot"),
];

impl Axes {
    pub fn new(prior: &Prior, worlds: &[Matrix4<f64>]) -> Self {
        let origin = |n: &str| -> Option<V> {
            prior
                .by_name
                .get(n)
                .map(|&j| worlds[j].fixed_view::<3, 1>(0, 3).into_owned())
        };
        let mut axes = vec![];
        if let (Some(hl), Some(hr), Some(sl), Some(sr)) = (
            origin("upperleg01.L"),
            origin("upperleg01.R"),
            origin("upperarm01.L"),
            origin("upperarm01.R"),
        ) {
            axes.push(Axis::new(
                "torso".into(),
                Part::Torso,
                (hl + hr) * 0.5,
                (sl + sr) * 0.5,
            ));
        }
        if let (Some(n), Some(h)) = (origin("neck01"), origin("head")) {
            axes.push(Axis::new("neck".into(), Part::Neck, n, h));
        }
        for (side, suffix) in SIDES {
            for (name, part, from, to) in LIMBS {
                if let (Some(p), Some(q)) = (
                    origin(&format!("{from}.{suffix}")),
                    origin(&format!("{to}.{suffix}")),
                ) {
                    axes.push(Axis::new(format!("{name}_{side}"), part, p, q));
                }
            }
        }
        let index = |name: &str| axes.iter().position(|x| x.name == name);
        let skeleton = prior.skeleton();
        let joint_axis: Vec<Option<usize>> = skeleton
            .joints
            .iter()
            .map(|j| {
                let n = j.name.to_lowercase();
                let side = if n.ends_with(".l") {
                    "l"
                } else if n.ends_with(".r") {
                    "r"
                } else {
                    ""
                };
                let starts = |p: &str| n.starts_with(p);
                if starts("upperarm") || starts("shoulder") {
                    index(&format!("upper_arm_{side}"))
                } else if starts("lowerarm") {
                    index(&format!("forearm_{side}"))
                } else if starts("upperleg") {
                    index(&format!("thigh_{side}"))
                } else if starts("lowerleg") {
                    index(&format!("calf_{side}"))
                } else if starts("neck") {
                    index("neck")
                } else if starts("spine")
                    || starts("breast")
                    || starts("pelvis")
                    || starts("clavicle")
                    || n == "root"
                {
                    index("torso")
                } else {
                    None
                }
            })
            .collect();
        let vertex_axis = (0..prior.body().bind.len())
            .map(|i| joint_axis[prior.dominant_joint(prior.body, i)])
            .collect();
        Self { axes, vertex_axis }
    }

    /// A point's segment within a limb chain by its position along the
    /// proximal bone (upper arm vs forearm at the elbow, thigh vs calf at
    /// the knee), not by its nearest prior vertex: a scan limb much thicker
    /// than MakeHuman's lies nearer the wrong segment's surface.
    pub fn split_chain(&self, k: usize, p: V) -> usize {
        let axis = &self.axes[k];
        let (proximal, distal) = match axis.part {
            Part::UpperArm | Part::Forearm => ("upper_arm", "forearm"),
            Part::Thigh | Part::Calf => ("thigh", "calf"),
            _ => return k,
        };
        let side = &axis.name[axis.name.len() - 1..];
        let find = |n: &str| {
            self.axes
                .iter()
                .position(|x| x.name == format!("{n}_{side}"))
        };
        let (Some(a), Some(b)) = (find(proximal), find(distal)) else {
            return k;
        };
        if self.axes[a].frame(p).t < self.axes[a].length() {
            a
        } else {
            b
        }
    }
}

/// Where a garment's gap came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GapSource {
    /// Measured at the garment's openings on the scan.
    Measured,
    /// `GarmentGuide::gap_mm`.
    Guidance,
    /// Default by tightness and layer.
    Prior,
}

/// One garment opening: the garment's edge ring and the skin ring past it.
#[derive(Debug, Clone, Serialize)]
pub struct Opening {
    pub axis: String,
    pub part: Part,
    /// Station of the garment-side ring along the axis, metres.
    pub t_m: f64,
    /// Skin is distal (towards the hand/foot/head) of the garment.
    pub skin_distal: bool,
    pub garment_radius_m: f64,
    pub skin_radius_m: f64,
    /// MakeHuman's own radius ratio garment station / skin station.
    pub body_taper: f64,
    /// Garment radius minus the skin radius carried to the garment station.
    pub gap_m: f64,
    /// Garment radius over that body radius.
    pub ease: f64,
}

impl Opening {
         pub fn is_collar(&self) -> bool {
        self.axis.starts_with("neck") || (self.part == Part::Torso && self.skin_distal)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GarmentGap {
    pub kind: GarmentKind,
    pub piece: String,
    pub gap_m: f64,
    pub source: GapSource,
    /// Weight of the body target under this garment, 0–1.
    pub confidence: f64,
    /// The target may pull the body inward too (otherwise outward only).
    pub two_sided: bool,
    pub tight: bool,
    pub layer: u32,
    pub openings: Vec<Opening>,
    /// Parts the gap holds for; empty: all. A gap measured at a sleeve holds
    /// for the arms and the torso, not for the same garment's skirt over
    /// the thighs, which gets `elsewhere`.
    pub covers: Vec<Part>,
    /// (gap, confidence, two-sided) on the parts not covered.
    pub elsewhere: (f64, f64, bool),
         pub torso_opening: bool,
}

impl GarmentGap {
    /// The gap on this part is a default (guessed), not measured or stated.
    pub fn guessed_on(&self, part: Part) -> bool {
        match self.source {
            GapSource::Prior => true,
            GapSource::Guidance => false,
            GapSource::Measured => {
                (part == Part::Torso && !self.tight && !self.torso_opening)
                    || !(self.covers.is_empty() || self.covers.contains(&part))
            }
        }
    }

    /// (gap, confidence, two-sided) under this garment on one body part.
    pub fn on(&self, part: Part) -> (f64, f64, bool) {
        // A loose garment's gap measured at a sleeve cuff or a trouser hem
        // is not its gap over the chest or the hips: a tunic's cuff flares
        // (3.5 cm on the flame), its body rests on the chest. Over the torso
        // it is the garment kind's default gap, no more (a tight shirt holds
        // its measured gap over the belly).
        if part == Part::Torso
            && self.source == GapSource::Measured
            && !self.tight
            && !self.torso_opening
        {
            return (
                self.gap_m.min(self.elsewhere.0),
                self.elsewhere.1,
                self.elsewhere.2,
            );
        }
        if self.covers.is_empty() || self.covers.contains(&part) {
            (self.gap_m, self.confidence, self.two_sided)
        } else {
            self.elsewhere
        }
    }
}

/// A soft radial target for one body vertex under a garment: along `dir`
/// from `center`, the body should sit at `radius`.
#[derive(Debug, Clone, Copy)]
pub struct Fill {
    pub center: V,
    pub dir: V,
    pub radius: f64,
         pub start: f64,
    pub weight: f64,
    pub two_sided: bool,
    pub piece: usize,
    pub part: Part,
    /// The gap is a default (guessed), not measured or stated.
    pub guessed: bool,
    /// Marks a ray whose cloth hangs free (no target; counted per garment
    /// and part, see `fill_targets`).
    pub draped: bool,
}

/// Radius of a cross-section ring (2-D points in its plane): a least-
/// squares circle (Kåsa), so an axis off the section's centre does not
/// matter and an arc missing part of its round (the inner arm, assigned to
/// the torso) still measures. `None` with too few points or an angular hole
/// of 150° or more around the fitted centre.
fn ring(points: &[[f64; 2]]) -> Option<f64> {
    if points.len() < 16 {
        return None;
    }
    let n = points.len() as f64;
    // Centred for conditioning.
    let m = points
        .iter()
        .fold([0.0, 0.0], |s, p| [s[0] + p[0] / n, s[1] + p[1] / n]);
    let mut ata = nalgebra::Matrix3::<f64>::zeros();
    let mut atb = nalgebra::Vector3::<f64>::zeros();
    for p in points {
        let (x, y) = (p[0] - m[0], p[1] - m[1]);
        let row = nalgebra::Vector3::new(x, y, 1.0);
        ata += row * row.transpose();
        atb += row * -(x * x + y * y);
    }
    let sol = ata.lu().solve(&atb)?;
    let (cx, cy) = (-sol[0] / 2.0, -sol[1] / 2.0);
    let mut angles: Vec<f64> = Vec::with_capacity(points.len());
    let mut sum = 0.0;
    for p in points {
        let (dx, dy) = (p[0] - m[0] - cx, p[1] - m[1] - cy);
        sum += dx.hypot(dy);
        angles.push(dy.atan2(dx));
    }
    angles.sort_by(f64::total_cmp);
    let wrap = angles[0] + std::f64::consts::TAU - angles[angles.len() - 1];
    let widest = angles.windows(2).map(|w| w[1] - w[0]).fold(wrap, f64::max);
    (widest < 150f64.to_radians() && sum.is_finite()).then_some(sum / n)
}

const SKIN: usize = usize::MAX;
/// Cross-section bin width, fraction of the body height (~2 cm).
const BIN: f64 = 0.012;
 const MAX_GAP: f64 = 0.04;

 const MOTION_GAP: f64 = 0.003;

fn smooth(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// What a garment is, by its name, for its default gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GarmentKind {
    Tight,
    Medium,
    Loose,
    Padded,
    Plate,
}

impl GarmentKind {
    pub fn of(name: &str, tight: bool) -> Self {
        let n = name.to_lowercase();
        let has = |words: &[&str]| words.iter().any(|w| n.contains(w));
        if has(&[
            "plate",
            "cuirass",
            "breastplate",
            "pauldron",
            "gauntlet",
            "greave",
        ]) {
            GarmentKind::Plate
        } else if has(&[
            "armour",
            "armor",
            "vest",
            "gambeson",
            "padded",
            "quilt",
            "jacket",
            "coat",
            "doublet",
            "jerkin",
            "brigandine",
        ]) {
            GarmentKind::Padded
        } else if tight {
            GarmentKind::Tight
        } else if has(&[
            "robe", "cloak", "cape", "dress", "gown", "skirt", "kimono", "hanfu", "poncho", "apron",
        ]) {
            GarmentKind::Loose
        } else {
            GarmentKind::Medium
        }
    }

    /// Typical gap between this garment and the skin when it is the only
    /// layer, metres at 1.75 m: material plus the air under it where it
    /// rests on the body.
    fn gap(self) -> f64 {
        match self {
            GarmentKind::Tight => 0.004,
            GarmentKind::Medium => 0.008,
            GarmentKind::Loose => 0.012,
            GarmentKind::Padded => 0.012,
            GarmentKind::Plate => 0.020,
        }
    }
}

 fn prior_gap(kind: GarmentKind, layer: u32, height: f64) -> f64 {
    (kind.gap() + 0.004 * layer as f64).min(0.025) * height / 1.75
}

/// Shell − gap beyond this many times the prior's body radius is cloth
/// hanging free (a robe's skirt, baggy trousers): no evidence of the body.
fn drape_limit(stylized: bool) -> f64 {
    if stylized { 1.5 } else { 1.45 }
}

/// The scan's surface samples (triangle centroids) by body segment, with
/// their label (`SKIN` or a garment piece). Hair, rigid parts and
/// unassigned surface are left out.
pub struct Samples {
    /// (axis, centroid, key).
    pub points: Vec<(usize, V, usize)>,
}

impl Samples {
    pub fn new(
        scan: &Scan,
        seg: &Segmentation,
        axes: &Axes,
        posed_body: &[Vec3],
        triangles: &[[u32; 3]],
        height: f64,
    ) -> Self {
        let query = build_query(posed_body, triangles);
        let stride = (scan.triangles.len() / 400_000).max(1);
        let points = (0..scan.triangles.len())
            .into_par_iter()
            .step_by(stride)
            .filter_map(|t| {
                let key = match seg.triangle_label[t] {
                    Label::Skin => SKIN,
                    Label::Hair => return None,
                    Label::Garment => {
                        let p = seg.triangle_piece[t]?;
                        if seg.piece_hair.get(p) == Some(&true)
                            || seg.piece_rigid.get(p) == Some(&true)
                        {
                            return None;
                        }
                        p
                    }
                };
                let c = scan.triangles[t]
                    .iter()
                    .fold(V::zeros(), |s, &i| s + v(scan.positions[i as usize]) / 3.0);
                let hit = query.closest_point(a(c), height * 0.15, None)?;
                let tri = triangles[hit.source[1]];
                let k =
                    (0..3).max_by(|&x, &y| hit.barycentric[x].total_cmp(&hit.barycentric[y]))?;
                let axis = axes.split_chain(axes.vertex_axis[tri[k] as usize]?, c);
                Some((axis, c, key))
            })
            .collect();
        Self { points }
    }
}

type Bins = BTreeMap<i64, BTreeMap<usize, Vec<[f64; 2]>>>;

/// Measures the gap at every garment opening and settles one gap per piece.
pub fn estimate_gaps(
    seg: &Segmentation,
    axes: &Axes,
    samples: &Samples,
    posed_body: &[Vec3],
    clearance: f64,
    height: f64,
) -> (Vec<GarmentGap>, BTreeMap<String, String>) {
    let width = BIN * height;
    let bin_of = |t: f64| (t / width).floor() as i64;
    // Per axis: bins of scan samples by key, and of the posed body.
    let mut scan_bins: Vec<Bins> = vec![BTreeMap::new(); axes.axes.len()];
    for &(k, c, key) in &samples.points {
        let f = axes.axes[k].frame(c);
        scan_bins[k]
            .entry(bin_of(f.t))
            .or_default()
            .entry(key)
            .or_default()
            .push(axes.axes[k].plane(f.radial));
    }
    let mut body_bins: Vec<BTreeMap<i64, Vec<[f64; 2]>>> = vec![BTreeMap::new(); axes.axes.len()];
    for (i, p) in posed_body.iter().enumerate() {
        if let Some(k) = axes.vertex_axis[i] {
            let f = axes.axes[k].frame(v(*p));
            body_bins[k]
                .entry(bin_of(f.t))
                .or_default()
                .push(axes.axes[k].plane(f.radial));
        }
    }
    let mut openings: BTreeMap<usize, Vec<Opening>> = BTreeMap::new();
    // Per axis, bin labels from its start: `s` skin, a piece's index (0–9,
    // then a–z), `.` mixed or sparse — for review.
    let mut profiles = BTreeMap::new();
    for (k, bins) in scan_bins.iter().enumerate() {
        let axis = &axes.axes[k];
        // Majority label of a bin (> 70 % of at least 16 samples).
        let class = |b: i64| -> Option<usize> {
            let row = bins.get(&b)?;
            let total: usize = row.values().map(Vec::len).sum();
            if total < 16 {
                return None;
            }
            row.iter()
                .find(|(_, p)| p.len() as f64 > 0.7 * total as f64)
                .map(|(key, _)| *key)
        };
        let Some((&lo, _)) = bins.first_key_value() else {
            continue;
        };
        let hi = *bins.last_key_value().unwrap().0;
        profiles.insert(
            axis.name.clone(),
            (lo..=hi)
                .map(|b| match class(b) {
                    None => '.',
                    Some(SKIN) => 's',
                    Some(p) => std::char::from_digit(p as u32 % 36, 36).unwrap_or('?'),
                })
                .collect::<String>(),
        );
        for b in lo..=hi {
            let Some(x) = class(b) else { continue };
            // Next classified bin, across up to two mixed ones (a folded
            // cuff or hem rim).
            let Some((b2, y)) = (b + 1..=b + 3).find_map(|c| class(c).map(|y| (c, y))) else {
                continue;
            };
            if x == y || (x != SKIN && y != SKIN) {
                continue;
            }
            let (piece, garment_bin, skin_bin) = if x == SKIN { (y, b2, b) } else { (x, b, b2) };
            let away = (garment_bin - skin_bin).signum();
            // Two rings on each side, one bin back from the edge when the
            // label holds there (a folded cuff rim or squeezed skin right at
            // the edge would bias the rings).
            let band = |start: i64, dir: i64, key: usize| -> Vec<i64> {
                let run: Vec<i64> = (0..3)
                    .map(|m| start + dir * m)
                    .take_while(|&c| class(c) == Some(key))
                    .collect();
                if run.len() >= 2 {
                    run[1..].to_vec()
                } else {
                    run
                }
            };
            let gb = band(garment_bin, away, piece);
            let sb = band(skin_bin, -away, SKIN);
            let mean =
                |xs: Vec<f64>| (!xs.is_empty()).then(|| xs.iter().sum::<f64>() / xs.len() as f64);
            let rings = |bs: &[i64], key: usize| {
                mean(
                    bs.iter()
                        .filter_map(|c| ring(bins.get(c)?.get(&key)?))
                        .collect(),
                )
            };
            let body_rings = |bs: &[i64]| {
                mean(
                    bs.iter()
                        .filter_map(|c| ring(body_bins[k].get(c)?))
                        .collect(),
                )
            };
            let (Some(garment_radius), Some(skin_radius)) = (rings(&gb, piece), rings(&sb, SKIN))
            else {
                continue;
            };
            let taper = match (body_rings(&gb), body_rings(&sb)) {
                (Some(g), Some(s)) if s > 1e-6 => (g / s).clamp(0.7, 1.4),
                _ => 1.0,
            };
            let body_radius = skin_radius * taper;
            let gap = garment_radius - body_radius;
            if !(-0.02 * height..=0.08 * height).contains(&gap) {
                continue;
            }
            let t = gb.iter().map(|&c| (c as f64 + 0.5) * width).sum::<f64>() / gb.len() as f64;
            openings.entry(piece).or_default().push(Opening {
                axis: axis.name.clone(),
                part: axis.part,
                t_m: t,
                skin_distal: skin_bin > garment_bin,
                garment_radius_m: garment_radius,
                skin_radius_m: skin_radius,
                body_taper: taper,
                gap_m: gap,
                ease: garment_radius / body_radius.max(1e-6),
            });
        }
    }
    let gaps = (0..seg.pieces.len())
        .map(|p| {
            let tight = seg.piece_tight.get(p).copied().unwrap_or(false);
            let layer = seg.piece_layer.get(p).copied().unwrap_or(0);
            let list = openings.remove(&p).unwrap_or_default();
            // Collars are reported but do not set the gap of the garment's
            // body (`Opening::is_collar`).
            let mut gaps: Vec<f64> = list
                .iter()
                .filter(|o| !o.is_collar())
                .map(|o| o.gap_m)
                .collect();
            gaps.sort_by(f64::total_cmp);
            let cap = MAX_GAP * height;
            let guided = seg
                .piece_gap_mm
                .get(p)
                .copied()
                .flatten()
                .filter(|g| g.is_finite() && *g >= 0.0);
            let kind = GarmentKind::of(&seg.pieces[p], tight);
            // A default gap by garment kind is a soft two-sided target where
            // the cloth is near the body (`fill_targets` drops cloth hanging
            // free); tight items are trusted more.
            let unmeasured = if let Some(g) = guided {
                (g / 1000.0, GapSource::Guidance, 0.8, true)
            } else {
                let confidence = if tight { 0.6 } else { 0.3 };
                (
                    prior_gap(kind, layer, height),
                    GapSource::Prior,
                    confidence,
                    true,
                )
            };
            let (gap, source, confidence, two_sided) = if !gaps.is_empty() {
                let m = gaps.len();
                let median = if m % 2 == 1 {
                    gaps[m / 2]
                } else {
                    0.5 * (gaps[m / 2 - 1] + gaps[m / 2])
                };
                (median, GapSource::Measured, 1.0, true)
            } else {
                unmeasured
            };
             
            let mut covers = vec![];
            if source == GapSource::Measured {
                covers.push(Part::Torso);
                for o in list.iter().filter(|o| !o.is_collar()) {
                    let family: &[Part] = match o.part {
                        Part::UpperArm | Part::Forearm => &[Part::UpperArm, Part::Forearm],
                        Part::Thigh | Part::Calf => &[Part::Thigh, Part::Calf],
                        _ => &[],
                    };
                    for f in family {
                        if !covers.contains(f) {
                            covers.push(*f);
                        }
                    }
                }
            }
            let torso_opening = source == GapSource::Measured
                && list.iter().any(|o| o.part == Part::Torso && !o.is_collar());
            GarmentGap {
                kind,
                piece: seg.pieces[p].clone(),
                gap_m: gap.clamp(clearance, cap),
                source,
                confidence,
                two_sided,
                tight,
                layer,
                openings: list,
                covers,
                elsewhere: (
                    unmeasured.0.clamp(clearance, cap),
                    unmeasured.2,
                    unmeasured.3,
                ),
                torso_opening,
            }
        })
        .collect();
    (gaps, profiles)
}

 #[allow(clippy::too_many_arguments)]
pub fn fill_targets(
    scan: &Scan,
    seg: &Segmentation,
    axes: &Axes,
    gaps: &[GarmentGap],
    posed_body: &[Vec3],
    triangles: &[[u32; 3]],
    clearance: f64,
    height: f64,
    stylized: bool,
) -> Vec<Option<Fill>> {
    let (reach, grow) = if stylized { (2.4, 1.8) } else { (1.65, 1.45) };
    // Area-weighted vertex normals of the posed body.
    let mut normals = vec![V::zeros(); posed_body.len()];
    for t in triangles {
        let [p0, p1, p2] = t.map(|i| v(posed_body[i as usize]));
        let n = (p1 - p0).cross(&(p2 - p0));
        for &i in t {
            normals[i as usize] += n;
        }
    }
    let out: Vec<Option<Fill>> = posed_body
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let k = axes.vertex_axis[i]?;
            let axis = &axes.axes[k];
            if axis.part == Part::Neck {
                return None;
            }
            let f = axis.frame(v(*p));
            // Between the joints only: over a shoulder or hip joint the
            // centreline ray runs along the surface, not out of it.
            if axis.part != Part::Torso && !(0.0..=axis.length()).contains(&f.t) {
                return None;
            }
            let r0 = f.radial.norm();
            if r0 < 0.008 * height {
                return None;
            }
            let dir = f.radial / r0;
             
            let top = if dir.z > 0.6 { 0.95 } else { 0.85 };
            if axis.part == Part::Torso && f.t > top * axis.length() {
                return None;
            }
            // A radial move must be an outward move of this surface (not
            // the top of a shoulder under a horizontal torso ray).
            if normals[i].try_normalize(1e-12)?.dot(&dir) < 0.6 {
                return None;
            }
            let shell = scan
                .query
                .first_ray_distance(a(f.center), a(dir), r0 * reach)?;
            // The surface leaving the body outward (the inside of the
            // garment); an entering face means the ray started outside the
            // scan (between the legs, past a robe's slit).
            let hit =
                scan.query
                    .closest_point(a(f.center + dir * shell), 2e-4, Some((a(dir), 0.5)))?;
            let t = hit.source[1];
            if seg.triangle_label[t] != Label::Garment {
                return None;
            }
            let piece = seg.triangle_piece[t]?;
            if seg.piece_hair.get(piece) == Some(&true) || seg.piece_rigid.get(piece) == Some(&true)
            {
                return None;
            }
            if shell < 0.6 * r0 {
                return None;
            }
            let g = gaps.get(piece)?;
            let (gap, confidence, two_sided) = g.on(axis.part);
            // Loose cloth hangs from the shoulders and rests on the chest
            // and the shoulder blades: there its gap is the cloth's
            // thickness, not the air gap of a guessed default, so the
            // shell's bust (or pectoral) profile drives the chest morphs.
            let resting = axis.part == Part::Torso
                && (0.55..=0.85).contains(&(f.t / axis.length()))
                && dir.z.abs() > 0.5;
            // A guessed gap holds only where the cloth is near the body:
            // baggy trousers and a robe's skirt hang free (minus 2 cm they
            // grew the legs fat).
            let guessed = g.guessed_on(axis.part);
            if guessed && (shell - gap) / r0 > drape_limit(stylized) {
                return Some(Fill {
                    center: f.center,
                    dir,
                    radius: r0,
                    start: r0,
                    weight: 0.0,
                    two_sided: false,
                    piece,
                    part: axis.part,
                    guessed,
                    draped: true,
                });
            }
            // Where the cloth rests on the body its shell is good evidence.
            let (gap, confidence) = if resting && guessed {
                (
                    gap.min(prior_gap(GarmentKind::Tight, g.layer, height)),
                    (2.0 * confidence).min(0.6),
                )
            } else {
                (gap, confidence)
            };
            // Room to move: a body right under the cloth at rest pokes
            // through it once the skinned arms and hips swing.
            let gap = gap.max(MOTION_GAP * height);
            let limit = shell - clearance;
            let mut confidence = confidence;
            let radius = if two_sided {
                (shell - gap).clamp(0.8 * r0, grow * r0).min(limit)
            } else {
                // A default gap under loose cloth says nothing: the shell of
                // baggy trousers minus 2 cm is not a thigh (they grew the
                // legs by the full 15 %). Only a shell near the body (within
                // 15 %, fading out by 30 %) is evidence.
                let ratio = (shell - gap) / r0;
                if ratio > 1.3 {
                    return None;
                }
                confidence *= 1.0 - smooth((ratio - 1.15) / 0.15);
                let r = (shell - gap).min(1.15 * r0).min(limit);
                if r <= r0 || confidence <= 0.0 {
                    return None;
                }
                r
            };
            // The shell along the ray is not the nearest cloth everywhere
            // (a sloping shoulder seam above a horizontal ray): growing
            // stops where any scan surface comes within 3/4 of the gap.
            let keep = 0.75 * gap;
            let free = |r: f64| {
                scan.query
                    .closest_point(a(f.center + dir * r), keep, None)
                    .is_none()
            };
            let radius = if radius > r0 && !free(radius) {
                if !free(r0) {
                    return None;
                }
                let (mut lo, mut hi) = (r0, radius);
                for _ in 0..8 {
                    let mid = 0.5 * (lo + hi);
                    if free(mid) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                if lo <= r0 + 1e-4 {
                    return None;
                }
                lo
            } else {
                radius
            };
            Some(Fill {
                center: f.center,
                dir,
                radius,
                start: r0,
                weight: confidence,
                two_sided,
                piece,
                part: axis.part,
                guessed,
                draped: false,
            })
        })
        .collect();
     
    let mut count: BTreeMap<(usize, u8), (usize, usize)> = BTreeMap::new();
    for f in out.iter().flatten().filter(|f| f.guessed) {
        let c = count.entry((f.piece, f.part as u8)).or_default();
        c.0 += f.draped as usize;
        c.1 += 1;
    }
    out.into_iter()
        .map(|f| {
            let f = f?;
            if f.draped {
                return None;
            }
            // Limbs only: over the torso cloth that stands off (an apron
            // vest) still rests on the chest, ray by ray.
            if f.guessed && f.part != Part::Torso {
                let (draped, total) = count[&(f.piece, f.part as u8)];
                if draped as f64 > 0.4 * total as f64 {
                    return None;
                }
            }
            Some(f)
        })
        .collect()
}

/// Per piece: how many body vertices got a target and how far it moves
/// them (mm, + outward).
pub fn fill_report(fill: &[Option<Fill>], seg: &Segmentation) -> Value {
    let mut by: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for f in fill.iter().flatten() {
        by.entry(seg.pieces[f.piece].clone())
            .or_default()
            .push((f.radius - f.start) * 1000.0);
    }
    by.into_iter()
        .map(|(k, mut d)| {
            d.sort_by(f64::total_cmp);
            let n = d.len();
            (
                k,
                json!({
                    "vertices": n,
                    "mean_mm": (d.iter().sum::<f64>() / n as f64 * 10.0).round() / 10.0,
                    "p10_mm": (d[n / 10] * 10.0).round() / 10.0,
                    "p90_mm": (d[(n * 9 / 10).min(n - 1)] * 10.0).round() / 10.0,
                }),
            )
        })
        .collect::<serde_json::Map<String, Value>>()
        .into()
}

/// Perimeter of the convex hull of 2-D points.
fn hull_perimeter(mut pts: Vec<[f64; 2]>) -> f64 {
    pts.sort_by(|p, q| p[0].total_cmp(&q[0]).then(p[1].total_cmp(&q[1])));
    pts.dedup();
    if pts.len() < 3 {
        return 0.0;
    }
    let cross = |o: [f64; 2], p: [f64; 2], q: [f64; 2]| {
        (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    };
    let mut hull: Vec<[f64; 2]> = vec![];
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &[f64; 2]>> = if pass == 0 {
            Box::new(pts.iter())
        } else {
            Box::new(pts.iter().rev())
        };
        for &p in iter {
            while hull.len() >= start + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    (0..hull.len())
        .map(|i| {
            let (p, q) = (hull[i], hull[(i + 1) % hull.len()]);
            (p[0] - q[0]).hypot(p[1] - q[1])
        })
        .sum()
}

 pub fn girths(axes: &Axes, points: &[(usize, V)], height: f64) -> Value {
    let slab = 0.006 * height;
    let find = |name: &str| axes.axes.iter().position(|x| x.name == name);
    let mut out = serde_json::Map::new();
    if let Some(k) = find("torso") {
        let torso = &axes.axes[k];
        let len = torso.length();
        let thighs: Vec<usize> = ["thigh_l", "thigh_r"]
            .iter()
            .filter_map(|n| find(n))
            .collect();
        for (name, t) in [
            ("chest", len - 0.06 * height),
            ("waist", 0.42 * len),
            ("hips", 0.015 * height),
        ] {
            let ring: Vec<[f64; 2]> = points
                .iter()
                .filter(|(j, _)| *j == k || thighs.contains(j))
                .filter_map(|(_, p)| {
                    let f = torso.frame(*p);
                    ((f.t - t).abs() < slab).then(|| [p.x, p.z])
                })
                .collect();
            out.insert(
                name.into(),
                json!((hull_perimeter(ring) * 1000.0).round() / 10.0),
            );
        }
    }
    for (name, fraction) in [
        ("neck", 0.5),
        ("upper_arm", 0.45),
        ("forearm", 0.3),
        ("thigh", 0.3),
        ("calf", 0.35),
    ] {
        let ks: Vec<usize> = if name == "neck" {
            find("neck").into_iter().collect()
        } else {
            ["l", "r"]
                .iter()
                .filter_map(|s| find(&format!("{name}_{s}")))
                .collect()
        };
        let values: Vec<f64> = ks
            .iter()
            .map(|&k| {
                let axis = &axes.axes[k];
                let t = fraction * axis.length();
                let ring: Vec<[f64; 2]> = points
                    .iter()
                    .filter(|(j, _)| *j == k)
                    .filter_map(|(_, p)| {
                        let f = axis.frame(*p);
                        ((f.t - t).abs() < slab).then(|| axis.plane(f.radial))
                    })
                    .collect();
                hull_perimeter(ring)
            })
            .filter(|g| *g > 0.0)
            .collect();
        if !values.is_empty() {
            let m = values.iter().sum::<f64>() / values.len() as f64;
            out.insert(name.into(), json!((m * 1000.0).round() / 10.0));
        }
    }
    out.into()
}

/// Body vertices as (segment, position) pairs for [`girths`].
pub fn body_points(axes: &Axes, positions: &[Vec3]) -> Vec<(usize, V)> {
    positions
        .iter()
        .enumerate()
        .filter_map(|(i, p)| Some((axes.vertex_axis[i]?, v(*p))))
        .collect()
}

/// Gap report rows.
pub fn gap_report(gaps: &[GarmentGap]) -> Value {
    gaps.iter()
        .map(|g| {
            (
                g.piece.clone(),
                json!({
                    "gap_mm": (g.gap_m * 10000.0).round() / 10.0,
                    "source": g.source,
                    "kind": g.kind,
                    "confidence": g.confidence,
                    "two_sided": g.two_sided,
                    "tight": g.tight,
                    "layer": g.layer,
                    "covers": g.covers,
                    "elsewhere_gap_mm": (g.elsewhere.0 * 10000.0).round() / 10.0,
                    "openings": g.openings.iter().map(|o| json!({
                        "axis": o.axis,
                        "t_m": (o.t_m * 1000.0).round() / 1000.0,
                        "skin_distal": o.skin_distal,
                        "garment_radius_mm": (o.garment_radius_m * 10000.0).round() / 10.0,
                        "skin_radius_mm": (o.skin_radius_m * 10000.0).round() / 10.0,
                        "body_taper": (o.body_taper * 1000.0).round() / 1000.0,
                        "gap_mm": (o.gap_m * 10000.0).round() / 10.0,
                        "ease": (o.ease * 1000.0).round() / 1000.0,
                    })).collect::<Vec<_>>(),
                }),
            )
        })
        .collect::<serde_json::Map<String, Value>>()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circle(r: f64, n: usize) -> Vec<[f64; 2]> {
        (0..n)
            .map(|i| {
                let a = i as f64 / n as f64 * std::f64::consts::TAU;
                [0.3 + r * a.cos(), -0.1 + r * a.sin()]
            })
            .collect()
    }

    #[test]
    fn ring_radius_ignores_the_axis_offset_and_rejects_partial_rings() {
        assert!((ring(&circle(0.05, 64)).unwrap() - 0.05).abs() < 1e-9);
        // Three quarters of the round still measure the full radius.
        let most: Vec<[f64; 2]> = circle(0.05, 64).into_iter().take(48).collect();
        assert!((ring(&most).unwrap() - 0.05).abs() < 1e-9);
        let half: Vec<[f64; 2]> = circle(0.05, 64).into_iter().take(32).collect();
        assert!(ring(&half).is_none());
        assert!(ring(&circle(0.05, 8)).is_none());
    }

    #[test]
    fn hull_perimeter_of_a_circle() {
        let r = 0.1;
        let p = hull_perimeter(circle(r, 360));
        assert!((p - std::f64::consts::TAU * r).abs() < 1e-3);
    }

    /// A sleeve over an upper arm: arm radius 5 cm tapering to 4 cm at the
    /// elbow, sleeve radius 7 cm ending half-way. The measured gap is the
    /// sleeve radius minus the arm radius at the sleeve's ring station, not
    /// at the skin's: MakeHuman's own taper carries the skin ring across the
    /// opening.
    #[test]
    fn gap_is_measured_at_a_sleeve_opening_with_the_body_taper() {
        let height = 1.75;
        let width = BIN * height;
        // Axis along +x from the shoulder; one limb axis only.
        let axis = Axis::new(
            "upper_arm_l".into(),
            Part::UpperArm,
            V::zeros(),
            V::new(0.3, 0.0, 0.0),
        );
        let axes = Axes {
            axes: vec![axis],
            vertex_axis: vec![],
        };
        let arm = |x: f64| 0.05 - 0.01 * (x / 0.3);
        let sleeve_end = 0.15;
        let mut points = vec![];
        let mut body = vec![];
        let mut x = 0.0;
        while x < 0.3 {
            for i in 0..48 {
                let ang = i as f64 / 48.0 * std::f64::consts::TAU;
                let (c, s) = (ang.cos(), ang.sin());
                let r = if x < sleeve_end { 0.07 } else { arm(x) };
                let key = if x < sleeve_end { 0 } else { SKIN };
                points.push((0, V::new(x, r * c, r * s), key));
                // The (slim) prior body has the same taper, 20 % thinner.
                let b = 0.8 * arm(x);
                body.push([x, b * c, b * s]);
            }
            x += width / 4.0;
        }
        let axes = Axes {
            vertex_axis: vec![Some(0); body.len()],
            ..axes
        };
        let seg = Segmentation {
            vertex_label: vec![],
            triangle_label: vec![],
            triangle_piece: vec![],
            pieces: vec!["shirt".into()],
            piece_hair: vec![false],
            piece_layer: vec![0],
            piece_tight: vec![false],
            piece_gap_mm: vec![None],
            piece_rigid: vec![false],
            strand: vec![],
            report: Value::Null,
        };
        let samples = Samples { points };
        let (gaps, _) = estimate_gaps(&seg, &axes, &samples, &body, 0.004, height);
        let g = &gaps[0];
        assert_eq!(g.source, GapSource::Measured);
        assert!(g.two_sided);
        let o = &g.openings[0];
        assert!(o.skin_distal);
        // Expected: sleeve 7 cm minus the arm radius at the sleeve's ring
        // station (the skin ring carried back by the taper).
        let expected = 0.07 - arm(o.t_m);
        assert!(
            (o.gap_m - expected).abs() < 0.0015,
            "gap {} expected {}",
            o.gap_m,
            expected
        );
        assert!((g.gap_m - o.gap_m).abs() < 1e-12);
        // Measured at a sleeve: holds for the arms; over the torso a loose
        // garment's cuff gap is capped by its kind's default (8 mm for a
        // shirt), and a skirt part over the thighs gets the default.
        assert_eq!(g.on(Part::Torso).0, 0.008f64.min(g.gap_m));
        assert!(g.guessed_on(Part::Torso) && !g.guessed_on(Part::Forearm));
        assert_eq!(g.on(Part::Forearm).0, g.gap_m);
        // (a shirt: a medium garment's default, 8 mm, a soft two-sided target)
        let (gap, _, two_sided) = g.on(Part::Thigh);
        assert!((gap - 0.008).abs() < 1e-12 && two_sided);
        assert!(o.ease > 1.3 && o.ease < 1.7);

        // No opening: the guidance's estimate, else a one-sided default.
        let none = Samples { points: vec![] };
        let mut seg2 = seg;
        seg2.piece_gap_mm = vec![Some(25.0)];
        let g = &estimate_gaps(&seg2, &axes, &none, &body, 0.004, height).0[0];
        assert_eq!(g.source, GapSource::Guidance);
        assert!((g.gap_m - 0.025).abs() < 1e-12);
        seg2.piece_gap_mm = vec![None];
        let g = &estimate_gaps(&seg2, &axes, &none, &body, 0.004, height).0[0];
        assert_eq!(g.source, GapSource::Prior);
        assert_eq!(g.kind, GarmentKind::Medium);
        assert!(g.two_sided && g.confidence < 0.5);
        assert!((g.gap_m - 0.008).abs() < 1e-12);
        // A huge measured gap is capped (7 cm at 1.75 m).
        assert!(prior_gap(GarmentKind::Plate, 9, height) <= 0.025 + 1e-12);
        assert_eq!(GarmentKind::of("armour", false), GarmentKind::Padded);
        assert_eq!(GarmentKind::of("Robe", false), GarmentKind::Loose);
        assert_eq!(GarmentKind::of("wrist_guards", true), GarmentKind::Tight);
        assert_eq!(GarmentKind::of("shirt", false), GarmentKind::Medium);
    }
}
