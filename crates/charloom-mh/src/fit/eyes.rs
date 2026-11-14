//! The eyes as a mirrored pair, and MakeHuman's eye openings.
//!
//! - [`openings`]: MakeHuman's eye openings, optically (where a front ray
//!   hits the eyeball before the lid skin): corner and lid-extreme vertices
//!   and the upper / lower lid-margin vertex loops.
//! - [`symmetrize`]: the marked eyes become one mirrored pair about the face
//!   midplane: the same opening shape, size and tilt on both sides, placed
//!   symmetrically. Only a disagreement beyond marking noise that both eyes'
//!   marks show consistently is kept (a whole eye shifted, both width and
//!   height larger), and reported.
//! - [`opening_warp`]: a smooth face-plane warp (thin-plate spline through
//!   the corners and lid points) carries MakeHuman's whole lid onto the
//!   marked outline before the fine fit, applied to the start shape *and* to
//!   the fit's reference shape. Pulling only the landmark vertices moved the
//!   front lid margin and left the lid rim and socket lining behind: the
//!   eyeball then showed through MakeHuman's own small opening, framed by a
//!   skin shelf between it and the marks.
use crate::geom::{V, proper_rotation, v};
use crate::prior::{PartKind, Prior, Region};
use crate::scan::build_query;
use charloom_core::Vec3;
use nalgebra::{DMatrix, Matrix3};
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// One MakeHuman eye opening (rest, optical), body vertex indices.
pub struct Opening {
    pub side: &'static str,
    /// Inner corner, outer corner, top and bottom lid-margin vertices.
    pub extremes: [usize; 4],
         pub upper: Vec<usize>,
    pub lower: Vec<usize>,
}

/// MakeHuman's eye openings, optically: on a dense grid of front-facing
/// rays, the opening is where the eyeball is hit before the lid skin. Its
/// extremes are the true corners and lid margins (MakeHuman's own eye
/// shape); its upper and lower edge per column give the margin loops.
pub fn openings(prior: &Prior) -> Vec<Opening> {
    let body = prior.body();
    let rest = &body.prim.positions;
    let Some(eyes) = prior.meshes.iter().find(|m| m.kind == PartKind::Eyes) else {
        return vec![];
    };
    let body_query = build_query(rest, &body.prim.triangles);
    let nearest = |p: (f64, f64, f64)| -> usize {
        let q = V::new(p.0, p.1, p.2);
        (0..rest.len())
            .min_by(|&a0, &b0| {
                (v(rest[a0]) - q)
                    .norm_squared()
                    .total_cmp(&(v(rest[b0]) - q).norm_squared())
            })
            .unwrap()
    };
    let mut out = vec![];
    for (side, joint) in [("l", "eye.L"), ("r", "eye.R")] {
        let Some(&j) = prior.by_name.get(joint) else {
            continue;
        };
        let keep: Vec<bool> = eyes
            .influences
            .iter()
            .map(|row| row.iter().any(|w| w.joint == j && w.weight > 0.5))
            .collect();
        let tris: Vec<[u32; 3]> = eyes
            .prim
            .triangles
            .iter()
            .filter(|t| t.iter().all(|&i| keep[i as usize]))
            .copied()
            .collect();
        if tris.is_empty() {
            continue;
        }
        let ball_query = build_query(&eyes.prim.positions, &tris);
        let ball: Vec<V> = eyes
            .prim
            .positions
            .iter()
            .zip(&keep)
            .filter(|(_, k)| **k)
            .map(|(p, _)| v(*p))
            .collect();
        let c = ball.iter().sum::<V>() / ball.len() as f64;
        let radius = ball.iter().map(|p| (p - c).norm()).fold(0.0, f64::max);
        let step = radius / 60.0;
        let mut opening: Vec<(f64, f64, f64)> = vec![];
        for iy in -90..=90 {
            for ix in -90..=90 {
                let (x, y) = (c.x + ix as f64 * step, c.y + iy as f64 * step);
                let origin = [x, y, c.z + radius * 4.0];
                let dir = [0.0, 0.0, -1.0];
                let Some(db) = ball_query.first_ray_distance(origin, dir, radius * 6.0) else {
                    continue;
                };
                if body_query
                    .first_ray_distance(origin, dir, radius * 6.0)
                    .is_none_or(|ds| db < ds)
                {
                    opening.push((x, y, origin[2] - db));
                }
            }
        }
        if opening.len() < 20 {
            continue;
        }
        let sign = if side == "l" { 1.0 } else { -1.0 };
        let lateral = |p: &(f64, f64, f64)| (p.0 - c.x) * sign;
        let inner = *opening
            .iter()
            .min_by(|a0, b0| lateral(a0).total_cmp(&lateral(b0)))
            .unwrap();
        let outer = *opening
            .iter()
            .max_by(|a0, b0| lateral(a0).total_cmp(&lateral(b0)))
            .unwrap();
        // Upper / lower edge of the opening in a column.
        let edge = |cx: f64, up: bool| -> Option<(f64, f64, f64)> {
            opening
                .iter()
                .filter(|p| (p.0 - cx).abs() < step * 1.5)
                .max_by(|a0, b0| {
                    if up {
                        a0.1.total_cmp(&b0.1)
                    } else {
                        b0.1.total_cmp(&a0.1)
                    }
                })
                .copied()
        };
        let cx = (inner.0 + outer.0) * 0.5;
        let (Some(top), Some(bottom)) = (edge(cx, true), edge(cx, false)) else {
            continue;
        };
        let extremes = [inner, outer, top, bottom].map(nearest);
        let mut loops = [vec![extremes[0]], vec![extremes[0]]];
        for k in 1..24 {
            let x = inner.0 + (outer.0 - inner.0) * k as f64 / 24.0;
            for (l, up) in loops.iter_mut().zip([true, false]) {
                if let Some(p) = edge(x, up) {
                    let i = nearest(p);
                    if !l.contains(&i) {
                        l.push(i);
                    }
                }
            }
        }
        for l in &mut loops {
            if !l.contains(&extremes[1]) {
                l.push(extremes[1]);
            }
        }
        let [upper, lower] = loops;
        out.push(Opening {
            side,
            extremes,
            upper,
            lower,
        });
    }
    out
}

/// The four opening marks of an eye.
pub const OPENING_MARKS: [&str; 4] = ["inner", "outer", "top", "bottom"];
/// The six lid-margin marks of an eye.
pub const LID_MARKS: [&str; 6] = [
    "upper_1", "upper_2", "upper_3", "lower_1", "lower_2", "lower_3",
];

/// The face's left/right mirror plane: `normal · p = offset`, the normal
/// pointing to the character's left (+X).
#[derive(Debug, Clone, Copy)]
pub struct Midplane {
    pub normal: V,
    pub offset: f64,
}

impl Midplane {
    /// From the paired face marks, one vote per feature (the eye openings,
    /// else the eye centres; the mouth corners): the normal is the mean
    /// left − right direction, the offset the median of the feature
    /// midpoints and the nose tip, so one eye marked off-centre does not
    /// move the plane with it. `None` without a pair.
    pub fn from_landmarks(lm: &BTreeMap<String, Vec3>) -> Option<Self> {
        // One vote per feature (the opening marks of an eye are one
        // feature): left − right direction and midpoint.
        let mut votes: Vec<(V, V)> = vec![];
        let opening: Vec<(V, V)> = OPENING_MARKS
            .iter()
            .filter_map(|k| {
                Some((
                    v(*lm.get(&format!("eye_l_{k}"))?),
                    v(*lm.get(&format!("eye_r_{k}"))?),
                ))
            })
            .collect();
        if !opening.is_empty() {
            let n = opening.len() as f64;
            let dir = opening.iter().fold(V::zeros(), |s, (l, r)| s + (l - r)) / n;
            let mid = opening
                .iter()
                .fold(V::zeros(), |s, (l, r)| s + (l + r) * 0.5)
                / n;
            votes.push((dir, mid));
        } else if let (Some(l), Some(r)) = (lm.get("eye_l"), lm.get("eye_r")) {
            votes.push((v(*l) - v(*r), (v(*l) + v(*r)) * 0.5));
        }
        if let (Some(l), Some(r)) = (lm.get("mouth_l"), lm.get("mouth_r")) {
            votes.push((v(*l) - v(*r), (v(*l) + v(*r)) * 0.5));
        }
        if votes.is_empty() {
            return None;
        }
        let sum = votes
            .iter()
            .filter(|(d, _)| d.norm() > 1e-9)
            .fold(V::zeros(), |s, (d, _)| s + d.normalize());
        if sum.norm() < 1e-9 {
            return None;
        }
        let normal = sum.normalize();
        // A plausible face frame only: facing +Z, left at +X.
        if normal.x < 0.8 {
            return None;
        }
        let mut offsets: Vec<f64> = votes.iter().map(|(_, m)| normal.dot(m)).collect();
        if let Some(n) = lm.get("nose_tip") {
            offsets.push(normal.dot(&v(*n)));
        }
        // With only two votes their mean; otherwise the median.
        offsets.sort_by(f64::total_cmp);
        let m = offsets.len();
        let offset = if m % 2 == 1 {
            offsets[m / 2]
        } else {
            (offsets[m / 2 - 1] + offsets[m / 2]) * 0.5
        };
        Some(Self { normal, offset })
    }

    pub fn mirror(&self, p: V) -> V {
        p - self.normal * (2.0 * (self.normal.dot(&p) - self.offset))
    }

    pub fn mirror_dir(&self, d: V) -> V {
        d - self.normal * (2.0 * self.normal.dot(&d))
    }
}

/// Marking noise: a whole-eye shift up to this (or a tenth of the eye's
/// width, whichever is larger) is not evidence of an asymmetric face.
const SHIFT_NOISE_M: f64 = 0.002;
const SHIFT_NOISE_OF_WIDTH: f64 = 0.1;
/// Size disagreement up to this ratio is marking noise; beyond it only when
/// width *and* height disagree in the same direction.
const SIZE_NOISE: f64 = 1.15;

/// The two marked eyes as one mirrored pair about the face midplane (see
/// the module notes). Marks of one side missing on the other are mirrored
/// over; an eye with no opening marks at all takes the other eye's,
/// mirrored. Other landmarks pass through.
pub fn symmetrize(lm: &BTreeMap<String, Vec3>) -> (BTreeMap<String, Vec3>, Value) {
    let mut out = lm.clone();
    let Some(plane) = Midplane::from_landmarks(lm) else {
        return (out, json!({"applied": false, "reason": "no face midplane"}));
    };
    let get = |side: &str, k: &str| lm.get(&format!("eye_{side}_{k}")).map(|p| v(*p));
    let full = |side: &str| OPENING_MARKS.iter().all(|k| get(side, k).is_some());
    let marks: Vec<&str> = OPENING_MARKS
        .iter()
        .chain(LID_MARKS.iter())
        .copied()
        .collect();
    let mut put = |side: &str, k: &str, p: V| {
        let name = if k.is_empty() {
            format!("eye_{side}")
        } else {
            format!("eye_{side}_{k}")
        };
        out.insert(name, [p.x, p.y, p.z]);
    };
    let plane_json = json!({"normal": [plane.normal.x, plane.normal.y, plane.normal.z], "offset_m": plane.offset});
    match (full("l"), full("r")) {
        (true, true) => {}
        (true, false) | (false, true) => {
            let (from, to) = if full("l") { ("l", "r") } else { ("r", "l") };
            for k in &marks {
                if let Some(p) = get(from, k) {
                    put(to, k, plane.mirror(p));
                }
            }
            return (
                out,
                json!({"applied": true, "plane": plane_json, "mirrored_from": from}),
            );
        }
        (false, false) => {
            return (
                out,
                json!({"applied": false, "reason": "eye openings not marked", "plane": plane_json}),
            );
        }
    }
    let centre = |side: &str| {
        OPENING_MARKS
            .iter()
            .map(|k| get(side, k).unwrap())
            .sum::<V>()
            / 4.0
    };
    let (c_l, c_r) = (centre("l"), centre("r"));
    let c_rm = plane.mirror(c_r);
    // Shapes about each eye's centre, the right one mirrored onto the left.
    let shape_l = |k: &str| get("l", k).map(|p| p - c_l);
    let shape_r = |k: &str| get("r", k).map(|p| plane.mirror_dir(p - c_r));
    let width = |s: &dyn Fn(&str) -> Option<V>| (s("outer").unwrap() - s("inner").unwrap()).norm();
    let height = |s: &dyn Fn(&str) -> Option<V>| (s("top").unwrap() - s("bottom").unwrap()).norm();
    let (w_l, w_r) = (width(&shape_l), width(&shape_r));
    let (h_l, h_r) = (height(&shape_l), height(&shape_r));
    // Position: keep only a whole-eye shift beyond marking noise.
    let shift = c_l - c_rm;
    let tol = SHIFT_NOISE_M.max(SHIFT_NOISE_OF_WIDTH * (w_l + w_r) * 0.5);
    let kept_shift = if shift.norm() > tol {
        shift * (1.0 - tol / shift.norm())
    } else {
        V::zeros()
    };
    // Size: kept only where width and height both disagree, same way.
    let (rw, rh) = ((w_l / w_r.max(1e-9)).ln(), (h_l / h_r.max(1e-9)).ln());
    let kept_log = if rw * rh > 0.0 {
        let k = rw.abs().min(rh.abs()) - SIZE_NOISE.ln();
        k.max(0.0) * rw.signum()
    } else {
        0.0
    };
    let (f_l, f_r) = ((kept_log * 0.5).exp(), (-kept_log * 0.5).exp());
    let mid = (c_l + c_rm) * 0.5;
    let (cl_new, crm_new) = (mid + kept_shift * 0.5, mid - kept_shift * 0.5);
    let mut disagreement = serde_json::Map::new();
    for k in &marks {
        let s = match (shape_l(k), shape_r(k)) {
            (Some(a0), Some(b0)) => {
                disagreement.insert((*k).to_string(), json!((a0 - b0).norm()));
                (a0 + b0) * 0.5
            }
            (Some(s), None) | (None, Some(s)) => s,
            (None, None) => continue,
        };
        put("l", k, cl_new + s * f_l);
        put("r", k, plane.mirror(crm_new + s * f_r));
    }
    // Coarse eye centres follow the same pairing.
    if let (Some(el), Some(er)) = (lm.get("eye_l"), lm.get("eye_r")) {
        let m = (v(*el) + plane.mirror(v(*er))) * 0.5;
        put("l", "", m + kept_shift * 0.5);
        put("r", "", plane.mirror(m - kept_shift * 0.5));
    }
    let symmetric = kept_shift.norm() == 0.0 && kept_log == 0.0;
    (
        out,
        json!({
            "applied": true,
            "plane": plane_json,
            "symmetric": symmetric,
            "centre_disagreement_m": shift.norm(),
            "centre_disagreement_vector_m": [shift.x, shift.y, shift.z],
            "shift_noise_m": tol,
            "kept_shift_m": kept_shift.norm(),
            "width_m": [w_l, w_r],
            "height_m": [h_l, h_r],
            "kept_size_ratio_l_over_r": kept_log.exp(),
            "shape_disagreement_m": disagreement,
        }),
    )
}

/// Eye-centre asymmetry the (symmetrized) marks keep: left centre minus the
/// mirrored right centre, with the plane; `None` without both openings.
pub fn kept_asymmetry(lm: &BTreeMap<String, Vec3>) -> Option<(Midplane, V)> {
    let plane = Midplane::from_landmarks(lm)?;
    let centre = |side: &str| -> Option<V> {
        let mut s = V::zeros();
        for k in OPENING_MARKS {
            s += v(*lm.get(&format!("eye_{side}_{k}"))?);
        }
        Some(s / 4.0)
    };
    Some((plane, centre("l")? - plane.mirror(centre("r")?)))
}

fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

 fn chord(p: V, inner: V, outer: V) -> f64 {
    let d = outer - inner;
    (p - inner).dot(&d) / d.norm_squared().max(1e-18)
}

 fn at_chord(points: &[V], inner: V, outer: V, f: f64) -> V {
    let fs: Vec<f64> = points.iter().map(|p| chord(*p, inner, outer)).collect();
    for k in 1..points.len() {
        if f <= fs[k] || k + 1 == points.len() {
            let span = fs[k] - fs[k - 1];
            let u = if span.abs() > 1e-12 {
                ((f - fs[k - 1]) / span).clamp(0.0, 1.0)
            } else {
                0.0
            };
            return points[k - 1] * (1.0 - u) + points[k] * u;
        }
    }
    points[0]
}

 pub fn lid_closing(
    prior: &Prior,
    openings: &[Opening],
    body: &[Vec3],
    worlds: &[nalgebra::Matrix4<f64>],
) -> Value {
    let mut upper_out = serde_json::Map::new();
    let mut lower_out = serde_json::Map::new();
    let middle = |loop_: &[usize]| -> Vec<usize> {
        let n = loop_.len();
        loop_[n / 4..(n - n / 4).max(n / 4 + 1).min(n)].to_vec()
    };
    for o in openings {
        let bone = |k: &str| {
            prior
                .by_name
                .get(&format!("{k}.{}", o.side.to_uppercase()))
                .copied()
        };
        let (Some(ju), Some(jl)) = (bone("orbicularis03"), bone("orbicularis04")) else {
            continue;
        };
        let (upper, lower) = (middle(&o.upper), middle(&o.lower));
        if upper.is_empty() || lower.is_empty() {
            continue;
        }
        // Mean margin angle about joint `j`'s local X, weighted by each
        // vertex's weight on `w_joint` (the optical margin loop also picks
        // rim and lining vertices hardly on the lid bones).
        let mean_angle = |j: usize, verts: &[usize], w_joint: usize| -> Option<f64> {
            let w = &worlds[j];
            let r = w.fixed_view::<3, 3>(0, 0).into_owned();
            let t = V::new(w[(0, 3)], w[(1, 3)], w[(2, 3)]);
            let (sum, total) = verts.iter().fold((0.0, 0.0), |(s0, t0), &i| {
                let p = r.transpose() * (v(body[i]) - t);
                let wt = body_influences(prior, i, w_joint).max(0.05);
                (s0 + p.z.atan2(p.y) * wt, t0 + wt)
            });
            (total > 1e-9).then(|| sum / total)
        };
        // The margin's (weight-weighted) mean lid weight: linear blend
        // skinning turns a vertex by about its weight's share of the bone's
        // turn. Dividing by it in full overshot (the lid skin above, fully
        // on the bone, folded down past the eye); by its square root (at
        // most ×1.4) the margins meet.
        let lag = |verts: &[usize], j: usize| -> f64 {
            let (s2, s1) = verts.iter().fold((0.0, 0.0), |(a0, b0), &i| {
                let w = body_influences(prior, i, j);
                (a0 + w * w, b0 + w)
            });
            (s2 / s1.max(1e-9)).clamp(0.5, 1.0).sqrt()
        };
        let (Some(uu), Some(lu), Some(ul), Some(ll)) = (
            mean_angle(ju, &upper, ju),
            mean_angle(ju, &lower, jl),
            mean_angle(jl, &upper, ju),
            mean_angle(jl, &lower, jl),
        ) else {
            continue;
        };
        let (up_turn, low_turn) = (
            0.7 * (lu - uu) / lag(&upper, ju),
            0.3 * (ul - ll) / lag(&lower, jl),
        );
        upper_out.insert(o.side.into(), json!(up_turn.clamp(-1.5, 1.5)));
        lower_out.insert(o.side.into(), json!(low_turn.clamp(-1.5, 1.5)));
    }
    json!({"upper": upper_out, "lower": lower_out})
}

/// Weight of body vertex `i` on joint `j` in MakeHuman's skinning.
fn body_influences(prior: &Prior, i: usize, j: usize) -> f64 {
    prior.body().influences[i]
        .iter()
        .filter(|x| x.joint == j)
        .map(|x| x.weight as f64)
        .sum()
}

/// Thin-plate spline (2D, with affine part) through point displacements.
struct Tps {
    centres: Vec<(f64, f64)>,
    weights: Vec<V>,
    affine: [V; 3],
}

impl Tps {
    fn kernel(r2: f64) -> f64 {
        if r2 < 1e-18 { 0.0 } else { 0.5 * r2 * r2.ln() }
    }

    /// Coordinates normalised to about unit size; `smoothing` relaxes the
    /// interpolation (close, slightly conflicting marks).
    fn fit(points: &[(f64, f64)], d: &[V], smoothing: f64) -> Option<Self> {
        let n = points.len();
        let m = n + 3;
        let mut l = DMatrix::<f64>::zeros(m, m);
        for i in 0..n {
            for j in 0..n {
                let (a0, b0) = (points[i], points[j]);
                l[(i, j)] = Self::kernel((a0.0 - b0.0).powi(2) + (a0.1 - b0.1).powi(2));
            }
            l[(i, i)] += smoothing;
            for (k, x) in [1.0, points[i].0, points[i].1].into_iter().enumerate() {
                l[(i, n + k)] = x;
                l[(n + k, i)] = x;
            }
        }
        let mut rhs = DMatrix::<f64>::zeros(m, 3);
        for i in 0..n {
            for c in 0..3 {
                rhs[(i, c)] = d[i][c];
            }
        }
        let sol = l.lu().solve(&rhs)?;
        let row = |i: usize| V::new(sol[(i, 0)], sol[(i, 1)], sol[(i, 2)]);
        Some(Self {
            centres: points.to_vec(),
            weights: (0..n).map(row).collect(),
            affine: [row(n), row(n + 1), row(n + 2)],
        })
    }

    fn eval(&self, p: (f64, f64)) -> V {
        let mut s = self.affine[0] + self.affine[1] * p.0 + self.affine[2] * p.1;
        for (c, w) in self.centres.iter().zip(&self.weights) {
            s += w * Self::kernel((p.0 - c.0).powi(2) + (p.1 - c.1).powi(2));
        }
        s
    }
}

/// Displacement of every body vertex (unwelded) that carries MakeHuman's eye
/// openings onto the marked outlines, in the posed frame and as the same
/// change of MakeHuman's rest shape (the fit's reference): (posed, rest,
/// report). Per eye a thin-plate spline in the eye's face plane maps the
/// corners and the lid margins at the marked lid points (or the top and
/// bottom marks) onto the marks; it acts fully around the opening, fading
/// out at about twice its size and behind the eyeball (socket back).
pub fn opening_warp(
    prior: &Prior,
    openings: &[Opening],
    posed: &[Vec3],
    landmarks: &BTreeMap<String, Vec3>,
) -> (Vec<V>, Vec<V>, Value) {
    let body = prior.body();
    let rest = &body.prim.positions;
    let n = posed.len();
    let mut d_posed = vec![V::zeros(); n];
    let mut d_rest = vec![V::zeros(); n];
    let mut report = serde_json::Map::new();
    let head: Vec<bool> = (0..n)
        .map(|i| prior.region[prior.dominant_joint(prior.body, i)] == Some(Region::Head))
        .collect();
    let plane = Midplane::from_landmarks(landmarks);
    for o in openings {
        let side = o.side;
        let mark = |k: &str| landmarks.get(&format!("eye_{side}_{k}")).map(|p| v(*p));
        let (Some(ti), Some(to)) = (mark("inner"), mark("outer")) else {
            continue;
        };
        let p = |i: usize| v(posed[i]);
        let [si, so, st, sb] = o.extremes.map(p);
        let upper: Vec<V> = o.upper.iter().map(|&i| p(i)).collect();
        let lower: Vec<V> = o.lower.iter().map(|&i| p(i)).collect();
         
        let mut pairs: Vec<(V, V)> = vec![(si, ti), (so, to)];
        for (part, extreme, src_extreme, line) in [
            ("upper", "top", st, &upper),
            ("lower", "bottom", sb, &lower),
        ] {
            let lid: Vec<V> = (1..=3)
                .filter_map(|k| mark(&format!("{part}_{k}")))
                .collect();
            if lid.len() == 3 {
                for t in lid {
                    pairs.push((at_chord(line, si, so, chord(t, ti, to)), t));
                }
            } else if let Some(t) = mark(extreme) {
                pairs.push((src_extreme, t));
            }
        }
        if pairs.len() < 4 {
            continue;
        }
        // Face-plane frame of MakeHuman's opening (posed).
        let e1 = (so - si).normalize();
        let e3 = (V::z() - e1 * e1.z).normalize();
        let e2 = e3.cross(&e1);
        let centre = (si + so + st + sb) / 4.0;
        let a_s = (so - si).norm() * 0.5;
        let a_t = (to - ti).norm() * 0.5;
        let uv = |q: V| ((q - centre).dot(&e1) / a_s, (q - centre).dot(&e2) / a_s);
        let mut points: Vec<(f64, f64)> = pairs.iter().map(|(s, _)| uv(*s)).collect();
        let mut disp: Vec<V> = pairs.iter().map(|(s, t)| t - s).collect();
        // Reach: the larger of the two openings; the target's height too.
        let grow = (a_t / a_s).max(1.0);
        let half_height = pairs
            .iter()
            .map(|(_, t)| ((t - centre).dot(&e2) / a_s).abs())
            .fold(0.0, f64::max);
        let (rx, ry) = (grow, (0.75 * grow).max(half_height * 1.3));
        // The fade-out band is at least three times the largest move wide:
        // a narrower one folded the skin under the eyes into a crease.
        let largest = disp.iter().map(|d| d.norm()).fold(0.0, f64::max) / a_s;
        let band = (3.0 * largest / rx.min(ry)).max(0.85);
         
        let ring = 1.15 + band;
        for k in 0..16 {
            let a0 = k as f64 * std::f64::consts::TAU / 16.0;
            points.push((rx * ring * a0.cos(), ry * ring * a0.sin()));
            disp.push(V::zeros());
        }
        let Some(tps) = Tps::fit(&points, &disp, 1e-3) else {
            continue;
        };
        // Nothing reaches the face midline (both eyes' fields met there and
        // pinched the nose bridge): the weight rises from the midplane to
        // full at the inner corner.
        let inner_side = plane.map(|m| m.normal.dot(&si) - m.offset);
        let weight = |q: V| -> f64 {
            let (u, w2) = uv(q);
            let depth = (q - centre).dot(&e3) / a_s;
            let rho = ((u / rx).powi(2) + (w2 / ry).powi(2)).sqrt();
            let midline = match (plane, inner_side) {
                (Some(m), Some(g)) if g.abs() > 1e-6 => {
                    // Distance from the midplane towards this eye.
                    let t = (m.normal.dot(&q) - m.offset) * g.signum();
                    smoothstep(t / g.abs())
                }
                _ => 1.0,
            };
            (1.0 - smoothstep((rho - 1.15) / band))
                * (1.0 - smoothstep((-depth - 1.0) / 1.0))
                * midline
        };
         
        let members: Vec<(usize, f64)> = (0..n)
            .filter(|&i| head[i])
            .map(|i| (i, weight(p(i))))
            .filter(|(_, w)| *w > 0.0)
            .collect();
        if members.len() < 10 {
            continue;
        }
        let (cp, cr) = members
            .iter()
            .fold((V::zeros(), V::zeros()), |(s0, s1), (i, _)| {
                (s0 + p(*i), s1 + v(rest[*i]))
            });
        let (cp, cr) = (cp / members.len() as f64, cr / members.len() as f64);
        let mut m = Matrix3::zeros();
        let mut var = 0.0;
        for (i, _) in &members {
            let (dp, dr) = (p(*i) - cp, v(rest[*i]) - cr);
            m += dp * dr.transpose();
            var += dr.norm_squared();
        }
        let r = proper_rotation(m);
        let s = ((r.transpose() * m).trace() / var.max(1e-20)).max(1e-6);
        let to_rest = r.transpose() / s;
        let mut moved_max = 0.0f64;
        for (i, w) in &members {
            let d = tps.eval(uv(p(*i))) * *w;
            moved_max = moved_max.max(d.norm());
            d_posed[*i] += d;
            d_rest[*i] += to_rest * d;
        }
        report.insert(
            side.to_string(),
            json!({
                "points": pairs.len(),
                "vertices": members.len(),
                "max_displacement_m": moved_max,
                "point_displacements_m": pairs.iter().map(|(s, t)| (t - s).norm()).collect::<Vec<_>>(),
                "fade_ring": ring,
                "width_ratio": a_t / a_s,
            }),
        );
    }
    (d_posed, d_rest, Value::Object(report))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eye(side: f64, dx: f64, w: f64) -> Vec<(String, Vec3)> {
        let s = if side > 0.0 { "l" } else { "r" };
        let c = [side * (0.03 + dx), 1.5, 0.1];
        [
            ("inner", [-w * 0.5, 0.0]),
            ("outer", [w * 0.5, 0.001]),
            ("top", [0.0, 0.004]),
            ("bottom", [0.0, -0.003]),
        ]
        .into_iter()
        .map(|(k, [x, y])| (format!("eye_{s}_{k}"), [c[0] + side * x, c[1] + y, c[2]]))
        .collect()
    }

    #[test]
    fn noisy_eyes_become_a_mirrored_pair() {
        let mut lm: BTreeMap<String, Vec3> = BTreeMap::new();
        lm.extend(eye(1.0, 0.0005, 0.020));
        lm.extend(eye(-1.0, -0.0005, 0.022));
        lm.insert("nose_tip".into(), [0.0, 1.47, 0.13]);
        let (out, report) = symmetrize(&lm);
        assert_eq!(report["symmetric"], json!(true));
        let plane = Midplane::from_landmarks(&lm).unwrap();
        for k in OPENING_MARKS {
            let (l, r) = (out[&format!("eye_l_{k}")], out[&format!("eye_r_{k}")]);
            assert!((v(l) - plane.mirror(v(r))).norm() < 1e-9, "{k}");
        }
        let w = |s: &str| {
            (v(out[&format!("eye_{s}_outer")]) - v(out[&format!("eye_{s}_inner")])).norm()
        };
        assert!((w("l") - 0.021).abs() < 1e-4);
    }

    #[test]
    fn a_clearly_shifted_eye_keeps_its_shift_beyond_noise() {
        let mut lm: BTreeMap<String, Vec3> = BTreeMap::new();
        lm.extend(eye(1.0, 0.0, 0.020));
        lm.extend(eye(-1.0, 0.0, 0.020));
        for k in OPENING_MARKS {
            lm.get_mut(&format!("eye_l_{k}")).unwrap()[0] += 0.006;
        }
        lm.insert("nose_tip".into(), [0.0, 1.47, 0.13]);
        lm.insert("mouth_l".into(), [0.02, 1.42, 0.11]);
        lm.insert("mouth_r".into(), [-0.02, 1.42, 0.11]);
        let (out, report) = symmetrize(&lm);
        assert_eq!(report["symmetric"], json!(false));
        // 6 mm further out on the left: 4 mm beyond the noise kept.
        let dx = out["eye_l_inner"][0] + out["eye_r_inner"][0];
        assert!((dx - 0.004).abs() < 1e-5, "{dx}");
    }

    #[test]
    fn one_marked_eye_is_mirrored_over() {
        let mut lm: BTreeMap<String, Vec3> = BTreeMap::new();
        lm.extend(eye(1.0, 0.0, 0.020));
        lm.insert("eye_l".into(), [0.03, 1.5, 0.1]);
        lm.insert("eye_r".into(), [-0.03, 1.5, 0.1]);
        lm.insert("mouth_l".into(), [0.02, 1.42, 0.11]);
        lm.insert("mouth_r".into(), [-0.02, 1.42, 0.11]);
        let (out, report) = symmetrize(&lm);
        assert_eq!(report["mirrored_from"], json!("l"));
        assert!((out["eye_r_outer"][0] + out["eye_l_outer"][0]).abs() < 1e-6);
    }

    #[test]
    fn thin_plate_spline_interpolates() {
        let pts = [(-1.0, 0.0), (1.0, 0.0), (0.0, 0.5), (0.0, -0.4), (0.5, 0.4)];
        let d: Vec<V> = pts
            .iter()
            .map(|p| V::new(p.0 * 0.1, p.1 * p.1, 0.01))
            .collect();
        let t = Tps::fit(&pts, &d, 0.0).unwrap();
        for (p, d) in pts.iter().zip(&d) {
            assert!((t.eval(*p) - d).norm() < 1e-9);
        }
    }
}
