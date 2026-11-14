//! Stage 2 — body *and* rig fit in one step.
//!
//! The posed MakeHuman body is deformed onto the scan with an
//! as-similar-as-possible (ARAP + bounded local scale) energy:
//! * visible **skin** is an equality target (closest scan point);
//! * **garment / hair** is a one-sided envelope: the body is pushed inside
//!   the cloth by `clearance`, never pulled out to it;
//! * hidden regions keep the MakeHuman prior shape through the regulariser.
//!
//! Each data target is gathered over the *whole* scan and classified by the
//! label of the triangle actually hit, so occluded body never snaps to a
//! distant visible-skin patch. Triangles that start to flip locally stiffen.
//!
//! The skeleton is then refitted from the same deformation: every joint is
//! carried by a weighted local affine fit of the vertices it drives, bones are
//! re-aimed at their child, and inverse bind matrices are recomputed. Skin
//! weights, the joint hierarchy, finger and face bones are kept as authored.
use crate::geom::{
    V, Weld, a, adjacency, distortion, face_normal, percentiles, proper_rotation, rotation_between,
    v, vertex_normals,
};
use crate::prior::{PartKind, Prior, Region};
use crate::scan::{Scan, build_query};
use crate::segment::{Label, Segmentation};
use crate::symmetry::{self, Evidence, Mirror};
use anyhow::{Result, ensure};
use charloom_core::{Transform, Vec3};
use nalgebra::{Matrix3, Matrix4, Rotation3, UnitQuaternion};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct FitParams {
    pub iterations: usize,
    pub stiffness_start: f64,
    pub stiffness_end: f64,
    pub clearance_m: f64,
    pub skin_weight: f64,
    pub envelope_weight: f64,
    /// Local scale range of the shape regulariser. Cartoon subjects (big
    /// hands, a round belly) need more than a realistic body.
    pub scale_range: (f64, f64),
    /// Body vertices (unwelded) that take no data at all and follow the
    /// shape regulariser only: fingers a creature hand does not have
    /// (`creature.rs`), which would merge into the scan's neighbouring
    /// fingers. Empty: none.
    pub no_data: Vec<bool>,
    /// Weight of the symmetric prior (`symmetry.rs`): a torso, arm, leg or
    /// scalp vertex with weak evidence (under cloth, no data, low trust) is
    /// pulled towards its mirror partner's shape: always where the partner
    /// is visible skin, otherwise only if this side is further from
    /// MakeHuman's own shape. 0 disables it; the asymmetry is reported
    /// either way.
    pub symmetry: f64,
    /// Body-size targets per unwelded body vertex (`bodysize.rs`): under a
    /// garment the body sits *shell − gap* along its centreline ray. Empty:
    /// none.
    pub fill: Vec<Option<crate::bodysize::Fill>>,
    /// Weight of those targets (times each target's confidence); a weak
    /// pull, applied only where this iteration found no skin.
    pub fill_weight: f64,
         pub mapped_hands: [bool; 2],
    /// The scan's own hands replace MakeHuman's (`hand_keep::HandRoute::KeepScan`):
    /// the hand's bounded fit keeps its own vertices only, and the forearm
    /// follows the scan's skin up to the wrist.
    pub kept_hands: bool,
    /// The scan's finger radius at each marked digit's tip and middle
    /// knuckle landmarks (`[left, right]`, MakeHuman digits 1..5, `[tip,
    /// mid]`), from the finger mapping.
    pub finger_tip_radius: [[[Option<f64>; 2]; 5]; 2],
    /// The body under the clothes keeps MakeHuman's shape (`fit/hidden.rs`):
    /// torso and limb vertices away from visible skin take no per-vertex
    /// fill, tight-cloth or mirror pulls, only the one-sided cloth push, and
    /// their shape regulariser is `1 +` this many times stiffer (local
    /// shrinking 8 % at most).
    /// 0: the per-vertex fit everywhere.
    pub hidden_stiffness: f64,
    /// Fold fairing on the whole body, not only the face: wherever the fit
    /// makes a crease MakeHuman's own surface does not have (an elbow fold,
    /// a belly crease, a knuckle ridge), those vertices relax towards
    /// MakeHuman's surface detail carried by the local frame.
    pub body_fairing: bool,
         pub body_clearance: bool,
    /// Marked eye openings: MakeHuman's lids warped onto the marked outline
    /// (start and lid reference) and their margins held there in the face
    /// plane (`eyes::opening_warp`). Off: only the landmark pulls.
    pub eye_openings: bool,
    /// Human first (`Config::human_first`): smooth capped residual, none
    /// under the clothes; head scale within `HUMAN_HEAD_SCALE`.
    pub human_first: bool,
    /// Body from the guidance (`Config::guided_body`).
    pub guided_body: bool,
         pub feet_measured: bool,
         pub bare_feet: [bool; 2],
    /// Guided: the visible skin's residual cap, share of the height
    /// (`shape::guided_limits`).
    pub skin_residual: f64,
    /// `subject.proportions` is `extreme` (guided).
    pub extreme: bool,
    /// Marked body outlines (`shape::outline_frames`, unwelded members) with
    /// their confidence: each station's cross-section is remapped across the
    /// view so its edges land on the marked ones.
    pub outlines: Vec<(Vec<(usize, f64)>, V, V, V, f64, f64, f64)>,
    /// The bodysuit edits' silhouettes (`fit::silhouette`): when given,
    /// they replace the outline rows in the fit.
    pub contours: Vec<crate::guide::SuitContour>,
    pub head_contours: Vec<crate::guide::SuitContour>,
    pub head_fallback: Option<(String, Vec<Vec3>)>,
    pub anatomical_head: Option<anatomical_head::Input>,
    pub anatomical_head_zero: bool,
    /// Free head (`fit::sculpt`): after the fit, every vertex of the head
    /// that sees the scan's skin is sculpted onto it; the measured skull,
    /// eye openings and landmarks hold.
    pub free_head: bool,
    /// Hold GPT-visible scalp during fitting and bridge only genuinely
    /// hidden exterior head vertices after sculpting.
    pub hidden_head: bool,
    /// Absolute graft gap in the scan's original working frame. The shape
    /// fit can change prior.height(), but the graft threshold does not.
    pub hidden_neck_limit_m: f64,
    /// Registration (`fit::register`): the body from one coarse-to-fine
    /// solve on MakeHuman's reposed rest mesh; the staged fit below is
    /// skipped.
    pub registration: bool,
    /// Joints marked in one view (landmark name, ray origin, direction).
    pub joint_rays: Vec<(String, Vec3, Vec3)>,
}

/// Human first: the largest residual of visible skin from MakeHuman's
/// shaped, posed body, share of the height (1.5 cm at 1.75 m).
pub const HUMAN_RESIDUAL: f64 = 0.0086;
 pub const GUIDED_SKIN_RESIDUAL: f64 = 0.04;
/// Human first: Laplacian smoothing passes on the residual (removes lumps,
/// folds and creases; keeps broad shape).
const HUMAN_SMOOTHING: usize = 12;
const GUIDED_SMOOTHING: usize = 0;
 pub const HUMAN_HEAD_SCALE: (f64, f64) = (0.87, 1.15);
/// Guided: the face marks size the head (anime faces' wider-set eyes ask
/// for more), still within a human-like range.
pub const GUIDED_HEAD_SCALE: (f64, f64) = (0.85, 1.3);

 const OUTLINE_FIT_WEIGHT: f64 = 2.0;

/// Reach of a marked outline along the body from its station, share of the
/// height (8.75 cm at 1.75 m): the residual fades out over it.
const OUTLINE_REACH: f64 = 0.05;
/// Weight of the silhouette pull (`fit::silhouette`), as the outline rows'.
const SILHOUETTE_WEIGHT: f64 = 2.0;
/// The farthest a body outline is pulled to the suit's (share of the
/// height, 7 cm at 1.75 m).
const SILHOUETTE_REACH: f64 = 0.04;
/// Distance along the body (share of the height, 1.4 cm at 1.75 m) within
/// which a marked outline row counts in full against farther ones.
const OUTLINE_NEAR: f64 = 0.008;
/// Free head: the marked eye openings kept out of the sculpt, radius as a
/// share of the corners' distance (the opening, its lid margins and lash
/// line).
const EYE_HOLE: f64 = 0.85;

pub struct BodyFit {
    /// Fitted rest positions of every body vertex (unwelded order).
    pub positions: Vec<Vec3>,
         pub pinned_head_rig_reference: Option<Vec<Vec3>>,
    /// Diagnostic colour per body vertex: what drove it in the last iteration.
    pub diagnostic: Vec<[f32; 4]>,
    /// Head cavity (mouth interior, nostrils, ear canals) per body vertex.
    pub cavity: Vec<bool>,
    /// Per body vertex, 1 where the texture bake must not take the scan:
    /// around a marked ear outline (`ears::Ears::bake_exclusion`).
    pub ear_exclude: Vec<f64>,
    /// Per body vertex in [0, 1]: a foot inside footwear (`fit/feet.rs`),
    /// skin the scan cannot show.
    pub shod_foot: Vec<f64>,
    pub report: Value,
}

mod ears;
pub mod eyes;
pub mod eye_fairing;
mod feet;
mod guard;
mod hands;
mod hidden;
pub mod hidden_head;
mod sculpt;
mod silhouette;
mod body_field;
mod suit_head;
mod head_registration;
mod register;
 
mod x110_head;
pub mod anatomical_head;

fn conjugate_gradient(
    adj: &[Vec<(u32, f64)>],
    diag_extra: &[f64],
    rhs: &[V],
    x: &mut [V],
    iterations: usize,
) -> f64 {
    let n = x.len();
    let degree: Vec<f64> = adj
        .iter()
        .zip(diag_extra)
        .map(|(r, d)| r.iter().map(|e| e.1).sum::<f64>() + d)
        .collect();
    let apply = |y: &[V]| -> Vec<V> {
        (0..n)
            .into_par_iter()
            .map(|i| {
                adj[i]
                    .iter()
                    .fold(y[i] * degree[i], |s, &(j, w)| s - y[j as usize] * w)
            })
            .collect()
    };
     
    let dot = |p: &[V], q: &[V]| {
        p.par_chunks(4096)
            .zip(q.par_chunks(4096))
            .map(|(p, q)| p.iter().zip(q).map(|(p, q)| p.dot(q)).sum::<f64>())
            .collect::<Vec<f64>>()
            .iter()
            .sum::<f64>()
    };
    let ax = apply(x);
    let mut r: Vec<V> = rhs.iter().zip(&ax).map(|(b, a)| b - a).collect();
    let mut z: Vec<V> = r.iter().zip(&degree).map(|(r, d)| r / *d).collect();
    let mut d = z.clone();
    let mut rz = dot(&r, &z);
    let reference = dot(rhs, rhs).sqrt().max(1e-12);
    for _ in 0..iterations {
        if dot(&r, &r).sqrt() / reference < 1e-7 {
            break;
        }
        let ad = apply(&d);
        let denom = dot(&d, &ad);
        if denom <= 1e-30 {
            break;
        }
        let alpha = rz / denom;
        for i in 0..n {
            x[i] += d[i] * alpha;
            r[i] -= ad[i] * alpha;
            z[i] = r[i] / degree[i];
        }
        let next = dot(&r, &z);
        let beta = next / rz.max(1e-300);
        for i in 0..n {
            d[i] = z[i] + d[i] * beta;
        }
        rz = next;
    }
    dot(&r, &r).sqrt() / reference
}

 const MOUTH_WARP_RANGE: (f64, f64) = (0.4, 1.0);

 fn mouth_warp(prior: &Prior, aligned: &[Vec3], landmarks: &BTreeMap<String, Vec3>) -> (Vec<V>, Value) {
    let body = prior.body();
    let rest = &body.prim.positions;
    let zero = vec![V::zeros(); rest.len()];
    let sets = face_landmark_sets(prior);
    let set = |name: &str| sets.iter().find(|(k, _)| *k == name).map(|(_, s)| s.clone());
    let (Some(l), Some(r), Some(tl), Some(tr)) =
        (set("mouth_l"), set("mouth_r"), landmarks.get("mouth_l"), landmarks.get("mouth_r"))
    else {
        return (zero, json!(null));
    };
    let centroid = |s: &[usize], pts: &[Vec3]| s.iter().fold(V::zeros(), |acc, &i| acc + v(pts[i])) / s.len().max(1) as f64;
    let (rl, rr) = (centroid(&l, rest), centroid(&r, rest));
    let width = (rl - rr).norm();
    let head: Vec<usize> = (0..rest.len())
        .filter(|&i| prior.region[prior.dominant_joint(prior.body, i)] == Some(Region::Head))
        .collect();
    let spread = |pts: &[Vec3]| {
        let m = centroid(&head, pts);
        (head.iter().map(|&i| (v(pts[i]) - m).norm_squared()).sum::<f64>() / head.len().max(1) as f64).sqrt()
    };
    let scale = spread(aligned) / spread(rest).max(1e-9);
    let marked = (v(*tl) - v(*tr)).norm();
    let ratio = (marked / (width * scale).max(1e-9)).clamp(MOUTH_WARP_RANGE.0, MOUTH_WARP_RANGE.1);
    if ratio > 0.97 {
        return (zero, json!({"ratio": ratio, "applied": false}));
    }
    let centre = (rl + rr) * 0.5;
    let out = rest
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if prior.region[prior.dominant_joint(prior.body, i)] != Some(Region::Head) {
                return V::zeros();
            }
            let d = v(*p) - centre;
            let x = ((d.norm() / width - 0.6) / 0.8).clamp(0.0, 1.0);
            let w = 1.0 - x * x * (3.0 - 2.0 * x);
            V::new(d.x * (ratio - 1.0) * w, 0.0, 0.0)
        })
        .collect();
    (out, json!({"ratio": ratio, "applied": true, "makehuman_width_m": width * scale, "marked_width_m": marked}))
}

/// Where each vision-guidance face landmark sits on the MakeHuman body, as
/// body vertex sets (their centroid is the landmark): eyes from the lid
/// bones, mouth corners from the outermost lip vertices, the nose tip from
/// the most forward midline point between eyes and mouth.
pub fn face_landmark_sets(prior: &Prior) -> Vec<(&'static str, Vec<usize>)> {
    let body = prior.body();
    let rest = &body.prim.positions;
    let name = |i: usize| {
        prior.skeleton().joints[prior.dominant_joint(prior.body, i)]
            .name
            .to_lowercase()
    };
    let by_prefix = |prefixes: &[&str]| -> Vec<usize> {
        (0..rest.len())
            .filter(|&i| prefixes.iter().any(|p| name(i).starts_with(p)))
            .collect()
    };
    let mut out = vec![];
    let eye_l = by_prefix(&["orbicularis03.l", "orbicularis04.l"]);
    let eye_r = by_prefix(&["orbicularis03.r", "orbicularis04.r"]);
    let lips = by_prefix(&["oris"]);
    let centroid =
        |s: &[usize]| s.iter().fold(V::zeros(), |acc, &i| acc + v(rest[i])) / s.len().max(1) as f64;
    // MakeHuman ".L" is the character's left, +X when facing +Z; check it.
    let (mut eye_l, mut eye_r) = (eye_l, eye_r);
    if !eye_l.is_empty() && !eye_r.is_empty() && centroid(&eye_l).x < centroid(&eye_r).x {
        std::mem::swap(&mut eye_l, &mut eye_r);
    }
    if !lips.is_empty() {
        let extreme = |sign: f64| {
            let best = lips
                .iter()
                .copied()
                .max_by(|&a0, &b0| (rest[a0][0] * sign).total_cmp(&(rest[b0][0] * sign)))
                .unwrap();
            vec![best]
        };
        out.push(("mouth_l", extreme(1.0)));
        out.push(("mouth_r", extreme(-1.0)));
    }
    if !eye_l.is_empty() && !eye_r.is_empty() && !lips.is_empty() {
        let (el, er, mouth) = (centroid(&eye_l), centroid(&eye_r), centroid(&lips));
        let mid_x = (el.x + er.x) * 0.5;
        let eye_y = (el.y + er.y) * 0.5;
        if let Some(tip) = (0..rest.len())
            .filter(|&i| {
                let p = rest[i];
                (p[0] - mid_x).abs() < 0.004
                    && p[1] < eye_y
                    && p[1] > mouth.y
                    && prior.region[prior.dominant_joint(prior.body, i)] == Some(Region::Head)
            })
            .max_by(|&a0, &b0| rest[a0][2].total_cmp(&rest[b0][2]))
        {
            out.push(("nose_tip", vec![tip]));
        }
    }
    // Eye openings, optically (`eyes::openings`): their extremes are the
    // true corners and lid margins (MakeHuman's own eye shape).
    for o in eyes::openings(prior) {
        let names: [&'static str; 4] = if o.side == "l" {
            ["eye_l_inner", "eye_l_outer", "eye_l_top", "eye_l_bottom"]
        } else {
            ["eye_r_inner", "eye_r_outer", "eye_r_top", "eye_r_bottom"]
        };
        for (name, i) in names.into_iter().zip(o.extremes) {
            out.push((name, vec![i]));
        }
    }
    if !eye_l.is_empty() {
        out.push(("eye_l", eye_l));
    }
    if !eye_r.is_empty() {
        out.push(("eye_r", eye_r));
    }
    out
}

/// The marked eye openings with a margin (free head: `sculpt`, `graft`):
/// centre of the four marks, radius EYE_HOLE times the corners' distance.
pub fn eye_holes(landmarks: &BTreeMap<String, Vec3>) -> Vec<(V, f64)> {
    ["l", "r"]
        .iter()
        .filter_map(|side| {
            let m: Vec<V> = ["inner", "outer", "top", "bottom"]
                .iter()
                .map(|k| landmarks.get(&format!("eye_{side}_{k}")).map(|p| v(*p)))
                .collect::<Option<Vec<V>>>()?;
            let c = m.iter().fold(V::zeros(), |s, p| s + p) / 4.0;
            Some((c, EYE_HOLE * (m[0] - m[1]).norm()))
        })
        .collect()
}

 pub fn eye_scales(prior: &Prior, landmarks: &BTreeMap<String, Vec3>) -> BTreeMap<String, f64> {
    let sets = face_landmark_sets(prior);
    let get = |n: &str| {
        sets.iter()
            .find(|s| s.0 == n)
            .map(|s| set_centroid(&prior.body().prim.positions, &s.1))
    };
    let mut out = BTreeMap::new();
    for (side, joint) in [("l", "eye.L"), ("r", "eye.R")] {
        let (i, o) = (format!("eye_{side}_inner"), format!("eye_{side}_outer"));
        if let (Some(mi), Some(mo), Some(li), Some(lo)) =
            (get(&i), get(&o), landmarks.get(&i), landmarks.get(&o))
        {
            let ratio = (v(*li) - v(*lo)).norm() / (mi - mo).norm().max(1e-6);
            // The raw ratio: `refit_rig` keeps it no smaller than the head's
            // own scale and under its cap (see there).
            out.insert(joint.to_string(), ratio.clamp(0.5, 3.0));
        }
    }
    out
}

fn set_centroid(points: &[Vec3], set: &[usize]) -> V {
    set.iter().fold(V::zeros(), |acc, &i| acc + v(points[i])) / set.len().max(1) as f64
}

/// Head alignment: similarity ICP (scale, rotation, translation) of the
/// MakeHuman face onto the scan's facial skin, anchored by the vision
/// landmarks when given, then a smooth local warp that carries the
/// MakeHuman eyes, nose and mouth exactly onto the landmarks (stylised faces
/// place them far from a realistic head). Blended into the neck by
/// head-subtree weight.
pub fn align_head(
    prior: &Prior,
    scan: &Scan,
    seg: &Segmentation,
    posed: &[Vec3],
    landmarks: &BTreeMap<String, Vec3>,
    scale_limit: Option<(f64, f64)>,
    marks_first: bool,
) -> (Vec<Vec3>, Value) {
    let height = prior.height();
    let body = prior.body();
    let head = prior.joint("head");
    let weight: Vec<f64> = body
        .influences
        .iter()
        .map(|row| {
            row.iter()
                .filter(|w| prior.is_ancestor(head, w.joint))
                .map(|w| w.weight)
                .sum()
        })
        .collect();
    let weld = Weld::new(&body.bind, 1e-7);
    let normals = vertex_normals(posed, &body.prim.triangles, &weld);
    let face: Vec<usize> = (0..posed.len())
        .filter(|&i| {
            weight[i] > 0.99 && normals[i][2] > 0.3 && weld.representative[weld.id[i]] == i
        })
        .collect();
    let guided: Vec<(&str, Vec<usize>, V)> = face_landmark_sets(prior)
        .into_iter()
        .filter_map(|(n, set)| Some((n, set, v(*landmarks.get(n)?))))
        .collect();
    let mut total = Matrix4::<f64>::identity();
    let mut current: Vec<V> = face.iter().map(|&i| v(posed[i])).collect();
    let mut residual = Value::Null;
    let mut used = 0;
    for iteration in 0..20 {
        let tau = height * (0.03 - 0.02 * iteration as f64 / 19.0);
        let mut pairs: Vec<(V, V, f64, f64)> = face
            .par_iter()
            .zip(&current)
            .filter_map(|(&i, p)| {
                let n = (total.fixed_view::<3, 3>(0, 0) * v(normals[i])).normalize();
                let hit = scan.query.closest_point(a(*p), tau, Some((a(n), 0.5)))?;
                (seg.triangle_label[hit.source[1]] == Label::Skin)
                    .then(|| (*p, v(hit.point), hit.distance_m, 1.0))
            })
            .collect();
        // Trim the worst 20 % (lips, lids, stray hair).
        pairs.sort_by(|x, y| x.2.total_cmp(&y.2));
        pairs.truncate(pairs.len() * 4 / 5);
        used = pairs.len();
        // Landmarks: each worth a tenth of the surface evidence.
        // Guided: the face marks place the head (each as much as the whole
        // surface), the surface only refines it: matched surface first, a
        // flat anime face turned the head 0.6-0.8 rad.
        let lw = (pairs.len() as f64 * if marks_first { 1.0 } else { 0.1 }).max(10.0);
        for (_, set, target) in &guided {
            let c = (total * set_centroid(posed, set).push(1.0)).xyz();
            pairs.push((c, *target, (c - target).norm(), lw));
        }
        if pairs.len() < 50 && guided.len() < 3 {
            break;
        }
        let wsum: f64 = pairs.iter().map(|p| p.3).sum();
        let ca = pairs.iter().fold(V::zeros(), |s, p| s + p.0 * p.3) / wsum;
        let cb = pairs.iter().fold(V::zeros(), |s, p| s + p.1 * p.3) / wsum;
        let m = pairs.iter().fold(Matrix3::zeros(), |s, p| {
            s + (p.1 - cb) * (p.0 - ca).transpose() * p.3
        });
        let r = proper_rotation(m);
        let var = pairs
            .iter()
            .map(|p| (p.0 - ca).norm_squared() * p.3)
            .sum::<f64>()
            .max(1e-20);
        let scale = ((r.transpose() * m).trace() / var).clamp(0.97, 1.03);
        let step = from_rt(&(r * scale), cb - r * scale * ca);
        total = step * total;
        current = current.iter().map(|p| (step * p.push(1.0)).xyz()).collect();
        residual = percentiles(pairs.iter().filter(|p| p.3 == 1.0).map(|p| p.2).collect());
    }
    // Human first: the head's scale held within human limits, about the
    // aligned face's centre.
    if let Some((lo, hi)) = scale_limit {
        let s = total.fixed_view::<3, 3>(0, 0).determinant().cbrt();
        let want = s.clamp(lo, hi);
        if (want - s).abs() > 1e-9 && !current.is_empty() {
            let c = current.iter().fold(V::zeros(), |a, p| a + p) / current.len() as f64;
            let fix = Matrix4::new_translation(&c)
                * Matrix4::new_scaling(want / s)
                * Matrix4::new_translation(&-c);
            total = fix * total;
        }
    }
    let basis = total.fixed_view::<3, 3>(0, 0).into_owned();
    let scale = basis.determinant().cbrt();
    let angle = Rotation3::from_matrix_unchecked(proper_rotation(basis)).angle();
    let axis = Rotation3::from_matrix_unchecked(proper_rotation(basis)).axis().map(|a| [a.x, a.y, a.z]);
    let marks_before: BTreeMap<&str, f64> = guided
        .iter()
        .map(|(n, set, t)| (*n, (set_centroid(posed, set) - t).norm()))
        .collect();
     
    let accepted = (0.55..=1.9).contains(&scale) && (angle < 0.45 || marks_first);
    let mut aligned: Vec<Vec3> = if accepted {
        posed
            .iter()
            .zip(&weight)
            .map(|(p, w)| a(v(*p) + ((total * v(*p).push(1.0)).xyz() - v(*p)) * *w))
            .collect()
    } else {
        posed.to_vec()
    };
    let rigid_rotation = proper_rotation(basis);
    let input_centre = face.iter().fold(V::zeros(), |s, &i| s + v(posed[i])) / face.len().max(1) as f64;
    let placed_centre = if accepted { (total * input_centre.push(1.0)).xyz() } else { input_centre };
    let rigid = if accepted { from_rt(&rigid_rotation, placed_centre - rigid_rotation * input_centre) }
        else { Matrix4::identity() };
    // Local feature warp: Gaussian RBF towards the coarse features (eye
    // centres, nose tip, mouth corners). The eye-opening points are left to
    // the fit, which pulls the lids itself: warping a few lid vertices by
    // millimetres with a narrow kernel creased the skin around the eyes
    // before the fit even started.
    let guided: Vec<(&str, Vec<usize>, V)> = guided
        .into_iter()
        .filter(|g| g.0.matches('_').count() < 2)
        .collect();
    let mut warp = Value::Null;
    if guided.len() >= 2 {
        let sources: Vec<V> = guided
            .iter()
            .map(|(_, set, _)| set_centroid(&aligned, set))
            .collect();
        let residuals: Vec<V> = guided
            .iter()
            .zip(&sources)
            .map(|((_, _, t), s)| t - s)
            .collect();
        let eyes: Vec<V> = guided
            .iter()
            .filter(|g| g.0.starts_with("eye"))
            .map(|g| g.2)
            .collect();
        let span = if eyes.len() == 2 {
            (eyes[0] - eyes[1]).norm()
        } else {
            height * 0.04
        };
        let sigma = span * 0.7;
        let k = sources.len();
        let kernel = |d2: f64| (-d2 / (2.0 * sigma * sigma)).exp();
        let mut km = nalgebra::DMatrix::<f64>::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                // Slight regularisation: an approximating, smoother warp.
                km[(i, j)] = kernel((sources[i] - sources[j]).norm_squared())
                    + if i == j { 0.05 } else { 0.0 };
            }
        }
        if let Some(inv) = km.try_inverse() {
            let coeff: Vec<V> = (0..k)
                .map(|i| (0..k).fold(V::zeros(), |s, j| s + residuals[j] * inv[(i, j)]))
                .collect();
            for (p, w) in aligned.iter_mut().zip(&weight) {
                if *w <= 0.0 {
                    continue;
                }
                let d = (0..k).fold(V::zeros(), |s, i| {
                    s + coeff[i] * kernel((v(*p) - sources[i]).norm_squared())
                });
                *p = a(v(*p) + d * *w);
            }
            warp = json!({"sigma_m":sigma,"residuals_before_warp_m":guided.iter().zip(&residuals).map(|(g, r)| (g.0, r.norm())).collect::<BTreeMap<_, _>>()});
        }
    }
    let shift =
        (total * origin(&prior.rest_world[head]).push(1.0)).xyz() - origin(&prior.rest_world[head]);
    (
        aligned,
        json!({"accepted":accepted,"rigid_from_input":rigid.as_slice(),"rotation_axis":axis,"marks_before_m":marks_before,"face_vertices":face.len(),"pairs":used,"landmarks":guided.iter().map(|g| g.0).collect::<Vec<_>>(),"scale":scale,"rotation_rad":angle,"head_shift_m":shift.norm(),"final_pair_distance_m":residual,"feature_warp":warp}),
    )
}

pub fn fit_body(
    prior: &Prior,
    scan: &Scan,
    seg: &Segmentation,
    posed: &[Vec3],
    p: &FitParams,
    landmarks: &BTreeMap<String, Vec3>,
) -> Result<BodyFit> {
    let suit_measured_pose = posed;
    let (mut aligned, head_report) = align_head(
        prior,
        scan,
        seg,
        posed,
        landmarks,
        p.human_first
            .then_some(if p.guided_body { GUIDED_HEAD_SCALE } else { HUMAN_HEAD_SCALE }),
        p.guided_body,
    );
    let rigid_values: Vec<f64> = head_report["rigid_from_input"].as_array().unwrap().iter().map(|v|v.as_f64().unwrap()).collect();
    let head_rigid = Matrix4::from_column_slice(&rigid_values);
    // Marked eye openings: MakeHuman's whole lid (margin, rim, socket
    // lining) warped onto the marked outline, in the start shape and in the
    // fit's reference (`eyes::opening_warp`).
    let eye_openings = eyes::openings(prior);
    let (eye_warp, eye_warp_rest, eye_warp_report) = if p.eye_openings {
        eyes::opening_warp(prior, &eye_openings, &aligned, landmarks)
    } else {
        let zero = vec![V::zeros(); aligned.len()];
        (zero.clone(), zero, json!({}))
    };
    for (q, d) in aligned.iter_mut().zip(&eye_warp) {
        *q = a(v(*q) + d);
    }
    let posed = aligned.as_slice();
    let height = prior.height();
    let body = prior.body();
    let weld = Weld::new(&body.bind, 1e-7);
    let neighbours = adjacency(&body.prim.triangles, &weld);
    let n = weld.count;
    let rest: Vec<V> = weld.representative.iter().map(|&i| v(posed[i])).collect();
    let head_shape_rest: Vec<V> = weld.representative.iter()
        .map(|&i| (head_rigid * v(suit_measured_pose[i]).push(1.0)).xyz()).collect();
     
    let shape_rest: Vec<V> = weld
        .representative
        .iter()
        .map(|&i| v(body.prim.positions[i]))
        .collect();
    // The regulariser's reference: MakeHuman's rest with the eye openings
    // and the mouth's width warped to the marks (the rest shape itself stays
    // for the mirror partners, cavities and ears).
    let (mouth_warp_rest, mouth_warp_report) = mouth_warp(prior, posed, landmarks);
    let shape_ref: Vec<V> = weld
        .representative
        .iter()
        .zip(&shape_rest)
        .map(|(&i, q)| q + eye_warp_rest[i] + mouth_warp_rest[i])
        .collect();
    let mut x = rest.clone();
    let region: Vec<Region> = weld
        .representative
        .iter()
        .map(|&i| prior.region[prior.dominant_joint(prior.body, i)].unwrap_or(Region::Torso))
        .collect();
    let welded_triangles: Vec<[u32; 3]> = body
        .prim
        .triangles
        .iter()
        .map(|t| t.map(|i| weld.id[i as usize] as u32))
        .collect();
    let rest_positions: Vec<Vec3> = rest.iter().map(|q| a(*q)).collect();
    // Inversions are judged against MakeHuman's rest surface, so folds that
    // posing introduced are detected (and released) too.
    let shape_positions: Vec<Vec3> = shape_rest.iter().map(|q| a(*q)).collect();
    let rest_normals: Vec<V> = {
        let reference: Vec<Vec3> = shape_ref.iter().map(|q| a(*q)).collect();
        welded_triangles
            .iter()
            .map(|t| face_normal(&reference, t))
            .collect()
    };
    // Eye sockets, lids and the mouth interior have no counterpart on a
    // scan's painted face: weak data, stiff shape. Identified by MakeHuman's
    // own face-bone weights.
    let protected: Vec<bool> = weld
        .representative
        .iter()
        .map(|&i| {
            let name = prior.skeleton().joints[prior.dominant_joint(prior.body, i)]
                .name
                .to_lowercase();
            ["eye", "orbicularis", "oculi", "oris", "tongue", "lip"]
                .iter()
                .any(|p| name.starts_with(p))
        })
        .collect();

    // Head cavities (mouth, nostrils, eye sockets, ear canals) look back
    // into the head: a short ray along the normal hits the head itself.
    // They have no counterpart on the scan and receive no data at all.
    let rest_normals_v: Vec<Vec3> = {
        let unwelded: Vec<Vec3> = weld.id.iter().map(|&w| rest_positions[w]).collect();
        let n_u = vertex_normals(&unwelded, &body.prim.triangles, &weld);
        weld.representative.iter().map(|&i| n_u[i]).collect()
    };
    // Classified on MakeHuman's own rest surface, independent of the fit.
    let prior_query = build_query(&shape_positions, &welded_triangles);
    let prior_normals: Vec<Vec3> = {
        let unwelded: Vec<Vec3> = weld.id.iter().map(|&w| shape_positions[w]).collect();
        let n_u = vertex_normals(&unwelded, &body.prim.triangles, &weld);
        weld.representative.iter().map(|&i| n_u[i]).collect()
    };
    let cavity: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|i| {
            region[i] == Region::Head && {
                // Visibility: a cavity vertex (mouth interior, nostril, ear
                // canal) sees out of the body in almost no direction; concave
                // skin (nose sides, mouth corners, under the eyes) sees open
                // space in many. 17 rays on two cones around the normal.
                let nrm = v(prior_normals[i]);
                let t1 = nrm
                    .cross(&if nrm.x.abs() < 0.9 { V::x() } else { V::y() })
                    .normalize();
                let t2 = nrm.cross(&t1);
                let origin = a(shape_rest[i] + nrm * 1e-4);
                let mut escaped = 0;
                for k in 0..17 {
                    let dir = if k == 0 {
                        nrm
                    } else {
                        let (ring, j) = ((k - 1) / 8, (k - 1) % 8);
                        let tilt: f64 = if ring == 0 { 0.7 } else { 2.1 };
                        let phi = std::f64::consts::TAU * (j as f64 + 0.5 * ring as f64) / 8.0;
                        (nrm + (t1 * phi.cos() + t2 * phi.sin()) * tilt).normalize()
                    };
                    if prior_query
                        .first_ray_distance(origin, a(dir), height * 0.2)
                        .is_none()
                    {
                        escaped += 1;
                    }
                }
                escaped <= 2
            }
        })
        .collect();
    // With a marked eye opening, the lids must reach it: a stylised opening
    // can be far taller than MakeHuman's almond, more than a stiff,
    // similarity-only lid can stretch (lid residuals stayed 5–10 mm). Lid
    // vertices then get normal stiffness and a wide scale range; the mouth
    // keeps its protection.
    let marked_eyes = landmarks.contains_key("eye_l_top") || landmarks.contains_key("eye_r_top");
    let lid: Vec<bool> = weld
        .representative
        .iter()
        .map(|&i| {
            let name = prior.skeleton().joints[prior.dominant_joint(prior.body, i)]
                .name
                .to_lowercase();
            marked_eyes && ["orbicularis", "oculi"].iter().any(|p| name.starts_with(p))
        })
        .collect();
    let mut stiffness: Vec<f64> = (0..n)
        .map(|i| {
            if lid[i] {
                1.0
            } else if protected[i] {
                4.0
            } else {
                1.0
            }
        })
        .collect();
    let mut trust: Vec<f64> = (0..n)
        .map(|i| {
            if cavity[i] || p.no_data.get(weld.representative[i]) == Some(&true) {
                0.0
            } else if protected[i] {
                0.15
            } else if region[i] == Region::Hand {
                0.4
            } else {
                1.0
            }
        })
        .collect();
    let relax_iterations = 12;
    // Head-subtree skin weight per welded vertex: 1 on the face, scalp and
    // ears, fading through the neck.
    let head_joint = prior.joint("head");
    let head_weight: Vec<f64> = weld
        .representative
        .iter()
        .map(|&r| {
            body.influences[r]
                .iter()
                .filter(|w| prior.is_ancestor(head_joint, w.joint))
                .map(|w| w.weight)
                .sum::<f64>()
        })
        .collect();
    let face_vertex: Vec<bool> = (0..n)
        .map(|i| !cavity[i] && head_weight[i] > 0.99)
        .collect();
    // Ears: rigid patches (a similarity of MakeHuman's rest pinna).
    let mut ears = ears::Ears::detect(
        &shape_rest,
        &prior_normals,
        &neighbours,
        &welded_triangles,
        &face_vertex,
        &prior_query,
        height,
    );
    // Marked ear outlines shape the pinna with MakeHuman's ear targets.
    ears.attach_outline(
        &shape_rest,
        &neighbours,
        &prior.ear_targets,
        &weld.representative,
        landmarks,
    );
    let mut ear_report = Value::Null;
     
    let mut ear_placed: Vec<V> = vec![];
     
    let rest_detail: Vec<V> = (0..n)
        .map(|i| {
            neighbours[i]
                .iter()
                .fold(V::zeros(), |s, &j| s + shape_ref[i] - shape_ref[j as usize])
        })
        .collect();
     
    let fold_watch: Vec<bool> = (0..n)
        .map(|i| {
            face_vertex[i]
                || (p.body_fairing
                    && !cavity[i]
                    && !ears.member[i]
                    && matches!(region[i], Region::Torso | Region::Arm | Region::Leg))
        })
        .collect();
    let fit_guard = guard::Guard::new(
        &shape_ref,
        &welded_triangles,
        &rest_normals,
        &neighbours,
        &fold_watch,
        0.012 * height / 1.75,
    );
    let stage_length = 10;
    let mut guard_reports = vec![];
    // Recompute the same rotation-aware count at each boundary. This separates
    // fitting/sculpting defects from changes made after the fitted body exits.
    let audit_inversions = |x: &[V]| -> Value {
        let frames: Vec<_> = (0..n).map(|i| {
            let mut m = Matrix3::zeros();
            for &j in &neighbours[i] {
                m += (x[i]-x[j as usize])*(shape_ref[i]-shape_ref[j as usize]).transpose();
            }
            proper_rotation(m)
        }).collect();
        let positions: Vec<_> = x.iter().map(|p|a(*p)).collect();
        let mut regions: BTreeMap<String,usize> = BTreeMap::new();
        let mut ids=vec![];
        for (k,t) in welded_triangles.iter().enumerate() {
            if inverted(&face_normal(&positions,t),&carried(&frames,t,&rest_normals[k])) {
                *regions.entry(format!("{:?}",region[t[0] as usize])).or_default()+=1;
                ids.push(k);
            }
        }
        json!({"total":ids.len(),"by_region":regions,"triangle_ids":ids})
    };
    let mut stage_inversions=BTreeMap::new();
    stage_inversions.insert("aligned",audit_inversions(&x));
    crate::geom::dump_stage("fit-reference",weld.id.iter().map(|&w|a(shape_ref[w])));
     
    let skin_samples: Vec<usize> = {
        let all: Vec<usize> = (0..scan.weld.count)
            .map(|w| scan.weld.representative[w])
            .filter(|&r| seg.vertex_label[r] == Label::Skin)
            .collect();
        let stride = (all.len() / 40_000).max(1);
        all.into_iter().step_by(stride).collect()
    };
     
    let (own_skin_tri, own_skin_vertex) = {
        let start: Vec<Vec3> = rest.iter().map(|q| a(*q)).collect();
        let start_query = build_query(&start, &welded_triangles);
        let own_at = |q: Vec3| -> bool {
            start_query.closest_point(q, height * 0.15, None).is_some_and(|h| {
                let t = welded_triangles[h.source[1]];
                let k = (0..3)
                    .max_by(|&x, &y| h.barycentric[x].total_cmp(&h.barycentric[y]))
                    .unwrap_or(0);
                matches!(region[t[k] as usize], Region::Hand | Region::Foot)
            })
        };
        let tri: Vec<bool> = (0..scan.triangles.len())
            .into_par_iter()
            .map(|t| {
                seg.triangle_label[t] == Label::Skin && {
                    let c = scan.triangles[t]
                        .iter()
                        .fold(V::zeros(), |acc, &k| acc + v(scan.positions[k as usize]))
                        / 3.0;
                    own_at(a(c))
                }
            })
            .collect();
        let mut vert = vec![false; scan.positions.len()];
        let flags: Vec<(usize, bool)> = skin_samples
            .par_iter()
            .map(|&k| (k, own_at(scan.positions[k])))
            .collect();
        for (k, f) in flags {
            vert[k] = f;
        }
        (tri, vert)
    };
    // Vision landmarks stay attached during the fit (welded vertex sets).
    let landmark_sets: Vec<(&str, Vec<usize>, V)> = face_landmark_sets(prior)
        .into_iter()
        // A marked eye opening supersedes the coarse eye-centre constraint.
        .filter(|(name, _)| {
            !((*name == "eye_l" || *name == "eye_r")
                && landmarks.contains_key(&format!("{name}_inner")))
        })
        .filter_map(|(name, set)| {
            let target = v(*landmarks.get(name)?);
            let mut ids: Vec<usize> = set.iter().map(|&i| weld.id[i]).collect();
            ids.sort_unstable();
            ids.dedup();
            Some((name, ids, target))
        })
        .collect();
    // A single-vertex eye landmark (corner, lid margin) is a point load:
    // pulled alone, an eye corner stretched into a wedge that showed the
    // eyeball. Its surface neighbourhood follows with the same offset at a
    // falling weight. (Spreading the mouth corners pushed the lips out.)
    let landmark_radius = 0.006 * height / 1.75;
    let landmark_patches: Vec<Vec<(usize, f64)>> = landmark_sets
        .iter()
        .map(|(name, set, _)| {
            if set.len() != 1 || !name.starts_with("eye_") {
                return vec![];
            }
            let mut seed = vec![false; n];
            seed[set[0]] = true;
            let d = guard::surface_distance(&shape_rest, &neighbours, &seed, landmark_radius);
            (0..n)
                .filter(|&j| j != set[0] && d[j].is_finite() && !cavity[j])
                .map(|j| (j, 0.5 * (1.0 - guard::smoothstep(d[j] / landmark_radius))))
                .collect()
        })
        .collect();
    // Landmark vertices stay where the landmarks pull them; fairing works
    // around them.
    let mut landmark_vertex = vec![false; n];
    for (_, set, _) in &landmark_sets {
        if set.len() == 1 {
            landmark_vertex[set[0]] = true;
        }
    }
    // Warped lid margins hold their place in the face plane (depth free):
    // left to the data, skin and regulariser closed a stylised opening back
    // towards MakeHuman's (a third of the marked opening covered by lid).
    let margin_hold: Vec<(usize, V)> = {
        let mut ids: Vec<usize> = eye_openings
            .iter()
            .filter(|o| eye_warp_report.get(o.side).is_some())
            .flat_map(|o| o.upper.iter().chain(&o.lower).map(|&i| weld.id[i]))
            .filter(|&w| !landmark_vertex[w])
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids.into_iter().map(|w| (w, rest[w])).collect()
    };
    // Symmetric prior: mirror partners on MakeHuman's (shaped) rest surface,
    // per-bone rest frames from the skin weights.
    let mirror = Mirror::new(&shape_rest, height * 0.0006);
    let influence_rows: Vec<&[charloom_core::Influence]> = weld
        .representative
        .iter()
        .map(|&r| body.influences[r].as_slice())
        .collect();
    let parents: Vec<Option<usize>> = prior.skeleton().joints.iter().map(|j| j.parent).collect();
    // Hands (posed fingers) and feet (in shoes, mirroring copied a foot
    // poking through its laces onto the other foot) keep their own fit, and
    // so do the face's own parts: lids and lips, ears (rigid patches),
    // eye-landmark patches and cavities. The torso, limbs and scalp (under
    // hair) may borrow from the mirror side; visible skin is gated out by
    // its evidence.
    let mut own_shape = vec![false; n];
    for i in 0..n {
        own_shape[i] = matches!(region[i], Region::Hand | Region::Foot)
            || cavity[i]
            || protected[i]
            || lid[i]
            || ears.member[i]
            || landmark_vertex[i]
            || p.no_data.get(weld.representative[i]) == Some(&true);
    }
    // On the head only the scalp (MakeHuman's `head` bone; face muscles keep
    // the face's own fit), and nothing within reach of an ear: mirroring
    // the scalp around a rigid ear patch folded its collar.
    let ear_ring =
        guard::surface_distance(&shape_rest, &neighbours, &ears.member, 0.02 * height / 1.75);
    for i in 0..n {
        if region[i] == Region::Head
            && (ear_ring[i].is_finite()
                || prior.skeleton().joints
                    [prior.dominant_joint(prior.body, weld.representative[i])]
                .name
                    != "head")
        {
            own_shape[i] = true;
        }
    }
    for patch in &ears.patches {
        for &(j, _) in &patch.collar {
            own_shape[j] = true;
        }
    }
    for patch in &landmark_patches {
        for &(j, _) in patch {
            own_shape[j] = true;
        }
    }
     
    let foot_side: Vec<Option<u8>> = weld
        .representative
        .iter()
        .zip(&region)
        .map(|(&r, &g)| {
            let name = prior.skeleton().joints[prior.dominant_joint(prior.body, r)]
                .name
                .to_lowercase();
            match g {
                Region::Foot if name.ends_with(".l") => Some(0),
                Region::Foot if name.ends_with(".r") => Some(1),
                _ => None,
            }
        })
        .collect();
    let mut shod_feet = {
        let (lo, hi) = p.scale_range;
        let posed_normals: Vec<V> = rest_normals_v.iter().map(|q| v(*q)).collect();
        let toe: Vec<bool> = weld
            .representative
            .iter()
            .map(|&r| {
                prior.skeleton().joints[prior.dominant_joint(prior.body, r)]
                    .name
                    .to_lowercase()
                    .starts_with("toe")
            })
            .collect();
        feet::Feet::detect(
            &foot_side,
            &toe,
            &rest,
            &posed_normals,
            &shape_rest,
            &neighbours,
            scan,
            seg,
            height,
            if p.feet_measured {
                (1.0, 1.0)
            } else {
                (1.0 + (lo - 1.0) * 0.75, 1.0 + (hi - 1.0) * 0.4)
            },
            p.bare_feet,
        )
    };
     
    let mut hand_fit = {
        let hand_side: Vec<Option<u8>> = weld
            .representative
            .iter()
            .zip(&region)
            .map(|(&r, &g)| {
                let name = prior.skeleton().joints[prior.dominant_joint(prior.body, r)]
                    .name
                    .to_lowercase();
                match g {
                    Region::Hand if name.ends_with(".l") => Some(0),
                    Region::Hand if name.ends_with(".r") => Some(1),
                    _ => None,
                }
            })
            .collect();
        let no_data: Vec<bool> = weld
            .representative
            .iter()
            .map(|&r| p.no_data.get(r) == Some(&true))
            .collect();
        let names: Vec<&str> = prior
            .skeleton()
            .joints
            .iter()
            .map(|j| j.name.as_str())
            .collect();
        hands::Hands::detect(
            &hand_side,
            &no_data,
            &influence_rows,
            &names,
            &rest,
            &shape_rest,
            &neighbours,
            height,
            p.scale_range,
            p.mapped_hands,
            &p.finger_tip_radius,
            p.extreme,
            Some((scan, seg)),
        )
    };
     
    if p.kept_hands {
        for i in 0..n {
            if region[i] != Region::Hand {
                hand_fit.weight[i] = 0.0;
            }
        }
    }
    // The body under the clothes keeps MakeHuman's shape (fit/hidden.rs).
    let hidden_excluded: Vec<bool> = (0..n)
        .map(|i| own_shape[i] || shod_feet.weight[i] > 0.0 || hand_fit.weight[i] > 0.0)
        .collect();
    let mut hidden_seed = if p.hidden_stiffness > 0.0 {
        hidden::initial_seeds(
            &rest,
            &rest_normals_v,
            scan,
            seg,
            &skin_samples,
            &welded_triangles,
            height,
        )
    } else {
        vec![]
    };
    let mut hidden_w = if hidden_seed.is_empty() {
        vec![0.0; n]
    } else {
        hidden::weights(
            &hidden_seed,
            &shape_rest,
            &neighbours,
            &region,
            &hidden_excluded,
            height,
        )
    };
    let symmetric_member: Vec<bool> = (0..n)
        .map(|i| {
            let m = mirror.partner[i] as usize;
            mirror.valid[i] && !own_shape[i] && !own_shape[m]
        })
        .collect();
    // Evidence per vertex, a running average over iterations: skin data at
    // its trust, a cloth envelope (an upper bound only) at 0.25, none at 0.
    let mut evidence = vec![f64::NAN; n];
    let mut evidence_class = vec![Evidence::None; n];
    let pre_fit_asymmetry = {
        let frames0 = symmetry::vertex_frames(&influence_rows, &parents, &shape_rest, &rest);
        symmetry::unpose(&frames0, &rest)
    };
    let outline_fit: Vec<(Vec<(usize, f64)>, V, V, V, f64, f64, f64)> = p
        .outlines
        .iter()
        .map(|(members, c, across, along, lo, hi, conf)| {
            let mut ids: Vec<(usize, f64)> = members.iter().map(|&(u, w)| (weld.id[u], w)).collect();
            ids.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            ids.dedup_by_key(|m| m.0);
            (ids, *c, *across, *along, *lo, *hi, *conf)
        })
        .collect();
    let silhouette_views: Vec<silhouette::View> = p.contours.iter().map(silhouette::View::new).collect();
     
    let on_neck: Vec<bool> = weld
        .representative
        .iter()
        .map(|&i| prior.skeleton().joints[prior.dominant_joint(prior.body, i)].name.to_lowercase().starts_with("neck"))
        .collect();
    let head_region: Vec<bool> = (0..n).map(|i| region[i] == Region::Head || on_neck[i]).collect();
    let scalp: Vec<bool> = weld.representative.iter().enumerate().map(|(w,&i)| {
        let name = &prior.skeleton().joints[prior.dominant_joint(prior.body, i)].name;
        // The bald views also show the nape hidden under hair. Treating all
        // neck rows as a zero boundary leaves a notch under a held occiput.
        name == "head" || on_neck[w] && scan.query.closest_point(a(x[w]),height*0.03,None)
            .is_some_and(|hit| matches!(seg.triangle_label[hit.source[1]], Label::Hair | Label::Skin))
    }).collect();
    let skull: Vec<bool> = weld.representative.iter().map(|&i| {
        prior.skeleton().joints[prior.dominant_joint(prior.body,i)].name=="head"
            && body.influences[i].iter().filter(|w|
                prior.skeleton().joints[w.joint].name.to_lowercase().starts_with("neck"))
                .map(|w|w.weight).sum::<f64>()<=1e-6
    }).collect();
    let internal: Vec<bool> = (0..n).map(|i| cavity[i] || protected[i]).collect();
    let boundary: Vec<bool> = (0..n).map(|i| ears.member[i] || ears.collar[i] || lid[i] || landmark_vertex[i]).collect();
    let transition_ears:Vec<_>=(0..n).map(|i|ears.member[i]||ears.collar[i]||ear_ring[i].is_finite()).collect();
    let transition_allowed:Vec<_>=(0..n).map(|i|!internal[i] && !lid[i] && !landmark_vertex[i]).collect();
    ensure!(p.anatomical_head.is_none() || p.head_fallback.is_some(), "anatomical head requires the validated x110 route");
    // Registration takes the skull from its own solve: the suit-head hold
    // (70 s of flame's 100 s fit) is not built.
    let scalp_hold = if p.anatomical_head.is_some() || p.registration {
        suit_head::Hold::disabled(&head_shape_rest, scan.positions.len())
    } else { suit_head::Hold::new(&head_shape_rest, &x, &neighbours, &welded_triangles, &head_region,
        &scalp, &skull, &internal, &boundary, &transition_allowed, &transition_ears, &on_neck, scan, seg, &p.head_contours,
        p.hidden_head && p.head_fallback.is_none(), height) };
    for i in 0..n {
        if scalp_hold.gpt_scalp[i] { x[i] = scalp_hold.target[i]; }
    }
    scalp_hold.enforce(&mut x);
    let mut stage_start = x.clone();
    stage_inversions.insert("held_start",audit_inversions(&x));
    crate::geom::dump_stage("head-held-start",weld.id.iter().map(|&w|a(x[w])));
    crate::geom::dump_stage("head-bounded-target",weld.id.iter().map(|&w|a(scalp_hold.target[w])));
    crate::geom::dump_stage("head-rest-target", weld.id.iter().map(|&w|a(head_shape_rest[w])));
    crate::geom::dump_stage("head-visibility", weld.id.iter().map(|&w|[scalp_hold.classes[w] as f64, internal[w] as u8 as f64, scalp_hold.weight[w]]));
    let silhouette_eligible: Vec<bool> = (0..n)
        .map(|i| !ears.member[i] && !cavity[i] && !on_neck[i])
        .zip(0..n)
        .map(|(ok, i)| ok && match region[i] {
            Region::Torso | Region::Arm | Region::Leg => true,
             
            Region::Head => p.head_fallback.is_some() && scalp[i],
            Region::Hand | Region::Foot => false,
        })
        .collect();
    let registration_report = p.registration.then(|| {
        let representative = &scan.weld.representative;
        let scan_points: Vec<V> = representative.iter().map(|&r| v(scan.positions[r])).collect();
        let scan_normals: Vec<V> = representative.iter().map(|&r| v(scan.normals[r])).collect();
        let scan_skin: Vec<bool> = representative.iter().map(|&r| seg.vertex_label[r] == Label::Skin).collect();
        let face_bone = |i: usize| {
            let name = prior.skeleton().joints[prior.dominant_joint(prior.body, weld.representative[i])].name.to_lowercase();
            ["orbicularis", "oris", "levator", "risorius", "tongue", "jaw", "eye", "temporalis", "special", "zygomat",
                "depressor", "mentalis", "nasal", "procerus", "frontalis", "platysma", "buccinator", "masseter", "occipit"]
                .iter().any(|k| name.contains(k))
        };
        let core: Vec<bool> = region.iter().map(|r| *r == Region::Torso).collect();
        let lips: Vec<bool> = (0..n).map(|i| {
            let name = prior.skeleton().joints[prior.dominant_joint(prior.body, weld.representative[i])].name.to_lowercase();
            name.starts_with("oris") || name.starts_with("lip")
        }).collect();
        // Lip contact surfaces and socket linings face another part of the
        // head a few millimetres away: pulled onto the scan's painted face
        // they turned outwards and the mouth opened.
        let no_data: Vec<bool> = (0..n).into_par_iter().map(|i| {
            cavity[i] || p.no_data.get(weld.representative[i]) == Some(&true) || (protected[i] && {
                let nrm = v(prior_normals[i]);
                prior_query.first_ray_distance(a(shape_rest[i] + nrm * 1e-4), a(nrm), 0.006 * height / 1.75).is_some()
            })
        }).collect();
        let hand_trust = std::env::var("MH_REG_HAND_TRUST").ok().and_then(|v| v.parse().ok()).unwrap_or(0.6);
        let trust: Vec<f64> = (0..n).map(|i| if protected[i] && !lid[i] { 0.4 } else if region[i] == Region::Hand { hand_trust } else { 1.0 }).collect();
        let stiffness: Vec<f64> = (0..n).map(|i| match region[i] {
            Region::Hand | Region::Foot => 6.0,
            // The head ends at the stiffness the face needs (its painted
            // relief is noise below that); the body goes softer.
            Region::Head => if no_data[i] || face_bone(i) { 1.5 } else { 1.43 },
            _ => 1.0,
        }).collect();
        let posed_shape: Vec<bool> = region.iter().map(|r| *r == Region::Hand).collect();
        let joint_rest: Vec<V> = prior.rest_world.iter().map(origin).collect();
        let roots: Vec<usize> = ["head", "foot.L", "foot.R"].iter().filter_map(|name| prior.by_name.get(*name).copied()).collect();
        let follow: Vec<bool> = (0..parents.len()).map(|q| roots.iter().any(|&r| r != q && prior.is_ancestor(r, q))).collect();
        let marks = landmark_sets.iter().map(|(name, ids, target)| register::Landmark {
            vertices: ids.clone(), target: *target, joint: false,
            weight: if name.starts_with("eye") || name.starts_with("mouth") || *name == "nose_tip" { 2.0 } else { 1.0 },
        }).collect();
         
        let mut marks: Vec<register::Landmark> = marks;
        for patch in &ears.patches {
            let side = if patch.outward.x > 0.0 { "l" } else { "r" };
            let pick = |key: &dyn Fn(&V) -> f64| patch.vertices.iter().copied()
                .max_by(|&i, &j| key(&shape_rest[i]).total_cmp(&key(&shape_rest[j])));
            for (name, vertex) in [("top", pick(&|q: &V| q.y)), ("lobe", pick(&|q: &V| -q.y)), ("back", pick(&|q: &V| -q.z))] {
                if let (Some(i), Some(target)) = (vertex, landmarks.get(&format!("ear_{side}_{name}"))) {
                    marks.push(register::Landmark { vertices: vec![i], target: v(*target), weight: 1.0, joint: false });
                }
            }
        }
        // The suit's head outlines, registered to the scan's head in each
        // view (front, side and three-quarter): the head's own views.
        let head_views: Vec<silhouette::View> = p.head_contours.iter().map(silhouette::View::new).collect();
        let on_head: Vec<bool> = region.iter().map(|r| *r == Region::Head).collect();
        let joints = crate::pose::LANDMARK_JOINTS.iter().filter_map(|(mark, joint)| {
            Some((*prior.by_name.get(*joint)?, v(*landmarks.get(*mark)?)))
        }).collect();
        let joint_rays = p.joint_rays.iter().filter_map(|(mark, origin, direction)| {
            let joint = crate::pose::LANDMARK_JOINTS.iter().find(|(m, _)| m == mark)?.1;
            Some((*prior.by_name.get(joint)?, v(*origin), v(*direction)))
        }).collect();
        let masks: Vec<Vec<bool>> = p.contours.iter().map(|c| register::suit_mask(c.camera.width, c.camera.height, &c.curves)).collect();
         
        let outside = |point: V, normal: V| {
            let side = normal.cross(&if normal.x.abs() < 0.8 { V::x() } else { V::y() }).normalize();
            let other = normal.cross(&side);
            [V::zeros(), side, -side, other, -other].iter().any(|t| {
                let d = (normal + t * 0.6).normalize();
                scan.query.first_ray_distance(a(point + d * 1e-4), a(d), 1.5).is_none()
            })
        };
        let (fitted, report) = register::fit(&register::Input {
            rest: &shape_rest, posed: &rest, triangles: &welded_triangles, neighbours: &neighbours,
            influences: &influence_rows, parents: &parents, follow: &follow, joint_rest: &joint_rest, stiffness: &stiffness,
            no_data: &no_data, trust: &trust, posed_shape: &posed_shape, lips: &lips, core: &core, scan_points: &scan_points, scan_normals: &scan_normals,
            scan_skin: &scan_skin, outside: &outside, views: &silhouette_views, masks: &masks, head_views: &head_views, head: &on_head, scalp: &scalp, landmarks: marks, joints, joint_rays,
        });
        x = fitted;
        crate::geom::dump_stage("registered", weld.id.iter().map(|&w| a(x[w])));
        report
    });
    let staged = !p.registration;
    let mut history = vec![];
    let mut last_kind = vec![u8::MAX; n];
    // Per-vertex rotations from the latest local step; a triangle is inverted
    // when it opposes its rest normal *carried by those rotations*, so a palm
    // turned 120° by the pose is not mistaken for a fold.
    let mut frames: Vec<Matrix3<f64>> = if p.registration {
        // Registration skips the staged local/global loop below. Its audit
        // still needs the fitted local rotations, not identity frames.
        (0..n).into_par_iter().map(|i| {
            proper_rotation(neighbours[i].iter().fold(Matrix3::zeros(), |s, &j| {
                let j = j as usize;
                s + (x[i] - x[j]) * (shape_ref[i] - shape_ref[j]).transpose()
            }))
        }).collect()
    } else { vec![Matrix3::identity(); n] };
    let mut reverse_hits = vec![0u32; n];
    for iteration in 0..if staged { p.iterations + relax_iterations } else { 0 } {
        let relaxing = iteration >= p.iterations;
        // Whatever has seen skin is visible: the hidden weight shrinks as
        // the fit reaches skin the pose had missed.
        if !hidden_seed.is_empty() && iteration > 0 {
            let mut grew = false;
            for i in 0..n {
                if evidence_class[i] == Evidence::Skin && !hidden_seed[i] {
                    hidden_seed[i] = true;
                    grew = true;
                }
            }
            if grew {
                hidden_w = hidden::weights(
                    &hidden_seed,
                    &shape_rest,
                    &neighbours,
                    &region,
                    &hidden_excluded,
                    height,
                );
            }
        }
        let t = (iteration as f64 / (p.iterations.max(2) - 1) as f64).min(1.0);
        if relaxing {
            // Release data around remaining inversions and let the shape
            // regulariser rebuild them consistently with their neighbours.
            let current: Vec<Vec3> = x.iter().map(|q| a(*q)).collect();
            let mut frozen = vec![false; n];
            for (k, tri) in welded_triangles.iter().enumerate() {
                if inverted(
                    &face_normal(&current, tri),
                    &carried(&frames, tri, &rest_normals[k]),
                ) {
                    tri.iter().for_each(|&i| frozen[i as usize] = true);
                }
            }
            for _ in 0..2 {
                let grow = frozen.clone();
                for i in 0..n {
                    if grow[i] {
                        neighbours[i]
                            .iter()
                            .for_each(|&j| frozen[j as usize] = true);
                    }
                }
            }
            for i in 0..n {
                if frozen[i] {
                    trust[i] = 0.0;
                    stiffness[i] = stiffness[i].max(16.0);
                }
            }
        }
        let lambda = p.stiffness_start * (p.stiffness_end / p.stiffness_start).powf(t);
        let tau = height * (0.05 * (1.0 - t) + 0.02 * t);
        let positions: Vec<Vec3> = x.iter().map(|q| a(*q)).collect();
        let unwelded: Vec<Vec3> = weld.id.iter().map(|&w| positions[w]).collect();
        let normals_u = vertex_normals(&unwelded, &body.prim.triangles, &weld);
        let normals: Vec<Vec3> = weld.representative.iter().map(|&i| normals_u[i]).collect();

        // With guidance, the tight pull is for garments marked `tight` only: a
        // loose robe pulled the lower back out into a bulge through its slit.
        let tight_garment = |t: usize| match seg.triangle_piece[t] {
            Some(pi) if !seg.piece_tight.is_empty() && seg.piece_tight.iter().any(|x| *x) => {
                seg.piece_tight[pi]
            }
            _ => true,
        };
        // Forward data term, classified by the label of the scan triangle hit,
        // and the cloth surface over the vertex (point, outward normal) where
        // the envelope applies: the symmetric prior may not push past it.
        type Forward = (Option<(V, f64, u8)>, Option<(V, V)>);
        let forward: Vec<Forward> = (0..n)
            .into_par_iter()
            .map(|i| -> Forward {
                if scalp_hold.gpt_scalp[i] || p.head_fallback.is_none() && scalp_hold.classes[i]==3 {
                    if scalp_hold.hard(i) { return (None, None); }
                    // A band vertex may acquire visible skin data as its neck
                    // boundary fits. Hair remains solely the bending obstacle.
                    let Some(hit)=scan.query.closest_point(positions[i],tau,Some((normals[i],0.3)))
                        else{return (None,None)};
                    let visible=seg.triangle_label[hit.source[1]]==Label::Skin && !seg.is_prop(hit.source[1])
                        && scan.triangles[hit.source[1]].iter().any(|&s|scalp_hold.scan_seen[s as usize]);
                    let nq=v(hit.normal);
                    return (visible.then(||(x[i]+nq*(v(hit.point)-x[i]).dot(&nq),p.skin_weight*trust[i],0)),None);
                }
                if cavity[i] {
                    // Cavities never take skin data, but may not poke out of
                    // the scan: one-sided push back inside (no orientation
                    // gate — cavity normals face inward).
                    let Some(hit) = scan.query.closest_point(positions[i], height * 0.01, None)
                    else {
                        return (None, None);
                    };
                    let (q, nq) = (v(hit.point), v(hit.normal));
                    let out = (x[i] - q).dot(&nq);
                    return (
                        (out > 0.0005).then(|| (x[i] - nq * (out + 0.001), p.envelope_weight, 1)),
                        None,
                    );
                }
                // On the head the body may never stick out of the scan: where
                // the orientation-gated match fails (nostril wings turned
                // sideways, a wider MakeHuman lip) the vertex would keep its
                // MakeHuman shape and bulge. Push it back inside, ungated.
                let reach = if region[i] == Region::Head {
                    0.006
                } else {
                    0.02
                };
                let push_inside = || {
                    let hit = scan
                        .query
                        .closest_point(positions[i], height * reach, None)?;
                    let (q, nq) = (v(hit.point), v(hit.normal));
                    let out = (x[i] - q).dot(&nq);
                    (out > 0.0005).then(|| (x[i] - nq * (out + 0.0005), p.envelope_weight, 1))
                };
                // With body-size targets the torso and limbs may never stick
                // out of a garment either where the gated match found
                // nothing (a grown trapezius under a collar).
                let push_out_of_cloth = || -> Forward {
                    if !p.fill.is_empty()
                        && matches!(region[i], Region::Torso | Region::Arm | Region::Leg)
                        && let Some(hit) =
                            scan.query.closest_point(positions[i], height * 0.02, None)
                    {
                        let (q, nq) = (v(hit.point), v(hit.normal));
                        let out = (x[i] - q).dot(&nq);
                        // Outer faces only: an inner face (a lining) read
                        // as is pushed a body correctly inside it outward,
                        // and read flipped, a garment layer running on
                        // inside the body pulled the body in after it.
                        if out > 0.0005
                            && nq.dot(&v(normals[i])) > 0.3
                            && seg.triangle_label[hit.source[1]] == Label::Garment
                        {
                            return (
                                Some((x[i] - nq * (out + p.clearance_m), p.envelope_weight, 1)),
                                Some((q, nq)),
                            );
                        }
                    }
                    (None, None)
                };
                let Some(hit) =
                    scan.query
                        .closest_point(positions[i], tau, Some((normals[i], 0.3)))
                else {
                    return if region[i] == Region::Head {
                        (push_inside(), None)
                    } else {
                        push_out_of_cloth()
                    };
                };
                let q = v(hit.point);
                let nq = v(hit.normal);
                let hand = trust[i];
                if region[i] == Region::Head && hand < 0.05 {
                     
                    return (push_inside(), None);
                }
                if hand <= 0.0 {
                    return (None, None);
                }
                // Point-to-plane targets: move only along the scan normal, so
                // the face and limbs never get dragged sideways into folds.
                let along = (q - x[i]).dot(&nq);
                let label = seg.triangle_label[hit.source[1]];
                if own_skin_tri[hit.source[1]] && !matches!(region[i], Region::Hand | Region::Foot) {
                    return (None, None);
                }
                match label {
                    Label::Skin => (Some((x[i] + nq * along, p.skin_weight * hand, 0)), None),
                    // Props (a quiver on the back, a bow across the chest)
                    // do not wrap the body: no envelope.
                    _ if seg.is_prop(hit.source[1]) => (None, None),
                    _ => {
                        // The envelope only ever pushes the body *inward* with
                        // respect to its pre-fit orientation; otherwise a body
                        // vertex can creep out along hairpins, straps or folds
                        // as its current normal turns.
                        let rest_n = v(rest_normals_v[i]);
                        if nq.dot(&rest_n) < 0.3 {
                            return (None, None);
                        }
                         
                        if (q - rest[i]).norm() > height * 0.05 {
                            return (None, None);
                        }
                        let target = x[i] + nq * (along - p.clearance_m);
                        let outside =
                            -along > -p.clearance_m && (target - x[i]).dot(&rest_n) <= 0.0;
                        let plane = Some((q, nq));
                        let term = if outside {
                            Some((target, p.envelope_weight, 1))
                        } else if label == Label::Garment
                            && matches!(region[i], Region::Torso | Region::Arm | Region::Leg)
                            && hit.distance_m < height * 0.006
                            && v(normals[i]).dot(&nq) > 0.7
                            && tight_garment(hit.source[1])
                        {
                            // Tight cloth only: a weak pull keeps the body close
                            // under fitted shirts and trousers. Never on hair,
                            // shoes or accessories, where it creeps outwards
                            // along thin protrusions.
                            Some((target, p.envelope_weight * 0.05, 2))
                        } else {
                            None
                        };
                        (term, plane)
                    }
                }
            })
            .collect();
        let mut target_sum = vec![V::zeros(); n];
        let mut weight = vec![0.0f64; n];
        let mut counts = [0usize; 4];
        for (i, (f, _)) in forward.iter().enumerate() {
            last_kind[i] = f.as_ref().map_or(u8::MAX, |f| f.2);
            if let Some((q, w, kind)) = f {
                // Under the clothes the tight-cloth pull goes: only the
                // one-sided push acts there, as a preference (the garments
                // follow the body, `garment::bind`).
                let w = if *kind == 2 {
                    w * (1.0 - hidden_w[i])
                } else {
                    *w
                };
                if w <= 0.0 {
                    continue;
                }
                target_sum[i] += q * w;
                weight[i] += w;
                counts[*kind as usize] += 1;
            }
        }
        // Reverse skin term: visible skin the body has not reached yet.
        let body_query = build_query(&positions, &welded_triangles);
        let reverse: Vec<(usize, V)> = skin_samples
            .par_iter()
            .filter_map(|&s| {
                let hit = body_query.closest_point(
                    scan.positions[s],
                    tau,
                    Some((scan.normals[s], 0.3)),
                )?;
                let tri = welded_triangles[hit.source[1]];
                let k =
                    (0..3).max_by(|&x, &y| hit.barycentric[x].total_cmp(&hit.barycentric[y]))?;
                let i = tri[k] as usize;
                if scalp_hold.hard(i) || (scalp_hold.gpt_scalp[i] || p.head_fallback.is_none() && scalp_hold.classes[i]==3) && !scalp_hold.scan_seen[s] { return None; }
                if own_skin_vertex[s] && !matches!(region[i], Region::Hand | Region::Foot) {
                    return None;
                }
                let nb = v(normals[i]);
                Some((i, x[i] + nb * (v(scan.positions[s]) - x[i]).dot(&nb)))
            })
            .collect();
        // Many scan samples can map to one body vertex (a protruding tip is
        // the nearest point to a whole patch). Average them and give the
        // vertex one bounded weight, or the tip runs away towards the patch.
        let mut reverse_sum = vec![V::zeros(); n];
        let mut reverse_count = vec![0u32; n];
        for (i, q) in &reverse {
            if protected[*i] || cavity[*i] || trust[*i] <= 0.0 {
                continue;
            }
            reverse_sum[*i] += q;
            reverse_count[*i] += 1;
        }
        for i in 0..n {
            if reverse_count[i] > 0 {
                let w = p.skin_weight * 0.5 * trust[i];
                reverse_hits[i] += reverse_count[i];
                target_sum[i] += reverse_sum[i] / reverse_count[i] as f64 * w;
                weight[i] += w;
            }
        }
        counts[3] = reverse.len();
        // Evidence of this iteration, then the symmetric prior.
        for i in 0..n {
            let skin = last_kind[i] == 0 || reverse_count[i] > 0;
            evidence_class[i] = if skin {
                Evidence::Skin
            } else if last_kind[i] == 1 || last_kind[i] == 2 {
                Evidence::Cloth
            } else {
                Evidence::None
            };
            let now = match evidence_class[i] {
                Evidence::Skin => trust[i].clamp(0.0, 1.0),
                Evidence::Cloth => 0.25,
                Evidence::None => 0.0,
            };
            evidence[i] = if evidence[i].is_nan() {
                now
            } else {
                0.6 * evidence[i] + 0.4 * now
            };
        }
        // Body size under the clothes: a weak radial pull towards *shell −
        // gap* where this iteration has no skin; never past the cloth.
        let mut filled = 0usize;
        if p.fill_weight > 0.0 {
            for i in 0..n {
                if evidence_class[i] == Evidence::Skin || trust[i] <= 0.0 {
                    continue;
                }
                let Some(f) = p.fill.get(weld.representative[i]).and_then(|f| f.as_ref()) else {
                    continue;
                };
                let d = x[i] - f.center;
                let delta = f.radius - d.dot(&f.dir);
                if !f.two_sided && delta <= 0.0 {
                    continue;
                }
                 
                if (d - f.dir * d.dot(&f.dir)).norm() > 0.015 * height {
                    continue;
                }
                let mut t = x[i] + f.dir * delta;
                if let Some((q, nq)) = forward[i].1 {
                    let out = (t - q).dot(&nq) + p.clearance_m;
                    if out > 0.0 {
                        t -= nq * out;
                    }
                }
                 
                let outward = (t - x[i]).dot(&f.dir) > 0.0;
                let hidden_gain = if outward { 1.0 + 4.0 * hidden_w[i] } else { 1.0 - 0.5 * hidden_w[i] };
                let w = p.fill_weight * f.weight * trust[i] * hidden_gain;
                if w <= 0.0 {
                    continue;
                }
                target_sum[i] += t * w;
                weight[i] += w;
                filled += 1;
            }
        }
        let mut symmetric_pulled = 0usize;
        let defect_margin = 0.0005 * height / 1.75;
        if p.symmetry > 0.0 {
            let frames_now = symmetry::vertex_frames(&influence_rows, &parents, &shape_rest, &x);
            let unposed = symmetry::unpose(&frames_now, &x);
            for i in (0..n).filter(|&i| symmetric_member[i]) {
                let w =
                    p.symmetry * (1.0 - evidence[i]).clamp(0.0, 1.0).powi(2) * (1.0 - hidden_w[i]);
                if w < 1e-4 {
                    continue;
                }
                if let Some(mut t) = symmetry::mirrored_target(&mirror, &frames_now, &unposed, i) {
                    // Only skin is shape evidence. Of two hidden sides, the
                    // one further from MakeHuman's own shape is the defect (a
                    // cloth-pushed dent, a bulge pulled out to a waistband):
                    // it copies the other, never the reverse. Averaging them
                    // put one hip's scallop on both hips; copying outwards
                    // only kept the dents.
                    let partner = mirror.partner[i] as usize;
                    if evidence_class[partner] != Evidence::Skin {
                        let own = (unposed[i] - shape_rest[i]).norm();
                        let other = (unposed[partner] - shape_rest[partner]).norm();
                        if own <= other + defect_margin {
                            continue;
                        }
                    }
                    // Never past the cloth: the envelope is an upper bound
                    // the mirrored shape must respect too.
                    if let Some((q, nq)) = forward[i].1 {
                        let out = (t - q).dot(&nq) + p.clearance_m;
                        if out > 0.0 {
                            t -= nq * out;
                        }
                    }
                    target_sum[i] += t * w;
                    weight[i] += w;
                    symmetric_pulled += 1;
                }
            }
        }
        if shod_feet.any() {
            let normals_now: Vec<V> = normals.iter().map(|q| v(*q)).collect();
            let foot_targets =
                shod_feet.targets(&rest, &x, &normals_now, scan, seg, p.clearance_m, height);
            for (i, q) in foot_targets {
                let w = shod_feet.weight[i];
                target_sum[i] *= 1.0 - w;
                weight[i] *= 1.0 - w;
                target_sum[i] += q * (p.envelope_weight * w);
                weight[i] += p.envelope_weight * w;
            }
        }
        {
            let normals_now: Vec<V> = normals.iter().map(|q| v(*q)).collect();
            let hand_targets =
                hand_fit.targets(&rest, &x, &normals_now, scan, seg, p.clearance_m, height);
            for (i, q) in hand_targets {
                let w = hand_fit.weight[i];
                target_sum[i] *= 1.0 - w;
                weight[i] *= 1.0 - w;
                target_sum[i] += q * (p.envelope_weight * w);
                weight[i] += p.envelope_weight * w;
            }
        }
        for (k, (name, set, target)) in landmark_sets.iter().enumerate() {
            let c = set.iter().fold(V::zeros(), |acc, &i| acc + x[i]) / set.len() as f64;
            let delta = target - c;
            let w = if name.matches('_').count() == 2 {
                8.0
            } else {
                3.0
            };
            for &i in set {
                target_sum[i] += (x[i] + delta) * w;
                weight[i] += w;
            }
            for &(j, f) in &landmark_patches[k] {
                target_sum[j] += (x[j] + delta) * (w * f);
                weight[j] += w * f;
            }
        }
         
        if relaxing && !ear_placed.is_empty() {
            for patch in &ears.patches {
                let held: Vec<(usize, f64)> =
                    patch.vertices.iter().map(|&i| (i, 1.0)).chain(patch.collar.iter().copied()).collect();
                let total: f64 = held.iter().map(|h| h.1).sum();
                let slide = held.iter().fold(V::zeros(), |s, &(i, f)| s + (ear_placed[i] - x[i]) * f)
                    / total.max(1e-12);
                for (i, f) in held {
                    let nb = v(normals[i]);
                    let w = p.skin_weight * f;
                    target_sum[i] += (x[i] + slide - nb * slide.dot(&nb)) * w;
                    weight[i] += w;
                }
            }
        }
        // Marked body outlines as a smooth residual on the shaped body: each
        // station's cross-section is measured in its slab, and the same
        // scaling across the view (to the marked width) applies to the
        // station's vertices within OUTLINE_REACH, fading with distance.
        // Width (front view) and depth (side view) corrections add; stations
        // overlapping on one axis average.
        if !silhouette_views.is_empty() {
            let (sum, count, hits, _) = silhouette::pulls(
                &silhouette_views,
                &x,
                &normals,
                &welded_triangles,
                &silhouette_eligible,
                SILHOUETTE_REACH * height,
            );
            for i in 0..n {
                if count[i] > 1e-9 {
                     
                    let w = SILHOUETTE_WEIGHT
                        * (count[i] / hits[i])
                        * match (region[i], evidence_class[i]) {
                            (Region::Head, Evidence::Skin) => 0.0,
                            (_, Evidence::Skin) => 0.25,
                            _ => 1.0,
                        };
                    if w <= 0.0 {
                        continue;
                    }
                    target_sum[i] += (x[i] + sum[i] / count[i]) * w;
                    weight[i] += w;
                }
            }
        }
        if !outline_fit.is_empty() && silhouette_views.is_empty() {
            let reach_along = OUTLINE_REACH * height;
            let mut disp = vec![[V::zeros(); 2]; n];
            let mut fsum = vec![[0.0f64; 2]; n];
            let mut cover = vec![[0.0f64; 2]; n];
            let mut conf_at = vec![0.0f64; n];
            let near = OUTLINE_NEAR * height;
            for (members, c, across, along, lo, hi, conf) in &outline_fit {
                let reach = 1.5 * (hi - lo);
                // The exact cut, not a slab's vertices (`shape::section_extent`).
                let mut member = vec![false; n];
                for &(i, w) in members {
                    member[i] = w >= crate::shape::CUT_MEMBER;
                }
                let Some((e_lo, e_hi)) = crate::shape::section_extent(
                    |i| x[i],
                    &welded_triangles,
                    &member,
                    *c,
                    *across,
                    *along,
                    reach,
                ) else {
                    continue;
                };
                let axis = usize::from(across.z.abs() > across.x.abs());
                for &(i, bone_w) in members {
                    let d = x[i] - c;
                    let t = d.dot(along).abs() / reach_along;
                    let u = d.dot(across);
                    if t >= 1.0 || u.abs() > reach {
                        continue;
                    }
                    let f = (1.0 - guard::smoothstep(t)) * bone_w;
                    // The nearest rows decide (inverse-square in the
                    // distance along the body): averaged over the whole
                    // reach, the dense rows' corrections blurred over 17 cm
                    // and a ledge narrower than that (MakeHuman's knee under
                    // a cartoon's thigh) stayed, rings down the legs.
                    let dd = d.dot(along);
                    let g = f / (dd * dd + near * near);
                     
                    let mid = 0.5 * (e_lo + e_hi);
                    let to = mid + (u - mid) * (hi - lo) / (e_hi - e_lo);
                    disp[i][axis] += across * ((to - u) * g);
                    fsum[i][axis] += g;
                    cover[i][axis] = cover[i][axis].max(f);
                    conf_at[i] = conf_at[i].max(conf * f);
                }
            }
            for i in 0..n {
                if conf_at[i] <= 0.0 {
                    continue;
                }
                let d = (0..2).fold(V::zeros(), |s, k| {
                    if fsum[i][k] > 0.0 { s + disp[i][k] / fsum[i][k] * cover[i][k] } else { s }
                });
                let w = OUTLINE_FIT_WEIGHT * conf_at[i];
                target_sum[i] += (x[i] + d) * w;
                weight[i] += w;
            }
        }
        for (i, q) in &margin_hold {
            let d = q - x[*i];
            let w = 8.0;
            target_sum[*i] += (x[*i] + d - V::z() * d.z) * w;
            weight[*i] += w;
        }
        scalp_hold.add_targets(&mut target_sum, &mut weight);
        // Weak anchor keeps unobserved regions from drifting as a whole.
        let anchor = 0.002;
        let targets: Vec<V> = (0..n)
            .map(|i| (target_sum[i] + rest[i] * anchor) / (weight[i] + anchor))
            .collect();
        let diag: Vec<f64> = weight.iter().map(|w| w + anchor).collect();

        // Local step: per-vertex similarity (rotation + bounded scale).
        let mut transforms: Vec<Matrix3<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut m = Matrix3::zeros();
                let mut rr = 0.0;
                for &j in &neighbours[i] {
                    let e0 = shape_ref[i] - shape_ref[j as usize];
                    m += (x[i] - x[j as usize]) * e0.transpose();
                    rr += e0.norm_squared();
                }
                let r = proper_rotation(m);
                let (lo, hi) = if lid[i] { (0.5, 2.0) } else { p.scale_range };
                // Hidden body shrinks by 8 % at most (a chest shrank into its
                // armour); it may still grow to bridge a cartoon's big
                // visible head and hands.
                let lo = lo + (lo.max(0.92) - lo) * hidden_w[i];
                let s = ((r.transpose() * m).trace() / rr.max(1e-20)).clamp(lo, hi);
                r * s
            })
            .collect();
        // Smooth the scale field once so proportions change gradually.
        let scales: Vec<f64> = transforms.iter().map(|m| m.determinant().cbrt()).collect();
        transforms = (0..n)
            .into_par_iter()
            .map(|i| {
                let s = (scales[i]
                    + neighbours[i]
                        .iter()
                        .map(|&j| scales[j as usize])
                        .sum::<f64>())
                    / (1 + neighbours[i].len()) as f64;
                transforms[i] / scales[i].max(1e-9) * s
            })
            .collect();
        frames = transforms.iter().map(|m| proper_rotation(*m)).collect();
        let edges: Vec<Vec<(u32, f64)>> = (0..n)
            .map(|i| {
                neighbours[i]
                    .iter()
                    .map(|&j| {
                        let k = |m: usize| stiffness[m] * (1.0 + p.hidden_stiffness * hidden_w[m]);
                        (j, lambda * 0.5 * (k(i) + k(j as usize)))
                    })
                    .collect()
            })
            .collect();
        let rhs: Vec<V> = (0..n)
            .into_par_iter()
            .map(|i| {
                edges[i].iter().fold(targets[i] * diag[i], |s, &(j, w)| {
                    let j = j as usize;
                    s + (transforms[i] + transforms[j]) * (shape_ref[i] - shape_ref[j]) * (0.5 * w)
                })
            })
            .collect();
        let previous = x.clone();
        let residual = scalp_hold.solve(&edges, &diag, &rhs, &mut x, 300);
        ensure!(
            x.iter().all(|q| q.iter().all(|c| c.is_finite())),
            "body_fit_nonfinite"
        );
        // Bound per-iteration motion.
        let step_limit = height * 0.01;
        for i in 0..n {
            let d = x[i] - previous[i];
            if d.norm() > step_limit {
                x[i] = previous[i] + d.normalize() * step_limit;
            }
        }
         
        {
            let folding = fit_guard.fold_vertices(&x);
            let mut region_mask = folding.clone();
            for i in 0..n {
                if folding[i] {
                    neighbours[i]
                        .iter()
                        .for_each(|&j| region_mask[j as usize] = true);
                }
            }
            let ids: Vec<usize> = (0..n)
                .filter(|&i| {
                    region_mask[i]
                        && !cavity[i]
                        && !ears.member[i]
                        && !landmark_vertex[i]
                        && !neighbours[i].is_empty()
                })
                .collect();
            for _ in 0..if relaxing { 6 } else { 2 } {
                let next: Vec<V> = ids
                    .iter()
                    .map(|&i| {
                        let k = neighbours[i].len() as f64;
                        let mean = neighbours[i]
                            .iter()
                            .fold(V::zeros(), |s, &j| s + x[j as usize])
                            / k;
                        mean + transforms[i] * rest_detail[i] / k
                    })
                    .collect();
                for (&i, q) in ids.iter().zip(next) {
                    x[i] = (x[i] + q) * 0.5;
                }
            }
        }
         
        if !ears.patches.is_empty() && !relaxing {
            let local_scale: Vec<f64> = transforms.iter().map(|m| m.determinant().cbrt()).collect();
            ear_report = ears.transport(
                &shape_rest,
                &mut x,
                &local_scale,
                Some((scan, &seg.vertex_label)),
            );
            ear_placed = x.clone();
        }
        // Deformation guard at the end of each stage: back the stage's update
        // off around new inversions and collapses; the vertices it held back
        // trust the data less in the next stage.
        if (iteration + 1) % stage_length == 0 || iteration + 1 == p.iterations + relax_iterations {
            let outcome = fit_guard.limit(&stage_start, &mut x, &frames);
            let mut grow = outcome.anchors.clone();
            for (ring, &held) in neighbours.iter().zip(&outcome.anchors) {
                if held {
                    ring.iter().for_each(|&j| grow[j as usize] = true);
                }
            }
            for (t, g) in trust.iter_mut().zip(&grow) {
                if *g {
                    *t *= 0.8;
                }
            }
            let mut report = outcome.report;
            report["iteration"] = json!(iteration);
            guard_reports.push(report);
            stage_start = x.clone();
        }
        scalp_hold.enforce(&mut x);
        // Flip guard: inverted triangles locally stiffen and lose data trust.
        let current: Vec<Vec3> = x.iter().map(|q| a(*q)).collect();
        let mut flipped = 0usize;
        for (k, tri) in welded_triangles.iter().enumerate() {
            if inverted(
                &face_normal(&current, tri),
                &carried(&frames, tri, &rest_normals[k]),
            ) {
                flipped += 1;
                for &i in tri {
                    stiffness[i as usize] = (stiffness[i as usize] * 1.5).min(64.0);
                    trust[i as usize] *= 0.6;
                }
            }
        }
        let defects = fit_guard.defects(&x, &frames);
        history.push(json!({"iteration":iteration,"defects":defects,"relax":relaxing,"stiffness":lambda,"tau_m":tau,"skin":counts[0],"envelope_push":counts[1],"envelope_tight":counts[2],"reverse_skin":counts[3],"symmetric_pulled":symmetric_pulled,"fill":filled,"flipped":flipped,"cg_residual":residual}));
    }
    // Human first: MakeHuman's (aligned, posed) shape plus a smooth residual
    // capped at HUMAN_RESIDUAL, none under the clothes. Hands, feet and ears
    // keep their own bounded fits (pose, size, digit cut; ear similarity).
    crate::geom::dump_stage("3-fine", weld.id.iter().map(|&w| a(x[w])));
    stage_inversions.insert("fine_fit",audit_inversions(&x));
    let human_report = (p.human_first && staged).then(|| {
        // Blend weight of the bounded hand and foot fits (1 inside, fading
        // over their bands): a hard switch tore the wrists and ankles.
        let own_w: Vec<f64> = (0..n)
            .map(|i| {
                if matches!(region[i], Region::Hand | Region::Foot) {
                    1.0
                } else {
                    hand_fit.weight[i].max(shod_feet.weight[i]).clamp(0.0, 1.0)
                }
            })
            .collect();
        let own = |i: usize| own_w[i] >= 1.0;
        let mut d: Vec<V> = (0..n).map(|i| x[i] - rest[i]).collect();
        let raw_max = d.iter().map(|v| v.norm()).fold(0.0, f64::max);
         
        let passes = if p.guided_body { GUIDED_SMOOTHING } else { HUMAN_SMOOTHING };
        for _ in 0..passes {
            d = (0..n)
                .map(|i| {
                    if own(i) || neighbours[i].is_empty() {
                        return d[i];
                    }
                    let m = neighbours[i]
                        .iter()
                        .fold(V::zeros(), |s, &j| s + d[j as usize])
                        / neighbours[i].len() as f64;
                    (d[i] + m) * 0.5
                })
                .collect();
        }
        let cap = if p.guided_body { p.skin_residual.max(GUIDED_SKIN_RESIDUAL) } else { HUMAN_RESIDUAL } * height;
        let mut capped = 0usize;
        for i in 0..n {
            if own(i) {
                continue;
            }
            let mut r = d[i] * (1.0 - hidden_w[i]);
            let len = r.norm();
            if p.guided_body {
                if len > 1e-9 {
                    r *= cap * (len / cap).tanh() / len;
                }
                if len > 0.5 * cap {
                    capped += 1;
                }
            } else if len > cap {
                r *= cap / len;
                capped += 1;
            }
            let w = own_w[i];
            x[i] = rest[i] + r * (1.0 - w) + (x[i] - rest[i]) * w;
        }
        // The per-vertex rotations again, for the inversion checks below:
        // those of the fit's last local step belong to the uncapped shape.
        frames = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut m = Matrix3::zeros();
                for &j in &neighbours[i] {
                    m += (x[i] - x[j as usize]) * (shape_ref[i] - shape_ref[j as usize]).transpose();
                }
                proper_rotation(m)
            })
            .collect();
        json!({"cap_m": cap, "smoothing_passes": passes, "raw_max_m": raw_max, "capped_vertices": capped})
    });
    if staged {
        scalp_hold.enforce(&mut x);
    }
    // Body under the clothes from the suit's whole outline (`fit::body_field`).
    let body_field_report = (staged && std::env::var("MH_BODY_FIELD").is_ok() && !silhouette_views.is_empty()).then(|| {
        let pinned: Vec<bool> = (0..n)
            .map(|i| matches!(region[i], Region::Head | Region::Hand | Region::Foot)
                || hand_fit.weight[i] > 0.0 || shod_feet.weight[i] > 0.0
                || (evidence_class[i] == Evidence::Skin && hidden_w[i] < 0.5))
            .collect();
        let env = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let settings = body_field::Settings {
            support: env("MH_BF_SUPPORT", 0.10), spacing: env("MH_BF_SPACING", 0.05),
            falloff: env("MH_BF_FALLOFF", 0.05), reach: env("MH_BF_REACH", 0.06),
            iterations: env("MH_BF_ITERS", 6.0) as usize, smooth: env("MH_BF_SMOOTH", 0.05),
            damping: env("MH_BF_DAMP", 0.002), cap: env("MH_BF_CAP", 0.06),
            shoulder: std::env::var("MH_BF_ARMS").is_err().then(|| env("MH_BF_SHOULDER", 0.12)),
        };
        crate::geom::dump_stage("body-field-pinned", weld.id.iter().map(|&w| [pinned[w] as u8 as f64, region[w] as u8 as f64, hidden_w[w]]));
        crate::geom::dump_stage("body-field-before", weld.id.iter().map(|&w| a(x[w])));
        let mut r = body_field::fit(&silhouette_views, &mut x, &welded_triangles, &neighbours, &region, &pinned, &settings);
         
        let fair = env("MH_BF_FAIR", 12.0) as usize;
        let w: Vec<f64> = (0..n).map(|i| if pinned[i] || matches!(region[i], Region::Arm) { 0.0 } else { hidden_w[i].clamp(0.0, 1.0) }).collect();
        let before = x.clone();
        for k in 0..2 * fair {
            let f = if k % 2 == 0 { 0.5 } else { -0.53 };
            let next: Vec<V> = (0..n)
                .map(|i| {
                    let nb = &neighbours[i];
                    if w[i] <= 0.0 || nb.is_empty() {
                        return x[i];
                    }
                    let mean = nb.iter().fold(V::zeros(), |s, &j| s + x[j as usize]) / nb.len() as f64;
                    x[i] + (mean - x[i]) * (f * w[i])
                })
                .collect();
            x.copy_from_slice(&next);
        }
        let moved: Vec<f64> = (0..n).map(|i| (x[i] - before[i]).norm()).collect();
        r["fairing"] = json!({"iterations": fair, "vertices": w.iter().filter(|&&v| v > 0.0).count(),
            "move_max_mm": moved.iter().cloned().fold(0.0, f64::max) * 1000.0});
        // Fine pass on the crotch alone (MH_BF_CROTCH radius, 0 = off).
        let radius = env("MH_BF_CROTCH", 0.0);
        if radius > 0.0 {
            if let Some((apex, patch)) = body_field::crotch_patch(&x, &neighbours, &region, radius) {
                let local: Vec<bool> = (0..n).map(|i| pinned[i] || !patch[i]).collect();
                let fine = body_field::Settings {
                    support: env("MH_BFC_SUPPORT", 0.04), spacing: env("MH_BFC_SPACING", 0.015),
                    falloff: env("MH_BFC_FALLOFF", 0.04), reach: env("MH_BFC_REACH", 0.04),
                    iterations: 6, smooth: env("MH_BFC_SMOOTH", 0.02), damping: 0.002,
                    cap: env("MH_BFC_CAP", 0.03), shoulder: None,
                };
                let mut c = body_field::fit(&silhouette_views, &mut x, &welded_triangles, &neighbours, &region, &local, &fine);
                c["apex"] = json!(a(x[apex]));
                r["crotch"] = c;
            }
        }
        let lift_radius = env("MH_BF_LIFT", 0.12);
        if lift_radius > 0.0 {
            r["crotch_lift"] = body_field::crotch_lift(&silhouette_views, &mut x, &welded_triangles, &region, &pinned, lift_radius, env("MH_BF_LIFT_CAP", 0.04));
        }
        crate::geom::dump_stage("body-field-after", weld.id.iter().map(|&w| a(x[w])));
        r
    });
    let fit_scalp_error_mm = scalp_hold.error_mm(&x);
    stage_inversions.insert("residual_cap",audit_inversions(&x));
    crate::geom::dump_stage("sculpt-before",weld.id.iter().map(|&w|a(x[w])));
    // Free head: the visible skin sculpted onto the scan, on the measured
    // armature (`fit::sculpt`).
    let sculpt_report = (p.free_head && staged).then(|| {
        let free: Vec<bool> = (0..n).map(|i| region[i] == Region::Head
            || p.head_fallback.is_none() && on_neck[i] && (scalp_hold.classes[i]==1 || scalp_hold.gpt_scalp[i])).collect();
        let held: std::collections::HashSet<usize> = margin_hold.iter().map(|m| m.0).collect();
        // The oculi bones also weight visible brow/upper-cheek skin; those
        // vertices are not all lid margins. Keep the actual measured opening
        // and landmark armature, allowing the surrounding skin to fit scan.
        let measured_lids:std::collections::HashSet<usize>=eye_openings.iter()
            .flat_map(|o|o.upper.iter().chain(&o.lower).map(|&i|weld.id[i])).collect();
        let armature: Vec<bool> = (0..n).map(|i| free[i] && (landmark_vertex[i] || held.contains(&i)
            || if p.head_fallback.is_some(){lid[i]}else{measured_lids.contains(&i)})).collect();
        let eye_socket: Vec<bool> = weld
            .representative
            .iter()
            .map(|&r| prior.skeleton().joints[prior.dominant_joint(prior.body, r)].name.to_lowercase().starts_with("eye"))
            .collect();
         
        let data: Vec<bool> = (0..n).map(|i| free[i] && !scalp_hold.hard(i) && !armature[i] && !cavity[i] && !eye_socket[i]
            && (!(ears.member[i] || ears.collar[i])
                || p.head_fallback.is_none() && scalp_hold.classes[i]==1)).collect();
        // The marked eye openings with a margin: centre of the four marks,
        // radius EYE_HOLE times the corners' distance.
        let eye_holes = eye_holes(landmarks);
        let openings:Vec<_>=if p.head_fallback.is_none() {
            ["l","r"].iter().filter_map(|side|crate::MarkedOpening::new(landmarks,side)).collect()
        }else{vec![]};
        let skin_only_query:Vec<_>=(0..n).map(|i|transition_ears[i]||on_neck[i]).collect();
        let report = sculpt::sculpt(
            &sculpt::Head { neighbours: &neighbours, triangles: &welded_triangles, free: &free, data: &data, armature: &armature, scalp_hold: &scalp_hold, skin_only_query:&skin_only_query, eyes: &eye_holes, openings:&openings, guard_surface:p.head_fallback.is_none(), height },
            &mut x,
            scan,
            seg,
        );
        frames = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut m = Matrix3::zeros();
                for &j in &neighbours[i] {
                    m += (x[i] - x[j as usize]) * (shape_ref[i] - shape_ref[j as usize]).transpose();
                }
                proper_rotation(m)
            })
            .collect();
        report
    });
    // The GPT scalp is already constrained. Only genuinely hidden exterior
    // vertices need a rest-shape bridge after sculpting.
    stage_inversions.insert("sculpt",audit_inversions(&x));
    let mut pinned_head_rig_reference = None;
    let hidden_head_report = (p.hidden_head && staged).then(|| {
        let head: Vec<bool> = (0..n).map(|i| region[i] == Region::Head).collect();
        let held: std::collections::HashSet<usize> = margin_hold.iter().map(|m| m.0).collect();
        let fixed: Vec<bool> = (0..n)
            .map(|i| {
                let r = weld.representative[i];
                // The ear's surroundings as fitted: restored round a grafted
                // ear, the skull came out through flame's ear roots.
                ears.member[i]
                    || ears.collar[i]
                    || ear_ring[i].is_finite()
                    || lid[i]
                    || landmark_vertex[i]
                    || cavity[i]
                    || protected[i]
                    || held.contains(&i)
                    || prior.skeleton().joints[prior.dominant_joint(prior.body, r)].name.to_lowercase().starts_with("eye")
            })
            .collect();
        crate::geom::dump_stage("hidden-head-before", weld.id.iter().map(|&w| a(x[w])));
        crate::geom::dump_stage("hidden-head-shape", weld.id.iter().map(|&w| a(head_shape_rest[w])));
        crate::geom::dump_stage("hidden-head-flags", weld.id.iter().map(|&w| [fixed[w] as u8 as f64, cavity[w] as u8 as f64, on_neck[w] as u8 as f64]));
        crate::geom::dump_stage("hidden-head-fixed-reasons", weld.id.iter().map(|&w| [(ears.member[w] || ears.collar[w] || ear_ring[w].is_finite()) as u8 as f64, protected[w] as u8 as f64, (lid[w] || landmark_vertex[w] || held.contains(&w)) as u8 as f64]));
        let report = if let Some((source, positions)) = &p.head_fallback {
            let anatomical_weight = |neck: bool| -> Vec<f64> {
                weld.representative.iter().map(|&i| body.influences[i].iter().filter(|w| {
                    let name = prior.skeleton().joints[w.joint].name.to_lowercase();
                    if neck { name.starts_with("neck") } else { name == "head" }
                }).map(|w| w.weight).sum()).collect()
            };
            let neck_weight = anatomical_weight(true);
            let skull_weight = anatomical_weight(false);
            let mut r = x110_head::restore(
                &x110_head::Input { neighbours: &neighbours, shape: &shape_rest,
                    head: &head, neck: &on_neck, neck_weight: &neck_weight,
                    skull_weight: &skull_weight, fixed: &fixed, height,
                    neck_limit: p.hidden_neck_limit_m }, &mut x, scan, seg);
            r["method"] = json!("x110_head_fallback");
            r["source"] = json!(source);
            r["fallback_stage"] = json!("original fit stage; no post-rig export reimport");
            r["validated_export_vertices"] = json!(positions.len());
            if let Some(input) = &p.anatomical_head {
                crate::geom::dump_stage("x119-x110-completion", weld.id.iter().map(|&w| a(x[w])));
                if input.x120.is_some() {
                    pinned_head_rig_reference = Some(weld.id.iter().map(|&w| a(x[w])).collect());
                }
                let anatomical = anatomical_head::fit(input, p.anatomical_head_zero,
                    &weld, &shape_rest, &welded_triangles, &head, &on_neck,
                    &cavity, &mut x, scan, seg);
                r["x110_completion"] = r.clone();
                r["method"] = json!(if input.x120.is_some() { "x120_pinned_anatomical_morph" } else { "x119_anatomical_morph" });
                r["anatomical"] = anatomical;
            }
            r
        } else {
            let mut report = scalp_hold.report.clone();
            report["hidden_bridge"] = hidden_head::restore(&neighbours, &head_shape_rest, &scalp_hold.classes, &fixed, &mut x);
            report["final_transition"] = scalp_hold.enforce_report(&mut x);
            report["fit_core_max_error_mm"] = json!(fit_scalp_error_mm);
            report["sculpt_core_max_error_mm"] = json!(scalp_hold.error_mm(&x));
            report
        };
        crate::geom::dump_stage("hidden-head-after", weld.id.iter().map(|&w| a(x[w])));
        frames = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut m = Matrix3::zeros();
                for &j in &neighbours[i] {
                    m += (x[i] - x[j as usize]) * (shape_ref[i] - shape_ref[j as usize]).transpose();
                }
                proper_rotation(m)
            })
            .collect();
        report
    });
    // The cloth stays outside the (stiff) hidden body: a smooth one-sided
    // clearance field (fit/hidden.rs).
    crate::geom::dump_stage("4-human-first", weld.id.iter().map(|&w| a(x[w])));
    stage_inversions.insert("head_completion",audit_inversions(&x));
    let clearance_report = (staged && !hidden_seed.is_empty() && p.body_clearance).then(|| {
        let positions: Vec<Vec3> = x.iter().map(|q| a(*q)).collect();
        let unwelded: Vec<Vec3> = weld.id.iter().map(|&w| positions[w]).collect();
        let normals_u = vertex_normals(&unwelded, &body.prim.triangles, &weld);
        let normals: Vec<Vec3> = weld.representative.iter().map(|&i| normals_u[i]).collect();
        hidden::clearance_pass(
            &mut x,
            &hidden_w,
            &normals,
            &welded_triangles,
            &shape_rest,
            &neighbours,
            &region,
            &hidden_excluded,
            scan,
            seg,
            p.clearance_m,
            height,
        )
    });
    // Face checks against the scan and against MakeHuman's own shape.
    // Bumps: face vertices above the scan surface (a lid corner past a
    // flatter stylised face, a nostril wing or lip that kept its shape).
    // Pushing them back after the fit folded the face, so they are reported.
    let above_scan = |x: &[V]| -> Vec<Option<(f64, V)>> {
        (0..n)
            .into_par_iter()
            .map(|i| {
                if !face_vertex[i] {
                    return None;
                }
                let hit = scan.query.closest_point(a(x[i]), height * 0.02, None)?;
                let nq = v(hit.normal);
                Some(((x[i] - v(hit.point)).dot(&nq), nq))
            })
            .collect()
    };
    let bump_stats = |above: &[Option<(f64, V)>]| {
        let mut out: Vec<f64> = above.iter().flatten().map(|a| a.0.max(0.0)).collect();
        out.sort_by(f64::total_cmp);
        json!({
            "vertices_over_1mm": out.iter().filter(|d| **d > 0.001).count(),
            "vertices_over_3mm": out.iter().filter(|d| **d > 0.003).count(),
            "p99_m": out.get(out.len() * 99 / 100).copied().unwrap_or(0.0),
            "max_m": out.last().copied().unwrap_or(0.0),
        })
    };
    let bump_above = above_scan(&x);
    let bumps = bump_stats(&bump_above);
    // New folds: an edge whose two triangles bend much more sharply than in
    // MakeHuman's rest shape is a crease the fit made (a flap at an eye
    // corner, a ridge under a nostril). Natural creases (lid margins, lips,
    // ears) are sharp in the rest shape already and do not count. Reported
    // and drawn green in fit-diagnostic.glb.
    let mut edge_tris: std::collections::HashMap<(u32, u32), Vec<usize>> = Default::default();
    for (t, tri) in welded_triangles.iter().enumerate() {
        for k in 0..3 {
            let (p, q) = (tri[k], tri[(k + 1) % 3]);
            edge_tris.entry((p.min(q), p.max(q))).or_default().push(t);
        }
    }
    let mut edge_pairs: Vec<((u32, u32), usize, usize)> = edge_tris
        .into_iter()
        .filter(|(_, ts)| ts.len() == 2)
        .map(|(e, ts)| (e, ts[0], ts[1]))
        .filter(|(e, _, _)| face_vertex[e.0 as usize] && face_vertex[e.1 as usize])
        .collect();
    edge_pairs.sort_unstable();
    let bend = |pos: &[Vec3], t0: usize, t1: usize| -> f64 {
        let (n0, n1) = (
            face_normal(pos, &welded_triangles[t0]),
            face_normal(pos, &welded_triangles[t1]),
        );
        let d = n0.norm() * n1.norm();
        if d < 1e-20 {
            0.0
        } else {
            (n0.dot(&n1) / d).clamp(-1.0, 1.0).acos()
        }
    };
    let rest_bend: Vec<f64> = edge_pairs
        .iter()
        .map(|&(_, t0, t1)| bend(&shape_positions, t0, t1))
        .collect();
    let folds = |x: &[V]| -> Vec<bool> {
        let pos: Vec<Vec3> = x.iter().map(|q| a(*q)).collect();
        let mut fold = vec![false; n];
        for (k, &((i, j), t0, t1)) in edge_pairs.iter().enumerate() {
            if bend(&pos, t0, t1) - rest_bend[k] > 0.8 {
                fold[i as usize] = true;
                fold[j as usize] = true;
            }
        }
        fold
    };
    let folds_prefit = folds(&rest).iter().filter(|f| **f).count();
    let fold0 = folds(&x);
    let folds_before = fold0.iter().filter(|f| **f).count();
    let mut fold_joints: BTreeMap<String, usize> = BTreeMap::new();
    for i in (0..n).filter(|&i| fold0[i]) {
        let name =
            &prior.skeleton().joints[prior.dominant_joint(prior.body, weld.representative[i])].name;
        *fold_joints
            .entry(format!(
                "{name}{}",
                if protected[i] { " (protected)" } else { "" }
            ))
            .or_default() += 1;
    }
    let fitted: Vec<Vec3> = weld.id.iter().map(|&w| a(x[w])).collect();
    let final_welded: Vec<Vec3> = x.iter().map(|q| a(*q)).collect();
    let mut flipped_vertex = vec![false; n];
    for (k, tri) in welded_triangles.iter().enumerate() {
        if inverted(
            &face_normal(&final_welded, tri),
            &carried(&frames, tri, &rest_normals[k]),
        ) {
            tri.iter().for_each(|&i| flipped_vertex[i as usize] = true);
        }
    }
    // red: inverted, green: new fold, white: face bump (>1 mm above the
    // scan), orange: skin, blue: cloth push, cyan: tight cloth,
    // magenta: protected face, yellow: reverse-skin only, grey: prior only.
    let diagnostic: Vec<[f32; 4]> = weld
        .id
        .iter()
        .map(|&w| {
            if flipped_vertex[w] {
                [1.0, 0.0, 0.0, 1.0]
            } else if fold0[w] {
                [0.0, 1.0, 0.0, 1.0]
            } else if bump_above[w].is_some_and(|b| b.0 > 0.001) {
                [1.0, 1.0, 1.0, 1.0]
            } else if protected[w] {
                [0.9, 0.1, 0.9, 1.0]
            } else {
                match last_kind[w] {
                    0 => [1.0, 0.6, 0.2, 1.0],
                    1 => [0.1, 0.3, 1.0, 1.0],
                    2 => [0.2, 0.9, 0.9, 1.0],
                    _ if reverse_hits[w] > 0 => [1.0, 1.0, 0.1, 1.0],
                    _ => [0.6, 0.6, 0.6, 1.0],
                }
            }
        })
        .collect();
    // Skin residual: distance from scan skin to the fitted body.
    let final_query = build_query(&fitted, &body.prim.triangles);
    let errors: Vec<(Region, f64)> = skin_samples
        .par_iter()
        .filter_map(|&s| {
            let h = final_query.closest_point(scan.positions[s], height * 0.1, None)?;
            let t = body.prim.triangles[h.source[1]][0] as usize;
            Some((region[weld.id[t]], h.distance_m))
        })
        .collect();
    let skin_error: Vec<f64> = errors.iter().map(|e| e.1).collect();
    let mut by_region: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    for (r, d) in &errors {
        by_region.entry(format!("{r:?}")).or_default().push(*d);
    }
    let skin_error_by_region: serde_json::Map<String, Value> = by_region
        .into_iter()
        .map(|(k, v)| (k, percentiles(v)))
        .collect();
    let mut flipped_by_region: std::collections::BTreeMap<String, usize> = Default::default();
    for (t, tri) in welded_triangles.iter().enumerate() {
        if inverted(
            &face_normal(&final_welded, tri),
            &carried(&frames, tri, &rest_normals[t]),
        ) {
            *flipped_by_region
                .entry(format!("{:?}", region[tri[0] as usize]))
                .or_default() += 1;
        }
    }
    // Largest local displacement discontinuities (vertex vs neighbour mean).
    let mut jumps: Vec<(f64, usize)> = (0..n)
        .map(|i| {
            let d = x[i] - rest[i];
            let m = neighbours[i]
                .iter()
                .fold(V::zeros(), |s, &j| s + (x[j as usize] - rest[j as usize]))
                / neighbours[i].len().max(1) as f64;
            ((d - m).norm(), i)
        })
        .collect();
    jumps.sort_by(|a, b| b.0.total_cmp(&a.0));
    let jumps: Vec<Value> = jumps[..8.min(n)]
        .iter()
        .map(|&(j, i)| json!({"jump_m":j,"joint":prior.skeleton().joints[prior.dominant_joint(prior.body, weld.representative[i])].name,"displacement_m":(x[i]-rest[i]).norm(),"rest":a(rest[i])}))
        .collect();
    let mut moved: Vec<(f64, usize)> = (0..n)
        .filter(|&i| region[i] == Region::Head)
        .map(|i| ((x[i] - rest[i]).norm(), i))
        .collect();
    moved.sort_by(|a, b| b.0.total_cmp(&a.0));
    let moved: Vec<Value> = moved
        .iter()
        .take(10)
        .map(|&(d, i)| json!({"moved_m":d,"rest":a(rest[i]),"delta":a(x[i]-rest[i]),"joint":prior.skeleton().joints[prior.dominant_joint(prior.body, weld.representative[i])].name,"trust":trust[i],"stiffness":stiffness[i],"cavity":cavity[i],"protected":protected[i],"last_kind":last_kind[i],"reverse_hits":reverse_hits[i]}))
        .collect();
    // Target minus fitted landmark (x right, y up, z forward).
    let landmark_offset: BTreeMap<&str, Vec3> = landmark_sets
        .iter()
        .map(|(name, set, target)| {
            let c = set.iter().fold(V::zeros(), |acc, &i| acc + x[i]) / set.len() as f64;
            (*name, a(target - c))
        })
        .collect();
    let landmark_fitted: BTreeMap<&str, Vec3> = landmark_sets
        .iter()
        .map(|(name, set, _)| {
            (
                *name,
                a(set.iter().fold(V::zeros(), |acc, &i| acc + x[i]) / set.len() as f64),
            )
        })
        .collect();
    let landmark_residual: BTreeMap<&str, f64> = landmark_sets
        .iter()
        .map(|(name, set, target)| {
            let c = set.iter().fold(V::zeros(), |acc, &i| acc + x[i]) / set.len() as f64;
            (*name, (c - target).norm())
        })
        .collect();
    // Where folds and bumps sit: ear, near an eye / the nose tip / a mouth
    // corner (2.5 cm), or elsewhere on the head.
    let zone_anchors: Vec<(&str, V)> = face_landmark_sets(prior)
        .into_iter()
        .filter(|(name, _)| ["eye_l", "eye_r", "nose_tip", "mouth_l", "mouth_r"].contains(name))
        .map(|(name, set)| (name, set_centroid(&body.prim.positions, &set)))
        .collect();
    let zones = |mask: &dyn Fn(usize) -> bool| {
        let mut zones: BTreeMap<&str, usize> = BTreeMap::new();
        for i in (0..n).filter(|&i| mask(i)) {
            let zone = if ears.member[i] {
                "ear"
            } else if ears.collar[i] {
                "ear_collar"
            } else {
                zone_anchors
                    .iter()
                    .map(|(name, c)| (*name, (shape_rest[i] - c).norm()))
                    .filter(|(_, d)| *d < 0.025 * height / 1.75)
                    .min_by(|p, q| p.1.total_cmp(&q.1))
                    .map_or("other", |z| z.0)
            };
            *zones.entry(zone).or_default() += 1;
        }
        zones
    };
    let fold_zones = zones(&|i| fold0[i]);
    // What drove each fold vertex in the last iteration.
    let mut fold_drivers: BTreeMap<String, usize> = BTreeMap::new();
    for i in (0..n).filter(|&i| fold0[i]) {
        let kind = if lid[i] {
            "lid"
        } else if protected[i] {
            "protected"
        } else {
            match last_kind[i] {
                0 => "skin",
                1 => "envelope_push",
                2 => "tight_cloth",
                _ if reverse_hits[i] > 0 => "reverse_skin",
                _ => "prior_only",
            }
        };
        *fold_drivers.entry(kind.to_string()).or_default() += 1;
    }
    let mut bump_drivers: BTreeMap<&str, usize> = BTreeMap::new();
    for i in (0..n).filter(|&i| bump_above[i].is_some_and(|b| b.0 > 0.001)) {
        let kind = if protected[i] {
            "protected"
        } else {
            match last_kind[i] {
                0 => "skin",
                1 => "envelope_push",
                _ => "no_data",
            }
        };
        *bump_drivers.entry(kind).or_default() += 1;
    }
    let bump_zones = zones(&|i| bump_above[i].is_some_and(|b| b.0 > 0.001));
    // Left/right asymmetry of the posed prior (the measure's floor) and of
    // the fitted body, in the skeleton's rest frame.
    let asymmetry = {
        let frames_end = symmetry::vertex_frames(&influence_rows, &parents, &shape_rest, &x);
        let unposed = symmetry::unpose(&frames_end, &x);
        let classes = |u: &[V]| symmetry::report(&mirror, u, &region, &evidence_class, &neighbours);
         
        let outside: Vec<bool> = (0..n)
            .into_par_iter()
            .map(|i| {
                scan.query
                    .closest_point(a(x[i]), height * 0.02, None)
                    .is_some_and(|h| (x[i] - v(h.point)).dot(&v(h.normal)) > 0.002)
            })
            .collect();
        let mut outside_by_region: BTreeMap<String, usize> = BTreeMap::new();
        for i in (0..n).filter(|&i| outside[i]) {
            *outside_by_region
                .entry(format!("{:?}", region[i]))
                .or_default() += 1;
        }
        json!({
            "outside_scan_2mm": {
                "by_region": outside_by_region,
                "prior_members": (0..n).filter(|&i| outside[i] && symmetric_member[i]).count(),
            },
            "note": "rest-frame distance between a vertex and its mirrored partner, metres; each pair once; visible = skin on both sides, mixed = one, hidden = neither",
            "mirror": mirror.report,
            "symmetric_prior_weight": p.symmetry,
            "pre_fit": classes(&pre_fit_asymmetry),
            "fitted": classes(&unposed),
        })
    };
    // The bone the most stretched edge belongs to (a scalp, a lid, a toe):
    // the maximum is one edge, so say where.
    let mut distortion_rest = distortion(&body.prim.positions, &fitted, &body.prim.triangles);
    if let Some(i) = distortion_rest["max_edge_vertices"][0].as_u64() {
        distortion_rest["max_edge_joint"] = json!(prior.skeleton().joints[prior.dominant_joint(prior.body, i as usize)].name);
    }
    let mut report = json!({
        "asymmetry": asymmetry,
        "face_landmark_residual_m": landmark_residual,
        "face_landmark_offset_m": landmark_offset,
        "face_landmark_fitted_m": landmark_fitted,
        "face_above_scan": bumps,
        "face_new_fold_vertices": {"pre_fit_aligned": folds_prefit, "fitted": folds_before, "in_ears": (0..n).filter(|&i| fold0[i] && ears.member[i]).count(), "by_joint": fold_joints},
        "guard": guard_reports,
        "fold_zones": fold_zones,
        "fold_drivers": fold_drivers,
        "bump_zones": bump_zones,
        "bump_drivers": bump_drivers,
        "defects_per_iteration": history.iter().map(|h| h["defects"].clone()).collect::<Vec<_>>(),
        "ears": ear_report,
        "ears_drift_after_placement_m": (!ear_placed.is_empty()).then(|| crate::geom::percentiles(
            (0..n).filter(|&i| ears.member[i]).map(|i| (x[i] - ear_placed[i]).norm()).collect(),
        )),
        "ears_after_release": (!ear_placed.is_empty()).then(|| {
            // Released ears against their placement: mean drift per side,
            // and how far placed and final pinnae stand outside the scan.
            let ids: Vec<usize> = (0..n).filter(|&i| ears.member[i]).collect();
            let side = |sgn: f64| {
                let ids: Vec<usize> = ids.iter().copied().filter(|&i| x[i].x * sgn > 0.0).collect();
                let k = ids.len().max(1) as f64;
                let c0 = ids.iter().map(|&i| ear_placed[i]).sum::<V>() / k;
                let d = ids.iter().map(|&i| x[i] - ear_placed[i]).sum::<V>() / k;
                // The head around it (within 3 cm, outside the ear).
                let ring: Vec<usize> = (0..n).filter(|&i| region[i] == Region::Head && !ears.member[i] && (ear_placed[i] - c0).norm() < 0.03 * height / 1.75).collect();
                let dr = ring.iter().map(|&i| x[i] - ear_placed[i]).sum::<V>() / ring.len().max(1) as f64;
                json!({"centroid_placed": a(c0), "mean_drift": a(d), "ring_mean_drift": a(dr), "ring": ring.len()})
            };
            // Placed ear against the scan: signed distance along the nearest
            // scan normal (positive = outside the scan).
            let outside: Vec<f64> = ids.iter().filter_map(|&i| {
                let h = scan.query.closest_point(a(ear_placed[i]), height * 0.05, None)?;
                Some((ear_placed[i] - v(h.point)).dot(&v(h.normal)))
            }).collect();
            let outside_final: Vec<f64> = ids.iter().filter_map(|&i| {
                let h = scan.query.closest_point(a(x[i]), height * 0.05, None)?;
                Some((x[i] - v(h.point)).dot(&v(h.normal)))
            }).collect();
            json!({"left": side(1.0), "right": side(-1.0),
                "placed_outside_scan_m": percentiles(outside), "final_outside_scan_m": percentiles(outside_final)})
        }),
        "human_first": human_report,
        "free_head": sculpt_report,
        "feet": shod_feet.report(),
        "hands": hand_fit.report(),
        "hidden_body": (!hidden_seed.is_empty())
            .then(|| hidden::report(&hidden_w, &hidden_seed, p.hidden_stiffness)),
        "cloth_clearance": clearance_report,
        "silhouette": if silhouette_views.is_empty() {
            Value::Null
        } else {
            let positions: Vec<Vec3> = x.iter().map(|q| a(*q)).collect();
            let unwelded: Vec<Vec3> = weld.id.iter().map(|&w| positions[w]).collect();
            let nu = vertex_normals(&unwelded, &body.prim.triangles, &weld);
            let nw: Vec<Vec3> = weld.representative.iter().map(|&i| nu[i]).collect();
            let (_, _, _, stats) = silhouette::pulls(&silhouette_views, &x, &nw, &welded_triangles, &silhouette_eligible, SILHOUETTE_REACH * height);
            json!(p.contours.iter().zip(&stats).map(|(c, s)| json!({"view": c.camera.id, "outline_vertices": s.0, "p50_m": s.1, "p95_m": s.2})).collect::<Vec<_>>())
        },
        "head_alignment": head_report,
        "eye_opening_warp": eye_warp_report,
        "mouth_warp": mouth_warp_report,
        "head_largest_displacements": moved,
        "largest_local_jumps": jumps,
        "cavity_vertices": cavity.iter().filter(|c| **c).count(),
        "protected_face_vertices": protected.iter().filter(|c| **c).count(),
        "inverted_triangles": flipped_by_region.values().sum::<usize>(),
        "inverted_note": "rotation-aware: triangle normal vs its rest normal carried by the solver's local frames; distortion_*.flipped_triangles compare raw normals and also count limbs posed past 90 degrees",
        "flipped_by_region": flipped_by_region,
        "vertices": body.bind.len(),
        "welded_vertices": n,
        "distortion_vs_prior_rest": distortion_rest,
        "distortion_vs_pose_stage": distortion(posed, &fitted, &body.prim.triangles),
        "scan_skin_to_body_m": percentiles(skin_error),
        "scan_skin_to_body_by_region_m": skin_error_by_region,
        "flipped_per_iteration": history.iter().map(|h| h["flipped"].clone()).collect::<Vec<_>>(),
        "last_iterations": history.iter().rev().take(3).collect::<Vec<_>>(),
        "first_iteration": history.first(),
    });
    report["registration"] = json!(registration_report);
    report["inversions_by_stage"] = json!(stage_inversions);
    report["hidden_head"] = json!(hidden_head_report);
    report["body_field"] = json!(body_field_report);
    Ok(BodyFit {
        cavity: weld.id.iter().map(|&w| cavity[w]).collect(),
        // Free head: the skin round the ears lies on the scan's, nothing to
        // keep from the bake.
        ear_exclude: if p.free_head { vec![0.0; weld.id.len()] } else { ears.bake_exclusion(&shape_rest, &x, &weld.id) },
        shod_foot: weld.id.iter().map(|&w| shod_feet.weight[w]).collect(),
        positions: fitted,
        pinned_head_rig_reference,
        diagnostic,
        report,
    })
}

pub struct Rig {
    pub worlds: Vec<Matrix4<f64>>,
    pub local: Vec<Transform>,
    /// Fitted rest positions per prior mesh (same order as `prior.meshes`).
    pub meshes: Vec<Vec<Vec3>>,
    pub report: Value,
}

fn origin(m: &Matrix4<f64>) -> V {
    V::new(m[(0, 3)], m[(1, 3)], m[(2, 3)])
}

/// Turns joint `j` about its own origin by the world rotation `g` (its rest
/// world and local transforms) with the vertices of prior mesh `mesh` that
/// it mainly carries (an eyeball). Descendant joints keep their world
/// transforms (their local ones are recomputed).
pub fn turn_joint(prior: &Prior, rig: &mut Rig, j: usize, g: &Matrix3<f64>, mesh: usize) {
    let w = rig.worlds[j];
    let o = origin(&w);
    let r = w.fixed_view::<3, 3>(0, 0).into_owned();
    rig.worlds[j] = from_rt(&(g * r), o);
    let skeleton = prior.skeleton();
    let local = |k: usize, worlds: &[Matrix4<f64>]| {
        let parent = skeleton.joints[k]
            .parent
            .map_or(Matrix4::identity(), |p| worlds[p]);
        decompose(&(parent.try_inverse().unwrap_or_else(Matrix4::identity) * worlds[k]))
    };
    rig.local[j] = local(j, &rig.worlds);
    for k in 0..skeleton.joints.len() {
        if skeleton.joints[k].parent == Some(j) {
            rig.local[k] = local(k, &rig.worlds);
        }
    }
    for (p, row) in rig.meshes[mesh]
        .iter_mut()
        .zip(&prior.meshes[mesh].influences)
    {
        if row.iter().any(|x| x.joint == j && x.weight > 0.5) {
            *p = a(o + g * (v(*p) - o));
        }
    }
}

fn from_rt(r: &Matrix3<f64>, t: V) -> Matrix4<f64> {
    let mut m = Matrix4::identity();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(r);
    m.fixed_view_mut::<3, 1>(0, 3).copy_from(&t);
    m
}

fn decompose(m: &Matrix4<f64>) -> Transform {
    let basis = m.fixed_view::<3, 3>(0, 0).into_owned();
    let scale = [0, 1, 2].map(|k| basis.column(k).norm().max(1e-12));
    let mut r = basis;
    for (k, s) in scale.iter().enumerate() {
        r.set_column(k, &(basis.column(k) / *s));
    }
    let q =
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(proper_rotation(r)));
    Transform {
        translation: a(origin(m)),
        rotation: [q.i, q.j, q.k, q.w],
        scale: scale.map(|s| if (s - 1.0).abs() < 1e-5 { 1.0 } else { s }),
    }
}

/// Local affine map around a joint, regularised toward the best similarity in
/// ill-conditioned directions (thin finger neighbourhoods).
fn local_affine(pairs: &[(V, V, f64)]) -> (Matrix3<f64>, V, V) {
    let wsum: f64 = pairs.iter().map(|p| p.2).sum::<f64>().max(1e-12);
    let ca = pairs.iter().fold(V::zeros(), |s, p| s + p.0 * p.2) / wsum;
    let cb = pairs.iter().fold(V::zeros(), |s, p| s + p.1 * p.2) / wsum;
    let mut saa = Matrix3::zeros();
    let mut sba = Matrix3::zeros();
    for (x, y, w) in pairs {
        saa += (x - ca) * (x - ca).transpose() * *w;
        sba += (y - cb) * (x - ca).transpose() * *w;
    }
    let r = proper_rotation(sba);
    let s = ((r.transpose() * sba).trace() / saa.trace().max(1e-20)).clamp(0.5, 2.0);
    let mu = saa.trace() / 3.0 * 0.15 + 1e-12;
    let affine = (sba + r * s * mu)
        * (saa + Matrix3::identity() * mu)
            .try_inverse()
            .unwrap_or_else(Matrix3::identity);
    (affine, ca, cb)
}

/// Eyeball placement from a fitted eye opening: (scale — `None` keeps the
/// head's — opening centre on the MakeHuman rest body, on the fitted body).
pub type EyeFit = BTreeMap<String, (Option<f64>, V, V)>;

 pub fn eye_fits(
    prior: &Prior,
    fitted_body: &[Vec3],
    landmarks: &BTreeMap<String, Vec3>,
    stylized: bool,
    paired: bool,
) -> EyeFit {
    let sets = face_landmark_sets(prior);
    let scales = eye_scales(prior, landmarks);
    let mut out = EyeFit::new();
    for (side, joint) in [("l", "eye.L"), ("r", "eye.R")] {
        let names = ["inner", "outer", "top", "bottom"].map(|k| format!("eye_{side}_{k}"));
        let opening: Vec<&Vec<usize>> = names
            .iter()
            .filter_map(|n| sets.iter().find(|s| s.0 == n.as_str()).map(|s| &s.1))
            .collect();
        if opening.len() < 4 || !names.iter().all(|n| landmarks.contains_key(n)) {
            continue;
        }
        let ids: Vec<usize> = opening.into_iter().flatten().copied().collect();
        let c_rest = set_centroid(&prior.body().prim.positions, &ids);
        let c_fit = set_centroid(fitted_body, &ids);
        if let Some(scale) = scales.get(joint) {
            out.insert(
                joint.to_string(),
                (stylized.then_some(*scale), c_rest, c_fit),
            );
        }
    }
    // A mirrored pair: one eyeball scale, and the fitted opening centres
    // mirror images about the face midplane, apart from the asymmetry the
    // marks keep (`eyes::symmetrize`).
    if !paired {
        return out;
    }
    if let (Some(l), Some(r), Some((plane, kept))) = (
        out.get("eye.L").copied(),
        out.get("eye.R").copied(),
        eyes::kept_asymmetry(landmarks),
    ) {
        let scale = match (l.0, r.0) {
            (Some(a0), Some(b0)) => Some((a0 * b0).sqrt()),
            (s0, s1) => s0.or(s1),
        };
        let mid = (l.2 + plane.mirror(r.2)) * 0.5;
        out.insert("eye.L".into(), (scale, l.1, mid + kept * 0.5));
        out.insert("eye.R".into(), (scale, r.1, plane.mirror(mid - kept * 0.5)));
    }
    out
}

pub fn refit_rig(
    prior: &Prior,
    fitted_body: &[Vec3],
    eyes: &EyeFit,
    held: &BTreeMap<usize, V>,
) -> Result<Rig> {
    let skeleton = prior.skeleton();
    let body = prior.body();
    let original = &body.prim.positions;
    let nodes = skeleton.joints.len();
    let old = &prior.rest_world;
    let is_joint: Vec<bool> = (0..nodes).map(|j| prior.skin_joints.contains(&j)).collect();
    // Vertex lists per joint: driven by the joint or its parent.
    let mut affine = vec![None::<(Matrix3<f64>, V, V)>; nodes];
    let mut fallback = vec![false; nodes];
    for &j in &prior.skin_joints {
        let parent = skeleton.joints[j].parent;
        let center = origin(&old[j]);
        let mut pairs: Vec<(V, V, f64)> = vec![];
        for (i, row) in body.influences.iter().enumerate() {
            let w: f64 = row
                .iter()
                .filter(|x| x.joint == j || Some(x.joint) == parent)
                .map(|x| x.weight)
                .sum();
            if w >= 0.25 {
                pairs.push((v(original[i]), v(fitted_body[i]), w));
            }
        }
        if pairs.len() < 16 {
            fallback[j] = true;
            let mut nearest: Vec<(f64, usize)> = original
                .iter()
                .enumerate()
                .map(|(i, p)| ((v(*p) - center).norm(), i))
                .collect();
            nearest.select_nth_unstable_by(31, |x, y| x.0.total_cmp(&y.0));
            pairs = nearest[..32]
                .iter()
                .map(|&(_, i)| (v(original[i]), v(fitted_body[i]), 1.0))
                .collect();
        }
        // Gaussian falloff around the joint keeps the fit local.
        let mut d: Vec<f64> = pairs.iter().map(|p| (p.0 - center).norm()).collect();
        d.sort_by(f64::total_cmp);
        let sigma = d[d.len() / 2].max(1e-4);
        for p in &mut pairs {
            p.2 *= (-(p.0 - center).norm_squared() / (2.0 * sigma * sigma)).exp() + 1e-3;
        }
        affine[j] = Some(local_affine(&pairs));
    }
    let mut position = vec![V::zeros(); nodes];
    let mut rotation = vec![Matrix3::identity(); nodes];
    let mut scale = vec![1.0f64; nodes];
    for j in 0..nodes {
        let old_r = proper_rotation(old[j].fixed_view::<3, 3>(0, 0).into_owned());
        match &affine[j] {
            Some((m, ca, cb)) => {
                position[j] = cb + m * (origin(&old[j]) - ca);
                rotation[j] = proper_rotation(*m) * old_r;
                scale[j] = m.determinant().abs().cbrt().clamp(0.7, 1.4);
            }
            None => {
                position[j] = origin(&old[j]);
                rotation[j] = old_r;
            }
        }
        // Joints placed elsewhere (`legs::held_joints`); re-aimed below.
        if let Some(p) = held.get(&j) {
            position[j] = *p;
        }
    }
    // Face bones sit on small, often thin neighbourhoods (lids, lips,
    // tongue) whose local fits are unstable. They follow the head rigidly in
    // rotation and scale; only well-supported ones keep their own position.
    let head = prior.joint("head");
    let head_old_r = proper_rotation(old[head].fixed_view::<3, 3>(0, 0).into_owned());
    let head_delta = rotation[head] * head_old_r.transpose();
    let is_face: Vec<bool> = (0..nodes)
        .map(|j| j != head && prior.is_ancestor(head, j))
        .collect();
    for j in 0..nodes {
        if !is_face[j] {
            continue;
        }
        let old_r = proper_rotation(old[j].fixed_view::<3, 3>(0, 0).into_owned());
        rotation[j] = head_delta * old_r;
        scale[j] = scale[head];
        if affine[j].is_none() {
            position[j] =
                position[head] + head_delta * (origin(&old[j]) - origin(&old[head])) * scale[head];
        }
         
        if let Some((s_eye, c_rest, c_fit)) = eyes.get(&skeleton.joints[j].name) {
             
            let wide = std::env::var("MH_EYE_WIDE").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(2.8);
            let s_eye = s_eye.map_or(scale[head], |s| s.max(scale[head]).min(if s > wide { 2.8 } else { 1.8 }));
            position[j] = c_fit + head_delta * (origin(&old[j]) - c_rest) * s_eye;
            scale[j] = s_eye;
        }
    }
     
    for j in 0..nodes {
        if !is_face[j] || eyes.contains_key(&skeleton.joints[j].name) {
            continue;
        }
        if let Some(e) = (0..nodes).find(|&e| {
            eyes.contains_key(&skeleton.joints[e].name)
                && (origin(&old[j]) - origin(&old[e])).norm() < 1e-4
        }) {
            position[j] = position[e];
            scale[j] = scale[e];
        }
    }
    // Re-aim single-child bones at their (moved) child.
    let mut realigned = 0;
    for j in 0..nodes {
        if !is_joint[j] || is_face[j] {
            continue;
        }
        let children: Vec<usize> = (0..nodes)
            .filter(|&c| is_joint[c] && skeleton.joints[c].parent == Some(j))
            .collect();
        if let [c] = children[..] {
            let old_r = proper_rotation(old[j].fixed_view::<3, 3>(0, 0).into_owned());
            let carried = rotation[j] * old_r.transpose() * (origin(&old[c]) - origin(&old[j]));
            let actual = position[c] - position[j];
            if carried.norm() > 1e-6 && actual.norm() > 1e-6 {
                rotation[j] = rotation_between(carried, actual) * rotation[j];
                realigned += 1;
            }
        }
    }
    let worlds: Vec<Matrix4<f64>> = (0..nodes)
        .map(|j| {
            if is_joint[j] {
                from_rt(&rotation[j], position[j])
            } else {
                old[j]
            }
        })
        .collect();
    let local: Vec<Transform> = (0..nodes)
        .map(|j| {
            let parent = skeleton.joints[j]
                .parent
                .map_or(Matrix4::identity(), |p| worlds[p]);
            decompose(&(parent.try_inverse().unwrap_or_else(Matrix4::identity) * worlds[j]))
        })
        .collect();
     
    let delta: Vec<Matrix4<f64>> = (0..nodes)
        .map(|j| {
            let old_r = proper_rotation(old[j].fixed_view::<3, 3>(0, 0).into_owned());
            let r = rotation[j] * old_r.transpose() * scale[j];
            from_rt(&r, position[j] - r * origin(&old[j]))
        })
        .collect();
    // Surface attachment for brows/lashes: triangle frame offsets.
    let body_query = build_query(original, &body.prim.triangles);
    let mut meshes = vec![];
    for (mi, mesh) in prior.meshes.iter().enumerate() {
        if mi == prior.body {
            meshes.push(fitted_body.to_vec());
            continue;
        }
        let moved: Vec<Vec3> = match mesh.kind {
            PartKind::Eyebrows | PartKind::Eyelashes => mesh
                .prim
                .positions
                .par_iter()
                .map(|p| attach(&body_query, &body.prim.triangles, original, fitted_body, *p))
                .collect(),
            _ => mesh
                .prim
                .positions
                .par_iter()
                .zip(&mesh.influences)
                .map(|(p, row)| {
                    let h = v(*p).push(1.0);
                    let s = row.iter().fold(nalgebra::Vector4::zeros(), |s, w| {
                        s + delta[w.joint] * h * w.weight
                    });
                    a(s.xyz())
                })
                .collect(),
        };
        meshes.push(moved);
    }
    let displacement: Vec<f64> = prior
        .skin_joints
        .iter()
        .map(|&j| (position[j] - origin(&old[j])).norm())
        .collect();
    let face_joints: Vec<Value> = ["head", "jaw", "eye.L", "eye.R", "tongue00", "neck03"]
        .iter()
        .filter_map(|name| prior.by_name.get(*name).map(|&j| (name, j)))
        .map(|(name, j)| {
            let m = delta[j].fixed_view::<3, 3>(0, 0).into_owned();
            let angle = Rotation3::from_matrix_unchecked(proper_rotation(m)).angle();
            json!({"joint":name,"moved_m":(position[j] - origin(&old[j])).norm(),"rotation_rad":angle,"scale":scale[j],"fallback_neighbourhood":fallback[j]})
        })
        .collect();
    let report = json!({
        "face_joints": face_joints,
        "joints": prior.skin_joints.len(),
        "bones_reaimed": realigned,
        "joint_displacement_m": percentiles(displacement),
        "weights_preserved": true,
        "eyes": eyes.iter().map(|(k, (sc, cr, cf))| (k.clone(), json!({"scale": prior.by_name.get(k).map(|&j| scale[j]), "width_ratio": sc, "head_scale": scale[head], "shift_m": (cf - cr).norm()}))).collect::<BTreeMap<_, _>>(),
        "inverse_bind_matrices": "recomputed from fitted rest",
    });
    Ok(Rig {
        worlds,
        local,
        meshes,
        report,
    })
}

fn attach(
    query: &charloom_engine::surface_query::SurfaceQuery,
    tris: &[[u32; 3]],
    old: &[Vec3],
    new: &[Vec3],
    p: Vec3,
) -> Vec3 {
    let Some(hit) = query.closest_point(p, 1.0, None) else {
        return p;
    };
    let t = tris[hit.source[1]];
    let frame = |pts: &[Vec3]| {
        let [p0, p1, p2] = t.map(|i| v(pts[i as usize]));
        let e = (p1 - p0).try_normalize(1e-12).unwrap_or(V::x());
        let nrm = (p1 - p0).cross(&(p2 - p0));
        let area = nrm.norm();
        let nrm = nrm / area.max(1e-20);
        let b = hit.barycentric;
        (
            p0 * b[0] + p1 * b[1] + p2 * b[2],
            Matrix3::from_columns(&[e, nrm.cross(&e), nrm]),
            area,
        )
    };
    let (q0, f0, a0) = frame(old);
    let (q1, f1, a1) = frame(new);
    // A collapsed triangle has no frame: carry the offset unrotated.
    if a0 < 1e-14 || a1 < 1e-14 || !(f0.iter().chain(f1.iter()).all(|x| x.is_finite())) {
        return a(q1 + (v(p) - q0));
    }
    let local = f0.transpose() * (v(p) - q0);
    a(q1 + f1 * local * (a1 / a0.max(1e-20)).sqrt().clamp(0.5, 2.0))
}

/// A triangle is inverted only if both normals are well defined and opposed.
fn inverted(now: &V, rest: &V) -> bool {
    let (a0, b0) = (now.norm(), rest.norm());
    a0 > 1e-14 && b0 > 1e-14 && now.dot(rest) < -1e-3 * a0 * b0
}

/// Rest normal of a triangle rotated by the mean local frame of its vertices.
fn carried(frames: &[Matrix3<f64>], tri: &[u32; 3], rest: &V) -> V {
    let m = tri
        .iter()
        .fold(Matrix3::zeros(), |s, &i| s + frames[i as usize]);
    proper_rotation(m) * rest
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inverted_ignores_degenerate_rest_triangles() {
        let up = V::new(0.0, 0.0, 1.0);
        assert!(inverted(&-up, &up));
        assert!(!inverted(&up, &up));
        assert!(
            !inverted(&-up, &V::zeros()),
            "zero-area rest triangles are not flips"
        );
    }

    #[test]
    fn local_affine_recovers_a_similarity_and_regularises_a_line() {
        let r = Rotation3::from_euler_angles(0.2, 0.4, -0.3).into_inner();
        let t = V::new(0.1, -0.2, 0.3);
        let pts = [
            V::new(0., 0., 0.),
            V::new(1., 0., 0.),
            V::new(0., 1., 0.),
            V::new(0., 0., 1.),
            V::new(1., 1., 1.),
        ];
        let pairs: Vec<(V, V, f64)> = pts.iter().map(|p| (*p, r * p * 1.1 + t, 1.0)).collect();
        let (m, ca, cb) = local_affine(&pairs);
        for (a0, b0, _) in &pairs {
            assert!((cb + m * (a0 - ca) - b0).norm() < 1e-6);
        }
        // Collinear samples (a finger): no blow-up in the unseen directions.
        let line: Vec<(V, V, f64)> = (0..10)
            .map(|i| {
                (
                    V::new(i as f64 * 0.01, 0., 0.),
                    V::new(i as f64 * 0.012, 0., 0.),
                    1.0,
                )
            })
            .collect();
        let (m, _, _) = local_affine(&line);
        assert!(m.iter().all(|x| x.is_finite()) && m.norm() < 3.0);
        assert!((m[(0, 0)] - 1.2).abs() < 0.05);
    }
}
