//! Stage 1a — articulated alignment. Rotates the MakeHuman skeleton (and
//! scales limb chains) so the posed body overlays the scan, using
//! bidirectional closest-point correspondences and per-bone Kabsch solves
//! about each joint, root to leaves. No vertex moves independently here, so
//! the body cannot tear; only the rig pose changes.
use crate::geom::{V, a, proper_rotation, v};
use crate::prior::Prior;
use crate::scan::{Scan, build_query};
use anyhow::Result;
use charloom_core::{Transform, Vec3};
use nalgebra::{Matrix3, Matrix4, Rotation3, UnitQuaternion};
use rayon::prelude::*;
use serde_json::{Value, json};

const CONTROLLED: &[&str] = &[
    "spine03",
    "spine01",
    "neck01",
    "head",
    "clavicle.L",
    "upperarm01.L",
    "lowerarm01.L",
    "wrist.L",
    "clavicle.R",
    "upperarm01.R",
    "lowerarm01.R",
    "wrist.R",
    "upperleg01.L",
    "lowerleg01.L",
    "foot.L",
    "upperleg01.R",
    "lowerleg01.R",
    "foot.R",
];
/// Limb chains whose length is estimated: (chain root, chain end).
const CHAINS: &[(&str, &str)] = &[
    ("upperarm01.L", "wrist.L"),
    ("upperarm01.R", "wrist.R"),
    ("upperleg01.L", "foot.L"),
    ("upperleg01.R", "foot.R"),
];

pub struct PoseFit {
    pub pose: Vec<Transform>,
    pub worlds: Vec<Matrix4<f64>>,
    pub report: Value,
}

struct Pair {
    p: V,
    q: V,
    /// Index into CONTROLLED, or usize::MAX for the root group.
    group: usize,
    w: f64,
}

 pub const LANDMARK_JOINTS: &[(&str, &str)] = &[
    ("neck", "neck01"),
    ("shoulder_l", "upperarm01.L"),
    ("shoulder_r", "upperarm01.R"),
    ("elbow_l", "lowerarm01.L"),
    ("elbow_r", "lowerarm01.R"),
    ("wrist_l", "wrist.L"),
    ("wrist_r", "wrist.R"),
    ("hip_l", "upperleg01.L"),
    ("hip_r", "upperleg01.R"),
    ("knee_l", "lowerleg01.L"),
    ("knee_r", "lowerleg01.R"),
    ("ankle_l", "foot.L"),
    ("ankle_r", "foot.R"),
];

fn landmark_of(joint: &str) -> Option<&'static str> {
    LANDMARK_JOINTS
        .iter()
        .find(|(_, j)| *j == joint)
        .map(|(l, _)| *l)
}

fn rotation_of(m: &Matrix4<f64>) -> Matrix3<f64> {
    proper_rotation(m.fixed_view::<3, 3>(0, 0).into_owned())
}

fn origin(m: &Matrix4<f64>) -> V {
    V::new(m[(0, 3)], m[(1, 3)], m[(2, 3)])
}

/// Applies a world-space rotation about joint `j`'s origin to its local TRS.
fn rotate_joint(
    prior: &Prior,
    pose: &mut [Transform],
    worlds: &[Matrix4<f64>],
    j: usize,
    r: &Matrix3<f64>,
) {
    let parent = prior.skeleton().joints[j]
        .parent
        .map_or(Matrix3::identity(), |p| rotation_of(&worlds[p]));
    let local = parent.transpose() * r * parent;
    let q = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(local));
    let old = charloom_core::unit_quat(pose[j].rotation);
    let new = q * old;
    pose[j].rotation = [new.i, new.j, new.k, new.w];
}

fn clamp_rotation(r: Matrix3<f64>, max_angle: f64, damping: f64) -> Matrix3<f64> {
    let rot = Rotation3::from_matrix_unchecked(r);
    match rot.axis_angle() {
        Some((axis, angle)) => {
            Rotation3::from_axis_angle(&axis, (angle * damping).min(max_angle)).into_inner()
        }
        None => Matrix3::identity(),
    }
}

/// Below an ankle, at least this area share of the scan is garment, and of
/// its upward-facing surface ahead of the ankle (instep and toes; skin there
/// is a sandal or a bare foot): closed footwear.
const SHOE_GARMENT_SHARE: f64 = 0.6;
/// Largest turn of a wrist from MakeHuman's rest hand, radians.
const WRIST_RANGE: f64 = 0.6;
/// Weight of a surface pair on cloth in the pose refinement (skin: 1).
const GARMENT_PAIR_WEIGHT: f64 = 0.3;
 const NECK_RANGE: (f64, f64) = (0.35, 1.3);
const SHOE_TOP_GARMENT_SHARE: f64 = 0.75;
/// The shoe's long axis needs this ratio of the horizontal spread's principal
/// variances; a rounder outline keeps the foot's own heading.
const SHOE_ELONGATION: f64 = 2.0;
/// Largest turn of the foot's heading away from MakeHuman's rest heading.
const SHOE_MAX_YAW: f64 = 0.6;
/// A shoe at least this much longer than MakeHuman's foot is oversized
/// (cartoon footwear): its closest points no longer hold the foot, which
/// turns inside it. A shoe close to the foot's size keeps the foot the pose
/// fit gives it (boots with a heel pitch it).
const SHOE_OVERSIZED: f64 = 1.5;
/// A foot whose sole is tilted further than this from flat is turned inside
/// its shoe, whatever the shoe's size.
const FOOT_MAX_TILT: f64 = 0.5;
/// Inner length of a shoe against its outer length (toe box, heel counter).
const SHOE_INNER_LENGTH: f64 = 0.8;
/// Largest foot scale filling a long shoe: realistic and stylised subjects
/// (the stylised end of the body fit's scale range).
const FOOT_SCALE_MAX: (f64, f64) = (1.15, 1.8);
const EXTREME_FOOT_SCALE_MAX: f64 = 2.6;

/// Closed footwear below one ankle, measured on the scan: where the foot
/// inside it points and how long it may be.
struct Shoe {
    side: &'static str,
    joint: usize,
         forward: V,
    /// Length along `forward` and the lowest point (2nd–98th percentiles).
    length: f64,
    floor: f64,
         rest_forward: V,
    rest_length: f64,
    rest_ankle_height: f64,
}

/// A scan triangle for the shoe measurement.
struct Face {
    centroid: V,
    area: f64,
    normal: V,
    label: crate::segment::Label,
}

/// Weighted principal horizontal (x–z) axis of `points`, turned to face +z,
/// and the ratio of the principal variances.
fn horizontal_axis(points: &[(V, f64)]) -> Option<(V, f64)> {
    let w: f64 = points.iter().map(|p| p.1).sum();
    if w <= 0.0 {
        return None;
    }
    let m = points.iter().fold(V::zeros(), |s, (p, k)| s + p * *k) / w;
    let (mut xx, mut zz, mut xz) = (0.0, 0.0, 0.0);
    for (p, k) in points {
        let d = p - m;
        xx += d.x * d.x * k;
        zz += d.z * d.z * k;
        xz += d.x * d.z * k;
    }
    let (xx, zz, xz) = (xx / w, zz / w, xz / w);
    let mean = (xx + zz) * 0.5;
    let spread = (((xx - zz) * 0.5).powi(2) + xz * xz).sqrt();
    let (major, minor) = (mean + spread, (mean - spread).max(1e-12));
    let angle = 0.5 * (2.0 * xz).atan2(xx - zz);
    let mut axis = V::new(angle.cos(), 0.0, angle.sin());
    if axis.z < 0.0 {
        axis = -axis;
    }
    Some((axis, major / minor))
}

/// Value at quantile `q` of `xs` (sorted in place).
fn quantile(xs: &mut [f64], q: f64) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[((xs.len() - 1) as f64 * q).round() as usize]
}

/// Yaw (about +y) from horizontal direction `from` to `to`.
fn yaw_between(from: V, to: V) -> f64 {
    from.cross(&to).y.atan2(from.dot(&to))
}

 fn measure_shoes(
    prior: &Prior,
    scan: &Scan,
    labels: &[crate::segment::Label],
    worlds: &[Matrix4<f64>],
    guide: &JointGuide,
) -> (Vec<Shoe>, Value) {
    use crate::segment::Label;
    let height = prior.height();
    let sides = [
        ("left", "foot.L", "ankle_l"),
        ("right", "foot.R", "ankle_r"),
    ];
    let ankle: Vec<V> = sides
        .iter()
        .map(|(_, j, l)| {
            let p = origin(&worlds[prior.joint(j)]);
            guide.target(l).map_or(p, |t| t.nearest(p))
        })
        .collect();
    let flat = |p: V| V::new(p.x, 0.0, p.z);
    let faces: Vec<Face> = scan
        .triangles
        .par_iter()
        .zip(labels)
        .map(|(t, &l)| {
            let [p0, p1, p2] = t.map(|i| v(scan.positions[i as usize]));
            let n = (p1 - p0).cross(&(p2 - p0));
            let area = n.norm() * 0.5;
            Face {
                centroid: (p0 + p1 + p2) / 3.0,
                area,
                normal: n / (2.0 * area).max(1e-20),
                label: l,
            }
        })
        .collect();
    // MakeHuman's rest feet.
    let rest = prior.skin_mesh(prior.body, &prior.rest_world);
    let mut shoes = vec![];
    let mut report = vec![];
    for (s, &(side, joint_name, _)) in sides.iter().enumerate() {
        let joint = prior.joint(joint_name);
        let (own, other) = (ankle[s], ankle[1 - s]);
        let below: Vec<&Face> = faces
            .iter()
            .filter(|f| {
                let c = f.centroid;
                let d = flat(c - own).norm();
                f.label != Label::Hair
                    && c.y < own.y
                    && d < 0.25 * height
                    && d < flat(c - other).norm()
            })
            .collect();
        let area = |keep: &dyn Fn(&Face) -> bool| -> f64 {
            below.iter().filter(|f| keep(f)).map(|f| f.area).sum()
        };
        let garment = |f: &Face| f.label == Label::Garment;
        let top = |f: &Face| f.normal.y > 0.5 && f.centroid.z > own.z;
        let garment_share = area(&garment) / area(&|_| true).max(1e-20);
        let top_share = area(&|f| top(f) && garment(f)) / area(&top).max(1e-20);
        let points: Vec<(V, f64)> = below.iter().map(|f| (f.centroid, f.area)).collect();
        let axis = horizontal_axis(&points);
        // MakeHuman's foot at rest (flat on the floor): heading and size.
        let foot: Vec<V> = (0..rest.len())
            .filter(|&i| prior.is_ancestor(joint, prior.dominant_joint(prior.body, i)))
            .map(|i| v(rest[i]))
            .collect();
        let rest_axis = horizontal_axis(&foot.iter().map(|p| (*p, 1.0)).collect::<Vec<_>>());
        let closed = below.len() >= 50
            && garment_share >= SHOE_GARMENT_SHARE
            && top_share >= SHOE_TOP_GARMENT_SHARE;
        let mut entry = json!({
            "side": side,
            "ankle": [own.x, own.y, own.z],
            "triangles": below.len(),
            "garment_share": garment_share,
            "top_garment_share": top_share,
            "closed": closed,
        });
         
        let bare_measured = !closed && guide.guided && guide.foot_lengths[s].is_some() && below.len() >= 50;
        entry["bare_measured"] = json!(bare_measured);
        if let (true, Some((forward, elongation)), Some((rest_forward, _))) =
            (closed || bare_measured, axis, rest_axis)
        {
            let along = |p: V| (p - own).dot(&forward);
            let mut t: Vec<f64> = below.iter().map(|f| along(f.centroid)).collect();
            let mut y: Vec<f64> = below.iter().map(|f| f.centroid.y).collect();
            let length = quantile(&mut t, 0.98) - quantile(&mut t, 0.02);
            let floor = quantile(&mut y, 0.02);
            let mut rt: Vec<f64> = foot.iter().map(|p| p.dot(&rest_forward)).collect();
            let rest_length = quantile(&mut rt, 1.0) - quantile(&mut rt, 0.0);
            let sole = foot.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
            let rest_ankle_height = origin(&prior.rest_world[joint]).y - sole;
            // A round outline has no reliable long axis: keep the heading.
            let yaw = if elongation >= SHOE_ELONGATION {
                yaw_between(rest_forward, forward).clamp(-SHOE_MAX_YAW, SHOE_MAX_YAW)
            } else {
                0.0
            };
            let forward =
                Rotation3::from_axis_angle(&nalgebra::Vector3::y_axis(), yaw) * rest_forward;
            // The foot as the pose fit has it so far: its sole's tilt.
            let up = rotation_of(&worlds[joint])
                * rotation_of(&prior.rest_world[joint]).transpose()
                * V::y();
            let tilt = up.dot(&V::y()).clamp(-1.0, 1.0).acos();
            let ratio = length / rest_length.max(1e-6);
            let set = ratio >= SHOE_OVERSIZED || tilt > FOOT_MAX_TILT;
            entry["length_ratio"] = json!(ratio);
            entry["foot_tilt_rad"] = json!(tilt);
            entry["foot_set_by_shoe"] = json!(set);
            entry["elongation"] = json!(elongation);
            entry["yaw_from_rest_rad"] = json!(yaw);
            entry["length_m"] = json!(length);
            entry["floor_m"] = json!(floor);
            entry["rest_foot_length_m"] = json!(rest_length);
            if !set && !bare_measured {
                report.push(entry);
                continue;
            }
            shoes.push(Shoe {
                side,
                joint,
                forward,
                length,
                floor,
                rest_forward,
                rest_length,
                rest_ankle_height,
            });
        }
        report.push(entry);
    }
    (shoes, json!(report))
}

/// Sets a foot in closed footwear: MakeHuman's rest foot (flat, sole on the
/// floor) turned to the shoe's heading about the ankle, and scaled up to
/// the shoe's inner length when the shoe is longer than the foot — bounded
/// by the room between the ankle and the shoe's floor, and by the subject's
/// scale range.
fn align_foot(
    prior: &Prior,
    pose: &mut [Transform],
    shoe: &Shoe,
    stylized: bool,
    measured: Option<f64>,
    extreme: bool,
) -> Result<Value> {
    let worlds = prior.worlds(pose)?;
    let j = shoe.joint;
     
    let yaw = if measured.is_some() { 0.0 } else { yaw_between(shoe.rest_forward, shoe.forward) };
    let target = Rotation3::from_axis_angle(&nalgebra::Vector3::y_axis(), yaw).into_inner()
        * rotation_of(&prior.rest_world[j]);
    let r = target * rotation_of(&worlds[j]).transpose();
    let turn = Rotation3::from_matrix_unchecked(r).angle();
    rotate_joint(prior, pose, &worlds, j, &r);
    let ankle = origin(&worlds[j]);
     
    let room = (ankle.y - shoe.floor) * 0.9 / shoe.rest_ankle_height.max(1e-6)
        * if extreme && measured.is_some() { 1.3 } else { 1.0 };
     
    let cap = if extreme && measured.is_some() {
        EXTREME_FOOT_SCALE_MAX
    } else if stylized {
        FOOT_SCALE_MAX.1
    } else {
        FOOT_SCALE_MAX.0
    };
    // The bare foot's measured length when the guidance has it (smaller
    // than MakeHuman's too); else grown to the shoe's inner length.
    let wanted = match measured {
        Some(len) => len / shoe.rest_length.max(1e-6),
        None => SHOE_INNER_LENGTH * shoe.length / shoe.rest_length.max(1e-6),
    };
    let scale = if measured.is_some() {
        wanted.clamp(0.7, cap).min(room)
    } else {
        wanted.min(cap).min(room).max(1.0)
    };
    pose[j].scale = prior.skeleton().joints[j].rest.scale.map(|r| r * scale);
    Ok(json!({
        "side": shoe.side,
        "turn_rad": turn,
        "scale": scale,
        "wanted_scale": wanted,
        "measured_length_m": measured,
        "room_scale": room,
    }))
}

 pub fn fit(
    prior: &Prior,
    scan: &Scan,
    iterations: usize,
    labels: Option<&[crate::segment::Label]>,
    start: Option<Vec<Transform>>,
    guide: &JointGuide,
) -> Result<PoseFit> {
    let guided: Vec<(usize, Target, f64)> = LANDMARK_JOINTS
        .iter()
        .filter_map(|(l, j)| {
            Some((
                prior.joint(j),
                guide.target(l)?,
                guide.confidence.get(*l).copied().unwrap_or(1.0),
            ))
        })
        .collect();
    let is_hair_triangle = |t: usize| labels.is_some_and(|l| l[t] == crate::segment::Label::Hair);
    // Refinement: cloth counts for less than skin. Loose sleeves, robes and
    // baggy trousers stand off the body; weighted like skin they turned an
    // archer's forearms inward and put her knees 7 cm off their marks.
    let cloth_weight = |t: usize| {
        if labels.is_some_and(|l| l[t] == crate::segment::Label::Garment) {
            GARMENT_PAIR_WEIGHT
        } else {
            1.0
        }
    };
    let skeleton = prior.skeleton();
    let height = prior.height();
    let controlled: Vec<usize> = CONTROLLED.iter().map(|n| prior.joint(n)).collect();
    // Group of every skeleton node: nearest controlled ancestor-or-self.
    let group_of: Vec<usize> = (0..skeleton.joints.len())
        .map(|mut j| {
            loop {
                if let Some(k) = controlled.iter().position(|&c| c == j) {
                    return k;
                }
                match skeleton.joints[j].parent {
                    Some(p) => j = p,
                    None => return usize::MAX,
                }
            }
        })
        .collect();
    // Controlled group g is in the subtree of controlled c?
    let in_subtree =
        |c: usize, g: usize| g != usize::MAX && prior.is_ancestor(controlled[c], controlled[g]);

    let wrists: Vec<usize> = ["wrist.L", "wrist.R"]
        .iter()
        .map(|n| prior.joint(n))
        .collect();
    let body = prior.body();
    let vertex_group: Vec<usize> = (0..body.bind.len())
        .map(|i| group_of[prior.dominant_joint(prior.body, i)])
        .collect();
    let scan_stride = (scan.positions.len() / 12_000).max(1);
    let scan_samples: Vec<usize> = (0..scan.weld.count)
        .step_by(scan_stride.max(1))
        .map(|w| scan.weld.representative[w])
        .collect();
    let hair_vertex: Vec<bool> = match labels {
        Some(l) => {
            let mut h = vec![false; scan.positions.len()];
            for (t, tri) in scan.triangles.iter().enumerate() {
                if l[t] == crate::segment::Label::Hair {
                    tri.iter().for_each(|&i| h[i as usize] = true);
                }
            }
            h
        }
        None => vec![],
    };
    let garment_vertex: Vec<bool> = match labels {
        Some(l) => {
            let mut g = vec![false; scan.positions.len()];
            for (t, tri) in scan.triangles.iter().enumerate() {
                if l[t] == crate::segment::Label::Garment {
                    tri.iter().for_each(|&i| g[i as usize] = true);
                }
            }
            g
        }
        None => vec![],
    };
    let scan_samples: Vec<usize> = scan_samples
        .into_iter()
        .filter(|&s| !hair_vertex.get(s).copied().unwrap_or(false))
        .collect();
    let refine = start.is_some();
     
    let head_group = CONTROLLED.iter().position(|n| *n == "head").unwrap();
    let neck_group = CONTROLLED.iter().position(|n| *n == "neck01").unwrap();
    let face_marks: Vec<(Vec<usize>, V, f64)> = if guide.guided {
        crate::fit::face_landmark_sets(prior)
            .into_iter()
            .filter_map(|(n, set)| {
                let target = v(*guide.points.get(n)?);
                Some((set, target, guide.confidence.get(n).copied().unwrap_or(1.0)))
            })
            .collect()
    } else {
        vec![]
    };
    let face_marks = if face_marks.len() >= 3 { face_marks } else { vec![] };
    let neck_chain: Vec<usize> = ["neck02", "neck03", "head"]
        .iter()
        .filter_map(|n| prior.by_name.get(*n).copied())
        .collect();
    let chain_length = |pose: &[Transform]| -> f64 {
        neck_chain.iter().map(|&j| v(pose[j].translation).norm()).sum()
    };
    let neck_rest = chain_length(&skeleton.rest_pose());

    let mut pose = start.unwrap_or_else(|| skeleton.rest_pose());
    let mut report_iterations = vec![];
    for iteration in 0..iterations {
        let progress = iteration as f64 / iterations.max(1) as f64;
        let tau = if refine {
            height * (0.04 * (1.0 - progress) + 0.02 * progress)
        } else {
            height * (0.12 * (1.0 - progress) + 0.03 * progress)
        };
        let worlds = prior.worlds(&pose)?;
        let posed = prior.skin_mesh(prior.body, &worlds);
        let normals = crate::geom::vertex_normals(
            &posed,
            &body.prim.triangles,
            &crate::geom::Weld::new(&body.bind, 1e-7),
        );
         
        let mut pairs: Vec<Pair> = posed
            .par_iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let hit = scan.query.closest_point(*p, tau, Some((normals[i], 0.2)))?;
                if is_hair_triangle(hit.source[1]) {
                    return None;
                }
                Some(Pair {
                    p: v(*p),
                    q: v(hit.point),
                    group: vertex_group[i],
                    w: cloth_weight(hit.source[1]),
                })
            })
            .collect();
         
        let body_query = build_query(&posed, &body.prim.triangles);
        let reverse: Vec<Pair> = scan_samples
            .par_iter()
            .filter_map(|&s| {
                let n = scan.normals[s];
                let hit = body_query.closest_point(scan.positions[s], tau, Some((n, 0.2)))?;
                let t = body.prim.triangles[hit.source[1]];
                let k =
                    (0..3).max_by(|&x, &y| hit.barycentric[x].total_cmp(&hit.barycentric[y]))?;
                Some(Pair {
                    p: v(hit.point),
                    q: v(scan.positions[s]),
                    group: vertex_group[t[k] as usize],
                    w: if garment_vertex.get(s).copied().unwrap_or(false) {
                        GARMENT_PAIR_WEIGHT
                    } else {
                        1.0
                    },
                })
            })
            .collect();
        let forward_count = pairs.len();
        pairs.extend(reverse);
         
        let landmark_weight = (pairs.len() as f64 * if guide.guided { 0.2 } else { 0.02 }).max(20.0);
        pairs.extend(guided.iter().map(|(j, target, c)| {
            let p = origin(&worlds[*j]);
            Pair {
                p,
                q: target.nearest(p),
                group: group_of[*j],
                w: landmark_weight * c,
            }
        }));
         
        let face_weight = landmark_weight * 4.0 / face_marks.len().max(1) as f64;
        let mut face_pairs: Vec<Pair> = face_marks
            .iter()
            .map(|(set, target, c)| Pair {
                p: set.iter().fold(V::zeros(), |s, &i| s + v(posed[i])) / set.len() as f64,
                q: *target,
                group: head_group,
                w: face_weight * c,
            })
            .collect();
        if pairs.len() < 100 {
            anyhow::bail!(
                "pose_fit_no_correspondences: the scan does not overlap the prior; check facing/height"
            );
        }

        // Root translation: robust (median) offset.
        let root = prior.joint("root");
        let mut delta = [0.0; 3];
        for (k, d) in delta.iter_mut().enumerate() {
            let mut xs: Vec<f64> = pairs.iter().map(|p| p.q[k] - p.p[k]).collect();
            xs.sort_by(f64::total_cmp);
            *d = xs[xs.len() / 2] * 0.8;
        }
        // With enough vision joints, the root follows them: loose clothing
        // and big hair bias the surface median, the LLM's joints do not.
        let total_conf: f64 = guided.iter().map(|g| g.2).sum();
        if guided.len() >= 4 && total_conf > 1e-6 {
            let mean = guided.iter().fold(V::zeros(), |s, (j, target, c)| {
                let p = origin(&worlds[*j]);
                s + (target.nearest(p) - p) * *c
            }) / total_conf;
             
            let alpha = if guide.guided { 1.0 } else { 0.7 } * (total_conf / guided.len() as f64);
            for k in 0..3 {
                delta[k] = delta[k] * (1.0 - alpha) + mean[k] * alpha * 0.8;
            }
        }
        let parent = skeleton.joints[root]
            .parent
            .map_or(Matrix3::identity(), |p| rotation_of(&worlds[p]));
        let local = parent.transpose() * v(delta);
        for k in 0..3 {
            pose[root].translation[k] += local[k];
        }
        for p in pairs.iter_mut().chain(face_pairs.iter_mut()) {
            p.p += v(delta);
        }

        // Hierarchical per-bone rotations.
        let mut worlds = prior.worlds(&pose)?;
        let mut rotations = vec![];
        for (c, &joint) in controlled.iter().enumerate() {
            let center = origin(&worlds[joint]);
            let mut m = Matrix3::zeros();
            let mut count = 0usize;
            for pair in pairs.iter().filter(|p| in_subtree(c, p.group)) {
                let w = pair.w * if pair.group == c { 1.0 } else { 0.5 };
                m += (pair.q - center) * (pair.p - center).transpose() * w;
                count += 1;
            }
            if c == head_group || c == neck_group {
                for pair in &face_pairs {
                    let w = pair.w * if c == head_group { 1.0 } else { 0.5 };
                    m += (pair.q - center) * (pair.p - center).transpose() * w;
                }
            }
            if count < 30 {
                continue;
            }
            let r = clamp_rotation(proper_rotation(m), 0.25, 0.6);
            rotate_joint(prior, &mut pose, &worlds, joint, &r);
            // Anatomical range: a wrist stays within WRIST_RANGE of
            // MakeHuman's rest hand (relative to the forearm). Per-step
            // clamps alone let a hand drift 90° over the iterations, fingers
            // pointing sideways, towards a bow or a sleeve beside it.
            if wrists.contains(&joint) {
                let rest = charloom_core::unit_quat(skeleton.joints[joint].rest.rotation);
                let now = charloom_core::unit_quat(pose[joint].rotation);
                let rel = rest.inverse() * now;
                if rel.angle() > WRIST_RANGE {
                    let limited = rest
                        * UnitQuaternion::from_axis_angle(
                            &rel.axis().unwrap_or(nalgebra::Vector3::x_axis()),
                            WRIST_RANGE,
                        );
                    pose[joint].rotation = [limited.i, limited.j, limited.k, limited.w];
                }
            }
            for pair in pairs.iter_mut().filter(|p| in_subtree(c, p.group)) {
                pair.p = center + r * (pair.p - center);
            }
            if in_subtree(c, head_group) {
                for pair in &mut face_pairs {
                    pair.p = center + r * (pair.p - center);
                }
            }
            worlds = prior.worlds(&pose)?;
            let angle = Rotation3::from_matrix_unchecked(r).angle();
            rotations.push(angle);
        }

         
        let mut scales = vec![];
        if iteration >= 2 {
            // (chain index, step, evidence weight)
            let mut steps: Vec<(usize, f64, f64)> = vec![];
            for (ci, &(start, end)) in CHAINS.iter().enumerate() {
                let (s, e) = (prior.joint(start), prior.joint(end));
                let c = controlled.iter().position(|&x| x == s).unwrap();
                let base = origin(&worlds[s]);
                let Some(axis) = (origin(&worlds[e]) - base).try_normalize(1e-9) else {
                    continue;
                };
                // Guided: the LLM's joints give the chain length directly
                // (measured across the viewing direction for single-view rays).
                if let (Some(ls), Some(le)) = (
                    landmark_of(start).and_then(|l| guide.target(l)),
                    landmark_of(end).and_then(|l| guide.target(l)),
                ) {
                    let (mh_s, mh_e) = (base, origin(&worlds[e]));
                    let ratio = (Target::span(&ls, &le, mh_s, mh_e)
                        / Target::span_model(&ls, &le, mh_s, mh_e).max(1e-6))
                    .clamp(0.8, 1.25);
                    let step = ratio.powf(0.5);
                    let conf = [start, end]
                        .iter()
                        .filter_map(|j| landmark_of(j))
                        .map(|l| guide.confidence.get(l).copied().unwrap_or(1.0))
                        .fold(1.0f64, f64::min);
                    steps.push((ci, step, conf.max(0.05)));
                    scales.push(json!({"chain":start,"step":step,"from":"landmarks"}));
                    continue;
                }
                let mut model: Vec<f64> = vec![];
                let mut target: Vec<f64> = vec![];
                for pair in pairs.iter().filter(|p| in_subtree(c, p.group)) {
                    model.push((pair.p - base).dot(&axis));
                    target.push((pair.q - base).dot(&axis));
                }
                if model.len() < 50 {
                    continue;
                }
                model.sort_by(f64::total_cmp);
                target.sort_by(f64::total_cmp);
                let at = |x: &[f64]| x[(x.len() - 1) * 97 / 100];
                let ratio = (at(&target) / at(&model).max(1e-6)).clamp(0.9, 1.1);
                let ratio = ratio.powf(0.5);
                steps.push((ci, ratio, 0.3));
                scales.push(json!({"chain":start,"step":ratio}));
            }
             
            let mut coupled = steps.clone();
            for pair in [[0usize, 1], [2, 3]] {
                let both: Vec<&(usize, f64, f64)> =
                    steps.iter().filter(|s| pair.contains(&s.0)).collect();
                if both.len() == 2 {
                    let w = both[0].2 + both[1].2;
                    let mean =
                        ((both[0].1.ln() * both[0].2 + both[1].1.ln() * both[1].2) / w).exp();
                    for c in coupled.iter_mut().filter(|s| pair.contains(&s.0)) {
                        c.1 = mean;
                    }
                }
            }
            for &(ci, step, _) in &coupled {
                let (start, end) = CHAINS[ci];
                let (s, e) = (prior.joint(start), prior.joint(end));
                // Scale every local offset on the chain below the chain root.
                let mut j = e;
                while j != s {
                    for k in 0..3 {
                        pose[j].translation[k] *= step;
                    }
                    j = skeleton.joints[j].parent.unwrap();
                }
            }
            scales.push(json!({"coupled_steps": coupled.iter().map(|c| (CHAINS[c.0].0, c.1)).collect::<Vec<_>>()}));
        }
        // Guided torso length, pelvis width and shoulder width: proportions
        // that limb chains cannot change (stylised characters differ most).
        if iteration >= 2 {
            let scale_offsets = |joints: &[&str], step: f64, pose: &mut [Transform]| {
                for j in joints {
                    if let Some(&k) = prior.by_name.get(*j) {
                        for c in 0..3 {
                            pose[k].translation[c] *= step;
                        }
                    }
                }
            };
            let at = |n: &str| origin(&worlds[prior.joint(n)]);
            if let (Some(neck), Some(hl), Some(hr)) = (
                guide.target("neck"),
                guide.target("hip_l"),
                guide.target("hip_r"),
            ) {
                let (mn, ml, mr) = (at("neck01"), at("upperleg01.L"), at("upperleg01.R"));
                let target = neck.nearest(mn) - (hl.nearest(ml) + hr.nearest(mr)) * 0.5;
                let model = mn - (ml + mr) * 0.5;
                let across = |d: V| match neck {
                    Target::Ray(_, dir) => (d - dir * d.dot(&dir)).norm(),
                    Target::Point(_) => d.norm(),
                };
                let step = (across(target) / across(model).max(1e-6))
                    .clamp(0.8, 1.25)
                    .powf(0.5);
                scale_offsets(
                    &[
                        "spine05", "spine04", "spine03", "spine02", "spine01", "neck01",
                    ],
                    step,
                    &mut pose,
                );
                scales.push(json!({"chain":"torso","step":step,"from":"landmarks"}));
            }
            for (l, r, joints) in [
                (
                    "hip_l",
                    "hip_r",
                    &["pelvis.L", "pelvis.R", "upperleg01.L", "upperleg01.R"][..],
                ),
                (
                    "shoulder_l",
                    "shoulder_r",
                    &[
                        "clavicle.L",
                        "clavicle.R",
                        "shoulder01.L",
                        "shoulder01.R",
                        "upperarm01.L",
                        "upperarm01.R",
                    ][..],
                ),
            ] {
                let (Some(tl), Some(tr)) = (guide.target(l), guide.target(r)) else {
                    continue;
                };
                let (jl, jr) = if l == "hip_l" {
                    ("upperleg01.L", "upperleg01.R")
                } else {
                    ("upperarm01.L", "upperarm01.R")
                };
                let (pl, pr) = (at(jl), at(jr));
                let step = (Target::span(&tl, &tr, pl, pr)
                    / Target::span_model(&tl, &tr, pl, pr).max(1e-6))
                .clamp(0.8, 1.25)
                .powf(0.5);
                scale_offsets(joints, step, &mut pose);
                scales
                    .push(json!({"chain":format!("{l}-{r} width"),"step":step,"from":"landmarks"}));
            }
        }
        // Guided neck length: where the face marks put the head joint (a
        // similarity of MakeHuman's marks onto the scan's, so a bigger
        // cartoon face moves the joint with it) against where it is.
        if iteration >= 2 && !face_pairs.is_empty() && neck_chain.len() == 3 {
            let wsum: f64 = face_pairs.iter().map(|p| p.w).sum::<f64>().max(1e-12);
            let ca = face_pairs.iter().fold(V::zeros(), |s, p| s + p.p * p.w) / wsum;
            let cb = face_pairs.iter().fold(V::zeros(), |s, p| s + p.q * p.w) / wsum;
            let m = face_pairs.iter().fold(Matrix3::zeros(), |s, p| {
                s + (p.q - cb) * (p.p - ca).transpose() * p.w
            });
            let r = proper_rotation(m);
            let var = face_pairs
                .iter()
                .map(|p| (p.p - ca).norm_squared() * p.w)
                .sum::<f64>()
                .max(1e-20);
            let s = ((r.transpose() * m).trace() / var).clamp(0.5, 2.0);
            let head = origin(&worlds[prior.joint("head")]);
            let base = origin(&worlds[prior.joint("neck01")]);
            let want = cb + r * (head - ca) * s;
            let ratio = ((want - base).norm() / (head - base).norm().max(1e-6))
                .clamp(0.8, 1.25)
                .powf(0.5);
            let now = chain_length(&pose);
            let step = (now * ratio).clamp(NECK_RANGE.0 * neck_rest, NECK_RANGE.1 * neck_rest)
                / now.max(1e-9);
            for &j in &neck_chain {
                for k in 0..3 {
                    pose[j].translation[k] *= step;
                }
            }
            scales.push(json!({"chain":"neck","step":step,"length_m":now * step,"rest_m":neck_rest,"from":"face marks"}));
        }
        let distances: Vec<f64> = pairs[..forward_count]
            .iter()
            .map(|p| (p.q - p.p).norm())
            .collect();
        report_iterations.push(json!({
            "iteration": iteration,
            "tau_m": tau,
            "pairs": pairs.len(),
            "forward_distance": crate::geom::percentiles(distances),
            "max_rotation_rad": rotations.iter().cloned().fold(0.0, f64::max),
            "chain_scale_steps": scales,
        }));
    }
    // A stylised subject's knees and ankles across their bones onto the
    // centres of the scan's visible legs (refinement only: the labels tell
    // skin from cloth). A realistic leg is about MakeHuman's own thickness:
    // its closest points and landmarks place it already.
    let legs_report = match labels {
        Some(l) if guide.legs && guide.stylized => crate::legs::align(prior, scan, l, &mut pose)?,
        _ => json!({"applied": false}),
    };
     
    let (shoes, shoe_report) = match labels {
        Some(l) => measure_shoes(prior, scan, l, &prior.worlds(&pose)?, guide),
        None => (vec![], json!(null)),
    };
     
    let mut shoes = shoes;
    if guide.guided && guide.foot_lengths.iter().all(|l| l.is_some()) && shoes.len() == 2 {
        let mirror = |f: V| V::new(-f.x, f.y, f.z);
        let (l, r) = if shoes[0].side == "left" { (0, 1) } else { (1, 0) };
        let avg = (shoes[l].forward + mirror(shoes[r].forward)).normalize();
        shoes[l].forward = avg;
        shoes[r].forward = mirror(avg);
    }
    let shoe_alignment = shoes
        .iter()
        .map(|shoe| {
            let measured = guide.foot_lengths[usize::from(shoe.side != "left")];
            align_foot(prior, &mut pose, shoe, guide.stylized, measured, guide.extreme)
        })
        .collect::<Result<Vec<_>>>()?;
    let worlds = prior.worlds(&pose)?;
    let residuals: serde_json::Map<String, Value> = LANDMARK_JOINTS
        .iter()
        .filter_map(|(l, j)| {
            let target = guide.target(l)?;
            let p = origin(&worlds[prior.joint(j)]);
            Some((l.to_string(), json!((target.nearest(p) - p).norm())))
        })
        .collect();
    let joints: serde_json::Map<String, Value> = LANDMARK_JOINTS
        .iter()
        .map(|(l, j)| {
            let p = origin(&worlds[prior.joint(j)]);
            (l.to_string(), json!([p.x, p.y, p.z]))
        })
        .collect();
    let face_residual: Vec<f64> = {
        let posed = prior.skin_mesh(prior.body, &worlds);
        face_marks
            .iter()
            .map(|(set, target, _)| {
                (set.iter().fold(V::zeros(), |s, &i| s + v(posed[i])) / set.len() as f64 - target).norm()
            })
            .collect()
    };
    Ok(PoseFit {
        report: json!({
            "iterations": report_iterations.last(),
            "history": report_iterations.len(),
            "joint_landmark_residual_m": residuals,
            "joints_m": joints,
            "face_marks": face_marks.len(),
            "face_mark_residual_m": face_residual,
            "neck_length_m": {"fitted": chain_length(&pose), "rest": neck_rest},
            "legs": legs_report,
            "feet_in_footwear": {"shoes": shoe_report, "alignment": shoe_alignment},
        }),
        pose,
        worlds,
    })
}

 pub fn facing_cost(prior: &Prior, scan: &Scan, yaw: f64) -> f64 {
    let r = Rotation3::from_axis_angle(&nalgebra::Vector3::y_axis(), yaw);
    let body = &prior.body().prim.positions;
    let center = body.iter().fold(V::zeros(), |s, p| s + v(*p)) / body.len() as f64;
    let limit = prior.height() * 0.1;
    let total: Vec<f64> = body
        .par_iter()
        .step_by(7)
        .map(|p| {
            let q: Vec3 = a(center + r * (v(*p) - center));
            scan.query
                .closest_point(q, limit, None)
                .map_or(limit, |h| h.distance_m)
        })
        .collect::<Vec<f64>>();
    crate::geom::ordered_sum(&total) / (body.len() / 7).max(1) as f64
}

/// Joint guidance from the vision stage: triangulated points and rays.
#[derive(Default)]
pub struct JointGuide {
    pub points: std::collections::BTreeMap<String, Vec3>,
    pub rays: std::collections::BTreeMap<String, (Vec3, Vec3)>,
    pub confidence: std::collections::BTreeMap<String, f64>,
    /// A stylised subject: feet may grow up to the stylised scale range to
    /// fill oversized cartoon footwear.
    pub stylized: bool,
    /// A stylised subject's knees and ankles on the scan's visible leg
    /// centres (`legs.rs`).
    pub legs: bool,
    /// Body from the guidance: the joint marks hold the skeleton, the
    /// surface only fills in (`Config::guided_body`).
    pub guided: bool,
    /// Bare foot lengths heel to toe (left, right; working frame m) from the
    /// guidance (`outline_foot_*_d`): a shod foot takes this size, not its
    /// shoe's.
    pub foot_lengths: [Option<f64>; 2],
    /// `subject.proportions` is `extreme`.
    pub extreme: bool,
}

#[derive(Clone, Copy)]
pub enum Target {
    Point(V),
    Ray(V, V),
}

impl JointGuide {
    pub fn target(&self, name: &str) -> Option<Target> {
        if let Some(p) = self.points.get(name) {
            return Some(Target::Point(v(*p)));
        }
        self.rays
            .get(name)
            .map(|(o, d)| Target::Ray(v(*o), v(*d).normalize()))
    }
}

impl Target {
    /// Closest point of the target to `p` (for a ray: its foot point).
    pub fn nearest(&self, p: V) -> V {
        match *self {
            Target::Point(q) => q,
            Target::Ray(o, d) => o + d * (p - o).dot(&d),
        }
    }
    fn direction(a0: &Target, b0: &Target) -> Option<V> {
        match (a0, b0) {
            (Target::Ray(_, d), _) | (_, Target::Ray(_, d)) => Some(*d),
            _ => None,
        }
    }
    /// Target length of a chain, across the viewing direction when a ray is
    /// involved (only that component is observed).
    pub fn span(a0: &Target, b0: &Target, pa: V, pb: V) -> f64 {
        let d = a0.nearest(pa) - b0.nearest(pb);
        match Self::direction(a0, b0) {
            Some(dir) => (d - dir * d.dot(&dir)).norm(),
            None => d.norm(),
        }
    }
    /// Model length measured the same way as [`Target::span`].
    pub fn span_model(a0: &Target, b0: &Target, pa: V, pb: V) -> f64 {
        let d = pa - pb;
        match Self::direction(a0, b0) {
            Some(dir) => (d - dir * d.dot(&dir)).norm(),
            None => d.norm(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A long outline turned 20° toward +x, seen from either end, reads as
    /// one heading toward +z; a round one is not elongated.
    #[test]
    fn horizontal_axis_finds_the_long_side_facing_forward() {
        let yaw = 20f64.to_radians();
        let (s, c) = yaw.sin_cos();
        let mut long = vec![];
        let mut round = vec![];
        for i in 0..40 {
            let t = i as f64 / 39.0 - 0.5;
            for k in [-1.0, 1.0] {
                // 0.45 long, 0.15 wide, rotated about +y.
                let (along, across) = (t * 0.45, k * 0.075);
                long.push((
                    V::new(along * s + across * c, 0.02, along * c - across * s),
                    1.0,
                ));
                let a = t * std::f64::consts::TAU;
                round.push((V::new(a.cos() * 0.1, 0.0, a.sin() * 0.1), 1.0));
            }
        }
        let (axis, elongation) = horizontal_axis(&long).unwrap();
        assert!(axis.z > 0.0 && axis.y == 0.0);
        assert!((yaw_between(V::z(), axis) - yaw).abs() < 1e-6);
        assert!(elongation > SHOE_ELONGATION);
        let (_, elongation) = horizontal_axis(&round).unwrap();
        assert!(elongation < 1.2);
        assert!(horizontal_axis(&[]).is_none());
    }
}
