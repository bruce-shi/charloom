 use crate::geom::{V, proper_rotation, rotation_between, v};
use crate::prior::Prior;
use crate::scan::Scan;
use anyhow::Result;
use charloom_core::{Skeleton, Transform, Vec3};
use nalgebra::{Matrix3, Matrix4, Rotation3, UnitQuaternion};
use serde_json::{Value, json};
use std::collections::BTreeMap;

 pub const HAND_MORPHS: &[&str] = &["hand-scale", "fingers-length"];

/// Hand frame: largest forearm twist and wrist swing (rad), uniform scale
/// range.
const HAND_TWIST: f64 = 1.7;
const HAND_ROTATION: f64 = 0.6;
const HAND_SCALE: (f64, f64) = (0.8, 1.3);

/// Per-bone bounds of one aiming step: largest rotation (rad) and range of
/// the length ratio applied to the segment it spans.
#[derive(Clone, Copy)]
pub struct Limit {
    pub angle: f64,
    pub ratio: (f64, f64),
}

/// Metacarpals sit inside the palm: small turns, little length change.
const METACARPAL: Limit = Limit {
    angle: 0.45,
    ratio: (0.75, 1.35),
};
 const THUMB_METACARPAL: Limit = Limit {
    angle: 1.3,
    ratio: (0.6, 1.35),
};
/// Phalanges: a full curl (about 90° per knuckle) and cartoon lengths.
const PHALANX: Limit = Limit {
    angle: 1.6,
    ratio: (0.65, 1.5),
};

/// One digit, base to tip. `anchors` are (landmark part, chain position):
/// position `i < bones.len()` is the head of `bones[i]`, `bones.len()` the tip
/// of the last bone, at `tip_local` in its frame.
pub struct Chain {
    pub name: String,
    pub bones: Vec<usize>,
    pub limits: Vec<Limit>,
    pub tip_local: V,
    pub anchors: Vec<(&'static str, usize, V)>,
}

fn origin(m: &Matrix4<f64>) -> V {
    V::new(m[(0, 3)], m[(1, 3)], m[(2, 3)])
}

fn rotation_of(m: &Matrix4<f64>) -> Matrix3<f64> {
    proper_rotation(m.fixed_view::<3, 3>(0, 0).into_owned())
}

/// Applies a world-space rotation about joint `j`'s origin to its local TRS.
fn rotate_joint(
    skeleton: &Skeleton,
    pose: &mut [Transform],
    worlds: &[Matrix4<f64>],
    j: usize,
    r: &Matrix3<f64>,
) {
    let parent = skeleton.joints[j]
        .parent
        .map_or(Matrix3::identity(), |p| rotation_of(&worlds[p]));
    let local = parent.transpose() * r * parent;
    let q = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(local));
    let new = q * charloom_core::unit_quat(pose[j].rotation);
    pose[j].rotation = [new.i, new.j, new.k, new.w];
}

/// `r` with its angle limited to `max` (same axis).
fn clamp_angle(r: Matrix3<f64>, max: f64) -> (Matrix3<f64>, f64, bool) {
    match Rotation3::from_matrix_unchecked(r).axis_angle() {
        Some((axis, angle)) => (
            Rotation3::from_axis_angle(&axis, angle.min(max)).into_inner(),
            angle.min(max),
            angle > max,
        ),
        None => (Matrix3::identity(), 0.0, false),
    }
}

fn point_at(chain: &Chain, worlds: &[Matrix4<f64>], at: usize) -> V {
    match chain.bones.get(at) {
        Some(&b) => origin(&worlds[b]),
        None => {
            let m = worlds[*chain.bones.last().unwrap()];
            (m * chain.tip_local.push(1.0)).xyz()
        }
    }
}

/// Poses one chain onto its anchors (see the module docs). Returns the
/// per-anchor report. A segment's length ratio moves the joints after the
/// pivot (bone offsets); the bones whose mesh spans the segment (the pivot
/// and those in between, before the distal one, which is scaled in `pose`)
/// get the same stretch along their axis in `axial`, for the skinning
/// ([`stretch_bones`]), so each knuckle crease stays on its joint.
pub fn map_chain(
    skeleton: &Skeleton,
    pose: &mut [Transform],
    chain: &Chain,
    axial: &mut BTreeMap<usize, f64>,
) -> Result<Value> {
    let n = chain.bones.len();
    let mut prev = 0usize;
    let mut steps = vec![];
    for &(part, at, target) in &chain.anchors {
        if at <= prev || at > n {
            continue;
        }
        let worlds = skeleton.world_matrices(pose)?;
        let pivot = origin(&worlds[chain.bones[prev]]);
        let model = point_at(chain, &worlds, at);
        let (dm, dt) = (model - pivot, target - pivot);
        let before = (target - model).norm();
        if dm.norm() < 1e-5 || dt.norm() < 1e-5 {
            prev = at;
            continue;
        }
        let limit = chain.limits[prev];
        let (r, angle, angle_clamped) = clamp_angle(rotation_between(dm, dt), limit.angle);
        rotate_joint(skeleton, pose, &worlds, chain.bones[prev], &r);
         
        let raw = dt.dot(&(r * dm)) / dm.norm_squared();
        let ratio = raw.clamp(limit.ratio.0, limit.ratio.1);
        // Offsets of the bones after the pivot up to the anchor; the tip
        // also stretches the last bone along its axis (local +Y).
        for &b in &chain.bones[prev + 1..at.min(n - 1) + 1] {
            for k in 0..3 {
                pose[b].translation[k] *= ratio;
            }
        }
        if at == n {
            pose[chain.bones[n - 1]].scale[1] *= ratio;
        }
        for &b in &chain.bones[prev..at.min(n - 1)] {
            *axial.entry(b).or_insert(1.0) *= ratio;
        }
        let worlds = skeleton.world_matrices(pose)?;
        let after = (target - point_at(chain, &worlds, at)).norm();
        steps.push(json!({
            "landmark": part,
            "rotation_rad": angle,
            "rotation_clamped": angle_clamped,
            "length_ratio": ratio,
            "length_clamped": ratio != raw,
            "residual_before_m": before,
            "residual_after_m": after,
        }));
        prev = at;
    }
    Ok(json!({"finger": chain.name, "steps": steps}))
}

/// Bounded hand-frame map: a twist about the forearm axis, a swing and a
/// uniform scale (about the wrist joint).
pub struct HandFrame {
    pub twist: f64,
    pub swing: Matrix3<f64>,
    pub swing_angle: f64,
    pub raw_twist: f64,
    pub raw_swing: f64,
    pub scale: f64,
}

/// Rotation (about `center`) and uniform scale taking points `p` onto `q`
/// (weighted least squares), split into a twist about `axis` and a swing,
/// each clamped. A scan hand often has its palm turned forward where
/// MakeHuman's faces the thigh: that is forearm twist, so it may be large.
pub fn hand_frame(center: V, axis: V, pairs: &[(V, V, f64)]) -> HandFrame {
    let m = pairs.iter().fold(Matrix3::zeros(), |m, (p, q, w)| {
        m + (q - center) * (p - center).transpose() * *w
    });
    let q =
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(proper_rotation(m)));
    // Shortest-arc sign (w ≥ 0): the twist is then within ±π.
    let q = if q.w < 0.0 {
        UnitQuaternion::new_unchecked(-q.into_inner())
    } else {
        q
    };
    let raw_twist = 2.0 * q.imag().dot(&axis).atan2(q.w);
    let unit = nalgebra::Unit::new_normalize(axis);
    let swing_q = q * UnitQuaternion::from_axis_angle(&unit, raw_twist).inverse();
    let raw_swing = swing_q.angle();
    let twist = raw_twist.clamp(-HAND_TWIST, HAND_TWIST);
    let (swing, swing_angle, _) =
        clamp_angle(swing_q.to_rotation_matrix().into_inner(), HAND_ROTATION);
    let r = swing * Rotation3::from_axis_angle(&unit, twist).into_inner();
    let (num, den) = pairs.iter().fold((0.0, 0.0), |(n, d), (p, q, w)| {
        let rp = r * (p - center);
        (n + (q - center).dot(&rp) * w, d + rp.norm_squared() * w)
    });
    let s = if den > 1e-12 { num / den } else { 1.0 };
    HandFrame {
        twist,
        swing,
        swing_angle,
        raw_twist,
        raw_swing,
        scale: s.clamp(HAND_SCALE.0, HAND_SCALE.1),
    }
}

/// End of the distal bone `j` in its own frame: the far end of the finger
/// on its axis (centre of the end cap, at the bone's full reach).
fn tip_local(prior: &Prior, j: usize) -> V {
    let body = prior.body();
    let inv = prior.rest_world[j]
        .try_inverse()
        .unwrap_or_else(Matrix4::identity);
    let local: Vec<V> = body
        .influences
        .iter()
        .zip(&body.prim.positions)
        .filter(|(row, _)| row.iter().any(|w| w.joint == j && w.weight >= 0.5))
        .map(|(_, p)| (inv * v(*p).push(1.0)).xyz())
        .collect();
    let reach = local.iter().map(|p| p.y).fold(0.0, f64::max);
    let cap: Vec<&V> = local.iter().filter(|p| p.y >= reach * 0.8).collect();
    if cap.is_empty() {
        return V::new(0.0, 0.02, 0.0);
    }
    let c = cap.iter().fold(V::zeros(), |s, p| s + **p) / cap.len() as f64;
    V::new(c.x, reach, c.z)
}

/// The finger's end beyond its `tip` landmark: the landmark is the centre of
/// the finger's cross-section at the nail, about one finger radius (its
/// distance to the scan surface) short of the end; a fat cartoon finger
/// ends much further out than MakeHuman's thin one.
fn finger_end(scan: &Scan, tip: V, from: V) -> (V, f64) {
    let radius = scan
        .query
        .closest_point(crate::geom::a(tip), 0.03, None)
        .map_or(0.0, |h| h.distance_m);
    let dir = (tip - from).try_normalize(1e-9).unwrap_or_else(V::zeros);
    (tip + dir * radius, radius)
}

/// Bones of digit `f` (1 = thumb) on side `suffix` ("L"/"R"), base to tip:
/// the thumb's three bones, or a metacarpal and three phalanges.
fn digit_bones(prior: &Prior, f: usize, suffix: &str) -> Option<Vec<usize>> {
    let names: Vec<String> = if f == 1 {
        (1..=3).map(|k| format!("finger1-{k}.{suffix}")).collect()
    } else {
        std::iter::once(format!("metacarpal{}.{suffix}", f - 1))
            .chain((1..=3).map(|k| format!("finger{f}-{k}.{suffix}")))
            .collect()
    };
    names
        .iter()
        .map(|n| prior.by_name.get(n).copied())
        .collect()
}

 fn chains(
    prior: &Prior,
    scan: &Scan,
    side: &str,
    hand_points: &BTreeMap<String, Vec3>,
    digits: usize,
) -> Vec<(Chain, [Option<f64>; 2])> {
    let suffix = if side == "l" { "L" } else { "R" };
    let mut out = vec![];
    // The scan's digit `k` (landmarks `finger{k}`) on MakeHuman's digit `f`.
    for (k, &f) in (1..).zip(crate::creature::kept(digits as u8)) {
        let Some(bones) = digit_bones(prior, f, suffix) else {
            continue;
        };
        let n = bones.len();
        let mut anchors: Vec<(&'static str, usize, V)> = [("base", 1), ("mid", 2), ("tip", n)]
            .into_iter()
            .filter_map(|(part, at)| {
                Some((
                    part,
                    at,
                    v(*hand_points.get(&format!("finger{k}_{side}_{part}"))?),
                ))
            })
            .collect();
         
        let mid_radius = anchors.iter().find(|a| a.0 == "mid").and_then(|a| {
            scan.query
                .closest_point(crate::geom::a(a.2), 0.03, None)
                .map(|h| h.distance_m)
        });
        let mut radius = None;
        if let [.., before, (_, at, tip)] = anchors.as_mut_slice()
            && *at == n
        {
            let (end, r) = finger_end(scan, *tip, before.2);
            *tip = end;
            radius = Some(r);
        }
        if anchors.is_empty() {
            continue;
        }
        let limits = (0..n)
            .map(|i| match (i, f) {
                (0, 1) => THUMB_METACARPAL,
                (0, _) => METACARPAL,
                _ => PHALANX,
            })
            .collect();
        out.push((
            Chain {
                name: format!("finger{f}_{side}"),
                tip_local: tip_local(prior, bones[n - 1]),
                bones,
                limits,
                anchors,
            },
            [radius, mid_radius],
        ));
    }
    out
}

 pub fn stretch_bones(worlds: &mut [Matrix4<f64>], axial: &BTreeMap<usize, f64>) {
    for (&j, &s) in axial {
        let mut m = Matrix4::identity();
        m[(1, 1)] = s;
        worlds[j] *= m;
    }
}

/// Least vertices (mainly carried by a hand bone) for its refit from the pose.
const REFIT_MIN_VERTICES: usize = 6;

 pub fn refit_hand_bones(
    prior: &Prior,
    worlds: &mut [Matrix4<f64>],
    local: &mut [Transform],
    posed_worlds: &[Matrix4<f64>],
    posed: &[Vec3],
    fitted: &[Vec3],
) -> Value {
    let skeleton = prior.skeleton();
    let body = prior.body();
    let mut report = serde_json::Map::new();
    for suffix in ["L", "R"] {
        let Some(&wrist) = prior.by_name.get(&format!("wrist.{suffix}")) else {
            continue;
        };
        let joints: Vec<usize> = prior
            .skin_joints
            .iter()
            .copied()
            .filter(|&j| prior.is_ancestor(wrist, j))
            .collect();
        let mut moved = vec![];
        for &j in &joints {
            let pairs: Vec<(V, V, f64)> = body
                .influences
                .iter()
                .enumerate()
                .filter_map(|(i, row)| {
                    let w: f64 = row.iter().filter(|x| x.joint == j).map(|x| x.weight).sum();
                    (w >= 0.5).then(|| (v(posed[i]), v(fitted[i]), w))
                })
                .collect();
            if pairs.len() < REFIT_MIN_VERTICES {
                continue;
            }
            let total: f64 = pairs.iter().map(|p| p.2).sum();
            let cp = pairs.iter().fold(V::zeros(), |s, p| s + p.0 * p.2) / total;
            let cf = pairs.iter().fold(V::zeros(), |s, p| s + p.1 * p.2) / total;
            let m = pairs.iter().fold(Matrix3::zeros(), |m, (p, q, w)| {
                m + (q - cf) * (p - cp).transpose() * *w
            });
            let r = proper_rotation(m);
            let posed_r = rotation_of(&posed_worlds[j]);
            let o = cf + r * (origin(&posed_worlds[j]) - cp);
            moved.push((origin(&worlds[j]) - o).norm());
            let mut world = Matrix4::identity();
            world.fixed_view_mut::<3, 3>(0, 0).copy_from(&(r * posed_r));
            world.fixed_view_mut::<3, 1>(0, 3).copy_from(&o);
            worlds[j] = world;
        }
        // Locals of the hand bones and of their children (helpers keep
        // their world matrices).
        let affected: Vec<usize> = (0..skeleton.joints.len())
            .filter(|&k| {
                joints.contains(&k)
                    || skeleton.joints[k]
                        .parent
                        .is_some_and(|p| joints.contains(&p))
            })
            .collect();
        for j in affected {
            let parent = skeleton.joints[j]
                .parent
                .map_or(Matrix4::identity(), |p| worlds[p]);
            let m = parent.try_inverse().unwrap_or_else(Matrix4::identity) * worlds[j];
            let q = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(
                proper_rotation(m.fixed_view::<3, 3>(0, 0).into_owned()),
            ));
            local[j] = Transform {
                translation: crate::geom::a(origin(&m)),
                rotation: [q.i, q.j, q.k, q.w],
                scale: [1.0; 3],
            };
        }
        moved.sort_by(f64::total_cmp);
        report.insert(
            format!("hand_{}", suffix.to_lowercase()),
            json!({
                "bones": moved.len(),
                "moved_from_generic_refit_m": {
                    "p50": moved.get(moved.len() / 2),
                    "max": moved.last(),
                },
            }),
        );
    }
    Value::Object(report)
}

/// Poses both hands onto the finger landmarks (no-op without them).
/// `digits` is the digit count of the `[left, right]` hand (thumb counted):
/// its digits are carried by `creature::kept` fingers, the others left alone. `pose` is updated in place,
/// and `axial` takes the bones' stretches for the skinning ([`stretch_bones`]).
pub fn fit(
    prior: &Prior,
    scan: &Scan,
    pose: &mut [Transform],
    hand_points: &BTreeMap<String, Vec3>,
    digits: [u8; 2],
    axial: &mut BTreeMap<usize, f64>,
) -> Result<Value> {
    let skeleton = prior.skeleton();
    let mut hands = serde_json::Map::new();
    for ((side, suffix), n) in [("l", "L"), ("r", "R")].into_iter().zip(digits) {
        let n = n as usize;
        let point = |name: String| hand_points.get(&name).map(|p| v(*p));
        let Some(&wrist) = prior.by_name.get(&format!("wrist.{suffix}")) else {
            continue;
        };
        let digit = |f: usize| digit_bones(prior, f, suffix);
         
        let kept = crate::creature::kept(n as u8);
        let marked = (1..=kept.len())
            .flat_map(|f| ["base", "mid", "tip"].map(|p| format!("finger{f}_{side}_{p}")))
            .filter(|n| hand_points.contains_key(n))
            .count();
        if marked == 0 {
            continue;
        }
        // Model points matching the base landmarks (the head of the bone
        // after the (thumb) metacarpal) and the palm (halfway from the wrist
        // to the mean knuckle of the kept fingers: the palm under the digits the
        // hand has).
        let pairs_at = |worlds: &[Matrix4<f64>]| -> Vec<(V, V, f64)> {
            let knuckle = |f: usize| digit(f).map(|b| origin(&worlds[b[1]]));
            let mut pairs: Vec<(V, V, f64)> = (1..)
                .zip(kept)
                .filter_map(|(k, &f)| {
                    Some((knuckle(f)?, point(format!("finger{k}_{side}_base"))?, 1.0))
                })
                .collect();
            if let (Some(q), Some(ks)) = (
                point(format!("palm_{side}")),
                kept.iter()
                    .filter(|&&f| f > 1)
                    .map(|&f| knuckle(f))
                    .collect::<Option<Vec<V>>>(),
            ) {
                let mean = ks.iter().fold(V::zeros(), |s, k| s + k) / ks.len() as f64;
                pairs.push(((origin(&worlds[wrist]) + mean) * 0.5, q, 1.0));
            }
            pairs
        };
        let mean_residual = |pairs: &[(V, V, f64)]| {
            pairs.iter().map(|(p, q, _)| (q - p).norm()).sum::<f64>() / pairs.len().max(1) as f64
        };
        let worlds = prior.worlds(pose)?;
        let pairs = pairs_at(&worlds);
        let forearm = skeleton.joints[wrist].parent;
        let mut frame = json!(null);
        if pairs.len() >= 3
            && let Some(forearm) = forearm
        {
            let center = origin(&worlds[wrist]);
            let axis = (center - origin(&worlds[forearm]))
                .try_normalize(1e-9)
                .unwrap_or_else(V::y);
            let f = hand_frame(center, axis, &pairs);
            // Half the twist turns the forearm's twist bone (about the axis
            // through the wrist), so the wrist does not wring.
            let half =
                Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), f.twist * 0.5)
                    .into_inner();
            rotate_joint(skeleton, pose, &worlds, forearm, &half);
            let worlds = prior.worlds(pose)?;
            rotate_joint(skeleton, pose, &worlds, wrist, &(f.swing * half));
            for k in 0..3 {
                pose[wrist].scale[k] *= f.scale;
            }
            let worlds = prior.worlds(pose)?;
            frame = json!({
                "twist_rad": f.twist,
                "swing_rad": f.swing_angle,
                "unclamped_twist_rad": f.raw_twist,
                "unclamped_swing_rad": f.raw_swing,
                "scale": f.scale,
                "points": pairs.len(),
                "mean_residual_before_m": mean_residual(&pairs),
                "mean_residual_after_m": mean_residual(&pairs_at(&worlds)),
            });
        }
         
        if frame.is_null() {
            hands.insert(format!("hand_{side}_unframed"), json!({"landmarks": marked, "skipped": "no hand frame"}));
            continue;
        }
        let mut fingers = vec![];
        for (chain, radius) in chains(prior, scan, side, hand_points, n) {
            let mut report = map_chain(skeleton, pose, &chain, axial)?;
            report["tip_radius_m"] = json!(radius[0]);
            report["mid_radius_m"] = json!(radius[1]);
            fingers.push(report);
        }
        hands.insert(
            format!("hand_{side}"),
            json!({"landmarks": marked, "digits": n, "frame": frame, "fingers": fingers}),
        );
    }
    let applied = hands.keys().any(|k| !k.ends_with("_unframed"));
    Ok(json!({"applied": applied, "hands": hands}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use charloom_core::Joint;

    fn joint(name: &str, parent: Option<usize>, t: [f64; 3]) -> Joint {
        Joint {
            id: name.into(),
            name: name.into(),
            parent,
            rest: Transform::translated(t),
            matrix: None,
            deform: true,
        }
    }

    /// A straight four-bone finger (metacarpal + three phalanges along +Y)
    /// is curled and lengthened onto base / mid / tip landmarks of a bent,
    /// longer target finger; an unreachable target is clamped, not followed.
    #[test]
    fn chain_curls_and_scales_onto_landmarks_within_bounds() {
        let skeleton = Skeleton {
            joints: vec![
                joint("wrist", None, [0.0; 3]),
                joint("metacarpal", Some(0), [0.0, 0.03, 0.0]),
                joint("p1", Some(1), [0.0, 0.07, 0.0]),
                joint("p2", Some(2), [0.0, 0.03, 0.0]),
                joint("p3", Some(3), [0.0, 0.025, 0.0]),
            ],
        };
        let tip_local = V::new(0.0, 0.02, 0.0);
        // Target: knuckle 1 cm sideways, proximal phalanx 20 % longer and
        // straight on, then curled 90° towards +Z with the distal part 30 %
        // longer.
        let base = V::new(0.01, 0.1, 0.0);
        let mid = base + V::new(0.0, 0.036, 0.0);
        let tip = mid + V::new(0.0, 0.0, (0.025 + 0.02) * 1.3);
        let mut pose = skeleton.rest_pose();
        let chain = Chain {
            name: "test".into(),
            bones: vec![1, 2, 3, 4],
            limits: vec![METACARPAL, PHALANX, PHALANX, PHALANX],
            tip_local,
            anchors: vec![("base", 1, base), ("mid", 2, mid), ("tip", 4, tip)],
        };
        let mut axial = BTreeMap::new();
        let report = map_chain(&skeleton, &mut pose, &chain, &mut axial).unwrap();
        let worlds = skeleton.world_matrices(&pose).unwrap();
        assert!((origin(&worlds[2]) - base).norm() < 1e-9);
        assert!((origin(&worlds[3]) - mid).norm() < 1e-9);
        assert!((point_at(&chain, &worlds, 4) - tip).norm() < 1e-9);
        let steps = report["steps"].as_array().unwrap();
        assert_eq!(steps.len(), 3);
        assert!((steps[1]["length_ratio"].as_f64().unwrap() - 1.2).abs() < 1e-9);
        assert!((steps[2]["length_ratio"].as_f64().unwrap() - 1.3).abs() < 1e-9);
        // Joint rotations stay proper and the distal bone is only stretched
        // along its axis.
        assert_eq!(pose[4].scale[0], 1.0);
        assert!((pose[4].scale[1] - 1.3).abs() < 1e-9);
        // The bones spanning each segment stretch with it for the skinning:
        // the proximal phalanx by the base-to-mid ratio, the middle one by
        // the mid-to-tip ratio (the distal one is scaled in the pose).
        let ratio = |k: usize| steps[k]["length_ratio"].as_f64().unwrap();
        assert_eq!(axial.len(), 3);
        for (bone, step) in [(1, 0), (2, 1), (3, 2)] {
            assert!((axial[&bone] - ratio(step)).abs() < 1e-12, "bone {bone}");
        }
        // A stretched bone scales its mesh along its own axis only.
        let mut stretched = worlds.clone();
        stretch_bones(&mut stretched, &axial);
        let along = stretched[2] * nalgebra::Vector4::new(0.0, 0.01, 0.0, 1.0);
        let head = origin(&worlds[2]);
        let reach = (worlds[2] * nalgebra::Vector4::new(0.0, 0.01, 0.0, 0.0)).xyz() * 1.2;
        assert!((along.xyz() - (head + reach)).norm() < 1e-12);
        assert_eq!(origin(&stretched[2]), head);

        // Behind the hand: the turn is bounded and the chain shortened, not
        // lengthened away from the target.
        let mut pose = skeleton.rest_pose();
        let far = Chain {
            anchors: vec![("tip", 4, V::new(0.0, 0.1 - 0.3, 0.0))],
            ..chain
        };
        let report = map_chain(&skeleton, &mut pose, &far, &mut BTreeMap::new()).unwrap();
        let step = &report["steps"][0];
        assert!(step["rotation_clamped"].as_bool().unwrap());
        assert!((step["rotation_rad"].as_f64().unwrap() - METACARPAL.angle).abs() < 1e-9);
        assert_eq!(step["length_ratio"].as_f64().unwrap(), METACARPAL.ratio.0);
        assert!(
            step["residual_after_m"].as_f64().unwrap()
                < step["residual_before_m"].as_f64().unwrap()
        );
    }

    #[test]
    fn hand_frame_splits_twist_and_swing_and_bounds_them() {
        let center = V::new(0.1, 0.2, 0.0);
        let axis = -V::y();
        let twist = Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), 1.2);
        let swing = Rotation3::from_axis_angle(&V::z_axis(), 0.3);
        let r = (swing * twist).into_inner();
        let ps = [
            V::new(0.0, -0.1, 0.0),
            V::new(0.02, -0.1, 0.01),
            V::new(-0.02, -0.09, 0.0),
            V::new(0.0, -0.05, 0.02),
        ];
        let pairs: Vec<(V, V, f64)> = ps
            .iter()
            .map(|p| (center + p, center + r * p * 1.1, 1.0))
            .collect();
        let f = hand_frame(center, axis, &pairs);
        assert!((f.twist - 1.2).abs() < 1e-9 && (f.swing_angle - 0.3).abs() < 1e-9);
        assert!((f.swing - swing.into_inner()).norm() < 1e-9 && (f.scale - 1.1).abs() < 1e-9);
        // A palm turned the other way round and a hand 3x too big: bounded.
        let r = Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), -2.5).into_inner();
        let pairs: Vec<(V, V, f64)> = ps
            .iter()
            .map(|p| (center + p, center + r * p * 3.0, 1.0))
            .collect();
        let f = hand_frame(center, axis, &pairs);
        assert!((f.twist + HAND_TWIST).abs() < 1e-9 && (f.raw_twist + 2.5).abs() < 1e-9);
        assert_eq!(f.scale, HAND_SCALE.1);
    }
}
