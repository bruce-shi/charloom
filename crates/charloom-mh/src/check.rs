 use crate::geom::{V, a, percentiles, v};
use crate::scan::build_query;
use anyhow::{Context, Result};
use charloom_core::{Influence, MotionClip, Skeleton, Vec3};
use charloom_engine::motion::{RetargetConfig, retarget};
use nalgebra::Matrix4;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::Path;

pub struct SkinnedPart<'a> {
    pub name: &'a str,
    pub positions: &'a [Vec3],
    pub triangles: &'a [[u32; 3]],
    pub influences: &'a [Vec<Influence>],
    /// Optional seam key per vertex (garments: scan welded id).
    pub seam: Option<&'a [usize]>,
}

 pub const RAISED_FOOT_PITCH_RAD: f64 = 0.52;

/// Downward pitch (radians below the horizontal) of each foot bone, from the
/// ankle (`foot.{side}`) to the mean of its toe roots (`toe{n}-1.{side}`).
fn foot_pitch(skeleton: &Skeleton, side: &str) -> Option<f64> {
    let index = |n: &str| skeleton.joints.iter().position(|j| j.name == n);
    let world = skeleton.world_matrices(&skeleton.rest_pose()).ok()?;
    let at = |i: usize| V::new(world[i][(0, 3)], world[i][(1, 3)], world[i][(2, 3)]);
    let ankle = at(index(&format!("foot.{side}"))?);
    let toes: Vec<V> = (1..=5)
        .filter_map(|n| index(&format!("toe{n}-1.{side}")).map(at))
        .collect();
    if toes.is_empty() {
        return None;
    }
    let d = (toes.iter().sum::<V>() / toes.len() as f64 - ankle).try_normalize(1e-9)?;
    Some((-d.y).clamp(-1.0, 1.0).asin())
}

 fn raised_feet(
    target: &Skeleton,
    reference: &Skeleton,
    mapped: &std::collections::HashSet<usize>,
) -> (Vec<String>, Vec<String>, Value) {
    let mut keep_reference = vec![];
    let mut keep_local = vec![];
    let mut report = serde_json::Map::new();
    for side in ["L", "R"] {
        let (Some(fitted), Some(rest)) = (foot_pitch(target, side), foot_pitch(reference, side))
        else {
            continue;
        };
        let raised = fitted - rest > RAISED_FOOT_PITCH_RAD;
        let foot = format!("foot.{side}");
        let mut local = vec![];
        if raised && let Some(f) = target.joints.iter().position(|j| j.name == foot) {
            if mapped.contains(&f) && target.joints[f].matrix.is_none() {
                keep_reference.push(foot.clone());
            }
            for (i, j) in target.joints.iter().enumerate() {
                let mut parent = j.parent;
                let mut below = false;
                while let Some(p) = parent {
                    if p == f {
                        below = true;
                        break;
                    }
                    parent = target.joints[p].parent;
                }
                if below && mapped.contains(&i) && j.matrix.is_none() {
                    local.push(j.name.clone());
                }
            }
        }
        keep_local.extend(local.iter().cloned());
        report.insert(
            side.into(),
            json!({
                "pitch_rad": fitted,
                "makehuman_pitch_rad": rest,
                "raised": raised,
                "toes_kept_at_rest": local,
            }),
        );
    }
    (keep_reference, keep_local, Value::Object(report))
}

/// Loads a clip and retargets it onto the fitted rig `target`. `reference`
/// is MakeHuman's own skeleton (same joint names), for the raised-foot test.
pub fn load_and_retarget(
    path: &Path,
    clip: Option<&str>,
    name: &str,
    target: &Skeleton,
    reference: &Skeleton,
    rest_local_hands: bool,
    rigid_wrists: &[String],
) -> Result<(MotionClip, Value)> {
    let source = charloom_io::load_motion_asset(path)?;
    let chosen = match clip {
        Some(c) => source
            .clips
            .iter()
            .find(|x| x.name == c)
            .with_context(|| format!("clip {c} not found"))?,
        // FBX files often carry an empty bind-pose take first; use the longest.
        None => source
            .clips
            .iter()
            .max_by(|x, y| {
                let (dx, dy) = (x.time_range(), y.time_range());
                (dx[1] - dx[0]).total_cmp(&(dy[1] - dy[0]))
            })
            .context("motion file has no clips")?,
    };
    // The trunk, neck and head keep their own rest posture and take the
    // clip's motion relative to it: aligned to the clip's upright spine,
    // a fitted trunk whose joints lean forward under an upright body (the
    // flame's neck joint 24° forward of its hips over a big belly) tipped
    // the whole upper body back by that much in the walk, chest out and
    // face up.
    let mapping = charloom_engine::motion::match_joints(&source.skeleton, target, &Default::default())?;
    let preserve_reference_joints: Vec<String> = mapping
        .mappings
        .iter()
        .map(|m| target.joints[m.target].name.clone())
        .filter(|n| {
            let n = n.to_lowercase();
            n.starts_with("spine") || n.starts_with("neck") || n == "head"
        })
        .collect();
    let mut config = RetargetConfig {
        name: name.to_string(),
        align_reference_chains: true,
        preserve_reference_joints,
        // A fitted hand is turned with the scan's (palms forward, say): its
        // palm alignment to the clip's reference must not wring the wrist
        // but turn the forearm (lowerarm01, lowerarm02, wrist) as pronation.
        distribute_palm_twist: true,
        root_motion: true,
        min_target_coverage: 0.15,
        ..RetargetConfig::default()
    };
    let mapped =
        charloom_engine::motion::match_joints(&source.skeleton, target, &config.joint_map)?
            .mappings
            .iter()
            .map(|m| m.target)
            .collect();
    let (keep_reference, keep_local, feet) = raised_feet(target, reference, &mapped);
    config.preserve_reference_joints.extend(keep_reference);
     
    let has_mapped_ancestor = |i: usize| {
        let mut parent = target.joints[i].parent;
        while let Some(p) = parent {
            if mapped.contains(&p) {
                return true;
            }
            parent = target.joints[p].parent;
        }
        false
    };
    let mut planted: Vec<String> = mapped
        .iter()
        .filter(|&&i| !has_mapped_ancestor(i))
        .map(|&i| target.joints[i].name.clone())
        .collect();
    planted.sort();
    planted.extend(["foot.L".to_string(), "foot.R".to_string()]);
    for n in planted {
        let Some(i) = target.joints.iter().position(|j| j.name == n) else {
            continue;
        };
        if mapped.contains(&i)
            && target.joints[i].matrix.is_none()
            && !config.preserve_reference_joints.contains(&n)
        {
            config.preserve_reference_joints.push(n);
        }
    }
     
    let legs = leg_scale(&source.skeleton, target, &config.joint_map);
    if let Some(scale) = legs {
        config.root_scale = Some(scale);
    }
     
    for &i in &mapped {
        let n = &target.joints[i].name;
        let wanted = n.starts_with("toe") || (rest_local_hands && n.starts_with("finger"))
            || rigid_wrists.contains(n);
        if wanted
            && !keep_local.contains(n)
            && !config.preserve_reference_joints.contains(n)
            && target.joints[i].matrix.is_none()
        {
            config.rest_local_reference_joints.push(n.clone());
        }
    }
    config.preserve_local_pose_joints = keep_local;
    let (mut motion, report) = retarget(&source.skeleton, chosen, target, &config)?;
    let toes = spread_toe_roll(target, &mut motion);
    let ground = plant_feet(
        &source.skeleton,
        chosen,
        target,
        &mut motion,
        &report.motion_roots,
        legs.unwrap_or(report.root_scale),
        &config.joint_map,
    )?;
    let feet_corrections: Vec<&String> = report
        .reference_chain_corrections
        .iter()
        .filter(|c| c.starts_with("foot.") || c.starts_with("toe"))
        .collect();
    Ok((
        motion,
        json!({"source":path,"clip":chosen.name,"retarget":report.mapping,"raised_feet":feet,"feet_reference":feet_corrections,"toe_roll":toes,"rest_local_reference":config.rest_local_reference_joints,"root_scale":report.root_scale,"leg_scale":legs,"ground":ground,"preserved_reference":config.preserve_reference_joints}),
    ))
}

/// Rest world positions of every joint.
fn rest_points(skeleton: &Skeleton) -> Option<Vec<V>> {
    let world = skeleton.world_matrices(&skeleton.rest_pose()).ok()?;
    Some(world.iter().map(|m| V::new(m[(0, 3)], m[(1, 3)], m[(2, 3)])).collect())
}

/// Source joint driving each of the target's leg joints (hip, knee, ankle)
/// per side, from the clip's joint mapping.
fn leg_chains(
    source: &Skeleton,
    target: &Skeleton,
    joint_map: &std::collections::BTreeMap<String, String>,
) -> Vec<[(usize, usize); 3]> {
    let Ok(mapping) = charloom_engine::motion::match_joints(source, target, joint_map) else {
        return vec![];
    };
    let by_target: HashMap<usize, usize> =
        mapping.mappings.iter().map(|m| (m.target, m.source)).collect();
    let pair = |n: String| {
        let t = target.joints.iter().position(|j| j.name == n)?;
        Some((by_target.get(&t).copied()?, t))
    };
    ["L", "R"]
        .iter()
        .filter_map(|s| {
            Some([
                pair(format!("upperleg01.{s}"))?,
                pair(format!("lowerleg01.{s}"))?,
                pair(format!("foot.{s}"))?,
            ])
        })
        .collect()
}

 fn leg_scale(
    source: &Skeleton,
    target: &Skeleton,
    joint_map: &std::collections::BTreeMap<String, String>,
) -> Option<f64> {
    let (sp, tp) = (rest_points(source)?, rest_points(target)?);
    let ratios: Vec<f64> = leg_chains(source, target, joint_map)
        .iter()
        .filter_map(|c| {
            let s = (sp[c[1].0] - sp[c[0].0]).norm() + (sp[c[2].0] - sp[c[1].0]).norm();
            let t = (tp[c[1].1] - tp[c[0].1]).norm() + (tp[c[2].1] - tp[c[1].1]).norm();
            (s > 1e-6 && t > 1e-6).then_some(t / s)
        })
        .collect();
    (!ratios.is_empty()).then(|| ratios.iter().sum::<f64>() / ratios.len() as f64)
}

 fn plant_feet(
    source: &Skeleton,
    clip: &MotionClip,
    target: &Skeleton,
    motion: &mut MotionClip,
    roots: &[String],
    scale: f64,
    joint_map: &std::collections::BTreeMap<String, String>,
) -> Result<Value> {
    use charloom_core::{Interpolation, Property};
    let ankles: Vec<(usize, usize)> =
        leg_chains(source, target, joint_map).iter().map(|c| c[2]).collect();
    let (Some(sp), Some(tp)) = (rest_points(source), rest_points(target)) else {
        return Ok(json!({"applied": false}));
    };
    let root = roots.iter().find_map(|n| target.joints.iter().position(|j| &j.name == n));
    let Some(track) = root.and_then(|r| {
        motion.tracks.iter().position(|t| {
            t.joint == r && t.property == Property::Translation && t.interpolation != Interpolation::CubicSpline
        })
    }) else {
        return Ok(json!({"applied": false}));
    };
    if ankles.is_empty() {
        return Ok(json!({"applied": false}));
    }
    let parent = target.joints[motion.tracks[track].joint].parent;
    let lowest = |points: &[V], rest: &[V], pick: fn(&(usize, usize)) -> usize| {
        ankles
            .iter()
            .map(|a| points[pick(a)].y - rest[pick(a)].y)
            .fold(f64::INFINITY, f64::min)
    };
    let times = motion.tracks[track].times.clone();
    let mut lifts = Vec::with_capacity(times.len());
    for &t in &times {
        let src = source.world_matrices(&clip.sample(source, t)?)?;
        let tgt = target.world_matrices(&motion.sample(target, t)?)?;
        let at = |m: &Vec<Matrix4<f64>>| m.iter().map(|m| V::new(m[(0, 3)], m[(1, 3)], m[(2, 3)])).collect::<Vec<V>>();
        let wanted = lowest(&at(&src), &sp, |a| a.0) * scale;
        let now = lowest(&at(&tgt), &tp, |a| a.1);
        let lift = wanted - now;
        // The lift is along the world's up; the root's key is in its parent's frame.
        let local = match parent {
            Some(p) => tgt[p]
                .fixed_view::<3, 3>(0, 0)
                .try_inverse()
                .context("singular root parent frame")?
                * V::new(0.0, lift, 0.0),
            None => V::new(0.0, lift, 0.0),
        };
        lifts.push((lift, local));
    }
    for (value, (_, local)) in motion.tracks[track].values.iter_mut().zip(&lifts) {
        for k in 0..3 {
            value[k] += local[k];
        }
    }
    let lifts: Vec<f64> = lifts.iter().map(|l| l.0).collect();
    let max = lifts.iter().fold(0.0f64, |m, l| m.max(l.abs()));
    let mean = lifts.iter().sum::<f64>() / lifts.len().max(1) as f64;
    Ok(json!({"applied": true, "keys": lifts.len(), "max_abs_lift_m": max, "mean_lift_m": mean}))
}

/// A clip has one toe bone per foot, mapped to MakeHuman's big toe
/// (`toe1-1`) alone: bent up at push-off while the other four stayed with
/// the foot, it stood out of the foot as a fin. The big toe's turn against
/// its rest, taken in the foot's frame, bends every toe root (`toe{2..5}-1`
/// sharing its parent) the same way. Returns per side the toes driven and
/// the largest turn (radians).
fn spread_toe_roll(target: &Skeleton, clip: &mut MotionClip) -> Value {
    use charloom_core::Property;
    use nalgebra::{Quaternion, UnitQuaternion};
    let q = |r: [f64; 4]| UnitQuaternion::from_quaternion(Quaternion::new(r[3], r[0], r[1], r[2]));
    let index = |n: &str| target.joints.iter().position(|j| j.name == n);
    let mut report = serde_json::Map::new();
    for side in ["L", "R"] {
        let Some(big) = index(&format!("toe1-1.{side}")) else {
            continue;
        };
        let Some(track) = clip
            .tracks
            .iter()
            .find(|t| t.joint == big && t.property == Property::Rotation)
            .cloned()
        else {
            continue;
        };
        let rest = q(target.joints[big].rest.rotation).inverse();
        let mut driven = vec![];
        let mut max_turn: f64 = 0.0;
        for n in 2..=5 {
            let Some(toe) = index(&format!("toe{n}-1.{side}")) else {
                continue;
            };
            if target.joints[toe].parent != target.joints[big].parent || target.joints[toe].matrix.is_some() {
                continue;
            }
            let own = q(target.joints[toe].rest.rotation);
            let values: Vec<[f64; 4]> = track
                .values
                .iter()
                .map(|r| {
                    let turn = q(*r) * rest;
                    max_turn = max_turn.max(turn.angle());
                    let out = (turn * own).into_inner();
                    [out.i, out.j, out.k, out.w]
                })
                .collect();
            clip.tracks.retain(|t| !(t.joint == toe && t.property == Property::Rotation));
            clip.tracks.push(charloom_core::Track { joint: toe, values, ..track.clone() });
            driven.push(target.joints[toe].name.clone());
        }
        report.insert(side.into(), json!({"toes": driven, "max_turn_rad": max_turn}));
    }
    Value::Object(report)
}

fn skin(parts: &SkinnedPart, palette: &HashMap<usize, Matrix4<f64>>) -> Vec<V> {
    parts
        .positions
        .par_iter()
        .zip(parts.influences)
        .map(|(p, row)| {
            let h = v(*p).push(1.0);
            row.iter()
                .fold(nalgebra::Vector4::zeros(), |s, w| {
                    s + palette[&w.joint] * h * w.weight
                })
                .xyz()
        })
        .collect()
}

/// Stretch of edges at least `min_len` long; generated meshes carry
/// micro-edges whose ratios are numerical noise, not visible tearing.
fn edge_ratios(
    rest: &[Vec3],
    posed: &[V],
    triangles: &[[u32; 3]],
    stride: usize,
    min_len: f64,
) -> Vec<f64> {
    triangles
        .par_iter()
        .step_by(stride)
        .flat_map_iter(|t| {
            [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
                .into_iter()
                .filter_map(|(i, j)| {
                    let r = (v(rest[i as usize]) - v(rest[j as usize])).norm();
                    (r > min_len).then(|| (posed[i as usize] - posed[j as usize]).norm() / r)
                })
        })
        .collect()
}

/// The exported rig: skeleton, skin palette (skeleton indices) and the rest
/// world matrices that define the inverse bind matrices.
pub struct RigView<'a> {
    pub skeleton: &'a Skeleton,
    pub palette: &'a [usize],
    pub rest_world: &'a [Matrix4<f64>],
}

/// Samples `clip` and reports per-frame defects. Joint indices of the parts
/// and the rig must match the clip.
pub fn check_clip(
    rig: &RigView,
    clip: &MotionClip,
    body: &SkinnedPart,
    garments: &[SkinnedPart],
    frames: usize,
    height: f64,
    // Per body triangle: hidden in the export. A graft (scan skin kept on
    // the free head, `graft.rs`) inside a hidden triangle does not count:
    // the scan's ear counted inside the MakeHuman ear it replaces. (Checked
    // against the visible triangles alone, the nearest surface across the
    // hole gave the wrong side.)
    hidden: Option<&[bool]>,
) -> Result<Value> {
    let RigView {
        skeleton,
        palette,
        rest_world,
    } = *rig;
    let ibm: Vec<Matrix4<f64>> = palette
        .iter()
        .map(|&j| {
            rest_world[j]
                .try_inverse()
                .unwrap_or_else(Matrix4::identity)
        })
        .collect();
    let [start, end] = clip.time_range();
    let duration = end - start;
    let mut rows = vec![];
    let mut worst = json!(null);
    let mut worst_rate = -1.0;
    let mut baseline = None;
    for f in 0..=frames {
        let time = if f == 0 {
            None
        } else {
            Some(start + duration * (f - 1) as f64 / (frames.max(2) - 1) as f64)
        };
        let pose = match time {
            None => skeleton.rest_pose(),
            Some(t) => clip.sample(skeleton, t)?,
        };
        let world = skeleton.world_matrices(&pose)?;
        let matrices: HashMap<usize, Matrix4<f64>> = palette
            .iter()
            .zip(&ibm)
            .map(|(&j, b)| (j, world[j] * b))
            .collect();
        let body_posed = skin(body, &matrices);
        let body_positions: Vec<Vec3> = body_posed.iter().map(|p| a(*p)).collect();
        let query = build_query(&body_positions, body.triangles);
        let lowest = body_posed.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let body_stretch = edge_ratios(
            body.positions,
            &body_posed,
            body.triangles,
            3,
            height * 0.001,
        );
        let mut garment_rows = vec![];
        let mut inside_total = 0usize;
        let mut sampled_total = 0usize;
        for g in garments {
            let skip_hidden = hidden.filter(|_| g.name.starts_with("graft"));
            let posed = skin(g, &matrices);
            // Area-weighted vertex normals of the posed cloth.
            let mut cloth_n = vec![V::zeros(); posed.len()];
            for t in g.triangles {
                let [i, j, k] = t.map(|x| x as usize);
                let n = (posed[j] - posed[i]).cross(&(posed[k] - posed[i]));
                cloth_n[i] += n;
                cloth_n[j] += n;
                cloth_n[k] += n;
            }
            let stride = (posed.len() / 20_000).max(1);
            // (depth, cloth faces out with the body): an inward-facing
            // lining or tucked layer below the skin is hidden behind the
            // garment's outer face; outward-facing cloth inside is the
            // poke-through one sees.
            let hits: Vec<(f64, bool)> = (0..posed.len())
                .into_par_iter()
                .step_by(stride)
                .filter_map(|i| {
                    let p = posed[i];
                    let hit = query.closest_point(a(p), height * 0.05, None)?;
                    if skip_hidden.is_some_and(|h| h.get(hit.source[1]) == Some(&true)) {
                        return None;
                    }
                    let d = (p - v(hit.point)).dot(&v(hit.normal));
                    let out = cloth_n[i]
                        .try_normalize(1e-20)
                        .is_some_and(|n| n.dot(&v(hit.normal)) > 0.0);
                    (d < -0.002).then_some((-d, out))
                })
                .collect();
            let depths: Vec<f64> = hits.iter().map(|h| h.0).collect();
            let outward = hits.iter().filter(|h| h.1).count();
            let sampled = posed.len().div_ceil(stride);
            let mut seam_gap: f64 = 0.0;
            if let Some(keys) = g.seam {
                let mut first: HashMap<usize, V> = HashMap::new();
                for (i, k) in keys.iter().enumerate() {
                    match first.get(k) {
                        Some(p) => seam_gap = seam_gap.max((p - posed[i]).norm()),
                        None => {
                            first.insert(*k, posed[i]);
                        }
                    }
                }
            }
            let stretch = edge_ratios(
                g.positions,
                &posed,
                g.triangles,
                (g.triangles.len() / 60_000).max(1),
                height * 0.001,
            );
            inside_total += depths.len();
            sampled_total += sampled;
            garment_rows.push(json!({"garment":g.name,"inside_fraction":depths.len() as f64 / sampled.max(1) as f64,"inside_outward_fraction":outward as f64 / sampled.max(1) as f64,"inside_depth_m":percentiles(depths),"edge_ratio":percentiles(stretch),"internal_seam_gap_m":seam_gap}));
        }
        let rate = inside_total as f64 / sampled_total.max(1) as f64;
        let row = json!({"time":time,"lowest_body_y_m":lowest,"body_edge_ratio":percentiles(body_stretch),"garments":garment_rows,"inside_fraction":rate});
        if time.is_none() {
            baseline = Some(row.clone());
        } else if rate > worst_rate {
            worst_rate = rate;
            worst = row.clone();
        }
        rows.push(row);
    }
    let rates: Vec<f64> = rows
        .iter()
        .skip(1)
        .map(|r| r["inside_fraction"].as_f64().unwrap_or(0.0))
        .collect();
    Ok(
        json!({"clip":clip.name,"duration_s":duration,"rest":baseline,"worst_frame":worst,"inside_fraction_over_frames":percentiles(rates),"frames":rows.len()-1,
            "sampled_frames": rows}),
    )
}

#[cfg(test)]
mod tests {
    /// Gait probe on exported runs: `CHARLOOM_GAIT_GLBS=a.glb:b.glb cargo test -p charloom-mh gait_probe -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gait_probe() {
        let glbs = std::env::var("CHARLOOM_GAIT_GLBS").unwrap();
        let clip = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/mixamo/walking.fbx");
        for g in glbs.split(':') {
            let asset = charloom_io::load_motion_asset(g).unwrap();
            let sk = &asset.skeleton;
            let (motion, report) = super::load_and_retarget(&clip, Some("mixamo.com"), "w", sk, sk, false, &[]).unwrap();
            let idx = |n: &str| sk.joints.iter().position(|j| j.name == n).unwrap();
            let foot = |w: &Vec<nalgebra::Matrix4<f64>>, s: &str| {
                let p = |i: usize| super::V::new(w[i][(0, 3)], w[i][(1, 3)], w[i][(2, 3)]);
                let a = p(idx(&format!("foot.{s}")));
                let t = (1..=5).map(|n| p(idx(&format!("toe{n}-1.{s}")))).sum::<super::V>() / 5.0;
                let d = (t - a).normalize();
                (a.y, (-d.y).asin().to_degrees())
            };
            let rest = sk.world_matrices(&sk.rest_pose()).unwrap();
            let [t0, t1] = motion.time_range();
            let mut line = format!("{} ground {} scale {}", g.split('/').rev().nth(2).unwrap_or(g), report["ground"], report["leg_scale"]);
            for s in ["L", "R"] {
                let r = foot(&rest, s);
                let mut low = (f64::INFINITY, 0.0);
                for k in 0..=60 {
                    let t = t0 + (t1 - t0) * k as f64 / 60.0;
                    let w = sk.world_matrices(&motion.sample(sk, t).unwrap()).unwrap();
                    let f = foot(&w, s);
                    if f.0 < low.0 { low = f; }
                }
                line += &format!(" | {s}: ankle {:+.1}cm pitch {:+.0}deg", (low.0 - r.0) * 100.0, low.1 - r.1);
            }
            println!("GAIT {line}");
        }
    }
    use super::*;
    use charloom_core::{Joint, Transform};

    /// Hips, ankle, toe roots (one for each foot), the big toe's distal bone.
    fn leg(ankle: [f64; 3], toe: [f64; 3]) -> Skeleton {
        let joint = |name: &str, parent: Option<usize>, t: [f64; 3]| Joint {
            id: name.into(),
            name: name.into(),
            parent,
            rest: Transform::translated(t),
            matrix: None,
            deform: true,
        };
        Skeleton {
            joints: vec![
                joint("root", None, [0.0, 0.8, 0.0]),
                joint("foot.L", Some(0), ankle),
                joint("toe1-1.L", Some(1), toe),
                joint("toe1-2.L", Some(2), [0.0, 0.0, 0.03]),
                joint("toe3-1.L", Some(1), [toe[0] + 0.03, toe[1], toe[2]]),
            ],
        }
    }

    #[test]
    fn a_raised_heel_keeps_its_foot_and_toes() {
        // MakeHuman: the foot bone ~20° below the horizontal.
        let human = leg([0.1, -0.74, 0.0], [0.0, -0.04, 0.11]);
        // Digitigrade: the ankle at the hock, the metatarsus nearly vertical.
        let paw = leg([0.1, -0.55, -0.05], [0.0, -0.2, 0.03]);
        let mapped: std::collections::HashSet<usize> = [1, 2, 3].into();
        let (reference, local, report) = raised_feet(&paw, &human, &mapped);
        assert_eq!(reference, vec!["foot.L".to_string()]);
        // Mapped toe bones only (toe3-1 is not driven by the clip).
        assert_eq!(local, vec!["toe1-1.L".to_string(), "toe1-2.L".to_string()]);
        assert_eq!(report["L"]["raised"], json!(true));
        assert!(report["L"]["pitch_rad"].as_f64().unwrap() > 1.3);
        // A human foot (and the missing right foot) is left to the retarget.
        let (reference, local, report) = raised_feet(&human, &human, &mapped);
        assert!(reference.is_empty() && local.is_empty());
        assert_eq!(report["L"]["raised"], json!(false));
        assert!(report.get("R").is_none());
        // Toes turned up (a cartoon shoe) are not a raised heel either.
        let up = leg([0.1, -0.74, 0.0], [0.0, 0.04, 0.11]);
        assert!(raised_feet(&up, &human, &mapped).0.is_empty());
    }
}
