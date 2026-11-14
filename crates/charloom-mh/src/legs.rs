 use crate::geom::{V, rotation_between, v};
use crate::prior::Prior;
use crate::scan::Scan;
use crate::segment::Label;
use anyhow::Result;
use charloom_core::Transform;
use nalgebra::{Matrix3, Matrix4, Rotation3, UnitQuaternion};
use serde_json::{Value, json};

/// Slices per segment.
const BINS: usize = 10;
/// Angular sectors around the bone; a scan slice counts when its skin
/// covers this many of them (a hem or collar covers the rest).
const SECTORS: usize = 12;
const MIN_SECTORS: usize = 11;
/// Largest move of a knee or ankle across its bone, share of the height.
const MAX_SHIFT: f64 = 0.03;

fn origin(m: &Matrix4<f64>) -> V {
    V::new(m[(0, 3)], m[(1, 3)], m[(2, 3)])
}

/// A limb segment in the posed frame: start, unit axis, length, and the
/// cross-section frame (`e[1]` forward, `e[0]` = axis × forward).
#[derive(Clone, Copy)]
pub struct Bone {
    pub a: V,
    pub u: V,
    pub length: f64,
    pub e: [V; 2],
}

impl Bone {
    pub fn new(a: V, b: V) -> Option<Self> {
        let length = (b - a).norm();
        let u = (b - a).try_normalize(1e-9)?;
        let fwd = (V::z() - u * u.z).try_normalize(1e-9)?;
        Some(Self {
            a,
            u,
            length,
            e: [u.cross(&fwd), fwd],
        })
    }

    /// (t along the bone in [0, 1], cross-section coordinates).
    pub fn local(&self, p: V) -> (f64, [f64; 2]) {
        let d = p - self.a;
        let t = d.dot(&self.u) / self.length;
        (t, [d.dot(&self.e[0]), d.dot(&self.e[1])])
    }

    fn distance(&self, p: V) -> f64 {
        let d = p - self.a;
        let t = d.dot(&self.u).clamp(0.0, self.length);
        (d - self.u * t).norm()
    }
}

/// Cross-section centre (mid-range of the 2nd–98th percentiles across both
/// section axes) per slice of a bone, for slices seen all round.
pub fn centres(bone: &Bone, points: &[V], min_sectors: usize) -> Vec<Option<[f64; 2]>> {
    let mut bins: Vec<Vec<[f64; 2]>> = vec![vec![]; BINS];
    for p in points {
        let (t, q) = bone.local(*p);
        if (0.0..1.0).contains(&t) {
            bins[((t * BINS as f64) as usize).min(BINS - 1)].push(q);
        }
    }
    bins.iter()
        .map(|qs| {
            if qs.len() < 12 {
                return None;
            }
            let mut sector = [false; SECTORS];
            for q in qs {
                let s = ((q[1].atan2(q[0]) + std::f64::consts::PI) / std::f64::consts::TAU
                    * SECTORS as f64) as usize;
                sector[s.min(SECTORS - 1)] = true;
            }
            if sector.iter().filter(|s| **s).count() < min_sectors {
                return None;
            }
            let mid = |k: usize| {
                let mut xs: Vec<f64> = qs.iter().map(|q| q[k]).collect();
                xs.sort_by(f64::total_cmp);
                let n = xs.len() - 1;
                0.5 * (xs[n * 2 / 100] + xs[n * 98 / 100])
            };
            Some([mid(0), mid(1)])
        })
        .collect()
}

/// Offset at the segment's start and end, and the slices used.
type OffsetLine = ([f64; 2], [f64; 2], usize);

/// Weighted least-squares line through per-slice offsets: the offset at
/// t = 0 and t = 1 (slice centres at (k + ½) / BINS), and the slices used.
fn offset_line(offsets: &[Option<[f64; 2]>]) -> Option<OffsetLine> {
    let pts: Vec<(f64, [f64; 2])> = offsets
        .iter()
        .enumerate()
        .filter_map(|(k, o)| Some(((k as f64 + 0.5) / BINS as f64, (*o)?)))
        .collect();
    if pts.len() < 3 {
        return None;
    }
    let (t0, t1) = (pts[0].0, pts[pts.len() - 1].0);
    if t1 - t0 < 0.3 {
        return None;
    }
    let n = pts.len() as f64;
    let mt = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let stt: f64 = pts.iter().map(|p| (p.0 - mt).powi(2)).sum();
    let mut at0 = [0.0; 2];
    let mut at1 = [0.0; 2];
    for k in 0..2 {
        let mq = pts.iter().map(|p| p.1[k]).sum::<f64>() / n;
        let slope = pts.iter().map(|p| (p.0 - mt) * (p.1[k] - mq)).sum::<f64>() / stt;
        // Extrapolation past the seen slices is damped towards their mean.
        let at = |t: f64| mq + slope * (t.clamp(t0, t1) - mt) + slope * 0.5 * (t - t.clamp(t0, t1));
        at0[k] = at(0.0);
        at1[k] = at(1.0);
    }
    Some((at0, at1, pts.len()))
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
        .map_or(Matrix3::identity(), |p| {
            crate::geom::proper_rotation(worlds[p].fixed_view::<3, 3>(0, 0).into_owned())
        });
    let local = parent.transpose() * r * parent;
    let q = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(local));
    let old = charloom_core::unit_quat(pose[j].rotation);
    let new = q * old;
    pose[j].rotation = [new.i, new.j, new.k, new.w];
}

/// Limb segments a scan skin point may belong to: legs, arms, hands, torso.
fn segments(prior: &Prior, worlds: &[Matrix4<f64>]) -> Vec<(String, Bone)> {
    let mut pairs: Vec<(String, String)> = vec![
        ("spine05".into(), "neck01".into()),
        ("root".into(), "spine05".into()),
    ];
    for s in ["L", "R"] {
        for (a, b) in [
            ("upperleg01", "lowerleg01"),
            ("lowerleg01", "foot"),
            ("foot", "toe3-1"),
            ("upperarm01", "lowerarm01"),
            ("lowerarm01", "wrist"),
            ("wrist", "finger3-3"),
        ] {
            pairs.push((format!("{a}.{s}"), format!("{b}.{s}")));
        }
    }
    pairs
        .into_iter()
        .filter_map(|(a, b)| {
            let (ja, jb) = (*prior.by_name.get(&a)?, *prior.by_name.get(&b)?);
            Some((a, Bone::new(origin(&worlds[ja]), origin(&worlds[jb]))?))
        })
        .collect()
}

/// Moves each knee and ankle across its bone so MakeHuman's leg centres
/// follow the scan's visible leg (module doc). `labels`: per scan triangle.
pub fn align(
    prior: &Prior,
    scan: &Scan,
    labels: &[Label],
    pose: &mut [Transform],
) -> Result<Value> {
    let height = prior.height();
    let mut skin = vec![false; scan.positions.len()];
    for (t, tri) in scan.triangles.iter().enumerate() {
        if labels[t] == Label::Skin {
            tri.iter().for_each(|&i| skin[i as usize] = true);
        }
    }
    let skin_points: Vec<V> = (0..scan.weld.count)
        .map(|w| scan.weld.representative[w])
        .filter(|&r| skin[r])
        .map(|r| v(scan.positions[r]))
        .collect();
    let names: Vec<String> = prior
        .skeleton()
        .joints
        .iter()
        .map(|j| j.name.clone())
        .collect();
    let body = prior.body();
    let mut report = serde_json::Map::new();
    for side in ["L", "R"] {
        let hip = prior.joint(&format!("upperleg01.{side}"));
        let knee = prior.joint(&format!("lowerleg01.{side}"));
        let ankle = prior.joint(&format!("foot.{side}"));
        let worlds = prior.worlds(pose)?;
        let posed = prior.skin_mesh(prior.body, &worlds);
        let segs = segments(prior, &worlds);
        let (h, k, an) = (
            origin(&worlds[hip]),
            origin(&worlds[knee]),
            origin(&worlds[ankle]),
        );
        let (Some(thigh), Some(shin)) = (Bone::new(h, k), Bone::new(k, an)) else {
            continue;
        };
        let mut shifts: Vec<(&str, Option<OffsetLine>)> = vec![];
        for (name, bone, prefix) in [("thigh", thigh, "upperleg"), ("shin", shin, "lowerleg")] {
            // Scan skin nearer to this segment than to the other limbs and
            // the torso (this leg's own segments share it by slice).
            let own_leg = |n: &str| {
                n.ends_with(&format!(".{side}"))
                    && (n.starts_with("upperleg")
                        || n.starts_with("lowerleg")
                        || n.starts_with("foot"))
            };
            let scan_points: Vec<V> = skin_points
                .iter()
                .filter(|p| {
                    let own = bone.distance(**p);
                    own < 0.15 * height
                        && segs
                            .iter()
                            .filter(|(n, _)| !own_leg(n))
                            .all(|(_, b)| b.distance(**p) > own)
                })
                .copied()
                .collect();
            let mh_points: Vec<V> = (0..body.bind.len())
                .filter(|&i| {
                    let n = &names[prior.dominant_joint(prior.body, i)];
                    n.starts_with(prefix) && n.ends_with(&format!(".{side}"))
                })
                .map(|i| v(posed[i]))
                .collect();
            let sc = centres(&bone, &scan_points, MIN_SECTORS);
            let mc = centres(&bone, &mh_points, 1);
            let offsets: Vec<Option<[f64; 2]>> = sc
                .iter()
                .zip(&mc)
                .map(|(s, m)| {
                    Some([
                        s.as_ref()?[0] - m.as_ref()?[0],
                        s.as_ref()?[1] - m.as_ref()?[1],
                    ])
                })
                .collect();
            shifts.push((name, offset_line(&offsets)));
        }
        let to3 = |bone: &Bone, d: [f64; 2]| bone.e[0] * d[0] + bone.e[1] * d[1];
        let clamp = |d: V| {
            let m = MAX_SHIFT * height;
            if d.norm() > m { d * (m / d.norm()) } else { d }
        };
        // Knee: the thigh's far end and the shin's near end, as seen.
        let at_knee: Vec<V> = shifts
            .iter()
            .filter_map(|(n, s)| {
                let (d0, d1, _) = (*s)?;
                Some(if *n == "thigh" {
                    to3(&thigh, d1)
                } else {
                    to3(&shin, d0)
                })
            })
            .collect();
        if at_knee.is_empty() {
            report.insert(
                side.into(),
                json!({"applied": false, "reason": "leg not seen as skin all round"}),
            );
            continue;
        }
        let knee_shift = clamp(at_knee.iter().sum::<V>() / at_knee.len() as f64);
        // Ankle: the shin's far end; without a seen shin, parallel.
        let ankle_shift = match shifts[1].1 {
            Some((_, d1, _)) => clamp(to3(&shin, d1)),
            None => knee_shift,
        };
        let k_target = k + knee_shift;
        let r = rotation_between(k - h, k_target - h);
        rotate_joint(prior, pose, &worlds, hip, &r);
        let worlds = prior.worlds(pose)?;
        let (k_now, a_now) = (origin(&worlds[knee]), origin(&worlds[ankle]));
        let a_target = an + ankle_shift;
        let r = rotation_between(a_now - k_now, a_target - k_now);
        rotate_joint(prior, pose, &worlds, knee, &r);

        let bend = |h: V, k: V, a: V| (k - h).angle(&(a - k)).to_degrees();
        let worlds = prior.worlds(pose)?;
        report.insert(
            side.into(),
            json!({
                "applied": true,
                "slices": shifts.iter().map(|(n, s)| (n.to_string(), json!(s.map_or(0, |s| s.2)))).collect::<serde_json::Map<_, _>>(),
                "knee_shift_m": [knee_shift.x, knee_shift.y, knee_shift.z],
                "ankle_shift_m": [ankle_shift.x, ankle_shift.y, ankle_shift.z],
                "knee_bend_deg": {
                    "before": bend(h, k, an),
                    "after": bend(origin(&worlds[hip]), origin(&worlds[knee]), origin(&worlds[ankle])),
                },
            }),
        );
    }
    Ok(Value::Object(report))
}

/// MakeHuman's own cross-section centres along one bone (bind or posed
/// frame), interpolated between slices: a limb grown about them keeps its
/// centre line.
pub struct CentreLine {
    bone: Bone,
    centres: Vec<[f64; 2]>,
}

impl CentreLine {
    pub fn new(bone: Bone, points: &[V]) -> Option<Self> {
        let raw = centres(&bone, points, 1);
        let seen: Vec<usize> = (0..BINS).filter(|&k| raw[k].is_some()).collect();
        if seen.is_empty() {
            return None;
        }
        // Unseen slices take the nearest seen one.
        let centres = (0..BINS)
            .map(|k| {
                let near = *seen.iter().min_by_key(|&&s| s.abs_diff(k)).unwrap();
                raw[near].unwrap()
            })
            .collect();
        Some(Self { bone, centres })
    }

    /// Offset of the centre line from the bone at `p`'s slice, across the
    /// bone.
    pub fn offset(&self, p: V) -> V {
        let (t, _) = self.bone.local(p);
        let x = (t * BINS as f64 - 0.5).clamp(0.0, (BINS - 1) as f64);
        let k = (x as usize).min(BINS - 2);
        let f = x - k as f64;
        let c = |j: usize| self.centres[k][j] * (1.0 - f) + self.centres[k + 1][j] * f;
        self.bone.e[0] * c(0) + self.bone.e[1] * c(1)
    }
}

 pub fn held_joints(
    prior: &Prior,
    pose_worlds: &[Matrix4<f64>],
    report: &Value,
) -> std::collections::BTreeMap<usize, V> {
    let mut out = std::collections::BTreeMap::new();
    for side in ["L", "R"] {
        if report[side]["applied"] != json!(true) {
            continue;
        }
        let j = |n: &str| prior.by_name.get(&format!("{n}.{side}")).copied();
        let (Some(knee), Some(shin2), Some(ankle)) = (j("lowerleg01"), j("lowerleg02"), j("foot"))
        else {
            continue;
        };
        let at = |n: usize| origin(&pose_worlds[n]);
        let rest = |n: usize| origin(&prior.rest_world[n]);
        let fraction = |from: usize, mid: usize, to: usize| {
            let d = rest(to) - rest(from);
            ((rest(mid) - rest(from)).dot(&d) / d.norm_squared()).clamp(0.0, 1.0)
        };
        let (k, a) = (at(knee), at(ankle));
        out.insert(knee, k);
        out.insert(shin2, k + (a - k) * fraction(knee, shin2, ankle));
        out.insert(ankle, a);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Points on a vertical elliptic cylinder around `c(t)`, bone along −y.
    fn cylinder(c: impl Fn(f64) -> [f64; 2], keep: impl Fn(f64) -> bool) -> Vec<V> {
        let mut out = vec![];
        for i in 0..200 {
            let t = (i as f64 + 0.5) / 200.0;
            for k in 0..36 {
                let phi = k as f64 * std::f64::consts::TAU / 36.0;
                if !keep(phi) {
                    continue;
                }
                let [cx, cz] = c(t);
                out.push(V::new(cx + 0.08 * phi.cos(), -t, cz + 0.05 * phi.sin()));
            }
        }
        out
    }

    #[test]
    fn centres_follow_the_section_centre_not_the_bone() {
        let bone = Bone::new(V::zeros(), V::new(0.0, -1.0, 0.0)).unwrap();
        // e[0] is axis × forward = −y × z = −x; e[1] forward = z.
        let pts = cylinder(|t| [0.02, -0.03 + 0.04 * t], |_| true);
        let c = centres(&bone, &pts, MIN_SECTORS);
        for (k, c) in c.iter().enumerate() {
            let t = (k as f64 + 0.5) / BINS as f64;
            let c = c.expect("seen all round");
            assert!((c[0] + 0.02).abs() < 2e-3, "{c:?}");
            assert!((c[1] - (-0.03 + 0.04 * t)).abs() < 3e-3, "{c:?}");
        }
        // A slice covered half by cloth is not trusted.
        let half = cylinder(|_| [0.0, 0.0], |phi| phi < std::f64::consts::PI);
        assert!(
            centres(&bone, &half, MIN_SECTORS)
                .iter()
                .all(Option::is_none)
        );
    }

    #[test]
    fn offset_line_fits_and_damps_extrapolation() {
        let offsets: Vec<Option<[f64; 2]>> = (0..BINS)
            .map(|k| {
                let t = (k as f64 + 0.5) / BINS as f64;
                (2..8).contains(&k).then_some([0.01 + 0.02 * t, -0.01])
            })
            .collect();
        let (at0, at1, n) = offset_line(&offsets).unwrap();
        assert_eq!(n, 6);
        // Seen from t = 0.25 to 0.75; beyond, half the slope.
        assert!((at0[0] - (0.01 + 0.02 * 0.25 - 0.01 * 0.25)).abs() < 1e-9);
        assert!((at1[0] - (0.01 + 0.02 * 0.75 + 0.01 * 0.25)).abs() < 1e-9);
        assert!((at0[1] + 0.01).abs() < 1e-9);
        // Too few slices: nothing.
        let few: Vec<Option<[f64; 2]>> = (0..BINS).map(|k| (k < 2).then_some([0.0, 0.0])).collect();
        assert!(offset_line(&few).is_none());
    }
}
