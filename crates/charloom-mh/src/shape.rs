 use crate::bodysize::Fill;
use crate::geom::{V, a, v};
use crate::prior::{Prior, Region};
use crate::scan::{Scan, build_query};
use crate::segment::{Label, Segmentation};
use anyhow::Result;
use charloom_core::Vec3;
use nalgebra::Matrix4;
use rayon::prelude::*;
use serde_json::{Value, json};

/// One signed parameter: `+` morph for positive values, `-` for negative.
struct Param {
    name: String,
    plus: Option<usize>,
    minus: Option<usize>,
}

fn params(morphs: &[(String, Vec<Vec3>)]) -> Vec<Param> {
    let mut out: Vec<Param> = vec![];
    for (i, (name, _)) in morphs.iter().enumerate() {
        let (base, plus) = match name.strip_suffix('+') {
            Some(b) => (b, true),
            None => match name.strip_suffix('-') {
                Some(b) => (b, false),
                None => (name.as_str(), true),
            },
        };
        let k = out.iter().position(|p| p.name == base).unwrap_or_else(|| {
            out.push(Param {
                name: base.to_string(),
                plus: None,
                minus: None,
            });
            out.len() - 1
        });
        if plus {
            out[k].plus = Some(i);
        } else {
            out[k].minus = Some(i);
        }
    }
    out
}

/// Weight of the body-size term against visible skin (5) and the envelope:
/// close to skin, which still wins where it shows. (At 3 the cloth's
/// evidence lost to the morphs' prior: the flame's chest stayed 3 cm inside
/// a vest the fit wanted it at.)
const FILL_WEIGHT: f64 = 6.0;

/// Weight of the one-sided envelope (body outside the cloth). Soft: the
/// garments follow the body (`garment::bind` moves cloth the fitted body
/// reaches out to the clearance), so the cloth is a preference, not a
/// ceiling that shrinks the body below the person the scan shows (at 20 it
/// held the stocky flame's chest 4 cm inside its vest).
const ENVELOPE_WEIGHT: f64 = 6.0;

/// The envelope's weight on the head and neck (hair, helmets, collars):
/// hard, as before the garments followed the body.
const HEAD_ENVELOPE_WEIGHT: f64 = 20.0;

 const STYLIZED_TORSO_REACH: f64 = 2.5;

/// Reach of the trunk's front/back offset (`trunk-offset`), share of the
/// height (3.5 cm at 1.75 m).
const TRUNK_OFFSET: f64 = 0.02;

 const OUTLINE_WEIGHT: f64 = 10.0;

/// Residual scale of an outline edge, share of the height (2 cm at 1.75 m).
const OUTLINE_SCALE: f64 = 0.011;
/// Guided: weight of the outlines' sizes in the shape fit. Every marked row
/// sizes MakeHuman's own body controls (build, torso, belly, hips, bust,
/// buttocks, limb girths), so the body is the nearest *human* one to the
/// suit's outline; only what the controls cannot reach is left to the
/// capped residual of `apply_outlines`.
const OUTLINE_SIZE_WEIGHT: f64 = 5.0;

/// Half thickness of the slab an outline measures the body in, share of
/// the height (1.4 cm at 1.75 m).
pub const OUTLINE_SLAB: f64 = 0.008;

/// Weight of the limb proportion prior: the spread of the limb girth
/// parameters (the four coupled girths and MakeHuman's upper-arm, thigh and
/// calf circumferences) around their mean.
const PROPORTION_WEIGHT: f64 = 0.08;

/// Reach of the stylised knee girth along the thigh and the shin, share of
/// each segment's length.
const KNEE_REACH: f64 = 0.35;

 const REGION_SMOOTHING: usize = 20;

/// Weight 1 of a regional control: +50 % of the region's size.
const REGION_STEP: f64 = 0.5;

 fn region_morphs(prior: &Prior) -> Vec<(String, Vec<Vec3>)> {
    let m = prior.body();
    let n = m.bind.len();
    let names: Vec<String> = prior.skeleton().joints.iter().map(|j| j.name.to_lowercase()).collect();
    let bind_origin = |name: &str| -> Option<V> {
        let j = *prior.by_name.get(name)?;
        let k = prior.skin_joints.iter().position(|&x| x == j)?;
        let inv = prior.ibm[k].try_inverse()?;
        Some(inv.fixed_view::<3, 1>(0, 3).into_owned())
    };
    let weight_on = |i: usize, f: &dyn Fn(&str) -> bool| -> f64 {
        m.influences[i].iter().filter(|x| f(&names[x.joint])).map(|x| x.weight).sum::<f64>().min(1.0)
    };
    let mut out: Vec<(String, Vec<Vec3>)> = vec![];
    // Each field smoothed over the mesh's own surface (REGION_SMOOTHING
    // passes of neighbour averaging): skin weights switch over a couple of
    // centimetres, and a region grown by them creased and flattened where
    // they did; smoothed it grows round, like a balloon.
    let weld = crate::geom::Weld::new(&m.bind, 1e-7);
    let neighbours = crate::geom::adjacency(&m.prim.triangles, &weld);
    let smooth = |delta: Vec<V>| -> Vec<V> {
        let mut d: Vec<V> = weld.representative.iter().map(|&r| delta[r]).collect();
        for _ in 0..REGION_SMOOTHING {
            d = (0..d.len())
                .map(|i| {
                    if neighbours[i].is_empty() {
                        return d[i];
                    }
                    let mean = neighbours[i].iter().fold(V::zeros(), |s, &j| s + d[j as usize])
                        / neighbours[i].len() as f64;
                    (d[i] + mean) * 0.5
                })
                .collect();
        }
        weld.id.iter().map(|&w| d[w]).collect()
    };
    let mut push = |id: String, delta: Vec<V>| {
        if delta.iter().all(|d| d.norm() < 1e-9) {
            return;
        }
        let delta = smooth(delta);
        for (sign, scale) in [("+", REGION_STEP), ("-", -0.3 * REGION_STEP)] {
            out.push((format!("{id}{sign}"), delta.iter().map(|d| a(d * scale)).collect()));
        }
    };
    // The trunk's centre line: the spine's joints, by height.
    let mut spine: Vec<V> = ["root", "spine05", "spine04", "spine03", "spine02", "spine01", "neck01"]
        .iter()
        .filter_map(|j| bind_origin(j))
        .collect();
    spine.sort_by(|a, b| a.y.total_cmp(&b.y));
    let (Some(hl), Some(hr), Some(neck)) = (bind_origin("upperleg01.L"), bind_origin("upperleg01.R"), bind_origin("neck01")) else {
        return out;
    };
    if spine.len() >= 2 {
        let centre = |y: f64| -> V {
            let k = spine.windows(2).position(|w| y <= w[1].y).unwrap_or(spine.len() - 2);
            let (p, q) = (spine[k], spine[k + 1]);
            let t = ((y - p.y) / (q.y - p.y).max(1e-9)).clamp(0.0, 1.0);
            p + (q - p) * t
        };
        let hip_y = 0.5 * (hl.y + hr.y);
        let torso = (neck.y - hip_y).max(1e-6);
        let trunk = |n: &str| n.starts_with("spine") || n.starts_with("pelvis") || n == "root" || n.starts_with("breast");
        let trunk_legs = |n: &str| trunk(n) || n.starts_with("upperleg");
        for (region, level, with_legs) in [("chest", 0.72, false), ("belly", 0.38, false), ("hips", 0.05, true)] {
            let yc = hip_y + level * torso;
            let sigma = torso / 6.0;
            let (mut width, mut depth, mut front) = (vec![V::zeros(); n], vec![V::zeros(); n], vec![V::zeros(); n]);
            for i in 0..n {
                let w = if with_legs { weight_on(i, &trunk_legs) } else { weight_on(i, &trunk) };
                if w <= 0.0 {
                    continue;
                }
                let p = v(m.bind[i]);
                let g = (-0.5 * ((p.y - yc) / sigma).powi(2)).exp() * w;
                let c = centre(p.y);
                width[i] = V::new((p.x - c.x) * g, 0.0, 0.0);
                depth[i] = V::new(0.0, 0.0, (p.z - c.z) * g);
                front[i] = V::new(0.0, 0.0, (p.z - c.z).max(0.0) * g);
            }
            push(format!("region-{region}-width"), width);
            push(format!("region-{region}-depth"), depth);
            if region == "belly" {
                push("region-belly-front".into(), front);
            }
        }
    }
     
    const BINS: usize = 24;
    for (first, second, joints, bones) in [
        ("thigh", "calf", ["upperleg01", "lowerleg01", "foot"], &["upperleg", "lowerleg"][..]),
        ("upperarm", "forearm", ["upperarm01", "lowerarm01", "wrist"], &["upperarm", "lowerarm"][..]),
    ] {
        let (mut d1, mut d2) = (vec![V::zeros(); n], vec![V::zeros(); n]);
        for side in [".l", ".r"] {
            let up = side.to_uppercase();
            let ps: Vec<V> = joints.iter().filter_map(|j| bind_origin(&format!("{j}{up}"))).collect();
            if ps.len() != 3 {
                continue;
            }
            // Station along the chain (0 start, 1 middle joint, 2 end), the
            // radial direction and distance from the nearer bone.
            let place = |p: V| -> (f64, V, f64) {
                let mut best = (f64::INFINITY, 0.0, V::zeros());
                for k in 0..2 {
                    let (a0, b0) = (ps[k], ps[k + 1]);
                    let len = (b0 - a0).norm().max(1e-6);
                    let u = (b0 - a0) / len;
                    let t = ((p - a0).dot(&u) / len).clamp(0.0, 1.0);
                    let r = p - (a0 + u * (t * len));
                    if r.norm() < best.0 {
                        best = (r.norm(), k as f64 + t, r);
                    }
                }
                (best.1, if best.0 > 1e-9 { best.2 / best.0 } else { V::zeros() }, best.0)
            };
             
            let arm = first == "upperarm";
            let ends = |nm: &str| arm && nm.ends_with(side) && (nm.starts_with("shoulder01") || nm.starts_with("wrist"));
            let members: Vec<(usize, f64, f64, f64, V, f64)> = (0..n)
                .filter_map(|i| {
                    let w = weight_on(i, &|nm: &str| nm.ends_with(side) && bones.iter().any(|b| nm.starts_with(b)));
                    let wa = (w + weight_on(i, &|nm: &str| ends(nm))).min(1.0);
                    (wa > 0.0).then(|| {
                        let (st, dir, r) = place(v(m.bind[i]));
                        (i, w, wa, st, dir, r)
                    })
                })
                .collect();
            let (u0, u1) = ((ps[1] - ps[0]).normalize(), (ps[2] - ps[1]).normalize());
            // Mean radius per station bin (of the vertices mostly on the limb),
            // smoothed along the limb.
            let bin = |st: f64| ((st / 2.0 * BINS as f64) as usize).min(BINS - 1);
            let (mut sum, mut cnt) = (vec![0.0; BINS], vec![0.0; BINS]);
            for &(_, w, _, st, _, r) in &members {
                if w > 0.5 {
                    sum[bin(st)] += r;
                    cnt[bin(st)] += 1.0;
                }
            }
            let mut radius: Vec<f64> = (0..BINS).map(|b| if cnt[b] > 0.0 { sum[b] / cnt[b] } else { f64::NAN }).collect();
            let known: Vec<usize> = (0..BINS).filter(|&b| radius[b].is_finite()).collect();
            if known.is_empty() {
                continue;
            }
            for b in 0..BINS {
                if !radius[b].is_finite() {
                    let k = *known.iter().min_by_key(|&&k| k.abs_diff(b)).unwrap();
                    radius[b] = radius[k];
                }
            }
            for _ in 0..3 {
                radius = (0..BINS)
                    .map(|b| (radius[b.saturating_sub(1)] + 2.0 * radius[b] + radius[(b + 1).min(BINS - 1)]) / 4.0)
                    .collect();
            }
            let smooth = |x: f64| {
                let x = x.clamp(0.0, 1.0);
                x * x * (3.0 - 2.0 * x)
            };
            // One radius for the whole limb: grown by its own radius per
            // station, a thick thigh grew more than the narrow knee below it
            // and the calf more than the ankle, and cartoon legs and arms
            // came out as rings (thigh, knee, calf, ankle: a Michelin man).
            // The marked outlines then shape the profile along the limb.
            let limb_radius = radius.iter().sum::<f64>() / BINS as f64;
            for &(i, w, wa, st, dir, _) in &members {
                let h1 = (1.5 - st).clamp(0.0, 1.0);
                let x = (st / 0.3).clamp(0.0, 1.0);
                let root = 0.5 + 0.5 * x * x * (3.0 - 2.0 * x);
                let rad = limb_radius;
                let (w, fade) = if arm {
                    let p = v(m.bind[i]);
                    let before = (-(p - ps[0]).dot(&u0)).max(0.0);
                    let past = (p - ps[2]).dot(&u1).max(0.0);
                    (wa, (1.0 - smooth(before / rad.max(1e-6))) * (1.0 - smooth(past / (0.5 * rad).max(1e-6))))
                } else {
                    (w, 1.0)
                };
                let out = dir * (rad * w * fade);
                d1[i] += out * (h1 * root);
                d2[i] += out * (1.0 - h1);
            }
        }
        push(format!("region-{first}"), d1);
        push(format!("region-{second}"), d2);
    }
    // The neck: radial about its bone, a Gaussian along it.
    {
        let mut delta = vec![V::zeros(); n];
        if let (Some(a0), Some(b0)) = (bind_origin("neck01"), bind_origin("head")) {
            let length = (b0 - a0).norm().max(1e-6);
            let u = (b0 - a0) / length;
            for i in 0..n {
                let w = weight_on(i, &|nm: &str| nm.starts_with("neck"));
                if w <= 0.0 {
                    continue;
                }
                let d = v(m.bind[i]) - a0;
                let t = d.dot(&u) / length;
                let g = (-0.5 * ((t - 0.5) / 0.45).powi(2)).exp() * w;
                delta[i] += (d - u * d.dot(&u)) * g;
            }
        }
        push("region-neck".into(), delta);
    }
     
    let mut delta = vec![V::zeros(); n];
    for side in [".l", ".r"] {
        let up = side.to_uppercase();
        let (Some(a0), Some(toe)) = (bind_origin(&format!("foot{up}")), bind_origin(&format!("toe3-1{up}"))) else {
            continue;
        };
        let along = V::new(toe.x - a0.x, 0.0, toe.z - a0.z);
        if along.norm() < 1e-6 {
            continue;
        }
        let lateral = along.normalize().cross(&V::y()).normalize();
        for i in 0..n {
            let w = weight_on(i, &|n: &str| n.ends_with(side) && (n.starts_with("foot") || n.starts_with("toe")));
            if w <= 0.0 {
                continue;
            }
            let d = v(m.bind[i]) - a0;
            delta[i] += lateral * (d.dot(&lateral) * w);
        }
    }
    push("region-foot-width".into(), delta);
    out
}

fn girth_morphs(prior: &Prior, grow: f64, leg_centres: bool) -> Vec<(String, Vec<Vec3>)> {
    let m = prior.body();
    let names: Vec<String> = prior
        .skeleton()
        .joints
        .iter()
        .map(|j| j.name.to_lowercase())
        .collect();
    let bind_origin = |name: &str| -> Option<V> {
        let j = *prior.by_name.get(name)?;
        let k = prior.skin_joints.iter().position(|&x| x == j)?;
        let inv = prior.ibm[k].try_inverse()?;
        Some(inv.fixed_view::<3, 1>(0, 3).into_owned())
    };
    let mut out = vec![];
    // A stylised knee as thick as the thigh and calf around it: MakeHuman's
    // knee is narrower than both, and grown with them it stayed 1–5 cm
    // inside a cartoon's leg — the fine fit dented it to reach the scan.
    let mut knee = vec![V::zeros(); m.bind.len()];
    for (id, prefix, from, to) in [
        ("girth-upperarm", "upperarm", "upperarm01", "lowerarm01"),
        ("girth-forearm", "lowerarm", "lowerarm01", "wrist"),
        ("girth-thigh", "upperleg", "upperleg01", "lowerleg01"),
        ("girth-calf", "lowerleg", "lowerleg01", "foot"),
    ] {
        let mut delta = vec![V::zeros(); m.bind.len()];
        let mut any = false;
        for side in ["l", "r"] {
            let (Some(a0), Some(b0)) = (
                bind_origin(&format!("{from}.{}", side.to_uppercase())),
                bind_origin(&format!("{to}.{}", side.to_uppercase())),
            ) else {
                continue;
            };
            let length = (b0 - a0).norm();
            let u = (b0 - a0) / length;
            let proximal = matches!(prefix, "upperarm" | "upperleg");
            // A stylised leg grows about its own cross-section centres
            // (`legs.rs`), not about the bone along the front of the shin.
            let centre_at = (leg_centres && prefix.ends_with("leg"))
                .then(|| {
                    let bone = crate::legs::Bone::new(a0, b0)?;
                    let own: Vec<V> = (0..m.bind.len())
                        .filter(|&i| {
                            let n = &names[prior.dominant_joint(prior.body, i)];
                            n.starts_with(prefix) && n.ends_with(&format!(".{side}"))
                        })
                        .map(|i| v(m.bind[i]))
                        .collect();
                    crate::legs::CentreLine::new(bone, &own)
                })
                .flatten();
             
            let (upper, lower) = if prefix.ends_with("arm") { ("upperarm", "lowerarm") } else { ("upperleg", "lowerleg") };
            let (j_mid, j_end, j_start) = if prefix.ends_with("arm") {
                ("lowerarm01", "wrist", "upperarm01")
            } else {
                ("lowerleg01", "foot", "upperleg01")
            };
            let (Some(s0), Some(mid), Some(e0)) = (
                bind_origin(&format!("{j_start}.{}", side.to_uppercase())),
                bind_origin(&format!("{j_mid}.{}", side.to_uppercase())),
                bind_origin(&format!("{j_end}.{}", side.to_uppercase())),
            ) else {
                continue;
            };
            let limb_dir = (e0 - s0).normalize();
            let band = 0.25 * (mid - s0).norm().min((e0 - mid).norm());
            for (i, row) in m.influences.iter().enumerate() {
                let w: f64 = row
                    .iter()
                    .filter(|x| {
                        let n = &names[x.joint];
                        (n.starts_with(upper) || n.starts_with(lower)) && n.ends_with(&format!(".{side}"))
                    })
                    .map(|x| x.weight)
                    .sum();
                if w > 0.0 {
                    let p = v(m.bind[i]);
                    let d = p - a0;
                    // Share of this girth: past the middle joint (signed,
                    // along the limb) the lower one's, smoothly across it.
                    let h = ((p - mid).dot(&limb_dir) + band) / (2.0 * band);
                    let lower_share = {
                        let t = h.clamp(0.0, 1.0);
                        t * t * (3.0 - 2.0 * t)
                    };
                    let share = if proximal { 1.0 - lower_share } else { lower_share };
                    let w = w * share;
                    if w <= 1e-6 {
                        continue;
                    }
                    let fade = if proximal {
                        {
                            let t = (d.dot(&u) / (0.35 * length)).clamp(0.0, 1.0);
                            t * t * (3.0 - 2.0 * t)
                        }
                    } else {
                        1.0
                    };
                    let centre = centre_at.as_ref().map_or(V::zeros(), |c| c.offset(p));
                    let radial = d - u * d.dot(&u) - centre;
                    delta[i] += radial * (w * fade);
                    any = true;
                    if centre_at.is_some() {
                        // Distance from the knee along this segment.
                        let s = if proximal {
                            length - d.dot(&u)
                        } else {
                            d.dot(&u)
                        };
                        let r = (s.abs() / (KNEE_REACH * length)).min(1.0);
                        knee[i] += radial * (w * (1.0 - r * r * (3.0 - 2.0 * r)));
                    }
                }
            }
        }
        if any {
            for (sign, scale) in [("+", grow), ("-", -0.3)] {
                out.push((
                    format!("{id}{sign}"),
                    delta.iter().map(|d| a(d * scale)).collect(),
                ));
            }
        }
    }
    if knee.iter().any(|d| d.norm_squared() > 0.0) {
        for (sign, scale) in [("+", grow), ("-", -0.3)] {
            out.push((
                format!("girth-knee{sign}"),
                knee.iter().map(|d| a(d * scale)).collect(),
            ));
        }
    }
    out
}

/// Per marked outline (`guide::OutlineMark`, `None` where unusable): the body vertices (unwelded)
/// of its station by their dominant bone, its centre, its frame in the view
/// plane (`across` from edge a to edge b, `along` the slab normal) and the
/// marked edges' offsets from the centre along `across`. Shared by the shape
/// fit and the fine fit.
pub fn outline_frames(
    prior: &Prior,
    ignore: &[bool],
    outlines: &[crate::guide::OutlineMark],
) -> Vec<Option<(Vec<(usize, f64)>, V, V, V, f64, f64)>> {
    let body = prior.body;
    let skeleton = prior.skeleton();
    let count = prior.meshes[body].bind.len();
    outlines
        .iter()
        .map(|m| {
            let (a0, b0, dir) = (v(m.a), v(m.b), v(m.dir).normalize());
            let span = (b0 - a0) - dir * (b0 - a0).dot(&dir);
            let length = span.norm();
            if length < 1e-4 {
                return None;
            }
            let across = span / length;
            let along = dir.cross(&across).normalize();
            let centre = (a0 + b0) * 0.5;
            let station = m.station.as_str();
            // Continuous rows (`leg_l12`) take their part's bones.
            let station = crate::guide::outline_part(station).unwrap_or(station);
            let (part, side) = match station.rsplit_once('_') {
                Some((p, s)) if s == "l" || s == "r" => {
                    (p, Some(if s == "l" { ".l" } else { ".r" }))
                }
                _ => (station, None),
            };
            let station_bone = |n: &str| {
                let on_side = side.is_none_or(|s| n.ends_with(s));
                let starts = |ps: &[&str]| ps.iter().any(|p| n.starts_with(p));
                on_side
                    && match part {
                        "chest" | "waist" => starts(&["spine", "breast", "clavicle"]) || n == "root",
                         
                        "trunk" => {
                            starts(&["spine", "breast", "clavicle", "pelvis", "upperleg"]) || n == "root"
                        }
                        "leg" => starts(&["upperleg", "lowerleg"]),
                        "neck" => starts(&["neck"]),
                        // The bare foot (`foot_l/_r` outlines): its width.
                        "foot" => starts(&["foot", "toe"]),
                        "arm" => starts(&["upperarm", "lowerarm", "shoulder"]),
                        "hips" => {
                            starts(&["spine04", "spine05", "pelvis", "upperleg01"]) || n == "root"
                        }
                        "upperarm" => starts(&["upperarm", "shoulder"]),
                        "forearm" => starts(&["lowerarm"]),
                        "thigh" => starts(&["upperleg", "pelvis"]),
                        "knee" => starts(&["upperleg02", "lowerleg01"]),
                        "shin" => starts(&["lowerleg"]),
                        // The cranium's skin is the head bone's; the face's
                        // own bones (lids, cheeks, jaw) keep it out of the
                        // face. Sized in the fine fit only (after the head
                        // is aligned by its face), not in `apply_outlines`.
                        "skull" => n == "head",
                        _ => false,
                    }
            };
            let bone_of: Vec<bool> = skeleton
                .joints
                .iter()
                .map(|j| station_bone(&j.name.to_lowercase()))
                .collect();
             
            let members: Vec<(usize, f64)> = (0..count)
                .filter(|&i| ignore.get(i) != Some(&true))
                .filter_map(|i| {
                    let w: f64 = prior.meshes[body].influences[i]
                        .iter()
                        .filter(|w| bone_of[w.joint])
                        .map(|w| w.weight)
                        .sum();
                    (w > 0.05).then_some((i, w.min(1.0)))
                })
                .collect();
            (!members.is_empty())
                .then(|| (members, centre, across, along, -0.5 * length, 0.5 * length))
        })
        .collect()
}

 pub const CUT_MEMBER: f64 = 0.2;

 pub fn section_extent(
    pos: impl Fn(usize) -> V,
    triangles: &[[u32; 3]],
    member: &[bool],
    c: V,
    across: V,
    along: V,
    reach: f64,
) -> Option<(f64, f64)> {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for t in triangles {
        let ids = t.map(|i| i as usize);
        if !ids.iter().all(|&i| member[i]) {
            continue;
        }
        let p = ids.map(&pos);
        let d = p.map(|q| (q - c).dot(&along));
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            if (d[a] > 0.0) == (d[b] > 0.0) {
                continue;
            }
            let q = p[a] + (p[b] - p[a]) * (d[a] / (d[a] - d[b]));
            let u = (q - c).dot(&across);
            if u.abs() <= reach {
                lo = lo.min(u);
                hi = hi.max(u);
            }
        }
    }
    (hi - lo > 1e-4).then_some((lo, hi))
}

/// Residual cap of `apply_outlines` in the guided body, share of the height
/// (4.4 cm at 1.75 m), for `subject.proportions` "human"; see
/// [`guide::Proportions`].
pub const GUIDE_RESIDUAL: f64 = 0.025;

/// Guided residual cap and control reach by `subject.proportions`:
/// (residual cap, torso control reach, limb girth growth).
/// (residual cap share of the height, torso controls' reach, the other
/// body controls' reach, the regional controls' reach, the visible skin's
/// residual cap share of the height in the fine fit).
pub fn guided_limits(p: crate::guide::Proportions) -> (f64, f64, f64, f64, f64) {
    use crate::guide::Proportions::*;
    match p {
        Human => (GUIDE_RESIDUAL, 1.0, 1.0, 0.3, 0.04),
        // MakeHuman's own targets stay within their sculpted 0-1 range for
        // every subject: pushed past it a target only multiplies its sculpt
        // (its dents and flats with it; at 3x the weight sagged into side
        // folds). Beyond human sizes the smooth regional fields grow.
        Stylized => (0.04, 1.0, 1.0, 1.5, 0.04),
         
        Extreme => (0.04, EXTREME_REACH, EXTREME_REACH, 5.0, 0.08),
    }
}

/// Reach of the body controls for `extreme` proportions.
const EXTREME_REACH: f64 = 1.0;
 const EXTREME_TORSO_WIDTH: f64 = 1.5;

/// The body's own controls, which the guided `proportions` may take past
/// MakeHuman's slider ends (head, face, hand and foot controls keep them).
const BODY_CONTROLS: &[&str] = &[
    "build-weight", "build-muscle", "stomach-pregnant", "torso-scale-depth", "torso-scale-horiz",
    "waist-circ", "hips-circ", "bust-circ", "pectoral", "dorsi", "buttocks-volume",
    "upperarm-circ", "thigh-circ", "calf-circ", "neck-circ", "breast-size",
];

/// Smoothing of a chain's scale profile along it (Gaussian sigma), share
/// of the height (1.75 cm at 1.75 m); see `apply_outlines`. The suit's rows
/// are dense (2-3 cm) and smooth; at 4.4 cm the corrections blurred past
/// MakeHuman's own ledges (the knee under a cartoon thigh), left as rings.
const PROFILE_SMOOTHING: f64 = 0.01;

/// Reach of a marked outline's deformation along the body from its
/// station, share of the height (10 cm at 1.75 m): it fades out over this.
const GUIDE_REACH: f64 = 0.057;

 pub fn apply_outlines(
    prior: &Prior,
    ignore: &[bool],
    outlines: &[crate::guide::OutlineMark],
    posed: &mut [Vec3],
    cap: Option<f64>,
) -> Value {
     
    const CHAINS: &[(&[&str], Option<(&[&str], &str)>)] = &[
        (&["chest", "waist", "hips", "trunk"], None),
        (&["neck"], None),
        (&["thigh_l", "knee_l", "shin_l", "leg_l"], Some((&["upperleg", "lowerleg"], ".l"))),
        (&["thigh_r", "knee_r", "shin_r", "leg_r"], Some((&["upperleg", "lowerleg"], ".r"))),
        (&["upperarm_l", "forearm_l", "arm_l"], Some((&["upperarm", "lowerarm"], ".l"))),
        (&["upperarm_r", "forearm_r", "arm_r"], Some((&["upperarm", "lowerarm"], ".r"))),
    ];
    let limb_weights = |prefixes: &[&str], side: &str| -> Vec<(usize, f64)> {
        let bone: Vec<bool> = prior
            .skeleton()
            .joints
            .iter()
            .map(|j| {
                let n = j.name.to_lowercase();
                n.ends_with(side) && prefixes.iter().any(|p| n.starts_with(p))
            })
            .collect();
        prior.meshes[prior.body]
            .influences
            .iter()
            .enumerate()
            .filter(|(i, _)| ignore.get(*i) != Some(&true))
            .filter_map(|(i, row)| {
                let w: f64 = row.iter().filter(|w| bone[w.joint]).map(|w| w.weight).sum();
                (w > 1e-3).then_some((i, w.min(1.0)))
            })
            .collect()
    };
    let height = prior.height();
    let reach_along = GUIDE_REACH * height;
    let n = posed.len();
    let frames = outline_frames(prior, ignore, outlines);
     
    let mut disp = vec![[V::zeros(); 2]; n];
    let mut wsum = vec![[0.0f64; 2]; n];
    let mut report = vec![];
    for &(chain, limb) in CHAINS {
        for (ki, kind) in ['w', 'd'].into_iter().enumerate() {
            // (axis coordinate, scale, centre, across, along)
            // (axis coordinate, scale, centre, across, along, edge shift)
            let mut rows: Vec<(f64, f64, V, V, V, f64)> = vec![];
            let mut member_w: std::collections::HashMap<usize, f64> = Default::default();
            for (m, frame) in outlines.iter().zip(&frames) {
                let Some((members, c, across, along, lo, hi)) = frame else {
                    continue;
                };
                let in_chain = chain.contains(&m.station.as_str())
                    || crate::guide::outline_part(&m.station).is_some_and(|p| chain.contains(&p));
                if m.kind != kind || !in_chain {
                    continue;
                }
                let mut member = vec![false; n];
                for &(i, w) in members {
                    member[i] = w >= CUT_MEMBER;
                }
                let Some((e_lo, e_hi)) = section_extent(
                    |i| v(posed[i]),
                    &prior.body().prim.triangles,
                    &member,
                    *c,
                    *across,
                    *along,
                    1.5 * (hi - lo),
                ) else {
                    report.push(json!({"station": m.station, "kind": kind.to_string(), "applied": false}));
                    continue;
                };
                let k = ((hi - lo) / (e_hi - e_lo).max(1e-4)).clamp(0.6, 2.5);
                 
                let mut shift = 0.5 * (lo + hi) - 0.5 * (e_lo + e_hi);
                let limit = if kind == 'd' { 0.35 } else { 0.1 } * (hi - lo);
                shift = shift.clamp(-limit, limit);
                rows.push((0.0, k, c + across * (0.5 * (e_lo + e_hi)), *across, *along, shift));
                for &(i, w) in members {
                    let e = member_w.entry(i).or_default();
                    *e = e.max(w);
                }
                report.push(json!({
                    "station": m.station, "kind": kind.to_string(), "view": m.view,
                    "marked_mm": ((hi - lo) * 1000.0).round(), "before_mm": ((e_hi - e_lo) * 1000.0).round(),
                    "scale": (k * 1000.0).round() / 1000.0,
                    "shift_mm": ((0.5 * (lo + hi) - 0.5 * (e_lo + e_hi)) * 1000.0).round(),
                }));
            }
            if rows.is_empty() {
                continue;
            }
            // The chain's axis: the first station's limb direction.
            let (o, axis_dir) = (rows[0].2, rows[0].4);
            for r in rows.iter_mut() {
                r.0 = (r.2 - o).dot(&axis_dir);
            }
            rows.sort_by(|a, b| a.0.total_cmp(&b.0));
             
            let sigma = PROFILE_SMOOTHING * height;
            let smoothed: Vec<(f64, f64)> = rows
                .iter()
                .map(|r| {
                    let (mut k, mut sh, mut ws) = (0.0, 0.0, 0.0);
                    for q in &rows {
                        let w = (-((q.0 - r.0) / sigma).powi(2) * 0.5).exp();
                        k += q.1 * w;
                        sh += q.5 * w;
                        ws += w;
                    }
                    (k / ws, sh / ws)
                })
                .collect();
            for (r, (k, sh)) in rows.iter_mut().zip(smoothed) {
                r.1 = k;
                r.5 = sh;
            }
            let ease = |t: f64| {
                let t = t.clamp(0.0, 1.0);
                1.0 - t * t * (3.0 - 2.0 * t)
            };
            if let Some((prefixes, side)) = limb {
                member_w = limb_weights(prefixes, side).into_iter().collect();
            }
            // Past its ends a chain keeps its end rows' values (the centre
            // running on along the last two rows' line: an A-pose leg
            // slants 3-4 cm hip to ankle) and its say fades out: `cover`
            // eases from 1 to 0 over GUIDE_REACH, except at a limb's
            // distal end (wrist, ankle), held to the limb's end. Held at
            // the leg's top too, the stocky woman's thighs grew flaps at
            // the hips.
            let n_rows = rows.len();
            let (first, last) = (rows[0], rows[n_rows - 1]);
            // The distal end: the one farther from the body's middle.
            let middle = posed.iter().fold(V::zeros(), |s, q| s + v(*q)) / posed.len().max(1) as f64;
            let first_distal = (first.2 - middle).norm() > (last.2 - middle).norm();
            let profile = |t: f64| -> (f64, V, V, f64, f64) {
                if t <= first.0 || t >= last.0 {
                    let low = t <= first.0;
                    let r = if low { first } else { last };
                    let (p0, p1) = if low { (rows[0], rows[1.min(n_rows - 1)]) } else { (rows[n_rows.saturating_sub(2)], rows[n_rows - 1]) };
                    let c = if (p1.0 - p0.0).abs() > 1e-6 {
                        p0.2 + (p1.2 - p0.2) * ((t - p0.0) / (p1.0 - p0.0))
                    } else {
                        r.2
                    };
                    let held = limb.is_some() && low == first_distal;
                    let cover = if held { 1.0 } else { ease((t - r.0).abs() / reach_along) };
                    return (r.1, c, r.3, r.5, cover);
                }
                let k = rows.windows(2).position(|w| t >= w[0].0 && t <= w[1].0).unwrap_or(0);
                let (a0, b0) = (rows[k], rows[k + 1]);
                let f = ((t - a0.0) / (b0.0 - a0.0).max(1e-9)).clamp(0.0, 1.0);
                (
                    a0.1 + (b0.1 - a0.1) * f,
                    a0.2 + (b0.2 - a0.2) * f,
                    (a0.3 + (b0.3 - a0.3) * f).normalize(),
                    a0.5 + (b0.5 - a0.5) * f,
                    1.0,
                )
            };
            for (&i, &w) in &member_w {
                let p = v(posed[i]);
                let (k, c, across, shift, cover) = profile((p - o).dot(&axis_dir));
                // Centre at this point of the chain: the profile's centre
                // carried along the axis to the vertex.
                let c = c + axis_dir * (p - c).dot(&axis_dir);
                let u = (p - c).dot(&across);
                 
                let w = w * cover;
                disp[i][ki] += across * ((u * (k - 1.0) + shift) * w);
                wsum[i][ki] += w;
            }
        }
    }
    let (mut moved, mut capped) = (0usize, 0usize);
    for i in 0..n {
        // Overlapping chains (hips and thighs) average; a vertex in one
        // chain takes its share by its skin weight.
        let mut d = (0..2).fold(V::zeros(), |s, k| s + disp[i][k] / wsum[i][k].max(1.0));
        // Only a residual on MakeHuman's controls (fitted to the same rows in
        // `fit_shape`): saturating smoothly at `cap`, so the body stays the
        // controls' human shape and takes no more than the last few
        // centimetres from the free deformation.
        if let Some(cap) = cap {
            let len = d.norm();
            if len > 1e-9 {
                d *= cap * (len / cap).tanh() / len;
            }
            capped += usize::from(len > 0.5 * cap);
        }
        if d.norm() > 1e-6 {
            posed[i] = a(v(posed[i]) + d);
            moved += 1;
        }
    }
    json!({"applied": true, "stations": report, "moved_vertices": moved, "cap_m": cap, "near_cap": capped})
}

/// Fits the prior body's shape morphs and writes the shaped bind and rest
/// positions back into the prior. `worlds` is the fitted pose; parameters
/// named in `skip` stay at zero. Body vertices marked in `ignore` (fingers a
/// creature hand does not have) take no part in the loss. With `fill`
/// (body-size targets per unwelded body vertex) the covered body is pulled
/// towards them and the limb girths are fitted too (to 1.8× on a `stylized`
/// subject, 1.4× otherwise).
#[allow(clippy::too_many_arguments)]
pub fn fit_shape(
    prior: &mut Prior,
    scan: &Scan,
    seg: &Segmentation,
    worlds: &mut Vec<Matrix4<f64>>,
    skip: &[&str],
    ignore: &[bool],
    fill: Option<&[Option<Fill>]>,
    stylized: bool,
    leg_centres: bool,
    outlines: &[crate::guide::OutlineMark],
    guided: bool,
     
    limits: Option<(f64, f64, f64)>,
    // Guided: the bodysuit edits' silhouettes and the eyes' midpoint, which
    // set the head's size controls (`head_from_suit`).
    contours: &[crate::guide::SuitContour],
    head_contours: &[crate::guide::SuitContour],
    fallback_controls: Option<[f64; 3]>,
    eyes: Option<V>,
) -> Result<Value> {
    let body = prior.body;
    let mut morphs = prior.meshes[body].morphs.clone();
     
    let head = prior.joint("head");
    let skeleton = prior.skeleton();
    let head_weight: Vec<f64> = prior.meshes[body]
        .influences
        .iter()
        .map(|row| {
            row.iter()
                .filter(|w| {
                    prior.is_ancestor(head, w.joint)
                        || skeleton.joints[w.joint]
                            .name
                            .to_lowercase()
                            .starts_with("neck")
                })
                .map(|w| w.weight)
                .sum::<f64>()
                .clamp(0.0, 1.0)
        })
        .collect();
    for (name, delta) in morphs.iter_mut() {
        if name.starts_with("build-") {
            for (d, w) in delta.iter_mut().zip(&head_weight) {
                *d = d.map(|c| c * (1.0 - w));
            }
        }
    }
     
    if fill.is_some() && !morphs.is_empty() {
        let trunk: Vec<f64> = prior.meshes[body]
            .influences
            .iter()
            .map(|row| {
                row.iter()
                    .filter(|w| {
                        let n = skeleton.joints[w.joint].name.to_lowercase();
                        ["spine", "breast", "clavicle", "pelvis"]
                            .iter()
                            .any(|p| n.starts_with(p))
                            || n == "root"
                    })
                    .map(|w| w.weight)
                    .sum::<f64>()
                    .clamp(0.0, 1.0)
            })
            .collect();
        let step = TRUNK_OFFSET * prior.height();
        for (sign, suffix) in [(1.0, '+'), (-1.0, '-')] {
            morphs.push((
                format!("trunk-offset{suffix}"),
                trunk.iter().map(|w| [0.0, 0.0, sign * step * w]).collect(),
            ));
        }
    }
     
    if !morphs.is_empty() && fill.is_some() && limits.is_none() {
        morphs.extend(girth_morphs(prior, if stylized { 0.8 } else { 0.4 }, stylized && leg_centres));
    }
    if !morphs.is_empty() && limits.is_some() {
        morphs.extend(region_morphs(prior));
    }
    if morphs.is_empty() {
        return Ok(
            json!({"applied": false, "reason": "prior has no shape morphs (use body-shape.glb)"}),
        );
    }
    let ps: Vec<Param> = params(&morphs)
        .into_iter()
        .filter(|p| !skip.contains(&p.name.as_str()))
        .filter(|p| !std::env::var("MH_SKIP_MORPHS").unwrap_or_default().split(',').any(|n| n == p.name))
        .collect();
    let bind0 = prior.meshes[body].bind.clone();
    let triangles = crate::creature::unmasked_triangles(&prior.meshes[body].prim.triangles, ignore);
    let height = prior.height();
    let shaped = |x: &[f64]| -> Vec<Vec3> {
        let mut bind = bind0.clone();
        for (p, &val) in ps.iter().zip(x) {
            let (m, w) = if val >= 0.0 {
                (p.plus, val)
            } else {
                (p.minus, -val)
            };
            let Some(m) = m else { continue };
            if w == 0.0 {
                continue;
            }
            for (b, d) in bind.iter_mut().zip(&morphs[m].1) {
                for k in 0..3 {
                    b[k] += d[k] * w;
                }
            }
        }
        bind
    };
    // Scan samples: visible skin (the body must reach it) and garment points
    // (the body must stay inside them).
    let skin: Vec<usize> = (0..scan.weld.count)
        .map(|w| scan.weld.representative[w])
        .filter(|&r| seg.vertex_label[r] == Label::Skin)
        .collect();
    let skin: Vec<usize> = skin
        .iter()
        .step_by((skin.len() / 6000).max(1))
        .copied()
        .collect();
    // The envelope holds the torso and limbs: hands and feet have their own
    // fit, and a wrist guard's rim held MakeHuman's hand small.
    let body_vertices: Vec<usize> = (0..bind0.len())
        .step_by((bind0.len() / 5000).max(1))
        .filter(|&i| ignore.get(i) != Some(&true))
        .filter(|&i| {
            matches!(
                prior.region[prior.dominant_joint(body, i)],
                Some(Region::Torso | Region::Arm | Region::Leg | Region::Head)
            )
        })
        .collect();
     
    let envelope_scale: Vec<f64> = {
        let skeleton = prior.skeleton();
        let head = prior.joint("head");
        body_vertices
            .iter()
            .map(|&i| {
                let j = prior.dominant_joint(body, i);
                if prior.is_ancestor(head, j)
                    || skeleton.joints[j].name.to_lowercase().starts_with("neck")
                {
                    HEAD_ENVELOPE_WEIGHT / ENVELOPE_WEIGHT
                } else if guided {
                    // Guided: cloth never shapes the body below the neck.
                    0.0
                } else {
                    1.0
                }
            })
            .collect()
    };
    let reach = height * 0.04;
    // Body normals of the unshaped posed body: the envelope counts cloth
    // facing the same way only.
    let body_normals: Vec<Vec3> = {
        let posed = prior.skin_positions(body, &bind0, worlds);
        let weld = crate::geom::Weld::new(&bind0, 1e-7);
        let welded = crate::geom::vertex_normals(&posed, &prior.meshes[body].prim.triangles, &weld);
        weld.id.iter().map(|&w| welded[w]).collect()
    };
    let find = |name: &str| ps.iter().position(|p| p.name == name);
    // Chest (bust circumference, lats) against the waist.
    let chest_index: Vec<usize> = ["bust-circ", "dorsi"]
        .iter()
        .filter_map(|n| find(n))
        .collect();
    let waist_index = find("waist-circ");
    let prior_scale: Vec<f64> = ps
        .iter()
        .map(|p| {
            if ["pectoral", "dorsi"].contains(&p.name.as_str()) {
                3.0
            } else if p.name.starts_with("build-") {
                // The whole-body build is the preferred explanation.
                0.5
            } else {
                1.0
            }
        })
        .collect();
    let girth_index: Vec<usize> = ps
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            p.name.starts_with("girth-")
                || ["upperarm-circ", "thigh-circ", "calf-circ"].contains(&p.name.as_str())
        })
        .map(|(k, _)| k)
        .collect();
    // Body-size targets on a subsample of the covered body.
    let fills: Vec<(usize, Fill)> = {
        let all: Vec<(usize, Fill)> = fill
            .unwrap_or(&[])
            .iter()
            .enumerate()
            .filter_map(|(i, f)| Some((i, (*f)?)))
            .filter(|(i, _)| ignore.get(*i) != Some(&true))
            .collect();
        let stride = (all.len() / 6000).max(1);
        all.into_iter().step_by(stride).collect()
    };
    let fill_mass: f64 = fills.iter().map(|(_, f)| f.weight).sum();
    // Body outside the clothes at one body vertex: distance outward past a
    // garment surface, squared and scaled by the reach.
    let envelope_at = |posed: &[Vec3], i: usize| -> f64 {
        let Some(h) = scan.query.closest_point(posed[i], reach, None) else {
            return 0.0;
        };
        if seg.triangle_label[h.source[1]] == Label::Skin || seg.is_prop(h.source[1]) {
            return 0.0;
        }
         
        let mut nq = v(h.normal);
        let facing = nq.dot(&v(body_normals[i]));
        if facing < -0.3 && h.distance_m < 0.01 * height {
            nq = -nq;
        } else if facing < 0.3 {
            return 0.0;
        }
        let out = (v(posed[i]) - v(h.point)).dot(&nq);
        if out > 0.0 {
            (out / reach).powi(2)
        } else {
            0.0
        }
    };
     
    let posed_worlds: Vec<Matrix4<f64>> = worlds.clone();
    let arm_chains: Vec<(usize, Vec<usize>, Vec<usize>)> = ["L", "R"]
        .iter()
        .filter_map(|side| {
            let root = *prior.by_name.get(&format!("upperarm01.{side}"))?;
            let members: Vec<usize> = (0..bind0.len())
                .filter(|&i| prior.dominant_joint(body, i) == root)
                .collect();
            let joints: Vec<usize> = (0..prior.skeleton().joints.len())
                .filter(|&j| prior.is_ancestor(root, j))
                .collect();
            (!members.is_empty()).then_some((root, members, joints))
        })
        .collect();
    let ibm_of: std::collections::HashMap<usize, (Matrix4<f64>, Matrix4<f64>)> = prior
        .skin_joints
        .iter()
        .zip(&prior.ibm)
        .filter_map(|(&j, m)| Some((j, (*m, m.try_inverse()?))))
        .collect();
    let arm_roots = |bind: &[Vec3]| -> (Vec<Matrix4<f64>>, Vec<f64>) {
        let mut out = posed_worlds.clone();
        let mut shifts = vec![];
        for (root, members, joints) in &arm_chains {
            let mut delta = members
                .iter()
                .fold(V::zeros(), |s, &i| s + v(bind[i]) - v(bind0[i]))
                / members.len() as f64;
            let cap = 0.04 * height;
            if delta.norm() > cap {
                delta *= cap / delta.norm();
            }
            shifts.push((delta.norm() * 1000.0).round() / 1000.0);
            // One world translation for the whole chain: the upper arm's
            // bind shift as posed. (Each joint's own skinning rotation of
            // the shift moved the joints of a posed hand apart: a curled
            // finger's bones then sat 1-2 cm off its mesh after the rig
            // refit, and the walk crushed the fingers.)
            let Some((ibm, _)) = ibm_of.get(root) else {
                continue;
            };
            let skinning = posed_worlds[*root] * ibm;
            let shift = Matrix4::new_translation(&-(skinning.fixed_view::<3, 3>(0, 0) * delta));
            for &j in joints {
                if ibm_of.contains_key(&j) {
                    out[j] = shift * posed_worlds[j];
                }
            }
        }
        (out, shifts)
    };
    // Marked body outlines (`guide::OutlineMark`): per mark, the body
    // vertices of its station (by their dominant bone), its frame in the
    // view plane (`across` from edge a to edge b, `along` the slab normal)
    // and the marked edges' offsets from its centre.
    let outline_members: Vec<Option<(Vec<(usize, f64)>, V, V, V, f64, f64)>> =
        outline_frames(prior, ignore, outlines);
    let slab = OUTLINE_SLAB * height;
    // Per mark: the body's edges across it, (min, max) offsets from its
    // centre over the station's vertices within the slab.
    let outline_edges = |posed: &[Vec3]| -> Vec<Option<(f64, f64)>> {
        outline_members
            .iter()
            .map(|frame| {
                let (members, c, across, along, lo, hi) = frame.as_ref()?;
                let reach = 1.5 * (hi - lo);
                let mut e = (f64::INFINITY, f64::NEG_INFINITY);
                let mut count = 0;
                for &(i, w) in members {
                    if w < 0.5 {
                        continue;
                    }
                    let d = v(posed[i]) - c;
                    if d.dot(along).abs() > slab {
                        continue;
                    }
                    let t = d.dot(across);
                    if t.abs() > reach {
                        continue;
                    }
                    e = (e.0.min(t), e.1.max(t));
                    count += 1;
                }
                (count >= 4).then_some(e)
            })
            .collect()
    };
     
    let placing: Vec<bool> = outlines
        .iter()
        .map(|m| {
            m.kind == 'd'
                && (["chest", "waist", "hips"].contains(&m.station.as_str())
                    || crate::guide::outline_part(&m.station) == Some("trunk"))
        })
        .collect();
    let outline_term = |posed: &[Vec3]| -> f64 {
        let count = placing.iter().filter(|p| **p).count();
        if count == 0 || outline_members.len() != placing.len() {
            return 0.0;
        }
        let scale = (OUTLINE_SCALE * height).powi(2);
        let edges = outline_edges(posed);
        let mut sum = 0.0;
        for ((e, frame), place) in edges.iter().zip(&outline_members).zip(&placing) {
            let Some((_, _, _, _, lo, hi)) = frame else {
                continue;
            };
            if !place {
                continue;
            }
            // A station the body does not reach counts as fully missed.
            let (a, b) = e.unwrap_or((*lo, *lo));
            sum += (0.5 * (a + b) - 0.5 * (lo + hi)).powi(2) / scale;
        }
        sum / count as f64
    };
     
    let size_cuts: Vec<Option<Vec<[u32; 3]>>> = if guided {
        let x0 = vec![0.0f64; ps.len()];
        let b0 = shaped(&x0);
        let p0 = prior.skin_positions(body, &b0, &arm_roots(&b0).0);
        outline_members
            .iter()
            .map(|frame| {
                let (members, c, _, along, _, _) = frame.as_ref()?;
                let mut member = vec![false; p0.len()];
                for &(i, w) in members {
                    member[i] = w >= CUT_MEMBER;
                }
                let band = 0.06 * height;
                let tris: Vec<[u32; 3]> = triangles
                    .iter()
                    .filter(|t| {
                        t.iter().all(|&i| member[i as usize])
                            && t.iter().any(|&i| (v(p0[i as usize]) - c).dot(along).abs() < band)
                    })
                    .copied()
                    .collect();
                (!tris.is_empty()).then_some(tris)
            })
            .collect()
    } else {
        vec![]
    };
    let size_term = |posed: &[Vec3]| -> f64 {
        if size_cuts.is_empty() {
            return 0.0;
        }
        let scale = (OUTLINE_SCALE * height).powi(2);
        let all = vec![true; posed.len()];
        let rows = size_cuts
            .par_iter()
            .zip(&outline_members)
            .filter_map(|(tris, frame)| {
                let tris = tris.as_ref()?;
                let (_, c, across, along, lo, hi) = frame.as_ref()?;
                // A row the body does not reach counts as fully missed.
                let missed = (hi - lo).powi(2) / scale;
                let r = section_extent(|i| v(posed[i]), tris, &all, *c, *across, *along, 1.5 * (hi - lo))
                    .map_or(missed, |(a, b)| (((b - a) - (hi - lo)).powi(2) / scale).min(missed));
                Some(r)
            })
            .collect::<Vec<f64>>();
        crate::geom::ordered_sum(&rows) / rows.len().max(1) as f64
    };
     
    let placing_now = std::cell::Cell::new(false);
    let loss = |x: &[f64]| -> (f64, f64, f64, f64) {
        let bind = shaped(x);
        let posed = prior.skin_positions(body, &bind, &arm_roots(&bind).0);
        let query = build_query(&posed, &triangles);
        // 1. Visible skin to the body (one-sided, robust).
        let skin_term: Vec<f64> = skin
            .par_iter()
            .map(|&s| {
                let d = query
                    .closest_point(scan.positions[s], reach, None)
                    .map_or(reach, |h| h.distance_m);
                (d / reach).powi(2)
            })
            .collect::<Vec<f64>>();
        let skin_term = crate::geom::ordered_sum(&skin_term) / skin.len().max(1) as f64;
        // 2. Body outside the clothes: distance outward past a garment surface.
        let envelope: Vec<f64> = body_vertices
            .par_iter()
            .zip(&envelope_scale)
            .map(|(&i, k)| k * envelope_at(&posed, i))
            .collect::<Vec<f64>>();
        let envelope = crate::geom::ordered_sum(&envelope) / body_vertices.len().max(1) as f64;
        // 3. Covered body to *shell − gap* along its centreline ray: two-
        //    sided for a measured or stated gap, outward only otherwise.
        let fill_term = if fill_mass > 0.0 {
            let rows = fills
                .par_iter()
                .map(|(i, f)| {
                    let r = (v(posed[*i]) - f.center).dot(&f.dir);
                    let d = if f.two_sided {
                        r - f.radius
                    } else {
                        (r - f.radius).min(0.0)
                    };
                    f.weight * (d / reach).powi(2)
                })
                .collect::<Vec<f64>>();
            crate::geom::ordered_sum(&rows) / fill_mass
        } else {
            0.0
        };
        // 4. Prior: stay near MakeHuman's own shape, limbs in proportion.
        let mut prior_term = x
            .iter()
            .zip(&prior_scale)
            .map(|(w, k)| w * w * k)
            .sum::<f64>()
            * 0.03;
        if girth_index.len() > 1 {
            let mean = girth_index.iter().map(|&k| x[k]).sum::<f64>() / girth_index.len() as f64;
            prior_term += PROPORTION_WEIGHT
                * girth_index
                    .iter()
                    .map(|&k| (x[k] - mean).powi(2))
                    .sum::<f64>();
        }
        // A chest grown far past the waist is a cartoon V, not a human
        // torso (the flame's chest swelled sideways over a narrow waist).
        if let (Some(w), false) = (waist_index, chest_index.is_empty()) {
            let chest = chest_index.iter().map(|&k| x[k]).sum::<f64>() / chest_index.len() as f64;
            prior_term += PROPORTION_WEIGHT * (chest - x[w] - 0.5).max(0.0).powi(2);
        }
        (
            5.0 * skin_term
                + ENVELOPE_WEIGHT * envelope
                + FILL_WEIGHT * fill_term
                + if placing_now.get() {
                    OUTLINE_WEIGHT * outline_term(&posed)
                } else {
                    0.0
                }
                + OUTLINE_SIZE_WEIGHT * size_term(&posed)
                + prior_term,
            skin_term,
            envelope,
            fill_term,
        )
    };
    let mut x = vec![0.0f64; ps.len()];
     
    let foot_width_k = ps.iter().position(|p| p.name == "region-foot-width");
    if let Some(k) = foot_width_k {
        let b0 = shaped(&x);
        let p0 = prior.skin_positions(body, &b0, &arm_roots(&b0).0);
        let skeleton = prior.skeleton();
        let mut ratios = vec![];
        for (side, station) in [(".l", "foot_l"), (".r", "foot_r")] {
            let Some(mark) = outlines.iter().find(|m| m.station == station && m.kind == 'w') else {
                continue;
            };
            let (ma, mb, md) = (v(mark.a), v(mark.b), v(mark.dir).normalize());
            let want = ((mb - ma) - md * (mb - ma).dot(&md)).norm();
            let foot: Vec<V> = (0..p0.len())
                .filter(|&i| {
                    let j = prior.dominant_joint(body, i);
                    let n = skeleton.joints[j].name.to_lowercase();
                    n.ends_with(side) && (n.starts_with("foot") || n.starts_with("toe"))
                })
                .map(|i| v(p0[i]))
                .collect();
            if foot.len() < 20 {
                continue;
            }
            // Long axis on the floor: the farthest pair's horizontal heading.
            let c = foot.iter().fold(V::zeros(), |s, q| s + q) / foot.len() as f64;
            let far = foot.iter().copied().max_by(|a, b| (a - c).xz().norm().total_cmp(&(b - c).xz().norm())).unwrap_or(c);
            let along = V::new(far.x - c.x, 0.0, far.z - c.z);
            if along.norm() < 1e-6 {
                continue;
            }
            let lateral = along.normalize().cross(&V::y()).normalize();
            let (lo, hi) = foot.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |e, q| {
                let u = (q - c).dot(&lateral);
                (e.0.min(u), e.1.max(u))
            });
            if hi - lo > 1e-4 {
                ratios.push(want / (hi - lo));
            }
        }
        if !ratios.is_empty() {
            let r = ratios.iter().sum::<f64>() / ratios.len() as f64;
            let reach = limits.map_or(1.0, |l| l.2);
            x[k] = ((r - 1.0) / REGION_STEP).clamp(-0.3, reach);
        }
    }
     
    let head_keys: Vec<usize> = ["head-scale-vert", "head-scale-horiz", "head-scale-depth"]
        .iter()
        .filter_map(|n| ps.iter().position(|p| p.name == *n))
        .collect();
    // Visible scan scalp is skin data. Hair and clothing are upper bounds
    // only. The bald GPT edits measure the scalp under them.
    let scan_on_skull = |eyes: V| {
        let seen = crate::fit::hidden_head::seen_from_outside(scan, eyes, 0.3);
        let head = prior.joint("head");
        let head_vertices: Vec<usize> = (0..bind0.len())
            .filter(|&i| {
                let j = prior.dominant_joint(body, i);
                j == head || prior.is_ancestor(head, j) && prior.region[j] == Some(Region::Head)
            })
            .collect();
        let mut in_head = vec![false; bind0.len()];
        for &i in &head_vertices {
            in_head[i] = true;
        }
        let head_triangles: Vec<[u32; 3]> =
            triangles.iter().copied().filter(|t| t.iter().all(|&i| in_head[i as usize])).collect();
        let above = move |p: V| p.y > eyes.y + SCALP_ABOVE_EYES_M && (p.xz() - eyes.xz()).norm() < 0.3;
        let scalp_skin: Vec<usize> = (0..scan.weld.count)
            .map(|w| scan.weld.representative[w])
            .filter(|&r| seg.vertex_label[r] == Label::Skin && seen[r] && above(v(scan.positions[r])))
            .collect();
        let scalp_skin: Vec<usize> = scalp_skin.iter().step_by((scalp_skin.len() / 2000).max(1)).copied().collect();
        let eye_vertices: Vec<usize> = crate::fit::face_landmark_sets(prior)
            .into_iter()
            .filter(|(n, _)| *n == "eye_l" || *n == "eye_r")
            .flat_map(|(_, s)| s)
            .collect();
        move |posed: &[Vec3]| -> (f64, f64) {
            // The head moved onto the marked eyes, as the outlines are (the
            // head's alignment places it there later; before it a cartoon's
            // MakeHuman head sits centimetres off its scan).
            let shift = if eye_vertices.is_empty() {
                V::zeros()
            } else {
                eyes - eye_vertices.iter().fold(V::zeros(), |s, &i| s + v(posed[i])) / eye_vertices.len() as f64
            };
            let posed: Vec<Vec3> = posed.iter().map(|p| a(v(*p) + shift)).collect();
            let posed = &posed[..];
            let scalp: Vec<usize> = head_vertices.iter().copied().filter(|&i| above(v(posed[i]))).collect();
            let out: Vec<f64> = scalp
                .par_iter()
                .map(|&i| {
                    scan.query.closest_point(posed[i], SCALP_REACH_M, None).map_or(0.0, |h| {
                        if seg.triangle_label[h.source[1]] == Label::Skin && !scan.triangles[h.source[1]].iter().any(|&k|seen[k as usize]) {return 0.0;}
                        (v(posed[i]) - v(h.point)).dot(&v(h.normal)).clamp(0.0, SCALP_REACH_M)
                    })
                })
                .collect();
             
            let out: Vec<f64> = out.iter().map(|o| o * o).collect();
            let out = (crate::geom::ordered_sum(&out) / out.len().max(1) as f64).sqrt();
            if scalp_skin.is_empty() {
                return (0.0, out);
            }
            let query = build_query(posed, &head_triangles);
            let on: Vec<f64> = scalp_skin
                .par_iter()
                .map(|&s| query.closest_point(scan.positions[s], SCALP_REACH_M, None).map_or(SCALP_REACH_M, |h| h.distance_m))
                .collect();
            (crate::geom::ordered_sum(&on) / on.len() as f64, out)
        }
    };
    let head_report = if let Some(controls) = fallback_controls.filter(|_|head_keys.len()==3) {
        for (&k, value) in head_keys.iter().zip(controls) { x[k]=value; }
        json!({"applied":true,"method":"x110_head_controls","controls":controls,"solved_controls":head_keys})
    } else { match (eyes, head_keys.len() == 3) {
        (Some(e), true) => head_from_suit(
            prior,
            head_contours,
            e,
            &mut x,
            &head_keys,
            &|x: &[f64]| {
                let b = shaped(x);
                prior.skin_positions(body, &b, &arm_roots(&b).0)
            },
            &scan_on_skull(e),
        ),
        _ => Value::Null,
    } };
    // The controls solved to a measure stay; any other is searched.
    let fixed_head: Vec<usize> = head_report
        .get("solved_controls")
        .and_then(|s| s.as_array())
        .map_or(vec![], |a| a.iter().filter_map(|k| k.as_u64().map(|k| k as usize)).collect());
    let start = loss(&x);
    let mut best: f64;
    let mut evaluations = 1usize;
     
    let build: Vec<bool> = ps.iter().map(|p| p.name.starts_with("build-")).collect();
    let place: Vec<bool> = ps.iter().map(|p| p.name == "trunk-offset").collect();
    let mut stages: Vec<Vec<usize>> = vec![];
    if build.iter().any(|b| *b) {
        stages.push((0..ps.len()).filter(|&k| build[k]).collect());
    }
    // Local morphs, then the trunk's place, then the local morphs again:
    // placed against MakeHuman's average torso (no bust, no belly) the
    // edges' centre sat back of the marks', and the trunk slid forward to
    // its bound on six of eight characters.
    let local: Vec<usize> = (0..ps.len())
        .filter(|&k| !build[k] && !place[k] && Some(k) != foot_width_k && !fixed_head.contains(&k))
        .collect();
    stages.push(local.clone());
    if place.iter().any(|b| *b) {
        stages.push((0..ps.len()).filter(|&k| place[k]).collect());
        stages.push(local);
    }
    let torso_param: Vec<bool> = ps
        .iter()
        .map(|p| {
            [
                "bust-circ",
                "torso-scale-depth",
                "torso-scale-horiz",
                "waist-circ",
                "hips-circ",
                "stomach-pregnant",
            ]
            .contains(&p.name.as_str())
        })
        .collect();
    for stage in &stages {
        placing_now.set(stage.iter().any(|&k| place[k]));
        best = loss(&x).0;
        evaluations += 1;
        let mut step = 0.5;
        for _round in 0..4 {
            for &k in stage {
                // A stylised torso may go half again past MakeHuman's
                // slider ends (the flame's chest stopped at the torso
                // morphs' limit, 15 cm inside its vest).
                let name = ps[k].name.as_str();
                let reach = match limits {
                     
                    Some(_) if name.contains("neck") => 1.0,
                    Some((_, _, region)) if name.starts_with("region-") => region,
                    Some((torso, other, _)) if BODY_CONTROLS.contains(&name) => {
                        let r = if torso_param[k] { torso } else { other };
                        if name == "torso-scale-horiz" && r > EXTREME_TORSO_WIDTH { EXTREME_TORSO_WIDTH } else { r }
                    }
                    None if stylized && torso_param[k] => STYLIZED_TORSO_REACH,
                    _ => 1.0,
                };
                let lo = if ps[k].minus.is_some() { -1.0 } else { 0.0 };
                let hi = if ps[k].plus.is_some() { reach } else { 0.0 };
                 
                let mut moves = 0;
                'line: while moves < 8 {
                    for dir in [1.0, -1.0] {
                        let mut trial = x.clone();
                        trial[k] = (x[k] + dir * step).clamp(lo, hi);
                        if trial[k] == x[k] {
                            continue;
                        }
                        let l = loss(&trial).0;
                        evaluations += 1;
                        if l < best {
                            best = l;
                            x = trial;
                            moves += 1;
                            continue 'line;
                        }
                    }
                    break;
                }
            }
            step *= 0.5;
        }
    }
    let end = loss(&x);
    // Per-parameter sensitivity at the result (CHARLOOM_SHAPE_SENSITIVITY=1):
    // loss terms with the parameter moved by ±0.25.
    let sensitivity: Option<Value> = std::env::var_os("CHARLOOM_SHAPE_SENSITIVITY").map(|_| {
        ps.iter()
            .enumerate()
            .map(|(k, p)| {
                let row: Vec<Value> = [-0.25, 0.25]
                    .iter()
                    .map(|d| {
                        let mut t = x.clone();
                        t[k] += d;
                        let l = loss(&t);
                        json!({"total": l.0 - end.0, "skin": 5.0 * (l.1 - end.1), "envelope": ENVELOPE_WEIGHT * (l.2 - end.2), "fill": FILL_WEIGHT * (l.3 - end.3)})
                    })
                    .collect();
                (p.name.clone(), json!(row))
            })
            .collect::<serde_json::Map<String, Value>>()
            .into()
    });
    let outline_report: Vec<Value> = {
        let report_for = |x: &[f64]| {
            let b = shaped(x);
            outline_edges(&prior.skin_positions(body, &b, &arm_roots(&b).0))
        };
        let (before, after) = (report_for(&vec![0.0; ps.len()]), report_for(&x));
        outlines
            .iter()
            .zip(&outline_members)
            .zip(before.iter().zip(&after))
            .filter_map(|((m, frame), (b, e))| {
                let (members, _, _, _, lo, hi) = frame.as_ref()?;
                let mm = |x: f64| (x * 1000.0).round();
                Some(json!({
                    "station": m.station, "kind": m.kind.to_string(), "view": m.view,
                    "marked_mm": [mm(*lo), mm(*hi)],
                    "before_mm": b.map(|b| [mm(b.0), mm(b.1)]),
                    "after_mm": e.map(|e| [mm(e.0), mm(e.1)]),
                    "members": members.len(),
                }))
            })
            .collect()
    };
    let bind = shaped(&x);
    // The arms stay where the pose put them on the scan: their roots move
    // with the shaped torso (a bounded pose refit of the arm chains).
    let (refit, root_shift) = arm_roots(&bind);
    *worlds = refit;
    let rest = prior.skin_positions(body, &bind, &prior.rest_world.clone());
    prior.meshes[body].bind = bind;
    prior.meshes[body].prim.positions = rest.iter().map(|p| a(v(*p))).collect();
    let weights: serde_json::Map<String, Value> = ps
        .iter()
        .zip(&x)
        .filter(|(_, w)| **w != 0.0)
        .map(|(p, w)| (p.name.clone(), json!((w * 1000.0).round() / 1000.0)))
        .collect();
    Ok(json!({
        "applied": true,
        "arm_root_shift_m": root_shift,
        "parameters": ps.len(),
        "skipped": skip,
        "evaluations": evaluations,
        "weights": weights,
        "loss": {"before": start.0, "after": end.0},
        "head_from_suit": head_report,
        "outline_size_term": {"weight": OUTLINE_SIZE_WEIGHT, "after": size_term(&prior.skin_positions(body, &shaped(&x), &arm_roots(&shaped(&x)).0))},
        "skin_term": {"before": start.1, "after": end.1},
        "envelope_term": {"before": start.2, "after": end.2},
        "fill_term": {"before": start.3, "after": end.3, "samples": fills.len()},
        "outlines": outline_report,
        "sensitivity": sensitivity,
    }))
}


/// Skull outline kept from the brow up: this share of the eyes-to-crown
/// height above the eyes (below it the front view's outline is ears,
/// cheeks and jaw, the scan's face).
const SKULL_FROM: f64 = 0.3;
/// The side view's back of the head is kept down to this share below the
/// eyes (about the ears' bottom; under it the nape is the neck's).
const BACK_TO: f64 = 0.3;
/// Half-width of the head's column, metres (clear of a raised hand).
const HEAD_COLUMN_M: f64 = 0.25;
/// Nearest-point distances are capped here, metres (a stray part of the
/// other outline does not dominate).
const SKULL_CAP_M: f64 = 0.10;
/// Symmetric reach for linear head-scale targets. A compact GPT skull
/// must be able to select a depth below -1, just as a large skull can
/// select a positive control above 1.
const HEAD_SCALE_REACH: f64 = 2.5;
 const SCALP_ABOVE_EYES_M: f64 = 0.02;
const SCALP_REACH_M: f64 = 0.05;
const SCAN_SKULL_WEIGHT: f64 = 10.0;

/// One view's skull outline: the bald suit's curve samples above the brow
/// (and the back of the head in the side view), in pixels.
struct SkullView {
    camera: crate::guide::Camera,
    eye: [f64; 2],
    h_px: f64,
    centre: f64,
    // Side view: +1 when the face is towards +px of the head's centre,
    // -1 the other way; 0 on the front view (above the brow only).
    face_sign: f64,
    suit: Vec<[f64; 2]>,
}

impl SkullView {
    fn keep(&self, p: [f64; 2]) -> bool {
        let up = (self.eye[1] - p[1]) / self.h_px.max(1e-6);
        if (p[0] - self.centre).abs() * self.camera.meters_per_pixel > HEAD_COLUMN_M {
            return false;
        }
        up > SKULL_FROM || (self.face_sign != 0.0 && (p[0] - self.centre) * self.face_sign < 0.0 && up > -BACK_TO)
    }

    /// The body's outline points: the outer boundary of its projected
    /// head (`triangles`, rasterized at the view's pixels, holes filled),
    /// moved so its eyes sit on the marked eyes.
    fn body(&self, p: &[Vec3], triangles: &[[u32; 3]], eye_vertices: &[usize]) -> Vec<[f64; 2]> {
        let mut eye = [0.0, 0.0];
        for &i in eye_vertices {
            let (x, y, _) = self.camera.project(p[i]);
            eye = [eye[0] + x, eye[1] + y];
        }
        let n = eye_vertices.len().max(1) as f64;
        let shift = if eye_vertices.is_empty() {
            [0.0, 0.0]
        } else {
            [self.eye[0] - eye[0] / n, self.eye[1] - eye[1] / n]
        };
        let proj = |i: u32| {
            let (x, y, _) = self.camera.project(p[i as usize]);
            [x + shift[0], y + shift[1]]
        };
        let tris: Vec<[[f64; 2]; 3]> = triangles.iter().map(|t| t.map(proj)).collect();
        let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for q in tris.iter().flatten() {
            (x0, y0, x1, y1) = (x0.min(q[0]), y0.min(q[1]), x1.max(q[0]), y1.max(q[1]));
        }
        if !x0.is_finite() {
            return vec![];
        }
        // One pixel of background all round.
        let (ox, oy) = (x0.floor() as i64 - 1, y0.floor() as i64 - 1);
        let (w, h) = ((x1.ceil() as i64 - ox + 2) as usize, (y1.ceil() as i64 - oy + 2) as usize);
        let mut mask = vec![0u8; w * h];
        for [a, b, c] in &tris {
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            if area.abs() < 1e-9 {
                continue;
            }
            let lx = (a[0].min(b[0]).min(c[0]).floor() as i64 - ox).max(0) as usize;
            let hx = ((a[0].max(b[0]).max(c[0]).ceil() as i64 - ox) as usize).min(w - 1);
            let ly = (a[1].min(b[1]).min(c[1]).floor() as i64 - oy).max(0) as usize;
            let hy = ((a[1].max(b[1]).max(c[1]).ceil() as i64 - oy) as usize).min(h - 1);
            for py in ly..=hy {
                for px in lx..=hx {
                    let q = [(px as i64 + ox) as f64 + 0.5, (py as i64 + oy) as f64 + 0.5];
                    let e = |u: &[f64; 2], v: &[f64; 2]| (v[0] - u[0]) * (q[1] - u[1]) - (v[1] - u[1]) * (q[0] - u[0]);
                    let (e0, e1, e2) = (e(a, b), e(b, c), e(c, a));
                    if (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0) {
                        mask[py * w + px] = 1;
                    }
                }
            }
        }
        // Outside: background reached from the border (2).
        let mut stack = vec![0usize];
        mask[0] = 2;
        while let Some(k) = stack.pop() {
            let (px, py) = (k % w, k / w);
            let mut push = |k2: usize| {
                if mask[k2] == 0 {
                    mask[k2] = 2;
                    stack.push(k2);
                }
            };
            if px > 0 { push(k - 1); }
            if px + 1 < w { push(k + 1); }
            if py > 0 { push(k - w); }
            if py + 1 < h { push(k + w); }
        }
        let mut out = vec![];
        for py in 1..h - 1 {
            for px in 1..w - 1 {
                let k = py * w + px;
                if mask[k] != 2 && [k - 1, k + 1, k - w, k + w].iter().any(|&k2| mask[k2] == 2) {
                    let q = [(px as i64 + ox) as f64 + 0.5, (py as i64 + oy) as f64 + 0.5];
                    if self.keep(q) {
                        out.push(q);
                    }
                }
            }
        }
        out
    }

    /// Mean capped nearest distance both ways (metres), and the body
    /// points' p95 to the curve.
    fn distance(&self, body: &[[f64; 2]]) -> (f64, f64) {
        let cap = SKULL_CAP_M / self.camera.meters_per_pixel;
        if body.is_empty() || self.suit.is_empty() {
            return (SKULL_CAP_M * 2.0, SKULL_CAP_M);
        }
        let near = |p: &[f64; 2], set: &[[f64; 2]]| {
            set.iter()
                .map(|q| ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt())
                .fold(f64::INFINITY, f64::min)
                .min(cap)
        };
        // Registered cameras share one physical head placement. Independent
        // per-view slides would hide a wrong rest shape from this objective.
        let mut to_suit: Vec<f64> = body.iter().map(|p| near(p, &self.suit)).collect();
        let reverse: Vec<f64> = self.suit.iter().map(|q| near(q, body)).collect();
        let to_body = reverse.iter().sum::<f64>() / reverse.len() as f64;
        let mean = to_suit.iter().sum::<f64>() / to_suit.len() as f64;
        to_suit.extend(reverse);
        to_suit.sort_by(f64::total_cmp);
        let p95 = to_suit[((to_suit.len() - 1) as f64 * 0.95) as usize];
        let m = self.camera.meters_per_pixel;
        ((mean + to_body) * m, p95 * m)
    }
}

 fn head_from_suit(
    prior: &Prior,
    contours: &[crate::guide::SuitContour],
    eyes: V,
    x: &mut [f64],
    keys: &[usize],
    posed: &dyn Fn(&[f64]) -> Vec<Vec3>,
    // (scan scalp skin to the body, body out of the scan), mean metres.
    scan_cost: &dyn Fn(&[Vec3]) -> (f64, f64),
) -> Value {
    let front = contours.iter().find(|c| v(c.camera.forward).z.abs() > 0.9);
    let side = contours.iter().find(|c| v(c.camera.forward).x.abs() > 0.9);
    let (Some(front), Some(_side)) = (front, side) else {
        return json!({"applied": false, "reason": "no front and side outline"});
    };
    // Crown: the front outline's top within a band about the eyes' column.
    let (ex, ey, _) = front.camera.project(a(eyes));
    let mpp = front.camera.meters_per_pixel;
    let Some(crown_py) = front
        .curves
        .iter()
        .flatten()
        .filter(|p| (p[0] - ex).abs() < 0.03 / mpp && p[1] < ey)
        .map(|p| p[1])
        .min_by(f64::total_cmp)
    else {
        return json!({"applied": false, "reason": "no crown"});
    };
    let crown_m = (ey - crown_py) * mpp;
    let view = |c: &crate::guide::SuitContour, side_view: bool| {
        let (ex, ey, _) = c.camera.project(a(eyes));
        let h_px = crown_m / c.camera.meters_per_pixel;
        // Side view: the head's centre between the face and the back of
        // the skull, from the run of the outline holding the eyes' column
        // halfway to the crown.
        let (centre, face_sign) = if side_view {
            let py = ey - 0.5 * h_px;
            let mut xs: Vec<f64> = vec![];
            for curve in &c.curves {
                for w in curve.windows(2) {
                    let (p, q) = (w[0], w[1]);
                    if (p[1] - py) * (q[1] - py) > 0.0 || (p[1] - q[1]).abs() < 1e-9 {
                        continue;
                    }
                    xs.push(p[0] + (q[0] - p[0]) * (py - p[1]) / (q[1] - p[1]));
                }
            }
            xs.sort_by(f64::total_cmp);
            let inside = |px: f64| xs.iter().filter(|&&x0| x0 > px).count() % 2 == 1;
            let gap = |r: &(f64, f64)| if ex < r.0 { r.0 - ex } else if ex > r.1 { ex - r.1 } else { 0.0 };
            match xs
                .windows(2)
                .map(|r| (r[0], r[1]))
                .filter(|r| r.1 - r.0 > 1.0 && inside(0.5 * (r.0 + r.1)))
                .min_by(|a, b| gap(a).total_cmp(&gap(b)))
            {
                Some((lo, hi)) => {
                    let mid = 0.5 * (lo + hi);
                    (mid, if ex >= mid { 1.0 } else { -1.0 })
                }
                None => (ex, 0.0),
            }
        } else {
            (ex, 0.0)
        };
        let mut sv = SkullView { camera: c.camera.clone(), eye: [ex, ey], h_px, centre, face_sign, suit: vec![] };
        sv.suit = c.curves.iter().flatten().copied().filter(|p| sv.keep(*p)).collect();
        sv
    };
    let views: Vec<SkullView> = contours.iter()
        .map(|c| view(c, v(c.camera.forward).x.abs() > 0.25))
        .filter(|v| v.suit.len() >= 20)
        .collect();
    if views.is_empty() {
        return json!({"applied": false, "reason": "no skull outline", "crown_m": crown_m});
    }
    let head = prior.joint("head");
    let head_vertices: Vec<usize> = (0..prior.body().bind.len())
        .filter(|&i| {
            let j = prior.dominant_joint(prior.body, i);
            j == head || prior.is_ancestor(head, j) && prior.region[j] == Some(Region::Head)
        })
        .collect();
    let in_head = {
        let mut m = vec![false; prior.body().bind.len()];
        for &i in &head_vertices {
            m[i] = true;
        }
        m
    };
    let head_triangles: Vec<[u32; 3]> = prior
        .body()
        .prim
        .triangles
        .iter()
        .copied()
        .filter(|t| t.iter().all(|&i| in_head[i as usize]))
        .collect();
    let eye_vertices: Vec<usize> = crate::fit::face_landmark_sets(prior)
        .into_iter()
        .filter(|(n, _)| *n == "eye_l" || *n == "eye_r")
        .flat_map(|(_, s)| s)
        .collect();
    let per_view = |x: &[f64]| -> (Vec<(f64, f64)>, (f64, f64)) {
        let p = posed(x);
        (views.iter().map(|vw| vw.distance(&vw.body(&p, &head_triangles, &eye_vertices))).collect(), scan_cost(&p))
    };
    let objective = |x: &[f64]| {
        let (d, (on, out)) = per_view(x);
        d.iter().map(|d| d.0 + 0.5*d.1).sum::<f64>() + on + SCAN_SKULL_WEIGHT * out
    };
    let (before, scan_before) = per_view(x);
    let mut best = objective(x);
    let mut evaluations = 1;
    let mut step = 0.5;
    for _round in 0..5 {
        for &k in keys {
            let mut moves = 0;
            'line: while moves < 8 {
                for dir in [1.0, -1.0] {
                    let mut trial = x.to_vec();
                    trial[k] = (x[k] + dir * step).clamp(-HEAD_SCALE_REACH, HEAD_SCALE_REACH);
                    if trial[k] == x[k] {
                        continue;
                    }
                    let l = objective(&trial);
                    evaluations += 1;
                    if l < best {
                        best = l;
                        x.copy_from_slice(&trial);
                        moves += 1;
                        continue 'line;
                    }
                }
                break;
            }
        }
        step *= 0.5;
    }
    let (after, scan_after) = per_view(x);
    let depth_probe: Vec<Value> = [-2.5, -2.0, -1.5, -1.25, -1.0, -0.75, -0.5].iter().map(|&depth| {
        let mut trial=x.to_vec();trial[keys[2]]=depth;
        let (d,(_,outside))=per_view(&trial);
        json!({"control":depth,"objective_mm":(d.iter().map(|v|v.0).sum::<f64>()+SCAN_SKULL_WEIGHT*outside)*1000.0,
            "view_chamfer_mm":d.iter().map(|v|v.0*1000.0).collect::<Vec<_>>()})
    }).collect();
    if let Some(path) = std::env::var_os("CHARLOOM_DUMP_SKULL") {
        let p = posed(x);
        let dump: Vec<Value> = views
            .iter()
            .map(|vw| json!({"suit": vw.suit, "body": vw.body(&p, &head_triangles, &eye_vertices), "eye": vw.eye, "centre": vw.centre, "face_sign": vw.face_sign, "h_px": vw.h_px}))
            .collect();
        let _ = std::fs::write(path, serde_json::to_string(&dump).unwrap_or_default());
    }
    let mm = |v: f64| v * 1000.0;
    let report = |d: &[(f64, f64)]| -> Vec<Value> {
        d.iter().map(|(m, p95)| json!({"chamfer_mm": mm(*m), "symmetric_skull_p95_mm": mm(*p95)})).collect()
    };
    json!({
        "applied": true,
        "crown_mm": mm(crown_m),
        "control_bounds": [-HEAD_SCALE_REACH, HEAD_SCALE_REACH],
        "depth_probe": depth_probe,
        "views": views.iter().map(|vw| json!({"view":vw.camera.id,"suit_points": vw.suit.len(), "back_of_head": vw.face_sign != 0.0})).collect::<Vec<_>>(),
        "before": report(&before),
        "after": report(&after),
        "scan_mm": {"scalp_skin_before": mm(scan_before.0), "scalp_skin_after": mm(scan_after.0), "outside_rms_before": mm(scan_before.1), "outside_rms_after": mm(scan_after.1)},
        "evaluations": evaluations,
        "controls": keys.iter().map(|&k| x[k]).collect::<Vec<f64>>(),
        "past_makehuman": keys.iter().filter(|&&k| x[k].abs() > 1.0).count(),
        "at_bound": keys.iter().filter(|&&k| x[k].abs() > HEAD_SCALE_REACH - 0.02).count(),
        "solved_controls": keys,
    })
}
