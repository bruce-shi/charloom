 use crate::geom::{V, percentiles, proper_rotation};
use crate::prior::Region;
use charloom_core::Influence;
use nalgebra::Matrix3;
use rayon::prelude::*;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, HashMap};

/// Mirror partner of every vertex of a symmetric rest surface.
pub struct Mirror {
    /// Partner index (the vertex itself on the midline).
    pub partner: Vec<u32>,
    /// Pairs that matched cleanly (involutive, within tolerance).
    pub valid: Vec<bool>,
    /// The mirror plane is x = plane.
    pub plane: f64,
    pub report: Value,
}

impl Mirror {
    /// Matches each vertex to the nearest mirrored rest position.
    /// `tolerance`: largest accepted match distance.
    pub fn new(rest: &[V], tolerance: f64) -> Self {
        let n = rest.len();
        let (lo, hi) = rest.iter().fold(
            (V::repeat(f64::INFINITY), V::repeat(f64::NEG_INFINITY)),
            |(l, h), p| (l.inf(p), h.sup(p)),
        );
        let plane = 0.5 * (lo.x + hi.x);
        let cell = ((hi - lo).norm() / 300.0).max(1e-6);
        let key = |p: &V| [0, 1, 2].map(|k| (p[k] / cell).floor() as i64);
        let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
        for (i, p) in rest.iter().enumerate() {
            grid.entry(key(p)).or_default().push(i as u32);
        }
        let matched: Vec<(u32, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let q = mirror_point(rest[i], plane);
                let k = key(&q);
                let mut best = (i as u32, f64::INFINITY);
                // A neighbour within `r - 1` cells is certainly the nearest
                // once the ring of radius r has been searched.
                for r in 1i64..=4 {
                    for dx in -r..=r {
                        for dy in -r..=r {
                            for dz in -r..=r {
                                if dx.abs().max(dy.abs()).max(dz.abs()) < r && r > 1 {
                                    continue;
                                }
                                let Some(cands) = grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz])
                                else {
                                    continue;
                                };
                                for &j in cands {
                                    let d = (rest[j as usize] - q).norm();
                                    if d < best.1 || (d == best.1 && j < best.0) {
                                        best = (j, d);
                                    }
                                }
                            }
                        }
                    }
                    if best.1 <= cell * (r - 1) as f64 + 1e-12 && best.1.is_finite() {
                        break;
                    }
                }
                best
            })
            .collect();
        let partner: Vec<u32> = matched.iter().map(|m| m.0).collect();
        let valid: Vec<bool> = (0..n)
            .map(|i| matched[i].1 <= tolerance && partner[partner[i] as usize] as usize == i)
            .collect();
        let involutive = (0..n)
            .filter(|&i| partner[partner[i] as usize] as usize == i)
            .count();
        let mut seen = vec![0u32; n];
        for &p in &partner {
            seen[p as usize] += 1;
        }
        let distances: Vec<f64> = matched.iter().map(|m| m.1).collect();
        let report = json!({
            "vertices": n,
            "plane_x_m": plane,
            "midline": (0..n).filter(|&i| partner[i] as usize == i).count(),
            "involutive": involutive,
            "injective": seen.iter().all(|&c| c == 1),
            "valid": valid.iter().filter(|v| **v).count(),
            "bijection": involutive == n && seen.iter().all(|&c| c == 1),
            "match_distance_m": percentiles(distances),
        });
        Self {
            partner,
            valid,
            plane,
            report,
        }
    }

    /// Mirror of a vector (direction: plane-free) or a point.
    pub fn point(&self, p: V) -> V {
        mirror_point(p, self.plane)
    }
}

fn mirror_point(p: V, plane: f64) -> V {
    V::new(2.0 * plane - p.x, p.y, p.z)
}

/// Rigid frame (rotation, translation) of every vertex that carries the rest
/// surface onto `x`: one weighted Kabsch fit per joint over the vertices it
/// drives, blended with the skin weights. Joints driving too little surface
/// take their parent's frame.
pub fn vertex_frames(
    influences: &[&[Influence]],
    parents: &[Option<usize>],
    rest: &[V],
    x: &[V],
) -> Vec<(Matrix3<f64>, V)> {
    let joints = parents.len();
    let mut wsum = vec![0.0f64; joints];
    let mut ra = vec![V::zeros(); joints];
    let mut xa = vec![V::zeros(); joints];
    for (i, row) in influences.iter().enumerate() {
        for w in row.iter() {
            wsum[w.joint] += w.weight;
            ra[w.joint] += rest[i] * w.weight;
            xa[w.joint] += x[i] * w.weight;
        }
    }
    let cr: Vec<V> = (0..joints).map(|j| ra[j] / wsum[j].max(1e-12)).collect();
    let cx: Vec<V> = (0..joints).map(|j| xa[j] / wsum[j].max(1e-12)).collect();
    let mut cov = vec![Matrix3::zeros(); joints];
    for (i, row) in influences.iter().enumerate() {
        for w in row.iter() {
            cov[w.joint] += (x[i] - cx[w.joint]) * (rest[i] - cr[w.joint]).transpose() * w.weight;
        }
    }
    let own: Vec<Option<(Matrix3<f64>, V)>> = (0..joints)
        .map(|j| {
            (wsum[j] >= 8.0).then(|| {
                let r = proper_rotation(cov[j]);
                (r, cx[j] - r * cr[j])
            })
        })
        .collect();
    let frame_of = |mut j: usize| loop {
        if let Some(f) = own[j] {
            return f;
        }
        match parents[j] {
            Some(p) => j = p,
            None => return (Matrix3::identity(), V::zeros()),
        }
    };
    let frames: Vec<(Matrix3<f64>, V)> = (0..joints).map(frame_of).collect();
    influences
        .par_iter()
        .map(|row| {
            let total: f64 = row.iter().map(|w| w.weight).sum::<f64>().max(1e-12);
            row.iter()
                .fold((Matrix3::zeros(), V::zeros()), |(m, t), w| {
                    let (r, b) = frames[w.joint];
                    (m + r * (w.weight / total), t + b * (w.weight / total))
                })
        })
        .map(|(m, t)| {
            if m.determinant().abs() < 1e-6 {
                (Matrix3::identity(), t)
            } else {
                (m, t)
            }
        })
        .collect()
}

/// `x` carried back into the rest frame by `frames`.
pub fn unpose(frames: &[(Matrix3<f64>, V)], x: &[V]) -> Vec<V> {
    frames
        .par_iter()
        .zip(x)
        .map(|((m, t), p)| m.try_inverse().unwrap_or_else(Matrix3::identity) * (p - t))
        .collect()
}

/// Per-vertex asymmetry in the rest frame: the vertex minus its partner's
/// mirrored position (zero for a symmetric body).
pub fn difference(mirror: &Mirror, u: &[V]) -> Vec<Option<V>> {
    (0..u.len())
        .map(|i| {
            mirror
                .valid
                .get(i)
                .copied()
                .unwrap_or(false)
                .then(|| u[i] - mirror.point(u[mirror.partner[i] as usize]))
        })
        .collect()
}

/// Where a vertex sits if it mirrors its partner: the partner's rest-frame
/// position mirrored and carried by this vertex's frame.
pub fn mirrored_target(
    mirror: &Mirror,
    frames: &[(Matrix3<f64>, V)],
    u: &[V],
    i: usize,
) -> Option<V> {
    if !mirror.valid[i] {
        return None;
    }
    let (m, t) = frames[i];
    Some(m * mirror.point(u[mirror.partner[i] as usize]) + t)
}

/// What constrains a vertex in the fit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Evidence {
    /// Visible scan skin (an equality target).
    Skin,
    /// Garment or hair: only a one-sided envelope.
    Cloth,
    /// No data: the shape regulariser alone.
    None,
}

/// Asymmetry report: each mirror pair once, classed by the evidence on its
/// two sides (`visible`: skin on both, `mixed`: skin on one, `hidden`: on
/// neither) and by region. `noise` is the part of the asymmetry field that
/// is not smooth over the surface (per-vertex fit noise).
pub fn report(
    mirror: &Mirror,
    u: &[V],
    region: &[Region],
    evidence: &[Evidence],
    neighbours: &[Vec<u32>],
) -> Value {
    let diff = difference(mirror, u);
    let pairs: Vec<usize> = (0..u.len())
        .filter(|&i| diff[i].is_some() && i <= mirror.partner[i] as usize)
        .collect();
    let class = |i: usize| {
        let skin = [i, mirror.partner[i] as usize]
            .iter()
            .filter(|&&k| evidence[k] == Evidence::Skin)
            .count();
        ["hidden", "mixed", "visible"][skin]
    };
    let mut by_class: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut by_region: BTreeMap<String, BTreeMap<&str, Vec<f64>>> = BTreeMap::new();
    for &i in &pairs {
        let d = diff[i].unwrap().norm();
        by_class.entry(class(i)).or_default().push(d);
        let r = by_region.entry(format!("{:?}", region[i])).or_default();
        r.entry(class(i)).or_default().push(d);
        r.entry("all").or_default().push(d);
    }
    // Smooth part: the field averaged over two rings.
    let mut smooth: Vec<V> = diff.iter().map(|d| d.unwrap_or_else(V::zeros)).collect();
    for _ in 0..4 {
        smooth = (0..smooth.len())
            .map(|i| {
                let ring = neighbours[i]
                    .iter()
                    .filter(|&&j| diff[j as usize].is_some())
                    .map(|&j| smooth[j as usize]);
                let (s, k) = ring.fold((smooth[i], 1.0), |(s, k), q| (s + q, k + 1.0));
                s / k
            })
            .collect();
    }
    let noise: Vec<f64> = pairs
        .iter()
        .map(|&i| (diff[i].unwrap() - smooth[i]).norm())
        .collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let short = |v: Vec<f64>| -> Value {
        let m = mean(&v);
        let mut p = percentiles(v);
        if let Some(o) = p.as_object_mut() {
            o.retain(|k, _| ["count", "p50", "p95", "max"].contains(&k.as_str()));
            o.insert("mean".into(), json!(m));
        }
        p
    };
    let all: Vec<f64> = pairs.iter().map(|&i| diff[i].unwrap().norm()).collect();
    json!({
        "all": short(all),
        "by_class": by_class.into_iter().map(|(k, v)| (k.to_string(), short(v))).collect::<Map<_, _>>(),
        "by_region": by_region
            .into_iter()
            .map(|(r, m)| (r, Value::Object(m.into_iter().map(|(k, v)| (k.to_string(), short(v))).collect())))
            .collect::<Map<_, _>>(),
        "noise": short(noise),
    })
}

/// Bone length left vs right for every `.L`/`.R` joint pair: relative
/// difference |l − r| / mean of the distance to the parent joint.
pub fn bone_lengths(names: &[&str], parents: &[Option<usize>], origins: &[V]) -> Value {
    let index: HashMap<&str, usize> = names.iter().enumerate().map(|(i, n)| (*n, i)).collect();
    let mut out: Vec<(f64, &str, f64, f64)> = vec![];
    for (j, name) in names.iter().enumerate() {
        let Some(base) = name.strip_suffix(".L") else {
            continue;
        };
        let Some(&k) = index.get(format!("{base}.R").as_str()) else {
            continue;
        };
        let (Some(pj), Some(pk)) = (parents[j], parents[k]) else {
            continue;
        };
        let (l, r) = (
            (origins[j] - origins[pj]).norm(),
            (origins[k] - origins[pk]).norm(),
        );
        if l + r < 1e-4 {
            continue;
        }
        out.push(((l - r).abs() / (0.5 * (l + r)), base, l, r));
    }
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    let main = [
        "clavicle",
        "upperarm01",
        "lowerarm01",
        "wrist",
        "upperleg01",
        "lowerleg01",
        "foot",
    ];
    json!({
        "max_relative": out.first().map_or(0.0, |o| o.0),
        "worst": out.first().map(|o| o.1),
        "limbs": out
            .iter()
            .filter(|o| main.contains(&o.1))
            .map(|o| (o.1.to_string(), json!({"relative": o.0, "l_m": o.2, "r_m": o.3})))
            .collect::<Map<_, _>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_body() -> Vec<V> {
        // A symmetric point set with a midline column.
        let mut pts = vec![];
        for y in 0..5 {
            for x in -3..=3 {
                pts.push(V::new(
                    x as f64 * 0.1 + 0.02,
                    y as f64 * 0.1,
                    (x * x) as f64 * 0.01,
                ));
            }
        }
        pts
    }

    #[test]
    fn mirror_map_is_a_clean_bijection() {
        let pts = grid_body();
        let m = Mirror::new(&pts, 1e-6);
        assert!((m.plane - 0.02).abs() < 1e-12);
        assert_eq!(m.report["bijection"], json!(true));
        assert_eq!(m.report["midline"], json!(5));
        for (i, &p) in m.partner.iter().enumerate() {
            assert!(m.valid[i]);
            assert!((m.point(pts[p as usize]) - pts[i]).norm() < 1e-12);
        }
        // A perturbed vertex beyond the tolerance is not trusted.
        let mut bent = pts.clone();
        bent[0].y += 0.01;
        let m = Mirror::new(&bent, 1e-3);
        assert!(!m.valid[0]);
    }

    #[test]
    fn mirrored_target_restores_symmetry_under_a_rigid_pose() {
        let rest = grid_body();
        let mirror = Mirror::new(&rest, 1e-6);
        // One joint drives everything; the pose is a rotation + shift.
        let infl: Vec<Vec<Influence>> = rest
            .iter()
            .map(|_| {
                vec![Influence {
                    joint: 0,
                    weight: 1.0,
                }]
            })
            .collect();
        let rows: Vec<&[Influence]> = infl.iter().map(Vec::as_slice).collect();
        let r = nalgebra::Rotation3::from_euler_angles(0.3, -0.2, 0.5).into_inner();
        let t = V::new(0.4, -0.1, 0.2);
        let mut x: Vec<V> = rest.iter().map(|p| r * p + t).collect();
        // A bump on one side only (index 3 is on the midline).
        let bumped = 1;
        x[bumped] += r * V::new(0.0, 0.0, 0.002);
        let frames = vertex_frames(&rows, &[None], &rest, &x);
        let u = unpose(&frames, &x);
        let d = difference(&mirror, &u);
        // The frame fit includes the bump, so it absorbs a little of it
        // (35 vertices here; a real joint drives hundreds).
        let partner = mirror.partner[bumped] as usize;
        assert!((d[bumped].unwrap().norm() - 0.002).abs() < 2e-4);
        assert!((d[partner].unwrap().norm() - 0.002).abs() < 2e-4);
        // Moving the bumped vertex to its mirrored target removes the bump.
        let target = mirrored_target(&mirror, &frames, &u, bumped).unwrap();
        assert!((target - (r * rest[bumped] + t)).norm() < 2e-4);
        // Unbumped symmetric vertices are their own target.
        let other = 10;
        let target = mirrored_target(&mirror, &frames, &u, other).unwrap();
        assert!((target - x[other]).norm() < 2e-4);
    }

    #[test]
    fn bone_lengths_compare_left_and_right() {
        let names = ["root", "arm.L", "arm.R"];
        let parents = [None, Some(0), Some(0)];
        let origins = [V::zeros(), V::new(0.3, 0.0, 0.0), V::new(-0.33, 0.0, 0.0)];
        let r = bone_lengths(&names, &parents, &origins);
        assert!((r["max_relative"].as_f64().unwrap() - 0.03 / 0.315).abs() < 1e-9);
        assert_eq!(r["worst"], json!("arm"));
    }
}
