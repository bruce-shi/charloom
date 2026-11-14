 use super::guard;
use crate::geom::{V, a, proper_rotation, v};
use crate::scan::Scan;
use crate::segment::{Label, Segmentation};
use nalgebra::Matrix3;
use rayon::prelude::*;
use serde_json::{Value, json};

/// Below this share of toe vertices facing scan skin (and twice that of the
/// whole foot) a foot is in footwear.
const SHOD_SKIN_FRACTION: f64 = 0.25;
/// Reach (share of the height) at which a foot vertex without a facing
/// surface nearby still finds the inside of its shoe.
const SHOE_REACH: f64 = 0.1;
/// Gap a shod foot keeps to the inside of its shoe, share of the height
/// (1.2 cm at 1.75 m).
const FOOT_GAP: f64 = 0.007;

struct Foot {
    side: &'static str,
    /// Welded vertices with a target, and their blend weight.
    members: Vec<(usize, f64)>,
    vertices: usize,
    /// Vertices facing the scan within reach of a shoe's inside.
    covered: usize,
    skin_fraction: f64,
    toe_skin_fraction: f64,
    shod: bool,
    /// Last similarity: scale, rotation (vs. the posed prior) and how far
    /// it moved the foot's centroid.
    scale: f64,
    rotation: Matrix3<f64>,
    shift: V,
    /// Share of the foot outside the shoe (beyond the clearance) at the last
    /// update, before and after the similarity.
    outside_before: f64,
    outside_after: f64,
    /// A measured bare foot: MakeHuman's posed foot stretched per axis onto
    /// the scan's foot (`fit_bare`), its targets and fitted scales.
    fitted: Vec<(usize, V)>,
    shape: Option<Value>,
}

pub(super) struct Feet {
    /// Per welded vertex: weight of the foot similarity target in [0, 1]
    /// (the per-vertex data keeps `1 − weight`).
    pub weight: Vec<f64>,
    /// Bounded uniform scale range of the similarity.
    scale_range: (f64, f64),
    feet: Vec<Foot>,
}

impl Feet {
    /// `side[i]`: 0 (left) or 1 (right) for welded vertices of the foot
    /// region (foot and toe bones). `rest`: the posed prior, `shape_rest`
    /// MakeHuman's rest surface (for surface distances).
    #[allow(clippy::too_many_arguments)]
    pub fn detect(
        side: &[Option<u8>],
        toe: &[bool],
        rest: &[V],
        rest_normals: &[V],
        shape_rest: &[V],
        neighbours: &[Vec<u32>],
        scan: &Scan,
        seg: &Segmentation,
        height: f64,
        scale_range: (f64, f64),
        bare: [bool; 2],
    ) -> Self {
        let n = rest.len();
        // What each vertex faces on the scan: skin, something else, nothing.
        let facing: Vec<Option<bool>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let hit = scan.query.closest_point(
                    a(rest[i]),
                    height * 0.03,
                    Some((a(rest_normals[i]), 0.3)),
                )?;
                Some(seg.triangle_label[hit.source[1]] == Label::Skin)
            })
            .collect();
        // A foot deep inside an oversized (cartoon) shoe is beyond that reach
        // of the shoe's walls: still surrounded by the scan, just not facing
        // anything near it. Such vertices count as covered for the shoe test.
        let surrounded: Vec<bool> = (0..n)
            .into_par_iter()
            .map(|i| {
                facing[i].is_some()
                    || (side[i].is_some()
                        && scan
                            .query
                            .closest_point(
                                a(rest[i]),
                                height * SHOE_REACH,
                                Some((a(rest_normals[i]), 0.3)),
                            )
                            .is_some_and(|h| seg.triangle_label[h.source[1]] != Label::Skin))
            })
            .collect();
        let band = 0.012 * height;
        let mut weight = vec![0.0f64; n];
        let mut feet = vec![];
        for (s, name) in [(0u8, "left"), (1u8, "right")] {
            let member: Vec<bool> = side.iter().map(|x| *x == Some(s)).collect();
            let vertices = member.iter().filter(|m| **m).count();
            if vertices == 0 {
                continue;
            }
            let (mut skin, mut hits, mut covered) = (0usize, 0usize, 0usize);
            for i in (0..n).filter(|&i| member[i]) {
                covered += surrounded[i] as usize;
                if let Some(k) = facing[i] {
                    hits += 1;
                    skin += k as usize;
                }
            }
            let skin_fraction = skin as f64 / hits.max(1) as f64;
            // Toes decide: a low shoe leaves the ankle (MakeHuman's foot
            // region reaches up to it) next to visible leg skin, while open
            // toes are a sandal or a bare foot.
            let (mut toe_skin, mut toe_hits) = (0usize, 0usize);
            for i in (0..n).filter(|&i| member[i] && toe[i]) {
                if let Some(k) = facing[i] {
                    toe_hits += 1;
                    toe_skin += k as usize;
                }
            }
            let toe_skin_fraction = toe_skin as f64 / toe_hits.max(1) as f64;
             
            let rigid = scale_range.0 == scale_range.1;
            let shod = rigid
                || !bare[s as usize]
                && covered * 2 > vertices
                && toe_hits > 0
                && toe_skin_fraction < SHOD_SKIN_FRACTION
                && skin_fraction < 2.0 * SHOD_SKIN_FRACTION;
            let mut members = vec![];
            if shod {
                // Signed surface distance from the foot's boundary: positive
                // into the foot, negative up the leg.
                let boundary: Vec<bool> = (0..n)
                    .map(|i| member[i] && neighbours[i].iter().any(|&j| !member[j as usize]))
                    .collect();
                let d = guard::surface_distance(shape_rest, neighbours, &boundary, band * 2.0);
                let mut w: Vec<f64> = (0..n)
                    .map(|i| {
                        if !d[i].is_finite() {
                            return if member[i] { 1.0 } else { 0.0 };
                        }
                        let t = if member[i] { d[i] } else { -d[i] };
                        // Observed skin keeps its own fit.
                        let hidden = rigid || facing[i] != Some(true);
                        guard::smoothstep((t + band) / (2.0 * band)) * hidden as u8 as f64
                    })
                    .collect();
                // Soften the cut where skin and hidden vertices meet.
                for _ in 0..3 {
                    let next: Vec<f64> = (0..n)
                        .map(|i| {
                            if w[i] == 0.0 && neighbours[i].iter().all(|&j| w[j as usize] == 0.0) {
                                return 0.0;
                            }
                            let sum: f64 = neighbours[i].iter().map(|&j| w[j as usize]).sum();
                            (w[i] + sum) / (1 + neighbours[i].len()) as f64
                        })
                        .collect();
                    w = next;
                }
                for (i, &wi) in w.iter().enumerate() {
                    if wi > 1e-3 {
                        members.push((i, wi));
                        weight[i] = weight[i].max(wi);
                    }
                }
            }
            feet.push(Foot {
                side: name,
                members,
                vertices,
                covered,
                skin_fraction,
                toe_skin_fraction,
                shod,
                scale: 1.0,
                rotation: Matrix3::identity(),
                shift: V::zeros(),
                outside_before: 0.0,
                outside_after: 0.0,
                fitted: vec![],
                shape: None,
            });
        }
        // Measured bare feet: MakeHuman's foot stretched per axis onto the
        // scan's (`fit_bare`), not left at the pose's uniform size.
        if scale_range.0 == scale_range.1 {
            let region: Vec<Vec<usize>> = (0..2u8)
                .map(|s| (0..n).filter(|&i| side[i] == Some(s)).collect())
                .collect();
            for foot in feet.iter_mut() {
                let s = if foot.side == "left" { 0 } else { 1 };
                if !bare[s] || foot.members.is_empty() {
                    continue;
                }
                if let Some((fitted, shape)) =
                    fit_bare(&region[s], &region[1 - s], &foot.members, rest, rest_normals, toe, scan, height)
                {
                    foot.fitted = fitted;
                    foot.shape = Some(shape);
                }
            }
        }
        Self {
            weight,
            scale_range,
            feet,
        }
    }

    pub fn any(&self) -> bool {
        self.feet.iter().any(|f| f.shod)
    }

    /// Similarity targets for this iteration, as `(vertex, target)`: each
    /// shod foot's posed prior moved by the similarity that best fits the
    /// current foot with its vertices outside the shoe pushed back inside.
    #[allow(clippy::too_many_arguments)]
    pub fn targets(
        &mut self,
        rest: &[V],
        x: &[V],
        normals: &[V],
        scan: &Scan,
        seg: &Segmentation,
        clearance: f64,
        height: f64,
    ) -> Vec<(usize, V)> {
        // Inward push of a vertex past (or within the clearance of) the
        // same-facing surface over it; skin is not a bound.
        let push = |p: V, nrm: V| -> Option<V> {
            let hit = scan
                .query
                .closest_point(a(p), height * 0.02, Some((a(nrm), 0.0)))
                .or_else(|| {
                    scan.query
                        .closest_point(a(p), height * SHOE_REACH * 0.5, Some((a(nrm), 0.5)))
                })?;
            if seg.triangle_label[hit.source[1]] == Label::Skin {
                return None;
            }
            let nq = v(hit.normal);
            let out = (p - v(hit.point)).dot(&nq);
            if out > -clearance {
                return Some(-nq * (out + clearance));
            }
             
            let gap = FOOT_GAP * height;
            (out < -gap).then(|| -nq * ((out + gap) * 0.5))
        };
        let (lo, hi) = self.scale_range;
        let mut out = vec![];
         
        if lo == hi {
            for foot in self.feet.iter().filter(|f| f.shod) {
                if foot.fitted.is_empty() {
                    out.extend(foot.members.iter().map(|&(i, _)| (i, rest[i])));
                } else {
                    out.extend(foot.fitted.iter().copied());
                }
            }
            return out;
        }
        for foot in self.feet.iter_mut().filter(|f| f.shod) {
            let pushes: Vec<Option<V>> = foot
                .members
                .par_iter()
                .map(|&(i, _)| push(x[i], normals[i]))
                .collect();
            let core = |w: f64| w > 0.5;
            let count_core = foot.members.iter().filter(|m| core(m.1)).count().max(1);
            foot.outside_before = foot
                .members
                .iter()
                .zip(&pushes)
                .zip(&foot.members)
                .filter(|((m, p), n)| {
                    core(m.1)
                        && p.is_some_and(|p| p.norm() > clearance && p.dot(&normals[n.0]) < 0.0)
                })
                .count() as f64
                / count_core as f64;
            let pairs: Vec<(V, V, f64)> = foot
                .members
                .iter()
                .zip(&pushes)
                .map(|(&(i, w), p)| (rest[i], x[i] + p.unwrap_or_else(V::zeros), w))
                .collect();
            let Some((s, r, t, shift)) = similarity(&pairs, (lo, hi)) else {
                continue;
            };
            foot.shift = shift;
            foot.scale = s;
            foot.rotation = r;
            let targets: Vec<(usize, V)> = foot
                .members
                .iter()
                .map(|&(i, _)| (i, r * rest[i] * s + t))
                .collect();
            foot.outside_after = targets
                .par_iter()
                .zip(&foot.members)
                .filter(|((i, q), m)| {
                    core(m.1)
                        && push(*q, normals[*i])
                            .is_some_and(|p| p.norm() > clearance && p.dot(&normals[*i]) < 0.0)
                })
                .count() as f64
                / count_core as f64;
            out.extend(targets);
        }
        out
    }

    pub fn report(&self) -> Value {
        json!(
            self.feet
                .iter()
                .map(|f| json!({
                    "side": f.side,
                    "vertices": f.vertices,
                    "covered": f.covered,
                    "skin_fraction": f.skin_fraction,
                    "toe_skin_fraction": f.toe_skin_fraction,
                    "shod": f.shod,
                    "similarity_vertices": f.members.len(),
                    "scale": f.scale,
                    "rotation_rad": ((f.rotation.trace() - 1.0) / 2.0).clamp(-1.0, 1.0).acos(),
                    "centroid_shift_m": f.shift.norm(),
                    "outside_shoe_fraction": {"wanted": f.outside_before, "similarity": f.outside_after},
                    "bare_fit": f.shape,
                }))
                .collect::<Vec<_>>()
        )
    }
}

/// Bounds of a measured bare foot's per-axis fit (`fit_bare`): its length
/// scale, its width and height scales at the heel and at the toes, and its
/// shifts along and across (shares of the foot's length).
const BARE_LENGTH: (f64, f64) = (0.85, 1.3);
const BARE_WIDTH: (f64, f64) = (0.7, 2.0);
const BARE_HEIGHT: (f64, f64) = (0.7, 2.2);
const BARE_SHIFT: f64 = 0.2;
/// Largest turn of a bare foot about its ankle (rad, out or in).
const BARE_YAW: f64 = 0.4;

 #[allow(clippy::too_many_arguments)]
fn fit_bare(
    own: &[usize],
    other: &[usize],
    members: &[(usize, f64)],
    rest: &[V],
    normals: &[V],
    toe: &[bool],
    scan: &Scan,
    height: f64,
) -> Option<(Vec<(usize, V)>, Value)> {
    if own.len() < 50 {
        return None;
    }
    let floor = own.iter().map(|&i| rest[i].y).fold(f64::INFINITY, f64::min);
    let centroid = |set: &[usize]| set.iter().fold(V::zeros(), |s, &i| s + rest[i]) / set.len().max(1) as f64;
    let c = centroid(own);
    let toes: Vec<usize> = own.iter().copied().filter(|&i| toe[i]).collect();
    if toes.len() < 10 {
        return None;
    }
    let tc = centroid(&toes);
    let u = V::new(tc.x - c.x, 0.0, tc.z - c.z);
    if u.norm() < 1e-6 {
        return None;
    }
    let u = u.normalize();
    let l = u.cross(&V::y()).normalize();
    let o = V::new(c.x, floor, c.z);
    let local = |p: V| V::new((p - o).dot(&u), (p - o).dot(&l), p.y - floor);
    let world = |q: V| o + u * q.x + l * q.y + V::y() * q.z;
    let src: Vec<V> = own.iter().map(|&i| local(rest[i])).collect();
    let (amin, amax) = src.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |e, q| (e.0.min(q.x), e.1.max(q.x)));
    let len = amax - amin;
    if len < 1e-3 {
        return None;
    }
    let mut hs: Vec<f64> = src.iter().map(|q| q.z).collect();
    hs.sort_by(f64::total_cmp);
    let top = hs[(hs.len() as f64 * 0.95) as usize].max(1e-3);
    let t_of = |a: f64| ((a - amin) / len).clamp(0.0, 1.0);
    // [su, tu, l0, l1, tl, y0, y1], and a turn about the ankle (a quarter
    // of the way from the heel) on the floor, fading out above the top.
    let pivot = amin + 0.25 * len;
    let map = |q: V, k: &[f64; 7], yaw: f64| -> V {
        let t = t_of(q.x);
        let fade = 1.0 - guard::smoothstep((q.z - top) / (0.5 * top));
        let a2 = q.x + ((k[0] - 1.0) * (q.x - amin) + k[1]) * fade;
        let sl = k[2] + (k[3] - k[2]) * t;
        let b2 = q.y * (1.0 + (sl - 1.0) * fade) + k[4] * fade;
        let sy = k[5] + (k[6] - k[5]) * t;
        let hk = (0.5 * top).min(0.9 * top / sy.max(1e-3));
        let h2 = if q.z <= hk {
            sy * q.z
        } else if q.z <= top {
            sy * hk + (q.z - hk) * (top - sy * hk) / (top - hk)
        } else {
            q.z
        };
        let (sn, cs) = (yaw * fade).sin_cos();
        let (x, y) = (a2 - pivot, b2);
        V::new(pivot + x * cs - y * sn, x * sn + y * cs, h2)
    };
    // The scan's foot.
    let reach = 0.04 * height;
    let nearest = |p: V, set: &[usize], pos: &dyn Fn(usize) -> V| -> (usize, f64) {
        set.iter()
            .map(|&i| (i, (pos(i) - p).norm_squared()))
            .min_by(|x, y| x.1.total_cmp(&y.1))
            .map_or((usize::MAX, f64::INFINITY), |(i, d)| (i, d.sqrt()))
    };
    let mut pts: Vec<V> = scan
        .weld
        .representative
        .par_iter()
        .map(|&r| v(scan.positions[r]))
        .filter(|p| {
            let q = local(*p);
            q.z > -0.01 * height && q.z < top && q.x > amin - 0.4 * len && q.x < amax + 0.4 * len && q.y.abs() < len
        })
        .filter(|p| {
            let (_, d) = nearest(*p, own, &|i| rest[i]);
            d < reach && d <= nearest(*p, other, &|i| rest[i]).1
        })
        .map(local)
        .collect();
    if pts.len() < 100 {
        return None;
    }
    let stride = pts.len().div_ceil(4000);
    pts = pts.into_iter().step_by(stride).collect();
    let mut k = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let residual = |k: &[f64; 7], yaw: f64| -> f64 {
        let mapped: Vec<V> = src.iter().map(|q| map(*q, k, yaw)).collect();
        let d = crate::geom::ordered_sum(
            &pts.par_iter()
                .map(|p| mapped.iter().map(|m| (m - p).norm_squared()).fold(f64::INFINITY, f64::min).sqrt())
                .collect::<Vec<f64>>(),
        );
        d / pts.len() as f64
    };
    let before = residual(&k, 0.0);
    let (mut forward, mut reverse) = (0usize, 0usize);
    let mut yaw = 0.0f64;
     
    for round in 0..41 {
        if round == 20 {
            yaw = (-8..=8)
                .map(|j| j as f64 * BARE_YAW / 8.0)
                .min_by(|x, y| residual(&k, *x).total_cmp(&residual(&k, *y)))
                .unwrap_or(0.0);
            continue;
        }
        let mapped: Vec<V> = src.iter().map(|q| map(*q, &k, yaw)).collect();
        // Foot to scan.
        let fwd: Vec<(V, V)> = (0..own.len())
            .into_par_iter()
            .filter_map(|j| {
                let q = world(mapped[j]);
                let hit = scan.query.closest_point(a(q), 0.03 * height, Some((a(normals[own[j]]), 0.0)))?;
                let h = local(v(hit.point));
                (h.z < top * 1.05 && nearest(v(hit.point), other, &|i| rest[i]).1 > reach * 0.25).then_some((src[j], h))
            })
            .collect();
        // Scan to foot.
        let rev: Vec<(V, V)> = pts
            .par_iter()
            .filter_map(|p| {
                let (j, d) = (0..mapped.len())
                    .map(|j| (j, (mapped[j] - p).norm_squared()))
                    .min_by(|x, y| x.1.total_cmp(&y.1))?;
                (d.sqrt() < reach).then_some((src[j], *p))
            })
            .collect();
        forward = fwd.len();
        reverse = rev.len();
        if forward + reverse < 50 {
            break;
        }
        let (wf, wr) = (1.0 / forward.max(1) as f64, 1.0 / reverse.max(1) as f64);
        // Targets turned back by the yaw: the stretch is solved unturned.
        let (sn, cs) = (-yaw).sin_cos();
        let unturn = |t: V| {
            let (x, y) = (t.x - pivot, t.y);
            V::new(pivot + x * cs - y * sn, x * sn + y * cs, t.z)
        };
        let pairs: Vec<(V, V, f64)> = fwd
            .iter()
            .map(|&(s, t)| (s, unturn(t), wf))
            .chain(rev.iter().map(|&(s, t)| (s, unturn(t), wr)))
            .collect();
        // Weighted ridge least squares, unknowns as offsets from identity.
        let solve = |rows: &mut dyn Iterator<Item = (Vec<f64>, f64, f64)>, m: usize| -> Vec<f64> {
            let mut n = nalgebra::DMatrix::<f64>::zeros(m, m);
            let mut r = nalgebra::DVector::<f64>::zeros(m);
            for (basis, y, w) in rows {
                for i in 0..m {
                    r[i] += w * basis[i] * y;
                    for j in 0..m {
                        n[(i, j)] += w * basis[i] * basis[j];
                    }
                }
            }
            for i in 0..m {
                n[(i, i)] += 0.05 * n[(i, i)] + 1e-12;
            }
            n.lu().solve(&r).map_or(vec![0.0; m], |x| x.iter().copied().collect())
        };
        let low = |s: &V| s.z < top;
        let x = solve(
            &mut pairs.iter().filter(|p| low(&p.0)).map(|(s, t, w)| (vec![s.x - amin, 1.0], t.x - s.x, *w)),
            2,
        );
        let y = solve(
            &mut pairs.iter().filter(|p| low(&p.0)).map(|(s, t, w)| {
                let tt = t_of(s.x);
                (vec![(1.0 - tt) * s.y, tt * s.y, 1.0], t.y - s.y, *w)
            }),
            3,
        );
        let z = solve(
            &mut pairs.iter().filter(|p| p.0.z < 0.5 * top).map(|(s, t, w)| {
                let tt = t_of(s.x);
                (vec![(1.0 - tt) * s.z, tt * s.z], t.z - s.z, *w)
            }),
            2,
        );
        let next = [
            (1.0 + x[0]).clamp(BARE_LENGTH.0, BARE_LENGTH.1),
            x[1].clamp(-BARE_SHIFT * len, BARE_SHIFT * len),
            (1.0 + y[0]).clamp(BARE_WIDTH.0, BARE_WIDTH.1),
            (1.0 + y[1]).clamp(BARE_WIDTH.0, BARE_WIDTH.1),
            y[2].clamp(-BARE_SHIFT * len, BARE_SHIFT * len),
            (1.0 + z[0]).clamp(BARE_HEIGHT.0, BARE_HEIGHT.1),
            (1.0 + z[1]).clamp(BARE_HEIGHT.0, BARE_HEIGHT.1),
        ];
        let change = next.iter().zip(&k).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        k = next;
        if change < 1e-3 && round > 20 {
            break;
        }
    }
    let after = residual(&k, yaw);
    let fitted = members.iter().map(|&(i, _)| (i, world(map(local(rest[i]), &k, yaw)))).collect();
    Some((
        fitted,
        json!({
            "length_scale": k[0],
            "shift_along": k[1] / len,
            "width_scale": {"heel": k[2], "toes": k[3]},
            "shift_across": k[4] / len,
            "height_scale": {"heel": k[5], "toes": k[6]},
            "yaw_rad": yaw,
            "scan_points": pts.len(),
            "pairs": {"foot_to_scan": forward, "scan_to_foot": reverse},
            "scan_to_foot_mean_m": {"before": before, "after": after},
        }),
    ))
}

/// Weighted similarity (Umeyama) taking each pair's first point onto its
/// second, with the uniform scale clamped to `range`: `(scale, rotation,
/// translation, centroid shift)`, so `q ≈ scale · R · p + t`.
fn similarity(pairs: &[(V, V, f64)], range: (f64, f64)) -> Option<(f64, Matrix3<f64>, V, V)> {
    let ws: f64 = pairs.iter().map(|p| p.2).sum();
    if ws <= 0.0 {
        return None;
    }
    let ms = pairs.iter().fold(V::zeros(), |acc, p| acc + p.0 * p.2) / ws;
    let md = pairs.iter().fold(V::zeros(), |acc, p| acc + p.1 * p.2) / ws;
    let (mut cov, mut var) = (Matrix3::zeros(), 0.0);
    for (src, dst, w) in pairs {
        let s = src - ms;
        cov += (dst - md) * s.transpose() * *w;
        var += s.norm_squared() * w;
    }
    let r = proper_rotation(cov);
    let s = ((r.transpose() * cov).trace() / var.max(1e-20)).clamp(range.0, range.1);
    Some((s, r, md - r * ms * s, md - ms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    #[test]
    fn similarity_recovers_a_scaled_rotation_and_clamps_the_scale() {
        let r = Rotation3::from_euler_angles(0.2, -0.4, 0.7).into_inner();
        let t = V::new(0.1, -0.05, 0.3);
        let pts = [
            V::new(0.0, 0.0, 0.0),
            V::new(0.1, 0.0, 0.02),
            V::new(0.0, 0.05, 0.0),
            V::new(0.02, 0.01, 0.2),
            V::new(-0.03, 0.04, 0.1),
        ];
        let pairs: Vec<(V, V, f64)> = pts.iter().map(|p| (*p, r * p * 0.9 + t, 1.0)).collect();
        let (s, r1, t1, _) = similarity(&pairs, (0.8, 1.2)).unwrap();
        assert!((s - 0.9).abs() < 1e-9 && (r1 - r).norm() < 1e-9 && (t1 - t).norm() < 1e-9);
        // A foot that would have to shrink further keeps the bound.
        let (s, ..) = similarity(&pairs, (0.95, 1.1)).unwrap();
        assert_eq!(s, 0.95);
        assert!(similarity(&[], (0.8, 1.2)).is_none());
    }
}
