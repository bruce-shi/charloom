//! Small, dependency-free triangle-mesh utilities shared by every stage.
use charloom_core::Vec3;
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use std::collections::HashMap;

pub type V = Vector3<f64>;

#[inline]
pub fn v(p: Vec3) -> V {
    V::new(p[0], p[1], p[2])
}
#[inline]
pub fn a(p: V) -> Vec3 {
    [p.x, p.y, p.z]
}

/// Nearest proper rotation to `m` (polar decomposition). For `m = Σ b aᵀ`
/// the result `R` minimises `Σ |R a − b|²`.
pub fn proper_rotation(m: Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    let (Some(u), Some(vt)) = (svd.u, svd.v_t) else {
        return Matrix3::identity();
    };
    let mut d = Matrix3::identity();
    d[(2, 2)] = (u * vt).determinant().signum();
    let r = u * d * vt;
    if r.iter().all(|x| x.is_finite()) {
        r
    } else {
        Matrix3::identity()
    }
}

/// Rotation taking unit direction `from` onto unit direction `to`.
pub fn rotation_between(from: V, to: V) -> Matrix3<f64> {
    let (Some(f), Some(t)) = (from.try_normalize(1e-12), to.try_normalize(1e-12)) else {
        return Matrix3::identity();
    };
    nalgebra::Rotation3::rotation_between(&f, &t)
        .map(|r| r.into_inner())
        .unwrap_or_else(|| {
            // Opposite directions: rotate 180° about any perpendicular axis.
            let axis = if f.x.abs() < 0.9 { V::x() } else { V::y() };
            let axis = nalgebra::Unit::new_normalize(f.cross(&axis));
            nalgebra::Rotation3::from_axis_angle(&axis, std::f64::consts::PI).into_inner()
        })
}

/// Area-weighted vertex normals accumulated over welded positions so UV seams
/// shade smoothly.
pub fn vertex_normals(positions: &[Vec3], triangles: &[[u32; 3]], weld: &Weld) -> Vec<Vec3> {
    let mut acc = vec![V::zeros(); weld.count];
    for t in triangles {
        let [p0, p1, p2] = t.map(|i| v(positions[i as usize]));
        let n = (p1 - p0).cross(&(p2 - p0));
        for &i in t {
            acc[weld.id[i as usize]] += n;
        }
    }
    weld.id
        .iter()
        .map(|&r| a(acc[r].try_normalize(1e-20).unwrap_or(V::y())))
        .collect()
}

pub fn face_normal(positions: &[Vec3], t: &[u32; 3]) -> V {
    let [p0, p1, p2] = t.map(|i| v(positions[i as usize]));
    (p1 - p0).cross(&(p2 - p0))
}

/// Collapses render-seam duplicates (identical positions) into one id.
#[derive(Clone)]
pub struct Weld {
    pub id: Vec<usize>,
    pub count: usize,
    /// One original vertex per welded id.
    pub representative: Vec<usize>,
}

impl Weld {
    pub fn new(positions: &[Vec3], tolerance: f64) -> Self {
        let q = |x: f64| (x / tolerance).round() as i64;
        let mut map = HashMap::<[i64; 3], usize>::with_capacity(positions.len());
        let mut id = Vec::with_capacity(positions.len());
        let mut representative = vec![];
        for (i, p) in positions.iter().enumerate() {
            let key = [q(p[0]), q(p[1]), q(p[2])];
            let next = map.len();
            let r = *map.entry(key).or_insert(next);
            if r == representative.len() {
                representative.push(i);
            }
            id.push(r);
        }
        Self {
            count: representative.len(),
            id,
            representative,
        }
    }
}

/// Symmetric neighbour lists on welded ids.
pub fn adjacency(triangles: &[[u32; 3]], weld: &Weld) -> Vec<Vec<u32>> {
    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); weld.count];
    for t in triangles {
        let w = t.map(|i| weld.id[i as usize] as u32);
        for (x, y) in [(w[0], w[1]), (w[1], w[2]), (w[2], w[0])] {
            if x != y {
                rows[x as usize].push(y);
                rows[y as usize].push(x);
            }
        }
    }
    rows.par_iter_mut().for_each(|r| {
        r.sort_unstable();
        r.dedup();
    });
    rows
}

/// Percentile summary used in every report.
pub fn percentiles(mut values: Vec<f64>) -> serde_json::Value {
    values.retain(|x| x.is_finite());
    if values.is_empty() {
        return serde_json::json!(null);
    }
    values.sort_by(f64::total_cmp);
    let at = |p: f64| values[((values.len() - 1) as f64 * p).round() as usize];
    serde_json::json!({"count":values.len(),"p50":at(0.5),"p95":at(0.95),"p99":at(0.99),"max":at(1.0),"min":at(0.0)})
}

 pub fn distortion(rest: &[Vec3], deformed: &[Vec3], triangles: &[[u32; 3]]) -> serde_json::Value {
    let mut ratios = Vec::with_capacity(triangles.len() * 3);
    // The most stretched edge's vertices (where the maximum is).
    let mut worst = (0.0f64, [0u32; 2]);
    let mut flipped = 0usize;
    let mut degenerate = 0usize;
    // Edges under 0.1 % of the height (sub-millimetre lid-margin and toe
    // edges) are ignored, as in the animation checks: a 0.1 mm edge growing
    // to 0.9 mm is invisible but dominated the maximum.
    let (lo, hi) = rest
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), p| {
            (l.min(p[1]), h.max(p[1]))
        });
    let min_edge = ((hi - lo) * 0.001).max(1e-9);
    for t in triangles {
        for (i, j) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let r = (v(rest[i as usize]) - v(rest[j as usize])).norm();
            if r > min_edge {
                let ratio = (v(deformed[i as usize]) - v(deformed[j as usize])).norm() / r;
                if ratio > worst.0 {
                    worst = (ratio, [i, j]);
                }
                ratios.push(ratio);
            }
        }
        let n0 = face_normal(rest, t);
        let n1 = face_normal(deformed, t);
        if n1.norm() < 1e-14 {
            degenerate += 1;
        } else if n0.norm() > 1e-14 && n0.dot(&n1) < 0.0 {
            flipped += 1;
        }
    }
    serde_json::json!({"edge_ratio":percentiles(ratios),"max_edge_vertices":worst.1,"min_edge_m":min_edge,"flipped_triangles":flipped,"degenerate_triangles":degenerate})
}

/// Closest point on triangle (Ericson), returned as barycentric weights.
pub fn closest_barycentric(p: V, a0: V, b0: V, c0: V) -> [f64; 3] {
    let ab = b0 - a0;
    let ac = c0 - a0;
    let ap = p - a0;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let bp = p - b0;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return [0.0, 1.0, 0.0];
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        return [1.0 - t, t, 0.0];
    }
    let cp = p - c0;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return [0.0, 0.0, 1.0];
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        return [1.0 - t, 0.0, t];
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return [0.0, 1.0 - t, t];
    }
    let denom = 1.0 / (va + vb + vc);
    let s = vb * denom;
    let t = vc * denom;
    [1.0 - s - t, s, t]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weld_merges_seam_duplicates_and_normals_agree() {
        let p = vec![
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
        ];
        let t = vec![[0, 1, 2], [3, 4, 2]];
        let w = Weld::new(&p, 1e-7);
        assert_eq!(w.count, 4);
        assert_eq!(w.id[1], w.id[3]);
        let n = vertex_normals(&p, &t, &w);
        assert_eq!(n[1], n[3]);
        assert!((n[0][2] - 1.0).abs() < 1e-12);
    }
    #[test]
    fn distortion_detects_flip() {
        let rest = vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]];
        let flipped = vec![[0., 0., 0.], [0., 1., 0.], [1., 0., 0.]];
        let d = distortion(&rest, &flipped, &[[0, 1, 2]]);
        assert_eq!(d["flipped_triangles"], 1);
    }
    #[test]
    fn proper_rotation_recovers_rotation() {
        let r = nalgebra::Rotation3::from_euler_angles(0.3, -0.2, 0.9).into_inner();
        let pts = [V::x(), V::y(), V::z(), V::new(1., 2., 3.)];
        let m = pts
            .iter()
            .fold(Matrix3::zeros(), |s, p| s + (r * p) * p.transpose());
        assert!((proper_rotation(m) - r).norm() < 1e-9);
    }
}

 pub fn ordered_sum(values: &[f64]) -> f64 {
    values.iter().sum()
}

pub fn dump_stage(name: &str, points: impl IntoIterator<Item = [f64; 3]>) {
    let Some(dir) = std::env::var_os("CHARLOOM_DUMP_STAGES") else {
        return;
    };
    let wide = std::env::var_os("CHARLOOM_DUMP_F64").is_some();
    let mut bytes = vec![];
    for p in points {
        for c in p {
            if wide {
                bytes.extend_from_slice(&c.to_le_bytes());
            } else {
                bytes.extend_from_slice(&(c as f32).to_le_bytes());
            }
        }
    }
    let dir = std::path::PathBuf::from(dir);
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join(format!("{name}.f32")), bytes);
}
