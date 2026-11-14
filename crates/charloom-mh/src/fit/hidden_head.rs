//! Scan visibility and interpolation only where neither the scan nor GPT
//! provides data. GPT scalp is held during fit/sculpt in `suit_head`.
use crate::geom::{V, a, proper_rotation, v};
use crate::scan::Scan;
use nalgebra::Matrix3;
use rayon::prelude::*;
const LOOKS: usize = 40;
const SEEN_SHARE: f64 = 0.35;

/// Least-squares similarity of a reference to anatomical correspondences.
/// Only the caller's observed vertices enter any part of the transform.
pub(super) fn reference_frame(
    reference: &[V],
    fitted: &[V],
    ids: &[usize],
) -> (Matrix3<f64>, f64, V) {
    let mean = |points: &[V]| ids.iter().fold(V::zeros(), |s, &i| s + points[i]) / ids.len() as f64;
    let (cx, cs) = (mean(fitted), mean(reference));
    let (mut m, mut ss) = (Matrix3::zeros(), 0.0);
    for &i in ids {
        let (p, q) = (fitted[i] - cx, reference[i] - cs);
        m += p * q.transpose();
        ss += q.norm_squared();
    }
    let rot = proper_rotation(m);
    // RMS radii also count residual shape mismatch as scale.
    let scale = (rot.transpose() * m).trace() / ss.max(1e-18);
    (rot, scale, cx - rot * cs * scale)
}

/// Biharmonic displacement from measured rest, with every observed point
/// held exactly. No silhouette field or skin/hair attraction is allowed.
pub(super) fn restore(
    neighbours: &[Vec<u32>],
    shape: &[V],
    classes: &[u8],
    fixed: &[bool],
    x: &mut [V],
) -> serde_json::Value {
    let ids: Vec<usize> = (0..x.len())
        .filter(|&i| classes[i] == 3 && !fixed[i])
        .collect();
    if ids.is_empty() {
        return serde_json::json!({"applied":false,"reason":"no truly hidden exterior vertices","vertices":0});
    }
    let rows = super::suit_head::bending_rows(shape, neighbours);
    let mut delta: Vec<V> = x.iter().zip(shape).map(|(x, s)| x - s).collect();
    let before = x.to_vec();
    let mut change: f64 = 0.0;
    let mut iterations = 0;
    for _ in 0..4000 {
        change = 0.0;
        for &i in &ids {
            let diagonal = rows[i].get(&i).copied().unwrap_or(0.0);
            if diagonal < 1e-12 {
                continue;
            }
            let off = rows[i]
                .iter()
                .filter(|(j, _)| **j != i)
                .fold(V::zeros(), |s, (&j, &w)| s + delta[j] * w);
            let next = -off / diagonal;
            change = change.max((next - delta[i]).norm());
            delta[i] = next;
        }
        iterations += 1;
        if change < 1e-10 {
            break;
        }
    }
    for &i in &ids {
        x[i] = shape[i] + delta[i];
    }
    serde_json::json!({"applied":true,"method":"truly_hidden_biharmonic_bridge","vertices":ids.len(),
        "iterations":iterations,"max_update_mm":change*1000.0,
        "max_move_mm":ids.iter().map(|&i|(x[i]-before[i]).norm()*1000.0).fold(0.0,f64::max)})
}

 pub fn seen_from_outside(scan: &Scan, centre: V, radius: f64) -> Vec<bool> {
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    let directions: Vec<V> = (0..LOOKS)
        .map(|k| {
            let y = 1.0 - 2.0 * (k as f64 + 0.5) / LOOKS as f64;
            let r = (1.0 - y * y).sqrt();
            let th = golden * k as f64;
            V::new(r * th.cos(), y, r * th.sin())
        })
        .collect();
    (0..scan.positions.len())
        .into_par_iter()
        .map(|i| {
            let p = v(scan.positions[i]);
            if (p - centre).norm() > radius {
                return true;
            }
            let normal = v(scan.normals[i]);
            let (mut facing, mut hits) = (0usize, 0usize);
            for d in &directions {
                if normal.dot(d) <= 0.2 {
                    continue;
                }
                facing += 1;
                if scan
                    .query
                    .first_ray_distance(a(p + d * 0.00005), a(*d), radius * 3.0)
                    .is_none()
                {
                    hits += 1;
                }
            }
            hits > 0 && hits as f64 >= SEEN_SHARE * facing.max(1) as f64
        })
        .collect()
}

pub(super) fn head_visibility(scan: &Scan, x: &[V], head: &[bool]) -> Vec<bool> {
    let ids: Vec<usize> = (0..x.len()).filter(|&i| head[i]).collect();
    let centre = ids.iter().fold(V::zeros(), |s, &i| s + x[i]) / ids.len().max(1) as f64;
    let radius = ids
        .iter()
        .map(|&i| (x[i] - centre).norm())
        .fold(0.0, f64::max)
        * 1.5;
    seen_from_outside(scan, centre, radius)
}

/// A hair shell bounds a point along the ray from inside the skull.
/// An outward shell crossing bounds the skull with clearance. Internal
/// braid roots are not radial skull targets. The bound never attracts a
/// point or grows the skull; no crossing means no hair constraint.
pub fn hair_ceiling(
    hair: &charloom_engine::surface_query::SurfaceQuery,
    centre: V,
    point: V,
    reach: f64,
    margin: f64,
) -> Option<V> {
    let delta = point - centre;
    let length = delta.norm();
    let direction = delta.try_normalize(1e-12)?;
    let boundary = hair.outward_ray_distance(a(centre), a(direction), reach)?;
    let limit = (boundary - margin).max(0.0);
    (length > limit).then(|| centre + direction * limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skull_frame_ignores_hidden_hair_shell() {
        let reference = vec![
            V::new(-0.04, 0.02, 0.03),
            V::new(0.04, 0.02, 0.03),
            V::new(0.0, -0.06, 0.02),
            V::new(0.0, 0.05, -0.02),
            V::new(0.0, 0.10, -0.07),
        ];
        let rotation = nalgebra::Rotation3::from_euler_angles(0.1, -0.15, 0.03).into_inner();
        let translation = V::new(0.02, 1.4, 0.1);
        let mut fitted: Vec<V> = reference
            .iter()
            .map(|p| rotation * p * 0.82 + translation)
            .collect();
        // Only this unobserved scalp vertex lies on an oversized hair shell.
        fitted[4] += V::new(0.03, 0.08, -0.1);
        let (r, scale, t) = reference_frame(&reference, &fitted, &[0, 1, 2, 3]);
        assert!((scale - 0.82).abs() < 1e-10);
        assert!((r - rotation).norm() < 1e-10);
        assert!((t - translation).norm() < 1e-10);
    }

    #[test]
    fn face_residual_does_not_inflate_similarity_scale() {
        let reference = vec![
            V::new(-1.0, -1.0, 0.0),
            V::new(1.0, -1.0, 0.0),
            V::new(-1.0, 1.0, 0.0),
            V::new(1.0, 1.0, 0.0),
        ];
        let mut fitted = reference.clone();
        for (i, z) in [0.4, -0.4, -0.4, 0.4].into_iter().enumerate() {
            fitted[i].z = z;
        }
        let (_, scale, _) = reference_frame(&reference, &fitted, &[0, 1, 2, 3]);
        assert!((scale - 1.0).abs() < 1e-10);
    }

    #[test]
    fn radial_hair_ceiling_keeps_clearance_without_growing_a_skull() {
        let hair = crate::scan::build_query(
            &[
                [-1.0, -1.0, 0.1],
                [1.0, -1.0, 0.1],
                [1.0, 1.0, 0.1],
                [-1.0, 1.0, 0.1],
            ],
            &[[0, 1, 2], [0, 2, 3]],
        );
        let centre = V::zeros();
        let limit = hair_ceiling(&hair, centre, V::new(0.0, 0.0, 0.12), 1.0, 0.002).unwrap();
        assert!((limit.z - 0.098).abs() < 1e-12);
        assert!(hair_ceiling(&hair, centre, V::new(0.0, 0.0, 0.08), 1.0, 0.002).is_none());
        assert!(hair_ceiling(&hair, centre, V::new(0.0, 0.0, -0.12), 1.0, 0.002).is_none());
    }
    #[test]
    fn visibility_does_not_see_skin_through_one_millimetre_hair() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [-0.1, -0.1, 0.001],
            [0.1, -0.1, 0.001],
            [0.1, 0.1, 0.001],
            [-0.1, 0.1, 0.001],
        ];
        let triangles = vec![[2, 3, 4], [2, 4, 5]];
        let scan = Scan {
            query: crate::scan::build_query(&positions, &triangles),
            weld: crate::geom::Weld::new(&positions, 1e-7),
            normals: vec![[0.0, 0.0, 1.0]; positions.len()],
            uvs: vec![],
            positions,
            triangles,
            triangle_material: vec![None; 2],
            transform: nalgebra::Matrix4::identity(),
            images: vec![],
            materials: vec![],
            textures: vec![],
            samplers: vec![],
            source_hash: String::new(),
        };
        let seen = seen_from_outside(&scan, V::zeros(), 1.0);
        assert!(!seen[0], "covered skin must be assigned to GPT/hidden data");
        assert!(seen[1], "uncovered skin remains observed");
    }
}
