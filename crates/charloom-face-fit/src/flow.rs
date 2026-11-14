//! Smooth spatial deformation: inner and outer surfaces share one displacement
//! field. Numerical Jacobian sampling limits each step; independent triangle
//! intersection checks remain necessary (this is not a proof of injectivity).
use crate::parallel::*;
use crate::surface::{Surface, V};
use crate::{Landmark, landmark_rmse, normals, smooth};
use nalgebra::{DMatrix, Matrix3};
use std::collections::BTreeMap;

pub fn fit(
    base: &[V],
    prior: &[V],
    triangles: &[[usize; 3]],
    body_ids: &[usize],
    eligible: &[bool],
    exterior: &[bool],
    regularize_interior: bool,
    blend: &[f64],
    landmarks: &[Landmark],
    surface: &Surface,
) -> (Vec<V>, Vec<serde_json::Value>) {
    let mut p = base.to_vec();
    let mut trace = vec![];
    let head_ids: Vec<_> = body_ids
        .iter()
        .copied()
        .filter(|&i| blend[i] > 0.99)
        .collect();
    let lowest = landmarks
        .iter()
        .map(|l| l.observed[2])
        .fold(f64::INFINITY, f64::min);
    let cutoff = lowest - 0.065;
    let full = lowest - 0.005;
    let posterior_cut =
        landmarks.iter().map(|l| l.observed[1]).sum::<f64>() / landmarks.len() as f64 + 0.055;
    let scalp_cut = landmarks
        .iter()
        .map(|l| l.observed[2])
        .fold(f64::NEG_INFINITY, f64::max)
        + 0.035;
    let support: Vec<_> = base
        .iter()
        .enumerate()
        .map(|(i, p)| {
            regularize_interior
                || eligible[i]
                || exterior[i]
                || p.y > posterior_cut
                || p.z > scalp_cut
        })
        .collect();
    for iteration in 0..32 {
        let sigma = if iteration < 10 {
            0.032
        } else if iteration < 22 {
            0.023
        } else {
            0.017
        };
        let ns = normals(&p, triangles);
        let mut samples: Vec<(V, V, f64)> = landmarks
            .iter()
            .map(|l| {
                (
                    p[l.vertex],
                    V::from(l.observed) - p[l.vertex],
                    100. * l.weight,
                )
            })
            .collect();
        // Spatial cells prevent denser topology from dominating registration.
        let mut cells = BTreeMap::<[i32; 3], usize>::new();
        for &i in &head_ids {
            if !support[i] {
                continue;
            }
            let cell = p[i].map(|x| (x / 0.014).floor() as i32);
            cells.entry([cell.x, cell.y, cell.z]).or_insert(i);
        }
        let mut observed = 0;
        for &i in cells.values() {
            if samples.iter().any(|(q, _, _)| (p[i] - q).norm() < 0.003) {
                continue;
            }
            if eligible[i] {
                if let Some((q, d)) = surface.nearest(p[i], 0.035, Some(ns[i])) {
                    samples.push((p[i], q - p[i], (0.005 / d.max(0.005)).max(0.1)));
                    observed += 1;
                }
            } else {
                 
                samples.push((p[i], prior[i] * 0.35 + base[i] * 0.65 - p[i], 0.20));
            }
        }
        let count = samples.len();
        let mut kernel = DMatrix::zeros(count, count);
        let mut rhs = DMatrix::zeros(count, 3);
        for i in 0..count {
            for j in 0..=i {
                let value =
                    (-(samples[i].0 - samples[j].0).norm_squared() / (2. * sigma * sigma)).exp();
                kernel[(i, j)] = value;
                kernel[(j, i)] = value;
            }
            kernel[(i, i)] += 0.08 / samples[i].2;
            let delta = samples[i].1;
            for axis in 0..3 {
                rhs[(i, axis)] = delta[axis];
            }
        }
        let Some(factor) = kernel.cholesky() else {
            break;
        };
        let coefficients = factor.solve(&rhs);
        let evaluate = |q: V| -> (V, Matrix3<f64>) {
            let mut value = V::zeros();
            let mut derivative = Matrix3::zeros();
            for (j, (center, _, _)) in samples.iter().enumerate() {
                let diff = q - center;
                let distance = diff.norm_squared();
                if distance > sigma * sigma * 36. {
                    continue;
                }
                let g = (-distance / (2. * sigma * sigma)).exp();
                let c = V::new(
                    coefficients[(j, 0)],
                    coefficients[(j, 1)],
                    coefficients[(j, 2)],
                );
                value += c * g;
                derivative += c * (-diff * (g / (sigma * sigma))).transpose();
            }
            (value, derivative)
        };
        let deformation: Vec<_> = p
            .par_iter()
            .map(|&q| {
                if q.z > cutoff {
                    evaluate(q)
                } else {
                    (V::zeros(), Matrix3::zeros())
                }
            })
            .collect();
        let max_gradient = deformation.iter().map(|(_, j)| j.norm()).fold(0., f64::max);
        let max_distance = deformation.iter().map(|(v, _)| v.norm()).fold(0., f64::max);
        let mut step = 0.8f64
            .min(0.35 / max_gradient.max(1e-9))
            .min(0.004 / max_distance.max(1e-9));
        // Face triangles must retain area and cannot turn over within one step.
        let mut accepted = false;
        for _ in 0..10 {
            let trial: Vec<_> = p
                .iter()
                .enumerate()
                .map(|(i, &q)| {
                    if blend[i] == 0. {
                        base[i]
                    } else {
                        q + deformation[i].0 * (step * smooth((q.z - cutoff) / (full - cutoff)))
                    }
                })
                .collect();
            if crate::valid_triangles(&p, &trial, triangles) {
                p = trial;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        trace.push(serde_json::json!({"stage":"spatial_flow","iteration":iteration,"control_points":count,"observed_cells":observed,"sigma_m":sigma,"maximum_sampled_gradient":max_gradient,"step":if accepted{step}else{0.},"landmark_rmse_mm":landmark_rmse(&p,landmarks)}));
        eprintln!(
            "flow {iteration}: {:.2} mm, step {:.3}",
            landmark_rmse(&p, landmarks),
            if accepted { step } else { 0. }
        );
    }
    (p, trace)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn spatial_fit_moves_both_sides_of_a_thin_shell_and_keeps_body_fixed() {
        let mut points = vec![];
        for y in [0., 0.004] {
            for z in 0..5 {
                for x in 0..5 {
                    points.push(V::new((x as f64 - 2.) * 0.02, y, 1.4 + z as f64 * 0.025));
                }
            }
        }
        let mut triangles = vec![];
        for layer in [0, 25] {
            for z in 0..4 {
                for x in 0..4 {
                    let a = layer + z * 5 + x;
                    triangles.extend([[a, a + 1, a + 6], [a, a + 6, a + 5]]);
                }
            }
        }
        let target: Vec<_> = points
            .iter()
            .map(|p| V::new(p.x, p.y - 0.006 * smooth((p.z - 1.42) / 0.07), p.z))
            .collect();
        let surface = Surface::new(&target, &triangles[..32]);
        let landmarks: Vec<_> = [10, 12, 14, 20, 22, 24]
            .iter()
            .map(|&i| Landmark {
                name: format!("anchor_{i}"),
                vertex: i,
                original: crate::arr(points[i]),
                observed: crate::arr(target[i]),
                weight: 1.,
            })
            .collect();
        let ids: Vec<_> = (0..points.len()).collect();
        let eligible: Vec<_> = ids.iter().map(|&i| i < 25).collect();
        let blend: Vec<_> = points
            .iter()
            .map(|p| if p.z <= 1.4 { 0. } else { 1. })
            .collect();
        let before = landmark_rmse(&points, &landmarks);
        let (fitted, _) = fit(
            &points, &points, &triangles, &ids, &eligible, &eligible, false, &blend, &landmarks,
            &surface,
        );
        assert!(landmark_rmse(&fitted, &landmarks) < before * 0.25);
        for i in 0..5 {
            assert!((fitted[i] - points[i]).norm() < 1e-12);
        }
        for i in 0..25 {
            assert!(fitted[i + 25].y - fitted[i].y > 0.002);
        }
        assert!((fitted[47].y - points[47].y).abs() > 0.002);
    }
}
