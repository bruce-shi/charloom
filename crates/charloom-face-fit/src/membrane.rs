//! Surface wrapping with an elastic displacement field on mesh connectivity.
//! Observations constrain only visible skin; hidden folded surfaces are carried
//! through their actual mesh neighbors, not through a shared image-depth column.
use crate::parallel::*;
use crate::surface::{Surface, V};
use crate::{Landmark, landmark_rmse, normals};
use std::collections::BTreeSet;

fn dot(a: &[V], b: &[V]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x.dot(y)).sum()
}
fn solve(diag: &[f64], edges: &[(usize, usize, f64)], rhs: &[V], x: &mut [V]) {
    let apply = |p: &[V]| {
        let mut q: Vec<_> = p.iter().zip(diag).map(|(v, d)| v * *d).collect();
        for &(i, j, w) in edges {
            q[i] -= p[j] * w;
            q[j] -= p[i] * w;
        }
        q
    };
    let ax = apply(x);
    let mut r: Vec<_> = rhs.iter().zip(ax).map(|(b, a)| b - a).collect();
    let mut z: Vec<_> = r.iter().zip(diag).map(|(v, d)| v / *d).collect();
    let mut direction = z.clone();
    let mut rz = dot(&r, &z);
    for _ in 0..140 {
        let q = apply(&direction);
        let den = dot(&direction, &q);
        if den < 1e-25 || rz < 1e-20 {
            break;
        }
        let alpha = rz / den;
        for i in 0..x.len() {
            x[i] += direction[i] * alpha;
            r[i] -= q[i] * alpha;
            z[i] = r[i] / diag[i];
        }
        let next = dot(&r, &z);
        let beta = next / rz;
        for i in 0..x.len() {
            direction[i] = z[i] + direction[i] * beta;
        }
        rz = next;
    }
}

pub fn fit(
    reference: &[V],
    faces: &[[usize; 3]],
    body_ids: &[usize],
    eligible: &[bool],
    blend: &[f64],
    landmarks: &[Landmark],
    skin: &Surface,
    complete: &Surface,
) -> (Vec<V>, Vec<serde_json::Value>) {
    let n = reference.len();
    let shape = Surface::new(reference, faces);
    let ns = normals(reference, faces);
    let mut is_body = vec![false; n];
    for &i in body_ids {
        is_body[i] = true;
    }
    let mut observed = vec![false; n];
    let features: Vec<_> = ["eye_", "mouth_"]
        .iter()
        .map(|prefix| {
            let points: Vec<_> = landmarks
                .iter()
                .filter(|l| l.name.starts_with(prefix))
                .map(|l| reference[l.vertex])
                .collect();
            let lo = points
                .iter()
                .fold(V::repeat(f64::INFINITY), |a, b| a.inf(b));
            let hi = points
                .iter()
                .fold(V::repeat(f64::NEG_INFINITY), |a, b| a.sup(b));
            (lo, hi)
        })
        .collect();
    for &i in body_ids {
        let p = reference[i];
        let interior = features.iter().any(|(lo, hi)| {
            p.x > lo.x - 0.005
                && p.x < hi.x + 0.005
                && p.z > lo.z - 0.005
                && p.z < hi.z + 0.005
                && p.y > hi.y + 0.002
        });
        observed[i] =
            eligible[i] && !interior && shape.front(p.x, p.z).is_some_and(|q| p.y - q.y < 0.0005);
    }
    let mut unique = BTreeSet::new();
    for f in faces {
        for (i, j) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            unique.insert((i.min(j), i.max(j)));
        }
    }
    let mut delta = vec![V::zeros(); n];
    let mut p = reference.to_vec();
    let mut trace = vec![];
    for iteration in 0..10 {
        let stiffness = if iteration < 3 {
            1.5
        } else if iteration < 6 {
            0.6
        } else {
            0.2
        };
        let edges: Vec<_> = unique.iter().map(|&(i, j)| (i, j, stiffness)).collect();
        let mut weights = vec![1e-5; n];
        let mut rhs = vec![V::zeros(); n];
        for i in 0..n {
            if blend[i] < 0.999 || !is_body[i] {
                weights[i] = 1000.;
            } else if !eligible[i] && ns[i].y > 0.1 {
                weights[i] = 0.2;
            }
        }
        let queries: Vec<_> = body_ids
            .par_iter()
            .filter_map(|&i| {
                if !observed[i] {
                    return None;
                }
                let (q, d) = skin.nearest(p[i], 0.035, Some(ns[i]))?;
                let first = complete.front(q.x, q.z)?;
                if (q.y - first.y).abs() > 0.0015 {
                    return None;
                }
                Some((i, q, if d < 0.02 { 1. } else { 0.3 }))
            })
            .collect();
        for &(i, q, w) in &queries {
            weights[i] += w;
            rhs[i] += (q - reference[i]) * w;
        }
        for l in landmarks {
            let w = 30. * l.weight;
            weights[l.vertex] += w;
            rhs[l.vertex] += (V::from(l.observed) - reference[l.vertex]) * w;
        }
        let mut diagonal = weights;
        for &(i, j, w) in &edges {
            diagonal[i] += w;
            diagonal[j] += w;
        }
        solve(&diagonal, &edges, &rhs, &mut delta);
        for i in 0..n {
            p[i] = reference[i] + delta[i] * blend[i];
        }
        trace.push(serde_json::json!({"stage":"mesh_elastic_wrap","iteration":iteration,"observed_vertices":queries.len(),"stiffness":stiffness,"landmark_rmse_mm":landmark_rmse(&p,landmarks)}));
    }
    // The helper cage is not part of the body surface graph. Carry it with the
    // nearest four canonical skin vertices; Blender only applies this result.
    for i in 0..n {
        if is_body[i] || blend[i] == 0. {
            continue;
        }
        let mut closest = [(f64::INFINITY, 0); 4];
        for &j in body_ids {
            let d = (reference[i] - reference[j]).norm_squared();
            if d < closest[3].0 {
                closest[3] = (d, j);
                closest.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
        }
        let mut displacement = V::zeros();
        let mut mass = 0.;
        for (d, j) in closest {
            let w = 1. / (d + 1e-6);
            displacement += delta[j] * w;
            mass += w;
        }
        p[i] = reference[i] + displacement / mass * blend[i];
    }
    (p, trace)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn membrane_transports_an_unobserved_neighbor() {
        let diag = [11., 2., 11.];
        let edges = [(0, 1, 1.), (1, 2, 1.)];
        let rhs = [V::y() * 0.01, V::zeros(), V::zeros()];
        let mut x = [V::zeros(); 3];
        solve(&diag, &edges, &rhs, &mut x);
        assert!(x[0].y > x[1].y && x[1].y > x[2].y && x[2].y > 0.);
        assert!((diag[1] * x[1].y - x[0].y - x[2].y).abs() < 1e-12);
    }
}
