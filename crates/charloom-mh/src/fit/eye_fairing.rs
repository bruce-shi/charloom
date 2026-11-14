//! Opt-in diagnostic fairing after eye clearing. The template identifies a
//! local eye neighbourhood; the measured opening extremes stay constrained.
//! This removes sharp clearing displacements, but is not a self-intersection
//! solver or a replacement for fitting the eyelid and socket coherently.
use crate::geom::{V, Weld, a, adjacency, v};
use crate::prior::{PartKind, Prior, Region};
use anyhow::{Context, Result, ensure};
use charloom_core::Vec3;
use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub struct Reference {
    weld: Weld,
    neighbours: Vec<Vec<u32>>,
    active: Vec<bool>,
}

impl Reference {
    /// Capture before the shape/pose stages mutate the template. The region
    /// is anatomical and independent of a character's current fit errors.
    pub fn new(prior: &Prior) -> Self {
        let body = prior.body();
        let weld = Weld::new(&body.prim.positions, 1e-6);
        let neighbours = adjacency(&body.prim.triangles, &weld);
        let mut balls = vec![];
        if let Some(eyes) = prior.meshes.iter().find(|m| m.kind == PartKind::Eyes) {
            for joint in ["eye.L", "eye.R"] {
                let Some(&j) = prior.by_name.get(joint) else {
                    continue;
                };
                let points: Vec<V> = eyes
                    .prim
                    .positions
                    .iter()
                    .zip(&eyes.influences)
                    .filter(|(_, row)| row.iter().any(|w| w.joint == j && w.weight > 0.5))
                    .map(|(p, _)| v(*p))
                    .collect();
                if points.is_empty() {
                    continue;
                }
                let centre = points.iter().sum::<V>() / points.len() as f64;
                let radius = points
                    .iter()
                    .map(|p| (p - centre).norm())
                    .fold(0.0, f64::max);
                balls.push((centre, radius * 2.4));
            }
        }
        let active = weld
            .representative
            .iter()
            .map(|&i| {
                prior.region[prior.dominant_joint(prior.body, i)] == Some(Region::Head)
                    && balls
                        .iter()
                        .any(|(c, r)| (v(body.prim.positions[i]) - c).norm() < *r)
            })
            .collect();
        Self {
            weld,
            neighbours,
            active,
        }
    }

    pub fn apply(
        &self,
        prior: &Prior,
        positions: &mut [Vec3],
        landmarks: &BTreeMap<String, Vec3>,
    ) -> Result<Value> {
        ensure!(
            positions.len() == self.weld.id.len(),
            "eye fairing topology mismatch"
        );
        let original: Vec<V> = self
            .weld
            .representative
            .iter()
            .map(|&i| v(positions[i]))
            .collect();
        let anchors: Vec<(usize, V)> = super::face_landmark_sets(prior)
            .into_iter()
            .filter(|(name, ids)| name.starts_with("eye_") && ids.len() == 1)
            .filter_map(|(name, ids)| landmarks.get(name).map(|p| (self.weld.id[ids[0]], v(*p))))
            .filter(|(i, _)| self.active[*i])
            .collect();
        if anchors.is_empty() || !self.active.iter().any(|x| *x) {
            return Ok(json!({"applied": false, "reason": "no marked eye neighbourhood"}));
        }
        let result = solve(&original, &self.neighbours, &self.active, &anchors, 8.0)?;
        let max_move = result
            .iter()
            .zip(&original)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        let anchor_error = anchors
            .iter()
            .map(|(i, p)| (result[*i] - p).norm())
            .fold(0.0, f64::max);
        for (i, p) in positions.iter_mut().enumerate() {
            if self.active[self.weld.id[i]] {
                *p = a(result[self.weld.id[i]]);
            }
        }
        Ok(json!({"applied": true, "experimental": true,
            "vertices": self.active.iter().filter(|x| **x).count(), "anchors": anchors.len(),
            "max_move_m": max_move, "anchor_max_error_m": anchor_error,
            "self_intersections_checked": false}))
    }
}

fn solve(
    original: &[V],
    neighbours: &[Vec<u32>],
    active: &[bool],
    anchors: &[(usize, V)],
    strength: f64,
) -> Result<Vec<V>> {
    if strength == 0.0 {
        return Ok(original.to_vec());
    }
    let ids: Vec<usize> = (0..original.len()).filter(|&i| active[i]).collect();
    if ids.is_empty() {
        return Ok(original.to_vec());
    }
    let mut local = vec![usize::MAX; original.len()];
    for (j, &i) in ids.iter().enumerate() {
        local[i] = j;
    }
    let n = ids.len();
    let mut triplets = vec![];
    let mut target = original.to_vec();
    let mut priority = vec![1.0; original.len()];
    for &(i, p) in anchors {
        if active[i] {
            target[i] = p;
            priority[i] = 1000.0;
        }
    }
    let mut rhs: Vec<V> = ids.iter().map(|&i| target[i] * priority[i]).collect();
    for (j, &i) in ids.iter().enumerate() {
        triplets.push(Triplet::new(j, j, priority[i]));
        if neighbours[i].is_empty() {
            continue;
        }
        let inv = -1.0 / neighbours[i].len() as f64;
        let mut terms = vec![(j, 1.0)];
        let mut boundary = V::zeros();
        for &other in &neighbours[i] {
            let other = other as usize;
            if active[other] {
                terms.push((local[other], inv));
            } else {
                boundary += original[other] * inv;
            }
        }
        for &(r, cr) in &terms {
            rhs[r] -= boundary * (strength * strength * cr);
            for &(c, cc) in &terms {
                triplets.push(Triplet::new(r, c, strength * strength * cr * cc));
            }
        }
    }
    let matrix = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets)
        .context("eye fairing sparse matrix")?;
    let factor = matrix
        .sp_cholesky(faer::Side::Lower)
        .context("eye fairing factorization")?;
    let b = faer::Mat::from_fn(n, 3, |r, c| rhs[r][c]);
    let solved = factor.solve(&b);
    let mut out = original.to_vec();
    for (j, &i) in ids.iter().enumerate() {
        out[i] = V::new(solved[(j, 0)], solved[(j, 1)], solved[(j, 2)]);
        ensure!(
            out[i].iter().all(|x| x.is_finite()),
            "non-finite eye fairing result"
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spike_is_reduced_while_boundary_is_exact() {
        let p = vec![
            V::new(-2.0, 0.0, 0.0),
            V::new(-1.0, 0.0, 0.0),
            V::new(0.0, 0.0, 1.0),
            V::new(1.0, 0.0, 0.0),
            V::new(2.0, 0.0, 0.0),
        ];
        let neighbours = vec![vec![1], vec![0, 2], vec![1, 3], vec![2, 4], vec![3]];
        let active = [false, true, true, true, false];
        let out = solve(&p, &neighbours, &active, &[], 8.0).unwrap();
        assert_eq!(out[0], p[0]);
        assert_eq!(out[4], p[4]);
        assert!(out[2].z < 0.2);
        assert_eq!(solve(&p, &neighbours, &active, &[], 0.0).unwrap(), p);
    }

    #[test]
    fn landmark_is_held_and_translation_is_equivariant() {
        let p = vec![
            V::new(-1.0, 0.0, 0.0),
            V::new(0.0, 0.0, 1.0),
            V::new(1.0, 0.0, 0.0),
        ];
        let neighbours = vec![vec![1], vec![0, 2], vec![1]];
        let active = [false, true, false];
        let anchor = V::new(0.0, 0.0, 0.2);
        let out = solve(&p, &neighbours, &active, &[(1, anchor)], 8.0).unwrap();
        assert!((out[1] - anchor).norm() < 0.02);
        let shift = V::new(2.0, -3.0, 4.0);
        let moved: Vec<_> = p.iter().map(|p| p + shift).collect();
        let result = solve(&moved, &neighbours, &active, &[(1, anchor + shift)], 8.0).unwrap();
        for (a, b) in out.iter().zip(result) {
            assert!((a + shift - b).norm() < 1e-10);
        }
    }
}
