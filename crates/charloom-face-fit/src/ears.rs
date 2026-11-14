//! Preserve authored pinnae as rigid patches attached to the nearby skull.
//! Frontal face matches and hair silhouettes do not identify ear anatomy.
use crate::{Landmark, arr, smooth, surface::V};
use nalgebra::Matrix3;
use serde_json::{Value, json};
use std::collections::{BTreeSet, VecDeque};

struct Patch {
    vertices: Vec<usize>,
    anchors: Vec<usize>,
    collar: Vec<(usize, f64)>,
    edges: Vec<(usize, usize)>,
}

pub struct Ears {
    reference: Vec<V>,
    protected: Vec<bool>,
    patches: Vec<Patch>,
}

impl Ears {
    pub fn new(
        reference: &[V],
        faces: &[[usize; 3]],
        preserve: &[usize],
        blend: &[f64],
        landmarks: &[Landmark],
    ) -> Self {
        let mut protected = vec![false; reference.len()];
        let fixed: BTreeSet<_> = landmarks.iter().map(|l| l.vertex).collect();
        let mut remaining: BTreeSet<_> = preserve.iter().copied().collect();
        for &i in &remaining {
            protected[i] = true;
        }
        let mut graph = vec![BTreeSet::new(); reference.len()];
        for &[a, b, c] in faces {
            for (a, b) in [(a, b), (b, c), (c, a)] {
                graph[a].insert(b);
                graph[b].insert(a);
            }
        }
        let mut patches = Vec::new();
        while let Some(seed) = remaining.pop_first() {
            let mut vertices = vec![seed];
            let mut cursor = 0;
            while cursor < vertices.len() {
                for &j in &graph[vertices[cursor]] {
                    if remaining.remove(&j) {
                        vertices.push(j);
                    }
                }
                cursor += 1;
            }
            vertices.sort_unstable();
            let center: V =
                vertices.iter().map(|&i| reference[i]).sum::<V>() / vertices.len() as f64;
            let radius = vertices
                .iter()
                .map(|&i| (reference[i] - center).norm())
                .fold(0., f64::max);
            let feather = (radius * 0.6).clamp(0.008, 0.025);
            let mut distance = vec![f64::INFINITY; reference.len()];
            let mut queue = VecDeque::new();
            for &i in &vertices {
                distance[i] = 0.;
                queue.push_back(i);
            }
            while let Some(i) = queue.pop_front() {
                for &j in &graph[i] {
                    let next = distance[i] + (reference[i] - reference[j]).norm();
                    if !protected[j] && next <= feather * 1.5 && next + 1e-12 < distance[j] {
                        distance[j] = next;
                        queue.push_back(j);
                    }
                }
            }
             
            let anchors = (0..reference.len())
                .filter(|&i| {
                    !protected[i]
                        && blend[i] > 0.999
                        && distance[i] >= feather
                        && distance[i] <= feather * 1.5
                })
                .collect();
            let collar = (0..reference.len())
                .filter(|&i| {
                    !protected[i]
                        && !fixed.contains(&i)
                        && blend[i] > 0.999
                        && distance[i] < feather
                })
                .map(|i| (i, 1. - smooth(distance[i] / feather)))
                .collect();
            let edges = vertices
                .iter()
                .flat_map(|&i| {
                    graph[i]
                        .iter()
                        .filter(move |&&j| i < j)
                        .map(move |&j| (i, j))
                })
                .filter(|&(_, j)| vertices.binary_search(&j).is_ok())
                .collect();
            patches.push(Patch {
                vertices,
                anchors,
                collar,
                edges,
            });
        }
        Self {
            reference: reference.to_vec(),
            protected,
            patches,
        }
    }

    pub fn contains(&self, vertex: usize) -> bool {
        self.protected[vertex]
    }

    pub fn transport(&self, points: &mut [V]) -> Value {
        let proposal = points.to_vec();
        let mut collar_weight = vec![0.; points.len()];
        let mut reports = Vec::new();
        for patch in &self.patches {
            let (rotation, translation) = rigid_frame(&self.reference, &proposal, &patch.anchors);
            for &i in &patch.vertices {
                points[i] = rotation * self.reference[i] + translation;
            }
            for &(i, weight) in &patch.collar {
                if weight > collar_weight[i] {
                    points[i] = proposal[i] * (1. - weight)
                        + (rotation * self.reference[i] + translation) * weight;
                    collar_weight[i] = weight;
                }
            }
            let strain = patch
                .edges
                .iter()
                .map(|&(a, b)| {
                    let rest = (self.reference[a] - self.reference[b]).norm();
                    ((points[a] - points[b]).norm() / rest.max(1e-12) - 1.).abs()
                })
                .fold(0., f64::max);
            reports.push(json!({"vertices":patch.vertices.len(),"attachment_anchors":patch.anchors.len(),
                "collar_vertices":patch.collar.len(),"maximum_relative_edge_change":strain,
                "translation_m":arr(translation),"rotation_degrees":((rotation.trace()-1.)/2.).clamp(-1.,1.).acos().to_degrees(),
                "attachment_status":if patch.anchors.len()>=3 {"skull_attached"} else {"insufficient_anchors_prior_retained"}}));
        }
        json!({"stage":"ear_shape_preservation","method":"rigid authored pinnae, skull attachment outside a geodesic collar",
            "status":if self.patches.is_empty(){"no_authored_ear_regions"}else{"preserved_prior"},
            "protected_vertices":self.protected.iter().filter(|&&p|p).count(),"patches":reports,
            "source_ear_shape_observed":false,"hair_is_ear_evidence":false,"user_approved":false})
    }
}

fn rigid_frame(reference: &[V], points: &[V], anchors: &[usize]) -> (Matrix3<f64>, V) {
    if anchors.len() < 3 {
        return (Matrix3::identity(), V::zeros());
    }
    let a = anchors.iter().map(|&i| reference[i]).sum::<V>() / anchors.len() as f64;
    let b = anchors.iter().map(|&i| points[i]).sum::<V>() / anchors.len() as f64;
    let covariance = anchors.iter().fold(Matrix3::zeros(), |m, &i| {
        m + (points[i] - b) * (reference[i] - a).transpose()
    });
    let svd = covariance.svd(true, true);
    if svd.singular_values[1] < 1e-12 {
        return (Matrix3::identity(), b - a);
    }
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    let mut handedness = Matrix3::identity();
    handedness[(2, 2)] = (u * vt).determinant().signum();
    let rotation = u * handedness * vt;
    (rotation, b - rotation * a)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn patch() -> (Vec<V>, Vec<[usize; 3]>, Vec<usize>) {
        let mut p = Vec::new();
        let mut faces = Vec::new();
        let mut ear = Vec::new();
        for y in 0..15 {
            for z in 0..15 {
                let d = ((y as f64 - 7.).powi(2) + (z as f64 - 7.).powi(2)).sqrt();
                p.push(V::new(
                    0.01 * (1. - d / 4.).max(0.),
                    y as f64 * 0.004,
                    z as f64 * 0.004,
                ));
                if d <= 2.1 {
                    ear.push(y * 15 + z);
                }
                if y < 14 && z < 14 {
                    let i = y * 15 + z;
                    faces.extend([[i, i + 15, i + 1], [i + 1, i + 15, i + 16]]);
                }
            }
        }
        (p, faces, ear)
    }
    #[test]
    fn pinna_recovers_shape_and_follows_skull_rotation_without_iteration_drift() {
        let (reference, faces, ids) = patch();
        let ears = Ears::new(&reference, &faces, &ids, &vec![1.; reference.len()], &[]);
        assert!(ears.patches[0].anchors.len() >= 3);
        let r = nalgebra::Rotation3::from_axis_angle(&V::z_axis(), 0.3).into_inner();
        let t = V::new(0.008, -0.003, 0.004);
        let expected: Vec<_> = reference.iter().map(|p| r * p + t).collect();
        let mut fitted = expected.clone();
        for &i in &ids {
            fitted[i].x *= 0.2;
            fitted[i].z *= 1.8;
        }
        for _ in 0..20 {
            ears.transport(&mut fitted);
        }
        for i in 0..fitted.len() {
            assert!((fitted[i] - expected[i]).norm() < 1e-10);
        }
        let report = ears.transport(&mut fitted);
        assert!(
            report["patches"][0]["maximum_relative_edge_change"]
                .as_f64()
                .unwrap()
                < 1e-10
        );
    }
    #[test]
    fn collar_does_not_move_landmarks_disconnected_geometry_or_lower_body() {
        let (mut reference, faces, ids) = patch();
        reference.push(reference[112]);
        let pinned = Landmark {
            name: "fixed".into(),
            vertex: 109,
            original: arr(reference[109]),
            observed: arr(reference[109]),
            weight: 1.,
        };
        let mut blend = vec![1.; reference.len()];
        blend[108] = 0.;
        let ears = Ears::new(&reference, &faces, &ids, &blend, &[pinned]);
        let mut fitted = reference.clone();
        for p in &mut fitted {
            p.x += 0.01;
        }
        fitted[109].z += 0.002;
        fitted[108].z += 0.003;
        let before = fitted.clone();
        ears.transport(&mut fitted);
        for i in [108, 109, reference.len() - 1] {
            assert_eq!(fitted[i], before[i]);
        }
    }
    #[test]
    fn absent_ear_annotations_leave_geometry_unchanged() {
        let (reference, faces, _) = patch();
        let ears = Ears::new(&reference, &faces, &[], &vec![1.; reference.len()], &[]);
        let mut fitted = reference.clone();
        assert_eq!(
            ears.transport(&mut fitted)["status"],
            "no_authored_ear_regions"
        );
        assert_eq!(fitted, reference);
    }
}
