//! Free outer-surface registration with local rotation/scale shape constraints.
//! A local/global ARAP-style solve keeps a healthy parametric reference while
//! allowing observed features to move beyond the morph space. Cavities are
//! excluded from correspondences and embedded in the outer head cage.
use crate::parallel::*;
use crate::surface::{Surface, V};
use crate::{FaceRegions, Landmark, landmark_rmse, normals};
use std::collections::{BTreeMap, BTreeSet};

fn observed_skin(surface: &Surface, p: V, normal: V) -> bool {
    normal.y < 0.2 && surface.front(p.x, p.z).is_some_and(|q| p.y - q.y < 0.00035)
}

fn reference_weight(blend: f64, has_skin_match: bool) -> f64 {
    if blend < 0.999 {
        1000.
    } else if has_skin_match {
        0.02
    } else {
        // Missing skin observations cannot justify dragging the hidden skull
        // with nearby facial landmarks. Keep the healthy parametric location;
        // a real matched surface still receives its independent fitting weight.
        10.
    }
}

#[derive(Clone, Copy)]
struct IncidentEdge {
    neighbor: usize,
    first_endpoint: bool,
}

/// Immutable reference data shared by all outer fitting iterations. Incident
/// edges retain their original order and orientation for identical f64 sums.
struct ShapeGraph {
    edges: Vec<(usize, usize)>,
    rest_edges: Vec<V>,
    squared: Vec<f64>,
    degree: Vec<f64>,
    rest_laplacian: Vec<V>,
    offsets: Vec<usize>,
    incidents: Vec<IncidentEdge>,
}
impl ShapeGraph {
    fn new(reference: &[V], edges: &[(usize, usize)]) -> Self {
        let n = reference.len();
        let mut offsets = vec![0; n + 1];
        let mut squared = vec![0.; n];
        let mut degree = vec![0.; n];
        let mut rest_laplacian = vec![V::zeros(); n];
        let mut rest_edges = Vec::with_capacity(edges.len());
        for &(a, b) in edges {
            offsets[a + 1] += 1;
            offsets[b + 1] += 1;
            let rest = reference[a] - reference[b];
            rest_edges.push(rest);
            squared[a] += rest.norm_squared();
            squared[b] += rest.norm_squared();
            degree[a] += 1.;
            degree[b] += 1.;
            rest_laplacian[a] += rest;
            rest_laplacian[b] -= rest;
        }
        for i in 0..n {
            offsets[i + 1] += offsets[i];
        }
        let mut cursor = offsets[..n].to_vec();
        let mut incidents = vec![
            IncidentEdge {
                neighbor: 0,
                first_endpoint: true
            };
            edges.len() * 2
        ];
        for &(a, b) in edges {
            incidents[cursor[a]] = IncidentEdge {
                neighbor: b,
                first_endpoint: true,
            };
            cursor[a] += 1;
            incidents[cursor[b]] = IncidentEdge {
                neighbor: a,
                first_endpoint: false,
            };
            cursor[b] += 1;
        }
        Self {
            edges: edges.to_vec(),
            rest_edges,
            squared,
            degree,
            rest_laplacian,
            offsets,
            incidents,
        }
    }
    fn row(&self, i: usize) -> &[IncidentEdge] {
        &self.incidents[self.offsets[i]..self.offsets[i + 1]]
    }
    fn laplacian(&self, points: &[V], out: &mut [V]) {
        for (i, value) in out.iter_mut().enumerate() {
            let center = points[i];
            let mut sum = V::zeros();
            for edge in self.row(i) {
                if edge.first_endpoint {
                    sum += center - points[edge.neighbor];
                } else {
                    sum -= points[edge.neighbor] - center;
                }
            }
            *value = sum;
        }
    }
    /// A = W + stiffness L + L B L. Fuse the first two edge traversals, keeping
    /// each accumulator's arithmetic order. The returned energy uses the same
    /// vertex-order dot product as the CG solver.
    fn apply(
        &self,
        points: &[V],
        weights: &[f64],
        bending: &[f64],
        stiffness: f64,
        detail: &mut [V],
        curvature: &mut [V],
        out: &mut [V],
    ) -> f64 {
        for i in 0..points.len() {
            let center = points[i];
            let mut weighted = center * weights[i];
            let mut laplacian = V::zeros();
            for edge in self.row(i) {
                if edge.first_endpoint {
                    let d = center - points[edge.neighbor];
                    weighted += d * stiffness;
                    laplacian += d;
                } else {
                    let d = points[edge.neighbor] - center;
                    weighted -= d * stiffness;
                    laplacian -= d;
                }
            }
            out[i] = weighted;
            detail[i] = laplacian * bending[i];
        }
        self.laplacian(detail, curvature);
        let mut energy = 0.;
        for i in 0..points.len() {
            out[i] += curvature[i];
            energy += points[i].dot(&out[i]);
        }
        energy
    }
}

 
fn shape_solve(
    points: &[V],
    weights: &[f64],
    targets: &[V],
    graph: &ShapeGraph,
    stiffness: f64,
) -> Vec<V> {
    let n = points.len();
    let mut covariance = vec![nalgebra::Matrix3::zeros(); n];
    for (&(a, b), rest) in graph.edges.iter().zip(&graph.rest_edges) {
        let rest = *rest;
        let current = points[a] - points[b];
        let c = current * rest.transpose();
        covariance[a] += c;
        covariance[b] += c;
    }
    let rotations: Vec<_> = covariance
        .iter()
        .map(|c| {
            let svd = c.svd(true, true);
            let u = svd.u.unwrap();
            let vt = svd.v_t.unwrap();
            let mut sign = nalgebra::Matrix3::identity();
            sign[(2, 2)] = (u * vt).determinant().signum();
            u * sign * vt
        })
        .collect();
    let mut numerator = vec![0.; n];
    for (&(a, b), rest) in graph.edges.iter().zip(&graph.rest_edges) {
        let rest = *rest;
        let current = points[a] - points[b];
        numerator[a] += current.dot(&(rotations[a] * rest));
        numerator[b] += current.dot(&(rotations[b] * rest));
    }
    let transforms: Vec<_> = (0..n)
        .map(|i| {
            rotations[i]
                * ((numerator[i] / graph.squared[i].max(1e-15) + 0.25) / 1.25).clamp(0.5, 2.)
        })
        .collect();
    let mut rhs = targets.to_vec();
    let mut diagonal = weights.to_vec();
    // Visible skin may depart from the anatomical prior; hidden skin needs
    // stronger curvature support because no observation constrains it.
    let bending: Vec<_> = weights
        .iter()
        .map(|w| (4. / (1. + w)).clamp(0.2, 4.))
        .collect();
    let detail: Vec<_> = graph
        .rest_laplacian
        .iter()
        .enumerate()
        .map(|(i, v)| transforms[i] * v)
        .collect();
    let weighted_detail: Vec<_> = detail.iter().zip(&bending).map(|(v, w)| v * *w).collect();
    let mut bending_rhs = vec![V::zeros(); n];
    graph.laplacian(&weighted_detail, &mut bending_rhs);
    for i in 0..n {
        rhs[i] += bending_rhs[i];
        diagonal[i] += graph.degree[i] * graph.degree[i] * bending[i];
    }
    for (&(a, b), rest) in graph.edges.iter().zip(&graph.rest_edges) {
        let desired = (transforms[a] + transforms[b]) * *rest * (stiffness * 0.5);
        rhs[a] += desired;
        rhs[b] -= desired;
        diagonal[a] += stiffness + bending[b];
        diagonal[b] += stiffness + bending[a];
    }
     
    let mut detail = vec![V::zeros(); n];
    let mut curvature = vec![V::zeros(); n];
    let mut apply = |p: &[V], out: &mut [V]| {
        graph.apply(
            p,
            weights,
            &bending,
            stiffness,
            &mut detail,
            &mut curvature,
            out,
        )
    };
    let dot = |a: &[V], b: &[V]| a.iter().zip(b).map(|(a, b)| a.dot(b)).sum::<f64>();
    let mut x = points.to_vec();
    let mut ad = vec![V::zeros(); n];
    apply(&x, &mut ad);
    let mut residual: Vec<_> = rhs.iter().zip(&ad).map(|(b, a)| b - a).collect();
    let mut direction: Vec<_> = residual
        .iter()
        .zip(&diagonal)
        .map(|(r, d)| r / d.max(1e-12))
        .collect();
    let mut rz = dot(&residual, &direction);
    let initial = rz;
    let mut z = vec![V::zeros(); n];
    for _ in 0..400 {
        if rz < 1e-22 || rz < initial * 1e-14 {
            break;
        }
        let denominator = apply(&direction, &mut ad);
        if denominator <= 0. || !denominator.is_finite() {
            return points.to_vec();
        }
        let alpha = rz / denominator;
        let mut next = 0.;
        for i in 0..n {
            x[i] += direction[i] * alpha;
            residual[i] -= ad[i] * alpha;
            z[i] = residual[i] / diagonal[i].max(1e-12);
            next += residual[i].dot(&z[i]);
        }
        for i in 0..n {
            direction[i] = z[i] + direction[i] * (next / rz);
        }
        rz = next;
    }
    if x.iter().any(|p| p.iter().any(|v| !v.is_finite())) {
        return points.to_vec();
    }
    x
}

pub fn fit(
    base: &[V],
    original: &[V],
    initial: &[V],
    healthy_prior: &[V],
    faces: &[[usize; 3]],
    regions: &FaceRegions,
    blend: &[f64],
    landmarks: &[Landmark],
    skin: &Surface,
    complete: &Surface,
    hair: &Surface,
    neck_start: f64,
    _groups: &BTreeMap<String, Vec<usize>>,
) -> (Vec<V>, Vec<serde_json::Value>) {
    let ids = &regions.outer;
    let mut lookup = vec![usize::MAX; base.len()];
    for (k, &i) in ids.iter().enumerate() {
        lookup[i] = k;
    }
    let outer_faces: Vec<_> = faces
        .iter()
        .filter(|f| f.iter().all(|&i| lookup[i] != usize::MAX))
        .copied()
        .collect();
    let mut edges = BTreeSet::new();
    for f in &outer_faces {
        for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            let (i, j) = (lookup[a], lookup[b]);
            edges.insert((i.min(j), i.max(j)));
        }
    }
    let edges: Vec<_> = edges.into_iter().collect();
    let lm = |name: &str| landmarks.iter().find(|l| l.name == name).unwrap();
    let scale = (lm("forehead").observed[2] - lm("chin").observed[2])
        .abs()
        .max(0.05);
    let ears = crate::ears::Ears::new(
        base,
        &outer_faces,
        &regions.scalp_preserve,
        blend,
        landmarks,
    );
    let mut reference = healthy_prior.to_vec();
    let mut ear_initialization = ears.transport(&mut reference);
    ear_initialization["stage"] = serde_json::json!("ear_shape_initialization");
    let reference = reference.as_slice();
    let mut p = initial.to_vec();
    ears.transport(&mut p);
    // ICP correspondences and contact backoff are discrete decisions. Suppress
    // sub-nanometre roundoff before those decisions so native/WASM arithmetic
    // cannot amplify an irrelevant last-bit difference into a visible crease.
    // Authored ears remain rigid; their vertices are never rounded individually.
    let stabilize = |points: &mut [V]| {
        for &i in ids {
            if blend[i] >= 0.999 && !ears.contains(i) {
                for v in points[i].iter_mut() {
                    *v = (*v * 1e9).round() / 1e9;
                }
            }
        }
    };
    let healthy: Vec<_> = ids.iter().map(|&i| reference[i]).collect();
    let shape_graph = ShapeGraph::new(&healthy, &edges);
    let reference_normal = normals(reference, &outer_faces);
    // Close observation apertures only for visibility queries. These temporary
    // triangles are never added to the model or exported topology.
    let mut occlusion_faces = outer_faces.clone();
    for names in [
        ["mouth_left", "mouth_upper", "mouth_right", "mouth_lower"],
        [
            "eye_inner_left",
            "eye_upper_left",
            "eye_outer_left",
            "eye_lower_left",
        ],
        [
            "eye_inner_right",
            "eye_upper_right",
            "eye_outer_right",
            "eye_lower_right",
        ],
    ] {
        let q = names.map(|name| lm(name).vertex);
        occlusion_faces.extend([[q[0], q[1], q[2]], [q[0], q[2], q[3]]]);
    }
    let occluder = Surface::new(reference, &occlusion_faces);
    let observed: Vec<_> = ids
        .par_iter()
        .map(|&i| observed_skin(&occluder, reference[i], reference_normal[i]))
        .collect();
    let mut radius = vec![0.003f64; ids.len()];
    for &(i, j) in &edges {
        let length = (reference[ids[i]] - reference[ids[j]]).norm();
        radius[i] = radius[i].min(length * 0.4);
        radius[j] = radius[j].min(length * 0.4);
    }
    let chin = lm("chin").observed[2];
    let forehead = lm("forehead").observed[2];
    let rear = landmarks
        .iter()
        .map(|l| l.observed[1])
        .fold(f64::NEG_INFINITY, f64::max)
        + 0.018;
    let mut trace = vec![
        serde_json::json!({"stage":"anatomical_partition","outer_vertices":ids.len(),"cavity_vertices":regions.cavities.iter().map(Vec::len).collect::<Vec<_>>(),"cavities_observed":false}),
        ear_initialization,
    ];
    let mut graph = vec![BTreeSet::new(); p.len()];
    for &(a, b) in &edges {
        graph[ids[a]].insert(ids[b]);
        graph[ids[b]].insert(ids[a]);
    }
    for (stage, stiffness) in [20., 8., 2., 0.5].into_iter().enumerate() {
        stabilize(&mut p);
        let before_stage = p.clone();
        for iteration in 0..10 {
            stabilize(&mut p);
            let ns = normals(&p, &outer_faces);
            let matches: Vec<_> = ids
                .par_iter()
                .enumerate()
                .map(|(k, &i)| {
                    let q = p[i];
                    if ears.contains(i)
                        || !observed[k]
                        || blend[i] < 0.999
                        || q.z < chin - 0.004
                        || q.z > forehead + 0.018
                        || q.y > rear
                        || reference_normal[i].y > 0.35
                    {
                        return None;
                    }
                    let (hit, distance) = skin.nearest(q, 0.025, Some(ns[i]))?;
                    // Discard a match hidden behind hair or another observed layer.
                    if complete
                        .front(hit.x, hit.z)
                        .is_none_or(|f| (f.y - hit.y).abs() > 0.0015)
                    {
                        return None;
                    }
                    Some((hit, 10. * (0.004 / distance.max(0.004)).powi(2)))
                })
                .collect();
            let mut w = vec![0.; ids.len()];
            let mut rhs = vec![V::zeros(); ids.len()];
            let mut count = 0;
            for (k, &i) in ids.iter().enumerate() {
                // No observation is not evidence for another facial surface.
                w[k] = reference_weight(blend[i], matches[k].is_some());
                if let Some((q, weight)) = matches[k] {
                    w[k] += weight;
                    rhs[k] += (q - reference[i]) * (weight / scale);
                    count += 1;
                }
            }
            for l in landmarks {
                let k = lookup[l.vertex];
                if k != usize::MAX {
                    let weight = 800. * l.weight;
                    w[k] += weight;
                    rhs[k] += (V::from(l.observed) - reference[l.vertex]) * (weight / scale);
                }
            }
            let targets: Vec<_> = ids
                .iter()
                .enumerate()
                .map(|(k, &i)| reference[i] * w[k] + rhs[k] * scale)
                .collect();
            let current: Vec<_> = ids.iter().map(|&i| p[i]).collect();
            let solved = shape_solve(&current, &w, &targets, &shape_graph, stiffness);
            let mut maximum = 0f64;
            for (k, &i) in ids.iter().enumerate() {
                if blend[i] < 0.999 {
                    continue;
                }
                let next = solved[k];
                let delta = next - p[i];
                maximum = maximum.max(delta.norm());
                p[i] += delta * (radius[k] / delta.norm().max(radius[k]));
            }
            stabilize(&mut p);
            ears.transport(&mut p);
            let mut unseen_drift: Vec<_> = ids
                .iter()
                .enumerate()
                .filter(|(k, i)| {
                    matches[*k].is_none() && blend[**i] >= 0.999 && !ears.contains(**i)
                })
                .map(|(_, &i)| (p[i] - reference[i]).norm() * 1000.)
                .collect();
            unseen_drift.sort_by(f64::total_cmp);
            let matched_error2: Vec<_> = ids
                .iter()
                .enumerate()
                .filter_map(|(k, &i)| {
                    matches[k]
                        .as_ref()
                        .map(|(hit, _)| (p[i] - hit).norm_squared())
                })
                .collect();
            let matched_rms = (!matched_error2.is_empty()).then(|| {
                (matched_error2.iter().sum::<f64>() / matched_error2.len() as f64).sqrt() * 1000.
            });
            trace.push(serde_json::json!({"stage":"outer_shape_icp","level":stage,"iteration":iteration,"stiffness":stiffness,"observed":count,"max_step_mm":maximum*1000.,"landmark_rmse_mm":landmark_rmse(&p,landmarks),
                "unmatched_prior_weight":10.,"unmatched_non_ear_vertices":unseen_drift.len(),
                "unmatched_prior_drift_p95_mm":unseen_drift.get(unseen_drift.len().saturating_sub(1)*95/100),
                "unmatched_prior_drift_max_mm":unseen_drift.last(),
                "retained_skin_match_rms_mm":matched_rms}));
        }
        let quality = crate::deformation_guard::limit_update(
            reference,
            &before_stage,
            &mut p,
            &outer_faces,
            &graph,
        );
        trace.push(
            serde_json::json!({"stage":"shape_quality_guard","level":stage,"quality":quality}),
        );
        eprintln!(
            "anatomical shape {stage}: {:.3} mm",
            landmark_rmse(&p, landmarks)
        );
    }
    trace.push(crate::attachment::contain_scalp(
        &mut p,
        ids,
        &edges,
        &outer_faces,
        landmarks,
        &regions.scalp_preserve,
        hair,
        complete,
    ));
     
    trace.push(crate::attachment::neck_transition(
        base, original, &mut p, faces, regions, landmarks, neck_start, blend,
    ));
    trace.push(ears.transport(&mut p));
    // A rest-space tetrahedral cage carries hidden anatomy with the complete
    // outer surface. Rim width alone cannot tell how the cheek volume changed.
    for binding in &regions.bindings {
        if blend[binding.vertex] > 0.999 {
            p[binding.vertex] = (0..4)
                .map(|k| p[binding.cage[k]] * binding.weights[k])
                .sum();
        }
    }
    for &i in &regions.unbound {
        if blend[i] < 0.999 {
            continue;
        }
        let mut near: Vec<_> = ids
            .iter()
            .map(|&j| ((base[i] - base[j]).norm_squared(), j))
            .collect();
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut delta = V::zeros();
        let mut mass = 0.;
        for &(distance, j) in near.iter().take(8) {
            let w = 1. / distance.max(1e-8);
            delta += (p[j] - base[j]) * w;
            mass += w;
        }
        p[i] = base[i] + delta / mass;
    }
    trace.push(serde_json::json!({"stage":"cavity_transport","method":"rest-space tetrahedral embedding in the outer head cage","bound":regions.bindings.len(),"unbound_local_transport":regions.unbound.len(),"landmark_rmse_mm":landmark_rmse(&p,landmarks)}));
    (p, trace)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached_shape_operator_matches_edge_order_with_reversed_and_isolated_vertices() {
        // Independent edge-scatter reference, including unsorted/reversed edges,
        // a duplicate, a self-edge and an isolated vertex. Reuse the graph while
        // changing both coefficients and points, as outer fitting iterations do.
        let edges = [(3, 0), (0, 2), (4, 1), (2, 1), (0, 2), (1, 3), (4, 4)];
        let reference: Vec<_> = (0..6).map(|i| V::new(i as f64, 0.1, -0.3)).collect();
        let graph = ShapeGraph::new(&reference, &edges);
        let laplacian = |points: &[V]| {
            let mut out = vec![V::zeros(); points.len()];
            for &(a, b) in &edges {
                let d = points[a] - points[b];
                out[a] += d;
                out[b] -= d;
            }
            out
        };
        assert_eq!(graph.rest_laplacian, laplacian(&reference));
        for scale in [0.0001, 1., 10000.] {
            let points: Vec<_> = (0..6)
                .map(|i| V::new((i as f64 - 2.3) * scale, -scale, 0.31))
                .collect();
            let weights: Vec<_> = (0..6).map(|i| 0.02 + i as f64 * scale).collect();
            let bending: Vec<_> = weights
                .iter()
                .map(|w| (4. / (1. + w)).clamp(0.2, 4.))
                .collect();
            for stiffness in [0., 0.5, 20.] {
                let mut expected: Vec<_> =
                    points.iter().zip(&weights).map(|(p, w)| p * *w).collect();
                for &(a, b) in &edges {
                    let d = (points[a] - points[b]) * stiffness;
                    expected[a] += d;
                    expected[b] -= d;
                }
                let detail: Vec<_> = laplacian(&points)
                    .iter()
                    .zip(&bending)
                    .map(|(p, w)| p * *w)
                    .collect();
                let curvature = laplacian(&detail);
                for (p, c) in expected.iter_mut().zip(curvature) {
                    *p += c;
                }
                let expected_energy = points
                    .iter()
                    .zip(&expected)
                    .map(|(p, q)| p.dot(q))
                    .sum::<f64>();
                let mut actual = vec![V::zeros(); 6];
                let energy = graph.apply(
                    &points,
                    &weights,
                    &bending,
                    stiffness,
                    &mut vec![V::zeros(); 6],
                    &mut vec![V::zeros(); 6],
                    &mut actual,
                );
                assert_eq!(actual, expected);
                assert_eq!(energy, expected_energy);
            }
        }
    }
    #[test]
    fn shape_prior_transports_a_curved_patch_without_shrinking_it() {
        let reference = vec![
            V::new(-0.02, 0., 0.),
            V::new(0.02, 0., 0.),
            V::new(0., -0.02, 0.),
            V::new(0., 0.02, 0.),
            V::new(0., 0., 0.01),
        ];
        let edges = [
            (0, 2),
            (2, 1),
            (1, 3),
            (3, 0),
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
        ];
        let rotation = nalgebra::Rotation3::from_euler_angles(0.3, -0.15, 0.1);
        let transformed: Vec<_> = reference
            .iter()
            .map(|p| rotation * p + V::new(0.2, -0.1, 0.05))
            .collect();
        let weights = vec![1.; reference.len()];
        let result = shape_solve(
            &transformed,
            &weights,
            &transformed,
            &ShapeGraph::new(&reference, &edges),
            8.,
        );
        for (a, b) in result.iter().zip(&transformed) {
            assert!((a - b).norm() < 1e-8);
        }
    }

    #[test]
    fn partial_front_observation_keeps_unseen_curved_skull_near_healthy_prior() {
        let mut reference = Vec::new();
        let mut edges = Vec::new();
        let mut faces = Vec::new();
        for row in 0..7 {
            for column in 0..7 {
                let x = (column as f64 - 3.) * 0.01;
                let y = (row as f64 - 3.) * 0.01;
                reference.push(V::new(x, y, 0.05 - 2. * x * x - 3. * y * y));
                let i = row * 7 + column;
                if column > 0 {
                    edges.push((i - 1, i));
                }
                if row > 0 {
                    edges.push((i - 7, i));
                }
                if row > 0 && column > 0 {
                    edges.push((i - 8, i));
                    faces.extend([[i - 8, i - 7, i], [i - 8, i, i - 1]]);
                }
            }
        }
        let graph = ShapeGraph::new(&reference, &edges);
        let observed_shift = V::new(0., -0.004, 0.002);
        let mut points: Vec<_> = reference
            .iter()
            .map(|p| p + V::new(0., -0.025, -0.02))
            .collect();
        let mut weights = Vec::new();
        let mut targets = Vec::new();
        for (i, p) in reference.iter().enumerate() {
            let observed = i / 7 < 2;
            let prior = reference_weight(1., observed);
            let skin = if observed { 10. } else { 0. };
            weights.push(prior + skin);
            targets.push(p * prior + (p + observed_shift) * skin);
        }
        for stiffness in [20., 8., 2., 0.5] {
            for _ in 0..10 {
                points = shape_solve(&points, &weights, &targets, &graph, stiffness);
            }
        }
        let front_error = (0..14)
            .map(|i| (points[i] - reference[i] - observed_shift).norm())
            .fold(0., f64::max);
        let unseen_error = (28..49)
            .map(|i| (points[i] - reference[i]).norm())
            .fold(0., f64::max);
        assert!(
            front_error < 0.0015,
            "observed front was pinned to prior: {front_error}"
        );
        assert!(
            unseen_error < 0.0015,
            "unseen skull followed the face: {unseen_error}"
        );
        for [a, b, c] in faces {
            let before = (reference[b] - reference[a]).cross(&(reference[c] - reference[a]));
            let after = (points[b] - points[a]).cross(&(points[c] - points[a]));
            assert!(
                before.dot(&after) > 0.,
                "partial observations folded the skull"
            );
        }
        for row in 4..6 {
            for column in 1..6 {
                let i = row * 7 + column;
                let delta = |j| points[j] - reference[j];
                let ridge = delta(i - 7) + delta(i + 7) - delta(i) * 2.;
                assert!(ridge.norm() < 0.0015, "unsupported ridge: {}", ridge.norm());
            }
        }
    }
    #[test]
    fn aperture_cap_prevents_the_visible_mouth_backwall_becoming_skin() {
        let points = vec![
            V::new(-1., 0.01, -1.),
            V::new(1., 0.01, -1.),
            V::new(0., 0.01, 1.),
            V::new(-1., 0., -1.),
            V::new(1., 0., -1.),
            V::new(0., 0., 1.),
        ];
        let open = Surface::new(&points, &[[0, 1, 2]]);
        let closed = Surface::new(&points, &[[0, 1, 2], [3, 4, 5]]);
        let back = V::new(0., 0.01, 0.);
        assert!(observed_skin(&open, back, -V::y()));
        assert!(!observed_skin(&closed, back, -V::y()));
        assert!(observed_skin(&closed, V::zeros(), -V::y()));
    }
}
