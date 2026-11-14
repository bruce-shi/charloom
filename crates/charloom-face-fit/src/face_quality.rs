//! Independent source-surface comparison. Fitting residuals and template strain
//! are not interchangeable: a large deformation supported by the source is valid.
use crate::{
    normals,
    surface::{Surface, V},
};
use anyhow::{Result, ensure};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Deserialize, Serialize, JsonSchema)]
pub struct Request {
    pub reference: Vec<[f64; 3]>,
    pub fitted: Vec<[f64; 3]>,
    pub triangles: Vec<[usize; 3]>,
    pub outer_vertices: Vec<usize>,
    pub source: Vec<[f64; 3]>,
    pub source_triangles: Vec<[usize; 3]>,
    #[serde(default)]
    pub occlusion_triangles: Vec<[usize; 3]>,
    #[serde(default)]
    pub unobserved_vertices: Vec<usize>,
}

pub fn run(input: Request) -> Result<Value> {
    let n = input.reference.len();
    let m = input.source.len();
    ensure!(
        n > 2
            && n == input.fitted.len()
            && n <= 1_000_000
            && m <= 1_000_000
            && input.triangles.len() <= 1_500_000
            && input.source_triangles.len() <= 1_500_000
            && input.occlusion_triangles.len() <= 1_500_000,
        "Invalid comparison mesh size"
    );
    ensure!(
        input
            .reference
            .iter()
            .chain(&input.fitted)
            .chain(&input.source)
            .flatten()
            .all(|v| v.is_finite())
            && input
                .triangles
                .iter()
                .flatten()
                .chain(&input.outer_vertices)
                .chain(&input.unobserved_vertices)
                .all(|&i| i < n)
            && input
                .source_triangles
                .iter()
                .chain(&input.occlusion_triangles)
                .flatten()
                .all(|&i| i < m),
        "Invalid comparison geometry"
    );
    let reference: Vec<_> = input.reference.iter().copied().map(V::from).collect();
    let fitted: Vec<_> = input.fitted.iter().copied().map(V::from).collect();
    let source: Vec<_> = input.source.iter().copied().map(V::from).collect();
    Ok(
        json!({"report":{"source_comparison":compare(&reference,&fitted,&input.triangles,&input.outer_vertices,&source,&input.source_triangles,&input.occlusion_triangles,&input.unobserved_vertices)}}),
    )
}

#[derive(Clone, Copy)]
struct Sample {
    point: V,
    normal: V,
    signed: f64,
    distance: f64,
    source_face: [usize; 3],
}

fn fingerprint(points: &[V], faces: &[[usize; 3]]) -> String {
    let mut hash = Sha256::new();
    hash.update((points.len() as u64).to_le_bytes());
    for p in points {
        for v in p.iter() {
            hash.update(v.to_le_bytes());
        }
    }
    hash.update((faces.len() as u64).to_le_bytes());
    for f in faces {
        for &i in f {
            hash.update((i as u64).to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

fn percentile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    Some(v[((v.len() - 1) as f64 * fraction).round() as usize])
}

pub fn compare(
    reference: &[V],
    fitted: &[V],
    triangles: &[[usize; 3]],
    outer_ids: &[usize],
    source: &[V],
    source_faces: &[[usize; 3]],
    complete_faces: &[[usize; 3]],
    unobserved_vertices: &[usize],
) -> Value {
    let outer: BTreeSet<_> = outer_ids.iter().copied().collect();
    let face_ids: Vec<_> = triangles
        .iter()
        .enumerate()
        .filter_map(|(i, f)| f.iter().all(|v| outer.contains(v)).then_some(i))
        .collect();
    let faces: Vec<_> = face_ids.iter().map(|&i| triangles[i]).collect();
    if faces.is_empty() {
        return json!({"version":1,"status":"insufficient_evidence","auto_pass":false,"issues":[],"coverage":{"observed_vertices":0,"total_vertices":outer.len()},"limits":["No outer skin supplied."]});
    }
    let low = outer
        .iter()
        .fold(V::repeat(f64::INFINITY), |v, &i| v.inf(&reference[i]));
    let high = outer
        .iter()
        .fold(V::repeat(f64::NEG_INFINITY), |v, &i| v.sup(&reference[i]));
    let extent = (high - low).max().max(0.05);
    let tolerance = (extent * 0.01).clamp(0.001, 0.005);
    let distance_limit = (extent * 0.08).clamp(0.015, 0.025);
    let boundary_margin = (extent * 0.008).clamp(0.001, 0.003);
    let skin = Surface::new(source, source_faces);
    let complete = Surface::new(
        source,
        if complete_faces.is_empty() {
            source_faces
        } else {
            complete_faces
        },
    );
    let source_normals = normals(source, source_faces);
    let current_normals = normals(fitted, &faces);
    let reference_normals = normals(reference, &faces);
    let semantic_unknown: BTreeSet<_> = unobserved_vertices.iter().copied().collect();
    let mut edge_counts = BTreeMap::new();
    let mut source_graph = vec![BTreeSet::new(); source.len()];
    for &[a, b, c] in source_faces {
        for (a, b) in [(a, b), (b, c), (c, a)] {
            *edge_counts.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            source_graph[a].insert(b);
            source_graph[b].insert(a);
        }
    }
    let mut boundary_distance = vec![f64::INFINITY; source.len()];
    let mut queue = VecDeque::new();
    for (&(a, b), &count) in &edge_counts {
        if count != 2 {
            for i in [a, b] {
                if boundary_distance[i] > 0. {
                    boundary_distance[i] = 0.;
                    queue.push_back(i);
                }
            }
        }
    }
    while let Some(i) = queue.pop_front() {
        for &j in &source_graph[i] {
            let next = boundary_distance[i] + (source[i] - source[j]).norm();
            if next < boundary_margin * 3. && next < boundary_distance[j] {
                boundary_distance[j] = next;
                queue.push_back(j);
            }
        }
    }
    let mut graph = vec![BTreeSet::new(); fitted.len()];
    let mut vertex_faces = vec![Vec::new(); fitted.len()];
    let mut edges = BTreeSet::new();
    for (&id, &[a, b, c]) in face_ids.iter().zip(&faces) {
        for i in [a, b, c] {
            vertex_faces[i].push(id);
        }
        for (a, b) in [(a, b), (b, c), (c, a)] {
            graph[a].insert(b);
            graph[b].insert(a);
            edges.insert((a.min(b), a.max(b)));
        }
    }
    let mut samples = vec![None; fitted.len()];
    let mut unknown = BTreeSet::new();
    let mut reason_counts = BTreeMap::<&str, usize>::new();
    for &i in &outer {
        if semantic_unknown.contains(&i) {
            unknown.insert(i);
            *reason_counts
                .entry("no_reviewed_part_correspondence")
                .or_default() += 1;
            continue;
        }
        let reason = match skin.nearest_detail(fitted[i], distance_limit, None) {
            None => Some("no_nearby_skin"),
            Some(hit) => {
                let f = source_faces[hit.triangle];
                let boundary = (0..3)
                    .map(|k| boundary_distance[f[k]].min(boundary_margin * 3.) * hit.barycentric[k])
                    .sum::<f64>();
                let normal = (0..3)
                    .map(|k| source_normals[f[k]] * hit.barycentric[k])
                    .sum::<V>()
                    .try_normalize(1e-12)
                    .unwrap_or(hit.normal);
                let visible = complete
                    .ray(
                        hit.point + normal * distance_limit,
                        -normal,
                        distance_limit * 1.1,
                    )
                    .is_some_and(|(q, _, _)| (q - hit.point).norm() < 0.001);
                if reference_normals[i].dot(&normal) < 0.25 {
                    Some("ambiguous_surface_correspondence")
                } else if boundary < boundary_margin {
                    Some("source_boundary_or_hole")
                } else if !visible {
                    Some("occluded_source_skin")
                } else {
                    samples[i] = Some(Sample {
                        point: hit.point,
                        normal,
                        signed: (fitted[i] - hit.point).dot(&normal),
                        distance: hit.distance,
                        source_face: f,
                    });
                    None
                }
            }
        };
        if let Some(reason) = reason {
            unknown.insert(i);
            *reason_counts.entry(reason).or_default() += 1;
        }
    }
    let mut flags = BTreeMap::<&str, BTreeSet<usize>>::new();
    let mut magnitudes = BTreeMap::<&str, BTreeMap<usize, f64>>::new();
    let mut add = |kind: &'static str, i: usize, magnitude: f64| {
        flags.entry(kind).or_default().insert(i);
        magnitudes
            .entry(kind)
            .or_default()
            .entry(i)
            .and_modify(|v| *v = v.max(magnitude))
            .or_insert(magnitude);
    };
    let mut distances = Vec::new();
    let mut inward = Vec::new();
    for &i in &outer {
        if let Some(sample) = samples[i] {
            distances.push(sample.distance * 1000.);
            inward.push((-sample.signed).max(0.) * 1000.);
            if sample.signed < -tolerance {
                add("inset", i, -sample.signed * 1000.);
            }
            if sample.signed > tolerance {
                add("outset", i, sample.signed * 1000.);
            }
            let nearby: BTreeSet<_> = graph[i]
                .iter()
                .copied()
                .chain(graph[i].iter().flat_map(|&j| graph[j].iter().copied()))
                .filter(|&j| {
                    j != i && (reference[j] - reference[i]).norm() < (extent * 0.08).max(0.012)
                })
                .collect();
            let local: Vec<_> = nearby
                .iter()
                .filter_map(|&j| samples[j].map(|s| s.signed))
                .collect();
            if local.len() >= 5 {
                let deviation = percentile(&local, 0.5).unwrap() - sample.signed;
                if sample.signed < -tolerance * 0.65 && deviation > tolerance * 0.5 {
                    add("dent", i, deviation * 1000.);
                }
            }
            if current_normals[i].dot(&sample.normal) < -0.15 {
                add("source_normal_reversal", i, 1.);
            }
        }
    }
    let mut high_strain = BTreeSet::new();
    let mut unsupported_strain = BTreeSet::new();
    let mut supported_large_change = 0;
    for &(a, b) in &edges {
        let rest = (reference[a] - reference[b]).norm();
        if rest < 1e-10 {
            continue;
        }
        let length = (fitted[a] - fitted[b]).norm();
        let ratio = length / rest;
        if !(0.55..=1.8).contains(&ratio) {
            high_strain.extend([a, b]);
            if let (Some(x), Some(y)) = (samples[a], samples[b]) {
                let target = (x.point - y.point).norm();
                let mismatch = length / target.max(1e-12);
                if target < rest * 0.25 || x.normal.dot(&y.normal) < 0.5 {
                    unsupported_strain.extend([a, b]);
                } else if !(0.65..=1.5).contains(&mismatch)
                    && x.distance.max(y.distance) > tolerance * 0.5
                {
                    for i in [a, b] {
                        add(
                            "stretch_mismatch",
                            i,
                            mismatch.max(1. / mismatch.max(1e-12)),
                        );
                    }
                } else {
                    supported_large_change += 1;
                }
            } else {
                unsupported_strain.extend([a, b]);
            }
        }
    }
    for &[a, b, c] in &faces {
        let before = (reference[b] - reference[a]).cross(&(reference[c] - reference[a]));
        let after = (fitted[b] - fitted[a]).cross(&(fitted[c] - fitted[a]));
        if before.norm() > 1e-12 && after.norm() < before.norm() * 0.05 {
            for i in [a, b, c] {
                add("collapsed_triangle", i, 1. - after.norm() / before.norm());
            }
        }
    }
    let before_crossings = Surface::new(reference, &faces).crossings();
    let fitted_surface = Surface::new(fitted, &faces);
    let after_crossings = fitted_surface.crossings();
    let new_crossings: Vec<_> = after_crossings
        .difference(&before_crossings)
        .copied()
        .collect();
    for pair in &new_crossings {
        for &f in pair {
            for &i in &faces[f] {
                add("self_crossing", i, 1.);
            }
        }
    }
    // Edge stretch without observed source support is a warning, not proof of
    // incorrect identity. Deliberate source-supported face proportions survive.
    if !unsupported_strain.is_empty() {
        flags.insert("unobserved_strain", unsupported_strain);
    }
    let mut issues = Vec::new();
    let mut defect_vertices = BTreeSet::<usize>::new();
    for (&kind, vertices) in &flags {
        let supported = kind != "unobserved_strain";
        if supported {
            defect_vertices.extend(vertices.iter().copied());
        }
        let mut pending = vertices.clone();
        while let Some(&seed) = pending.first() {
            pending.remove(&seed);
            let mut component = vec![seed];
            let mut index = 0;
            while index < component.len() {
                let i = component[index];
                index += 1;
                for &j in &graph[i] {
                    if pending.remove(&j) {
                        component.push(j);
                    }
                }
            }
            component.sort_unstable();
            let affected: BTreeSet<_> = component
                .iter()
                .flat_map(|&i| vertex_faces[i].iter().copied())
                .collect();
            let center = component.iter().map(|&i| fitted[i]).sum::<V>() / component.len() as f64;
            let source_triangles: BTreeSet<_> = component
                .iter()
                .filter_map(|&i| samples[i].map(|s| s.source_face))
                .collect();
            let maximum = magnitudes.get(kind).and_then(|values| {
                component
                    .iter()
                    .filter_map(|i| values.get(i))
                    .copied()
                    .reduce(f64::max)
            });
            issues.push(json!({"source_triangles":source_triangles,"kind":kind,"severity":if supported{"error"}else{"warning"},"evidence":if ["collapsed_triangle","self_crossing"].contains(&kind){"mesh_geometry"}else if supported{"source_skin"}else{"unobserved"},"vertices":component,"faces":affected,"center":crate::arr(center),"maximum":maximum}));
        }
    }
    let observed = samples.iter().flatten().count();
    let reverse: Vec<_> = source_faces
        .iter()
        .filter_map(|f| {
            if f.iter().any(|&i| boundary_distance[i] < boundary_margin) {
                return None;
            }
            let q = (source[f[0]] + source[f[1]] + source[f[2]]) / 3.;
            Some(q)
        })
        .collect();
    let reverse_distances: Vec<_> = reverse
        .iter()
        .filter_map(|&p| {
            fitted_surface
                .nearest(p, distance_limit, None)
                .map(|(_, d)| d * 1000.)
        })
        .collect();
    json!({"version":1,"status":if !defect_vertices.is_empty(){"problems_found"}else if observed==0{"insufficient_evidence"}else{"no_detected_error_in_observed_skin"},
        "geometry_hashes":{"source_skin_sha256":fingerprint(source,source_faces),"fitted_sha256":fingerprint(fitted,triangles),"reference_sha256":fingerprint(reference,triangles)},
        "auto_pass":defect_vertices.is_empty()&&unknown.is_empty(),"user_approved":false,"frame":"source-aligned Blender Z-up meters; before body assembly translation",
        "coverage":{"total_vertices":outer.len(),"observed_vertices":observed,"unobserved_vertices":unknown.len(),"observed_fraction":observed as f64/outer.len().max(1) as f64,"unobserved_reasons":reason_counts},
        "thresholds":{"surface_deviation_mm":tolerance*1000.,"source_boundary_margin_mm":boundary_margin*1000.,"correspondence_limit_mm":distance_limit*1000.,"template_edge_ratio_warning":[0.55,1.8],"source_edge_ratio_error":[0.65,1.5]},
        "surface_distance_mm":{"median":percentile(&distances,0.5),"p95":percentile(&distances,0.95),"maximum_inset":percentile(&inward,1.),"source_to_fit_p95":percentile(&reverse_distances,0.95),"source_to_fit_samples":reverse_distances.len(),"source_to_fit_unmatched_samples":reverse.len()-reverse_distances.len()},
        "defect_vertices":defect_vertices.len(),"source_supported_large_change_edges":supported_large_change,"high_strain_vertices":high_strain.len(),"new_self_crossings":new_crossings.len(),
        "issue_counts":flags.iter().map(|(k,v)|(k,v.len())).collect::<BTreeMap<_,_>>(),"issues":issues,"unobserved_vertex_ids":unknown,
        "limits":["Observed skin only: hair, source holes, occlusion and missing anatomy are not evidence of a skin defect.","A large template deformation supported by the source is not classified as erroneous stretching.","Thresholds detect geometric discrepancy, not identity or expression correctness. Coplanar overlaps and animation remain unverified."]})
}

#[cfg(test)]
mod tests {
    use super::*;
    fn patch() -> (Vec<V>, Vec<[usize; 3]>, Vec<usize>) {
        let mut p = Vec::new();
        let mut f = Vec::new();
        for y in 0..17 {
            for x in 0..17 {
                let i = y * 17 + x;
                p.push(V::new(x as f64 * 0.004, y as f64 * 0.004, 0.));
                if x < 16 && y < 16 {
                    f.extend([[i, i + 1, i + 17], [i + 1, i + 18, i + 17]]);
                }
            }
        }
        (p, f, (0..289).collect())
    }
    #[test]
    fn detects_a_dent_against_the_source_but_not_the_same_authored_concavity() {
        let (rest, f, ids) = patch();
        let mut dent = rest.clone();
        for p in &mut dent {
            let d = (p.x - 0.032).hypot(p.y - 0.032);
            p.z -= 0.006 * (-d * d / 0.00008).exp();
        }
        let bad = compare(&rest, &dent, &f, &ids, &rest, &f, &f, &[]);
        assert!(bad["issue_counts"]["dent"].as_u64().unwrap_or(0) > 0);
        assert_eq!(bad["status"], "problems_found");
        let matching = compare(&rest, &dent, &f, &ids, &dent, &f, &f, &[]);
        assert_eq!(matching["defect_vertices"], 0);
    }
    #[test]
    fn missing_source_skin_is_unknown_and_supported_stretch_is_not_an_error() {
        let (rest, f, ids) = patch();
        let stretched: Vec<_> = rest.iter().map(|p| V::new(p.x * 2., p.y, p.z)).collect();
        let matching = compare(&rest, &stretched, &f, &ids, &stretched, &f, &f, &[]);
        assert_eq!(matching["defect_vertices"], 0);
        assert!(
            matching["source_supported_large_change_edges"]
                .as_u64()
                .unwrap()
                > 0
        );
        let missing = compare(&rest, &stretched, &f, &ids, &stretched, &[], &[], &[]);
        assert_eq!(missing["status"], "insufficient_evidence");
        assert_eq!(missing["auto_pass"], false);
    }
    #[test]
    fn detects_local_stretch_against_observed_skin() {
        let (rest, f, ids) = patch();
        let mut fitted = rest.clone();
        fitted[144].z += 0.014;
        let report = compare(&rest, &fitted, &f, &ids, &rest, &f, &f, &[]);
        assert!(
            report["issue_counts"]["stretch_mismatch"]
                .as_u64()
                .unwrap_or(0)
                > 0
        );
        assert_eq!(report["auto_pass"], false);
        assert!(report["issues"].as_array().unwrap().iter().any(|issue| {
            issue["kind"] == "stretch_mismatch"
                && !issue["source_triangles"].as_array().unwrap().is_empty()
        }));
    }
    #[test]
    fn holes_and_occluding_layers_do_not_become_false_dent_evidence() {
        let (rest, faces, ids) = patch();
        let mut dent = rest.clone();
        dent[144].z -= 0.006;
        let open: Vec<_> = faces
            .iter()
            .copied()
            .filter(|f| {
                let center = f.iter().map(|&i| rest[i]).sum::<V>() / 3.;
                (center.x - 0.032).hypot(center.y - 0.032) > 0.018
            })
            .collect();
        let hole = compare(&rest, &dent, &faces, &ids, &rest, &open, &open, &[]);
        assert!(
            hole["unobserved_vertex_ids"]
                .as_array()
                .unwrap()
                .contains(&json!(144))
        );
        assert!(hole["issue_counts"]["dent"].as_u64().unwrap_or(0) == 0);
        let mut layered = rest.clone();
        layered.extend(rest.iter().map(|p| p + V::z() * 0.008));
        let mut occlusion = faces.clone();
        occlusion.extend(faces.iter().map(|f| f.map(|i| i + rest.len())));
        let hidden = compare(
            &rest,
            &dent,
            &faces,
            &ids,
            &layered,
            &faces,
            &occlusion,
            &[],
        );
        assert_eq!(hidden["defect_vertices"], 0);
        assert_eq!(hidden["status"], "insufficient_evidence");
        assert!(
            hidden["coverage"]["unobserved_reasons"]["occluded_source_skin"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert_eq!(hidden["auto_pass"], false);
    }
    #[test]
    fn missing_source_cannot_hide_a_collapsed_triangle() {
        let (rest, faces, ids) = patch();
        let mut fitted = rest.clone();
        fitted[144] = fitted[145];
        let report = compare(&rest, &fitted, &faces, &ids, &[], &[], &[], &[]);
        assert!(
            report["issue_counts"]["collapsed_triangle"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert_eq!(report["coverage"]["observed_vertices"], 0);
        assert_eq!(report["status"], "problems_found");
    }
}
