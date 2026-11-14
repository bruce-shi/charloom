//! Landmark-initialized parametric and constrained nonrigid head registration.
//! This crate has no image, Blender, LLM, or GLTF dependency. Inputs retain the
//! canonical full-body vertex indices; unobserved anatomy is a prior, not evidence.
#[cfg(test)]
mod annotation_tests;
mod attachment;
pub mod authoring;
pub mod body_guidance;
pub mod body_warp;
pub mod browser_mesh;
mod browser_remesh;
#[cfg(test)]
mod contact;
mod deformation_guard;
mod ears;
pub mod face_quality;
#[cfg(test)]
mod flow;
#[cfg(test)]
mod membrane;
mod parallel;
mod registration;
pub mod shape_hints;
mod surface;
mod wrap;
use crate::parallel::*;
use anyhow::{Context, Result, ensure};
use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use surface::{Surface, V};

#[derive(Deserialize, schemars::JsonSchema)]
pub struct Morph {
    pub name: String,
    pub deltas: Vec<(usize, f64, f64, f64)>,
}
#[derive(Deserialize, Default, schemars::JsonSchema)]
pub struct FaceRegions {
    pub outer: Vec<usize>,
    pub cavities: Vec<Vec<usize>>,
    /// Authored pinnae transported rigidly with the skull, excluded from face/hair projection.
    #[serde(default)]
    pub scalp_preserve: Vec<usize>,
    #[serde(default)]
    pub bindings: Vec<CageBinding>,
    #[serde(default)]
    pub unbound: Vec<usize>,
}
#[derive(Deserialize, schemars::JsonSchema)]
pub struct CageBinding {
    pub vertex: usize,
    pub cage: [usize; 4],
    pub weights: [f64; 4],
}
#[derive(Deserialize, schemars::JsonSchema)]
pub struct NeckGirth {
    #[serde(default = "one")]
    pub width_scale: f64,
    #[serde(default = "one")]
    pub depth_scale: f64,
}
impl Default for NeckGirth {
    fn default() -> Self {
        Self {
            width_scale: 1.,
            depth_scale: 1.,
        }
    }
}
#[derive(Deserialize, schemars::JsonSchema)]
pub struct Input {
    #[serde(default)]
    #[schemars(required)]
    pub anatomy: Option<charloom_core::anatomy::AnatomyReview>,
    pub id: String,
    pub template: Vec<[f64; 3]>,
    pub body_fit: Vec<[f64; 3]>,
    pub triangles: Vec<[usize; 3]>,
    pub body_ids: Vec<usize>,
    pub source: Vec<[f64; 3]>,
    pub source_triangles: Vec<[usize; 3]>,
    #[serde(default)]
    pub hair_triangles: Vec<[usize; 3]>,
    pub landmark_triangles: Vec<[usize; 3]>,
    pub morphs: Vec<Morph>,
    pub groups: BTreeMap<String, Vec<usize>>,
    #[serde(default)]
    pub face_regions: FaceRegions,
    #[serde(default)]
    pub neck_girth: NeckGirth,
}
#[derive(Deserialize, schemars::JsonSchema)]
pub struct Annotation {
    pub name: String,
    pub template_xz: [f64; 2],
    #[serde(default)]
    pub template_vertex: Option<usize>,
    pub source_xz: [f64; 2],
         #[serde(default)]
    pub source_y: Option<f64>,
    #[serde(default = "one")]
    pub weight: f64,
}
fn one() -> f64 {
    1.
}
#[derive(Serialize, Deserialize)]
pub struct Landmark {
    pub name: String,
    pub vertex: usize,
    pub original: [f64; 3],
    pub observed: [f64; 3],
    pub weight: f64,
}
#[derive(Serialize, Deserialize)]
pub struct Output {
    pub schema_version: u32,
    pub id: String,
    pub aligned: Vec<[f64; 3]>,
    pub parametric: Vec<[f64; 3]>,
    pub fitted: Vec<[f64; 3]>,
    pub assembled: Vec<[f64; 3]>,
    pub morph_weights: BTreeMap<String, f64>,
    pub landmarks: Vec<Landmark>,
    pub report: serde_json::Value,
}
fn arr(v: V) -> [f64; 3] {
    [v.x, v.y, v.z]
}
fn smooth(x: f64) -> f64 {
    let x = x.clamp(0., 1.);
    x * x * (3. - 2. * x)
}
fn normals(points: &[V], faces: &[[usize; 3]]) -> Vec<V> {
    let mut n = vec![V::zeros(); points.len()];
    for &[a, b, c] in faces {
        let v = (points[b] - points[a]).cross(&(points[c] - points[a]));
        for i in [a, b, c] {
            n[i] += v;
        }
    }
    for v in &mut n {
        *v = v.try_normalize(1e-15).unwrap_or(V::z());
    }
    n
}

fn front_landmark(surface: &Surface, xz: [f64; 2]) -> Option<V> {
    if let Some(p) = surface.front(xz[0], xz[1]) {
        return Some(p);
    }
    // Keep the legacy silhouette calibration tolerance for 2D observations.
    for radius in [0.0005, 0.001, 0.002] {
        for (dx, dz) in [(0., -1.), (0., 1.), (-1., 0.), (1., 0.)] {
            if let Some(p) = surface.front(xz[0] + dx * radius, xz[1] + dz * radius) {
                return Some(p);
            }
        }
    }
    None
}

fn observed_landmark(surface: &Surface, annotation: &Annotation) -> Result<V> {
    ensure!(
        annotation.source_xz.iter().all(|v| v.is_finite())
            && annotation.source_y.is_none_or(|v| v.is_finite()),
        "Invalid source landmark coordinates"
    );
    if let Some(y) = annotation.source_y {
        let p = V::new(annotation.source_xz[0], y, annotation.source_xz[1]);
        return surface
            .nearest(p, 0.003, None)
            .map(|(p, _)| p)
            .with_context(|| {
                format!(
                    "{}: 3D source landmark is more than 3 mm from observed skin",
                    annotation.name
                )
            });
    }
    front_landmark(surface, annotation.source_xz)
        .with_context(|| format!("{}: source landmark misses observed skin", annotation.name))
}

/// Similarity registration separates pose from identity and disallows reflection.
fn similarity(from: &[V], to: &[V], weights: &[f64]) -> (Matrix3<f64>, V) {
    let total = weights.iter().sum::<f64>();
    let mean = |p: &[V]| {
        p.iter()
            .zip(weights)
            .fold(V::zeros(), |s, (v, w)| s + v * *w)
            / total
    };
    let a = mean(from);
    let b = mean(to);
    let mut cov = Matrix3::zeros();
    let mut denominator = 0.;
    for ((p, q), w) in from.iter().zip(to).zip(weights) {
        cov += (q - b) * (p - a).transpose() * *w;
        denominator += (p - a).norm_squared() * *w;
    }
    let svd = cov.svd(true, true);
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    let mut d = Matrix3::identity();
    d[(2, 2)] = (u * vt).determinant().signum();
    let rot = u * d * vt;
    let scale = from
        .iter()
        .zip(to)
        .zip(weights)
        .map(|((p, q), w)| (q - b).dot(&(rot * (p - a))) * *w)
        .sum::<f64>()
        / denominator.max(1e-12);
    let matrix = rot * scale.clamp(0.5, 2.);
    (matrix, b - matrix * a)
}

pub fn solve(input: Input, annotations: Vec<Annotation>) -> Result<Output> {
    let began = parallel::Timer::start();
    let anatomy = charloom_core::anatomy::preflight(&charloom_core::anatomy::AnatomyRequest {
        review: input.anatomy.clone(),
        source_hash: body_guidance::fingerprint_geometry(&input.source, &input.source_triangles),
        prior_hash: body_guidance::fingerprint_geometry(&input.template, &input.triangles),
        usage: charloom_core::anatomy::AnatomyUse::FitFace,
        preserved_part_ids: vec![],
    })?;
    let n = input.template.len();
    ensure!(n == input.body_fit.len() && n > 3, "template size mismatch");
    ensure!(
        annotations.len() >= 8,
        "At least eight named facial landmarks are required"
    );
    ensure!(
        input
            .template
            .iter()
            .chain(&input.body_fit)
            .chain(&input.source)
            .flatten()
            .all(|x| x.is_finite()),
        "nonfinite input"
    );
    ensure!(
        input.triangles.iter().flatten().all(|&i| i < n)
            && input
                .source_triangles
                .iter()
                .flatten()
                .all(|&i| i < input.source.len()),
        "invalid face index"
    );
    ensure!(input.body_ids.iter().all(|&i| i < n), "invalid body index");
    let original: Vec<V> = input.template.iter().map(|v| V::from(*v)).collect();
    let body: Vec<V> = input.body_fit.iter().map(|v| V::from(*v)).collect();
    let source: Vec<V> = input.source.iter().map(|v| V::from(*v)).collect();
    ensure!(
        input
            .hair_triangles
            .iter()
            .flatten()
            .all(|&i| i < source.len()),
        "invalid hair index"
    );
    let hair_surface = Surface::new(&source, &input.hair_triangles);
    // Upstream masks may have been revised independently. Hair and reliable
    // skin observations must be mutually exclusive even for legacy bundles.
    let hair_keys: BTreeSet<_> = input
        .hair_triangles
        .iter()
        .map(|f| {
            let mut key = *f;
            key.sort();
            key
        })
        .collect();
    let skin_triangles: Vec<_> = input
        .source_triangles
        .iter()
        .filter(|f| {
            let mut key = **f;
            key.sort();
            !hair_keys.contains(&key)
        })
        .copied()
        .collect();
    let excluded_hair_from_skin_matches = input.source_triangles.len() - skin_triangles.len();
    let surface = Surface::new(&source, &skin_triangles);
    ensure!(
        input
            .landmark_triangles
            .iter()
            .flatten()
            .all(|&i| i < source.len()),
        "invalid landmark surface index"
    );
     
    let landmark_surface = Surface::new(&source, &input.landmark_triangles);
    let original_surface = Surface::new(&original, &input.triangles);
    let lowest_annotation = annotations
        .iter()
        .map(|a| a.template_xz[1])
        .fold(f64::INFINITY, f64::min);
    let mut exterior = vec![false; n];
    let visible: Vec<_> = input
        .body_ids
        .par_iter()
        .map(|&i| {
            let p = original[i];
            (
                i,
                p.z > lowest_annotation - 0.08
                    && original_surface
                        .front(p.x, p.z)
                        .is_some_and(|q| p.y - q.y < 0.0015),
            )
        })
        .collect();
    for (i, value) in visible {
        exterior[i] = value;
    }
    let mut landmarks = Vec::new();
    let mut used = BTreeSet::new();
    for a in annotations {
        ensure!(
            a.weight > 0. && a.weight.is_finite(),
            "Invalid landmark weight"
        );
        let observed = observed_landmark(&landmark_surface, &a)?;
        let hint = front_landmark(&original_surface, a.template_xz).unwrap_or(V::new(
            a.template_xz[0],
            -0.13,
            a.template_xz[1],
        ));
        let automatic_vertex = *input
            .body_ids
            .iter()
            .filter(|i| !used.contains(*i) && exterior[**i])
            .min_by(|&&i, &&j| {
                let dist = |v: V| {
                    (v.x - a.template_xz[0]).powi(2)
                        + (v.z - a.template_xz[1]).powi(2)
                        + 0.12 * (v.y - hint.y).powi(2)
                };
                dist(original[i]).total_cmp(&dist(original[j]))
            })
            .context("No template landmark vertex")?;
        let vertex = a.template_vertex.unwrap_or(automatic_vertex);
        ensure!(
            vertex < n && input.body_ids.contains(&vertex) && !used.contains(&vertex),
            "Invalid or repeated template landmark"
        );
        ensure!(
            (original[vertex] - hint).norm() < 0.035,
            "{}: landmark is too far from template surface",
            a.name
        );
        used.insert(vertex);
        landmarks.push(Landmark {
            name: a.name,
            vertex,
            original: arr(original[vertex]),
            observed: arr(observed),
            weight: a.weight,
        });
    }
    let chin = landmarks
        .iter()
        .find(|l| l.name == "chin")
        .context("chin landmark is required")?;
    let forehead = landmarks
        .iter()
        .find(|l| l.name == "forehead")
        .context("forehead landmark is required")?;
    let neck_start = chin.original[2] - 0.085;
    let neck_end = chin.original[2] - 0.005;
    let from: Vec<V> = landmarks.iter().map(|l| V::from(l.original)).collect();
    let to: Vec<V> = landmarks.iter().map(|l| V::from(l.observed)).collect();
    let lm_weights: Vec<f64> = landmarks.iter().map(|l| l.weight).collect();
    let (matrix, translation) = similarity(&from, &to, &lm_weights);
    let blend = attachment::neck_blend(&original, neck_start, neck_end);
    let head: Vec<V> = original.iter().map(|p| matrix * p + translation).collect();
    let (base, neck_alignment) = attachment::initial_neck(
        &original,
        &body,
        &head,
        matrix.column(2).into_owned(),
        &blend,
        &input.body_ids,
        &input.triangles,
        neck_start,
        neck_end,
    )?;
    let source_low = chin.observed[2] - 0.012;
    let source_high = forehead.observed[2] + 0.030;
    let max_x = landmarks
        .iter()
        .map(|l| l.observed[0].abs())
        .fold(0., f64::max)
        + 0.014;
    let front_y = landmarks
        .iter()
        .map(|l| l.observed[1])
        .fold(f64::INFINITY, f64::min);
    let rear_y = landmarks
        .iter()
        .map(|l| l.observed[1])
        .fold(f64::NEG_INFINITY, f64::max)
        + 0.025;
    let active: Vec<bool> = (0..n).map(|i| blend[i] > 0.).collect();
    let eligible: Vec<bool> = base
        .iter()
        .enumerate()
        .map(|(i, p)| {
            active[i]
                && exterior[i]
                && p.z > source_low
                && p.z < source_high
                && p.x.abs() < max_x
                && p.y < rear_y
                && p.y > front_y - 0.05
        })
        .collect();
    let mut is_body = vec![false; n];
    for &i in &input.body_ids {
        is_body[i] = true;
    }
    // Every target displacement is transformed by the fitted head frame. Targets
    // are bounded and retain their official names; no inferred face is baked in.
    let mut basis = Vec::new();
    let mut names = Vec::new();
    for m in &input.morphs {
        // Pose is already solved by registration. Do not explain identity using
        // whole-head translations/tilts or re-age the cranial prior.
        if m.name.starts_with("head/head-trans-")
            || m.name.starts_with("head/head-angle-")
            || m.name.starts_with("head/head-age-")
        {
            continue;
        }
        ensure!(
            m.deltas
                .iter()
                .all(|(i, x, y, z)| *i < n && [x, y, z].iter().all(|v| v.is_finite())),
            "invalid morph {}",
            m.name
        );
        let entries: Vec<_> = m
            .deltas
            .iter()
            .filter(|(i, _, _, _)| active[*i])
            .map(|&(i, x, y, z)| (i, matrix * V::new(x, y, z) * blend[i]))
            .collect();
        if entries
            .iter()
            .any(|(i, v)| is_body[*i] && v.norm() > 0.00001)
        {
            basis.push(entries);
            names.push(m.name.clone());
        }
    }
    let mut coefficients = vec![0.; basis.len()];
    let mut parameter = base.clone();
    let repair_neck_root = neck_alignment["compression_repair_applied"]
        .as_bool()
        .unwrap_or(false);
    let mut trace = vec![neck_alignment];
    let targets = |p: &[V], limit: f64| -> (Vec<f64>, Vec<V>, usize) {
        let ns = normals(p, &input.triangles);
        let hits: Vec<_> = (0..n)
            .into_par_iter()
            .map(|i| {
                if !eligible[i] || !is_body[i] {
                    return None;
                }
                surface.nearest(p[i], limit, Some(ns[i]))
            })
            .collect();
        let mut w = vec![0.; n];
        let mut rhs = vec![V::zeros(); n];
        let mut count = 0;
        for (i, h) in hits.into_iter().enumerate() {
            if let Some((q, d)) = h {
                let weight = (0.006 / d.max(0.006)).max(0.08);
                w[i] = weight;
                rhs[i] = q * weight;
                count += 1;
            }
        }
        for l in &landmarks {
            let weight = 80. * l.weight;
            w[l.vertex] += weight;
            rhs[l.vertex] += V::from(l.observed) * weight;
        }
        (w, rhs, count)
    };
    for iteration in 0..12 {
        let (mut w, mut rhs, count) =
            targets(&parameter, if iteration < 3 { 0.055 } else { 0.028 });
        for i in 0..n {
            if is_body[i] && active[i] {
                let prior = if eligible[i] { 0.20 } else { 3. };
                w[i] += prior;
                rhs[i] += base[i] * prior;
            }
        }
        // Bounded coordinate descent on the current robust point correspondences.
        // Updating the residual in place exploits sparse official morph support.
        let mut residual: Vec<V> = (0..n).map(|i| parameter[i] * w[i] - rhs[i]).collect();
         
        let reg = 0.002;
        let diagonal: Vec<f64> = basis
            .iter()
            .map(|entries| {
                entries
                    .iter()
                    .map(|(i, v)| v.norm_squared() * w[*i])
                    .sum::<f64>()
                    + reg
            })
            .collect();
        for _ in 0..24 {
            let mut largest = 0f64;
            for (k, entries) in basis.iter().enumerate() {
                let g = entries
                    .iter()
                    .map(|(i, v)| v.dot(&residual[*i]))
                    .sum::<f64>()
                    + reg * coefficients[k];
                let next = (coefficients[k] - g / diagonal[k]).clamp(0., 1.);
                let delta = next - coefficients[k];
                coefficients[k] = next;
                largest = largest.max(delta.abs());
                if delta.abs() > 1e-10 {
                    for &(i, v) in entries {
                        parameter[i] += v * delta;
                        residual[i] += v * (delta * w[i]);
                    }
                }
            }
            if largest < 1e-5 {
                break;
            }
        }
        trace.push(serde_json::json!({"stage":"parameters","iteration":iteration,"observed_vertices":count,"landmark_rmse_mm":landmark_rmse(&parameter,&landmarks)}));
        eprintln!(
            "parameters {iteration}: {:.2} mm",
            landmark_rmse(&parameter, &landmarks)
        );
    }
    let (mut fitted, wrap_trace) = wrap::fit(
        &base,
        &input.triangles,
        &input.body_ids,
        &eligible,
        &blend,
        &landmarks,
        &surface,
        &landmark_surface,
    );
    trace.extend(wrap_trace);
    if !input.face_regions.outer.is_empty() {
        let regions = &input.face_regions;
        let outer: BTreeSet<_> = regions.outer.iter().copied().collect();
        ensure!(!regions.cavities.is_empty(), "missing cavity anatomy");
        ensure!(
            !regions.bindings.is_empty(),
            "missing rest-space cage bindings; regenerate template anatomy"
        );
        for b in &regions.bindings {
            ensure!(
                b.vertex < n
                    && !outer.contains(&b.vertex)
                    && b.cage.iter().all(|i| outer.contains(i))
                    && b.weights.iter().all(|w| w.is_finite() && *w >= -1e-7)
                    && (b.weights.iter().sum::<f64>() - 1.).abs() < 1e-8,
                "invalid interior cage binding"
            );
            let rest: V = (0..4).map(|k| original[b.cage[k]] * b.weights[k]).sum();
            ensure!(
                (rest - original[b.vertex]).norm() < 1e-8,
                "cage embedding does not reproduce template"
            );
        }
        ensure!(
            regions.unbound.iter().all(|&i| i < n),
            "invalid unbound index"
        );
        ensure!(
            outer.len() == regions.outer.len() && outer.iter().all(|&i| i < n && is_body[i]),
            "invalid outer anatomy"
        );
        ensure!(
            regions.scalp_preserve.iter().all(|i| outer.contains(i)),
            "invalid scalp preservation patch"
        );
        for cavity in &regions.cavities {
            ensure!(
                cavity.iter().all(|&i| i < n && is_body[i])
                    && cavity.iter().filter(|i| outer.contains(i)).count() >= 3,
                "invalid cavity anatomy"
            );
        }
        ensure!(
            input.groups.values().flatten().all(|&i| i < n),
            "invalid helper index"
        );
        let (refined, refinement_trace) = registration::fit(
            &base,
            &original,
            &fitted,
            &parameter,
            &input.triangles,
            regions,
            &blend,
            &landmarks,
            &surface,
            &landmark_surface,
            &hair_surface,
            neck_start,
            &input.groups,
        );
        fitted = refined;
        trace.extend(refinement_trace);
    }
    let distance_stats = |p: &[V]| {
        let mut d: Vec<_> = input
            .body_ids
            .par_iter()
            .filter(|&&i| eligible[i])
            .filter_map(|&i| surface.nearest(p[i], 0.08, None).map(|(_, d)| d * 1000.))
            .collect();
        d.sort_by(f64::total_cmp);
        if d.is_empty() {
            return serde_json::json!({"count":0});
        }
        serde_json::json!({"count":d.len(),"mean_mm":d.iter().sum::<f64>()/d.len() as f64,"p95_mm":d[(d.len()-1)*95/100]})
    };
    let maximum_below_neck = (0..n)
        .filter(|&i| blend[i] == 0.)
        .map(|i| (fitted[i] - body[i]).norm())
        .fold(0., f64::max);
    ensure!(
        maximum_below_neck < 1e-12,
        "unrelated body vertices changed"
    );
    ensure!(
        fitted.iter().all(|p| p.iter().all(|x| x.is_finite())),
        "nonfinite fitted geometry"
    );
    let ear_report = trace
        .iter()
        .rev()
        .find(|step| step["stage"] == "ear_shape_preservation")
        .cloned();
    let temple_report = trace
        .iter()
        .rev()
        .find(|step| step["stage"] == "temple_transition")
        .cloned();
    let report = serde_json::json!({"anatomy":anatomy,"method":if input.face_regions.outer.is_empty() {"Rust coarse face sheet (legacy input without cavity regions)"} else {"Rust free anatomical surface fit with a parametric shape prior and cavity transport"},"seconds":began.seconds(),"ear_preservation":ear_report,"temple_transition":temple_report,
      "landmark_initialization":"reviewed source image annotations projected in Rust, paired semantic standard-template indices","landmark_count":landmarks.len(),"excluded_hair_from_skin_matches":excluded_hair_from_skin_matches,
      "landmark_rmse_mm":{"aligned":landmark_rmse(&base,&landmarks),"parametric":landmark_rmse(&parameter,&landmarks),"fitted":landmark_rmse(&fitted,&landmarks)},
      "surface_distance":{"aligned":distance_stats(&base),"parametric":distance_stats(&parameter),"fitted":distance_stats(&fitted)},
      "template_vertices":n,"template_triangles":input.triangles.len(),"topology_unchanged":true,"source_head_stitched":false,
      "maximum_below_neck_change_m":maximum_below_neck,"neck_blend_original_z":[neck_start,neck_end],"deformation":if input.face_regions.outer.is_empty(){"bounded facial chart warp"}else{"local rotation/scale shape constraints, free observed-surface fitting, geometry backoff, tetrahedral interior transport"},
      "normals_preserve_reference_hemisphere":valid_triangles(&base,&fitted,&input.triangles),"user_approved":false,"animation_validated":false,
      "limits":["Observed-surface distances are fit diagnostics, not identity scores.","Hidden scalp and mouth interiors remain template estimates.","Expression and eyelid collision validation is still required."],"trace":trace});
    let (mut assembled, head_offset) = attachment::assemble(&body, &fitted, &blend, chin.vertex);
    let mut report = report;
    report["attachment"] = serde_json::json!({"method":"rigid head assembly toward body-prior chin height, never raising the source head; smooth neck transition", "head_translation_m":arr(head_offset),"requested_body_prior_translation_m":body[chin.vertex].z-fitted[chin.vertex].z,"upward_translation_clamped":body[chin.vertex].z>fitted[chin.vertex].z,"source_frame_fitted_preserved":true,"source_hair_translation_m":arr(head_offset),"maximum_below_neck_change_m":maximum_below_neck,"anatomical_proportion_approved":false});
    if repair_neck_root {
        let root = attachment::relax_neck_root(
            &body,
            &original,
            &mut assembled,
            &input.triangles,
            &input.face_regions,
            &landmarks,
        )?;
        let start = root["start_z"].as_f64().unwrap();
        let maximum = original
            .iter()
            .enumerate()
            .filter(|(_, p)| p.z <= start)
            .map(|(i, _)| (assembled[i] - body[i]).norm())
            .fold(0., f64::max);
        ensure!(maximum < 1e-12, "Neck-root correction changed lower body");
        report["source_frame_neck_blend_original_z"] = report["neck_blend_original_z"].clone();
        report["neck_blend_original_z"][0] = serde_json::json!(start);
        report["attachment"]["neck_root"] = root;
        report["attachment"]["maximum_below_neck_change_m"] = serde_json::json!(maximum);
    }
    report["attachment"]["neck_girth"] = attachment::neck_girth(
        &original,
        &mut assembled,
        &input.body_ids,
        &input.triangles,
        &landmarks,
        &input.face_regions,
        &input.neck_girth,
    )?;
    let source_frame_final: Vec<_> = assembled.iter().map(|p| p - head_offset).collect();
    report["source_comparison"] = face_quality::compare(
        &base,
        &source_frame_final,
        &input.triangles,
        &input.face_regions.outer,
        &source,
        &skin_triangles,
        &input.landmark_triangles,
        &input.face_regions.scalp_preserve,
    );
    report["quality_status"] = serde_json::json!(if report["source_comparison"]["status"]
        == "problems_found"
    {
        "problems_found"
    } else {
        "needs_review"
    });
    report["source_comparison"]["evaluated_stage"] =
        serde_json::json!("final_assembled_head_with_assembly_translation_removed");
    report["seconds"] = serde_json::json!(began.seconds());
    Ok(Output {
        schema_version: 1,
        id: input.id,
        aligned: base.into_iter().map(arr).collect(),
        parametric: parameter.into_iter().map(arr).collect(),
        fitted: fitted.into_iter().map(arr).collect(),
        assembled: assembled.into_iter().map(arr).collect(),
        morph_weights: names
            .into_iter()
            .zip(coefficients)
            .filter(|(_, v)| *v > 1e-5)
            .collect(),
        landmarks,
        report,
    })
}

fn landmark_rmse(p: &[V], landmarks: &[Landmark]) -> f64 {
    (landmarks
        .iter()
        .map(|l| (p[l.vertex] - V::from(l.observed)).norm_squared() * l.weight)
        .sum::<f64>()
        / landmarks.iter().map(|l| l.weight).sum::<f64>())
    .sqrt()
        * 1000.
}
fn valid_triangles(reference: &[V], p: &[V], faces: &[[usize; 3]]) -> bool {
    faces.iter().all(|&[a, b, c]| {
        let old = (reference[b] - reference[a]).cross(&(reference[c] - reference[a]));
        let new = (p[b] - p[a]).cross(&(p[c] - p[a]));
        old.norm_squared() < 1e-20
            || (new.norm_squared() > old.norm_squared() * 0.02 && old.dot(&new) > 0.)
    })
}
#[cfg(test)]
fn cg(
    start: &[V],
    weights: &[f64],
    rhs: &[V],
    edges: &[(usize, usize)],
    lambda: f64,
    iterations: usize,
) -> Vec<V> {
    let apply = |p: &[V]| {
        let mut v: Vec<_> = p.iter().zip(weights).map(|(x, w)| x * *w).collect();
        for &(a, b) in edges {
            let d = (p[a] - p[b]) * lambda;
            v[a] += d;
            v[b] -= d;
        }
        v
    };
    let mut diagonal = weights.to_vec();
    for &(a, b) in edges {
        diagonal[a] += lambda;
        diagonal[b] += lambda;
    }
    let mut x = start.to_vec();
    let ax = apply(&x);
    let mut r: Vec<_> = rhs.iter().zip(ax).map(|(b, a)| b - a).collect();
    let mut z: Vec<_> = r.iter().zip(&diagonal).map(|(r, d)| r / *d).collect();
    let mut p = z.clone();
    let dot = |a: &[V], b: &[V]| a.iter().zip(b).map(|(a, b)| a.dot(b)).sum::<f64>();
    let mut rz = dot(&r, &z);
    for _ in 0..iterations {
        if rz < 1e-20 {
            break;
        }
        let ap = apply(&p);
        let alpha = rz / dot(&p, &ap).max(1e-30);
        for i in 0..x.len() {
            x[i] += p[i] * alpha;
            r[i] -= ap[i] * alpha;
            z[i] = r[i] / diagonal[i];
        }
        let next = dot(&r, &z);
        let beta = next / rz;
        for i in 0..p.len() {
            p[i] = z[i] + p[i] * beta;
        }
        rz = next;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn similarity_recovers_pose_without_reflecting() {
        let a = vec![
            V::new(0., 0., 0.),
            V::new(1., 0., 0.),
            V::new(0., 1., 0.),
            V::new(0., 0., 1.),
        ];
        let rot = nalgebra::Rotation3::from_euler_angles(0.1, 0.2, -0.3).into_inner();
        let offset = V::new(2., 3., 4.);
        let b: Vec<_> = a.iter().map(|p| rot * p * 1.3 + offset).collect();
        let (m, t) = similarity(&a, &b, &[1.; 4]);
        assert!(m.determinant() > 0.);
        assert!(a.iter().zip(b).all(|(p, q)| (m * p + t - q).norm() < 1e-10));
    }
    #[test]
    fn deformation_keeps_pinned_boundary_and_rejects_fold() {
        let rest = vec![V::new(0., 0., 0.), V::new(1., 0., 0.), V::new(0., 1., 0.)];
        let weights = [1e12, 1., 1.];
        let rhs = vec![rest[0] * 1e12, V::new(1., 0., 0.2), V::new(0., 1., 0.2)];
        let result = cg(&rest, &weights, &rhs, &[(0, 1), (0, 2), (1, 2)], 0.1, 40);
        assert!((result[0] - rest[0]).norm() < 1e-11);
        assert!(result[1].z > 0.1);
        assert!(!valid_triangles(
            &rest,
            &[rest[0], rest[2], rest[1]],
            &[[0, 1, 2]]
        ));
    }
}
