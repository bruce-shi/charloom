//! Sampled noncoplanar surface-crossing diagnostics. Regions select where to
//! look; they do not pretend that a fused mesh has separate clothing layers.
use anyhow::{Context, Result, ensure};
use charloom_core::{Asset, Vec3, v3};
use charloom_io::{load_gltf, write_json};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::Path};

use crate::{
    surface_query::SurfaceQuery,
    surface_region::{SurfaceMask, validate_mask},
};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct SurfaceContactConfig {
    pub regions: Vec<SurfaceMask>,
    pub clip: Option<String>,
    pub times: Vec<f64>,
    pub epsilon_m: f64,
    pub maximum_reported_pairs: usize,
}
impl Default for SurfaceContactConfig {
    fn default() -> Self {
        Self {
            regions: vec![],
            clip: None,
            times: vec![],
            epsilon_m: 0.000001,
            maximum_reported_pairs: 200,
        }
    }
}
fn points_for_pair(asset: &Asset, positions: &[Vec<Vec3>], pair: &[[usize; 2]; 2]) -> Value {
    json!(pair.map(|[m, t]| asset.meshes[m].triangles[t].map(|v| positions[m][v as usize])))
}

pub fn audit_asset(asset: &Asset, config: &SurfaceContactConfig) -> Result<Value> {
    ensure!(
        config.epsilon_m.is_finite() && config.epsilon_m > 0.0 && config.epsilon_m <= 0.001,
        "invalid_contact_epsilon"
    );
    ensure!(
        !config.regions.is_empty(),
        "surface_contacts_requires_explicit_regions"
    );
    ensure!(
        config.times.len() <= 64 && config.maximum_reported_pairs <= 10000,
        "contact_report_budget_exceeded"
    );
    for region in &config.regions {
        validate_mask(region, &asset.meshes)?;
    }
    let mut selected = BTreeSet::new();
    for region in &config.regions {
        let vertices: BTreeSet<_> = region.vertices.iter().copied().collect();
        for (t, tri) in asset.meshes[region.mesh].triangles.iter().enumerate() {
            if tri.iter().any(|v| vertices.contains(&(*v as usize))) {
                selected.insert([region.mesh, t]);
            }
        }
    }
    let rest = asset.world_meshes()?;
    let baseline =
        SurfaceQuery::new(&asset.meshes, &rest).crossing_pairs(&selected, config.epsilon_m);
    let baseline_examples: Vec<_> = baseline
        .iter()
        .take(config.maximum_reported_pairs)
        .map(|pair| json!({"triangles":pair,"points":points_for_pair(asset,&rest,pair)}))
        .collect();
    let mut samples = Vec::new();
    if let Some(name) = &config.clip {
        let clips: Vec<_> = asset.clips.iter().filter(|c| c.name == *name).collect();
        ensure!(clips.len() == 1, "surface_contacts_clip_must_be_unique");
        ensure!(
            !config.times.is_empty(),
            "surface_contacts_requires_explicit_times"
        );
        let clip = clips.first().context("surface_contacts_clip_missing")?;
        let [start, end] = clip.time_range();
        for &time in &config.times {
            ensure!(
                time.is_finite() && time >= start && time <= end,
                "surface_contacts_time_out_of_range"
            );
            let positions = asset.deformed_positions(
                &clip.sample(&asset.skeleton, time)?,
                &clip.sample_morphs(time)?,
            )?;
            let pairs = SurfaceQuery::new(&asset.meshes, &positions)
                .crossing_pairs(&selected, config.epsilon_m);
            let mut minimum_area_ratio = f64::INFINITY;
            let mut collapsed_triangles = 0;
            for &[m, t] in &selected {
                let area = |ps: &[Vec<Vec3>]| {
                    let [a, b, c] = asset.meshes[m].triangles[t].map(|v| v3(ps[m][v as usize]));
                    (b - a).cross(&(c - a)).norm()
                };
                let reference_area = area(&rest);
                if reference_area > 1e-12 {
                    let ratio = area(&positions) / reference_area;
                    minimum_area_ratio = minimum_area_ratio.min(ratio);
                    if ratio < 0.001 {
                        collapsed_triangles += 1;
                    }
                }
            }
            let newly_crossing: Vec<_> = pairs.difference(&baseline).collect();
            let examples:Vec<_>=newly_crossing.iter().take(config.maximum_reported_pairs).map(|pair|json!({"triangles":pair,"points":points_for_pair(asset,&positions,pair)})).collect();
            samples.push(json!({"time_s":time,"crossing_pairs":pairs.len(),"new_crossing_pairs":newly_crossing.len(),"collapsed_triangles":collapsed_triangles,"minimum_area_ratio":if minimum_area_ratio.is_finite(){Some(minimum_area_ratio)}else{None},"examples_truncated":newly_crossing.len()>config.maximum_reported_pairs,"examples":examples}));
        }
    } else {
        ensure!(
            config.times.is_empty(),
            "surface_contacts_times_require_clip"
        );
    }
    let summary = json!({"source_hash":asset.source_hash,"status":"diagnostic_only","method":"bvh_strict_noncoplanar_triangle_crossings_v1","selected_triangles":selected.len(),"epsilon_m":config.epsilon_m,"baseline_crossing_pairs":baseline.len(),"baseline_examples_truncated":baseline.len()>config.maximum_reported_pairs,"baseline_examples":baseline_examples,"clip":config.clip,"samples":samples,"limits":["Only pairs touching the exact selected regions are tested.","Shared-vertex contacts and coplanar overlap are excluded; enclosed surfaces without crossings are not detected.","Triangle pairs are diagnostics, not semantic clothing layers or penetration depth.","Finite pose samples do not certify collision-free animation; no physical simulation or asset mutation is performed."]});
    Ok(summary)
}

pub fn audit_path(input: &Path, config: &SurfaceContactConfig, out: &Path) -> Result<Value> {
    let source = load_gltf(input)?;
    let summary = audit_asset(&source.asset, config)?;
    crate::pipeline::publish_bundle(out, |dir| {
        write_json(&dir.join("summary.json"), &summary)?;
        write_json(&dir.join("config.json"), config)
    })?;
    Ok(json!({"out":std::fs::canonicalize(out)?,"summary":summary}))
}
