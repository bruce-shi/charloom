//! Final roster assembly on an existing transfer. Keep the original skin,
//! clips, morphs and materials while applying the measured surface corrections.
pub mod body_surface;
pub mod clips;
pub mod collar;
pub mod chart_shell;
mod face_binding;
pub mod retained_mouth;
pub mod walk;
pub use face_binding::bind_face;
mod rim_curve;
mod native_lips;
pub use native_lips::fit_native;

use crate::geom::{a, v};
use crate::guide::{ViewSet, file_sha256};
use anyhow::{Context, Result, ensure};
use charloom_core::Vec3;
use charloom_engine::surface_query::SurfaceQuery;
use charloom_io::load_gltf;
use nalgebra::Matrix4;
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn read<T: DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )
    .with_context(|| format!("parse {}", path.display()))
}

#[derive(Deserialize)]
struct Trace {
    view: String,
    image: PathBuf,
    image_sha256: String,
    source_sha256: String,
    pixels: Vec<[f64; 2]>,
}

/// Project the traced image pixels onto the segmentation in its working frame.
pub fn mouth_trace(run: &Path, trace: &Path) -> Result<Vec<Vec3>> {
    let trace: Trace = read(trace)?;
    let report: Value = read(&run.join("report.json"))?;
    let source = Path::new(
        report["source"]
            .as_str()
            .context("missing transfer source")?,
    );
    let config: Value = read(&run.join("config.json"))?;
    let guide = Path::new(config["guidance"].as_str().context("missing guidance")?);
    let dir = guide.parent().context("invalid guidance path")?;
    let views: ViewSet = read(&dir.join("views.json"))?;
    ensure!(
        Some(&*views.source_hash) == report["source_hash"].as_str(),
        "camera set belongs to a different scan"
    );
    ensure!(
        file_sha256(source)? == trace.source_sha256,
        "mouth trace source hash mismatch"
    );
    let camera = views
        .cameras
        .iter()
        .find(|c| c.id == trace.view)
        .context("trace view missing")?;
    let image = dir.join("views").join(format!("{}.png", camera.id));
    ensure!(
        file_sha256(&image)? == trace.image_sha256 && image.file_name() == trace.image.file_name(),
        "mouth trace camera image mismatch"
    );
    let matrix: [f64; 16] =
        serde_json::from_value(report["working_frame"]["source_to_working"].clone())?;
    let camera = camera.transformed(&Matrix4::from_column_slice(&matrix));
    let segment = load_gltf(run.join("1-segment/segmentation.glb"))?;
    let meshes = &segment.asset.meshes;
    let positions = meshes
        .iter()
        .map(|m| m.positions.clone())
        .collect::<Vec<_>>();
    let query = SurfaceQuery::new(meshes, &positions);
    ensure!(
        trace.pixels.len() >= 2,
        "at least two mouth samples required"
    );
    let mut points = Vec::with_capacity(trace.pixels.len());
    for [x, y] in trace.pixels {
        let (origin, direction) = camera.ray(x, y);
        let hit = query
            .first_ray_distance(a(origin), a(direction), 200.)
            .with_context(|| format!("mouth pixel [{x}, {y}] misses scan"))?;
        points.push(a(origin + direction * hit));
    }
    points.sort_by(|a, b| a[0].total_cmp(&b[0]));
    ensure!(
        points.windows(2).all(|p| p[1][0] > p[0][0]),
        "mouth trace must span distinct horizontal positions"
    );
    Ok(points)
}
