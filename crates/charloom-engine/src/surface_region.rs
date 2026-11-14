//! Exact vertex regions produced by calibrated views or explicit authoring.
use anyhow::{Result, ensure};
use charloom_core::Mesh;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SurfaceMask {
    pub mesh: usize,
    /// Hash of local position and triangle arrays, independent of skin weights.
    pub geometry_hash: String,
    pub vertices: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WeightRegion {
    pub id: String,
    pub mask: SurfaceMask,
    #[serde(default)]
    pub allowed_joints: Vec<String>,
    #[serde(default)]
    pub excluded_joints: Vec<String>,
    pub evidence: Vec<String>,
}

/// Explicit source-provenance evidence that render copies came from one
/// physical vertex. This couples skin weights without welding any geometry.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SeamCoupling {
    pub id: String,
    pub mask: SurfaceMask,
    pub evidence: Vec<String>,
}

pub fn geometry_hash(mesh: &Mesh) -> String {
    let mut hash = Sha256::new();
    hash.update((mesh.positions.len() as u64).to_le_bytes());
    for p in &mesh.positions {
        for v in p {
            hash.update(v.to_le_bytes());
        }
    }
    hash.update((mesh.triangles.len() as u64).to_le_bytes());
    for t in &mesh.triangles {
        for i in t {
            hash.update(i.to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

pub fn validate_mask(mask: &SurfaceMask, meshes: &[Mesh]) -> Result<()> {
    ensure!(mask.mesh < meshes.len(), "surface_mask_mesh_out_of_bounds");
    validate_mask_with_geometry_hash(mask, meshes, &geometry_hash(&meshes[mask.mesh]))
}

/// A binding operation may validate thousands of source-identity groups on
/// one mesh. The caller computes each immutable geometry hash once.
pub(crate) fn validate_mask_with_geometry_hash(
    mask: &SurfaceMask,
    meshes: &[Mesh],
    hash: &str,
) -> Result<()> {
    ensure!(mask.mesh < meshes.len(), "surface_mask_mesh_out_of_bounds");
    let mesh = &meshes[mask.mesh];
    ensure!(mask.geometry_hash == hash, "surface_mask_geometry_mismatch");
    ensure!(
        !mask.vertices.is_empty() && mask.vertices.iter().all(|&i| i < mesh.positions.len()),
        "invalid_surface_mask_vertices"
    );
    ensure!(
        mask.vertices.windows(2).all(|p| p[0] < p[1]),
        "surface_mask_vertices_must_be_sorted_unique"
    );
    Ok(())
}
