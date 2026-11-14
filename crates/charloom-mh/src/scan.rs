//! The generated (Tripo/Meshy) model, merged into one triangle soup and moved
//! into the working frame: Y-up metres, facing +Z, feet on y = 0, centred.
use crate::geom::{Weld, a, v, vertex_normals};
use crate::source::{ImageData, MaterialData, read_images, read_materials, read_primitives};
use anyhow::{Result, ensure};
use charloom_core::{Mesh, Vec3};
use charloom_engine::surface_query::SurfaceQuery;
use charloom_io::load_gltf;
use nalgebra::{Matrix4, Rotation3, Vector3};
use std::path::Path;

pub fn query_mesh(positions: &[Vec3], triangles: &[[u32; 3]]) -> (Mesh, Vec<Vec3>) {
    (
        Mesh {
            name: String::new(),
            node: 0,
            primitive: 0,
            positions: positions.to_vec(),
            triangles: triangles.to_vec(),
            skin: None,
            influences: vec![],
            morph_targets: vec![],
            morph_weights: vec![],
        },
        positions.to_vec(),
    )
}

pub fn build_query(positions: &[Vec3], triangles: &[[u32; 3]]) -> SurfaceQuery {
    let (mesh, p) = query_mesh(positions, triangles);
    SurfaceQuery::new(&[mesh], &[p])
}

pub struct Scan {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<[f32; 2]>,
    pub triangles: Vec<[u32; 3]>,
    pub triangle_material: Vec<Option<usize>>,
    pub weld: Weld,
    pub query: SurfaceQuery,
         pub transform: Matrix4<f64>,
    pub images: Vec<ImageData>,
    pub materials: Vec<MaterialData>,
    pub textures: Vec<serde_json::Value>,
    pub samplers: Vec<serde_json::Value>,
    pub source_hash: String,
}

pub struct RawScan {
    pub positions: Vec<Vec3>,
    pub uvs: Vec<[f32; 2]>,
    pub triangles: Vec<[u32; 3]>,
    pub triangle_material: Vec<Option<usize>>,
    pub images: Vec<ImageData>,
    pub materials: Vec<MaterialData>,
    pub textures: Vec<serde_json::Value>,
    pub samplers: Vec<serde_json::Value>,
    pub source_hash: String,
}

impl RawScan {
    pub fn load(path: &Path) -> Result<Self> {
        let loaded = load_gltf(path)?;
        let world = loaded.asset.world_meshes()?;
        let prims = read_primitives(&loaded, &world)?;
        let mut out = Self {
            positions: vec![],
            uvs: vec![],
            triangles: vec![],
            triangle_material: vec![],
            images: read_images(&loaded)?,
            materials: read_materials(&loaded)?,
            textures: json_array(&loaded.document.json["textures"]),
            samplers: json_array(&loaded.document.json["samplers"]),
            source_hash: loaded.asset.source_hash.clone(),
        };
        for p in prims {
            let offset = out.positions.len() as u32;
            out.positions.extend(p.positions);
            out.uvs.extend(p.uvs);
            out.triangle_material
                .extend(std::iter::repeat_n(p.material, p.triangles.len()));
            out.triangles
                .extend(p.triangles.iter().map(|t| t.map(|i| i + offset)));
        }
        ensure!(!out.triangles.is_empty(), "source has no triangles");
        Ok(out)
    }

    /// Similarity putting the source into the working frame for a given yaw
    /// (radians, applied before scaling) and target height.
    pub fn working_transform(&self, yaw: f64, height: f64, center_xz: [f64; 2]) -> Matrix4<f64> {
        let r = Rotation3::from_axis_angle(&Vector3::y_axis(), yaw).to_homogeneous();
        let (mut lo, mut hi) = (V3::repeat(f64::INFINITY), V3::repeat(f64::NEG_INFINITY));
        for p in &self.positions {
            let q = (r * v(*p).push(1.0)).xyz();
            lo = lo.inf(&q);
            hi = hi.sup(&q);
        }
        let s = height / (hi.y - lo.y).max(1e-9);
        let shift = Vector3::new(-(lo.x + hi.x) * 0.5, -lo.y, -(lo.z + hi.z) * 0.5);
        Matrix4::new_translation(&Vector3::new(center_xz[0], 0.0, center_xz[1]))
            * Matrix4::new_scaling(s)
            * Matrix4::new_translation(&shift)
            * r
    }

    pub fn into_scan(self, transform: Matrix4<f64>) -> Scan {
        let positions: Vec<Vec3> = self
            .positions
            .iter()
            .map(|p| a((transform * v(*p).push(1.0)).xyz()))
            .collect();
        let weld = Weld::new(&positions, 1e-7);
        let normals = vertex_normals(&positions, &self.triangles, &weld);
        let query = build_query(&positions, &self.triangles);
        Scan {
            positions,
            normals,
            uvs: self.uvs,
            triangles: self.triangles,
            triangle_material: self.triangle_material,
            weld,
            query,
            transform,
            images: self.images,
            materials: self.materials,
            textures: self.textures,
            samplers: self.samplers,
            source_hash: self.source_hash,
        }
    }
}

type V3 = Vector3<f64>;

/// A ray hit on the scan: triangle index, barycentric point, ray distance.
#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    pub triangle: usize,
    pub barycentric: [f64; 3],
    pub distance: f64,
}

pub fn json_array(value: &serde_json::Value) -> Vec<serde_json::Value> {
    value.as_array().cloned().unwrap_or_default()
}

impl Scan {
    /// Base colour (sRGB, [0,1]) at a barycentric point of a triangle.
    pub fn color_at(&self, triangle: usize, bary: [f64; 3]) -> Option<[f64; 3]> {
        let material = self.materials.get(self.triangle_material[triangle]?)?;
        let t = self.triangles[triangle];
        let uv = [0, 1].map(|k| {
            (0..3)
                .map(|i| self.uvs[t[i] as usize][k] as f64 * bary[i])
                .sum()
        });
        let factor = material.base_color_factor;
        let c = match material.base_color {
            Some(i) => crate::source::sample(&self.images.get(i)?.decoded, uv),
            None => [1.0; 3],
        };
        Some(std::array::from_fn(|k| c[k] * factor[k]))
    }

    /// First scan surface facing the ray along `origin + t·direction`,
    /// `t ≤ maximum`: the selected-to-active projection of a texture bake.
    /// Back faces are passed through: a cage point that starts inside a
    /// neighbouring part (the other thigh, the head behind an ear) leaves it
    /// before reaching the surface it should see.
    pub fn first_facing_hit(&self, origin: Vec3, direction: Vec3, maximum: f64) -> Option<RayHit> {
        let d = v(direction).try_normalize(1e-12)?;
        let mut o = v(origin);
        let mut travelled = 0.0;
        for _ in 0..6 {
            let t = self
                .query
                .first_ray_distance(a(o), a(d), maximum - travelled)?;
            let p = o + d * t;
            travelled += t;
            // The ray query reports distance only: recover the triangle as
            // the facing one through the hit point.
            if let Some(h) = self.query.closest_point(a(p), 2e-4, Some((a(-d), 0.0))) {
                return Some(RayHit {
                    triangle: h.source[1],
                    barycentric: h.barycentric,
                    distance: travelled,
                });
            }
            o = p + d * 1e-5;
            travelled += 1e-5;
            if travelled >= maximum {
                return None;
            }
        }
        None
    }

    pub fn vertex_color(&self, vertex: usize, triangle: usize) -> Option<[f64; 3]> {
        let t = self.triangles[triangle];
        let k = t.iter().position(|&i| i as usize == vertex)?;
        let mut bary = [0.0; 3];
        bary[k] = 1.0;
        self.color_at(triangle, bary)
    }

    pub fn height(&self) -> f64 {
        let (lo, hi) = self
            .positions
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), p| {
                (l.min(p[1]), h.max(p[1]))
            });
        hi - lo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sheet_scan() -> Scan {
        // Two unit squares in z = 0.5 (facing −z) and z = 0 (facing +z).
        let mut positions = vec![];
        let mut triangles = vec![];
        for (z, flip) in [(0.5, true), (0.0, false)] {
            let o = positions.len() as u32;
            positions.extend([
                [-1.0, -1.0, z],
                [1.0, -1.0, z],
                [1.0, 1.0, z],
                [-1.0, 1.0, z],
            ]);
            let (t0, t1) = ([o, o + 1, o + 2], [o, o + 2, o + 3]);
            if flip {
                triangles.extend([[t0[0], t0[2], t0[1]], [t1[0], t1[2], t1[1]]]);
            } else {
                triangles.extend([t0, t1]);
            }
        }
        RawScan {
            positions,
            uvs: vec![[0.0; 2]; 8],
            triangle_material: vec![None; triangles.len()],
            triangles,
            images: vec![],
            materials: vec![],
            textures: vec![],
            samplers: vec![],
            source_hash: String::new(),
        }
        .into_scan(Matrix4::identity())
    }

    #[test]
    fn facing_ray_passes_back_faces_and_respects_its_length() {
        let scan = sheet_scan();
        // From above both sheets, looking down: the z = 0.5 sheet faces away.
        let hit = scan
            .first_facing_hit([0.2, 0.1, 1.0], [0.0, 0.0, -1.0], 2.0)
            .unwrap();
        assert!(hit.triangle >= 2 && (hit.distance - 1.0).abs() < 1e-4);
        assert!(
            scan.first_facing_hit([0.2, 0.1, 1.0], [0.0, 0.0, -1.0], 0.9)
                .is_none()
        );
        // From below looking up, the z = 0 sheet faces away, z = 0.5 faces it.
        let hit = scan
            .first_facing_hit([0.2, 0.1, -0.5], [0.0, 0.0, 1.0], 2.0)
            .unwrap();
        assert!(hit.triangle < 2 && (hit.distance - 1.0).abs() < 1e-4);
    }
}
