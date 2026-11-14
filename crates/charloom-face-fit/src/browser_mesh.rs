//! Portable mesh operations for the Three.js workspace. All coordinates are
//! world-space metres; attribute and object ownership remains with Three.js.
use crate::{
    browser_remesh::{self, SkinRemeshConfig},
    surface::{Surface, V},
};
use anyhow::{Result, ensure};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MeshData {
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[usize; 3]>,
    #[serde(default)]
    pub uv: Vec<[f32; 2]>,
    #[serde(default)]
    pub materials: Vec<usize>,
}
impl MeshData {
    pub fn check(&self) -> Result<()> {
        ensure!(
            self.vertices.len() >= 3 && !self.triangles.is_empty(),
            "Mesh has no triangles"
        );
        ensure!(
            self.vertices.len() <= 1_000_000 && self.triangles.len() <= 1_500_000,
            "Mesh exceeds the interactive processing limit"
        );
        ensure!(
            self.vertices.iter().flatten().all(|v| v.is_finite()),
            "Nonfinite vertex"
        );
        ensure!(
            self.triangles
                .iter()
                .flatten()
                .all(|&i| i < self.vertices.len()),
            "Invalid triangle index"
        );
        ensure!(
            self.uv.is_empty() || self.uv.len() == self.vertices.len(),
            "UV count mismatch"
        );
        ensure!(
            self.materials.is_empty() || self.materials.len() == self.triangles.len(),
            "Material count mismatch"
        );
        Ok(())
    }
    fn points(&self) -> Vec<V> {
        self.vertices.iter().copied().map(V::from).collect()
    }
}
/// Portable, world-space mesh operations. Bind/transport store surface
/// correspondences; garment is a bounded clearance projection, not cloth
/// simulation, donor registration, skeletal binding or project adoption.
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum Request {
    Inspect {
        mesh: MeshData,
    },
    Remesh {
        mesh: MeshData,
        config: SkinRemeshConfig,
    },
    Garment {
        mesh: MeshData,
        body: MeshData,
        /// Requested outward gap in metres, in the range 0..=0.03.
        clearance_m: f64,
        /// Local search distance in metres, in the range 0.001..=0.2.
        maximum_distance_m: f64,
    },
    Transport {
        mesh: MeshData,
        deformed: Vec<[f64; 3]>,
        points: Vec<[f64; 3]>,
    },
    Bind {
        /// Body surface in its exact rest vertex order.
        mesh: MeshData,
        /// Registered garment vertices in the same world-space metres.
        points: Vec<[f64; 3]>,
    },
}
fn inspect(mesh: &MeshData) -> Value {
    let points = mesh.points();
    let mut edges = BTreeMap::<[usize; 2], usize>::new();
    let mut degenerate = 0;
    for &[a, b, c] in &mesh.triangles {
        if (points[b] - points[a])
            .cross(&(points[c] - points[a]))
            .norm()
            < 1e-12
        {
            degenerate += 1;
        }
        for (a, b) in [(a, b), (b, c), (c, a)] {
            *edges.entry([a.min(b), a.max(b)]).or_default() += 1;
        }
    }
    json!({"vertices":points.len(),"triangles":mesh.triangles.len(),"degenerate_triangles":degenerate,
        "boundary_edges":edges.values().filter(|&&n| n==1).count(),"nonmanifold_edges":edges.values().filter(|&&n|n>2).count(),
        "self_crossings":Surface::new(&points,&mesh.triangles).crossings().len(),
        "crossing_definition":"strict noncoplanar triangle crossings; coplanar contacts excluded","user_approved":false})
}
pub fn process(request: &str) -> Result<String> {
    let request: Request = serde_json::from_str(request)?;
    let output = match request {
        Request::Inspect { mesh } => {
            mesh.check()?;
            json!({"report":inspect(&mesh)})
        }
        Request::Remesh { mesh, config } => {
            mesh.check()?;
            let result = browser_remesh::remesh(
                &mesh,
                &mesh
                    .triangles
                    .iter()
                    .map(|f| f.map(|i| i as u32))
                    .collect::<Vec<_>>(),
                &mesh.vertices,
                (!mesh.uv.is_empty()).then_some(mesh.uv.as_slice()),
                &config,
            )?;
            json!({"triangles":result.faces,"face_sources":result.face_sources,"report":result.report})
        }
        Request::Transport {
            mesh,
            deformed,
            points,
        } => {
            mesh.check()?;
            ensure!(
                deformed.len() == mesh.vertices.len()
                    && deformed
                        .iter()
                        .chain(&points)
                        .flatten()
                        .all(|v| v.is_finite()),
                "Transport input mismatch"
            );
            let source = mesh.points();
            let surface = Surface::new(&source, &mesh.triangles);
            let mut transported = Vec::with_capacity(points.len());
            for p in points {
                let p = V::from(p);
                let hit = surface
                    .nearest_detail(p, 10., None)
                    .ok_or_else(|| anyhow::anyhow!("Unbound accessory vertex"))?;
                let delta = mesh.triangles[hit.triangle]
                    .iter()
                    .zip(hit.barycentric)
                    .fold(V::zeros(), |sum, (&i, w)| {
                        sum + (V::from(deformed[i]) - source[i]) * w
                    });
                let q = p + delta;
                transported.push([q.x, q.y, q.z]);
            }
            json!({"vertices":transported,"report":{"method":"rest_surface_barycentric_displacement_transport"}})
        }
        Request::Bind { mesh, points } => {
            mesh.check()?;
            ensure!(
                points.len() <= 1_000_000 && points.iter().flatten().all(|v| v.is_finite()),
                "Invalid garment binding points"
            );
            let source = mesh.points();
            let surface = Surface::new(&source, &mesh.triangles);
            let mut indices = Vec::with_capacity(points.len() * 3);
            let mut weights = Vec::with_capacity(points.len() * 3);
            let mut maximum_distance = 0.0f64;
            for point in &points {
                let point = V::from(*point);
                let hit = surface.nearest_detail(point, 10., None).ok_or_else(|| {
                    anyhow::anyhow!("Garment lies outside the body binding range")
                })?;
                maximum_distance = maximum_distance.max((point - hit.point).norm());
                indices.extend(mesh.triangles[hit.triangle]);
                weights.extend(hit.barycentric);
            }
            json!({"indices":indices,"weights":weights,"count":points.len(),"maxDistance":maximum_distance,
                "report":{"method":"rest_surface_barycentric_garment_binding","bound_vertices":points.len(),"maximum_distance_m":maximum_distance}})
        }
        Request::Garment {
            mesh,
            body,
            clearance_m,
            maximum_distance_m,
        } => {
            mesh.check()?;
            body.check()?;
            ensure!(
                (0.0..=0.03).contains(&clearance_m) && (0.001..=0.2).contains(&maximum_distance_m),
                "Garment distance out of range"
            );
            let surface = Surface::new(&body.points(), &body.triangles);
            let mut points = mesh.points();
            let mut moved = 0;
            let mut maximum = 0.0f64;
            for p in &mut points {
                let before = *p;
                for _ in 0..5 {
                    let Some(hit) = surface.nearest_detail(*p, maximum_distance_m, None) else {
                        break;
                    };
                    let signed = (*p - hit.point).dot(&hit.normal);
                    if signed >= clearance_m - 1e-7 {
                        break;
                    }
                    *p += hit.normal * (clearance_m - signed).min(maximum_distance_m);
                }
                let delta = (*p - before).norm();
                if delta > 1e-8 {
                    moved += 1
                };
                maximum = maximum.max(delta);
            }
            let unresolved = points
                .iter()
                .filter(|&&p| {
                    surface
                        .nearest_detail(p, maximum_distance_m, None)
                        .is_some_and(|h| (p - h.point).dot(&h.normal) < clearance_m - 1e-5)
                })
                .count();
            let mut combined = body.points();
            let offset = combined.len();
            combined.extend(&points);
            let mut faces = body.triangles.clone();
            let face_offset = faces.len();
            faces.extend(mesh.triangles.iter().map(|f| f.map(|i| i + offset)));
            let crossings = Surface::new(&combined, &faces)
                .crossings()
                .into_iter()
                .filter(|f| f[0] < face_offset && f[1] >= face_offset)
                .count();
            json!({"vertices":points.iter().map(|p|[p.x,p.y,p.z]).collect::<Vec<_>>(),"report":{"method":"body_surface_clearance_projection","moved_vertices":moved,"maximum_displacement_m":maximum,"clearance_m":clearance_m,"unresolved_vertices":unresolved,"body_garment_crossings":crossings,"cloth_simulation":false}})
        }
    };
    Ok(serde_json::to_string(&output)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn garment_clearance_and_accessory_transport_are_real_geometry_operations() {
        let body = json!({"vertices":[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],"triangles":[[0,1,2]]});
        let mesh = json!({"vertices":[[0.1,0.1,-0.01],[0.2,0.1,-0.01],[0.1,0.2,-0.01]],"triangles":[[0,1,2]]});
        let result:Value=serde_json::from_str(&process(&json!({"operation":"garment","mesh":mesh,"body":body,"clearance_m":0.003,"maximum_distance_m":0.1}).to_string()).unwrap()).unwrap();
        assert_eq!(result["report"]["unresolved_vertices"], 0);
        for p in result["vertices"].as_array().unwrap() {
            assert!((p[2].as_f64().unwrap() - 0.003).abs() < 1e-9)
        }
        let result:Value=serde_json::from_str(&process(&json!({"operation":"transport","mesh":body,"deformed":[[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]],"points":[[0.2,0.2,0.02]]}).to_string()).unwrap()).unwrap();
        assert!((result["vertices"][0][2].as_f64().unwrap() - 1.02).abs() < 1e-9);
        let binding: Value = serde_json::from_str(
            &process(
                &json!({"operation":"bind","mesh":body,"points":[[0.2,0.2,0.02]]}).to_string(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(binding["indices"], json!([0, 1, 2]));
        let weights: Vec<f64> = binding["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert!((weights.iter().sum::<f64>() - 1.).abs() < 1e-12);
        assert!((weights[1] - 0.2).abs() < 1e-12);
        assert!((binding["maxDistance"].as_f64().unwrap() - 0.02).abs() < 1e-12);
    }
    #[test]
    fn remesh_reduces_planar_faces_and_keeps_uv_boundary() {
        let n = 16;
        let mut vertices = vec![];
        let mut triangles = vec![];
        let mut uv = vec![];
        for y in 0..n {
            for x in 0..n {
                vertices.push([x as f64 * 0.005, y as f64 * 0.005, 0.]);
                uv.push([x as f32 / 15., y as f32 / 15.]);
            }
        }
        for y in 0..n - 1 {
            for x in 0..n - 1 {
                let a = y * n + x;
                triangles.extend([[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]);
            }
        }
        let result:Value=serde_json::from_str(&process(&json!({"operation":"remesh","mesh":{"vertices":vertices,"triangles":triangles,"uv":uv},"config":{"target_edge_m":0.015,"target_face_fraction":0.65}}).to_string()).unwrap()).unwrap();
        assert!(result["triangles"].as_array().unwrap().len() < triangles.len());
        let used: std::collections::BTreeSet<_> = result["triangles"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|f| {
                f.as_array()
                    .unwrap()
                    .iter()
                    .map(|i| i.as_u64().unwrap() as usize)
            })
            .collect();
        for i in 0..n {
            for v in [i, (n - 1) * n + i, i * n, i * n + n - 1] {
                assert!(used.contains(&v))
            }
        }
    }
}
