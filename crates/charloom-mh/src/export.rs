//! GLB assembly on top of `charloom_io::GlbDocument`.
use crate::source::{ImageData, MaterialData};
use anyhow::{Context, Result, ensure};
use charloom_core::{Influence, Mat4, MotionClip, Skeleton, Vec3};
use charloom_io::GlbDocument;
use nalgebra::Matrix4;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::Path;

pub struct OutMesh<'a> {
    pub name: &'a str,
    pub positions: &'a [Vec3],
    pub normals: &'a [Vec3],
    pub uvs: Option<&'a [[f32; 2]]>,
    pub colors: Option<&'a [[f32; 4]]>,
    pub triangles: &'a [[u32; 3]],
    /// Global joint indices of the exported skeleton.
    pub influences: Option<&'a [Vec<Influence>]>,
    pub material: Option<usize>,
}

pub struct Builder {
    pub doc: GlbDocument,
    joint_offset: usize,
    skin: Option<(usize, HashMap<usize, usize>)>,
}

/// Shifts every `{ "index": n }` texture reference under a material.
fn offset_textures(value: &mut Value, offset: usize) {
    match value {
        Value::Object(map) => {
            for (key, child) in map.iter_mut() {
                if key.ends_with("Texture")
                    && let Some(i) = child.get("index").and_then(Value::as_u64)
                {
                    child["index"] = json!(i as usize + offset);
                }
                offset_textures(child, offset);
            }
        }
        Value::Array(items) => items.iter_mut().for_each(|c| offset_textures(c, offset)),
        _ => {}
    }
}

fn bytes_f32(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(f32::to_le_bytes).collect()
}

impl Builder {
    pub fn new() -> Self {
        Self {
            doc: GlbDocument::new(),
            joint_offset: 0,
            skin: None,
        }
    }

    pub fn add_image(&mut self, bytes: &[u8], mime: &str) -> usize {
        let view = self.doc.append_view(bytes, None);
        let images = self.doc.array("images");
        images.push(json!({"bufferView":view,"mimeType":mime}));
        images.len() - 1
    }

    pub fn add_texture(&mut self, image: usize) -> usize {
        let textures = self.doc.array("textures");
        textures.push(json!({"source":image}));
        textures.len() - 1
    }

    pub fn add_material(&mut self, material: Value) -> usize {
        let materials = self.doc.array("materials");
        materials.push(material);
        materials.len() - 1
    }

    /// Copies a source document's images/textures/samplers/materials and
    /// returns the new material index of every source material.
    pub fn import_materials(
        &mut self,
        images: &[ImageData],
        textures: &[Value],
        samplers: &[Value],
        materials: &[MaterialData],
    ) -> Vec<usize> {
        let image_offset = self.doc.array("images").len();
        for image in images {
            self.add_image(&image.bytes, &image.mime);
        }
        let sampler_offset = self.doc.array("samplers").len();
        self.doc.array("samplers").extend(samplers.iter().cloned());
        let texture_offset = self.doc.array("textures").len();
        for texture in textures {
            let mut t = texture.clone();
            if let Some(s) = t.get("source").and_then(Value::as_u64) {
                t["source"] = json!(s as usize + image_offset);
            }
            if let Some(s) = t.get("sampler").and_then(Value::as_u64) {
                t["sampler"] = json!(s as usize + sampler_offset);
            }
            self.doc.array("textures").push(t);
        }
        materials
            .iter()
            .map(|m| {
                let mut json = m.json.clone();
                offset_textures(&mut json, texture_offset);
                self.add_material(json)
            })
            .collect()
    }

    pub fn set_skeleton(
        &mut self,
        skeleton: &Skeleton,
        palette: &[usize],
        world: &[Matrix4<f64>],
    ) -> Result<()> {
        self.joint_offset = self.doc.append_skeleton(skeleton)?;
        let ibm: Vec<Mat4> = palette
            .iter()
            .map(|&j| {
                let m = world[j].try_inverse().unwrap_or_else(Matrix4::identity);
                std::array::from_fn(|c| std::array::from_fn(|r| m[(r, c)]))
            })
            .collect();
        let accessor = self.doc.append_matrices(&ibm);
        let skins = self.doc.array("skins");
        skins.push(json!({"name":"makehuman","joints":palette.iter().map(|j| j + self.joint_offset).collect::<Vec<_>>(),"inverseBindMatrices":accessor}));
        let index = skins.len() - 1;
        self.skin = Some((
            index,
            palette.iter().enumerate().map(|(k, &j)| (j, k)).collect(),
        ));
        Ok(())
    }

    pub fn add_mesh(&mut self, m: &OutMesh) -> Result<usize> {
        ensure!(!m.triangles.is_empty(), "mesh {} is empty", m.name);
        let n = m.positions.len();
        let mut attributes = json!({
            "POSITION": self.doc.append_vec3(m.positions, Some(34962), true),
            "NORMAL": self.doc.append_vec3(m.normals, Some(34962), false),
        });
        if let Some(uv) = m.uvs {
            attributes["TEXCOORD_0"] = json!(self.doc.append_accessor(
                &bytes_f32(uv.iter().flatten().copied()),
                5126,
                "VEC2",
                n,
                Some(34962),
                None
            ));
        }
        if let Some(c) = m.colors {
            attributes["COLOR_0"] = json!(self.doc.append_accessor(
                &bytes_f32(c.iter().flatten().copied()),
                5126,
                "VEC4",
                n,
                Some(34962),
                None
            ));
        }
        let mut skinned = false;
        if let (Some(rows), Some((_, local))) = (m.influences, &self.skin) {
            ensure!(rows.len() == n, "influence count mismatch in {}", m.name);
            let mut jb = Vec::with_capacity(n * 8);
            let mut wb = Vec::with_capacity(n * 16);
            for row in rows {
                ensure!(!row.is_empty(), "unweighted vertex in {}", m.name);
                let sum: f64 = row.iter().take(4).map(|w| w.weight).sum();
                for k in 0..4 {
                    let (j, w) = row
                        .get(k)
                        .map_or((0, 0.0), |w| (local[&w.joint], w.weight / sum));
                    jb.extend_from_slice(&(j as u16).to_le_bytes());
                    wb.extend_from_slice(&(w as f32).to_le_bytes());
                }
            }
            attributes["JOINTS_0"] =
                json!(
                    self.doc
                        .append_accessor(&jb, 5123, "VEC4", n, Some(34962), None)
                );
            attributes["WEIGHTS_0"] =
                json!(
                    self.doc
                        .append_accessor(&wb, 5126, "VEC4", n, Some(34962), None)
                );
            skinned = true;
        }
        let indices: Vec<u8> = m
            .triangles
            .iter()
            .flatten()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let indices = self.doc.append_accessor(
            &indices,
            5125,
            "SCALAR",
            m.triangles.len() * 3,
            Some(34963),
            None,
        );
        let mut primitive = json!({"attributes":attributes,"indices":indices});
        if let Some(material) = m.material {
            primitive["material"] = json!(material);
        }
        let meshes = self.doc.array("meshes");
        meshes.push(json!({"name":m.name,"primitives":[primitive]}));
        let mesh = meshes.len() - 1;
        let mut node = json!({"name":m.name,"mesh":mesh});
        if skinned {
            node["skin"] = json!(self.skin.as_ref().unwrap().0);
        }
        let nodes = self.doc.array("nodes");
        nodes.push(node);
        let index = nodes.len() - 1;
        self.doc.json["scenes"][0]["nodes"]
            .as_array_mut()
            .expect("scene nodes")
            .push(json!(index));
        Ok(index)
    }

    /// glTF morph targets (POSITION deltas, named in `extras.targetNames`,
    /// default weight 0) on the mesh of `node` from [`Builder::add_mesh`].
    /// Targets touching under half the vertices are stored sparse.
    pub fn add_morph_targets(
        &mut self,
        node: usize,
        targets: &[(String, Vec<Vec3>)],
    ) -> Result<()> {
        if targets.is_empty() {
            return Ok(());
        }
        let mesh = self.doc.json["nodes"][node]["mesh"]
            .as_u64()
            .context("morph targets on a node without a mesh")? as usize;
        let mut list = vec![];
        for (name, deltas) in targets {
            let quantised: Vec<[f32; 3]> = deltas.iter().map(|d| d.map(|x| x as f32)).collect();
            let moved: Vec<u32> = (0..quantised.len() as u32)
                .filter(|&i| quantised[i as usize] != [0.0; 3])
                .collect();
            let n = deltas.len();
            let accessor = if !moved.is_empty() && moved.len() * 2 < n {
                let bounds = charloom_core::Bounds::from_points(deltas.iter().copied())
                    .map(|b| (json!(b.min), json!(b.max)))
                    .context("empty morph target")?;
                let indices: Vec<u8> = moved.iter().flat_map(|i| i.to_le_bytes()).collect();
                let values = bytes_f32(moved.iter().flat_map(|&i| quantised[i as usize]));
                let iv = self.doc.append_view(&indices, None);
                let vv = self.doc.append_view(&values, None);
                let accessors = self.doc.array("accessors");
                accessors.push(json!({"componentType":5126,"count":n,"type":"VEC3","min":bounds.0,"max":bounds.1,
                    "sparse":{"count":moved.len(),"indices":{"bufferView":iv,"componentType":5125},"values":{"bufferView":vv}}}));
                accessors.len() - 1
            } else {
                ensure!(!moved.is_empty(), "morph target {name} on {node} is empty");
                self.doc.append_vec3(deltas, Some(34962), true)
            };
            list.push(json!({ "POSITION": accessor }));
        }
        let m = &mut self.doc.json["meshes"][mesh];
        for p in m["primitives"].as_array_mut().context("mesh primitives")? {
            p["targets"] = json!(list);
        }
        m["weights"] = json!(vec![0.0; targets.len()]);
        m["extras"]["targetNames"] = json!(targets.iter().map(|t| &t.0).collect::<Vec<_>>());
        Ok(())
    }

    /// Sets `extras[key]` on the mesh of `node`.
    pub fn set_mesh_extra(&mut self, node: usize, key: &str, value: Value) {
        if let Some(mesh) = self.doc.json["nodes"][node]["mesh"].as_u64() {
            self.doc.json["meshes"][mesh as usize]["extras"][key] = value;
        }
    }

    pub fn add_animation(&mut self, clip: &MotionClip) -> Result<()> {
        self.doc.append_animation(clip, self.joint_offset)?;
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.doc.save(path)
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}
