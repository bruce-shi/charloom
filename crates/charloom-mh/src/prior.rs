//! The rigged MakeHuman (MPFB) body: skeleton, single skin, body mesh and the
//! face helpers (eyes, brows, lashes, teeth, tongue).
use crate::geom::{a, v};
use crate::source::{Primitive, read_primitives};
use anyhow::{Context, Result, ensure};
use charloom_core::{Influence, Transform, Vec3, matrix_from};
use charloom_io::{LoadedAsset, load_gltf};
use nalgebra::Matrix4;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PartKind {
    Body,
    Eyes,
    Eyebrows,
    Eyelashes,
    Teeth,
    Tongue,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Region {
    Head,
    Torso,
    Arm,
    Hand,
    Leg,
    Foot,
}

pub struct PriorMesh {
    pub kind: PartKind,
    /// Rest world positions plus UVs and triangles.
    pub prim: Primitive,
    /// Bind-space positions (glTF POSITION).
    pub bind: Vec<Vec3>,
    /// Global node indices.
    pub influences: Vec<Vec<Influence>>,
    /// Morph targets in bind space, by name (MakeHuman shape targets exported
    /// by scripts/mpfb/export-shape-targets.py; empty for plain priors).
    pub morphs: Vec<(String, Vec<Vec3>)>,
}

pub struct Prior {
    pub loaded: LoadedAsset,
    pub meshes: Vec<PriorMesh>,
    pub body: usize,
    /// Skin palette as node indices, in glTF skin order.
    pub skin_joints: Vec<usize>,
    pub ibm: Vec<Matrix4<f64>>,
    pub rest_world: Vec<Matrix4<f64>>,
    pub by_name: HashMap<String, usize>,
    pub region: Vec<Option<Region>>,
    /// MakeHuman's per-side ear targets (`ears/l-ear-scale-incr`, ...) as
    /// rest-space offsets of the body vertices, from the sidecar
    /// `ear-targets.glb` next to the prior (scripts/mpfb/export-shape-targets.py
    /// with CHARLOOM_TARGET_SET=ears); empty when it is absent. They shape
    /// each ear to the marked ear outline (fit::ears).
    pub ear_targets: Vec<(String, Vec<Vec3>)>,
}

/// glTF `meshes[].extras.targetNames` by mesh name (read from the GLB JSON).
fn morph_target_names(path: &Path) -> HashMap<String, Vec<String>> {
    let parse = || -> Option<HashMap<String, Vec<String>>> {
        let bytes = std::fs::read(path).ok()?;
        let n = u32::from_le_bytes(bytes.get(12..16)?.try_into().ok()?) as usize;
        let json: serde_json::Value = serde_json::from_slice(bytes.get(20..20 + n)?).ok()?;
        Some(
            json["meshes"]
                .as_array()?
                .iter()
                .filter_map(|m| {
                    let names = m["extras"]["targetNames"].as_array()?;
                    Some((
                        m["name"].as_str()?.to_string(),
                        names
                            .iter()
                            .filter_map(|x| x.as_str().map(String::from))
                            .collect(),
                    ))
                })
                .collect(),
        )
    };
    parse().unwrap_or_default()
}

fn classify(name: &str) -> PartKind {
    let n = name.to_lowercase();
    if n.contains("eyebrow") {
        PartKind::Eyebrows
    } else if n.contains("eyelash") {
        PartKind::Eyelashes
    } else if n.contains("teeth") {
        PartKind::Teeth
    } else if n.contains("tongue") {
        PartKind::Tongue
    } else if n.contains("eye") || n.contains("low-poly") {
        PartKind::Eyes
    } else {
        PartKind::Other
    }
}

fn region_by_name(name: &str) -> Option<Region> {
    let n = name.to_lowercase();
    let starts = |p: &str| n.starts_with(p);
    if starts("wrist") || starts("finger") || starts("metacarpal") {
        Some(Region::Hand)
    } else if starts("upperarm") || starts("lowerarm") || starts("shoulder") {
        Some(Region::Arm)
    } else if starts("foot") || starts("toe") {
        Some(Region::Foot)
    } else if starts("upperleg") || starts("lowerleg") || starts("pelvis") {
        Some(Region::Leg)
    } else if n == "head" {
        Some(Region::Head)
    } else if starts("spine")
        || starts("breast")
        || starts("clavicle")
        || starts("neck")
        || n == "root"
    {
        Some(Region::Torso)
    } else {
        None
    }
}

impl Prior {
    pub fn load(path: &Path) -> Result<Self> {
        let loaded = load_gltf(path)?;
        let asset = &loaded.asset;
        ensure!(
            asset.skins.len() == 1,
            "prior must contain exactly one skin"
        );
        let skin = &asset.skins[0];
        let rest_world = asset.skeleton.world_matrices(&asset.skeleton.rest_pose())?;
        let rest = asset.skinned_positions(&asset.skeleton.rest_pose())?;
        let prims = read_primitives(&loaded, &rest)?;
        let target_names = morph_target_names(path);
        let mut meshes = vec![];
        for (prim, mesh) in prims.into_iter().zip(&asset.meshes) {
            ensure!(
                mesh.skin == Some(0),
                "prior mesh {} is not skinned",
                mesh.name
            );
            meshes.push(PriorMesh {
                kind: classify(&mesh.name),
                prim,
                bind: mesh.positions.clone(),
                influences: mesh.influences.clone(),
                morphs: target_names
                    .get(&mesh.name)
                    .filter(|names| names.len() == mesh.morph_targets.len())
                    .map(|names| {
                        names
                            .iter()
                            .cloned()
                            .zip(mesh.morph_targets.iter().cloned())
                            .collect()
                    })
                    .unwrap_or_default(),
            });
        }
        let body = (0..meshes.len())
            .max_by_key(|&i| meshes[i].bind.len())
            .context("prior has no meshes")?;
        meshes[body].kind = PartKind::Body;
        let by_name: HashMap<_, _> = asset
            .skeleton
            .joints
            .iter()
            .enumerate()
            .map(|(i, j)| (j.name.clone(), i))
            .collect();
        for needed in [
            "root",
            "head",
            "upperarm01.L",
            "upperleg01.R",
            "wrist.L",
            "foot.R",
        ] {
            ensure!(
                by_name.contains_key(needed),
                "prior is not an MPFB default rig: missing {needed}"
            );
        }
        let region = (0..asset.skeleton.joints.len())
            .map(|i| {
                let mut at = Some(i);
                while let Some(j) = at {
                    if let Some(r) = region_by_name(&asset.skeleton.joints[j].name) {
                        return Some(r);
                    }
                    at = asset.skeleton.joints[j].parent;
                }
                None
            })
            .collect();
        let mut prior = Self {
            meshes,
            body,
            skin_joints: skin.joints.clone(),
            ibm: skin
                .inverse_bind_matrices
                .iter()
                .map(|m| matrix_from(*m))
                .collect(),
            rest_world,
            by_name,
            region,
            loaded,
            ear_targets: vec![],
        };
        if let Some(dir) = path.parent() {
            prior.ear_targets = prior.load_ear_targets(&dir.join("ear-targets.glb"));
        }
        Ok(prior)
    }

    /// Ear targets from a sidecar export of the same MakeHuman body (same
    /// vertex order and rest surface), carried to rest space. Anything that
    /// does not match the prior's body is ignored.
    fn load_ear_targets(&self, path: &Path) -> Vec<(String, Vec<Vec3>)> {
        let Ok(loaded) = load_gltf(path) else {
            return vec![];
        };
        let names = morph_target_names(path);
        let body = self.body();
        let Some(mesh) = loaded.asset.meshes.iter().find(|m| {
            m.positions.len() == body.bind.len()
                && names
                    .get(&m.name)
                    .is_some_and(|n| n.len() == m.morph_targets.len() && !n.is_empty())
        }) else {
            return vec![];
        };
        let same = mesh
            .positions
            .iter()
            .zip(&body.bind)
            .step_by(7)
            .all(|(p, q)| (v(*p) - v(*q)).norm() < 1e-4);
        if !same {
            return vec![];
        }
        let rest = self.skin_positions(self.body, &body.bind, &self.rest_world);
        names[&mesh.name]
            .iter()
            .zip(&mesh.morph_targets)
            .filter(|(n, _)| n.starts_with("ears/"))
            .map(|(n, d)| {
                let bind: Vec<Vec3> = body
                    .bind
                    .iter()
                    .zip(d)
                    .map(|(b, d)| a(v(*b) + v(*d)))
                    .collect();
                let moved = self.skin_positions(self.body, &bind, &self.rest_world);
                let delta = moved
                    .iter()
                    .zip(&rest)
                    .map(|(m, r)| a(v(*m) - v(*r)))
                    .collect();
                (n.clone(), delta)
            })
            .collect()
    }

    pub fn skeleton(&self) -> &charloom_core::Skeleton {
        &self.loaded.asset.skeleton
    }

    pub fn joint(&self, name: &str) -> usize {
        self.by_name[name]
    }

    pub fn body(&self) -> &PriorMesh {
        &self.meshes[self.body]
    }

    pub fn worlds(&self, pose: &[Transform]) -> Result<Vec<Matrix4<f64>>> {
        Ok(self.skeleton().world_matrices(pose)?)
    }

    /// Linear blend skinning of one prior mesh's bind positions.
    pub fn skin_mesh(&self, mesh: usize, worlds: &[Matrix4<f64>]) -> Vec<Vec3> {
        self.skin_positions(mesh, &self.meshes[mesh].bind, worlds)
    }

    /// Linear blend skinning of given bind positions with one mesh's weights.
    pub fn skin_positions(&self, mesh: usize, bind: &[Vec3], worlds: &[Matrix4<f64>]) -> Vec<Vec3> {
        let palette: HashMap<usize, Matrix4<f64>> = self
            .skin_joints
            .iter()
            .zip(&self.ibm)
            .map(|(&j, ibm)| (j, worlds[j] * ibm))
            .collect();
        let m = &self.meshes[mesh];
        bind.par_iter()
            .zip(&m.influences)
            .map(|(p, row)| {
                let p = v(*p).push(1.0);
                let s = row.iter().fold(nalgebra::Vector4::zeros(), |s, w| {
                    s + palette[&w.joint] * p * w.weight
                });
                a(s.xyz())
            })
            .collect()
    }

    pub fn dominant_joint(&self, mesh: usize, vertex: usize) -> usize {
        self.meshes[mesh].influences[vertex]
            .iter()
            .max_by(|x, y| x.weight.total_cmp(&y.weight))
            .map_or(0, |w| w.joint)
    }

    pub fn height(&self) -> f64 {
        let ys = self.body().prim.positions.iter().map(|p| p[1]);
        let (lo, hi) = ys.fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), y| {
            (l.min(y), h.max(y))
        });
        hi - lo
    }

    pub fn is_ancestor(&self, ancestor: usize, mut node: usize) -> bool {
        loop {
            if node == ancestor {
                return true;
            }
            match self.skeleton().joints[node].parent {
                Some(p) => node = p,
                None => return false,
            }
        }
    }
}
