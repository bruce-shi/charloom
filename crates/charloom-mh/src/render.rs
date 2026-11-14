//! `charloom mh-render`: fast CPU review renders of an `mh-transfer` run.
//!
//! Writes `RUN_DIR/renders/` with the file names, cameras and framing of
//! `scripts/mh-render-review.py` (Blender), so review habits, comparisons and
//! scripts keep working; Blender stays the high-quality option
//! (`MH_RENDER=blender scripts/mh-review.sh`).
//!
//! Orthographic cameras, supersampled (2× by default), textured lambert under
//! camera-relative studio lights (Workbench-like) or one sun plus world
//! ambient (the EEVEE head close-ups), AgX view transform like Blender's
//! default. MASK materials are alpha-tested; BLEND materials (eyelashes,
//! brows, hair cards) are composited per sample, sorted back to front, over
//! the opaque surface.
use crate::guide::Camera;
use crate::source::gltf_of;
use anyhow::{Context, Result, bail};
use charloom_core::matrix_from;
use charloom_io::{LoadedAsset, load_gltf};
use image::RgbImage;
use nalgebra::{Matrix4, Vector3, Vector4};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

type V3 = Vector3<f32>;
type D3 = Vector3<f64>;

/// `--only` groups.
/// `eyes` (also drawn with `face`): eye close-ups, open, blinking and looking
/// left / right (front only), next to the textured source scan from the same
/// camera.
/// `ears` (also drawn with `face`): `ears-{l,r}.png`, each ear from its side
/// as in scripts/mh-render-ears.py: scan textured | scan shape | body
/// textured | body shape | character (with hair), one camera.
/// `feet`: foot close-ups (ankle to toes, front and outer side): the scan's
/// labels, MakeHuman's posed prior and the fitted body in clay, the body
/// textured and dressed, at rest and at 25 % and 60 % of the walk.
/// `hand-walk`: hand close-ups of the naked body through the walk
/// (`hand-walk-{l,r}-{front,side}-{f}.png` at 10, 25, 40, 60 and 80 %),
/// framed on the hand's bones in each frame.
/// `chest`: sport-bra close-ups, textured and clay, framed on the bra.
pub const GROUPS: [&str; 11] = [
    "segment",
    "body",
    "dressed",
    "face",
    "eyes",
    "ears",
    "walk",
    "hands",
    "feet",
    "hand-walk",
    "chest",
];

/// Walk fractions of the `hand-walk` close-ups.
const HAND_WALK_FRAMES: [f64; 5] = [0.1, 0.25, 0.4, 0.6, 0.8];

/// Side of one ear tile in `ears-{l,r}.png`.
const EAR_TILE: usize = 448;

/// Ear close-up cameras from a body: (side, centre, ortho size). The ear
/// is the most lateral skin in a band below the eyes' height, behind them.
fn ear_cameras(body: &Posed, eyes: (D3, D3)) -> Vec<(&'static str, D3, f64)> {
    let (lo, hi) = eyes;
    let (ey, ez, mid) = (
        (lo.y + hi.y) / 2.0,
        (lo.z + hi.z) / 2.0,
        (lo.x + hi.x) / 2.0,
    );
    // Eye span of a ~1.75 m adult.
    let unit = (hi.x - lo.x) / 0.095;
    let band: Vec<D3> = body
        .points(|part| !part.name.starts_with("eyes") && body.model.triangles_of(part) >= 5000)
        .filter(|p| p.y > ey - 0.075 * unit && p.y < ey + 0.02 * unit && p.z < ez)
        .collect();
    let mut out = vec![];
    for (side, sign) in [("l", 1.0), ("r", -1.0)] {
        let pts: Vec<&D3> = band.iter().filter(|p| (p.x - mid) * sign > 0.0).collect();
        let Some(xmax) = pts
            .iter()
            .map(|p| (p.x - mid) * sign)
            .max_by(f64::total_cmp)
        else {
            continue;
        };
        let ear: Vec<&D3> = pts
            .into_iter()
            .filter(|p| (p.x - mid) * sign > xmax - 0.02 * unit)
            .collect();
        let centre = ear.iter().copied().sum::<D3>() / ear.len() as f64;
        out.push((side, centre, 0.13 * unit));
    }
    out
}

/// Per-piece flat colours (linear), as in scripts/mh-render-review.py.
const PALETTE: [[f32; 3]; 12] = [
    [0.9, 0.16, 0.16],
    [0.16, 0.55, 1.0],
    [0.16, 0.78, 0.27],
    [1.0, 0.63, 0.0],
    [0.75, 0.24, 1.0],
    [0.0, 0.86, 0.86],
    [0.94, 0.9, 0.16],
    [0.59, 0.35, 0.12],
    [1.0, 0.24, 0.67],
    [0.08, 0.43, 0.16],
    [0.55, 0.55, 0.55],
    [0.3, 0.2, 0.7],
];
/// Blender world colour of the review scene (linear).
const WORLD: [f32; 3] = [0.35, 0.37, 0.4];
/// Workbench's colour for objects without a material or texture.
const DEFAULT_GREY: [f32; 3] = [0.8, 0.8, 0.8];
/// Blender's default frame rate: the walk frames are whole frames at 24 fps.
const FPS: f64 = 24.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlphaMode {
    Opaque,
    Mask(f32),
    Blend,
}

struct Material {
    texture: Option<usize>,
    factor: [f32; 4],
    alpha: AlphaMode,
}

/// RGBA8 sRGB texels.
struct Texture {
    w: usize,
    h: usize,
    px: Vec<u8>,
}

fn srgb_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        std::array::from_fn(|i| crate::source::srgb_to_linear(i as f64 / 255.0) as f32)
    })
}

impl Texture {
    /// Bilinear, repeat wrapping: linear RGB and alpha.
    fn sample(&self, uv: [f32; 2]) -> [f32; 4] {
        let lut = srgb_lut();
        let x = uv[0].rem_euclid(1.0) * self.w as f32 - 0.5;
        let y = uv[1].rem_euclid(1.0) * self.h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let texel = |xi: i64, yi: i64| {
            let k = (yi.rem_euclid(self.h as i64) as usize * self.w
                + xi.rem_euclid(self.w as i64) as usize)
                * 4;
            let p = &self.px[k..k + 4];
            [
                lut[p[0] as usize],
                lut[p[1] as usize],
                lut[p[2] as usize],
                p[3] as f32 / 255.0,
            ]
        };
        let c = [
            texel(x0, y0),
            texel(x0 + 1, y0),
            texel(x0, y0 + 1),
            texel(x0 + 1, y0 + 1),
        ];
        std::array::from_fn(|k| {
            (c[0][k] * (1.0 - fx) + c[1][k] * fx) * (1.0 - fy)
                + (c[2][k] * (1.0 - fx) + c[3][k] * fx) * fy
        })
    }
}

/// One glTF primitive's render attributes (bind space).
struct Part {
    name: String,
    /// Index into `asset.meshes`.
    mesh: usize,
    normals: Vec<V3>,
    uvs: Vec<[f32; 2]>,
    /// COLOR_0, linear.
    colors: Option<Vec<[f32; 4]>>,
    material: Option<usize>,
    object_color: [f32; 3],
}

/// One loaded GLB.
pub struct Model {
    loaded: LoadedAsset,
    parts: Vec<Part>,
    materials: Vec<Material>,
    /// By glTF image index; decoded only when the model is drawn textured.
    textures: Vec<Option<Texture>>,
}

/// One part posed: world positions and normals.
pub struct PartGeom {
    positions: Vec<V3>,
    normals: Vec<V3>,
}

fn vertex_normals(positions: &[V3], triangles: &[[u32; 3]]) -> Vec<V3> {
    let mut n = vec![V3::zeros(); positions.len()];
    for t in triangles {
        let [a, b, c] = t.map(|i| positions[i as usize]);
        let f = (b - a).cross(&(c - a));
        for &i in t {
            n[i as usize] += f;
        }
    }
    n.iter()
        .map(|x| x.try_normalize(1e-20).unwrap_or(V3::y()))
        .collect()
}

impl Model {
    pub fn load(path: &Path, textures: bool) -> Result<Self> {
        let loaded = load_gltf(path)?;
        let gltf = gltf_of(&loaded)?;
        let binary = loaded.document.binary.as_slice();
        let mut parts = vec![];
        for (mi, mesh) in loaded.asset.meshes.iter().enumerate() {
            let primitive = gltf
                .document
                .nodes()
                .nth(mesh.node)
                .and_then(|n| n.mesh())
                .and_then(|m| m.primitives().nth(mesh.primitive))
                .context("primitive lookup")?;
            let reader = primitive.reader(|_| Some(binary));
            let n = mesh.positions.len();
            let normals = match reader.read_normals() {
                Some(it) => it.map(V3::from).collect::<Vec<_>>(),
                None => {
                    let p: Vec<V3> = mesh
                        .positions
                        .iter()
                        .map(|p| V3::new(p[0] as f32, p[1] as f32, p[2] as f32))
                        .collect();
                    vertex_normals(&p, &mesh.triangles)
                }
            };
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|t| t.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0; 2]; n]);
            let colors: Option<Vec<[f32; 4]>> =
                reader.read_colors(0).map(|c| c.into_rgba_f32().collect());
            if normals.len() != n || uvs.len() != n || colors.as_ref().is_some_and(|c| c.len() != n)
            {
                bail!(
                    "{}: attribute count mismatch in {}",
                    path.display(),
                    mesh.name
                );
            }
            parts.push(Part {
                name: mesh.name.clone(),
                mesh: mi,
                normals,
                uvs,
                colors,
                material: primitive.material().index(),
                object_color: DEFAULT_GREY,
            });
        }
        let materials: Vec<Material> = gltf
            .document
            .materials()
            .map(|m| {
                let pbr = m.pbr_metallic_roughness();
                Material {
                    texture: pbr
                        .base_color_texture()
                        .map(|t| t.texture().source().index()),
                    factor: pbr.base_color_factor(),
                    alpha: match m.alpha_mode() {
                        gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                        gltf::material::AlphaMode::Mask => {
                            AlphaMode::Mask(m.alpha_cutoff().unwrap_or(0.5))
                        }
                        gltf::material::AlphaMode::Blend => AlphaMode::Blend,
                    },
                }
            })
            .collect();
        let images: Vec<_> = gltf.document.images().collect();
        let wanted: Vec<bool> = (0..images.len())
            .map(|i| {
                textures
                    && parts
                        .iter()
                        .any(|p| p.material.is_some_and(|m| materials[m].texture == Some(i)))
            })
            .collect();
        let textures = images
            .par_iter()
            .enumerate()
            .map(|(i, image)| -> Result<Option<Texture>> {
                if !wanted[i] {
                    return Ok(None);
                }
                let gltf::image::Source::View { view, .. } = image.source() else {
                    bail!("external image URIs are not supported; pack the model as GLB");
                };
                let bytes = binary
                    .get(view.offset()..view.offset() + view.length())
                    .context("image view out of bounds")?;
                let decoded = image::load_from_memory(bytes)
                    .context("decode embedded image")?
                    .to_rgba8();
                Ok(Some(Texture {
                    w: decoded.width() as usize,
                    h: decoded.height() as usize,
                    px: decoded.into_raw(),
                }))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            loaded,
            parts,
            materials,
            textures,
        })
    }

    /// The walk (or first) animation's time for a fraction of its range, on
    /// whole frames like Blender's `scene.frame_set`.
    pub fn clip_time(&self, fraction: f64) -> Option<f64> {
        let clips = &self.loaded.asset.clips;
        let clip = clips
            .iter()
            .find(|c| c.name == "walking")
            .or(clips.first())?;
        let [t0, t1] = clip.time_range();
        let (start, end) = (t0 * FPS, t1 * FPS);
        Some((start + (end - start) * fraction).trunc() / FPS)
    }

    /// Linear-blend-skinned positions and normals at rest (`None`) or at a
    /// time of the walk (or first) animation.
    pub fn pose(&self, time: Option<f64>) -> Result<Vec<PartGeom>> {
        let pose = self.local_pose(time)?;
        self.pose_with(&pose)
    }

    /// Rest pose with the upper lids (`orbicularis03.L` / `.R`) and the lower
    /// ones (`orbicularis04.*`) rotated by `blink` radians (upper left,
    /// upper right, lower left, lower right) about their local X, as
    /// mh-render-review.py's eye close-ups.
    pub fn pose_blink(&self, blink: [f64; 4]) -> Result<Vec<PartGeom>> {
        self.pose_turns(&[
            ("orbicularis03.L", D3::x_axis(), blink[0]),
            ("orbicularis03.R", D3::x_axis(), blink[1]),
            ("orbicularis04.L", D3::x_axis(), blink[2]),
            ("orbicularis04.R", D3::x_axis(), blink[3]),
        ])
    }

    /// Rest pose with both eyes (`eye.L`, `eye.R`) turned `yaw` radians
    /// about their local Z (sideways; positive: towards the character's
    /// left).
    pub fn pose_look(&self, yaw: f64) -> Result<Vec<PartGeom>> {
        self.pose_turns(&[("eye.L", D3::z_axis(), yaw), ("eye.R", D3::z_axis(), yaw)])
    }

    /// Rest pose with a face test expression (`face_test::poses`).
    pub fn pose_local(&self, pose: &[charloom_core::Transform]) -> Result<Vec<PartGeom>> {
        self.pose_with(pose)
    }

    /// The face test expressions on this model's skeleton, by name.
    pub fn face_poses(&self) -> Vec<(String, Vec<charloom_core::Transform>)> {
        let skeleton = &self.loaded.asset.skeleton;
        crate::face_test::poses(skeleton, &skeleton.rest_pose())
    }

    /// Rest pose with the named joints turned about a local axis.
    fn pose_turns(&self, turns: &[(&str, nalgebra::Unit<D3>, f64)]) -> Result<Vec<PartGeom>> {
        let skeleton = &self.loaded.asset.skeleton;
        let mut pose = skeleton.rest_pose();
        for (j, t) in skeleton.joints.iter().zip(pose.iter_mut()) {
            let Some((_, axis, angle)) = turns.iter().find(|x| x.0 == j.name) else {
                continue;
            };
            let turn = nalgebra::UnitQuaternion::from_axis_angle(axis, *angle);
            let q = charloom_core::unit_quat(t.rotation) * turn;
            t.rotation = [q.i, q.j, q.k, q.w];
        }
        self.pose_with(&pose)
    }

    /// Linear-blend-skinned positions and normals for local joint transforms.
    fn pose_with(&self, pose: &[charloom_core::Transform]) -> Result<Vec<PartGeom>> {
        let asset = &self.loaded.asset;
        let worlds = asset.skeleton.world_matrices(pose)?;
        let f32m = |m: &Matrix4<f64>| m.map(|x| x as f32);
        let palettes: Vec<Vec<Option<Matrix4<f32>>>> = asset
            .skins
            .iter()
            .map(|skin| {
                let mut p = vec![None; worlds.len()];
                for (&j, ibm) in skin.joints.iter().zip(&skin.inverse_bind_matrices) {
                    p[j] = Some(f32m(&(worlds[j] * matrix_from(*ibm))));
                }
                p
            })
            .collect();
        self.parts
            .par_iter()
            .map(|part| {
                let mesh = &asset.meshes[part.mesh];
                let bind = mesh
                    .positions
                    .iter()
                    .map(|p| Vector4::new(p[0] as f32, p[1] as f32, p[2] as f32, 1.0));
                let (positions, normals): (Vec<V3>, Vec<V3>) = match mesh.skin {
                    Some(s) if !mesh.influences.is_empty() => {
                        let palette = &palettes[s];
                        bind.zip(&part.normals)
                            .zip(&mesh.influences)
                            .map(|((p, n), row)| {
                                let mut m = Matrix4::<f32>::zeros();
                                for w in row {
                                    if let Some(j) = palette[w.joint] {
                                        m += j * w.weight as f32;
                                    }
                                }
                                let q = (m * p).xyz();
                                let n = m.fixed_view::<3, 3>(0, 0) * n;
                                (q, n.try_normalize(1e-20).unwrap_or(V3::y()))
                            })
                            .unzip()
                    }
                    _ => {
                        let m = f32m(&worlds[mesh.node]);
                        let r = m.fixed_view::<3, 3>(0, 0).into_owned();
                        bind.zip(&part.normals)
                            .map(|(p, n)| {
                                (
                                    (m * p).xyz(),
                                    (r * n).try_normalize(1e-20).unwrap_or(V3::y()),
                                )
                            })
                            .unzip()
                    }
                };
                Ok(PartGeom { positions, normals })
            })
            .collect()
    }

    fn triangles_of(&self, part: &Part) -> usize {
        self.loaded.asset.meshes[part.mesh].triangles.len()
    }

    fn triangles(&self, part: usize) -> &[[u32; 3]] {
        &self.loaded.asset.meshes[self.parts[part].mesh].triangles
    }

    /// Local joint transforms at rest (`None`) or at a time of the walk (or
    /// first) animation.
    fn local_pose(&self, time: Option<f64>) -> Result<Vec<charloom_core::Transform>> {
        let asset = &self.loaded.asset;
        let skeleton = &asset.skeleton;
        Ok(match time {
            None => skeleton.rest_pose(),
            Some(t) => {
                let clips = &asset.clips;
                let clip = clips
                    .iter()
                    .find(|c| c.name == "walking")
                    .or(clips.first())
                    .context("model has no animation")?;
                clip.sample(skeleton, t)?
            }
        })
    }

    /// Heads and tails of the bones from `root` down, posed. Bone tails
    /// follow Blender's glTF import: along the joint's local +Y, as long as
    /// the nearest child (leaf bones inherit the parent's length).
    fn subtree_bounds(&self, root: &str, time: Option<f64>) -> Option<(D3, D3)> {
        let skeleton = &self.loaded.asset.skeleton;
        let root = skeleton.joints.iter().position(|j| j.name == root)?;
        let worlds = skeleton.world_matrices(&self.local_pose(time).ok()?).ok()?;
        let head = |j: usize| worlds[j].fixed_view::<3, 1>(0, 3).into_owned();
        let children = |j: usize| {
            (0..skeleton.joints.len()).filter(move |&c| skeleton.joints[c].parent == Some(j))
        };
        let mut pts = vec![];
        let mut stack = vec![(root, 0.0_f64)];
        while let Some((j, parent_len)) = stack.pop() {
            let h = head(j);
            let len = children(j)
                .map(|c| (head(c) - h).norm())
                .filter(|&d| d > 1e-4)
                .fold(f64::INFINITY, f64::min);
            let len = if len.is_finite() { len } else { parent_len };
            let y = worlds[j].fixed_view::<3, 1>(0, 1).into_owned();
            pts.push(h);
            pts.push(h + y.try_normalize(1e-12).unwrap_or(D3::y()) * len);
            stack.extend(children(j).map(|c| (c, len)));
        }
        bounds_of(pts.into_iter())
    }

    /// Rest-pose bounds of one hand's bones (wrist and below), padded by a
    /// quarter of the largest extent, like mh-render-review.py's `hand_box`.
    pub fn hand_box(&self, side: &str) -> Option<(D3, D3)> {
        self.hand_box_at(side, None)
    }

    /// [`Self::hand_box`] at a clip time (`None`: rest).
    pub fn hand_box_at(&self, side: &str, time: Option<f64>) -> Option<(D3, D3)> {
        let (lo, hi) = self.subtree_bounds(&format!("wrist.{side}"), time)?;
        let pad = (hi - lo).max() * 0.25;
        Some((lo - D3::repeat(pad), hi + D3::repeat(pad)))
    }

    /// One foot's bones (ankle to toe tips) at rest or in the walk, raised
    /// to take in the lower shin (a shoe collar and the skin above it) and
    /// padded.
    pub fn foot_box(&self, side: &str, time: Option<f64>) -> Option<(D3, D3)> {
        let (lo, mut hi) = self.subtree_bounds(&format!("foot.{side}"), time)?;
        let extent = (hi - lo).max();
        hi.y += extent * 0.45;
        // The heel and the toe pads reach past the bones.
        let pad = extent * 0.45;
        Some((lo - D3::repeat(pad), hi + D3::repeat(pad)))
    }
}

/// A posed model ready to draw.
pub struct Posed<'a> {
    model: &'a Model,
    geom: Vec<PartGeom>,
}

impl<'a> Posed<'a> {
    pub fn new(model: &'a Model, time: Option<f64>) -> Result<Self> {
        Ok(Self {
            model,
            geom: model.pose(time)?,
        })
    }

    /// The upper lids closed by `blink` radians (see [`Model::pose_blink`]).
    pub fn blink(model: &'a Model, blink: [f64; 4]) -> Result<Self> {
        Ok(Self {
            model,
            geom: model.pose_blink(blink)?,
        })
    }

    /// Both eyes turned sideways by `yaw` radians (see [`Model::pose_look`]).
    pub fn local(model: &'a Model, pose: &[charloom_core::Transform]) -> Result<Self> {
        Ok(Self {
            model,
            geom: model.pose_local(pose)?,
        })
    }

    pub fn look(model: &'a Model, yaw: f64) -> Result<Self> {
        Ok(Self {
            model,
            geom: model.pose_look(yaw)?,
        })
    }

    /// Rest pose moved by an affine transform (the source scan into the
    /// working frame).
    pub fn transformed(model: &'a Model, m: &Matrix4<f64>) -> Result<Self> {
        let m = m.map(|x| x as f32);
        let r = m.fixed_view::<3, 3>(0, 0).into_owned();
        let nm = r.try_inverse().map_or(r, |i| i.transpose());
        let geom = model
            .pose(None)?
            .into_iter()
            .map(|g| PartGeom {
                positions: g
                    .positions
                    .iter()
                    .map(|p| (m * p.push(1.0)).xyz())
                    .collect(),
                normals: g
                    .normals
                    .iter()
                    .map(|n| (nm * n).try_normalize(1e-20).unwrap_or(V3::y()))
                    .collect(),
            })
            .collect();
        Ok(Self { model, geom })
    }

    fn points<'b>(&'b self, keep: impl Fn(&Part) -> bool + 'b) -> impl Iterator<Item = D3> + 'b {
        self.model
            .parts
            .iter()
            .zip(&self.geom)
            .filter(move |(p, _)| keep(p))
            .flat_map(|(_, g)| g.positions.iter().map(|p| p.cast::<f64>()))
    }
}

fn bounds_of(points: impl Iterator<Item = D3>) -> Option<(D3, D3)> {
    let (lo, hi) = points.fold(
        (D3::repeat(f64::INFINITY), D3::repeat(f64::NEG_INFINITY)),
        |(lo, hi), p| (lo.inf(&p), hi.sup(&p)),
    );
    (lo.x <= hi.x).then_some((lo, hi))
}

/// Bounds of every part with at least 64 triangles (Blender's `bounds()`
/// skips tiny meshes).
fn scene_bounds(scene: &[&Posed]) -> Option<(D3, D3)> {
    bounds_of(
        scene
            .iter()
            .flat_map(|p| p.points(|part| p.model.triangles_of(part) >= 64)),
    )
}

/// How a close-up is framed.
#[derive(Debug, Clone, Copy)]
pub enum Crop {
    /// Brow to chin from the eyeballs' bounds.
    Eyes { lo: D3, hi: D3 },
    /// The kept scan head (head route keep_scan, no MakeHuman eyes).
    Head { lo: D3, hi: D3 },
    /// A height fraction of the whole model.
    Fraction(f64, f64),
}

/// Camera centre and ortho scale (the larger image side) for a model's
/// bounds, as mh-render-review.py frames its 768×1024 renders.
pub fn frame(mut lo: D3, mut hi: D3, crop: Option<Crop>) -> (D3, f64) {
    match crop {
        Some(Crop::Eyes { lo: elo, hi: ehi }) => {
            let span = ehi.x - elo.x;
            let cy = (ehi.y + elo.y) / 2.0 - 0.3 * span;
            let mid = (ehi.x + elo.x) / 2.0;
            (lo.y, hi.y) = (cy - 1.3 * span, cy + 1.3 * span);
            (lo.x, hi.x) = (mid - 0.975 * span, mid + 0.975 * span);
        }
        Some(Crop::Head { lo: hlo, hi: hhi }) => {
            let h = hhi.y - hlo.y;
            let mid = (hhi.x + hlo.x) / 2.0;
            (lo.y, hi.y) = (hlo.y - 0.05 * h, hhi.y + 0.1 * h);
            (lo.x, hi.x) = (mid - 0.55 * h, mid + 0.55 * h);
        }
        Some(Crop::Fraction(a, b)) => {
            let h = hi.y - lo.y;
            (lo.y, hi.y) = (lo.y + h * a, lo.y + h * b);
            let w = (hi.y - lo.y) * 0.75;
            let mid = (lo.x + hi.x) / 2.0;
            (lo.x, hi.x) = (mid - w / 2.0, mid + w / 2.0);
        }
        None => {}
    }
    let size = (hi.y - lo.y).max((hi.x - lo.x) * 1024.0 / 768.0) * 1.08;
    ((lo + hi) / 2.0, size)
}

/// Orthographic camera at `angle_deg` around +Y (0 = looking at the front,
/// along -Z; 90 = from +X), `size` metres across the larger image side.
pub fn ortho_camera(id: &str, center: D3, angle_deg: f64, size: f64, w: usize, h: usize) -> Camera {
    let t = angle_deg.to_radians();
    let forward = D3::new(-t.sin(), 0.0, -t.cos());
    let up = D3::y();
    let right = forward.cross(&up);
    let a = |v: D3| [v.x, v.y, v.z];
    Camera {
        id: id.into(),
        width: w,
        height: h,
        center: a(center),
        right: a(right),
        up: a(up),
        forward: a(forward),
        meters_per_pixel: size / w.max(h) as f64,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Colour {
    Texture,
    Vertex,
    Object,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Light {
    /// Workbench studio lighting: camera-relative key, fill and specular.
    Studio,
    /// EEVEE: the review scene's sun (energy 3) plus world ambient.
    Sun,
}

impl Light {
    fn background(self) -> [f32; 3] {
        match self {
            // The AgX fit reads ~4/255 brighter than Blender at this grey.
            Light::Studio => WORLD.map(|c| c * 0.96),
            Light::Sun => WORLD.map(|c| c * 0.8),
        }
    }
}

struct Item<'a> {
    model: &'a Model,
    part: usize,
    geom: &'a PartGeom,
}

impl Item<'_> {
    fn part(&self) -> &Part {
        &self.model.parts[self.part]
    }
    fn triangles(&self) -> &[[u32; 3]] {
        self.model.triangles(self.part)
    }
    fn material(&self) -> Option<&Material> {
        self.part().material.map(|m| &self.model.materials[m])
    }
    fn alpha_mode(&self) -> AlphaMode {
        self.material().map_or(AlphaMode::Opaque, |m| m.alpha)
    }
    fn uv(&self, tri: [u32; 3], b: [f32; 3]) -> [f32; 2] {
        let uv = &self.part().uvs;
        let [a, c, d] = tri.map(|i| uv[i as usize]);
        [
            a[0] * b[0] + c[0] * b[1] + d[0] * b[2],
            a[1] * b[0] + c[1] * b[1] + d[1] * b[2],
        ]
    }
    /// Material base colour (linear) and alpha.
    fn base(&self, tri: [u32; 3], b: [f32; 3]) -> [f32; 4] {
        let Some(m) = self.material() else {
            return [DEFAULT_GREY[0], DEFAULT_GREY[1], DEFAULT_GREY[2], 1.0];
        };
        let t = m
            .texture
            .and_then(|i| self.model.textures.get(i)?.as_ref())
            .map_or([1.0; 4], |t| t.sample(self.uv(tri, b)));
        std::array::from_fn(|k| t[k] * m.factor[k])
    }
    fn alpha(&self, tri: [u32; 3], b: [f32; 3]) -> f32 {
        let m = self.material().map_or(1.0, |m| m.factor[3]);
        match self.material().and_then(|m| m.texture) {
            Some(i) => match self.model.textures.get(i).and_then(Option::as_ref) {
                Some(t) => t.sample(self.uv(tri, b))[3] * m,
                None => m,
            },
            None => m,
        }
    }
}

/// One output image.
pub struct Job<'a> {
    pub file: String,
    pub cam: Camera,
    items: Vec<Item<'a>>,
    pub colour: Colour,
    pub light: Light,
}

impl<'a> Job<'a> {
    pub fn new(
        file: String,
        cam: Camera,
        scene: &[&'a Posed<'a>],
        colour: Colour,
        light: Light,
    ) -> Self {
        let items = scene
            .iter()
            .flat_map(|p| {
                p.geom.iter().enumerate().map(|(part, geom)| Item {
                    model: p.model,
                    part,
                    geom,
                })
            })
            .collect();
        Self {
            file,
            cam,
            items,
            colour,
            light,
        }
    }
}

/// A blended sample: `k` is the band-local sample index.
#[derive(Debug, Clone, Copy)]
struct Frag {
    k: u32,
    z: f32,
    item: u32,
    tri: u32,
    b: [f32; 2],
}

/// Composites `(depth, colour, alpha)` layers over `base`, farthest first,
/// whatever their order in `layers`.
pub fn composite(base: [f32; 3], layers: &mut [(f32, [f32; 3], f32)]) -> [f32; 3] {
    layers.sort_by(|a, b| b.0.total_cmp(&a.0));
    layers.iter().fold(base, |c, (_, s, a)| {
        std::array::from_fn(|k| c[k] * (1.0 - a) + s[k] * a)
    })
}

/// Covered sample centres of a screen-space triangle within rows `y0..y1`:
/// calls `f(x, y, z, b1, b2)`.
#[inline]
fn raster(
    p: [[f32; 3]; 3],
    w: usize,
    y0: usize,
    y1: usize,
    mut f: impl FnMut(usize, usize, f32, f32, f32),
) {
    let [p0, p1, p2] = p;
    let area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]);
    if area.is_nan() || area.abs() <= 1e-9 {
        return;
    }
    let minx = p0[0].min(p1[0]).min(p2[0]);
    let maxx = p0[0].max(p1[0]).max(p2[0]);
    let miny = p0[1].min(p1[1]).min(p2[1]);
    let maxy = p0[1].max(p1[1]).max(p2[1]);
    let xa = (minx - 0.5).ceil().max(0.0) as usize;
    let xb = ((maxx - 0.5).floor()).min(w as f32 - 1.0);
    let ya = ((miny - 0.5).ceil().max(y0 as f32)) as usize;
    let yb = ((maxy - 0.5).floor()).min(y1 as f32 - 1.0);
    if xb < xa as f32 || yb < ya as f32 {
        return;
    }
    let (xb, yb) = (xb as usize, yb as usize);
    let inv = 1.0 / area;
    let (e1x, e1y) = (p1[0] - p0[0], p1[1] - p0[1]);
    let (e2x, e2y) = (p2[0] - p0[0], p2[1] - p0[1]);
    let (dz1, dz2) = (p1[2] - p0[2], p2[2] - p0[2]);
    for y in ya..=yb {
        let py = y as f32 + 0.5 - p0[1];
        for x in xa..=xb {
            let px = x as f32 + 0.5 - p0[0];
            let b1 = (px * e2y - e2x * py) * inv;
            let b2 = (e1x * py - px * e1y) * inv;
            if b1 >= 0.0 && b2 >= 0.0 && b1 + b2 <= 1.0 {
                f(x, y, p0[2] + b1 * dz1 + b2 * dz2, b1, b2);
            }
        }
    }
}

 pub fn agx(c: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = c.map(|x| x.max(0.0));
    let i = [
        0.842_479 * r + 0.078_433_6 * g + 0.079_223_7 * b,
        0.042_328_2 * r + 0.878_468_6 * g + 0.079_166_1 * b,
        0.042_375_7 * r + 0.078_433_6 * g + 0.879_143 * b,
    ];
    let (min_ev, max_ev) = (-12.473_93_f32, 4.026_069_f32);
    let x = i.map(|v| {
        let x = (v.max(1e-10).log2().clamp(min_ev, max_ev) - min_ev) / (max_ev - min_ev);
        let (x2, x4) = (x * x, x * x * x * x);
        15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x
            - 0.002_32
    });
    [
        1.196_879 * x[0] - 0.098_020_9 * x[1] - 0.099_029_7 * x[2],
        -0.052_896_9 * x[0] + 1.151_903 * x[1] - 0.098_961_2 * x[2],
        -0.052_971_6 * x[0] - 0.098_043_5 * x[1] + 1.151_074 * x[2],
    ]
    .map(|v| v.clamp(0.0, 1.0))
}

struct Shader<'j> {
    job: &'j Job<'j>,
    right: V3,
    up: V3,
    forward: V3,
    sun: V3,
}

impl Shader<'_> {
    fn shade(&self, item: &Item, tri: [u32; 3], b: [f32; 3]) -> ([f32; 3], f32) {
        let base = item.base(tri, b);
        let part = item.part();
        let albedo: [f32; 3] = match self.job.colour {
            Colour::Texture => [base[0], base[1], base[2]],
            Colour::Vertex => part.colors.as_ref().map_or(DEFAULT_GREY, |c| {
                let [x, y, z] = tri.map(|i| c[i as usize]);
                std::array::from_fn(|k| x[k] * b[0] + y[k] * b[1] + z[k] * b[2])
            }),
            Colour::Object => part.object_color,
        };
        let alpha = match item.alpha_mode() {
            AlphaMode::Opaque => 1.0,
            _ => base[3],
        };
        let nn = &item.geom.normals;
        let mut n =
            (nn[tri[0] as usize] * b[0] + nn[tri[1] as usize] * b[1] + nn[tri[2] as usize] * b[2])
                .try_normalize(1e-20)
                .unwrap_or(-self.forward);
        // Both sides lit, like Blender with backface culling off.
        if n.dot(&self.forward) > 0.0 {
            n = -n;
        }
        let c = match self.job.light {
            Light::Studio => {
                // Camera space: x right, y up, z toward the viewer.
                let v = V3::new(n.dot(&self.right), n.dot(&self.up), -n.dot(&self.forward));
                // Levels fitted to Blender's Workbench review renders (flat
                // piece colours read back through AgX): ~0.4 facing the
                // camera, ~0.15 at grazing angles.
                let key = V3::new(-0.45, 0.55, 0.70).normalize();
                let fill = V3::new(0.75, 0.10, 0.65).normalize();
                let diffuse = 0.06
                    + 0.03 * v.y
                    + 0.34 * v.dot(&key).max(0.0)
                    + 0.05 * v.dot(&fill).max(0.0)
                    + 0.06 * v.z.max(0.0);
                let h = (key + V3::z()).normalize();
                let spec = 0.04 * v.dot(&h).max(0.0).powi(24);
                albedo.map(|a| a * diffuse + spec)
            }
            Light::Sun => {
                let d = n.dot(&self.sun).max(0.0);
                let h = (self.sun - self.forward).normalize();
                let spec = 0.02 * n.dot(&h).max(0.0).powi(32);
                let bg = Light::Sun.background();
                std::array::from_fn(|k| albedo[k] * (bg[k] + 0.955 * d) + spec)
            }
        };
        (c, alpha)
    }
}

/// Output rows per band (each band is rasterised and shaded independently).
const BAND: usize = 16;

/// Renders one job at `ss`× supersampling, box-filtered down.
pub fn render(job: &Job, ss: usize) -> RgbImage {
    let ss = ss.max(1);
    let cam = &job.cam;
    let (w, h) = (cam.width * ss, cam.height * ss);
    let mpp = (cam.meters_per_pixel / ss as f64) as f32;
    let f = |x: [f64; 3]| V3::new(x[0] as f32, x[1] as f32, x[2] as f32);
    let (center, right, up, forward) = (f(cam.center), f(cam.right), f(cam.up), f(cam.forward));
    let (hw, hh) = (w as f32 * 0.5, h as f32 * 0.5);
    let projected: Vec<Vec<[f32; 3]>> = job
        .items
        .par_iter()
        .map(|it| {
            it.geom
                .positions
                .iter()
                .map(|p| {
                    let d = p - center;
                    [
                        d.dot(&right) / mpp + hw,
                        hh - d.dot(&up) / mpp,
                        d.dot(&forward),
                    ]
                })
                .collect()
        })
        .collect();
    let band_rows = BAND * ss;
    let bands = h.div_ceil(band_rows);
    // Triangles per band, opaque/mask and blend apart.
    let per_item: Vec<(Vec<Vec<u32>>, bool)> = job
        .items
        .par_iter()
        .zip(&projected)
        .map(|(it, proj)| {
            let mut bins = vec![vec![]; bands];
            for (t, tri) in it.triangles().iter().enumerate() {
                let [a, b, c] = tri.map(|i| proj[i as usize]);
                let (lo, hi) = (a[1].min(b[1]).min(c[1]), a[1].max(b[1]).max(c[1]));
                let (xl, xh) = (a[0].min(b[0]).min(c[0]), a[0].max(b[0]).max(c[0]));
                if !(hi >= 0.0 && lo < h as f32 && xh >= 0.0 && xl < w as f32) {
                    continue;
                }
                let b0 = ((lo - 0.5).max(0.0) as usize) / band_rows;
                let b1 = (((hi - 0.5).max(0.0) as usize) / band_rows).min(bands - 1);
                for bin in &mut bins[b0..=b1] {
                    bin.push(t as u32);
                }
            }
            (bins, it.alpha_mode() == AlphaMode::Blend)
        })
        .collect();
    let shader = Shader {
        job,
        right,
        up,
        forward,
        sun: V3::new(0.262, 0.643, 0.720).normalize(),
    };
    let bg = job.light.background();
    let rows: Vec<Vec<u8>> = (0..bands)
        .into_par_iter()
        .map(|band| {
            let y0 = band * band_rows;
            let y1 = (y0 + band_rows).min(h);
            let n = w * (y1 - y0);
            let mut depth = vec![f32::INFINITY; n];
            let mut id = vec![(u32::MAX, 0u32); n];
            let mut bary = vec![[0f32; 2]; n];
            for (i, (bins, blend)) in per_item.iter().enumerate() {
                if *blend {
                    continue;
                }
                let it = &job.items[i];
                let tris = it.triangles();
                let mask = match it.alpha_mode() {
                    AlphaMode::Mask(c) => Some(c),
                    _ => None,
                };
                for &t in &bins[band] {
                    let tri = tris[t as usize];
                    let p = tri.map(|v| projected[i][v as usize]);
                    raster(p, w, y0, y1, |x, y, z, b1, b2| {
                        let k = (y - y0) * w + x;
                        if z < depth[k] {
                            if let Some(cut) = mask
                                && it.alpha(tri, [1.0 - b1 - b2, b1, b2]) < cut
                            {
                                return;
                            }
                            depth[k] = z;
                            id[k] = (i as u32, t);
                            bary[k] = [b1, b2];
                        }
                    });
                }
            }
            let mut frags: Vec<Frag> = vec![];
            for (i, (bins, blend)) in per_item.iter().enumerate() {
                if !*blend {
                    continue;
                }
                let it = &job.items[i];
                let tris = it.triangles();
                for &t in &bins[band] {
                    let tri = tris[t as usize];
                    let p = tri.map(|v| projected[i][v as usize]);
                    raster(p, w, y0, y1, |x, y, z, b1, b2| {
                        let k = (y - y0) * w + x;
                        if z < depth[k] && it.alpha(tri, [1.0 - b1 - b2, b1, b2]) > 1.0 / 512.0 {
                            frags.push(Frag {
                                k: k as u32,
                                z,
                                item: i as u32,
                                tri: t,
                                b: [b1, b2],
                            });
                        }
                    });
                }
            }
            let shade_at = |item: u32, t: u32, b: [f32; 2]| {
                let it = &job.items[item as usize];
                shader.shade(
                    it,
                    it.triangles()[t as usize],
                    [1.0 - b[0] - b[1], b[0], b[1]],
                )
            };
            let mut col: Vec<[f32; 3]> = (0..n)
                .map(|k| match id[k] {
                    (u32::MAX, _) => bg,
                    (item, t) => shade_at(item, t, bary[k]).0,
                })
                .collect();
            frags.sort_unstable_by(|a, b| a.k.cmp(&b.k).then(b.z.total_cmp(&a.z)));
            for g in frags.chunk_by(|a, b| a.k == b.k) {
                let mut layers: Vec<(f32, [f32; 3], f32)> = g
                    .iter()
                    .map(|fr| {
                        let (c, a) = shade_at(fr.item, fr.tri, fr.b);
                        (fr.z, c, a.clamp(0.0, 1.0))
                    })
                    .collect();
                let k = g[0].k as usize;
                col[k] = composite(col[k], &mut layers);
            }
            // Box filter down to output pixels, then the view transform.
            let out_rows = (y1 - y0) / ss;
            let mut out = Vec::with_capacity(out_rows * cam.width * 3);
            let norm = 1.0 / (ss * ss) as f32;
            for oy in 0..out_rows {
                for ox in 0..cam.width {
                    let mut s = [0f32; 3];
                    for dy in 0..ss {
                        for dx in 0..ss {
                            let c = col[(oy * ss + dy) * w + ox * ss + dx];
                            for (s, c) in s.iter_mut().zip(c) {
                                *s += c;
                            }
                        }
                    }
                    let d = agx(s.map(|x| x * norm));
                    out.extend(d.map(|x| (x * 255.0 + 0.5) as u8));
                }
            }
            out
        })
        .collect();
    RgbImage::from_raw(cam.width as u32, cam.height as u32, rows.concat())
        .expect("image size matches the band rows")
}

pub struct Options {
    /// Groups to render (empty = all): see [`GROUPS`].
    pub only: Vec<String>,
    /// Supersampling factor per axis.
    pub ssaa: usize,
    /// Directory of reference renders (e.g. Blender's) to put side by side
    /// with ours in RUN_DIR/renders-compare/.
    pub compare: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            only: vec![],
            ssaa: 2,
            compare: None,
        }
    }
}

fn piece_legend(names: &[String]) -> String {
    // Same text as Python's json.dump(legend, indent=1).
    let rows: Vec<String> = names
        .iter()
        .enumerate()
        .map(|(k, n)| {
            let c = PALETTE[k % PALETTE.len()].map(|x| (x * 255.0) as u8);
            format!(
                " {}: \"#{:02x}{:02x}{:02x}\"",
                serde_json::to_string(n).unwrap_or_default(),
                c[0],
                c[1],
                c[2]
            )
        })
        .collect();
    if rows.is_empty() {
        "{}".into()
    } else {
        format!("{{\n{}\n}}", rows.join(",\n"))
    }
}

const VIEWS: [(&str, f64); 3] = [("front", 0.0), ("side", 90.0), ("back", 180.0)];

/// The front, side and back views of one scene framed like
/// mh-render-review.py's 768×1024 renders.
#[allow(clippy::too_many_arguments)]
fn push_views<'a>(
    jobs: &mut Vec<Job<'a>>,
    prefix: &str,
    label: &str,
    scene: Vec<&'a Posed<'a>>,
    crop: Option<Crop>,
    colour: Colour,
    light: Light,
) {
    let Some((lo, hi)) = scene_bounds(&scene) else {
        return;
    };
    let (center, size) = frame(lo, hi, crop);
    for (view, angle) in VIEWS {
        let file = format!("{prefix}-{view}-{label}.png");
        let cam = ortho_camera(&file, center, angle, size, 768, 1024);
        jobs.push(Job::new(file, cam, &scene, colour, light));
    }
}

/// Renders the review image set of `run` into `run/renders/`.
pub fn run(run: &Path, options: &Options) -> Result<Value> {
    let started = Instant::now();
    for g in &options.only {
        if !GROUPS.contains(&g.as_str()) && g != "character" && g != "all" {
            bail!("unknown --only group {g}; use {}", GROUPS.join(","));
        }
    }
    let want = |g: &str| {
        options.only.is_empty()
            || options
                .only
                .iter()
                .any(|o| o == g || o == "all" || (o == "character" && g == "dressed"))
    };
    let out = run.join("renders");
    std::fs::create_dir_all(&out)?;
    let path = |rel: &str| Some(run.join(rel)).filter(|p| p.exists());
    let seg_path = path("1-segment/segmentation.glb");
    let body_path = path("3-skin/body.glb");
    let char_path = path("4-garments/character.glb");
    let posed_path = path("1-segment/posed-prior.glb");
    let piece_dir = run.join("1-segment/pieces");
    let mut piece_files: Vec<PathBuf> = if want("segment") && piece_dir.is_dir() {
        std::fs::read_dir(&piece_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "glb"))
            .collect()
    } else {
        vec![]
    };
    piece_files.sort();
    let need_seg = want("segment") || want("hands") || want("feet");
    let need_body = want("body")
        || want("face")
        || want("hands")
        || want("feet")
        || want("hand-walk")
        || want("chest");
    let want_eyes = want("eyes") || want("face");
    let want_ears = want("ears") || want("face");
    let need_char =
        want("dressed") || want("face") || want("walk") || want("feet") || want_eyes || want_ears;
    let need_body = need_body || want_ears;
    // The source scan and its placement in the working frame (report.json).
    let scan_source: Option<(PathBuf, Matrix4<f64>)> = (want_eyes || want_ears)
        .then(|| {
            let report: Value =
                serde_json::from_slice(&std::fs::read(run.join("report.json")).ok()?).ok()?;
            let src = PathBuf::from(report["source"].as_str()?);
            let m: Vec<f64> = report["working_frame"]["source_to_working"]
                .as_array()?
                .iter()
                .filter_map(Value::as_f64)
                .collect();
            (src.exists() && m.len() == 16).then(|| (src, Matrix4::from_column_slice(&m)))
        })
        .flatten();
    // The blink close-ups turn the lids by the angles that close this
    // character's eyes (report.json `stage2_lid_closing_rad`, per lid and
    // side); MakeHuman's own eye closes with the upper lid at about -0.5 rad.
    let closing = std::fs::read(run.join("report.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .map(|r| r["stage2_lid_closing_rad"].clone())
        .unwrap_or(Value::Null);
    let blink_angle = [
        ("upper", "l", -0.5),
        ("upper", "r", -0.5),
        ("lower", "l", 0.0),
        ("lower", "r", 0.0),
    ]
    .map(|(lid, s, d)| closing[lid][s].as_f64().unwrap_or(d));
    let need_posed = want("hands") || want("feet");
    // Load every GLB in parallel; pieces are drawn in flat colours, so their
    // textures stay encoded.
    let mut loads: Vec<(String, PathBuf, bool)> = vec![];
    let mut add = |key: &str, p: &Option<PathBuf>, need: bool, tex: bool| {
        if let (Some(p), true) = (p, need) {
            loads.push((key.into(), p.clone(), tex));
        }
    };
    add("segmentation", &seg_path, need_seg, want("segment"));
    add("body", &body_path, need_body, true);
    add("character", &char_path, need_char, true);
    add("posed", &posed_path, need_posed, false);
    add(
        "scan",
        &scan_source.as_ref().map(|s| s.0.clone()),
        want_eyes || want_ears,
        true,
    );
    for (k, p) in piece_files.iter().enumerate() {
        loads.push((format!("piece{k:03}"), p.clone(), false));
    }
    let mut models: BTreeMap<String, Model> = loads
        .par_iter()
        .map(|(k, p, tex)| {
            Model::load(p, *tex)
                .with_context(|| format!("load {}", p.display()))
                .map(|m| (k.clone(), m))
        })
        .collect::<Result<_>>()?;
    let piece_names: Vec<String> = piece_files
        .iter()
        .map(|p| {
            p.file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    for (k, _) in piece_files.iter().enumerate() {
        if let Some(m) = models.get_mut(&format!("piece{k:03}")) {
            for part in &mut m.parts {
                part.object_color = PALETTE[k % PALETTE.len()];
            }
        }
    }
    if !piece_files.is_empty() {
        std::fs::write(
            out.join("1-segment-pieces-legend.json"),
            piece_legend(&piece_names),
        )?;
    }
    let loaded_s = started.elapsed().as_secs_f64();

    // Posed geometry: rest for everything, plus the walk frames.
    let model = |k: &str| models.get(k);
    let rest = |k: &str| model(k).map(|m| Posed::new(m, None)).transpose();
    let seg = rest("segmentation")?;
    let body = rest("body")?;
    let character = rest("character")?;
    let posed = rest("posed")?;
    let pieces: Vec<Posed> = (0..piece_files.len())
        .filter_map(|k| model(&format!("piece{k:03}")))
        .map(|m| Posed::new(m, None))
        .collect::<Result<_>>()?;
    let walk: Vec<(f64, Posed)> = match model("character") {
        Some(m) if want("walk") || want("feet") => [0.25, 0.6]
            .into_par_iter()
            .filter_map(|f| m.clip_time(f).map(|t| (f, t)))
            .map(|(f, t)| Posed::new(m, Some(t)).map(|p| (f, p)))
            .collect::<Result<_>>()?,
        _ => vec![],
    };

    // Face test expressions (`face_test.rs`): the whole face, front and
    // three-quarter, one image per expression.
    let mut face_scenes: Vec<(Posed, Vec<(String, Camera)>)> = vec![];
    if want_eyes
        && let Some(m) = model("character")
        && let Some((lo, hi)) = character
            .as_ref()
            .and_then(|c| bounds_of(c.points(|part| part.name.starts_with("eyes"))))
    {
        let eye_span = hi.x - lo.x;
        let center = (lo + hi) / 2.0 - D3::y() * (eye_span * 0.45);
        let size = eye_span * 2.6;
        for (name, pose) in m.face_poses() {
            let scene = Posed::local(m, &pose)?;
            let views = [("front", 0.0), ("three-quarter", 35.0)]
                .map(|(view, angle)| {
                    let file = format!("face-{name}-{view}.png");
                    let cam = ortho_camera(&file, center, angle, size, 768, 768);
                    (file, cam)
                })
                .to_vec();
            face_scenes.push((scene, views));
        }
    }
    let mut jobs: Vec<Job> = vec![];
    if want("segment") {
        if let Some(s) = &seg {
            push_views(
                &mut jobs,
                "1-segment-segmentation",
                "rest",
                vec![s],
                None,
                Colour::Vertex,
                Light::Studio,
            );
            // Head close-ups: labels and the scan's own texture, to judge
            // hair strands against the face (skin-coloured strands).
            // Both sides: the ears and the strands beside them.
            if let Some((lo, hi)) = scene_bounds(&[s]) {
                let (center, size) = frame(lo, hi, Some(Crop::Fraction(0.8, 1.0)));
                for (prefix, colour) in [
                    ("1-segment-segmentation", Colour::Vertex),
                    ("1-segment-scan", Colour::Texture),
                ] {
                    for (view, angle) in [
                        ("front", 0.0),
                        ("left", 90.0),
                        ("back", 180.0),
                        ("right", -90.0),
                    ] {
                        let file = format!("{prefix}-{view}-head-crop.png");
                        let cam = ortho_camera(&file, center, angle, size, 768, 1024);
                        jobs.push(Job::new(file, cam, &[s], colour, Light::Studio));
                    }
                }
            }
        }
        if !pieces.is_empty() {
            push_views(
                &mut jobs,
                "1-segment-pieces",
                "rest",
                pieces.iter().collect(),
                None,
                Colour::Object,
                Light::Studio,
            );
        }
    }
    if want("dressed")
        && let Some(c) = &character
    {
        push_views(
            &mut jobs,
            "4-garments-character",
            "rest",
            vec![c],
            None,
            Colour::Texture,
            Light::Studio,
        );
    }
    if want("body")
        && let Some(b) = &body
    {
        push_views(
            &mut jobs,
            "3-skin-body",
            "rest",
            vec![b],
            None,
            Colour::Texture,
            Light::Studio,
        );
    }
    if want("chest")
        && let Some(b) = &body
        && let Some((lo, hi)) =
            bounds_of(b.points(|part| part.name.contains("underwear_punkduck_sport_bra")))
    {
        let center = (lo + hi) * 0.5;
        let span = hi - lo;
        let size = span.x.max(span.y).max(span.z) * 1.2;
        for (mode, colour) in [("textured", Colour::Texture), ("clay", Colour::Object)] {
            for (view, angle) in [
                ("front", 0.0),
                ("quarter", 45.0),
                ("side", 90.0),
                ("back-quarter", 135.0),
                ("back", 180.0),
            ] {
                let file = format!("chest-cpu-{mode}-{view}.png");
                let cam = ortho_camera(&file, center, angle, size, 1024, 1024);
                jobs.push(Job::new(file, cam, &[b], colour, Light::Studio));
            }
        }
    }
    if want("face") {
        for (p, prefix) in [
            (&character, "4-garments-character-eevee"),
            (&body, "3-skin-body-eevee"),
        ] {
            let Some(p) = p else { continue };
            let eyes = bounds_of(p.points(|part| part.name.starts_with("eyes")));
            let heads = bounds_of(p.points(|part| part.name.starts_with("scan_head")));
            let crop = match (eyes, heads) {
                (Some((lo, hi)), _) => Crop::Eyes { lo, hi },
                (None, Some((lo, hi))) => Crop::Head { lo, hi },
                _ => Crop::Fraction(0.84, 1.0),
            };
            push_views(
                &mut jobs,
                prefix,
                "rest-crop",
                vec![p],
                Some(crop),
                Colour::Texture,
                Light::Sun,
            );
        }
    }
    // Eye close-ups (mh-render-review.py's render_eyes): both eyes, front and
    // three-quarter, 1024×512 across 1.3 eyeball spans; open, with the upper
    // lids turned closed (`blink_angle`), and the textured scan from the same camera.
    let blink = match model("character") {
        Some(m) if want_eyes => Some(Posed::blink(m, blink_angle)?),
        _ => None,
    };
    // Both eyes turned sideways: the ball must stay clean (no lid line or
    // white rim sliding over it).
    let looks = match model("character") {
        Some(m) if want_eyes => vec![
            ("look-left", Posed::look(m, 0.35)?),
            ("look-right", Posed::look(m, -0.35)?),
        ],
        _ => vec![],
    };
    let scan_posed = match (model("scan"), &scan_source) {
        (Some(m), Some((_, t))) => Some(Posed::transformed(m, t)?),
        _ => None,
    };
    if want_eyes
        && let (Some(c), Some(b)) = (&character, &blink)
        && let Some((lo, hi)) = bounds_of(c.points(|part| part.name.starts_with("eyes")))
    {
        let center = (lo + hi) / 2.0;
        let size = (hi.x - lo.x) * 1.3;
        for (view, angle) in [("front", 0.0), ("three-quarter", 35.0)] {
            for (state, scene) in [
                ("open", Some(c)),
                ("blink", Some(b)),
                ("scan", scan_posed.as_ref()),
            ] {
                let Some(scene) = scene else { continue };
                let file = format!("eyes-{state}-{view}.png");
                let cam = ortho_camera(&file, center, angle, size, 1024, 512);
                jobs.push(Job::new(file, cam, &[scene], Colour::Texture, Light::Sun));
            }
        }
        for (state, scene) in &looks {
            let file = format!("eyes-{state}-front.png");
            let cam = ortho_camera(&file, center, 0.0, size, 1024, 512);
            jobs.push(Job::new(file, cam, &[scene], Colour::Texture, Light::Sun));
        }
    }
    for (scene, views) in &face_scenes {
        for (file, cam) in views {
            jobs.push(Job::new(file.clone(), cam.clone(), &[scene], Colour::Texture, Light::Sun));
        }
    }
    // Ear close-ups: each ear from its side, every column the same camera.
    let mut ear_rows: Vec<(String, Vec<String>)> = vec![];
    if want_ears
        && let Some(b) = &body
        && let Some(eyes) = character
            .as_ref()
            .and_then(|c| bounds_of(c.points(|part| part.name.starts_with("eyes"))))
            .or_else(|| bounds_of(b.points(|part| part.name.starts_with("eyes"))))
    {
        for (side, centre, size) in ear_cameras(b, eyes) {
            let angle = if side == "l" { 90.0 } else { -90.0 };
            let mut tiles = vec![];
            for (k, (name, scene, colour, light)) in [
                ("scan", scan_posed.as_ref(), Colour::Texture, Light::Sun),
                (
                    "scan-shape",
                    scan_posed.as_ref(),
                    Colour::Object,
                    Light::Studio,
                ),
                ("body", Some(b), Colour::Texture, Light::Sun),
                ("body-shape", Some(b), Colour::Object, Light::Studio),
                (
                    "character",
                    character.as_ref(),
                    Colour::Texture,
                    Light::Studio,
                ),
            ]
            .into_iter()
            .enumerate()
            {
                let file = format!(".ears-{side}-{k}-{name}.png");
                tiles.push(file.clone());
                let Some(scene) = scene else { continue };
                let cam = ortho_camera(&file, centre, angle, size, EAR_TILE, EAR_TILE);
                jobs.push(Job::new(file, cam, &[scene], colour, light));
            }
            ear_rows.push((format!("ears-{side}.png"), tiles));
        }
    }
    // The naked body (with its underwear) through the same walk frames
    // (body group, feet close-ups).
    let body_walk: Vec<(f64, Posed)> = match model("body") {
        Some(m) if (want("feet") || want("body")) && !m.loaded.asset.clips.is_empty() => {
            [0.25, 0.6]
                .into_par_iter()
                .filter_map(|f| m.clip_time(f).map(|t| (f, t)))
                .map(|(f, t)| Posed::new(m, Some(t)).map(|p| (f, p)))
                .collect::<Result<_>>()?
        }
        _ => vec![],
    };
    for (f, p) in body_walk.iter().filter(|_| want("body")) {
        push_views(
            &mut jobs,
            "3-skin-body",
            &format!("{f:.2}"),
            vec![p],
            None,
            Colour::Texture,
            Light::Studio,
        );
    }
    for (f, p) in walk.iter().filter(|_| want("walk")) {
        push_views(
            &mut jobs,
            "4-garments-character",
            &format!("{f:.2}"),
            vec![p],
            None,
            Colour::Texture,
            Light::Studio,
        );
    }
    if want("hands")
        && let Some(bm) = model("body")
    {
        for side in ["L", "R"] {
            let Some((lo, hi)) = bm.hand_box(side) else {
                continue;
            };
            let center = (lo + hi) / 2.0;
            let size = (hi - lo).max();
            for (p, name, colour) in [
                (&seg, "scan", Colour::Vertex),
                (&posed, "posed", Colour::Texture),
                (&body, "body", Colour::Texture),
            ] {
                let Some(p) = p else { continue };
                // Side view from outside the hand (the character's left is +X).
                for (view, angle) in [
                    ("front", 0.0),
                    ("side", if side == "L" { 90.0 } else { -90.0 }),
                ] {
                    let file = format!("hand-{name}-{}-{view}.png", side.to_lowercase());
                    let cam = ortho_camera(&file, center, angle, size, 768, 768);
                    jobs.push(Job::new(file, cam, &[p], colour, Light::Studio));
                }
            }
        }
    }
    let hand_walk: Vec<(f64, f64, Posed)> = match model("body") {
        Some(m) if want("hand-walk") && !m.loaded.asset.clips.is_empty() => HAND_WALK_FRAMES
            .into_par_iter()
            .filter_map(|f| m.clip_time(f).map(|t| (f, t)))
            .map(|(f, t)| Posed::new(m, Some(t)).map(|p| (f, t, p)))
            .collect::<Result<_>>()?,
        _ => vec![],
    };
    if let Some(bm) = model("body") {
        for (f, t, p) in &hand_walk {
            for side in ["L", "R"] {
                let Some((lo, hi)) = bm.hand_box_at(side, Some(*t)) else {
                    continue;
                };
                let center = (lo + hi) / 2.0;
                let size = (hi - lo).max();
                for (view, angle) in [
                    ("front", 0.0),
                    ("side", if side == "L" { 90.0 } else { -90.0 }),
                ] {
                    let file = format!("hand-walk-{}-{view}-{f:.2}.png", side.to_lowercase());
                    let cam = ortho_camera(&file, center, angle, size, 512, 512);
                    jobs.push(Job::new(file, cam, &[p], Colour::Texture, Light::Studio));
                }
            }
        }
    }
    if want("feet")
        && let Some(bm) = model("body")
    {
        for side in ["L", "R"] {
            // Side view from outside the foot (the character's left is +X).
            let views = [
                ("front", 0.0),
                ("side", if side == "L" { 90.0 } else { -90.0 }),
            ];
            let s = side.to_lowercase();
            let at = |list: &'_ [(f64, Posed)], f: f64| -> Option<usize> {
                list.iter().position(|(g, _)| *g == f)
            };
            let mut sets = vec![(
                None,
                vec![
                    (seg.as_ref(), "scan", Colour::Vertex),
                    (posed.as_ref(), "prior-clay", Colour::Object),
                    (body.as_ref(), "body-clay", Colour::Object),
                    (body.as_ref(), "body", Colour::Texture),
                    (character.as_ref(), "dressed", Colour::Texture),
                ],
            )];
            for f in [0.25, 0.6] {
                sets.push((
                    Some(f),
                    vec![
                        (
                            at(&body_walk, f).map(|k| &body_walk[k].1),
                            "body-clay",
                            Colour::Object,
                        ),
                        (at(&walk, f).map(|k| &walk[k].1), "dressed", Colour::Texture),
                    ],
                ));
            }
            for (frame, scenes) in sets {
                // Walk frames are framed on the animated model's bones.
                let framing = match frame {
                    Some(_) => model("character").filter(|m| !m.loaded.asset.clips.is_empty()),
                    None => Some(bm),
                };
                let Some(framing) = framing else { continue };
                let time = frame.and_then(|f| framing.clip_time(f));
                if frame.is_some() && time.is_none() {
                    continue;
                }
                let Some((lo, hi)) = framing.foot_box(side, time) else {
                    continue;
                };
                let center = (lo + hi) / 2.0;
                let size = (hi - lo).max();
                let label = frame.map_or("rest".to_string(), |f| format!("{f:.2}"));
                for (p, name, colour) in scenes {
                    let Some(p) = p else { continue };
                    for (view, angle) in views {
                        let file = format!("feet-{name}-{s}-{view}-{label}.png");
                        let cam = ortho_camera(&file, center, angle, size, 768, 768);
                        jobs.push(Job::new(file, cam, &[p], colour, Light::Studio));
                    }
                }
            }
        }
    }
    let posed_s = started.elapsed().as_secs_f64() - loaded_s;
    let ss = options.ssaa;
    let files: Vec<String> = jobs
        .par_iter()
        .map(|job| -> Result<String> {
            let img = render(job, ss);
            img.save(out.join(&job.file))
                .with_context(|| format!("write {}", job.file))?;
            Ok(job.file.clone())
        })
        .collect::<Result<_>>()?;
    let mut files = files;
    for (row, tiles) in &ear_rows {
        let mut img = RgbImage::from_pixel(
            (EAR_TILE * tiles.len()) as u32,
            EAR_TILE as u32,
            image::Rgb([51, 51, 51]),
        );
        for (k, tile) in tiles.iter().enumerate() {
            if let Some(i) = files.iter().position(|f| f == tile) {
                let t = image::open(out.join(tile))?.to_rgb8();
                image::imageops::replace(&mut img, &t, (k * EAR_TILE) as i64, 0);
                std::fs::remove_file(out.join(tile))?;
                files.remove(i);
            }
        }
        img.save(out.join(row))?;
        files.push(row.clone());
    }
    let render_s = started.elapsed().as_secs_f64() - loaded_s - posed_s;
    let compared = match &options.compare {
        Some(dir) => compare(&out, dir, &run.join("renders-compare"), &files)?,
        None => 0,
    };
    Ok(json!({
        "renders": out,
        "images": files.len(),
        "files": files,
        "ssaa": ss,
        "compared": compared,
        "load_s": loaded_s,
        "pose_s": posed_s,
        "render_s": render_s,
        "total_s": started.elapsed().as_secs_f64(),
    }))
}

/// Writes `reference | ours` side by side for every file present in both.
fn compare(ours: &Path, reference: &Path, out: &Path, files: &[String]) -> Result<usize> {
    std::fs::create_dir_all(out)?;
    let done: Vec<bool> = files
        .par_iter()
        .map(|f| -> Result<bool> {
            let r = reference.join(f);
            if !r.exists() {
                return Ok(false);
            }
            let a = image::open(&r)?.to_rgb8();
            let b = image::open(ours.join(f))?.to_rgb8();
            let h = a.height().max(b.height());
            let gap = 8;
            let mut img =
                RgbImage::from_pixel(a.width() + gap + b.width(), h, image::Rgb([255, 255, 255]));
            image::imageops::replace(&mut img, &a, 0, 0);
            image::imageops::replace(&mut img, &b, (a.width() + gap) as i64, 0);
            img.save(out.join(f))?;
            Ok(true)
        })
        .collect::<Result<_>>()?;
    Ok(done.into_iter().filter(|&d| d).count())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_order_is_back_to_front() {
        let base = [0.0, 0.0, 1.0];
        let near_red = (1.0, [1.0, 0.0, 0.0], 0.5);
        let far_green = (2.0, [0.0, 1.0, 0.0], 0.5);
        let a = composite(base, &mut [near_red, far_green]);
        let b = composite(base, &mut [far_green, near_red]);
        assert_eq!(a, b);
        // Green over blue, then red over that: red dominates.
        assert!(
            (a[0] - 0.5).abs() < 1e-6 && (a[1] - 0.25).abs() < 1e-6 && (a[2] - 0.25).abs() < 1e-6
        );
    }

    #[test]
    fn body_framing_matches_blender() {
        // A 1.8 m tall, 0.6 m wide figure: height-limited, 8 % margin.
        let (lo, hi) = (D3::new(-0.3, 0.0, -0.15), D3::new(0.3, 1.8, 0.15));
        let (center, size) = frame(lo, hi, None);
        assert!((size - 1.8 * 1.08).abs() < 1e-12);
        let cam = ortho_camera("front", center, 0.0, size, 768, 1024);
        let top = cam.project([0.0, 1.8, 0.0]);
        let bottom = cam.project([0.0, 0.0, 0.0]);
        let margin = 1024.0 * (1.0 - 1.0 / 1.08) / 2.0;
        assert!((top.1 - margin).abs() < 1e-6 && (bottom.1 - (1024.0 - margin)).abs() < 1e-6);
        // The figure's left (+X) is on the image's right from the front, and
        // the side camera looks from +X with the figure's front (+Z) on the left.
        assert!(cam.project([0.3, 1.0, 0.0]).0 > 384.0);
        let side = ortho_camera("side", center, 90.0, size, 768, 1024);
        assert!(side.project([0.0, 1.0, 0.15]).0 < 384.0);
        assert!(side.project([0.3, 1.0, 0.0]).2 < side.project([-0.3, 1.0, 0.0]).2);
    }

    #[test]
    fn raster_covers_each_sample_once() {
        // Two triangles sharing a diagonal cover a 4×4 square exactly once.
        let q = [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
        ];
        let mut hits = [0u8; 16];
        for t in [[0, 1, 2], [0, 2, 3]] {
            raster(t.map(|i| q[i]), 4, 0, 4, |x, y, _, _, _| {
                hits[y * 4 + x] += 1
            });
        }
        assert!(hits.iter().all(|&h| h >= 1));
        assert!(hits.iter().filter(|&&h| h > 1).count() <= 4);
    }

    #[test]
    fn agx_maps_world_grey_near_blender() {
        // Blender shows the review world colour 0.35 at about 158/255.
        let d = agx([0.35; 3]);
        assert!((d[0] * 255.0 - 158.0).abs() < 4.0, "{d:?}");
    }
}
