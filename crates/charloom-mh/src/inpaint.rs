 use crate::source::{lab, linear_to_srgb, srgb_to_linear};
use crate::texture::{box_blur, dilate, pull_push};
use anyhow::{Context, Result, bail, ensure};
use charloom_io::GlbDocument;
use image::imageops::FilterType;
use image::{GrayImage, Luma, Rgba, RgbaImage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// Side of every exported patch image (the usual image-generator size).
pub const PATCH_SIDE: u32 = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Eye zone filled from the skin around it (the bake's smudge).
    EyeSurround,
    /// Skin the scan never showed (under clothes, hair or shoes): MakeHuman
    /// texture toned to the scan.
    Hidden,
         Outlier,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    /// Square crop side in texture texels.
    pub side: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Patch {
    pub id: String,
    pub kind: Kind,
    /// Dominant MakeHuman joints under the masked area.
    pub region: Vec<String>,
    /// Fraction of the masked area on body triangles that stay in the
    /// dressed `character.glb` (0: only seen on the naked `body.glb`).
    pub visible_dressed: f64,
    /// Crop of the base-colour texture, in texels (top-left origin).
    pub crop: Rect,
    /// Quarter turns clockwise from the crop to the patch image, so the
    /// body's up points up in the patch (the face island lies sideways).
    pub rotation_cw: u32,
    /// Patch image side in pixels (crop resampled to it).
    pub patch_side: u32,
    pub image: String,
    pub mask: String,
    pub context: String,
         pub edited: String,
    /// Texels that may change (mask > 0.5).
    pub masked_texels: usize,
    /// Median L*a*b* of observed scan skin around the patch.
    pub reference_lab: [f64; 3],
    /// Median L*a*b* of the current (synthesized) fill under the mask.
    pub current_lab: [f64; 3],
    pub prompt: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tolerances {
    /// Mean L*a*b* shift of the keep area (global tone drift of the edit).
    pub keep_drift_delta_e: f64,
    /// Per-pixel ΔE outside the mask after removing that drift.
    pub keep_delta_e_p50: f64,
    pub keep_delta_e_p90: f64,
    /// Median of the blended masked area vs `reference_lab` (or, where the
    /// bake's fill is already further off, that distance + 2 while staying
    /// within this of the fill).
    pub tone_delta_e: f64,
    /// 95th percentile of masked pixels vs `reference_lab` (dark marks,
    /// painted eyes or hair).
    pub tone_delta_e_p95: f64,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            keep_drift_delta_e: 12.0,
            keep_delta_e_p50: 6.0,
            keep_delta_e_p90: 14.0,
            tone_delta_e: 10.0,
            tone_delta_e_p95: 28.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub run: PathBuf,
         pub texture: String,
    pub provenance: String,
    pub texture_size: u32,
    pub instructions: String,
    pub tolerances: Tolerances,
    pub patches: Vec<Patch>,
}

const INSTRUCTIONS: &str = "Each patch-NN.png is a square crop of a game character's skin texture in UV-atlas layout (flattened, not a photo). patch-NN.mask.png is white where the image may be repainted and black where it must stay as it is; patch-NN.context.png shows where the patch sits on the body (red) and in the atlas (yellow box). Edit patch-NN.png so the white area becomes plausible, evenly lit human skin that matches the colour, tone and pore detail of the surrounding (black-mask) skin, keep the framing, scale and everything under the black mask unchanged (no shift, crop, border, text or watermark), and save it as patch-NN.edited.png at exactly the same size. Never add eyes, eyelashes, hair, clothing, tattoos, shadows or lighting; the 3D eyeballs, hair and clothes are separate meshes.";

const INSTRUCTIONS_MD: &str = include_str!("inpaint_instructions.md");

fn prompt_for(kind: Kind, region: &[String]) -> String {
    let specific = match kind {
        Kind::EyeSurround => {
            "The white area surrounds an eye opening on the unwrapped face. The eyeball is a separate 3D mesh behind it, so paint only eyelid and eye-socket skin: soft lid skin with a faint crease, a little pinker or darker than the cheek next to it. No eyeball, iris, sclera, pupil, eyelashes, eyeliner or make-up."
        }
        Kind::Hidden => {
            "The white area is skin the scan never showed (under clothes, hair or shoes); it currently holds a generic body texture toned to this character. Keep its anatomy marks where they are (navel, nipples, creases, knuckles, scalp stubble) and add realistic skin detail and subtle colour variation in the character's tone; do not add new features, body hair, underwear or tan lines."
        }
        Kind::Outlier => {
            "The white area held scan colours rejected as not skin (painted hair strands, cloth shadows or dirt). Paint plain skin continuing the surrounding skin."
        }
    };
    format!(
        "Fill the white area with plausible skin matching the surrounding skin tone and pore detail; keep black areas pixel-identical; no eyes, no hair, no shadows. Body region: {}. {specific}",
        region.join(", ")
    )
}

/// The bake's provenance decoded per texel.
struct Provenance {
    n: usize,
    /// In the UV islands.
    covered: Vec<bool>,
    /// 1 = synthesized, 0 = observed scan colour.
    synth: Vec<f32>,
    kind: Vec<Option<Kind>>,
}

impl Provenance {
    fn new(img: &RgbaImage) -> Self {
        let n = img.width() as usize;
        let mut covered = vec![false; n * n];
        let mut synth = vec![0f32; n * n];
        let mut kind = vec![None; n * n];
        for (k, p) in img.pixels().enumerate() {
            if p[3] < 128 {
                continue;
            }
            covered[k] = true;
            synth[k] = 1.0 - p[0] as f32 / 255.0;
            if synth[k] > 0.5 {
                kind[k] = Some(if p[1] >= 8 {
                    Kind::EyeSurround
                } else if p[2] >= 128 {
                    Kind::Outlier
                } else {
                    Kind::Hidden
                });
            }
        }
        Self {
            n,
            covered,
            synth,
            kind,
        }
    }
}

/// The body mesh of `3-skin/body.glb` for context renders and region names.
struct BodyMesh {
    positions: Vec<[f64; 3]>,
    uvs: Vec<[f32; 2]>,
    triangles: Vec<[u32; 3]>,
    joint: Vec<String>,
}

fn load_body(path: &Path) -> Result<BodyMesh> {
    let loaded = charloom_io::load_gltf(path)?;
    let world = loaded.asset.world_meshes()?;
    let prims = crate::source::read_primitives(&loaded, &world)?;
    let mi = loaded
        .asset
        .meshes
        .iter()
        .position(|m| m.name == "body")
        .context("no mesh named body in body.glb")?;
    let mesh = &loaded.asset.meshes[mi];
    let joint = mesh
        .influences
        .iter()
        .map(|inf| {
            inf.iter()
                .max_by(|a, b| a.weight.total_cmp(&b.weight))
                .map_or("-".into(), |i| {
                    loaded.asset.skeleton.joints[i.joint].name.clone()
                })
        })
        .collect();
    Ok(BodyMesh {
        positions: world[mi].clone(),
        uvs: prims[mi].uvs.clone(),
        triangles: mesh.triangles.clone(),
        joint,
    })
}

fn to_lin(p: &Rgba<u8>) -> [f32; 3] {
    [0, 1, 2].map(|j| srgb_to_linear(p[j] as f64 / 255.0) as f32)
}

fn lab8(p: &Rgba<u8>) -> [f64; 3] {
    lab([0, 1, 2].map(|j| p[j] as f64 / 255.0))
}

/// The bake's ΔE (lightness half-weighted, so shading differences count
/// less than hue).
fn delta_e(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) * 0.5 + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn median3(mut v: Vec<[f64; 3]>) -> [f64; 3] {
    std::array::from_fn(|j| {
        v.sort_by(|a, b| a[j].total_cmp(&b[j]));
        v.get(v.len() / 2).map_or(0.0, |c| c[j])
    })
}

fn percentile(mut v: Vec<f64>, q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// Feathered whole-texture mask of one kind: the core (kind texels in kept
/// components) eroded by `2r`, then blurred twice by `r`, so the ramp lies
/// inside the synthesized area and observed texels never change.
fn feathered(prov: &Provenance, core: &[bool], r: usize) -> Vec<f32> {
    let n = prov.n;
    let c: Vec<f32> = core.iter().map(|&b| b as u8 as f32).collect();
    let eroded: Vec<f32> = box_blur(&c, n, n, 2 * r)
        .iter()
        .map(|&v| (v > 0.999) as u8 as f32)
        .collect();
    let m = box_blur(&box_blur(&eroded, n, n, r), n, n, r);
    m.iter()
        .zip(&prov.synth)
        .zip(&prov.covered)
        .map(|((m, s), c)| if *c { m * s } else { 0.0 })
        .collect()
}

/// Connected components (4-neighbour) of one kind on a coarse grid; returns
/// per component its texel count, cell bbox and cell list.
fn components(prov: &Provenance, kind: Kind, cell: usize) -> Vec<(usize, [usize; 4], Vec<usize>)> {
    let n = prov.n;
    let g = n.div_ceil(cell);
    let mut count = vec![0usize; g * g];
    for k in 0..n * n {
        if prov.kind[k] == Some(kind) {
            count[(k / n / cell) * g + (k % n) / cell] += 1;
        }
    }
    let occupied: Vec<bool> = count.iter().map(|&c| c * 4 >= cell * cell).collect();
    let mut seen = vec![false; g * g];
    let mut out = vec![];
    for start in 0..g * g {
        if !occupied[start] || seen[start] {
            continue;
        }
        let mut stack = vec![start];
        seen[start] = true;
        let mut cells = vec![];
        let (mut total, mut bb) = (0, [usize::MAX, usize::MAX, 0, 0]);
        while let Some(c) = stack.pop() {
            let (cx, cy) = (c % g, c / g);
            cells.push(c);
            total += count[c];
            bb = [bb[0].min(cx), bb[1].min(cy), bb[2].max(cx), bb[3].max(cy)];
            let mut push = |x: usize, y: usize| {
                let d = y * g + x;
                if occupied[d] && !seen[d] {
                    seen[d] = true;
                    stack.push(d);
                }
            };
            if cx > 0 {
                push(cx - 1, cy);
            }
            if cx + 1 < g {
                push(cx + 1, cy);
            }
            if cy > 0 {
                push(cx, cy - 1);
            }
            if cy + 1 < g {
                push(cx, cy + 1);
            }
        }
        out.push((total, bb, cells));
    }
    out
}

/// Square crops covering a texel bbox: one square if it fits `max`, else
/// overlapping tiles of side `max`.
fn crops(bb: [usize; 4], n: usize, min: usize, max: usize) -> Vec<Rect> {
    let (w, h) = (bb[2] - bb[0], bb[3] - bb[1]);
    let clamp = |c: usize, side: usize| c.min(n - side) as u32;
    if w.max(h) <= max {
        let side = w.max(h).max(min).min(n);
        let cx = (bb[0] + bb[2]) / 2;
        let cy = (bb[1] + bb[3]) / 2;
        return vec![Rect {
            x: clamp(cx.saturating_sub(side / 2), side),
            y: clamp(cy.saturating_sub(side / 2), side),
            side: side as u32,
        }];
    }
    let overlap = max / 8;
    let tiles = |len: usize| (len.saturating_sub(overlap)).div_ceil(max - overlap).max(1);
    let (tx, ty) = (tiles(w), tiles(h));
    let step = |len: usize, t: usize| {
        if t > 1 {
            (len - max) as f64 / (t - 1) as f64
        } else {
            0.0
        }
    };
    let (sx, sy) = (step(w.max(max), tx), step(h.max(max), ty));
    let mut out = vec![];
    for j in 0..ty {
        for i in 0..tx {
            let x =
                bb[0] as f64 + sx * i as f64 - if w < max { (max - w) as f64 / 2.0 } else { 0.0 };
            let y =
                bb[1] as f64 + sy * j as f64 - if h < max { (max - h) as f64 / 2.0 } else { 0.0 };
            out.push(Rect {
                x: clamp(x.max(0.0).round() as usize, max),
                y: clamp(y.max(0.0).round() as usize, max),
                side: max as u32,
            });
        }
    }
    out
}

/// Ramp from 0 at the crop border to 1 `ramp` texels inside, so a tile's
/// mask never ends in a hard edge.
fn border_ramp(rect: Rect, x: u32, y: u32, ramp: f32) -> f32 {
    let d = (x - rect.x)
        .min(y - rect.y)
        .min(rect.x + rect.side - 1 - x)
        .min(rect.y + rect.side - 1 - y) as f32;
    let t = (d / ramp).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn crop_mask(mask: &[f32], n: usize, rect: Rect, ramp: f32) -> GrayImage {
    GrayImage::from_fn(rect.side, rect.side, |i, j| {
        let (x, y) = (rect.x + i, rect.y + j);
        let m = mask[y as usize * n + x as usize] * border_ramp(rect, x, y, ramp);
        Luma([(m.clamp(0.0, 1.0) * 255.0).round() as u8])
    })
}

fn resize_rgba(img: &RgbaImage, side: u32) -> RgbaImage {
    if img.width() == side {
        img.clone()
    } else {
        image::imageops::resize(img, side, side, FilterType::CatmullRom)
    }
}

fn resize_gray(img: &GrayImage, side: u32) -> GrayImage {
    if img.width() == side {
        img.clone()
    } else {
        image::imageops::resize(img, side, side, FilterType::Triangle)
    }
}

/// Median L*a*b* of observed texels around a crop: the crop first, grown
/// until at least 2000 observed texels are inside.
fn reference_lab(tex: &RgbaImage, prov: &Provenance, rect: Rect) -> [f64; 3] {
    let n = prov.n as i64;
    let mut grow = 0i64;
    loop {
        let (x0, y0) = ((rect.x as i64 - grow).max(0), (rect.y as i64 - grow).max(0));
        let (x1, y1) = (
            (rect.x as i64 + rect.side as i64 + grow).min(n),
            (rect.y as i64 + rect.side as i64 + grow).min(n),
        );
        let mut v = vec![];
        for y in (y0..y1).step_by(2) {
            for x in (x0..x1).step_by(2) {
                let k = (y * n + x) as usize;
                if prov.covered[k] && prov.synth[k] < 0.1 {
                    v.push(lab8(tex.get_pixel(x as u32, y as u32)));
                }
            }
        }
        if v.len() >= 500 || (x0 == 0 && y0 == 0 && x1 == n && y1 == n) {
            return median3(v);
        }
        grow += rect.side as i64 / 2;
    }
}

/// Front view, back view and atlas, 512 px each: the patch's masked area
/// in red on the body, the crop as a yellow box on the atlas.
fn context_image(
    tex: &RgbaImage,
    body: &BodyMesh,
    mask: &[f32],
    n: usize,
    rect: Rect,
) -> RgbaImage {
    const S: u32 = 512;
    let mut out = RgbaImage::from_pixel(S * 3, S, Rgba([40, 40, 44, 255]));
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in &body.positions {
        for j in 0..2 {
            lo[j] = lo[j].min(p[j]);
            hi[j] = hi[j].max(p[j]);
        }
    }
    let scale = (S as f64 - 24.0) / (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-6);
    let mid = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0];
    let at_uv = |uv: [f64; 2]| -> usize {
        let x = (uv[0] * n as f64).clamp(0.0, n as f64 - 1.0) as usize;
        let y = (uv[1] * n as f64).clamp(0.0, n as f64 - 1.0) as usize;
        y * n + x
    };
    for (panel, side) in [(0u32, 1.0f64), (1, -1.0)] {
        let mut depth = vec![f64::NEG_INFINITY; (S * S) as usize];
        for t in &body.triangles {
            let p = t.map(|i| body.positions[i as usize]);
            let q = p.map(|p| {
                [
                    S as f64 / 2.0 + side * (p[0] - mid[0]) * scale,
                    S as f64 / 2.0 - (p[1] - mid[1]) * scale,
                    side * p[2],
                ]
            });
            let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
            let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
            let nrm = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let len = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
            if len < 1e-12 {
                continue;
            }
            let shade = 0.35 + 0.65 * (nrm[2] / len).abs();
            let uv = [0, 1].map(|j| {
                t.iter()
                    .map(|&i| body.uvs[i as usize][j] as f64)
                    .sum::<f64>()
                    / 3.0
            });
            let k = at_uv(uv);
            let c = tex.get_pixel((k % n) as u32, (k / n) as u32);
            let red = (mask[k] * 1.5).min(1.0) as f64 * 0.7;
            let col: [u8; 3] = std::array::from_fn(|j| {
                let base = c[j] as f64 * shade;
                let target = if j == 0 { 255.0 } else { 30.0 };
                (base * (1.0 - red) + target * red).clamp(0.0, 255.0) as u8
            });
            let area = (q[1][0] - q[0][0]) * (q[2][1] - q[0][1])
                - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1]);
            if area.abs() < 1e-9 {
                continue;
            }
            let x0 = q
                .iter()
                .map(|v| v[0])
                .fold(f64::INFINITY, f64::min)
                .floor()
                .max(0.0) as u32;
            let x1 = (q
                .iter()
                .map(|v| v[0])
                .fold(f64::NEG_INFINITY, f64::max)
                .ceil() as u32)
                .min(S - 1);
            let y0 = q
                .iter()
                .map(|v| v[1])
                .fold(f64::INFINITY, f64::min)
                .floor()
                .max(0.0) as u32;
            let y1 = (q
                .iter()
                .map(|v| v[1])
                .fold(f64::NEG_INFINITY, f64::max)
                .ceil() as u32)
                .min(S - 1);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                    let l1 =
                        ((q[2][0] - px) * (q[0][1] - py) - (q[0][0] - px) * (q[2][1] - py)) / area;
                    let l2 =
                        ((q[0][0] - px) * (q[1][1] - py) - (q[1][0] - px) * (q[0][1] - py)) / area;
                    let l0 = 1.0 - l1 - l2;
                    if l0 < 0.0 || l1 < 0.0 || l2 < 0.0 {
                        continue;
                    }
                    let z = l0 * q[0][2] + l1 * q[1][2] + l2 * q[2][2];
                    let d = &mut depth[(y * S + x) as usize];
                    if z > *d {
                        *d = z;
                        out.put_pixel(panel * S + x, y, Rgba([col[0], col[1], col[2], 255]));
                    }
                }
            }
        }
    }
    let atlas = resize_rgba(tex, S);
    let f = n as f64 / S as f64;
    for y in 0..S {
        for x in 0..S {
            let k = ((y as f64 + 0.5) * f) as usize * n + ((x as f64 + 0.5) * f) as usize;
            let p = atlas.get_pixel(x, y);
            let red = (mask[k] * 1.5).min(1.0) * 0.6;
            let c: [u8; 3] = std::array::from_fn(|j| {
                let t = if j == 0 { 255.0 } else { 30.0 };
                (p[j] as f32 * (1.0 - red) + t * red) as u8
            });
            out.put_pixel(2 * S + x, y, Rgba([c[0], c[1], c[2], 255]));
        }
    }
    let (rx0, ry0) = ((rect.x as f64 / f) as u32, (rect.y as f64 / f) as u32);
    let rx1 = (((rect.x + rect.side) as f64 / f) as u32).min(S - 1);
    let ry1 = (((rect.y + rect.side) as f64 / f) as u32).min(S - 1);
    for t in 0..2 {
        for x in rx0..=rx1 {
            for y in [ry0 + t, ry1.saturating_sub(t)] {
                out.put_pixel(2 * S + x, y, Rgba([255, 220, 0, 255]));
            }
        }
        for y in ry0..=ry1 {
            for x in [rx0 + t, rx1.saturating_sub(t)] {
                out.put_pixel(2 * S + x, y, Rgba([255, 220, 0, 255]));
            }
        }
    }
    out
}

/// The body triangles kept in the dressed character (fully covered ones
/// are removed there).
fn dressed_triangles(path: &Path) -> Option<std::collections::HashSet<[u32; 3]>> {
    let loaded = charloom_io::load_gltf(path).ok()?;
    let mesh = loaded.asset.meshes.iter().find(|m| m.name == "body")?;
    Some(mesh.triangles.iter().copied().collect())
}

struct MaskInfo {
    region: Vec<String>,
    visible_dressed: f64,
    rotation_cw: u32,
}

/// Region names, dressed visibility and the upright rotation of a patch
/// mask, from the body triangles under it.
fn mask_info(
    body: &BodyMesh,
    mask: &[f32],
    n: usize,
    dressed: Option<&std::collections::HashSet<[u32; 3]>>,
) -> MaskInfo {
    let mut joints: Vec<(String, f64)> = vec![];
    let (mut total, mut kept) = (0.0, 0.0);
    let mut up = [0.0f64; 2];
    for t in &body.triangles {
        let uv = t.map(|i| body.uvs[i as usize].map(|c| c as f64 * n as f64));
        let c = [0, 1].map(|j| uv.iter().map(|p| p[j]).sum::<f64>() / 3.0);
        let k =
            c[1].clamp(0.0, n as f64 - 1.0) as usize * n + c[0].clamp(0.0, n as f64 - 1.0) as usize;
        let m = mask[k] as f64;
        if m <= 0.25 {
            continue;
        }
        let name = &body.joint[t[0] as usize];
        match joints.iter_mut().find(|(j, _)| j == name) {
            Some(e) => e.1 += m,
            None => joints.push((name.clone(), m)),
        }
        total += m;
        if dressed.is_none_or(|d| d.contains(t)) {
            kept += m;
        }
        // Texture-space gradient of world height: where "up" points.
        let p = t.map(|i| body.positions[i as usize][1]);
        let (a, b) = (
            [uv[1][0] - uv[0][0], uv[1][1] - uv[0][1]],
            [uv[2][0] - uv[0][0], uv[2][1] - uv[0][1]],
        );
        let det = a[0] * b[1] - a[1] * b[0];
        if det.abs() > 1e-9 {
            let (dy1, dy2) = (p[1] - p[0], p[2] - p[0]);
            let gx = (dy1 * b[1] - dy2 * a[1]) / det;
            let gy = (dy2 * a[0] - dy1 * b[0]) / det;
            let len = (gx * gx + gy * gy).sqrt();
            if len > 0.0 {
                up[0] += m * gx / len;
                up[1] += m * gy / len;
            }
        }
    }
    joints.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    // Clockwise quarter turn maps an image vector (x, y) to (-y, x); pick
    // the turn that brings "up" closest to (0, -1).
    let turn = |k: u32| (0..k).fold(up, |v, _| [-v[1], v[0]]);
    let rotation_cw = (0..4u32)
        .min_by(|&i, &j| turn(i)[1].total_cmp(&turn(j)[1]).then(i.cmp(&j)))
        .unwrap_or(0);
    MaskInfo {
        region: joints.into_iter().take(4).map(|j| j.0).collect(),
        visible_dressed: if total > 0.0 { kept / total } else { 0.0 },
        rotation_cw,
    }
}

fn rotate_cw<P: image::Pixel + 'static>(
    img: &image::ImageBuffer<P, Vec<P::Subpixel>>,
    quarter_turns: u32,
) -> image::ImageBuffer<P, Vec<P::Subpixel>> {
    match quarter_turns % 4 {
        1 => image::imageops::rotate90(img),
        2 => image::imageops::rotate180(img),
        3 => image::imageops::rotate270(img),
        _ => img.clone(),
    }
}

fn skin_dir(run: &Path) -> PathBuf {
    run.join("3-skin")
}

/// `charloom mh-texture-export RUN_DIR`: writes `RUN_DIR/inpaint/` (or
/// `out`) with patches, masks, context images and `patches.json`.
pub fn export(run: &Path, out: Option<&Path>) -> Result<Value> {
    let out = out.map_or_else(|| run.join("inpaint"), Path::to_path_buf);
    std::fs::create_dir_all(&out)?;
    let skin = skin_dir(run);
    let prov_path = skin.join("body-provenance.png");
    ensure!(
        prov_path.exists(),
        "{} is missing: re-run mh-transfer with a version that writes the bake provenance",
        prov_path.display()
    );
     
    let base_path = out.join("base.png");
    let tex = if base_path.exists() {
        image::open(&base_path)?.to_rgba8()
    } else {
        let t = image::open(skin.join("body-basecolor.png"))?.to_rgba8();
        t.save(&base_path)?;
        t
    };
    let prov_img = image::open(&prov_path)?.to_rgba8();
    ensure!(
        tex.dimensions() == prov_img.dimensions() && tex.width() == tex.height(),
        "texture and provenance sizes differ"
    );
    prov_img.save(out.join("provenance.png"))?;
    let prov = Provenance::new(&prov_img);
    let n = prov.n;
    let body = load_body(&skin.join("body.glb"))?;
    let covered = prov.covered.iter().filter(|c| **c).count();
    let cell = (n / 256).max(1);
    let (min_side, max_side) = (n / 4, n / 2);
    let r = (n / 512).max(1);
    let ramp = (n / 64) as f32;
    let dressed = dressed_triangles(&run.join("4-garments").join("character.glb"));
    let g = n.div_ceil(cell);
    let mut patches = vec![];
    let (mut small, mut small_texels, mut sockets) = (0usize, 0usize, 0usize);
    for kind in [Kind::EyeSurround, Kind::Hidden, Kind::Outlier] {
        let min_area = match kind {
            Kind::EyeSurround => 64,
            Kind::Hidden => covered / 200,
            Kind::Outlier => covered / 500,
        };
        let comps = components(&prov, kind, cell);
        let mut comp_of_cell = vec![usize::MAX; g * g];
        let mut core = vec![false; n * n];
        let mut rects: Vec<(Rect, Vec<usize>)> = vec![];
        for (ci, (area, bb, cells)) in comps.iter().enumerate() {
            if *area < min_area {
                // Small holes: the bake's pull-push fill is fine there.
                small += 1;
                small_texels += area;
                continue;
            }
            let margin = n / 64;
            let tb = [
                (bb[0] * cell).saturating_sub(margin),
                (bb[1] * cell).saturating_sub(margin),
                ((bb[2] + 1) * cell + margin).min(n),
                ((bb[3] + 1) * cell + margin).min(n),
            ];
            // An eye zone with no observed skin around it is the socket's
            // own island behind the eyeball: never seen.
            if kind == Kind::EyeSurround {
                let observed = (tb[1]..tb[3])
                    .flat_map(|y| (tb[0]..tb[2]).map(move |x| y * n + x))
                    .filter(|&k| prov.covered[k] && prov.synth[k] < 0.1)
                    .count();
                if observed < 200 {
                    sockets += 1;
                    continue;
                }
            }
            for &c in cells {
                comp_of_cell[c] = ci;
                let (cx, cy) = (c % g, c / g);
                for y in cy * cell..((cy + 1) * cell).min(n) {
                    for x in cx * cell..((cx + 1) * cell).min(n) {
                        if prov.kind[y * n + x] == Some(kind) {
                            core[y * n + x] = true;
                        }
                    }
                }
            }
            rects.extend(
                crops(tb, n, min_side, max_side)
                    .into_iter()
                    .map(|r| (r, vec![ci])),
            );
        }
        // Merge small squares of one kind (two eyes) into one when the
        // union stays small; tiles of one large area stay separate.
        let mut merged = true;
        while merged {
            merged = false;
            'outer: for i in 0..rects.len() {
                for j in i + 1..rects.len() {
                    let (a, b) = (rects[i].0, rects[j].0);
                    let x0 = a.x.min(b.x);
                    let y0 = a.y.min(b.y);
                    let side = ((a.x + a.side).max(b.x + b.side) - x0)
                        .max((a.y + a.side).max(b.y + b.side) - y0);
                    let larger = a.side.max(b.side);
                    if (larger as usize) < max_side
                        && side as usize <= max_side
                        && side * 2 <= larger * 3
                    {
                        let mut ids = rects[i].1.clone();
                        ids.extend(rects[j].1.iter().copied());
                        rects[i] = (
                            Rect {
                                x: x0.min(n as u32 - side),
                                y: y0.min(n as u32 - side),
                                side,
                            },
                            ids,
                        );
                        rects.remove(j);
                        merged = true;
                        break 'outer;
                    }
                }
            }
        }
        if rects.is_empty() {
            continue;
        }
        let mask = feathered(&prov, &core, r);
        for (rect, ids) in rects {
            let id = format!("patch-{:02}", patches.len() + 1);
            // Only this patch's own areas (another area in the same crop
            // has its own patch).
            let own: Vec<f32> = (0..n * n)
                .map(|k| {
                    let c = comp_of_cell[(k / n / cell) * g + (k % n) / cell];
                    if ids.contains(&c) { mask[k] } else { 0.0 }
                })
                .collect();
            let crop = crop_mask(&own, n, rect, ramp);
            let masked_texels = crop.pixels().filter(|p| p[0] > 127).count();
            if masked_texels < 32 {
                continue;
            }
            // Whole-texture mask of this patch (for the context image).
            let mut patch_mask = vec![0f32; n * n];
            for (i, j, p) in crop.enumerate_pixels() {
                patch_mask[(rect.y + j) as usize * n + (rect.x + i) as usize] = p[0] as f32 / 255.0;
            }
            let info = mask_info(&body, &patch_mask, n, dressed.as_ref());
            let texels =
                image::imageops::crop_imm(&tex, rect.x, rect.y, rect.side, rect.side).to_image();
            let current: Vec<[f64; 3]> = crop
                .enumerate_pixels()
                .filter(|(_, _, p)| p[0] > 127)
                .map(|(i, j, _)| lab8(texels.get_pixel(i, j)))
                .collect();
            // Opaque RGB: image generators mishandle transparent gutters.
            image::DynamicImage::ImageRgba8(rotate_cw(
                &resize_rgba(&texels, PATCH_SIDE),
                info.rotation_cw,
            ))
            .to_rgb8()
            .save(out.join(format!("{id}.png")))?;
            rotate_cw(&resize_gray(&crop, PATCH_SIDE), info.rotation_cw)
                .save(out.join(format!("{id}.mask.png")))?;
            context_image(&tex, &body, &patch_mask, n, rect)
                .save(out.join(format!("{id}.context.png")))?;
            patches.push(Patch {
                prompt: prompt_for(kind, &info.region),
                id: id.clone(),
                kind,
                region: info.region,
                visible_dressed: info.visible_dressed,
                crop: rect,
                rotation_cw: info.rotation_cw,
                patch_side: PATCH_SIDE,
                image: format!("{id}.png"),
                mask: format!("{id}.mask.png"),
                context: format!("{id}.context.png"),
                edited: format!("{id}.edited.png"),
                masked_texels,
                reference_lab: reference_lab(&tex, &prov, rect),
                current_lab: median3(current),
            });
        }
    }
    let manifest = Manifest {
        version: 1,
        run: run.to_path_buf(),
        texture: "base.png".into(),
        provenance: "provenance.png".into(),
        texture_size: n as u32,
        instructions: INSTRUCTIONS.into(),
        tolerances: Tolerances::default(),
        patches,
    };
    charloom_io::write_json(&out.join("patches.json"), &manifest)?;
    std::fs::write(out.join("INSTRUCTIONS.md"), INSTRUCTIONS_MD)?;
    Ok(json!({
        "patches_dir": out,
        "patches": manifest.patches.iter().map(|p| json!({"id": p.id, "kind": p.kind, "region": p.region, "crop": p.crop, "rotation_cw": p.rotation_cw, "masked_texels": p.masked_texels, "visible_dressed": p.visible_dressed})).collect::<Vec<_>>(),
        "skipped": {"small_components": small, "small_component_texels": small_texels, "eye_socket_components": sockets},
        "synthesized_fraction": prov.synth.iter().zip(&prov.covered).filter(|(s, c)| **c && **s > 0.5).count() as f64 / covered.max(1) as f64,
    }))
}

/// Result of validating and blending one edited patch.
#[derive(Debug, Serialize)]
pub struct PatchResult {
    pub id: String,
    pub accepted: bool,
    pub reasons: Vec<String>,
    pub metrics: Value,
}

/// Validates `edited` against the exported patch and, if accepted, blends
/// it into `texture` inside the feathered mask. `original`, `mask` and
/// `island` are at patch resolution.
pub fn apply_patch(
    texture: &mut RgbaImage,
    patch: &Patch,
    original: &RgbaImage,
    mask: &GrayImage,
    island: &GrayImage,
    edited: &RgbaImage,
    tol: &Tolerances,
) -> PatchResult {
    let side = patch.patch_side;
    let mut reasons = vec![];
    if edited.dimensions() != (side, side) {
        return PatchResult {
            id: patch.id.clone(),
            accepted: false,
            reasons: vec![format!(
                "size {}x{} != {side}x{side}: save the edit at exactly the patch size",
                edited.width(),
                edited.height()
            )],
            metrics: json!({}),
        };
    }
    let s = side as usize;
    // Keep area: island pixels outside the mask.
    let keep: Vec<bool> = (0..s * s)
        .map(|k| mask.as_raw()[k] == 0 && island.as_raw()[k] > 127)
        .collect();
    let lab_o: Vec<[f64; 3]> = original.pixels().map(lab8).collect();
    let lab_e: Vec<[f64; 3]> = edited.pixels().map(lab8).collect();
    let n_keep = keep.iter().filter(|k| **k).count();
    let drift: [f64; 3] = std::array::from_fn(|j| {
        (0..s * s)
            .filter(|&k| keep[k])
            .map(|k| lab_e[k][j] - lab_o[k][j])
            .sum::<f64>()
            / n_keep.max(1) as f64
    });
    let keep_de: Vec<f64> = (0..s * s)
        .filter(|&k| keep[k])
        .map(|k| delta_e(std::array::from_fn(|j| lab_e[k][j] - drift[j]), lab_o[k]))
        .collect();
    let drift_de = delta_e(drift, [0.0; 3]);
    let (p50, p90) = (percentile(keep_de.clone(), 0.5), percentile(keep_de, 0.9));
    if n_keep > 0 {
        if drift_de > tol.keep_drift_delta_e {
            reasons.push(format!(
                "keep area tone drifted by ΔE {drift_de:.1} > {}",
                tol.keep_drift_delta_e
            ));
        }
        if p50 > tol.keep_delta_e_p50 || p90 > tol.keep_delta_e_p90 {
            reasons.push(format!(
                "keep area changed (ΔE p50 {p50:.1}, p90 {p90:.1} > {}, {}): content moved or was repainted outside the mask",
                tol.keep_delta_e_p50, tol.keep_delta_e_p90
            ));
        }
    }
    // Low-frequency tone anchoring: the edit's offset from the original
    // where it must stay unchanged, interpolated into the mask, so the
    // blend keeps the bake's tone and takes the edit's detail.
    let m: Vec<f32> = mask.as_raw().iter().map(|&v| v as f32 / 255.0).collect();
    let lin_o: Vec<[f32; 3]> = original.pixels().map(to_lin).collect();
    let lin_e: Vec<[f32; 3]> = edited.pixels().map(to_lin).collect();
    let diff: Vec<[f32; 3]> = (0..s * s)
        .map(|k| std::array::from_fn(|j| lin_o[k][j] - lin_e[k][j]))
        .collect();
    let anchor_w: Vec<f32> = (0..s * s)
        .map(|k| {
            if island.as_raw()[k] > 127 {
                (1.0 - m[k]).powi(2)
            } else {
                0.0
            }
        })
        .collect();
    let offset = if anchor_w.iter().any(|w| *w > 0.0) {
        pull_push(&diff, &anchor_w, s, s)
    } else {
        vec![[0.0; 3]; s * s]
    };
    let anchored: Vec<[f32; 3]> = (0..s * s)
        .map(|k| std::array::from_fn(|j| (lin_e[k][j] + offset[k][j]).clamp(0.0, 1.0)))
        .collect();
    let srgb = |c: [f32; 3]| c.map(|x| linear_to_srgb(x as f64));
    let masked: Vec<usize> = (0..s * s).filter(|&k| m[k] > 0.5).collect();
    let tone_raw = median3(masked.iter().map(|&k| lab_e[k]).collect());
    let tone_lab: Vec<[f64; 3]> = masked.iter().map(|&k| lab(srgb(anchored[k]))).collect();
    let tone = median3(tone_lab.clone());
    let tone_de = delta_e(tone, patch.reference_lab);
    let tone_p95 = percentile(
        tone_lab
            .iter()
            .map(|c| delta_e(*c, patch.reference_lab))
            .collect(),
        0.95,
    );
    // Where the bake's own fill already differs from the observed skin
    // (scalp stubble, soles), the edit may differ as much, plus a little,
    // as long as it stays close to that fill.
    let fill_de = delta_e(patch.current_lab, patch.reference_lab);
    let vs_fill = delta_e(tone, patch.current_lab);
    let bound = tol.tone_delta_e.max(fill_de + 2.0);
    if tone_de > bound || (tone_de > tol.tone_delta_e && vs_fill > tol.tone_delta_e) {
        reasons.push(format!(
            "masked tone ΔE {tone_de:.1} from the surrounding observed skin > {bound:.1} (ΔE {vs_fill:.1} from the current fill)",
        ));
    }
    if tone_p95 > tol.tone_delta_e_p95 {
        reasons.push(format!(
            "masked area has strong marks (ΔE p95 {tone_p95:.1} > {}): eyes, hair, shadows or seams painted in",
            tol.tone_delta_e_p95
        ));
    }
    let metrics = json!({
        "keep_pixels": n_keep,
        "keep_drift_lab": drift,
        "keep_drift_delta_e": drift_de,
        "keep_delta_e_p50": p50,
        "keep_delta_e_p90": p90,
        "masked_pixels": masked.len(),
        "masked_lab_raw": tone_raw,
        "masked_lab_blended": tone,
        "reference_lab": patch.reference_lab,
        "tone_delta_e": tone_de,
        "tone_delta_e_raw": delta_e(tone_raw, patch.reference_lab),
        "tone_delta_e_p95": tone_p95,
        "current_fill_delta_e": fill_de,
        "tone_vs_current_fill_delta_e": vs_fill,
    });
    let accepted = reasons.is_empty();
    if accepted {
        // Blend in texel space: bilinear patch samples at each texel centre.
        let r = patch.crop;
        let f = side as f32 / r.side as f32;
        let bil = |x: f32, y: f32| -> ([f32; 3], f32) {
            let (x, y) = (
                (x * f - 0.5).clamp(0.0, side as f32 - 1.0),
                (y * f - 0.5).clamp(0.0, side as f32 - 1.0),
            );
            let (x0, y0) = (x.floor() as usize, y.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(s - 1), (y0 + 1).min(s - 1));
            let (tx, ty) = (x - x0 as f32, y - y0 as f32);
            let wts = [
                (y0 * s + x0, (1.0 - tx) * (1.0 - ty)),
                (y0 * s + x1, tx * (1.0 - ty)),
                (y1 * s + x0, (1.0 - tx) * ty),
                (y1 * s + x1, tx * ty),
            ];
            let mut c = [0f32; 3];
            let mut mm = 0f32;
            for (k, w) in wts {
                for j in 0..3 {
                    c[j] += anchored[k][j] * w;
                }
                mm += m[k] * w;
            }
            (c, mm)
        };
        for j in 0..r.side {
            for i in 0..r.side {
                let (c, mm) = bil(i as f32 + 0.5, j as f32 + 0.5);
                if mm <= 0.0 {
                    continue;
                }
                let p = texture.get_pixel_mut(r.x + i, r.y + j);
                let base = to_lin(p);
                for ch in 0..3 {
                    let v = base[ch] * (1.0 - mm) + c[ch] * mm;
                    p[ch] = (linear_to_srgb(v as f64) * 255.0).round() as u8;
                }
            }
        }
    }
    PatchResult {
        id: patch.id.clone(),
        accepted,
        reasons,
        metrics,
    }
}

/// Re-derives the UV gutter the bake grew around the islands, from the
/// (possibly edited) island texels.
fn regrow_gutter(texture: &mut RgbaImage, prov: &Provenance) {
    let mut filled = prov.covered.clone();
    for (k, p) in texture.pixels_mut().enumerate() {
        if !filled[k] {
            *p = Rgba([0, 0, 0, 0]);
        }
    }
    dilate(texture, &mut filled, 6);
}

 fn repack_glb(path: &Path, png: &[u8], allowed: &[&RgbaImage]) -> Result<Value> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let glb = gltf::binary::Glb::from_slice(&bytes)?;
    let json: Value = serde_json::from_slice(&glb.json)?;
    let bin = glb.bin.context("GLB without a binary chunk")?.into_owned();
    let mesh = json["meshes"]
        .as_array()
        .and_then(|m| m.iter().position(|m| m["name"] == "body"))
        .context("no mesh named body")?;
    let material = json["meshes"][mesh]["primitives"][0]["material"]
        .as_u64()
        .context("body has no material")? as usize;
    let texture = json["materials"][material]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        .as_u64()
        .context("body material has no base-colour texture")? as usize;
    let image = json["textures"][texture]["source"]
        .as_u64()
        .context("texture source")? as usize;
    let view = json["images"][image]["bufferView"]
        .as_u64()
        .context("image bufferView")? as usize;
    let views = json["bufferViews"]
        .as_array()
        .context("bufferViews")?
        .clone();
    let shared = json["images"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .any(|(i, im)| i != image && im["bufferView"].as_u64() == Some(view as u64))
        || json["accessors"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|a| a["bufferView"].as_u64() == Some(view as u64));
    ensure!(!shared, "body texture's bufferView is shared");
    let range = |v: &Value| -> Result<std::ops::Range<usize>> {
        let o = v["byteOffset"].as_u64().unwrap_or(0) as usize;
        let l = v["byteLength"].as_u64().context("byteLength")? as usize;
        ensure!(o + l <= bin.len(), "bufferView out of bounds");
        Ok(o..o + l)
    };
    let old = image::load_from_memory(&bin[range(&views[view])?])?.to_rgba8();
    ensure!(
        allowed.iter().any(|a| **a == old),
        "{}: embedded body texture is neither the bake nor a previous import",
        path.display()
    );
    let mut doc = GlbDocument {
        json: json.clone(),
        binary: vec![],
    };
    let mut new_views = vec![];
    for (i, v) in views.iter().enumerate() {
        let data: &[u8] = if i == view { png } else { &bin[range(v)?] };
        while !doc.binary.len().is_multiple_of(4) {
            doc.binary.push(0);
        }
        let mut nv = v.clone();
        nv["byteOffset"] = json!(doc.binary.len());
        nv["byteLength"] = json!(data.len());
        doc.binary.extend_from_slice(data);
        new_views.push(nv);
    }
    doc.json["bufferViews"] = json!(new_views);
    charloom_io::write_atomic(path, &doc.to_glb()?)?;
    Ok(
        json!({"glb": path, "image": image, "image_bytes_before": range(&views[view])?.len(), "image_bytes_after": png.len()}),
    )
}

/// `charloom mh-texture-import RUN_DIR`: validates and blends every
/// `patch-NN.edited.png`, writes `3-skin/body-basecolor.png` and re-packs
/// the body and character GLBs. Deterministic given the edited images.
pub fn import(run: &Path, patches_dir: Option<&Path>) -> Result<Value> {
    let dir = patches_dir.map_or_else(|| run.join("inpaint"), Path::to_path_buf);
    let manifest: Manifest = serde_json::from_slice(
        &std::fs::read(dir.join("patches.json"))
            .context("read patches.json (run mh-texture-export first)")?,
    )?;
    let base = image::open(dir.join(&manifest.texture))?.to_rgba8();
    let prov_img = image::open(dir.join(&manifest.provenance))?.to_rgba8();
    let prov = Provenance::new(&prov_img);
    let mut texture = base.clone();
    let mut results = vec![];
    for patch in &manifest.patches {
        let edited_path = dir.join(&patch.edited);
        if !edited_path.exists() {
            results.push(PatchResult {
                id: patch.id.clone(),
                accepted: false,
                reasons: vec!["not edited (no edited image)".into()],
                metrics: json!({"skipped": true}),
            });
            continue;
        }
        let edited = image::open(&edited_path)
            .with_context(|| format!("decode {}", edited_path.display()))?
            .to_rgba8();
        let back = (4 - patch.rotation_cw % 4) % 4;
        let edited = rotate_cw(&edited, back);
        let r = patch.crop;
        // Originals from the base texture, as exported.
        let crop =
            |img: &RgbaImage| image::imageops::crop_imm(img, r.x, r.y, r.side, r.side).to_image();
        let original = resize_rgba(&crop(&base), patch.patch_side);
        let mask = rotate_cw(&image::open(dir.join(&patch.mask))?.to_luma8(), back);
        if mask.dimensions() != (patch.patch_side, patch.patch_side) {
            bail!("{}: mask size does not match patch_side", patch.id);
        }
        let island_crop = GrayImage::from_fn(r.side, r.side, |i, j| {
            Luma([
                if prov.covered[(r.y + j) as usize * prov.n + (r.x + i) as usize] {
                    255
                } else {
                    0
                },
            ])
        });
        let island = image::imageops::resize(
            &island_crop,
            patch.patch_side,
            patch.patch_side,
            FilterType::Nearest,
        );
        results.push(apply_patch(
            &mut texture,
            patch,
            &original,
            &mask,
            &island,
            &edited,
            &manifest.tolerances,
        ));
    }
    let accepted = results.iter().filter(|r| r.accepted).count();
    let skin = skin_dir(run);
    let mut glbs = vec![];
    if accepted > 0 || skin.join("body-basecolor.png").exists() {
        regrow_gutter(&mut texture, &prov);
        let current = image::open(skin.join("body-basecolor.png"))?.to_rgba8();
        let mut png = vec![];
        texture.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)?;
        for glb in [
            skin.join("body.glb"),
            run.join("4-garments").join("character.glb"),
        ] {
            if glb.exists() {
                glbs.push(repack_glb(&glb, &png, &[&base, &current])?);
            }
        }
        charloom_io::write_atomic(&skin.join("body-basecolor.png"), &png)?;
    }
    let changed = texture
        .pixels()
        .zip(base.pixels())
        .filter(|(a, b)| a != b)
        .count();
    let report = json!({
        "patches_dir": dir,
        "accepted": accepted,
        "rejected": results.iter().filter(|r| !r.accepted && !r.metrics["skipped"].as_bool().unwrap_or(false)).count(),
        "changed_texels": changed,
        "texture": skin.join("body-basecolor.png"),
        "glbs": glbs,
        "patches": results,
    });
    charloom_io::write_json(&dir.join("import-report.json"), &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skin(x: u32, y: u32) -> Rgba<u8> {
        // Skin-like tone with a little deterministic texture.
        let t = ((x * 7 + y * 13) % 9) as u8;
        Rgba([200 + t, 150 + t / 2, 130, 255])
    }

    fn setup() -> (Patch, RgbaImage, GrayImage, GrayImage) {
        let side = 64;
        let original = RgbaImage::from_fn(side, side, skin);
        // Mask: a feathered disc in the middle.
        let mask = GrayImage::from_fn(side, side, |x, y| {
            let d = ((x as f32 - 32.0).powi(2) + (y as f32 - 32.0).powi(2)).sqrt();
            Luma([((1.0 - (d - 12.0) / 6.0).clamp(0.0, 1.0) * 255.0) as u8])
        });
        let island = GrayImage::from_pixel(side, side, Luma([255]));
        let patch = Patch {
            id: "patch-01".into(),
            kind: Kind::Hidden,
            region: vec![],
            visible_dressed: 1.0,
            rotation_cw: 0,
            crop: Rect { x: 16, y: 8, side },
            patch_side: side,
            image: String::new(),
            mask: String::new(),
            context: String::new(),
            edited: String::new(),
            masked_texels: 0,
            reference_lab: median3(original.pixels().map(lab8).collect()),
            current_lab: [0.0; 3],
            prompt: String::new(),
        };
        (patch, original, mask, island)
    }

    fn texture_with(original: &RgbaImage, rect: Rect) -> RgbaImage {
        let mut t = RgbaImage::from_pixel(128, 128, Rgba([10, 20, 30, 255]));
        image::imageops::replace(&mut t, original, rect.x as i64, rect.y as i64);
        t
    }

    #[test]
    fn accepted_edit_changes_only_masked_texels() {
        let (patch, original, mask, island) = setup();
        let mut edited = original.clone();
        // Slightly different skin detail inside the mask.
        for (x, y, p) in edited.enumerate_pixels_mut() {
            if mask.get_pixel(x, y)[0] > 0 {
                p[0] = p[0].saturating_sub(((x + y) % 5) as u8 * 2);
            }
        }
        let mut tex = texture_with(&original, patch.crop);
        let before = tex.clone();
        let r = apply_patch(
            &mut tex,
            &patch,
            &original,
            &mask,
            &island,
            &edited,
            &Tolerances::default(),
        );
        assert!(r.accepted, "{:?}", r.reasons);
        let mut changed_inside = 0;
        for (x, y, p) in tex.enumerate_pixels() {
            let inside = x >= patch.crop.x
                && y >= patch.crop.y
                && x < patch.crop.x + patch.crop.side
                && y < patch.crop.y + patch.crop.side
                && mask.get_pixel(x - patch.crop.x, y - patch.crop.y)[0] > 0;
            if !inside {
                // The bilinear mask footprint reaches one texel past the
                // mask's own support at most.
                let near = x + 1 >= patch.crop.x
                    && y + 1 >= patch.crop.y
                    && (x.saturating_sub(patch.crop.x) as i64 - 32).abs() < 20
                    && (y.saturating_sub(patch.crop.y) as i64 - 32).abs() < 20;
                assert!(near || p == before.get_pixel(x, y), "texel {x},{y} changed");
            } else if p != before.get_pixel(x, y) {
                changed_inside += 1;
            }
        }
        assert!(changed_inside > 100);
        // Deterministic.
        let mut tex2 = texture_with(&original, patch.crop);
        apply_patch(
            &mut tex2,
            &patch,
            &original,
            &mask,
            &island,
            &edited,
            &Tolerances::default(),
        );
        assert!(tex == tex2);
    }

    #[test]
    fn rejects_wrong_size_moved_content_and_off_tone() {
        let (patch, original, mask, island) = setup();
        let tol = Tolerances::default();
        let mut tex = texture_with(&original, patch.crop);
        let before = tex.clone();
        let small = RgbaImage::new(32, 32);
        let r = apply_patch(&mut tex, &patch, &original, &mask, &island, &small, &tol);
        assert!(!r.accepted && r.reasons[0].contains("size"));
        // Repainted outside the mask (a dark band across the keep area).
        let mut moved = original.clone();
        for (_, y, p) in moved.enumerate_pixels_mut() {
            if y < 20 {
                *p = Rgba([60, 40, 30, 255]);
            }
        }
        let r = apply_patch(&mut tex, &patch, &original, &mask, &island, &moved, &tol);
        assert!(
            !r.accepted && r.reasons.iter().any(|s| s.contains("keep area")),
            "{:?}",
            r.reasons
        );
        // Green inside the mask: off tone and strong marks.
        let mut green = original.clone();
        for (x, y, p) in green.enumerate_pixels_mut() {
            if mask.get_pixel(x, y)[0] > 200 {
                *p = Rgba([60, 200, 60, 255]);
            }
        }
        let r = apply_patch(&mut tex, &patch, &original, &mask, &island, &green, &tol);
        assert!(
            !r.accepted && r.reasons.iter().any(|s| s.contains("tone")),
            "{:?}",
            r.reasons
        );
        assert!(
            tex == before,
            "rejected patches must not change the texture"
        );
    }

    #[test]
    fn tone_anchoring_removes_a_uniform_shift() {
        let (patch, original, mask, island) = setup();
        // The whole edit 3 % brighter: anchoring brings the masked area back.
        let edited = RgbaImage::from_fn(64, 64, |x, y| {
            let p = skin(x, y);
            Rgba([
                p[0].saturating_add(6),
                p[1].saturating_add(6),
                p[2].saturating_add(6),
                255,
            ])
        });
        let mut tex = texture_with(&original, patch.crop);
        let r = apply_patch(
            &mut tex,
            &patch,
            &original,
            &mask,
            &island,
            &edited,
            &Tolerances::default(),
        );
        assert!(r.accepted, "{:?}", r.reasons);
        let (cx, cy) = (patch.crop.x + 32, patch.crop.y + 32);
        let got = tex.get_pixel(cx, cy);
        let want = original.get_pixel(32, 32);
        for j in 0..3 {
            assert!(
                (got[j] as i32 - want[j] as i32).abs() <= 2,
                "{got:?} vs {want:?}"
            );
        }
    }

    #[test]
    fn crops_tile_large_boxes_and_cover_them() {
        let n = 2048;
        let one = crops([100, 100, 300, 260], n, 512, 1024);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].side, 512);
        let many = crops([0, 0, 1600, 1500], n, 512, 1024);
        assert!(many.len() >= 4);
        for &(x, y) in &[(0u32, 0u32), (1599, 1499), (800, 750)] {
            assert!(
                many.iter()
                    .any(|r| x >= r.x && y >= r.y && x < r.x + r.side && y < r.y + r.side)
            );
        }
    }
}
