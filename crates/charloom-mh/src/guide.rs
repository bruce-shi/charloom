//! Stage 0 — vision-LLM guidance.
//!
//! `prepare` renders calibrated orthographic views of the generated model
//! (four around the body, four around the head) with pixel rulers, plus a
//! JSON template and instructions for a vision LLM. The LLM answers in
//! *pixels*: landmark points (eyes, nose, mouth corners, body joints) and
//! polygons naming garments, hair and skin. `resolve` re-rasterises the same
//! cameras with a triangle-ID buffer, so every annotated pixel maps to an
//! exact scan triangle; joint landmarks take the centre of the limb along the
//! viewing ray, and two views of one landmark are triangulated.
use crate::geom::{V, a, v};
use crate::scan::{RawScan, Scan};
use crate::segment::Label;
use anyhow::{Context, Result, ensure};
use charloom_core::Vec3;
use image::{Rgba, RgbaImage};
use nalgebra::Matrix4;
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;

pub const SCHEMA: &str = "charloom-mh-guidance-v1";

/// Orthographic camera. Pixel (0,0) is the image's top-left corner.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Camera {
    pub id: String,
    pub width: usize,
    pub height: usize,
    pub center: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    /// Viewing direction, from the camera into the scene.
    pub forward: Vec3,
    pub meters_per_pixel: f64,
}

impl Camera {
    pub fn project(&self, p: Vec3) -> (f64, f64, f64) {
        let d = v(p) - v(self.center);
        (
            d.dot(&v(self.right)) / self.meters_per_pixel + self.width as f64 * 0.5,
            self.height as f64 * 0.5 - d.dot(&v(self.up)) / self.meters_per_pixel,
            d.dot(&v(self.forward)),
        )
    }
    /// A point on the viewing ray of a pixel, far in front of the model.
    pub fn ray(&self, px: f64, py: f64) -> (V, V) {
        let o = v(self.center)
            + v(self.right) * ((px - self.width as f64 * 0.5) * self.meters_per_pixel)
            + v(self.up) * ((self.height as f64 * 0.5 - py) * self.meters_per_pixel)
            - v(self.forward) * 100.0;
        (o, v(self.forward))
    }
    /// The same camera after a similarity transform of the scene.
    pub fn transformed(&self, m: &Matrix4<f64>) -> Camera {
        let basis = m.fixed_view::<3, 3>(0, 0).into_owned();
        let s = basis.determinant().cbrt();
        let dir = |d: Vec3| a((basis * v(d)) / s);
        Camera {
            center: a((m * v(self.center).push(1.0)).xyz()),
            right: dir(self.right),
            up: dir(self.up),
            forward: dir(self.forward),
            meters_per_pixel: self.meters_per_pixel * s,
            ..self.clone()
        }
    }
}

/// The silhouette of a bodysuit edit in one view (`scripts/mh-suit-contours.py`):
/// the view's camera in the working frame and its closed curves in pixels.
#[derive(Debug, Clone)]
pub struct SuitContour {
    pub camera: Camera,
    pub curves: Vec<Vec<[f64; 2]>>,
         pub weights: Vec<Vec<f64>>,
    /// Accepted visible-skin texture correspondences (registered GPT -> scan),
    /// in original camera pixels. The held-out e3 member is never included.
    pub skin_anchors: Vec<([f64; 2], [f64; 2])>,
         pub rotation_limits_deg: [f64; 2],
    /// Back-view scale is calibrated by its same-sheet profile, never by a faceless back.
    pub scale_tied_to: Option<String>,
    /// Precalibrated visible scan-skin exterior anchors (GPT -> scan).
    pub skin_outline_anchors: Vec<([f64; 2], [f64; 2])>,
}

fn head_rotation_limits(data:&Value,id:&str)->[f64;2] {
    let (mut lo,mut hi)=(-5f64,5f64);
    if let Some(members)=data["head_registration"]["members"].as_object() {
        for (name,views) in members {
            if name=="e3" || views[id]["accepted"].as_bool()!=Some(true){continue;}
            if let Some(angle)=views[id]["similarity"]["rotation_deg"].as_f64() {
                lo=lo.max(-5.-angle);hi=hi.min(5.-angle);
            }
        }
    }
    [lo.min(0.),hi.max(0.)]
}

fn registered_skin_anchors(data:&Value,id:&str)->Vec<([f64;2],[f64;2])> {
    registered_anchors(data, id, "texture_matches_detail")
}

fn registered_anchors(data:&Value,id:&str,field:&str)->Vec<([f64;2],[f64;2])> {
    let mut anchors=vec![];
    let Some(members)=data["head_registration"]["members"].as_object()else{return anchors;};
    for (name,views) in members {
        if name=="e3" {continue;}
        let r=&views[id];if r["accepted"].as_bool()!=Some(true){continue;}
        let m=&r["similarity"]["matrix"];
        let Some(values)=(0..6).map(|i|m[i/3][i%3].as_f64()).collect::<Option<Vec<_>>>()else{continue;};
        let Some(pairs)=r[field].as_array()else{continue;};
        for pair in pairs {
            let Some(p)=(0..4).map(|i|pair[i/2][i%2].as_f64()).collect::<Option<Vec<_>>>()else{continue;};
            anchors.push(([values[0]*p[2]+values[1]*p[3]+values[2],values[3]*p[2]+values[4]*p[3]+values[5]],[p[0],p[1]]));
        }
    }
    anchors
}

/// `views/<file>` beside the guidance, else `views/suit_contours.json`,
/// its views' cameras carried into the working frame (none when neither
/// file is there).
pub fn load_suit_contours(guidance_path: &std::path::Path, file: &str, views: &ViewSet, scan: &Scan) -> Vec<SuitContour> {
    load_contour_channel(guidance_path, file, views, scan, false)
}

/// Independently registered head channel. Older contour files retain their
/// original behavior; an explicitly empty head channel means no accepted view.
pub fn load_suit_head_contours(guidance_path: &std::path::Path, file: &str, views: &ViewSet, scan: &Scan) -> Vec<SuitContour> {
    load_contour_channel(guidance_path, file, views, scan, true)
}

fn load_contour_channel(guidance_path: &std::path::Path, file: &str, views: &ViewSet, scan: &Scan, head: bool) -> Vec<SuitContour> {
    let Some(dir) = guidance_path.parent() else {
        return vec![];
    };
    let Ok(text) = std::fs::read_to_string(dir.join("views").join(file))
        .or_else(|_| std::fs::read_to_string(dir.join("views").join("suit_contours.json")))
    else {
        return vec![];
    };
    let Ok(mut json) = serde_json::from_str::<serde_json::Value>(&text) else {
        return vec![];
    };
    // Supplementary back channel: existing body/head inputs remain immutable.
    // Only the all-view head solve consumes this observation, not body sizing.
    if head && file == "suit_contours.ensemble-all.json" {
        if let Ok(text) = std::fs::read_to_string(dir.join("views/suit-back/registration/head-contours.json")) {
            if let Ok(extra) = serde_json::from_str::<Value>(&text) {
                for key in ["head_views", "head_weights", "head_scale_ties"] {
                    if let Some(add) = extra[key].as_object() {
                        if !json[key].is_object() { json[key] = serde_json::json!({}); }
                        json[key].as_object_mut().unwrap().extend(add.clone());
                    }
                }
                if let Some(members) = extra["head_registration"]["members"].as_object() {
                    for (name, views) in members {
                        if name == "e3" { continue; }
                        let dst = &mut json["head_registration"]["members"][name];
                        if !dst.is_object() { *dst = serde_json::json!({}); }
                        if let Some(add) = views.as_object() { dst.as_object_mut().unwrap().extend(add.clone()); }
                    }
                }
            }
        }
    }
    let local_head = head && json.get("head_views").is_some();
    let (view_key, weight_key) = if local_head { ("head_views", "head_weights") } else { ("views", "weights") };
    let Some(map) = json[view_key].as_object() else {
        return vec![];
    };
    map.iter()
        .filter_map(|(id, curves)| {
            let camera = views.cameras.iter().find(|c| &c.id == id)?.transformed(&scan.transform);
            let curves: Vec<Vec<[f64; 2]>> = curves
                .as_array()?
                .iter()
                .filter_map(|c| {
                    c.as_array()?
                        .iter()
                        .map(|p| Some([p.get(0)?.as_f64()?, p.get(1)?.as_f64()?]))
                        .collect::<Option<Vec<_>>>()
                })
                .collect();
            let weights = curves
                .iter()
                .enumerate()
                .map(|(k, c)| {
                    let w = json[weight_key][id.as_str()][k].as_array();
                    (0..c.len())
                        .map(|j| w.filter(|w| w.len() == c.len()).and_then(|w| w[j].as_f64()).unwrap_or(1.0).clamp(0.0, 1.0))
                        .collect()
                })
                .collect();
            let skin_anchors=if local_head {registered_skin_anchors(&json,id)}else{vec![]};
            let rotation_limits_deg=if local_head {head_rotation_limits(&json,id)}else{[-5.,5.]};
            let scale_tied_to = json["head_scale_ties"][id.as_str()].as_str().map(str::to_owned);
            let skin_outline_anchors = if local_head {registered_anchors(&json,id,"outline_matches_detail")} else {vec![]};
            (!curves.is_empty()).then_some(SuitContour { camera, curves, weights, skin_anchors, rotation_limits_deg, scale_tied_to, skin_outline_anchors })
        })
        .collect()
}

fn cameras_for(lo: V, hi: V) -> Vec<Camera> {
    let mut out = vec![];
    let height = hi.y - lo.y;
    let center = (lo + hi) * 0.5;
    // Head: top 15 % of the model, centred on the topmost slab.
    let head_center = V::new(center.x, hi.y - height * 0.085, center.z);
    for (prefix, c, w, h, extent) in [
        ("body", center, 768usize, 1024usize, height * 1.08),
        ("head", head_center, 768, 768, height * 0.18),
    ] {
        for deg in [0, 90, 180, 270] {
            out.push(ring_camera(prefix, c, w, h, extent, lo, hi, deg as f64));
        }
    }
    out
}

/// One camera of the ring round the model at `deg` yaw (`body_y045`).
#[allow(clippy::too_many_arguments)]
fn ring_camera(prefix: &str, c: V, w: usize, h: usize, extent: f64, lo: V, hi: V, deg: f64) -> Camera {
    let t = deg.to_radians();
    let forward = V::new(-t.sin(), 0.0, -t.cos());
    let up = V::y();
    let right = forward.cross(&up);
    let mpp = if prefix == "head" {
        extent / h as f64
    } else {
        let half_w = (hi - lo).component_mul(&right.abs()).sum() * 0.5 * 1.08;
        (extent / h as f64).max(half_w * 2.0 / w as f64)
    };
    Camera {
        id: format!("{prefix}_y{:03}", (deg.round() as i64).rem_euclid(360)),
        width: w,
        height: h,
        center: a(c),
        right: a(right),
        up: a(up),
        forward: a(forward),
        meters_per_pixel: mpp,
    }
}

/// Extra body views at these yaws (degrees, on the ring of `body_y000` …:
/// `body_y045`, `body_y315`), rendered and added to `views.json` like the
/// four: the bodysuit edits' 3/4 silhouettes (`scripts/mh-suit-ensemble.py`),
/// which give the body's cross-sections their shape where the front and
/// side give only width and depth.
pub fn body_views(guidance_path: &Path, yaws: &[f64]) -> Result<Value> {
    let (_, mut views) = load(guidance_path)?;
    let dir = guidance_path.parent().unwrap_or(Path::new("."));
    let scan = load_source_scan(&views)?;
    let (lo, hi) = scan.positions.iter().fold(
        (V::repeat(f64::INFINITY), V::repeat(f64::NEG_INFINITY)),
        |(l, h), p| (l.inf(&v(*p)), h.sup(&v(*p))),
    );
    let height = hi.y - lo.y;
    let cams: Vec<Camera> = yaws
        .iter()
        .map(|&deg| ring_camera("body", (lo + hi) * 0.5, 768, 1024, height * 1.08, lo, hi, deg))
        .collect();
    views.cameras.retain(|c| !cams.iter().any(|n| n.id == c.id));
    let images = render_views(dir, &scan, &cams, &mut views)?;
    views.cameras.extend(cams.iter().cloned());
    charloom_io::write_json(&dir.join("views.json"), &views)?;
    Ok(json!({"views": images}))
}

/// Visible triangle per pixel (u32::MAX = background) and its depth.
pub struct IdBuffer {
    pub id: Vec<u32>,
    pub depth: Vec<f32>,
}

pub fn rasterize(cam: &Camera, positions: &[Vec3], triangles: &[[u32; 3]]) -> IdBuffer {
    let (w, h) = (cam.width, cam.height);
    let projected: Vec<(f64, f64, f64)> = positions.par_iter().map(|p| cam.project(*p)).collect();
    let mut id = vec![u32::MAX; w * h];
    let mut depth = vec![f32::INFINITY; w * h];
    for (t, tri) in triangles.iter().enumerate() {
        let [p0, p1, p2] = tri.map(|i| projected[i as usize]);
        let area = (p1.0 - p0.0) * (p2.1 - p0.1) - (p2.0 - p0.0) * (p1.1 - p0.1);
        if area.abs() < 1e-12 {
            continue;
        }
        let x0 = p0.0.min(p1.0).min(p2.0).floor().max(0.0) as i64;
        let x1 = (p0.0.max(p1.0).max(p2.0).ceil() as i64).min(w as i64 - 1);
        let y0 = p0.1.min(p1.1).min(p2.1).floor().max(0.0) as i64;
        let y1 = (p0.1.max(p1.1).max(p2.1).ceil() as i64).min(h as i64 - 1);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                let b1 = ((px - p0.0) * (p2.1 - p0.1) - (p2.0 - p0.0) * (py - p0.1)) / area;
                let b2 = ((p1.0 - p0.0) * (py - p0.1) - (px - p0.0) * (p1.1 - p0.1)) / area;
                let b0 = 1.0 - b1 - b2;
                if b0 < -1e-9 || b1 < -1e-9 || b2 < -1e-9 {
                    continue;
                }
                let z = (p0.2 * b0 + p1.2 * b1 + p2.2 * b2) as f32;
                let k = y as usize * w + x as usize;
                if z < depth[k] {
                    depth[k] = z;
                    id[k] = t as u32;
                }
            }
        }
    }
    IdBuffer { id, depth }
}

/// Barycentric coordinates of a pixel centre inside a triangle.
fn pixel_bary(cam: &Camera, positions: &[Vec3], tri: [u32; 3], px: f64, py: f64) -> [f64; 3] {
    let [p0, p1, p2] = tri.map(|i| cam.project(positions[i as usize]));
    let area = (p1.0 - p0.0) * (p2.1 - p0.1) - (p2.0 - p0.0) * (p1.1 - p0.1);
    let b1 = ((px - p0.0) * (p2.1 - p0.1) - (p2.0 - p0.0) * (py - p0.1)) / area;
    let b2 = ((p1.0 - p0.0) * (py - p0.1) - (px - p0.0) * (p1.1 - p0.1)) / area;
    [1.0 - b1 - b2, b1, b2]
}

// 3×5 digit glyphs for the pixel rulers.
const DIGITS: [[u8; 5]; 10] = [
    [7, 5, 5, 5, 7],
    [2, 6, 2, 2, 7],
    [7, 1, 7, 4, 7],
    [7, 1, 7, 1, 7],
    [5, 5, 7, 1, 1],
    [7, 4, 7, 1, 7],
    [7, 4, 7, 5, 7],
    [7, 1, 1, 1, 1],
    [7, 5, 7, 5, 7],
    [7, 5, 7, 1, 7],
];

fn draw_number(img: &mut RgbaImage, mut x: u32, y: u32, n: usize, color: Rgba<u8>) {
    for ch in n.to_string().bytes() {
        let g = DIGITS[(ch - b'0') as usize];
        for (row, bits) in g.iter().enumerate() {
            for col in 0..3 {
                if bits & (4 >> col) != 0 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let (px, py) = (x + col * 2 + dx, y + row as u32 * 2 + dy);
                            if px < img.width() && py < img.height() {
                                img.put_pixel(px, py, color);
                            }
                        }
                    }
                }
            }
        }
        x += 8;
    }
}

fn render(cam: &Camera, scan: &Scan, buffer: &IdBuffer, grid: bool) -> RgbaImage {
    let (w, h) = (cam.width, cam.height);
    let light = (v(cam.forward) * -1.0 + V::new(0.3, 0.5, 0.0)).normalize();
    let pixels: Vec<[u8; 4]> = (0..w * h)
        .into_par_iter()
        .map(|k| {
            let t = buffer.id[k];
            if t == u32::MAX {
                return [96, 100, 108, 255];
            }
            let tri = scan.triangles[t as usize];
            let (px, py) = ((k % w) as f64 + 0.5, (k / w) as f64 + 0.5);
            let b = pixel_bary(cam, &scan.positions, tri, px, py);
            let n = (0..3).fold(V::zeros(), |s, j| {
                s + v(scan.normals[tri[j] as usize]) * b[j]
            });
            let shade = 0.45 + 0.55 * n.normalize().dot(&light).abs();
            let c = scan.color_at(t as usize, b).unwrap_or([0.7; 3]);
            let q = |x: f64| ((x * shade).clamp(0.0, 1.0).powf(1.0 / 1.1) * 255.0) as u8;
            [q(c[0]), q(c[1]), q(c[2]), 255]
        })
        .collect();
    let mut img = RgbaImage::new(w as u32, h as u32);
    for (k, p) in pixels.into_iter().enumerate() {
        img.put_pixel((k % w) as u32, (k / w) as u32, Rgba(p));
    }
    if grid {
        let line = Rgba([255, 255, 0, 255]);
        for x in (0..w).step_by(50) {
            for y in (0..h).step_by(if x % 100 == 0 { 3 } else { 6 }) {
                let p = img.get_pixel_mut(x as u32, y as u32);
                for c in 0..3 {
                    p[c] = ((p[c] as u16 + line[c] as u16) / 2) as u8;
                }
            }
            if x % 100 == 0 && x > 0 {
                draw_number(&mut img, x as u32 + 2, 2, x, line);
            }
        }
        for y in (0..h).step_by(50) {
            for x in (0..w).step_by(if y % 100 == 0 { 3 } else { 6 }) {
                let p = img.get_pixel_mut(x as u32, y as u32);
                for c in 0..3 {
                    p[c] = ((p[c] as u16 + line[c] as u16) / 2) as u8;
                }
            }
            if y % 100 == 0 && y > 0 {
                draw_number(&mut img, 2, y as u32 + 2, y, line);
            }
        }
    }
    img
}

/// Renders `view` of the (working-frame) scan with coloured crosses at world
/// points: a QA overlay of guidance landmarks against fitted joints.
pub fn overlay(
    views: &ViewSet,
    view: &str,
    scan: &Scan,
    marks: &[(Vec3, [u8; 3])],
) -> Option<RgbaImage> {
    let cam = views
        .cameras
        .iter()
        .find(|c| c.id == view)?
        .transformed(&scan.transform);
    let buffer = rasterize(&cam, &scan.positions, &scan.triangles);
    let mut img = render(&cam, scan, &buffer, true);
    for (p, color) in marks {
        let (x, y, _) = cam.project(*p);
        for d in -7i64..=7 {
            for (dx, dy) in [(d, 0), (0, d), (d, 1), (1, d)] {
                let (px, py) = (x as i64 + dx, y as i64 + dy);
                if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                    img.put_pixel(
                        px as u32,
                        py as u32,
                        Rgba([color[0], color[1], color[2], 255]),
                    );
                }
            }
        }
    }
    Some(img)
}

// ---------------------------------------------------------------- schema

/// How much of a landmark is in view in one observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    /// The spot itself is in view.
    Visible,
    /// Partly hidden (a strand, cuff or strap over part of it).
    Partial,
    /// Hidden (under clothing or hair): placed from proportions.
    Occluded,
}

/// One pixel observation of a landmark in a named view.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Point {
    pub view: String,
    pub px: [f64; 2],
    /// 1 = clearly visible; lower for joints estimated under clothing.
    #[serde(default = "one")]
    pub confidence: f64,
    /// Optional. `occluded` halves the confidence, `partial` keeps 80 %.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visibility: Option<Visibility>,
    /// Optional: why the point is here or what hides it.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reason: String,
}

impl Point {
    /// Confidence after the visibility discount, in [0, 1].
    pub fn effective_confidence(&self) -> f64 {
        let factor = match self.visibility {
            Some(Visibility::Occluded) => 0.5,
            Some(Visibility::Partial) => 0.8,
            _ => 1.0,
        };
        (self.confidence * factor).clamp(0.0, 1.0)
    }
}

fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Polygon {
    pub view: String,
    /// Pixel vertices, at least three, in order around the region.
    pub polygon: Vec<[f64; 2]>,
}

/// A band along a thin item (strap, moustache, single strand, cord).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Stroke {
    pub view: String,
    /// Centre line in pixels, in order along the item (one point = a dot).
    pub points: Vec<[f64; 2]>,
    /// Full width of the band in pixels: the item's visible width.
    pub width_px: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GarmentGuide {
    /// Output mesh name, e.g. "robe", "trousers", "boots", "hair", "hairpin".
    pub name: String,
    /// `garment` or `hair` (hair accessories count as hair).
    pub label: Label,
    /// 0 = worn directly on the skin; outer layers have larger numbers.
    #[serde(default)]
    pub layer: u32,
    /// Fits closely (shirt, leggings): the body is allowed to hug it.
    #[serde(default)]
    pub tight: bool,
    /// Distance from the skin to the item's outer surface (material plus
    /// air), millimetres: how far the hidden body sits inside it. Optional;
    /// a gap measured at the item's openings (a cuff or hem with skin just
    /// past it) wins, and without either a conservative default by
    /// `tight` and `layer` applies (`bodysize.rs`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gap_mm: Option<f64>,
    #[serde(default)]
    pub regions: Vec<Polygon>,
    /// Bands along thin parts of the item that a polygon cannot cover.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub strokes: Vec<Stroke>,
         #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub rigid: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Subject {
    /// human, stylized_humanoid, nonhuman (only humanoids are supported).
    pub kind: String,
    /// female / male / unknown — which MakeHuman prior fits best.
    #[serde(default)]
    pub sex: String,
    #[serde(default)]
    pub notes: String,
    /// Head route for this character: `fit` (MakeHuman head fitted, colour
    /// baked) or `keep_scan` (the scan's own head on the MakeHuman body).
    /// `Config::head`, when set, wins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head: Option<crate::head_keep::HeadRoute>,
    /// Hand route (`fit` or `keep_scan`), measured by
    /// `scripts/mh-hand-landmarks.py`: five digits whose landmarks read as a
    /// hand are fitted, any other hand is kept. `Config::hands`, when set,
    /// wins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hands: Option<crate::hand_keep::HandRoute>,
    /// Digits per hand, thumb counted (2–5): 4 = no little finger, 3 = no
    /// ring or little finger, 2 = thumb and index only. One number for both
    /// hands or `{"l": n, "r": n}`. MakeHuman's surplus fingers are left
    /// out of the fit and cut from the output (bones stay). Unset: 5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hand_digits: Option<HandDigits>,
         #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proportions: Option<Proportions>,
}

/// `Subject::proportions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Proportions {
    Human,
    Stylized,
    Extreme,
}

/// Digits per hand (thumb counted): both hands, or per side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum HandDigits {
    Both(u8),
    PerSide { l: u8, r: u8 },
}

impl HandDigits {
    /// `[left, right]`.
    pub fn per_side(self) -> [u8; 2] {
        match self {
            HandDigits::Both(n) => [n, n],
            HandDigits::PerSide { l, r } => [l, r],
        }
    }
}

impl Subject {
    /// `proportions`, or its default from `kind`.
    pub fn proportions(&self) -> Proportions {
        self.proportions.unwrap_or(if self.kind.starts_with("stylized") {
            Proportions::Stylized
        } else {
            Proportions::Human
        })
    }

    /// Digits of the `[left, right]` hand (5 when unset).
    pub fn digits(&self) -> [u8; 2] {
        self.hand_digits.map_or([5, 5], HandDigits::per_side)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Guidance {
    pub schema: String,
    /// `source_hash` from views.json; guidance for another file is refused.
    pub source_hash: String,
    #[serde(default)]
    pub subject: Subject,
    /// Id of the body view in which the character faces the viewer.
    pub front_view: String,
         #[serde(default)]
    pub landmarks: BTreeMap<String, Vec<Point>>,
    #[serde(default)]
    pub garments: Vec<GarmentGuide>,
    /// Bare skin regions (face, arms, legs …) — seeds and skin-colour samples.
    #[serde(default)]
    pub skin: Vec<Polygon>,
    /// Bands of bare skin too thin for a polygon (a strip between two items).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skin_strokes: Vec<Stroke>,
    /// Skin areas whose painted texture must not be baked: hair strands or
    /// garment shadows painted onto skin, stray straps.
    #[serde(default)]
    pub skin_cleanup: Vec<Polygon>,
    /// Thin painted marks on skin (a painted strand or strap) not to bake.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skin_cleanup_strokes: Vec<Stroke>,
    /// The painted eyebrows, one stroke (or a few) per brow along its middle
    /// from the head by the nose to the tail, `width_px` its thickest part,
    /// in a frontal `face_*`/`head_*` view. The skin bake copies the scan
    /// there as painted (never feathered, mirrored or removed as a strand).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub brow_strokes: Vec<Stroke>,
         #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub view_sha256: BTreeMap<String, String>,
}

pub const SURFACE_LANDMARKS: &[&str] = &[
    "eye_l",
    "eye_r",
    "nose_tip",
    "mouth_l",
    "mouth_r",
    // Eye openings (refine pass, `face_*` views): corners and lid midpoints.
    "eye_l_inner",
    "eye_l_outer",
    "eye_l_top",
    "eye_l_bottom",
    "eye_r_inner",
    "eye_r_outer",
    "eye_r_top",
    "eye_r_bottom",
];
/// Optional lid-margin points (refine pass, `face_*` or eye `zoom_*` views):
/// on the edge where each lid meets the eyeball, at about 25 / 50 / 75 % of
/// the way from the inner to the outer corner. Surface points like the eye
/// corners; not required.
pub const LID_LANDMARKS: &[&str] = &[
    "eye_l_upper_1",
    "eye_l_upper_2",
    "eye_l_upper_3",
    "eye_l_lower_1",
    "eye_l_lower_2",
    "eye_l_lower_3",
    "eye_r_upper_1",
    "eye_r_upper_2",
    "eye_r_upper_3",
    "eye_r_lower_1",
    "eye_r_lower_2",
    "eye_r_lower_3",
];

pub fn is_lid_landmark(name: &str) -> bool {
    LID_LANDMARKS.contains(&name)
}

/// Optional ear outline (side head views or an ear `zoom_*`): only what is
/// visible, hair-covered parts left out. `top`: highest point of the helix;
/// `back`: its rearmost point; `lobe`: bottom of the lobe; `front_top`: where
/// the helix joins the head above the tragus; `front_bottom`: where the lobe
/// joins the head; `tragus`: the tip of the tragus. Surface points; MakeHuman's
/// ear is shaped to them (fit::ears) and the scan's ear is baked only inside
/// them.
pub const EAR_LANDMARKS: &[&str] = &[
    "ear_l_top",
    "ear_l_back",
    "ear_l_lobe",
    "ear_l_front_top",
    "ear_l_front_bottom",
    "ear_l_tragus",
    "ear_r_top",
    "ear_r_back",
    "ear_r_lobe",
    "ear_r_front_top",
    "ear_r_front_bottom",
    "ear_r_tragus",
];

/// The ear outline's parts in drawing order round the ear (from the front
/// top over the helix, down to the lobe and up the front past the tragus).
pub const EAR_OUTLINE: [&str; 6] = ["front_top", "top", "back", "lobe", "front_bottom", "tragus"];

pub fn is_ear_landmark(name: &str) -> bool {
    EAR_LANDMARKS.contains(&name)
}

/// Every landmark name guidance may use.
pub fn is_known_landmark(name: &str) -> bool {
    SURFACE_LANDMARKS.contains(&name)
        || is_lid_landmark(name)
        || is_ear_landmark(name)
        || is_joint_landmark(name)
        || is_outline_landmark(name)
}

 pub const OUTLINE_STATIONS: &[&str] = &[
    "chest",
    "waist",
    "hips",
    "upperarm_l",
    "upperarm_r",
    "forearm_l",
    "forearm_r",
    "thigh_l",
    "thigh_r",
    "knee_l",
    "knee_r",
    "shin_l",
    "shin_r",
    // The bald skull above the ears (`scripts/mh-bodysuit.py` shaves the
    // head): its width and depth under the hair.
    "skull",
     
    "foot_l",
    "foot_r",
];

/// `outline_{station}_{w1,w2,d1,d2}`.
pub fn is_outline_landmark(name: &str) -> bool {
    outline_parts(name).is_some()
}

 pub const OUTLINE_PARTS: &[&str] = &["trunk", "neck", "leg_l", "leg_r", "arm_l", "arm_r"];

/// The part of a continuous outline row (`leg_l12` -> `leg_l`).
pub fn outline_part(station: &str) -> Option<&'static str> {
    let base = station.trim_end_matches(|c: char| c.is_ascii_digit());
    (base.len() < station.len())
        .then(|| OUTLINE_PARTS.iter().copied().find(|p| *p == base))
        .flatten()
}

/// (station, 'w' or 'd', 1 or 2) of an outline mark.
pub fn outline_parts(name: &str) -> Option<(&str, char, u8)> {
    let rest = name.strip_prefix("outline_")?;
    let (station, edge) = rest.rsplit_once('_')?;
    let mut chars = edge.chars();
    let kind = chars.next()?;
    let index = chars.next()?.to_digit(10)? as u8;
    ((OUTLINE_STATIONS.contains(&station) || outline_part(station).is_some())
        && matches!(kind, 'w' | 'd')
        && matches!(index, 1 | 2)
        && chars.next().is_none())
    .then_some((station, kind, index))
}

/// One marked body outline: the body's two edges at a station as seen in
/// one view, as viewing rays (origin, shared unit direction) in the working
/// frame. Only the components across the viewing direction are measured.
#[derive(Debug, Clone, Serialize)]
pub struct OutlineMark {
    pub station: String,
    /// 'w': across a front or back view; 'd': across a side view.
    pub kind: char,
    pub view: String,
    pub a: Vec3,
    pub b: Vec3,
    pub dir: Vec3,
    pub confidence: f64,
}

pub const JOINT_LANDMARKS: &[&str] = &[
    "neck",
    "shoulder_l",
    "shoulder_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
    "hip_l",
    "hip_r",
    "knee_l",
    "knee_r",
    "ankle_l",
    "ankle_r",
];

/// Optional hand landmarks, resolved like joints: `finger{1..5}_{l,r}_{base,
/// mid,tip}` (1 = thumb; base = knuckle where the finger leaves the palm,
/// mid = middle knuckle, tip = finger tip) and `palm_l` / `palm_r`.
pub fn hand_landmarks() -> Vec<String> {
    let mut out = vec![];
    for side in ["l", "r"] {
        for finger in 1..=5 {
            for part in ["base", "mid", "tip"] {
                out.push(format!("finger{finger}_{side}_{part}"));
            }
        }
        out.push(format!("palm_{side}"));
    }
    out
}

pub fn is_hand_landmark(name: &str) -> bool {
    if let Some(side) = name.strip_prefix("palm_") {
        return side == "l" || side == "r";
    }
    let Some(rest) = name.strip_prefix("finger") else {
        return false;
    };
    let parts: Vec<&str> = rest.split('_').collect();
    matches!(
        parts.as_slice(),
        [f, "l" | "r", "base" | "mid" | "tip"] if matches!(*f, "1" | "2" | "3" | "4" | "5")
    )
}

/// Landmarks inside the body (joint centres), as opposed to surface points.
pub fn is_joint_landmark(name: &str) -> bool {
    JOINT_LANDMARKS.contains(&name) || is_hand_landmark(name)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewImage {
    /// sha256 of `views/<id>.png` and `views/<id>-grid.png` as rendered.
    pub sha256: String,
    pub grid_sha256: String,
    pub rendered_unix: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewSet {
    pub schema: String,
    pub source: String,
    pub source_hash: String,
    pub cameras: Vec<Camera>,
         #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub images: BTreeMap<String, ViewImage>,
}

const INSTRUCTIONS: &str = include_str!("guide_instructions.md");

pub(crate) fn file_sha256(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Renders the plain and ruled image of every camera into `dir/views` and
/// records their hashes in `views.images`.
fn render_views(
    dir: &Path,
    scan: &Scan,
    cams: &[Camera],
    views: &mut ViewSet,
) -> Result<Vec<Value>> {
    let rendered: Vec<Result<(Value, ViewImage)>> = cams
        .par_iter()
        .map(|cam| {
            let buffer = rasterize(cam, &scan.positions, &scan.triangles);
            let clean = dir.join("views").join(format!("{}.png", cam.id));
            let grid = dir.join("views").join(format!("{}-grid.png", cam.id));
            render(cam, scan, &buffer, false).save(&clean)?;
            render(cam, scan, &buffer, true).save(&grid)?;
            Ok((
                json!({"id":cam.id,"image":format!("views/{}-grid.png", cam.id),"clean":format!("views/{}.png", cam.id),"width":cam.width,"height":cam.height}),
                ViewImage {
                    sha256: file_sha256(&clean)?,
                    grid_sha256: file_sha256(&grid)?,
                    rendered_unix: unix_now(),
                },
            ))
        })
        .collect();
    let mut out = vec![];
    for (cam, r) in cams.iter().zip(rendered) {
        let (info, mut image) = r?;
        // An identical re-render (refine with unchanged eye marks) keeps its
        // time, so marks made on it are not reported as stale.
        if let Some(old) = views.images.get(&cam.id)
            && old.sha256 == image.sha256
        {
            image.rendered_unix = old.rendered_unix;
        }
        views.images.insert(cam.id.clone(), image);
        out.push(info);
    }
    Ok(out)
}

pub fn prepare(source: &Path, out: &Path) -> Result<Value> {
    ensure!(!out.exists(), "output_exists: {}", out.display());
    let raw = RawScan::load(source)?;
    let scan = raw.into_scan(Matrix4::identity());
    let (lo, hi) = scan.positions.iter().fold(
        (V::repeat(f64::INFINITY), V::repeat(f64::NEG_INFINITY)),
        |(l, h), p| (l.inf(&v(*p)), h.sup(&v(*p))),
    );
    let cameras = cameras_for(lo, hi);
    std::fs::create_dir_all(out.join("views"))?;
    let mut set = ViewSet {
        schema: SCHEMA.into(),
        source: source
            .canonicalize()
            .unwrap_or_else(|_| source.to_path_buf())
            .display()
            .to_string(),
        source_hash: scan.source_hash.clone(),
        cameras: cameras.clone(),
        images: BTreeMap::new(),
    };
    let images = render_views(out, &scan, &cameras, &mut set)?;
    charloom_io::write_json(&out.join("views.json"), &set)?;
    let template = Guidance {
        schema: SCHEMA.into(),
        source_hash: scan.source_hash.clone(),
        subject: Subject {
            kind: "human".into(),
            sex: "unknown".into(),
            notes: String::new(),
            head: None,
            hand_digits: None,
            hands: None,
            proportions: None,
        },
        front_view: "body_y000".into(),
        landmarks: SURFACE_LANDMARKS
            .iter()
            .chain(LID_LANDMARKS)
            .chain(EAR_LANDMARKS)
            .chain(JOINT_LANDMARKS)
            .map(|n| (n.to_string(), vec![]))
            .chain(hand_landmarks().into_iter().map(|n| (n, vec![])))
            .collect(),
        view_sha256: set
            .images
            .iter()
            .map(|(k, i)| (k.clone(), i.sha256.clone()))
            .collect(),
        ..Default::default()
    };
    charloom_io::write_json(&out.join("guidance.template.json"), &template)?;
    charloom_io::write_json(
        &out.join("guidance.schema.json"),
        &schemars::schema_for!(Guidance),
    )?;
    std::fs::write(out.join("INSTRUCTIONS.md"), INSTRUCTIONS)?;
    Ok(
        json!({"output":out,"views":images,"template":out.join("guidance.template.json"),"instructions":out.join("INSTRUCTIONS.md")}),
    )
}

fn load_source_scan(views: &ViewSet) -> Result<Scan> {
    let raw = RawScan::load(Path::new(&views.source))?;
    ensure!(
        raw.source_hash == views.source_hash,
        "source file changed since mh-guide"
    );
    Ok(raw.into_scan(Matrix4::identity()))
}

/// Camera looking along `forward` (horizontal) at `center`.
fn horizontal_camera(id: String, center: V, forward: V, size: usize, extent: f64) -> Camera {
    let forward = V::new(forward.x, 0.0, forward.z).normalize();
    let up = V::y();
    let right = forward.cross(&up);
    Camera {
        id,
        width: size,
        height: size,
        center: a(center),
        right: a(right),
        up: a(up),
        forward: a(forward),
        meters_per_pixel: extent / size as f64,
    }
}

/// `guidance` without what refers to views missing from `views` (marks
/// copied from another folder, face views not rendered yet), for the probes
/// of `refine` and `zoom`; also the ids of the missing views.
fn known_views_only(guidance: &Guidance, views: &ViewSet) -> (Guidance, Vec<String>) {
    let known = |v: &str| views.cameras.iter().any(|c| c.id == v);
    let mut missing: Vec<String> = vec![];
    let mut keep = |v: &str| {
        let k = known(v);
        if !k && !missing.iter().any(|m| m == v) {
            missing.push(v.to_string());
        }
        k
    };
    let mut g = guidance.clone();
    for obs in g.landmarks.values_mut() {
        obs.retain(|o| keep(&o.view));
    }
    for item in &mut g.garments {
        item.regions.retain(|r| keep(&r.view));
        item.strokes.retain(|s| keep(&s.view));
    }
    for polys in [&mut g.skin, &mut g.skin_cleanup] {
        polys.retain(|r| keep(&r.view));
    }
    for strokes in [
        &mut g.skin_strokes,
        &mut g.skin_cleanup_strokes,
        &mut g.brow_strokes,
    ] {
        strokes.retain(|s| keep(&s.view));
    }
    g.front_view = "body_y000".into();
    (g, missing)
}

/// Refine pass: after the LLM has marked the eyes and mouth, render close-up
/// `face_*` views centred on them (front and ±35°), add them to views.json
/// and the template, so eye openings can be marked precisely.
pub fn refine(guidance_path: &Path) -> Result<Value> {
    let (guidance, mut views) = load(guidance_path)?;
    let dir = guidance_path.parent().unwrap_or(Path::new("."));
    let scan = load_source_scan(&views)?;
    // Resolve in the source frame (identity transform) to find the face.
    let (probe, missing) = known_views_only(&guidance, &views);
    let resolved = resolve(&probe, &views, &scan)?;
    let (Some(el), Some(er)) = (
        resolved.landmarks.get("eye_l"),
        resolved.landmarks.get("eye_r"),
    ) else {
        anyhow::bail!(
            "refine needs eye_l and eye_r marked in a head view first{}",
            if missing.is_empty() {
                String::new()
            } else {
                format!(
                    " (marks in {} were ignored: those views are not in views.json yet)",
                    missing.join(", ")
                )
            }
        );
    };
    let (el, er) = (v(*el), v(*er));
    let mouth = ["mouth_l", "mouth_r"]
        .iter()
        .filter_map(|n| resolved.landmarks.get(*n))
        .map(|p| v(*p))
        .collect::<Vec<_>>();
    let mouth_c = if mouth.is_empty() {
        (el + er) * 0.5 - V::y() * (el - er).norm()
    } else {
        mouth.iter().sum::<V>() / mouth.len() as f64
    };
    let center = ((el + er) * 0.5 + mouth_c) * 0.5;
    let span = (el - er).norm();
    // Front direction of the face from the chosen front view.
    let front_deg: f64 = guidance
        .front_view
        .strip_prefix("body_y")
        .and_then(|d| d.parse().ok())
        .unwrap_or(0.0);
    views.cameras.retain(|c| !c.id.starts_with("face_"));
    let mut cams = vec![];
    for (suffix, offset) in [("y000", 0.0f64), ("l35", 35.0), ("r35", -35.0)] {
        let t = (front_deg + offset).to_radians();
        cams.push(horizontal_camera(
            format!("face_{suffix}"),
            center,
            V::new(-t.sin(), 0.0, -t.cos()),
            768,
            span * 3.2,
        ));
    }
    render_views(dir, &scan, &cams, &mut views)?;
    let added: Vec<String> = cams.iter().map(|c| c.id.clone()).collect();
    views.cameras.extend(cams);
    charloom_io::write_json(&dir.join("views.json"), &views)?;
    Ok(
        json!({"views_added": added, "next": "mark eye_*_inner/_outer/_top/_bottom (and optionally refine eye_l, eye_r, nose_tip, mouth_l, mouth_r) in the face_* views; with open eyes also the lid margins eye_*_upper_1..3 / eye_*_lower_1..3 (here or in an mh-guide-zoom --landmark eye_l / eye_r close-up); with painted brows one brow_strokes stroke per brow in face_y000"}),
    )
}

/// What `zoom` centres on.
pub enum ZoomTarget {
    /// A landmark already marked in guidance.json.
    Landmark(String),
    /// A pixel box `[x0, y0, x1, y1]` in a view.
    Box(String, [f64; 4]),
}

 pub fn zoom(guidance_path: &Path, target: &ZoomTarget) -> Result<Value> {
    zoom_orbit(guidance_path, target, &[30.0])
}

/// `zoom` with the obliques at these yaw angles (degrees; positive turns
/// towards the area's side, `_o30`; negative away, `_om30`): a ring of
/// views around a hand for the multi-view hand landmarks
/// (`scripts/mh-hand-landmarks.py`).
pub fn zoom_orbit(guidance_path: &Path, target: &ZoomTarget, orbit: &[f64]) -> Result<Value> {
    let (guidance, mut views) = load(guidance_path)?;
    let dir = guidance_path.parent().unwrap_or(Path::new("."));
    let scan = load_source_scan(&views)?;
    let unit = unit_of(&scan);
    let find = |id: &str| -> Result<Camera> {
        views
            .cameras
            .iter()
            .find(|c| c.id == id)
            .cloned()
            .with_context(|| format!("unknown view {id}"))
    };
    let (slug, reference, center, extent) = match target {
        ZoomTarget::Landmark(name) if name == "ear_l" || name == "ear_r" => {
            // The whole ear: centred on its marked outline points.
            let prefix = format!("{name}_");
            let marked: Vec<&Point> = guidance
                .landmarks
                .iter()
                .filter(|(n, _)| n.starts_with(&prefix))
                .flat_map(|(_, o)| o.iter())
                .collect();
            let obs = marked
                .iter()
                .max_by(|a, b| {
                    a.effective_confidence()
                        .total_cmp(&b.effective_confidence())
                })
                .with_context(|| {
                    format!("no {prefix}* point is marked; mark the ear in the side head view first, or use --box")
                })?;
            let cam = find(&obs.view)?;
            let resolved = resolve(&known_views_only(&guidance, &views).0, &views, &scan)?;
            let points: Vec<V> = resolved
                .landmarks
                .iter()
                .filter(|(n, _)| n.starts_with(&prefix))
                .map(|(_, p)| v(*p))
                .collect();
            ensure!(
                !points.is_empty(),
                "no {prefix}* point resolves onto the model"
            );
            let centre = points.iter().sum::<V>() / points.len() as f64;
            (name.clone(), cam, centre, 0.14 * unit)
        }
        ZoomTarget::Landmark(name) => {
            let obs = guidance
                .landmarks
                .get(name)
                .and_then(|o| {
                    o.iter().max_by(|a, b| {
                        a.effective_confidence()
                            .total_cmp(&b.effective_confidence())
                    })
                })
                .with_context(|| format!("{name} is not marked in guidance.json"))?;
            let cam = find(&obs.view)?;
            let resolved = resolve(&known_views_only(&guidance, &views).0, &views, &scan)?;
            let point = resolved
                .landmarks
                .get(name)
                .or_else(|| resolved.joint_section.get(name))
                .map(|p| v(*p))
                .or_else(|| {
                    // A ray without a section: its first surface.
                    let buffer = rasterize(&cam, &scan.positions, &scan.triangles);
                    let (x, y) = (obs.px[0].floor() as usize, obs.px[1].floor() as usize);
                    let k = y.checked_mul(cam.width)? + x;
                    (x < cam.width && y < cam.height && buffer.id[k] != u32::MAX).then(|| {
                        let (o, d) = cam.ray(obs.px[0], obs.px[1]);
                        o + d * (buffer.depth[k] as f64 + 100.0)
                    })
                })
                .with_context(|| format!("{name} does not resolve onto the model"))?;
            let extent = if is_hand_landmark(name) {
                0.08
            } else if is_joint_landmark(name) {
                0.16
            } else {
                0.05
            } * unit;
            (name.clone(), cam, point, extent)
        }
        ZoomTarget::Box(view, [x0, y0, x1, y1]) => {
            let cam = find(view)?;
            let (x0, x1) = (x0.min(*x1), x0.max(*x1));
            let (y0, y1) = (y0.min(*y1), y0.max(*y1));
            ensure!(
                x1 - x0 >= 4.0 && y1 - y0 >= 4.0,
                "box must be at least 4 px wide and high"
            );
            let buffer = rasterize(&cam, &scan.positions, &scan.triangles);
            let mut depths = vec![];
            for y in (y0.max(0.0) as usize)..(y1.ceil() as usize).min(cam.height) {
                for x in (x0.max(0.0) as usize)..(x1.ceil() as usize).min(cam.width) {
                    let k = y * cam.width + x;
                    if buffer.id[k] != u32::MAX {
                        depths.push(buffer.depth[k] as f64);
                    }
                }
            }
            ensure!(!depths.is_empty(), "box covers no part of the model");
            depths.sort_by(f64::total_cmp);
            let (o, d) = cam.ray((x0 + x1) * 0.5, (y0 + y1) * 0.5);
            let point = o + d * (depths[depths.len() / 2] + 100.0);
            let extent = (x1 - x0).max(y1 - y0) * cam.meters_per_pixel * 1.15;
            (
                format!("{view}_{:.0}_{:.0}_{:.0}_{:.0}", x0, y0, x1, y1),
                cam,
                point,
                extent,
            )
        }
    };
    let slug: String = slug
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    // Oblique: turn the camera towards the side of the model the area is on.
    let (lo, hi) = scan.positions.iter().fold(
        (V::repeat(f64::INFINITY), V::repeat(f64::NEG_INFINITY)),
        |(l, h), p| (l.inf(&v(*p)), h.sup(&v(*p))),
    );
    let mid = (lo + hi) * 0.5;
    let (f, r) = (v(reference.forward), v(reference.right));
    let side = if (center - mid).dot(&r) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let mut cams = vec![horizontal_camera(format!("zoom_{slug}"), center, f, 768, extent)];
    for &deg in orbit {
        let t = deg.to_radians();
        let oblique = f * t.cos() - r * (side * t.sin());
        let tag = if deg >= 0.0 { format!("o{:.0}", deg) } else { format!("om{:.0}", -deg) };
        cams.push(horizontal_camera(format!("zoom_{slug}_{tag}"), center, oblique, 768, extent));
    }
    views.cameras.retain(|c| !cams.iter().any(|n| n.id == c.id));
    let images = render_views(dir, &scan, &cams, &mut views)?;
    views.cameras.extend(cams.iter().cloned());
    charloom_io::write_json(&dir.join("views.json"), &views)?;
    Ok(json!({
        "views_added": images,
        "meters_per_pixel": extent / 768.0,
        "from_view": reference.id,
        "next": "mark points, polygons or strokes in the zoom_* views like in any other view; replace the coarser observation of the same landmark rather than adding a conflicting one",
    }))
}

// ---------------------------------------------------------------- check

const PALETTE: [[u8; 3]; 10] = [
    [230, 40, 40],
    [40, 140, 255],
    [40, 200, 70],
    [255, 160, 0],
    [190, 60, 255],
    [0, 220, 220],
    [240, 230, 40],
    [150, 90, 30],
    [255, 60, 170],
    [20, 110, 40],
];

/// Left/right landmark pairs: in a frontal view the character's left is on
/// the image's right.
const PAIRS: &[(&str, &str)] = &[
    ("eye_l", "eye_r"),
    ("mouth_l", "mouth_r"),
    ("eye_l_inner", "eye_r_inner"),
    ("eye_l_outer", "eye_r_outer"),
    ("eye_l_upper_1", "eye_r_upper_1"),
    ("eye_l_upper_2", "eye_r_upper_2"),
    ("eye_l_upper_3", "eye_r_upper_3"),
    ("eye_l_lower_1", "eye_r_lower_1"),
    ("eye_l_lower_2", "eye_r_lower_2"),
    ("eye_l_lower_3", "eye_r_lower_3"),
    ("shoulder_l", "shoulder_r"),
    ("elbow_l", "elbow_r"),
    ("wrist_l", "wrist_r"),
    ("hip_l", "hip_r"),
    ("knee_l", "knee_r"),
    ("ankle_l", "ankle_r"),
    ("palm_l", "palm_r"),
];

 const CHAINS: &[(&str, &str, &str, f64)] = &[
    ("shoulder", "elbow", "wrist", 0.55),
    ("hip", "knee", "ankle", 0.53),
];

/// Views of one joint closer than this are one view: no depth from them.
const MIN_VIEW_ANGLE_DEG: f64 = 15.0;

fn blend(img: &mut RgbaImage, x: i64, y: i64, c: [u8; 3], alpha: f32) {
    if x < 0 || y < 0 || x >= img.width() as i64 || y >= img.height() as i64 {
        return;
    }
    let p = img.get_pixel_mut(x as u32, y as u32);
    for (channel, target) in p.0.iter_mut().zip(c) {
        *channel = (*channel as f32 * (1.0 - alpha) + target as f32 * alpha) as u8;
    }
}

fn draw_polygon(img: &mut RgbaImage, poly: &[[f64; 2]], c: [u8; 3]) {
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in poly {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    for y in lo[1].floor() as i64..=hi[1].ceil() as i64 {
        for x in lo[0].floor() as i64..=hi[0].ceil() as i64 {
            if inside(poly, x as f64 + 0.5, y as f64 + 0.5) {
                blend(img, x, y, c, 0.3);
            }
        }
    }
    for i in 0..poly.len() {
        let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
        draw_line(img, p, q, c);
    }
}

fn draw_line(img: &mut RgbaImage, p: [f64; 2], q: [f64; 2], c: [u8; 3]) {
    let steps = (q[0] - p[0]).abs().max((q[1] - p[1]).abs()).ceil().max(1.0) as i64;
    for s in 0..=steps {
        let t = s as f64 / steps as f64;
        let (x, y) = (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t);
        for (dx, dy) in [(0, 0), (1, 0), (0, 1)] {
            blend(img, x as i64 + dx, y as i64 + dy, c, 1.0);
        }
    }
}

/// A stroke: its band translucent, its centre line solid.
fn draw_stroke(img: &mut RgbaImage, s: &Stroke, c: [u8; 3]) {
    let half = s.width_px.max(1.0) * 0.5;
    let Some((lo, hi)) = stroke_bounds(s) else {
        return;
    };
    for y in lo[1].floor() as i64..=hi[1].ceil() as i64 {
        for x in lo[0].floor() as i64..=hi[0].ceil() as i64 {
            if polyline_distance(&s.points, x as f64 + 0.5, y as f64 + 0.5) <= half {
                blend(img, x, y, c, 0.45);
            }
        }
    }
    for w in s.points.windows(2) {
        draw_line(img, w[0], w[1], c);
    }
    if s.points.len() == 1 {
        draw_line(img, s.points[0], s.points[0], c);
    }
}

fn draw_cross(img: &mut RgbaImage, px: [f64; 2], c: [u8; 3]) {
    let (x, y) = (px[0] as i64, px[1] as i64);
    for d in -6i64..=6 {
        for (dx, dy) in [(d, 0), (0, d), (d, 1), (1, d)] {
            blend(img, x + dx, y + dy, c, 1.0);
        }
    }
}

/// Overlay colour of a lid-margin landmark (upper cyan, lower green).
fn lid_colour(name: &str) -> Option<[u8; 3]> {
    if !is_lid_landmark(name) {
        None
    } else if name.contains("_upper_") {
        Some([0, 224, 255])
    } else {
        Some([80, 255, 80])
    }
}

fn draw_small_cross(img: &mut RgbaImage, px: [f64; 2], c: [u8; 3]) {
    let (x, y) = (px[0] as i64, px[1] as i64);
    for d in -3i64..=3 {
        blend(img, x + d, y, c, 1.0);
        blend(img, x, y + d, c, 1.0);
    }
}

fn draw_thin_line(img: &mut RgbaImage, p: [f64; 2], q: [f64; 2], c: [u8; 3]) {
    let steps = (q[0] - p[0]).abs().max((q[1] - p[1]).abs()).ceil().max(1.0) as i64;
    for s in 0..=steps {
        let t = s as f64 / steps as f64;
        let (x, y) = (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t);
        blend(img, x as i64, y as i64, c, 0.8);
    }
}

/// Small ring at a resolved point projected into a view.
fn draw_dot_ring(img: &mut RgbaImage, px: [f64; 2], c: [u8; 3]) {
    for k in 0..32 {
        let t = k as f64 / 32.0 * std::f64::consts::TAU;
        blend(
            img,
            (px[0] + 4.0 * t.cos()).round() as i64,
            (px[1] + 4.0 * t.sin()).round() as i64,
            c,
            1.0,
        );
    }
}

 fn draw_ring(img: &mut RgbaImage, px: [f64; 2], c: [u8; 3]) {
    for k in 0..64 {
        let t = k as f64 / 64.0 * std::f64::consts::TAU;
        for r in [9.0, 10.0] {
            blend(
                img,
                (px[0] + r * t.cos()) as i64,
                (px[1] + r * t.sin()) as i64,
                c,
                1.0,
            );
        }
    }
}

 fn view_hash_warnings(
    dir: &Path,
    guidance_path: &Path,
    guidance: &Guidance,
    views: &ViewSet,
) -> Vec<String> {
    let mut used: Vec<&str> = guidance
        .landmarks
        .values()
        .flatten()
        .map(|p| p.view.as_str())
        .chain(guidance.garments.iter().flat_map(|g| {
            g.regions
                .iter()
                .map(|r| r.view.as_str())
                .chain(g.strokes.iter().map(|s| s.view.as_str()))
        }))
        .chain(
            guidance
                .skin
                .iter()
                .chain(&guidance.skin_cleanup)
                .map(|r| r.view.as_str()),
        )
        .chain(
            guidance
                .skin_strokes
                .iter()
                .chain(&guidance.skin_cleanup_strokes)
                .chain(&guidance.brow_strokes)
                .map(|s| s.view.as_str()),
        )
        .collect();
    used.sort_unstable();
    used.dedup();
    let mut out = vec![];
    if views.images.is_empty() {
        if !used.is_empty() {
            out.push("views.json records no image hashes (made by an older mh-guide): changed view files cannot be detected".into());
        }
        return out;
    }
    let saved = std::fs::metadata(guidance_path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs());
    let mut unbound = vec![];
    for view in used {
        let Some(record) = views.images.get(view) else {
            unbound.push(view);
            continue;
        };
        let file = dir.join("views").join(format!("{view}.png"));
        match file_sha256(&file) {
            Ok(h) if h != record.sha256 => out.push(format!(
                "{view}: views/{view}.png changed since it was rendered (hash differs from views.json); the marks may not match the picture"
            )),
            Err(_) => out.push(format!("{view}: views/{view}.png is missing")),
            _ => {}
        }
        if let Some(h) = guidance.view_sha256.get(view) {
            if *h != record.sha256 {
                out.push(format!(
                    "{view}: guidance was marked on an earlier render of this view (view_sha256 differs); re-mark its points and polygons"
                ));
            }
        } else if saved.is_some_and(|s| s < record.rendered_unix) {
            out.push(format!(
                "{view} was re-rendered after guidance.json was last saved; re-mark its points and polygons"
            ));
        }
    }
    if !unbound.is_empty() {
        out.push(format!(
            "no image hash in views.json for {} (rendered by an older mh-guide): changes to those files cannot be detected",
            unbound.join(", ")
        ));
    }
    out
}

/// Likely finger-numbering mistakes on one hand in one view: knuckles of
/// fingers 2–5 out of order across the hand, a tip nearer the palm centre
/// than its own knuckle, two hand points on the same pixel.
fn hand_warnings(view: &str, side: &str, here: &BTreeMap<&str, [f64; 2]>) -> Vec<String> {
    let mut out = vec![];
    let get = |n: String| here.get(n.as_str()).copied();
    let dist = |p: [f64; 2], q: [f64; 2]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
    let bases: Vec<(usize, [f64; 2])> = (2..=5)
        .filter_map(|f| Some((f, get(format!("finger{f}_{side}_base"))?)))
        .collect();
    if bases.len() >= 3 {
        let (p0, p1) = (bases[0].1, bases[bases.len() - 1].1);
        let axis = [p1[0] - p0[0], p1[1] - p0[1]];
        let along: Vec<f64> = bases
            .iter()
            .map(|(_, p)| (p[0] - p0[0]) * axis[0] + (p[1] - p0[1]) * axis[1])
            .collect();
        if along.windows(2).any(|w| w[1] < w[0]) {
            out.push(format!(
                "{view}: the knuckles finger2…finger5_{side}_base are not in order across the hand (index, middle, ring, little); check the finger numbers"
            ));
        }
    }
    let palm = get(format!("palm_{side}"));
    for f in 1..=5 {
        let (base, tip) = (
            get(format!("finger{f}_{side}_base")),
            get(format!("finger{f}_{side}_tip")),
        );
        if let (Some(p), Some(b), Some(t)) = (palm, base, tip)
            && dist(t, p) < dist(b, p)
        {
            out.push(format!(
                "{view}: finger{f}_{side}_tip is nearer palm_{side} than its knuckle; check base and tip are not swapped (fine for a curled finger)"
            ));
        }
    }
    let names: Vec<(&str, [f64; 2])> = here
        .iter()
        .filter(|(n, _)| is_hand_landmark(n) && n.contains(&format!("_{side}")))
        .map(|(n, p)| (*n, *p))
        .collect();
    for (i, (a, p)) in names.iter().enumerate() {
        for (b, q) in &names[i + 1..] {
            if dist(*p, *q) < 2.0 {
                out.push(format!(
                    "{view}: {a} and {b} are on the same pixel; one of them is probably a copy"
                ));
            }
        }
    }
    out
}

/// Lid-margin sanity, on the resolved points projected into the front face
/// view (pixels, y down; the character's left on the image's right): each
/// lid's three points in order from the inner to the outer corner at about
/// 25 / 50 / 75 %, the upper lid above the lower one, every point inside the
/// box of the four eye-opening points (± 20 % of the eye width), and each
/// eye's points on its own side of the face.
fn lid_warnings(pts: &BTreeMap<String, [f64; 2]>) -> Vec<String> {
    let mut out = vec![];
    let middle = match (pts.get("eye_l_inner"), pts.get("eye_r_inner")) {
        (Some(l), Some(r)) => Some((l[0] + r[0]) * 0.5),
        _ => None,
    };
    for side in ["l", "r"] {
        let get = |n: String| pts.get(&n).copied();
        let lid = |part: &str, k: usize| get(format!("eye_{side}_{part}_{k}"));
        let marked: Vec<&str> = LID_LANDMARKS
            .iter()
            .copied()
            .filter(|n| n.starts_with(&format!("eye_{side}_")) && pts.contains_key(*n))
            .collect();
        if marked.is_empty() {
            continue;
        }
        if marked.len() < 6 {
            out.push(format!(
                "eye_{side}: {} of the 6 lid-margin points are marked (eye_{side}_upper_1..3, eye_{side}_lower_1..3); mark all six or none",
                marked.len()
            ));
        }
        if let Some(mid) = middle {
            // The character's left eye is on the image's right.
            let sign = if side == "l" { 1.0 } else { -1.0 };
            for n in &marked {
                if (pts[*n][0] - mid) * sign <= 0.0 {
                    out.push(format!(
                        "{n} is on the other eye's side of the face; left/right are the character's (its left eye is on the image's right in the front view)"
                    ));
                }
            }
        }
        let (Some(inner), Some(outer)) = (
            get(format!("eye_{side}_inner")),
            get(format!("eye_{side}_outer")),
        ) else {
            out.push(format!(
                "eye_{side}: lid-margin points are marked but eye_{side}_inner / eye_{side}_outer are not; mark the corners too"
            ));
            continue;
        };
        let axis = [outer[0] - inner[0], outer[1] - inner[1]];
        let width = axis[0].hypot(axis[1]);
        if width < 1e-6 {
            continue;
        }
        let along = |p: [f64; 2]| {
            ((p[0] - inner[0]) * axis[0] + (p[1] - inner[1]) * axis[1]) / width.powi(2)
        };
        // Unit normal of the corner line pointing up in the image (y down).
        let mut normal = [-axis[1] / width, axis[0] / width];
        if normal[1] > 0.0 {
            normal = [-normal[0], -normal[1]];
        }
        let height = |p: [f64; 2]| (p[0] - inner[0]) * normal[0] + (p[1] - inner[1]) * normal[1];
        for part in ["upper", "lower"] {
            let fractions: Vec<(usize, f64)> = (1..=3)
                .filter_map(|k| Some((k, along(lid(part, k)?))))
                .collect();
            let ordered = fractions.windows(2).all(|w| w[1].1 > w[0].1)
                && fractions.iter().all(|(_, f)| *f > 0.0 && *f < 1.0);
            if !ordered {
                let at: Vec<String> = fractions
                    .iter()
                    .map(|(k, f)| format!("{k}: {:.0} %", f * 100.0))
                    .collect();
                out.push(format!(
                    "eye_{side}_{part}_1..3 are not in order from eye_{side}_inner to eye_{side}_outer ({}); 1 is nearest the inner corner (by the nose), 3 nearest the outer one",
                    at.join(", ")
                ));
                continue;
            }
            for (k, f) in fractions {
                let target = 0.25 * k as f64;
                if (f - target).abs() > 0.15 {
                    out.push(format!(
                        "eye_{side}_{part}_{k} is {:.0} % of the way from the inner to the outer corner; place it at about {:.0} %",
                        f * 100.0,
                        target * 100.0
                    ));
                }
            }
        }
        for k in 1..=3 {
            if let (Some(u), Some(l)) = (lid("upper", k), lid("lower", k))
                && height(u) <= height(l)
            {
                out.push(format!(
                    "eye_{side}_upper_{k} is not above eye_{side}_lower_{k}; upper and lower lid swapped?"
                ));
            }
        }
        let corners: Vec<[f64; 2]> = ["inner", "outer", "top", "bottom"]
            .iter()
            .filter_map(|k| get(format!("eye_{side}_{k}")))
            .collect();
        let margin = 0.2 * width;
        let lo =
            [0, 1].map(|i| corners.iter().map(|p| p[i]).fold(f64::INFINITY, f64::min) - margin);
        let hi = [0, 1].map(|i| {
            corners
                .iter()
                .map(|p| p[i])
                .fold(f64::NEG_INFINITY, f64::max)
                + margin
        });
        for n in &marked {
            let p = pts[*n];
            if (0..2).any(|i| p[i] < lo[i] || p[i] > hi[i]) {
                out.push(format!(
                    "{n} at [{:.0}, {:.0}] lies outside the eye opening (the box of eye_{side}_inner/_outer/_top/_bottom ± 20 % of the eye width); mark it on the lid margin",
                    p[0], p[1]
                ));
            }
        }
    }
    out
}

/// Ear outline checks on the resolved 3D points: `fwd` the character's
/// forward, `up`, `left` its left; `eyes` the middle between the eyes (or
/// another face point), `unit` the length of 1 m on a 1.7 m character.
fn ear_warnings(
    pts: &BTreeMap<String, V>,
    fwd: V,
    up: V,
    left: V,
    eyes: Option<V>,
    unit: f64,
) -> Vec<String> {
    let mut out = vec![];
    let mm = |d: f64| d / unit * 1000.0;
    let mut heights = vec![];
    for side in ["l", "r"] {
        let get = |k: &str| pts.get(&format!("ear_{side}_{k}")).copied();
        let marked: Vec<(&str, V)> = EAR_OUTLINE
            .iter()
            .filter_map(|k| Some((*k, get(k)?)))
            .collect();
        if marked.is_empty() {
            continue;
        }
        let name = |k: &str| format!("ear_{side}_{k}");
        if marked.len() == 1 {
            out.push(format!(
                "ear_{side}: only {} is marked; one point only moves MakeHuman's ear, add any other outline point that is visible (top, back, lobe, front_top, front_bottom, tragus)",
                name(marked[0].0)
            ));
        }
        let middle = eyes.or_else(|| {
            let other = if side == "l" { "r" } else { "l" };
            let theirs: Vec<V> = EAR_OUTLINE
                .iter()
                .filter_map(|k| pts.get(&format!("ear_{other}_{k}")).copied())
                .collect();
            let mine = marked.iter().map(|m| m.1).sum::<V>() / marked.len() as f64;
            (!theirs.is_empty())
                .then(|| (mine + theirs.iter().sum::<V>() / theirs.len() as f64) * 0.5)
        });
        let sign = if side == "l" { 1.0 } else { -1.0 };
        if let Some(mid) = middle {
            for (k, p) in &marked {
                if (p - mid).dot(&left) * sign < 0.03 * unit {
                    out.push(format!(
                        "{} is not on the character's {} side of the head; left/right are the character's (its left ear is seen from +X, the image's right in the front view)",
                        name(k),
                        if side == "l" { "left" } else { "right" }
                    ));
                }
            }
        }
        let (h, f) = (|p: V| p.dot(&up), |p: V| p.dot(&fwd));
        let tol = 0.004 * unit;
        for (k, test, what) in [
            ("top", 0, "highest"),
            ("lobe", 1, "lowest"),
            ("back", 2, "rearmost"),
        ] {
            let Some(p) = get(k) else { continue };
            for (q, other) in marked.iter().filter(|m| m.0 != k) {
                let beyond = match test {
                    0 => h(*other) - h(p),
                    1 => h(p) - h(*other),
                    _ => f(p) - f(*other),
                };
                if beyond > tol {
                    out.push(format!(
                        "{} is not the {what} ear point: {} is {:.0} mm {} (top = highest point of the helix, lobe = bottom of the lobe, back = rearmost point of the helix)",
                        name(k),
                        name(q),
                        mm(beyond),
                        match test {
                            0 => "higher",
                            1 => "lower",
                            _ => "further back",
                        }
                    ));
                }
            }
        }
        if let (Some(t), Some(b)) = (get("front_top"), get("front_bottom"))
            && h(t) <= h(b)
        {
            out.push(format!(
                "{} is not above {}; front_top is where the helix joins the head above the tragus, front_bottom where the lobe joins it",
                name("front_top"),
                name("front_bottom")
            ));
        }
        if let (Some(t), Some(b), Some(g)) = (get("front_top"), get("front_bottom"), get("tragus"))
            && !(h(b) - tol..=h(t) + tol).contains(&h(g))
        {
            out.push(format!(
                "{} is not between {} and {} in height",
                name("tragus"),
                name("front_top"),
                name("front_bottom")
            ));
        }
        if let Some(b) = get("back") {
            for k in ["front_top", "front_bottom", "tragus"] {
                if let Some(p) = get(k)
                    && f(p) - f(b) < 0.01 * unit
                {
                    out.push(format!(
                        "{} is not in front of {} (the front of the ear joins the head, the back is the helix rim)",
                        name(k),
                        name("back")
                    ));
                }
            }
        }
        // A plausible ear: 3–16 cm tall (cartoon ears up to twice a human's
        // 6 cm), behind the eyes and about at their height.
        if let (Some(t), Some(l)) = (get("top"), get("lobe")) {
            let height = h(t) - h(l);
            heights.push((side, height, (h(t) + h(l)) * 0.5));
            if !(0.03 * unit..=0.16 * unit).contains(&height) {
                out.push(format!(
                    "ear_{side}: top to lobe is {:.0} mm (for a 1.7 m character); an ear is 30–160 mm tall: check the points are on the ear",
                    mm(height)
                ));
            }
        }
        let front = ["front_top", "front_bottom", "tragus"]
            .iter()
            .filter_map(|k| get(k))
            .map(f)
            .fold(f64::NEG_INFINITY, f64::max);
        if let Some(b) = get("back")
            && front.is_finite()
            && !(0.01 * unit..=0.1 * unit).contains(&(front - f(b)))
        {
            out.push(format!(
                "ear_{side}: front to back is {:.0} mm (for a 1.7 m character); an ear is 10–100 mm deep from the front where it joins the head to the back of the helix",
                mm(front - f(b))
            ));
        }
        if let Some(e) = eyes {
            let c = marked.iter().map(|m| m.1).sum::<V>() / marked.len() as f64;
            let (behind, below) = (f(e) - f(c), h(e) - h(c));
            if !(0.02 * unit..=0.2 * unit).contains(&behind)
                || !(-0.08 * unit..=0.14 * unit).contains(&below)
            {
                out.push(format!(
                    "ear_{side}: its points lie {:.0} mm behind and {:.0} mm below the eyes (for a 1.7 m character); ears sit 20–200 mm behind and about level with the eyes: check they are on the ear",
                    mm(behind),
                    mm(below)
                ));
            }
        }
    }
    if let [(_, hl, cl), (_, hr, cr)] = heights[..]
        && ((hl - hr).abs() > 0.3 * hl.max(hr) || (cl - cr).abs() > 0.03 * unit)
    {
        out.push(format!(
            "the ears differ: left {:.0} mm tall at {:.0} mm, right {:.0} mm tall at {:.0} mm (heights of their middles); fine for an asymmetric character, otherwise check top and lobe on both sides",
            mm(hl),
            mm(cl),
            mm(hr),
            mm(cr)
        ));
    }
    out
}

/// Overlay colour of the ear outline.
const EAR_COLOUR: [u8; 3] = [255, 150, 0];
/// Brow strokes in the overlays (violet).
const BROW_COLOUR: [u8; 3] = [170, 80, 255];

/// The camera lid checks and lid curves are judged in: `face_y000`, else the
/// head view of the front direction, else the front body view.
fn lid_camera<'a>(cameras: &BTreeMap<&str, &'a Camera>, front_view: &str) -> Option<&'a Camera> {
    let head = front_view.replace("body_", "head_");
    ["face_y000", head.as_str(), front_view]
        .iter()
        .find_map(|id| cameras.get(id).copied())
}

 pub fn check(guidance_path: &Path) -> Result<Value> {
    let (guidance, views) = load(guidance_path)?;
    let dir = guidance_path.parent().unwrap_or(Path::new("."));
    let scan = load_source_scan(&views)?;
    let resolved = resolve(&guidance, &views, &scan)?;
    let cameras: BTreeMap<&str, &Camera> =
        views.cameras.iter().map(|c| (c.id.as_str(), c)).collect();
    let mut warnings: Vec<String> = view_hash_warnings(dir, guidance_path, &guidance, &views);
    let mut errors: Vec<String> = vec![];

    let mut rejected: Vec<(String, [f64; 2])> = vec![];
    if let Some(problems) = resolved.report["landmark_problems"].as_array() {
        for p in problems {
            let line = format!(
                "{} in {} at {}: {}",
                p["landmark"].as_str().unwrap_or("?"),
                p["view"].as_str().unwrap_or("?"),
                p["px"],
                p["problem"].as_str().unwrap_or("?")
            );
            if p["severity"] == "error" {
                errors.push(line);
                if let (Some(view), Some(px)) = (p["view"].as_str(), p["px"].as_array()) {
                    let px = [0, 1].map(|k| px.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0));
                    rejected.push((view.to_string(), px));
                }
            } else {
                warnings.push(line);
            }
        }
    }
    if let Some(snaps) = resolved.report["landmark_snaps"].as_array() {
        for s in snaps {
            warnings.push(format!(
                "{} in {} at {}: {} (used the skin {:.1} mm behind the first surface, {:.1} px away); better mark it on the visible skin",
                s["landmark"].as_str().unwrap_or("?"),
                s["view"].as_str().unwrap_or("?"),
                s["px"],
                s["note"].as_str().unwrap_or(""),
                s["behind_first_surface_mm"].as_f64().unwrap_or(0.0),
                s["moved_px"].as_f64().unwrap_or(0.0),
            ));
        }
    }
    if let Some(seeds) = resolved.report["garment_seeds"].as_array() {
        for s in seeds.iter().filter(|s| s["seed_triangles"] == 0) {
            warnings.push(format!(
                "garment {}: its polygons and strokes cover no visible surface (or a smaller polygon wins everywhere)",
                s["garment"]
            ));
        }
    }
    // Joint depth: one view (or near-parallel views) leaves it to the probe.
    let mut single: Vec<String> = vec![];
    if let Some(info) = resolved.report["landmark_views"].as_object() {
        for (name, i) in info {
            if !JOINT_LANDMARKS.contains(&name.as_str()) {
                continue;
            }
            let n = i["views"].as_u64().unwrap_or(0);
            let spread = i["spread_deg"].as_f64().unwrap_or(0.0);
            if n >= 2 && spread < MIN_VIEW_ANGLE_DEG {
                warnings.push(format!(
                    "{name}: its {n} views are only {spread:.0}° apart, so they give no depth; mark it in a side view (body_y090/body_y270) instead of a second frontal one"
                ));
            } else if n == 1 {
                single.push(name.clone());
            }
        }
    }
    if !single.is_empty() {
        warnings.push(format!(
            "seen in one view only (depth comes from the cross-section probe and the fit): {}; add a side-view observation where the joint is clearly visible there (a guess under a collar or robe is worse than none)",
            single.join(", ")
        ));
    }
    if let Some(sections) = resolved.report["joint_section_depth"].as_object() {
        for (name, s) in sections {
            if s["ambiguous"] == true {
                warnings.push(format!(
                    "{name}: the viewing ray crosses {} surface layers and no body layer is marked; its depth is uncertain (mark the garment over it, or add a side view)",
                    s["crossings"]
                ));
            }
        }
    }
    let has_face_views = cameras.keys().any(|k| k.starts_with("face_"));
    let mut required: Vec<&str> = SURFACE_LANDMARKS
        .iter()
        .copied()
        .filter(|n| has_face_views || !n.contains("_l_") && !n.contains("_r_"))
        .collect();
    required.extend(JOINT_LANDMARKS);
    for name in required {
        if guidance.landmarks.get(name).is_none_or(|o| o.is_empty()) {
            warnings.push(format!("{name}: not marked"));
        }
    }
    for name in guidance.landmarks.keys() {
        if !is_known_landmark(name) {
            errors.push(format!(
                "{name}: unknown landmark name (see INSTRUCTIONS.md); it is ignored"
            ));
        }
    }
    for p in resolved.report["outline_problems"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        warnings.push(p.to_string());
    }
    // Lid margins: judged on the resolved (skin) points seen from the front.
    if let Some(cam) = lid_camera(&cameras, &guidance.front_view) {
        let pts: BTreeMap<String, [f64; 2]> = resolved
            .landmarks
            .iter()
            .filter(|(n, _)| n.starts_with("eye_l_") || n.starts_with("eye_r_"))
            .map(|(n, p)| {
                let (x, y, _) = cam.project(*p);
                (n.clone(), [x, y])
            })
            .collect();
        warnings.extend(lid_warnings(&pts));
    }
    // Ear outlines: judged in 3D in the character's frame.
    if let Some(cam) = cameras.get(guidance.front_view.as_str()) {
        let pts: BTreeMap<String, V> = resolved
            .landmarks
            .iter()
            .filter(|(n, _)| is_ear_landmark(n))
            .map(|(n, p)| (n.clone(), v(*p)))
            .collect();
        let get = |n: &str| resolved.landmarks.get(n).map(|p| v(*p));
        let eyes = match (get("eye_l"), get("eye_r")) {
            (Some(l), Some(r)) => Some((l + r) * 0.5),
            _ => None,
        };
        warnings.extend(ear_warnings(
            &pts,
            -v(cam.forward),
            v(cam.up),
            v(cam.right),
            eyes,
            unit_of(&scan),
        ));
    }
    for (what, strokes) in guidance
        .garments
        .iter()
        .map(|g| (g.name.as_str(), &g.strokes))
        .chain([
            ("skin", &guidance.skin_strokes),
            ("skin_cleanup", &guidance.skin_cleanup_strokes),
            ("brow", &guidance.brow_strokes),
        ])
    {
        for s in strokes {
            if s.points.is_empty() || !s.width_px.is_finite() || s.width_px < 1.0 {
                errors.push(format!(
                    "{what}: a stroke in {} needs at least one point and width_px ≥ 1",
                    s.view
                ));
            } else if s.width_px > 40.0 {
                warnings.push(format!(
                    "{what}: a stroke in {} is {:.0} px wide; use a polygon for wide items",
                    s.view, s.width_px
                ));
            }
        }
    }
    if !has_face_views {
        warnings.push(
            "no face_* views yet: run `charloom mh-guide-refine` and mark the eye openings".into(),
        );
    }
    let hair_in_head = guidance.garments.iter().any(|g| {
        g.label == Label::Hair
            && (g.regions.iter().any(|r| r.view.starts_with("head_"))
                || g.strokes.iter().any(|r| r.view.starts_with("head_")))
    });
    let any_hair = guidance.garments.iter().any(|g| g.label == Label::Hair);
    if any_hair && !hair_in_head {
        warnings.push(
            "hair is not marked in any head_* view: the head views decide the hairline".into(),
        );
    }

    // Left/right order and limb proportions, per view.
    let front = cameras
        .get(guidance.front_view.as_str())
        .map(|c| v(c.forward));
    let mut seen: BTreeMap<&str, BTreeMap<&str, [f64; 2]>> = BTreeMap::new();
    for (name, obs) in &guidance.landmarks {
        for o in obs {
            seen.entry(o.view.as_str())
                .or_default()
                .insert(name.as_str(), o.px);
        }
    }
    for (view, here) in &seen {
        let Some(cam) = cameras.get(view) else {
            continue;
        };
        let facing = front.map_or(0.0, |f| f.dot(&v(cam.forward)));
        if facing.abs() > 0.9 {
            for (l, r) in PAIRS {
                if let (Some(pl), Some(pr)) = (here.get(l), here.get(r)) {
                    // Front: left on the image's right. Back: the opposite.
                    if (pl[0] - pr[0]) * facing.signum() < 0.0 {
                        warnings.push(format!(
                            "{view}: {l} and {r} look swapped (the character's left is on the image's {} in this view)",
                            if facing > 0.0 { "right" } else { "left" }
                        ));
                    }
                }
            }
        }
        for side in ["l", "r"] {
            for (a, b, c, frac) in CHAINS {
                let get = |n: &str| here.get(format!("{n}_{side}").as_str()).copied();
                let (Some(pa), Some(pb), Some(pc)) = (get(a), get(b), get(c)) else {
                    continue;
                };
                let len = ((pc[0] - pa[0]).powi(2) + (pc[1] - pa[1]).powi(2)).sqrt();
                if len < 40.0 {
                    continue;
                }
                let e = [
                    pa[0] + frac * (pc[0] - pa[0]),
                    pa[1] + frac * (pc[1] - pa[1]),
                ];
                let off = ((pb[0] - e[0]).powi(2) + (pb[1] - e[1]).powi(2)).sqrt() / len;
                if off > 0.1 {
                    warnings.push(format!(
                        "{view}: {b}_{side} is {:.0} % of the limb away from its usual place ({:.0} % along {a}→{c}, near [{:.0}, {:.0}]); check it is not on a cuff, hem or belt",
                        off * 100.0,
                        frac * 100.0,
                        e[0],
                        e[1]
                    ));
                }
            }
            warnings.extend(hand_warnings(view, side, here));
        }
    }
    if let Some(list) = resolved.report["landmark_disagreements"].as_array() {
        for d in list {
            warnings.push(format!(
                "{}: its observations in {} land on the skin up to {:.1} mm apart; keep the one in the sharpest view (face_*/zoom_*) and drop the others",
                d["landmark"].as_str().unwrap_or("?"),
                d["views"],
                d["max_offset_mm"].as_f64().unwrap_or(0.0)
            ));
        }
    }
    // Finger segments in 3D: a mid knuckle that took the depth of another
    // finger in front of it makes one segment much longer than the other.
    let unit = unit_of(&scan);
    for side in ["l", "r"] {
        for f in 1..=5 {
            let p = |part: &str| {
                resolved
                    .hand_points
                    .get(&format!("finger{f}_{side}_{part}"))
                    .map(|p| v(*p))
            };
            if let (Some(b), Some(m), Some(t)) = (p("base"), p("mid"), p("tip")) {
                let (bm, mt) = ((m - b).norm(), (t - m).norm());
                let ratio = bm.max(mt) / bm.min(mt).max(1e-9);
                let bend = ((m - b).dot(&(t - m)) / (bm * mt).max(1e-18))
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees();
                if ratio > 2.5 || bend > 120.0 {
                    warnings.push(format!(
                        "finger{f}_{side}: base→mid {:.0} mm, mid→tip {:.0} mm (for a 1.7 m character), bent {bend:.0}° at the mid knuckle; one of the points probably took the depth of another finger in front of or behind it: mark it in a second view at least 15° away (a side zoom)",
                        bm / unit * 1000.0,
                        mt / unit * 1000.0
                    ));
                }
            }
        }
    }
    // Digits beyond `subject.hand_digits` are numbered past the scan's hand.
    let digits = guidance.subject.digits();
    for name in guidance.landmarks.keys().filter(|n| is_hand_landmark(n)) {
        let digit = name
            .strip_prefix("finger")
            .and_then(|r| r.split('_').next())
            .and_then(|f| f.parse::<u8>().ok());
        let side = usize::from(name.contains("_r_"));
        if let Some(f) = digit
            && f > digits[side]
            && !guidance.landmarks[name].is_empty()
        {
            warnings.push(format!(
                "{name}: the hand has {} digits (subject.hand_digits); number the scan's digits 1..{} from the thumb and leave finger{f} out",
                digits[side], digits[side]
            ));
        }
    }
    for name in guidance.landmarks.keys().filter(|n| is_hand_landmark(n)) {
        if !guidance.landmarks[name].is_empty() && !resolved.hand_points.contains_key(name) {
            warnings.push(format!(
                "{name}: no 3D point (no finger cross-section along its ray); mark it where the finger is seen across, not edge-on"
            ));
        }
    }

    // Overlays.
    let out_dir = dir.join("views").join("check");
    std::fs::create_dir_all(&out_dir)?;
    let mut legend = serde_json::Map::new();
    for (i, g) in guidance.garments.iter().enumerate() {
        let c = PALETTE[i % PALETTE.len()];
        legend.insert(
            g.name.clone(),
            json!(format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])),
        );
    }
    legend.insert("skin".into(), json!("#ffffff outline / band"));
    legend.insert("skin_cleanup".into(), json!("#000000 outline / band"));
    legend.insert(
        "grown seeds".into(),
        json!("dotted tint in the item's colour (white = skin): surface the seeds grew onto up to the colour edge"),
    );
    legend.insert("face points".into(), json!("#ff2020 cross"));
    legend.insert("body joints".into(), json!("#ffe000 cross"));
    legend.insert("rejected points".into(), json!("#ff00ff ring"));
    legend.insert(
        "ear outline".into(),
        json!("#ff9600: small crosses where marked; small rings joined by a thin closed line (front_top, top, back, lobe, front_bottom, tragus) where the resolved 3D points project and are visible"),
    );
    legend.insert(
        "lid margins".into(),
        json!("upper #00e0ff, lower #50ff50: small crosses where marked; small rings joined by a thin line (inner corner, 1, 2, 3, outer corner) where the resolved 3D points project and are visible"),
    );
    // Skin-coloured hair strands (strands.rs) as the pipeline would judge
    // them from these marks: magenta = becomes hair, cyan = a thin floating
    // layer kept as skin.
    let strand_flags = crate::strands::guide_flags(&scan, &resolved);
    legend.insert(
        "strands".into(),
        json!("magenta tint: skin-coloured strand the pipeline makes hair; cyan tint: thin floating skin it keeps (ear rim? strand fused to the face?)"),
    );
    // Per view: the overlay path and, per strand region, its mean pixel and
    // visible pixel count.
    type Seen = Vec<(u32, [f64; 2], usize)>;
    let written: Vec<Result<(String, Seen)>> = views
        .cameras
        .par_iter()
        .map(|cam| {
            let buffer = rasterize(cam, &scan.positions, &scan.triangles);
            let mut img = render(cam, &scan, &buffer, true);
            // Strand regions: tint, and visible pixels per region (mean px).
            let mut seen_px: BTreeMap<u32, ([f64; 2], usize)> = BTreeMap::new();
            for (k, &t) in buffer.id.iter().enumerate() {
                if t == u32::MAX {
                    continue;
                }
                let f = strand_flags.flag[t as usize];
                if f == 0 {
                    continue;
                }
                let (x, y) = ((k % cam.width) as i64, (k / cam.width) as i64);
                blend(
                    &mut img,
                    x,
                    y,
                    if f == 1 { [255, 0, 200] } else { [0, 220, 255] },
                    0.55,
                );
                let r = strand_flags.region_of[t as usize];
                if r != u32::MAX {
                    let e = seen_px.entry(r).or_insert(([0.0; 2], 0));
                    e.0[0] += x as f64;
                    e.0[1] += y as f64;
                    e.1 += 1;
                }
            }
            let seen_px: Vec<(u32, [f64; 2], usize)> = seen_px
                .into_iter()
                .map(|(r, (s, c))| (r, [s[0] / c as f64, s[1] / c as f64], c))
                .collect();
            // Grown seeds: dotted tint, so the polygons stay readable.
            for (k, &t) in buffer.id.iter().enumerate() {
                let (x, y) = ((k % cam.width) as i64, (k / cam.width) as i64);
                if t == u32::MAX || !resolved.seed_grown[t as usize] || (x + y) % 3 != 0 {
                    continue;
                }
                let c = match resolved.garment_seed[t as usize] {
                    Some((gi, _)) => PALETTE[gi % PALETTE.len()],
                    None => [255, 255, 255],
                };
                blend(&mut img, x, y, c, 0.7);
            }
            for (i, g) in guidance.garments.iter().enumerate() {
                for r in g.regions.iter().filter(|r| r.view == cam.id) {
                    draw_polygon(&mut img, &r.polygon, PALETTE[i % PALETTE.len()]);
                }
                for s in g.strokes.iter().filter(|s| s.view == cam.id) {
                    draw_stroke(&mut img, s, PALETTE[i % PALETTE.len()]);
                }
            }
            for (polys, strokes, c) in [
                (&guidance.skin, &guidance.skin_strokes, [255, 255, 255]),
                (
                    &guidance.skin_cleanup,
                    &guidance.skin_cleanup_strokes,
                    [0, 0, 0],
                ),
            ] {
                for r in polys.iter().filter(|r| r.view == cam.id) {
                    draw_polygon(&mut img, &r.polygon, c);
                }
                for s in strokes.iter().filter(|s| s.view == cam.id) {
                    draw_stroke(&mut img, s, c);
                }
            }
            for s in guidance.brow_strokes.iter().filter(|s| s.view == cam.id) {
                draw_stroke(&mut img, s, BROW_COLOUR);
            }
            for (name, obs) in &guidance.landmarks {
                let lid = lid_colour(name);
                let ear = is_ear_landmark(name);
                let c = if is_joint_landmark(name) {
                    [255, 224, 0]
                } else if ear {
                    EAR_COLOUR
                } else {
                    lid.unwrap_or([255, 32, 32])
                };
                for o in obs.iter().filter(|o| o.view == cam.id) {
                    if lid.is_some() || ear {
                        draw_small_cross(&mut img, o.px, c);
                    } else {
                        draw_cross(&mut img, o.px, c);
                    }
                }
            }
             
            let visible = |p: &Vec3| -> Option<[f64; 2]> {
                let (x, y, z) = cam.project(*p);
                let (xi, yi) = (x.floor(), y.floor());
                if xi < 0.0 || yi < 0.0 || xi >= cam.width as f64 || yi >= cam.height as f64 {
                    return None;
                }
                let k = yi as usize * cam.width + xi as usize;
                (z <= buffer.depth[k] as f64 + 0.002 * unit).then_some([x, y])
            };
            for side in ["l", "r"] {
                for part in ["upper", "lower"] {
                    let chain: Vec<String> = std::iter::once(format!("eye_{side}_inner"))
                        .chain((1..=3).map(|k| format!("eye_{side}_{part}_{k}")))
                        .chain(std::iter::once(format!("eye_{side}_outer")))
                        .collect();
                    if !chain[1..4]
                        .iter()
                        .any(|n| resolved.landmarks.contains_key(n))
                    {
                        continue;
                    }
                    let c = lid_colour(&chain[1]).unwrap_or([255, 255, 255]);
                    let at: Vec<Option<[f64; 2]>> = chain
                        .iter()
                        .map(|n| resolved.landmarks.get(n).and_then(&visible))
                        .collect();
                    for w in at.windows(2) {
                        if let (Some(p), Some(q)) = (w[0], w[1]) {
                            draw_thin_line(&mut img, p, q, c);
                        }
                    }
                    for p in at[1..4].iter().flatten() {
                        draw_dot_ring(&mut img, *p, c);
                    }
                }
            }
            // Resolved ear outlines, closed round the ear, where visible.
            for side in ["l", "r"] {
                let at: Vec<Option<[f64; 2]>> = EAR_OUTLINE
                    .iter()
                    .filter_map(|k| resolved.landmarks.get(&format!("ear_{side}_{k}")))
                    .map(&visible)
                    .collect();
                for (k, p) in at.iter().enumerate() {
                    if (at.len() > 2 || k + 1 < at.len())
                        && let (Some(p), Some(q)) = (p, at[(k + 1) % at.len()])
                    {
                        draw_thin_line(&mut img, *p, q, EAR_COLOUR);
                    }
                }
                for p in at.iter().flatten() {
                    draw_dot_ring(&mut img, *p, EAR_COLOUR);
                }
            }
            for (view, px) in &rejected {
                if *view == cam.id {
                    draw_ring(&mut img, *px, [255, 0, 255]);
                }
            }
            let path = out_dir.join(format!("{}.png", cam.id));
            img.save(&path)?;
            Ok((path.display().to_string(), seen_px))
        })
        .collect();
    let written = written.into_iter().collect::<Result<Vec<_>>>()?;
    // Each flagged strand where it is seen best (head/face views first).
    let mut best: BTreeMap<u32, (String, [f64; 2], usize)> = BTreeMap::new();
    for ((_, seen), cam) in written.iter().zip(&views.cameras) {
        let head = cam.id.starts_with("head_") || cam.id.starts_with("face_");
        for &(r, px, c) in seen {
            let c = if head { c * 4 } else { c };
            if best.get(&r).is_none_or(|b| c > b.2) {
                best.insert(r, (cam.id.clone(), px, c));
            }
        }
    }
    let describe = |hair: bool| -> (usize, Vec<String>) {
        let mut list: Vec<_> = best
            .iter()
            .filter(|(r, _)| strand_flags.regions[**r as usize].verdict.is_hair() == hair)
            .collect();
        // Longest first.
        list.sort_by(|a, b| {
            let len = |r: &u32| strand_flags.regions[*r as usize].length_m;
            len(b.0).total_cmp(&len(a.0))
        });
        let total = list.len();
        let shown = list
            .into_iter()
            .take(12)
            .map(|(r, (view, px, _))| {
                let g = &strand_flags.regions[*r as usize];
                format!(
                    "{view} [{:.0}, {:.0}] ({:.0} mm long, {})",
                    px[0],
                    px[1],
                    g.length_m / unit * 1000.0,
                    g.verdict.name()
                )
            })
            .collect();
        (total, shown)
    };
    let (total, to_hair) = describe(true);
    if total > 0 {
        warnings.push(format!(
            "{total} skin-coloured strand(s) will be relabelled hair (magenta in views/check; longest first, lengths for a 1.7 m character): {}; confirm each is a hair strand, and cover any that is real skin (an ear rim) with a skin_strokes stroke",
            to_hair.join("; ")
        ));
    }
    let (total, kept) = describe(false);
    if total > 0 {
        warnings.push(format!(
            "{total} thin floating skin layer(s) stay skin (cyan in views/check): {}; if one is a hair strand painted skin colour, mark it with a hair stroke",
            kept.join("; ")
        ));
    }
    let written: Vec<String> = written.into_iter().map(|(p, _)| p).collect();
    Ok(json!({
        "errors": errors,
        "warnings": warnings,
        "overlays": written,
        "legend": legend,
        "resolve": {
            "landmarks": resolved.report["landmarks"],
            "garment_seeds": resolved.report["garment_seeds"],
            "skin_seed_triangles": resolved.report["skin_seed_triangles"],
            "skin_grown_triangles": resolved.report["skin_grown_triangles"],
            "cleanup_triangles": resolved.report["cleanup_triangles"],
            "skin_over_garment_triangles": resolved.report["skin_over_garment_triangles"],
            "joint_section_depth": resolved.report["joint_section_depth"],
            "hand_points": resolved.report["hand_points"],
        },
    }))
}

// ---------------------------------------------------------------- resolve

/// Guidance mapped onto the working-frame scan.
pub struct Resolved {
    pub yaw: f64,
    /// Surface landmarks (snapped onto the skin) and joints seen in two or
    /// more views at least 15° apart (triangulated).
    pub landmarks: BTreeMap<String, Vec3>,
    /// Joints seen in one view: the viewing ray (origin, unit direction).
    /// Depth is left to the body fit — under loose clothing the first surface
    /// along the ray is cloth, not the limb.
    pub joint_rays: BTreeMap<String, (Vec3, Vec3)>,
    /// Single-view (or low-confidence) joints: the centre of the limb's
    /// cross-section along the ray (body layer when one is found under the
    /// marked cloth). Reported; the pose fit still uses the ray.
    pub joint_section: BTreeMap<String, Vec3>,
    /// Hand landmarks (`finger{1..5}_{l,r}_{base,mid,tip}`, `palm_{l,r}`) as
    /// 3D points: triangulated when seen in two views ≥ 15° apart, else the
    /// centre of the finger's (palm's) cross-section along the viewing ray.
    pub hand_points: BTreeMap<String, Vec3>,
    /// Marked body outlines (`outline_*`), pairs in one view.
    pub outlines: Vec<OutlineMark>,
    /// Mean effective observation confidence per landmark.
    pub confidence: BTreeMap<String, f64>,
    /// Per scan triangle: (garment index into `garments`, depth inside polygon in px).
    /// Includes stroke seeds and grown seeds (depth 0).
    pub garment_seed: Vec<Option<(usize, f32)>>,
    /// Per scan triangle: depth inside the skin polygon in px, where skin wins.
    pub skin_seed: Vec<Option<f32>>,
    /// Per scan triangle: its seed (garment or skin) forces its label: deep
    /// inside a polygon, on a stroke, or colour-consistent with its item.
    pub seed_strong: Vec<bool>,
         pub seed_grown: Vec<bool>,
    /// Per scan triangle: seeded by a stroke (garment or skin), which forces
    /// its label even where the neckline refinement would move it.
    pub seed_stroke: Vec<bool>,
    pub cleanup: Vec<bool>,
    /// Per scan triangle: inside a marked brow (`brow_strokes`).
    pub brow: Vec<bool>,
    /// Per scan triangle: visible in at least one view the guidance uses.
    pub seen: Vec<bool>,
    pub garments: Vec<GarmentGuide>,
    pub report: Value,
}

/// Outline pairs from the guidance: both edges of a station and kind,
/// marked in the same view (the first observation of each). Returns the
/// marks and the problems (a lone edge, edges in different views).
pub fn resolve_outlines(
    guidance: &Guidance,
    cameras: &BTreeMap<String, Camera>,
) -> (Vec<OutlineMark>, Vec<String>) {
    let mut out = vec![];
    let mut problems = vec![];
    let stations: std::collections::BTreeSet<&str> = guidance
        .landmarks
        .keys()
        .filter_map(|n| outline_parts(n).map(|p| p.0))
        .collect();
    for station in stations {
        for kind in ['w', 'd'] {
            let get = |i: u8| {
                guidance
                    .landmarks
                    .get(&format!("outline_{station}_{kind}{i}"))
                    .and_then(|o| o.first())
            };
            match (get(1), get(2)) {
                (None, None) => {}
                (Some(p), Some(q)) => {
                    if p.view != q.view {
                        problems.push(format!(
                            "outline_{station}_{kind}1/2: marked in different views ({}, {}); mark both edges in one view",
                            p.view, q.view
                        ));
                        continue;
                    }
                    let Some(cam) = cameras.get(&p.view) else {
                        continue;
                    };
                    let (a0, d) = cam.ray(p.px[0], p.px[1]);
                    let (b0, _) = cam.ray(q.px[0], q.px[1]);
                    out.push(OutlineMark {
                        station: station.to_string(),
                        kind,
                        view: p.view.clone(),
                        a: crate::geom::a(a0),
                        b: crate::geom::a(b0),
                        dir: crate::geom::a(d),
                        confidence: 0.5 * (p.effective_confidence() + q.effective_confidence()),
                    });
                }
                _ => problems.push(format!(
                    "outline_{station}_{kind}1/2: only one edge marked; mark both"
                )),
            }
        }
    }
    (out, problems)
}

pub fn load(path: &Path) -> Result<(Guidance, ViewSet)> {
    let guidance: Guidance = serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )
    .context("parse guidance")?;
    ensure!(
        guidance.schema == SCHEMA,
        "guidance schema must be {SCHEMA}"
    );
    ensure!(
        guidance
            .subject
            .digits()
            .iter()
            .all(|n| (2..=8).contains(n)),
        "guidance subject.hand_digits must be 2–8 per hand (thumb counted)"
    );
    let views_path = path.parent().unwrap_or(Path::new(".")).join("views.json");
    let views: ViewSet =
        serde_json::from_slice(&std::fs::read(&views_path).with_context(|| {
            format!(
                "read {} (guidance must sit next to the views it annotates)",
                views_path.display()
            )
        })?)?;
    ensure!(
        views.source_hash == guidance.source_hash,
        "guidance_source_mismatch: views and guidance are for different files"
    );
    Ok((guidance, views))
}

/// Yaw (radians) that turns the guidance's front view into a +Z-facing model.
pub fn facing_yaw(guidance: &Guidance) -> Result<f64> {
    let deg: f64 = guidance
        .front_view
        .strip_prefix("body_y")
        .and_then(|d| d.parse().ok())
        .context("front_view must be one of body_y000/body_y090/body_y180/body_y270")?;
    // Camera at yaw θ sees the side facing (sin θ, 0, cos θ); rotate it to +Z.
    Ok(-deg.to_radians())
}

fn inside(poly: &[[f64; 2]], x: f64, y: f64) -> bool {
    let mut c = false;
    let n = poly.len();
    for i in 0..n {
        let (a0, b0) = (poly[i], poly[(i + n - 1) % n]);
        if (a0[1] > y) != (b0[1] > y) && x < (b0[0] - a0[0]) * (y - a0[1]) / (b0[1] - a0[1]) + a0[0]
        {
            c = !c;
        }
    }
    c
}

fn segment_distance(p: [f64; 2], q: [f64; 2], x: f64, y: f64) -> f64 {
    let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
    let t = (((x - p[0]) * dx + (y - p[1]) * dy) / (dx * dx + dy * dy).max(1e-12)).clamp(0.0, 1.0);
    ((x - p[0] - t * dx).powi(2) + (y - p[1] - t * dy).powi(2)).sqrt()
}

fn edge_distance(poly: &[[f64; 2]], x: f64, y: f64) -> f64 {
    let n = poly.len();
    (0..n)
        .map(|i| segment_distance(poly[i], poly[(i + 1) % n], x, y))
        .fold(f64::INFINITY, f64::min)
}

/// Distance from a pixel to a stroke's centre line (a single point is a dot).
fn polyline_distance(points: &[[f64; 2]], x: f64, y: f64) -> f64 {
    match points {
        [] => f64::INFINITY,
        [p] => ((x - p[0]).powi(2) + (y - p[1]).powi(2)).sqrt(),
        _ => points
            .windows(2)
            .map(|w| segment_distance(w[0], w[1], x, y))
            .fold(f64::INFINITY, f64::min),
    }
}

fn stroke_bounds(s: &Stroke) -> Option<([f64; 2], [f64; 2])> {
    if s.points.is_empty() {
        return None;
    }
    let half = s.width_px.max(1.0) * 0.5;
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in &s.points {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k] - half);
            hi[k] = hi[k].max(p[k] + half);
        }
    }
    Some((lo, hi))
}

/// Visible triangles of the pixels inside `[lo, hi]` for which `depth` (px
/// inside the region) is `Some`, each with its largest depth.
fn region_triangles(
    cam: &Camera,
    buffer: &IdBuffer,
    lo: [f64; 2],
    hi: [f64; 2],
    depth: impl Fn(f64, f64) -> Option<f64>,
) -> Vec<(u32, f32)> {
    let (w, h) = (cam.width, cam.height);
    if !(lo[0].is_finite() && lo[1].is_finite() && hi[0].is_finite() && hi[1].is_finite())
        || hi[0] < 0.0
        || hi[1] < 0.0
    {
        return vec![];
    }
    let x0 = lo[0].floor().max(0.0) as usize;
    let x1 = (hi[0].ceil().max(0.0) as usize).min(w - 1);
    let y0 = lo[1].floor().max(0.0) as usize;
    let y1 = (hi[1].ceil().max(0.0) as usize).min(h - 1);
    let mut best: BTreeMap<u32, f32> = BTreeMap::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            let t = buffer.id[y * w + x];
            if t == u32::MAX {
                continue;
            }
            if let Some(d) = depth(x as f64 + 0.5, y as f64 + 0.5) {
                let e = best.entry(t).or_insert(0.0);
                *e = e.max(d as f32);
            }
        }
    }
    best.into_iter().collect()
}

/// Visible triangles inside a polygon with each one's distance to the edge.
fn polygon_triangles(cam: &Camera, buffer: &IdBuffer, poly: &[[f64; 2]]) -> Vec<(u32, f32)> {
    if poly.len() < 3 {
        return vec![];
    }
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in poly {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    region_triangles(cam, buffer, lo, hi, |x, y| {
        inside(poly, x, y).then(|| edge_distance(poly, x, y))
    })
}

/// Visible triangles under a stroke's band, with their distance inside the
/// band's edge (px).
fn stroke_triangles(cam: &Camera, buffer: &IdBuffer, s: &Stroke) -> Vec<(u32, f32)> {
    let Some((lo, hi)) = stroke_bounds(s) else {
        return vec![];
    };
    if !s.width_px.is_finite() || s.width_px < 1.0 {
        return vec![];
    }
    let half = s.width_px * 0.5;
    region_triangles(cam, buffer, lo, hi, |x, y| {
        let d = polyline_distance(&s.points, x, y);
        (d <= half).then_some(half - d)
    })
}

fn stroke_area_px(s: &Stroke) -> f64 {
    let length: f64 = s
        .points
        .windows(2)
        .map(|w| ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt())
        .sum();
    (length + s.width_px) * s.width_px
}

/// Length unit for geometric tolerances: they are given for a 1.7 m tall
/// character and scale with the scan (the check runs in source units).
fn unit_of(scan: &Scan) -> f64 {
    (scan.height() / 1.7).max(1e-9)
}

// ------------------------------------------------- multi-layer ray queries

/// One surface crossing along a pixel's viewing ray.
#[derive(Debug, Clone, Copy)]
struct Hit {
    /// Camera depth (along `forward`).
    depth: f64,
    tri: u32,
    /// Negative: the triangle's winding faces the camera (entering a volume).
    facing: f64,
}

 struct Layers {
    x0: i64,
    y0: i64,
    w: i64,
    h: i64,
    hits: Vec<Vec<Hit>>,
}

impl Layers {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cam: &Camera,
        projected: &[(f64, f64, f64)],
        positions: &[Vec3],
        triangles: &[[u32; 3]],
        x0: i64,
        y0: i64,
        x1: i64,
        y1: i64,
    ) -> Layers {
        let (x0, y0) = (x0.max(0), y0.max(0));
        let (x1, y1) = (x1.min(cam.width as i64 - 1), y1.min(cam.height as i64 - 1));
        let (w, h) = ((x1 - x0 + 1).max(0), (y1 - y0 + 1).max(0));
        let mut hits = vec![vec![]; (w * h) as usize];
        if w == 0 || h == 0 {
            return Layers { x0, y0, w, h, hits };
        }
        let forward = v(cam.forward);
        let (fx0, fx1, fy0, fy1) = (x0 as f64, (x1 + 1) as f64, y0 as f64, (y1 + 1) as f64);
        for (t, tri) in triangles.iter().enumerate() {
            let [p0, p1, p2] = tri.map(|i| projected[i as usize]);
            if p0.0.max(p1.0).max(p2.0) < fx0
                || p0.0.min(p1.0).min(p2.0) > fx1
                || p0.1.max(p1.1).max(p2.1) < fy0
                || p0.1.min(p1.1).min(p2.1) > fy1
            {
                continue;
            }
            let area = (p1.0 - p0.0) * (p2.1 - p0.1) - (p2.0 - p0.0) * (p1.1 - p0.1);
            if area.abs() < 1e-12 {
                continue;
            }
            let [q0, q1, q2] = tri.map(|i| v(positions[i as usize]));
            let facing = (q1 - q0).cross(&(q2 - q0)).dot(&forward).signum();
            let bx0 = (p0.0.min(p1.0).min(p2.0).floor() as i64).max(x0);
            let bx1 = (p0.0.max(p1.0).max(p2.0).ceil() as i64).min(x1);
            let by0 = (p0.1.min(p1.1).min(p2.1).floor() as i64).max(y0);
            let by1 = (p0.1.max(p1.1).max(p2.1).ceil() as i64).min(y1);
            for y in by0..=by1 {
                for x in bx0..=bx1 {
                    let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                    let b1 = ((px - p0.0) * (p2.1 - p0.1) - (p2.0 - p0.0) * (py - p0.1)) / area;
                    let b2 = ((p1.0 - p0.0) * (py - p0.1) - (px - p0.0) * (p1.1 - p0.1)) / area;
                    let b0 = 1.0 - b1 - b2;
                    if b0 < -1e-9 || b1 < -1e-9 || b2 < -1e-9 {
                        continue;
                    }
                    hits[((y - y0) * w + (x - x0)) as usize].push(Hit {
                        depth: p0.2 * b0 + p1.2 * b1 + p2.2 * b2,
                        tri: t as u32,
                        facing,
                    });
                }
            }
        }
        for list in &mut hits {
            list.sort_by(|a, b| a.depth.total_cmp(&b.depth));
            // A pixel centre on a shared edge hits both triangles: one crossing.
            list.dedup_by(|b, a| (b.depth - a.depth).abs() < 1e-9 && a.facing == b.facing);
        }
        Layers { x0, y0, w, h, hits }
    }

    fn at(&self, x: i64, y: i64) -> &[Hit] {
        if x < self.x0 || y < self.y0 || x >= self.x0 + self.w || y >= self.y0 + self.h {
            return &[];
        }
        &self.hits[((y - self.y0) * self.w + (x - self.x0)) as usize]
    }
}

/// A surface landmark placed on the skin.
#[derive(Debug, Clone)]
struct Snap {
    point: V,
    tri: u32,
    /// Layers walked past at the chosen pixel (strands, marked garments).
    skipped: usize,
    /// Camera depth from the first surface to the chosen one.
    behind_first: f64,
    /// Distance from the marked pixel to the pixel used.
    moved_px: f64,
         first_occluder: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
enum SnapError {
    Misses,
    /// No skin within the search radius at a plausible depth.
    NoSkin {
        first_occluder: Option<u32>,
        /// How much nearer than the surrounding skin the first surface is.
        nearer: Option<f64>,
    },
}

/// Thresholds for the surface snap, in metres of a 1.7 m character.
const SNAP_RING_M: f64 = 0.006;
const SNAP_SEARCH_M: f64 = 0.003;
const SNAP_STRAND_M: f64 = 0.008;
const SNAP_DEEP_M: f64 = 0.012;
/// Snapped observations of one surface landmark further apart than this
/// from their mean are reported (the views disagree about the spot).
const SURFACE_AGREE_M: f64 = 0.003;

 #[allow(clippy::too_many_arguments)]
fn snap_surface(
    cam: &Camera,
    projected: &[(f64, f64, f64)],
    positions: &[Vec3],
    triangles: &[[u32; 3]],
    px: [f64; 2],
    occluder: &dyn Fn(u32) -> bool,
    unit: f64,
    strand_test: bool,
) -> Result<Snap, SnapError> {
    let mpp = cam.meters_per_pixel;
    let ring = ((SNAP_RING_M * unit / mpp).ceil() as i64).clamp(2, 48);
    let search = (SNAP_SEARCH_M * unit / mpp).min(24.0);
    let (cx, cy) = (px[0].floor() as i64, px[1].floor() as i64);
    if cx < 0 || cy < 0 || cx >= cam.width as i64 || cy >= cam.height as i64 {
        return Err(SnapError::Misses);
    }
    let layers = Layers::new(
        cam,
        projected,
        positions,
        triangles,
        cx - ring,
        cy - ring,
        cx + ring,
        cy + ring,
    );
    let centre = layers.at(cx, cy);
    if centre.is_empty() {
        return Err(SnapError::Misses);
    }
    // Skin depth around the point: the first unmarked layer on a ring.
    let mut around: Vec<f64> = vec![];
    for dy in -ring..=ring {
        for dx in -ring..=ring {
            let d2 = dx * dx + dy * dy;
            if d2 * 4 < ring * ring || d2 > ring * ring {
                continue;
            }
            if let Some(h) = layers
                .at(cx + dx, cy + dy)
                .iter()
                .find(|h| !occluder(h.tri))
            {
                around.push(h.depth);
            }
        }
    }
    let reference = (around.len() >= 8).then(|| {
        around.sort_by(f64::total_cmp);
        around[around.len() / 2]
    });
    let accept = |hits: &[Hit]| -> Option<usize> {
        // Facing of the outside of the model (the first layer faces the camera).
        let outside = hits.first()?.facing;
        let mut walked = false;
        for (i, h) in hits.iter().enumerate() {
            if occluder(h.tri) {
                walked = true;
                continue;
            }
            // The far side of a layer walked past is not skin.
            if walked && h.facing != outside {
                continue;
            }
            if let Some(r) = reference {
                if strand_test && h.depth < r - SNAP_STRAND_M * unit {
                    walked = true;
                    continue;
                }
                if h.depth > r + SNAP_DEEP_M * unit {
                    return None;
                }
            }
            return Some(i);
        }
        None
    };
    let first_occluder = occluder(centre[0].tri).then_some(centre[0].tri);
    let point_at = |tri: u32, x: f64, y: f64| {
        let t = triangles[tri as usize];
        let b = pixel_bary(cam, positions, t, x, y);
        (0..3).fold(V::zeros(), |s, j| s + v(positions[t[j] as usize]) * b[j])
    };
    if let Some(i) = accept(centre) {
        return Ok(Snap {
            point: point_at(centre[i].tri, px[0], px[1]),
            tri: centre[i].tri,
            skipped: i,
            behind_first: centre[i].depth - centre[0].depth,
            moved_px: 0.0,
            first_occluder,
        });
    }
    // Nearest pixel within the search radius that has skin.
    let r = search.ceil() as i64;
    let mut candidates: Vec<(f64, i64, i64)> = vec![];
    for dy in -r..=r {
        for dx in -r..=r {
            let (x, y) = (cx + dx, cy + dy);
            let d = ((x as f64 + 0.5 - px[0]).powi(2) + (y as f64 + 0.5 - px[1]).powi(2)).sqrt();
            if (dx, dy) != (0, 0) && d <= search.max(1.0) {
                candidates.push((d, x, y));
            }
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (d, x, y) in candidates {
        let hits = layers.at(x, y);
        if let Some(i) = accept(hits) {
            return Ok(Snap {
                point: point_at(hits[i].tri, x as f64 + 0.5, y as f64 + 0.5),
                tri: hits[i].tri,
                skipped: i,
                behind_first: hits[i].depth - centre[0].depth,
                moved_px: d,
                first_occluder,
            });
        }
    }
    Err(SnapError::NoSkin {
        first_occluder,
        nearer: reference.map(|r| r - centre[0].depth),
    })
}

/// Cross-section of a limb along one viewing ray.
#[derive(Debug, Clone, Serialize)]
struct Section {
    /// Camera depth of the first surface along the ray.
    first: f64,
    near: f64,
    far: f64,
    /// Surface crossings inside the chosen outer interval.
    crossings: usize,
    /// The interval is a body layer found under marked cloth.
    body_layer: bool,
}

impl Section {
    fn centre(&self) -> f64 {
        (self.near + self.far) * 0.5
    }
}

/// Paired ray intervals (as `visual_fit`): the first closed interval along
/// the ray — nesting by the crossings' facing, the first two crossings when
/// the winding is unusable — and, inside it, the body layer under the cloth:
/// a crossing pair on unmarked surface inside an interval whose outside is
/// marked hair/garment. An interval thinner than the limb can be (`thin`:
/// a robe flap, a sleeve sheet or a hand in front of it) is skipped when
/// more surface follows.
fn section_probe(hits: &[Hit], cloth: &dyn Fn(u32) -> bool, thin: f64) -> Option<Section> {
    let mut start = 0;
    while start + 1 < hits.len() {
        let rest = &hits[start..];
        let mut end = None;
        if rest[0].facing < 0.0 {
            let mut depth = 0i32;
            for (i, h) in rest.iter().enumerate() {
                depth += if h.facing < 0.0 { 1 } else { -1 };
                if depth <= 0 {
                    end = Some(i);
                    break;
                }
            }
        }
        let end = end.unwrap_or(1);
        let comp = &rest[..=end];
        let (near, far) = (comp[0].depth, comp[end].depth);
        if far - near < thin && start + end + 2 < hits.len() {
            start += end + 1;
            continue;
        }
        let mut body = None;
        if comp.iter().any(|h| cloth(h.tri)) {
            let inner: Vec<&Hit> = comp[1..].iter().filter(|h| !cloth(h.tri)).collect();
            if let Some(i) = inner.iter().position(|h| h.facing < 0.0)
                && let Some(j) = inner[i + 1..].iter().position(|h| h.facing > 0.0)
            {
                body = Some((inner[i].depth, inner[i + 1 + j].depth));
            }
        }
        return Some(match body {
            Some((n, f)) => Section {
                first: hits[0].depth,
                near: n,
                far: f,
                crossings: comp.len(),
                body_layer: true,
            },
            None => Section {
                first: hits[0].depth,
                near,
                far,
                crossings: comp.len(),
                body_layer: false,
            },
        });
    }
    None
}

// ------------------------------------------------------------ seed growth

/// Colour of an item's seeds: median CIELAB and the radius that holds most
/// of its samples.
#[derive(Debug, Clone, Copy)]
struct ColourModel {
    median: [f64; 3],
    tolerance: f64,
}

/// CIELAB distance with lightness down-weighted, as in `segment` (shading).
fn lab_distance(c: &[f64; 3], m: &[f64; 3]) -> f64 {
    ((c[0] - m[0]).powi(2) * 0.5 + (c[1] - m[1]).powi(2) + (c[2] - m[2]).powi(2)).sqrt()
}

impl ColourModel {
    fn from_samples(samples: &[[f64; 3]]) -> Option<ColourModel> {
        if samples.len() < 12 {
            return None;
        }
        let med = |k: usize| {
            let mut xs: Vec<f64> = samples.iter().map(|c| c[k]).collect();
            xs.sort_by(f64::total_cmp);
            xs[xs.len() / 2]
        };
        let median = [med(0), med(1), med(2)];
        let mut d: Vec<f64> = samples.iter().map(|c| lab_distance(c, &median)).collect();
        d.sort_by(f64::total_cmp);
        let p90 = d[(d.len() * 9) / 10];
        Some(ColourModel {
            median,
            tolerance: p90.clamp(GROW_MIN_TOLERANCE, GROW_MAX_TOLERANCE),
        })
    }
    /// Lightness counts half again: baked shading and occlusion darken an
    /// item (the inside of an arm) without changing its hue.
    fn distance(&self, c: &[f64; 3]) -> f64 {
        let m = &self.median;
        ((c[0] - m[0]).powi(2) * 0.25 + (c[1] - m[1]).powi(2) + (c[2] - m[2]).powi(2)).sqrt()
    }
    fn accepts(&self, c: &[f64; 3]) -> bool {
        self.distance(c) <= self.tolerance
    }
}

/// A garment polygon's triangle that shows bare skin: near the skin colour
/// and much nearer it than the item's (whose tolerance can be wide enough
/// to take the skin too: a red tunic polygon over a tanned neck).
fn seed_on_skin(item: &ColourModel, skin: &ColourModel, c: &[f64; 3]) -> bool {
    let (di, ds) = (item.distance(c), skin.distance(c));
    ds <= skin.tolerance.max(12.0) && ds < 0.5 * di
}

/// Seed growth limits: px of the seed's view, colour step between
/// neighbouring triangles, and the item colour radius bounds (ΔE).
const GROW_PX: f64 = 24.0;
const GROW_STEP: f64 = 12.0;
const GROW_MIN_TOLERANCE: f64 = 8.0;
const GROW_MAX_TOLERANCE: f64 = 28.0;

/// Triangles around each welded vertex (CSR).
struct Incident {
    start: Vec<u32>,
    tris: Vec<u32>,
}

impl Incident {
    fn new(triangles: &[[u32; 3]], weld_id: &[usize], count: usize) -> Incident {
        let mut start = vec![0u32; count + 1];
        for tri in triangles {
            for &i in tri {
                start[weld_id[i as usize] + 1] += 1;
            }
        }
        for k in 0..count {
            start[k + 1] += start[k];
        }
        let mut fill = start.clone();
        let mut tris = vec![0u32; start[count] as usize];
        for (t, tri) in triangles.iter().enumerate() {
            for &i in tri {
                let w = weld_id[i as usize];
                tris[fill[w] as usize] = t as u32;
                fill[w] += 1;
            }
        }
        Incident { start, tris }
    }
    fn around(&self, w: usize) -> &[u32] {
        &self.tris[self.start[w] as usize..self.start[w + 1] as usize]
    }
}

/// Grows seed regions over connected surface into unseeded triangles, so a
/// polygon drawn a little short of an item still covers it up to its edge.
/// Every owner grows at once (Dijkstra on surface distance), so skin and
/// garments compete for what lies between them; a front stops at a colour
/// edge (a step above `GROW_STEP` between neighbours, or a colour outside
/// its item's model; for a garment, clearly bare skin) and after `budget`
/// metres from its seed. `models` ends with the skin's. `owner` holds
/// the seeds (`None` = unseeded); only triangles in `sources` start a front.
/// Returns the grown triangles.
#[allow(clippy::too_many_arguments)]
fn grow_seeds(
    triangles: &[[u32; 3]],
    weld_id: &[usize],
    incident: &Incident,
    lab: &[Option<[f64; 3]>],
    centroid: &[V],
    owner: &mut [Option<usize>],
    sources: &[bool],
    budget: &[f64],
    models: &[Option<ColourModel>],
) -> Vec<bool> {
    // The skin is the last owner.
    let skin_owner = models.len() - 1;
    let n = triangles.len();
    let mut cost = vec![f64::INFINITY; n];
    let mut limit = vec![0.0f64; n];
    let mut grown = vec![false; n];
    let mut heap = std::collections::BinaryHeap::new();
    for t in 0..n {
        if let Some(o) = owner[t]
            && sources[t]
            && models.get(o).is_some_and(|m| m.is_some())
            && budget[t] > 0.0
        {
            cost[t] = 0.0;
            limit[t] = budget[t];
            heap.push((std::cmp::Reverse(OrderedF64(0.0)), t));
        }
    }
    while let Some((std::cmp::Reverse(OrderedF64(c)), t)) = heap.pop() {
        if c > cost[t] {
            continue;
        }
        let Some(o) = owner[t] else { continue };
        let Some(model) = models[o] else { continue };
        let Some(lt) = lab[t] else { continue };
        for &i in &triangles[t] {
            for &u in incident.around(weld_id[i as usize]) {
                let u = u as usize;
                if u == t || (owner[u].is_some() && !grown[u]) {
                    continue;
                }
                let Some(lu) = lab[u] else { continue };
                let next = c + (centroid[u] - centroid[t]).norm();
                // A garment front does not grow onto clearly bare skin.
                let onto_skin = o != skin_owner
                    && models[skin_owner].is_some_and(|s| seed_on_skin(&model, &s, &lu));
                if next > limit[t]
                    || next >= cost[u]
                    || lab_distance(&lt, &lu) > GROW_STEP
                    || !model.accepts(&lu)
                    || onto_skin
                {
                    continue;
                }
                cost[u] = next;
                limit[u] = limit[t];
                owner[u] = Some(o);
                grown[u] = true;
                heap.push((std::cmp::Reverse(OrderedF64(next)), u));
            }
        }
    }
    grown
}

/// Thinnest plausible cross-section through a joint (metres, 1.7 m
/// character): thinner intervals along its ray are something in front of it.
fn min_limb_thickness(name: &str) -> f64 {
    if name.starts_with("palm") {
        0.01
    } else if is_hand_landmark(name) {
        0.005
    } else if name.starts_with("wrist") {
        0.025
    } else if name.starts_with("elbow") || name.starts_with("ankle") {
        0.04
    } else {
        0.06
    }
}

/// Total order for non-NaN f64 heap keys.
#[derive(PartialEq, Clone, Copy)]
struct OrderedF64(f64);
impl Eq for OrderedF64 {}
impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Mean colour of a triangle (centroid and corners) in CIELAB.
fn triangle_lab(scan: &Scan, t: usize) -> Option<[f64; 3]> {
    let third = 1.0 / 3.0;
    let mut sum = [0.0; 3];
    for b in [
        [third, third, third],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ] {
        let c = scan.color_at(t, b)?;
        for k in 0..3 {
            sum[k] += c[k] * 0.25;
        }
    }
    Some(crate::source::lab(sum))
}

fn max_angle_deg(dirs: &[V]) -> f64 {
    let mut best = 0.0f64;
    for i in 0..dirs.len() {
        for j in i + 1..dirs.len() {
            best = best.max(dirs[i].dot(&dirs[j]).clamp(-1.0, 1.0).acos().to_degrees());
        }
    }
    best
}

/// Least-squares intersection of weighted viewing rays.
fn triangulate(rays: &[(V, V, f64)]) -> Option<V> {
    let mut m = nalgebra::Matrix3::zeros();
    let mut r = V::zeros();
    for (o, d, w) in rays {
        let proj = (nalgebra::Matrix3::identity() - d * d.transpose()) * w.max(1e-3);
        m += proj;
        r += proj * o;
    }
    m.try_inverse().map(|inv| inv * r)
}

 pub fn resolve(guidance: &Guidance, views: &ViewSet, scan: &Scan) -> Result<Resolved> {
    let cameras: BTreeMap<String, Camera> = views
        .cameras
        .iter()
        .map(|c| (c.id.clone(), c.transformed(&scan.transform)))
        .collect();
    let mut needed: Vec<&str> = guidance
        .landmarks
        .values()
        .flatten()
        .map(|p| p.view.as_str())
        .collect();
    needed.extend(guidance.garments.iter().flat_map(|g| {
        g.regions
            .iter()
            .map(|r| r.view.as_str())
            .chain(g.strokes.iter().map(|s| s.view.as_str()))
    }));
    needed.extend(
        guidance
            .skin
            .iter()
            .chain(&guidance.skin_cleanup)
            .map(|r| r.view.as_str()),
    );
    needed.extend(
        guidance
            .skin_strokes
            .iter()
            .chain(&guidance.skin_cleanup_strokes)
            .chain(&guidance.brow_strokes)
            .map(|s| s.view.as_str()),
    );
    needed.sort_unstable();
    needed.dedup();
    for n in &needed {
        ensure!(
            cameras.contains_key(*n),
            "guidance refers to unknown view {n}"
        );
    }
    let buffers: BTreeMap<&str, IdBuffer> = needed
        .par_iter()
        .map(|n| {
            (
                *n,
                rasterize(&cameras[*n], &scan.positions, &scan.triangles),
            )
        })
        .collect();
    let unit = unit_of(scan);
    let mut report = serde_json::Map::new();

    // ---- Regions: polygon and stroke seeds.
    let n = scan.triangles.len();
    let mut garment_seed: Vec<Option<(usize, f32)>> = vec![None; n];
    // Where polygons overlap, the smaller one wins: a hairpin drawn inside the
    // hair, a belt drawn over the robe, a face drawn inside a hair outline
    // that the LLM drew around the whole head. Areas are compared in m², as
    // the views have different scales. Strokes compete with their band area.
    let area = |r: &Polygon| {
        let poly = &r.polygon;
        let px = (0..poly.len())
            .map(|i| {
                let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
                p[0] * q[1] - q[0] * p[1]
            })
            .sum::<f64>()
            .abs()
            * 0.5;
        px * cameras[&r.view].meters_per_pixel.powi(2)
    };
    let stroke_area = |s: &Stroke| stroke_area_px(s) * cameras[&s.view].meters_per_pixel.powi(2);
    // Per triangle: winning area, metres per pixel of its view, from a stroke.
    let mut seed_area = vec![f64::INFINITY; n];
    let mut seed_mpp = vec![0.0f64; n];
    let mut seed_stroke = vec![false; n];
    let mut hits_per_garment = vec![0usize; guidance.garments.len()];
    for (gi, g) in guidance.garments.iter().enumerate() {
        let polys = g.regions.iter().map(|r| {
            (
                area(r),
                &r.view,
                false,
                polygon_triangles(&cameras[&r.view], &buffers[r.view.as_str()], &r.polygon),
            )
        });
        let strokes = g.strokes.iter().map(|s| {
            (
                stroke_area(s),
                &s.view,
                true,
                stroke_triangles(&cameras[&s.view], &buffers[s.view.as_str()], s),
            )
        });
        for (a0, view, is_stroke, tris) in polys.chain(strokes).collect::<Vec<_>>() {
            for (t, depth) in tris {
                let t = t as usize;
                if a0 < seed_area[t]
                    || (a0 == seed_area[t] && garment_seed[t].is_none_or(|(_, d)| depth > d))
                {
                    garment_seed[t] = Some((gi, depth));
                    seed_area[t] = a0;
                    seed_mpp[t] = cameras[view].meters_per_pixel;
                    seed_stroke[t] = is_stroke;
                    hits_per_garment[gi] += 1;
                }
            }
        }
    }
    let mut skin_area = vec![f64::INFINITY; n];
    let mut skin_depth = vec![0.0f32; n];
    let mut skin_mpp = vec![0.0f64; n];
    let mut skin_stroke = vec![false; n];
    let skin_regions = guidance
        .skin
        .iter()
        .map(|r| {
            (
                area(r),
                &r.view,
                false,
                polygon_triangles(&cameras[&r.view], &buffers[r.view.as_str()], &r.polygon),
            )
        })
        .chain(guidance.skin_strokes.iter().map(|s| {
            (
                stroke_area(s),
                &s.view,
                true,
                stroke_triangles(&cameras[&s.view], &buffers[s.view.as_str()], s),
            )
        }))
        .collect::<Vec<_>>();
    for (a0, view, is_stroke, tris) in skin_regions {
        for (t, depth) in tris {
            let t = t as usize;
            if a0 < skin_area[t] {
                skin_mpp[t] = cameras[view].meters_per_pixel;
                skin_stroke[t] = is_stroke;
            }
            skin_area[t] = skin_area[t].min(a0);
            skin_depth[t] = skin_depth[t].max(depth);
        }
    }
    let mut skin_over_garment = 0usize;
    let mut skin_seed: Vec<Option<f32>> = (0..n)
        .map(|t| {
            let wins = skin_area[t] < seed_area[t];
            if wins && garment_seed[t].take().is_some() {
                skin_over_garment += 1;
            }
            wins.then_some(skin_depth[t])
        })
        .collect();
    let mut cleanup = vec![false; n];
    for r in &guidance.skin_cleanup {
        for (t, _) in polygon_triangles(&cameras[&r.view], &buffers[r.view.as_str()], &r.polygon) {
            cleanup[t as usize] = true;
        }
    }
    for s in &guidance.skin_cleanup_strokes {
        for (t, _) in stroke_triangles(&cameras[&s.view], &buffers[s.view.as_str()], s) {
            cleanup[t as usize] = true;
        }
    }
    let mut brow = vec![false; n];
    for s in &guidance.brow_strokes {
        for (t, _) in stroke_triangles(&cameras[&s.view], &buffers[s.view.as_str()], s) {
            brow[t as usize] = true;
        }
    }
    report.insert(
        "brow_triangles".into(),
        json!(brow.iter().filter(|b| **b).count()),
    );
    // Hair/garment regions as drawn (before growth): the surface landmarks
    // must not sit on them.
    let marked_cloth: Vec<Option<usize>> =
        garment_seed.iter().map(|s| s.map(|(gi, _)| gi)).collect();

    // ---- Seed growth up to the colour edge.
    let g_count = guidance.garments.len();
    let skin_owner = g_count;
    let any_seed =
        garment_seed.iter().any(|s| s.is_some()) || skin_seed.iter().any(|s| s.is_some());
    let mut seed_strong = vec![false; n];
    let mut seed_grown = vec![false; n];
    let mut grown_per_owner = vec![0usize; g_count + 1];
    let mut strong_per_owner = vec![0usize; g_count + 1];
    let mut models: Vec<Option<ColourModel>> = vec![None; g_count + 1];
    let mut garment_on_skin = 0usize;
    let mut off_label = 0usize;
    if any_seed {
        let lab: Vec<Option<[f64; 3]>> = (0..n)
            .into_par_iter()
            .map(|t| triangle_lab(scan, t))
            .collect();
        let mut owner: Vec<Option<usize>> = (0..n)
            .map(|t| {
                garment_seed[t]
                    .map(|(gi, _)| gi)
                    .or(skin_seed[t].map(|_| skin_owner))
            })
            .collect();
        let depth_of = |t: usize| -> f32 {
            garment_seed[t]
                .map(|(_, d)| d)
                .or(skin_seed[t])
                .unwrap_or(0.0)
        };
        let from_stroke = |t: usize| {
            if garment_seed[t].is_some() {
                seed_stroke[t]
            } else {
                skin_stroke[t]
            }
        };
        // Item colour models from the polygon cores and strokes.
        let mut samples: Vec<Vec<[f64; 3]>> = vec![vec![]; g_count + 1];
        for t in 0..n {
            if let (Some(o), Some(c)) = (owner[t], lab[t])
                && (depth_of(t) >= 3.0 || from_stroke(t))
            {
                samples[o].push(c);
            }
        }
        for (o, s) in samples.iter_mut().enumerate() {
            let stride = (s.len() / 4000).max(1);
            let picked: Vec<[f64; 3]> = s.iter().step_by(stride).copied().collect();
            models[o] = ColourModel::from_samples(&picked);
        }
         
        let mut dropped = vec![false; n];
        for t in 0..n {
            let (Some(o), Some(c)) = (owner[t], lab[t]) else { continue };
            // Explicit rigid parts can be skin-coloured (kept ears, horns,
            // fangs). Colour disagreement must not perforate their marked
            // surface by turning its interior back into body skin/hair.
            if o == skin_owner || from_stroke(t) || guidance.garments[o].rigid {
                continue;
            }
            let Some(own) = models[o] else { continue };
            let hair = guidance.garments[o].label == Label::Hair;
            let other = (0..g_count).any(|o2| {
                o2 != o
                    && (guidance.garments[o2].label == Label::Hair) != hair
                    && models[o2].is_some_and(|m| seed_on_skin(&own, &m, &c))
            });
            if other {
                owner[t] = None;
                dropped[t] = true;
                off_label += 1;
            }
        }
        for t in 0..n {
            if let Some(o) = owner[t] {
                let on_skin = o != skin_owner
                    && !guidance.garments[o].rigid
                    && !from_stroke(t)
                    && matches!((models[o], models[skin_owner], lab[t]),
                        (Some(m), Some(s), Some(c)) if seed_on_skin(&m, &s, &c));
                if on_skin {
                    garment_on_skin += 1;
                }
                seed_strong[t] = !on_skin
                    && (depth_of(t) >= 8.0
                        || from_stroke(t)
                        || matches!((models[o], lab[t]), (Some(m), Some(c)) if m.accepts(&c)));
            }
        }
        let centroid: Vec<V> = scan
            .triangles
            .par_iter()
            .map(|t| {
                t.iter()
                    .fold(V::zeros(), |s, &i| s + v(scan.positions[i as usize]))
                    / 3.0
            })
            .collect();
        let budget: Vec<f64> = (0..n)
            .map(|t| {
                let mpp = if garment_seed[t].is_some() {
                    seed_mpp[t]
                } else {
                    skin_mpp[t]
                };
                GROW_PX * mpp
            })
            .collect();
        let incident = Incident::new(&scan.triangles, &scan.weld.id, scan.weld.count);
        seed_grown = grow_seeds(
            &scan.triangles,
            &scan.weld.id,
            &incident,
            &lab,
            &centroid,
            &mut owner,
            &seed_strong,
            &budget,
            &models,
        );
        for t in 0..n {
            if dropped[t] {
                garment_seed[t] = None;
            }
        }
        for t in 0..n {
            if !seed_grown[t] {
                continue;
            }
            let o = owner[t].expect("grown triangles have an owner");
            if o == skin_owner {
                skin_seed[t] = Some(0.0);
            } else {
                garment_seed[t] = Some((o, 0.0));
            }
            seed_strong[t] = true;
            grown_per_owner[o] += 1;
        }
        for t in 0..n {
            if seed_strong[t]
                && let Some(o) = owner[t]
            {
                strong_per_owner[o] += 1;
            }
        }
    }
    let counts: Vec<Value> = guidance
        .garments
        .iter()
        .enumerate()
        .map(|(gi, g)| {
            json!({
                "garment": g.name,
                "seed_triangles": hits_per_garment[gi],
                "grown_triangles": grown_per_owner[gi],
                "strong_triangles": strong_per_owner[gi],
                "colour_lab": models[gi].map(|m| m.median),
                "colour_tolerance": models[gi].map(|m| m.tolerance),
            })
        })
        .collect();
    report.insert("garment_seeds".into(), json!(counts));
    report.insert(
        "skin_over_garment_triangles".into(),
        json!(skin_over_garment),
    );
    report.insert(
        "garment_seed_on_skin_triangles".into(),
        json!(garment_on_skin),
    );
    report.insert("seed_off_label_colour_triangles".into(), json!(off_label));
    report.insert(
        "skin_seed_triangles".into(),
        json!(skin_seed.iter().filter(|x| x.is_some()).count()),
    );
    report.insert(
        "skin_grown_triangles".into(),
        json!(grown_per_owner[skin_owner]),
    );
    report.insert(
        "skin_colour".into(),
        json!(models[skin_owner].map(|m| json!({"lab": m.median, "tolerance": m.tolerance}))),
    );
    report.insert(
        "cleanup_triangles".into(),
        json!(cleanup.iter().filter(|x| **x).count()),
    );

    // ---- Landmarks.
    let mut landmark_views: Vec<&str> = guidance
        .landmarks
        .values()
        .flatten()
        .map(|p| p.view.as_str())
        .collect();
    landmark_views.sort_unstable();
    landmark_views.dedup();
    let projected: BTreeMap<&str, Vec<(f64, f64, f64)>> = landmark_views
        .par_iter()
        .map(|id| {
            let cam = &cameras[*id];
            (
                *id,
                scan.positions.iter().map(|p| cam.project(*p)).collect(),
            )
        })
        .collect();
    let occluder = |t: u32| marked_cloth[t as usize].is_some();
    let cloth = |t: u32| garment_seed[t as usize].is_some();
    struct Observed<'a> {
        point: &'a Point,
        ray: (V, V),
        weight: f64,
        surface: Option<Snap>,
        section: Option<(V, Section)>,
        problem: Option<Value>,
        snap_note: Option<Value>,
    }
    let names: Vec<(&String, &Vec<Point>)> = guidance
        .landmarks
        .iter()
        .filter(|(n, p)| !p.is_empty() && !is_outline_landmark(n))
        .collect();
    let observed: Vec<Vec<Observed>> = names
        .par_iter()
        .map(|(name, points)| {
            let joint = is_joint_landmark(name);
            points
                .par_iter()
                .map(|p| {
                    let cam = &cameras[&p.view];
                    let proj = &projected[p.view.as_str()];
                    let ray = cam.ray(p.px[0], p.px[1]);
                    let mut o = Observed {
                        point: p,
                        ray,
                        weight: p.effective_confidence(),
                        surface: None,
                        section: None,
                        problem: None,
                        snap_note: None,
                    };
                    let problem = |text: String| {
                        Some(json!({"landmark":name,"view":p.view,"px":p.px,"problem":text,"severity":"error"}))
                    };
                    if joint {
                        let (x, y) = (p.px[0].floor() as i64, p.px[1].floor() as i64);
                        let layers = Layers::new(cam, proj, &scan.positions, &scan.triangles, x - 1, y - 1, x + 1, y + 1);
                        if layers.at(x, y).is_empty() {
                            o.problem = problem("pixel misses the model".into());
                            return o;
                        }
                        // Median section centre over the 3×3 pixels around it.
                        let thin = min_limb_thickness(name) * unit;
                        let centre = section_probe(layers.at(x, y), &cloth, thin);
                        if let Some(s) = centre {
                            let mut depths: Vec<f64> = vec![];
                            for dy in -1..=1 {
                                for dx in -1..=1 {
                                    if let Some(q) = section_probe(layers.at(x + dx, y + dy), &cloth, thin)
                                        && q.centre() > s.near - (s.far - s.near)
                                        && q.centre() < s.far + (s.far - s.near)
                                    {
                                        depths.push(q.centre());
                                    }
                                }
                            }
                            depths.sort_by(f64::total_cmp);
                            let z = depths.get(depths.len() / 2).copied().unwrap_or(s.centre());
                            o.section = Some((ray.0 + ray.1 * (z + 100.0), s));
                        }
                        return o;
                    }
                    match snap_surface(
                        cam,
                        proj,
                        &scan.positions,
                        &scan.triangles,
                        p.px,
                        &occluder,
                        unit,
                        // The nose tip and a pinna stand out of the skin
                        // around them like a strand would.
                        name.as_str() != "nose_tip" && !is_ear_landmark(name),
                    ) {
                        Ok(s) => {
                            if s.skipped > 0 || s.moved_px > 0.0 {
                                let note = match s.first_occluder {
                                    Some(t) => format!(
                                        "lies on your `{}` region",
                                        marked_cloth[t as usize].map_or("?", |gi| guidance.garments[gi].name.as_str())
                                    ),
                                    None => "lies on a layer much nearer the camera than the skin around it (probably a hair strand)".into(),
                                };
                                o.snap_note = Some(json!({
                                    "landmark": name, "view": p.view, "px": p.px, "note": note,
                                    "skipped_layers": s.skipped,
                                    "behind_first_surface_mm": s.behind_first / unit * 1000.0,
                                    "moved_px": s.moved_px,
                                    "triangle": s.tri,
                                }));
                            }
                            o.surface = Some(s);
                        }
                        Err(SnapError::Misses) => o.problem = problem("pixel misses the model".into()),
                        Err(SnapError::NoSkin { first_occluder, nearer }) => {
                            let what = match (first_occluder, nearer) {
                                (Some(t), _) => format!(
                                    "lies on your `{}` region",
                                    marked_cloth[t as usize].map_or("?", |gi| guidance.garments[gi].name.as_str())
                                ),
                                (None, Some(d)) if d > SNAP_STRAND_M * unit => format!(
                                    "lies on a layer {:.0} mm nearer the camera than the skin around it (a hair strand?)",
                                    d / unit * 1000.0
                                ),
                                _ => "lies on no skin layer at a plausible depth".into(),
                            };
                            o.problem = problem(format!(
                                "{what}, and no skin was found behind it or within 3 mm; rejected — mark it on the visible skin"
                            ));
                        }
                    }
                    o
                })
                .collect()
        })
        .collect();
    let mut landmarks = BTreeMap::new();
    let mut joint_rays = BTreeMap::new();
    let mut joint_section = BTreeMap::new();
    let mut confidence = BTreeMap::new();
    let mut problems = vec![];
    let mut snaps = vec![];
    let mut sections = serde_json::Map::new();
    let mut view_info = serde_json::Map::new();
    let mut disagree = vec![];
    for ((name, points), obs) in names.iter().zip(observed) {
        confidence.insert(
            name.to_string(),
            (points.iter().map(|p| p.effective_confidence()).sum::<f64>() / points.len() as f64)
                .clamp(0.0, 1.0),
        );
        let joint = is_joint_landmark(name);
        let mut usable = vec![];
        for o in obs {
            if let Some(p) = o.problem.clone() {
                problems.push(p);
                continue;
            }
            if let Some(s) = o.snap_note.clone() {
                snaps.push(s);
            }
            usable.push(o);
        }
        if usable.is_empty() {
            continue;
        }
        let dirs: Vec<V> = usable.iter().map(|o| o.ray.1).collect();
        let spread = max_angle_deg(&dirs);
        let notes: Vec<Value> = usable
            .iter()
            .filter(|o| o.point.visibility.is_some() || !o.point.reason.is_empty())
            .map(|o| {
                json!({
                    "view": o.point.view,
                    "visibility": o.point.visibility,
                    "reason": o.point.reason,
                    "effective_confidence": o.weight,
                })
            })
            .collect();
        view_info.insert(
            name.to_string(),
            json!({"views": usable.len(), "spread_deg": spread, "effective_confidence": confidence[*name], "observations": notes}),
        );
        let best = usable
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.weight.total_cmp(&b.1.weight).then(b.0.cmp(&a.0)))
            .map(|(i, _)| i)
            .unwrap();
        let triangulated = (usable.len() >= 2 && spread >= MIN_VIEW_ANGLE_DEG)
            .then(|| {
                triangulate(
                    &usable
                        .iter()
                        .map(|o| (o.ray.0, o.ray.1, o.weight))
                        .collect::<Vec<_>>(),
                )
            })
            .flatten();
        if joint {
            let conf = confidence[*name];
            match triangulated {
                Some(p) => {
                    landmarks.insert(name.to_string(), a(p));
                }
                None => {
                    let o = &usable[best];
                    joint_rays.insert(name.to_string(), (a(o.ray.0), a(o.ray.1)));
                    if let Some((p, _)) = &o.section {
                        joint_section.insert(name.to_string(), a(*p));
                    }
                }
            }
            if triangulated.is_none() || conf < 0.6 {
                let probes: Vec<Value> = usable
                    .iter()
                    .filter_map(|o| {
                        let (p, s) = o.section.as_ref()?;
                        Some(json!({
                            "view": o.point.view,
                            "point": a(*p),
                            "behind_first_surface_m": cameras[&o.point.view].project(a(*p)).2 - s.first,
                            "thickness_m": s.far - s.near,
                            "crossings": s.crossings,
                            "layer": if s.body_layer { "body" } else { "outer" },
                            "ambiguous": !s.body_layer && s.crossings > 2,
                            "offset_from_triangulated_m": triangulated.map(|t| (t - p).norm()),
                        }))
                    })
                    .collect();
                if let Some(first) = probes.first() {
                    let mut entry = first.clone();
                    entry["observations"] = json!(probes);
                    sections.insert(name.to_string(), entry);
                }
            }
        } else {
            // Surface points: the confidence-weighted mean of the snapped skin
            // points (each on the skin; ray triangulation would leave the
            // surface by the pixel error over the view angle).
            let snapped: Vec<(V, f64)> = usable
                .iter()
                .filter_map(|o| Some((o.surface.as_ref()?.point, o.weight.max(1e-3))))
                .collect();
            if !snapped.is_empty() {
                let total: f64 = snapped.iter().map(|s| s.1).sum();
                let p = snapped.iter().map(|(p, w)| p * *w).sum::<V>() / total;
                let worst = snapped
                    .iter()
                    .map(|(q, _)| (q - p).norm())
                    .fold(0.0, f64::max);
                if worst > SURFACE_AGREE_M * unit {
                    disagree.push(json!({
                        "landmark": name,
                        "views": usable.iter().map(|o| o.point.view.as_str()).collect::<Vec<_>>(),
                        "max_offset_mm": worst / unit * 1000.0,
                    }));
                }
                landmarks.insert(name.to_string(), a(p));
            }
        }
    }
    report.insert("landmarks".into(), json!(landmarks));
    report.insert("joint_rays".into(), json!(joint_rays));
    report.insert("joint_section_depth".into(), Value::Object(sections));
    report.insert("landmark_views".into(), Value::Object(view_info));
    report.insert("landmark_problems".into(), json!(problems));
    report.insert("landmark_snaps".into(), json!(snaps));
    report.insert("landmark_disagreements".into(), json!(disagree));
    let hands: BTreeMap<String, Vec3> = landmarks
        .iter()
        .chain(&joint_section)
        .filter(|(n, _)| is_hand_landmark(n))
        .map(|(n, p)| (n.clone(), *p))
        .collect();
    report.insert("hand_points".into(), json!(hands));
    let hand_points = hands;
    report.insert("subject".into(), json!(guidance.subject));
    // Views used only for lid-margin or ear points (close-ups) do not widen
    // what counts as seen: those marks must not change the segmentation.
    let lid_only = |view: &str| {
        !guidance.landmarks.iter().any(|(n, obs)| {
            !is_lid_landmark(n) && !is_ear_landmark(n) && obs.iter().any(|o| o.view == view)
        }) && !guidance.garments.iter().any(|g| {
            g.regions.iter().any(|r| r.view == view) || g.strokes.iter().any(|s| s.view == view)
        }) && !guidance
            .skin
            .iter()
            .chain(&guidance.skin_cleanup)
            .any(|r| r.view == view)
            && !guidance
                .skin_strokes
                .iter()
                .chain(&guidance.skin_cleanup_strokes)
                .any(|s| s.view == view)
    };
    let mut seen = vec![false; scan.triangles.len()];
    for (_, b) in buffers.iter().filter(|(view, _)| !lid_only(view)) {
        for &t in b.id.iter().filter(|&&t| t != u32::MAX) {
            seen[t as usize] = true;
        }
    }
    let seed_stroke: Vec<bool> = (0..n)
        .map(|t| {
            (garment_seed[t].is_some() && seed_stroke[t])
                || (skin_seed[t].is_some() && skin_stroke[t])
        })
        .collect();
    let (outlines, outline_problems) = resolve_outlines(guidance, &cameras);
    report.insert("outlines".into(), json!(outlines));
    report.insert("outline_problems".into(), json!(outline_problems));
    Ok(Resolved {
        yaw: facing_yaw(guidance)?,
        landmarks,
        joint_rays,
        joint_section,
        hand_points,
        outlines,
        confidence,
        garment_seed,
        skin_seed,
        seed_strong,
        seed_grown,
        seed_stroke,
        cleanup,
        brow,
        seen,
        garments: guidance.garments.clone(),
        report: Value::Object(report),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creature_fields_parse_and_stay_out_of_default_output() {
        let s: Subject = serde_json::from_str(r#"{"kind": "x", "hand_digits": 4}"#).unwrap();
        assert_eq!(s.digits(), [4, 4]);
        let s: Subject =
            serde_json::from_str(r#"{"kind": "x", "hand_digits": {"l": 3, "r": 5}}"#).unwrap();
        assert_eq!(s.digits(), [3, 5]);
        assert_eq!(Subject::default().digits(), [5, 5]);
        assert!(
            !json!(Subject::default())
                .to_string()
                .contains("hand_digits")
        );
        let g: GarmentGuide =
            serde_json::from_str(r#"{"name": "horns", "label": "garment", "rigid": true}"#)
                .unwrap();
        assert!(g.rigid);
        let g: GarmentGuide =
            serde_json::from_str(r#"{"name": "robe", "label": "garment"}"#).unwrap();
        assert!(!g.rigid && !json!(g).to_string().contains("rigid"));
    }

    #[test]
    fn brow_strokes_parse_and_stay_out_of_default_output() {
        let g: Guidance = serde_json::from_str(
            r#"{"schema": "x", "source_hash": "h", "front_view": "body_y000",
                "brow_strokes": [{"view": "face_y000", "points": [[10, 20], [40, 15]], "width_px": 12}]}"#,
        )
        .unwrap();
        assert_eq!(g.brow_strokes.len(), 1);
        assert_eq!(g.brow_strokes[0].width_px, 12.0);
        assert!(
            !json!(Guidance::default())
                .to_string()
                .contains("brow_strokes")
        );
    }

    #[test]
    fn camera_projection_and_ray_roundtrip() {
        let cams = cameras_for(V::new(-0.5, 0.0, -0.2), V::new(0.5, 1.8, 0.2));
        let p: Vec3 = [0.12, 1.1, 0.05];
        for cam in &cams {
            let (x, y, _) = cam.project(p);
            let (o, d) = cam.ray(x, y);
            let t = (v(p) - o).dot(&d);
            assert!((o + d * t - v(p)).norm() < 1e-9, "{}", cam.id);
        }
        // Front view looks along −Z with +X on the right.
        assert_eq!(cams[0].forward, [0.0, 0.0, -1.0]);
        assert!((cams[0].right[0] - 1.0).abs() < 1e-12);
    }
    #[test]
    fn polygon_inside_and_edge_distance() {
        let sq = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        assert!(inside(&sq, 5.0, 5.0) && !inside(&sq, 11.0, 5.0));
        assert!((edge_distance(&sq, 5.0, 2.0) - 2.0).abs() < 1e-12);
    }
    #[test]
    fn facing_yaw_turns_each_view_to_plus_z() {
        for (view, deg) in [
            ("body_y000", 0.0f64),
            ("body_y090", 90.0),
            ("body_y180", 180.0),
        ] {
            let g = Guidance {
                schema: SCHEMA.into(),
                front_view: view.into(),
                ..Default::default()
            };
            let yaw = facing_yaw(&g).unwrap();
            let r = nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::y_axis(), yaw);
            let t = deg.to_radians();
            let seen = r * V::new(t.sin(), 0.0, t.cos());
            assert!((seen - V::z()).norm() < 1e-9, "{view}");
        }
    }

    /// Front camera over [-0.05, 0.05]² at 1 mm per pixel, looking along −Z.
    fn test_camera() -> Camera {
        Camera {
            id: "t".into(),
            width: 100,
            height: 100,
            center: [0.0, 0.0, 0.0],
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, -1.0],
            meters_per_pixel: 0.001,
        }
    }

    /// Axis-aligned quad at height z facing +Z (towards the camera).
    fn quad(
        positions: &mut Vec<Vec3>,
        triangles: &mut Vec<[u32; 3]>,
        lo: [f64; 2],
        hi: [f64; 2],
        z: f64,
    ) {
        let o = positions.len() as u32;
        positions.extend([
            [lo[0], lo[1], z],
            [hi[0], lo[1], z],
            [hi[0], hi[1], z],
            [lo[0], hi[1], z],
        ]);
        triangles.extend([[o, o + 1, o + 2], [o, o + 2, o + 3]]);
    }

    #[test]
    fn surface_snap_walks_past_strands_and_marked_hair_and_rejects_without_skin() {
        let cam = test_camera();
        let (mut positions, mut triangles) = (vec![], vec![]);
        // Skin: a large face plane at z = 0.
        quad(
            &mut positions,
            &mut triangles,
            [-0.05, -0.05],
            [0.05, 0.05],
            0.0,
        );
        // A thin strand 12 mm in front of it, 4 mm wide, crossing the centre.
        quad(
            &mut positions,
            &mut triangles,
            [-0.05, -0.002],
            [0.05, 0.002],
            0.012,
        );
        let proj: Vec<_> = positions.iter().map(|p| cam.project(*p)).collect();
        let none = |_: u32| false;
        // Unit 1: tolerances in metres. Centre pixel (on the strand).
        let s = snap_surface(
            &cam,
            &proj,
            &positions,
            &triangles,
            [50.2, 50.3],
            &none,
            1.0,
            true,
        )
        .unwrap();
        assert_eq!(s.skipped, 1, "walked past the strand");
        assert!(s.point[2].abs() < 1e-9 && (s.behind_first - 0.012).abs() < 1e-6);
        assert!((s.point[0] - 0.0002).abs() < 1e-9, "kept the marked pixel");
        // Off the strand: the first hit is the skin.
        let s = snap_surface(
            &cam,
            &proj,
            &positions,
            &triangles,
            [50.5, 30.5],
            &none,
            1.0,
            true,
        )
        .unwrap();
        assert_eq!((s.skipped, s.moved_px), (0, 0.0));
        // The strand marked as hair is skipped without the depth test too.
        let hair = |t: u32| t >= 2;
        let s = snap_surface(
            &cam,
            &proj,
            &positions,
            &triangles,
            [50.5, 50.5],
            &hair,
            1.0,
            false,
        )
        .unwrap();
        assert_eq!(s.skipped, 1);
        assert!(s.first_occluder.is_some_and(|t| t >= 2));
         
        let (mut p2, mut t2) = (vec![], vec![]);
        quad(&mut p2, &mut t2, [-0.05, -0.05], [-0.004, 0.05], 0.0);
        quad(&mut p2, &mut t2, [0.004, -0.05], [0.05, 0.05], 0.0);
        quad(&mut p2, &mut t2, [-0.004, -0.05], [0.004, -0.004], 0.0);
        quad(&mut p2, &mut t2, [-0.004, 0.004], [0.004, 0.05], 0.0);
        quad(&mut p2, &mut t2, [-0.05, -0.002], [0.05, 0.002], 0.012);
        let proj2: Vec<_> = p2.iter().map(|p| cam.project(*p)).collect();
        let r = snap_surface(&cam, &proj2, &p2, &t2, [50.5, 50.5], &none, 1.0, true);
        match r {
            Err(SnapError::NoSkin {
                first_occluder: None,
                nearer: Some(d),
            }) => {
                assert!((d - 0.012).abs() < 1e-6)
            }
            other => panic!("{other:?}"),
        }
        // Missing the model entirely.
        let r = snap_surface(
            &cam,
            &proj,
            &positions[..4],
            &triangles[..0],
            [50.5, 50.5],
            &none,
            1.0,
            true,
        );
        assert_eq!(r.unwrap_err(), SnapError::Misses);
    }

    #[test]
    fn surface_snap_moves_to_skin_within_3_mm() {
        let cam = test_camera();
        let (mut positions, mut triangles) = (vec![], vec![]);
        // Skin with a 2 mm wide gap at x ∈ [-1, 1] mm, a strand in front of the gap.
        quad(
            &mut positions,
            &mut triangles,
            [-0.05, -0.05],
            [-0.001, 0.05],
            0.0,
        );
        quad(
            &mut positions,
            &mut triangles,
            [0.001, -0.05],
            [0.05, 0.05],
            0.0,
        );
        quad(
            &mut positions,
            &mut triangles,
            [-0.0015, -0.05],
            [0.0015, 0.05],
            0.010,
        );
        let proj: Vec<_> = positions.iter().map(|p| cam.project(*p)).collect();
        let s = snap_surface(
            &cam,
            &proj,
            &positions,
            &triangles,
            [50.5, 50.5],
            &|_| false,
            1.0,
            true,
        )
        .unwrap();
        assert!(s.moved_px > 0.5 && s.moved_px <= 3.0, "{s:?}");
        assert!(s.point[2].abs() < 1e-9);
    }

    #[test]
    fn section_probe_finds_the_body_under_marked_cloth() {
        let h = |depth: f64, tri: u32, facing: f64| Hit { depth, tri, facing };
        // Cloth (tri 0/3, marked) around a body layer (tri 1/2).
        let hits = [
            h(0.0, 0, -1.0),
            h(0.02, 1, -1.0),
            h(0.08, 2, 1.0),
            h(0.12, 3, 1.0),
        ];
        let cloth = |t: u32| t == 0;
        let s = section_probe(&hits, &cloth, 0.008).unwrap();
        assert!(s.body_layer && (s.centre() - 0.05).abs() < 1e-12);
        // Nothing marked: the whole outer interval.
        let s = section_probe(&hits, &|_| false, 0.008).unwrap();
        assert!(!s.body_layer && (s.centre() - 0.06).abs() < 1e-12 && s.crossings == 4);
        // Single shell: centre of the limb.
        let s = section_probe(
            &hits[..1]
                .iter()
                .chain(&hits[3..])
                .copied()
                .collect::<Vec<_>>(),
            &cloth,
            0.008,
        )
        .unwrap();
        assert!((s.centre() - 0.06).abs() < 1e-12);
        // A thick cloth sheet (two marked crossings 3 mm apart) is skipped.
        let sheet = [
            h(0.0, 0, -1.0),
            h(0.003, 0, 1.0),
            h(0.02, 1, -1.0),
            h(0.08, 2, 1.0),
        ];
        let s = section_probe(&sheet, &cloth, 0.008).unwrap();
        assert!((s.centre() - 0.05).abs() < 1e-12);
        // The first limb along the ray, not the torso behind it.
        let two = [
            h(0.0, 1, -1.0),
            h(0.06, 2, 1.0),
            h(0.1, 1, -1.0),
            h(0.3, 2, 1.0),
        ];
        let s = section_probe(&two, &|_| false, 0.008).unwrap();
        assert!((s.centre() - 0.03).abs() < 1e-12);
        assert!(section_probe(&two[..1], &|_| false, 0.008).is_none());
    }

    #[test]
    fn seeds_grow_over_connected_surface_up_to_the_colour_edge() {
        // A strip of 20 triangles in a row (each shares a vertex with the
        // next), 1 cm apart: 0..10 skin colour, 10..20 garment blue.
        let n = 20;
        let triangles: Vec<[u32; 3]> = (0..n as u32).map(|t| [t, t + 1, t + 100]).collect();
        let weld_id: Vec<usize> = (0..200).collect();
        let incident = Incident::new(&triangles, &weld_id, 200);
        let centroid: Vec<V> = (0..n).map(|t| V::new(t as f64 * 0.01, 0.0, 0.0)).collect();
        let skin = [70.0, 20.0, 30.0];
        let blue = [40.0, 0.0, -40.0];
        let lab: Vec<Option<[f64; 3]>> = (0..n)
            .map(|t| Some(if t < 10 { skin } else { blue }))
            .collect();
        // Skin seed at 2, garment seed at 17; skin = owner 1, garment = 0.
        let mut owner = vec![None; n];
        owner[2] = Some(1);
        owner[17] = Some(0);
        let sources = vec![true; n];
        let budget = vec![0.1; n];
        let models = vec![
            Some(ColourModel {
                median: blue,
                tolerance: 10.0,
            }),
            Some(ColourModel {
                median: skin,
                tolerance: 10.0,
            }),
        ];
        let grown = grow_seeds(
            &triangles, &weld_id, &incident, &lab, &centroid, &mut owner, &sources, &budget,
            &models,
        );
        assert!((0..10).all(|t| owner[t] == Some(1)), "{owner:?}");
        assert!((10..20).all(|t| owner[t] == Some(0)), "{owner:?}");
        assert!(!grown[2] && !grown[17] && grown[0] && grown[9] && grown[10]);
        // A short budget stops the front: 2.5 cm from the seed.
        let mut owner = vec![None; n];
        owner[2] = Some(1);
        let budget = vec![0.025; n];
        grow_seeds(
            &triangles, &weld_id, &incident, &lab, &centroid, &mut owner, &sources, &budget,
            &models,
        );
        assert_eq!(owner.iter().filter(|o| o.is_some()).count(), 5, "{owner:?}"); // 0..=4
        // A seed that is not a source (weak, off-colour) does not grow.
        let mut owner = vec![None; n];
        owner[2] = Some(1);
        let sources = vec![false; n];
        let grown = grow_seeds(
            &triangles, &weld_id, &incident, &lab, &centroid, &mut owner, &sources, &budget,
            &models,
        );
        assert!(!grown.iter().any(|g| *g));
    }

    #[test]
    fn garment_seeds_on_bare_skin_are_recognised() {
        let skin = ColourModel {
            median: [64.0, 24.0, 36.0],
            tolerance: 15.0,
        };
        // A red tunic whose wide tolerance also takes the tanned neck.
        let tunic = ColourModel {
            median: [41.0, 45.0, 33.0],
            tolerance: 28.0,
        };
        assert!(tunic.accepts(&skin.median));
        assert!(seed_on_skin(&tunic, &skin, &[62.0, 25.0, 35.0]));
        assert!(!seed_on_skin(&tunic, &skin, &[48.0, 40.0, 34.0]));
        // A cream collar on pale skin stays the collar's.
        let pale = ColourModel {
            median: [77.0, 20.0, 32.0],
            tolerance: 28.0,
        };
        let cream = ColourModel {
            median: [90.0, 4.0, 18.0],
            tolerance: 8.0,
        };
        assert!(!seed_on_skin(&cream, &pale, &[86.0, 6.0, 19.0]));
        assert!(!seed_on_skin(&cream, &pale, &[70.0, 4.0, 18.0]));
    }

    #[test]
    fn strokes_rasterise_to_a_band_of_triangles() {
        let cam = test_camera();
        let (mut positions, mut triangles) = (vec![], vec![]);
        // A 10×10 grid of 1 cm quads.
        for i in 0..10 {
            for j in 0..10 {
                let (x, y) = (-0.05 + i as f64 * 0.01, -0.05 + j as f64 * 0.01);
                quad(
                    &mut positions,
                    &mut triangles,
                    [x, y],
                    [x + 0.01, y + 0.01],
                    0.0,
                );
            }
        }
        let buffer = rasterize(&cam, &positions, &triangles);
        // A horizontal 4 px stroke through the middle row of pixels 45..=55.
        let s = Stroke {
            view: "t".into(),
            points: vec![[5.0, 55.0], [95.0, 55.0]],
            width_px: 4.0,
        };
        let tris = stroke_triangles(&cam, &buffer, &s);
        // Rows of quads: pixel y 55 ± 2 lies in quad row j = 4 (y ∈ [-0.01, 0]).
        let rows: std::collections::BTreeSet<usize> =
            tris.iter().map(|(t, _)| (*t as usize / 2) % 10).collect();
        assert_eq!(rows.into_iter().collect::<Vec<_>>(), vec![4]);
        let cols: std::collections::BTreeSet<usize> =
            tris.iter().map(|(t, _)| *t as usize / 20).collect();
        assert_eq!(cols.len(), 10);
        assert!(tris.iter().all(|(_, d)| *d >= 0.0 && *d <= 2.0));
        // A dot.
        let dot = Stroke {
            view: "t".into(),
            points: vec![[25.0, 25.0]],
            width_px: 3.0,
        };
        assert!(!stroke_triangles(&cam, &buffer, &dot).is_empty());
        assert!((stroke_area_px(&s) - 94.0 * 4.0).abs() < 1e-9);
        let bad = Stroke {
            view: "t".into(),
            points: vec![[25.0, 25.0]],
            width_px: 0.0,
        };
        assert!(stroke_triangles(&cam, &buffer, &bad).is_empty());
    }

    #[test]
    fn layers_list_every_surface_under_a_pixel() {
        let cam = test_camera();
        let (mut positions, mut triangles) = (vec![], vec![]);
        quad(
            &mut positions,
            &mut triangles,
            [-0.05, -0.05],
            [0.05, 0.05],
            0.0,
        );
        quad(
            &mut positions,
            &mut triangles,
            [-0.05, -0.05],
            [0.05, 0.05],
            -0.02,
        );
        let proj: Vec<_> = positions.iter().map(|p| cam.project(*p)).collect();
        let l = Layers::new(&cam, &proj, &positions, &triangles, 40, 40, 60, 60);
        let hits = l.at(50, 50);
        assert_eq!(hits.len(), 2);
        assert!(hits[0].depth < hits[1].depth && hits[0].facing < 0.0);
        assert!(l.at(10, 10).is_empty());
    }

    #[test]
    fn hand_warnings_catch_finger_numbering_mistakes() {
        let mut here: BTreeMap<&str, [f64; 2]> = BTreeMap::new();
        here.insert("palm_r", [100.0, 100.0]);
        here.insert("finger2_r_base", [80.0, 140.0]);
        here.insert("finger3_r_base", [95.0, 145.0]);
        here.insert("finger4_r_base", [110.0, 145.0]);
        here.insert("finger5_r_base", [125.0, 140.0]);
        here.insert("finger3_r_tip", [95.0, 200.0]);
        assert!(hand_warnings("v", "r", &here).is_empty());
        // Ring and middle swapped; a tip marked on the knuckle side; a copy.
        here.insert("finger3_r_base", [110.0, 145.0]);
        here.insert("finger4_r_base", [95.0, 145.0]);
        here.insert("finger5_r_tip", [125.0, 120.0]);
        here.insert("finger5_r_mid", [125.5, 120.5]);
        let w = hand_warnings("v", "r", &here);
        assert_eq!(w.len(), 3, "{w:?}");
        assert!(w[0].contains("not in order") && w[1].contains("finger5_r_tip"));
        assert!(w[2].contains("same pixel"));
        // The other hand is not affected.
        assert!(hand_warnings("v", "l", &here).is_empty());
    }

    #[test]
    fn lid_warnings_catch_lid_margin_mistakes() {
        // Front view: the character's left eye on the image's right.
        let mut pts: BTreeMap<String, [f64; 2]> = BTreeMap::new();
        let mut put = |n: &str, p: [f64; 2]| {
            pts.insert(n.into(), p);
        };
        for (side, inner, outer) in [("l", 440.0, 560.0), ("r", 320.0, 200.0)] {
            let x = |f: f64| inner + (outer - inner) * f;
            put(&format!("eye_{side}_inner"), [inner, 280.0]);
            put(&format!("eye_{side}_outer"), [outer, 270.0]);
            put(&format!("eye_{side}_top"), [x(0.5), 245.0]);
            put(&format!("eye_{side}_bottom"), [x(0.5), 300.0]);
            for (k, up, down) in [(1, 255.0, 292.0), (2, 246.0, 299.0), (3, 252.0, 290.0)] {
                let f = 0.25 * k as f64;
                put(&format!("eye_{side}_upper_{k}"), [x(f), up]);
                put(&format!("eye_{side}_lower_{k}"), [x(f), down]);
            }
        }
        assert!(lid_warnings(&pts).is_empty(), "{:?}", lid_warnings(&pts));
        assert!(LID_LANDMARKS.iter().all(|n| is_known_landmark(n)));
        assert!(!is_known_landmark("eye_l_upper_4") && !is_lid_landmark("eye_l_top"));

        // Upper 1 and 3 swapped (out of order), upper/lower 2 swapped.
        let mut bad = pts.clone();
        let (u1, u3) = (bad["eye_l_upper_1"], bad["eye_l_upper_3"]);
        bad.insert("eye_l_upper_1".into(), u3);
        bad.insert("eye_l_upper_3".into(), u1);
        let (u2, l2) = (bad["eye_l_lower_2"], bad["eye_l_upper_2"]);
        bad.insert("eye_l_upper_2".into(), u2);
        bad.insert("eye_l_lower_2".into(), l2);
        let w = lid_warnings(&bad);
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(w[0].contains("eye_l_upper_1..3 are not in order"), "{w:?}");
        assert!(
            w[1].contains("eye_l_upper_2 is not above eye_l_lower_2"),
            "{w:?}"
        );

        // A point far above the eye (on the brow), one off its 25 % place.
        let mut bad = pts.clone();
        bad.insert("eye_r_upper_2".into(), [260.0, 180.0]);
        bad.insert("eye_r_lower_1".into(), [310.0, 292.0]);
        let w = lid_warnings(&bad);
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(w[0].contains("eye_r_lower_1 is 7 %"), "{w:?}");
        assert!(
            w[1].contains("eye_r_upper_2 at [260, 180] lies outside"),
            "{w:?}"
        );

        // Left and right lid points swapped; an incomplete lid.
        let mut bad = pts.clone();
        for n in LID_LANDMARKS.iter().filter(|n| n.starts_with("eye_l_")) {
            let r = n.replace("eye_l_", "eye_r_");
            let (pl, pr) = (bad[*n], bad[&r]);
            bad.insert(n.to_string(), pr);
            bad.insert(r, pl);
        }
        let w = lid_warnings(&bad);
        assert!(
            w.iter()
                .any(|w| w.starts_with("eye_l_upper_1 is on the other eye's side"))
        );
        assert!(
            w.iter()
                .any(|w| w.starts_with("eye_r_lower_3 is on the other eye's side"))
        );
        let mut partial = pts.clone();
        partial.remove("eye_r_lower_3");
        partial.remove("eye_l_inner");
        let w = lid_warnings(&partial);
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(w[0].contains("eye_l_inner / eye_l_outer are not"), "{w:?}");
        assert!(w[1].contains("5 of the 6"), "{w:?}");
    }

    #[test]
    fn probes_ignore_marks_in_views_not_rendered_yet() {
        let cam = |id: &str| Camera {
            id: id.into(),
            ..test_camera()
        };
        let views = ViewSet {
            schema: SCHEMA.into(),
            source: String::new(),
            source_hash: String::new(),
            cameras: vec![cam("head_y000")],
            images: BTreeMap::new(),
        };
        let point = |view: &str| Point {
            view: view.into(),
            px: [1.0, 2.0],
            confidence: 1.0,
            visibility: None,
            reason: String::new(),
        };
        let mut g = Guidance {
            schema: SCHEMA.into(),
            front_view: "body_y090".into(),
            ..Default::default()
        };
        g.landmarks
            .insert("eye_l".into(), vec![point("head_y000"), point("face_y000")]);
        g.skin.push(Polygon {
            view: "face_y000".into(),
            polygon: vec![[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]],
        });
        let (probe, missing) = known_views_only(&g, &views);
        assert_eq!(missing, vec!["face_y000".to_string()]);
        assert_eq!(probe.landmarks["eye_l"].len(), 1);
        assert!(probe.skin.is_empty() && probe.front_view == "body_y000");
    }

    #[test]
    fn ear_warnings_catch_outline_mistakes() {
        assert!(
            EAR_LANDMARKS
                .iter()
                .all(|n| is_known_landmark(n) && !is_joint_landmark(n))
        );
        assert!(!is_known_landmark("ear_l_middle"));
        // Character frame: forward +Z, up +Y, left +X; a 6 cm left ear
        // 9 cm behind and 3 cm below the eyes, 7 cm out.
        let (fwd, up, left) = (V::z(), V::y(), V::x());
        let eyes = V::new(0.0, 1.6, 0.1);
        let at = |f: f64, u: f64| V::new(0.07, 1.57 + u, 0.01 + f);
        let good: BTreeMap<String, V> = [
            ("ear_l_top", at(-0.01, 0.03)),
            ("ear_l_back", at(-0.02, 0.01)),
            ("ear_l_lobe", at(0.0, -0.03)),
            ("ear_l_front_top", at(0.01, 0.02)),
            ("ear_l_front_bottom", at(0.01, -0.02)),
            ("ear_l_tragus", at(0.012, 0.0)),
        ]
        .into_iter()
        .map(|(n, p)| (n.to_string(), p))
        .collect();
        assert!(ear_warnings(&good, fwd, up, left, Some(eyes), 1.0).is_empty());
        let mut bad = good.clone();
        // Top and lobe swapped, the back in front of the tragus.
        let (t, l) = (bad["ear_l_top"], bad["ear_l_lobe"]);
        bad.insert("ear_l_top".into(), l);
        bad.insert("ear_l_lobe".into(), t);
        bad.insert("ear_l_back".into(), at(0.03, 0.01));
        let w = ear_warnings(&bad, fwd, up, left, Some(eyes), 1.0);
        assert!(
            w.iter()
                .any(|w| w.starts_with("ear_l_top is not the highest")),
            "{w:?}"
        );
        assert!(
            w.iter()
                .any(|w| w.starts_with("ear_l_lobe is not the lowest")),
            "{w:?}"
        );
        assert!(
            w.iter()
                .any(|w| w.starts_with("ear_l_back is not the rearmost")),
            "{w:?}"
        );
        // On the right side of the head, and a single point.
        let mut wrong = BTreeMap::new();
        wrong.insert("ear_l_lobe".to_string(), V::new(-0.07, 1.54, 0.01));
        let w = ear_warnings(&wrong, fwd, up, left, Some(eyes), 1.0);
        assert!(
            w.iter().any(|w| w.contains("only ear_l_lobe is marked")),
            "{w:?}"
        );
        assert!(
            w.iter()
                .any(|w| w.contains("not on the character's left side")),
            "{w:?}"
        );
    }

    #[test]
    fn hand_landmark_names_and_visibility() {
        let names = hand_landmarks();
        assert_eq!(names.len(), 32);
        assert!(
            names
                .iter()
                .all(|n| is_hand_landmark(n) && is_joint_landmark(n))
        );
        assert!(is_hand_landmark("finger1_l_tip") && !is_hand_landmark("finger6_l_tip"));
        assert!(!is_hand_landmark("finger1_x_tip") && !is_hand_landmark("palm_c"));
        let p: Point = serde_json::from_str(
            r#"{"view":"body_y000","px":[1,2],"confidence":0.6,"visibility":"occluded","reason":"under the sleeve"}"#,
        )
        .unwrap();
        assert!((p.effective_confidence() - 0.3).abs() < 1e-12);
        let old: Point = serde_json::from_str(r#"{"view":"body_y000","px":[1,2]}"#).unwrap();
        assert_eq!(old.effective_confidence(), 1.0);
    }
}

#[cfg(test)]
mod registered_skin_anchor_tests {
    use super::*;
    #[test]
    fn skin_texture_anchors_apply_head_transform_and_exclude_held_out_or_rejected_members() {
        let accepted=json!({"body_y000":{"accepted":true,"similarity":{"matrix":[[2.,0.,10.],[0.,2.,20.]],"rotation_deg":4.5},
            "texture_matches_detail":[[[4.,5.],[1.,2.],0.9]],
            "outline_matches_detail":[[[7.,8.],[3.,4.],1.0]]}});
        let rejected=json!({"body_y000":{"accepted":false,"similarity":{"matrix":[[2.,0.,10.],[0.,2.,20.]],"rotation_deg":4.5},
            "texture_matches_detail":[[[40.,50.],[10.,20.],0.9]],
            "outline_matches_detail":[[[70.,80.],[30.,40.],1.0]]}});
        let data=json!({"head_registration":{"members":{"e1":accepted.clone(),"e2":rejected,"e3":accepted}}});
        assert_eq!(registered_skin_anchors(&data,"body_y000"),vec![([12.,24.],[4.,5.])]);
        assert_eq!(registered_anchors(&data,"body_y000","outline_matches_detail"),vec![([16.,28.],[7.,8.])]);
        assert_eq!(head_rotation_limits(&data,"body_y000"),[-5.,0.5]);
    }
}
