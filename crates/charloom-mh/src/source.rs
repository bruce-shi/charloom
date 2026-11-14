//! Reading render attributes (UVs, textures, materials) that `charloom_io`'s
//! `Asset` deliberately does not carry.
use anyhow::{Context, Result, bail, ensure};
use charloom_core::Vec3;
use charloom_io::LoadedAsset;
use image::RgbaImage;

/// One mesh primitive, positions in the frame chosen by the caller.
#[derive(Clone)]
pub struct Primitive {
    pub name: String,
    /// Index into `asset.meshes`.
    pub asset_mesh: usize,
    pub positions: Vec<Vec3>,
    pub uvs: Vec<[f32; 2]>,
    pub triangles: Vec<[u32; 3]>,
    pub material: Option<usize>,
}

/// Encoded image bytes kept verbatim for re-embedding, plus a decoded copy.
pub struct ImageData {
    pub mime: String,
    pub bytes: Vec<u8>,
    pub decoded: RgbaImage,
}

pub struct MaterialData {
    pub name: String,
    pub json: serde_json::Value,
    pub base_color: Option<usize>,
    pub base_color_factor: [f64; 4],
}

pub fn gltf_of(loaded: &LoadedAsset) -> Result<gltf::Gltf> {
    gltf::Gltf::from_slice(&serde_json::to_vec(&loaded.document.json)?)
        .context("re-parse glTF document")
}

/// Reads every `asset.meshes` entry with the given per-mesh positions.
pub fn read_primitives(loaded: &LoadedAsset, positions: &[Vec<Vec3>]) -> Result<Vec<Primitive>> {
    let gltf = gltf_of(loaded)?;
    let binary = loaded.document.binary.as_slice();
    let mut out = vec![];
    for (mi, mesh) in loaded.asset.meshes.iter().enumerate() {
        let primitive = gltf
            .document
            .nodes()
            .nth(mesh.node)
            .and_then(|n| n.mesh())
            .and_then(|m| m.primitives().nth(mesh.primitive))
            .context("primitive lookup")?;
        let reader = primitive.reader(|_| Some(binary));
        let uvs: Vec<[f32; 2]> = reader
            .read_tex_coords(0)
            .map(|t| t.into_f32().collect())
            .unwrap_or_else(|| vec![[0.0; 2]; mesh.positions.len()]);
        ensure!(uvs.len() == mesh.positions.len(), "uv_count_mismatch");
        ensure!(
            positions[mi].len() == mesh.positions.len(),
            "position_count_mismatch"
        );
        out.push(Primitive {
            name: mesh.name.clone(),
            asset_mesh: mi,
            positions: positions[mi].clone(),
            uvs,
            triangles: mesh.triangles.clone(),
            material: primitive.material().index(),
        });
    }
    Ok(out)
}

pub fn read_images(loaded: &LoadedAsset) -> Result<Vec<ImageData>> {
    let gltf = gltf_of(loaded)?;
    let binary = &loaded.document.binary;
    let mut out = vec![];
    for image in gltf.document.images() {
        let (bytes, mime) = match image.source() {
            gltf::image::Source::View { view, mime_type } => {
                let start = view.offset();
                let end = start + view.length();
                ensure!(end <= binary.len(), "image_view_out_of_bounds");
                (binary[start..end].to_vec(), mime_type.to_string())
            }
            gltf::image::Source::Uri { .. } => {
                bail!("external image URIs are not supported; pack the model as GLB")
            }
        };
        let decoded = image::load_from_memory(&bytes)
            .context("decode embedded image")?
            .to_rgba8();
        out.push(ImageData {
            mime,
            bytes,
            decoded,
        });
    }
    Ok(out)
}

pub fn read_materials(loaded: &LoadedAsset) -> Result<Vec<MaterialData>> {
    let gltf = gltf_of(loaded)?;
    Ok(gltf
        .document
        .materials()
        .enumerate()
        .map(|(i, m)| {
            let pbr = m.pbr_metallic_roughness();
            MaterialData {
                name: m.name().unwrap_or("material").to_string(),
                json: loaded.document.json["materials"][i].clone(),
                base_color: pbr
                    .base_color_texture()
                    .map(|t| t.texture().source().index()),
                base_color_factor: pbr.base_color_factor().map(f64::from),
            }
        })
        .collect())
}

/// Bilinear sRGB sample with repeat wrapping; returns sRGB in [0, 1].
pub fn sample(image: &RgbaImage, uv: [f64; 2]) -> [f64; 3] {
    let (w, h) = (image.width() as i64, image.height() as i64);
    let x = uv[0].rem_euclid(1.0) * w as f64 - 0.5;
    let y = uv[1].rem_euclid(1.0) * h as f64 - 0.5;
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let px = |xi: i64, yi: i64| {
        let p = image.get_pixel(xi.rem_euclid(w) as u32, yi.rem_euclid(h) as u32);
        [p[0] as f64, p[1] as f64, p[2] as f64]
    };
    let (x0, y0) = (x0 as i64, y0 as i64);
    let c = [
        px(x0, y0),
        px(x0 + 1, y0),
        px(x0, y0 + 1),
        px(x0 + 1, y0 + 1),
    ];
    std::array::from_fn(|k| {
        ((c[0][k] * (1. - fx) + c[1][k] * fx) * (1. - fy)
            + (c[2][k] * (1. - fx) + c[3][k] * fx) * fy)
            / 255.0
    })
}

pub fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
pub fn linear_to_srgb(c: f64) -> f64 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// CIE L*a*b* (D65) from sRGB in [0, 1].
pub fn lab(rgb: [f64; 3]) -> [f64; 3] {
    let [r, g, b] = rgb.map(srgb_to_linear);
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    let f = |t: f64| {
        if t > 0.008856 {
            t.cbrt()
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}
