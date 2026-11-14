//! MakeHuman's eyelashes on the fitted lids.
//!
//! The lash mesh is one card strip per lid: a root row on the lid margin and
//! rows fanning out to the tips. The rig refit carries every lash vertex on
//! its nearest body triangle with that triangle's area stretch, so where the
//! fit widens the lids the tips are pushed out up to twice as far and
//! neighbouring tips follow different triangles: long, splayed lashes. It
//! also attaches them to the shape-morphed rest body, while the lash mesh has
//! no shape morphs and only sits on the unmorphed lids, so the morph offset
//! near the eyes is carried into the lashes as well.
//! MakeHuman's default lashes are also heavy (a dense "false lash" texture,
//! the lower set nearly as long as the upper) next to the fine lash lines of
//! generated scans.
//!
//! Here every strip is rebuilt from its roots: each lash vertex keeps its
//! rest offset from its root on the lid margin, turned with the lid surface
//! (rotation only, smoothed along the margin) and scaled by the strip's own
//! margin stretch (bounded) times a length factor, shorter for the lower lid.
//! The lower lashes' texture is made more transparent. Weights are untouched:
//! the lashes stay on the lid bones, so blinks still carry them.
use crate::geom::{V, a, proper_rotation, v};
use crate::prior::Prior;
use crate::scan::build_query;
use crate::source::{ImageData, MaterialData};
use anyhow::Result;
use charloom_core::Vec3;
use nalgebra::Matrix3;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct Lashes {
    /// Export MakeHuman's eyelash mesh, rebuilt on the fitted lids. Off by
    /// default: the scan's painted lash line and eyeliner are baked onto the
    /// lids, and MakeHuman's card lashes would draw a second, spiky set.
    pub keep_makehuman: bool,
    /// Upper lash length relative to MakeHuman's, about the lid margin.
    pub upper_length: f64,
    /// Lower lash length relative to MakeHuman's.
    pub lower_length: f64,
    /// Opacity multiplier of the lower lashes' texture.
    pub lower_opacity: f64,
    /// Opacity multiplier of the upper lashes' texture.
    pub upper_opacity: f64,
}

impl Default for Lashes {
    fn default() -> Self {
        Self {
            keep_makehuman: false,
            upper_length: 0.7,
            lower_length: 0.4,
            lower_opacity: 0.5,
            upper_opacity: 1.0,
        }
    }
}

/// Root row: strip-boundary vertices this close to the rest body surface.
const ROOT_M: f64 = 0.0006;
/// Lid rotations are averaged over roots this close (rest), along the margin.
const SMOOTH_M: f64 = 0.002;
/// Bounds of the strip's margin stretch applied to lash length.
const STRETCH: (f64, f64) = (0.85, 1.2);

/// One lash strip (connected component of the lash mesh).
struct Strip {
    vertices: Vec<usize>,
    lower: bool,
}

pub struct Shaped {
    pub positions: Vec<Vec3>,
    /// Vertex lies on a lower-lid strip.
    pub lower: Vec<bool>,
    pub report: Value,
}

fn components(n: usize, tris: &[[u32; 3]]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for t in tris {
        for k in 1..3 {
            let (x, y) = (
                find(&mut parent, t[0] as usize),
                find(&mut parent, t[k] as usize),
            );
            parent[x] = y;
        }
    }
    let mut by_root: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        by_root.entry(r).or_default().push(i);
    }
    let mut out: Vec<Vec<usize>> = by_root.into_values().collect();
    out.sort();
    out
}

/// The body surface under `p`: its point on the rest and on the fitted body
/// and the surface rotation between them (triangle frames, as in the rig
/// refit's surface attachment).
fn surface_map(
    query: &charloom_engine::surface_query::SurfaceQuery,
    tris: &[[u32; 3]],
    rest: &[Vec3],
    fitted: &[Vec3],
    p: Vec3,
) -> Option<(V, V, Matrix3<f64>)> {
    let hit = query.closest_point(p, 0.01, None)?;
    let t = tris[hit.source[1]];
    let b = hit.barycentric;
    let frame = |pts: &[Vec3]| {
        let [p0, p1, p2] = t.map(|i| v(pts[i as usize]));
        let e = (p1 - p0).try_normalize(1e-12)?;
        let n = (p1 - p0).cross(&(p2 - p0)).try_normalize(1e-20)?;
        Some((
            p0 * b[0] + p1 * b[1] + p2 * b[2],
            Matrix3::from_columns(&[e, n.cross(&e), n]),
        ))
    };
    let ((q0, f0), (q1, f1)) = (frame(rest)?, frame(fitted)?);
    let r = f1 * f0.transpose();
    (r.iter().all(|x| x.is_finite())).then_some((q0, q1, r))
}

 pub fn shape(
    prior: &Prior,
    mesh: usize,
    rest_body: &[Vec3],
    fitted_body: &[Vec3],
    carried: &[Vec3],
    params: &Lashes,
) -> Shaped {
    let lash = &prior.meshes[mesh].prim;
    shape_strips(
        &lash.positions,
        &lash.triangles,
        rest_body,
        &prior.body().prim.triangles,
        fitted_body,
        carried,
        params,
    )
}

fn shape_strips(
    rest: &[Vec3],
    tris: &[[u32; 3]],
    rest_body: &[Vec3],
    body_tris: &[[u32; 3]],
    fitted_body: &[Vec3],
    carried: &[Vec3],
    params: &Lashes,
) -> Shaped {
    let n = rest.len();
    let query = build_query(rest_body, body_tris);
    let dist: Vec<f64> = rest
        .iter()
        .map(|p| {
            query
                .closest_point(*p, 0.05, None)
                .map_or(f64::INFINITY, |h| h.distance_m)
        })
        .collect();
    let mut edge_use: HashMap<(usize, usize), u32> = HashMap::new();
    for t in tris {
        for k in 0..3 {
            let (x, y) = (t[k] as usize, t[(k + 1) % 3] as usize);
            *edge_use.entry((x.min(y), x.max(y))).or_default() += 1;
        }
    }
    let mut boundary = vec![false; n];
    let mut adjacent: Vec<Vec<usize>> = vec![vec![]; n];
    for (&(x, y), &c) in &edge_use {
        adjacent[x].push(y);
        adjacent[y].push(x);
        if c == 1 {
            boundary[x] = true;
            boundary[y] = true;
        }
    }
    let is_root: Vec<bool> = (0..n).map(|i| boundary[i] && dist[i] < ROOT_M).collect();
    // Nearest root along the strip (Dijkstra over rest edge lengths).
    let mut root_of: Vec<Option<usize>> = (0..n).map(|i| is_root[i].then_some(i)).collect();
    let mut best = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::new();
    for i in (0..n).filter(|&i| is_root[i]) {
        best[i] = 0.0;
        heap.push(Reverse((0u64, i)));
    }
    let key = |d: f64| (d * 1e12) as u64;
    while let Some(Reverse((d, i))) = heap.pop() {
        if d > key(best[i]) {
            continue;
        }
        for &j in &adjacent[i] {
            let nd = best[i] + (v(rest[i]) - v(rest[j])).norm();
            if nd < best[j] {
                best[j] = nd;
                root_of[j] = root_of[i];
                heap.push(Reverse((key(nd), j)));
            }
        }
    }
    let strips: Vec<Strip> = components(n, tris)
        .into_iter()
        .map(|vertices| {
            // Lower lid: the lashes hang down from their roots.
            let drop: f64 = vertices
                .iter()
                .filter_map(|&i| root_of[i].map(|r| rest[i][1] - rest[r][1]))
                .sum();
            Strip {
                lower: drop < 0.0,
                vertices,
            }
        })
        .collect();
    let mut positions = carried.to_vec();
    let mut lower = vec![false; n];
    let mut reports = vec![];
    for strip in &strips {
        let roots: Vec<usize> = strip
            .vertices
            .iter()
            .copied()
            .filter(|&i| is_root[i])
            .collect();
        if roots.len() < 3 {
            reports.push(json!({"vertices": strip.vertices.len(), "roots": roots.len(), "kept": "no root row"}));
            continue;
        }
        for &i in &strip.vertices {
            lower[i] = strip.lower;
        }
        let mapped: HashMap<usize, (V, Matrix3<f64>)> = roots
            .iter()
            .filter_map(|&r| {
                let (q0, q1, m) = surface_map(&query, body_tris, rest_body, fitted_body, rest[r])?;
                Some((r, (q1 + m * (v(rest[r]) - q0), m)))
            })
            .collect();
        // Margin stretch: spread of the root row about its centroid.
        let spread = |pts: &dyn Fn(usize) -> V| {
            let c = roots.iter().fold(V::zeros(), |s, &r| s + pts(r)) / roots.len() as f64;
            (roots
                .iter()
                .map(|&r| (pts(r) - c).norm_squared())
                .sum::<f64>()
                / roots.len() as f64)
                .sqrt()
        };
        let at = |r: usize| mapped.get(&r).map_or(v(carried[r]), |m| m.0);
        let stretch = (spread(&at) / spread(&|r| v(rest[r])).max(1e-9)).clamp(STRETCH.0, STRETCH.1);
        let length = if strip.lower {
            params.lower_length
        } else {
            params.upper_length
        } * stretch;
        let rotation: HashMap<usize, Matrix3<f64>> = roots
            .iter()
            .map(|&r| {
                let sum = mapped
                    .iter()
                    .filter(|(q, _)| (v(rest[**q]) - v(rest[r])).norm() < SMOOTH_M)
                    .fold(Matrix3::zeros(), |s, (_, m)| s + m.1);
                (r, proper_rotation(sum))
            })
            .collect();
        let (mut before, mut after) = (0.0f64, 0.0f64);
        for &i in &strip.vertices {
            let Some(r) = root_of[i] else { continue };
            let offset = v(rest[i]) - v(rest[r]);
            let p = at(r) + rotation[&r] * offset * length;
            before = before.max((v(carried[i]) - v(carried[r])).norm());
            after = after.max((p - at(r)).norm());
            positions[i] = a(p);
        }
        reports.push(json!({
            "lid": if strip.lower { "lower" } else { "upper" },
            "vertices": strip.vertices.len(),
            "roots": roots.len(),
            "margin_stretch": stretch,
            "length_factor": length,
            "rest_max_length_m": strip.vertices.iter().filter_map(|&i| root_of[i].map(|r| (v(rest[i]) - v(rest[r])).norm())).fold(0.0, f64::max),
            "carried_max_length_m": before,
            "max_length_m": after,
        }));
    }
    Shaped {
        positions,
        lower,
        report: json!({"strips": reports, "params": params}),
    }
}

/// Scales the lash texture's alpha under the lower (and upper) strips' UVs.
/// Skipped when the lash image is shared with another material.
pub fn fade_texture(
    prior: &Prior,
    mesh: usize,
    lower: &[bool],
    materials: &[MaterialData],
    images: &mut [ImageData],
    params: &Lashes,
) -> Result<Value> {
    let prim = &prior.meshes[mesh].prim;
    let Some(index) = prim.material.and_then(|m| materials[m].base_color) else {
        return Ok(json!({"applied": false, "reason": "no lash texture"}));
    };
    let users = materials
        .iter()
        .filter(|m| m.base_color == Some(index))
        .count();
    if users != 1 {
        return Ok(json!({"applied": false, "reason": "lash texture shared"}));
    }
    if (params.lower_opacity - 1.0).abs() < 1e-9 && (params.upper_opacity - 1.0).abs() < 1e-9 {
        return Ok(json!({"applied": false, "reason": "opacity 1"}));
    }
    let img = &mut images[index].decoded;
    let (w, h) = (img.width() as usize, img.height() as usize);
    // 0 = untouched, 1 = upper, 2 = lower; upper wins where cards overlap.
    let mut mask = vec![0u8; w * h];
    for lid_lower in [true, false] {
        for t in prim
            .triangles
            .iter()
            .filter(|t| lower[t[0] as usize] == lid_lower)
        {
            let uv = t.map(|i| {
                let [u, s] = prim.uvs[i as usize];
                [u as f64 * w as f64, s as f64 * h as f64]
            });
            let lo = |k: usize| uv.iter().map(|p| p[k]).fold(f64::INFINITY, f64::min);
            let hi = |k: usize| uv.iter().map(|p| p[k]).fold(f64::NEG_INFINITY, f64::max);
            let area = (uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1])
                - (uv[2][0] - uv[0][0]) * (uv[1][1] - uv[0][1]);
            if area.abs() < 1e-12 {
                continue;
            }
            let x0 = (lo(0).floor().max(0.0) as usize).saturating_sub(1);
            let y0 = (lo(1).floor().max(0.0) as usize).saturating_sub(1);
            let x1 = ((hi(0).ceil() as usize) + 1).min(w);
            let y1 = ((hi(1).ceil() as usize) + 1).min(h);
            for y in y0..y1 {
                for x in x0..x1 {
                    let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                    let edge = |p: [f64; 2], q: [f64; 2]| {
                        ((q[0] - p[0]) * (py - p[1]) - (px - p[0]) * (q[1] - p[1])) / area
                    };
                    // One texel of slack (in barycentric units of the edge).
                    let b = [edge(uv[1], uv[2]), edge(uv[2], uv[0]), edge(uv[0], uv[1])];
                    let slack = |k: usize| {
                        let (p, q) = (uv[(k + 1) % 3], uv[(k + 2) % 3]);
                        ((q[0] - p[0]).hypot(q[1] - p[1])) / area.abs()
                    };
                    if (0..3).all(|k| b[k] >= -slack(k)) {
                        mask[y * w + x] = if lid_lower { 2 } else { 1 };
                    }
                }
            }
        }
    }
    let mut faded = [0usize; 2];
    for (k, p) in img.pixels_mut().enumerate() {
        let scale = match mask[k] {
            1 => params.upper_opacity,
            2 => params.lower_opacity,
            _ => continue,
        };
        faded[mask[k] as usize - 1] += 1;
        p[3] = (p[3] as f64 * scale.clamp(0.0, 1.0)).round() as u8;
    }
    let mut png = vec![];
    image::DynamicImage::ImageRgba8(img.clone())
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)?;
    images[index].bytes = png;
    images[index].mime = "image/png".into();
    Ok(json!({"applied": true, "upper_texels": faded[0], "lower_texels": faded[1]}))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat lid (z = 0) and two lash strips on its margin y = 0: one
    /// fanning up and forward, one hanging down. Each strip: roots at
    /// x = 0..4 mm on the surface, tips 6 mm out.
    type Mesh = (Vec<Vec3>, Vec<[u32; 3]>);

    /// (lash mesh, body mesh)
    fn fixture() -> (Mesh, Mesh) {
        let body = vec![
            [-0.05, -0.05, 0.0],
            [0.05, -0.05, 0.0],
            [0.05, 0.05, 0.0],
            [-0.05, 0.05, 0.0],
        ];
        let body_tris = vec![[0, 1, 2], [0, 2, 3]];
        let mut lash = vec![];
        let mut tris = vec![];
        for dir in [[0.0, 0.004, 0.0045], [0.0, -0.003, 0.005]] {
            let base = lash.len() as u32;
            for i in 0..5 {
                let x = i as f64 * 0.001;
                lash.push([x, 0.0, 0.0]);
                lash.push([x, dir[1], dir[2]]);
            }
            for i in 0..4u32 {
                let (r0, t0, r1, t1) = (
                    base + 2 * i,
                    base + 2 * i + 1,
                    base + 2 * i + 2,
                    base + 2 * i + 3,
                );
                tris.push([r0, r1, t1]);
                tris.push([r0, t1, t0]);
            }
        }
        ((lash, tris), (body, body_tris))
    }

    #[test]
    fn lashes_shortened_about_root_and_bounded() {
        let ((lash, tris), (body, body_tris)) = fixture();
        // The fit doubles the lid and turns it 90 degrees about y.
        let turn = |p: &Vec3| [p[2] * 2.0, p[1] * 2.0, -p[0] * 2.0];
        let fitted: Vec<Vec3> = body.iter().map(turn).collect();
        // The rig refit's surface carry, doubling the offsets too.
        let carried: Vec<Vec3> = lash.iter().map(turn).collect();
        let params = Lashes::default();
        let out = shape_strips(&lash, &tris, &body, &body_tris, &fitted, &carried, &params);
        for (i, p) in lash.iter().enumerate() {
            let lower = i >= 10;
            assert_eq!(out.lower[i], lower, "vertex {i}");
            let root = i & !1;
            let rest_len = (v(*p) - v(lash[root])).norm();
            let len = (v(out.positions[i]) - v(out.positions[root])).norm();
            let factor = if lower {
                params.lower_length
            } else {
                params.upper_length
            };
            // Margin stretch 2 is bounded to STRETCH.1.
            assert!(
                (len - rest_len * factor * STRETCH.1).abs() < 1e-9,
                "vertex {i}: {len}"
            );
            // Roots follow the lid; offsets turn with it.
            let want = v(turn(&lash[root])) + v(turn(p)) / 2.0 * factor * STRETCH.1
                - v(turn(&lash[root])) / 2.0 * factor * STRETCH.1;
            assert!((v(out.positions[i]) - want).norm() < 1e-9, "vertex {i}");
        }
    }

    #[test]
    fn strip_without_root_row_is_kept() {
        let ((mut lash, tris), (body, body_tris)) = fixture();
        for p in &mut lash {
            p[2] += 0.01;
        }
        let out = shape_strips(
            &lash,
            &tris,
            &body,
            &body_tris,
            &body,
            &lash,
            &Lashes::default(),
        );
        assert_eq!(out.positions, lash);
        assert!(out.lower.iter().all(|l| !l));
    }
}
