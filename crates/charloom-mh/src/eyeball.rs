//! MakeHuman's eyeball dressed from the scan without projecting it.
//!
//! The eye is a ball inside the lids: MakeHuman's clean eyeball texture
//! (sclera, round iris, pupil) stays on the unchanged sphere. Projecting the
//! scan's painted eye onto it (`Config::eye_projection`) also carried the lid
//! lines, eyeliner, lid shadow and white rims onto the ball, where they slid
//! when the eye turned. From the scan only:
//! - the iris colour (hue and brightness of the painted iris ring), keeping
//!   MakeHuman's iris pattern;
//! - the iris size against the opening (the iris disc scaled in the eyeball
//!   texture, bounded: stylised big-iris eyes scale more);
//! - the gaze: the eye bone's rest rotation turns the iris centre towards the
//!   painted iris centre (bounded).
//!
//! Both eyes are one mirrored pair: same size and colour, mirrored gaze.
use crate::geom::{self, V, rotation_between};
use crate::prior::{PartKind, Prior};
use crate::{MarkedOpening, Opening, fit, scan, source};
use anyhow::Result;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Largest iris scale either way, and the largest gaze turn.
const IRIS_SCALE: (f64, f64) = (0.8, 1.6);
const GAZE_MAX_RAD: f64 = 0.3;

 #[derive(Clone, Copy)]
struct ScanIris {
    s: f64,
    t: f64,
    radius: f64,
    colour: [f64; 3],
}

fn lum(c: &[f64; 3]) -> f64 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn percentile(mut x: Vec<f64>, q: f64) -> f64 {
    x.sort_by(f64::total_cmp);
    x[((x.len() - 1) as f64 * q).round() as usize]
}

/// Samples the scan inside the marked opening on a face-plane grid (rays
/// from just behind the painted surface, along the opening's normal) and
/// finds the iris: the pupil (darkest samples) and the dark run around it
/// along its row and column. Shares of the whole opening took shaded corners
/// and lid shadow too (a 26 mm iris on a realistic eye).
fn measure(scan: &scan::Scan, m: &MarkedOpening) -> Option<(ScanIris, usize)> {
    let n = m.u.cross(&m.up).normalize();
    let n = if n.z < 0.0 { -n } else { n };
    let (first, last) = (m.opening.upper.first()?.0, m.opening.upper.last()?.0);
    const N: usize = 48;
    // grid[a][b]: column a (s), row b (t from the lower to the upper lid).
    let mut grid: Vec<Vec<Option<(f64, f64, f64, [f64; 3])>>> = vec![vec![None; N + 1]; N + 1];
    for (a0, column) in grid.iter_mut().enumerate() {
        let s = first + (last - first) * a0 as f64 / N as f64;
        let (Some(up), Some(lo)) = (
            Opening::margin(&m.opening.upper, s),
            Opening::margin(&m.opening.lower, s),
        ) else {
            continue;
        };
        for (b0, cell) in column.iter_mut().enumerate() {
            let t = lo + (up - lo) * b0 as f64 / N as f64;
            let depth = m.opening.depth(s, t);
            let p = m.centre + m.u * s + m.up * t;
            let origin = p - n * 0.015;
            let Some(d) = scan
                .query
                .first_ray_distance(geom::a(origin), geom::a(n), 0.04)
            else {
                continue;
            };
            let Some(h) = scan
                .query
                .closest_point(geom::a(origin + n * d), 0.002, None)
            else {
                continue;
            };
            if let Some(c) = scan.color_at(h.source[1], h.barycentric) {
                *cell = Some((s, t, depth, c));
            }
        }
    }
    // Away from the lid margins (eyeliner, lash line, lid shadow).
    let inside: Vec<(usize, usize, (f64, f64, f64, [f64; 3]))> = grid
        .iter()
        .enumerate()
        .flat_map(|(a0, col)| {
            col.iter()
                .enumerate()
                .filter_map(move |(b0, c)| c.filter(|x| x.2 > 0.2).map(|x| (a0, b0, x)))
        })
        .collect();
    if inside.len() < 40 {
        return None;
    }
    let lums: Vec<f64> = inside.iter().map(|x| lum(&x.2.3)).collect();
    let (light, darkest) = (percentile(lums.clone(), 0.9), percentile(lums, 0.05));
    let mean = |keep: &dyn Fn(f64) -> bool| -> [f64; 3] {
        let sel: Vec<[f64; 3]> = inside
            .iter()
            .map(|x| x.2.3)
            .filter(|c| keep(lum(c)))
            .collect();
        std::array::from_fn(|k| sel.iter().map(|c| c[k]).sum::<f64>() / sel.len().max(1) as f64)
    };
    let (sclera, pupil) = (mean(&|l| l >= light), mean(&|l| l <= darkest));
    let dist =
        |c: &[f64; 3], d: &[f64; 3]| ((0..3).map(|k| (c[k] - d[k]).powi(2)).sum::<f64>()).sqrt();
    // Iris: further from the white's colour than a third of the way to the
    // pupil (sRGB). Luminance alone dropped a light brown iris ring (a
    // stylised big iris measured as its dark centre).
    let span = dist(&pupil, &sclera);
    let is_iris = |c: &[f64; 3]| dist(c, &sclera) > 0.35 * span;
    let dark = |a0: usize, b0: usize| -> Option<bool> {
        grid.get(a0)?
            .get(b0)?
            .filter(|x| x.2 > 0.2)
            .map(|x| is_iris(&x.3))
    };
    // Per column: the length of iris (metres) across the opening there.
    // Counts of samples weighed every column alike, and a shaded corner,
    // short but all dark, outweighed the iris.
    let height = |a0: usize| -> f64 {
        let s = first + (last - first) * a0 as f64 / N as f64;
        match (
            Opening::margin(&m.opening.upper, s),
            Opening::margin(&m.opening.lower, s),
        ) {
            (Some(up), Some(lo)) => (up - lo).max(0.0),
            _ => 0.0,
        }
    };
    let length: Vec<f64> = (0..=N)
        .map(|a0| {
            let n = (0..=N).filter(|&b0| dark(a0, b0) == Some(true)).count();
            n as f64 / (N + 1) as f64 * height(a0)
        })
        .collect();
    let smooth = |a0: usize| -> f64 {
        (a0.saturating_sub(1)..=(a0 + 1).min(N))
            .map(|q| length[q])
            .sum()
    };
     
    let pa = (0..=N).max_by(|&a0, &b0| {
        smooth(a0)
            .total_cmp(&smooth(b0))
            .then(b0.abs_diff(N / 2).cmp(&a0.abs_diff(N / 2)))
    })?;
    if length[pa] <= 0.0 {
        return None;
    }
    // The iris's columns: outwards from there while a column holds at least
    // 30 % of that iris length (one weaker column is bridged). A dark run
    // along one row stopped at the painted catchlight inside a stylised
    // iris: a 3 mm iris measured in a 7 mm one, shrunk and set off-centre.
    let enough = |a0: isize| -> bool {
        (0..=N as isize).contains(&a0) && length[a0 as usize] >= 0.3 * length[pa]
    };
    let reach = |dir: isize| -> isize {
        let mut q = 0isize;
        loop {
            let a0 = pa as isize + dir * (q + 1);
            if enough(a0) {
                q += 1;
            } else if enough(a0 + dir) {
                q += 2;
            } else {
                return q;
            }
        }
    };
    let (left, right) = (pa as isize - reach(-1), pa as isize + reach(1));
    let s_of = |a0: isize| first + (last - first) * a0 as f64 / N as f64;
    let cell = (last - first) / N as f64;
    let (sc, radius) = (
        (s_of(left) + s_of(right)) / 2.0,
        (s_of(right) - s_of(left)) / 2.0 + cell * 0.5,
    );
    // Vertically, in the column through the iris centre: its highest and
    // lowest iris samples, cut where they reach a lid (the last sample
    // inside the opening there).
    let ac = (((sc - first) / cell).round() as usize).min(N);
    let ac = if (0..=N).any(|b0| dark(ac, b0) == Some(true)) {
        ac
    } else {
        pa
    };
    let inside_rows: Vec<usize> = (0..=N).filter(|&b0| dark(ac, b0).is_some()).collect();
    let rows: Vec<usize> = (0..=N).filter(|&b0| dark(ac, b0) == Some(true)).collect();
    let (Some(&b_lo), Some(&b_hi)) = (rows.first(), rows.last()) else {
        return None;
    };
    let (top_cut, bottom_cut) = (
        Some(&b_hi) == inside_rows.last(),
        Some(&b_lo) == inside_rows.first(),
    );
    let t_at = |b0: usize| grid[ac][b0].map(|x| x.1);
    let (t0, t1) = (t_at(b_lo)?, t_at(b_hi)?);
    let tc = match (top_cut, bottom_cut) {
        (false, false) => (t0 + t1) / 2.0,
        (true, false) => t0 + radius,
        (false, true) => t1 - radius,
        // Both lids over the iris: the middle of what shows.
        (true, true) => (t0 + t1) / 2.0,
    };
    // Iris ring colour: the dark samples within the iris, without the
    // pupil (darkest third) and highlights.
    let mut ring: Vec<[f64; 3]> = inside
        .iter()
        .filter(|x| {
            is_iris(&x.2.3) && ((x.2.0 - sc).powi(2) + (x.2.1 - tc).powi(2)).sqrt() < radius
        })
        .map(|x| x.2.3.map(source::srgb_to_linear))
        .collect();
    if ring.len() < 10 {
        return None;
    }
    ring.sort_by(|a0, b0| lum(a0).total_cmp(&lum(b0)));
    let n0 = ring.len();
    let ring = &ring[n0 / 3..(n0 * 9 / 10).max(n0 / 3 + 1)];
    let colour = std::array::from_fn(|k| percentile(ring.iter().map(|c| c[k]).collect(), 0.5));
    Some((
        ScanIris {
            s: sc,
            t: tc,
            radius,
            colour,
        },
        n0,
    ))
}

/// One eyeball (side) of MakeHuman's eye mesh: its triangles, and UV ↔ 3D.
struct Ball<'a> {
    uvs: &'a [[f32; 2]],
    positions: &'a [charloom_core::Vec3],
    tris: Vec<[u32; 3]>,
}

impl Ball<'_> {
    /// The 3D point at a UV position (inside one of the ball's triangles).
    fn at(&self, uv: [f64; 2]) -> Option<V> {
        for t in &self.tris {
            let q = t.map(|i| {
                let p = self.uvs[i as usize];
                [p[0] as f64, p[1] as f64]
            });
            let area = (q[1][0] - q[0][0]) * (q[2][1] - q[0][1])
                - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1]);
            if area.abs() < 1e-14 {
                continue;
            }
            let l1 = ((q[2][0] - uv[0]) * (q[0][1] - uv[1])
                - (q[0][0] - uv[0]) * (q[2][1] - uv[1]))
                / area;
            let l2 = ((q[0][0] - uv[0]) * (q[1][1] - uv[1])
                - (q[1][0] - uv[0]) * (q[0][1] - uv[1]))
                / area;
            let l0 = 1.0 - l1 - l2;
            if l0 >= -1e-9 && l1 >= -1e-9 && l2 >= -1e-9 {
                let p = |k: usize| geom::v(self.positions[t[k] as usize]);
                return Some(p(0) * l0 + p(1) * l1 + p(2) * l2);
            }
        }
        None
    }

    fn uv_box(&self) -> ([f64; 2], [f64; 2]) {
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        for t in &self.tris {
            for &i in t {
                let p = self.uvs[i as usize];
                for k in 0..2 {
                    lo[k] = lo[k].min(p[k] as f64);
                    hi[k] = hi[k].max(p[k] as f64);
                }
            }
        }
        (lo, hi)
    }
}

 pub fn dress(
    prior: &Prior,
    rig: &mut fit::Rig,
    images: &mut [source::ImageData],
    scan: &scan::Scan,
    landmarks: &BTreeMap<String, charloom_core::Vec3>,
) -> Result<Value> {
    let Some((mi, eyes)) = prior
        .meshes
        .iter()
        .enumerate()
        .find(|(_, m)| m.kind == PartKind::Eyes)
    else {
        return Ok(Value::Null);
    };
    let materials = source::read_materials(&prior.loaded)?;
    let Some(index) = eyes.prim.material.and_then(|m| materials[m].base_color) else {
        return Ok(Value::Null);
    };
    // The painted irises, as one mirrored pair.
    let mut found = vec![];
    for side in ["l", "r"] {
        if let Some(m) = MarkedOpening::new(landmarks, side)
            && let Some((iris, n)) = measure(scan, &m)
        {
            found.push((side, m, iris, n));
        }
    }
    if found.is_empty() {
        return Ok(Value::Null);
    }
    let k = found.len() as f64;
    let pair = ScanIris {
        s: found.iter().map(|f| f.2.s).sum::<f64>() / k,
        t: found.iter().map(|f| f.2.t).sum::<f64>() / k,
        radius: found.iter().map(|f| f.2.radius).sum::<f64>() / k,
        colour: std::array::from_fn(|c| found.iter().map(|f| f.2.colour[c]).sum::<f64>() / k),
    };
    // MakeHuman's iris discs in the texture (one per eyeball island).
    let img = &mut images[index].decoded;
    let (w, h) = (img.width() as usize, img.height() as usize);
    let px_lum =
        |p: &image::Rgba<u8>| lum(&[0, 1, 2].map(|k| source::srgb_to_linear(p[k] as f64 / 255.0)));
    let sclera_lum = percentile(img.pixels().map(px_lum).collect(), 0.9);
    let sclera_linear: [f64; 3] = {
        let light: Vec<[f64; 3]> = img
            .pixels()
            .filter(|p| px_lum(p) >= sclera_lum)
            .map(|p| [0, 1, 2].map(|k| source::srgb_to_linear(p[k] as f64 / 255.0)))
            .collect();
        std::array::from_fn(|k| light.iter().map(|c| c[k]).sum::<f64>() / light.len().max(1) as f64)
    };
    let is_iris = |p: &image::Rgba<u8>| px_lum(p) < sclera_lum * 0.3;
    let mut report = serde_json::Map::new();
    let mut discs = vec![];
    for (side, joint) in [("l", "eye.L"), ("r", "eye.R")] {
        let Some(&j) = prior.by_name.get(joint) else {
            continue;
        };
        let on: Vec<bool> = eyes
            .influences
            .iter()
            .map(|row| row.iter().any(|x| x.joint == j && x.weight > 0.5))
            .collect();
        let ball = Ball {
            uvs: &eyes.prim.uvs,
            positions: &rig.meshes[mi],
            tris: eyes
                .prim
                .triangles
                .iter()
                .filter(|t| t.iter().all(|&i| on[i as usize]))
                .copied()
                .collect(),
        };
        if ball.tris.is_empty() {
            continue;
        }
        let (lo, hi) = ball.uv_box();
        let (mut sum, mut n) = ([0.0f64; 2], 0usize);
        for y in (lo[1] * h as f64) as usize..((hi[1] * h as f64) as usize).min(h) {
            for x in (lo[0] * w as f64) as usize..((hi[0] * w as f64) as usize).min(w) {
                if is_iris(img.get_pixel(x as u32, y as u32)) {
                    sum[0] += x as f64 + 0.5;
                    sum[1] += y as f64 + 0.5;
                    n += 1;
                }
            }
        }
        if n < 20 {
            continue;
        }
        let c_px = [sum[0] / n as f64, sum[1] / n as f64];
        let r_px = (n as f64 / std::f64::consts::PI).sqrt();
        let c_uv = [c_px[0] / w as f64, c_px[1] / h as f64];
        let Some(centre_3d) = ball.at(c_uv) else {
            continue;
        };
        // World iris radius: mean chord to points on the disc's edge.
        let edge: Vec<f64> = (0..12)
            .filter_map(|q| {
                let a0 = std::f64::consts::TAU * q as f64 / 12.0;
                ball.at([
                    c_uv[0] + r_px / w as f64 * a0.cos(),
                    c_uv[1] + r_px / h as f64 * a0.sin(),
                ])
                .map(|p| (p - centre_3d).norm())
            })
            .collect();
        if edge.len() < 6 {
            continue;
        }
        let r_world = edge.iter().sum::<f64>() / edge.len() as f64;
        discs.push((side, j, c_px, r_px, r_world, centre_3d));
    }
    if discs.is_empty() {
        return Ok(Value::Null);
    }
    // One iris scale for the pair.
    let r_mh = discs.iter().map(|d| d.4).sum::<f64>() / discs.len() as f64;
    let scale = (pair.radius / r_mh).clamp(IRIS_SCALE.0, IRIS_SCALE.1);
    let source_img = img.clone();
    for &(_, _, c, r, _, _) in &discs {
         
        let (a0, b0) = (r * scale * 1.06, r * scale * 1.06 + r * 0.5);
        let reach = b0.ceil() as i64 + 2;
        for y in (c[1] as i64 - reach).max(0)..(c[1] as i64 + reach).min(h as i64) {
            for x in (c[0] as i64 - reach).max(0)..(c[0] as i64 + reach).min(w as i64) {
                let d = V::new(x as f64 + 0.5 - c[0], y as f64 + 0.5 - c[1], 0.0);
                let dist = d.norm();
                if dist >= b0 {
                    continue;
                }
                let from = if dist <= a0 {
                    dist / scale
                } else {
                    let f = (dist - a0) / (b0 - a0);
                    (a0 / scale) * (1.0 - f) + b0 * f
                };
                let q = if dist > 1e-9 { d * (from / dist) } else { d };
                let uv = [(c[0] + q.x) / w as f64, (c[1] + q.y) / h as f64];
                let s = source::sample(&source_img, uv);
                let p = img.get_pixel_mut(x as u32, y as u32);
                for k in 0..3 {
                    p[k] = (s[k] * 255.0).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    // The scan's iris colour, MakeHuman's iris pattern (relative luminance).
    let target = pair.colour;
    let (mut lin_sum, mut n) = ([0.0f64; 3], 0usize);
    for p in img.pixels().filter(|p| is_iris(p)) {
        for k in 0..3 {
            lin_sum[k] += source::srgb_to_linear(p[k] as f64 / 255.0);
        }
        n += 1;
    }
    if n > 0 {
        let mean_lum = lum(&lin_sum) / n as f64;
        // The iris ring is the brighter part of the dark disc: its pattern
        // relative to the ring's own mean, the pupil stays dark.
        for p in img.pixels_mut() {
            if !is_iris(p) {
                continue;
            }
            let lin: [f64; 3] =
                std::array::from_fn(|k| source::srgb_to_linear(p[k] as f64 / 255.0));
            let rel = lum(&lin) / mean_lum.max(1e-6);
            for k in 0..3 {
                p[k] = (source::linear_to_srgb(target[k] * rel.min(3.0)) * 255.0).round() as u8;
            }
        }
    }
    let mut png = vec![];
    image::DynamicImage::ImageRgba8(img.clone())
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)?;
    images[index].bytes = png;
    images[index].mime = "image/png".into();
    // Gaze: the iris centre turned towards the painted iris centre (the same
    // opening coordinates on both sides: mirrored).
    for &(side, j, _, _, _, centre_3d) in &discs {
        let Some(m) = MarkedOpening::new(landmarks, side) else {
            continue;
        };
        let o = rig.worlds[j].fixed_view::<3, 1>(0, 3).into_owned();
        let target = m.centre + m.u * pair.s + m.up * pair.t;
        let (d0, d1) = ((centre_3d - o).normalize(), (target - o).normalize());
        let angle = d0.angle(&d1);
        let g = if angle > GAZE_MAX_RAD {
            let axis = nalgebra::Unit::new_normalize(d0.cross(&d1));
            nalgebra::Rotation3::from_axis_angle(&axis, GAZE_MAX_RAD).into_inner()
        } else {
            rotation_between(d0, d1)
        };
        fit::turn_joint(prior, rig, j, &g, mi);
        let measured = found.iter().find(|f| f.0 == side);
        report.insert(
            side.into(),
            json!({
                "scan_iris_centre_st_m": measured.map(|f| [f.2.s, f.2.t]),
                "scan_iris_radius_m": measured.map(|f| f.2.radius),
                "iris_samples": measured.map(|f| f.3),
                "makehuman_iris_radius_m": discs.iter().find(|d| d.0 == side).map(|d| d.4),
                "gaze_turn_rad": angle.min(GAZE_MAX_RAD),
                "gaze_wanted_rad": angle,
            }),
        );
    }
    Ok(json!({
        "method": "makehuman_eyeball_dressed",
        "pair": {"iris_centre_st_m": [pair.s, pair.t], "iris_radius_m": pair.radius, "iris_linear": pair.colour},
        "iris_scale": scale,
        "iris_scale_wanted": pair.radius / r_mh,
        "sides": report,
        "sclera_linear": sclera_linear,
    }))
}
