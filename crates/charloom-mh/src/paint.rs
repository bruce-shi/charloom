 use crate::source::{lab, linear_to_srgb, srgb_to_linear};
use crate::texture::{box_blur, pull_push};
use charloom_core::Vec3;
use rayon::prelude::*;
use serde_json::{Value, json};

/// The body's UV atlas rasterised: per texel its triangle (`u32::MAX`
/// outside the islands) and barycentric coordinates.
pub(crate) struct Atlas<'a> {
    pub w: usize,
    pub h: usize,
    pub owner: &'a [u32],
    pub bary: &'a [[f32; 3]],
    pub triangles: &'a [[u32; 3]],
    pub positions: &'a [Vec3],
    pub uvs: &'a [[f32; 2]],
}

impl Atlas<'_> {
    /// Per triangle: the surface derivatives along texel x and y (metres
    /// per texel), zero for a degenerate UV triangle.
    fn jacobians(&self) -> Vec<[[f64; 3]; 2]> {
        self.triangles
            .par_iter()
            .map(|tri| {
                let u = tri.map(|i| {
                    let p = self.uvs[i as usize];
                    [p[0] as f64 * self.w as f64, p[1] as f64 * self.h as f64]
                });
                let p = tri.map(|i| self.positions[i as usize]);
                let (d1, d2) = (
                    [u[1][0] - u[0][0], u[1][1] - u[0][1]],
                    [u[2][0] - u[0][0], u[2][1] - u[0][1]],
                );
                let det = d1[0] * d2[1] - d2[0] * d1[1];
                if det.abs() < 1e-12 {
                    return [[0.0; 3]; 2];
                }
                let e1: [f64; 3] = std::array::from_fn(|j| p[1][j] - p[0][j]);
                let e2: [f64; 3] = std::array::from_fn(|j| p[2][j] - p[0][j]);
                [
                    std::array::from_fn(|j| (e1[j] * d2[1] - e2[j] * d1[1]) / det),
                    std::array::from_fn(|j| (e2[j] * d1[0] - e1[j] * d2[0]) / det),
                ]
            })
            .collect()
    }

    /// Metres per texel of a triangle with surface derivatives `jac`.
    fn scale(&self, jac: &[[f64; 3]; 2]) -> f64 {
        let [jx, jy] = jac;
        let c = [
            jx[1] * jy[2] - jx[2] * jy[1],
            jx[2] * jy[0] - jx[0] * jy[2],
            jx[0] * jy[1] - jx[1] * jy[0],
        ];
        (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt().sqrt()
    }

    /// Median metres per texel over the texels `keep` selects.
    fn median_scale(&self, keep: impl Fn(usize) -> bool) -> Option<f64> {
        let jac = self.jacobians();
        let mut s: Vec<f64> = (0..self.w * self.h)
            .step_by(3)
            .filter(|&k| self.owner[k] != u32::MAX && keep(k))
            .map(|k| self.scale(&jac[self.owner[k] as usize]))
            .filter(|s| *s > 0.0)
            .collect();
        s.sort_by(f64::total_cmp);
        s.get(s.len() / 2).copied()
    }

    fn position(&self, k: usize) -> [f64; 3] {
        let tri = self.triangles[self.owner[k] as usize];
        let b = self.bary[k];
        std::array::from_fn(|j| {
            (0..3)
                .map(|i| self.positions[tri[i] as usize][j] * b[i] as f64)
                .sum()
        })
    }
}

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Texels of painted hair strands (see the module docs). `colour`: the
/// observed scan colour (sRGB) per texel; `hair`: texels whose scan surface
/// is hair; `protect`: > 0 in the eye surround; `detail`: > 0.5 on the face
/// (sets the texel scale).
pub(crate) fn painted_strands(
    atlas: &Atlas,
    colour: &[Option<[f64; 3]>],
    hair: &[bool],
    protect: &[f32],
    detail: &[f32],
) -> (Vec<bool>, Value) {
    let (w, h) = (atlas.w, atlas.h);
    let n = w * h;
    let jac = atlas.jacobians();
    // Metres per texel on the observed face (MakeHuman gives the face more
    // atlas than the body), else on all observed skin.
    let Some(scale) = atlas
        .median_scale(|k| colour[k].is_some() && detail[k] > 0.5)
        .or_else(|| atlas.median_scale(|k| colour[k].is_some()))
    else {
        return (
            vec![false; n],
            json!({"applied": false, "reason": "no observed skin"}),
        );
    };
    let px = |m: f64| ((m / scale).round() as usize).clamp(1, 32);
    let radius = px(0.003);
    let labs: Vec<Option<[f32; 3]>> = colour
        .par_iter()
        .map(|c| c.map(|c| lab(c).map(|x| x as f32)))
        .collect();
    let lightness: Vec<f32> = labs.iter().map(|c| c.map_or(f32::NAN, |c| c[0])).collect();
    // Morphological closing: dark features narrower than 2·radius vanish.
    let closed = extreme(
        &extreme(&lightness, w, h, radius, true),
        w,
        h,
        radius,
        false,
    );
    let darkness: Vec<f32> = (0..n)
        .map(|k| {
            if lightness[k].is_nan() || closed[k].is_nan() {
                0.0
            } else {
                (closed[k] - lightness[k]).max(0.0)
            }
        })
        .collect();
    let grow = |field: &[bool], r: usize| -> Vec<bool> {
        let f: Vec<f32> = field.iter().map(|x| *x as u8 as f32).collect();
        box_blur(&f, w, h, r).iter().map(|x| *x > 1e-6).collect()
    };
    let protected = grow(
        &protect.iter().map(|p| *p > 0.0).collect::<Vec<_>>(),
        px(0.002),
    );
    let near_hair = grow(hair, px(0.0015));
    let candidate: Vec<bool> = (0..n)
        .map(|k| {
            let Some(c) = labs[k] else { return false };
            // Dark and hair-coloured: not a red lip line or a lid crease.
            darkness[k] > 12.0 && c[1] < c[2] + 10.0 && !protected[k]
        })
        .collect();
    // Line direction: the structure tensor of the darkness field. Across a
    // line its gradient dominates; the line runs perpendicular to it.
    let (mut gxx, mut gxy, mut gyy) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    for y in 1..h.saturating_sub(1) {
        for x in 1..w - 1 {
            let k = y * w + x;
            if atlas.owner[k] == u32::MAX {
                continue;
            }
            let gx = (darkness[k + 1] - darkness[k - 1]) * 0.5;
            let gy = (darkness[k + w] - darkness[k - w]) * 0.5;
            gxx[k] = gx * gx;
            gxy[k] = gx * gy;
            gyy[k] = gy * gy;
        }
    }
    let r2 = radius / 2 + 1;
    let (gxx, gxy, gyy) = (
        box_blur(&gxx, w, h, r2),
        box_blur(&gxy, w, h, r2),
        box_blur(&gyy, w, h, r2),
    );
    // Per candidate: Some(true) along a hanging line, Some(false) along a
    // line across the face (a brow, a lid or lip line), None for no clear
    // direction.
    let direction: Vec<Option<bool>> = (0..n)
        .into_par_iter()
        .map(|k| {
            if !candidate[k] {
                return None;
            }
            let (a, b, c) = (gxx[k] as f64, gxy[k] as f64, gyy[k] as f64);
            let half = ((a - c) * (a - c) * 0.25 + b * b).sqrt();
            let (l1, l2) = ((a + c) * 0.5 + half, (a + c) * 0.5 - half);
            if l1 <= 1e-9 || (l1 - l2) / (l1 + l2) < 0.3 {
                return None;
            }
            // Eigenvector of l1 (across the line), then the line direction.
            let e = if (l1 - a).abs() + b.abs() > 1e-12 {
                [b, l1 - a]
            } else {
                [1.0, 0.0]
            };
            let t = [-e[1], e[0]];
            let [jx, jy] = jac[atlas.owner[k] as usize];
            let d: [f64; 3] = std::array::from_fn(|j| jx[j] * t[0] + jy[j] * t[1]);
            let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            // Y is up: more than 25° from horizontal.
            (norm > 0.0).then(|| d[1].abs() / norm > 0.42)
        })
        .collect();
    let along_down: Vec<bool> = direction.iter().map(|d| *d == Some(true)).collect();
    // Never removed, not even as part of a strand crossing it.
    let across: Vec<bool> = direction.iter().map(|d| *d == Some(false)).collect();
    // Connected candidate regions (8-neighbours).
    let mut seen = vec![false; n];
    let mut strand = vec![false; n];
    let (mut regions, mut removed_regions) = (0usize, 0usize);
    for start in 0..n {
        if !candidate[start] || seen[start] {
            continue;
        }
        regions += 1;
        let mut members = vec![start];
        seen[start] = true;
        let mut i = 0;
        while i < members.len() {
            let k = members[i];
            i += 1;
            let (x, y) = ((k % w) as i64, (k / w) as i64);
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let q = ny as usize * w + nx as usize;
                    if candidate[q] && !seen[q] {
                        seen[q] = true;
                        members.push(q);
                    }
                }
            }
        }
        if !members.iter().any(|&k| near_hair[k]) {
            continue;
        }
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for &k in &members {
            let p = atlas.position(k);
            for j in 0..3 {
                lo[j] = lo[j].min(p[j]);
                hi[j] = hi[j].max(p[j]);
            }
        }
        let extent = (0..3).map(|j| (hi[j] - lo[j]).powi(2)).sum::<f64>().sqrt();
        if extent < 0.006 {
            continue;
        }
        let down = members.iter().filter(|&&k| along_down[k]).count();
        // A mostly hanging strand goes as a whole (its curved bits too);
        // otherwise only its hanging part (a strand across a brow).
        let whole = down as f64 >= 0.7 * members.len() as f64;
        let mut any = false;
        for &k in &members {
            if (whole && !across[k]) || along_down[k] {
                strand[k] = true;
                any = true;
            }
        }
        removed_regions += any as usize;
    }
    // The line's soft edge and its painted highlight: slightly darker or
    // lighter texels next to it (morphological opening for the lighter),
    // never another dark line's texels (a brow it crosses).
    let opened = extreme(
        &extreme(&lightness, w, h, radius, false),
        w,
        h,
        radius,
        true,
    );
    let halo = grow(&strand, (radius / 2).max(1));
    for k in 0..n {
        let bright = if lightness[k].is_nan() || opened[k].is_nan() {
            0.0
        } else {
            lightness[k] - opened[k]
        };
        if halo[k]
            && !strand[k]
            && !candidate[k]
            && (darkness[k] > 6.0 || bright > 4.0)
            && !protected[k]
        {
            strand[k] = true;
        }
    }
    let removed = strand.iter().filter(|x| **x).count();
    (
        strand,
        json!({
            "applied": true,
            "texel_scale_mm": scale * 1000.0,
            "closing_radius_px": radius,
            "candidate_texels": candidate.iter().filter(|x| **x).count(),
            "regions": regions,
            "removed_regions": removed_regions,
            "removed_texels": removed,
        }),
    )
}

 pub(crate) fn edge_shadows(
    atlas: &Atlas,
    colour: &[Option<([f64; 3], f32)>],
    cloth: &[bool],
    skip: &[f32],
) -> (Vec<f32>, Value) {
    let (w, h) = (atlas.w, atlas.h);
    let n = w * h;
    let Some(scale) = atlas.median_scale(|k| colour[k].is_some() && skip[k] < 0.5) else {
        return (vec![1.0; n], json!({"applied": false}));
    };
    let band = ((0.01 / scale).round() as usize).clamp(2, 48);
    let near: Vec<f32> = box_blur(
        &cloth.iter().map(|c| *c as u8 as f32).collect::<Vec<_>>(),
        w,
        h,
        band,
    );
    let lin: Vec<[f32; 3]> = colour
        .iter()
        .map(|o| o.map_or([0.0; 3], |(c, _)| c.map(|x| srgb_to_linear(x) as f32)))
        .collect();
    // Skin away from the edges, extended into the band.
    let interior: Vec<f32> = (0..n)
        .map(|k| match colour[k] {
            Some((_, wt)) if skip[k] < 0.5 && near[k] <= 1e-6 => wt,
            _ => 0.0,
        })
        .collect();
    let reference = pull_push(&lin, &interior, w, h);
    let factor: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|k| match colour[k] {
            Some((c, _)) if near[k] > 1e-6 && skip[k] < 0.5 => {
                let r = lab(reference[k].map(|x| linear_to_srgb(x as f64)));
                let dl = (r[0] - lab(c)[0]) as f32;
                1.0 - smooth((dl - 4.0) / 8.0)
            }
            _ => 1.0,
        })
        .collect();
    let shaded = factor.iter().filter(|f| **f < 0.5).count();
    (
        factor,
        json!({"applied": true, "band_px": band, "shadow_texels": shaded}),
    )
}

/// Separable square max (`max`) or min filter of radius `r`, ignoring NaN
/// (NaN where the whole window is NaN).
fn extreme(src: &[f32], w: usize, h: usize, r: usize, max: bool) -> Vec<f32> {
    let pick = |a: f32, b: f32| {
        if a.is_nan() {
            b
        } else if b.is_nan() {
            a
        } else if max {
            a.max(b)
        } else {
            a.min(b)
        }
    };
    let mut tmp = vec![f32::NAN; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let line = &src[y * w..(y + 1) * w];
        for (x, out) in row.iter_mut().enumerate() {
            let (lo, hi) = (x.saturating_sub(r), (x + r).min(w - 1));
            *out = line[lo..=hi].iter().fold(f32::NAN, |acc, &v| pick(acc, v));
        }
    });
    let mut out = vec![f32::NAN; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let (lo, hi) = (y.saturating_sub(r), (y + r).min(h - 1));
        for (x, o) in row.iter_mut().enumerate() {
            *o = (lo..=hi).fold(f32::NAN, |acc, yy| pick(acc, tmp[yy * w + x]));
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Positions, UVs, triangles, texel owners and barycentrics.
    type Plane = (
        Vec<Vec3>,
        Vec<[f32; 2]>,
        Vec<[u32; 3]>,
        Vec<u32>,
        Vec<[f32; 3]>,
    );

    /// A flat square atlas of skin, 0.25 mm per texel, facing +Z (Y up).
    fn plane(w: usize) -> Plane {
        let s = w as f64 * 0.00025;
        let positions = vec![[0.0, 0.0, 0.0], [s, 0.0, 0.0], [0.0, s, 0.0], [s, s, 0.0]];
        // Texel y grows downwards in the atlas, world Y upwards.
        let uvs = vec![[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]];
        let triangles = vec![[0, 1, 2], [1, 3, 2]];
        let mut owner = vec![u32::MAX; w * w];
        let mut bary = vec![[0f32; 3]; w * w];
        for y in 0..w {
            for x in 0..w {
                let p = [(x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / w as f64];
                for (t, tri) in triangles.iter().enumerate() {
                    let q = tri.map(|i| uvs[i as usize].map(|c| c as f64));
                    let area = (q[1][0] - q[0][0]) * (q[2][1] - q[0][1])
                        - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1]);
                    let l1 = ((q[2][0] - p[0]) * (q[0][1] - p[1])
                        - (q[0][0] - p[0]) * (q[2][1] - p[1]))
                        / area;
                    let l2 = ((q[0][0] - p[0]) * (q[1][1] - p[1])
                        - (q[1][0] - p[0]) * (q[0][1] - p[1]))
                        / area;
                    let l0 = 1.0 - l1 - l2;
                    if l0 >= -1e-9 && l1 >= -1e-9 && l2 >= -1e-9 {
                        owner[y * w + x] = t as u32;
                        bary[y * w + x] = [l0 as f32, l1 as f32, l2 as f32];
                        break;
                    }
                }
            }
        }
        (positions, uvs, triangles, owner, bary)
    }

    /// A dark vertical strand from hair at the top is removed; a dark
    /// horizontal brow and a thick dark patch (a beard) of the same colour
    /// stay, and so does a strand that never touches hair.
    #[test]
    fn hanging_dark_lines_touching_hair_are_strands() {
        let w = 160;
        let (positions, uvs, triangles, owner, bary) = plane(w);
        let atlas = Atlas {
            w,
            h: w,
            owner: &owner,
            bary: &bary,
            triangles: &triangles,
            positions: &positions,
            uvs: &uvs,
        };
        let skin = [0.85, 0.65, 0.52];
        let dark = [0.25, 0.17, 0.12];
        let mut colour = vec![Some(skin); w * w];
        let mut hair = vec![false; w * w];
        for y in 0..10 {
            for x in 0..w {
                colour[y * w + x] = None;
                hair[y * w + x] = true;
            }
        }
        let paint = |c: &mut Vec<Option<[f64; 3]>>, x0: usize, x1: usize, y0: usize, y1: usize| {
            for y in y0..y1 {
                for x in x0..x1 {
                    c[y * w + x] = Some(dark);
                }
            }
        };
        paint(&mut colour, 30, 36, 10, 90); // strand from the hairline
        paint(&mut colour, 60, 120, 60, 66); // brow
        paint(&mut colour, 70, 130, 100, 150); // beard
        paint(&mut colour, 140, 146, 80, 150); // line away from hair
        let none = vec![0.0; w * w];
        let face = vec![1.0; w * w];
        let (strand, report) = painted_strands(&atlas, &colour, &hair, &none, &face);
        let at = |x: usize, y: usize| strand[y * w + x];
        assert!(at(32, 50) && at(33, 20), "{report}");
        assert!(!at(90, 63), "brow");
        assert!(!at(100, 125), "beard");
        assert!(!at(143, 120), "line not touching hair");
        // The eye surround is protected.
        let mut eye = vec![0.0; w * w];
        for y in 0..w {
            for x in 20..50 {
                eye[y * w + x] = 1.0;
            }
        }
        let (strand, _) = painted_strands(&atlas, &colour, &hair, &eye, &face);
        assert!(!strand[50 * w + 32]);
    }

    /// Darker samples next to a garment are dropped; samples there like the
    /// skin further in are kept.
    #[test]
    fn edge_shadows_drop_dark_samples_by_garments() {
        let w = 64;
        let (positions, uvs, triangles, owner, bary) = plane(w);
        let atlas = Atlas {
            w,
            h: w,
            owner: &owner,
            bary: &bary,
            triangles: &triangles,
            positions: &positions,
            uvs: &uvs,
        };
        let skin = [0.9, 0.7, 0.55];
        let shadow = [0.65, 0.48, 0.35];
        let colour: Vec<Option<([f64; 3], f32)>> = (0..w * w)
            .map(|k| match k % w {
                0..4 => None,
                4..8 => Some((shadow, 1.0)),
                _ => Some((skin, 1.0)),
            })
            .collect();
        let cloth: Vec<bool> = (0..w * w).map(|k| k % w < 4).collect();
        let (f, report) = edge_shadows(&atlas, &colour, &cloth, &vec![0.0; w * w]);
        let r = 30 * w;
        assert!(
            f[r + 5] < 0.1 && f[r + 12] > 0.9 && f[r + 50] == 1.0,
            "{report}"
        );
        // On the face nothing is dropped.
        let (f, _) = edge_shadows(&atlas, &colour, &cloth, &vec![1.0; w * w]);
        assert!(f.iter().all(|x| *x == 1.0));
    }
}
