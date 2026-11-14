 use crate::geom::{V, a, v};
use crate::scan::Scan;
use crate::segment::Label;
use crate::source::{linear_to_srgb, sample, srgb_to_linear};
use charloom_core::Vec3;
use image::{Rgba, RgbaImage};
use rayon::prelude::*;
use serde_json::{Value, json};

pub struct Bake {
    pub image: RgbaImage,
         pub provenance: RgbaImage,
    pub report: Value,
}

pub struct BakeInput<'a> {
    pub scan: &'a Scan,
    /// Scan label per original scan vertex.
    pub labels: &'a [Label],
    /// Fitted body (rest) positions, normals, UVs and triangles.
    pub positions: &'a [Vec3],
    pub normals: &'a [Vec3],
    pub uvs: &'a [[f32; 2]],
    pub triangles: &'a [[u32; 3]],
    /// MakeHuman base-colour texture used where the scan shows no skin.
    pub original: Option<&'a RgbaImage>,
    /// Per body vertex in [0, 1]: 1 keeps the scan texture out (eye sockets).
    pub exclude: &'a [f64],
         pub eye_surround: &'a [f64],
    /// Per body vertex in [0, 1]: socket lining inside a marked eye opening
    /// (seen past the eyeball's edge in the corners), with the scan's sclera
    /// colour (linear) it takes. Empty / None: none.
    pub eye_lining: &'a [f64],
    pub sclera: Option<[f64; 3]>,
    /// Per body vertex: brow zone weight [left, right] (around MakeHuman's
    /// eyebrows), and the vertex's mirror partner. Where one brow is
    /// clearly less observed than the other (hair hanging over it), the
    /// clean brow's texels are mirrored onto it. Empty: none.
    pub brows: &'a [[f64; 2]],
    pub mirror: &'a [u32],
         pub marked_brow: &'a [bool],
    /// Per scan vertex sampling weight (see [`skin_sample_weights`]).
    pub scan_weight: &'a [f32],
    /// Per body vertex: 1 where painted detail is real (face), so outlier
    /// rejection does not erase lips, brows or freckles.
    pub detail: &'a [f64],
    /// Per body vertex: head cavity. Exposed cavity skin (a nostril wall
    /// outside a smaller scan nose) takes the nearby scan skin colour.
    pub cavity: &'a [bool],
    /// Per body vertex in [0, 1]: skin the scan cannot show however close a
    /// skin-labelled surface is (a foot inside a shoe): its samples are
    /// dropped by this share, so a mislabelled shoe patch neither paints nor
    /// tones the foot. Empty: none.
    pub hidden: &'a [f64],
         pub scalp: &'a [f64],
    pub size: u32,
         pub reach: f64,
}

pub fn bake(input: &BakeInput) -> Bake {
    let BakeInput {
        scan,
        labels,
        positions,
        normals,
        uvs,
        triangles,
        original,
        exclude,
        eye_surround,
        eye_lining,
        sclera,
        brows,
        mirror,
        marked_brow,
        scan_weight,
        detail,
        cavity,
        hidden,
        scalp,
        size,
        reach,
    } = *input;
    let (w, h) = (size as usize, size as usize);
     
    let mut owner = vec![u32::MAX; w * h];
    let mut bary = vec![[0f32; 3]; w * h];
    for (t, tri) in triangles.iter().enumerate() {
        let uv = tri.map(|i| {
            let p = uvs[i as usize];
            [p[0] as f64 * w as f64, p[1] as f64 * h as f64]
        });
        let area = (uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1])
            - (uv[2][0] - uv[0][0]) * (uv[1][1] - uv[0][1]);
        if area.abs() < 1e-12 {
            continue;
        }
        let x0 = uv
            .iter()
            .map(|p| p[0])
            .fold(f64::INFINITY, f64::min)
            .floor()
            .max(0.0) as usize;
        let x1 = (uv
            .iter()
            .map(|p| p[0])
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil() as usize)
            .min(w - 1);
        let y0 = uv
            .iter()
            .map(|p| p[1])
            .fold(f64::INFINITY, f64::min)
            .floor()
            .max(0.0) as usize;
        let y1 = (uv
            .iter()
            .map(|p| p[1])
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil() as usize)
            .min(h - 1);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                let l1 =
                    ((uv[2][0] - px) * (uv[0][1] - py) - (uv[0][0] - px) * (uv[2][1] - py)) / area;
                let l2 =
                    ((uv[0][0] - px) * (uv[1][1] - py) - (uv[1][0] - px) * (uv[0][1] - py)) / area;
                let l0 = 1.0 - l1 - l2;
                if l0 >= -1e-6 && l1 >= -1e-6 && l2 >= -1e-6 {
                    owner[y * w + x] = t as u32;
                    bary[y * w + x] = [l0 as f32, l1 as f32, l2 as f32];
                }
            }
        }
    }
    // Sample the scan at every covered texel.
    // Per-texel exclusion (eye sockets: MakeHuman draws real eyes there).
    let excluded: Vec<f32> = (0..w * h)
        .into_par_iter()
        .map(|k| {
            let t = owner[k];
            if t == u32::MAX {
                return 0.0;
            }
            let tri = triangles[t as usize];
            (0..3)
                .map(|j| exclude[tri[j] as usize] as f32 * bary[k][j])
                .sum::<f32>()
                .clamp(0.0, 1.0)
        })
        .collect();
    let surround_eye: Vec<f32> = if eye_surround.is_empty() {
        vec![0.0; w * h]
    } else {
        (0..w * h)
            .into_par_iter()
            .map(|k| {
                let t = owner[k];
                if t == u32::MAX {
                    return 0.0;
                }
                let tri = triangles[t as usize];
                (0..3)
                    .map(|j| eye_surround[tri[j] as usize] as f32 * bary[k][j])
                    .sum::<f32>()
                    .clamp(0.0, 1.0)
            })
            .collect()
    };
    let lining: Vec<f32> = match sclera {
        Some(_) if !eye_lining.is_empty() => (0..w * h)
            .into_par_iter()
            .map(|k| {
                let t = owner[k];
                if t == u32::MAX {
                    return 0.0;
                }
                let tri = triangles[t as usize];
                (0..3)
                    .map(|j| eye_lining[tri[j] as usize] as f32 * bary[k][j])
                    .sum::<f32>()
                    .clamp(0.0, 1.0)
            })
            .collect(),
        _ => vec![0.0; w * h],
    };
    let sclera_srgb = sclera.map(|c| c.map(linear_to_srgb));
    // 1. Feathered sampling: weight from the scan (distance to garment/hair
    //    edges, LLM cleanup areas), zero for non-skin.
    //    Projection as a selected-to-active bake does it: a ray along the
    //    body normal from a cage `cage` outside the body, inward up to
    //    `ray` long, takes the first scan surface facing it. Closest point
    //    folds many texels onto one ridge where the body runs beside a
    //    curved scan (temples, lids, the nose sides, under a hairline) and
    //    smears it; a normal ray keeps the scan's layout. Where the first
    //    surface is hair or cloth the texel is occluded in the scan, so no
    //    skin is looked for behind or beside it. Closest point is only the
    //    fallback where the ray finds no surface at all.
    //    For a 1.7 m figure (`reach` 34 mm): 25 mm cage, 60 mm ray.
    let cage = reach * 0.75;
    let ray = reach * 1.75;
    // Third field: the scan surface found is hair (a strand over the skin,
    // the hairline; bit 1) or cloth (bit 2), for crate::paint.
    let sampled: Vec<(Option<Sample>, Sampled, u8)> = (0..w * h)
        .into_par_iter()
        .map(|k| {
            let t = owner[k];
            if t == u32::MAX {
                return (None, Sampled::Skipped, 0);
            }
            let tri = triangles[t as usize];
            let b = bary[k];
            let p = (0..3).fold(V::zeros(), |s, j| {
                s + v(positions[tri[j] as usize]) * b[j] as f64
            });
            let n = (0..3).fold(V::zeros(), |s, j| {
                s + v(normals[tri[j] as usize]) * b[j] as f64
            });
            let in_cavity = tri.iter().filter(|&&i| cavity[i as usize]).count() >= 2;
            let (triangle, barycentric, kind) = if in_cavity {
                // Cavity normals face inward: no orientation gate, short reach.
                match scan.query.closest_point(a(p), reach.min(0.004), None) {
                    Some(h) => (h.source[1], h.barycentric, Sampled::Cavity),
                    None => return (None, Sampled::Missed, 0),
                }
            } else {
                let ray_hit = n
                    .try_normalize(1e-9)
                    .and_then(|n| scan.first_facing_hit(a(p + n * cage), a(-n), ray));
                match ray_hit {
                    Some(h) => (h.triangle, h.barycentric, Sampled::Ray),
                    None => match scan.query.closest_point(a(p), reach, Some((a(n), 0.2))) {
                        Some(h) => (h.source[1], h.barycentric, Sampled::Closest),
                        None => return (None, Sampled::Missed, 0),
                    },
                }
            };
            let st = scan.triangles[triangle];
            let in_brow = marked_brow.get(triangle).copied().unwrap_or(false);
            // A marked brow is painted skin, even where its colour made the
            // segmentation call it hair; never cloth.
            let not_skin = |l: Label| l != Label::Skin && !(in_brow && l == Label::Hair);
            if st.iter().any(|&i| not_skin(labels[i as usize])) {
                let kind = if kind == Sampled::Ray {
                    Sampled::RayOccluded
                } else {
                    kind
                };
                let on = |l: Label| st.iter().any(|&i| labels[i as usize] == l);
                return (
                    None,
                    kind,
                    on(Label::Hair) as u8 | (on(Label::Garment) as u8) << 1,
                );
            }
            let weight: f32 = if in_brow {
                1.0
            } else {
                (0..3)
                    .map(|j| scan_weight[st[j] as usize] * barycentric[j] as f32)
                    .sum()
            };
            let shown: f32 = 1.0
                - (0..3)
                    .map(|j| hidden.get(tri[j] as usize).copied().unwrap_or(0.0) as f32 * b[j])
                    .sum::<f32>()
                    .clamp(0.0, 1.0);
            let weight = weight * shown;
            if weight < 0.01 {
                return (None, kind, 0);
            }
            (
                scan.color_at(triangle, barycentric).map(|c| (c, weight)),
                kind,
                (in_brow as u8) << 2,
            )
        })
        .collect();
    let count = |s: Sampled| sampled.iter().filter(|x| x.1 == s).count();
    let ray_report = json!({
        "cage_extrusion_m": cage,
        "max_ray_distance_m": ray,
        "ray_hit_texels": count(Sampled::Ray),
        "ray_hit_skin_texels": sampled.iter().filter(|x| x.1 == Sampled::Ray && x.0.is_some()).count(),
        "ray_occluded_texels": count(Sampled::RayOccluded),
        "closest_fallback_texels": count(Sampled::Closest),
        "closest_fallback_skin_texels": sampled.iter().filter(|x| x.1 == Sampled::Closest && x.0.is_some()).count(),
        "cavity_texels": count(Sampled::Cavity),
        "missed_texels": count(Sampled::Missed),
    });
    let on_hair: Vec<bool> = sampled.iter().map(|x| x.2 & 1 != 0).collect();
    let on_cloth: Vec<bool> = sampled.iter().map(|x| x.2 & 2 != 0).collect();
    // Texels showing a marked brow: kept as painted.
    let in_brow: Vec<bool> = sampled.iter().map(|x| x.2 & 4 != 0).collect();
    let brow_texels = in_brow.iter().filter(|b| **b).count();
    let mut observed: Vec<Option<Sample>> = sampled.into_iter().map(|x| x.0).collect();
    let original_at = |k: usize| -> [f64; 3] {
        match original {
            Some(img) => sample(
                img,
                [
                    ((k % w) as f64 + 0.5) / w as f64,
                    ((k / w) as f64 + 0.5) / h as f64,
                ],
            ),
            None => [0.8, 0.65, 0.55],
        }
    };
    let detail_at = |k: usize| -> f64 {
        let t = owner[k];
        if t == u32::MAX {
            return 0.0;
        }
        let tri = triangles[t as usize];
        (0..3)
            .map(|j| detail[tri[j] as usize] * bary[k][j] as f64)
            .sum()
    };
    let hidden_at = |k: usize| -> f64 {
        let t = owner[k];
        if t == u32::MAX || hidden.is_empty() {
            return 0.0;
        }
        let tri = triangles[t as usize];
        (0..3)
            .map(|j| hidden[tri[j] as usize] * bary[k][j] as f64)
            .sum::<f64>()
            .clamp(0.0, 1.0)
    };
    // Paint that is not skin (see crate::paint): contact shadows along
    // garment edges away from the face, and hair
    // strands painted on the skin. Their samples are dropped before the
    // tone field and the fill read them.
    let detail_tex: Vec<f32> = (0..w * h).map(|k| detail_at(k) as f32).collect();
    let skip: Vec<f32> = detail_tex
        .iter()
        .zip(&excluded)
        .zip(&in_brow)
        .map(|((d, e), b)| if *e > 0.0 || *b { 1.0 } else { *d })
        .collect();
    let atlas = crate::paint::Atlas {
        w,
        h,
        owner: &owner,
        bary: &bary,
        triangles,
        positions,
        uvs,
    };
    let (shade, shadow_report) = crate::paint::edge_shadows(&atlas, &observed, &on_cloth, &skip);
    for (o, f) in observed.iter_mut().zip(&shade) {
        if let Some((_, wt)) = o {
            *wt *= f;
            if *wt < 0.01 {
                *o = None;
            }
        }
    }
    let colours: Vec<Option<[f64; 3]>> = observed.iter().map(|o| o.map(|(c, _)| c)).collect();
    let protect: Vec<f32> = excluded
        .iter()
        .zip(&in_brow)
        .map(|(e, b)| if *b { 1.0 } else { *e })
        .collect();
    let (strand, strand_report) =
        crate::paint::painted_strands(&atlas, &colours, &on_hair, &protect, &detail_tex);
    for (o, s) in observed.iter_mut().zip(&strand) {
        if *s {
            *o = None;
        }
    }
     
    let mut scalp_hair = 0usize;
    if !scalp.is_empty() {
        let seen: Vec<[f64; 3]> = (0..w * h).step_by(5).filter(|&k| excluded[k] <= 0.0 && !in_brow[k])
            .filter_map(|k| observed[k].filter(|o| o.1 > 0.5).map(|o| lab(o.0))).collect();
        if seen.len() > 200 {
            let median: [f64; 3] = std::array::from_fn(|j| {
                let mut v: Vec<f64> = seen.iter().map(|c| c[j]).collect();
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            });
            for k in 0..w * h {
                let t = owner[k];
                let Some((c, _)) = observed[k] else { continue };
                if t == u32::MAX {
                    continue;
                }
                let tri = triangles[t as usize];
                let on_scalp: f64 = (0..3).map(|j| scalp[tri[j] as usize] * bary[k][j] as f64).sum();
                 
                let (l, de) = (lab(c), delta_e(lab(c), median));
                let greyer = l[1].hypot(l[2]) < median[1].hypot(median[2]) - 6.0;
                if on_scalp > 0.2 && (de > 28.0 || (de > 14.0 && greyer)) {
                    observed[k] = None;
                    scalp_hair += 1;
                }
            }
        }
    }
     
    let mut mh_lin: Vec<[f64; 3]> = (0..w * h)
        .into_par_iter()
        .map(|k| original_at(k).map(srgb_to_linear))
        .collect();
    let scalp_report = destubble(&mut mh_lin, w, h, size, &|k| {
        let t = owner[k];
        if t == u32::MAX || scalp.is_empty() {
            return 0.0;
        }
        let tri = triangles[t as usize];
        (0..3).map(|j| scalp[tri[j] as usize] * bary[k][j] as f64).sum::<f64>().clamp(0.0, 1.0)
    }, &|k| detail_at(k), &|k| owner[k] != u32::MAX);
     
    if std::env::var("MH_TEX_FLAT").map_or(true, |v| v != "0") {
        let r = (size as usize / 512).max(2) * 12;
        let inside: Vec<f32> = (0..w * h).map(|k| (owner[k] != u32::MAX) as u8 as f32).collect();
        let cover = box_blur(&box_blur(&inside, w, h, r), w, h, r);
        let mut mean = [0.0f64; 3];
        let count = inside.iter().filter(|x| **x > 0.0).count().max(1) as f64;
        let mut broad: Vec<[f32; 3]> = vec![[0.0; 3]; w * h];
        for j in 0..3 {
            let channel: Vec<f32> = (0..w * h).map(|k| mh_lin[k][j] as f32 * inside[k]).collect();
            mean[j] = channel.iter().map(|x| *x as f64).sum::<f64>() / count;
            let blurred = box_blur(&box_blur(&channel, w, h, r), w, h, r);
            for k in 0..w * h {
                broad[k][j] = blurred[k] / cover[k].max(1e-4);
            }
        }
        for k in (0..w * h).filter(|&k| inside[k] > 0.0) {
            for j in 0..3 {
                mh_lin[k][j] = (mh_lin[k][j] / (broad[k][j] as f64).max(0.02)).clamp(0.5, 1.6) * mean[j];
            }
        }
    }
    let log_ratio: Vec<[f32; 3]> = (0..w * h)
        .into_par_iter()
        .map(|k| match observed[k] {
            Some((c, _)) => std::array::from_fn(|j| {
                (srgb_to_linear(c[j]).max(1e-3) / mh_lin[k][j].max(0.02))
                    .clamp(0.05, 12.0)
                    .ln() as f32
            }),
            None => [0.0; 3],
        })
        .collect();
    // The eye surround's painted sclera and eyeliner are not skin tone, nor
    // is a marked brow (its dark tone spread over the unobserved forehead).
    let ratio_weight: Vec<f32> = observed
        .iter()
        .zip(&excluded)
        .zip(&in_brow)
        .map(|((o, e), b)| {
            if *b {
                0.0
            } else {
                o.map_or(0.0, |(_, w0)| w0) * (1.0 - e)
            }
        })
        .collect();
    let mut tone = pull_push(&log_ratio, &ratio_weight, w, h);
    // Far from any observed skin (the torso under clothes) the extrapolated
    // tone is a guess from a few edge samples, and a collar or shadow colour
    // there tinted the whole hidden torso. Blend towards the body's median
    // tone with distance from observed skin.
    let global: [f32; 3] = std::array::from_fn(|j| {
        let mut v: Vec<f32> = (0..w * h)
            .step_by(5)
            .filter(|&k| ratio_weight[k] > 0.5)
            .map(|k| log_ratio[k][j])
            .collect();
        v.sort_by(f32::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(0.0)
    });
    let far = (size as usize / 512).max(2) * 16;
    let cover = box_blur(&box_blur(&ratio_weight, w, h, far), w, h, far);
    // Texel-space filters stop at UV seams: the foot islands never saw the
    // leg's tone just across the ankle seam, and the seam showed as a line
    // where observed skin met the toned texture. The same fields spread
    // over the welded surface carry across seams.
    let surface = SurfaceSpread::new(positions, triangles, &owner, &bary);
    let (cover3, tone3) = surface.spread(&log_ratio, &ratio_weight, 12);
    for k in 0..w * h {
        let c = ((cover[k] - 0.05) / 0.25).clamp(0.0, 1.0);
         
        let c = c * c * (3.0 - 2.0 * c) * (1.0 - hidden_at(k) as f32);
        let c3 = ((cover3[k] - 0.05) / 0.25).clamp(0.0, 1.0);
        let c3 = c3 * c3 * (3.0 - 2.0 * c3);
        for j in 0..3 {
            let outer = tone3[k][j] * c3 + global[j] * (1.0 - c3);
            tone[k][j] = tone[k][j] * c + outer * (1.0 - c);
        }
    }
     
    let brightest = {
        let mut v: Vec<f64> = (0..w * h)
            .step_by(3)
            .filter(|&k| ratio_weight[k] > 0.5)
            .filter_map(|k| observed[k].map(|(c, _)| c.iter().map(|x| srgb_to_linear(*x)).fold(0.0, f64::max)))
            .collect();
        v.sort_by(f64::total_cmp);
        v.get(v.len().saturating_sub(1) * 95 / 100).copied().unwrap_or(f64::INFINITY)
    };
    let expected = |k: usize| -> [f64; 3] {
        let lin: [f64; 3] = std::array::from_fn(|j| mh_lin[k][j] * (tone[k][j] as f64).exp());
        let top = lin.iter().copied().fold(0.0, f64::max);
        let gain = if top > brightest { brightest / top } else { 1.0 };
        std::array::from_fn(|j| linear_to_srgb(lin[j] * gain))
    };
    // 3. Outliers away from the face (painted strands, cloth shadows) are
    //    not copied: they differ strongly from the expected local skin.
    let mut outliers = 0usize;
    let mut rejected = vec![false; w * h];
    let mut mask = vec![0f32; w * h];
    for k in 0..w * h {
        let Some((c, wt)) = observed[k] else { continue };
        // The eye surround is decided below, against the skin around it.
        if excluded[k] > 0.0 {
            continue;
        }
        let mut m = wt;
        if detail_at(k) < 0.5 && !in_brow[k] {
            let e = expected(k);
            let de = delta_e(lab(c), lab(e));
            if de > 22.0 {
                let f = (1.0 - (de - 22.0) / 10.0).clamp(0.0, 1.0) as f32;
                m *= f;
                if f < 0.5 {
                    outliers += 1;
                    rejected[k] = true;
                }
            }
        }
        mask[k] = m;
    }
    // Small holes inside observed skin (nostril wings wider than the scan's
    // nose, a lid crease) are inpainted from the scan colours; only large
    // unobserved areas (under clothes) fall back to toned MakeHuman texture.
    let observed_lin: Vec<[f32; 3]> = observed
        .iter()
        .map(|o| o.map_or([0.0; 3], |(c, _)| c.map(|x| srgb_to_linear(x) as f32)))
        .collect();
     
    let face_lab: Vec<[f64; 3]> = (0..w * h)
        .step_by(7)
        .filter(|&k| mask[k] > 0.5 && detail_at(k) >= 0.5)
        .filter_map(|k| observed[k].map(|(c, _)| lab(c)))
        .collect();
    let face_median: [f64; 3] = std::array::from_fn(|j| {
        let mut v: Vec<f64> = face_lab.iter().map(|c| c[j]).collect();
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(0.0)
    });
    let around_mask: Vec<f32> = (0..w * h)
        .map(|k| match observed[k] {
            Some((c, _)) if !face_lab.is_empty() && delta_e(lab(c), face_median) > 18.0 => 0.0,
            _ => mask[k],
        })
        .collect();
    let skin_around = pull_push(&observed_lin, &around_mask, w, h);
    for k in 0..w * h {
        let Some((c, wt)) = observed[k] else { continue };
        if excluded[k] <= 0.0 && surround_eye[k] <= 0.0 {
            continue;
        }
        let e = skin_around[k].map(|x| linear_to_srgb(x as f64));
        let (lc, le) = (lab(c), lab(e));
        let de = delta_e(lc, le);
         
        let f = if lc[0] < le[0] - 8.0 {
            1.0
        } else {
            (1.0 - (de - 10.0) / 6.0).clamp(0.0, 1.0) as f32
        };
        // On the lid skin around the opening only the rejection applies.
        let zone = if excluded[k] > 0.0 {
            1.0
        } else {
            surround_eye[k]
        };
        let f = 1.0 - (1.0 - f) * zone;
        if f < 0.5 {
            outliers += 1;
            rejected[k] = true;
        }
        let wt = if excluded[k] > 0.0 { wt } else { mask[k] };
        mask[k] = wt * (1.0 - excluded[k]) * f;
    }
     
    let fill_mask: Vec<f32> = (0..w * h)
        .map(|k| match observed[k] {
            Some((c, _))
                if in_brow[k] && !face_lab.is_empty() && lab(c)[0] < face_median[0] - 8.0 =>
            {
                0.0
            }
            _ => mask[k],
        })
        .collect();
    let inpaint = pull_push(&observed_lin, &fill_mask, w, h);
    let radius = (size as usize / 512).max(2) * 2;
    let blurred = box_blur(&box_blur(&mask, w, h, radius), w, h, radius);
    let wide = (size as usize / 512).max(2) * 6;
    let surround = box_blur(&box_blur(&fill_mask, w, h, wide), w, h, wide);
    // Observed skin colour across UV seams (see the tone above): the toned
    // side of a seam fades in from the observed colour on the other side.
    let (surround3, fill3) = surface.spread(&observed_lin, &fill_mask, 6);
    let mut image = RgbaImage::new(size, size);
    let mut provenance = RgbaImage::new(size, size);
    let mut filled = vec![false; w * h];
    let mut n_obs = 0usize;
    let mut observed_at = vec![0f32; w * h];
    for k in 0..w * h {
        if owner[k] == u32::MAX {
            continue;
        }
        let local = ((surround[k] - 0.3) / 0.4).clamp(0.0, 1.0) as f64;
        // Excluded eye texels take the surrounding scan skin, never
        // MakeHuman's texture there (its painted socket eye shows white).
        let local = local * local * (3.0 - 2.0 * local);
        // On a foot inside a shoe the texel-space fill only spreads the few
        // shaded collar samples (a dark blot on the heel).
        let local = local * (1.0 - hidden_at(k));
        // A seam's two sides share their vertices: half the surface
        // coverage there means the other side is fully observed.
        let local3 = ((surround3[k] - 0.05) / 0.4).clamp(0.0, 1.0) as f64;
        let local3 = local3 * local3 * (3.0 - 2.0 * local3);
        // Across a seam (more surface than texel coverage) the fill is the
        // surface-spread colour.
        let across = if local3 > local {
            1.0 - local / local3
        } else {
            0.0
        };
        let local = local.max(local3);
        let local = local + (1.0 - local) * excluded[k] as f64;
        let toned = expected(k);
        let e = excluded[k] as f64;
        // Deep inside the opening the body surface is MakeHuman's socket
        // lining and inner-corner tissue. Where the scan paints white there,
        // skin would show as a flap over the eye: take the scan's white.
        // (Not on the lid rim: white there would draw a second eye.)
        let deep = ((e - 0.7) / 0.25).clamp(0.0, 1.0);
        let skin_fill: [f64; 3] = std::array::from_fn(|j| linear_to_srgb(skin_around[k][j] as f64));
        let eye_fill = match observed[k] {
            Some((c, _)) if deep > 0.0 && lab(c)[0] > lab(skin_fill)[0] + 8.0 => {
                std::array::from_fn(|j| c[j] * deep + skin_fill[j] * (1.0 - deep))
            }
            _ => skin_fill,
        };
        let base: [f64; 3] = std::array::from_fn(|j| {
            let observed_fill = inpaint[k][j] as f64 * (1.0 - across) + fill3[k][j] as f64 * across;
            let fill = linear_to_srgb(observed_fill) * (1.0 - e) + eye_fill[j] * e;
            toned[j] * (1.0 - local) + fill * local
        });
        // A marked brow is not blended into the fill at its edge: next to
        // hair (fringe, strands) the feathered surround is thin, and the
        // brow faded into a smear.
        let m = if in_brow[k] {
            mask[k].min(1.0) as f64
        } else {
            (blurred[k] * mask[k]).min(1.0) as f64
        };
        let (x, y) = ((k % w) as u32, (k / w) as u32);
        let observed_weight = if observed[k].is_some() { m } else { 0.0 };
        observed_at[k] = observed_weight as f32;
        provenance.put_pixel(
            x,
            y,
            Rgba([
                to8(observed_weight),
                to8(excluded[k] as f64),
                if rejected[k] || strand[k] { 255 } else { 0 },
                255,
            ]),
        );
        let c = match observed[k] {
            Some((o, _)) if m > 0.0 => {
                n_obs += 1;
                std::array::from_fn(|j| o[j] * m + base[j] * (1.0 - m))
            }
            _ => base,
        };
        // Socket lining showing in the opening's corners is the eye's white.
        let c: [f64; 3] = match sclera_srgb {
            Some(s) if lining[k] > 0.0 => {
                let l = lining[k] as f64;
                std::array::from_fn(|j| c[j] * (1.0 - l) + s[j] * l)
            }
            _ => c,
        };
        image.put_pixel(x, y, Rgba([to8(c[0]), to8(c[1]), to8(c[2]), 255]));
        filled[k] = true;
    }
    // Marked brows are the scan's own: nothing is mirrored over them.
    let brow_report = if brow_texels > 0 {
        json!({"marked_brow_texels": brow_texels, "mirrored_onto": null})
    } else {
        mirror_brows(
            &mut image,
            &owner,
            &bary,
            triangles,
            uvs,
            brows,
            mirror,
            &observed_at,
        )
    };
    dilate(&mut image, &mut filled, 6);
    let covered = owner.iter().filter(|o| **o != u32::MAX).count();
    let mean_gain: [f64; 3] = std::array::from_fn(|j| {
        let (sum, n) = (0..w * h)
            .filter(|&k| owner[k] != u32::MAX)
            .fold((0.0, 0usize), |(s0, n0), k| {
                (s0 + (tone[k][j] as f64).exp(), n0 + 1)
            });
        sum / n.max(1) as f64
    });
    Bake {
        image,
        provenance,
        report: json!({"excluded_texels":excluded.iter().filter(|e| **e > 0.5).count(),"size":size,"uv_texels":covered,"observed_skin_texels":n_obs,"observed_fraction":n_obs as f64 / covered.max(1) as f64,"rejected_outlier_texels":outliers,"painted_strands":strand_report,"edge_shadows":shadow_report,"mean_tone_gain_linear":mean_gain,"projection":ray_report,"brows":brow_report,"scalp":scalp_report,"scalp_hair_texels":scalp_hair}),
    }
}

/// Brow symmetry: the observed share of each brow zone (texel weight of the
/// scan colour); where one brow is clearly less observed than the other
/// (hair strands hanging over it: blurred fill and paint residue), the other
/// brow's texels are mirrored onto it through the body's mirror partners
/// (same barycentric point of the mirrored triangle), blended out at the
/// zone's edge.
#[allow(clippy::too_many_arguments)]
fn mirror_brows(
    image: &mut RgbaImage,
    owner: &[u32],
    bary: &[[f32; 3]],
    triangles: &[[u32; 3]],
    uvs: &[[f32; 2]],
    brows: &[[f64; 2]],
    mirror: &[u32],
    observed: &[f32],
) -> Value {
    if brows.is_empty() || mirror.is_empty() {
        return Value::Null;
    }
    let zone = |k: usize, s: usize| -> f64 {
        let t = owner[k];
        if t == u32::MAX {
            return 0.0;
        }
        let tri = triangles[t as usize];
        (0..3)
            .map(|j| brows[tri[j] as usize][s] * bary[k][j] as f64)
            .sum()
    };
    let mut cover = [0.0f64; 2];
    let mut total = [0.0f64; 2];
    for k in 0..owner.len() {
        for s in 0..2 {
            let z = zone(k, s);
            cover[s] += z * observed[k] as f64;
            total[s] += z;
        }
    }
    let share = [0, 1].map(|s| cover[s] / total[s].max(1e-9));
    let (good, bad) = if share[0] >= share[1] { (0, 1) } else { (1, 0) };
    // Mirrored only for a clear difference with a well-observed source.
    let strength = (((share[good] - share[bad]) - 0.1) / 0.1).clamp(0.0, 1.0)
        * ((share[good] - 0.25) / 0.15).clamp(0.0, 1.0);
    let mut texels = 0usize;
    if strength > 0.0 {
        let source = image.clone();
        for k in 0..owner.len() {
            let z = zone(k, bad);
            if z <= 0.0 {
                continue;
            }
            let tri = triangles[owner[k] as usize];
            let mut uv = [0.0f64; 2];
            for j in 0..3 {
                let p = uvs[mirror[tri[j] as usize] as usize];
                uv[0] += p[0] as f64 * bary[k][j] as f64;
                uv[1] += p[1] as f64 * bary[k][j] as f64;
            }
            let c = sample(&source, uv);
            let a = z * strength;
            let w = image.width() as usize;
            let p = image.get_pixel_mut((k % w) as u32, (k / w) as u32);
            for j in 0..3 {
                p[j] = (p[j] as f64 * (1.0 - a) + c[j] * 255.0 * a)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
            texels += 1;
        }
    }
    json!({
        "observed_share": {"l": share[0], "r": share[1]},
        "mirrored_onto": (strength > 0.0).then_some(if bad == 0 { "l" } else { "r" }),
        "strength": strength,
        "texels": texels,
    })
}

/// Texel fields spread over the welded body surface, across UV seams.
struct SurfaceSpread {
    /// Welded vertex of each body vertex, and welded neighbours.
    id: Vec<usize>,
    neighbours: Vec<Vec<u32>>,
    /// Per texel: owning triangle (`u32::MAX` outside the atlas).
    owner: Vec<u32>,
    bary: Vec<[f32; 3]>,
    triangles: Vec<[u32; 3]>,
}

impl SurfaceSpread {
    fn new(positions: &[Vec3], triangles: &[[u32; 3]], owner: &[u32], bary: &[[f32; 3]]) -> Self {
        let weld = crate::geom::Weld::new(positions, 1e-7);
        let neighbours = crate::geom::adjacency(triangles, &weld);
        Self {
            id: weld.id,
            neighbours,
            owner: owner.to_vec(),
            bary: bary.to_vec(),
            triangles: triangles.to_vec(),
        }
    }

    /// Per texel: the surface coverage of `weights` (weighted share of the
    /// texels around each welded vertex, smoothed `passes` times over its
    /// ring) and the weighted mean of `values` there.
    fn spread(
        &self,
        values: &[[f32; 3]],
        weights: &[f32],
        passes: usize,
    ) -> (Vec<f32>, Vec<[f32; 3]>) {
        let n = self.neighbours.len();
        let (mut total, mut cover, mut sum) = (vec![0f64; n], vec![0f64; n], vec![[0f64; 3]; n]);
        for (k, &t) in self.owner.iter().enumerate() {
            if t == u32::MAX {
                continue;
            }
            for (j, &i) in self.triangles[t as usize].iter().enumerate() {
                let (b, wi) = (self.bary[k][j] as f64, self.id[i as usize]);
                total[wi] += b;
                let wk = weights[k] as f64 * b;
                if wk > 0.0 {
                    cover[wi] += wk;
                    for c in 0..3 {
                        sum[wi][c] += values[k][c] as f64 * wk;
                    }
                }
            }
        }
        // Coverage A and coverage-weighted values S = A · value per vertex.
        let mut a: Vec<f64> = (0..n).map(|i| cover[i] / total[i].max(1e-12)).collect();
        let mut s: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let m = cover[i].max(1e-12);
                std::array::from_fn(|c| sum[i][c] / m * a[i])
            })
            .collect();
        for _ in 0..passes {
            let step = |i: usize| {
                let ring = &self.neighbours[i];
                let k = (1 + ring.len()) as f64;
                let a1 = ring.iter().fold(a[i], |x, &j| x + a[j as usize]) / k;
                let s1: [f64; 3] = std::array::from_fn(|c| {
                    ring.iter().fold(s[i][c], |x, &j| x + s[j as usize][c]) / k
                });
                (a1, s1)
            };
            let next: Vec<(f64, [f64; 3])> = (0..n).into_par_iter().map(step).collect();
            for (i, (a1, s1)) in next.into_iter().enumerate() {
                a[i] = a1;
                s[i] = s1;
            }
        }
        let texels: Vec<(f32, [f32; 3])> = (0..self.owner.len())
            .into_par_iter()
            .map(|k| {
                let t = self.owner[k];
                if t == u32::MAX {
                    return (0.0, [0.0; 3]);
                }
                let (mut ak, mut sk) = (0.0, [0.0; 3]);
                for (j, &i) in self.triangles[t as usize].iter().enumerate() {
                    let (b, wi) = (self.bary[k][j] as f64, self.id[i as usize]);
                    ak += a[wi] * b;
                    for c in 0..3 {
                        sk[c] += s[wi][c] * b;
                    }
                }
                let value = std::array::from_fn(|c| (sk[c] / ak.max(1e-9)) as f32);
                (ak as f32, value)
            })
            .collect();
        texels.into_iter().unzip()
    }
}

/// A scan colour (sRGB) and its sampling weight.
type Sample = ([f64; 3], f32);

/// How a texel found its scan sample.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Sampled {
    /// Not in the atlas, or an excluded eye texel.
    Skipped,
    /// The normal ray hit the scan.
    Ray,
    /// The normal ray's first surface is hair or cloth.
    RayOccluded,
    /// The ray missed; closest-point fallback.
    Closest,
    /// Head cavity: short closest-point query.
    Cavity,
    /// No scan surface in reach.
    Missed,
}

fn lab(c: [f64; 3]) -> [f64; 3] {
    crate::source::lab(c)
}

/// MakeHuman's scalp stubble out of its texture (linear colours, in
/// place): on the scalp (`scalp_at`), a texel whose local colour is off the
/// texture's own face skin (median over `detail_at` texels) takes that skin
/// colour, keeping half its fine detail (its ratio to the local mean).
fn destubble(
    mh: &mut [[f64; 3]],
    w: usize,
    h: usize,
    size: u32,
    scalp_at: &dyn Fn(usize) -> f64,
    detail_at: &dyn Fn(usize) -> f64,
    inside: &dyn Fn(usize) -> bool,
) -> Value {
    let face: Vec<usize> = (0..w * h).step_by(3).filter(|&k| detail_at(k) > 0.9 && scalp_at(k) < 0.1).collect();
    if face.is_empty() {
        return json!({"applied": false});
    }
    let skin: [f64; 3] = std::array::from_fn(|j| {
        let mut v: Vec<f64> = face.iter().map(|&k| mh[k][j]).collect();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    });
    let skin_lab = lab(skin.map(crate::source::linear_to_srgb));
    let r = (size as usize / 512).max(1) * 4;
    // The local mean over the UV island only: the empty atlas round it
    // darkened the island's edge, a line down the back of the head.
    let m: Vec<f32> = (0..w * h).map(|k| if inside(k) { 1.0 } else { 0.0 }).collect();
    let cover = box_blur(&box_blur(&m, w, h, r), w, h, r);
    let blurred: Vec<Vec<f32>> = (0..3)
        .map(|j| {
            let c: Vec<f32> = mh.iter().zip(&m).map(|(p, m)| p[j] as f32 * m).collect();
            box_blur(&box_blur(&c, w, h, r), w, h, r).iter().zip(&cover).map(|(c, w)| c / w.max(1e-4)).collect()
        })
        .collect();
    let mut changed = 0usize;
    for k in 0..w * h {
        let s = scalp_at(k);
        if s <= 0.0 {
            continue;
        }
        let local: [f64; 3] = std::array::from_fn(|j| blurred[j][k] as f64);
        let off = delta_e(lab(local.map(crate::source::linear_to_srgb)), skin_lab);
        let f = s * ((off - 3.0) / 10.0).clamp(0.0, 1.0);
        if f <= 0.0 {
            continue;
        }
        changed += 1;
        for j in 0..3 {
            let ratio = (mh[k][j] / local[j].max(1e-4)).clamp(0.7, 1.3).sqrt();
            mh[k][j] = mh[k][j] * (1.0 - f) + skin[j] * ratio * f;
        }
    }
    json!({"applied": true, "texels": changed, "skin_srgb": skin.map(crate::source::linear_to_srgb)})
}

fn delta_e(a0: [f64; 3], b0: [f64; 3]) -> f64 {
    ((a0[0] - b0[0]).powi(2) * 0.5 + (a0[1] - b0[1]).powi(2) + (a0[2] - b0[2]).powi(2)).sqrt()
}

/// Pull-push scattered-data interpolation on an image: weighted
/// down-sampling to 1×1, then up-sampling that fills low-weight texels from
/// coarser levels. Smooth, fast and defined everywhere.
/// One pull-push level: values, weights, width, height.
type Level = (Vec<[f32; 3]>, Vec<f32>, usize, usize);

pub(crate) fn pull_push(values: &[[f32; 3]], weights: &[f32], w: usize, h: usize) -> Vec<[f32; 3]> {
    let mut levels: Vec<Level> = vec![(
        values.to_vec(),
        weights.iter().map(|x| x.clamp(0.0, 1.0)).collect(),
        w,
        h,
    )];
    while levels.last().is_some_and(|l| l.2 > 1 || l.3 > 1) {
        let (v0, w0, lw, lh) = levels.last().unwrap();
        let (nw, nh) = (lw.div_ceil(2), lh.div_ceil(2));
        let mut nv = vec![[0f32; 3]; nw * nh];
        let mut nwt = vec![0f32; nw * nh];
        for y in 0..*lh {
            for x in 0..*lw {
                let k = y * lw + x;
                let c = (y / 2) * nw + x / 2;
                for (acc, val) in nv[c].iter_mut().zip(v0[k]) {
                    *acc += val * w0[k];
                }
                nwt[c] += w0[k];
            }
        }
        for c in 0..nw * nh {
            if nwt[c] > 0.0 {
                let inv = 1.0 / nwt[c];
                nv[c].iter_mut().for_each(|x| *x *= inv);
            }
            nwt[c] = nwt[c].min(1.0);
        }
        levels.push((nv, nwt, nw, nh));
    }
    for l in (0..levels.len() - 1).rev() {
        let (head, tail) = levels.split_at_mut(l + 1);
        let (coarse, fine) = (&tail[0], &mut head[l]);
        let (cv, _, cw, ch) = coarse;
        let (fv, fw, lw, lh) = fine;
        // Bilinear, not nearest: a hole filled from a coarse level would
        // otherwise show as flat squares of that level's texel size.
        let sample = |fx: f32, fy: f32| -> [f32; 3] {
            let (u, v) = (
                (fx * 0.5 - 0.5).clamp(0.0, (*cw - 1) as f32),
                (fy * 0.5 - 0.5).clamp(0.0, (*ch - 1) as f32),
            );
            let (x0, y0) = (u.floor() as usize, v.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(*cw - 1), (y0 + 1).min(*ch - 1));
            let (tx, ty) = (u - x0 as f32, v - y0 as f32);
            let at = |x: usize, y: usize| cv[y * cw + x];
            let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
            std::array::from_fn(|j| {
                (a[j] * (1.0 - tx) + b[j] * tx) * (1.0 - ty) + (c[j] * (1.0 - tx) + d[j] * tx) * ty
            })
        };
        for y in 0..*lh {
            for x in 0..*lw {
                let k = y * *lw + x;
                let c = sample(x as f32 + 0.5, y as f32 + 0.5);
                let a0 = fw[k];
                for j in 0..3 {
                    fv[k][j] = fv[k][j] * a0 + c[j] * (1.0 - a0);
                }
                fw[k] = 1.0;
            }
        }
    }
    levels.swap_remove(0).0
}

/// Per scan vertex sampling weight for skin restoration: 0 on non-skin and
/// LLM cleanup triangles, rising smoothly to 1 over `radius` (geodesic) away
/// from garment/hair edges so cuff and collar colours do not bleed in, and
/// over `cleanup_radius` away from cleanup triangles (a marked painted
/// strand is thin: a garment-edge feather around it would blank the brow or
/// cheek beside it).
pub fn skin_sample_weights(
    scan: &Scan,
    labels: &[Label],
    cleanup: Option<&[bool]>,
    radius: f64,
    cleanup_radius: f64,
) -> Vec<f32> {
    let n = scan.weld.count;
    let adj = crate::geom::adjacency(&scan.triangles, &scan.weld);
    let mut blocked = vec![false; n];
    if let Some(c) = cleanup {
        for (t, tri) in scan.triangles.iter().enumerate() {
            if c[t] {
                tri.iter()
                    .for_each(|&i| blocked[scan.weld.id[i as usize]] = true);
            }
        }
    }
    let edge: Vec<bool> = (0..n)
        .map(|w0| labels[scan.weld.representative[w0]] != Label::Skin)
        .collect();
    let a = geodesic_ramp(scan, &adj, &edge, radius);
    let b = geodesic_ramp(scan, &adj, &blocked, cleanup_radius);
    scan.weld.id.iter().map(|&w0| a[w0] * b[w0]).collect()
}

/// Per welded vertex: 0 on `seeds`, rising smoothly to 1 at geodesic
/// distance `radius` from them.
fn geodesic_ramp(scan: &Scan, adj: &[Vec<u32>], seeds: &[bool], radius: f64) -> Vec<f32> {
    let n = seeds.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut heap = std::collections::BinaryHeap::new();
    for w0 in (0..n).filter(|&w0| seeds[w0]) {
        dist[w0] = 0.0;
        heap.push((std::cmp::Reverse(Key(0.0)), w0));
    }
    while let Some((std::cmp::Reverse(Key(d)), i)) = heap.pop() {
        if d > dist[i] || d > radius {
            continue;
        }
        let p = v(scan.positions[scan.weld.representative[i]]);
        for &j in &adj[i] {
            let j = j as usize;
            let nd = d + (v(scan.positions[scan.weld.representative[j]]) - p).norm();
            if nd < dist[j] {
                dist[j] = nd;
                heap.push((std::cmp::Reverse(Key(nd)), j));
            }
        }
    }
    dist.iter()
        .map(|&d| {
            let t = (d / radius.max(1e-9)).clamp(0.0, 1.0);
            (t * t * (3.0 - 2.0 * t)) as f32
        })
        .collect()
}

#[derive(PartialEq, Clone, Copy)]
struct Key(f64);
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

fn to8(x: f64) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0).round() as u8
}

pub(crate) fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut tmp = vec![0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let line = &src[y * w..(y + 1) * w];
        let mut acc: f32 = line[..(r + 1).min(w)].iter().sum();
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            row[x] = acc / (hi - lo + 1) as f32;
            if x + r + 1 < w {
                acc += line[x + r + 1];
            }
            if x >= r {
                acc -= line[x - r];
            }
        }
    });
    let mut out = vec![0f32; w * h];
    for x in 0..w {
        let mut acc = 0.0;
        for y in 0..(r + 1).min(h) {
            acc += tmp[y * w + x];
        }
        for y in 0..h {
            let lo = y.saturating_sub(r);
            let hi = (y + r).min(h - 1);
            out[y * w + x] = acc / (hi - lo + 1) as f32;
            if y + r + 1 < h {
                acc += tmp[(y + r + 1) * w + x];
            }
            if y >= r {
                acc -= tmp[(y - r) * w + x];
            }
        }
    }
    out
}

/// Grows filled texels into the UV gutter to avoid seams under mip-mapping.
pub(crate) fn dilate(image: &mut RgbaImage, filled: &mut [bool], passes: usize) {
    let (w, h) = (image.width() as usize, image.height() as usize);
    for _ in 0..passes {
        let mut next = filled.to_vec();
        let mut writes = vec![];
        for y in 0..h {
            for x in 0..w {
                if filled[y * w + x] {
                    continue;
                }
                let mut acc = [0u32; 3];
                let mut n = 0;
                for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let k = ny as usize * w + nx as usize;
                    if filled[k] {
                        let p = image.get_pixel(nx as u32, ny as u32);
                        for j in 0..3 {
                            acc[j] += p[j] as u32;
                        }
                        n += 1;
                    }
                }
                if n > 0 {
                    writes.push((x, y, acc.map(|c| (c / n) as u8)));
                    next[y * w + x] = true;
                }
            }
        }
        for (x, y, c) in writes {
            image.put_pixel(x as u32, y as u32, Rgba([c[0], c[1], c[2], 255]));
        }
        filled.copy_from_slice(&next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn box_blur_keeps_constants_and_spreads_an_impulse() {
        let (w, h) = (16, 12);
        let flat = vec![0.7f32; w * h];
        assert!(
            box_blur(&flat, w, h, 3)
                .iter()
                .all(|v| (v - 0.7).abs() < 1e-6)
        );
        let mut impulse = vec![0f32; w * h];
        impulse[6 * w + 8] = 1.0;
        let b = box_blur(&impulse, w, h, 2);
        assert!(b[6 * w + 8] > 0.0 && b[6 * w + 10] > 0.0 && b[6 * w + 11] == 0.0);
    }

    /// Two triangles share an edge on the surface but lie in separate UV
    /// islands (a seam): colour observed only on one reaches the other.
    #[test]
    fn surface_spread_crosses_uv_seams() {
        // Unwelded: 0-1-2 and 3-4-5, with 1 = 3 and 2 = 4 in space.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let triangles = vec![[0, 1, 2], [3, 5, 4]];
        // One texel per triangle, at its centroid; texel 0 observed red.
        let owner = vec![0, 1];
        let bary = vec![[1.0 / 3.0; 3]; 2];
        let surface = SurfaceSpread::new(&positions, &triangles, &owner, &bary);
        let values = vec![[1.0, 0.0, 0.0], [0.0; 3]];
        let (cover, value) = surface.spread(&values, &[1.0, 0.0], 0);
        assert!(cover[0] > cover[1] && cover[1] > 0.3, "{cover:?}");
        assert!((value[1][0] - 1.0).abs() < 1e-5 && value[1][1].abs() < 1e-6);
        let (smoothed, _) = surface.spread(&values, &[1.0, 0.0], 3);
        assert!(smoothed[1] > 0.3 && smoothed[0] < 1.0);
    }
}
