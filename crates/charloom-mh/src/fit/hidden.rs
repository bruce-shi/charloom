 use super::guard;
use crate::geom::{V, a, face_normal};
use crate::prior::Region;
use crate::scan::Scan;
use crate::segment::{Label, Segmentation};
use charloom_core::Vec3;
use rayon::prelude::*;
use serde_json::{Value, json};

/// Blend band around visible skin, fraction of the body height (7 cm at
/// 1.75 m).
const BLEND: f64 = 0.04;

/// Vertices that see visible skin before the fit: facing scan skin, or the
/// nearest body vertex of a scan skin sample.
pub fn initial_seeds(
    rest: &[V],
    rest_normals: &[Vec3],
    scan: &Scan,
    seg: &Segmentation,
    skin_samples: &[usize],
    welded_triangles: &[[u32; 3]],
    height: f64,
) -> Vec<bool> {
    let reach = 0.05 * height;
    let mut seed: Vec<bool> = (0..rest.len())
        .into_par_iter()
        .map(|i| {
            scan.query
                .closest_point(a(rest[i]), reach, Some((rest_normals[i], 0.3)))
                .is_some_and(|h| seg.triangle_label[h.source[1]] == Label::Skin)
        })
        .collect();
     
    let positions: Vec<Vec3> = rest.iter().map(|q| a(*q)).collect();
    let body_query = crate::scan::build_query(&positions, welded_triangles);
    let nearest: Vec<usize> = skin_samples
        .par_iter()
        .filter_map(|&s| {
            let hit = body_query.closest_point(
                scan.positions[s],
                2.5 * reach,
                Some((scan.normals[s], 0.3)),
            )?;
            let tri = welded_triangles[hit.source[1]];
            let k = (0..3).max_by(|&x, &y| hit.barycentric[x].total_cmp(&hit.barycentric[y]))?;
            Some(tri[k] as usize)
        })
        .collect();
    for i in nearest {
        seed[i] = true;
    }
    seed
}

/// Hidden weight per welded vertex: 0 on and near skin seeds (and outside
/// the torso and limbs, and where `excluded`), rising smoothly to 1 at
/// `BLEND` from them.
pub fn weights(
    seed: &[bool],
    shape_rest: &[V],
    neighbours: &[Vec<u32>],
    region: &[Region],
    excluded: &[bool],
    height: f64,
) -> Vec<f64> {
    let blend = BLEND * height;
    let distance = guard::surface_distance(shape_rest, neighbours, seed, blend);
    (0..seed.len())
        .map(|i| {
            if excluded[i] || !matches!(region[i], Region::Torso | Region::Arm | Region::Leg) {
                0.0
            } else if distance[i].is_finite() {
                guard::smoothstep(distance[i] / blend)
            } else {
                1.0
            }
        })
        .collect()
}

pub fn report(weight: &[f64], seed: &[bool], stiffness: f64) -> Value {
    json!({
        "note": "torso/arm/leg vertices away from visible skin: no per-vertex fill, tight-cloth or mirror pulls; only the one-sided cloth push, on a stiffened shape that shrinks locally by 8 % at most",
        "seed_vertices": seed.iter().filter(|s| **s).count(),
        "hidden_vertices": weight.iter().filter(|w| **w > 0.99).count(),
        "blend_vertices": weight.iter().filter(|w| **w > 0.0 && **w <= 0.99).count(),
        "stiffness": stiffness,
    })
}

/// Slope of the clearance dish: a 1 cm dent spreads over 8 cm.
const SLOPE: f64 = 0.12;

 #[allow(clippy::too_many_arguments)]
pub fn clearance_pass(
    x: &mut [V],
    hidden: &[f64],
    normals: &[Vec3],
    triangles: &[[u32; 3]],
    shape_rest: &[V],
    neighbours: &[Vec<u32>],
    region: &[Region],
    excluded: &[bool],
    scan: &Scan,
    seg: &Segmentation,
    clearance: f64,
    height: f64,
) -> Value {
    let n = x.len();
    let raw: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            if hidden[i] <= 0.0
                || excluded[i]
                || !matches!(region[i], Region::Torso | Region::Arm | Region::Leg)
            {
                return 0.0;
            }
            let Some(h) = scan.query.closest_point(a(x[i]), 0.02 * height, None) else {
                return 0.0;
            };
            if seg.triangle_label[h.source[1]] != Label::Garment || seg.is_prop(h.source[1]) {
                return 0.0;
            }
             
            let nq = crate::geom::v(h.normal);
            if nq.dot(&crate::geom::v(normals[i])) < 0.3 {
                return 0.0;
            }
            ((x[i] - crate::geom::v(h.point)).dot(&nq) + clearance).clamp(0.0, 0.012 * height)
        })
        .collect();
    let field = smooth_field(&raw, neighbours, shape_rest);
    // Only under the clothes; the blend band around visible skin takes part
    // of it, visible skin none.
    let mut field: Vec<f64> = field.iter().zip(hidden).map(|(f, h)| f * h).collect();
    // Moving along converging normals (an armpit, the crotch) can fold the
    // surface: where a triangle turns over, the field is halved around it.
    let start = x.to_vec();
    let before: Vec<Vec3> = start.iter().map(|q| a(*q)).collect();
    let flipped = |field: &[f64]| -> Vec<usize> {
        let moved: Vec<Vec3> = (0..n)
            .map(|i| a(start[i] - crate::geom::v(normals[i]) * field[i]))
            .collect();
        (0..triangles.len())
            .filter(|&t| {
                let tri = &triangles[t];
                tri.iter().any(|&k| field[k as usize] > 0.0)
                    && face_normal(&moved, tri).dot(&face_normal(&before, tri)) <= 0.0
            })
            .collect()
    };
    let mut backed_off = 0;
    for _ in 0..6 {
        let bad = flipped(&field);
        if bad.is_empty() {
            break;
        }
        for t in bad {
            for &k in &triangles[t] {
                let k = k as usize;
                field[k] *= 0.5;
                for &j in &neighbours[k] {
                    field[j as usize] *= 0.5;
                }
            }
            backed_off += 1;
        }
    }
    for i in 0..n {
        if field[i] > 0.0 {
            x[i] = start[i] - crate::geom::v(normals[i]) * field[i];
        }
    }
    let mut moved: Vec<f64> = field
        .iter()
        .filter(|f| **f > 1e-5)
        .map(|f| f * 1000.0)
        .collect();
    moved.sort_by(f64::total_cmp);
    let q = |f: f64| {
        moved
            .get(((moved.len() as f64 - 1.0) * f).round() as usize)
            .map_or(0.0, |x| (x * 10.0).round() / 10.0)
    };
    json!({
        "note": "one-sided smooth inward field after the fit: torso/limb vertices nearest to cloth sit the clearance inside it",
        "violating_vertices": raw.iter().filter(|r| **r > 0.0).count(),
        "backed_off_triangles": backed_off,
        "moved_vertices": moved.len(),
        "moved_mm": {"p50": q(0.5), "p90": q(0.9), "max": q(1.0)},
    })
}

/// A smooth field at least `floor` everywhere: cone dilation of bounded
/// slope over the edge graph (exact, deterministic order), then relaxation
/// that never drops below the floor.
fn smooth_field(floor: &[f64], neighbours: &[Vec<u32>], positions: &[V]) -> Vec<f64> {
    let n = floor.len();
    let mut field = floor.to_vec();
    loop {
        let mut changed = false;
        for i in 0..n {
            for &j in &neighbours[i] {
                let j = j as usize;
                let c = field[j] - SLOPE * (positions[i] - positions[j]).norm();
                if c > field[i] + 1e-9 {
                    field[i] = c;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    for _ in 0..6 {
        field = (0..n)
            .map(|i| {
                if neighbours[i].is_empty() {
                    return field[i];
                }
                let mean = neighbours[i]
                    .iter()
                    .map(|&j| field[j as usize])
                    .sum::<f64>()
                    / neighbours[i].len() as f64;
                (0.5 * (field[i] + mean)).max(floor[i])
            })
            .collect();
    }
    field
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1 cm dent at one vertex of a line of 1 cm edges becomes a dish:
    /// never below the dent, falling by at most the slope per edge, zero
    /// well away, symmetric.
    #[test]
    fn clearance_field_is_a_bounded_slope_dish() {
        let n = 41;
        let positions: Vec<V> = (0..n).map(|i| V::new(i as f64 * 0.01, 0.0, 0.0)).collect();
        let neighbours: Vec<Vec<u32>> = (0..n)
            .map(|i| {
                let mut r = vec![];
                if i > 0 {
                    r.push(i as u32 - 1);
                }
                if i + 1 < n {
                    r.push(i as u32 + 1);
                }
                r
            })
            .collect();
        let mut floor = vec![0.0; n];
        floor[20] = 0.01;
        let f = smooth_field(&floor, &neighbours, &positions);
        assert!((f[20] - 0.01).abs() < 1e-12);
        assert!(f.iter().zip(&floor).all(|(a, b)| a >= b));
        for i in 1..n {
            assert!((f[i] - f[i - 1]).abs() <= SLOPE * 0.01 + 1e-9);
        }
        assert!(f[20 + 8] > 0.0 && f[0] < 1e-6 && f[n - 1] < 1e-6);
        for k in 0..20 {
            assert!((f[20 - k] - f[20 + k]).abs() < 1e-12);
        }
    }
}
