 use crate::garment::Garment;
use crate::geom::{V, a, v};
use crate::prior::{Prior, Region};
use crate::scan::{Scan, build_query};
use crate::segment::{Label, Segmentation};
use charloom_core::Vec3;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};

/// Scan skin farther than this from the body is not covered, share of the
/// height.
pub(crate) const GAP: f64 = 0.002;
/// The body's own skin within this of a graft is hidden, share of the
/// height (`lib.rs`).
pub const BODY_UNDER: f64 = 0.0025;
/// Rings of scan triangles each graft grows onto the skin round it.
const MARGIN_RINGS: usize = 12;
/// Largest island of skin a graft takes in as a hole, square metres.
const HOLE_MAX: f64 = 0.002;
/// Smallest patch kept, square metres (a stray triangle or a lip crease is
/// left to the bake).
const MIN_AREA: f64 = 0.0002;
/// A patch is a part to keep only if it stands this far out of the body
/// somewhere, share of the height.
const MIN_OUT: f64 = 0.008;
/// The marked eye openings (centre, radius) are the eyeballs', not grafts.
pub fn uncovered(
    prior: &Prior,
    body_positions: &[Vec3],
    scan: &Scan,
    seg: &Segmentation,
    eyes: &[(V, f64)],
    height: f64,
    restored_nape: bool,
) -> (Vec<Garment>, Value) {
    let body = prior.body();
    let query = build_query(body_positions, &body.prim.triangles);
    let weld = crate::geom::Weld::new(&body.bind, 1e-7);
    let body_normals = crate::geom::vertex_normals(body_positions, &body.prim.triangles, &weld);
    let gap = GAP * height;
     
    let head: Vec<V> = (0..body_positions.len())
        .filter(|&i| prior.region[prior.dominant_joint(prior.body, i)] == Some(Region::Head))
        .map(|i| v(body_positions[i]))
        .collect();
    let centre = head.iter().fold(V::zeros(), |s, p| s + p) / head.len().max(1) as f64;
    let radius = head.iter().map(|p| (p - centre).norm()).fold(0.0, f64::max) * 1.6;
    let seen = crate::fit::hidden_head::seen_from_outside(scan, centre, radius);
    // Candidate skin triangles: not in a piece, beside the head, off the body.
    let off: Vec<bool> = (0..scan.triangles.len())
        .into_par_iter()
        .map(|t| {
            if seg.triangle_label[t] != Label::Skin || seg.triangle_piece[t].is_some() {
                return false;
            }
            if !scan.triangles[t].iter().any(|&i| seen[i as usize]) {
                return false;
            }
            let c = scan.triangles[t].iter().fold(V::zeros(), |s, &i| s + v(scan.positions[i as usize])) / 3.0;
            if eyes.iter().any(|(e, r)| (c - e).norm() < *r) {
                return false;
            }
             
            scan.triangles[t].iter().all(|&i| {
                let p = v(scan.positions[i as usize]);
                let Some(h) = query.closest_point(a(p), 0.1 * height, None) else {
                    return false;
                };
                let tri = body.prim.triangles[h.source[1]];
                let k = (0..3).max_by(|&p, &q| h.barycentric[p].total_cmp(&h.barycentric[q])).unwrap_or(0);
                let j = prior.dominant_joint(prior.body, tri[k] as usize);
                // Neck influence is split over three bones. A lower
                // occiput vertex can therefore have `head` as its largest
                // single influence while belonging to the neck transition.
                // Grafting that transition past a restored/fair skull
                // reattached the scan's jagged nape as a rigid skin shelf.
                // Keep protruding head parts (ears), never the neck seam.
                let neck_weight: f64 = (0..3).map(|k| {
                    body.influences[tri[k] as usize].iter().filter(|w|
                        prior.skeleton().joints[w.joint].name.to_lowercase().starts_with("neck")
                    ).map(|w| w.weight).sum::<f64>() * h.barycentric[k]
                }).sum();
                let on_head = prior.region[j] == Some(Region::Head)
                    && (!restored_nape || neck_weight <= 1e-6);
                // Out of the body only: skin inside it is the body bulging
                // past the scan (a fit error, not a part to keep), and grafted
                // it sat inside the head from the rest pose on.
                let n = (0..3).fold(V::zeros(), |s, k| s + v(body_normals[tri[k] as usize]) * h.barycentric[k]);
                on_head && (p - v(h.point)).dot(&n) > gap
            })
        })
        .collect();
    // Connected patches over shared welded edges.
    let mut edge_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (t, tri) in scan.triangles.iter().enumerate() {
        if !off[t] {
            continue;
        }
        let w = tri.map(|i| scan.weld.id[i as usize]);
        for k in 0..3 {
            let (p, q) = (w[k], w[(k + 1) % 3]);
            edge_tris.entry((p.min(q), p.max(q))).or_default().push(t);
        }
    }
    let area = |t: usize| {
        let [p, q, r] = scan.triangles[t].map(|i| v(scan.positions[i as usize]));
        0.5 * (q - p).cross(&(r - p)).norm()
    };
    let mut seen = vec![false; scan.triangles.len()];
    let mut patches: Vec<Vec<usize>> = vec![];
    for t0 in 0..scan.triangles.len() {
        if !off[t0] || seen[t0] {
            continue;
        }
        let mut patch = vec![];
        let mut stack = vec![t0];
        seen[t0] = true;
        while let Some(t) = stack.pop() {
            patch.push(t);
            let w = scan.triangles[t].map(|i| scan.weld.id[i as usize]);
            for k in 0..3 {
                let (p, q) = (w[k], w[(k + 1) % 3]);
                for &u in &edge_tris[&(p.min(q), p.max(q))] {
                    if !seen[u] {
                        seen[u] = true;
                        stack.push(u);
                    }
                }
            }
        }
        if patch.iter().map(|&t| area(t)).sum::<f64>() >= MIN_AREA {
            patches.push(patch);
        }
    }
    // A part to keep stands well out of the body somewhere (an ear, a
    // horn). A patch that nowhere leaves it by MIN_OUT is the fit's
    // residual on the face (a lid, a lip corner, a nostril rim): kept, it
    // lay as a shard over the body's own skin a few millimetres under it
    // and the two showed through each other.
    let min_out = std::env::var("MH_GRAFT_MIN_OUT").ok().and_then(|s| s.parse().ok()).unwrap_or(MIN_OUT) * height;
    let reach: Vec<f64> = patches
        .par_iter()
        .map(|patch| {
            patch.iter().flat_map(|&t| scan.triangles[t]).map(|i| {
                let p = v(scan.positions[i as usize]);
                query.closest_point(a(p), 0.1 * height, None).map_or(0.0, |h| (p - v(h.point)).norm())
            }).fold(0.0, f64::max)
        })
        .collect();
    let dropped: Vec<Value> = patches.iter().zip(&reach).filter(|(_, r)| **r < min_out)
        .map(|(p, r)| json!({"triangles": p.len(), "reach_mm": (r * 1e4).round() / 10.0})).collect();
    let kept_reach: Vec<f64> = reach.iter().copied().filter(|&r| r >= min_out).map(|r| (r * 1e4).round() / 10.0).collect();
    let mut k = 0;
    patches.retain(|_| { k += 1; reach[k - 1] >= min_out });
     
    let free_skin = |t: usize| seg.triangle_label[t] == Label::Skin && seg.triangle_piece[t].is_none();
    let mut vert_tris: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut in_patch = vec![false; scan.triangles.len()];
    for &t in patches.iter().flatten() {
        in_patch[t] = true;
    }
    let mut frontier: Vec<usize> = patches.iter().flatten().copied().collect();
    let mut grown = 0usize;
    for _ in 0..MARGIN_RINGS {
        if vert_tris.is_empty() {
            for (t, tri) in scan.triangles.iter().enumerate() {
                if free_skin(t) {
                    for &i in tri {
                        vert_tris.entry(scan.weld.id[i as usize]).or_default().push(t);
                    }
                }
            }
        }
        let mut next = vec![];
        for &t in &frontier {
            for &i in &scan.triangles[t] {
                for &u in vert_tris.get(&scan.weld.id[i as usize]).map_or(&[][..], |v| v.as_slice()) {
                    if !in_patch[u] {
                        let c = scan.triangles[u].iter().fold(V::zeros(), |s, &i| s + v(scan.positions[i as usize])) / 3.0;
                        if eyes.iter().any(|(e, r)| (c - e).norm() < *r) {
                            continue;
                        }
                        in_patch[u] = true;
                        next.push(u);
                    }
                }
            }
        }
        grown += next.len();
        frontier = next;
    }
     
    let candidate = |t: usize| {
        free_skin(t) && !in_patch[t] && {
            let c = scan.triangles[t].iter().fold(V::zeros(), |s, &i| s + v(scan.positions[i as usize])) / 3.0;
            !eyes.iter().any(|(e, r)| (c - e).norm() < *r)
        }
    };
    let mut seen_hole = vec![false; scan.triangles.len()];
    let mut filled = 0usize;
    let mut holes: Vec<usize> = vec![];
    let seeds: Vec<usize> = (0..scan.triangles.len())
        .filter(|&t| in_patch[t])
        .flat_map(|t| scan.triangles[t].iter().flat_map(|&i| vert_tris.get(&scan.weld.id[i as usize]).cloned().unwrap_or_default()))
        .filter(|&u| candidate(u))
        .collect();
    for s0 in seeds {
        if seen_hole[s0] || in_patch[s0] {
            continue;
        }
        let mut island = vec![];
        let mut stack = vec![s0];
        seen_hole[s0] = true;
        let mut size = 0.0;
        while let Some(t) = stack.pop() {
            island.push(t);
            size += area(t);
            if size > HOLE_MAX {
                break;
            }
            for &i in &scan.triangles[t] {
                for &u in vert_tris.get(&scan.weld.id[i as usize]).map_or(&[][..], |v| v.as_slice()) {
                    if !seen_hole[u] && candidate(u) {
                        seen_hole[u] = true;
                        stack.push(u);
                    }
                }
            }
        }
        if size <= HOLE_MAX {
            filled += island.len();
            holes.extend(island);
        }
    }
    for t in holes {
        in_patch[t] = true;
    }
    let all: Vec<usize> = (0..scan.triangles.len()).filter(|&t| in_patch[t]).collect();
    // One rigid piece per material.
    let mut groups: BTreeMap<Option<usize>, Vec<usize>> = BTreeMap::new();
    for &t in &all {
        groups.entry(scan.triangle_material[t]).or_default().push(t);
    }
    let mut out = vec![];
    for (k, (material, triangles)) in groups.into_iter().enumerate() {
        let mut remap = vec![u32::MAX; scan.positions.len()];
        let mut g = Garment {
            name: if k == 0 { "graft".into() } else { format!("graft_{k}") },
            hair: false,
            layer: 0,
            material,
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            triangles: vec![],
            weld_id: vec![],
            influences: vec![],
            rigid: true,
        };
        for t in triangles {
            let tri = scan.triangles[t].map(|i| {
                let i = i as usize;
                if remap[i] == u32::MAX {
                    remap[i] = g.positions.len() as u32;
                    g.positions.push(scan.positions[i]);
                    g.normals.push(scan.normals[i]);
                    g.uvs.push(scan.uvs[i]);
                    g.weld_id.push(scan.weld.id[i]);
                }
                remap[i]
            });
            g.triangles.push(tri);
        }
        out.push(g);
    }
    let report = json!({
        "gap_m": gap,
        "patches": patches.iter().map(|p| json!({"triangles": p.len(), "scan_triangles": p, "centroid_m": a(p.iter().fold(V::zeros(), |s, &t| s + scan.triangles[t].iter().fold(V::zeros(), |s, &i| s + v(scan.positions[i as usize])) / 3.0) / p.len().max(1) as f64), "area_cm2": (p.iter().map(|&t| area(t)).sum::<f64>() * 1e4 * 10.0).round() / 10.0})).collect::<Vec<_>>(),
        "pieces": out.len(),
        "kept_reach_mm": kept_reach,
        "dropped": dropped,
        "margin_triangles": grown,
        "hole_triangles": filled,
    });
    (out, report)
}
