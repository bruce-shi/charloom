//! Stage 3 — garment extraction, clearance and weight transfer.
//!
//! Garments keep the scan's exact geometry, UVs and material. Weights come
//! from the fitted MakeHuman body (barycentric blend at the closest
//! normal-compatible body point) and are then diffused along the garment
//! surface — strongly where the cloth hangs away from the body (skirts,
//! sleeves), weakly where it is tight — so neighbouring vertices never jump
//! between bones. Weights are computed per *welded* scan vertex, so pieces
//! cut from one fused mesh share identical weights along their cut and the
//! seam cannot open during animation.
use crate::geom::{V, Weld, a, percentiles, v};
use crate::scan::{Scan, build_query};
use crate::segment::Segmentation;
use charloom_core::{Influence, Vec3};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub struct Garment {
    pub name: String,
    pub hair: bool,
    /// 0 = worn on the skin; anchors only to garments of lower layers.
    pub layer: u32,
    pub material: Option<usize>,
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<[f32; 2]>,
    pub triangles: Vec<[u32; 3]>,
    /// Scan welded id of each vertex.
    pub weld_id: Vec<usize>,
    pub influences: Vec<Vec<Influence>>,
    /// Hard part (claw, horn, tusk): each connected piece is bound to one
    /// bone, see [`bind`].
    pub rigid: bool,
}

/// Reach of the cloth push across separate fragments and layers, share of
/// the height (5.25 cm at 1.75 m): `spatial_push`. Layers keep their order
/// while the push fades by less than their spacing: at most 1.5 x the push
/// over the reach, so 3 cm pushes ask for 4.5 cm. Over 1.4 cm an inner
/// bodice pushed 16 mm passed the collar 5 mm over it (pushed 11 mm).
const PUSH_REACH: f64 = 0.03;

/// One garment per (piece, material) with compacted vertices.
pub fn extract(scan: &Scan, seg: &Segmentation) -> Vec<Garment> {
    let mut groups: BTreeMap<(usize, Option<usize>), Vec<usize>> = BTreeMap::new();
    for (t, piece) in seg.triangle_piece.iter().enumerate() {
        if let Some(p) = piece {
            groups
                .entry((*p, scan.triangle_material[t]))
                .or_default()
                .push(t);
        }
    }
    let mut out = vec![];
    for ((piece, material), triangles) in groups {
        let mut remap = vec![u32::MAX; scan.positions.len()];
        let mut g = Garment {
            hair: seg.piece_hair[piece],
            layer: seg.piece_layer[piece],
            name: seg.pieces[piece].clone(),
            material,
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            triangles: vec![],
            weld_id: vec![],
            influences: vec![],
            rigid: seg.piece_rigid.get(piece).copied().unwrap_or(false),
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
    // Disambiguate names when a piece spans several materials.
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for g in &mut out {
        let n = seen.entry(g.name.clone()).or_default();
        if *n > 0 {
            g.name = format!("{}_{}", g.name, n);
        }
        *n += 1;
    }
    out
}

fn blend(rows: &[(&[Influence], f64)], limit: usize) -> Vec<Influence> {
    let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
    for (row, w) in rows {
        for i in row.iter() {
            *acc.entry(i.joint).or_default() += i.weight * w;
        }
    }
    let mut out: Vec<Influence> = acc
        .into_iter()
        .filter(|(_, w)| *w > 1e-6)
        .map(|(joint, weight)| Influence { joint, weight })
        .collect();
    out.sort_by(|x, y| y.weight.total_cmp(&x.weight));
    out.truncate(limit);
    let sum: f64 = out.iter().map(|x| x.weight).sum();
    for x in &mut out {
        x.weight /= sum.max(1e-12);
    }
    out
}

pub struct BodyRest<'a> {
    pub positions: &'a [Vec3],
    pub triangles: &'a [[u32; 3]],
    pub influences: &'a [Vec<Influence>],
    pub weld: &'a Weld,
}

 pub fn bind(
    garments: &mut [Garment],
    body: &BodyRest,
    height: f64,
    clearance: f64,
    rigid_bone: impl Fn(usize) -> usize,
    foot_of: impl Fn(usize) -> Option<usize>,
    is_hand: impl Fn(usize) -> bool + Sync,
) -> Value {
    let query = build_query(body.positions, body.triangles);
    let normals = crate::geom::vertex_normals(body.positions, body.triangles, body.weld);
    // Unique welded scan vertices used by any garment.
    let mut ids: Vec<usize> = garments
        .iter()
        .flat_map(|g| g.weld_id.iter().copied())
        .collect();
    ids.sort_unstable();
    ids.dedup();
    let index: std::collections::HashMap<usize, usize> =
        ids.iter().enumerate().map(|(k, &w)| (w, k)).collect();
    let mut position = vec![[0.0; 3]; ids.len()];
    let mut normal = vec![[0.0; 3]; ids.len()];
    let mut hair = vec![false; ids.len()];
    for g in garments.iter() {
        for (i, &w) in g.weld_id.iter().enumerate() {
            position[index[&w]] = g.positions[i];
            normal[index[&w]] = g.normals[i];
            hair[index[&w]] |= g.hair;
        }
    }
    let mut adjacency: Vec<Vec<u32>> = vec![vec![]; ids.len()];
    for g in garments.iter() {
        for t in &g.triangles {
            let k = t.map(|i| index[&g.weld_id[i as usize]] as u32);
            for (x, y) in [(k[0], k[1]), (k[1], k[2]), (k[2], k[0])] {
                if x != y {
                    adjacency[x as usize].push(y);
                    adjacency[y as usize].push(x);
                }
            }
        }
    }
    adjacency.par_iter_mut().for_each(|r| {
        r.sort_unstable();
        r.dedup();
    });
    struct Sample {
        weights: Vec<Influence>,
        distance: f64,
        /// How far the cloth faces the body's way (cosine), the body's
        /// smoothed normal and the signed distance along it, where the cloth
        /// faces the same way (not hair).
        push: Option<(f64, V, f64)>,
    }
    // `triangles[t]` is the body triangle behind query triangle `t`.
    let sample =
        |k: usize, query: &charloom_engine::surface_query::SurfaceQuery, triangles: &[usize]| {
            let p = position[k];
            let hit = query
                .closest_point(p, height, Some((normal[k], 0.0)))
                .or_else(|| query.closest_point(p, height, None));
            let Some(hit) = hit else {
                return Sample {
                    weights: vec![],
                    distance: f64::INFINITY,
                    push: None,
                };
            };
            let tri = body.triangles[triangles[hit.source[1]]];
            let rows: Vec<(&[Influence], f64)> = (0..3)
                .map(|j| {
                    (
                        body.influences[tri[j] as usize].as_slice(),
                        hit.barycentric[j],
                    )
                })
                .collect();
            let nb = (0..3).fold(nalgebra::Vector3::zeros(), |s, j| {
                s + v(normals[tri[j] as usize]) * hit.barycentric[j]
            });
            let nb = nb.try_normalize(1e-12).unwrap_or(v(hit.normal));
            let signed = (v(p) - v(hit.point)).dot(&nb);
            // Never on hair, nor on cloth turned away from the body (deep
            // folds and layers tucked inside it are hidden, not dragged out)
            // unless it is only just inside it (an inner trouser leg against a
            // fuller thigh).
            let facing = v(normal[k]).dot(&nb);
            let push = (!hair[k] && (facing > 0.5 || (signed < 0.0 && signed > -height * 0.008)))
                .then_some((facing, nb, signed));
            Sample {
                weights: blend(&rows, 8),
                distance: hit.distance_m,
                push,
            }
        };
    let all_triangles: Vec<usize> = (0..body.triangles.len()).collect();
    let mut samples: Vec<Sample> = (0..ids.len())
        .into_par_iter()
        .map(|k| sample(k, &query, &all_triangles))
        .collect();
    let mut piece = vec![usize::MAX; ids.len()];
    for (gi, g) in garments.iter().enumerate() {
        for &w in &g.weld_id {
            let k = index[&w];
            if piece[k] == usize::MAX {
                piece[k] = gi;
            }
        }
    }
    // Hand domain: a hand hanging against the hip must not lend its bones to
    // the shorts there (the vertex would follow the arm in the walk). Pieces
    // that are mostly not on the hands re-bind such vertices to the body
    // without the hands; gloves and sleeve cuffs over the hand keep theirs.
    let on_hand = |s: &Sample| {
        s.weights
            .iter()
            .filter(|w| is_hand(w.joint))
            .map(|w| w.weight)
            .sum::<f64>()
            > 0.5
    };
    let mut hand_share = vec![(0usize, 0usize); garments.len()];
    for k in 0..ids.len() {
        hand_share[piece[k]].1 += 1;
        if on_hand(&samples[k]) {
            hand_share[piece[k]].0 += 1;
        }
    }
    let rebind: Vec<usize> = (0..ids.len())
        .filter(|&k| {
            let (h, n) = hand_share[piece[k]];
            (h as f64) < 0.2 * n as f64 && on_hand(&samples[k])
        })
        .collect();
    let mut hand_rebound = 0usize;
    if !rebind.is_empty() {
        let kept: Vec<usize> = (0..body.triangles.len())
            .filter(|&t| {
                body.triangles[t].iter().all(|&i| {
                    body.influences[i as usize]
                        .iter()
                        .filter(|w| is_hand(w.joint))
                        .map(|w| w.weight)
                        .sum::<f64>()
                        < 0.5
                })
            })
            .collect();
        let tris: Vec<[u32; 3]> = kept.iter().map(|&t| body.triangles[t]).collect();
        let no_hands = build_query(body.positions, &tris);
        let redone: Vec<(usize, Sample)> = rebind
            .par_iter()
            .map(|&k| (k, sample(k, &no_hands, &kept)))
            .collect();
        for (k, s) in redone {
            if !s.weights.is_empty() {
                samples[k] = s;
                hand_rebound += 1;
            }
        }
    }
    // The garment fits the body: where the (fuller) fitted body reaches the
    // cloth, the cloth moves out to the clearance, each layer a little
    // further than the one under it, by a field spread smoothly over the
    // cloth. Down to 2 cm deep (5 cm where the cloth faces squarely out); deeper
    // cloth is a fold or a tucked layer and stays hidden.
    let raw: Vec<V> = (0..ids.len())
        .map(|k| {
            let Some((facing, nb, signed)) = samples[k].push else {
                return V::zeros();
            };
            let layer = garments.get(piece[k]).map_or(0, |g| g.layer.min(4)) as f64;
            let want = clearance + 0.0025 * layer * height / 1.75;
            // Cloth that faces squarely the body's way is the outside of the
            // garment over a fuller body, not a fold: it comes out from deeper
            // (a sash or trousers over a curved hip, up to ~5 cm).
            // A rigid prop moves as a whole by its deepest push: a deep
            // one along one normal shoves a bow lying across the chest into
            // the body elsewhere, so props keep the shallow limit.
            let rigid = garments.get(piece[k]).is_some_and(|g| g.rigid);
            let depth = if facing > 0.8 && !rigid { 0.03 } else { 0.012 };
            if signed < want && signed > -height * depth {
                // Along the body normal only: the closest point is not
                // always straight under the cloth (it added a sideways jump).
                nb * (want - signed)
            } else {
                V::zeros()
            }
        })
        .collect();
     
    let mut raw = raw;
    {
        let mut cloth_positions: Vec<Vec3> = vec![];
        let mut cloth_triangles: Vec<[u32; 3]> = vec![];
        let mut cloth_vertex: Vec<usize> = vec![];
        for g in garments.iter().filter(|g| !g.hair && !g.rigid) {
            let o = cloth_positions.len() as u32;
            for &w in &g.weld_id {
                cloth_positions.push(position[index[&w]]);
                cloth_vertex.push(index[&w]);
            }
            cloth_triangles.extend(g.triangles.iter().map(|t| t.map(|i| i + o)));
        }
        // Open edges (necklines, cuffs, hems): skin past an edge finds the
        // edge as its nearest cloth and tore it out into spikes.
        let mut edge_uses: std::collections::HashMap<(usize, usize), u32> = Default::default();
        for t in &cloth_triangles {
            for j in 0..3 {
                let (x, y) = (cloth_vertex[t[j] as usize], cloth_vertex[t[(j + 1) % 3] as usize]);
                *edge_uses.entry((x.min(y), x.max(y))).or_default() += 1;
            }
        }
        let mut open_edge = vec![false; ids.len()];
        for ((x, y), n) in &edge_uses {
            if *n == 1 {
                open_edge[*x] = true;
                open_edge[*y] = true;
            }
        }
        if !cloth_triangles.is_empty() {
            let cloth = build_query(&cloth_positions, &cloth_triangles);
            let reach = height * 0.03;
            let lend: Vec<Option<([usize; 3], [f64; 3], V)>> = (0..body.positions.len())
                .into_par_iter()
                .filter(|&i| body.weld.representative[body.weld.id[i]] == i)
                .map(|i| {
                    let p = body.positions[i];
                    let nb = v(normals[i]).try_normalize(1e-12)?;
                    let hit = cloth.closest_point(p, reach, None)?;
                    let tri = cloth_triangles[hit.source[1]];
                    let ks = tri.map(|c| cloth_vertex[c as usize]);
                    let nc = (0..3)
                        .fold(V::zeros(), |s, j| s + v(normal[ks[j]]) * hit.barycentric[j])
                        .try_normalize(1e-12)?;
                    // Cloth over the body facing its way (not a fold).
                    if nc.dot(&nb) < 0.5 {
                        return None;
                    }
                    let layer = ks
                        .iter()
                        .map(|&k| garments.get(piece[k]).map_or(0, |g| g.layer.min(4)))
                        .max()
                        .unwrap_or(0) as f64;
                    let want = clearance + 0.0025 * layer * height / 1.75;
                    if ks.iter().any(|&k| open_edge[k]) {
                        return None;
                    }
                    let d = v(p) - v(hit.point);
                    let out = d.dot(&nc);
                    // Only cloth straight over the point (not beside it),
                    // and a tip, not a whole region past the cloth.
                    if (d - nc * out).norm() > height * 0.004 || out > height * 0.025 {
                        return None;
                    }
                    (out > -want).then(|| (ks, hit.barycentric, nc * (out + want)))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .collect();
            // To the triangle and its ring: the closest cloth triangle to a
            // tip is not always the one it pierces.
            for (ks, _, d) in lend.into_iter().flatten() {
                for k in ks.into_iter().chain(ks.iter().flat_map(|&k| adjacency[k].iter().map(|&j| j as usize))) {
                    if !open_edge[k] && d.norm() > raw[k].norm() {
                        raw[k] = d;
                    }
                }
            }
        }
    }
    let pushed = raw.iter().filter(|d| d.norm() > 0.0).count();
    let stretch_report = |push: &[V]| -> Value {
        let mut per: Vec<(String, usize)> = vec![];
        for g in garments.iter().filter(|g| !g.hair) {
            let mut bad = 0usize;
            for t in &g.triangles {
                let k = t.map(|i| index[&g.weld_id[i as usize]]);
                let before = (0..3).map(|j| (v(position[k[j]]) - v(position[k[(j + 1) % 3]])).norm()).fold(0.0, f64::max);
                let after = (0..3)
                    .map(|j| ((v(position[k[j]]) + push[k[j]]) - (v(position[k[(j + 1) % 3]]) + push[k[(j + 1) % 3]])).norm())
                    .fold(0.0, f64::max);
                if after > 2.0 * before + 0.002 {
                    bad += 1;
                }
            }
            per.push((g.name.clone(), bad));
        }
        json!(per)
    };
     
    let movable: Vec<bool> = vec![true; ids.len()];
    let raw = spatial_push(&position, &raw, &movable, PUSH_REACH * height);
    let cloth_push = spread_cloth_push(&adjacency, &raw);
    let stretched = stretch_report(&cloth_push);
    let mut anchored: Vec<bool> = samples.iter().map(|s| s.distance < height * 0.01).collect();
    let mut initial: Vec<Vec<Influence>> = samples.iter().map(|s| s.weights.clone()).collect();
     
    let mut layered = 0usize;
    let layered_guidance = garments.iter().any(|g| g.layer > 0);
    for (gi, _) in garments.iter().enumerate() {
        let mut inner_positions = vec![];
        let mut inner_triangles = vec![];
        let mut inner_ids = vec![];
        for (gj, g) in garments.iter().enumerate() {
            // With guided layers an outer garment rests only on inner ones;
            // without them any other garment may be the support.
            if gj == gi || g.hair || (layered_guidance && g.layer >= garments[gi].layer) {
                continue;
            }
            for t in &g.triangles {
                let kk = t.map(|i| index[&g.weld_id[i as usize]]);
                if kk.iter().all(|&q| anchored[q] && piece[q] == gj) {
                    let o = inner_positions.len() as u32;
                    for &q in &kk {
                        inner_positions.push(position[q]);
                        inner_ids.push(q);
                    }
                    inner_triangles.push([o, o + 1, o + 2]);
                }
            }
        }
        if inner_triangles.is_empty() {
            continue;
        }
        let inner = build_query(&inner_positions, &inner_triangles);
        let found: Vec<(usize, Vec<Influence>)> = (0..ids.len())
            .into_par_iter()
            .filter(|&q| piece[q] == gi && !anchored[q] && !hair[q])
            .filter_map(|q| {
                let hit = inner.closest_point(position[q], height * 0.012, None)?;
                let t = inner_triangles[hit.source[1]];
                let rows: Vec<(&[Influence], f64)> = (0..3)
                    .map(|j| {
                        (
                            samples[inner_ids[t[j] as usize]].weights.as_slice(),
                            hit.barycentric[j],
                        )
                    })
                    .collect();
                Some((q, blend(&rows, 8)))
            })
            .collect();
        for (q, w) in found {
            initial[q] = w;
            anchored[q] = true;
            layered += 1;
        }
    }
    let (weights, harmonic) = harmonic_weights(
        &initial,
        &anchored,
        &adjacency,
        &position,
        &piece,
        height * 0.008,
    );
    let mut final_weights: Vec<Vec<Influence>> = weights
        .iter()
        .map(|w| blend(&[(w.as_slice(), 1.0)], 4))
        .collect();
    // Rigid parts: every connected piece follows the one bone that dominates
    // the body where it attaches (its vertices nearest the body).
    let rigid_pieces: Vec<Value> = garments
        .iter()
        .filter(|g| g.rigid)
        .flat_map(|g| {
            let keys: Vec<usize> = g.weld_id.iter().map(|w| index[w]).collect();
            rigid_components(&keys, &g.triangles)
                .into_iter()
                .filter_map(|component| {
                    let attached = attachment(&component, |k| samples[k].distance, height * 0.01);
                    let folded: Vec<Vec<Influence>> = attached
                        .iter()
                        .map(|&(k, _)| {
                            samples[k]
                                .weights
                                .iter()
                                .map(|x| Influence {
                                    joint: rigid_bone(x.joint),
                                    weight: x.weight,
                                })
                                .collect()
                        })
                        .collect();
                    let rows: Vec<(&[Influence], f64)> = folded
                        .iter()
                        .zip(&attached)
                        .map(|(row, &(_, w))| (row.as_slice(), w))
                        .collect();
                    let bone = blend(&rows, 1).first()?.joint;
                    for &k in &component {
                        final_weights[k] = vec![Influence {
                            joint: bone,
                            weight: 1.0,
                        }];
                    }
                    Some(json!({"garment": g.name, "vertices": component.len(), "joint": bone}))
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let footwear = stiffen_footwear(garments, &index, &mut final_weights, &foot_of);
    let unbound = final_weights.iter().filter(|w| w.is_empty()).count();
    for g in garments.iter_mut() {
        g.influences = g
            .weld_id
            .iter()
            .map(|w| final_weights[index[w]].clone())
            .collect();
        if g.rigid {
            // A rigid part (claw, horn, quiver, bow) keeps its exact shape
            // but is not left inside a fuller body: each connected piece
            // moves out as a whole by the deepest push among its vertices.
            let keys: Vec<usize> = g.weld_id.iter().map(|w| index[w]).collect();
            for component in rigid_components(&keys, &g.triangles) {
                let shift = component
                    .iter()
                    .map(|&k| cloth_push[k])
                    .max_by(|p, q| p.norm().total_cmp(&q.norm()))
                    .unwrap_or_else(V::zeros);
                if shift.norm() > 0.0 {
                    let members: std::collections::HashSet<usize> = component.into_iter().collect();
                    for (i, w) in g.weld_id.iter().enumerate() {
                        if members.contains(&index[w]) {
                            g.positions[i] = a(v(g.positions[i]) + shift);
                        }
                    }
                }
            }
            continue;
        }
        for (i, w) in g.weld_id.iter().enumerate() {
            let d = cloth_push[index[w]];
            if d.norm() > 0.0 {
                g.positions[i] = a(v(g.positions[i]) + d);
            }
        }
    }
    json!({
        "garments": garments.iter().map(|g| json!({"name":g.name,"vertices":g.positions.len(),"triangles":g.triangles.len()})).collect::<Vec<_>>(),
        "unique_vertices": ids.len(),
        "unbound_vertices": unbound,
        "clearance_pushed_vertices": pushed,
        "stretched_triangles": stretched,
        "clearance_push_m": {
            "max": raw.iter().map(|d| d.norm()).fold(0.0, f64::max),
            "spread_vertices": cloth_push.iter().filter(|d| d.norm() > 0.0).count(),
        },
        "anchored_vertices": anchored.iter().filter(|a| **a).count(),
        "layer_anchored_vertices": layered,
        "harmonic": harmonic,
        "rigid_pieces": rigid_pieces,
        "footwear": footwear,
        "hand_domain_rebound_vertices": hand_rebound,
        "garment_to_body_distance_m": percentiles(samples.iter().map(|s| s.distance).collect()),
    })
}

/// Share of the transferred toe-bone influence footwear keeps.
pub const FOOTWEAR_TOE_SHARE: f64 = 0.15;

fn is_footwear_name(name: &str) -> bool {
    let n = name.to_lowercase();
    [
        "shoe", "boot", "sandal", "sneaker", "slipper", "footwear", "clog", "heel", "loafer",
        "trainer", "moccasin", "geta", "sabaton",
    ]
    .iter()
    .any(|k| n.contains(k))
}

/// Caps toe influence on footwear (see [`bind`]). Weights are per welded
/// vertex, so a seam shared with a trouser leg stays closed.
fn stiffen_footwear(
    garments: &[Garment],
    index: &std::collections::HashMap<usize, usize>,
    weights: &mut [Vec<Influence>],
    foot_of: &impl Fn(usize) -> Option<usize>,
) -> Value {
    let mut out = vec![];
    for g in garments.iter().filter(|g| !g.hair && !g.rigid) {
        let keys: Vec<usize> = g.weld_id.iter().map(|w| index[w]).collect();
        let (mut on_feet, mut total) = (0.0, 0.0);
        for &k in &keys {
            for x in &weights[k] {
                total += x.weight;
                if foot_of(x.joint).is_some() {
                    on_feet += x.weight;
                }
            }
        }
        let share = on_feet / total.max(1e-12);
        let by_name = is_footwear_name(&g.name);
        if !(by_name || share > 0.6) {
            continue;
        }
        let mut changed = 0usize;
        let mut toe_before: f64 = 0.0;
        let mut toe_after: f64 = 0.0;
        for &k in &keys {
            let row = &weights[k];
            let toe: f64 = row
                .iter()
                .filter(|x| foot_of(x.joint).is_some_and(|f| f != x.joint))
                .map(|x| x.weight)
                .sum();
            if toe <= 0.0 {
                continue;
            }
            let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
            for x in row {
                match foot_of(x.joint).filter(|&f| f != x.joint) {
                    Some(foot) => {
                        *acc.entry(x.joint).or_default() += x.weight * FOOTWEAR_TOE_SHARE;
                        *acc.entry(foot).or_default() += x.weight * (1.0 - FOOTWEAR_TOE_SHARE);
                    }
                    None => *acc.entry(x.joint).or_default() += x.weight,
                }
            }
            let rows: Vec<Influence> = acc
                .into_iter()
                .map(|(joint, weight)| Influence { joint, weight })
                .collect();
            let new = blend(&[(rows.as_slice(), 1.0)], 4);
            toe_before = toe_before.max(toe);
            toe_after = toe_after.max(
                new.iter()
                    .filter(|x| foot_of(x.joint).is_some_and(|f| f != x.joint))
                    .map(|x| x.weight)
                    .sum(),
            );
            weights[k] = new;
            changed += 1;
        }
        out.push(json!({"garment": g.name, "by_name": by_name, "foot_weight_share": share, "vertices_changed": changed, "max_toe_weight_before": toe_before, "max_toe_weight_after": toe_after}));
    }
    json!(out)
}

/// Connected components (sorted keys) of a piece whose vertex `i` has
/// key `keys[i]` (welded: pieces cut from one fused mesh stay connected).
fn rigid_components(keys: &[usize], triangles: &[[u32; 3]]) -> Vec<Vec<usize>> {
    let mut parent: std::collections::BTreeMap<usize, usize> =
        keys.iter().map(|&k| (k, k)).collect();
    fn find(parent: &mut std::collections::BTreeMap<usize, usize>, mut k: usize) -> usize {
        while parent[&k] != k {
            let up = parent[&parent[&k]];
            parent.insert(k, up);
            k = up;
        }
        k
    }
    for t in triangles {
        let r0 = find(&mut parent, keys[t[0] as usize]);
        for &i in &t[1..] {
            let r = find(&mut parent, keys[i as usize]);
            if r != r0 {
                parent.insert(r, r0);
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let all: Vec<usize> = parent.keys().copied().collect();
    for k in all {
        let r = find(&mut parent, k);
        groups.entry(r).or_default().push(k);
    }
    groups.into_values().collect()
}

/// The attachment of a rigid piece: its vertices within `band` of its
/// closest approach to the body, weighted down linearly towards the band's
/// edge.
fn attachment(
    component: &[usize],
    distance: impl Fn(usize) -> f64,
    band: f64,
) -> Vec<(usize, f64)> {
    let nearest = component
        .iter()
        .map(|&k| distance(k))
        .fold(f64::INFINITY, f64::min);
    if !nearest.is_finite() {
        return vec![];
    }
    component
        .iter()
        .map(|&k| (k, 1.0 - (distance(k) - nearest) / band))
        .filter(|(_, w)| *w > 0.0)
        .collect()
}

/// Body triangles completely covered by cloth (every vertex sees a garment
/// along its outward normal within `reach`), eroded by one ring so the body
/// stays intact under cuffs, collars and hems.
pub fn covered_body_triangles(body: &BodyRest, garments: &[Garment], reach: f64) -> Vec<bool> {
    let mut positions = vec![];
    let mut triangles = vec![];
    // Hair never hides the body: it is no closed shell (the scalp and skull
    // show between strands and when the hair swings), so the head under it
    // stays.
    for g in garments.iter().filter(|g| !g.hair) {
        let o = positions.len() as u32;
        positions.extend_from_slice(&g.positions);
        triangles.extend(g.triangles.iter().map(|t| t.map(|i| i + o)));
    }
    if triangles.is_empty() {
        return vec![false; body.triangles.len()];
    }
    let query = build_query(&positions, &triangles);
    let normals = crate::geom::vertex_normals(body.positions, body.triangles, body.weld);
    let n = body.weld.count;
    let covered: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|w| {
            let i = body.weld.representative[w];
            let origin = a(v(body.positions[i]) + v(normals[i]) * 1e-4);
            query
                .first_ray_distance(origin, normals[i], reach)
                .is_some()
        })
        .collect();
    // Only body well inside the cloth goes: a hole edge right at a collar,
    // cuff or hem shows the empty body wherever the garment stands off the
    // skin or moves in animation. Keep a band of `margin` (surface distance)
    // under every garment edge.
    let adjacency = crate::geom::adjacency(body.triangles, body.weld);
    let margin = reach * 0.6;
    let mut dist = vec![f64::INFINITY; n];
    let mut heap = std::collections::BinaryHeap::new();
    for w in 0..n {
        if !covered[w] {
            dist[w] = 0.0;
            heap.push(std::cmp::Reverse((0u64, w)));
        }
    }
    let pos = |w: usize| v(body.positions[body.weld.representative[w]]);
    // Heap key: distance in micrometres.
    while let Some(std::cmp::Reverse((key, w))) = heap.pop() {
        let d = key as f64 * 1e-6;
        if d > dist[w] + 1e-6 || d > margin {
            continue;
        }
        for &j in &adjacency[w] {
            let j = j as usize;
            let nd = d + (pos(j) - pos(w)).norm();
            if nd < dist[j] {
                dist[j] = nd;
                heap.push(std::cmp::Reverse(((nd * 1e6) as u64, j)));
            }
        }
    }
    let eroded: Vec<bool> = (0..n).map(|w| covered[w] && dist[w] > margin).collect();
    body.triangles
        .iter()
        .map(|t| t.iter().all(|&i| eroded[body.weld.id[i as usize]]))
        .collect()
}

 fn harmonic_weights(
    initial: &[Vec<Influence>],
    anchored: &[bool],
    adjacency: &[Vec<u32>],
    position: &[Vec3],
    piece: &[usize],
    cell: f64,
) -> (Vec<Vec<Influence>>, Value) {
    let n = initial.len();
    let mut joints: Vec<usize> = initial.iter().flatten().map(|i| i.joint).collect();
    joints.sort_unstable();
    joints.dedup();
    let column: std::collections::HashMap<usize, usize> =
        joints.iter().enumerate().map(|(c, &j)| (j, c)).collect();
    let k = joints.len().max(1);
    let mut fine = vec![0f32; n * k];
    for (i, row) in initial.iter().enumerate() {
        for w in row {
            fine[i * k + column[&w.joint]] = w.weight as f32;
        }
    }
    // Voxel clusters per piece.
    let mut cluster_of = vec![0usize; n];
    let mut keys: std::collections::HashMap<(i64, i64, i64, usize), usize> = Default::default();
    for i in 0..n {
        let p = position[i];
        let key = (
            (p[0] / cell).floor() as i64,
            (p[1] / cell).floor() as i64,
            (p[2] / cell).floor() as i64,
            piece[i],
        );
        let next = keys.len();
        cluster_of[i] = *keys.entry(key).or_insert(next);
    }
    let m = keys.len();
    let mut coarse = vec![0f32; m * k];
    let mut members = vec![0f32; m];
    let mut anchor_votes = vec![0i32; m];
    for i in 0..n {
        let c = cluster_of[i];
        members[c] += 1.0;
        anchor_votes[c] += if anchored[i] { 1 } else { -1 };
        for j in 0..k {
            coarse[c * k + j] += fine[i * k + j];
        }
    }
    for c in 0..m {
        for j in 0..k {
            coarse[c * k + j] /= members[c].max(1.0);
        }
    }
    let mut coarse_adj: Vec<Vec<u32>> = vec![vec![]; m];
    for i in 0..n {
        for &j in &adjacency[i] {
            let (a0, b0) = (cluster_of[i], cluster_of[j as usize]);
            if a0 != b0 {
                coarse_adj[a0].push(b0 as u32);
            }
        }
    }
    coarse_adj.par_iter_mut().for_each(|r| {
        r.sort_unstable();
        r.dedup();
    });
    // Only where the nearest-bone assignment jumps between neighbours (the
    // midline between legs, a split robe, a sleeve hanging over the ribs)
    // does closest-point transfer tear cloth. Loose tubes around one limb are
    // already right. Solve only in a band around those jumps.
    let l1 = |a0: usize, b0: usize| {
        (0..k)
            .map(|j| (coarse[a0 * k + j] - coarse[b0 * k + j]).abs())
            .sum::<f32>()
    };
    let mut seam: Vec<bool> = (0..m)
        .map(|c| coarse_adj[c].iter().any(|&d| l1(c, d as usize) > 0.6))
        .collect();
    let seam_clusters = seam.iter().filter(|x| **x).count();
    for _ in 0..3 {
        let grow = seam.clone();
        for c in 0..m {
            if grow[c] {
                for &d in &coarse_adj[c] {
                    seam[d as usize] = true;
                }
            }
        }
    }
    let fixed: Vec<bool> = (0..m).map(|c| anchor_votes[c] > 0 || !seam[c]).collect();
    // Jacobi relaxation of the free clusters (Dirichlet problem).
    let free_clusters = fixed.iter().filter(|f| !**f).count();
    let mut iterations = 0;
    let mut change = 0f32;
    if fixed.iter().any(|f| *f) && free_clusters > 0 {
        for it in 0..1500 {
            let next: Vec<f32> = (0..m)
                .into_par_iter()
                .flat_map_iter(|c| {
                    let row: Vec<f32> = if fixed[c] || coarse_adj[c].is_empty() {
                        coarse[c * k..(c + 1) * k].to_vec()
                    } else {
                        let mut acc = vec![0f32; k];
                        for &d in &coarse_adj[c] {
                            for j in 0..k {
                                acc[j] += coarse[d as usize * k + j];
                            }
                        }
                        let inv = 1.0 / coarse_adj[c].len() as f32;
                        acc.iter().map(|v| v * inv).collect()
                    };
                    row
                })
                .collect();
            change = next
                .iter()
                .zip(&coarse)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            coarse = next;
            iterations = it + 1;
            if change < 1e-4 {
                break;
            }
        }
    }
     
    let free: Vec<bool> = (0..n)
        .map(|i| !anchored[i] && !fixed[cluster_of[i]])
        .collect();
    for i in 0..n {
        if free[i] {
            let c = cluster_of[i];
            fine[i * k..(i + 1) * k].copy_from_slice(&coarse[c * k..(c + 1) * k]);
        }
    }
    for _ in 0..6 {
        let next: Vec<f32> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let row: Vec<f32> = if !free[i] || adjacency[i].is_empty() {
                    fine[i * k..(i + 1) * k].to_vec()
                } else {
                    let mut acc: Vec<f32> = fine[i * k..(i + 1) * k].to_vec();
                    for &d in &adjacency[i] {
                        for j in 0..k {
                            acc[j] += fine[d as usize * k + j];
                        }
                    }
                    let inv = 1.0 / (1 + adjacency[i].len()) as f32;
                    acc.iter().map(|v| v * inv).collect()
                };
                row
            })
            .collect();
        fine = next;
    }
    let weights: Vec<Vec<Influence>> = (0..n)
        .map(|i| {
            let row: Vec<Influence> = (0..k)
                .filter(|&j| fine[i * k + j] > 1e-5)
                .map(|j| Influence {
                    joint: joints[j],
                    weight: fine[i * k + j] as f64,
                })
                .collect();
            blend(&[(row.as_slice(), 1.0)], 8)
        })
        .collect();
    (
        weights,
        json!({"clusters":m,"seam_clusters":seam_clusters,"free_clusters":free_clusters,"free_vertices":free.iter().filter(|f| **f).count(),"cell_m":cell,"iterations":iterations,"final_change":change,"joints":k}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
         #[test]
    fn seam_band_turns_a_weight_jump_into_a_smooth_ramp() {
        let n: usize = 60;
        let position: Vec<Vec3> = (0..n).map(|i| [i as f64 * 0.01, 0.0, 0.0]).collect();
        let adjacency: Vec<Vec<u32>> = (0..n)
            .map(|i| {
                [i.checked_sub(1), (i + 1 < n).then_some(i + 1)]
                    .into_iter()
                    .flatten()
                    .map(|j| j as u32)
                    .collect()
            })
            .collect();
        let initial: Vec<Vec<Influence>> = (0..n)
            .map(|i| {
                vec![Influence {
                    joint: usize::from(i >= n / 2),
                    weight: 1.0,
                }]
            })
            .collect();
        let anchored: Vec<bool> = (0..n).map(|i| !(10..50).contains(&i)).collect();
        let (weights, report) = harmonic_weights(
            &initial,
            &anchored,
            &adjacency,
            &position,
            &vec![0; n],
            0.005,
        );
        assert!(report["seam_clusters"].as_u64().unwrap() >= 2);
        let w1: Vec<f64> = weights
            .iter()
            .map(|r| r.iter().find(|x| x.joint == 1).map_or(0.0, |x| x.weight))
            .collect();
        for pair in w1.windows(2) {
            assert!(pair[1] + 1e-6 >= pair[0], "ramp must be monotone");
            assert!(pair[1] - pair[0] < 0.25, "no tearing jump: {pair:?}");
        }
        assert!(
            w1[5] < 1e-6 && w1[55] > 1.0 - 1e-6,
            "anchors keep their weights"
        );
        for r in &weights {
            assert!((r.iter().map(|x| x.weight).sum::<f64>() - 1.0).abs() < 1e-6);
        }
    }

    /// Two rigid spikes over a body whose weights blend from joint 0
    /// (x < 0) to joint 1 (x > 0): each follows the joint where it
    /// attaches, even where it leans over the other side, and keeps its
    /// shape; a non-rigid copy takes blended weights.
    #[test]
    fn rigid_pieces_follow_the_bone_where_they_attach() {
        let body_positions: Vec<Vec3> = vec![
            [-1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
        ];
        let body_triangles = vec![[0u32, 1, 2], [0, 2, 3]];
        let influences: Vec<Vec<Influence>> = body_positions
            .iter()
            .map(|p| {
                vec![Influence {
                    joint: usize::from(p[0] > 0.0),
                    weight: 1.0,
                }]
            })
            .collect();
        let weld = Weld::new(&body_positions, 1e-9);
        // A spike rooted at x = root, its tip leaning to x = root - lean.
        let spike = |root: f64, lean: f64, id0: usize, rigid: bool| Garment {
            name: "spike".into(),
            hair: false,
            layer: 0,
            material: None,
            positions: vec![
                [root - 0.02, 0.001, 0.0],
                [root + 0.02, 0.001, 0.0],
                [root - lean, 0.3, 0.0],
                [root - lean, 0.3, 0.05],
            ],
            normals: vec![[0.0, 1.0, 0.0]; 4],
            uvs: vec![[0.0, 0.0]; 4],
            triangles: vec![[0, 1, 2], [1, 3, 2]],
            weld_id: (id0..id0 + 4).collect(),
            influences: vec![],
            rigid,
        };
        let mut garments = vec![
            spike(0.5, 0.9, 0, true),
            spike(-0.6, -0.9, 10, true),
            spike(0.5, 0.9, 20, false),
        ];
        let before: Vec<Vec<Vec3>> = garments.iter().map(|g| g.positions.clone()).collect();
        let report = bind(
            &mut garments,
            &BodyRest {
                positions: &body_positions,
                triangles: &body_triangles,
                influences: &influences,
                weld: &weld,
            },
            1.0,
            0.004,
            |j| j,
            |_| None,
            |_| false,
        );
        for (g, joint) in garments[..2].iter().zip([1, 0]) {
            for row in &g.influences {
                assert_eq!(row.len(), 1);
                assert_eq!((row[0].joint, row[0].weight), (joint, 1.0));
            }
        }
        // A rigid piece keeps its exact shape: if it moves out of the body,
        // all of it moves by the same shift.
        let shift = |k: usize| [0, 1, 2].map(|c| garments[0].positions[k][c] - before[0][k][c]);
        for k in 1..before[0].len() {
            for c in 0..3 {
                assert!((shift(k)[c] - shift(0)[c]).abs() < 1e-9);
            }
        }
        assert_eq!(report["rigid_pieces"].as_array().unwrap().len(), 2);
        // Without `rigid` the leaning tip blends towards joint 0.
        let tip = &garments[2].influences[2];
        assert!(tip.iter().any(|x| x.joint == 0 && x.weight > 0.1));
    }

    /// A shoe over a foot whose front half is weighted to a toe bone keeps
    /// only a small toe share; a piece of another name weighted mostly to the
    /// feet is treated the same.
    #[test]
    fn footwear_follows_the_foot_not_the_toes() {
        let body_positions: Vec<Vec3> = vec![
            [-0.05, 0.0, -0.1],
            [0.05, 0.0, -0.1],
            [0.05, 0.0, 0.15],
            [-0.05, 0.0, 0.15],
        ];
        let body_triangles = vec![[0u32, 2, 1], [0, 3, 2]];
        // Joint 0: lower leg, 1: foot, 2: toe (child of the foot).
        let influences: Vec<Vec<Influence>> = body_positions
            .iter()
            .map(|p| {
                vec![Influence {
                    joint: if p[2] > 0.0 { 2 } else { 1 },
                    weight: 1.0,
                }]
            })
            .collect();
        let weld = Weld::new(&body_positions, 1e-9);
        let piece = |name: &str, id0: usize| Garment {
            name: name.into(),
            hair: false,
            layer: 0,
            material: None,
            positions: vec![
                [-0.04, 0.005, -0.08],
                [0.04, 0.005, -0.08],
                [0.04, 0.005, 0.14],
                [-0.04, 0.005, 0.14],
            ],
            normals: vec![[0.0, 1.0, 0.0]; 4],
            uvs: vec![[0.0, 0.0]; 4],
            triangles: vec![[0, 2, 1], [0, 3, 2]],
            weld_id: (id0..id0 + 4).collect(),
            influences: vec![],
            rigid: false,
        };
        let mut garments = vec![piece("shoes", 0), piece("wrap", 10)];
        let report = bind(
            &mut garments,
            &BodyRest {
                positions: &body_positions,
                triangles: &body_triangles,
                influences: &influences,
                weld: &weld,
            },
            1.0,
            0.004,
            |j| j,
            |j| match j {
                1 | 2 => Some(1),
                _ => None,
            },
            |_| false,
        );
        let toe = |g: &Garment| {
            g.influences
                .iter()
                .map(|r| {
                    r.iter()
                        .filter(|x| x.joint == 2)
                        .map(|x| x.weight)
                        .sum::<f64>()
                })
                .fold(0.0, f64::max)
        };
        // Both are weighted to the feet: capped.
        for g in &garments[..2] {
            let t = toe(g);
            assert!(t > 0.0 && t <= FOOTWEAR_TOE_SHARE + 1e-9, "{}: {t}", g.name);
        }
        assert_eq!(report["footwear"].as_array().unwrap().len(), 2);
        assert_eq!(report["footwear"][0]["by_name"], json!(true));
        assert_eq!(report["footwear"][1]["by_name"], json!(false));
    }

    /// Shorts over a hip (joint 0) with a hand (joint 1) resting against one
    /// corner: that corner must not take the hand's bone.
    #[test]
    fn a_hand_against_the_hip_lends_no_weight_to_the_shorts() {
        let mut body_positions: Vec<Vec3> = vec![
            [-0.2, 0.0, -0.2],
            [0.2, 0.0, -0.2],
            [0.2, 0.0, 0.2],
            [-0.2, 0.0, 0.2],
        ];
        // A small hand quad just above the shorts' corner.
        body_positions.extend([
            [0.17, 0.009, 0.17],
            [0.19, 0.009, 0.17],
            [0.19, 0.009, 0.19],
            [0.17, 0.009, 0.19],
        ]);
        let body_triangles = vec![[0u32, 2, 1], [0, 3, 2], [4, 6, 5], [4, 7, 6]];
        let influences: Vec<Vec<Influence>> = (0..8)
            .map(|i| {
                vec![Influence {
                    joint: usize::from(i >= 4),
                    weight: 1.0,
                }]
            })
            .collect();
        let weld = Weld::new(&body_positions, 1e-9);
        let n = 11;
        let positions: Vec<Vec3> = (0..n * n)
            .map(|i| {
                let (x, z) = (
                    (i % n) as f64 / (n - 1) as f64,
                    (i / n) as f64 / (n - 1) as f64,
                );
                [-0.18 + 0.36 * x, 0.008, -0.18 + 0.36 * z]
            })
            .collect();
        let mut triangles = vec![];
        for r in 0..n - 1 {
            for c in 0..n - 1 {
                let i = (r * n + c) as u32;
                let m = n as u32;
                triangles.push([i, i + m, i + 1]);
                triangles.push([i + 1, i + m, i + m + 1]);
            }
        }
        let mut garments = vec![Garment {
            name: "shorts".into(),
            hair: false,
            layer: 0,
            material: None,
            normals: vec![[0.0, 1.0, 0.0]; positions.len()],
            uvs: vec![[0.0, 0.0]; positions.len()],
            weld_id: (0..positions.len()).collect(),
            positions,
            triangles,
            influences: vec![],
            rigid: false,
        }];
        let report = bind(
            &mut garments,
            &BodyRest {
                positions: &body_positions,
                triangles: &body_triangles,
                influences: &influences,
                weld: &weld,
            },
            1.0,
            0.004,
            |j| j,
            |_| None,
            |j| j == 1,
        );
        assert!(report["hand_domain_rebound_vertices"].as_u64().unwrap() > 0);
        for row in &garments[0].influences {
            assert!(row.iter().all(|x| x.joint == 0), "{row:?}");
        }
    }

    #[test]
    fn tube_garment_away_from_any_seam_keeps_transferred_weights() {
        let n: usize = 30;
        let position: Vec<Vec3> = (0..n).map(|i| [i as f64 * 0.01, 0.0, 0.0]).collect();
        let adjacency: Vec<Vec<u32>> = (0..n)
            .map(|i| {
                [i.checked_sub(1), (i + 1 < n).then_some(i + 1)]
                    .into_iter()
                    .flatten()
                    .map(|j| j as u32)
                    .collect()
            })
            .collect();
        let initial: Vec<Vec<Influence>> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                vec![
                    Influence {
                        joint: 3,
                        weight: 1.0 - t * 0.3,
                    },
                    Influence {
                        joint: 4,
                        weight: t * 0.3,
                    },
                ]
            })
            .collect();
        let (weights, _) = harmonic_weights(
            &initial,
            &vec![false; n],
            &adjacency,
            &position,
            &vec![0; n],
            0.005,
        );
        for (a, b) in weights.iter().zip(&initial) {
            let wa = a.iter().find(|x| x.joint == 3).unwrap().weight;
            let wb = b.iter().find(|x| x.joint == 3).unwrap().weight
                / b.iter().map(|x| x.weight).sum::<f64>();
            assert!((wa - wb).abs() < 1e-5);
        }
    }
}

 fn spatial_push(position: &[Vec3], raw: &[V], movable: &[bool], reach: f64) -> Vec<V> {
    use std::collections::HashMap;
    let cell = reach * 0.5;
    let key = |p: V| [p.x, p.y, p.z].map(|x| (x / cell).floor() as i64);
    // Per cell: summed position, summed push, count, strongest push size.
    let mut cells: HashMap<[i64; 3], (V, V, f64, f64)> = HashMap::new();
    for k in (0..raw.len()).filter(|&k| movable[k] && raw[k].norm() > 0.0) {
        let p = v(position[k]);
        let e = cells.entry(key(p)).or_insert((V::zeros(), V::zeros(), 0.0, 0.0));
        e.0 += p;
        e.1 += raw[k];
        e.2 += 1.0;
        e.3 = e.3.max(raw[k].norm());
    }
    if cells.is_empty() {
        return raw.to_vec();
    }
    (0..raw.len())
        .into_par_iter()
        .map(|k| {
            if !movable[k] {
                return raw[k];
            }
            let p = v(position[k]);
            let c = key(p);
            let (mut size, mut dir) = (raw[k].norm(), raw[k]);
            let mut any = false;
            for dx in -2..=2 {
                for dy in -2..=2 {
                    for dz in -2..=2 {
                        let Some((sp, sd, n, top)) = cells.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) else { continue };
                        let x = ((sp / *n - p).norm() / reach).min(1.0);
                        let w = 1.0 - x * x * (3.0 - 2.0 * x);
                        if w <= 0.0 {
                            continue;
                        }
                        any = true;
                        size = size.max(top * w);
                        dir += sd * w;
                    }
                }
            }
            match dir.try_normalize(1e-12) {
                Some(d) if any => d * size,
                _ => raw[k],
            }
        })
        .collect()
}

fn spread_cloth_push(adjacency: &[Vec<u32>], raw: &[V]) -> Vec<V> {
    let n = raw.len();
    let mut d = raw.to_vec();
    let mean = |d: &[V], k: usize| -> V {
        if adjacency[k].is_empty() {
            return V::zeros();
        }
        adjacency[k]
            .iter()
            .fold(V::zeros(), |s, &j| s + d[j as usize])
            / adjacency[k].len() as f64
    };
    for _ in 0..6 {
        d = (0..n)
            .map(|k| {
                let m = mean(&d, k) * 0.7;
                if m.norm() > raw[k].norm() { m } else { raw[k] }
            })
            .collect();
    }
    for _ in 0..2 {
        d = (0..n)
            .map(|k| {
                let m = (d[k] + mean(&d, k)) * 0.5;
                if m.norm() > raw[k].norm() { m } else { raw[k] }
            })
            .collect();
    }
    d
}

#[cfg(test)]
mod push_tests {
    use super::*;

    #[test]
    fn cloth_push_spreads_without_dropping_below_the_raw_push() {
        let n: usize = 21;
        let adjacency: Vec<Vec<u32>> = (0..n)
            .map(|i| {
                [i.checked_sub(1), (i + 1 < n).then_some(i + 1)]
                    .into_iter()
                    .flatten()
                    .map(|j| j as u32)
                    .collect()
            })
            .collect();
        let mut raw = vec![V::zeros(); n];
        raw[10] = V::new(0.0, 0.0, 0.01);
        let d = spread_cloth_push(&adjacency, &raw);
        assert!((d[10] - raw[10]).norm() < 1e-12);
        assert!(d[11].z > 0.0 && d[11].z < 0.01 && d[13].z < d[11].z);
        assert!((d[9].z - d[11].z).abs() < 1e-12);
    }
}
