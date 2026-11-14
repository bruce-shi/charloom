//! Creature-like humanoids with fewer digits per hand (`subject.hand_digits`
//! in the guidance: a wolfman's four-fingered paw, a cartoon three-digit
//! hand).
//!
//! MakeHuman's hand keeps all five fingers up to the output, so every stage
//! works on MakeHuman's own topology:
//!
//! * the finger mapping (`fingers.rs`) skips the missing digits;
//! * the shape fit and the body fit take no data on them ([`digit_mask`]),
//!   so they are never pulled onto the scan's neighbouring fingers (that
//!   merged fingers and inverted triangles), and garments do not bind to
//!   them;
//! * where the final body mesh is assembled, [`cut_digits`] removes their
//!   triangles (vertices dominated by the digit's `finger{F}-1..3` bones;
//!   the metacarpal and palm stay), caps each hole with a small dome whose
//!   UVs, normals and weights come from the boundary ring, compacts the mesh
//!   and moves the stump's remaining weight on the digit onto its
//!   metacarpal. The bones stay in the skeleton: the rig is MakeHuman's
//!   standard one and animations still retarget.
//!
//! The fingers kept span the palm ([`kept`]): a thumb and three fingers keep
//! MakeHuman's index, middle and little finger (the ring finger is cut), a
//! thumb and two the index and little finger, a thumb and one the middle
//! finger. Cut from the outside in, the kept fingers bunched on the thumb
//! side of a palm still five fingers wide, with a stump on its outer edge;
//! a cartoon hand's few fingers spread across the whole palm. The scan's
//! digits are still numbered from the thumb (`finger1..n` landmarks) and
//! carried by the kept fingers in order. The cap over a cut finger between
//! two kept ones reads as a knuckle (a flat cap showed a hard facet).
use crate::geom::{V, Weld, a, v};
use charloom_core::{Influence, Vec3};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};

/// Summed weight on a missing digit's bones from which a vertex is cut.
pub const CUT_WEIGHT: f64 = 0.5;

/// Dome height of a cap over its boundary radius.
const CAP_HEIGHT: f64 = 0.5;
/// Rings of a cap between the boundary and the apex.
const CAP_RINGS: usize = 2;

/// One exported skin mesh (positions, shading, UVs, compact influences).
pub struct SkinMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<[f32; 2]>,
    pub triangles: Vec<[u32; 3]>,
    pub influences: Vec<Vec<Influence>>,
}

/// MakeHuman's digits (1 = thumb … 5 = little finger) that carry the `n`
/// digits of a hand, from the thumb out: the fingers spread over the palm.
pub fn kept(n: u8) -> &'static [usize] {
    match n {
        0..=2 => &[1, 3],
        3 => &[1, 2, 5],
        4 => &[1, 2, 3, 5],
        _ => &[1, 2, 3, 4, 5],
    }
}

/// MakeHuman's digits cut from a hand with `n` digits.
pub fn missing(n: u8) -> Vec<usize> {
    (1..=5).filter(|f| !kept(n).contains(f)).collect()
}

/// Phalanx bones (`finger{F}-1..3.{L,R}`) of the missing digits of the
/// `[left, right]` hands, each with the bone that takes over its weight on
/// the stump (`metacarpal{F-1}`, else the bone `index` finds for the wrist).
/// `index` maps a bone name to its index in the caller's joint numbering.
pub fn missing_bones(
    index: impl Fn(&str) -> Option<usize>,
    digits: [u8; 2],
) -> Vec<(usize, usize)> {
    let mut out = vec![];
    for (n, suffix) in digits.into_iter().zip(["L", "R"]) {
        for f in missing(n) {
            let Some(to) = index(&format!("metacarpal{}.{suffix}", f - 1))
                .or_else(|| index(&format!("wrist.{suffix}")))
            else {
                continue;
            };
            for k in 1..=3 {
                if let Some(bone) = index(&format!("finger{f}-{k}.{suffix}")) {
                    out.push((bone, to));
                }
            }
        }
    }
    out
}

/// Per vertex: dominated by the missing digits' bones (summed weight at
/// least [`CUT_WEIGHT`]). All false without missing digits.
pub fn digit_mask(influences: &[Vec<Influence>], bones: &[(usize, usize)]) -> Vec<bool> {
    influences
        .iter()
        .map(|row| {
            !bones.is_empty()
                && row
                    .iter()
                    .filter(|w| bones.iter().any(|b| b.0 == w.joint))
                    .map(|w| w.weight)
                    .sum::<f64>()
                    >= CUT_WEIGHT
        })
        .collect()
}

/// Triangles touching none of the `masked` vertices.
pub fn unmasked_triangles(triangles: &[[u32; 3]], masked: &[bool]) -> Vec<[u32; 3]> {
    triangles
        .iter()
        .filter(|t| !t.iter().any(|&i| masked.get(i as usize) == Some(&true)))
        .copied()
        .collect()
}

/// Cuts the missing digits out of the exported skin and caps the holes (see
/// the module docs). `hidden` (per triangle) is carried along; cap triangles
/// take the state of the triangle they close. Returns the report.
pub fn cut_digits(mesh: &mut SkinMesh, hidden: &mut Vec<bool>, bones: &[(usize, usize)]) -> Value {
    let remove = digit_mask(&mesh.influences, bones);
    let mut report = cut_and_cap(mesh, hidden, &remove);
    // The stump follows the hand.
    report["stump_vertices_reweighted"] = json!(redirect_weights(&mut mesh.influences, bones));
    report
}

/// Moves weight on the cut digits' bones to the bone that takes it over
/// (the metacarpal), so what remains near a cut digit follows the hand.
/// Returns the number of rows changed.
pub fn redirect_weights(rows: &mut [Vec<Influence>], bones: &[(usize, usize)]) -> usize {
    let mut changed = 0usize;
    for row in rows {
        if !row.iter().any(|w| bones.iter().any(|b| b.0 == w.joint)) {
            continue;
        }
        changed += 1;
        let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
        for w in row.iter() {
            let joint = bones
                .iter()
                .find(|b| b.0 == w.joint)
                .map_or(w.joint, |b| b.1);
            *acc.entry(joint).or_default() += w.weight;
        }
        *row = normalized(acc);
    }
    changed
}

fn normalized(acc: BTreeMap<usize, f64>) -> Vec<Influence> {
    let mut out: Vec<Influence> = acc
        .into_iter()
        .filter(|(_, w)| *w > 1e-9)
        .map(|(joint, weight)| Influence { joint, weight })
        .collect();
    out.sort_by(|x, y| y.weight.total_cmp(&x.weight).then(x.joint.cmp(&y.joint)));
    out.truncate(4);
    let sum: f64 = out.iter().map(|w| w.weight).sum();
    for w in &mut out {
        w.weight /= sum.max(1e-12);
    }
    out
}

/// One boundary edge of a hole: welded ends and the unwelded vertices the
/// kept triangle uses there, in the kept triangle's winding.
#[derive(Clone, Copy)]
struct Edge {
    w: [usize; 2],
    u: [u32; 2],
    triangle: usize,
}

/// Removes every triangle touching a `remove` vertex (welded by position,
/// so UV seams do not count as cuts), closes each new boundary loop with a
/// dome and drops vertices no triangle uses any more.
pub fn cut_and_cap(mesh: &mut SkinMesh, hidden: &mut Vec<bool>, remove: &[bool]) -> Value {
    let weld = Weld::new(&mesh.positions, 1e-7);
    let mut removed_w = vec![false; weld.count];
    for (i, r) in remove.iter().enumerate() {
        if *r {
            removed_w[weld.id[i]] = true;
        }
    }
    let wid = |i: u32| weld.id[i as usize];
    let cut: Vec<bool> = mesh
        .triangles
        .iter()
        .map(|t| t.iter().any(|&i| removed_w[wid(i)]))
        .collect();
    let removed_triangles = cut.iter().filter(|c| **c).count();
    if removed_triangles == 0 {
        return json!({"applied": false});
    }
     
    let mut uses: HashMap<[usize; 2], (usize, usize)> = HashMap::new();
    for (t, tri) in mesh.triangles.iter().enumerate() {
        for k in 0..3 {
            let (x, y) = (wid(tri[k]), wid(tri[(k + 1) % 3]));
            let e = uses.entry([x.min(y), x.max(y)]).or_default();
            if cut[t] {
                e.1 += 1;
            } else {
                e.0 += 1;
            }
        }
    }
    // New boundary: kept once, cut on the other side.
    let mut outgoing: HashMap<usize, Vec<Edge>> = HashMap::new();
    for (t, tri) in mesh.triangles.iter().enumerate() {
        if cut[t] {
            continue;
        }
        for k in 0..3 {
            let (p, q) = (tri[k], tri[(k + 1) % 3]);
            let (x, y) = (wid(p), wid(q));
            if uses[&[x.min(y), x.max(y)]] == (1, 1) {
                outgoing.entry(x).or_default().push(Edge {
                    w: [x, y],
                    u: [p, q],
                    triangle: t,
                });
            }
        }
    }
    // Chain the edges into loops (the kept side's winding runs one way round
    // each hole).
    let mut loops: Vec<Vec<Edge>> = vec![];
    let mut open = 0usize;
    let mut starts: Vec<usize> = outgoing.keys().copied().collect();
    starts.sort_unstable();
    for s in starts {
        while let Some(first) = outgoing.get_mut(&s).and_then(Vec::pop) {
            let mut chain = vec![first];
            let closed = loop {
                let at = chain.last().unwrap().w[1];
                if at == s {
                    break true;
                }
                match outgoing.get_mut(&at).and_then(Vec::pop) {
                    Some(e) => chain.push(e),
                    None => break false,
                }
            };
            if closed && chain.len() >= 3 {
                loops.push(chain);
            } else {
                open += 1;
            }
        }
    }
     
    let mut ring_neighbours: HashMap<usize, Vec<usize>> = HashMap::new();
    for (t, tri) in mesh.triangles.iter().enumerate() {
        if !cut[t] {
            continue;
        }
        for &i in tri {
            if removed_w[wid(i)] {
                for &j in tri {
                    if !removed_w[wid(j)] {
                        ring_neighbours.entry(wid(j)).or_default().push(i as usize);
                    }
                }
            }
        }
    }
    let mut kept_triangles: Vec<[u32; 3]> = vec![];
    let mut kept_hidden: Vec<bool> = vec![];
    for (t, tri) in mesh.triangles.iter().enumerate() {
        if !cut[t] {
            kept_triangles.push(*tri);
            kept_hidden.push(hidden.get(t).copied().unwrap_or(false));
        }
    }
    let mut caps = vec![];
    for chain in &loops {
        let ring: Vec<V> = chain
            .iter()
            .map(|e| v(mesh.positions[e.u[0] as usize]))
            .collect();
        let c = ring.iter().fold(V::zeros(), |s, p| s + p) / ring.len() as f64;
        // Newell normal of the loop, turned towards the cut side.
        let mut axis = (0..ring.len()).fold(V::zeros(), |s, k| {
            s + (ring[k] - c).cross(&(ring[(k + 1) % ring.len()] - c))
        });
        let beyond: Vec<V> = chain
            .iter()
            .flat_map(|e| ring_neighbours.get(&e.w[0]).into_iter().flatten())
            .map(|&i| v(mesh.positions[i]))
            .collect();
        let side = beyond.iter().fold(V::zeros(), |s, p| s + p) / beyond.len().max(1) as f64 - c;
        if axis.dot(&side) < 0.0 {
            axis = -axis;
        }
        let Some(axis) = axis.try_normalize(1e-12) else {
            open += 1;
            continue;
        };
        let radius = ring.iter().map(|p| (p - c).norm()).sum::<f64>() / ring.len() as f64;
        let height = radius * CAP_HEIGHT;
        // The apex takes the ring's mean weights; ring k blends towards it.
        let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
        for e in chain {
            for w in &mesh.influences[e.u[0] as usize] {
                *acc.entry(w.joint).or_default() += w.weight;
            }
        }
        let apex_weights = normalized(acc);
        let blend = |row: &[Influence], t: f64| {
            let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
            for w in row {
                *acc.entry(w.joint).or_default() += w.weight * (1.0 - t);
            }
            for w in &apex_weights {
                *acc.entry(w.joint).or_default() += w.weight * t;
            }
            normalized(acc)
        };
        // Ring vertex k of unwelded boundary vertex u (seam copies stay
        // apart, each with its own UV): the boundary colour carried inward.
        let mut ring_vertex: HashMap<(usize, u32), u32> = HashMap::new();
        let first_new = mesh.positions.len();
        let add = |mesh: &mut SkinMesh, p: V, n: V, uv: [f32; 2], w: Vec<Influence>| {
            mesh.positions.push(a(p));
            mesh.normals.push(a(n));
            mesh.uvs.push(uv);
            mesh.influences.push(w);
            (mesh.positions.len() - 1) as u32
        };
        let level = |k: usize| {
            let phi = std::f64::consts::FRAC_PI_2 * k as f64 / (CAP_RINGS + 1) as f64;
            (phi.cos(), phi.sin())
        };
        let mut vertex_at = |mesh: &mut SkinMesh, k: usize, u: u32| -> u32 {
            if k == 0 {
                return u;
            }
            *ring_vertex.entry((k, u)).or_insert_with(|| {
                let (cs, sn) = level(k);
                let d = v(mesh.positions[u as usize]) - c;
                let radial = (d - axis * d.dot(&axis)).try_normalize(1e-12).unwrap_or(d);
                let p = c + d * cs + axis * (height * sn);
                let n =
                    (radial * (cs / radius.max(1e-9)) + axis * (sn / height.max(1e-9))).normalize();
                let uv = mesh.uvs[u as usize];
                let w = blend(&mesh.influences[u as usize], 1.0 - cs);
                add(mesh, p, n, uv, w)
            })
        };
        let mut triangles = vec![];
        for e in chain {
            let h = hidden.get(e.triangle).copied().unwrap_or(false);
            // Reversed edge: the cap continues the kept surface's winding.
            let (p, q) = (e.u[1], e.u[0]);
            for k in 0..CAP_RINGS {
                let (p0, q0) = (vertex_at(mesh, k, p), vertex_at(mesh, k, q));
                let (p1, q1) = (vertex_at(mesh, k + 1, p), vertex_at(mesh, k + 1, q));
                triangles.push(([p0, q0, q1], h));
                triangles.push(([p0, q1, p1], h));
            }
            // One apex per edge (UV of that edge), all at one point with one
            // set of weights, so the cap is closed when posed.
            let (p1, q1) = (vertex_at(mesh, CAP_RINGS, p), vertex_at(mesh, CAP_RINGS, q));
            let uv = [p, q].map(|i| mesh.uvs[i as usize]);
            let apex = add(
                mesh,
                c + axis * height,
                axis,
                [(uv[0][0] + uv[1][0]) * 0.5, (uv[0][1] + uv[1][1]) * 0.5],
                apex_weights.clone(),
            );
            triangles.push(([p1, q1, apex], h));
        }
        caps.push(json!({
            "boundary_edges": chain.len(),
            "radius_m": radius,
            "center": a(c),
            "axis": a(axis),
            "triangles": triangles.len(),
            "new_vertices": mesh.positions.len() - first_new,
        }));
        for (t, h) in triangles {
            kept_triangles.push(t);
            kept_hidden.push(h);
        }
    }
    // Compact: drop vertices only the cut triangles used.
    let mut used = vec![false; mesh.positions.len()];
    for t in &kept_triangles {
        for &i in t {
            used[i as usize] = true;
        }
    }
    let mut referenced = vec![false; mesh.positions.len()];
    for t in &mesh.triangles {
        for &i in t {
            referenced[i as usize] = true;
        }
    }
    let keep: Vec<bool> = (0..mesh.positions.len())
        .map(|i| used[i] || (i < referenced.len() && !referenced[i]))
        .collect();
    let mut map = vec![u32::MAX; keep.len()];
    let mut next = 0u32;
    for (i, k) in keep.iter().enumerate() {
        if *k {
            map[i] = next;
            next += 1;
        }
    }
    let pick = |i: &usize| keep[*i];
    mesh.positions = (0..keep.len())
        .filter(pick)
        .map(|i| mesh.positions[i])
        .collect();
    mesh.normals = (0..keep.len())
        .filter(pick)
        .map(|i| mesh.normals[i])
        .collect();
    mesh.uvs = (0..keep.len()).filter(pick).map(|i| mesh.uvs[i]).collect();
    mesh.influences = (0..keep.len())
        .filter(pick)
        .map(|i| std::mem::take(&mut mesh.influences[i]))
        .collect();
    let dropped = keep.iter().filter(|k| !**k).count();
    mesh.triangles = kept_triangles
        .iter()
        .map(|t| t.map(|i| map[i as usize]))
        .collect();
    *hidden = kept_hidden;
    json!({
        "applied": true,
        "removed_triangles": removed_triangles,
        "removed_vertices": dropped,
        "capped_holes": caps.len(),
        "uncapped_boundaries": open,
        "caps": caps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// A closed tube along +Y (a finger on a palm): rings every 1 cm, fan
    /// caps at both ends, a UV seam along angle 0 (duplicated vertices with
    /// u = 0 and u = 1). Joint 0 drives y < 3 cm, joint 1 above, blended in
    /// between.
    fn finger() -> SkinMesh {
        let (segments, rings) = (12usize, 9usize);
        let mut m = SkinMesh {
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            triangles: vec![],
            influences: vec![],
        };
        let weights = |y: f64| {
            let t = ((y - 0.02) / 0.03).clamp(0.0, 1.0);
            [(0usize, 1.0 - t), (1, t)]
                .into_iter()
                .filter(|(_, w)| *w > 0.0)
                .map(|(joint, weight)| Influence { joint, weight })
                .collect::<Vec<_>>()
        };
        let push = |m: &mut SkinMesh, p: [f64; 3], uv: [f32; 2]| {
            m.positions.push(p);
            m.normals.push([p[0] / 0.008, 0.0, p[2] / 0.008]);
            m.uvs.push(uv);
            m.influences.push(weights(p[1]));
            (m.positions.len() - 1) as u32
        };
        // Ring r, column s (s = segments is the seam copy of s = 0).
        let mut grid = vec![vec![0u32; segments + 1]; rings];
        for (r, row) in grid.iter_mut().enumerate() {
            let y = r as f64 * 0.01;
            for (s, slot) in row.iter_mut().enumerate() {
                let phi = std::f64::consts::TAU * s as f64 / segments as f64;
                *slot = push(
                    &mut m,
                    [0.008 * phi.cos(), y, 0.008 * phi.sin()],
                    [s as f32 / segments as f32, r as f32 / rings as f32],
                );
            }
        }
        for r in 0..rings - 1 {
            for s in 0..segments {
                let (a0, b0, a1, b1) = (
                    grid[r][s],
                    grid[r][s + 1],
                    grid[r + 1][s],
                    grid[r + 1][s + 1],
                );
                // Outward winding (angle runs from +X towards +Z).
                m.triangles.push([a0, a1, b0]);
                m.triangles.push([b0, a1, b1]);
            }
        }
        let bottom = push(&mut m, [0.0, 0.0, 0.0], [0.5, 0.0]);
        let top = push(&mut m, [0.0, 0.08, 0.0], [0.5, 1.0]);
        for s in 0..segments {
            m.triangles.push([bottom, grid[0][s], grid[0][s + 1]]);
            m.triangles
                .push([top, grid[rings - 1][s + 1], grid[rings - 1][s]]);
        }
        m
    }

    /// Directed welded edges of a mesh: closed and consistently wound when
    /// every one appears exactly once and its reverse too.
    fn closed_and_oriented(m: &SkinMesh) -> bool {
        let w = Weld::new(&m.positions, 1e-7);
        let mut seen: HashMap<(usize, usize), usize> = HashMap::new();
        for t in &m.triangles {
            for k in 0..3 {
                let e = (w.id[t[k] as usize], w.id[t[(k + 1) % 3] as usize]);
                *seen.entry(e).or_default() += 1;
            }
        }
        seen.iter()
            .all(|(&(x, y), &n)| n == 1 && seen.get(&(y, x)) == Some(&1))
    }

    fn signed_volume(m: &SkinMesh) -> f64 {
        m.triangles
            .iter()
            .map(|t| {
                let [p, q, r] = t.map(|i| v(m.positions[i as usize]));
                p.dot(&q.cross(&r)) / 6.0
            })
            .sum()
    }

    #[test]
    fn test_finger_is_closed() {
        let m = finger();
        assert!(closed_and_oriented(&m));
        assert!(signed_volume(&m) > 0.0);
    }

    #[test]
    fn cut_removes_the_digit_and_closes_the_hole_with_an_outward_dome() {
        let mut m = finger();
        let triangles = m.triangles.len();
        let mut hidden = vec![false; triangles];
        // Hide one triangle at the cut ring and one far below it.
        hidden[2 * 12 * 2] = true;
        hidden[0] = true;
        let report = cut_digits(&mut m, &mut hidden, &[(1, 0)]);
        assert_eq!(report["applied"], true);
        assert_eq!(report["capped_holes"], 1);
        assert_eq!(report["uncapped_boundaries"], 0);
        assert_eq!(hidden.len(), m.triangles.len());
        assert!(hidden[0], "hidden state of kept triangles is carried");
        // Nothing of the digit is left; the stump is watertight and wound
        // outwards, and the cap bulges out past the cut (no spike: within
        // the dome height).
        assert!(closed_and_oriented(&m));
        assert!(signed_volume(&m) > 0.0);
        let top = m.positions.iter().map(|p| p[1]).fold(f64::MIN, f64::max);
        let cut_y = 0.03; // last ring with joint-1 weight below 0.5
        assert!(
            top > cut_y && top <= cut_y + 0.008 * CAP_HEIGHT + 1e-9,
            "top {top}"
        );
        for p in &m.positions {
            assert!((p[0].powi(2) + p[2].powi(2)).sqrt() <= 0.008 + 1e-9);
        }
        // Weight on the cut bone moved to its metacarpal; rows normalized.
        for row in &m.influences {
            assert!(row.iter().all(|w| w.joint == 0));
            assert!((row.iter().map(|w| w.weight).sum::<f64>() - 1.0).abs() < 1e-9);
        }
        // Cap normals point out of the stump; seam copies keep their UVs.
        let w = Weld::new(&m.positions, 1e-7);
        for (p, n) in m.positions.iter().zip(&m.normals) {
            if p[1] > cut_y + 1e-6 {
                assert!(n[1] > 0.0);
            }
        }
        let uvs_at_seam: HashSet<u32> = m
            .positions
            .iter()
            .zip(&m.uvs)
            .filter(|(p, _)| p[1] > cut_y + 1e-6 && p[2].abs() < 1e-12 && p[0] > 1e-6)
            .map(|(_, uv)| uv[0].to_bits())
            .collect();
        assert!(uvs_at_seam.len() >= 2, "seam copies keep u = 0 and u = 1");
        assert!(w.count < m.positions.len());
        // Every vertex is used.
        let mut used = vec![false; m.positions.len()];
        m.triangles
            .iter()
            .flatten()
            .for_each(|&i| used[i as usize] = true);
        assert!(used.iter().all(|u| *u));
    }

    #[test]
    fn nothing_to_cut_leaves_the_mesh_alone() {
        let mut m = finger();
        let before = (m.positions.clone(), m.triangles.clone());
        let mut hidden = vec![false; m.triangles.len()];
        let report = cut_digits(&mut m, &mut hidden, &[]);
        assert_eq!(report["applied"], false);
        assert_eq!((m.positions, m.triangles), before);
    }

    #[test]
    fn missing_digits_leave_fingers_spread_over_the_palm() {
        let names: Vec<String> = ["L", "R"]
            .iter()
            .flat_map(|s| {
                (1..=5).flat_map(move |f| {
                    let mut v = vec![format!("metacarpal{}.{s}", f - 1)];
                    v.extend((1..=3).map(|k| format!("finger{f}-{k}.{s}")));
                    v
                })
            })
            .collect();
        let index = |n: &str| names.iter().position(|x| x == n);
        assert!(missing_bones(index, [5, 5]).is_empty());
        let bones = missing_bones(index, [4, 3]);
        let named: Vec<(&str, &str)> = bones
            .iter()
            .map(|&(b, to)| (names[b].as_str(), names[to].as_str()))
            .collect();
        // Four digits: the ring finger goes; three: middle and ring.
        assert_eq!(named.len(), 3 + 6);
        assert!(named.contains(&("finger4-2.L", "metacarpal3.L")));
        assert!(named.contains(&("finger3-1.R", "metacarpal2.R")));
        assert!(named.contains(&("finger4-1.R", "metacarpal3.R")));
        assert!(!named.iter().any(|(b, _)| b.starts_with("finger5")));
        assert!(!named.iter().any(|(b, _)| b.starts_with("finger3") && b.ends_with(".L")));
    }
}
