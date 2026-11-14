//! One level of Loop subdivision for the exported body: every triangle into
//! four, the surface smoothed on its welded topology (UV seams split
//! vertices, not the surface), so a body grown far past MakeHuman's sizes
//! (a cartoon's belly and limbs) shows no polygons. UVs and skin weights of
//! a new edge vertex are its edge's average (corners keep theirs); normals
//! are recomputed. Fitting runs on MakeHuman's own mesh; only the output
//! is subdivided.
use crate::geom::{V, Weld, a, v};
use charloom_core::{Influence, Vec3};
use std::collections::HashMap;

pub struct Subdivided {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<[f32; 2]>,
    pub influences: Vec<Vec<Influence>>,
    pub triangles: Vec<[u32; 3]>,
    /// The original triangle of each new one.
    pub parent: Vec<usize>,
}

/// Mean of two weight sets, the four largest kept and renormalised.
fn mix(x: &[Influence], y: &[Influence]) -> Vec<Influence> {
    let mut acc: HashMap<usize, f64> = HashMap::new();
    for w in x.iter().chain(y) {
        *acc.entry(w.joint).or_default() += 0.5 * w.weight;
    }
    let mut out: Vec<Influence> = acc.into_iter().map(|(joint, weight)| Influence { joint, weight }).collect();
    out.sort_by(|p, q| q.weight.total_cmp(&p.weight).then(p.joint.cmp(&q.joint)));
    out.truncate(4);
    let sum: f64 = out.iter().map(|w| w.weight).sum();
    if sum > 0.0 {
        for w in &mut out {
            w.weight /= sum;
        }
    }
    out
}

pub fn loop_subdivide(
    positions: &[Vec3],
    uvs: &[[f32; 2]],
    influences: &[Vec<Influence>],
    triangles: &[[u32; 3]],
) -> Subdivided {
    let weld = Weld::new(positions, 1e-7);
    let wp: Vec<V> = weld.representative.iter().map(|&r| v(positions[r])).collect();
    // Welded edges: their opposite vertices (one on a boundary).
    let key = |p: usize, q: usize| (p.min(q), p.max(q));
    let mut opposite: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut ring: Vec<Vec<usize>> = vec![vec![]; weld.count];
    for t in triangles {
        let w = t.map(|i| weld.id[i as usize]);
        for k in 0..3 {
            let (p, q, o) = (w[k], w[(k + 1) % 3], w[(k + 2) % 3]);
            opposite.entry(key(p, q)).or_default().push(o);
            if !ring[p].contains(&q) {
                ring[p].push(q);
            }
            if !ring[q].contains(&p) {
                ring[q].push(p);
            }
        }
    }
    let boundary = |p: usize, q: usize| opposite.get(&key(p, q)).is_none_or(|o| o.len() != 2);
    // Edge points (welded).
    let edge_point = |p: usize, q: usize| -> V {
        match opposite.get(&key(p, q)) {
            Some(o) if o.len() == 2 => (wp[p] + wp[q]) * 0.375 + (wp[o[0]] + wp[o[1]]) * 0.125,
            _ => (wp[p] + wp[q]) * 0.5,
        }
    };
    // Moved original vertices (welded): Loop's weights inside, the boundary
    // curve's on an open edge.
    let moved: Vec<V> = (0..weld.count)
        .map(|p| {
            let rim: Vec<usize> = ring[p].iter().copied().filter(|&q| boundary(p, q)).collect();
            if rim.len() >= 2 {
                return wp[p] * 0.75 + (wp[rim[0]] + wp[rim[1]]) * 0.125;
            }
            let k = ring[p].len();
            if k < 3 {
                return wp[p];
            }
            let c = 0.375 + 0.25 * (2.0 * std::f64::consts::PI / k as f64).cos();
            let beta = (0.625 - c * c) / k as f64;
            let sum = ring[p].iter().fold(V::zeros(), |s, &q| s + wp[q]);
            wp[p] * (1.0 - k as f64 * beta) + sum * beta
        })
        .collect();
    let mut out_pos: Vec<Vec3> = (0..positions.len()).map(|i| a(moved[weld.id[i]])).collect();
    let mut out_uv: Vec<[f32; 2]> = uvs.to_vec();
    let mut out_inf: Vec<Vec<Influence>> = influences.to_vec();
    // One new vertex per unwelded edge (the two sides of a UV seam get
    // their own, at the same position).
    let mut mid: HashMap<(u32, u32), u32> = HashMap::new();
    let mut midpoint = |i: u32, j: u32, out_pos: &mut Vec<Vec3>, out_uv: &mut Vec<[f32; 2]>, out_inf: &mut Vec<Vec<Influence>>| -> u32 {
        let k = (i.min(j), i.max(j));
        if let Some(&m) = mid.get(&k) {
            return m;
        }
        let (iu, ju) = (i as usize, j as usize);
        let m = out_pos.len() as u32;
        out_pos.push(a(edge_point(weld.id[iu], weld.id[ju])));
        out_uv.push(if uvs.is_empty() {
            [0.0, 0.0]
        } else {
            [(uvs[iu][0] + uvs[ju][0]) * 0.5, (uvs[iu][1] + uvs[ju][1]) * 0.5]
        });
        out_inf.push(if influences.is_empty() { vec![] } else { mix(&influences[iu], &influences[ju]) });
        mid.insert(k, m);
        m
    };
    let mut tris = Vec::with_capacity(triangles.len() * 4);
    let mut parent = Vec::with_capacity(triangles.len() * 4);
    for (f, t) in triangles.iter().enumerate() {
        let [i0, i1, i2] = *t;
        let m01 = midpoint(i0, i1, &mut out_pos, &mut out_uv, &mut out_inf);
        let m12 = midpoint(i1, i2, &mut out_pos, &mut out_uv, &mut out_inf);
        let m20 = midpoint(i2, i0, &mut out_pos, &mut out_uv, &mut out_inf);
        for tri in [[i0, m01, m20], [i1, m12, m01], [i2, m20, m12], [m01, m12, m20]] {
            tris.push(tri);
            parent.push(f);
        }
    }
    if uvs.is_empty() {
        out_uv.clear();
    }
    // Smooth normals on the new welded surface.
    let weld2 = Weld::new(&out_pos, 1e-7);
    let mut acc = vec![V::zeros(); weld2.count];
    for t in &tris {
        let p = t.map(|i| v(out_pos[i as usize]));
        let n = (p[1] - p[0]).cross(&(p[2] - p[0]));
        for &i in t {
            acc[weld2.id[i as usize]] += n;
        }
    }
    let normals: Vec<Vec3> = (0..out_pos.len())
        .map(|i| {
            let n = acc[weld2.id[i]];
            a(if n.norm() > 1e-12 { n.normalize() } else { V::y() })
        })
        .collect();
    Subdivided { positions: out_pos, normals, uvs: out_uv, influences: out_inf, triangles: tris, parent }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subdivides_a_closed_tetrahedron_into_four_times_the_triangles() {
        let p = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let t = vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]];
        let inf: Vec<Vec<Influence>> = (0..4).map(|j| vec![Influence { joint: j, weight: 1.0 }]).collect();
        let s = loop_subdivide(&p, &[[0.0, 0.0]; 4], &inf, &t);
        assert_eq!(s.triangles.len(), 16);
        assert_eq!(s.positions.len(), 4 + 6);
        // An edge vertex carries both ends' joints, half each.
        let e = &s.influences[4];
        assert_eq!(e.len(), 2);
        assert!((e[0].weight - 0.5).abs() < 1e-9);
        // Smoothing pulls the corners in.
        assert!(v(s.positions[1]).norm() < 1.0);
    }
}
