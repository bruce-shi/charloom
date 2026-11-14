//! Read-only closest-point queries over an explicitly supplied observed surface.
use nalgebra::Vector3;
pub type V = Vector3<f64>;

#[derive(Clone)]
struct Triangle {
    id: usize,
    p: [V; 3],
    min: V,
    max: V,
    normal: V,
}
struct Node {
    min: V,
    max: V,
    range: std::ops::Range<usize>,
    children: Option<[usize; 2]>,
}
pub struct Surface {
    triangles: Vec<Triangle>,
    nodes: Vec<Node>,
}
pub struct Hit {
    pub point: V,
    pub distance: f64,
    pub normal: V,
    pub triangle: usize,
    pub barycentric: [f64; 3],
}

impl Surface {
    pub fn new(points: &[V], faces: &[[usize; 3]]) -> Self {
        let triangles = faces
            .iter()
            .enumerate()
            .filter_map(|(id, f)| {
                let p = f.map(|i| points[i]);
                let normal = (p[1] - p[0]).cross(&(p[2] - p[0])).try_normalize(1e-12)?;
                Some(Triangle {
                    id,
                    min: p[0].inf(&p[1]).inf(&p[2]),
                    max: p[0].sup(&p[1]).sup(&p[2]),
                    p,
                    normal,
                })
            })
            .collect();
        let mut s = Self {
            triangles,
            nodes: vec![],
        };
        if !s.triangles.is_empty() {
            s.build(0, s.triangles.len());
        }
        s
    }
    fn build(&mut self, start: usize, end: usize) -> usize {
        let min = self.triangles[start..end]
            .iter()
            .fold(V::repeat(f64::INFINITY), |v, t| v.inf(&t.min));
        let max = self.triangles[start..end]
            .iter()
            .fold(V::repeat(f64::NEG_INFINITY), |v, t| v.sup(&t.max));
        let id = self.nodes.len();
        self.nodes.push(Node {
            min,
            max,
            range: start..end,
            children: None,
        });
        if end - start > 12 {
            let ext = max - min;
            let axis = (0..3).max_by(|&a, &b| ext[a].total_cmp(&ext[b])).unwrap();
            let mid = (start + end) / 2;
            self.triangles[start..end].select_nth_unstable_by(mid - start, |a, b| {
                (a.min[axis] + a.max[axis]).total_cmp(&(b.min[axis] + b.max[axis]))
            });
            self.nodes[id].children = Some([self.build(start, mid), self.build(mid, end)]);
        }
        id
    }
    pub fn nearest(&self, p: V, limit: f64, normal: Option<V>) -> Option<(V, f64)> {
        self.nearest_triangle(p, limit, normal)
            .map(|(_, point, squared)| (point, squared.sqrt()))
    }
    pub fn nearest_detail(&self, p: V, limit: f64, normal: Option<V>) -> Option<Hit> {
        let (t, point, squared) = self.nearest_triangle(p, limit, normal)?;
        // Barycentric coordinates are needed only for the final hit, never for
        // intermediate candidates or distance-only ICP queries.
        let u = t.p[1] - t.p[0];
        let v = t.p[2] - t.p[0];
        let w = point - t.p[0];
        let denominator = u.dot(&u) * v.dot(&v) - u.dot(&v).powi(2);
        let b = (v.dot(&v) * w.dot(&u) - u.dot(&v) * w.dot(&v)) / denominator;
        let c = (u.dot(&u) * w.dot(&v) - u.dot(&v) * w.dot(&u)) / denominator;
        Some(Hit {
            point,
            distance: squared.sqrt(),
            normal: t.normal,
            triangle: t.id,
            barycentric: [1. - b - c, b, c],
        })
    }
    fn nearest_triangle(&self, p: V, limit: f64, normal: Option<V>) -> Option<(&Triangle, V, f64)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best = limit * limit;
        let mut result = None;
        let mut stack = vec![0];
        while let Some(i) = stack.pop() {
            let n = &self.nodes[i];
            let bound = (0..3)
                .map(|k| (p[k] - p[k].clamp(n.min[k], n.max[k])).powi(2))
                .sum::<f64>();
            if bound > best {
                continue;
            }
            if let Some(ch) = n.children {
                stack.extend(ch);
                continue;
            }
            for t in &self.triangles[n.range.clone()] {
                if normal.is_some_and(|v| v.dot(&t.normal) < 0.15) {
                    continue;
                }
                let q = closest(p, t.p);
                let d = (p - q).norm_squared();
                if d < best {
                    best = d;
                    result = Some((t, q, d));
                }
            }
        }
        result
    }
    /// Strict noncoplanar triangle crossings, excluding shared-point contact.
    /// Same geometric definition as the existing surface-contacts auditor.
    pub fn crossings(&self) -> std::collections::BTreeSet<[usize; 2]> {
        self.crossings_selected(None)
    }
    /// Exact crossings with at least one changed face. Each selected triangle
    /// still queries the entire surface, including stationary disconnected parts.
    pub fn crossings_involving(&self, changed: &[bool]) -> std::collections::BTreeSet<[usize; 2]> {
        self.crossings_selected(Some(changed))
    }
    fn crossings_selected(
        &self,
        changed: Option<&[bool]>,
    ) -> std::collections::BTreeSet<[usize; 2]> {
        let mut out = std::collections::BTreeSet::new();
        if self.nodes.is_empty() {
            return out;
        }
        let mut pending = Vec::with_capacity(32);
        for a in &self.triangles {
            if changed.is_some_and(|mask| !mask[a.id]) {
                continue;
            }
            pending.clear();
            pending.push(0);
            while let Some(i) = pending.pop() {
                let n = &self.nodes[i];
                if (0..3).any(|k| a.max[k] < n.min[k] - 1e-6 || n.max[k] < a.min[k] - 1e-6) {
                    continue;
                }
                if let Some(ch) = n.children {
                    pending.extend(ch);
                    continue;
                }
                for b in &self.triangles[n.range.clone()] {
                    if a.id == b.id
                        || (a.id > b.id && changed.is_none_or(|mask| mask[b.id]))
                        || (0..3).any(|k| a.max[k] < b.min[k] - 1e-6 || b.max[k] < a.min[k] - 1e-6)
                    {
                        continue;
                    }
                    if crosses(a, b) {
                        out.insert([a.id.min(b.id), a.id.max(b.id)]);
                    }
                }
            }
        }
        out
    }
    pub fn front(&self, x: f64, z: f64) -> Option<V> {
        if self.nodes.is_empty() || !x.is_finite() || !z.is_finite() {
            return None;
        }
        let origin = V::new(x, self.nodes[0].min.y - 1., z);
        let dir = V::y();
        let mut best = f64::INFINITY;
        let mut result = None;
        let mut stack = vec![0];
        while let Some(i) = stack.pop() {
            let node = &self.nodes[i];
            if x < node.min.x
                || x > node.max.x
                || z < node.min.z
                || z > node.max.z
                || node.min.y - origin.y > best
            {
                continue;
            }
            if let Some(children) = node.children {
                stack.extend(children);
                continue;
            }
            for t in &self.triangles[node.range.clone()] {
                if x < t.min.x || x > t.max.x || z < t.min.z || z > t.max.z {
                    continue;
                }
                let e1 = t.p[1] - t.p[0];
                let e2 = t.p[2] - t.p[0];
                let h = dir.cross(&e2);
                let a = e1.dot(&h);
                if a.abs() < 1e-12 {
                    continue;
                }
                let s = origin - t.p[0];
                let u = s.dot(&h) / a;
                if !(0. ..=1.).contains(&u) {
                    continue;
                }
                let q = s.cross(&e1);
                let v = dir.dot(&q) / a;
                if v < 0. || u + v > 1. {
                    continue;
                }
                let d = e2.dot(&q) / a;
                if d > 0. && d < best {
                    best = d;
                    result = Some(origin + dir * d);
                }
            }
        }
        result
    }

    /// First positive hit along an arbitrary normalized ray, with the original
    /// triangle index. Used for scalp containment, not unsigned nearest points.
    pub fn ray(&self, origin: V, direction: V, limit: f64) -> Option<(V, V, usize)> {
        let dir = direction.try_normalize(1e-12)?;
        if self.nodes.is_empty() {
            return None;
        }
        let intersects = |low: V, high: V, best: f64| {
            let mut near = 0f64;
            let mut far = best;
            for k in 0..3 {
                if dir[k].abs() < 1e-12 {
                    if origin[k] < low[k] || origin[k] > high[k] {
                        return false;
                    }
                } else {
                    let a = (low[k] - origin[k]) / dir[k];
                    let b = (high[k] - origin[k]) / dir[k];
                    near = near.max(a.min(b));
                    far = far.min(a.max(b));
                    if near > far {
                        return false;
                    }
                }
            }
            true
        };
        let mut best = limit;
        let mut hit = None;
        let mut stack = vec![0];
        while let Some(i) = stack.pop() {
            let node = &self.nodes[i];
            if !intersects(node.min, node.max, best) {
                continue;
            }
            if let Some(children) = node.children {
                stack.extend(children);
                continue;
            }
            for t in &self.triangles[node.range.clone()] {
                let e1 = t.p[1] - t.p[0];
                let e2 = t.p[2] - t.p[0];
                let h = dir.cross(&e2);
                let a = e1.dot(&h);
                if a.abs() < 1e-12 {
                    continue;
                }
                let s = origin - t.p[0];
                let u = s.dot(&h) / a;
                if !(-1e-9..=1. + 1e-9).contains(&u) {
                    continue;
                }
                let q = s.cross(&e1);
                let v = dir.dot(&q) / a;
                if v < -1e-9 || u + v > 1. + 1e-9 {
                    continue;
                }
                let distance = e2.dot(&q) / a;
                if distance > 1e-8 && distance < best {
                    best = distance;
                    hit = Some((origin + dir * distance, t.normal, t.id));
                }
            }
        }
        hit
    }
}

fn crosses(a: &Triangle, b: &Triangle) -> bool {
    let e = 1e-6;
    if a.p
        .iter()
        .any(|p| b.p.iter().any(|q| (p - q).norm_squared() < e * e))
    {
        return false;
    }
    let pierces = |points: &[V; 3], face: &Triangle| {
        let distance = points.map(|p| (p - face.p[0]).dot(&face.normal));
        if distance.iter().all(|&d| d >= -e) || distance.iter().all(|&d| d <= e) {
            return false;
        }
        let e1 = face.p[1] - face.p[0];
        let e2 = face.p[2] - face.p[0];
        let d00 = e1.dot(&e1);
        let d01 = e1.dot(&e2);
        let d11 = e2.dot(&e2);
        let den = d00 * d11 - d01 * d01;
        if den <= 1e-24 {
            return false;
        }
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            if distance[i] * distance[j] >= -e * e {
                continue;
            }
            let t = distance[i] / (distance[i] - distance[j]);
            let p = points[i] + (points[j] - points[i]) * t - face.p[0];
            let d20 = p.dot(&e1);
            let d21 = p.dot(&e2);
            let u = (d11 * d20 - d01 * d21) / den;
            let v = (d00 * d21 - d01 * d20) / den;
            if u >= -1e-9 && v >= -1e-9 && u + v <= 1. + 1e-9 {
                return true;
            }
        }
        false
    };
    pierces(&a.p, b) || pierces(&b.p, a)
}

pub(crate) fn closest(p: V, [a, b, c]: [V; 3]) -> V {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0. && d2 <= 0. {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0. && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0. && d1 >= 0. && d3 <= 0. {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0. && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0. && d2 >= 0. && d6 <= 0. {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0. && d4 - d3 >= 0. && d5 - d6 >= 0. {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let inv = 1. / (va + vb + vc);
    a + ab * (vb * inv) + ac * (vc * inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_crossings_match_full_scan_including_lower_id_stationary_faces() {
        let mut points = Vec::new();
        let mut faces = Vec::new();
        for i in 0..30 {
            let x = i as f64 * 0.03;
            let n = points.len();
            points.extend([
                V::new(x - 0.02, -0.02, 0.),
                V::new(x + 0.02, -0.02, 0.),
                V::new(x, 0.02, 0.),
                V::new(x, -0.01, -0.01),
                V::new(x, -0.01, 0.01),
                V::new(x, 0.02, 0.003),
            ]);
            faces.extend([[n, n + 1, n + 2], [n + 3, n + 4, n + 5]]);
        }
        // A degenerate face must not change the mapping to original face IDs.
        faces.insert(0, [0, 0, 0]);
        let surface = Surface::new(&points, &faces);
        let full = surface.crossings();
        assert!(!full.is_empty());
        for selected in 0..faces.len() {
            let mut changed = vec![false; faces.len()];
            changed[selected] = true;
            let expected = full
                .iter()
                .copied()
                .filter(|p| p.contains(&selected))
                .collect();
            assert_eq!(surface.crossings_involving(&changed), expected);
        }
        assert_eq!(surface.crossings_involving(&vec![true; faces.len()]), full);
        assert!(
            surface
                .crossings_involving(&vec![false; faces.len()])
                .is_empty()
        );
    }
    #[test]
    fn front_query_selects_nearest_layer_across_bvh_branches() {
        let mut points = vec![];
        let mut faces = vec![];
        for i in 0..40 {
            let y = i as f64 * 0.01 - 3.;
            let n = points.len();
            points.extend([V::new(-1., y, 0.), V::new(1., y, 0.), V::new(0., y, 1.)]);
            faces.push([n, n + 1, n + 2]);
        }
        let s = Surface::new(&points, &faces);
        assert!((s.front(0., 0.3).unwrap().y + 3.).abs() < 1e-12);
        assert!(s.front(2., 0.3).is_none());
    }
    #[test]
    fn ignores_unobserved_space_and_hits_triangle_interior() {
        let s = Surface::new(
            &[V::new(-1., 0., 0.), V::new(1., 0., 0.), V::new(0., 0., 1.)],
            &[[0, 1, 2]],
        );
        assert!(
            (s.nearest(V::new(0., -0.2, 0.4), 1., None).unwrap().0 - V::new(0., 0., 0.4)).norm()
                < 1e-12
        );
        assert!(s.front(2., 0.4).is_none());
        assert!((s.front(0., 0.4).unwrap().y).abs() < 1e-12);
    }
}
