//! Static triangle queries for internal landmark fitting. The BVH is an analysis
//! accelerator; it never replaces or edits the render surface.
use charloom_core::{Mesh, Vec3, v3};
use nalgebra::Vector3;
use std::collections::BTreeSet;

struct Triangle {
    a: Vector3<f64>,
    e1: Vector3<f64>,
    e2: Vector3<f64>,
    min: Vector3<f64>,
    max: Vector3<f64>,
    center: Vector3<f64>,
    source: [usize; 2],
}
struct Node {
    min: Vector3<f64>,
    max: Vector3<f64>,
    children: Option<[usize; 2]>,
    start: usize,
    end: usize,
}
pub struct SurfaceQuery {
    triangles: Vec<Triangle>,
    nodes: Vec<Node>,
}

#[derive(Debug, Clone)]
pub struct SurfaceHit {
    pub point: Vec3,
    pub normal: Vec3,
    pub barycentric: [f64; 3],
    pub source: [usize; 2],
    pub distance_m: f64,
}

impl SurfaceQuery {
    pub fn new(meshes: &[Mesh], positions: &[Vec<Vec3>]) -> Self {
        let mut triangles = Vec::new();
        for (m, (mesh, points)) in meshes.iter().zip(positions).enumerate() {
            for (t, tri) in mesh.triangles.iter().enumerate() {
                let [a, b, c] = tri.map(|i| v3(points[i as usize]));
                if (b - a).cross(&(c - a)).norm_squared() < 1e-24 {
                    continue;
                }
                triangles.push(Triangle {
                    a,
                    e1: b - a,
                    e2: c - a,
                    min: a.inf(&b).inf(&c),
                    max: a.sup(&b).sup(&c),
                    center: (a + b + c) / 3.0,
                    source: [m, t],
                });
            }
        }
        let mut query = Self {
            triangles,
            nodes: vec![],
        };
        if !query.triangles.is_empty() {
            query.build(0, query.triangles.len());
        }
        query
    }
    fn build(&mut self, start: usize, end: usize) -> usize {
        let min = self.triangles[start..end]
            .iter()
            .map(|t| t.min)
            .fold(Vector3::repeat(f64::INFINITY), |a, b| a.inf(&b));
        let max = self.triangles[start..end]
            .iter()
            .map(|t| t.max)
            .fold(Vector3::repeat(f64::NEG_INFINITY), |a, b| a.sup(&b));
        let at = self.nodes.len();
        self.nodes.push(Node {
            min,
            max,
            children: None,
            start,
            end,
        });
        if end - start > 12 {
            let extent = max - min;
            let axis = (0..3)
                .max_by(|&a, &b| extent[a].total_cmp(&extent[b]))
                .unwrap();
            let middle = start + (end - start) / 2;
            self.triangles[start..end].select_nth_unstable_by(middle - start, |a, b| {
                a.center[axis].total_cmp(&b.center[axis])
            });
            let left = self.build(start, middle);
            let right = self.build(middle, end);
            self.nodes[at].children = Some([left, right]);
        }
        at
    }

    /// Euclidean closest point, optionally gated by an oriented surface normal.
    /// A filtered query must be built from the declared observed-skin faces;
    /// proximity alone never turns a garment into a skin observation.
    pub fn closest_point(
        &self,
        point: Vec3,
        maximum_distance_m: f64,
        normal: Option<(Vec3, f64)>,
    ) -> Option<SurfaceHit> {
        if self.nodes.is_empty() || !maximum_distance_m.is_finite() || maximum_distance_m <= 0.0 {
            return None;
        }
        let p = charloom_core::v3(point);
        let bound = |node: &Node| {
            (0..3)
                .map(|k| (p[k] - p[k].clamp(node.min[k], node.max[k])).powi(2))
                .sum::<f64>()
        };
        let mut best = maximum_distance_m.powi(2);
        let mut hit = None;
        let mut pending = vec![0];
        while let Some(i) = pending.pop() {
            let node = &self.nodes[i];
            if bound(node) > best {
                continue;
            }
            if let Some([a, b]) = node.children {
                if bound(&self.nodes[a]) < bound(&self.nodes[b]) {
                    pending.extend([b, a]);
                } else {
                    pending.extend([a, b]);
                }
                continue;
            }
            for triangle in &self.triangles[node.start..node.end] {
                let n = triangle.e1.cross(&triangle.e2).normalize();
                if let Some((wanted, cosine)) = normal
                    && n.dot(&charloom_core::v3(wanted)) < cosine
                {
                    continue;
                }
                let bary = closest_barycentric(
                    p,
                    triangle.a,
                    triangle.a + triangle.e1,
                    triangle.a + triangle.e2,
                );
                let q = triangle.a + triangle.e1 * bary[1] + triangle.e2 * bary[2];
                let distance = (p - q).norm_squared();
                if distance <= best {
                    best = distance;
                    hit = Some(SurfaceHit {
                        point: charloom_core::a3(q),
                        normal: charloom_core::a3(n),
                        barycentric: bary,
                        source: triangle.source,
                        distance_m: distance.sqrt(),
                    });
                }
            }
        }
        hit
    }

    /// First outward-facing boundary along a positive ray. The origin may be in
    /// an empty garment cavity: cloth thickness is not a filled body volume.
    /// Opposing coincident sheets are ambiguous and are skipped.
    pub fn outward_ray_distance(
        &self,
        origin: Vec3,
        direction: Vec3,
        maximum_distance_m: f64,
    ) -> Option<f64> {
        self.ray_distance(origin, direction, maximum_distance_m, true)
    }
    /// First surface hit, including inward-facing open hair or cloth sheets.
    pub fn first_ray_distance(
        &self,
        origin: Vec3,
        direction: Vec3,
        maximum_distance_m: f64,
    ) -> Option<f64> {
        self.ray_distance(origin, direction, maximum_distance_m, false)
    }
    fn ray_distance(
        &self,
        origin: Vec3,
        direction: Vec3,
        maximum_distance_m: f64,
        outward_only: bool,
    ) -> Option<f64> {
        if self.nodes.is_empty() || !maximum_distance_m.is_finite() || maximum_distance_m <= 0.0 {
            return None;
        }
        let o = charloom_core::v3(origin);
        let d = charloom_core::v3(direction).try_normalize(1e-12)?;
        let mut pending = vec![0];
        let mut hits = vec![];
        while let Some(i) = pending.pop() {
            let node = &self.nodes[i];
            let (mut near, mut far) = (0.0f64, maximum_distance_m);
            for k in 0..3 {
                if d[k].abs() < 1e-14 {
                    if o[k] < node.min[k] || o[k] > node.max[k] {
                        far = -1.;
                        break;
                    }
                } else {
                    let a = (node.min[k] - o[k]) / d[k];
                    let b = (node.max[k] - o[k]) / d[k];
                    near = near.max(a.min(b));
                    far = far.min(a.max(b));
                }
            }
            if near > far {
                continue;
            }
            if let Some(children) = node.children {
                pending.extend(children);
                continue;
            }
            for triangle in &self.triangles[node.start..node.end] {
                let h = d.cross(&triangle.e2);
                let det = triangle.e1.dot(&h);
                if det.abs() < 1e-14 {
                    continue;
                }
                let s = o - triangle.a;
                let u = s.dot(&h) / det;
                if !(-1e-9..=1. + 1e-9).contains(&u) {
                    continue;
                }
                let q = s.cross(&triangle.e1);
                let v = d.dot(&q) / det;
                if v < -1e-9 || u + v > 1. + 1e-9 {
                    continue;
                }
                let t = triangle.e2.dot(&q) / det;
                if t > 1e-7 && t <= maximum_distance_m {
                    hits.push((t, triangle.e1.cross(&triangle.e2).dot(&d).signum()));
                }
            }
        }
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut at = 0;
        while at < hits.len() {
            let mut end = at + 1;
            let mut positive = hits[at].1 > 0.;
            let mut negative = hits[at].1 < 0.;
            while end < hits.len() && (hits[end].0 - hits[at].0).abs() < 1e-7 {
                positive |= hits[end].1 > 0.;
                negative |= hits[end].1 < 0.;
                end += 1;
            }
            if !outward_only || positive && !negative {
                return Some(hits[at].0);
            }
            at = end;
        }
        None
    }
    pub fn line_intervals(&self, origin: Vector3<f64>, direction: Vector3<f64>) -> Vec<[f64; 2]> {
        if self.nodes.is_empty() || direction.norm_squared() < 1e-20 {
            return vec![];
        }
        let direction = direction.normalize();
        let mut pending = vec![0];
        let mut hits = Vec::new();
        while let Some(at) = pending.pop() {
            let node = &self.nodes[at];
            let (mut near, mut far) = (f64::NEG_INFINITY, f64::INFINITY);
            for a in 0..3 {
                if direction[a].abs() < 1e-14 {
                    if origin[a] < node.min[a] - 1e-10 || origin[a] > node.max[a] + 1e-10 {
                        near = f64::INFINITY;
                        far = f64::NEG_INFINITY;
                        break;
                    }
                } else {
                    let x = (node.min[a] - origin[a]) / direction[a];
                    let y = (node.max[a] - origin[a]) / direction[a];
                    near = near.max(x.min(y));
                    far = far.min(x.max(y));
                }
            }
            if near > far + 1e-10 {
                continue;
            }
            if let Some(children) = node.children {
                pending.extend(children);
                continue;
            }
            for triangle in &self.triangles[node.start..node.end] {
                let h = direction.cross(&triangle.e2);
                let det = triangle.e1.dot(&h);
                if det.abs() < 1e-14 {
                    continue;
                }
                let s = origin - triangle.a;
                let u = s.dot(&h) / det;
                if !(-1e-9..=1.0 + 1e-9).contains(&u) {
                    continue;
                }
                let q = s.cross(&triangle.e1);
                let v = direction.dot(&q) / det;
                if v < -1e-9 || u + v > 1.0 + 1e-9 {
                    continue;
                }
                let t = triangle.e2.dot(&q) / det;
                hits.push((t, triangle.e1.cross(&triangle.e2).dot(&direction).signum()));
            }
        }
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Adjacent triangles hit along a shared edge describe one crossing.
        // Opposing coincident sheets are ambiguous and must not certify volume.
        let mut crossings: Vec<(f64, f64)> = Vec::new();
        for hit in hits {
            if let Some(last) = crossings.last_mut()
                && (last.0 - hit.0).abs() < 1e-7
            {
                if last.1 != hit.1 {
                    return vec![];
                }
            } else {
                crossings.push(hit);
            }
        }
        if !crossings.len().is_multiple_of(2) {
            return vec![];
        }
        crossings
            .chunks_exact(2)
            .filter_map(|pair| {
                (pair[0].1 * pair[1].1 < 0.0 && pair[1].0 - pair[0].0 > 1e-7)
                    .then_some([pair[0].0, pair[1].0])
            })
            .collect()
    }
    pub fn enclosing_interval(
        &self,
        origin: Vector3<f64>,
        direction: Vector3<f64>,
    ) -> Option<[f64; 2]> {
        self.line_intervals(origin, direction)
            .into_iter()
            .find(|p| p[0] < -1e-7 && p[1] > 1e-7)
    }

    /// Strict noncoplanar crossings touching the requested source triangles.
    /// Coplanar overlap, mere proximity, and shared-vertex contact are excluded.
    /// This is a diagnostic for sampled poses, not continuous collision detection.
    pub fn crossing_pairs(
        &self,
        selected: &BTreeSet<[usize; 2]>,
        epsilon: f64,
    ) -> BTreeSet<[[usize; 2]; 2]> {
        let mut result = BTreeSet::new();
        for a in self
            .triangles
            .iter()
            .filter(|t| selected.contains(&t.source))
        {
            let mut pending = vec![0];
            while let Some(index) = pending.pop() {
                let node = &self.nodes[index];
                if (0..3)
                    .any(|k| a.max[k] < node.min[k] - epsilon || node.max[k] < a.min[k] - epsilon)
                {
                    continue;
                }
                if let Some(children) = node.children {
                    pending.extend(children);
                    continue;
                }
                for b in &self.triangles[node.start..node.end] {
                    if a.source == b.source
                        || (selected.contains(&b.source) && a.source > b.source)
                        || (0..3)
                            .any(|k| a.max[k] < b.min[k] - epsilon || b.max[k] < a.min[k] - epsilon)
                    {
                        continue;
                    }
                    if proper_crossing(a, b, epsilon) {
                        result.insert(if a.source < b.source {
                            [a.source, b.source]
                        } else {
                            [b.source, a.source]
                        });
                    }
                }
            }
        }
        result
    }
}

fn closest_barycentric(
    p: Vector3<f64>,
    a: Vector3<f64>,
    b: Vector3<f64>,
    c: Vector3<f64>,
) -> [f64; 3] {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return [1., 0., 0.];
    }
    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return [0., 1., 0.];
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return [1. - v, v, 0.];
    }
    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return [0., 0., 1.];
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return [1. - w, 0., w];
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return [0., 1. - w, w];
    }
    let total = va + vb + vc;
    let v = vb / total;
    let w = vc / total;
    [1. - v - w, v, w]
}

fn proper_crossing(a: &Triangle, b: &Triangle, epsilon: f64) -> bool {
    let vertices = |t: &Triangle| [t.a, t.a + t.e1, t.a + t.e2];
    let av = vertices(a);
    let bv = vertices(b);
    if av.iter().any(|p| {
        bv.iter()
            .any(|q| (p - q).norm_squared() < epsilon * epsilon)
    }) {
        return false;
    }
    let pierces = |points: &[Vector3<f64>; 3], face: &Triangle| {
        let n = face.e1.cross(&face.e2).normalize();
        let distances = points.map(|p| (p - face.a).dot(&n));
        if distances.iter().all(|d| *d >= -epsilon) || distances.iter().all(|d| *d <= epsilon) {
            return false;
        }
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            if distances[i] * distances[j] >= -epsilon * epsilon {
                continue;
            }
            let t = distances[i] / (distances[i] - distances[j]);
            let p = points[i] + (points[j] - points[i]) * t - face.a;
            let d00 = face.e1.dot(&face.e1);
            let d01 = face.e1.dot(&face.e2);
            let d11 = face.e2.dot(&face.e2);
            let d20 = p.dot(&face.e1);
            let d21 = p.dot(&face.e2);
            let denominator = d00 * d11 - d01 * d01;
            if denominator <= 1e-24 {
                continue;
            }
            let u = (d11 * d20 - d01 * d21) / denominator;
            let v = (d00 * d21 - d01 * d20) / denominator;
            if u >= -1e-9 && v >= -1e-9 && u + v <= 1.0 + 1e-9 {
                return true;
            }
        }
        false
    };
    pierces(&av, b) || pierces(&bv, a)
}

#[cfg(test)]
mod tests {
    use super::*;
    pub(super) fn cube(z: f64) -> Mesh {
        Mesh {
            name: "cube".into(),
            node: 0,
            primitive: 0,
            positions: vec![
                [-1., -1., z - 1.],
                [1., -1., z - 1.],
                [1., 1., z - 1.],
                [-1., 1., z - 1.],
                [-1., -1., z + 1.],
                [1., -1., z + 1.],
                [1., 1., z + 1.],
                [-1., 1., z + 1.],
            ],
            triangles: vec![
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
            skin: None,
            influences: vec![],
            morph_targets: vec![],
            morph_weights: vec![],
        }
    }
    #[test]
    fn ray_pairs_distinguish_interiors_empty_gaps_and_open_sheets() {
        let meshes = vec![cube(-2.), cube(2.)];
        let points = meshes
            .iter()
            .map(|m| m.positions.clone())
            .collect::<Vec<_>>();
        let query = SurfaceQuery::new(&meshes, &points);
        assert_eq!(
            query.line_intervals(Vector3::zeros(), Vector3::z()),
            vec![[-3., -1.], [1., 3.]]
        );
        assert!(
            query
                .enclosing_interval(Vector3::zeros(), Vector3::z())
                .is_none()
        );
        assert_eq!(
            query.enclosing_interval(Vector3::new(0., 0., 2.), Vector3::z()),
            Some([-1., 1.])
        );
        let mut sheet = cube(0.);
        sheet.triangles = vec![[4, 5, 6], [4, 6, 7]];
        let query = SurfaceQuery::new(std::slice::from_ref(&sheet), &[sheet.positions.clone()]);
        assert!(
            query
                .line_intervals(Vector3::zeros(), Vector3::z())
                .is_empty()
        );
    }
    #[test]
    fn triangle_audit_reports_crossings_but_ignores_neighbors_and_coplanar_contact() {
        let mut mesh = cube(0.);
        mesh.positions = vec![
            [-1., -1., 0.],
            [1., -1., 0.],
            [0., 1., 0.],
            [0., -0.5, -1.],
            [0., -0.5, 1.],
            [0., 0.5, 1.],
            [-1., -1., 0.],
            [0., 1., 0.],
            [-2., 0., 0.],
        ];
        mesh.triangles = vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]];
        let query = SurfaceQuery::new(std::slice::from_ref(&mesh), &[mesh.positions.clone()]);
        let pairs = query.crossing_pairs(&BTreeSet::from([[0, 0]]), 1e-6);
        assert_eq!(pairs, BTreeSet::from([[[0, 0], [0, 1]]]));
        assert!(
            query
                .crossing_pairs(&BTreeSet::from([[0, 2]]), 1e-6)
                .is_empty()
        );
    }
}
#[test]
fn first_hit_includes_an_inward_open_sheet_before_an_outer_surface() {
    let mesh = Mesh {
        name: "open_hair".into(),
        node: 0,
        primitive: 0,
        positions: vec![
            [-1., -1., 0.8],
            [1., -1., 0.8],
            [0., 1., 0.8],
            [-1., -1., 1.],
            [1., -1., 1.],
            [0., 1., 1.],
        ],
        triangles: vec![[0, 2, 1], [3, 4, 5]],
        skin: None,
        influences: vec![],
        morph_targets: vec![],
        morph_weights: vec![],
    };
    let query = SurfaceQuery::new(
        std::slice::from_ref(&mesh),
        std::slice::from_ref(&mesh.positions),
    );
    assert!((query.first_ray_distance([0.; 3], [0., 0., 1.], 2.).unwrap() - 0.8).abs() < 1e-12);
    assert!(
        (query
            .outward_ray_distance([0.; 3], [0., 0., 1.], 2.)
            .unwrap()
            - 1.)
            .abs()
            < 1e-12
    );
}
