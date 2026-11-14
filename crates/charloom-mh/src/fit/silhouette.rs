//! Silhouette fit to the bodysuit edits' outlines (`guide::SuitContour`,
//! `scripts/mh-suit-contours.py`): the body's own outline in each view
//! (vertices on the edge of its projected shape, grazing the view) pulled
//! onto the nearest point of the suit's smooth curve, and every curve point
//! pulling the nearest outline vertex: closest points both ways, as many as
//! the curve has samples. The outline rows (`outline_*`) sampled that curve
//! every centimetre and sized each row on its own; between them the body
//! rippled into rings (thigh, knee, calf: a Michelin man).
use crate::geom::{V, v};
use crate::guide::{Camera, SuitContour};
use charloom_core::Vec3;
use std::collections::HashMap;

/// Body normals within this of the view plane (|n · forward|) may lie on
/// the outline.
const GRAZING: f64 = 0.45;
/// A vertex within this many pixels of its projected shape's edge is on it.
const EDGE_PX: i32 = 2;
/// Grid cell (pixels) of the nearest-point searches.
const CELL: f64 = 8.0;

pub struct View {
    camera: Camera,
    grazing: f64,
    external_only: bool,
    samples: Vec<[f64; 2]>,
    /// Per sample, how far the edits agree there (`SuitContour::weights`).
    weights: Vec<f64>,
    grid: HashMap<(i32, i32), Vec<u32>>,
}

impl View {
    pub fn camera(&self) -> &Camera {
        &self.camera
    }
    pub fn samples(&self) -> &[[f64; 2]] {
        &self.samples
    }
}

fn cell_of(p: [f64; 2]) -> (i32, i32) {
    ((p[0] / CELL).floor() as i32, (p[1] / CELL).floor() as i32)
}

fn nearest(
    points: &[[f64; 2]],
    grid: &HashMap<(i32, i32), Vec<u32>>,
    p: [f64; 2],
    r: f64,
) -> Option<u32> {
    let (cx, cy) = cell_of(p);
    let k = (r / CELL).ceil() as i32;
    let mut best: Option<(f64, u32)> = None;
    for dx in -k..=k {
        for dy in -k..=k {
            let Some(list) = grid.get(&(cx + dx, cy + dy)) else {
                continue;
            };
            for &j in list {
                let q = points[j as usize];
                let d2 = (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2);
                if d2 <= r * r && best.is_none_or(|b| d2 < b.0) {
                    best = Some((d2, j));
                }
            }
        }
    }
    best.map(|b| b.1)
}

fn grid_of(points: &[[f64; 2]]) -> HashMap<(i32, i32), Vec<u32>> {
    let mut grid: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
    for (j, p) in points.iter().enumerate() {
        grid.entry(cell_of(*p)).or_default().push(j as u32);
    }
    grid
}

impl View {
    pub fn new(c: &SuitContour) -> Self {
        let samples: Vec<[f64; 2]> = c.curves.iter().flatten().copied().collect();
        let weights: Vec<f64> = c
            .curves
            .iter()
            .enumerate()
            .flat_map(|(k, curve)| {
                (0..curve.len()).map(move |j| {
                    c.weights
                        .get(k)
                        .and_then(|w| w.get(j))
                        .copied()
                        .unwrap_or(1.0)
                })
            })
            .collect();
        let grid = grid_of(&samples);
        View {
            camera: c.camera.clone(),
            grazing: GRAZING,
            external_only: false,
            samples,
            weights,
            grid,
        }
    }

    /// Crop and supersample the same orthographic camera for head data.
    /// No normal gate: folds at a hidden nape are still observed outlines.
    pub fn head(c: &SuitContour, x: &[V], region: &[bool]) -> Self {
        let mut view = Self::new(c);
        let points: Vec<_> = x
            .iter()
            .zip(region)
            .filter(|(_, on)| **on)
            .map(|(p, _)| c.camera.project(crate::geom::a(*p)))
            .collect();
        let minx = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min) - 25.0;
        let maxx = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max) + 25.0;
        let miny = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min) - 25.0;
        let maxy = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max) + 25.0;
        let factor = 3.0;
        let width = ((maxx - minx) * factor).ceil() as usize;
        let height = ((maxy - miny) * factor).ceil() as usize;
        let cx = minx + width as f64 / (2.0 * factor);
        let cy = miny + height as f64 / (2.0 * factor);
        view.camera.center = crate::geom::a(
            v(c.camera.center)
                + v(c.camera.right)
                    * (cx - c.camera.width as f64 / 2.0)
                    * c.camera.meters_per_pixel
                - v(c.camera.up) * (cy - c.camera.height as f64 / 2.0) * c.camera.meters_per_pixel,
        );
        view.camera.width = width;
        view.camera.height = height;
        view.camera.meters_per_pixel /= factor;
        // A cropped camera must also crop its data. Full-body samples below
        // the anatomical head must not pull a nape edge down toward a shoulder.
        let paired:Vec<_>=view.samples.iter().zip(&view.weights)
            .filter(|(p,_)|p[0]>=minx&&p[0]<=maxx&&p[1]>=miny&&p[1]<=maxy-25.0)
            .map(|(p,&w)| ([(p[0]-minx)*factor,(p[1]-miny)*factor],w)).collect();
        view.samples=paired.iter().map(|(p,_)|*p).collect();
        view.weights=paired.iter().map(|(_,w)|*w).collect();
        view.grid = grid_of(&view.samples);
        view.grazing = 1.01;
        view.external_only = true;
        view
    }

    fn to_world(&self, dx: f64, dy: f64) -> V {
        let mpp = self.camera.meters_per_pixel;
        v(self.camera.right) * (dx * mpp) - v(self.camera.up) * (dy * mpp)
    }

    /// Lift each registered outline point to the nearest rendered silhouette
    /// depth, then test its outward head-centred ray against scan hair. Hair
    /// only removes contradictory evidence; it never supplies an equality.
    pub(super) fn hair_arbiter(&mut self, x:&[V], triangles:&[[u32;3]], region:&[bool],
        centre:V, hair:&charloom_engine::surface_query::SurfaceQuery, reach:f64,
        clearance:f64, trust:f64) -> usize {
        let (constraints,_,_)=rendered_constraints(std::slice::from_ref(self),x,triangles,region,0.060);
        let surface:Vec<_>=constraints.iter().map(|c|c.vertices.iter().zip(c.bary)
            .fold(V::zeros(),|s,(&i,b)|s+x[i as usize]*b)).collect();
        let pixels:Vec<_>=surface.iter().zip(&constraints).map(|(&p,c)|{
            let(u,v,_)=self.camera.project(crate::geom::a(p+c.delta));[u,v]
        }).collect();
        let grid=grid_of(&pixels);let mut rejected=0;
        for k in 0..self.samples.len() {
            self.weights[k]*=trust;
            let q=self.samples[k];
            let Some(j)=nearest(&pixels,&grid,q,0.060/self.camera.meters_per_pixel)else{continue;};
            let p=surface[j as usize];let(u,v,_)=self.camera.project(crate::geom::a(p));
            let target=p+self.to_world(q[0]-u,q[1]-v);let radial=target-centre;
            let Some(dir)=radial.try_normalize(1e-12)else{continue;};
            if hair.outward_ray_distance(crate::geom::a(centre),crate::geom::a(dir),reach)
                .is_some_and(|h|radial.norm()>(h-clearance).max(0.0)) {
                self.weights[k]=0.0;rejected+=1;
            }
        }
        rejected
    }

    pub(super) fn evidence(&self)->serde_json::Value {
        serde_json::json!({"camera":self.camera,"samples":self.samples,"weights":self.weights})
    }
}

/// The body's outline vertices in one view: projected positions, and per
/// vertex whether it lies on the edge of the projected body (any vertex,
/// eligible or not, takes part in the shape).
pub(super) fn outline(
    cam: &Camera,
    x: &[V],
    normals: &[Vec3],
    triangles: &[[u32; 3]],
    grazing: f64,
    external_only: bool,
) -> (Vec<[f64; 2]>, Vec<bool>) {
    let (w, h) = (cam.width as i32, cam.height as i32);
    let proj: Vec<[f64; 2]> = x
        .iter()
        .map(|p| {
            let (px, py, _) = cam.project(crate::geom::a(*p));
            [px, py]
        })
        .collect();
    let mut mask = vec![false; (w * h) as usize];
    for t in triangles {
        let [a, b, c] = t.map(|i| proj[i as usize]);
        let x0 = a[0].min(b[0]).min(c[0]).floor().max(0.0) as i32;
        let x1 = a[0].max(b[0]).max(c[0]).ceil().min((w - 1) as f64) as i32;
        let y0 = a[1].min(b[1]).min(c[1]).floor().max(0.0) as i32;
        let y1 = a[1].max(b[1]).max(c[1]).ceil().min((h - 1) as f64) as i32;
        if x1 < x0 || y1 < y0 {
            continue;
        }
        let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        if area.abs() < 1e-9 {
            continue;
        }
        for py in y0..=y1 {
            for px in x0..=x1 {
                let p = [px as f64 + 0.5, py as f64 + 0.5];
                let e = |u: [f64; 2], q: [f64; 2]| {
                    (q[0] - u[0]) * (p[1] - u[1]) - (q[1] - u[1]) * (p[0] - u[0])
                };
                let (e0, e1, e2) = (e(a, b), e(b, c), e(c, a));
                if (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0) {
                    mask[(py * w + px) as usize] = true;
                }
            }
        }
    }
    let mut exterior = vec![false; mask.len()];
    if external_only {
        let mut queue = std::collections::VecDeque::new();
        for py in 0..h {
            for px in 0..w {
                if px == 0 || py == 0 || px == w - 1 || py == h - 1 {
                    let k = (py * w + px) as usize;
                    if !mask[k] && !exterior[k] {
                        exterior[k] = true;
                        queue.push_back((px, py));
                    }
                }
            }
        }
        while let Some((px, py)) = queue.pop_front() {
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let (x, y) = (px + dx, py + dy);
                if x < 0 || y < 0 || x >= w || y >= h {
                    continue;
                }
                let k = (y * w + x) as usize;
                if !mask[k] && !exterior[k] {
                    exterior[k] = true;
                    queue.push_back((x, y));
                }
            }
        }
    }
    let outside = |px: i32, py: i32| {
        if px < 0 || py < 0 || px >= w || py >= h {
            return true;
        }
        let k = (py * w + px) as usize;
        if external_only { exterior[k] } else { !mask[k] }
    };
    let at =
        |px: i32, py: i32| px >= 0 && py >= 0 && px < w && py < h && mask[(py * w + px) as usize];
    let near_edge = |px: i32, py: i32| {
        for dy in -EDGE_PX..=EDGE_PX {
            for dx in -EDGE_PX..=EDGE_PX {
                let (qx, qy) = (px + dx, py + dy);
                if at(qx, qy)
                    && (outside(qx + 1, qy)
                        || outside(qx - 1, qy)
                        || outside(qx, qy + 1)
                        || outside(qx, qy - 1))
                {
                    return true;
                }
            }
        }
        false
    };
    let fwd = v(cam.forward);
    let on: Vec<bool> = proj
        .iter()
        .zip(normals)
        .map(|(p, n)| v(*n).dot(&fwd).abs() < grazing && near_edge(p[0] as i32, p[1] as i32))
        .collect();
    (proj, on)
}

/// Per welded vertex: the in-plane pulls onto the views' curves summed by
/// each curve point's weight (the edits' agreement there), the weights'
/// sum and how many pulls there are (`eligible` vertices only; `reach_m`
/// the farthest pull), and per view (outline vertices, p50 and p95
/// distance in metres). `sum / weight` is the pull; `weight / hits` how
/// far to trust it.
pub fn pulls(
    views: &[View],
    x: &[V],
    normals: &[Vec3],
    triangles: &[[u32; 3]],
    eligible: &[bool],
    reach_m: f64,
) -> (Vec<V>, Vec<f64>, Vec<f64>, Vec<(usize, f64, f64)>) {
    let n = x.len();
    let mut sum = vec![V::zeros(); n];
    let mut count = vec![0.0f64; n];
    let mut hits = vec![0.0f64; n];
    let mut stats = vec![];
    for view in views {
        let r = reach_m / view.camera.meters_per_pixel;
        let (proj, on) = outline(
            &view.camera,
            x,
            normals,
            triangles,
            view.grazing,
            view.external_only,
        );
        // The whole outline, eligible or not: a part of the curve nearest an
        // excluded part (the face, the neck, a hand) pulls nothing, where
        // nearest among the eligible alone the jaw's curve pulled the scalp
        // and the shoulders.
        let ids: Vec<usize> = (0..n).filter(|&i| on[i]).collect();
        let pts: Vec<[f64; 2]> = ids.iter().map(|&i| proj[i]).collect();
        let body_grid = grid_of(&pts);
        let mut dist = vec![];
        let mut used = 0;
        // Body outline -> curve.
        for (k, &i) in ids.iter().enumerate() {
            if !eligible[i] {
                continue;
            }
            used += 1;
            if let Some(j) = nearest(&view.samples, &view.grid, pts[k], r) {
                let q = view.samples[j as usize];
                let w = view.weights[j as usize];
                let (dx, dy) = (q[0] - pts[k][0], q[1] - pts[k][1]);
                sum[i] += view.to_world(dx, dy) * w;
                count[i] += w;
                hits[i] += 1.0;
                dist.push((dx * dx + dy * dy).sqrt() * view.camera.meters_per_pixel);
            }
        }
        // Curve -> body outline (every part of the curve is reached).
        for (q, &w) in view.samples.iter().zip(&view.weights) {
            if let Some(k) = nearest(&pts, &body_grid, *q, r) {
                let i = ids[k as usize];
                if !eligible[i] {
                    continue;
                }
                let p = pts[k as usize];
                sum[i] += view.to_world(q[0] - p[0], q[1] - p[1]) * w;
                count[i] += w;
                hits[i] += 1.0;
            }
        }
        dist.sort_by(f64::total_cmp);
        let pct = |f: f64| {
            dist.get(((dist.len() as f64 - 1.0) * f) as usize)
                .copied()
                .unwrap_or(0.0)
        };
        stats.push((used, pct(0.5), pct(0.95)));
    }
    (sum, count, hits, stats)
}

/// A correspondence on the *rendered surface*, with barycentric support.
/// It is consumed by a smooth deformation field, never applied to a rim vertex.
pub struct SurfaceConstraint {
    pub vertices: [u32; 3],
    pub bary: [f64; 3],
    pub delta: V,
    pub axes: [V; 2],
    pub weight: f64,
}

/// Rasterize the whole mesh with depth and triangle ownership, flood the
/// exterior, and use its external boundary pixels in both correspondence
/// directions. Hidden triangles, holes, and nearby internal rim vertices do
/// not supply silhouette correspondences.
pub fn rendered_constraints(
    views: &[View],
    x: &[V],
    triangles: &[[u32; 3]],
    region: &[bool],
    reach_m: f64,
) -> (Vec<SurfaceConstraint>, Vec<bool>, Vec<(usize, f64, f64)>) {
    rendered_constraints_with(views, x, triangles, region, reach_m, false)
}

 pub fn rendered_constraints_with(
    views: &[View],
    x: &[V],
    triangles: &[[u32; 3]],
    region: &[bool],
    reach_m: f64,
    no_crossing: bool,
) -> (Vec<SurfaceConstraint>, Vec<bool>, Vec<(usize, f64, f64)>) {
    let mut constraints = Vec::new();
    let mut observed = vec![false; x.len()];
    let mut stats = Vec::new();
    for view in views {
        let cam = &view.camera;
        let (w, h) = (cam.width, cam.height);
        let projected: Vec<_> = x.iter().map(|p| cam.project(crate::geom::a(*p))).collect();
        let mut owner = vec![usize::MAX; w * h];
        let mut depth = vec![f64::INFINITY; w * h];
        let mut bary = vec![[0.0; 3]; w * h];
        for (ti, t) in triangles.iter().enumerate() {
            let p = t.map(|i| projected[i as usize]);
            let minx = p.iter().map(|p| p.0).fold(f64::INFINITY, f64::min).floor().max(0.0) as usize;
            let maxx = p.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max).ceil().min(w as f64 - 1.0) as usize;
            let miny = p.iter().map(|p| p.1).fold(f64::INFINITY, f64::min).floor().max(0.0) as usize;
            let maxy = p.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max).ceil().min(h as f64 - 1.0) as usize;
            if minx >= w || miny >= h || minx > maxx || miny > maxy { continue; }
            let det = (p[1].1 - p[2].1) * (p[0].0 - p[2].0) + (p[2].0 - p[1].0) * (p[0].1 - p[2].1);
            if det.abs() < 1e-12 { continue; }
            for py in miny..=maxy {
                for px in minx..=maxx {
                    let (u, v) = (px as f64 + 0.5, py as f64 + 0.5);
                    let a = ((p[1].1 - p[2].1) * (u - p[2].0) + (p[2].0 - p[1].0) * (v - p[2].1)) / det;
                    let b = ((p[2].1 - p[0].1) * (u - p[2].0) + (p[0].0 - p[2].0) * (v - p[2].1)) / det;
                    let c = 1.0 - a - b;
                    if a < -1e-9 || b < -1e-9 || c < -1e-9 { continue; }
                    let z = a * p[0].2 + b * p[1].2 + c * p[2].2;
                    let k = py * w + px;
                    if z < depth[k] { depth[k] = z; owner[k] = ti; bary[k] = [a, b, c]; }
                }
            }
        }
        let mut exterior = vec![false; w * h];
        let mut queue = std::collections::VecDeque::new();
        for py in 0..h { for px in 0..w {
            let k = py * w + px;
            if (px == 0 || py == 0 || px + 1 == w || py + 1 == h) && owner[k] == usize::MAX {
                exterior[k] = true; queue.push_back((px, py));
            }
        }}
        while let Some((px, py)) = queue.pop_front() {
            for (dx, dy) in [(-1,0),(1,0),(0,-1),(0,1)] {
                let (xx, yy) = (px as isize + dx, py as isize + dy);
                if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize { continue; }
                let k = yy as usize * w + xx as usize;
                if owner[k] == usize::MAX && !exterior[k] { exterior[k] = true; queue.push_back((xx as usize, yy as usize)); }
            }
        }
        let mut pixels = Vec::new();
        for py in 1..h-1 { for px in 1..w-1 {
            let k = py*w+px;
            if owner[k] == usize::MAX { continue; }
            if [k-1,k+1,k-w,k+w].iter().any(|&q| exterior[q]) { pixels.push(k); }
        }}
        let pts: Vec<_> = pixels.iter().map(|&k| [(k%w) as f64+0.5, (k/w) as f64+0.5]).collect();
        let grid = grid_of(&pts);
        let reach = reach_m / cam.meters_per_pixel;
        let mut distances = Vec::new();
        let mut add = |j: usize, q: [f64;2], weight: f64| {
            let k = pixels[j]; let t = triangles[owner[k]];
            if !t.iter().any(|&i| region[i as usize]) { return; }
            if no_crossing {
                let (from, steps) = (pts[j], ((q[0] - pts[j][0]).hypot(q[1] - pts[j][1])).ceil() as usize);
                let (mut open, mut again) = (0, 0);
                for s in 1..steps {
                    let f = s as f64 / steps as f64;
                    let (px, py) = (from[0] + (q[0] - from[0]) * f, from[1] + (q[1] - from[1]) * f);
                    if px < 0.0 || py < 0.0 || px >= w as f64 || py >= h as f64 { break; }
                    if owner[py as usize * w + px as usize] == usize::MAX { open += 1; again = 0; }
                    else if open >= 2 { again += 1; if again >= 3 { return; } }
                }
            }
            for (&i, &b) in t.iter().zip(&bary[k]) { if b > 0.02 { observed[i as usize] = true; } }
            let delta = view.to_world(q[0]-pts[j][0], q[1]-pts[j][1]);
            distances.push(delta.norm());
            constraints.push(SurfaceConstraint { vertices: t, bary: bary[k], delta,
                axes: [v(cam.right), v(cam.up)], weight });
        };
        // Sample by image arc density, independently of mesh tessellation.
        for (j, p) in pts.iter().enumerate().step_by(2) {
            if let Some(k) = nearest(&view.samples, &view.grid, *p, reach) {
                add(j, view.samples[k as usize], view.weights[k as usize]);
            }
        }
        for (&q, &weight) in view.samples.iter().zip(&view.weights) {
            if let Some(j) = nearest(&pts, &grid, q, reach) { add(j as usize, q, weight); }
        }
        distances.sort_by(f64::total_cmp);
        let pct = |f:f64| distances.get(((distances.len().saturating_sub(1)) as f64*f) as usize).copied().unwrap_or(0.0);
        stats.push((distances.len(), pct(0.5), pct(0.95)));
    }
    (constraints, observed, stats)
}

#[cfg(test)]
mod surface_tests {
    use super::*;
    #[test]
    fn rendered_boundary_uses_surface_barycentrics_and_excludes_occluded_triangles() {
        let camera=Camera{id:"test".into(),width:24,height:24,center:[0.0;3],right:[1.0,0.0,0.0],up:[0.0,1.0,0.0],forward:[0.0,0.0,1.0],meters_per_pixel:0.1};
        let contour=SuitContour{camera,curves:vec![vec![[3.0,3.0],[21.0,3.0],[21.0,21.0],[3.0,21.0]]],weights:vec![vec![1.0;4]],skin_anchors:vec![],rotation_limits_deg:[-5.,5.],scale_tied_to:None,skin_outline_anchors:vec![]};
        let x=vec![V::new(-0.8,-0.8,-1.0),V::new(0.8,-0.8,-1.0),V::new(0.8,0.8,-1.0),V::new(-0.8,0.8,-1.0),
            V::new(-0.4,-0.4,1.0),V::new(0.4,-0.4,1.0),V::new(0.0,0.4,1.0)];
        let (s,visible,_)=rendered_constraints(&[View::new(&contour)],&x,&[[0,1,2],[0,2,3],[4,5,6]],&vec![true;x.len()],1.0);
        assert!(!s.is_empty());
        assert!(s.iter().all(|s|s.vertices.iter().all(|&i|i<4)));
        assert!(s.iter().any(|s|s.bary.iter().filter(|&&b|b>0.1).count()>=2));
        assert!(visible[4..].iter().all(|v|!*v));
    }
}
