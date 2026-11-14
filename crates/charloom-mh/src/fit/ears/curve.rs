 use super::{Patch, rotation_fit};
use crate::geom::{V, v};
use charloom_core::Vec3;
use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Unit};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};

/// Outline parts in drawing order (as `guide::EAR_OUTLINE`).
pub const PARTS: [&str; 6] = ["front_top", "top", "back", "lobe", "front_bottom", "tragus"];

/// Residual weights (forward, up, out) per part: a support point is pinned
/// along its direction, loosely along the rim; depth (out) weighs less (a
/// marked point is on the scan's skin, but a rim seen edge-on is less sure).
const WEIGHTS: [[f64; 3]; 6] = [
    [1.0, 1.0, 0.6],
    [0.3, 1.0, 0.6],
    [1.0, 0.3, 0.6],
    [0.3, 1.0, 0.6],
    [1.0, 1.0, 0.6],
    [1.0, 1.0, 0.6],
];

/// MakeHuman ear targets used. (parameter, + target, − target or none,
/// prior width).
const TARGETS: [(&str, &str, Option<&str>, f64); 13] = [
    ("scale", "scale-incr", Some("scale-decr"), 0.5),
    (
        "scale-vert",
        "scale-vert-incr",
        Some("scale-vert-decr"),
        0.7,
    ),
    (
        "scale-depth",
        "scale-depth-incr",
        Some("scale-depth-decr"),
        0.7,
    ),
    ("trans-vert", "trans-up", Some("trans-down"), 0.7),
    ("trans-horiz", "trans-forward", Some("trans-backward"), 0.7),
    ("rot", "rot-forward", Some("rot-backward"), 0.7),
    ("lobe", "lobe-incr", Some("lobe-decr"), 0.7),
    ("flap", "flap-incr", Some("flap-decr"), 0.7),
    ("wing", "wing-incr", Some("wing-decr"), 0.7),
    ("shape-pointed", "shape-pointed", None, 0.12),
    ("shape-round", "shape-round", None, 0.2),
    ("shape-square", "shape-square", None, 0.2),
    ("shape-triangle", "shape-triangle", None, 0.12),
];

struct Param {
    name: &'static str,
    /// Offsets over `Shape::local` for positive and negative values.
    plus: Vec<V>,
    minus: Option<Vec<V>>,
    tau: f64,
}

/// One ear's shape model and its marked outline.
pub struct Shape {
    pub side: &'static str,
    /// Welded vertices MakeHuman's ear targets move (this side).
    local: Vec<usize>,
    /// Per local vertex, how far it follows the placed ear (1 on the pinna,
    /// the collar's feather weight around it).
    hold: Vec<f64>,
    /// The pinna (local indices of the ear patch): its outline parts are
    /// taken from these.
    ear: Vec<usize>,
    params: Vec<Param>,
         fixed: [usize; 3],
    /// Marked parts (index into `PARTS`) and their target points.
    pub marks: Vec<(usize, V)>,
    /// Target names found for this side (report).
    found: Vec<String>,
}

/// The skull placement of one ear and its side-plane frame.
pub struct Frame {
    pub r: Matrix3<f64>,
    pub s: f64,
    pub ca: V,
    pub cb: V,
    pub fwd: V,
    pub up: V,
    pub out: V,
    pub pivot: V,
    /// 1 m of a 1.7 m character, times the head scale.
    pub unit: f64,
}

impl Frame {
         fn base(&self, q: V) -> V {
        self.cb + self.r * (q - self.ca) * self.s
    }
}

/// Fitted parameters of one ear.
pub struct Fitted {
    flare: Matrix3<f64>,
    hinge: V,
    weights: Vec<f64>,
    shift: V,
    scale: f64,
    /// Turn, tilt and depth stretch (linear, about the pivot).
    rotation: Matrix3<f64>,
}

impl Shape {
    /// `targets`: the prior's ear targets (rest-space offsets per body
    /// vertex); `representative`: a body vertex per welded vertex. None
    /// when nothing of this ear is marked or the targets are missing.
    pub fn new(
        patch: &Patch,
        rest: &[V],
        neighbours: &[Vec<u32>],
        targets: &[(String, Vec<Vec3>)],
        representative: &[usize],
        landmarks: &BTreeMap<String, Vec3>,
    ) -> Option<Self> {
        let side = if patch.outward.x > 0.0 { "l" } else { "r" };
        let marks: Vec<(usize, V)> = PARTS
            .iter()
            .enumerate()
            .filter_map(|(k, part)| Some((k, v(*landmarks.get(&format!("ear_{side}_{part}"))?))))
            .collect();
        if marks.is_empty() {
            return None;
        }
        let by_name: HashMap<&str, &Vec<Vec3>> =
            targets.iter().map(|(n, d)| (n.as_str(), d)).collect();
        let pinna = pinna(patch, rest, neighbours);
        let in_ear: std::collections::HashSet<usize> = pinna.iter().copied().collect();
        // Feathered into the fitted head over the collar's width (as
        // `Ears::detect`), from the whole pinna.
        let height = rest.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max)
            - rest.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let unit = height / 1.75;
        let c = pinna.iter().map(|&i| rest[i]).sum::<V>() / pinna.len() as f64;
        let radius = pinna
            .iter()
            .map(|&i| (rest[i] - c).norm())
            .fold(0.0, f64::max);
        let feather = radius.clamp(0.012 * unit, 0.03 * unit);
        let mut distance: HashMap<usize, f64> = pinna.iter().map(|&i| (i, 0.0)).collect();
        let mut queue: std::collections::VecDeque<usize> = pinna.iter().copied().collect();
        while let Some(i) = queue.pop_front() {
            let di = distance[&i];
            for &j in &neighbours[i] {
                let j = j as usize;
                let next = di + (rest[i] - rest[j]).norm();
                if next < feather && distance.get(&j).is_none_or(|d| next + 1e-12 < *d) {
                    distance.insert(j, next);
                    queue.push_back(j);
                }
            }
        }
        let mut local: Vec<usize> = pinna.clone();
        let mut hold = vec![1.0; local.len()];
        let mut ring: Vec<(usize, f64)> = distance
            .iter()
            .filter(|(i, d)| **d > 0.0 && !in_ear.contains(i))
            .map(|(i, d)| (*i, *d))
            .collect();
        ring.sort_by_key(|r| r.0);
        for (i, d) in ring {
            let t = (d / feather).clamp(0.0, 1.0);
            local.push(i);
            hold.push(1.0 - t * t * (3.0 - 2.0 * t));
        }
        let ear: Vec<usize> = (0..pinna.len()).collect();
        let offsets = |t: &str| -> Option<Vec<V>> {
            let d = by_name.get(format!("ears/{side}-ear-{t}").as_str())?;
            Some(local.iter().map(|&i| v(d[representative[i]])).collect())
        };
        let mut found = vec![];
        let params: Vec<Param> = TARGETS
            .iter()
            .filter_map(|&(name, plus, minus, tau)| {
                let p = offsets(plus)?;
                found.push(format!("ears/{side}-ear-{plus}"));
                let m = minus.and_then(|m| {
                    let o = offsets(m)?;
                    found.push(format!("ears/{side}-ear-{m}"));
                    Some(o)
                });
                Some(Param {
                    name,
                    plus: p,
                    minus: m,
                    tau,
                })
            })
            .collect();
         
        let h = |k: usize| rest[local[ear[k]]].y;
        let f = |k: usize| rest[local[ear[k]]].z;
        let (lo, hi) = (0..ear.len()).fold((f64::INFINITY, f64::NEG_INFINITY), |(l, u), k| {
            (l.min(h(k)), u.max(h(k)))
        });
        let mut fs: Vec<f64> = (0..ear.len()).map(f).collect();
        fs.sort_by(f64::total_cmp);
        let middle = fs[fs.len() / 2];
        // The attachment loop: pinna vertices next to the head.
        let root: Vec<usize> = (0..ear.len())
            .filter(|&k| {
                f(k) > middle
                    && neighbours[local[ear[k]]]
                        .iter()
                        .any(|&n| !in_ear.contains(&(n as usize)))
            })
            .collect();
        let pick = |set: &[usize], key: &dyn Fn(usize) -> f64| {
            set.iter()
                .copied()
                .max_by(|&a, &b| key(a).total_cmp(&key(b)))
        };
        let front_top = pick(&root, &|k| h(k))?;
        let front_bottom = pick(&root, &|k| -h(k))?;
        let mid: Vec<usize> = (0..ear.len())
            .filter(|&k| (h(k) - (lo + (hi - lo) * 0.45)).abs() < (hi - lo) * 0.15)
            .collect();
        let tragus = pick(&mid, &|k| f(k))?;
        Some(Self {
            side,
            local,
            hold,
            ear,
            params,
            fixed: [front_top, front_bottom, tragus],
            marks,
            found,
        })
    }

    /// Welded vertices that are part of the placed pinna (followed at least
    /// halfway).
    pub fn members(&self) -> std::collections::HashSet<usize> {
        self.local
            .iter()
            .zip(&self.hold)
            .filter(|(_, h)| **h >= 0.5)
            .map(|(i, _)| *i)
            .collect()
    }

    /// Parameter count: the targets, then forward / up / out shift, log
    /// scale, turn, flare, tilt and log depth stretch (all scaled to about 1
    /// at their prior width).
    fn dims(&self) -> usize {
        self.params.len() + 8
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        let mut b: Vec<(f64, f64)> = self
            .params
            .iter()
            .map(|p| (if p.minus.is_some() { -1.0 } else { 0.0 }, 1.0))
            .collect();
        b.extend([
            (-5.0, 5.0),
            (-5.0, 5.0),
            (-1.5, 0.75),
            (0.8f64.ln() / 1.4f64.ln(), 2.2f64.ln() / 1.4f64.ln()),
            (-1.67, 1.67),
            (-0.6, 2.3),
            (-1.7, 2.6),
            (0.8f64.ln() / 1.4f64.ln(), 2.0f64.ln() / 1.4f64.ln()),
        ]);
        b
    }

    fn decode(&self, theta: &[f64], frame: &Frame, rest: &[V]) -> Fitted {
        let rest_of = |j: usize| rest[self.local[j]];
        let k = self.params.len();
        let u = frame.unit;
        let turn = theta[k + 4] * 0.3;
        let axis = Unit::new_normalize(frame.fwd.cross(&frame.up));
        // Flare: out about a vertical hinge at the front attachment (the
        // back of the helix turns outward for positive values).
        let hinge_axis = Unit::new_normalize(frame.out.cross(&frame.fwd));
        let hinge = self.fixed[..2]
            .iter()
            .map(|&k| frame.base(rest_of(self.ear[k])))
            .sum::<V>()
            / 2.0;
        Fitted {
            flare: Rotation3::from_axis_angle(&hinge_axis, theta[k + 5] * 0.35).into_inner(),
            hinge,
            weights: theta[..k].to_vec(),
            shift: frame.fwd * theta[k] * 0.01 * u
                + frame.up * theta[k + 1] * 0.01 * u
                + frame.out * theta[k + 2] * 0.004 * u,
            scale: (theta[k + 3] * 1.4f64.ln()).exp(),
            // Turn in the side plane, tilt the top out of the head (about
            // the forward axis), stretch front to back (a round cartoon ear
            // is twice as deep as MakeHuman's, beyond its `scale-depth`).
            rotation: Rotation3::from_axis_angle(&axis, turn).into_inner()
                * Rotation3::from_axis_angle(
                    &Unit::new_normalize(frame.up.cross(&frame.out)),
                    theta[k + 6] * 0.35,
                )
                .into_inner()
                * (Matrix3::identity()
                    + frame.fwd
                        * frame.fwd.transpose()
                        * ((theta[k + 7] * 1.4f64.ln()).exp() - 1.0)),
        }
    }

    /// Rest-space offset of local vertex `j` under the weights.
    fn offset(&self, weights: &[f64], j: usize) -> V {
        let mut d = V::zeros();
        for (p, &w) in self.params.iter().zip(weights) {
            if w > 0.0 {
                d += p.plus[j] * w;
            } else if w < 0.0
                && let Some(m) = &p.minus
            {
                d += m[j] * -w;
            }
        }
        d
    }

    /// Placed position of local vertex `j`.
    fn place(&self, fit: &Fitted, frame: &Frame, rest: &[V], j: usize) -> V {
        let b = frame.base(rest[self.local[j]] + self.offset(&fit.weights, j));
        let f = fit.hinge + fit.flare * (b - fit.hinge);
        frame.pivot + fit.shift + fit.rotation * (f - frame.pivot) * fit.scale
    }

    /// MakeHuman's outline parts under `fit`.
    fn parts(&self, fit: &Fitted, frame: &Frame, rest: &[V]) -> [V; 6] {
        let pts: Vec<V> = self
            .ear
            .iter()
            .map(|&j| self.place(fit, frame, rest, j))
            .collect();
        parts_of(&pts, self.fixed, frame)
    }

    fn residuals(&self, theta: &[f64], frame: &Frame, rest: &[V]) -> Vec<f64> {
        let fit = self.decode(theta, frame, rest);
        let parts = self.parts(&fit, frame, rest);
        let sigma = 0.004 * frame.unit;
        let mut r = vec![];
        for &(k, target) in &self.marks {
            let d = parts[k] - target;
            for (axis, w) in [frame.fwd, frame.up, frame.out].iter().zip(WEIGHTS[k]) {
                r.push(d.dot(axis) * w / sigma);
            }
        }
        for (p, t) in self.params.iter().zip(theta) {
            r.push(t / p.tau);
        }
        r.extend(&theta[self.params.len()..]);
        r
    }

    /// Bounded Levenberg–Marquardt on the marked parts. Returns the
    /// placement of every local vertex (welded id, position) and a report.
    pub fn fit(&self, frame: &Frame, rest: &[V]) -> (Vec<(usize, V, f64)>, Value) {
        let n = self.dims();
        let cost = |t: &[f64]| {
            self.residuals(t, frame, rest)
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
        };
        let start = cost(&vec![0.0; n]);
        // Two starts: MakeHuman's ear, and one turned out 0.5 rad (a
        // cartoon ear standing out of the head is a separate minimum).
        let (theta, current) = [0.0, 0.5 / 0.35]
            .iter()
            .map(|&flare| {
                let mut t = vec![0.0; n];
                t[self.params.len() + 5] = flare;
                self.descend(t, frame, rest)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        self.report(theta, current, start, frame, rest)
    }

    /// Bounded Levenberg–Marquardt from `theta`: (solution, cost).
    fn descend(&self, mut theta: Vec<f64>, frame: &Frame, rest: &[V]) -> (Vec<f64>, f64) {
        let n = self.dims();
        let bounds = self.bounds();
        let cost = |t: &[f64]| {
            self.residuals(t, frame, rest)
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
        };
        let mut current = cost(&theta);
        let mut mu = 1e-2;
        for _ in 0..40 {
            let r0 = DVector::from_vec(self.residuals(&theta, frame, rest));
            let mut jac = DMatrix::zeros(r0.len(), n);
            for c in 0..n {
                let mut t = theta.clone();
                let h = 1e-4;
                t[c] += h;
                let r1 = self.residuals(&t, frame, rest);
                for (row, x) in r1.iter().enumerate() {
                    jac[(row, c)] = (x - r0[row]) / h;
                }
            }
            let jt = jac.transpose();
            let g = &jt * &r0;
            let h = &jt * &jac;
            let mut improved = false;
            for _ in 0..8 {
                let mut a = h.clone();
                for d in 0..n {
                    a[(d, d)] += mu * (1.0 + h[(d, d)]);
                }
                let Some(step) = a.lu().solve(&(-&g)) else {
                    mu *= 10.0;
                    continue;
                };
                let next: Vec<f64> = theta
                    .iter()
                    .zip(step.iter())
                    .zip(&bounds)
                    .map(|((t, s), (lo, hi))| (t + s).clamp(*lo, *hi))
                    .collect();
                let c = cost(&next);
                if c < current {
                    theta = next;
                    current = c;
                    mu = (mu / 3.0).max(1e-7);
                    improved = true;
                    break;
                }
                mu *= 4.0;
            }
            if !improved || g.norm() < 1e-9 {
                break;
            }
        }
        (theta, current)
    }

    fn report(
        &self,
        theta: Vec<f64>,
        current: f64,
        start: f64,
        frame: &Frame,
        rest: &[V],
    ) -> (Vec<(usize, V, f64)>, Value) {
        let fit = self.decode(&theta, frame, rest);
        let placed: Vec<(usize, V, f64)> = (0..self.local.len())
            .map(|j| {
                (
                    self.local[j],
                    self.place(&fit, frame, rest, j),
                    self.hold[j],
                )
            })
            .collect();
        let parts = self.parts(&fit, frame, rest);
        let unit = frame.unit / frame.s.max(1e-9);
        let residual: BTreeMap<String, Value> = self
            .marks
            .iter()
            .map(|&(k, t)| {
                let d = parts[k] - t;
                (
                    PARTS[k].to_string(),
                    json!({
                        "in_plane_mm": (d - frame.out * d.dot(&frame.out)).norm() / unit * 1000.0,
                        "out_mm": d.dot(&frame.out) / unit * 1000.0,
                    }),
                )
            })
            .collect();
        let k = self.params.len();
        let weights: BTreeMap<&str, f64> = self
            .params
            .iter()
            .zip(&theta)
            .filter(|(_, w)| w.abs() > 1e-3)
            .map(|(p, w)| (p.name, *w))
            .collect();
        let report = json!({
            "applied": true,
            "side": self.side,
            "method": "marked ear outline (MakeHuman ear targets + side-plane similarity)",
            "marked": self.marks.iter().map(|m| PARTS[m.0]).collect::<Vec<_>>(),
            "targets_available": self.found.len(),
            "weights": weights,
            "scale": fit.scale,
            "turn_rad": theta[k + 4] * 0.3,
            "flare_rad": theta[k + 5] * 0.35,
            "tilt_rad": theta[k + 6] * 0.35,
            "depth_stretch": (theta[k + 7] * 1.4f64.ln()).exp(),
            "shift_cm": [theta[k] * frame.unit / unit, theta[k + 1] * frame.unit / unit, theta[k + 2] * 0.4 * frame.unit / unit],
            "residual": residual,
            "cost": {"start": start, "end": current},
        });
        (placed, report)
    }

    /// The outline parts of the placed ear in `x` (frame from the skull):
    /// marked points where marked, MakeHuman's own elsewhere.
    pub fn outline(&self, x: &[V], frame: &Frame) -> [V; 6] {
        let pts: Vec<V> = self.ear.iter().map(|&j| x[self.local[j]]).collect();
        let mut parts = parts_of(&pts, self.fixed, frame);
        for &(k, t) in &self.marks {
            parts[k] = t;
        }
        parts
    }
}

/// MakeHuman's pinna on the rest surface: the thin-shell patch plus what
/// is seen from the side inside its outline (concha floor, tragus), not
/// the head hidden behind the pinna (an outward ray from it meets the
/// shell). Connected to the patch.
fn pinna(patch: &Patch, rest: &[V], neighbours: &[Vec<u32>]) -> Vec<usize> {
    let out = patch.outward;
    let plane = |p: V| [p.z, p.y];
    let shell: std::collections::HashSet<usize> = patch.vertices.iter().copied().collect();
    let hull = convex_hull(
        &patch
            .vertices
            .iter()
            .map(|&i| plane(rest[i]))
            .collect::<Vec<_>>(),
    );
    let tris: Vec<[usize; 3]> = patch
        .window
        .iter()
        .map(|t| t.map(|i| i as usize))
        .filter(|t| t.iter().all(|i| shell.contains(i)))
        .collect();
    // Lateral height of the shell over a side-plane point, if any.
    let over = |q: [f64; 2]| -> Option<f64> {
        let mut best: Option<f64> = None;
        for t in &tris {
            let p = t.map(|i| plane(rest[i]));
            let det = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
            if det.abs() < 1e-14 {
                continue;
            }
            let l1 = ((q[0] - p[0][0]) * (p[2][1] - p[0][1])
                - (p[2][0] - p[0][0]) * (q[1] - p[0][1]))
                / det;
            let l2 = ((p[1][0] - p[0][0]) * (q[1] - p[0][1])
                - (q[0] - p[0][0]) * (p[1][1] - p[0][1]))
                / det;
            if l1 < 0.0 || l2 < 0.0 || l1 + l2 > 1.0 {
                continue;
            }
            let h = (rest[t[0]] * (1.0 - l1 - l2) + rest[t[1]] * l1 + rest[t[2]] * l2).dot(&out);
            best = Some(best.map_or(h, |b: f64| b.max(h)));
        }
        best
    };
    let height = rest.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max)
        - rest.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
    let margin = 0.003 * height / 1.75;
    let candidate = |i: usize| {
        let q = plane(rest[i]);
        signed_distance(&hull, q) < margin
            && over(q).is_none_or(|h| h < rest[i].dot(&out) + 0.002 * height / 1.75)
    };
    let mut seen = shell.clone();
    let mut queue: std::collections::VecDeque<usize> = patch.vertices.iter().copied().collect();
    let mut out_set = patch.vertices.clone();
    while let Some(i) = queue.pop_front() {
        for &j in &neighbours[i] {
            let j = j as usize;
            if !seen.contains(&j) && rest[j].x * out.x > 0.0 && candidate(j) {
                seen.insert(j);
                out_set.push(j);
                queue.push_back(j);
            }
        }
    }
    out_set.sort_unstable();
    out_set
}

/// Convex hull (counter-clockwise) of 2D points.
fn convex_hull(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut p = points.to_vec();
    p.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut hull: Vec<[f64; 2]> = vec![];
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &[f64; 2]>> = if pass == 0 {
            Box::new(p.iter())
        } else {
            Box::new(p.iter().rev())
        };
        for &q in iter {
            while hull.len() >= start + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], q) <= 0.0
            {
                hull.pop();
            }
            hull.push(q);
        }
        hull.pop();
    }
    hull
}

/// Outline parts of placed pinna vertices `pts` (`Shape::ear` order).
fn parts_of(pts: &[V], fixed: [usize; 3], frame: &Frame) -> [V; 6] {
    let tau = 0.0015 * frame.unit;
    let support = |dir: V| {
        let m = pts
            .iter()
            .map(|p| p.dot(&dir))
            .fold(f64::NEG_INFINITY, f64::max);
        let (mut sum, mut total) = (V::zeros(), 0.0);
        for p in pts {
            let w = ((p.dot(&dir) - m) / tau).exp();
            sum += p * w;
            total += w;
        }
        sum / total.max(1e-300)
    };
    [
        pts[fixed[0]],
        support(frame.up),
        support(-frame.fwd),
        support(-frame.up),
        pts[fixed[1]],
        pts[fixed[2]],
    ]
}

/// Skull frame of a patch in `x` (as the transport's placement).
pub fn frame(patch: &Patch, rest: &[V], x: &[V], skull: &[usize], s: f64, unit: f64) -> Frame {
    let (r, ca, cb) = if skull.len() >= 3 {
        rotation_fit(rest, x, skull)
    } else {
        rotation_fit(rest, x, &patch.vertices)
    };
    let out = (r * patch.outward).normalize();
    let up = (r * V::y()).normalize();
    let up = (up - out * up.dot(&out)).normalize();
    let fwd = up.cross(&out);
    let fwd = if fwd.dot(&(r * V::z())) < 0.0 {
        -fwd
    } else {
        fwd
    };
    let base = |q: V| cb + r * (q - ca) * s;
    let c3 = patch.vertices.iter().map(|&i| base(rest[i])).sum::<V>() / patch.vertices.len() as f64;
    let ring = skull.iter().map(|&i| x[i]).sum::<V>() / skull.len().max(1) as f64;
    let pivot = c3 - out * (c3 - ring).dot(&out);
    Frame {
        r,
        s,
        ca,
        cb,
        fwd,
        up,
        out,
        pivot,
        unit: unit * s,
    }
}

/// Closed Catmull–Rom curve through `pts` (2D), `per` samples a segment.
pub fn closed_curve(pts: &[[f64; 2]], per: usize) -> Vec<[f64; 2]> {
    let n = pts.len();
    let mut out = vec![];
    for i in 0..n {
        let p = |k: usize| pts[(i + n + k - 1) % n];
        let (p0, p1, p2, p3) = (p(0), p(1), p(2), p(3));
        for s in 0..per {
            let t = s as f64 / per as f64;
            let (t2, t3) = (t * t, t * t * t);
            let c = |a: f64, b: f64, c: f64, d: f64| {
                0.5 * (2.0 * b
                    + (-a + c) * t
                    + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
                    + (-a + 3.0 * b - 3.0 * c + d) * t3)
            };
            out.push([c(p0[0], p1[0], p2[0], p3[0]), c(p0[1], p1[1], p2[1], p3[1])]);
        }
    }
    out
}

/// Signed distance to a closed polygon (negative inside).
pub fn signed_distance(poly: &[[f64; 2]], p: [f64; 2]) -> f64 {
    let n = poly.len();
    let mut inside = false;
    let mut best = f64::INFINITY;
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let x = a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            if p[0] < x {
                inside = !inside;
            }
        }
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy).max(1e-30))
            .clamp(0.0, 1.0);
        best = best.min((p[0] - a[0] - t * dx).hypot(p[1] - a[1] - t * dy));
    }
    if inside { -best } else { best }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_curve_passes_through_its_points_and_contains_the_middle() {
        let square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let c = closed_curve(&square, 8);
        assert_eq!(c.len(), 32);
        for (k, p) in square.iter().enumerate() {
            let q = c[k * 8];
            assert!((q[0] - p[0]).abs() < 1e-12 && (q[1] - p[1]).abs() < 1e-12);
        }
        assert!(signed_distance(&c, [0.5, 0.5]) < -0.4);
        // Rounded: the curve bulges 1/8 out of the square between points.
        assert!((signed_distance(&c, [2.0, 0.5]) - 0.875).abs() < 0.01);
    }
}
