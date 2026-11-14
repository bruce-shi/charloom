 use super::guard;
use crate::geom::{V, a, proper_rotation, v};
use crate::scan::Scan;
use crate::segment::{Label, Segmentation};
use charloom_core::Influence;
use nalgebra::{Matrix3, Rotation3, SMatrix, SVector, Unit};
use rayon::prelude::*;
use serde_json::{Value, json};

/// Largest rotation of the similarity (rad) and its scale range as a share
/// of the body fit's local scale range, for an unmarked and a mapped hand.
const ROTATION: [f64; 2] = [0.3, 0.1];
const SCALE_SHARE: [f64; 2] = [0.25, 0.3];
/// Girth gain range as shares of the body fit's local scale range (below
/// and above 1): a cartoon hand's fingers may be much thicker, hardly
/// thinner; a marked hand, whose finger radius is measured, up to three
/// times (stylised).
const GIRTH_SHARE: (f64, [f64; 2]) = (0.25, [1.5, 2.5]);
/// Measured radius ratios outside this range are a misplaced landmark.
const RADIUS_RATIO: (f64, f64) = (0.5, 4.0);
 const WRIST_GROWN: f64 = 1.15;
/// Face-like fit of a mapped five-digit hand: after the shape controls,
/// each vertex moves onto the scan's skin where the hit lies on its own
/// digit (the hit's nearest bone is of the same digit: a finger never pulls
/// onto its neighbour), capped at this share of the height, the move
/// smoothed over the hand's surface (`SURFACE_SMOOTHING` passes).
const SURFACE_CAP: f64 = 0.012;
const SURFACE_SMOOTHING: usize = 6;
const DIGIT_SHARE: f64 = 0.8;
/// Half-width of the wrist fade (share of the height).
const BAND: f64 = 0.012;
 const ANCHOR: f64 = 8.0;
/// Reach of the scan match (share of the height) and the distance at which
/// a match counts half (a fused finger web, a stray strap).
const REACH: f64 = 0.02;
const SOFT: f64 = 0.006;
/// Girth groups: the palm (wrist and metacarpal bones), the thumb and the
/// four fingers together. Per-finger girths followed misalignment (a finger
/// beside its scan finger read as thin, its neighbour on a fused web as
/// fat); a cartoon hand's fingers are thick alike.
const GROUPS: usize = 3;
 const CUT_GIRTH: f64 = 0.2;
/// Radial group of a cut finger's bones (gain [`CUT_GIRTH`]).
const CUT_GROUP: usize = usize::MAX;
/// Curled fingers (digits 2..5) and the curl range per knuckle (rad):
/// towards the palm, hardly back.
const CURLED: usize = 4;
const CURL: (f64, f64) = (-0.15, 0.8);
/// Hand shape parameters: girths, then curls.
const PARAMS: usize = GROUPS + CURLED;
type Params = [f64; PARAMS];
/// MakeHuman's own hand: unit girths, no curl.
const NEUTRAL: Params = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];

/// The phalanx index (1..3) of a finger bone, `finger{F}-{k}`.
fn phalanx_of(name: &str) -> Option<usize> {
    let n = name.to_lowercase();
    let rest = n.strip_prefix("finger")?;
    let k = rest.split('-').nth(1)?.chars().next()?.to_digit(10)? as usize;
    (1..=3).contains(&k).then_some(k)
}

/// Digit of a hand bone (0 for the palm: wrist and metacarpals, else
/// 1..5) and its side (0 left, 1 right).
fn group_of(name: &str) -> Option<(u8, usize)> {
    let n = name.to_lowercase();
    let side = if n.ends_with(".l") {
        0
    } else if n.ends_with(".r") {
        1
    } else {
        return None;
    };
    if n.starts_with("wrist") || n.starts_with("metacarpal") {
        return Some((side, 0));
    }
    let digit = n.strip_prefix("finger")?.chars().next()?.to_digit(10)? as usize;
    (1..=5).contains(&digit).then_some((side, digit))
}

/// A hand bone: its digit (0: palm) and axis (through `center` along `axis`,
/// from its vertices on the posed hand).
struct Bone {
    joint: usize,
    group: usize,
    center: V,
    axis: V,
    /// A finger bone of a digit the character lacks (most of the vertices
    /// it dominates take no data).
    cut: bool,
}

struct Hand {
    side: &'static str,
    mapped: bool,
    /// Welded vertices with a target, their blend weight, and whether they
    /// take scan evidence (digits the character lacks do not).
    members: Vec<(usize, f64, bool)>,
    /// Per member: radial direction scaled by skin weight, per girth group
    /// (the move at gain 2).
    radial: Vec<Vec<(usize, V)>>,
    /// Per member: (curled finger, phalanx 1..3, skin weight).
    curled: Vec<Vec<(usize, usize, f64)>>,
    /// Per curled finger: its knuckles (index 1..3) and flexion axis.
    knuckles: [[Option<V>; 4]; CURLED],
    flexion: [V; CURLED],
    vertices: usize,
    scale_range: (f64, f64),
    bounds: (Params, Params),
    /// Shape the parameters are held towards and the share of their own
    /// evidence that holds them (MakeHuman's hand lightly; a measured finger
    /// radius as strongly as the data); measured radius ratios per digit.
    prior: Params,
    hold: Params,
    /// Fitted forearm girth over the hand's at the wrist.
    wrist_ratio: Option<f64>,
    /// The scan's hand size over MakeHuman's posed hand (`size_ratio`).
    size_ratio: Option<f64>,
    measured: [Option<f64>; 5],
    max_rotation: f64,
    params: Params,
    /// Last similarity.
    scale: f64,
    rotation: Matrix3<f64>,
    raw_rotation: f64,
    raw_scale: f64,
    shift: V,
    /// Mean distance to the scan of the matched core vertices, of the
    /// current hand and of its new target; matched share.
    residual_current: f64,
    residual_target: f64,
    matched: f64,
    /// The face-like surface fit of a mapped five-digit hand (`SURFACE`):
    /// per member its digit (0 palm, 1..5, `usize::MAX` off the hand), the
    /// hand's bone segments by digit, the members' neighbours (member
    /// indices), and its last mean residual and matched share.
    surface: bool,
    digit: Vec<usize>,
    segments: Vec<(usize, V, V)>,
    nbr: Vec<Vec<usize>>,
    surface_residual: f64,
    surface_matched: f64,
}

impl Hand {
    /// Member `k` of the posed hand shaped by `p`: girths about the bone
    /// axes, then the finger curls (each knuckle turned about its flexion
    /// axis, the outer ones first), blended by the skin weights.
    fn model(&self, rest: &[V], k: usize, p: &Params) -> V {
        let s = self.radial[k]
            .iter()
            .fold(rest[self.members[k].0], |q, &(g, r)| {
                q + r * (if g == CUT_GROUP { CUT_GIRTH } else { p[g] } - 1.0)
            });
        let mut out = s;
        for &(d, phalanx, w) in &self.curled[k] {
            let angle = p[GROUPS + d];
            if angle == 0.0 {
                continue;
            }
            let turn = Rotation3::from_axis_angle(&Unit::new_unchecked(self.flexion[d]), angle);
            let mut t = s;
            for j in (1..=phalanx).rev() {
                if let Some(c) = self.knuckles[d][j] {
                    t = c + turn * (t - c);
                }
            }
            out += (t - s) * w;
        }
        out
    }
}

pub(super) struct Hands {
    /// Per welded vertex: weight of the hand target in [0, 1] (the
    /// per-vertex data keeps `1 − weight`).
    pub weight: Vec<f64>,
    hands: Vec<Hand>,
}

impl Hands {
         #[allow(clippy::too_many_arguments)]
    pub fn detect(
        side: &[Option<u8>],
        no_data: &[bool],
        influences: &[&[Influence]],
        names: &[&str],
        rest: &[V],
        shape_rest: &[V],
        neighbours: &[Vec<u32>],
        height: f64,
        scale_range: (f64, f64),
        mapped: [bool; 2],
        tip_radius: &[[[Option<f64>; 2]; 5]; 2],
        full_scale: bool,
        scan: Option<(&Scan, &Segmentation)>,
    ) -> Self {
        let n = side.len();
        let band = BAND * height;
        let groups: Vec<Option<(u8, usize)>> = names.iter().map(|n| group_of(n)).collect();
        let mut weight = vec![0.0f64; n];
        let mut hands = vec![];
        for (s, name) in [(0u8, "left"), (1u8, "right")] {
            let member: Vec<bool> = side.iter().map(|x| *x == Some(s)).collect();
            let vertices = member.iter().filter(|m| **m).count();
            if vertices == 0 {
                continue;
            }
            // Signed surface distance from the hand's boundary at the wrist:
            // positive into the hand, negative up the forearm.
            let boundary: Vec<bool> = (0..n)
                .map(|i| member[i] && neighbours[i].iter().any(|&j| !member[j as usize]))
                .collect();
            let d = guard::surface_distance(shape_rest, neighbours, &boundary, band * 2.0);
            let mut w: Vec<f64> = (0..n)
                .map(|i| {
                    if !d[i].is_finite() {
                        return if member[i] { 1.0 } else { 0.0 };
                    }
                    let t = if member[i] { d[i] } else { -d[i] };
                    guard::smoothstep((t + band) / (2.0 * band))
                })
                .collect();
            for _ in 0..2 {
                let next: Vec<f64> = (0..n)
                    .map(|i| {
                        if w[i] == 0.0 && neighbours[i].iter().all(|&j| w[j as usize] == 0.0) {
                            return 0.0;
                        }
                        let sum: f64 = neighbours[i].iter().map(|&j| w[j as usize]).sum();
                        (w[i] + sum) / (1 + neighbours[i].len()) as f64
                    })
                    .collect();
                w = next;
            }
            let mut members = vec![];
            for (i, &wi) in w.iter().enumerate() {
                if wi > 1e-3 {
                    members.push((i, wi, !no_data.get(i).copied().unwrap_or(false)));
                    weight[i] = weight[i].max(wi);
                }
            }
            // Bone axes from the vertices each bone dominates: the principal
            // axis of a phalanx is the finger's; the palm's axis runs from
            // the wrist to the knuckles.
            let mut bones: Vec<Bone> = vec![];
            let bone_index = |bones: &Vec<Bone>, j: usize| bones.iter().position(|b| b.joint == j);
            let mut owned: Vec<Vec<V>> = vec![];
            let mut owned_cut: Vec<usize> = vec![];
            for (i, row) in influences.iter().enumerate() {
                for inf in row.iter() {
                    let Some((bs, g)) = groups[inf.joint] else {
                        continue;
                    };
                    if bs != s || inf.weight < 0.5 {
                        continue;
                    }
                    let k = match bone_index(&bones, inf.joint) {
                        Some(k) => k,
                        None => {
                            bones.push(Bone {
                                joint: inf.joint,
                                group: g,
                                center: V::zeros(),
                                axis: V::zeros(),
                                cut: false,
                            });
                            owned.push(vec![]);
                            owned_cut.push(0);
                            bones.len() - 1
                        }
                    };
                    owned[k].push(rest[i]);
                    if no_data.get(i).copied().unwrap_or(false) {
                        owned_cut[k] += 1;
                    }
                }
            }
            let centroid = |ps: &[V]| ps.iter().fold(V::zeros(), |s, p| s + p) / ps.len() as f64;
            let palm_axis = {
                let of = |pred: &dyn Fn(&Bone) -> bool| -> Option<V> {
                    let ps: Vec<V> = bones
                        .iter()
                        .zip(&owned)
                        .filter(|(b, _)| pred(b))
                        .flat_map(|(_, o)| o.iter().copied())
                        .collect();
                    (!ps.is_empty()).then(|| centroid(&ps))
                };
                match (of(&|b| b.group == 0), of(&|b| b.group >= 2)) {
                    (Some(p), Some(f)) => (f - p).try_normalize(1e-9),
                    _ => None,
                }
            };
            for ((b, ps), &c) in bones.iter_mut().zip(&owned).zip(&owned_cut) {
                b.cut = b.group >= 2 && 2 * c > ps.len();
                if ps.len() < 6 {
                    continue;
                }
                b.center = centroid(ps);
                b.axis = if b.group == 0 {
                    palm_axis.unwrap_or_else(V::zeros)
                } else {
                    let cov = ps.iter().fold(Matrix3::zeros(), |m, p| {
                        let d = p - b.center;
                        m + d * d.transpose()
                    });
                    let e = cov.symmetric_eigen();
                    let k = e.eigenvalues.imax();
                    e.eigenvectors.column(k).into_owned()
                };
            }
             
            let wrist_ratio = palm_axis.and_then(|ax| {
                let ring = |pick: &dyn Fn(usize) -> bool| -> Option<f64> {
                    let ps: Vec<V> = (0..n).filter(|&i| pick(i)).map(|i| rest[i]).collect();
                    if ps.len() < 8 {
                        return None;
                    }
                    let c = centroid(&ps);
                    Some(ps.iter().map(|p| { let q = p - c; (q - ax * q.dot(&ax)).norm() }).sum::<f64>() / ps.len() as f64)
                };
                let hand = ring(&|i| member[i] && d[i].is_finite() && d[i] < 0.2 * band)?;
                let arm = ring(&|i| !member[i] && d[i].is_finite() && d[i] > band && d[i] < 2.0 * band)?;
                (hand > 1e-6).then(|| arm / hand)
            });
            let radial: Vec<Vec<(usize, V)>> = members
                .iter()
                .map(|&(i, _, _)| {
                    let mut out: Vec<(usize, V)> = vec![];
                    for inf in influences[i].iter() {
                        let Some(b) = bone_index(&bones, inf.joint).map(|k| &bones[k]) else {
                            continue;
                        };
                        if b.axis == V::zeros() {
                            continue;
                        }
                        let d = rest[i] - b.center;
                        let r = (d - b.axis * d.dot(&b.axis)) * inf.weight;
                        let g = if b.cut {
                            CUT_GROUP
                        } else {
                            b.group.min(GROUPS - 1)
                        };
                        match out.iter_mut().find(|(h, _)| *h == g) {
                            Some(e) => e.1 += r,
                            None => out.push((g, r)),
                        }
                    }
                    out
                })
                .collect();
            // Finger curl: knuckle centres from the vertices shared by a
            // phalanx and its parent bone, flexion axes across the finger
            // and the palm's normal, turned so that a positive curl follows
            // MakeHuman's own slight curl (towards the palm).
            let id = |name: String| names.iter().position(|n| n.eq_ignore_ascii_case(&name));
            let suffix = if s == 0 { "L" } else { "R" };
            let bone_center = |j: Option<usize>| {
                j.and_then(|j| bone_index(&bones, j))
                    .map(|k| &bones[k])
                    .filter(|b| b.axis != V::zeros())
            };
            let palm_normal = {
                let ps: Vec<V> = bones
                    .iter()
                    .zip(&owned)
                    .filter(|(b, _)| b.group == 0)
                    .flat_map(|(_, o)| o.iter().copied())
                    .collect();
                (ps.len() >= 6).then(|| {
                    let c = centroid(&ps);
                    let cov = ps.iter().fold(Matrix3::zeros(), |m, p| {
                        let d = p - c;
                        m + d * d.transpose()
                    });
                    let e = cov.symmetric_eigen();
                    e.eigenvectors.column(e.eigenvalues.imin()).into_owned()
                })
            };
             
            let size_ratio = match (palm_axis, scan) {
                (Some(ax), Some((scan, seg))) => (|| {
                    let hand: Vec<V> = (0..n).filter(|&i| member[i] && (d[i].is_nan() || d[i] > band)).map(|i| rest[i]).collect();
                    let ring: Vec<V> = (0..n).filter(|&i| member[i] && d[i] < 0.2 * band).map(|i| rest[i]).collect();
                    if hand.len() < 50 || ring.len() < 8 {
                        return None;
                    }
                    let wc = centroid(&ring);
                    let length = hand.iter().map(|p| (p - wc).norm()).fold(0.0, f64::max);
                     
                    let cand: Vec<bool> = scan
                        .weld
                        .representative
                        .par_iter()
                        .map(|&r| {
                            let p = v(scan.positions[r]);
                            (p - wc).dot(&ax) > -0.2 * length
                                && (p - wc).norm() < 1.6 * length
                                && scan
                                    .query
                                    .closest_point(a(p), 1e-3, None)
                                    .is_some_and(|h| seg.triangle_label[h.source[1]] == Label::Skin)
                        })
                        .collect();
                    let mut adj: Vec<Vec<usize>> = vec![vec![]; cand.len()];
                    for t in &scan.triangles {
                        let w = t.map(|i| scan.weld.id[i as usize]);
                        if w.iter().all(|&k| cand[k]) {
                            for k in 0..3 {
                                adj[w[k]].push(w[(k + 1) % 3]);
                                adj[w[(k + 1) % 3]].push(w[k]);
                            }
                        }
                    }
                    let pos = |k: usize| v(scan.positions[scan.weld.representative[k]]);
                    let mut seeds: Vec<usize> = (0..cand.len()).filter(|&k| cand[k]).collect();
                    seeds.sort_by(|&x, &y| (pos(x) - wc).norm().total_cmp(&(pos(y) - wc).norm()));
                    seeds.truncate(20);
                    let mut seen = vec![false; cand.len()];
                    let mut stack = seeds.clone();
                    for &k in &seeds {
                        seen[k] = true;
                    }
                    while let Some(k) = stack.pop() {
                        for &j in &adj[k] {
                            if !seen[j] {
                                seen[j] = true;
                                stack.push(j);
                            }
                        }
                    }
                    let pts: Vec<V> = (0..cand.len())
                        .filter(|&k| seen[k] && (pos(k) - wc).dot(&ax) > 0.0)
                        .map(pos)
                        .collect();
                    let size = |ps: &[V]| -> Option<f64> {
                        if ps.len() < 50 {
                            return None;
                        }
                        let c = centroid(ps);
                        let cov = ps.iter().fold(Matrix3::zeros(), |m, p| {
                            let q = p - c;
                            m + q * q.transpose()
                        });
                        let e = cov.symmetric_eigen();
                        let mut order = [0usize, 1, 2];
                        order.sort_by(|&x, &y| e.eigenvalues[y].total_cmp(&e.eigenvalues[x]));
                        let spread = |k: usize| {
                            let axis = e.eigenvectors.column(order[k]).into_owned();
                            let mut t: Vec<f64> = ps.iter().map(|p| (p - c).dot(&axis)).collect();
                            t.sort_by(f64::total_cmp);
                            t[t.len() * 95 / 100] - t[t.len() * 5 / 100]
                        };
                        Some(spread(0))
                    };
                    let own = size(&hand)?;
                    (own > 1e-4).then_some(size(&pts)? / own)
                })(),
                _ => None,
            };
            let mut knuckles = [[None; 4]; CURLED];
            let mut flexion = [V::zeros(); CURLED];
            for (d, f) in (2..=5usize).enumerate() {
                for (k, slot) in knuckles[d].iter_mut().enumerate().skip(1) {
                    let parent = if k == 1 {
                        id(format!("metacarpal{}.{suffix}", f - 1))
                    } else {
                        id(format!("finger{f}-{}.{suffix}", k - 1))
                    };
                    let (Some(parent), Some(child)) =
                        (parent, id(format!("finger{f}-{k}.{suffix}")))
                    else {
                        continue;
                    };
                    let shared: Vec<V> = influences
                        .iter()
                        .enumerate()
                        .filter_map(|(i, row)| {
                            let w = |j: usize| {
                                row.iter()
                                    .filter(|x| x.joint == j)
                                    .map(|x| x.weight)
                                    .sum::<f64>()
                            };
                            // A knuckle of the palm may share its weight
                            // over neighbouring metacarpals.
                            let wp = if k == 1 {
                                row.iter()
                                    .filter(|x| groups[x.joint].is_some_and(|g| g == (s, 0)))
                                    .map(|x| x.weight)
                                    .sum::<f64>()
                            } else {
                                w(parent)
                            };
                            let wc = w(child);
                            ((0.3..=0.7).contains(&wc) && wc + wp > 0.8).then_some(rest[i])
                        })
                        .collect();
                    if shared.len() >= 3 {
                        *slot = Some(centroid(&shared));
                    }
                }
                let (Some(first), Some(last), Some(normal), Some(base)) = (
                    bone_center(id(format!("finger{f}-1.{suffix}"))),
                    bone_center(id(format!("finger{f}-3.{suffix}"))),
                    palm_normal,
                    knuckles[d][1],
                ) else {
                    continue;
                };
                let reach = last.center - base;
                let along = if first.axis.dot(&reach) < 0.0 {
                    -first.axis
                } else {
                    first.axis
                };
                let Some(mut axis) = along.cross(&normal).try_normalize(1e-9) else {
                    continue;
                };
                let bend = reach - along * reach.dot(&along);
                if axis.cross(&reach).dot(&bend) < 0.0 {
                    axis = -axis;
                }
                flexion[d] = axis;
            }
            let curled: Vec<Vec<(usize, usize, f64)>> = members
                .iter()
                .map(|&(i, _, _)| {
                    influences[i]
                        .iter()
                        .filter_map(|inf| {
                            let (bs, g) = groups[inf.joint]?;
                            let phalanx = phalanx_of(names[inf.joint])?;
                            (bs == s && g >= 2 && flexion[g - 2] != V::zeros()).then_some((
                                g - 2,
                                phalanx,
                                inf.weight,
                            ))
                        })
                        .collect()
                })
                .collect();
            let k = mapped[s as usize] as usize;
            let share = |lo: f64, hi: f64| {
                (
                    1.0 + (scale_range.0 - 1.0) * lo,
                    1.0 + (scale_range.1 - 1.0) * hi,
                )
            };
            let bounds: (Params, Params) = {
                let girth = share(GIRTH_SHARE.0, GIRTH_SHARE.1[k]);
                let curl = if mapped[s as usize] { (0.0, 0.0) } else { CURL };
                (
                    std::array::from_fn(|j| if j < GROUPS { girth.0 } else { curl.0 }),
                    std::array::from_fn(|j| if j < GROUPS { girth.1 } else { curl.1 }),
                )
            };
             
            let mut prior = NEUTRAL;
            let mut hold = [0.01; PARAMS];
            let mut measured = [None; 5];
            if mapped[s as usize] {
                for (f, m) in (1..=5usize).zip(measured.iter_mut()) {
                    // Phalanx after the mark: the distal one at the tip, the
                    // one after the middle knuckle (the thumb's distal) there.
                    let at = [3, if f == 1 { 3 } else { 2 }];
                    for (scan_r, k) in tip_radius[s as usize][f - 1].iter().zip(at) {
                        let (Some(scan_r), Some(j)) =
                            (*scan_r, id(format!("finger{f}-{k}.{suffix}")))
                        else {
                            continue;
                        };
                        let Some(kb) = bone_index(&bones, j) else {
                            continue;
                        };
                        let b = &bones[kb];
                        if b.axis == V::zeros() {
                            continue;
                        }
                        let mut r: Vec<f64> = owned[kb]
                            .iter()
                            .map(|p| {
                                let d = p - b.center;
                                (d - b.axis * d.dot(&b.axis)).norm()
                            })
                            .collect();
                        r.sort_by(f64::total_cmp);
                        let own = r[r.len() / 2];
                        let ratio = scan_r / own.max(1e-6);
                        if (RADIUS_RATIO.0..=RADIUS_RATIO.1).contains(&ratio) {
                            *m = Some(m.map_or(ratio, |x: f64| x.max(ratio)));
                        }
                        if m.is_some_and(|x| x >= 1.0) {
                            break;
                        }
                    }
                }
                let mut fingers: Vec<f64> = measured[1..].iter().flatten().copied().collect();
                fingers.sort_by(f64::total_cmp);
                 
                if let Some(&mid) = fingers.get(fingers.len() / 2) {
                    let thicken = |r: f64, j: usize| r.max(1.0).clamp(bounds.0[j], bounds.1[j]);
                    prior[2] = thicken(mid, 2);
                    hold[2] = 1.0;
                    prior[1] = thicken(measured[0].unwrap_or(mid), 1);
                    hold[1] = 1.0;
                }
            }
            // Grown forearms: the palm takes their ratio and the digits most
            // of it, unless a finger measure thickens them more.
            let grown = wrist_ratio.filter(|k| *k > WRIST_GROWN);
            if let Some(k) = grown {
                let fit = |r: f64, j: usize| r.clamp(bounds.0[j], bounds.1[j]);
                let digits = 1.0 + (k - 1.0) * DIGIT_SHARE;
                prior[0] = fit(prior[0].max(k), 0);
                hold[0] = 1.0;
                for j in [1, 2] {
                    prior[j] = fit(prior[j].max(digits), j);
                    hold[j] = 1.0;
                }
            }
            // Surface fit: a mapped hand with all five digits (no cut).
            let surface = mapped[s as usize] && !bones.iter().any(|b| b.cut);
            let digit: Vec<usize> = members
                .iter()
                .map(|&(i, _, _)| {
                    influences[i]
                        .iter()
                        .filter_map(|inf| groups[inf.joint].filter(|(bs, _)| *bs == s).map(|(_, g)| (g, inf.weight)))
                        .fold(std::collections::BTreeMap::<usize, f64>::new(), |mut m, (g, w)| {
                            *m.entry(g).or_default() += w;
                            m
                        })
                        .into_iter()
                        .filter(|(_, w)| *w > 0.5)
                        .max_by(|a, b| a.1.total_cmp(&b.1))
                        .map_or(usize::MAX, |(g, _)| g)
                })
                .collect();
            let segments: Vec<(usize, V, V)> = bones
                .iter()
                .zip(&owned)
                .filter(|(b, ps)| b.axis != V::zeros() && ps.len() >= 6)
                .map(|(b, ps)| {
                    let (lo, hi) = ps.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |e, p| {
                        let t = (p - b.center).dot(&b.axis);
                        (e.0.min(t), e.1.max(t))
                    });
                    (b.group, b.center + b.axis * lo, b.center + b.axis * hi)
                })
                .collect();
            let local: std::collections::HashMap<usize, usize> =
                members.iter().enumerate().map(|(k, &(i, _, _))| (i, k)).collect();
            let nbr: Vec<Vec<usize>> = members
                .iter()
                .map(|&(i, _, _)| neighbours[i].iter().filter_map(|&j| local.get(&(j as usize)).copied()).collect())
                .collect();
            hands.push(Hand {
                surface,
                digit,
                segments,
                nbr,
                surface_residual: 0.0,
                surface_matched: 0.0,
                side: name,
                mapped: mapped[s as usize],
                wrist_ratio,
                size_ratio,
                members,
                radial,
                curled,
                knuckles,
                flexion,
                vertices,
                 
                scale_range: if full_scale { scale_range } else { share(SCALE_SHARE[k], SCALE_SHARE[k]) },
                bounds,
                prior,
                hold,
                measured,
                max_rotation: ROTATION[k],
                params: prior,
                scale: 1.0,
                rotation: Matrix3::identity(),
                raw_rotation: 0.0,
                raw_scale: 1.0,
                shift: V::zeros(),
                residual_current: 0.0,
                residual_target: 0.0,
                matched: 0.0,
            });
        }
        Self { weight, hands }
    }

    /// Hand targets for this iteration, as `(vertex, target)`: each hand's
    /// posed prior with its girths, moved by the bounded similarity that
    /// best fits the current hand with its vertices moved onto the scan
    /// hand; the girths are then updated against the same data.
    #[allow(clippy::too_many_arguments)]
    pub fn targets(
        &mut self,
        rest: &[V],
        x: &[V],
        normals: &[V],
        scan: &Scan,
        seg: &Segmentation,
        clearance: f64,
        height: f64,
    ) -> Vec<(usize, V)> {
        let (reach, soft) = (REACH * height, SOFT * height);
        // Move of a vertex onto the scan: along the scan normal onto skin,
        // back inside (by the clearance) past cloth; with the distance.
        let delta = |p: V, nrm: V| -> Option<(V, f64)> {
            let hit = scan.query.closest_point(a(p), reach, Some((a(nrm), 0.3)))?;
            let nq = v(hit.normal);
            let along = (v(hit.point) - p).dot(&nq);
            if seg.triangle_label[hit.source[1]] == Label::Skin {
                Some((nq * along, along.abs()))
            } else if seg.is_prop(hit.source[1]) {
                // A bow or quiver next to the hand is no sleeve to stay in.
                None
            } else {
                let out = -along;
                (out > -clearance).then(|| (-nq * (out + clearance), out.max(0.0)))
            }
        };
        let mut out = vec![];
        for hand in &mut self.hands {
            let moves: Vec<Option<(V, f64)>> = hand
                .members
                .par_iter()
                .map(|&(i, _, data)| if data { delta(x[i], normals[i]) } else { None })
                .collect();
            let core = |w: f64| w > 0.5;
            let shaped: Vec<V> = (0..hand.members.len())
                .into_par_iter()
                .map(|k| hand.model(rest, k, &hand.params))
                .collect();
            let (mut sum, mut count) = (0.0, 0usize);
            // (member, wanted position, data weight)
            let mut wanted = Vec::with_capacity(hand.members.len());
            let mut pairs = Vec::with_capacity(hand.members.len());
            for (k, (&(i, w, data), m)) in hand.members.iter().zip(&moves).enumerate() {
                if !data {
                    continue;
                }
                let (d, dist) = m.unwrap_or((V::zeros(), 0.0));
                if core(w) && m.is_some() {
                    sum += dist;
                    count += 1;
                }
                // A far match (a web between fused fingers) counts less.
                let robust = 1.0 / (1.0 + (dist / soft).powi(2));
                if m.is_some() {
                    wanted.push((k, x[i] + d, w * robust));
                }
                pairs.push((
                    shaped[k],
                    x[i] + d,
                    w * robust + ANCHOR * 4.0 * w * (1.0 - w),
                ));
            }
            let core_count = hand.members.iter().filter(|m| core(m.1) && m.2).count();
            hand.matched = count as f64 / core_count.max(1) as f64;
            hand.residual_current = sum / count.max(1) as f64;
             
            let floor = hand
                .size_ratio
                .filter(|k| !hand.mapped && *k > WRIST_GROWN)
                .map_or(0.0, |k| k.min(hand.scale_range.1));
            let range = (hand.scale_range.0.max(floor), hand.scale_range.1.max(floor));
            let Some(sim) = similarity(&pairs, range, hand.max_rotation) else {
                continue;
            };
            hand.scale = sim.scale;
            hand.rotation = sim.rotation;
            hand.raw_rotation = sim.raw_angle;
            hand.raw_scale = sim.raw_scale;
            hand.shift = sim.shift;
            let mut targets: Vec<(usize, V)> = hand
                .members
                .iter()
                .zip(&shaped)
                .map(|(&(i, _, _), p)| (i, sim.rotation * p * sim.scale + sim.translation))
                .collect();
            if hand.surface {
                // The bones where the similarity put them; a scan point's
                // digit is its nearest bone's.
                let segs: Vec<(usize, V, V)> = hand
                    .segments
                    .iter()
                    .map(|&(g, p, q)| {
                        let m = |u: V| sim.rotation * u * sim.scale + sim.translation;
                        (g, m(p), m(q))
                    })
                    .collect();
                let digit_at = |q: V| -> usize {
                    segs.iter()
                        .map(|&(g, a0, b0)| {
                            let ab = b0 - a0;
                            let t = ((q - a0).dot(&ab) / ab.norm_squared().max(1e-12)).clamp(0.0, 1.0);
                            (g, (q - (a0 + ab * t)).norm())
                        })
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .map_or(usize::MAX, |(g, _)| g)
                };
                let cap = SURFACE_CAP * height;
                let raw: Vec<Option<V>> = targets
                    .par_iter()
                    .zip(&hand.members)
                    .zip(&hand.digit)
                    .map(|((&(i, q), &(_, w, data)), &g)| {
                        if !data || g == usize::MAX || w < 0.5 {
                            return None;
                        }
                        let hit = scan.query.closest_point(a(q), reach, Some((a(normals[i]), 0.3)))?;
                        if seg.triangle_label[hit.source[1]] != Label::Skin || digit_at(v(hit.point)) != g {
                            return None;
                        }
                        let r = v(hit.point) - q;
                        Some(if r.norm() > cap { r * (cap / r.norm()) } else { r })
                    })
                    .collect();
                let matched = raw.iter().filter(|r| r.is_some()).count();
                hand.surface_matched = matched as f64 / raw.len().max(1) as f64;
                hand.surface_residual = raw.iter().flatten().map(|r| r.norm()).sum::<f64>() / matched.max(1) as f64;
                // Smoothed over the hand: matched vertices hold their move
                // (twice a neighbour's say), the others take their
                // neighbours'.
                let mut f: Vec<V> = raw.iter().map(|r| r.unwrap_or_else(V::zeros)).collect();
                for _ in 0..SURFACE_SMOOTHING {
                    f = (0..f.len())
                        .map(|k| {
                            let own = if raw[k].is_some() { 2.0 } else { 0.0 };
                            let sum = hand.nbr[k].iter().fold(raw[k].map_or(V::zeros(), |r| r * own), |acc, &j| acc + f[j]);
                            let n = own + hand.nbr[k].len() as f64;
                            if n > 0.0 { sum / n } else { f[k] }
                        })
                        .collect();
                }
                for ((t, &(_, w, _)), d) in targets.iter_mut().zip(&hand.members).zip(&f) {
                    t.1 += d * guard::smoothstep((w - 0.5) * 2.0);
                }
            }
            let after: Vec<f64> = targets
                .par_iter()
                .zip(&hand.members)
                .filter(|(_, m)| core(m.1) && m.2)
                .filter_map(|((i, q), _)| delta(*q, normals[*i]).map(|d| d.1))
                .collect();
            hand.residual_target = after.iter().sum::<f64>() / after.len().max(1) as f64;
            // Girths and curls against the same data, in the posed hand's
            // frame.
            let back = |q: V| sim.rotation.transpose() * (q - sim.translation) / sim.scale;
            let rows: Vec<(usize, V, f64)> =
                wanted.iter().map(|&(k, q, c)| (k, back(q), c)).collect();
            let h: &Hand = hand;
            let step = gauss_newton(
                |k, p| h.model(rest, k, p),
                &rows,
                &hand.params,
                &hand.bounds,
                (&hand.prior, &hand.hold),
            );
            for (p, q) in hand.params.iter_mut().zip(step) {
                *p = 0.5 * (*p + q);
            }
            // One hand: the thumb stays in proportion to the fingers (a
            // strap or quiver beside the thumb read as a 2.2× thumb).
            let fingers = hand.params[2];
            hand.params[1] = hand.params[1].clamp(fingers * 0.8, fingers * 1.25);
            out.extend(targets);
        }
        // A pair of hands is one pair, like the eyes: unmarked hands share
        // their girths (the mean); asymmetric evidence is noise, not anatomy.
        if let [l, r] = self.hands.as_mut_slice()
            && !l.mapped
            && !r.mapped
        {
            for g in 0..GROUPS {
                let m = 0.5 * (l.params[g] + r.params[g]);
                l.params[g] = m;
                r.params[g] = m;
            }
        }
        out
    }

    pub fn report(&self) -> Value {
        json!(
            self.hands
                .iter()
                .map(|h| json!({
                    "side": h.side,
                    "mapped": h.mapped,
                    "surface_fit": h.surface.then(|| json!({"mean_move_m": h.surface_residual, "matched": h.surface_matched})),
                    "vertices": h.vertices,
                    "target_vertices": h.members.len(),
                    "scale": h.scale,
                    "scale_range": [h.scale_range.0, h.scale_range.1],
                    "unclamped_scale": h.raw_scale,
                    "rotation_rad": ((h.rotation.trace() - 1.0) / 2.0).clamp(-1.0, 1.0).acos(),
                    "unclamped_rotation_rad": h.raw_rotation,
                    "max_rotation_rad": h.max_rotation,
                    "centroid_shift_m": h.shift.norm(),
                    "girth": {"palm": h.params[0], "thumb": h.params[1], "fingers": h.params[2], "range": [h.bounds.0[0], h.bounds.1[0]], "measured_tip_radius_ratio": h.measured, "held_towards": &h.prior[..GROUPS], "wrist_ratio": h.wrist_ratio, "size_ratio": h.size_ratio},
                    "curl_rad": {"fingers_2_to_5": &h.params[GROUPS..], "range": [h.bounds.0[GROUPS], h.bounds.1[GROUPS]], "knuckles": h.knuckles.iter().map(|k| k.iter().flatten().count()).collect::<Vec<_>>()},
                    "matched_fraction": h.matched,
                    "mean_scan_distance_m": {"current": h.residual_current, "target": h.residual_target},
                }))
                .collect::<Vec<_>>()
        )
    }
}

/// One damped Gauss-Newton step of the hand shape `p` towards the wanted
/// positions `rows` (member, position, weight) of `model`, held towards
/// `prior` by `hold` (a share of each parameter's own evidence) and clamped
/// to `bounds`; numeric derivatives.
fn gauss_newton(
    model: impl Fn(usize, &Params) -> V + Sync,
    rows: &[(usize, V, f64)],
    p: &Params,
    bounds: &(Params, Params),
    (prior, hold): (&Params, &Params),
) -> Params {
    const EPS: f64 = 1e-4;
    let free: Vec<usize> = (0..PARAMS).filter(|&j| bounds.0[j] < bounds.1[j]).collect();
    let (m, b) = rows
        .par_iter()
        .map(|&(k, y, c)| {
            let at = model(k, p);
            let cols: Vec<V> = free
                .iter()
                .map(|&j| {
                    let mut q = *p;
                    q[j] += EPS;
                    (model(k, &q) - at) / EPS
                })
                .collect();
            let mut m = SMatrix::<f64, PARAMS, PARAMS>::zeros();
            let mut b = SVector::<f64, PARAMS>::zeros();
            for (u, &j) in free.iter().enumerate() {
                b[j] += cols[u].dot(&(y - at)) * c;
                for (w, &l) in free.iter().enumerate() {
                    m[(j, l)] += cols[u].dot(&cols[w]) * c;
                }
            }
            (m, b)
        })
        .collect::<Vec<_>>()
        .into_iter()
        // In row order (`geom::ordered_sum`): a parallel reduce made
        // identical runs differ.
        .fold((SMatrix::zeros(), SVector::zeros()), |x, y| (x.0 + y.0, x.1 + y.1));
    let (mut m, mut b) = (m, b);
    // Each parameter is held by a share of its own evidence, one without
    // evidence by a floor.
    let floor = (0..PARAMS).map(|j| m[(j, j)]).fold(0.0, f64::max) * 1e-4 + 1e-12;
    for j in 0..PARAMS {
        let h = m[(j, j)] * hold[j] + floor;
        m[(j, j)] += h;
        b[j] += h * (prior[j] - p[j]);
    }
    let d = m.cholesky().map_or(SVector::zeros(), |c| c.solve(&b));
    std::array::from_fn(|j| (p[j] + d[j]).clamp(bounds.0[j], bounds.1[j]))
}

struct Similarity {
    scale: f64,
    rotation: Matrix3<f64>,
    translation: V,
    shift: V,
    raw_angle: f64,
    raw_scale: f64,
}

/// Weighted similarity (Umeyama) taking each pair's first point onto its
/// second, with the rotation angle limited to `max_angle` and the uniform
/// scale clamped to `range`, so `q ≈ scale · R · p + t`.
fn similarity(pairs: &[(V, V, f64)], range: (f64, f64), max_angle: f64) -> Option<Similarity> {
    let ws: f64 = pairs.iter().map(|p| p.2).sum();
    if ws <= 1e-12 {
        return None;
    }
    let ms = pairs.iter().fold(V::zeros(), |acc, p| acc + p.0 * p.2) / ws;
    let md = pairs.iter().fold(V::zeros(), |acc, p| acc + p.1 * p.2) / ws;
    let (mut cov, mut var) = (Matrix3::zeros(), 0.0);
    for (src, dst, w) in pairs {
        let s = src - ms;
        cov += (dst - md) * s.transpose() * *w;
        var += s.norm_squared() * w;
    }
    let full = proper_rotation(cov);
    let (r, raw_angle) = match Rotation3::from_matrix_unchecked(full).axis_angle() {
        Some((axis, angle)) => (
            Rotation3::from_axis_angle(&axis, angle.min(max_angle)).into_inner(),
            angle,
        ),
        None => (Matrix3::identity(), 0.0),
    };
    let raw_scale = (r.transpose() * cov).trace() / var.max(1e-20);
    let s = raw_scale.clamp(range.0, range.1);
    Some(Similarity {
        scale: s,
        rotation: r,
        translation: md - r * ms * s,
        shift: md - ms,
        raw_angle,
        raw_scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points() -> Vec<V> {
        vec![
            V::new(0.0, 0.0, 0.0),
            V::new(0.08, 0.0, 0.01),
            V::new(0.0, 0.03, 0.0),
            V::new(0.02, 0.01, 0.1),
            V::new(-0.03, 0.02, 0.05),
        ]
    }

    #[test]
    fn similarity_recovers_a_small_turn_and_bounds_a_large_one() {
        let r = Rotation3::from_euler_angles(0.05, -0.1, 0.08).into_inner();
        let t = V::new(0.01, -0.02, 0.005);
        let pairs: Vec<(V, V, f64)> = points()
            .iter()
            .map(|p| (*p, r * p * 1.05 + t, 1.0))
            .collect();
        let s = similarity(&pairs, (0.9, 1.1), 0.3).unwrap();
        assert!((s.scale - 1.05).abs() < 1e-9 && (s.rotation - r).norm() < 1e-9);
        assert!((s.translation - t).norm() < 1e-9);
        // A hand turned 1 rad and twice the size: the turn and the growth
        // stop at their bounds, about the same axis.
        let big = Rotation3::from_axis_angle(&V::z_axis(), 1.0).into_inner();
        let pairs: Vec<(V, V, f64)> = points().iter().map(|p| (*p, big * p * 2.0, 1.0)).collect();
        let s = similarity(&pairs, (0.9, 1.1), 0.3).unwrap();
        assert!((s.raw_angle - 1.0).abs() < 1e-9 && s.scale == 1.1);
        let want = Rotation3::from_axis_angle(&V::z_axis(), 0.3).into_inner();
        assert!((s.rotation - want).norm() < 1e-9);
        assert!(similarity(&[], (0.9, 1.1), 0.3).is_none());
    }

    /// A girth on a ring (30 % thicker wanted) and a curl about a knuckle
    /// (0.5 rad wanted, past its bound of 0.4): recovered within bounds by
    /// repeated steps, the unused parameters held at MakeHuman's hand.
    #[test]
    fn gauss_newton_fits_girth_and_curl_within_bounds() {
        let ring: Vec<V> = (0..8)
            .map(|k| {
                let phi = k as f64 * 0.785;
                V::new(phi.cos(), phi.sin(), 0.0) * 0.008
            })
            .collect();
        let tip: Vec<V> = (1..6).map(|k| V::new(0.0, 0.0, 0.01 * k as f64)).collect();
        let axis = V::x();
        let model = |k: usize, p: &Params| -> V {
            if k < ring.len() {
                ring[k] * p[1]
            } else {
                Rotation3::from_axis_angle(&Unit::new_unchecked(axis), p[GROUPS])
                    * tip[k - ring.len()]
            }
        };
        let want = {
            let mut p = NEUTRAL;
            p[1] = 1.3;
            p[GROUPS] = 0.5;
            p
        };
        let rows: Vec<(usize, V, f64)> = (0..ring.len() + tip.len())
            .map(|k| (k, model(k, &want), 1.0))
            .collect();
        let mut bounds = ([0.8; PARAMS], [1.5; PARAMS]);
        bounds.0[GROUPS..].fill(-0.1);
        bounds.1[GROUPS..].fill(0.4);
        let mut p = NEUTRAL;
        for _ in 0..20 {
            p = gauss_newton(model, &rows, &p, &bounds, (&NEUTRAL, &[0.01; PARAMS]));
        }
        assert!((p[1] - 1.3).abs() < 0.01, "{p:?}");
        assert!((p[GROUPS] - 0.4).abs() < 1e-9, "{p:?}");
        assert!((p[0] - 1.0).abs() < 1e-9 && p[GROUPS + 1] == 0.0);
        assert_eq!(phalanx_of("finger3-2.L"), Some(2));
        assert_eq!(phalanx_of("metacarpal2.L"), None);
    }

    #[test]
    fn hand_bones_group_by_digit_and_side() {
        assert_eq!(group_of("wrist.L"), Some((0, 0)));
        assert_eq!(group_of("metacarpal3.R"), Some((1, 0)));
        assert_eq!(group_of("finger1-2.R"), Some((1, 1)));
        assert_eq!(group_of("finger5-3.L"), Some((0, 5)));
        assert_eq!(group_of("lowerarm02.L"), None);
        assert_eq!(group_of("finger2-1"), None);
    }

    /// A strip of vertices along +X, the "hand" from x > 0: full weight in
    /// the hand, none well up the "forearm", a monotone fade between.
    #[test]
    fn wrist_fade_is_monotone_across_the_boundary() {
        let n: usize = 61;
        let shape: Vec<V> = (0..n)
            .map(|i| V::new((i as f64 - 30.0) * 0.002, 0.0, 0.0))
            .collect();
        let neighbours: Vec<Vec<u32>> = (0..n)
            .map(|i| {
                [i.wrapping_sub(1), i + 1]
                    .into_iter()
                    .filter(|&j| j < n)
                    .map(|j| j as u32)
                    .collect()
            })
            .collect();
        let side: Vec<Option<u8>> = (0..n).map(|i| (i > 30).then_some(0)).collect();
        let rows: Vec<&[Influence]> = vec![&[]; n];
        let hands = Hands::detect(
            &side,
            &vec![false; n],
            &rows,
            &[],
            &shape,
            &shape,
            &neighbours,
            1.75,
            (0.8, 1.25),
            [false, false],
            &[[[None; 2]; 5]; 2],
            false,
            None,
        );
        let w = &hands.weight;
        assert!(w[0] == 0.0 && w[n - 1] > 0.999);
        assert!(w.windows(2).all(|p| p[1] >= p[0] - 1e-12));
        assert!(w[30] > 0.2 && w[30] < 0.8);
    }
}
