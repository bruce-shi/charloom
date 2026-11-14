 use crate::fit::Rig;
use crate::garment::Garment;
use crate::geom::{V, Weld, a, v, vertex_normals};
use crate::head_keep::{
    Bridge, Frame, HeadPart, KeptHead, Ring, Split, boundary_loops, colour_at_uv, end_tangent,
    erode_bowties, finish, hermite, keep_large_components, mix, neck_loop, plane_offset, smoothstep,
    zipper,
};
use crate::prior::Prior;
use crate::scan::Scan;
use crate::segment::{Label, Segmentation};
use anyhow::{Context, Result, bail, ensure};
use charloom_core::{Influence, Vec3};
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HandRoute {
    /// MakeHuman's hand, mapped onto the hand's landmarks and fitted to the
    /// scan digit by digit.
    #[default]
    Fit,
    /// The scan's own hand, skinned to MakeHuman's hand bones.
    KeepScan,
}

pub struct Input<'a> {
    pub prior: &'a Prior,
    pub rig: &'a Rig,
    pub scan: &'a Scan,
    pub seg: &'a Segmentation,
    pub height: f64,
    /// 0 left, 1 right.
    pub side: u8,
    /// The hand's digit count (thumb counted).
    pub digits: u8,
         pub measured: usize,
}

/// Share of the height: the body's cut up the forearm from the wrist joint,
/// the gap to the scan's cut, the weight feather beyond it.
/// Most the forearm grows to meet a kept hand's wrist (ratio of the scan's
/// wrist ring to MakeHuman's at the cut).
const WRIST_GROWTH_MAX: f64 = 2.0;
/// Length of that growth's fade, in body ring radii, without an elbow joint.
const WRIST_GROWTH_BAND: f64 = 2.5;
/// A garment vertex this near a scan wrist-ring vertex (fraction of the
/// height) touches the ring.
const CUFF_TOUCH: f64 = 0.006;
/// Share of the scan's wrist ring touched by a garment that makes it a
/// cuff's rim: no forearm growth.
const CUFF_RING_SHARE: f64 = 0.25;
const BODY_CUT: f64 = 0.008;
const GAP: f64 = 0.012;
const FEATHER: f64 = 0.02;
/// Blur of the kept hand's weights over its surface (share of the height).
const BLUR: f64 = 0.012;

fn origin(m: &nalgebra::Matrix4<f64>) -> V {
    V::new(m[(0, 3)], m[(1, 3)], m[(2, 3)])
}

impl KeptHead {
    /// MakeHuman's whole fitted body as a kept body with no part cut yet
    /// (the head fitted, hands to be kept).
    pub fn from_body(prior: &Prior, rig: &Rig) -> Self {
        let body = prior.body();
        let positions = rig.meshes[prior.body].clone();
        let weld = Weld::new(&body.bind, 1e-7);
        let normals = vertex_normals(&positions, &body.prim.triangles, &weld);
        let n = body.prim.triangles.len();
        Self {
            body_positions: positions,
            body_normals: normals,
            body_uvs: body.prim.uvs.clone(),
            body_triangles: body.prim.triangles.clone(),
            body_influences: body.influences.clone(),
            body_triangle_origin: (0..n).collect(),
            body_near_seam: vec![false; n],
            heads: vec![],
            bridges: vec![],
            head: false,
            report: Value::Null,
        }
    }
}

 pub fn place_bones(prior: &Prior, rig: &mut Rig, side: u8, digits: u8, points: &BTreeMap<String, Vec3>) -> Value {
    let suffix = if side == 0 { "L" } else { "R" };
    let tag = if side == 0 { "l" } else { "r" };
    let skeleton = prior.skeleton();
    let kept_digits = crate::creature::kept(digits);
    let mut changed: Vec<usize> = vec![];
    let mut moved = vec![];
    for (k, &f) in kept_digits.iter().enumerate() {
        let get = |part: &str| points.get(&format!("finger{}_{tag}_{part}", k + 1)).map(|p| v(*p));
        let (Some(base), Some(mid), Some(tip)) = (get("base"), get("mid"), get("tip")) else {
            continue;
        };
        let names: Vec<String> = if f == 1 {
            (1..=3).map(|i| format!("finger1-{i}.{suffix}")).collect()
        } else {
            std::iter::once(format!("metacarpal{}.{suffix}", f - 1))
                .chain((1..=3).map(|i| format!("finger{f}-{i}.{suffix}")))
                .collect()
        };
        let Some(bones) = names.iter().map(|n| prior.by_name.get(n).copied()).collect::<Option<Vec<usize>>>() else {
            continue;
        };
         
        let (l1, l2) = ((mid - base).norm(), (tip - mid).norm());
        let at = |t: f64| {
            let d = t * (l1 + l2);
            if d <= l1 { base + (mid - base) * (d / l1.max(1e-12)) } else { mid + (tip - mid) * ((d - l1) / l2.max(1e-12)) }
        };
        let head0 = origin(&rig.worlds[bones[0]]);
        let heads: Vec<V> = if f == 1 { vec![head0, base, at(0.5)] } else { vec![head0, base, at(0.45), at(0.75)] };
        for (i, &j) in bones.iter().enumerate() {
            let tail = heads.get(i + 1).copied().unwrap_or(tip);
            let Some(dir) = (tail - heads[i]).try_normalize(1e-9) else {
                continue;
            };
            let w = rig.worlds[j];
            let r_old: nalgebra::Matrix3<f64> = w.fixed_view::<3, 3>(0, 0).into_owned();
            let y_old = (r_old * V::y()).normalize();
            let turn = nalgebra::Rotation3::rotation_between(&y_old, &dir)
                .unwrap_or_else(|| nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(r_old * V::x()), std::f64::consts::PI));
            let r_new = turn.matrix() * r_old;
            let mut m = nalgebra::Matrix4::identity();
            m.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_new);
            m.fixed_view_mut::<3, 1>(0, 3).copy_from(&heads[i]);
            moved.push((origin(&w) - heads[i]).norm());
            rig.worlds[j] = m;
            changed.push(j);
        }
    }
    let affected: Vec<usize> = (0..skeleton.joints.len())
        .filter(|&j| changed.contains(&j) || skeleton.joints[j].parent.is_some_and(|p| changed.contains(&p)))
        .collect();
    for j in affected {
        let parent = skeleton.joints[j].parent.map_or(nalgebra::Matrix4::identity(), |p| rig.worlds[p]);
        let m = parent.try_inverse().unwrap_or_else(nalgebra::Matrix4::identity) * rig.worlds[j];
        let q = nalgebra::UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(
            crate::geom::proper_rotation(m.fixed_view::<3, 3>(0, 0).into_owned()),
        ));
        rig.local[j] = charloom_core::Transform {
            translation: a(origin(&m)),
            rotation: [q.i, q.j, q.k, q.w],
            scale: [1.0; 3],
        };
    }
    moved.sort_by(f64::total_cmp);
    json!({"side": tag, "bones": moved.len(), "moved_m": {"p50": moved.get(moved.len() / 2), "max": moved.last()}})
}

pub fn keep_scan_hand(kept: &mut KeptHead, inp: &Input, garments: &mut [Garment]) -> Result<Value> {
    let Input {
        prior,
        rig,
        scan,
        seg,
        height,
        side,
        digits,
        measured,
    } = *inp;
    let rigid = measured < digits.min(5) as usize;
    let suffix = if side == 0 { "L" } else { "R" };
    let tag = if side == 0 { "l" } else { "r" };
    let skeleton = prior.skeleton();
    let wrist = *prior
        .by_name
        .get(&format!("wrist.{suffix}"))
        .with_context(|| format!("prior has no wrist.{suffix}"))?;
    let forearm = skeleton.joints[wrist].parent.context("wrist without a parent")?;
    let w0 = origin(&rig.worlds[wrist]);
    let axis = (w0 - origin(&rig.worlds[forearm]))
        .try_normalize(1e-9)
        .context("degenerate forearm")?;
    let up = if axis.y.abs() < 0.9 { V::y() } else { V::z() };
    let e1 = (up - axis * axis.dot(&up)).normalize();
    let e2 = axis.cross(&e1);
    let frame = Frame {
        origin: w0,
        axis,
        e1,
        e2,
    };
    // MakeHuman's hand bones of this side that carry a digit the hand has.
    let kept_digits = crate::creature::kept(digits);
    let hand_bone = |j: usize| -> Option<usize> {
        let n = skeleton.joints[j].name.to_lowercase();
        if !n.ends_with(&format!(".{}", tag)) {
            return None;
        }
        if n.starts_with("wrist") || n.starts_with("metacarpal") {
            return Some(0);
        }
        let d = n.strip_prefix("finger")?.chars().next()?.to_digit(10)? as usize;
        kept_digits.contains(&d).then_some(d)
    };
    let bones: Vec<(usize, usize)> = (0..skeleton.joints.len())
        .filter_map(|j| hand_bone(j).map(|g| (j, g)))
        .collect();
    ensure!(bones.len() >= 4, "hand_keep: too few hand bones on side {tag}");
    // Bone segments: joint to its first hand child; an end bone continues
    // its parent's direction for 80 % of the parent's length.
    let segments: Vec<(usize, usize, V, V)> = bones
        .iter()
        .map(|&(j, g)| {
            let p = origin(&rig.worlds[j]);
            let child = bones
                .iter()
                .find(|&&(c, _)| skeleton.joints[c].parent == Some(j))
                .map(|&(c, _)| origin(&rig.worlds[c]));
            let q = child.unwrap_or_else(|| {
                let parent = skeleton.joints[j].parent.map(|pj| origin(&rig.worlds[pj])).unwrap_or(w0);
                p + (p - parent) * 0.8
            });
            (j, g, p, q)
        })
        .collect();
    let seg_dist = |x: V, p: V, q: V| -> f64 {
        let d = q - p;
        let t = ((x - p).dot(&d) / d.norm_squared().max(1e-12)).clamp(0.0, 1.0);
        (x - (p + d * t)).norm()
    };
    // The hand's ball: its kept body vertices (dominant hand bone of this
    // side), their farthest from the wrist, with room for the scan's bigger
    // hand.
    let names: Vec<String> = skeleton.joints.iter().map(|j| j.name.to_lowercase()).collect();
    let dominant = |row: &[Influence]| row.iter().max_by(|x, y| x.weight.total_cmp(&y.weight)).map(|x| x.joint);
    let hand_name = |n: &str| {
        n.ends_with(&format!(".{}", tag)) && (n.starts_with("wrist") || n.starts_with("metacarpal") || n.starts_with("finger"))
    };
     
    let arm_name = |n: &str| hand_name(n) || (n.ends_with(&format!(".{}", tag)) && n.starts_with("lowerarm"));
    let on_arm: Vec<bool> = kept
        .body_influences
        .iter()
        .map(|r| dominant(r).is_some_and(|j| arm_name(&names[j])))
        .collect();
    // The scan side's ball: the posed hand's reach (95th percentile of its
    // vertices' distances from the wrist), with room for a bigger hand.
    let mut dists: Vec<f64> = kept
        .body_positions
        .iter()
        .zip(&kept.body_influences)
        .filter(|(_, r)| dominant(r).is_some_and(|j| hand_name(&names[j])))
        .map(|(p, _)| (v(*p) - w0).norm())
        .collect();
    dists.sort_by(f64::total_cmp);
    let reach = dists.get(dists.len() * 95 / 100).copied().unwrap_or(0.0).max(height * 0.06);
     
    let ball = reach * 3.0;
    let inside = |p: V| (p - w0).norm() < ball;
    let body_cut = -BODY_CUT * height;
    let scan_cut = body_cut + GAP * height;

    // ---- Body side: MakeHuman's hand beyond the body plane goes ----
    let body_weld = Weld::new(&kept.body_positions, 1e-7);
    let tris = kept.body_triangles.clone();
    let bpos = kept.body_positions.clone();
    let bnrm = kept.body_normals.clone();
    let buv = kept.body_uvs.clone();
    let binf = kept.body_influences.clone();
    let bs = Split::new((&bpos, &bnrm, &buv, &tris), &body_weld, &frame, body_cut, |t| {
        tris[t].iter().all(|&i| on_arm[i as usize])
    });
    let identity: Vec<usize> = (0..skeleton.joints.len()).collect();
    let body_infl: Vec<Vec<Influence>> = binf
        .iter()
        .cloned()
        .chain(bs.ends.iter().map(|&(i, j, f)| mix(&[(&binf[i], 1.0 - f), (&binf[j], f)], &identity, 4)))
        .collect();
    let bweld = &bs.weld;
    let removed: Vec<bool> = (0..bweld.count)
        .map(|w| {
            let i = bweld.representative[w];
            let p = v(bs.positions[i]);
            i < bs.first_new && plane_offset(&frame, p, body_cut) > 0.0 && on_arm[i]
        })
        .collect();
    let mut body_keep: Vec<bool> = bs
        .triangles
        .iter()
        .map(|t| !t.iter().any(|&i| removed[bweld.id[i as usize]]))
        .collect();
    let body_eroded = erode_bowties(&bs.triangles, &mut body_keep, bweld);
    keep_large_components(&bs.triangles, &mut body_keep, bweld, 0.05);
    let loops = boundary_loops(&bs.triangles, &body_keep, bweld);
    let bpoint = |w: usize| v(bs.positions[bweld.representative[w]]);
    let loop_note: Vec<String> = loops
        .iter()
        .map(|l| {
            let pts: Vec<V> = l.iter().map(|&w| bpoint(w)).collect();
            let mean = pts.iter().map(|p| frame.axial(*p)).sum::<f64>() / pts.len() as f64;
            format!("{} vertices, winding {:.2}, axial {:.3}", l.len(), crate::head_keep::winding(&frame, &pts), mean)
        })
        .collect();
     
    let pick = loops
        .iter()
        .filter_map(|l| {
            let pts: Vec<V> = l.iter().map(|&w| bpoint(w)).collect();
            let c = pts.iter().sum::<V>() / pts.len() as f64;
            let mean = frame.axial(c);
            (frame.radial(c).norm() < reach && (mean - body_cut).abs() < height * 0.03).then_some(((mean - body_cut).abs(), l.clone(), c))
        })
        .min_by(|x, y| x.0.total_cmp(&y.0));
    let Some((_, body_ring_ids, centre)) = pick else {
        bail!("hand_keep_no_body_ring: no closed MakeHuman wrist boundary (cut {body_cut:.3}; loops: {loop_note:?})");
    };
    // Moved across the axis only: the cuts' offsets along it stay.
    let frame = Frame {
        origin: centre - axis * (centre - w0).dot(&axis),
        ..frame
    };
    let body_ring: HashSet<usize> = body_ring_ids.iter().copied().collect();
    let mut blv: Vec<Vec3> = bs.positions.clone();
    for (i, p) in blv.iter_mut().enumerate() {
        if body_ring.contains(&bweld.id[i]) {
            let d = body_cut - frame.axial(v(*p));
            if d > 0.0 && d < height * 0.015 {
                *p = a(v(*p) + frame.axis * d);
            }
        }
    }
    let bpoint = |w: usize| v(blv[bweld.representative[w]]);
     
    let old_origin = kept.body_triangle_origin.clone();
    let old_near = kept.body_near_seam.clone();
    let mut new_index = vec![u32::MAX; bs.positions.len()];
    let (mut np, mut nn, mut nu, mut nt, mut ni, mut no, mut nnear) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    let mut kept_split = vec![];
    for (t, tri) in bs.triangles.iter().enumerate() {
        if !body_keep[t] {
            continue;
        }
        let tri2 = tri.map(|i| {
            let i = i as usize;
            if new_index[i] == u32::MAX {
                new_index[i] = np.len() as u32;
                np.push(blv[i]);
                nn.push(bs.normals[i]);
                nu.push(bs.uvs[i]);
                ni.push(body_infl[i].clone());
            }
            new_index[i]
        });
        nt.push(tri2);
        no.push(old_origin[bs.origin[t]]);
        nnear.push(old_near[bs.origin[t]] || tri.iter().any(|&i| body_ring.contains(&bweld.id[i as usize])));
        kept_split.push(t);
    }
    let removed_triangles = tris.len() - nt.len();

    // ---- Scan side: the skin beyond the scan plane joined to its ring ----
    let ss = Split::new(
        (&scan.positions, &scan.normals, &scan.uvs, &scan.triangles),
        &scan.weld,
        &frame,
        scan_cut,
        |t| {
            seg.triangle_label[t] == Label::Skin
                && scan.triangles[t].iter().all(|&i| inside(v(scan.positions[i as usize])))
        },
    );
    let sweld = &ss.weld;
    let slabel: Vec<Label> = ss.origin.iter().map(|&t| seg.triangle_label[t]).collect();
    let smaterial: Vec<Option<usize>> = ss.origin.iter().map(|&t| scan.triangle_material[t]).collect();
    let above: Vec<bool> = (0..sweld.count)
        .map(|w| {
            let i = sweld.representative[w];
            let p = v(ss.positions[i]);
            (i >= ss.first_new || plane_offset(&frame, p, scan_cut) > 0.0) && inside(p)
        })
        .collect();
    let mut region: Vec<bool> = ss
        .triangles
        .iter()
        .zip(&slabel)
        .map(|(t, l)| *l == Label::Skin && t.iter().all(|&i| above[sweld.id[i as usize]]))
        .collect();
    // Before the ring's clean-up: the whole skin beyond the cut (islands and
    // bow-ties along a decimated scan's cracks are hand too).
    let full_region = region.clone();
    let scan_eroded = erode_bowties(&ss.triangles, &mut region, sweld);
    let loops = boundary_loops(&ss.triangles, &region, sweld);
    let spoint0 = |w: usize| v(ss.positions[sweld.representative[w]]);
    let scan_ring_ids =
        neck_loop(&frame, loops, spoint0, scan_cut).context("hand_keep_no_scan_ring: no closed scan wrist boundary")?;
    // Only the skin joined to the ring, over the whole skin region except
    // what the clean-up took at the ring itself.
    let ring_set: HashSet<usize> = scan_ring_ids.iter().copied().collect();
    for (t, tri) in ss.triangles.iter().enumerate() {
        if full_region[t] && !region[t] && !tri.iter().any(|&i| ring_set.contains(&sweld.id[i as usize])) {
            region[t] = true;
        }
    }
    {
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for (t, tri) in ss.triangles.iter().enumerate() {
            if region[t] {
                let w = tri.map(|i| sweld.id[i as usize]);
                for k in 0..3 {
                    adj.entry(w[k]).or_default().push(w[(k + 1) % 3]);
                    adj.entry(w[(k + 1) % 3]).or_default().push(w[k]);
                }
            }
        }
         
        let eps = height * 0.003;
        let key = |p: V| [p.x, p.y, p.z].map(|x| (x / eps).floor() as i64);
        let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
        for &w in adj.keys() {
            grid.entry(key(v(ss.positions[sweld.representative[w]]))).or_default().push(w);
        }
        let mut seen: HashSet<usize> = scan_ring_ids.iter().copied().collect();
        let mut stack: Vec<usize> = scan_ring_ids.clone();
        while let Some(x) = stack.pop() {
            let px = v(ss.positions[sweld.representative[x]]);
            let k = key(px);
            let mut next: Vec<usize> = adj.get(&x).cloned().unwrap_or_default();
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(c) = grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                            next.extend(c.iter().copied().filter(|&y| (v(ss.positions[sweld.representative[y]]) - px).norm() < eps));
                        }
                    }
                }
            }
            for y in next {
                if seen.insert(y) {
                    stack.push(y);
                }
            }
        }
        for (t, tri) in ss.triangles.iter().enumerate() {
            if region[t] && !tri.iter().all(|&i| seen.contains(&sweld.id[i as usize])) {
                region[t] = false;
            }
        }
    }
    let hand_triangles: Vec<usize> = (0..ss.triangles.len()).filter(|&t| region[t]).collect();
    ensure!(hand_triangles.len() > 100, "hand_keep_empty: only {} scan skin triangles beyond the wrist", hand_triangles.len());
    let scan_ring: HashSet<usize> = scan_ring_ids.iter().copied().collect();
    let mut spos: Vec<Vec3> = ss.positions.clone();
    for (i, p) in spos.iter_mut().enumerate() {
        if scan_ring.contains(&sweld.id[i]) {
            let d = frame.axial(v(*p)) - scan_cut;
            if d > 0.0 && d < height * 0.01 {
                *p = a(v(*p) - frame.axis * d);
            }
        }
    }

    // ---- Weights: the digit's bones and the palm's, feathered at the seam ----
    let feather = FEATHER * height;
    let mut weld_ids: Vec<usize> = hand_triangles
        .iter()
        .flat_map(|&t| ss.triangles[t].map(|i| sweld.id[i as usize]))
        .chain(scan_ring_ids.iter().copied())
        .collect();
    weld_ids.sort_unstable();
    weld_ids.dedup();
    let floor = (height * 0.003).powi(4);
    let ring_row: Vec<Influence> = {
        let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
        for &w in &body_ring_ids {
            for x in &body_infl[bweld.representative[w]] {
                *acc.entry(x.joint).or_default() += x.weight / body_ring_ids.len() as f64;
            }
        }
        // Two bones: with the forearm's three and the palm's in the seam's
        // blend, neighbours kept different fourth bones and stretched.
        finish(acc, 2)
    };
    let rows: Vec<Vec<Influence>> = weld_ids
        .par_iter()
        .map(|&w| {
            let p = v(spos[sweld.representative[w]]);
            let d: Vec<f64> = segments.iter().map(|&(_, _, s0, s1)| seg_dist(p, s0, s1)).collect();
            let near = (0..segments.len()).min_by(|&x, &y| d[x].total_cmp(&d[y])).unwrap_or(0);
            let g = segments[near].1;
            let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
            for (k, &(j, gj, _, _)) in segments.iter().enumerate() {
                if gj == g || gj == 0 {
                    *acc.entry(j).or_default() += 1.0 / (d[k].powi(4) + floor);
                }
            }
            let mut hand_row = if rigid {
                vec![Influence { joint: wrist, weight: 1.0 }]
            } else {
                finish(acc, 3)
            };
             
            let f = smoothstep((frame.axial(p) - scan_cut) / feather);
            if f < 1.0 {
                hand_row = mix(&[(&ring_row, 1.0 - f), (&hand_row, f)], &identity, 4);
            }
            hand_row
        })
        .collect();
    let weight_of: HashMap<usize, usize> = weld_ids.iter().enumerate().map(|(k, &w)| (w, k)).collect();
    // Smoothed over the hand's surface (and across cracks): by nearest digit
    // alone, a palm vertex beside a finger's base took palm bones only and
    // its neighbour the finger's, and the walk's curl tore the edge between
    // them 100-fold into a needle.
    let rows: Vec<Vec<Influence>> = {
        let mut nbr: Vec<Vec<usize>> = vec![vec![]; weld_ids.len()];
        for &t in &hand_triangles {
            let w = ss.triangles[t].map(|i| weight_of[&sweld.id[i as usize]]);
            for k in 0..3 {
                nbr[w[k]].push(w[(k + 1) % 3]);
                nbr[w[(k + 1) % 3]].push(w[k]);
            }
        }
        let eps = height * 0.003;
        let pos = |k: usize| v(spos[sweld.representative[weld_ids[k]]]);
        let key = |p: V| [p.x, p.y, p.z].map(|x| (x / eps).floor() as i64);
        let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
        for k in 0..weld_ids.len() {
            grid.entry(key(pos(k))).or_default().push(k);
        }
        for k in 0..weld_ids.len() {
            let (p, c) = (pos(k), key(pos(k)));
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(list) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                            for &j in list {
                                if j != k && (pos(j) - p).norm() < eps {
                                    nbr[k].push(j);
                                }
                            }
                        }
                    }
                }
            }
        }
        for n in &mut nbr {
            n.sort_unstable();
            n.dedup();
        }
        let ring_fixed: Vec<bool> = weld_ids.iter().map(|w| scan_ring.contains(w)).collect();
        // Passes for a blur of BLUR of the height whatever the mesh's
        // density (a random walk spreads as the root of its steps).
        let (mut sum, mut count) = (0.0, 0usize);
        for (k, n) in nbr.iter().enumerate() {
            for &j in n {
                sum += (pos(j) - pos(k)).norm();
                count += 1;
            }
        }
        let edge = (sum / count.max(1) as f64).max(1e-9);
        let passes = ((BLUR * height / edge).powi(2).round() as usize).clamp(3, 80);
        let mut cur = rows;
        for _ in 0..passes {
            cur = (0..cur.len())
                .into_par_iter()
                .map(|k| {
                    if ring_fixed[k] || nbr[k].is_empty() {
                        return cur[k].clone();
                    }
                    let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
                    for x in &cur[k] {
                        *acc.entry(x.joint).or_default() += 0.5 * x.weight;
                    }
                    let share = 0.5 / nbr[k].len() as f64;
                    for &j in &nbr[k] {
                        for x in &cur[j] {
                            *acc.entry(x.joint).or_default() += share * x.weight;
                        }
                    }
                     
                    finish(acc, 4)
                })
                .collect();
        }
        cur.into_iter()
            .map(|r| finish(r.into_iter().map(|x| (x.joint, x.weight)).collect(), 4))
            .collect()
    };

     
    let sring = Ring::new(
        &frame,
        scan_ring_ids.clone(),
        |w| v(spos[sweld.representative[w]]),
        |w| v(ss.normals[sweld.representative[w]]),
    );
    let body_radius = body_ring_ids.iter().map(|&w| frame.radial(bpoint(w)).norm()).sum::<f64>()
        / body_ring_ids.len().max(1) as f64;
     
    let cuff_reach = CUFF_TOUCH * height;
    let cuff_cell = cuff_reach.max(1e-6);
    let ckey = |p: V| [p.x, p.y, p.z].map(|x| (x / cuff_cell).floor() as i64);
    let mut cuff_grid: HashSet<[i64; 3]> = HashSet::new();
    for g in garments.iter().filter(|g| !g.hair) {
        for p in &g.positions {
            let p = v(*p);
            if inside(p) {
                cuff_grid.insert(ckey(p));
            }
        }
    }
    let touched = scan_ring_ids
        .iter()
        .filter(|&&w| {
            let c = ckey(v(spos[sweld.representative[w]]));
            (-1..=1).any(|dx| (-1..=1).any(|dy| (-1..=1).any(|dz| cuff_grid.contains(&[c[0] + dx, c[1] + dy, c[2] + dz]))))
        })
        .count();
    let cuff = touched as f64 / scan_ring_ids.len().max(1) as f64;
    let ratio = sring.mean_radius(&frame) / body_radius.max(1e-6);
    let growth = if cuff >= CUFF_RING_SHARE { 1.0 } else { ratio.clamp(1.0, WRIST_GROWTH_MAX) };
    let mut grown = 0usize;
    let mut moved = vec![false; np.len()];
    let mut kept_source = vec![0usize; np.len()];
    for (i, &k) in new_index.iter().enumerate() {
        if k != u32::MAX {
            kept_source[k as usize] = i;
        }
    }
    if growth > 1.02 {
         
        let elbow = prior
            .by_name
            .get(&format!("lowerarm01.{suffix}"))
            .map(|&j| origin(&rig.worlds[j]));
        let band = elbow
            .map(|e| body_cut - frame.axial(e))
            .filter(|b| *b > body_radius)
            .unwrap_or(WRIST_GROWTH_BAND * body_radius);
        for i in 0..blv.len() {
            if i < bs.first_new && !on_arm[i] {
                continue;
            }
            let p = v(blv[i]);
            let d = -plane_offset(&frame, p, body_cut);
            if !(0.0..=band).contains(&d) {
                continue;
            }
            let x = d / band;
            let fade = 1.0 - x * x * (3.0 - 2.0 * x);
            blv[i] = a(p + frame.radial(p) * ((growth - 1.0) * fade));
            if new_index[i] != u32::MAX {
                np[new_index[i] as usize] = blv[i];
                moved[new_index[i] as usize] = true;
            }
            grown += 1;
        }
         
        let w = Weld::new(&np, 1e-7);
        let fresh = vertex_normals(&np, &nt, &w);
        for (k, m) in moved.iter().enumerate() {
            if *m && !body_ring.contains(&bweld.id[kept_source[k]]) {
                nn[k] = fresh[k];
            }
        }
    }
    let bpoint = |w: usize| v(blv[bweld.representative[w]]);

    // ---- Loft ----
    let bring = Ring::new(&frame, body_ring_ids, bpoint, |w| v(bs.normals[bweld.representative[w]]));
    let body_row = |k: usize| body_infl[bweld.representative[bring.ids[k]]].clone();
    let scan_row = |k: usize| rows[weight_of[&sring.ids[k]]].clone();
    let rings = 6usize;
    let samples = 96usize;
    let mut br = Bridge {
        name: format!("wrist_bridge_{tag}"),
        positions: vec![],
        normals: vec![],
        uvs: vec![],
        triangles: vec![],
        influences: vec![],
        png: vec![],
        bottom: vec![],
        top: vec![],
    };
    let push = |br: &mut Bridge, p: V, uv: [f64; 2], infl: Vec<Influence>| -> u32 {
        br.positions.push(a(p));
        br.uvs.push([uv[0] as f32, uv[1] as f32]);
        br.influences.push(infl);
        (br.positions.len() - 1) as u32
    };
    let mut row_index: Vec<Vec<u32>> = vec![];
    let mut row_params: Vec<Vec<f64>> = vec![];
    {
        let n = bring.len();
        let mut idx = vec![];
        for k in 0..=n {
            idx.push(push(&mut br, bring.points[k % n], [bring.params[k], 0.0], body_row(k % n)));
            if k < n {
                br.bottom.push((bring.params[k], bs.uvs[bweld.representative[bring.ids[k]]]));
            }
        }
        row_index.push(idx);
        row_params.push(bring.params.clone());
    }
    let blend = |ring: &Ring, row: &dyn Fn(usize) -> Vec<Influence>, u: f64| {
        let (i, j, f) = ring.locate(u);
        let (ri, rj) = (row(i), row(j));
        mix(&[(&ri, 1.0 - f), (&rj, f)], &identity, 8)
    };
    let ends: Vec<(V, V, V, V)> = (0..samples)
        .map(|j| {
            let u = j as f64 / samples as f64;
            let (lo, ln) = bring.sample(u);
            let (hi, hn) = sring.sample(u);
            let delta = hi - lo;
            (lo, end_tangent(&frame, ln, delta), hi, end_tangent(&frame, hn, delta))
        })
        .collect();
    for r in 1..rings {
        let t = r as f64 / rings as f64;
        let mut pts: Vec<V> = ends.iter().map(|(lo, t0, hi, t1)| hermite(*lo, *t0, *hi, *t1, t)).collect();
        let strength = (std::f64::consts::PI * t).sin();
        for _ in 0..2 {
            let prev = pts.clone();
            for j in 0..samples {
                let mut acc = V::zeros();
                for d in -3i64..=3 {
                    acc += prev[(j as i64 + d).rem_euclid(samples as i64) as usize];
                }
                pts[j] = prev[j] * (1.0 - strength) + acc / 7.0 * strength;
            }
        }
        let s = smoothstep(t);
        let mut idx = vec![];
        let mut params = vec![];
        for j in 0..=samples {
            let u = j as f64 / samples as f64;
            let lw = blend(&bring, &body_row, u);
            let hw = blend(&sring, &scan_row, u);
            idx.push(push(&mut br, pts[j % samples], [u, t], mix(&[(&lw, 1.0 - s), (&hw, s)], &identity, 4)));
            params.push(u);
        }
        row_index.push(idx);
        row_params.push(params);
    }
    {
        let n = sring.len();
        let mut idx = vec![];
        for k in 0..=n {
            idx.push(push(&mut br, sring.points[k % n], [sring.params[k], 1.0], scan_row(k % n)));
            if k < n {
                br.top.push((sring.params[k], None));
            }
        }
        row_index.push(idx);
        row_params.push(sring.params.clone());
    }
    // Scan colour at the ring.
    {
        let mut colour: HashMap<usize, ([f64; 3], f64)> = HashMap::new();
        for (t, tri) in ss.triangles.iter().enumerate() {
            if !region[t] {
                continue;
            }
            for &i in tri {
                let w = sweld.id[i as usize];
                if scan_ring.contains(&w)
                    && let Some(col) = colour_at_uv(scan, smaterial[t], ss.uvs[i as usize])
                {
                    let e = colour.entry(w).or_insert(([0.0; 3], 0.0));
                    for (sum, c) in e.0.iter_mut().zip(col) {
                        *sum += crate::source::srgb_to_linear(c);
                    }
                    e.1 += 1.0;
                }
            }
        }
        for (k, entry) in br.top.iter_mut().enumerate() {
            entry.1 = colour.get(&sring.ids[k]).map(|(sum, n)| sum.map(|c| crate::source::linear_to_srgb(c / n)));
        }
    }
    for r in 0..rings {
        zipper(&row_index[r], &row_params[r], &row_index[r + 1], &row_params[r + 1], &mut br.triangles);
    }
    let inward = br
        .triangles
        .iter()
        .filter(|t| {
            let p = t.map(|i| v(br.positions[i as usize]));
            let n = (p[1] - p[0]).cross(&(p[2] - p[0]));
            n.dot(&frame.radial((p[0] + p[1] + p[2]) / 3.0)) < 0.0
        })
        .count();
    if inward * 2 > br.triangles.len() {
        for t in &mut br.triangles {
            t.swap(1, 2);
        }
    }
    // Seam normals over both sides of each ring.
    let mut body_seam: HashMap<usize, V> = HashMap::new();
    for &t in &kept_split {
        let tri = bs.triangles[t];
        let n = crate::geom::face_normal(&blv, &tri);
        for &i in &tri {
            let w = bweld.id[i as usize];
            if body_ring.contains(&w) {
                *body_seam.entry(w).or_insert(V::zeros()) += n;
            }
        }
    }
    let mut scan_seam: HashMap<usize, V> = HashMap::new();
    for &t in &hand_triangles {
        let tri = ss.triangles[t];
        let n = crate::geom::face_normal(&spos, &tri);
        for &i in &tri {
            let w = sweld.id[i as usize];
            if scan_ring.contains(&w) {
                *scan_seam.entry(w).or_insert(V::zeros()) += n;
            }
        }
    }
    let bottom_n = bring.len() + 1;
    let top_start = br.positions.len() - (sring.len() + 1);
    for t in &br.triangles {
        let n = crate::geom::face_normal(&br.positions, t);
        for &i in t {
            let i = i as usize;
            if i < bottom_n {
                *body_seam.entry(bring.ids[i % bring.len()]).or_insert(V::zeros()) += n;
            } else if i >= top_start {
                *scan_seam.entry(sring.ids[(i - top_start) % sring.len()]).or_insert(V::zeros()) += n;
            }
        }
    }
    let bw = Weld::new(&br.positions, 1e-9);
    br.normals = vertex_normals(&br.positions, &br.triangles, &bw);
    let unit = |n: V| a(n.try_normalize(1e-20).unwrap_or(frame.axis));
    for k in 0..bottom_n {
        br.normals[k] = unit(body_seam[&bring.ids[k % bring.len()]]);
    }
    for k in top_start..br.positions.len() {
        br.normals[k] = unit(scan_seam[&sring.ids[(k - top_start) % sring.len()]]);
    }
    for (old, &n2) in new_index.iter().enumerate() {
        if n2 != u32::MAX
            && let Some(n) = body_seam.get(&bweld.id[old])
        {
            nn[n2 as usize] = unit(*n);
        }
    }

    // ---- Hand parts, one per scan material ----
    let mut groups: BTreeMap<Option<usize>, Vec<usize>> = BTreeMap::new();
    for &t in &hand_triangles {
        groups.entry(smaterial[t]).or_default().push(t);
    }
    let multi = groups.len() > 1;
    let mut parts = vec![];
    for (gi, (material, list)) in groups.into_iter().enumerate() {
        let mut remap: HashMap<u32, u32> = HashMap::new();
        let mut part = HeadPart {
            name: if multi { format!("scan_hand_{tag}_{gi}") } else { format!("scan_hand_{tag}") },
            material,
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            triangles: vec![],
            influences: vec![],
        };
        for t in list {
            let tri = ss.triangles[t].map(|i| {
                *remap.entry(i).or_insert_with(|| {
                    let w = sweld.id[i as usize];
                    part.positions.push(spos[i as usize]);
                    part.normals.push(match scan_seam.get(&w) {
                        Some(n) => unit(*n),
                        None => ss.normals[i as usize],
                    });
                    part.uvs.push(ss.uvs[i as usize]);
                    part.influences.push(rows[weight_of[&w]].clone());
                    (part.positions.len() - 1) as u32
                })
            });
            part.triangles.push(tri);
        }
        parts.push(part);
    }

    // ---- Closed seam ----
    let mut all_pos: Vec<Vec3> = np.clone();
    let mut all_tri: Vec<[u32; 3]> = nt.clone();
    for (p, t) in parts.iter().map(|h| (&h.positions, &h.triangles)).chain(std::iter::once((&br.positions, &br.triangles))) {
        let o = all_pos.len() as u32;
        all_pos.extend_from_slice(p);
        all_tri.extend(t.iter().map(|x| x.map(|i| i + o)));
    }
    let all_weld = Weld::new(&all_pos, 1e-7);
    let mut edge_faces: HashMap<(usize, usize), u32> = HashMap::new();
    for t in &all_tri {
        let w = t.map(|i| all_weld.id[i as usize]);
        for (x, y) in [(w[0], w[1]), (w[1], w[2]), (w[2], w[0])] {
            if x != y {
                *edge_faces.entry((x.min(y), x.max(y))).or_default() += 1;
            }
        }
    }
    let base = all_pos.len() - br.positions.len();
    let (mut open, mut nonmanifold) = (0usize, 0usize);
    for (start, len) in [(0usize, bring.len()), (top_start, sring.len())] {
        for k in 0..len {
            let (x, y) = (all_weld.id[base + start + k], all_weld.id[base + start + k + 1]);
            match edge_faces.get(&(x.min(y), x.max(y))).copied().unwrap_or(0) {
                2 => {}
                0 | 1 => open += 1,
                _ => nonmanifold += 1,
            }
        }
    }
    if open > 0 || nonmanifold > 0 {
        bail!("hand_keep_open_seam: {open} open, {nonmanifold} non-manifold seam edges");
    }

     
    let hand_pts: Vec<(V, usize)> = weld_ids.iter().enumerate().map(|(k, &w)| (v(spos[sweld.representative[w]]), k)).collect();
    let cell = height * 0.02;
    let hkey = |p: V| [p.x, p.y, p.z].map(|x| (x / cell).floor() as i64);
    let mut hgrid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (idx, (p, _)) in hand_pts.iter().enumerate() {
        hgrid.entry(hkey(*p)).or_default().push(idx);
    }
    let nearest_row = |p: V| -> Option<usize> {
        let c = hkey(p);
        let mut best: Option<(f64, usize)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for &idx in hgrid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]).map_or(&[][..], |v| v.as_slice()) {
                        let d = (hand_pts[idx].0 - p).norm();
                        if d < cell && best.is_none_or(|b| d < b.0) {
                            best = Some((d, hand_pts[idx].1));
                        }
                    }
                }
            }
        }
        best.map(|b| b.1)
    };
    let digit_or_palm = |j: usize| {
        let n = &names[j];
        n.ends_with(&format!(".{}", tag)) && (n.starts_with("finger") || n.starts_with("metacarpal"))
    };
    let (mut reweighted, mut near_hand, mut folded) = (0usize, 0usize, 0usize);
    for g in garments.iter_mut() {
        for i in 0..g.positions.len() {
            if let Some(&k) = weight_of.get(&g.weld_id[i]) {
                g.influences[i] = rows[k].clone();
                reweighted += 1;
                continue;
            }
            let p = v(g.positions[i]);
            if frame.axial(p) > scan_cut
                && inside(p)
                && let Some(k) = nearest_row(p)
            {
                g.influences[i] = rows[k].clone();
                near_hand += 1;
            } else if g.influences[i].iter().any(|x| digit_or_palm(x.joint)) {
                // Before the cut: this side's finger and palm bones (from
                // MakeHuman's hand, now moved into the scan's digits) fold
                // onto the wrist.
                let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
                for x in &g.influences[i] {
                    let j = if digit_or_palm(x.joint) { wrist } else { x.joint };
                    *acc.entry(j).or_default() += x.weight;
                }
                g.influences[i] = finish(acc, 4);
                folded += 1;
            }
        }
    }

    kept.body_positions = np;
    kept.body_normals = nn;
    kept.body_uvs = nu;
    kept.body_triangles = nt;
    kept.body_influences = ni;
    kept.body_triangle_origin = no;
    kept.body_near_seam = nnear;
    let hand_vertices: usize = parts.iter().map(|h| h.positions.len()).sum();
    let names_used: Vec<String> = parts.iter().map(|p| p.name.clone()).collect();
    kept.heads.extend(parts);
    let bridge_triangles = br.triangles.len();
    kept.bridges.push(br);
    Ok(json!({
        "route": "keep_scan",
        "side": tag,
        "digits": digits,
        "skinned_digits": kept_digits,
        "wrist_origin": a(w0),
        "axis": a(axis),
        "ball_radius_m": ball,
        "body_offset_m": body_cut,
        "scan_offset_m": scan_cut,
        "seam_vertices": {"body": bring.len(), "scan": sring.len()},
        "ring_radius_m": {"body": bring.mean_radius(&frame), "scan": sring.mean_radius(&frame)},
        "forearm_growth": {"ratio": growth, "ring_ratio": ratio, "cuff_share": cuff, "body_radius_before_m": body_radius, "vertices": grown},
        "body_removed_triangles": removed_triangles,
        "body_bowtie_eroded_triangles": body_eroded,
        "scan_bowtie_eroded_triangles": scan_eroded,
        "hand_triangles": hand_triangles.len(),
        "hand_vertices": hand_vertices,
        "parts": names_used,
        "bridge_triangles": bridge_triangles,
        "open_seam_edges": open,
        "garment_vertices_reweighted": reweighted,
        "garment_vertices_near_hand_reweighted": near_hand,
        "garment_vertices_hand_weights_folded_to_wrist": folded,
        "measured_digits": measured,
        "rigid": rigid,
        "skinning": "nearest digit's bones and the palm's, inverse fourth power of the distance to each bone's segment, three at most; MakeHuman's weights at the seam, smoothstep to the hand over feather_m",
        "feather_m": feather,
    }))
}
