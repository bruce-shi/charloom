//! Preserve the skull measured from the bald-suit views, before scan fitting.
//! GPT residuals are measured before the hold. Scan skin has priority where visible;
//! GPT scalp is positional data, with a bounded biharmonic transition to skin.
use super::{guard, hidden_head, silhouette, head_registration::Registration};
use crate::{
    geom::{V, a},
    guide::SuitContour,
    scan::Scan,
    segment::{Label, Segmentation},
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(super) const BAND_M: f64 = 0.020;
const HAIR_CLEARANCE_M: f64 = 0.0018;

/// Measure the residual skull shape the three global MakeHuman controls
/// cannot express. This is the existing GPT silhouette pull, solved in a
/// bending field BEFORE the rest hold; it never consumes hair as a target.
fn fit_scalp(
    reference: &[V],
    aligned: &[V],
    neighbours: &[Vec<u32>],
    triangles: &[[u32; 3]],
    scalp: &[bool],
    _region: &[bool],
    registration: &Registration,
    centre: V,
    hair: &charloom_engine::surface_query::SurfaceQuery,
    reach: f64,
    lower_start: Option<f64>,
) -> (Vec<V>, Value) {
    if !scalp.iter().any(|&v| v) {
        return (reference.to_vec(), json!({"applied":false}));
    }
    let confidence=registration.relative_strength().max(1e-6);
    let bending_scale=0.05*confidence;
    let ridge=0.0025*confidence;
    let lower: Vec<_> = reference.iter().map(|p| lower_start.map_or(0., |start|
        ((centre.z + 0.02 - p.z) / 0.04).clamp(0., 1.)
        * ((start - p.y) / 0.010).clamp(0., 1.))).collect();
    let density: Vec<_> = lower.iter().map(|w| 1. + 19. * w).collect();
    let rows = bending_rows_weighted(reference, neighbours, Some(&density));
    let ids: Vec<_> = (0..reference.len()).filter(|&i| scalp[i]).collect();
    let mut x: Vec<_> = (0..reference.len())
        .map(|i| if scalp[i] { reference[i] } else { aligned[i] })
        .collect();
    let (views, hair_evidence) = registration.weighted_views(&x, triangles, scalp, centre, hair, reach);
    // The lower solve must bend into the actual scan-aligned collar. Zero
    // displacement here would anchor the nape to an unrelated rest collar.
    let mut offset: Vec<_> = (0..x.len()).map(|i| if scalp[i] {V::zeros()}
        else {(aligned[i] - reference[i]) * lower[i]}).collect();
    let mut history = vec![];
    for iteration in 0..60 {
        let (constraints, _, stats) =
            silhouette::rendered_constraints(&views, &x, triangles, scalp, 0.025);
        let constraints: Vec<_> = constraints
            .into_iter()
            .filter(|c| {
                c.vertices
                    .iter()
                    .zip(c.bary)
                    .filter(|(i, _)| scalp[**i as usize])
                    .map(|(_, b)| b)
                    .sum::<f64>()
                    > 0.5
                    && c.weight > 0.0
            })
            .collect();
        if constraints.is_empty() {
            break;
        }
        let project = |c: &silhouette::SurfaceConstraint, p: V| {
            c.axes[0] * c.axes[0].dot(&p) + c.axes[1] * c.axes[1].dot(&p)
        };
        let mut rhs = vec![V::zeros(); x.len()];
        let mut diagonal = vec![1.0; x.len()];
        for &i in &ids {
            diagonal[i] = bending_scale * rows[i][&i] + ridge;
        }
        for c in &constraints {
            let current = c
                .vertices
                .iter()
                .zip(c.bary)
                .fold(V::zeros(), |s, (&i, b)| s + offset[i as usize] * b);
            let target = project(c, current + c.delta) * c.weight;
            for (&i, b) in c.vertices.iter().zip(c.bary) {
                let i = i as usize;
                if scalp[i] {
                    rhs[i] += target * b;
                    diagonal[i] += c.weight * b * b;
                }
            }
        }
        let apply = |p: &[V]| -> Vec<V> {
            let mut y = vec![V::zeros(); x.len()];
            for &i in &ids {
                y[i] = rows[i]
                    .iter()
                    .fold(V::zeros(), |s, (&j, &a)| s + p[j] * (bending_scale * a))
                    + p[i] * ridge;
            }
            for c in &constraints {
                let q = c
                    .vertices
                    .iter()
                    .zip(c.bary)
                    .fold(V::zeros(), |s, (&i, b)| s + p[i as usize] * b);
                let q = project(c, q) * c.weight;
                for (&i, b) in c.vertices.iter().zip(c.bary) {
                    let i = i as usize;
                    if scalp[i] {
                        y[i] += q * b;
                    }
                }
            }
            y
        };
        let dot = |a: &[V], b: &[V]| ids.iter().map(|&i| a[i].dot(&b[i])).sum::<f64>();
        let applied = apply(&offset);
        let mut r: Vec<_> = rhs.iter().zip(applied).map(|(b, a)| b - a).collect();
        let mut d: Vec<_> = r.iter().zip(&diagonal).map(|(r, a)| r / *a).collect();
        let mut rz = dot(&r, &d);
        let initial = dot(&r, &r).max(1e-30);
        let mut solved = offset.clone();
        for _ in 0..500 {
            if dot(&r, &r) < initial * 1e-16 {
                break;
            }
            let ad = apply(&d);
            let den = dot(&d, &ad);
            if den <= 1e-30 {
                break;
            }
            let alpha = rz / den;
            for &i in &ids {
                solved[i] += d[i] * alpha;
                r[i] -= ad[i] * alpha;
            }
            let z: Vec<_> = r.iter().zip(&diagonal).map(|(r, a)| r / *a).collect();
            let next = dot(&r, &z);
            let beta = next / rz.max(1e-300);
            for &i in &ids {
                d[i] = z[i] + d[i] * beta;
            }
            rz = next;
        }
        let mut change: f64 = 0.0;
        for &i in &ids {
            let delta = (solved[i] - offset[i]) * if iteration < 8 {0.4} else {0.2};
            change = change.max(delta.norm());
            offset[i] += delta;
            x[i] = reference[i] + offset[i];
        }
        history.push(json!({"max_update_mm":change*1000.0,"views":stats}));
        if change < 0.00005 {
            break;
        }
    }
    let target = (0..x.len()).map(|i| reference[i] + offset[i]).collect();
    (
        target,
        json!({"applied":true,"method":"GPT_silhouette_bending_field_before_hold", "iterations":history,
        "hair_evidence":hair_evidence,"mean_registration_confidence":confidence,"bending_scale":bending_scale,
        "lower_bending_multiplier":if lower_start.is_some(){20.0}else{1.0},"lower_scan_boundary_displacement":lower_start.is_some(),
        "max_offset_mm":offset.iter().map(|d|d.norm()*1000.0).fold(0.0,f64::max)}),
    )
}

/// Fit -> register against the whole rendered head -> fit, with one shared
/// three-round budget across monotone scan-priority reclassification passes.
fn measured_scalp(reference:&[V], aligned:&[V], neighbours:&[Vec<u32>], triangles:&[[u32;3]],
    scalp:&[bool], region:&[bool], registration:&mut Registration, scan:&Scan, seg:&Segmentation,
    centre:V, hair:&charloom_engine::surface_query::SurfaceQuery, reach:f64, lower_start:Option<f64>)->(Vec<V>,Value) {
    let (mut measured,mut report)=fit_scalp(reference,aligned,neighbours,triangles,scalp,region,registration,centre,hair,reach,lower_start);
    if !scalp.iter().any(|v|*v) {return (measured,report);}
    while !registration.converged && registration.rounds.len()<3 {
        // Visible face, ears and neck remain in their scan-aligned positions.
        let whole:Vec<_>=(0..reference.len()).map(|i|if scalp[i]{measured[i]}else{aligned[i]}).collect();
        registration.update(&whole,triangles,region,scan,seg);
        (measured,report)=fit_scalp(reference,aligned,neighbours,triangles,scalp,region,registration,centre,hair,reach,lower_start);
    }
    report["registration_3d"]=registration.report();
    (measured,report)
}

fn occiput_levels(reference:&[V], aligned:&[V], skull:&[bool], ear:&[bool], neck:&[bool],
    visible:&[bool], contours:&[SuitContour], centre:V)->(f64,f64,Value) {
    let values:Vec<_>=reference.iter().zip(skull).filter(|(_,s)|**s).map(|(p,_)|p.y).collect();
    let low=values.iter().copied().fold(f64::INFINITY,f64::min);
    let top=values.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    if !low.is_finite() || !top.is_finite() {return (centre.y,centre.y-0.03,json!({"method":"empty_skull"}));}
    let mut ears:Vec<_>=reference.iter().zip(ear).filter(|(_,e)|**e).map(|(p,_)|p.y).collect();
    ears.sort_by(f64::total_cmp);
    let ear_y=ears.get(ears.len()/2).copied().unwrap_or(low+(top-low)*0.45);
    let fallback=(ear_y+0.35*(top-low)).min(top-0.01);
    let mut candidates=vec![];
    for c in contours.iter().filter(|c|c.camera.forward[0].abs()>0.92) {
        let mut rear=vec![];
        for p in c.curves.iter().flatten() {
            let world=crate::geom::v(c.camera.center)+crate::geom::v(c.camera.right)*((p[0]-c.camera.width as f64/2.)*c.camera.meters_per_pixel)
                +crate::geom::v(c.camera.up)*((c.camera.height as f64/2.-p[1])*c.camera.meters_per_pixel);
            if world.y>ear_y && world.y<top+0.02 && world.z<centre.z {rear.push(world);}
        }
        let back=rear.iter().map(|p|p.z).fold(f64::INFINITY,f64::min);
        let bulge:Vec<_>=rear.iter().filter(|p|p.z<back+0.001).collect();
        if !bulge.is_empty(){candidates.push(bulge.iter().map(|p|p.y).sum::<f64>()/bulge.len()as f64);}
    }
    let side_y=if candidates.is_empty(){None}else{Some(candidates.iter().sum::<f64>()/candidates.len()as f64)};
    let valid=side_y.filter(|y|*y>ear_y+0.1*(top-low)&&*y<top-0.01);
    let start=valid.unwrap_or(fallback);
    let end=(0..reference.len()).filter(|&i|neck[i]&&visible[i]&&reference[i].z<centre.z+0.02)
        .map(|i|aligned[i].y).max_by(f64::total_cmp).unwrap_or(ear_y-0.04).min(start-0.015);
    (start,end,json!({"method":if valid.is_some(){"side_GPT_maximum_back_point"}else{"ear_canal_plus_35_percent_skull_height"},
        "ear_canal_proxy_y_m":ear_y,"skull_height_mm":(top-low)*1000.,"side_bulge_candidate_y_m":side_y,
        "ramp_start_y_m":start,"visible_neck_end_y_m":end,"ramp_height_mm":(start-end)*1000.,"outline_gate_min_y_m":start}))
}

 fn fairing_force(reference:&[V],neighbours:&[Vec<u32>],weight:&[f64])->Vec<V> {
    let mut force=vec![V::zeros();reference.len()];
    for i in 0..reference.len() {
        if weight[i]<=0. {continue;}
        let mut row=vec![];let mut diagonal=0.;let mut lap=V::zeros();
        for &j in &neighbours[i] {let j=j as usize;let w=0.005/(reference[i]-reference[j]).norm().max(1e-5);
            diagonal+=w;row.push((j,-w));lap+=(reference[i]-reference[j])*w;}
        row.push((i,diagonal));
        for (j,w) in row {force[j]+=lap*(w*weight[i]);}
    }
    force
}

 pub(super) fn bending_rows(x: &[V], neighbours: &[Vec<u32>]) -> Vec<BTreeMap<usize, f64>> {
    bending_rows_weighted(x, neighbours, None)
}

/// L^T W L, retaining symmetry when only a local observed patch needs stronger
/// bending regularization. Scaling completed matrix rows would break CG.
fn bending_rows_weighted(x: &[V], neighbours: &[Vec<u32>], density: Option<&[f64]>) -> Vec<BTreeMap<usize, f64>> {
    let n = x.len();
    let mut rows = vec![BTreeMap::new(); n];
    for i in 0..n {
        let mut row = Vec::with_capacity(neighbours[i].len() + 1);
        let mut diagonal = 0.0;
        for &j in &neighbours[i] {
            let j = j as usize;
            let w = 0.005 / (x[i] - x[j]).norm().max(1e-5);
            diagonal += w;
            row.push((j, -w));
        }
        row.push((i, diagonal));
        for &(j, u) in &row {
            for &(k, v) in &row {
                *rows[j].entry(k).or_insert(0.0) += u * v * density.map_or(1., |d| d[i]);
            }
        }
    }
    rows
}

/// Minimise ||L w||^2 with Dirichlet 0/1 boundaries and 0 <= w <= 1.
/// These are bounded biharmonic weights, not a threshold or a smoothstep.
fn blend_weights(
    x: &[V],
    neighbours: &[Vec<u32>],
    eligible: &[bool],
    band_m: f64,
) -> (Vec<f64>, Value) {
    let zero: Vec<bool> = eligible.iter().map(|v| !v).collect();
    let distance = guard::surface_distance(x, neighbours, &zero, band_m);
    let core: Vec<bool> = (0..x.len())
        .map(|i| eligible[i] && distance[i] >= band_m)
        .collect();
    let free: Vec<usize> = (0..x.len()).filter(|&i| eligible[i] && !core[i]).collect();
    let mut w: Vec<f64> = (0..x.len())
        .map(|i| {
            if core[i] {
                1.0
            } else if eligible[i] {
                (distance[i] / band_m).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect();
    let rows = bending_rows(x, neighbours);
    // Start from the unconstrained biharmonic solution. Stationary descent
    // alone converges very slowly on long, narrow strips (L^T L).
    let mut index = vec![usize::MAX; x.len()];
    for (k, &i) in free.iter().enumerate() {
        index[i] = k;
    }
    let apply = |p: &[f64]| -> Vec<f64> {
        free.iter()
            .map(|&i| {
                rows[i]
                    .iter()
                    .filter(|(j, _)| index[**j] != usize::MAX)
                    .map(|(&j, &a)| a * p[index[j]])
                    .sum()
            })
            .collect()
    };
    let rhs: Vec<f64> = free
        .iter()
        .map(|&i| {
            -rows[i]
                .iter()
                .filter(|(j, _)| index[**j] == usize::MAX)
                .map(|(&j, &a)| a * w[j])
                .sum::<f64>()
        })
        .collect();
    let diagonal: Vec<f64> = free.iter().map(|&i| rows[i][&i].max(1e-12)).collect();
    let mut solution: Vec<f64> = free.iter().map(|&i| w[i]).collect();
    let applied = apply(&solution);
    let mut r: Vec<f64> = rhs.iter().zip(applied).map(|(b, a)| b - a).collect();
    let mut direction: Vec<f64> = r.iter().zip(&diagonal).map(|(r, d)| r / d).collect();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    let initial = dot(&r, &r).max(1e-30);
    let mut rz = dot(&r, &direction);
    for _ in 0..2000 {
        if dot(&r, &r) < initial * 1e-22 {
            break;
        }
        let ad = apply(&direction);
        let denominator = dot(&direction, &ad);
        if denominator <= 1e-30 {
            break;
        }
        let alpha = rz / denominator;
        for k in 0..free.len() {
            solution[k] += alpha * direction[k];
            r[k] -= alpha * ad[k];
        }
        let z: Vec<f64> = r.iter().zip(&diagonal).map(|(r, d)| r / d).collect();
        let next = dot(&r, &z);
        let beta = next / rz.max(1e-300);
        for k in 0..free.len() {
            direction[k] = z[k] + beta * direction[k];
        }
        rz = next;
    }
    for (k, &i) in free.iter().enumerate() {
        w[i] = solution[k].clamp(0.0, 1.0);
    }
    let mut change: f64 = 0.0;
    let mut iterations = 0;
    for _ in 0..10000 {
        change = 0.0;
        for &i in &free {
            let diagonal = rows[i].get(&i).copied().unwrap_or(0.0);
            if diagonal < 1e-12 {
                continue;
            }
            let off = rows[i]
                .iter()
                .filter(|(j, _)| **j != i)
                .map(|(&j, &a)| a * w[j])
                .sum::<f64>();
            let next = (-off / diagonal).clamp(0.0, 1.0);
            change = change.max((next - w[i]).abs());
            w[i] = next;
        }
        iterations += 1;
        if change < 1e-9 {
            break;
        }
    }
    (
        w,
        json!({"method":"bounded_biharmonic", "band_mm": band_m*1000.0,
        "core_vertices":core.iter().filter(|v|**v).count(), "transition_vertices":free.len(),
        "iterations":iterations, "max_weight_update":change, "converged":change<1e-9}),
    )
}

/// Solve a displacement field over the smooth, pre-measurement surface.
/// The visible skin collar has soft scan targets; protected anatomical rows
/// remain Dirichlet boundaries. Hair inequalities use inward offsets projected
/// from the SAME Hessian after the unconstrained solve, never vertex clipping.
#[allow(clippy::too_many_arguments)]
fn transition_field(
    reference: &[V],
    scan_fit: &[V],
    target: &[V],
    hold: &[f64],
    free: &[bool],
    hair_eligible: &[bool],
    data_fidelity: &[f64],
    core_fidelity: f64,
    rows: &[BTreeMap<usize, f64>],
    bending_force: Option<&[V]>,
    centre: V,
    hair: &charloom_engine::surface_query::SurfaceQuery,
    reach: f64,
) -> (Vec<V>, Value) {
    let ids: Vec<_> = (0..reference.len()).filter(|&i| free[i]).collect();
    if ids.is_empty() {
        return (scan_fit.to_vec(), json!({"applied":false}));
    }
    let n = reference.len();
    let mut d: Vec<_> = scan_fit.iter().zip(reference).map(|(x, r)| x - r).collect();
    let fidelity: Vec<_> = hold.iter().enumerate().map(|(i,w)| data_fidelity[i]+core_fidelity*w.powi(4)).collect();
    let rhs: Vec<_> = (0..n)
        .map(|i| {
            if !free[i] {
                return V::zeros();
            }
            let desired = (target[i] - reference[i]) * hold[i] + d[i] * (1.0 - hold[i]);
            desired * fidelity[i] - bending_force.map_or(V::zeros(), |f|f[i])
                - rows[i]
                    .iter()
                    .filter(|(j, _)| !free[**j])
                    .fold(V::zeros(), |s, (&j, &a)| s + d[j] * a)
        })
        .collect();
    let dot = |a: &[V], b: &[V]| ids.iter().map(|&i| a[i].dot(&b[i])).sum::<f64>();
    let apply = |p: &[V]| -> Vec<V> {
        let mut y = vec![V::zeros(); n];
        for &i in &ids {
            y[i] = rows[i].iter().filter(|(j, _)| free[**j])
                .fold(V::zeros(), |s, (&j, &a)| s + p[j] * a) + p[i] * fidelity[i];
        }
        y
    };
    let diagonal: Vec<_> = (0..n).map(|i| rows[i].get(&i).copied().unwrap_or(0.0)+fidelity[i]).collect();
    let ad = apply(&d);
    let mut r: Vec<_> = (0..n).map(|i| if free[i] {rhs[i]-ad[i]} else {V::zeros()}).collect();
    let mut p: Vec<_> = r.iter().zip(&diagonal).map(|(r,a)| r / *a).collect();
    let mut rz = dot(&r,&p);
    let initial = dot(&r,&r).max(1e-30);
    for _ in 0..1500 {
        if dot(&r,&r) < initial*1e-20 { break; }
        let ap = apply(&p); let den=dot(&p,&ap);
        if den<=1e-30 { break; }
        let alpha=rz/den;
        for &i in &ids { d[i]+=p[i]*alpha; r[i]-=ap[i]*alpha; }
        let z:Vec<_>=r.iter().zip(&diagonal).map(|(r,a)|r / *a).collect();
        let next=dot(&r,&z); let beta=next/rz.max(1e-300);
        for &i in &ids { p[i]=z[i]+p[i]*beta; }
        rz=next;
    }
     
    let unbounded:Vec<_>=(0..n).map(|i|reference[i]+d[i]).collect();
    let dirs:Vec<_>=unbounded.iter().enumerate().map(|(i,p)|if hair_eligible[i] {(p-centre).try_normalize(1e-12).unwrap_or(V::zeros())}else{V::zeros()}).collect();
    let mut index=vec![usize::MAX;n];
    for (k,&i) in ids.iter().enumerate(){index[i]=k;}
    let lower:Vec<_>=ids.iter().map(|&i|{
        if !hair_eligible[i] { return 0.0; }
        let radius=(unbounded[i]-centre).norm();
        hair.outward_ray_distance(a(centre),a(dirs[i]),reach)
            .map_or(0.0,|h|(radius-(h-HAIR_CLEARANCE_M).max(0.0)).max(0.0))
    }).collect();
    let matrix:Vec<Vec<(usize,f64)>>=ids.iter().map(|&i|{
        rows[i].iter().filter(|(j,_)|free[**j]).map(|(&j,&a)|
            (index[j],a*dirs[i].dot(&dirs[j])+if i==j{fidelity[i]}else{0.0})).collect()
    }).collect();
    let diag:Vec<_>=ids.iter().map(|&i|diagonal[i]).collect();
    let mut t=lower.clone();
    let mut active;
    let mul=|v:&[f64]| -> Vec<f64> {matrix.iter().map(|row|row.iter().map(|&(j,a)|a*v[j]).sum()).collect()};
    let mut iterations=0;
    let mut kkt:f64=0.0;
    for outer in 0..200 {
        iterations=outer+1;
        let gradient=mul(&t);
        kkt=0.0;
        active=vec![false;ids.len()];
        for k in 0..ids.len(){
            active[k]=t[k]-gradient[k]/diag[k] <= lower[k]+1e-12;
            let residual=if t[k]<=lower[k]+1e-10 {gradient[k].min(0.0)}else{gradient[k]};
            kkt=kkt.max(residual.abs()/diag[k]);
        }
        if kkt<1e-9 { break; }
        // Primal active-set update, with feasibility-preserving line search.
        let mut step=vec![0.0;ids.len()];
        let mut rr:Vec<_>=(0..ids.len()).map(|k|if active[k]{0.0}else{-gradient[k]}).collect();
        let mut pp:Vec<_>=rr.iter().zip(&diag).map(|(r,d)|r/d).collect();
        let dot=|a:&[f64],b:&[f64]|a.iter().zip(b).map(|(a,b)|a*b).sum::<f64>();
        let mut rz=dot(&rr,&pp);let initial=dot(&rr,&rr).max(1e-30);
        for _ in 0..1500 {
            if dot(&rr,&rr)<initial*1e-20 {break;}
            let mut ap=mul(&pp);
            for k in 0..ids.len(){if active[k]{ap[k]=0.0;}}
            let den=dot(&pp,&ap);if den<=1e-30{break;}
            let alpha=rz/den;
            for k in 0..ids.len(){step[k]+=alpha*pp[k];rr[k]-=alpha*ap[k];}
            let z:Vec<_>=rr.iter().zip(&diag).map(|(r,d)|r/d).collect();
            let next=dot(&rr,&z);let beta=next/rz.max(1e-300);
            for k in 0..ids.len(){pp[k]=z[k]+pp[k]*beta;}
            rz=next;
        }
        let mut alpha:f64=1.0;
        for k in 0..ids.len(){if step[k] < -1e-15{alpha=alpha.min((t[k]-lower[k]).max(0.0)/(-step[k]));}}
        if alpha>1e-10 {
            for k in 0..ids.len(){t[k]+=alpha*step[k];}
        }
        // Coordinate minimisation resolves simultaneous contact changes. The
        // bound is on the variational offset, not a post-solve vertex clip.
        for _ in 0..8 {for k in 0..ids.len(){
            let off:f64=matrix[k].iter().filter(|(j,_)|*j!=k).map(|&(j,a)|a*t[j]).sum();
            t[k]=(-off/diag[k]).max(lower[k]);
        }}
    }
    let mut result=unbounded;
    for (k,&i) in ids.iter().enumerate(){result[i]-=dirs[i]*t[k];}
    let violation=lower.iter().zip(&t).map(|(l,t)|(l-t).max(0.0)).fold(0.0,f64::max);
    (result,json!({"applied":true,"method":"displacement_bending_fixed_ray_inward_obstacle",
        "vertices":ids.len(),"iterations":iterations,"kkt_offset_mm":kkt*1000.0,
        "active_contacts":lower.iter().filter(|&&x|x>0.0).count(),
        "max_hair_violation_mm":violation*1000.0,"hair_converged":violation<1e-7&&kkt<1e-8}))

}

pub(super) struct Hold {
    /// 0: outside/protected internal topology; 1: scan skin; 2: GPT; 3: hidden.
    pub classes: Vec<u8>,
    pub weight: Vec<f64>,
    pub target: Vec<V>,
    pub gpt_scalp: Vec<bool>,
    pub scan_seen: Vec<bool>,
    /// Observed lower scalp stays full-strength GPT data, but is solved jointly
    /// with bending rather than frozen to a noisy per-vertex estimate.
    soft_observed: Vec<bool>,
    transition_region: Vec<bool>,
    hair_eligible: Vec<bool>,
    data_fidelity: Vec<f64>,
    pub report: Value,
    centre: V,
    hair: charloom_engine::surface_query::SurfaceQuery,
    reach: f64,
    bending: Vec<BTreeMap<usize, f64>>,
    reference: Vec<V>,
    bending_force: Vec<V>,
}

impl Hold {
         pub fn disabled(reference: &[V], scan_vertices: usize) -> Self {
        let n = reference.len();
        Self { classes: vec![0; n], weight: vec![0.; n], target: reference.to_vec(),
            gpt_scalp: vec![false; n], scan_seen: vec![false; scan_vertices],
            soft_observed: vec![false; n], transition_region: vec![false; n],
            hair_eligible: vec![false; n], data_fidelity: vec![0.; n],
            report: json!({"method":"disabled_x119","applied":false}), centre: V::zeros(),
            hair: crate::scan::build_query(&[], &[]), reach: 0.,
            bending: vec![], reference: reference.to_vec(), bending_force: vec![] }
    }

    pub fn new(
        reference: &[V],
        aligned: &[V],
        neighbours: &[Vec<u32>],
        triangles: &[[u32; 3]],
        region: &[bool],
        scalp: &[bool],
        skull: &[bool],
        fixed: &[bool],
        boundary: &[bool],
        transition_allowed: &[bool],
        ear: &[bool],
        neck: &[bool],
        scan: &Scan,
        seg: &Segmentation,
        contours: &[SuitContour],
        enabled: bool,
        height: f64,
    ) -> Self {
        let n = reference.len();
        let seen = hidden_head::head_visibility(scan, aligned, region);
        let mut visible: Vec<bool> = (0..n)
            .map(|i| {
                region[i]
                    && !fixed[i]
                    && scan
                        .query
                        .closest_point(a(aligned[i]), 0.012 * height, None)
                        .is_some_and(|h| {
                            seg.triangle_label[h.source[1]] == Label::Skin
                                && !seg.is_prop(h.source[1])
                                && scan.triangles[h.source[1]]
                                    .iter()
                                    .any(|&s| seen[s as usize])
                        })
            })
            .collect();
        // Forward nearest points can hit a strand and miss the visible skin
        // beside it. Reserve every reverse-visible skin correspondence too,
        // so the held scalp cannot steal vertices needed to fit visible skin.
        // Classification is geometric, not normal-gated: before fitting, a
        // brow/forehead normal may disagree with the scan. Rejecting that
        // correspondence incorrectly made visible forehead a hard GPT hold.
        let body = crate::scan::build_query(
            &aligned.iter().map(|p| a(*p)).collect::<Vec<_>>(),
            triangles,
        );
        for &s in &scan.weld.representative {
            if !seen[s] || seg.vertex_label[s] != Label::Skin {
                continue;
            }
            let Some(hit) = body.closest_point(scan.positions[s], height * 0.03, None) else {
                continue;
            };
            for (&i, b) in triangles[hit.source[1]].iter().zip(hit.barycentric) {
                let i = i as usize;
                if region[i] && !fixed[i] && b > 0.05 {
                    visible[i] = true;
                }
            }
        }
        // Reserve observed ear/neck skin even when a nearer hair strand
        // obscures the ordinary closest-point lookup. Only visible scan skin
        // enters this query; the GPT scalp cannot claim these data vertices.
        let skin_triangles:Vec<_>=scan.triangles.iter().enumerate()
            .filter(|(t,tri)|seg.triangle_label[*t]==Label::Skin && !seg.is_prop(*t)
                && tri.iter().any(|&i|seen[i as usize])).map(|(_,t)|*t).collect();
        let visible_skin=crate::scan::build_query(&scan.positions,&skin_triangles);
        for i in 0..n {
            if (ear[i]||neck[i]) && region[i] && !fixed[i]
                && visible_skin.closest_point(a(aligned[i]),height*0.012,None).is_some(){visible[i]=true;}
        }
        let has_back = contours.iter().any(|c| c.scale_tied_to.is_some() && c.camera.forward[2] > 0.92);
        let skull_ids: Vec<_> = (0..n).filter(|&i| skull[i]).collect();
        let posterior_centre = skull_ids.iter().fold(V::zeros(), |s, &i| s + reference[i]) / skull_ids.len().max(1) as f64;
        let (_, neck_y, _) = occiput_levels(reference, aligned, skull, ear, neck, &visible, contours, posterior_centre);
        // Back and profile see the posterior skull/nape down to observed neck.
        // Cavities, ears and all visible scan skin keep their original priority.
        let observed_posterior: Vec<_> = (0..n).map(|i| has_back && scalp[i]
            && reference[i].z < posterior_centre.z && reference[i].y >= neck_y - 0.010
            && !ear[i] && !boundary[i]).collect();
        // Bald edits expose the exterior skull, not only vertices sitting
        // on a raster silhouette. Cavities are not exterior data.
        let mut classes: Vec<u8> = (0..n)
            .map(|i| {
                if !region[i] || fixed[i] {
                    0
                } else if visible[i] {
                    1
                } else if !contours.is_empty() && scalp[i] && (skull[i] || observed_posterior[i]) && !boundary[i] {
                    2
                } else {
                    3
                }
            })
            .collect();
        let ids: Vec<_> = (0..n).filter(|&i| region[i] && !neck[i] && !fixed[i]).collect();
        let centre =
            ids.iter().fold(V::zeros(), |s, &i| s + reference[i]) / ids.len().max(1) as f64;
        let reach = ids
            .iter()
            .map(|&i| (reference[i] - centre).norm())
            .fold(0.0, f64::max)
            * 3.0;
        let hair_triangles: Vec<_> = scan
            .triangles
            .iter()
            .enumerate()
            .filter(|(i, _)| seg.triangle_label[*i] == Label::Hair && !seg.is_prop(*i))
            .map(|(_, t)| *t)
            .collect();
        let hair = crate::scan::build_query(&scan.positions, &hair_triangles);
        let mut registration=Registration::new(contours);
        let (initial_ramp_start,_,_)=occiput_levels(reference,aligned,skull,ear,neck,&visible,contours,centre);
         
        let mut reclassified = Vec::new();
        let (gpt_scalp, measured, mut measurement) = loop {
            let gpt_scalp: Vec<_> = (0..n)
                .map(|i| enabled && scalp[i] && (skull[i] || observed_posterior[i]) && !boundary[i] && classes[i] == 2)
                .collect();
            let measured_mask: Vec<_> = (0..n).map(|i| gpt_scalp[i] && (has_back || skull[i] && reference[i].y>=initial_ramp_start)).collect();
            let (measured, measurement) = measured_scalp(
                reference,
                aligned,
                neighbours,
                triangles,
                &measured_mask,
                region,
                &mut registration, scan, seg, centre, &hair, reach, has_back.then_some(initial_ramp_start),
            );
            let mut changed = false;
            for i in 0..n {
                if !gpt_scalp[i] {
                    continue;
                }
                if scan
                    .query
                    .closest_point(a(measured[i]), 0.012 * height, None)
                    .is_some_and(|h| {
                        seg.triangle_label[h.source[1]] == Label::Skin
                            && !seg.is_prop(h.source[1])
                            && scan.triangles[h.source[1]]
                                .iter()
                                .any(|&s| seen[s as usize])
                    })
                {
                    classes[i] = 1;
                    reclassified.push(i);
                    changed = true;
                }
            }
            if !changed {
                break (gpt_scalp, measured, measurement);
            }
        };
        measurement["scan_priority_reclassified_vertices"] = json!(reclassified);
        // Distances and bending come from the smooth original surface. Measuring
        // geodesics on the displaced skull can promote a stretched seam edge
        // directly to a hard core, skipping the entire transition band.
        let head_ids: Vec<_> = (0..n).filter(|&i| skull[i]).collect();
        let head_height = head_ids
            .iter()
            .map(|&i| reference[i].y)
            .fold(f64::NEG_INFINITY, f64::max)
            - head_ids
                .iter()
                .map(|&i| reference[i].y)
                .fold(f64::INFINITY, f64::min);
        let band_m = if head_height.is_finite() {
            (head_height * 0.13).clamp(0.015, 0.025)
        } else {
            BAND_M
        };
        let (mut weight, mut blend) = blend_weights(reference, neighbours, &gpt_scalp, band_m);
        let (ramp_start,ramp_end,occiput)=occiput_levels(reference,aligned,skull,ear,neck,&visible,&registration.contours,centre);
        let lower_occiput:Vec<_>=(0..n).map(|i|{
            let back=((centre.z+0.02-reference[i].z)/0.04).clamp(0.,1.);
            let t=((reference[i].y-ramp_end)/(ramp_start-ramp_end).max(0.015)).clamp(0.,1.);
            let fade=back*(1.-t*t*(3.-2.*t));
            if region[i] && !fixed[i] && !ear[i] {fade}else{0.}
        }).collect();
        if has_back {
            let (neck_weight, _) = blend_weights(reference, neighbours, &gpt_scalp, 0.010);
            for i in 0..n {
                let posterior = ((centre.z - reference[i].z) / 0.020).clamp(0., 1.);
                let below = ((ramp_start - reference[i].y) / 0.010).clamp(0., 1.);
                let mix = posterior * below;
                weight[i] = weight[i] * (1. - mix) + neck_weight[i] * mix;

            }
        } else {
            for i in 0..n {weight[i]*=1.-lower_occiput[i];}
        }
        blend["occiput_transition"]=occiput;
        blend["occiput_transition"]["back_observed"] = json!(has_back);
        blend["occiput_transition"]["neck_blend_mm"] = json!(if has_back {10.0} else {band_m*1000.0});
        blend["occiput_transition"]["lower_observation_mode"] = json!(if has_back {"full_GPT_positional_data_joint_bending"} else {"unobserved_fill"});
        blend["head_height_mm"] = json!(head_height * 1000.0);
        blend["distance_reference"] = json!("smooth_suit_shaped_rest_before_GPT_residual");
        // The geodesic band also includes the observed skin collar and the
        // unseen neck. Their GPT hold remains zero; skin is a strong soft data
        // target, so fitted folds are not imposed as an immutable boundary.
        let soft_observed: Vec<_> = (0..n).map(|i| has_back && gpt_scalp[i]
            && reference[i].y < ramp_start && reference[i].z < centre.z + 0.02).collect();
        let from_scalp=guard::surface_distance(reference,neighbours,&gpt_scalp,band_m);
        let transition_region:Vec<_>=(0..n).map(|i| enabled && region[i] && !fixed[i]
            && transition_allowed[i] && (classes[i]!=1 || ear[i] || neck[i])
            && (gpt_scalp[i] || classes[i]==3 || from_scalp[i]<band_m || ear[i])).collect();
        let hair_eligible=gpt_scalp.clone();
        let absolute_bending:Vec<_>=(0..n).map(|i|if region[i]&&!fixed[i]&&!ear[i] {
            ((centre.z+0.02-reference[i].z)/0.04).clamp(0.,1.)*((ramp_start-reference[i].y)/0.01).clamp(0.,1.)
        }else{0.}).collect();
        let data_fidelity:Vec<_>=(0..n).map(|i|if classes[i]==1 {if neck[i]{12000.0}else{4000.0}}else{40.0*(1.-absolute_bending[i]).powi(4)+0.01}).collect();
        let mut bending = bending_rows(reference, neighbours);
        let mut bending_force=fairing_force(reference,neighbours,&absolute_bending);
        let (target, transition) = transition_field(
            reference, aligned, &measured, &weight, &transition_region, &hair_eligible, &data_fidelity, 200.0, &bending, Some(&bending_force), centre, &hair, reach,
        );
        for row in &mut bending {for value in row.values_mut(){*value*=10.0;}}
        for force in &mut bending_force {*force*=10.0;}
        let count = |k| classes.iter().filter(|&&v| v == k).count();
        let report = json!({"method":"suit_rest_hold", "enabled":enabled, "applied":enabled,
            "visibility":{"scan_skin_visible":count(1),"gpt_visible":count(2),"truly_hidden":count(3)},
            "protected_or_outside_vertices":count(0),
            "gpt_scalp_vertices":gpt_scalp.iter().filter(|v|**v).count(),
            "blend":blend,"scan_data_weight_on_gpt_core":0.0,"scan_data_weight_in_band":"1-hold_weight","hair_pull_weight_on_gpt_scalp":0.0,
            "reference":"smooth suit-shaped rest displacement with GPT core and scan boundary",
            "measurement":measurement,
            "transition":transition,
            "hair_bound":{"method":"one_sided_constraints_in_displacement_bending_system",
                "clearance_mm":HAIR_CLEARANCE_M*1000.0,"centre":a(centre)}});
        Self {
            classes,
            weight,
            target,
            gpt_scalp,
            scan_seen: seen,
            soft_observed,
            transition_region,
            hair_eligible,
            data_fidelity,
            report,
            centre,
            hair,
            reach,
            bending,
            reference: reference.to_vec(),
            bending_force,
        }
    }

    pub fn hard(&self, i: usize) -> bool {
        self.weight[i] >= 1.0 - 1e-10 && !self.soft_observed[i]
    }

    /// The hard core also gets an exact projection after every operator.
    /// The transition uses the biharmonic weight as an anchor confidence.
    pub fn add_targets(&self, sum: &mut [V], weight: &mut [f64]) {
        for i in 0..sum.len() {
            if self.hard(i) {
                weight[i] = 1.0;
                sum[i] = self.target[i] * weight[i];
            } else if self.weight[i] > 0.0 {
                sum[i] *= 1.0 - self.weight[i];
                weight[i] *= 1.0 - self.weight[i];
                let w = 40.0 * self.weight[i].powi(2) / (1.0 - self.weight[i]).max(0.1);
                sum[i] += self.target[i] * w;
                weight[i] += w;
            }
        }
    }

    /// Eliminate the held rows/columns before CG. A huge penalty would
    /// dominate CG's relative tolerance and degrade the unrelated body fit.
    pub fn solve(
        &self,
        adj: &[Vec<(u32, f64)>],
        diag: &[f64],
        rhs: &[V],
        x: &mut [V],
        iterations: usize,
    ) -> f64 {
        let mut reduced = vec![vec![]; x.len()];
        let mut diagonal = diag.to_vec();
        let mut b = rhs.to_vec();
        for i in 0..x.len() {
            if self.hard(i) {
                diagonal[i] = 1.0;
                b[i] = self.target[i];
                x[i] = self.target[i];
            } else {
                for &(j, w) in &adj[i] {
                    if self.hard(j as usize) {
                        diagonal[i] += w;
                        b[i] += self.target[j as usize] * w;
                    } else {
                        reduced[i].push((j, w));
                    }
                }
            }
        }
        let residual = super::conjugate_gradient(&reduced, &diagonal, &b, x, iterations);
        self.enforce(x);
        residual
    }

    pub fn enforce(&self, x: &mut [V]) {
        self.enforce_report(x);
    }

    pub fn enforce_report(&self, x: &mut [V]) -> Value {
        for (i, p) in x.iter_mut().enumerate() {
            if self.hard(i) {
                *p = self.target[i];
            }
        }
         
        let free: Vec<bool> = (0..x.len())
            .map(|i| self.transition_region[i] && !self.hard(i))
            .collect();
        if free.iter().any(|&v| v) {
            let (solved, report) = transition_field(
                &self.reference,
                x,
                &self.target,
                &self.weight,
                &free,
                &self.hair_eligible,
                &self.data_fidelity,
                400.0,
                &self.bending,
                Some(&self.bending_force),
                self.centre,
                &self.hair,
                self.reach,
            );
            for i in 0..x.len() {
                if free[i] {
                    x[i] = solved[i];
                }
            }
            report
        } else {
            json!({"applied":false})
        }
    }

    pub fn error_mm(&self, x: &[V]) -> f64 {
        (0..x.len())
            .filter(|&i| self.hard(i))
            .map(|i| (x[i] - self.target[i]).norm() * 1000.0)
            .fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_bending_density_preserves_cg_symmetry_and_rigid_translation_nullspace() {
        let x: Vec<_> = (0..8).map(|i| V::new(i as f64 * 0.007, 0., 0.)).collect();
        let neighbours: Vec<Vec<u32>> = (0usize..8).map(|i|
            (0usize..8).filter(|&j| i.abs_diff(j) == 1).map(|j| j as u32).collect()).collect();
        let density = [1., 1., 1., 4., 12., 20., 20., 20.];
        let rows = bending_rows_weighted(&x, &neighbours, Some(&density));
        for (i, row) in rows.iter().enumerate() {
            assert!(row.values().sum::<f64>().abs() < 1e-12);
            for (&j, &a) in row { assert!((a - rows[j][&i]).abs() < 1e-12); }
        }
        let d: Vec<_> = (0..8).map(|i| (i as f64 * 1.7).sin()).collect();
        let energy: f64 = rows.iter().enumerate().map(|(i, row)|
            d[i] * row.iter().map(|(&j, &a)| a * d[j]).sum::<f64>()).sum();
        assert!(energy > 0.);
    }

    #[test]
    fn visible_skin_reserves_unfitted_surface_despite_normal_disagreement() {
        // An unfitted forehead lies 20 mm behind a visible scan patch and
        // faces the wrong way. Its vertices are outside the forward reach;
        // reverse surface support must still reserve all three for skin.
        let x = vec![
            V::new(-0.05, -0.05, 0.0),
            V::new(0.05, -0.05, 0.0),
            V::new(0.0, 0.05, 0.0),
        ];
        let triangles = vec![[0, 1, 2]];
        let neighbours = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let positions = vec![
            [-0.001, -0.001, 0.02],
            [0.0, 0.001, 0.02],
            [0.001, -0.001, 0.02],
        ];
        let scan = Scan {
            query: crate::scan::build_query(&positions, &triangles),
            weld: crate::geom::Weld::new(&positions, 1e-7),
            normals: vec![[0.0, 0.0, -1.0]; 3],
            uvs: vec![],
            positions,
            triangles: triangles.clone(),
            triangle_material: vec![None],
            transform: nalgebra::Matrix4::identity(),
            images: vec![],
            materials: vec![],
            textures: vec![],
            samplers: vec![],
            source_hash: String::new(),
        };
        let seg = Segmentation {
            vertex_label: vec![Label::Skin; 3],
            triangle_label: vec![Label::Skin],
            triangle_piece: vec![None],
            pieces: vec![],
            piece_hair: vec![],
            piece_layer: vec![],
            piece_tight: vec![],
            piece_gap_mm: vec![],
            piece_rigid: vec![],
            strand: vec![false],
            report: json!({}),
        };
        let hold = Hold::new(
            &x,
            &x,
            &neighbours,
            &triangles,
            &[true; 3],
            &[true; 3],
            &[true; 3],
            &[false; 3],
            &[false; 3],
            &[true; 3],
            &[false;3],
            &[false;3],
            &scan,
            &seg,
            &[],
            true,
            1.0,
        );
        assert_eq!(hold.classes, vec![1, 1, 1]);
        assert!(hold.gpt_scalp.iter().all(|v| !v));
        assert!(hold.weight.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn distant_hair_cannot_attract_the_transition_outward() {
        let x: Vec<_> = (0..41)
            .map(|i| V::new((i as f64 - 20.0) * 0.001, 0.0, 0.0))
            .collect();
        let neighbours: Vec<Vec<u32>> = (0usize..41)
            .map(|i| {
                (0usize..41)
                    .filter(|&j| i.abs_diff(j) == 1)
                    .map(|j| j as u32)
                    .collect()
            })
            .collect();
        let free: Vec<_> = (0..41).map(|i| i > 1 && i < 39).collect();
        let rows = bending_rows(&x, &neighbours);
        let (hold, _) = blend_weights(&x, &neighbours, &free, 0.015);
        let target: Vec<_> = x.iter().map(|p| p + V::new(0.0, 0.0, 0.004)).collect();
        let empty = crate::scan::build_query(&[], &[]);
        let far = crate::scan::build_query(
            &[
                [-1.0, -1.0, 0.04],
                [1.0, -1.0, 0.04],
                [1.0, 1.0, 0.04],
                [-1.0, 1.0, 0.04],
            ],
            &[[0, 1, 2], [0, 2, 3]],
        );
        let (a, _) = transition_field(
            &x,
            &x,
            &target,
            &hold,
            &free,
            &free,
            &vec![0.25;free.len()],
            200.0,
            &rows,
            None,
            V::new(0.0, 0.0, -1.0),
            &empty,
            2.0,
        );
        let (b, _) = transition_field(
            &x,
            &x,
            &target,
            &hold,
            &free,
            &free,
            &vec![0.25;free.len()],
            200.0,
            &rows,
            None,
            V::new(0.0, 0.0, -1.0),
            &far,
            2.0,
        );
        assert!(a.iter().zip(b).all(|(a, b)| (a - b).norm() < 1e-12));
        assert_eq!(a[0], x[0]);
        assert_eq!(a[40], x[40]);
    }

    #[test]
    fn geodesic_biharmonic_band_has_fixed_ends_and_smooth_interior() {
        let points: Vec<_> = (0..41)
            .map(|i| V::new(i as f64 * 0.001, 0.0, 0.0))
            .collect();
        let neighbours: Vec<Vec<u32>> = (0usize..41)
            .map(|i| {
                (0..41)
                    .filter(|&j| i.abs_diff(j) == 1)
                    .map(|j| j as u32)
                    .collect()
            })
            .collect();
        let eligible: Vec<_> = (0..41).map(|i| i > 0).collect();
        let (w, report) = blend_weights(&points, &neighbours, &eligible, BAND_M);
        assert_eq!(w[0], 0.0);
        assert_eq!(w[21], 1.0);
        assert!(w[10] > 0.25 && w[10] < 0.75);
        assert!(w.windows(2).all(|p| p[1] >= p[0] - 1e-8));
        assert!(w[1] < 0.05 && w[20] - w[19] < 0.05);
        assert_eq!(report["converged"], true);
    }

    #[test]
    fn transition_carries_scan_boundary_slope_and_hair_obstacle_in_one_system() {
        let reference: Vec<_> = (0..61)
            .map(|i| V::new((i as f64 - 30.0) * 0.001, 0.0, 0.0))
            .collect();
        let neighbours: Vec<Vec<u32>> = (0usize..61)
            .map(|i| {
                (0usize..61)
                    .filter(|&j| i.abs_diff(j) == 1)
                    .map(|j| j as u32)
                    .collect()
            })
            .collect();
        let free: Vec<_> = (0..61).map(|i| i > 1 && i < 59).collect();
        let rows = bending_rows(&reference, &neighbours);
        let (hold, _) = blend_weights(&reference, &neighbours, &free, 0.02);
        let target: Vec<_> = reference
            .iter()
            .map(|p| p + V::new(0.0, 0.0, 0.008))
            .collect();
        let empty = crate::scan::build_query(&[], &[]);
        let (smooth, _) = transition_field(
            &reference,
            &reference,
            &target,
            &hold,
            &free,
            &free,
            &vec![0.25;free.len()],
            200.0,
            &rows,
            None,
            V::new(0.0, 0.0, -1.0),
            &empty,
            2.0,
        );
        assert_eq!(smooth[0], reference[0]);
        assert_eq!(smooth[60], reference[60]);
        assert!(smooth[30].z > 0.0079);
        assert!(
            smooth
                .windows(3)
                .all(|p| (p[0].z - 2.0 * p[1].z + p[2].z).abs() < 0.0003)
        );
        let hair = crate::scan::build_query(
            &[
                [-1.0, -1.0, 0.003],
                [1.0, -1.0, 0.003],
                [1.0, 1.0, 0.003],
                [-1.0, 1.0, 0.003],
            ],
            &[[0, 1, 2], [0, 2, 3]],
        );
        let (bounded, report) = transition_field(
            &reference,
            &reference,
            &target,
            &hold,
            &free,
            &free,
            &vec![0.25;free.len()],
            200.0,
            &rows,
            None,
            V::new(0.0, 0.0, -1.0),
            &hair,
            2.0,
        );
        assert_eq!(report["hair_converged"], true);
        assert!(bounded.iter().all(|p| p.z <= 0.001201));
        assert!(
            bounded
                .windows(3)
                .all(|p| (p[0].z - 2.0 * p[1].z + p[2].z).abs() < 0.0003)
        );
        assert_eq!(bounded[0], reference[0]);
    }

    #[test]
    fn absolute_bending_removes_a_reference_shelf_between_fixed_ends() {
        let reference:Vec<_>=(0..61).map(|i|V::new(i as f64*0.001,0.,if i>=30 {0.012}else{0.})).collect();
        let neighbours:Vec<Vec<u32>>=(0usize..61).map(|i|(0usize..61).filter(|&j|i.abs_diff(j)==1).map(|j|j as u32).collect()).collect();
        let free:Vec<_>=(0..61).map(|i|i>1&&i<59).collect();
        let rows=bending_rows(&reference,&neighbours);
        let force=fairing_force(&reference,&neighbours,&vec![1.;61]);
        let empty=crate::scan::build_query(&[],&[]);
        let (smooth,_) = transition_field(&reference,&reference,&reference,&vec![0.;61],&free,&vec![false;61],
            &vec![0.001;61],400.,&rows,Some(&force),V::zeros(),&empty,1.);
        let energy=|x:&[V]|(0..61).map(|i|x[i].z*rows[i].iter().map(|(&j,&a)|x[j].z*a).sum::<f64>()).sum::<f64>();
        assert!(energy(&smooth)<energy(&reference)*0.2);
        assert_eq!(smooth[0],reference[0]);assert_eq!(smooth[60],reference[60]);
    }

    #[test]
    fn hard_scalp_discards_scan_targets_and_cannot_drift() {
        let hold = Hold {
            soft_observed: vec![false; 2],
            classes: vec![2, 1],
            weight: vec![1.0, 0.0],
            target: vec![V::new(0.0, 1.0, 0.0), V::zeros()],
            gpt_scalp: vec![true, false],
            scan_seen: vec![],
            transition_region: vec![true,false],
            hair_eligible: vec![true,false],
            data_fidelity:vec![0.05,40.0],
            report: json!({}),
            centre: V::zeros(),
            hair: crate::scan::build_query(&[], &[]),
            reach: 0.0,
            bending: vec![BTreeMap::new(); 2],
            reference: vec![V::zeros(); 2],
            bending_force: vec![V::zeros(); 2],
        };
        let mut sum = vec![V::repeat(20.0); 2];
        let mut weight = vec![100.0; 2];
        hold.add_targets(&mut sum, &mut weight);
        assert!((sum[0] / weight[0] - hold.target[0]).norm() < 1e-12);
        assert_eq!(sum[1], V::repeat(20.0));
        let mut x = vec![V::repeat(8.0); 2];
        hold.enforce(&mut x);
        assert_eq!(hold.error_mm(&x), 0.0);
        assert_eq!(x[1], V::repeat(8.0));
    }
}
