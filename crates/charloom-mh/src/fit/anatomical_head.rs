 use super::{hidden_head, silhouette};
use crate::{geom::{V, Weld, a, v}, guide::{Camera, SuitContour}, scan::{Scan, build_query}, segment::{Label, Segmentation}};
use anyhow::{Result, ensure};
use nalgebra::{DMatrix, DVector};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::{Path, PathBuf}};

mod pinned;
mod symmetry;
#[derive(Debug, Clone, Deserialize)]
pub struct SymmetrySettings {
    strength: f64,
}
#[derive(Debug, Clone, Deserialize)]
pub struct PinnedSettings {
    pin_falloff_m: f64, residual_below_cap_m: f64, residual_centres: usize,
    bending_weight: f64, projection_iterations: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Parameter {
    name: String, min: f64, max: f64,
    plus: Vec<[f64; 3]>, minus: Vec<[f64; 3]>,
}
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Basis { prior_sha256: String, vertex_count: usize, parameters: Vec<Parameter> }
#[derive(Debug, Clone, Deserialize)]
pub struct Observation { camera: Camera, samples: Vec<[f64; 2]>, weights: Vec<f64> }
#[derive(Debug, Clone, Deserialize)]
pub struct Input {
    name: String, basis_path: PathBuf, reference_report: PathBuf, reference_sha256: String,
    initial: BTreeMap<String, f64>, inion_y_m: f64, views: Vec<Observation>,
    hidden_mask: Vec<bool>, visible_mask: Vec<bool>, held_out_used: bool,
    morph_iterations: usize, residual_iterations: usize, skin_weight: f64,
    hair_weight: f64, prior_weight: f64, residual_support_m: f64,
    residual_fade_m: f64, residual_cap_m: f64,
    #[serde(default)] pub x120: Option<PinnedSettings>,
    #[serde(default)] pub x121: Option<SymmetrySettings>,
    #[serde(skip)] pub basis: Basis,
}
pub fn load(path: &Path, prior_path: &Path) -> Result<Input> {
    let mut input: Input = serde_json::from_slice(&std::fs::read(path)?)?;
    input.basis = serde_json::from_slice(&std::fs::read(&input.basis_path)?)?;
    ensure!(input.basis.prior_sha256 == format!("{:x}", Sha256::digest(std::fs::read(prior_path)?)),
        "anatomical basis prior bytes differ");
    ensure!(format!("{:x}", Sha256::digest(std::fs::read(&input.reference_report)?)) == input.reference_sha256,
        "frozen x118 target report changed");
    ensure!(!input.held_out_used && input.views.len() >= if input.x121.is_some() {3} else {4}, "training views missing or contaminated by held-out data");
    if let Some(settings) = &input.x121 {
        ensure!(input.x120.is_some() && settings.strength > 0. && settings.strength.is_finite(), "invalid x121 symmetry settings");
    }
    if let Some(settings) = &input.x120 {
        ensure!(settings.pin_falloff_m >= 0.020 && (12..=20).contains(&settings.residual_centres)
            && input.residual_support_m >= 0.060 && input.residual_fade_m >= 0.030
            && input.residual_cap_m > 0. && input.residual_cap_m <= 0.030
            && settings.residual_below_cap_m > 0. && settings.residual_below_cap_m <= 0.010
            && settings.bending_weight > 0., "invalid x120 pinned residual bounds");
    } else {
        ensure!(input.residual_support_m >= 0.030 && input.residual_fade_m >= 0.030
            && input.residual_cap_m <= 0.006, "invalid anatomical residual bounds");
    }
    let n=input.basis.vertex_count;
    ensure!(input.visible_mask.len()==n && input.hidden_mask.len()==n, "anatomical mask indexing differs");
    for p in &input.basis.parameters {
        ensure!(p.plus.len()==n && p.minus.len()==n && p.min>=-1. && p.max<=1., "invalid anatomical morph");
        ensure!(p.plus.iter().chain(&p.minus).flatten().all(|x|x.is_finite()), "nonfinite anatomical morph");
    }
    for view in &input.views {
        ensure!(view.samples.len()==view.weights.len() && !view.samples.is_empty(), "invalid frozen view weights");
    }
    Ok(input)
}

struct Field { plus: Vec<V>, minus: Vec<V>, lower: f64, upper: f64 }
fn field_value(field: &Field, weight: f64, i: usize) -> V {
    if weight >= 0. { field.plus[i]*weight } else { field.minus[i]*(-weight) }
}
fn derivative(field: &Field, weight: f64, i: usize) -> V {
    if weight.abs()<1e-8 && field.lower>=0. { field.plus[i] }
    else if weight.abs()<1e-8 { (field.plus[i]-field.minus[i])*0.5 }
    else if weight>0. { field.plus[i] } else { -field.minus[i] }
}
fn displacement(fields: &[Field], weights: &[f64], initial: &[f64], n: usize) -> Vec<V> {
    (0..n).map(|i| fields.iter().zip(weights).zip(initial)
        .fold(V::zeros(), |sum, ((f,&w),&b)|sum+field_value(f,w,i)-field_value(f,b,i))).collect()
}
fn bounded_start(base: &[V], fields: &[Field], initial: &[f64]) -> (Vec<f64>, Vec<V>) {
     
    let weights: Vec<f64> = fields.iter().zip(initial).map(|(f,&w)|w.clamp(f.lower,f.upper)).collect();
    let delta=displacement(fields,&weights,initial,base.len());
    let x=base.iter().zip(delta).map(|(p,d)|p+d).collect();
    (weights,x)
}
fn cap_coefficients(fields: &[Field], weights: &mut [f64], n: usize, cap: Option<f64>) {
    if let Some(cap)=cap {
        let max=displacement(fields,weights,&vec![0.;weights.len()],n).iter().map(V::norm).fold(0.,f64::max);
        if max>cap { for w in weights { *w*=cap/max; } }
    }
}
fn wendland(distance: f64, support: f64) -> f64 {
    let r=(distance/support).clamp(0.,1.);(1.-r).powi(4)*(4.*r+1.)
}
fn fade(y: f64, inion: f64, width: f64) -> f64 {
    let t=((y-inion)/width).clamp(0.,1.);t*t*t*(10.+t*(-15.+6.*t))
}

struct Row { ids: [usize;3], bary: [f64;3], axis: V, delta: f64, weight: f64 }
fn at(points: &[V], row: &Row) -> V {
    (0..3).fold(V::zeros(),|s,k|s+points[row.ids[k]]*row.bary[k])
}
struct Data<'a> {
    input: &'a Input, views: Vec<silhouette::View>, triangles: &'a [[u32;3]],
    region: Vec<bool>, visible: Vec<bool>, hidden: Vec<bool>,
    skin: charloom_engine::surface_query::SurfaceQuery,
    skin_samples: Vec<[f64;3]>, hair: charloom_engine::surface_query::SurfaceQuery,
    centre: V,
}
impl Data<'_> {
    fn rows(&self,x:&[V])->(Vec<Row>,Value) {
        let (constraints,_,stats)=silhouette::rendered_constraints(&self.views,x,self.triangles,&self.region,0.12);
        let mut rows=vec![];let mut gpt=0;
        let total=constraints.iter().filter(|r|r.vertices.iter().zip(r.bary).map(|(&i,b)|if self.visible[i as usize]{b}else{0.}).sum::<f64>()<0.5)
            .map(|r|r.weight.max(0.)).sum::<f64>().max(1.);
        for c in constraints {
            // Visible scan skin always owns the correspondence, even on a GPT rim.
            if c.vertices.iter().zip(c.bary).map(|(&i,b)|if self.visible[i as usize]{b}else{0.}).sum::<f64>()>=0.5 {continue;}
            if c.weight<=0. {continue;}
            for axis in c.axes { rows.push(Row { ids:c.vertices.map(|i|i as usize), bary:c.bary,
                axis, delta:c.delta.dot(&axis),weight:c.weight/total }); }
            gpt+=1;
        }
        let mut skinrows=vec![];
        for (i,&visible) in self.visible.iter().enumerate() {
            if !visible {continue;}
            if let Some(hit)=self.skin.closest_point(a(x[i]),0.05,None) {
                for axis in [V::x(),V::y(),V::z()] { skinrows.push(Row {ids:[i;3],bary:[1.,0.,0.],axis,
                    delta:(v(hit.point)-x[i]).dot(&axis),weight:1.}); }
            }
        }
        let body=build_query(&x.iter().map(|p|a(*p)).collect::<Vec<_>>(),self.triangles);
        for &q in &self.skin_samples {
            if let Some(hit)=body.closest_point(q,0.05,None) {
                let ids=self.triangles[hit.source[1]].map(|i|i as usize);
                if !ids.iter().any(|&i|self.region[i]){continue;}
                for axis in [V::x(),V::y(),V::z()] {skinrows.push(Row {ids,bary:hit.barycentric,axis,
                    delta:(v(q)-v(hit.point)).dot(&axis),weight:1.});}
            }
        }
        let skin_count=skinrows.len();let w=self.input.skin_weight/skin_count.max(1)as f64;
        for r in &mut skinrows{r.weight=w;}
        rows.extend(skinrows);
        let mut hairrows=vec![];let mut hair_max:f64=0.;
        for (i,&hidden) in self.hidden.iter().enumerate() {
            if !hidden || self.visible[i] {continue;}
            if let Some(bound)=hidden_head::hair_ceiling(&self.hair,self.centre,x[i],0.5,0.0018) {
                let dir=(x[i]-self.centre).normalize();let excess=(x[i]-bound).norm();hair_max=hair_max.max(excess);
                hairrows.push(Row {ids:[i;3],bary:[1.,0.,0.],axis:dir,delta:-excess,weight:1.});
            }
        }
        let contacts=hairrows.len();let hw=self.input.hair_weight/self.hidden.iter().filter(|&&v|v).count().max(1)as f64;
        for r in &mut hairrows{r.weight=hw;}
        rows.extend(hairrows);
        (rows,json!({"gpt_surface_pairs":gpt,"skin_rows":skin_count,"hair_contacts":contacts,
            "hair_max_violation_mm":hair_max*1000.,"views":stats.iter().zip(&self.input.views).map(|(s,v)|
                json!({"view":v.camera.id,"pairs":s.0,"p50_mm":s.1*1000.,"p95_mm":s.2*1000.})).collect::<Vec<_>>()}))
    }
}

fn optimize(data:&Data,base:&[V],fields:&[Field],initial:&[f64],iterations:usize,prior:f64,cap:Option<f64>) -> (Vec<V>,Vec<f64>,Vec<Value>) {
    let n=base.len();let k=fields.len();let (mut weights,mut x)=bounded_start(base,fields,initial);let mut history=vec![];
    for iteration in 0..iterations {
        let (rows,stats)=data.rows(&x);let mut matrix=DMatrix::<f64>::zeros(k,k);let mut rhs=DVector::<f64>::zeros(k);
        for row in &rows {
            let jac:Vec<f64>=fields.iter().zip(&weights).map(|(f,&w)|(0..3).map(|j|derivative(f,w,row.ids[j]).dot(&row.axis)*row.bary[j]).sum()).collect();
            for a in 0..k {rhs[a]+=row.weight*jac[a]*row.delta;for b in 0..=a {matrix[(a,b)]+=row.weight*jac[a]*jac[b];}}
        }
        for a in 0..k {for b in 0..a {matrix[(b,a)]=matrix[(a,b)];}matrix[(a,a)]+=prior+1e-12;rhs[a]-=prior*weights[a];}
        // Deterministic box-constrained coordinate descent on the small system.
        let mut step=DVector::<f64>::zeros(k);
        for _ in 0..100 {for j in 0..k {
            let off=(0..k).filter(|&l|l!=j).map(|l|matrix[(j,l)]*step[l]).sum::<f64>();
            step[j]=((rhs[j]-off)/matrix[(j,j)]).clamp((fields[j].lower-weights[j]).max(-0.25),(fields[j].upper-weights[j]).min(0.25));
        }}
        let energy=|candidate:&[V],ws:&[f64]|rows.iter().map(|r|{
            let prediction=(at(candidate,r)-at(&x,r)).dot(&r.axis);
            r.weight*(prediction-r.delta).powi(2)
        }).sum::<f64>()+prior*ws.iter().map(|w|w*w).sum::<f64>();
        let before=energy(&x,&weights);let mut accepted=0.;let mut after=before;let mut movement:f64=0.;
        for alpha in [1.,0.5,0.25,0.125,0.0625,0.03125] {
            let mut next:Vec<f64>=(0..k).map(|j|(weights[j]+step[j]*alpha).clamp(fields[j].lower,fields[j].upper)).collect();
            cap_coefficients(fields,&mut next,n,cap);
            let d=displacement(fields,&next,initial,n);let candidate:Vec<V>=base.iter().zip(d).map(|(p,d)|p+d).collect();
            let score=energy(&candidate,&next);
            if score<before-1e-15 {movement=x.iter().zip(&candidate).map(|(a,b)|(a-b).norm()).fold(0.,f64::max);weights=next;x=candidate;accepted=alpha;after=score;break;}
        }
        history.push(json!({"iteration":iteration,"energy_before":before,"energy_after":after,"step_fraction":accepted,"max_step_mm":movement*1000.,"correspondences":stats}));
        if accepted==0. || movement<0.000002 {break;}
    }
    (x,weights,history)
}

#[allow(clippy::too_many_arguments)]
pub fn fit(input:&Input,zero:bool,weld:&Weld,reference:&[V],triangles:&[[u32;3]],head:&[bool],neck:&[bool],cavity:&[bool],x:&mut [V],scan:&Scan,seg:&Segmentation)->Value {
    if input.x120.is_some() {
        return pinned::fit(input,weld,reference,triangles,head,neck,cavity,x,scan,seg);
    }
    assert_eq!(weld.id.len(),input.basis.vertex_count);
    let n=x.len();let baseline=x.to_vec();
    let initial:Vec<f64>=input.basis.parameters.iter().map(|p|input.initial.get(&p.name).copied().unwrap_or(0.)).collect();
    if zero {return json!({"method":"MakeHuman_morph_space","identity_probe":true,"applied":false,
        "zero_added_max_delta_mm":0.,"morph_weights":input.basis.parameters.iter().zip(&initial).map(|(p,w)|json!({"name":p.name,"initial":w,"solved":w,"added":0.})).collect::<Vec<_>>(),
        "inion_y_m":input.inion_y_m,"per_vertex_skull_below_inion":false,"old_skull_holds":false});}
    let visible:Vec<bool>=weld.representative.iter().map(|&i|input.visible_mask[i]).collect();
    let hidden:Vec<bool>=weld.representative.iter().map(|&i|input.hidden_mask[i]).collect();
    let ids:Vec<usize>=(0..n).filter(|&i|head[i]&&visible[i]&&!cavity[i]).collect();
    let (rotation,scale,translation)=hidden_head::reference_frame(reference,&baseline,&ids);
    let fields:Vec<Field>=input.basis.parameters.iter().map(|p|Field {
        plus:weld.representative.iter().map(|&i|rotation*v(p.plus[i])*scale).collect(),
        minus:weld.representative.iter().map(|&i|rotation*v(p.minus[i])*scale).collect(),lower:p.min,upper:p.max}).collect();
    let region:Vec<bool>=(0..n).map(|i|(head[i]||neck[i])&&!cavity[i]).collect();
    let seen=hidden_head::head_visibility(scan,&baseline,head);
    let skin_triangles:Vec<_>=scan.triangles.iter().enumerate().filter(|(t,tri)|seg.triangle_label[*t]==Label::Skin && !seg.is_prop(*t) && tri.iter().any(|&i|seen[i as usize])).map(|(_,t)|*t).collect();
    let hair_triangles:Vec<_>=scan.triangles.iter().enumerate().filter(|(t,_)|seg.triangle_label[*t]==Label::Hair && !seg.is_prop(*t)).map(|(_,t)|*t).collect();
    let body=build_query(&baseline.iter().map(|p|a(*p)).collect::<Vec<_>>(),triangles);
    let mut skin_samples=vec![];
    for &s in &scan.weld.representative {
        if !seen[s] || seg.vertex_label[s]!=Label::Skin {continue;}
        if let Some(h)=body.closest_point(scan.positions[s],0.03,None) {
            if triangles[h.source[1]].iter().any(|&i|region[i as usize]) {skin_samples.push(scan.positions[s]);}
        }
    }
     
    let stride=(skin_samples.len()/6000).max(1);skin_samples=skin_samples.into_iter().step_by(stride).collect();
    let centre=(0..n).filter(|&i|head[i]).fold(V::zeros(),|s,i|s+baseline[i])/head.iter().filter(|&&v|v).count().max(1)as f64;
    let views=input.views.iter().map(|o|silhouette::View::new(&SuitContour {camera:o.camera.clone(),curves:vec![o.samples.clone()],weights:vec![o.weights.clone()],skin_anchors:vec![],rotation_limits_deg:[0.,0.],scale_tied_to:None,skin_outline_anchors:vec![]})).collect();
    let data=Data {input,views,triangles,region,visible,hidden,skin:build_query(&scan.positions,&skin_triangles),skin_samples,hair:build_query(&scan.positions,&hair_triangles),centre};
    let (morphed,weights,history)=optimize(&data,&baseline,&fields,&initial,input.morph_iterations,input.prior_weight,None);
    // Six smooth, broad fields (18 coefficients), supported in native scalp
    // space. The native inion and zero band are frozen BEFORE morph fitting.
    let candidates:Vec<usize>=(0..n).filter(|&i|head[i]&&!cavity[i]&&data.hidden[i]&&baseline[i].y>input.inion_y_m+0.020).collect();
    let mut centres=vec![];
    if let Some(&top)=candidates.iter().max_by(|&&a,&&b|baseline[a].y.total_cmp(&baseline[b].y)) {centres.push(top);}
    while centres.len()<6 && centres.len()<candidates.len() {
        let best=candidates.iter().filter(|i|!centres.contains(i)).max_by(|&&a,&&b|{
            let distance=|i:usize|centres.iter().map(|&j|(baseline[i]-baseline[j]).norm_squared()).fold(f64::INFINITY,f64::min);
            distance(a).total_cmp(&distance(b))
        });if let Some(&best)=best {centres.push(best);}else{break;}
    }
    let mut residual_fields=vec![];
    for &c in &centres {for axis in [V::x(),V::y(),V::z()] {
        let plus:Vec<V>=(0..n).map(|i|if head[i]&&!cavity[i] {axis*input.residual_cap_m*wendland((baseline[i]-baseline[c]).norm(),input.residual_support_m)*fade(baseline[i].y,input.inion_y_m,input.residual_fade_m)}else{V::zeros()}).collect();
        residual_fields.push(Field {minus:plus.iter().map(|p|-p).collect(),plus,lower:-1.,upper:1.});
    }}
    let (result,residual_weights,residual_history)=optimize(&data,&morphed,&residual_fields,&vec![0.;residual_fields.len()],input.residual_iterations,input.prior_weight*0.25,Some(input.residual_cap_m));
    let residual:Vec<V>=result.iter().zip(&morphed).map(|(a,b)|a-b).collect();
    let max=residual.iter().map(V::norm).fold(0.,f64::max);
    let lower=(0..n).filter(|&i|baseline[i].y<=input.inion_y_m).map(|i|residual[i].norm()).fold(0.,f64::max);
    assert!(max<=0.0060000001 && lower<1e-12);
    x.copy_from_slice(&result);
    let (_,final_stats)=data.rows(x);
    json!({"method":"MakeHuman_morph_space","applied":true,"identity_probe":false,"name":input.name,
        "basis":input.basis_path,"target_report":input.reference_report,"held_out_used":false,"old_skull_holds":false,
        "initial_projection":input.basis.parameters.iter().zip(&initial).filter(|(p,w)|**w<p.min||**w>p.max).map(|(p,w)|json!({"name":p.name,"x110_measurement":w,"first_feasible_weight":w.clamp(p.min,p.max)})).collect::<Vec<_>>(),
        "transport":{"method":"one_whole_head_similarity_from_visible_skin","scale":scale,"rotation":rotation.as_slice(),"translation":a(translation)},
        "morph_weights":input.basis.parameters.iter().zip(&initial).zip(&weights).map(|((p,b),w)|json!({"name":p.name,"initial":b,"solved":w,"added":w-b,"min":p.min,"max":p.max})).collect::<Vec<_>>(),
        "morph_history":history,"residual_history":residual_history,"final_constraints":final_stats,
        "inion_y_m":input.inion_y_m,"per_vertex_skull_below_inion":false,
        "residual":{"method":"six_Wendland_C2_RBFs_18_coefficients","support_m":input.residual_support_m,"fade_m":input.residual_fade_m,
            "cap_mm":input.residual_cap_m*1000.,"max_mm":max*1000.,"below_inion_max_mm":lower*1000.,
            "centres":centres.iter().map(|&i|a(baseline[i])).collect::<Vec<_>>(),"coefficients":residual_weights,
            "cap_method":"global_coefficient_scaling_preserves_frequency"},"skin_samples":data.skin_samples.len(),"hair_clearance_mm":1.8})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn residual_is_smooth_and_exactly_zero_at_and_below_inion() {
        assert_eq!(fade(-0.001,0.,0.03),0.);assert_eq!(fade(0.,0.,0.03),0.);
        assert!(fade(0.000001,0.,0.03)<1e-10);assert_eq!(wendland(0.06,0.06),0.);
        assert_eq!(wendland(0.,0.06),1.);assert!(wendland(0.03,0.06)>0.);
    }
    #[test] fn zero_added_morphs_are_bit_exact_and_global_cap_preserves_field() {
        let f=Field {plus:vec![V::new(0.01,0.,0.),V::new(0.02,0.,0.)],minus:vec![V::new(-0.01,0.,0.),V::new(-0.02,0.,0.)],lower:-1.,upper:1.};
        let fields=vec![f];let d=displacement(&fields,&[-0.3],&[-0.3],2);assert!(d.iter().all(|d|*d==V::zeros()));
        let mut w=vec![1.];cap_coefficients(&fields,&mut w,2,Some(0.006));let d=displacement(&fields,&w,&[0.],2);
        assert!((d[1].norm()-0.006).abs()<1e-12);assert!((d[1].x/d[0].x-2.).abs()<1e-12);
    }
    #[test] fn out_of_range_x110_measurement_is_projected_without_losing_its_origin() {
        let fields=vec![Field {plus:vec![V::new(0.,0.02,0.)],minus:vec![V::new(0.,-0.02,0.)],lower:-1.,upper:1.}];
        let base=vec![V::new(0.,1.5,0.)];let initial=[1.90625];
        let (weights,x)=bounded_start(&base,&fields,&initial);
        assert_eq!(weights,[1.]);assert!((x[0].y-(1.5-0.90625*0.02)).abs()<1e-12);
        let lower=(fields[0].lower-weights[0]).max(-0.25);
        let upper=(fields[0].upper-weights[0]).min(0.25);
        assert!(lower<=upper);assert_eq!(0.1_f64.clamp(lower,upper),0.);
        let (inside,unchanged)=bounded_start(&base,&fields,&[0.5]);
        assert_eq!(inside,[0.5]);assert_eq!(unchanged,base);
    }
}
