 use super::*;
use std::{cmp::Ordering, collections::BinaryHeap};

#[derive(PartialEq)]
struct Queue(f64, usize);
impl Eq for Queue {}
impl PartialOrd for Queue { fn partial_cmp(&self, other:&Self)->Option<Ordering>{Some(self.cmp(other))} }
impl Ord for Queue { fn cmp(&self,other:&Self)->Ordering {other.0.total_cmp(&self.0).then_with(||other.1.cmp(&self.1))} }

fn pin_envelope(base:&[V], triangles:&[[u32;3]], visible:&[bool], region:&[bool], width:f64)->Vec<f64> {
    let mut neighbours=vec![vec![];base.len()];
    for t in triangles {for j in 0..3 {let a=t[j]as usize;let b=t[(j+1)%3]as usize;
        let length=(base[a]-base[b]).norm();neighbours[a].push((b,length));neighbours[b].push((a,length));}}
    let mut distance=vec![f64::INFINITY;base.len()];let mut queue=BinaryHeap::new();
    for i in 0..base.len() {if visible[i] || !region[i] {distance[i]=0.;queue.push(Queue(0.,i));}}
    while let Some(Queue(d,i))=queue.pop() {
        if d>distance[i] || d>=width {continue;}
        for &(j,length) in &neighbours[i] {let next=d+length;if next<distance[j] {distance[j]=next;queue.push(Queue(next,j));}}
    }
    distance.into_iter().map(|d|fade(d,0.,width)).collect()
}

fn cap_field(fields:&[Field],weights:&mut[f64],caps:&[f64]) {
    let d=displacement(fields,weights,&vec![0.;weights.len()],caps.len());
    let ratio=d.iter().zip(caps).map(|(v,c)|v.norm()/c).fold(1.,f64::max);
    if ratio>1. {for w in weights {*w/=ratio;}}
}

fn bending(centres:&[V],strength:f64)->DMatrix<f64> {
    let k=centres.len()*3;let mut penalty=DMatrix::zeros(k,k);
    for (c,i) in centres.iter().enumerate() {
        let mut near:Vec<_>=centres.iter().enumerate().filter(|(j,_)|*j!=c)
            .map(|(j,v)|(j,(i-v).norm())).collect();
        near.sort_by(|a,b|a.1.total_cmp(&b.1));near.truncate(4);
        let sum=near.iter().map(|(_,d)|1./d.max(0.001)).sum::<f64>();
        for axis in 0..3 {let mut row=vec![(c*3+axis,1.)];
            row.extend(near.iter().map(|(j,d)|(j*3+axis,-1./d.max(0.001)/sum)));
            for &(a,wa) in &row {for &(b,wb) in &row {penalty[(a,b)]+=strength*wa*wb;}}
        }
    }
    penalty
}

fn hair_rows(data:&Data,x:&[V])->Vec<Row> {
    let mut rows=vec![];
    for (i,&hidden) in data.hidden.iter().enumerate() {
        if !hidden || data.visible[i] {continue;}
        if let Some(bound)=hidden_head::hair_ceiling(&data.hair,data.centre,x[i],0.5,0.0018) {
            rows.push(Row {ids:[i;3],bary:[1.,0.,0.],axis:(x[i]-data.centre).normalize(),
                delta:-(x[i]-bound).norm(),weight:data.input.hair_weight/data.hidden.iter().filter(|&&b|b).count().max(1)as f64});
        }
    }
    rows
}

fn field_feasible(reference:&[V],candidate:&[V],triangles:&[[u32;3]])->bool {
    triangles.iter().all(|t| {
        let [a,b,c]=t.map(|i|i as usize);
        let normal=(reference[b]-reference[a]).cross(&(reference[c]-reference[a]));
        let next=(candidate[b]-candidate[a]).cross(&(candidate[c]-candidate[a]));
        if normal.norm_squared()>1e-20 && normal.dot(&next)<=0. {return false;}
        [(a,b),(b,c),(c,a)].iter().all(|&(i,j)| {
            let length=(reference[i]-reference[j]).norm();
            length<1e-8 || ((candidate[i]-reference[i])-(candidate[j]-reference[j])).norm()<=length*0.75+1e-12
        })
    })
}

#[allow(clippy::too_many_arguments)]
fn solve(data:&Data,reference:&[V],base:&[V],fields:&[Field],origin:&[f64],start:&[f64],iterations:usize,
    prior:f64,penalty:&DMatrix<f64>,caps:Option<&[f64]>,projection:bool,sym:Option<&symmetry::Symmetry>)->(Vec<V>,Vec<f64>,Vec<Value>) {
    let n=base.len();let k=fields.len();let mut weights=start.to_vec();
    let positions=|w:&[f64]|base.iter().zip(displacement(fields,w,origin,n)).map(|(p,d)|p+d).collect::<Vec<_>>();
    let mut x=positions(&weights);let mut history=vec![];
    for iteration in 0..iterations {
        let (rows,stats)=if projection {let r=hair_rows(data,&x);let stats=json!({"hair_contacts":r.len(),
            "hair_max_violation_mm":r.iter().map(|r|r.delta.abs()*1000.).fold(0.,f64::max)});(r,stats)}else{data.rows(&x)};
        if rows.is_empty() {break;}
        let mut matrix=DMatrix::<f64>::zeros(k,k);let mut rhs=DVector::<f64>::zeros(k);
        for row in &rows {
            let jac:Vec<_>=fields.iter().zip(&weights).map(|(f,&w)|(0..3).map(|j|derivative(f,w,row.ids[j]).dot(&row.axis)*row.bary[j]).sum::<f64>()).collect();
            for a in 0..k {rhs[a]+=row.weight*jac[a]*row.delta;for b in 0..=a {matrix[(a,b)]+=row.weight*jac[a]*jac[b];}}
        }
        for a in 0..k {for b in 0..a {matrix[(b,a)]=matrix[(a,b)];}}
        matrix+=penalty;
        if let Some(sym)=sym {sym.add_system(fields,&weights,&x,reference,&mut matrix,&mut rhs);}
        for a in 0..k {matrix[(a,a)]+=prior+1e-12;
            rhs[a]-=prior*(weights[a]-if projection {start[a]}else{0.});
            rhs[a]-=(0..k).map(|b|penalty[(a,b)]*weights[b]).sum::<f64>();}
        let mut step=DVector::<f64>::zeros(k);
        for _ in 0..100 {for j in 0..k {
            let off=(0..k).filter(|&l|l!=j).map(|l|matrix[(j,l)]*step[l]).sum::<f64>();
            step[j]=((rhs[j]-off)/matrix[(j,j)]).clamp((fields[j].lower-weights[j]).max(-0.25),(fields[j].upper-weights[j]).min(0.25));
        }}
        let energy=|candidate:&[V],ws:&[f64]|rows.iter().map(|r|{
            let prediction=(at(candidate,r)-at(&x,r)).dot(&r.axis);r.weight*(prediction-r.delta).powi(2)
        }).sum::<f64>()+prior*ws.iter().enumerate().map(|(j,w)|(w-if projection {start[j]}else{0.}).powi(2)).sum::<f64>()
            +(0..k).map(|a|(0..k).map(|b|ws[a]*penalty[(a,b)]*ws[b]).sum::<f64>()).sum::<f64>();
        let energy=|candidate:&[V],ws:&[f64]|energy(candidate,ws)+sym.map_or(0.,|s|s.energy(candidate,reference));
        let view_before=if sym.is_some() {symmetry::view_distances(data,&x,reference)}else{vec![]};
        let mut view_after=view_before.clone();let mut rejected_views=0;
        let before=energy(&x,&weights);let mut accepted=0.;let mut after=before;let mut movement:f64=0.;
        let mut direction_correction=false;
        'directions: for attempt in 0..if sym.is_some() {2} else {1} {
          let trial_step=if attempt==0 {step.clone()}else{direction_correction=true;symmetry::common_direction(data,&x,reference,fields,&weights,&matrix,&step)};
          for alpha in [1.,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.00390625] {
            let mut next:Vec<_>=(0..k).map(|j|(weights[j]+trial_step[j]*alpha).clamp(fields[j].lower,fields[j].upper)).collect();
            if let Some(caps)=caps {cap_field(fields,&mut next,caps);}
            let candidate=positions(&next);
            if !field_feasible(reference,&candidate,data.triangles) {continue;}
            let score=energy(&candidate,&next);
            if score<before-1e-15 {
                if sym.is_some() {
                    let trial=symmetry::view_distances(data,&candidate,reference);
                    if !symmetry::no_view_worse(&view_before,&trial) {rejected_views+=1;continue;}
                    view_after=trial;
                }
                movement=x.iter().zip(&candidate).map(|(a,b)|(a-b).norm()).fold(0.,f64::max);
                weights=next;x=candidate;accepted=alpha;after=score;break 'directions;}
        }
          }
        history.push(json!({"iteration":iteration,"energy_before":before,"energy_after":after,"step_fraction":accepted,"max_step_mm":movement*1000.,"correspondences":stats,"view_before":symmetry::view_json(data,&view_before),"view_after":symmetry::view_json(data,&view_after),"view_rejected_steps":rejected_views,"common_direction_attempted":direction_correction}));
        if accepted==0. || movement<0.000002 {break;}
    }
    (x,weights,history)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fit(input:&Input,weld:&Weld,reference:&[V],triangles:&[[u32;3]],head:&[bool],neck:&[bool],cavity:&[bool],x:&mut[V],scan:&Scan,seg:&Segmentation)->Value {
    let settings=input.x120.as_ref().unwrap();assert_eq!(weld.id.len(),input.basis.vertex_count);
    let n=x.len();let baseline=x.to_vec();
    let initial:Vec<_>=input.basis.parameters.iter().map(|p|input.initial.get(&p.name).copied().unwrap_or(0.)).collect();
    // Union over seam duplicates: one frozen visible corner pins the welded point.
    let mut visible=vec![false;n];let mut hidden=vec![false;n];
    for (i,&w) in weld.id.iter().enumerate() {visible[w]|=input.visible_mask[i];hidden[w]|=input.hidden_mask[i];}
    let region:Vec<_>=(0..n).map(|i|(head[i]||neck[i])&&!cavity[i]).collect();
    let mut support=visible.clone();
    for t in triangles {if t.iter().any(|&i|visible[i as usize]) {for &i in t {support[i as usize]=true;}}}
    let pin=pin_envelope(&baseline,triangles,&support,&region,settings.pin_falloff_m);
    let ids:Vec<_>=(0..n).filter(|&i|head[i]&&visible[i]&&!cavity[i]).collect();
    let (rotation,scale,translation)=hidden_head::reference_frame(reference,&baseline,&ids);
    let fields:Vec<_>=input.basis.parameters.iter().zip(&initial).map(|(p,&b)|{
        let frozen=matches!(p.name.as_str(),"head-scale-vert"|"head-scale-horiz"|"head-scale-depth");
        Field {plus:weld.representative.iter().enumerate().map(|(w,&i)|rotation*v(p.plus[i])*scale*pin[w]).collect(),
            minus:weld.representative.iter().enumerate().map(|(w,&i)|rotation*v(p.minus[i])*scale*pin[w]).collect(),
            lower:if frozen {b}else{p.min},upper:if frozen {b}else{p.max}}
    }).collect();
    let seen=hidden_head::head_visibility(scan,&baseline,head);
    let skin_triangles:Vec<_>=scan.triangles.iter().enumerate().filter(|(t,tri)|seg.triangle_label[*t]==Label::Skin&&!seg.is_prop(*t)&&tri.iter().any(|&i|seen[i as usize])).map(|(_,t)|*t).collect();
    let hair_triangles:Vec<_>=scan.triangles.iter().enumerate().filter(|(t,_)|seg.triangle_label[*t]==Label::Hair&&!seg.is_prop(*t)).map(|(_,t)|*t).collect();
    let body=build_query(&baseline.iter().map(|p|a(*p)).collect::<Vec<_>>(),triangles);let mut skin_samples=vec![];
    for &s in &scan.weld.representative {if !seen[s]||seg.vertex_label[s]!=Label::Skin {continue;}
        if let Some(h)=body.closest_point(scan.positions[s],0.03,None) {if triangles[h.source[1]].iter().any(|&i|region[i as usize]) {skin_samples.push(scan.positions[s]);}}}
    let stride=(skin_samples.len()/6000).max(1);skin_samples=skin_samples.into_iter().step_by(stride).collect();
    let centre=(0..n).filter(|&i|head[i]).fold(V::zeros(),|s,i|s+baseline[i])/head.iter().filter(|&&b|b).count().max(1)as f64;
    let views=input.views.iter().map(|o|silhouette::View::new(&SuitContour {camera:o.camera.clone(),curves:vec![o.samples.clone()],weights:vec![o.weights.clone()],skin_anchors:vec![],rotation_limits_deg:[0.,0.],scale_tied_to:None,skin_outline_anchors:vec![]})).collect();
    let data=Data {input,views,triangles,region,visible,hidden,skin:build_query(&scan.positions,&skin_triangles),skin_samples,hair:build_query(&scan.positions,&hair_triangles),centre};
    let sym=input.x121.as_ref().map(|s|symmetry::Symmetry::new(reference,head,&data.region,rotation*V::x(),translation,s.strength));
    let initial_views=if sym.is_some() {symmetry::view_distances(&data,&baseline,&baseline)}else{vec![]};
    let (morphed,weights,history)=solve(&data,&baseline,&baseline,&fields,&initial,&initial,input.morph_iterations,input.prior_weight,&DMatrix::zeros(fields.len(),fields.len()),None,false,sym.as_ref());
    let candidates:Vec<_>=(0..n).filter(|&i|data.region[i]&&data.hidden[i]&&pin[i]>0.).collect();
    let mut centres:Vec<V>=vec![];
    if let Some(sym)=&sym {
         
        let paired:Vec<V>=sym.pairs.iter().filter(|&&(i,j)|candidates.contains(&i)&&candidates.contains(&j))
            .map(|&(i,j)|(baseline[i]+sym.mirror_point(baseline[j]))*0.5)
            .map(|p|if (p-sym.point).dot(&sym.normal)<0. {sym.mirror_point(p)}else{p})
            .filter(|p|(p-sym.point).dot(&sym.normal)>0.004).collect();
        while centres.len()<settings.residual_centres {
            let best=paired.iter().max_by(|a,b|{
                let distance=|p:&V|if centres.is_empty() {p.y}else{centres.iter().map(|q|(p-q).norm_squared()).fold(f64::INFINITY,f64::min)};
                distance(a).total_cmp(&distance(b))
            });
            if let Some(&p)=best {centres.push(p);centres.push(sym.mirror_point(p));}else{break;}
        }
        assert_eq!(centres.len(),settings.residual_centres);
    } else {
        if let Some(&top)=candidates.iter().max_by(|&&a,&&b|baseline[a].y.total_cmp(&baseline[b].y)) {centres.push(baseline[top]);}
        while centres.len()<settings.residual_centres && centres.len()<candidates.len() {
            let best=candidates.iter().max_by(|&&a,&&b|{
                let distance=|i:usize|centres.iter().map(|p|(baseline[i]-p).norm_squared()).fold(f64::INFINITY,f64::min);
                distance(a).total_cmp(&distance(b))
            });if let Some(&best)=best {centres.push(baseline[best]);}else{break;}
        }
    }
    // 10 mm below inion; a C2 30 mm transition reaches 30 mm above it.
    // The same geodesic pin envelope takes both fields to zero on visible neck.
    let caps:Vec<_>=baseline.iter().map(|p|settings.residual_below_cap_m+(input.residual_cap_m-settings.residual_below_cap_m)*fade(p.y,input.inion_y_m,input.residual_fade_m)).collect();
    let mut residual_fields=vec![];
    for c in &centres {for axis in [V::x(),V::y(),V::z()] {
        let plus:Vec<_>=(0..n).map(|i|axis*caps[i]*pin[i]*wendland((baseline[i]-c).norm(),input.residual_support_m)).collect();
        residual_fields.push(Field {minus:plus.iter().map(|p|-p).collect(),plus,lower:-1.,upper:1.});
    }}
    let penalty=bending(&centres,settings.bending_weight);let zero=vec![0.;residual_fields.len()];
    let (result,residual_weights,residual_history)=solve(&data,&baseline,&morphed,&residual_fields,&zero,&zero,input.residual_iterations,input.prior_weight*0.25,&penalty,Some(&caps),false,sym.as_ref());
    let before_projection=hair_rows(&data,&result).iter().map(|r|r.delta.abs()).fold(0.,f64::max);
    let (result,residual_weights,projection_history)=if before_projection>0.00001 && settings.projection_iterations>0 {
        solve(&data,&baseline,&morphed,&residual_fields,&zero,&residual_weights,settings.projection_iterations,1e-8,&(&penalty*0.01),Some(&caps),true,sym.as_ref())
    }else{(result,residual_weights,vec![])};
    let residual:Vec<_>=result.iter().zip(&morphed).map(|(a,b)|(a-b).norm()).collect();
    let stats=|above:bool|{let mut values:Vec<_>=(0..n).filter(|&i|data.region[i]&&((baseline[i].y>input.inion_y_m)==above)).map(|i|residual[i]*1000.).collect();values.sort_by(f64::total_cmp);
        json!({"count":values.len(),"max_mm":values.last().copied().unwrap_or(0.),"p95_mm":values.get(((values.len().saturating_sub(1))as f64*0.95).ceil()as usize).copied().unwrap_or(0.)})};
    let pinned_max=(0..n).filter(|&i|data.visible[i]).map(|i|(result[i]-baseline[i]).norm()).fold(0.,f64::max);
    assert_eq!(pinned_max,0.);assert!((0..n).all(|i|residual[i]<=caps[i]+1e-12));
    for (p,(&w,&b)) in input.basis.parameters.iter().zip(weights.iter().zip(&initial)) {if p.name.starts_with("head-scale-") {assert_eq!(w,b);}}
    x.copy_from_slice(&result);let (_,final_stats)=data.rows(x);
    json!({"method":if sym.is_some() {"x121_symmetric_pinned_MakeHuman_and_RBF"}else{"x120_pinned_MakeHuman_and_RBF"},"applied":true,"identity_probe":false,"name":input.name,
        "basis":input.basis_path,"target_report":input.reference_report,"held_out_used":false,"old_skull_holds":false,"initial_projection":[],
        "transport":{"method":"one_whole_head_similarity_from_visible_skin","scale":scale,"rotation":rotation.as_slice(),"translation":a(translation)},
        "morph_weights":input.basis.parameters.iter().zip(&initial).zip(&weights).zip(&fields).map(|(((p,b),w),f)|json!({"name":p.name,"initial":b,"solved":w,"added":w-b,"min":f.lower,"max":f.upper,"frozen":f.lower==f.upper})).collect::<Vec<_>>(),
        "symmetry":sym.as_ref().map(|s|s.stats(&result,&baseline)),"per_view_descent":{"initial":symmetry::view_json(&data,&initial_views),"final":symmetry::view_json(&data,&if sym.is_some() {symmetry::view_distances(&data,&result,&baseline)}else{vec![]}),"metric":"positive-trust bidirectional upper-skull RMS and p95; every accepted view; 1e-9 mm roundoff only"},
        "morph_history":history,"residual_history":residual_history,"hair_projection_history":projection_history,"final_constraints":final_stats,
        "pinning":{"method":"frozen_visible_skin_with_one_triangle_support_collar_geodesic_C2_falloff","falloff_mm":settings.pin_falloff_m*1000.,"visible_vertices":data.visible.iter().filter(|&&b|b).count(),"support_vertices":support.iter().filter(|&&b|b).count(),"max_displacement_mm":pinned_max*1000.,"applies_to_morphs_and_residual":true},
        "field_feasibility":{"triangle_orientation_preserved":field_feasible(&baseline,&result,triangles),"maximum_relative_adjacent_displacement":0.75},
        "inion_y_m":input.inion_y_m,"per_vertex_skull_below_inion":false,
        "residual":{"method":"Wendland_C2_RBFs","support_m":input.residual_support_m,"fade_m":input.residual_fade_m,"cap_mm":input.residual_cap_m*1000.,"below_cap_mm":settings.residual_below_cap_m*1000.,"above_inion":stats(true),"below_inion":stats(false),"max_mm":residual.iter().copied().fold(0.,f64::max)*1000.,
            "centres":centres.iter().map(|&p|a(p)).collect::<Vec<_>>(),"coefficients":residual_weights,"bending_weight":settings.bending_weight,"bending_method":"squared_normalized_four_neighbour_coefficient_Laplacian","cap_method":"global_coefficient_scaling_against_smooth_local_caps"},
        "hair_projection":{"method":"same_pinned_RBF_coefficient_space","before_max_violation_mm":before_projection*1000.},"skin_samples":data.skin_samples.len(),"hair_clearance_mm":1.8})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn coefficient_candidate_guard_rejects_folds_and_steep_pinned_fields() {
        let base=vec![V::zeros(),V::x(),V::y()];let triangles=[[0,1,2]];
        assert!(field_feasible(&base,&base,&triangles));
        assert!(!field_feasible(&base,&[V::zeros(),-V::x(),V::y()],&triangles));
        assert!(!field_feasible(&base,&[V::zeros(),V::x()+V::z(),V::y()],&triangles));
        assert!(field_feasible(&base,&[V::zeros(),V::x()+V::z()*0.2,V::y()],&triangles));
    }
    #[test] fn geodesic_pins_follow_edges_not_spatial_shortcuts() {
        let p=vec![V::new(0.,0.,0.),V::new(0.01,0.,0.),V::new(0.02,0.,0.),V::new(0.001,0.,0.)];
        let w=pin_envelope(&p,&[[0,1,1],[1,2,2],[2,3,3]],&[true,false,false,false],&[true;4],0.03);
        assert_eq!(w[0],0.);assert!(w[1]>0. && w[1]<w[2]);assert_eq!(w[3],1.);
    }
    #[test] fn frozen_out_of_range_scale_has_zero_step_and_exact_origin() {
        let f=Field {plus:vec![V::y()*0.02],minus:vec![-V::y()*0.02],lower:1.90625,upper:1.90625};
        let (w,x)=bounded_start(&[V::y()*1.5],&[f],&[1.90625]);assert_eq!(w,[1.90625]);assert_eq!(x,[V::y()*1.5]);
    }
    #[test] fn cap_scales_coefficients_and_bending_preserves_constant_field() {
        let fields=vec![Field {plus:vec![V::x()*0.02,V::x()*0.06],minus:vec![-V::x()*0.02,-V::x()*0.06],lower:-1.,upper:1.}];
        let mut w=vec![1.];cap_field(&fields,&mut w,&[0.01,0.03]);assert_eq!(w,[0.5]);
        let b=bending(&[V::zeros(),V::x(),V::y()],1.);
        assert!((&b*DVector::from_element(9,1.)).norm()<1e-12);
        let mut curved=DVector::zeros(9);curved[0]=1.;assert!(curved.dot(&(&b*&curved))>0.);
    }
}
