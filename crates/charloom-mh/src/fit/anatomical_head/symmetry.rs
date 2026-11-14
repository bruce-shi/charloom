 use super::*;

pub(super) struct Symmetry {
    pub normal: V,
    pub point: V,
    pub pairs: Vec<(usize, usize)>,
    strength: f64,
}
impl Symmetry {
    pub fn new(reference: &[V], head: &[bool], region: &[bool], normal: V, point: V, strength: f64) -> Self {
        // Correspondence comes from the unchanged anatomical reference, not a
        // deformed candidate. Midline points pair with themselves.
        let ids: Vec<_> = (0..reference.len()).filter(|&i|head[i] && region[i]).collect();
        let mut pairs = Vec::new();
        for &i in &ids {
            let q = V::new(-reference[i].x, reference[i].y, reference[i].z);
            if let Some(&j) = ids.iter().min_by(|&&a,&&b|(reference[a]-q).norm_squared().total_cmp(&(reference[b]-q).norm_squared())) {
                if i <= j && (reference[j]-q).norm() < 0.003 { pairs.push((i,j)); }
            }
        }
        assert!(!pairs.is_empty(), "no anatomical sagittal pairs");
        Self {normal: normal.normalize(), point, pairs, strength}
    }
    pub fn reflect(&self, d: V) -> V { d - self.normal * (2. * d.dot(&self.normal)) }
    pub fn mirror_point(&self, p: V) -> V { self.point + self.reflect(p-self.point) }
    fn error(&self, x: &[V], baseline: &[V], i: usize, j: usize) -> V {
        x[i]-baseline[i]-self.reflect(x[j]-baseline[j])
    }
    pub fn energy(&self, x: &[V], baseline: &[V]) -> f64 {
        self.strength/self.pairs.len() as f64 * self.pairs.iter().map(|&(i,j)|self.error(x,baseline,i,j).norm_squared()).sum::<f64>()
    }
    pub fn add_system(&self, fields: &[Field], weights: &[f64], x: &[V], baseline: &[V], matrix: &mut DMatrix<f64>, rhs: &mut DVector<f64>) {
        let weight = self.strength/self.pairs.len() as f64;
        for &(i,j) in &self.pairs {
            let error = self.error(x,baseline,i,j);
            let jac: Vec<_> = fields.iter().zip(weights).map(|(f,&w)|derivative(f,w,i)-self.reflect(derivative(f,w,j))).collect();
            for a in 0..fields.len() {
                rhs[a] -= weight * jac[a].dot(&error);
                for b in 0..fields.len() { matrix[(a,b)] += weight * jac[a].dot(&jac[b]); }
            }
        }
    }
    pub fn stats(&self, x: &[V], baseline: &[V]) -> Value {
        let mut errors: Vec<_> = self.pairs.iter().map(|&(i,j)|self.error(x,baseline,i,j).norm()*1000.).collect();
        errors.sort_by(f64::total_cmp);
        json!({"pairs":errors.len(),"strength":self.strength,"plane_normal":a(self.normal),"plane_point":a(self.point),
            "displacement_mirror_max_mm":errors.last(),"displacement_mirror_p95_mm":errors[((errors.len()-1)as f64*0.95).ceil()as usize],
            "displacement_mirror_rms_mm":(errors.iter().map(|v|v*v).sum::<f64>()/errors.len()as f64).sqrt(),
            "definition":"total added displacement at anatomical mirror pairs; baseline asymmetry is not a target"})
    }
}

/// Positive-trust bidirectional surface correspondences above the frozen inion.
/// Re-render every trial so silhouette ownership changes cannot evade descent.
pub(super) fn view_distances(data: &Data, x: &[V], baseline: &[V]) -> Vec<(usize,f64,f64)> {
    data.views.iter().map(|view| {
        let (rows,_,_) = silhouette::rendered_constraints(std::slice::from_ref(view),x,data.triangles,&data.region,0.12);
        let mut distances: Vec<_> = rows.iter().filter(|r|r.weight>0. &&
            r.vertices.iter().zip(r.bary).map(|(&i,b)|if data.hidden[i as usize] && !data.visible[i as usize] && baseline[i as usize].y>=data.input.inion_y_m {b}else{0.}).sum::<f64>()>0.5)
            .map(|r|(r.delta.norm(),r.weight)).collect();
        distances.sort_by(|a,b|a.0.total_cmp(&b.0));
        let total=distances.iter().map(|p|p.1).sum::<f64>();
        if total==0. {return (0,f64::INFINITY,f64::INFINITY);}
        let rms=(distances.iter().map(|p|p.0*p.0*p.1).sum::<f64>()/total).sqrt();
        let mut sum=0.;let mut p95=0.;
        for &(distance,weight) in &distances {sum+=weight;p95=distance;if sum>=total*0.95 {break;}}
        (distances.len(),rms,p95)
    }).collect()
}
pub(super) fn no_view_worse(before: &[(usize,f64,f64)], after: &[(usize,f64,f64)]) -> bool {
    before.len()==after.len() && before.iter().zip(after).all(|(b,a)|
        b.0>0 && a.0>0 && a.1.is_finite() && a.2.is_finite() && a.1<=b.1+1e-12 && a.2<=b.2+1e-12)
}
pub(super) fn view_json(data: &Data, values: &[(usize,f64,f64)]) -> Value {
    json!(values.iter().zip(&data.input.views).map(|(v,o)|json!({"view":o.camera.id,"samples":v.0,"rms_mm":v.1*1000.,"p95_mm":v.2*1000.})).collect::<Vec<_>>())
}

fn project_halfspaces(step:&mut DVector<f64>, gradients:&[DVector<f64>], diagonal:&[f64], lower:&[f64], upper:&[f64]) {
    for _ in 0..100 {
        for g in gradients {
            let excess=g.dot(step);
            let norm=(0..step.len()).map(|j|g[j]*g[j]/diagonal[j]).sum::<f64>();
            if excess>0. && norm>1e-30 {
                for j in 0..step.len() {step[j]-=excess/norm*g[j]/diagonal[j];}
            }
        }
        for j in 0..step.len() {step[j]=step[j].clamp(lower[j],upper[j]);}
    }
}
/// Only a proposed search direction. The fresh raster guard must still pass.
pub(super) fn common_direction(data:&Data,x:&[V],baseline:&[V],fields:&[Field],weights:&[f64],matrix:&DMatrix<f64>,proposed:&DVector<f64>)->DVector<f64> {
    let k=fields.len();let mut gradients=vec![];
    for view in &data.views {
        let (rows,_,_)=silhouette::rendered_constraints(std::slice::from_ref(view),x,data.triangles,&data.region,0.12);
        let mut rows:Vec<_>=rows.into_iter().filter(|r|r.weight>0. &&
            r.vertices.iter().zip(r.bary).map(|(&i,b)|if data.hidden[i as usize] && !data.visible[i as usize] && baseline[i as usize].y>=data.input.inion_y_m {b}else{0.}).sum::<f64>()>0.5).collect();
        rows.sort_by(|a,b|a.delta.norm_squared().total_cmp(&b.delta.norm_squared()));
        for start in [0,rows.len()*9/10] {
            let mut g=DVector::zeros(k);
            for r in &rows[start..] {
                for j in 0..k {
                    if fields[j].lower==fields[j].upper {continue;}
                    let d=(0..3).fold(V::zeros(),|s,a|s+derivative(&fields[j],weights[j],r.vertices[a]as usize)*r.bary[a]);
                    g[j]-=r.weight*r.axes.iter().map(|axis|d.dot(axis)*r.delta.dot(axis)).sum::<f64>();
                }
            }
            gradients.push(g);
        }
    }
    let diagonal:Vec<_>=(0..k).map(|j|matrix[(j,j)].max(1e-12)).collect();
    let lower:Vec<_>=(0..k).map(|j|(fields[j].lower-weights[j]).max(-0.25)).collect();
    let upper:Vec<_>=(0..k).map(|j|(fields[j].upper-weights[j]).min(0.25)).collect();
    let mut step=proposed.clone();project_halfspaces(&mut step,&gradients,&diagonal,&lower,&upper);step
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn mirror_penalty_detects_crown_lean_but_not_symmetric_expansion() {
        let reference=vec![-V::x(),V::x()];
        let s=Symmetry::new(&reference,&[true;2],&[true;2],V::x(),V::zeros(),1.);
        assert_eq!(s.energy(&[-V::x()*1.1,V::x()*1.1],&reference),0.);
        assert!(s.energy(&[-V::x()+V::x()*0.1,V::x()+V::x()*0.1],&reference)>0.);
        let tilted=V::new(1.,1.,0.).normalize();let s=Symmetry {normal:tilted,..s};
        assert!((s.reflect(tilted)+tilted).norm()<1e-12);
    }
    #[test] fn common_descent_projection_respects_each_view_and_frozen_coefficients() {
        let mut step=DVector::from_vec(vec![1.,-1.,1.]);
        let gradients=vec![DVector::from_vec(vec![1.,0.,0.]),DVector::from_vec(vec![0.,1.,0.])];
        project_halfspaces(&mut step,&gradients,&[1.;3],&[-2.,-2.,0.],&[2.,2.,0.]);
        assert!(gradients.iter().all(|g|g.dot(&step)<=1e-12));assert_eq!(step[2],0.);assert!(step[1]<0.);
    }
    #[test] fn pooled_progress_cannot_hide_one_worsening_view() {
        let before=[(20,0.02,0.03),(20,0.01,0.015)];
        assert!(!no_view_worse(&before,&[(20,0.01,0.015),(20,0.011,0.016)]));
        assert!(no_view_worse(&before,&[(20,0.019,0.029),(20,0.009,0.014)]));
        assert!(!no_view_worse(&before,&[(20,0.019,0.029),(0,0.,0.)]));
    }
}
