//! Local positional contact constraints for a thin folded facial membrane.
//! Stock-template contacts are tracked separately; they are not new failures.
use crate::surface::{Surface, V};

fn separation(a: [V; 3], b: [V; 3]) -> V {
    let edges = |t: [V; 3]| [t[1] - t[0], t[2] - t[1], t[0] - t[2]];
    let ea = edges(a);
    let eb = edges(b);
    let mut axes = vec![ea[0].cross(&ea[1]), eb[0].cross(&eb[1])];
    for x in ea {
        for y in eb {
            axes.push(x.cross(&y));
        }
    }
    let mut best = V::repeat(f64::INFINITY);
    for axis in axes {
        let Some(n) = axis.try_normalize(1e-15) else {
            continue;
        };
        let interval = |t: [V; 3]| {
            t.iter()
                .fold([f64::INFINITY, f64::NEG_INFINITY], |[lo, hi], p| {
                    let d = p.dot(&n);
                    [lo.min(d), hi.max(d)]
                })
        };
        let [amin, amax] = interval(a);
        let [bmin, bmax] = interval(b);
        let forward = bmax - amin;
        let back = amax - bmin;
        let delta = if forward < back {
            n * (forward.max(0.) + 2e-5)
        } else {
            -n * (back.max(0.) + 2e-5)
        };
        if delta.norm_squared() < best.norm_squared() {
            best = delta;
        }
    }
    best
}

pub fn repair(base: &[V], fitted: &mut [V], faces: &[[usize; 3]], low_z: f64) -> serde_json::Value {
    let faces: Vec<_> = faces
        .iter()
        .copied()
        .filter(|f| f.iter().all(|&i| base[i].z > low_z))
        .collect();
    let baseline = Surface::new(base, &faces).crossings();
    let original = fitted.to_vec();
    let mut history = vec![];
    for _ in 0..16 {
        let contacts = Surface::new(fitted, &faces).crossings();
        let introduced: Vec<_> = contacts.difference(&baseline).copied().collect();
        history.push(introduced.len());
        if introduced.is_empty() {
            break;
        }
        let mut delta = vec![V::zeros(); fitted.len()];
        let mut count = vec![0.; fitted.len()];
        for [a, b] in introduced {
            let correction =
                separation(faces[a].map(|i| fitted[i]), faces[b].map(|i| fitted[i])) * 0.55;
            let correction = correction * (0.0004 / correction.norm().max(0.0004));
            for (face, sign) in [(faces[a], 1.), (faces[b], -1.)] {
                for i in face {
                    delta[i] += correction * sign;
                    count[i] += 1.;
                }
            }
        }
        for i in 0..fitted.len() {
            if count[i] > 0. {
                fitted[i] += delta[i] / count[i];
            }
        }
    }
    let after = Surface::new(fitted, &faces).crossings();
    let max_shift = fitted
        .iter()
        .zip(&original)
        .map(|(a, b)| (a - b).norm())
        .fold(0., f64::max);
    serde_json::json!({"stage":"thin_membrane_contact_constraints","baseline_pairs":baseline.len(),"new_pairs":after.difference(&baseline).count(),"new_pairs_per_iteration":history,"maximum_contact_correction_mm":max_shift*1000.,"scope":"head skin including internal surfaces; excludes eyeball and hair objects"})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn separates_a_new_piercing_without_moving_unrelated_vertices() {
        let base = vec![
            V::new(-1., 0., 0.),
            V::new(1., 0., 0.),
            V::new(0., 1., 0.),
            V::new(0., 0.2, 0.001),
            V::new(0.2, 0.2, 0.001),
            V::new(0., 0.7, 0.001),
            V::new(3., 3., 3.),
        ];
        let faces = vec![[0, 1, 2], [3, 4, 5]];
        let mut p = base.clone();
        p[3].z = -0.0001;
        p[4].z = 0.0001;
        p[5].z = 0.0001;
        assert_eq!(Surface::new(&p, &faces).crossings().len(), 1);
        let report = repair(&base, &mut p, &faces, -1.);
        assert_eq!(report["new_pairs"], 0);
        assert_eq!(p[6], base[6]);
        assert!(report["maximum_contact_correction_mm"].as_f64().unwrap() < 0.5);
    }
}
