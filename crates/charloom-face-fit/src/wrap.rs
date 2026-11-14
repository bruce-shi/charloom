//! A face is fitted as a smooth outer sheet. The chart warp is shared through
//! depth, so a mouth/eye interior receives the same displacement as its cover,
//! rather than being independently attracted onto the observed face surface.
use crate::parallel::*;
use crate::surface::{Surface, V};
use crate::{Landmark, landmark_rmse, smooth};
use std::collections::BTreeMap;

 struct Field {
    samples: Vec<(V, V, f64)>,
    sigma: f64,
    zero_weight: f64,
}
impl Field {
    fn new(samples: Vec<(V, V, f64)>, sigma: f64, zero_weight: f64) -> Self {
        Self {
            samples,
            sigma,
            zero_weight,
        }
    }
    fn at(&self, p: V) -> V {
        let mut value = V::zeros();
        let mut mass = self.zero_weight;
        for (q, delta, weight) in &self.samples {
            let d = p - q;
            let k = weight * (-(d.x * d.x + d.z * d.z) / (2. * self.sigma * self.sigma)).exp();
            value += delta * k;
            mass += k;
        }
        value / mass
    }
}

pub fn fit(
    base: &[V],
    faces: &[[usize; 3]],
    body_ids: &[usize],
    eligible: &[bool],
    blend: &[f64],
    landmarks: &[Landmark],
    skin: &Surface,
    complete: &Surface,
) -> (Vec<V>, Vec<serde_json::Value>) {
    let mut p = base.to_vec();
    let original_surface = Surface::new(base, faces);
    let chin_base = base[landmarks.iter().find(|l| l.name == "chin").unwrap().vertex].z;
    // Preserve the rear cranium. Cavities just behind the facial skin are
    // transported fully; only deeper, unobserved skull tissue fades to zero.
    let mask: Vec<_> = base
        .par_iter()
        .enumerate()
        .map(|(i, &q)| {
            if blend[i] == 0. {
                return 0.;
            }
            // A failed silhouette ray is NOT evidence for the front skin.
            // Also keep facial chart corrections out of the lower neck: equal
            // X/Z coordinates do not make a neck point part of a cheek.
            let Some(front) = original_surface.front(q.x, q.z) else {
                return 0.;
            };
            let gap = q.y - front.y;
            let neck = smooth((q.z - (chin_base - 0.025)) / 0.025);
            blend[i] * neck * (1. - smooth((gap - 0.035) / 0.055))
        })
        .collect();
    let mut trace = vec![];
    let chin = landmarks
        .iter()
        .find(|l| l.name == "chin")
        .unwrap()
        .observed[2];
    let forehead = landmarks
        .iter()
        .find(|l| l.name == "forehead")
        .unwrap()
        .observed[2];
    // First slide the elastic sheet in its frontal chart, preserving feature
    // correspondence. The boundary is softly held outside the observed face.
    for iteration in 0..48 {
        let sigma = if iteration < 12 {
            0.028
        } else if iteration < 28 {
            0.012
        } else {
            0.010
        };
        let mut samples: Vec<_> = landmarks
            .iter()
            .map(|l| {
                let q = V::from(l.observed) - p[l.vertex];
                let d = V::new(q.x, 0., q.z);
                (p[l.vertex], d * (0.003 / d.norm().max(0.003)), l.weight)
            })
            .collect();
        for x in [-0.13, -0.09, 0., 0.09, 0.13] {
            for z in [chin - 0.07, forehead + 0.08] {
                samples.push((V::new(x, 0., z), V::zeros(), 1.));
            }
        }
        let field = Field::new(samples, sigma, 0.03);
        let displacement: Vec<_> = p
            .par_iter()
            .enumerate()
            .map(|(i, &q)| {
                if mask[i] == 0. {
                    V::zeros()
                } else {
                    field.at(q) * mask[i]
                }
            })
            .collect();
        let mut step = 0.7;
        // Require a small positive local chart Jacobian; the depth translation
        // is shared by both layers of the sheet, including the mouth interior.
        for &i in body_ids {
            if blend[i] < 0.99 {
                continue;
            }
            let q = p[i];
            let e = 0.0001;
            let dx = (field.at(q + V::x() * e) - field.at(q - V::x() * e)) / (2. * e);
            let dz = (field.at(q + V::z() * e) - field.at(q - V::z() * e)) / (2. * e);
            while step > 0.003
                && (1. + step * dx.x) * (1. + step * dz.z) - step * step * dx.z * dz.x < 0.6
            {
                step *= 0.5;
            }
        }
        for i in 0..p.len() {
            p[i] += displacement[i] * step;
        }
        trace.push(serde_json::json!({"stage":"sheet_landmark_slide","iteration":iteration,"step":step,"sigma_m":sigma}));
    }
    let mut samples: Vec<_> = landmarks
        .iter()
        .map(|l| {
            (
                p[l.vertex],
                V::new(0., l.observed[1] - base[l.vertex].y, 0.),
                15. * l.weight,
            )
        })
        .collect();
    let feature_regions: Vec<_> = ["eye_", "mouth_"]
        .iter()
        .map(|prefix| {
            let ids: Vec<_> = landmarks
                .iter()
                .filter(|l| l.name.starts_with(prefix))
                .map(|l| l.vertex)
                .collect();
            let low = ids
                .iter()
                .fold(V::repeat(f64::INFINITY), |q, &i| q.inf(&base[i]));
            let high = ids
                .iter()
                .fold(V::repeat(f64::NEG_INFINITY), |q, &i| q.sup(&base[i]));
            (low, high, ids)
        })
        .collect();
    let mut cells = BTreeMap::new();
    for &i in body_ids {
        if !eligible[i] {
            continue;
        }
        if original_surface
            .front(base[i].x, base[i].z)
            .is_none_or(|q| base[i].y - q.y > 0.0015)
        {
            continue;
        }
        let is_interior = feature_regions.iter().any(|(low, high, _)| {
            base[i].x > low.x - 0.005
                && base[i].x < high.x + 0.005
                && base[i].z > low.z - 0.005
                && base[i].z < high.z + 0.005
                && base[i].y > high.y + 0.002
        });
        if is_interior {
            continue;
        }
        let key = [
            (p[i].x / 0.004).floor() as i32,
            (p[i].z / 0.004).floor() as i32,
        ];
        cells.entry(key).or_insert(i);
    }
    let mut hits = 0;
    for &i in cells.values() {
        if samples
            .iter()
            .any(|(q, _, _)| (q.x - p[i].x).hypot(q.z - p[i].z) < 0.0015)
        {
            continue;
        }
        let front = complete.front(p[i].x, p[i].z);
        let hit = skin
            .front(p[i].x, p[i].z)
            .filter(|q| front.is_some_and(|f| (q.y - f.y).abs() < 0.001));
        if let Some(q) = hit {
            let delta = q.y - base[i].y;
            if delta.abs() < 0.045 {
                samples.push((p[i], V::new(0., delta, 0.), 1.));
                hits += 1;
            }
        }
    }
    // Weak priors extend the observed sheet smoothly into the hidden cranium.
    for x in [-0.13, -0.09, 0., 0.09, 0.13] {
        for z in [chin - 0.06, forehead + 0.10] {
            samples.push((V::new(x, 0., z), V::zeros(), 0.2));
        }
    }
    let field = Field::new(samples, 0.011, 0.03);
    for i in 0..p.len() {
        if mask[i] == 0. {
            continue;
        }
        p[i].y = base[i].y + field.at(p[i]).y * mask[i];
    }
    trace.push(serde_json::json!({"stage":"elastic_sheet_surface_attachment","observed_cells":hits,"sigma_m":0.011,"landmark_rmse_mm":landmark_rmse(&p,landmarks),"attachment":"same depth displacement for outer skin, sockets, oral cavity and helpers"}));
    eprintln!(
        "sheet wrap: {:.3} mm, {hits} observed patches",
        landmark_rmse(&p, landmarks)
    );
    (p, trace)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn field_cannot_overshoot_samples_and_decays_when_unobserved() {
        let field = Field::new(
            vec![
                (V::zeros(), V::new(0.003, -0.01, 0.002), 1.),
                (V::x() * 0.01, V::new(-0.002, 0.005, -0.001), 5.),
            ],
            0.02,
            0.03,
        );
        for i in -200..200 {
            let d = field.at(V::new(i as f64 * 0.001, 0., 0.));
            assert!(d.x >= -0.002 && d.x <= 0.003);
            assert!(d.y >= -0.01 && d.y <= 0.005);
            assert!(d.z >= -0.001 && d.z <= 0.002);
        }
        assert!(field.at(V::new(0., 0., 0.2)).norm() < 1e-10);
    }
    #[test]
    fn depth_transport_preserves_a_thin_shell_gap() {
        let samples = vec![
            (V::new(-0.02, 0., 0.), V::new(0., -0.006, 0.), 1.),
            (V::new(0.02, 0., 0.), V::new(0., 0.004, 0.), 1.),
        ];
        let field = Field::new(samples, 0.03, 0.001);
        for x in [-0.03, -0.01, 0., 0.01, 0.03] {
            let a = V::new(x, -0.01, 0.);
            let b = V::new(x, -0.006, 0.);
            assert!(((a + field.at(a)) - (b + field.at(b)) - (a - b)).norm() < 1e-12);
        }
    }
}
