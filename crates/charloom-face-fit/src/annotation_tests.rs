use super::*;

fn fixture() -> (Vec<V>, Vec<[usize; 3]>, Annotation) {
    let points = [-0.2, 0.1]
        .into_iter()
        .flat_map(|y| [V::new(-0.1, y, 0.), V::new(0.1, y, 0.), V::new(0., y, 0.2)])
        .collect();
    let annotation = serde_json::from_value(serde_json::json!({
        "name":"nose_tip", "template_xz":[0.,0.05], "source_xz":[0.,0.05]
    }))
    .unwrap();
    (points, vec![[0, 1, 2], [3, 4, 5]], annotation)
}

#[test]
fn explicit_depth_selects_the_reviewed_layer_instead_of_the_front_layer() {
    let (points, faces, mut annotation) = fixture();
    let surface = Surface::new(&points, &faces);
    assert!((observed_landmark(&surface, &annotation).unwrap().y + 0.2).abs() < 1e-10);
    annotation.source_y = Some(0.1);
    let observed = observed_landmark(&surface, &annotation).unwrap();
    assert!((observed - V::new(0., 0.1, 0.05)).norm() < 1e-10);
}

#[test]
fn invalid_or_unobserved_depth_does_not_silently_revert_to_a_front_ray() {
    let (points, faces, mut annotation) = fixture();
    let surface = Surface::new(&points, &faces);
    annotation.source_y = Some(0.);
    assert!(
        observed_landmark(&surface, &annotation)
            .unwrap_err()
            .to_string()
            .contains("3 mm")
    );
    annotation.source_y = Some(f64::NAN);
    assert!(observed_landmark(&surface, &annotation).is_err());
}
