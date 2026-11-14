use charloom_core::*;
use charloom_io::{load_gltf, mesh_document};
use serde_json::json;

#[test]
fn default_and_animated_morphs_survive_roundtrip() {
    let mut doc = mesh_document(
        "morph",
        &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        &[[0, 1, 2]],
    )
    .unwrap();
    let deltas = vec![0.0_f32, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0];
    let accessor = doc.append_accessor(
        &deltas
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
        5126,
        "VEC3",
        3,
        Some(34962),
        Some((json!([0.0, 0.0, 2.0]), json!([0.0, 0.0, 2.0]))),
    );
    doc.json["meshes"][0]["primitives"][0]["targets"] = json!([{"POSITION":accessor}]);
    doc.json["meshes"][0]["weights"] = json!([0.25]);
    doc.append_animation(
        &MotionClip {
            name: "morph".into(),
            tracks: Vec::new(),
            morph_tracks: vec![MorphTrack {
                node: 0,
                interpolation: Interpolation::Linear,
                times: vec![0.0, 1.0],
                values: vec![vec![0.0], vec![1.0]],
            }],
        },
        0,
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("morph.glb");
    doc.save(&path).unwrap();
    let asset = load_gltf(path).unwrap().asset;
    let pose = asset.skeleton.rest_pose();
    assert!((asset.skinned_positions(&pose).unwrap()[0][0][2] - 0.5).abs() < 1e-8);
    let clip = &asset.clips[0];
    assert!(
        (asset
            .deformed_positions(&pose, &clip.sample_morphs(0.5).unwrap())
            .unwrap()[0][0][2]
            - 1.0)
            .abs()
            < 1e-8
    );
}
