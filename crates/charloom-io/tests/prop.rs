use charloom_core::*;
use charloom_io::{load_gltf, mesh_document};
use serde_json::json;

#[test]
fn rigid_prop_merge_preserves_resources_transforms_and_original_character() {
    let folder = tempfile::tempdir().unwrap();
    let triangle = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]];
    let mut character = mesh_document("character", &triangle, &[[0, 1, 2]]).unwrap();
    let original_binary = character.binary.clone();
    let original_node = character.json["nodes"][0].clone();
    let mut prop = mesh_document("part", &triangle, &[[0, 1, 2]]).unwrap();
    prop.json["nodes"][0]["translation"] = json!([0.0, 0.0, 0.4]);
    prop.json["images"] = json!([{"uri":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="}]);
    prop.json["textures"] = json!([{"source":0}]);
    prop.json["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] =
        json!({"index":0,"texCoord":0});
    prop.save(&folder.path().join("prop.glb")).unwrap();
    let source = load_gltf(folder.path().join("prop.glb")).unwrap();
    let node = character
        .append_rigid_prop(
            &source,
            "held_object",
            Transform::translated([0.8, 1.0, 0.0]),
        )
        .unwrap();
    assert_eq!(character.json["nodes"][0], original_node);
    assert_eq!(&character.binary[..original_binary.len()], original_binary);
    character.save(&folder.path().join("merged.glb")).unwrap();
    let merged = load_gltf(folder.path().join("merged.glb")).unwrap();
    assert_eq!(merged.asset.meshes.len(), 2);
    assert_eq!(merged.asset.skeleton.joints[node].name, "held_object");
    let points = merged.asset.world_meshes().unwrap();
    assert!(
        points[0]
            .iter()
            .zip(triangle)
            .all(|(a, b)| (v3(*a) - v3(b)).norm() < 1e-7)
    );
    assert!((v3(points[1][0]) - v3([0.8, 1.0, 0.4])).norm() < 1e-7);
    let material = merged.document.json["meshes"][1]["primitives"][0]["material"]
        .as_u64()
        .unwrap() as usize;
    assert_ne!(material, 0);
    assert_eq!(
        merged.document.json["materials"][material]["pbrMetallicRoughness"]["baseColorTexture"]["index"],
        0
    );
    assert!(merged.document.json["images"][0]["bufferView"].is_number());
    assert!(
        character
            .append_rigid_prop(&source, "held_object", Transform::default())
            .is_err()
    );
}
