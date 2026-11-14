 use std::path::PathBuf;

#[test]
#[ignore = "needs CHARLOOM_TEST_MH_SOURCE / _PRIOR / _MOTION assets"]
fn generated_scan_becomes_a_clean_rigged_makehuman_character() {
    let var = |k: &str| PathBuf::from(std::env::var(k).unwrap_or_else(|_| panic!("set {k}")));
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("run");
    let guidance = std::env::var("CHARLOOM_TEST_MH_GUIDANCE")
        .ok()
        .map(PathBuf::from);
    let head = std::env::var("CHARLOOM_TEST_MH_HEAD")
        .ok()
        .map(|h| serde_json::from_value(serde_json::json!(h)).expect("head route"));
    let config = charloom_mh::Config {
        guidance,
        head,
        motions: vec![charloom_mh::Motion {
            path: var("CHARLOOM_TEST_MH_MOTION"),
            clip: None,
            name: "walk".into(),
        }],
        check_frames: 12,
        ..Default::default()
    };
    let r = charloom_mh::run(
        &var("CHARLOOM_TEST_MH_SOURCE"),
        &var("CHARLOOM_TEST_MH_PRIOR"),
        &out,
        &config,
    )
    .unwrap();
    let body = &r["stage2_body_fit"];
    assert!(
        body["inverted_triangles"].as_u64().unwrap() < 50,
        "body folds: {}",
        body["flipped_by_region"]
    );
    let fit = &body["distortion_vs_prior_rest"];
    assert!(
        fit["edge_ratio"]["max"].as_f64().unwrap() < 6.0,
        "body stretch: {fit}"
    );
    assert!(r["export_roundtrip_max_error_m"].as_f64().unwrap() < 1e-5);
    assert!(r["stage4_garments"]["unbound_vertices"].as_u64().unwrap() == 0);
    let check = &r["stage5_checks"][0];
    assert!(
        check["inside_fraction_over_frames"]["max"]
            .as_f64()
            .unwrap()
            < 0.02,
        "poke-through: {check}"
    );
    for g in check["worst_frame"]["garments"].as_array().unwrap() {
        assert_eq!(
            g["internal_seam_gap_m"].as_f64().unwrap(),
            0.0,
            "cut seam opened: {g}"
        );
    }
    if head == Some(charloom_mh::head_keep::HeadRoute::KeepScan) {
        let h = &r["stage2_head"];
        assert_eq!(h["route"], "keep_scan", "fell back: {h}");
        assert_eq!(h["open_seam_edges"], 0, "{h}");
        assert_eq!(h["nonmanifold_seam_edges"], 0, "{h}");
        // Crossings can remain in the top band where the ring follows a
        // garment edge (hidden under it); the loft itself must not fold.
        assert!(
            h["bridge_crossing_pairs"].as_u64().unwrap() * 20
                < h["bridge_triangles"].as_u64().unwrap(),
            "{h}"
        );
        assert_eq!(h["seam_gap_over_clips"][0]["max_gap_m"], 0.0, "{h}");
    }
    for f in [
        "1-segment/segmentation.glb",
        "3-skin/body.glb",
        "4-garments/character.glb",
        "report.json",
    ] {
        assert!(out.join(f).exists(), "{f}");
    }
}
