use charloom_local::{Change, store::Store};
use serde_json::json;
fn set(path: &str, value: serde_json::Value) -> Change {
    Change::Set {
        path: path.into(),
        value,
    }
}
#[test]
fn revisions_transactions_restart_and_single_writer() {
    let root = tempfile::tempdir().unwrap();
    let store = Store::new(root.path()).unwrap();
    assert!(
        Store::new(root.path())
            .err()
            .unwrap()
            .to_string()
            .contains("local_data_in_use")
    );
    let p = store
        .create("Local".into(), Some("test".into()), None)
        .unwrap();
    let hash = store.blob(b"immutable source").unwrap();
    let bytes = std::fs::read(store.blob_path(&hash).unwrap()).unwrap();
    let first = store
        .edit(
            &p.id,
            0,
            vec![set("/datasets/vision", json!({"confidence":0.8}))],
            "cli".into(),
            "request1".into(),
        )
        .unwrap();
    assert_eq!(first.revision, 1);
    assert!(
        store
            .edit(
                &p.id,
                0,
                vec![set("/name", json!("stale"))],
                "web".into(),
                "stale".into()
            )
            .unwrap_err()
            .to_string()
            .contains("revision_conflict:1")
    );
    assert_eq!(store.read(&p.id).unwrap().name, "Local");
    assert!(
        store
            .edit(
                &p.id,
                1,
                vec![
                    set("/name", json!("must rollback")),
                    set("/missing/child", json!(1))
                ],
                "cli".into(),
                "invalid".into()
            )
            .is_err()
    );
    assert_eq!(store.read(&p.id).unwrap().revision, 1);
    // Retrying a lost acknowledgement does not execute a second change.
    let retried = store
        .edit(
            &p.id,
            0,
            vec![set("/datasets/vision", json!({"confidence":0.8}))],
            "cli".into(),
            "request1".into(),
        )
        .unwrap();
    assert_eq!(retried.revision, 1);
    drop(store);
    let reopened = Store::new(root.path()).unwrap();
    let saved = reopened.read("test").unwrap();
    assert_eq!(saved.document["datasets"]["vision"]["confidence"], 0.8);
    assert_eq!(saved.revision, 1);
    assert_eq!(
        std::fs::read(reopened.blob_path(&hash).unwrap()).unwrap(),
        bytes
    );
    assert_eq!(reopened.list().unwrap()[0].revision, 1);
}
#[test]
fn invalid_input_never_creates_an_empty_project() {
    let root = tempfile::tempdir().unwrap();
    let source = tempfile::tempdir().unwrap();
    let store = Store::new(root.path()).unwrap();
    let bad = source.path().join("bad.json");
    std::fs::write(&bad, b"{broken").unwrap();
    assert!(store.open_path(&bad, None, None, None).is_err());
    assert!(store.list().unwrap().is_empty());
    let obj = source.path().join("body.obj");
    std::fs::write(&obj, b"v 0 0 0\n").unwrap();
    assert!(
        store
            .open_path(&obj, None, None, Some("invalid".into()))
            .is_err()
    );
    assert!(store.list().unwrap().is_empty());
}
#[test]
fn native_request_reads_canonical_bindings_and_publishes_candidate() {
    let root = tempfile::tempdir().unwrap();
    let store = Store::new(root.path()).unwrap();
    let p = store.create("Fit".into(), None, None).unwrap();
    store
        .edit(
            &p.id,
            0,
            vec![set("/datasets/points", json!([[0.2, 0.2, 0.02]]))],
            "web".into(),
            "points".into(),
        )
        .unwrap();
    let request = json!({"schema_version":1,"operation":"mesh","input":{"operation":"bind","mesh":{"vertices":[[0,0,0],[1,0,0],[0,1,0]],"triangles":[[0,1,2]]},"points":[]}});
    let output = store
        .run(
            &p.id,
            1,
            Some(request),
            None,
            vec![charloom_local::Binding {
                target: "/input/points".into(),
                source: "/datasets/points".into(),
            }],
            Some("binding_candidate".into()),
        )
        .unwrap();
    assert_eq!(output["status"], "published");
    let current = store.read(&p.id).unwrap();
    assert_eq!(current.revision, 2);
    assert_eq!(
        current.document["datasets"]["binding_candidate"]["indices"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert!(
        root.path()
            .join("results")
            .join(&p.id)
            .join(format!("{}.json", output["result_id"].as_str().unwrap()))
            .is_file()
    );
}
#[test]
fn shared_parameter_shapes_and_pointer_keys_are_enforced() {
    let root = tempfile::tempdir().unwrap();
    let store = Store::new(root.path()).unwrap();
    let p = store.create("Fit".into(), None, None).unwrap();
    for changes in [
        vec![set(
            "/fit",
            json!({"annotations":[{"name":"nose","weight":"bad"}]}),
        )],
        vec![set("/fit", json!({"bodyGuidance":{"version":"invented"}}))],
        vec![set("/datasets/__proto__", json!({}))],
        vec![set("/id", json!("other"))],
    ] {
        assert!(
            store
                .edit(&p.id, 0, changes, "cli".into(), "bad".into())
                .is_err()
        );
    }
    assert_eq!(store.read(&p.id).unwrap().revision, 0);
    store
        .edit(
            &p.id,
            0,
            vec![set("/datasets/a~1b~0c", json!(3))],
            "cli".into(),
            "escaped".into(),
        )
        .unwrap();
    assert_eq!(store.read(&p.id).unwrap().document["datasets"]["a/b~c"], 3);
}

#[test]
fn retries_remain_idempotent_after_another_writer_and_restart() {
    let root = tempfile::tempdir().unwrap();
    let store = Store::new(root.path()).unwrap();
    let p = store.create("Retries".into(), None, None).unwrap();
    store
        .edit(
            &p.id,
            0,
            vec![set("/datasets/items", json!([]))],
            "web".into(),
            "init".into(),
        )
        .unwrap();
    let append = vec![set("/datasets/items/-", json!("once"))];
    store
        .edit(&p.id, 1, append.clone(), "web".into(), "lost-ack".into())
        .unwrap();
    store
        .edit(
            &p.id,
            2,
            vec![set("/datasets/agent", json!(true))],
            "mcp".into(),
            "intervening".into(),
        )
        .unwrap();
    drop(store);
    let store = Store::new(root.path()).unwrap();
    let receipt = store
        .edit(&p.id, 1, append, "web".into(), "lost-ack".into())
        .unwrap();
    assert_eq!(receipt.revision, 3);
    assert_eq!(
        store.read(&p.id).unwrap().document["datasets"]["items"],
        json!(["once"])
    );
    assert!(
        store
            .edit(
                &p.id,
                3,
                vec![set("/datasets/items/-", json!("different"))],
                "web".into(),
                "lost-ack".into()
            )
            .is_err()
    );
}

#[test]
fn buffered_atomic_write_preserves_previous_file_after_partial_serialization_failure() {
    use serde::ser::{Error, SerializeSeq};
    struct FailsAfterWriting;
    impl serde::Serialize for FailsAfterWriting {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut seq = serializer.serialize_seq(None)?;
            // More than the output buffer, so part of the temporary file has
            // reached disk before the serializer reports its error.
            seq.serialize_element(&vec![7_u32; 600_000])?;
            Err(S::Error::custom("injected serialization failure"))
        }
    }
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("project.json");
    charloom_local::store::atomic_json(&path, &json!({"saved":true})).unwrap();
    let previous = std::fs::read(&path).unwrap();
    assert!(charloom_local::store::atomic_json(&path, &FailsAfterWriting).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), previous);
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
}
