use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::PathBuf};
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn main() -> Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    ensure!(
        args.len() == 4,
        "Usage: shape-hints input.json result.json annotations.json new-result.json"
    );
    let input_bytes = fs::read(&args[0])?;
    let result_bytes = fs::read(&args[1])?;
    let hint_bytes = fs::read(&args[2])?;
    let input: charloom_face_fit::Input = serde_json::from_slice(&input_bytes).context("input")?;
    let mut result: charloom_face_fit::Output =
        serde_json::from_slice(&result_bytes).context("result")?;
    let hints: charloom_face_fit::shape_hints::HintDocument =
        serde_json::from_slice(&hint_bytes).context("hints")?;
    ensure!(
        hints.reference_result_sha256 == sha(&result_bytes),
        "Stale annotations: reference result SHA256 mismatch"
    );
    ensure!(
        result.report["provenance"]["input_sha256"].as_str() == Some(sha(&input_bytes).as_str()),
        "Input SHA256 does not match the reference result"
    );
    let path = PathBuf::from(&args[3]);
    ensure!(!path.exists(), "Use a new output path");
    charloom_face_fit::shape_hints::apply(&input, &mut result, &hints)?;
    result.report["shape_hints"]["provenance"] = serde_json::json!({"parent_result_sha256":sha(&result_bytes),"annotations_sha256":sha(&hint_bytes),"executable_sha256":sha(&fs::read(std::env::current_exe()?)?),"input_sha256":sha(&input_bytes)});
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("{}.partial", std::process::id()));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    serde_json::to_writer(&mut file, &result)?;
    file.flush()?;
    drop(file);
    fs::hard_link(&temporary, &path)?;
    fs::remove_file(&temporary)?;
    println!("{}", serde_json::to_string(&result.report["shape_hints"])?);
    Ok(())
}
