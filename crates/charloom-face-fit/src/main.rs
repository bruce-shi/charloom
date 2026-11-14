use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    anyhow::ensure!(
        args.len() == 3,
        "Usage: charloom-face-fit input.json landmarks.json output.json"
    );
    let output = PathBuf::from(&args[2]);
    anyhow::ensure!(
        !output.exists(),
        "Use a new output file; existing experiments are not overwritten"
    );
    let input = serde_json::from_reader(fs::File::open(&args[0])?).context("input")?;
    let annotations = serde_json::from_reader(fs::File::open(&args[1])?).context("landmarks")?;
    let mut fitted = charloom_face_fit::solve(input, annotations)?;
    let mut sources = Sha256::new();
    for bytes in [
        include_bytes!("main.rs").as_slice(),
        include_bytes!("lib.rs").as_slice(),
        include_bytes!("wrap.rs").as_slice(),
        include_bytes!("registration.rs").as_slice(),
        include_bytes!("attachment.rs").as_slice(),
        include_bytes!("ears.rs").as_slice(),
        include_bytes!("face_quality.rs").as_slice(),
        include_bytes!("deformation_guard.rs").as_slice(),
        include_bytes!("membrane.rs").as_slice(),
        include_bytes!("contact.rs").as_slice(),
        include_bytes!("surface.rs").as_slice(),
    ] {
        sources.update(bytes);
    }
    fitted.report["provenance"] = serde_json::json!({
        "compiled_sources_sha256":format!("{:x}",sources.finalize()),
        "executable_sha256":format!("{:x}",Sha256::digest(fs::read(std::env::current_exe()?)?)),
        "input_sha256":format!("{:x}",Sha256::digest(fs::read(&args[0])?)),
        "landmarks_sha256":format!("{:x}",Sha256::digest(fs::read(&args[1])?))
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = output.with_extension(format!("{}.partial", std::process::id()));
    let mut writer = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?,
    );
    serde_json::to_writer(&mut writer, &fitted)?;
    writer.flush()?;
    drop(writer);
    // Creating the final name is atomic and cannot overwrite another run.
    fs::hard_link(&temporary, &output)?;
    fs::remove_file(&temporary)?;
    println!("{}", serde_json::to_string(&fitted.report)?);
    Ok(())
}
