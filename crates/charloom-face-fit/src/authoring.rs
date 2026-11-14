//! One serialized operation contract for the browser Worker, CLI and MCP.
use anyhow::{Result, ensure};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    AnatomyPreflight {
        input: charloom_core::anatomy::AnatomyRequest,
    },
    FitBody {
        input: crate::body_guidance::Input,
    },
    FitFace {
        #[schemars(with = "crate::Input")]
        input: Value,
        #[schemars(with = "Vec<crate::Annotation>")]
        annotations: Value,
    },
    CompareFace {
        input: crate::face_quality::Request,
    },
    Mesh {
        #[schemars(with = "crate::browser_mesh::Request")]
        input: Value,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_schema_exposes_the_existing_operation_contracts_and_roundtrips() {
        let schema = serde_json::to_value(schemars::schema_for!(Request)).unwrap();
         
        let outer = schema["oneOf"].as_array().unwrap();
        let variant = outer
            .iter()
            .find(|v| v["properties"]["operation"]["const"] == "mesh")
            .unwrap();
        let reference = variant["properties"]["input"]["$ref"].as_str().unwrap();
        let nested = schema
            .pointer(reference.strip_prefix('#').unwrap())
            .unwrap();
        let operations: Vec<_> = nested["oneOf"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["properties"]["operation"]["const"].as_str().unwrap())
            .collect();
        assert_eq!(
            operations,
            ["inspect", "remesh", "garment", "transport", "bind"]
        );

        let surface = json!({"vertices":[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],"triangles":[[0,1,2]]});
        let request = json!({"schema_version":1,"operation":"mesh","input":{
            "operation":"bind","mesh":surface,"points":[[0.25,0.25,0.01]]}});
        let decoded: Request = serde_json::from_value(request).unwrap();
        let roundtrip = serde_json::to_value(&decoded).unwrap();
        let output = run(serde_json::from_value(roundtrip).unwrap()).unwrap();
        assert_eq!(output["indices"], json!([0, 1, 2]));
        assert_eq!(output["weights"], json!([0.5, 0.25, 0.25]));
        assert_eq!(
            output["report"]["method"],
            "rest_surface_barycentric_garment_binding"
        );
        assert_eq!(output["schema_version"], 1);
    }
}
#[derive(Serialize, Deserialize, JsonSchema)]
pub struct Request {
    pub schema_version: u32,
    #[serde(flatten)]
    pub operation: Operation,
}
pub fn capabilities() -> Value {
    json!({"schema_version":1,"anatomy_review_required_for_priors":true,"anatomy_classifier":false,"operations":["anatomy_preflight","fit_body","fit_face","compare_face","mesh"],"shared_runtimes":["native_cli","native_mcp","browser_wasm"],"body_coordinate_system":"gltf_y_up_m","face_coordinate_system":"blender_z_up_m","body_guidance":"explicit_confirmed_landmarks_and_semantic_surface_regions","unique_hidden_anatomy_recovery":false,"visual_approval_required":true})
}
pub fn run(request: Request) -> Result<Value> {
    ensure!(request.schema_version == 1, "authoring_schema_version");
    let (name, mut output) = match request.operation {
        Operation::AnatomyPreflight { input } => (
            "anatomy_preflight",
            json!({"report":charloom_core::anatomy::preflight(&input)?}),
        ),
        Operation::FitBody { input } => ("fit_body", crate::body_guidance::solve(input)?),
        Operation::FitFace { input, annotations } => {
            let input: crate::Input = serde_json::from_value(input)?;
            let annotations: Vec<crate::Annotation> = serde_json::from_value(annotations)?;
            validate_face(&input, &annotations)?;
            (
                "fit_face",
                serde_json::to_value(crate::solve(input, annotations)?)?,
            )
        }
        Operation::Mesh { input } => (
            "mesh",
            serde_json::from_str(&crate::browser_mesh::process(&input.to_string())?)?,
        ),
        Operation::CompareFace { input } => ("compare_face", crate::face_quality::run(input)?),
    };
    output["schema_version"] = json!(1);
    output["operation"] = json!(name);
    Ok(output)
}
pub fn process(request: &str) -> Result<String> {
    Ok(serde_json::to_string(&run(serde_json::from_str(
        request,
    )?)?)?)
}
pub fn validate_face(input: &crate::Input, annotations: &[crate::Annotation]) -> Result<()> {
    let n = input.template.len();
    ensure!(
        n <= 1_000_000 && input.source.len() <= 1_000_000 && input.triangles.len() <= 1_500_000,
        "Fit input exceeds the interactive processing limit"
    );
    let valid_id = |i: &usize| *i < n;
    ensure!(
        input.groups.values().flatten().all(valid_id)
            && input.face_regions.outer.iter().all(valid_id)
            && input.face_regions.cavities.iter().flatten().all(valid_id)
            && input.face_regions.scalp_preserve.iter().all(valid_id)
            && input.face_regions.unbound.iter().all(valid_id)
            && input.face_regions.bindings.iter().all(|b| b.vertex < n
                && b.cage.iter().all(valid_id)
                && b.weights.iter().all(|v| v.is_finite()))
            && annotations.iter().all(|a| a
                .template_xz
                .iter()
                .chain(&a.source_xz)
                .all(|v| v.is_finite())
                && a.source_y.is_none_or(|v| v.is_finite())
                && a.template_vertex.is_none_or(|i| i < n)
                && a.weight.is_finite()
                && a.weight > 0.)
            && input
                .landmark_triangles
                .iter()
                .flatten()
                .all(|&i| i < input.source.len())
            && input
                .morphs
                .iter()
                .flat_map(|m| &m.deltas)
                .all(|&(i, x, y, z)| i < n && [x, y, z].iter().all(|v| v.is_finite())),
        "Invalid anatomical indices or landmark coordinates"
    );
    Ok(())
}
#[cfg(not(target_arch = "wasm32"))]
pub fn run_file(input: &std::path::Path, out: &std::path::Path) -> Result<Value> {
    use std::io::Write;
    ensure!(!out.exists(), "output_exists: {}", out.display());
    let metadata = std::fs::metadata(input)?;
    ensure!(
        metadata.len() <= 256 * 1024 * 1024,
        "authoring_request_size_limit"
    );
    let result = run(serde_json::from_slice(&std::fs::read(input)?)?)?;
    let bytes = serde_json::to_vec(&result)?;
    // create_new also protects against another writer appearing while we solve.
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out)?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(json!({"output":out,"operation":result["operation"],"report":result["report"]}))
}
