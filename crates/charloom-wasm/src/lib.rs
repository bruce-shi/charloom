//! Browser adapter for the same local-deformation code used by the native CLI.
//! File access, timing, scheduling and cancellation belong to the Web Worker.
use charloom_face_fit::shape_hints::{DeformationMesh, HintDocument, LocalDeformer};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
struct Bundle {
    schema_version: u32,
    character_id: String,
    reference_result_sha256: String,
    mesh: DeformationMesh,
}
fn error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[wasm_bindgen]
pub struct Deformer {
    character_id: String,
    reference_result_sha256: String,
    solver: LocalDeformer,
    report: String,
}
#[wasm_bindgen]
impl Deformer {
    #[wasm_bindgen(constructor)]
    pub fn new(bundle: &str) -> Result<Deformer, JsValue> {
        let data: Bundle = serde_json::from_str(bundle).map_err(error)?;
        if data.schema_version != 1 {
            return Err(error("Unsupported deformation data version"));
        }
        Ok(Self {
            character_id: data.character_id,
            reference_result_sha256: data.reference_result_sha256,
            solver: LocalDeformer::new(data.mesh).map_err(error)?,
            report: "null".into(),
        })
    }
    /// Fast preview only: no self-intersection search, no acceptance claims.
    pub fn preview(&mut self, request: &str) -> Result<Vec<f32>, JsValue> {
        let doc = self.document(request)?;
        let points = self.solver.preview(&doc.points).map_err(error)?;
        Ok(self.solver.render_offsets(&points))
    }
    /// Full body-surface fold/collapse and new-crossing checks with guarded backoff.
    /// Existing contacts are measured against the immutable selected result.
    pub fn validate(&mut self, request: &str) -> Result<Vec<f32>, JsValue> {
        let doc = self.document(request)?;
        let (points, report) = self.solver.checked(&doc.points).map_err(error)?;
        self.report = report.to_string();
        Ok(self.solver.render_offsets(&points))
    }
    pub fn report(&self) -> String {
        self.report.clone()
    }
}

/// Full anatomical registration, morph fitting, hidden-anatomy transport and
/// neck assembly. No filesystem, Python, Blender or server process is involved.
#[wasm_bindgen]
pub fn fit_face(input: &str, annotations: &str) -> Result<String, JsValue> {
    let input: charloom_face_fit::Input = serde_json::from_str(input).map_err(error)?;
    let annotations: Vec<charloom_face_fit::Annotation> =
        serde_json::from_str(annotations).map_err(error)?;
    charloom_face_fit::authoring::validate_face(&input, &annotations).map_err(error)?;
    let result = charloom_face_fit::solve(input, annotations).map_err(error)?;
    serde_json::to_string(&result).map_err(error)
}

#[wasm_bindgen]
pub fn process_mesh(request: &str) -> Result<String, JsValue> {
    charloom_face_fit::browser_mesh::process(request).map_err(error)
}
impl Deformer {
    fn document(&self, request: &str) -> Result<HintDocument, JsValue> {
        let doc: HintDocument = serde_json::from_str(request).map_err(error)?;
        if doc.character_id != self.character_id
            || doc.reference_result_sha256 != self.reference_result_sha256
            || doc.schema_version != 1
            || doc.coordinate_system != "gltf_y_up_m"
        {
            return Err(error(
                "Deformation request does not match this model version",
            ));
        }
        // An empty request is an exact reset. Nonempty requests use the CLI limits.
        if !doc.points.is_empty() {
            doc.validate().map_err(error)?;
        }
        Ok(doc)
    }
}

#[wasm_bindgen]
pub fn authoring(request: &str) -> Result<String, JsValue> {
    charloom_face_fit::authoring::process(request).map_err(error)
}

/// Persistent body input for repeated parameter edits. Geometry validation,
/// fingerprints, adjacency and existing crossings belong to this immutable input.
#[wasm_bindgen]
pub struct BodyFitter {
    solver: charloom_face_fit::body_guidance::PreparedBody,
}
#[wasm_bindgen]
impl BodyFitter {
    #[wasm_bindgen(constructor)]
    pub fn new(input: &str) -> Result<BodyFitter, JsValue> {
        let input: charloom_face_fit::body_guidance::Input =
            serde_json::from_str(input).map_err(error)?;
        let solver = charloom_face_fit::body_guidance::PreparedBody::new(input.mesh, input.source)
            .map_err(error)?;
        Ok(Self { solver })
    }
    pub fn solve(&self, guidance: &str) -> Result<String, JsValue> {
        let guidance = serde_json::from_str(guidance).map_err(error)?;
        let mut result = self.solver.solve(&guidance).map_err(error)?;
        result["schema_version"] = serde_json::json!(1);
        result["operation"] = serde_json::json!("fit_body");
        serde_json::to_string(&result).map_err(error)
    }
}
