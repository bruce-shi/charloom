 use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AnatomyClass {
    Human,
    StylizedHumanoid,
    Nonhuman,
    Unknown,
}
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum AnatomyKind {
    Head,
    Body,
    Hand,
    Foot,
    Chain,
    Appendage,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AnatomyStrategy {
    FitPrior,
    PreserveSource,
    Unresolved,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnatomyPart {
    pub id: String,
    pub kind: AnatomyKind,
         pub source_form: String,
    pub prior_form: String,
    pub strategy: AnatomyStrategy,
    pub evidence: String,
    pub confirmed: bool,
    #[serde(default)]
    pub preserved_part_ids: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnatomyReview {
    pub version: u32,
    pub anatomy_class: AnatomyClass,
    pub source_hash: String,
    pub prior_hash: String,
    pub author: String,
    pub evidence: String,
    pub parts: Vec<AnatomyPart>,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AnatomyUse {
    FitBody,
    ReconstructBody,
    CompleteBody,
    FitFace,
    MpfbBatch,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnatomyRequest {
    pub review: Option<AnatomyReview>,
    pub source_hash: String,
    pub prior_hash: String,
    pub usage: AnatomyUse,
    #[serde(default)]
    pub preserved_part_ids: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AnatomyPlan {
    pub anatomy_class: AnatomyClass,
    pub route: String,
    pub fit_parts: Vec<String>,
    pub preserved_parts: Vec<String>,
    pub automatic_classification: bool,
}
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct AnatomyError(pub String);
fn check(condition: bool, code: &str) -> Result<(), AnatomyError> {
    if condition {
        Ok(())
    } else {
        Err(AnatomyError(code.into()))
    }
}
pub fn preflight(request: &AnatomyRequest) -> Result<AnatomyPlan, AnatomyError> {
    use AnatomyKind::*;
    let r = request
        .review
        .as_ref()
        .ok_or_else(|| AnatomyError("anatomy_review_required".into()))?;
    check(r.version == 1, "anatomy_review_version")?;
    check(
        r.anatomy_class != AnatomyClass::Unknown,
        "anatomy_class_unknown",
    )?;
    check(
        !matches!(request.usage, AnatomyUse::MpfbBatch)
            || r.anatomy_class != AnatomyClass::Nonhuman,
        "anatomy_mpfb_requires_compatible_hybrid_route",
    )?;
    check(
        !request.source_hash.is_empty() && r.source_hash == request.source_hash,
        "anatomy_source_hash_mismatch",
    )?;
    check(
        !request.prior_hash.is_empty() && r.prior_hash == request.prior_hash,
        "anatomy_prior_hash_mismatch",
    )?;
    check(
        ["user", "agent", "mixed"].contains(&r.author.as_str()) && !r.evidence.trim().is_empty(),
        "anatomy_review_evidence_required",
    )?;
    check(
        !r.parts.is_empty() && r.parts.len() <= 128,
        "anatomy_part_budget",
    )?;
    let mut ids = BTreeSet::new();
    let mut kinds = BTreeSet::new();
    let mut bound = BTreeSet::new();
    let available: BTreeSet<_> = request.preserved_part_ids.iter().cloned().collect();
    check(
        available.len() == request.preserved_part_ids.len(),
        "anatomy_duplicate_preserved_part",
    )?;
    check(
        available
            .iter()
            .all(|id| !id.trim().is_empty() && id.len() <= 128),
        "anatomy_invalid_preserved_part",
    )?;
    let mut plan = AnatomyPlan {
        anatomy_class: r.anatomy_class,
        route: "reviewed_compatible_prior".into(),
        fit_parts: vec![],
        preserved_parts: vec![],
        automatic_classification: false,
    };
    for p in &r.parts {
        check(
            !p.id.trim().is_empty() && p.id.len() <= 128 && ids.insert(p.id.clone()),
            "anatomy_duplicate_or_empty_part",
        )?;
        if matches!(request.usage, AnatomyUse::FitFace) && p.kind != Head {
            continue;
        }
        kinds.insert(p.kind);
        check(
            p.confirmed && !p.evidence.trim().is_empty(),
            "anatomy_part_confirmation_required",
        )?;
        check(
            !p.source_form.trim().is_empty()
                && !p.prior_form.trim().is_empty()
                && !["unknown", "unresolved"].iter().any(|label| {
                    p.source_form.trim().eq_ignore_ascii_case(label)
                        || p.prior_form.trim().eq_ignore_ascii_case(label)
                }),
            "anatomy_structure_unknown",
        )?;
        check(
            r.anatomy_class != AnatomyClass::Human
                || p.kind == Appendage
                || p.source_form == "human",
            "anatomy_human_class_conflict",
        )?;
        match p.strategy {
            AnatomyStrategy::Unresolved => {
                return Err(AnatomyError("anatomy_part_unresolved".into()));
            }
            AnatomyStrategy::FitPrior => {
                check(p.source_form == p.prior_form, "anatomy_incompatible_prior")?;
                check(
                    p.preserved_part_ids.is_empty(),
                    "anatomy_fit_part_has_preservation",
                )?;
                // The legacy MPFB batch has human head/hand/foot assumptions.
                check(
                    !matches!(request.usage, AnatomyUse::MpfbBatch) || p.source_form == "human",
                    "anatomy_mpfb_human_structure_required",
                )?;
                plan.fit_parts.push(p.id.clone());
            }
            AnatomyStrategy::PreserveSource => {
                check(
                    matches!(request.usage, AnatomyUse::ReconstructBody),
                    "anatomy_hybrid_reconstruction_required",
                )?;
                check(
                    !p.preserved_part_ids.is_empty(),
                    "anatomy_preserved_part_binding_required",
                )?;
                for id in &p.preserved_part_ids {
                    check(
                        available.contains(id) && bound.insert(id.clone()),
                        "anatomy_preserved_part_missing_or_duplicate",
                    )?;
                    plan.preserved_parts.push(id.clone());
                }
            }
        }
    }
    let required: &[_] = if matches!(request.usage, AnatomyUse::FitFace) {
        &[Head]
    } else {
        &[Head, Body, Hand, Foot, Chain]
    };
    check(
        required.iter().all(|k| kinds.contains(k)),
        "anatomy_required_part_review_missing",
    )?;
    check(available == bound, "anatomy_unreviewed_preserved_part")?;
    if !bound.is_empty() {
        plan.route = "hybrid_preserve_source".into();
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> AnatomyRequest {
        AnatomyRequest {
            source_hash: "source".into(),
            prior_hash: "prior".into(),
            usage: AnatomyUse::FitBody,
            preserved_part_ids: vec![],
            review: Some(AnatomyReview {
                version: 1,
                anatomy_class: AnatomyClass::Human,
                source_hash: "source".into(),
                prior_hash: "prior".into(),
                author: "agent".into(),
                evidence: "Synthetic reviewed test fixture".into(),
                parts: [
                    AnatomyKind::Head,
                    AnatomyKind::Body,
                    AnatomyKind::Hand,
                    AnatomyKind::Foot,
                    AnatomyKind::Chain,
                ]
                .into_iter()
                .enumerate()
                .map(|(i, kind)| AnatomyPart {
                    id: i.to_string(),
                    kind,
                    source_form: "human".into(),
                    prior_form: "human".into(),
                    strategy: AnatomyStrategy::FitPrior,
                    evidence: "matching synthetic structure".into(),
                    confirmed: true,
                    preserved_part_ids: vec![],
                })
                .collect(),
            }),
        }
    }
    #[test]
    fn unknown_missing_stale_and_unreviewed_inputs_never_default_to_human() {
        let good = request();
        assert!(preflight(&good).is_ok());
        for case in 0..6 {
            let mut r = good.clone();
            match case {
                0 => r.review = None,
                1 => r.review.as_mut().unwrap().anatomy_class = AnatomyClass::Unknown,
                2 => r.source_hash = "stale".into(),
                3 => r.prior_hash = "stale".into(),
                4 => r.review.as_mut().unwrap().parts[0].confirmed = false,
                _ => {
                    r.review.as_mut().unwrap().parts.pop();
                }
            }
            assert!(preflight(&r).is_err(), "case {case}");
        }
    }
    #[test]
    fn stylized_ratios_do_not_waive_structural_compatibility() {
        let mut r = request();
        let p = r.review.as_mut().unwrap();
        p.anatomy_class = AnatomyClass::StylizedHumanoid;
        p.parts[2].source_form = "three_digit_hand".into();
        assert!(preflight(&r).unwrap_err().0.contains("incompatible_prior"));
        r.review.as_mut().unwrap().parts[2].prior_form = "three_digit_hand".into();
        assert!(preflight(&r).is_ok());
    }
    #[test]
    fn nonhuman_preservation_requires_an_executed_hybrid_binding() {
        let mut r = request();
        r.review.as_mut().unwrap().anatomy_class = AnatomyClass::Nonhuman;
        let foot = &mut r.review.as_mut().unwrap().parts[3];
        foot.source_form = "cloven_hoof".into();
        foot.strategy = AnatomyStrategy::PreserveSource;
        foot.preserved_part_ids = vec!["original-hooves".into()];
        assert!(preflight(&r).is_err());
        r.usage = AnatomyUse::ReconstructBody;
        assert!(preflight(&r).is_err());
        r.preserved_part_ids = vec!["original-hooves".into()];
        assert_eq!(preflight(&r).unwrap().route, "hybrid_preserve_source");
        r.preserved_part_ids.push("unreviewed-tail".into());
        assert!(preflight(&r).is_err());
        r.usage = AnatomyUse::MpfbBatch;
        assert!(preflight(&r).is_err());
    }
}
