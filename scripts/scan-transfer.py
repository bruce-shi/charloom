"""Fit a MakeHuman template and bind the source wardrobe from a calibrated scan."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from roster.mesh import Glb
from transfer.scan import Scan, transform_points
from transfer.guide import load_annotations, resolve
from transfer.body import fit_body
from transfer.segment import segment, reviewed_owners
from transfer.scaffold import fit_template
from transfer.wardrobe import write_models, write_segmentation



def transfer(
    source,
    prior,
    guidance,
    out,
    size=1024,
    motion=None,
    trace=None,
    decisions=None,
    evidence=None,
    review_out=None,
):
    source, prior, guidance, out = (
        Path(p).resolve() for p in (source, prior, guidance, out)
    )
    if out.exists():
        raise FileExistsError(out)
    scan, template = Scan(source), Glb(prior)
    guide, _, cameras = load_annotations(scan, guidance)
    frame = scan.working_frame(template, guide["front_view"])
    vertices = transform_points(scan.vertices, frame)
    resolved = resolve(scan, guide, cameras, frame)
    names, owners = segment(scan, guide, cameras)
    review, brows = reviewed_owners(
        scan,
        owners,
        names,
        decisions,
        evidence,
        review_out or out.with_name(out.name + ".segment-review.json"),
        vertices,
    )
    resolved["brow_triangle_ranges"] = brows
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as working:
        stage = Path(working) / out.name
        (stage / "0-guide").mkdir(parents=True)
        (stage / "0-guide/resolved.json").write_text(json.dumps(resolved, indent=2))
        (stage / "config.json").write_text(
            json.dumps({"guidance": str(guidance)}, indent=2)
        )
        write_segmentation(scan, vertices, owners, stage)
        (stage / "1-segment/review.json").write_text(json.dumps(review, indent=2))
        pose = fit_template(template, resolved)
        surface = fit_body(template, guidance, cameras, frame)
        wardrobe = write_models(
            template,
            scan,
            vertices,
            names,
            owners,
            stage,
            size,
            Glb(prior),
            guide,
            guidance.parent,
        )
        motion_report = []
        if motion:
            from transfer.motion import sample_motion, retarget

            samples = Path(working) / "motion.npz"
            sample_motion(motion, samples, {}, None)
            with np.load(samples) as take:
                for file in ("3-skin/body.glb", "4-garments/character.glb"):
                    motion_report.append(
                        retarget(
                            take,
                            stage / file,
                            "walking",
                            {"align_reference_chains": True},
                        )
                    )
        if trace:
            import shutil

            shutil.copyfile(trace, stage / "mouth-trace.json")
        report = dict(
            stage5_motions=motion_report,
            operation="python_scan_transfer",
            source=str(source),
            source_hash=scan.source_hash,
            prior=str(prior),
            working_frame={"source_to_working": frame.T.ravel().tolist()},
            stage2_rig=pose,
            stage2_body_fit=surface,
            stage4_garments=wardrobe,
        )
        (stage / "report.json").write_text(json.dumps(report, indent=2))
        stage.rename(out)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    for flag in ("prior", "guidance", "out"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--motion", type=Path)
    p.add_argument("--mouth-trace", type=Path)
    p.add_argument(
        "--review-out",
        type=Path,
        help="Write uncertain components here before requesting decisions",
    )
    p.add_argument(
        "--segment-review",
        type=Path,
        help="Agent decisions for uncertain scan components",
    )
    p.add_argument(
        "--segment-evidence",
        type=Path,
        help="Previously reviewed source triangle ranges",
    )
    p.add_argument("--texture-size", type=int, default=1024)
    args = p.parse_args()
    if args.texture_size < 1:
        p.error("Texture size must be positive")
    print(
        json.dumps(
            transfer(
                args.source,
                args.prior,
                args.guidance,
                args.out,
                size=args.texture_size,
                motion=args.motion,
                trace=args.mouth_trace,
                decisions=args.segment_review,
                evidence=args.segment_evidence,
                review_out=args.review_out,
            )
        )
    )
