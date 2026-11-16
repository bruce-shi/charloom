"""Versioned, agent-directed skin edits and deformation checks for a GLB."""

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .surface import Surface


def digest(path):
    with Path(path).open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def head(report, config):
    return max(
        report["max_edge_growth_m"] / max(config.get("max_edge_growth_m", 0.03), 1e-12),
        report["max_seam_gap_m"] / max(config.get("max_seam_gap_m", 0.001), 1e-12),
    )


def save_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, allow_nan=False)
    os.replace(f.name, path)


def start(source, dest, config):
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        run = Path(tmp) / "session"
        run.mkdir()
        model = run / "round-000.glb"
        from roster.mesh import Glb

        Glb(source).write(model)
        surface = Surface(model)
        if not any(p.weights is not None for p in surface.parts):
            raise ValueError("Skin repair requires a skinned mesh")
        report = surface.analyze(config)
        if config.get("surface_contacts"):
            report["surface_contacts"] = surface.contacts(config["surface_contacts"])
        state = dict(
            source=str(source.resolve()),
            config=config,
            round=0,
            best=model.name,
            best_sha256=digest(model),
            status="in_progress",
            diagnostics=report,
            history=[],
        )
        save_json(run / "session.json", state)
        run.rename(dest)
    return state


def append_attr(g, values, kind):
    values = np.ascontiguousarray(values)
    g.blob.extend(bytes(-len(g.blob) % 4))
    view = len(g.doc["bufferViews"])
    g.doc["bufferViews"].append(
        dict(buffer=0, byteOffset=len(g.blob), byteLength=values.nbytes)
    )
    g.blob.extend(values.tobytes())
    idx = len(g.doc["accessors"])
    g.doc["accessors"].append(
        dict(bufferView=view, componentType=kind, count=len(values), type="VEC4")
    )
    return idx


def edit(surface, plan, contact=None):
    strategy = plan["strategy"]
    if strategy == "separate_surfaces":
        if not contact or not contact.get("times"):
            raise ValueError(
                "Surface separation requires a clip, regions and sample times"
            )
        from . import separation

        constraints = separation.constraints(surface, contact)
    if strategy not in (
        "separate_surfaces",
        "weld_seams",
        "relax_strain",
        "exclude_influences",
    ):
        raise ValueError(f"Unsupported repair strategy {strategy!r}")
    g = surface.scene.glb
    budget = plan.get("max_weight_change_l1", 0.5)
    iterations = plan.get("iterations", 1)
    if not 0 <= budget <= 2 or iterations < 1:
        raise ValueError("Invalid weight-change budget or iteration count")
    changes = 0
    for i, part in enumerate(surface.parts):
        if part.weights is None:
            continue
        n = len(part.positions)
        ws = np.zeros((n, len(part.bones)))
        np.add.at(ws, (np.arange(n)[:, None], part.palette), part.weights)
        original = ws.copy()
        names = [surface.scene.names[k] for k in part.bones]
        protected = np.isin(names, plan.get("protect_joints", []))
        editable = np.zeros(n, bool)
        ban = np.zeros(ws.shape, bool)
        for region in plan.get("regions", []):
            if region.get("mask") is not None:
                if region["mask"]["mesh"] != i:
                    continue
                selected = np.zeros(n, bool)
                selected[surface.mask(region["mask"])] = True
            else:
                selected = (
                    (surface.rest[i] >= region["min"])
                    & (surface.rest[i] <= region["max"])
                ).all(1)
            editable |= selected
            ban |= selected[:, None] & np.isin(names, region.get("exclude_joints", []))
        if not plan.get("regions") and strategy == "weld_seams":
            editable[:] = True
        if strategy == "separate_surfaces" and not plan.get("regions"):
            for mask in contact["regions"]:
                if mask["mesh"] == i:
                    editable[surface.mask(mask)] = True
        if not editable.any():
            continue
        if strategy == "weld_seams":
            if not len(part.seams):
                continue
            a, b = part.seams.T
            graph = csr_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
            _, groups = connected_components(graph, directed=False)
            for group in np.unique(groups[a]):
                ids = np.flatnonzero(groups == group)
                if (
                    editable[ids].all()
                    and not protected[np.flatnonzero(ws[ids].sum(0))].any()
                ):
                    ws[ids] = ws[ids].mean(0)
        elif strategy == "separate_surfaces":
            if ban.any():
                raise ValueError("Use exclude_influences in a separate repair step")
            separation.apply(
                part,
                ws,
                constraints[i],
                editable,
                protected,
                plan,
                max(4 * contact.get("epsilon_m", 1e-6), 1e-7),
            )
        elif strategy == "relax_strain":
            a, b = part.edges.T
            matrix = csr_matrix(
                (np.ones(len(a) * 2), (np.r_[a, b], np.r_[b, a])), shape=(n, n)
            )
            degree = np.maximum(1, np.asarray(matrix.sum(1)))
            for _ in range(iterations):
                ws[editable] = (ws[editable] + (matrix @ ws / degree)[editable]) * 0.5
        ws[ban] = 0.0
        ws[:, protected] = original[:, protected]
        left = 1.0 - ws[:, protected].sum(1)
        allowed = ws[:, ~protected]
        if np.any((allowed.sum(1) < 1e-15) & (left > 1e-8)):
            raise ValueError("Exclusions remove all available deform weights")
        ws[:, ~protected] = (
            allowed * (left / np.maximum(allowed.sum(1), 1e-20))[:, None]
        )
        ws[~editable] = original[~editable]
        # Keep a protected influence even when smoothing introduces several new neighbors.
        ranked = np.where((ws > 0) & protected, 2.0 + ws, ws)
        indices = np.argsort(-ranked, axis=1, kind="stable")[:, : part.palette.shape[1]]
        weights = np.take_along_axis(ws, indices, axis=1)
        fixed = protected[indices]
        total = np.where(~fixed, weights, 0).sum(1)
        weights *= np.where(
            fixed,
            1.0,
            ((1 - np.where(fixed, weights, 0).sum(1)) / np.maximum(total, 1e-20))[
                :, None
            ],
        )
        new = np.zeros_like(ws)
        np.put_along_axis(new, indices, weights, axis=1)
        l1 = np.abs(new - original).sum(1)
        if (l1 > budget + 1e-6).any():
            raise ValueError(
                f"Weight change {l1.max():.3f} exceeds plan budget {budget}"
            )
        mask = l1 > 1e-7
        if not mask.any():
            continue
        ji = part.palette.astype("<u2").copy()
        wt = part.weights.astype("<f4").copy()
        width = indices.shape[1]
        ji[mask], wt[mask] = 0, 0
        ji[mask, :width] = indices[mask]
        wt[mask, :width] = weights[mask]
        attrs = part.prim["attributes"]
        for offset in range(0, ji.shape[1], 4):
            attrs[f"JOINTS_{offset // 4}"] = append_attr(
                g, ji[:, offset : offset + 4], 5123
            )
            attrs[f"WEIGHTS_{offset // 4}"] = append_attr(
                g, wt[:, offset : offset + 4], 5126
            )
        changes += int(mask.sum())
    return changes


def step(folder, plan):
    path = folder / "session.json"
    with (folder / ".repair.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text())
        if state["round"] != plan["expected_round"]:
            raise ValueError(
                "Skin repair round changed; read this session before editing"
            )
        cfg = state["config"]
        best = folder / state["best"]
        if digest(best) != state["best_sha256"]:
            raise ValueError("Best skin candidate changed outside the session")
        if state["status"] != "in_progress" and not plan.get("resume_after_review"):
            raise ValueError("A closed review requires resume_after_review")
        strategy = plan["strategy"]
        number = state["round"] + 1
        entry = dict(
            round=number,
            strategy=strategy,
            diagnosis=plan["diagnosis"],
            evidence=plan.get("evidence", []),
        )
        if strategy == "update_budget":
            cfg["max_rounds"] = plan.get("round_budget")
        elif strategy in ("accept", "report_unresolved"):
            state["status"] = "accepted" if strategy == "accept" else "unresolved"
        else:
            if cfg.get("max_rounds") is not None and number > cfg["max_rounds"]:
                raise ValueError("Repair session round budget exhausted")
            source = best
            if plan.get("continue_from_round") is not None:
                source = folder / f"round-{plan['continue_from_round']:03d}.glb"
            if (
                plan.get("expected_candidate_hash")
                and digest(source) != plan["expected_candidate_hash"]
            ):
                raise ValueError("Chosen repair candidate hash changed")
            candidate = folder / f"round-{number:03d}.glb"
            src = Surface(source)
            if "surface_contacts" in plan:
                contact = plan["surface_contacts"]
                if strategy != "separate_surfaces" or (
                    cfg.get("surface_contacts") and cfg["surface_contacts"] != contact
                ):
                    raise ValueError(
                        "A repair step may add a crossing check but cannot replace it"
                    )
                if not cfg.get("surface_contacts"):
                    cfg["surface_contacts"] = contact
                    state["diagnostics"]["surface_contacts"] = Surface(best).contacts(
                        contact
                    )
            entry["changed_vertices"] = edit(src, plan, cfg.get("surface_contacts"))
            with tempfile.TemporaryDirectory(dir=folder) as tmp:
                model = Path(tmp) / candidate.name
                src.scene.glb.doc["buffers"][0]["byteLength"] = len(src.scene.glb.blob)
                src.scene.glb.write(model)
                surface = Surface(model)
                result = surface.analyze(cfg)
                if cfg.get("surface_contacts"):
                    result["surface_contacts"] = surface.contacts(
                        cfg["surface_contacts"]
                    )
                entry.update(
                    file=candidate.name, sha256=digest(model), diagnostics=result
                )

                def crossings(d):
                    return sum(
                        r["new_crossings"]
                        for r in d.get("surface_contacts", {}).get("frames", [])
                    )

                if head(result, cfg) <= head(state["diagnostics"], cfg) and crossings(
                    result
                ) <= crossings(state["diagnostics"]):
                    state.update(
                        best=candidate.name,
                        best_sha256=entry["sha256"],
                        diagnostics=result,
                    )
                model.replace(candidate)
            state["status"] = "in_progress"
        state["history"].append(entry)
        state["round"] = number
        save_json(path, state)
        return state
