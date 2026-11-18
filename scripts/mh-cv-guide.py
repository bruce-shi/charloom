"""Computer-vision guidance prototype: MediaPipe body, hand and face
landmarks on the guidance views, compared with the vision agent's marks.

    out/tools/cv/.venv/bin/python scripts/mh-cv-guide.py OUT_DIR GUIDE_DIR...

Per character it detects
  * the 13 body joints in the front body view (the original render and the
    bodysuit edit, `views/bodysuit_<view>-reg.png`, when there is one);
  * the finger and palm points in every hand zoom the agent marked;
  * the eye corners and lids, nose tip and mouth corners in `face_y000`;
and reports each detected point's distance to the agent's mark of the same
name, normalised to a 1.75 m figure (mm), plus what failed to detect.
`--write GUIDE_DIR...` puts MediaPipe's body joints from the bodysuit
edits (front and side view) into guidance.json instead (see `write`).
Writes OUT_DIR/cv-guide.json and OUT_DIR/<name>-{body,face}.png overlays
(agent's marks red crosses, detected blue/green rings).
"""
import json
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageDraw

MODELS = Path(__file__).resolve().parent.parent / "out/tools/cv/models"

POSE = {
    "shoulder_l": 11, "shoulder_r": 12, "elbow_l": 13, "elbow_r": 14,
    "wrist_l": 15, "wrist_r": 16, "hip_l": 23, "hip_r": 24,
    "knee_l": 25, "knee_r": 26, "ankle_l": 27, "ankle_r": 28,
}
FACE = {
    "eye_l_inner": 362, "eye_l_outer": 263, "eye_l_top": 386, "eye_l_bottom": 374,
    "eye_r_inner": 133, "eye_r_outer": 33, "eye_r_top": 159, "eye_r_bottom": 145,
    "nose_tip": 4, "mouth_l": 291, "mouth_r": 61,
}
# Agent's finger{f}_{side}_{base,mid,tip} (1 = thumb) -> MediaPipe hand index.
HAND = {
    (1, "base"): 2, (1, "mid"): 3, (1, "tip"): 4,
    (2, "base"): 5, (2, "mid"): 6, (2, "tip"): 8,
    (3, "base"): 9, (3, "mid"): 10, (3, "tip"): 12,
    (4, "base"): 13, (4, "mid"): 14, (4, "tip"): 16,
    (5, "base"): 17, (5, "mid"): 18, (5, "tip"): 20,
}


def load(path):
    img = Image.open(path).convert("RGB")
    return img, mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img))


def figure_rows(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    bg = np.median(np.stack([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]]), axis=0)
    rows = np.where((np.abs(a - bg).sum(axis=2) > 40).any(axis=1))[0]
    return rows.min(), rows.max()


def figure_height_px(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    bg = np.median(np.stack([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]]), axis=0)
    fg = np.abs(a - bg).sum(axis=2) > 40
    rows = np.where(fg.any(axis=1))[0]
    return rows.max() - rows.min() + 1


def marks_in(g, view):
    out = {}
    for name, obs in g["landmarks"].items():
        for o in obs:
            if o["view"] == view:
                out[name] = o["px"]
    return out


def main(out_dir, folders):
    out_dir.mkdir(parents=True, exist_ok=True)
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_heavy.task")),
        num_poses=1, min_pose_detection_confidence=0.3))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
        num_hands=2, min_hand_detection_confidence=0.2))
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
        num_faces=1, min_face_detection_confidence=0.2))
    report = {}
    for folder in folders:
        name = folder.name.removeprefix("mh-guide-")
        g = json.loads((folder / "guidance.json").read_text())
        cams = {c["id"]: c for c in json.loads((folder / "views.json").read_text())["cameras"]}
        front = g["front_view"]
        fh = figure_height_px(folder / "views" / f"{front}.png")
        mm_per = lambda view: cams[view]["meters_per_pixel"] / (fh * cams[front]["meters_per_pixel"]) * 1750.0
        r = {"body": {}, "body_suit": {}, "hand": {}, "face": {}, "missing": [], "points": {}}
        top, bottom = figure_rows(folder / "views" / f"{front}.png")
        r["figure_rows"] = [int(top), int(bottom)]
        # Body joints, front view: original and bodysuit edit.
        agent = marks_in(g, front)
        for key, path in (("body", folder / "views" / f"{front}.png"),
                          ("body_suit", folder / "views" / f"bodysuit_{front}-reg.png")):
            if not path.exists():
                continue
            img, mimg = load(path)
            res = pose.detect(mimg)
            if not res.pose_landmarks:
                r["missing"].append(key)
                continue
            lm = res.pose_landmarks[0]
            det = {k: (lm[i].x * img.width, lm[i].y * img.height) for k, i in POSE.items()}
            # Neck: the agent marks the neck's base; the shoulders' midpoint
            # lifted by a fifth of the shoulder span.
            s0, s1 = det["shoulder_l"], det["shoulder_r"]
            span = abs(s0[0] - s1[0])
            det["neck"] = ((s0[0] + s1[0]) / 2, (s0[1] + s1[1]) / 2 - 0.2 * span)
            r["points"][key] = {k: [round(p[0], 1), round(p[1], 1)] for k, p in det.items()}
            r["points"]["agent"] = {k: agent[k] for k in det if k in agent}
            for k, p in det.items():
                if k in agent:
                    q = agent[k]
                    r[key][k] = round(float(np.hypot(p[0] - q[0], p[1] - q[1])) * mm_per(front), 1)
            if True:
                d = ImageDraw.Draw(img)
                for k, p in det.items():
                    d.ellipse([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], outline=(0, 90, 255), width=3)
                for k, q in agent.items():
                    if k in det:
                        d.line([q[0] - 6, q[1], q[0] + 6, q[1]], fill=(255, 0, 0), width=3)
                        d.line([q[0], q[1] - 6, q[0], q[1] + 6], fill=(255, 0, 0), width=3)
                img.save(out_dir / f"{name}-{key}.png")
        # Hands: every zoom view with finger marks.
        for view in sorted({o["view"] for k, obs in g["landmarks"].items()
                            if k.startswith(("finger", "palm")) for o in obs}):
            path = folder / "views" / f"{view}.png"
            if not path.exists() or view not in cams:
                continue
            agent = marks_in(g, view)
            fingers = {k: v for k, v in agent.items() if k.startswith("finger")}
            img, mimg = load(path)
            res = hands.detect(mimg)
            if not res.hand_landmarks:
                r["missing"].append(f"hand:{view}")
                continue
            # The detected hand nearest the marked fingers.
            mx = np.mean([v[0] for v in fingers.values()]) if fingers else img.width / 2
            my = np.mean([v[1] for v in fingers.values()]) if fingers else img.height / 2
            best = min(res.hand_landmarks, key=lambda h: np.hypot(
                np.mean([p.x for p in h]) * img.width - mx, np.mean([p.y for p in h]) * img.height - my))
            for k, q in fingers.items():
                f, side, part = k[6], k.split("_")[1], k.split("_")[2]
                i = HAND.get((int(f), part))
                if i is None:
                    continue
                p = (best[i].x * img.width, best[i].y * img.height)
                r["hand"][f"{view}:{k}"] = round(float(np.hypot(p[0] - q[0], p[1] - q[1])) * mm_per(view), 1)
        # Face: face_y000.
        if (folder / "views" / "face_y000.png").exists() and "face_y000" in cams:
            agent = marks_in(g, "face_y000")
            img, mimg = load(folder / "views" / "face_y000.png")
            res = face.detect(mimg)
            if not res.face_landmarks:
                r["missing"].append("face")
            else:
                lm = res.face_landmarks[0]
                d = ImageDraw.Draw(img)
                for k, i in FACE.items():
                    p = (lm[i].x * img.width, lm[i].y * img.height)
                    d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], outline=(0, 255, 0), width=2)
                    if k in agent:
                        q = agent[k]
                        r["face"][k] = round(float(np.hypot(p[0] - q[0], p[1] - q[1])) * mm_per("face_y000"), 1)
                        d.line([q[0] - 5, q[1], q[0] + 5, q[1]], fill=(255, 0, 0), width=2)
                        d.line([q[0], q[1] - 5, q[0], q[1] + 5], fill=(255, 0, 0), width=2)
                img.save(out_dir / f"{name}-face.png")
        report[name] = r
        summ = lambda d: (round(float(np.median(list(d.values()))), 1), round(float(np.percentile(list(d.values()), 90)), 1)) if d else None
        print(f"{name:16} body {summ(r['body'])} suit {summ(r['body_suit'])} hand {summ(r['hand'])} ({len(r['hand'])}) face {summ(r['face'])} missing {r['missing']}")
    (out_dir / "cv-guide.json").write_text(json.dumps(report, indent=1))


# Joints `--write` takes from the bodysuit edit. Ankles stay the agent's:
# MediaPipe's ankle is the malleolus, 35-60 mm above our ankle joint, and
# the edit swaps shoes for bare feet. The neck agrees to 8 mm already.
WRITE_JOINTS = ["shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r",
                "hip_l", "hip_r", "knee_l", "knee_r"]


def write(folders):
    """Replace the agent's body joint marks with MediaPipe's on the
    registered bodysuit edits (front and side view), where it sees them
    (visibility >= 0.5); backs up guidance.json once to
    guidance.before-cv.json. Run `scripts/mh-bodysuit.py --write` after it
    so the outline stations move to the new joint rows."""
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_heavy.task")),
        num_poses=1, min_pose_detection_confidence=0.3))
    for folder in folders:
        g = json.loads((folder / "guidance.json").read_text())
        front = g["front_view"]
        side = f"body_y{(int(front.split('_y')[1]) + 90) % 360:03d}"
        backup = folder / "guidance.before-cv.json"
        if not backup.exists():
            backup.write_text(json.dumps(g, indent=1))
        changed = {}
        for view in (front, side):
            path = folder / "views" / f"bodysuit_{view}-reg.png"
            if not path.exists():
                continue
            img, mimg = load(path)
            res = pose.detect(mimg)
            if not res.pose_landmarks:
                print(f"{folder.name}: no pose in {path.name}")
                continue
            lm = res.pose_landmarks[0]
            for k in WRITE_JOINTS:
                q = lm[POSE[k]]
                if (q.visibility or 0) < 0.5:
                    continue
                px = [round(q.x * img.width, 1), round(q.y * img.height, 1)]
                if view == side:
                    # The side view gives depth only: the views share the
                    # camera's height and scale, so a side joint off its
                    # front height by more than 3 % of the image is another
                    # joint (the orc's side shoulder_r sat at its neck, its
                    # wrists on its belly); kept, it is moved to that height.
                    f = [o for o in g["landmarks"].get(k, []) if o["view"] == front]
                    if not f or abs(px[1] - f[0]["px"][1]) > 0.03 * img.height:
                        g["landmarks"][k] = [o for o in g["landmarks"].get(k, []) if o["view"] != view]
                        continue
                    px[1] = f[0]["px"][1]
                obs = [o for o in g["landmarks"].get(k, []) if o["view"] != view]
                obs.append({"view": view, "px": px,
                            "confidence": 0.8, "reason": "MediaPipe pose on the bodysuit edit"})
                g["landmarks"][k] = obs
                changed.setdefault(view, []).append(k)
        (folder / "guidance.json").write_text(json.dumps(g, indent=1))
        print(f"{folder.name}: " + "; ".join(f"{v} {len(ks)}" for v, ks in changed.items()))


if __name__ == "__main__":
    if sys.argv[1] == "--write":
        write([Path(p) for p in sys.argv[2:]])
    else:
        main(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
