"""Intermediate evidence of a guided run, as images the web IDE shows first
(`scripts/mh-ide-sync.py` lists `renders/evidence-*.jpg`).

    out/tools/cv/.venv/bin/python scripts/mh-evidence.py RUN_DIR GUIDE_DIR [RUN_DIR GUIDE_DIR ...]

  evidence-1-sheet.jpg      the bodysuit sheet the image model returned
                            (front + side in one edit, bald, natural hands)
  evidence-2-front.jpg      front view, original | bodysuit: MediaPipe's
  evidence-3-side.jpg       side view, the same
      blue skeleton; cyan rings = joints written into the guidance
      (visibility >= 0.5; grey = seen too poorly, not used) with their
      visibility; red crosses = the vision agent's joint marks before
      MediaPipe (guidance.before-cv.json); yellow lines = the outline
      stations measured on the suit's silhouette (the body's sizes).
The run's own `0-guide/qa-{front,side}.png` (marks red, fitted joints
green) are listed after these.
"""
import json
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageDraw, ImageFont

MODELS = Path(__file__).resolve().parent.parent / "out/tools/cv/models"
EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32), (27, 29), (28, 30)]
USED = {11: "shoulder_l", 12: "shoulder_r", 13: "elbow_l", 14: "elbow_r", 15: "wrist_l",
        16: "wrist_r", 23: "hip_l", 24: "hip_r", 25: "knee_l", 26: "knee_r"}
JOINTS = set(USED.values()) | {"ankle_l", "ankle_r", "neck"}


def font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return None


def panel(guide, view, pose, before, outlines):
    """original | bodysuit for one view, annotated."""
    tiles = []
    suit = guide / "views" / f"bodysuit_{view}-reg.png"
    im = Image.open(suit).convert("RGB")
    res = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(im)))
    f = font(15)
    for src, title in ((guide / "views" / f"{view}.png", "original"), (suit, "bodysuit")):
        base = Image.open(src).convert("RGB")
        d = ImageDraw.Draw(base)
        for k, (v, p) in outlines.items():
            if v == view and k.endswith("1"):
                q = outlines[k[:-1] + "2"][1]
                d.line([tuple(p), tuple(q)], fill=(255, 220, 0), width=2)
                d.text((q[0] + 4, q[1] - 8), k[8:-3], fill=(255, 220, 0), font=f)
        if res.pose_landmarks:
            lm = res.pose_landmarks[0]
            at = lambda i: (lm[i].x * im.width, lm[i].y * im.height)
            for a, b in EDGES:
                d.line([at(a), at(b)], fill=(0, 140, 255), width=3)
            for i, name in USED.items():
                x, y = at(i)
                vis = lm[i].visibility or 0.0
                col = (0, 255, 255) if vis >= 0.5 else (140, 140, 140)
                d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=col, width=3)
                d.text((x + 9, y - 8), f"{name} {vis:.2f}", fill=col, font=f)
        for k, obs in before.get("landmarks", {}).items():
            if k in JOINTS:
                for o in obs:
                    if o["view"] == view:
                        x, y = o["px"]
                        d.line([x - 7, y, x + 7, y], fill=(255, 40, 40), width=3)
                        d.line([x, y - 7, x, y + 7], fill=(255, 40, 40), width=3)
        d.text((10, 8), f"{view} {title}", fill=(255, 255, 255), font=font(20))
        tiles.append(base)
    out = Image.new("RGB", (sum(t.width for t in tiles), max(t.height for t in tiles)))
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width
    return out


def main(pairs):
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_heavy.task")),
        num_poses=1, min_pose_detection_confidence=0.3))
    for run, guide in pairs:
        g = json.loads((guide / "guidance.json").read_text())
        b = guide / "guidance.before-cv.json"
        before = json.loads(b.read_text()) if b.exists() else {}
        front = g["front_view"]
        side = f"body_y{(int(front.split('_y')[1]) + 90) % 360:03d}"
        outlines = json.loads((guide / "views" / "bodysuit.json").read_text()).get("marks", {})
        dst = run / "renders"
        dst.mkdir(exist_ok=True)
        sheet = guide / "views" / "bodysuit_sheet-2k.png"
        if sheet.exists():
            Image.open(sheet).convert("RGB").save(dst / "evidence-1-sheet.jpg", quality=88)
        for i, view in ((2, front), (3, side)):
            if (guide / "views" / f"bodysuit_{view}-reg.png").exists():
                panel(guide, view, pose, before, outlines).save(dst / f"evidence-{i}-{'front' if i == 2 else 'side'}.jpg", quality=88)
        print("wrote", dst)


if __name__ == "__main__":
    a = [Path(p) for p in sys.argv[1:]]
    main(list(zip(a[0::2], a[1::2])))
