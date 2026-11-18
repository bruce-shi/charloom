"""Bald back view: GPT edits of the side (profile) and back views together,
so the occiput and nape under the hair have a GPT target too.

    out/tools/cv/.venv/bin/python scripts/mh-bald-back.py out/examples/mh-guide-NAME [...] [--runs 3]

The suit ensemble (scripts/mh-suit-ensemble.py) edits the front, profile and
two front 3/4 views: no edit sees the back of the head, so the skull below
the inion was interpolated and came out boxy with a shelf over the neck
(x116/x117, 2026-11-09). This asks for RUNS edits of a 1x2 sheet, the
profile beside the back view, with the same bald/green-suit prompt. The
profile panel ties the new back edit to the existing edits' head size.

Writes only views/suit-back/ (input_sheet.png, e<k>_sheet.png and the
panels e<k>_<view>.png, 768x1024, plus back.json). Existing edits are never
touched; nothing is registered or fitted here.
"""
import base64
import json
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path
from importlib import import_module

sys.path.insert(0, str(Path(__file__).resolve().parent))
mhb = import_module("mh-bodysuit")

W, H = 768, 1024
SHEET_W, SHEET_H = 2048, 1360
SHEET = (
    "This sheet shows the SAME 3D character in 2 panels side by side: the side view "
    "(profile, left) and the back view (seen from directly behind, right). Edit both panels "
    "the same way so they show one consistent body and one consistent head. In the back "
    "view the bald scalp, the back of the skull and the nape of the neck must be clearly "
    "visible, with the natural rounded shape of a real skull curving smoothly into the neck. "
)


def views_of(g):
    yaw = int(g["front_view"].split("_y")[1])
    return [f"body_y{(yaw + 90) % 360:03d}", f"body_y{(yaw + 180) % 360:03d}"]


def main(folder: Path, runs: int):
    g = json.loads((folder / "guidance.json").read_text())
    views = views_of(g)
    missing = [v for v in views if not (folder / "views" / f"{v}.png").exists()]
    if missing:
        raise SystemExit(f"{folder.name}: missing renders {missing}")
    out_dir = folder / "views" / "suit-back"
    out_dir.mkdir(exist_ok=True)
    sheet_in = out_dir / "input_sheet.png"
    tiles = [str(folder / "views" / f"{v}.png") for v in views]
    subprocess.run(["magick", *tiles, "+append", "-resize", f"{SHEET_W}x{SHEET_H}!", str(sheet_in)], check=True)
    prompt = SHEET + mhb.PROMPT.replace("this 3D character render", "the sheet")
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in (("model", mhb.MODEL), ("prompt", prompt), ("size", f"{SHEET_W}x{SHEET_H}"),
                        ("quality", "high"), ("n", str(runs))):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="in.png"\r\n'
                 f'Content-Type: image/png\r\n\r\n'.encode() + sheet_in.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        "https://api.openai.com/v1/images/edits", data=b"".join(parts),
        headers={"Authorization": "Bearer " + mhb.key(),
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=1500) as r:
        out = json.loads(r.read())
    for k, d in enumerate(out["data"], 1):
        sheet = out_dir / f"e{k}_sheet.png"
        sheet.write_bytes(base64.b64decode(d["b64_json"]))
        for i, v in enumerate(views):
            subprocess.run(["magick", str(sheet), "-resize", f"{2 * W}x{H}!", "-crop", f"{W}x{H}+{i * W}+0",
                            "+repage", str(out_dir / f"e{k}_{v}.png")], check=True)
    report = {"model": mhb.MODEL, "views": views, "members": len(out["data"]),
              "usage": out.get("usage"), "prompt": prompt}
    (out_dir / "back.json").write_text(json.dumps(report, indent=1))
    print(folder.name, views, len(out["data"]), "members")


if __name__ == "__main__":
    args = sys.argv[1:]
    runs = int(args[args.index("--runs") + 1]) if "--runs" in args else 3
    folders = [Path(a) for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--runs")]
    for f in folders:
        main(f, runs)
