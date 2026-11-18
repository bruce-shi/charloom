"""Multi-view hand landmarks, digit count and hand route, measured.

    out/tools/cv/.venv/bin/python -I scripts/mh-hand-landmarks.py [--write] GUIDE_DIR...

The agent's finger marks were the hands' weak point (the troll's: fingertips
only, 10-20 cm off; the brute's 6-11 cm; the orc's five digits on a
four-digit hand). Measured here instead, on the scan:

- **Landmarks.** For each hand a ring of close-ups (`charloom
  mh-guide-zoom --box ... --orbit ...`: the front view and every 30 degrees
  around), MediaPipe's hand landmarker on each (over grey and colour,
  padded, mirrored: its palm detector found none of the troll's green hands
  in colour), and each of the 21 points triangulated from the views' rays
  (orthographic views: a pixel is a ray), dropping views that disagree.
- **Digits.** The scan's hand surface beyond the wrist, each vertex's
  distance along it from the wrist, and the peaks of that distance standing
  out by a cleft of at least PEAK of the hand's size (`measure_digits`):
  fused fingers are one digit. Per digit its chain on the surface (base,
  mid, tip), following its curl. The character's count is the larger of
  its two hands'.
- **Route.** Five digits whose MediaPipe points read as a hand on both
  sides (`plausible`): `subject.hands = "fit"`, the marks MediaPipe's.
  Otherwise `"keep_scan"` (`hand_keep.rs`), the marks the surface chains
  (MakeHuman's bones are posed into the scan's digits for its skinning).

With `--write` guidance.json gets `subject.hand_digits`, `subject.hands`
and per hand `finger{1..n}_{side}_{base,mid,tip}` and `palm_{side}`, each
the projection of its 3D point into two well-separated views (marks of
hand landmarks triangulate, so they resolve to that point). Writes
`hands3d.json` and `views/hands3d-<side>.png`; backs guidance.json up once
to guidance.before-hands.json.
"""
import json
import subprocess
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "out/tools/cv/models"
GUIDE_TOOL = ROOT / "scripts/scan-guide.py"
ORBIT = [30, 60, 90, 120, 150, 180, -30, -60, -90, -120, -150]
# MediaPipe index per guidance digit (thumb = 1): base, mid, tip.
DIGITS = {1: (2, 3, 4), 2: (5, 6, 8), 3: (9, 10, 12), 4: (13, 14, 16), 5: (17, 18, 20)}
MIN_ANGLE = 25.0
# Views agreeing within this share of the hand's size (wrist to middle
# knuckle) count for a point.
AGREE = 0.35
# A digit's tip stands this far (share of the wrist-to-knuckle size) above
# the cleft joining it to its neighbour: digits measured 0.46-0.97, knuckle
# bumps up to 0.27 (orc, hiker, brute).
PEAK = 0.37


def detector():
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
        num_hands=2, min_hand_detection_confidence=0.2, min_hand_presence_confidence=0.2))


def detect(det, img):
    """The hand nearest the view's centre, per point the median over
    detections on variants of the view: grey and original colour, padded
    1.0-2.4x, mirrored or not. MediaPipe's palm detector is trained on
    photos: on the troll's green cartoon hands it found none of 24 views in
    colour, every one with grey and padding."""
    found = []
    bg = img.getpixel((2, 2))
    for colour in (img, ImageOps.grayscale(img).convert("RGB")):
        for f in (1.0, 1.4, 1.8, 2.4):
            W = int(img.width * f)
            off = ((W - img.width) // 2, (W - img.height) // 2)
            canvas = Image.new("RGB", (W, W), bg if colour is img else (bg[0],) * 3)
            canvas.paste(colour, off)
            for mirror in (False, True):
                im = ImageOps.mirror(canvas) if mirror else canvas
                res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(im)))
                for lm in res.hand_landmarks:
                    xs = np.array([(1 - q.x if mirror else q.x) * W - off[0] for q in lm])
                    ys = np.array([q.y * W - off[1] for q in lm])
                    found.append(np.stack([xs, ys], 1))
    if not found:
        return None
    c = np.array([img.width / 2, img.height / 2])
    # Detections of the centred hand: middle knuckle nearest the centre,
    # and those agreeing with it.
    ref = min(found, key=lambda a: np.linalg.norm(a[9] - c))
    near = [a for a in found if np.linalg.norm(a[9] - ref[9]) < 0.15 * img.width]
    med = np.median(np.stack(near), 0)
    return [tuple(p) for p in med]


def mask_rows(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    bg = a[2, 2]
    fg = np.abs(a - bg).sum(-1) > 30
    rows = np.flatnonzero(fg.any(1))
    return rows.min(), rows.max()


def ray(cam, px, py):
    c, r, u, f = (np.array(cam[k], float) for k in ("center", "right", "up", "forward"))
    m = cam["meters_per_pixel"]
    o = c + r * (px - cam["width"] / 2) * m + u * (cam["height"] / 2 - py) * m
    return o, f / np.linalg.norm(f)


def project(cam, p):
    c, r, u = (np.array(cam[k], float) for k in ("center", "right", "up"))
    m = cam["meters_per_pixel"]
    d = p - c
    return [round(float(d @ r / m + cam["width"] / 2), 1), round(float(cam["height"] / 2 - d @ u / m), 1)]


def closest(rays):
    """Least-squares point nearest to all rays (origin, unit direction)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for o, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    return np.linalg.lstsq(A, b, rcond=None)[0]


def dist(p, o, d):
    q = p - o
    return float(np.linalg.norm(q - d * (q @ d)))


def angle(d1, d2):
    return float(np.degrees(np.arccos(np.clip(abs(d1 @ d2), -1, 1))))


def triangulate(obs, size):
    """obs: [(view, origin, dir)]. Robust: the best pair, then every view
    within AGREE * size, refit."""
    best = None
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            if angle(obs[i][2], obs[j][2]) < MIN_ANGLE:
                continue
            p = closest([obs[i][1:], obs[j][1:]])
            inl = [k for k, (_, o, d) in enumerate(obs) if dist(p, o, d) < AGREE * size]
            score = (len(inl), -sum(dist(p, o, d) for _, o, d in obs if dist(p, o, d) < AGREE * size))
            if best is None or score > best[0]:
                best = (score, inl)
    if best is None:
        return None
    inl = best[1]
    p = closest([obs[k][1:] for k in inl])
    dirs = [obs[k][2] for k in inl]
    spread = max(angle(a, b) for a in dirs for b in dirs) if len(dirs) > 1 else 0.0
    if len(inl) < 2 or spread < MIN_ANGLE:
        return None
    res = [dist(p, obs[k][1], obs[k][2]) for k in inl]
    return p, [obs[k][0] for k in inl], float(np.mean(res))


def hand_box(g, front, cams, folder):
    """Pixel box around each hand in the front view: from the wrist mark,
    outwards along the forearm (elbow to wrist), half a hand on."""
    top, bottom = mask_rows(folder / "views" / f"{front}.png")
    fh = bottom - top
    out = {}
    for side in ("l", "r"):
        pt = lambda k: next((o["px"] for o in g["landmarks"].get(k, []) if o["view"] == front), None)
        w, e = pt(f"wrist_{side}"), pt(f"elbow_{side}")
        if w is None:
            continue
        d = np.array([0.0, 1.0]) if e is None else np.array(w, float) - np.array(e, float)
        d /= np.linalg.norm(d) + 1e-9
        c = np.array(w, float) + d * 0.07 * fh
        half = 0.14 * fh
        out[side] = [round(c[0] - half, 1), round(c[1] - half, 1), round(c[0] + half, 1), round(c[1] + half, 1)]
    return out


def load_scan(path):
    """Scan positions (welded by position) and triangles, in the views' frame."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    meshes = importlib.import_module("mh-match").glb_meshes(Path(path))
    pos, tri, off = [], [], 0
    for _, p, t in meshes:
        pos.append(p)
        tri.append(t + off)
        off += len(p)
    pos, tri = np.concatenate(pos), np.concatenate(tri)
    key = np.round(pos / 1e-6).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return pos[first], inv.reshape(-1)[tri]


def measure_digits(pos, tri, pts, size):
    """The scan's digits, measured on its surface: the hand beyond the wrist
    (joined to the middle knuckle), each vertex's distance along the surface
    from the wrist's cut, and the peaks of that distance that stand out from
    their neighbours by a cleft of at least PEAK of the wrist-to-knuckle size
    (persistence of the distance's superlevel sets). Fused fingers are one
    digit; the knuckles' bumps are no peak. Per digit its chain on the scan,
    following its own curl: `tip` (the branch's centre a little short of its
    end), `mid` (halfway), `base` (just above the cleft, carried a third of
    the size back into the palm for a finger: the knuckle sits below the
    web), and `cleft` (its surface distance)."""
    import heapq
    w, k = np.array(pts[0]), np.array(pts[9])
    axis = (k - w) / (np.linalg.norm(k - w) + 1e-12)
    tips = np.array([pts[i] for i in (4, 8, 12, 16, 20)])
    reach = max(np.linalg.norm(tips - w, axis=1).max(), 2.0 * size) * 1.4
    d = pos - w
    near = (np.linalg.norm(d, axis=1) < reach) & (d @ axis > -0.15 * size)
    idx = np.flatnonzero(near)
    if len(idx) < 50:
        return None
    local = -np.ones(len(pos), np.int64)
    local[idx] = np.arange(len(idx))
    lt = local[tri[near[tri].all(1)]]
    adj = [[] for _ in idx]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        for x, y in zip(lt[:, a], lt[:, b]):
            adj[x].append(y)
            adj[y].append(x)
    P = pos[idx]
    start = int(np.argmin(np.linalg.norm(P - k, axis=1)))
    comp = np.zeros(len(idx), bool)
    comp[start] = True
    stack = [start]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if not comp[y]:
                comp[y] = True
                stack.append(y)
    h = (P - w) @ axis
    dist = np.full(len(idx), np.inf)
    heap = [(0.0, int(x)) for x in np.flatnonzero(comp & (h < 0.05 * size))]
    if not heap:
        return None
    for _, x in heap:
        dist[x] = 0.0
    heapq.heapify(heap)
    while heap:
        dx, x = heapq.heappop(heap)
        if dx > dist[x]:
            continue
        for y in adj[x]:
            nd = dx + float(np.linalg.norm(P[x] - P[y]))
            if nd < dist[y]:
                dist[y] = nd
                heapq.heappush(heap, (nd, y))
    order = [x for x in np.argsort(-dist) if comp[x] and np.isfinite(dist[x])]
    parent, birth, label, died = {}, {}, {}, {}
    merged = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for x in order:
        parent[x] = x
        roots = sorted({find(y) for y in adj[x] if y in parent and y != x}, key=lambda r: -birth[r])
        if not roots:
            birth[x] = dist[x]
            label[x] = x
            continue
        oldest = roots[0]
        parent[x] = oldest
        label[x] = oldest
        for r in roots[1:]:
            died[r] = dist[x]
            merged[r] = oldest
            parent[r] = oldest
    top = max(birth, key=lambda r: birth[r])
    persist = {r: birth[r] - died.get(r, 0.0) for r in birth}
    sig = [r for r in birth if persist[r] > PEAK * size]
    # A vertex's digit: its label's chain through the bumps merged into a
    # digit, stopping at the first digit.
    def owner(r):
        while r not in sig and r in merged:
            r = merged[r]
        return r if r in sig else None
    cleft = {r: died[r] for r in sig if r in died}
    for r in sig:
        if r not in cleft:
            under = [died[q] for q in sig if q != r and owner(merged.get(q, q)) == r]
            cleft[r] = max(under) if under else 0.0
    branch = {r: [] for r in sig}
    for x in order:
        o = owner(label[x])
        if o is not None and dist[x] >= cleft[o]:
            branch[o].append(x)
    digits = []
    for r in sig:
        xs = np.array(branch[r])
        if len(xs) < 10:
            continue
        dd, pp = dist[xs], P[xs]
        lo, hi = cleft[r], birth[r]

        def band(a, b):
            m = (dd >= a) & (dd <= b)
            return pp[m].mean(0) if m.any() else None
        tip = band(hi - 0.25 * size, hi - 0.08 * size)
        mid = band((lo + hi) / 2 - 0.07 * size, (lo + hi) / 2 + 0.07 * size)
        base = band(lo, lo + 0.12 * size)
        if tip is None or mid is None or base is None:
            continue
        digits.append({"tip": tip, "mid": mid, "base": base, "end": P[r], "cleft": float(lo), "length": float(hi - lo),
                       "persistence": float(persist[r])})
    return {"count": len(sig), "digits": digits,
            "peaks_per_size": sorted((round(float(q / size), 2) for q in persist.values()), reverse=True)[:8]}


def order_digits(digits, pts):
    """Thumb first (the digit whose end is nearest MediaPipe's thumb tip, or
    with the deepest cleft), then by distance of the base from the thumb's.
    A digit's measured base sits at its web, partway up the finger: each
    finger's is carried down its own axis to the knuckles' plane (across the
    palm through MediaPipe's knuckles, square to wrist-to-middle-knuckle; its
    knuckle line holds whatever the digit count), the thumb's to MediaPipe's
    thumb knuckle when it lies on the thumb."""
    if not digits:
        return []
    P = {i: np.array(pts[i]) for i in pts}
    if 4 in P:
        thumb = min(digits, key=lambda g: np.linalg.norm(g["end"] - P[4]))
    else:
        thumb = min(digits, key=lambda g: g["cleft"])
    rest = sorted((g for g in digits if g is not thumb), key=lambda g: np.linalg.norm(g["base"] - thumb["base"]))
    out = []
    thumb = dict(thumb)
    if 2 in P and np.linalg.norm(P[2] - thumb["base"]) < 1.5 * thumb["length"]:
        thumb["base"] = P[2]
    out.append(thumb)
    have = all(i in P for i in (0, 5, 9, 13, 17))
    if have:
        axis = P[9] - P[0]
        axis /= np.linalg.norm(axis) + 1e-12
        level = np.mean([P[i] for i in (5, 9, 13, 17)], axis=0) @ axis
    for g in rest:
        g = dict(g)
        if have:
            u = g["base"] - g["mid"]
            u /= np.linalg.norm(u) + 1e-12
            drop = (g["base"] @ axis - level) / max(-(u @ axis), 0.3)
            g["base"] = g["base"] + u * float(np.clip(drop, 0.0, 1.2 * g["length"]))
        out.append(g)
    return out


def plausible(points, size):
    """MediaPipe's triangulated hand reads as a hand: palm width, finger and
    thumb lengths (shares of wrist to middle knuckle) and view agreement."""
    P = {i: np.array(points[i]["p"]) for i in points}
    L = lambda a, b: np.linalg.norm(P[a] - P[b]) / size
    chain = lambda i: L(i[0], i[1]) + L(i[1], i[2])
    checks = {
        "palm_width": (L(5, 17), 0.45, 1.5),
        "thumb": (chain((2, 3, 4)), 0.35, 1.7),
        **{f"finger{f}": (chain(DIGITS[f]), 0.45, 2.1) for f in (2, 3, 4, 5)},
        "residual": (float(np.mean([points[i]["residual_m"] for i in points])) / size, 0.0, 0.25),
    }
    bad = {k: round(v, 2) for k, (v, lo, hi) in checks.items() if not lo <= v <= hi}
    return not bad, bad


def main(folders, write):
    det = detector()
    for folder in folders:
        g = json.loads((folder / "guidance.json").read_text())
        front = g["front_view"]
        views = json.loads((folder / "views.json").read_text())
        boxes = hand_box(g, front, {c["id"]: c for c in views["cameras"]}, folder)
        result = {"digits": None, "hands": {}}
        scan, hands = None, {}
        for side, box in boxes.items():
            zoom_name = "zoom_" + front + "_" + "_".join(str(round(p)) for p in box)
            existing = [c["id"] for c in views["cameras"] if c["id"] == zoom_name or c["id"].startswith(zoom_name+"_o")]
            p = None if existing else subprocess.run([sys.executable, str(GUIDE_TOOL), views["source"],
                                "--zoom", str(folder / "guidance.json"), "--name", zoom_name, "--view", front,
                                "--box", *map(str, box), "--yaw", *map(str, ORBIT)],
                               capture_output=True, text=True, cwd=ROOT)
            if p is not None and p.returncode != 0:
                print(folder.name, side, "zoom failed", p.stderr[-300:])
                continue
            ids = existing if p is None else [x["id"] for x in json.loads(p.stdout)["views_added"]]
            views = json.loads((folder / "views.json").read_text())
            cams = {c["id"]: c for c in views["cameras"]}
            per_view, tiles = {}, []
            for vid in ids:
                img = Image.open(folder / "views" / f"{vid}.png").convert("RGB")
                pts = detect(det, img)
                d = ImageDraw.Draw(img)
                if pts is not None:
                    per_view[vid] = pts
                    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                                 (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]:
                        d.line([pts[a], pts[b]], fill=(255, 220, 0), width=2)
                    for x, y in pts:
                        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 40, 40))
                d.text((6, 6), vid + (" ok" if vid in per_view else " none"), fill=(255, 255, 255))
                tiles.append(img.resize((256, 256)))
            sheet = Image.new("RGB", (256 * 6, 256 * ((len(tiles) + 5) // 6)))
            for i, t in enumerate(tiles):
                sheet.paste(t, (256 * (i % 6), 256 * (i // 6)))
            sheet.save(folder / "views" / f"hands3d-{side}.png")
            # Hand size from the detections (wrist to middle knuckle, metres).
            sizes = [np.hypot(pts[9][0] - pts[0][0], pts[9][1] - pts[0][1]) * cams[v]["meters_per_pixel"] for v, pts in per_view.items()]
            size = float(np.median(sizes)) if sizes else 0.0
            points = {}
            for i in range(21):
                obs = [(v, *ray(cams[v], *pts[i])) for v, pts in per_view.items()]
                t = triangulate(obs, size) if size > 0 else None
                if t:
                    points[i] = {"p": t[0].tolist(), "views": t[1], "residual_m": round(t[2], 4)}
            measured = None
            if all(i in points for i in (0, 4, 8, 9, 12, 16, 20)):
                if scan is None:
                    scan = load_scan(views["source"])
                measured = measure_digits(*scan, {i: points[i]["p"] for i in points}, size)
            ok, bad = plausible(points, size) if len(points) == 21 else (False, {"points": len(points)})
            hands[side] = {"cams": cams, "points": points, "size": size, "measured": measured, "plausible": ok}
            result["hands"][side] = {"views": ids, "detected": sorted(per_view), "size_m": round(size, 4),
                                     "digits": measured["count"] if measured else None,
                                     "peaks_per_size": measured["peaks_per_size"] if measured else None,
                                     "mediapipe_plausible": ok, "implausible": bad, "points": points}
            print(f"{folder.name} {side}: detected in {len(per_view)}/{len(ids)} views, "
                  f"{len(points)}/21 points, size {size * 100:.1f} cm, digits {result['hands'][side]['digits']} "
                  f"(peaks / size {result['hands'][side]['peaks_per_size']}), mediapipe {'ok' if ok else bad}")
        # The character's digit count: the most either hand shows (a curled
        # hand's fingertips touching its palm hide clefts: the troll's left
        # read 3, its right 5).
        counts = [h["measured"]["count"] for h in hands.values() if h["measured"]]
        n = max(counts) if counts else None
        result["digits"] = n
        # The route: MakeHuman's hands fitted when both have five digits and
        # MediaPipe's points read as a hand; else the scan's hands kept.
        route = "fit" if n == 5 and hands and all(h["plausible"] for h in hands.values()) else "keep_scan"
        result["route"] = route
        print(f"{folder.name}: {n} digits, hands route {route}")
        for side, h in hands.items():
            cams, points, measured = h["cams"], h["points"], h["measured"]
            chains = order_digits(measured["digits"], {i: points[i]["p"] for i in points}) if measured else []
            # Five digits: MediaPipe's 21 points when they read as a hand,
            # else the chains measured on the surface. Other counts: the
            # measured chains (MediaPipe always sees five).
            if route == "fit" and h["plausible"]:
                source = "mediapipe"
            elif chains:
                source = "surface"
            else:
                source = None
            result["hands"][side]["marks_from"] = source
            print(f"{folder.name} {side}: marks from {source}")
            if not write or source is None:
                continue
            backup = folder / "guidance.before-hands.json"
            if not backup.exists():
                backup.write_text((folder / "guidance.json").read_text())
            ring_ids = [v for v in result["hands"][side]["views"]]

            def put(name, p3, seen, reason):
                dirs = {v: np.array(cams[v]["forward"], float) for v in seen}
                a, b = max(((x, y) for x in seen for y in seen if x < y), key=lambda q: angle(dirs[q[0]], dirs[q[1]]))
                g["landmarks"][name] = [{"view": v, "px": project(cams[v], p3), "confidence": 0.8, "reason": reason} for v in (a, b)]

            for f in range(1, 6):
                for part in ("base", "mid", "tip"):
                    g["landmarks"].pop(f"finger{f}_{side}_{part}", None)
            if source == "mediapipe":
                why = "MediaPipe hand, triangulated from the ring of close-ups"
                for f, idx in DIGITS.items():
                    for part, i in zip(("base", "mid", "tip"), idx):
                        put(f"finger{f}_{side}_{part}", np.array(points[i]["p"]), points[i]["views"], why)
            else:
                # Two views 90 degrees apart from the ring: the front close-up
                # and its quarter turn.
                pair = [ring_ids[0], next((v for v in ring_ids if v.endswith("_o90")), ring_ids[-1])]
                why = "measured on the scan's surface (digit branch of the distance from the wrist)"
                for f, chain in enumerate(chains[:n], 1):
                    for part in ("base", "mid", "tip"):
                        put(f"finger{f}_{side}_{part}", chain[part], pair, why)
            if all(i in points for i in (0, 5, 9, 13, 17)):
                knuck = np.mean([points[i]["p"] for i in (5, 9, 13, 17)], axis=0)
                palm = (np.array(points[0]["p"]) + knuck) / 2
                seen = sorted(set(points[0]["views"]) & set(points[9]["views"]))
                if len(seen) >= 2:
                    put(f"palm_{side}", palm, seen, "MediaPipe hand, triangulated from the ring of close-ups")
        if write and n:
            g.setdefault("subject", {})["hand_digits"] = n
            g["subject"]["hands"] = route
        (folder / "hands3d.json").write_text(json.dumps(result, indent=1))
        if write:
            (folder / "guidance.json").write_text(json.dumps(g, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    write = "--write" in args
    main([Path(a) for a in args if a != "--write"], write)
