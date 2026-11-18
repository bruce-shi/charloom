# Guidance for `charloom mh-transfer` (vision agent task)

Current builds use the Python guide CLI. Run it with the source GLB and guide paths from the repository root. The annotation fields below apply to this route. References to `charloom mh-*` commands in older sections describe the retired CLI.

| Operation | Current command |
| --- | --- |
| New calibrated views | `"$PY" scripts/scan-guide.py "$SCAN" --out "$GUIDE"` |
| Face close-ups | `"$PY" scripts/scan-guide.py "$SCAN" --refine "$GUIDE/guidance.json"` |
| Pixel-box close-up | `"$PY" scripts/scan-guide.py "$SCAN" --zoom "$GUIDE/guidance.json" --view VIEW --box X0 Y0 X1 Y1 --name NAME` |
| Lip seam | `"$PY" scripts/scan-guide.py "$SCAN" --trace "$GUIDE/guidance.json" --view VIEW --pixels POINTS.json --output "$GUIDE/mouth-trace.json"` |
| Validate and resolve | `"$PY" scripts/scan-guide.py "$SCAN" --check "$GUIDE/guidance.json"` |

For runtime setup, use the [environment instructions](../skills/charloom/references/setup.md).


You are looking at orthographic renders of ONE generated 3D character. Your
answer guides a numerical pipeline that turns it into a MakeHuman-rigged
character: garments and hair are cut out as separate meshes, the MakeHuman
body is fitted underneath, and the face is aligned to your landmarks.

Copy `guidance.template.json` to `guidance.json` in this folder and fill it.
Answer only with what you can see. Every coordinate is a **pixel in the named
view's image**, `[x, y]`, origin at the top-left corner. The `-grid.png`
images have rulers every 50 px (labels every 100 px); the plain `.png` images
are identical without the grid.

Views (`views.json` holds the exact cameras):

| id | what it shows |
| --- | --- |
| `body_y000`, `body_y090`, `body_y180`, `body_y270` | whole body, camera turned 0/90/180/270° around the vertical axis |
| `head_y000` … `head_y270` | the top 15 % of the model (head) from the same four directions |
| `face_*` | close-ups of the face, added by `charloom mh-guide-refine` (section 3) |
| `zoom_*` | close-ups you ask for with `charloom mh-guide-zoom` (section 7) |

## 1. `front_view`

The body view in which the character's face looks at you, e.g. `"body_y000"`.
"Left"/"right" below are **the character's own** left/right: in the front view
the character's left is on the image's right.

## 2. `subject`

`kind`: `human` or `stylized_humanoid` (big head, cartoon proportions) or
`nonhuman`. `sex`: `female` / `male` / `unknown` (chooses the MakeHuman base).
`notes`: anything unusual (tail, wings, extra limbs, props held in hands).
The optional `subject.head` controls the base transfer. The default `fit`
fits the MakeHuman head and rig. For a rigid head, `keep_scan` attaches the
textured scan to the head bone and needs an exposed neck seam. The run config
can override this field. The final assembly can bind the original scan head on a fitted body.

`hand_digits` (optional, 2–5, thumb counted): how many digits each hand has
when it is not five — `4` for a thumb and three fingers, `3` for a thumb and
two fingers, `2` for a thumb and one finger (MakeHuman's fingers kept are
spread over the palm: index, middle and little; index and little; middle). One number for both hands, or
`{"l": 4, "r": 5}` when they differ. Count in a close-up of each hand (a claw
is not a digit; a dewclaw high on the wrist is not a digit either). MakeHuman's
surplus fingers are left out of the fit and cut from the output; the hand's
digits are matched from the thumb outwards. Omit it for five digits.

`proportions` (optional): how far the body leaves human proportions —
`human` (any real person, thin to very heavy), `stylized` (an anime or game
hero: longer legs, a bigger chest or slimmer waist than people have, still
a human figure) or `extreme` (a cartoon build no person has: a huge belly
on stubby legs, a head a third of the height, arms like barrels). It widens
how far the body may move from MakeHuman's shapes to match the outlines; an
unneeded `extreme` lets measuring noise through, so choose the least that
fits. Omit it for `human` on a human, `stylized` on a stylized_humanoid.

### Choose the animated head route

Use the original textured scan head for every new animated face. Use `fit`
in the base transfer to prepare the body and rig, then assemble the `volume`
head from the scan the transfer consumed. The volume builder preserves the
original head triangles and their texture around the new mouth and eye openings.

Inspect the front, oblique and profile with texture on and off. Identify which
eyes, lids, lashes, teeth and tongue have their own geometry. Record those
observations and the calibrated views in `subject.notes`. Supply reviewed
triangle ranges tied to the current scan for separate facial parts. Mark the
actual eye openings and mouth, including painted ones. Preserve the scan and
view hashes with the lip trace. For a projecting muzzle inspect the lower
jaw beneath the upper lip in profile.

Follow [the face and dialogue skill](../skills/charloom-face-dialogue/SKILL.md) for assembly
and playback. Compare the neutral profile with the original, close both
eyes, change gaze and play dialogue on the bare and dressed exports. Check
the chin and teeth at peak opening and the neck through head turns.
Document any unresolved feature.

## 3. `landmarks`

Name → list of observations
`{"view": id, "px": [x, y], "confidence": c, "visibility": v, "reason": "…"}`.
One observation is enough. For joints and hand points, two from views at
least 15° apart (front + side) are triangulated and are more accurate; two
from (nearly) the same direction give no depth and count as one. Surface
points (face) are each placed on the skin and averaged, so mark them once,
in the sharpest view that shows them (`face_*` or `zoom_*`). Omit what you
cannot see.

`visibility` (optional): `visible`, `partial` (a strand, cuff or strap covers
part of the spot) or `occluded` (hidden under clothing or hair, placed from
proportions). `occluded` halves the confidence, `partial` keeps 80 % of it.
`reason` (optional): a few words on why the point is there or what hides it
("under the sleeve, from proportions"); it is kept for review.

`confidence` (0–1, default 1): how sure you are of the spot. Use 1 for a
visible eye or a bare wrist, about 0.5 for a joint inside a boot or a tight
sleeve, and 0.2–0.4 for joints hidden under loose clothing (hips under a robe
or tunic, knees in baggy trousers). Low-confidence joints only nudge the fit;
the geometry decides.

Face (surface points, use `head_*` views; click the exact spot):
`eye_l`, `eye_r` (centre of the pupil / iris; for painted or closed eyes the
centre of the eye opening), `nose_tip`, `mouth_l`, `mouth_r` (mouth corners).

Surface points are placed **on the skin**, not on the first surface under the
pixel: the pipeline walks past surface inside your hair/garment polygons and
past layers much nearer the camera than the skin around the point (a loose
strand), and uses the skin behind them, or the nearest skin within 3 mm. If
there is none, the point is **rejected** (an error in the check). Still mark
the visible skin: a strand in front of an eye corner hides where the corner
really is.

Eye openings (second pass): after saving `guidance.json`, run
`charloom mh-guide-refine guidance.json`. It adds close-up `face_y000`,
`face_l35`, `face_r35` views centred on the face. In `face_y000` mark, per
eye, `eye_l_inner` / `eye_l_outer` (the corners where the lids meet),
`eye_l_top` / `eye_l_bottom` (the upper and lower lid edge above and below the
pupil), and the same for `eye_r_*`. You may also re-mark `eye_l`, `eye_r`,
`nose_tip`, `mouth_l`, `mouth_r` there for precision; replace the head-view
observations rather than adding conflicting ones. The eye opening decides
the MakeHuman eyeball size and where the scan's eyeliner and lids are kept.

Lid margins (optional; mark them whenever the eyes are open and the lids are
visible): six points per eye on the **lid margin**, the edge where the lid
meets the eyeball (the inner edge of the lash line or eyeliner, where the
white or iris begins), not the outer edge of the eyeliner, a wing or the tips
of the lashes:
- `eye_l_upper_1`, `eye_l_upper_2`, `eye_l_upper_3` on the upper lid at about
  25 %, 50 % and 75 % of the way from the inner corner (by the nose) to the
  outer corner;
- `eye_l_lower_1`, `eye_l_lower_2`, `eye_l_lower_3` the same on the lower lid;
- the same for `eye_r_*` (`l`/`r` are the character's, as for the corners).
They pin MakeHuman's lid curve between the four opening points. Mark them in
`face_y000` or, sharper, in a close-up of each eye
(`charloom mh-guide-zoom guidance.json --landmark eye_l`, then mark in
`zoom_eye_l`; the `zoom_eye_l_o30` view is turned towards the outer corner
and shows its lid margins better). Like the other face points they are
placed on the skin under the pixel; one view is enough, a second one is
averaged. Mark all six of an eye or none; leave them out for closed or
hidden eyes. The check warns when a lid's points are not in order from the
inner to the outer corner, when an upper point is not above its lower one,
when a point lies outside the opening's box, and when left and right are
swapped; its overlays draw each resolved lid (inner corner, 1, 2, 3, outer
corner) in every view where it is visible (upper cyan, lower green): it must
follow the lid margin in the oblique views too.

Ear outline (optional; mark it whenever an ear is at least partly
visible): points on the **visible** ear in the side head view that shows it
(`head_y090` / `head_y270`, or the side views of your `front_view`), better
in a close-up of each ear (`charloom mh-guide-zoom guidance.json --box
head_y090 X0 Y0 X1 Y1` around the ear, or `--landmark ear_l` once one ear_l
point is marked). Per ear (`l`/`r` the character's):
- `ear_l_top`: the highest point of the helix (the rim);
- `ear_l_back`: the rearmost point of the helix;
- `ear_l_lobe`: the bottom of the lobe;
- `ear_l_front_top`: where the top of the ear joins the head, above the
  tragus (the root of the helix);
- `ear_l_front_bottom`: where the lobe joins the head (cheek or jaw);
- `ear_l_tragus`: the tip of the tragus, the small flap in front of the ear
  canal.
Mark **only what you can see**: leave out every point that hair, a strand,
a hat or a garment covers (MakeHuman's own ear shape fills the hidden part),
and never mark a strand's edge as the rim. Two points are the minimum to
place the ear; three or more shape it. MakeHuman's ear is placed, scaled
and shaped (its own ear targets) to these points, and the scan's ear colour
is baked onto it only inside the outline they draw, so the generated ear is
not painted a second time on the head beside it. The check warns when the
points are out of order (top not the highest, lobe not the lowest, back not
the rearmost, front points not in front of the back), on the wrong side of
the head, implausibly sized or placed, or when the two ears differ a lot;
its overlays draw each ear's resolved outline (orange, closed from
front_top over the top, back and lobe to front_bottom and the tragus).

Body joints (centre of the joint as seen in that view; with one view only
the two image coordinates are used, and the depth comes from a cross-section
probe along the viewing ray — the middle of the limb, or of the body layer
under a garment you marked — and from the fit): `neck` (base of the neck between
the collarbones), `shoulder_l/_r` (shoulder joint, not the shoulder top),
`elbow_l/_r`, `wrist_l/_r`, `hip_l/_r` (hip joint, roughly the leg crease),
`knee_l/_r`, `ankle_l/_r`. Under clothing, estimate where the joint is from
body proportions, not from the clothing's own features:
- **Hips.** The hip joints are at the leg crease, about a hand's width below
  the waist. A belt, sash or waistband sits at the waist, above the hips.
  For an adult the hips are about half the standing height above the floor.
- **Knees.** Each knee is about halfway between the hip joint and the ankle.
  Baggy trousers or a robe hem do not move it.
- **Elbows.** Each elbow is about halfway along the arm between shoulder and
  wrist. A sleeve cuff or bracer edge does not mark it.

Mark each joint in a side view (`body_y090` / `body_y270`) as well where it is
clearly visible there (a bare elbow, knee or ankle); the check warns about
joints seen in one view only. Leave a joint the side view does not show (a
neck under a collar, hips under a robe) single-view: a guessed side mark
pulls the joint's depth, a single view leaves it to the geometry.

Hands are measured, not marked, once the bodysuit edit and MediaPipe body
joints are written: `out/tools/cv/.venv/bin/python -I
scripts/mh-hand-landmarks.py --write GUIDE_DIR` renders a ring of close-ups
around each hand, triangulates MediaPipe's 21 hand points, counts the
scan's digits on its surface, and writes `subject.hand_digits`,
`subject.hands` (`fit` for five digits whose points read as a hand, else
`keep_scan`: the scan's own hands kept and skinned to MakeHuman's bones) and
the finger and palm marks below. Mark hands by hand only where it reports
no hand.

Hands (optional but valuable: they map MakeHuman's finger bones onto the
scan's fingers). Mark them in a close-up of each hand (section 7, `--box`
around the hand in the body view where you see the back or the palm of the
hand flat, not edge-on). Names: `finger{F}_{S}_{P}` with
- `F` = 1 thumb, 2 index, 3 middle, 4 ring, 5 little finger;
- `S` = `l` / `r`, the character's side;
- `P` = `base`: the knuckle where the finger leaves the palm (for the thumb,
  the knuckle at the web between thumb and index finger); `mid`: the middle
  knuckle (fingers: the second knuckle; thumb: its only visible knuckle);
  `tip`: the end of the finger, at the nail;
- plus `palm_l`, `palm_r`: the centre of the palm (or of the back of the hand).

Mark the centre of the finger across its width, as for joints: the depth is
the middle of the finger along the viewing ray. A finger hidden behind
another can be left out; for fused or mitten-like cartoon fingers mark the
ones you can tell apart. With `subject.hand_digits` below 5, number the
hand's digits from the thumb (`1` thumb, then `2`, `3`, … across the hand)
and leave the higher numbers out: on a thumb + three fingers hand the
outermost finger is `finger4`, and there is no `finger5`. Mark the same names in a second close-up at least
15° away (the `_o30` one) when the fingers are clear there too.

**Body outlines** (recommended whenever clothes hide the body's shape): at
each station mark the **body's** two edges as you judge them, not the
cloth's. Under tight clothes that is the cloth's edge; under loose clothes,
a robe, armour or a shirt hanging off the back, it is where the skin would
be (the shape the cloth drapes over, the curve of the belly, chest and
buttocks). The body's size and its place front to back are fitted to these
marks; without them they are guessed from the cloth's surface.
- Stations: `chest` (at the bust or nipples), `waist` (narrowest point of
  the torso), `hips` (widest point of the buttocks), and per side
  `upperarm_l/_r` (mid upper arm), `forearm_l/_r` (mid forearm), `thigh_l/_r`
  (mid thigh), `knee_l/_r`, `shin_l/_r` (mid calf), and `skull` (the
  skull's width and depth above the ears, under the hair: only where the
  scalp's line can be judged; `scripts/mh-bodysuit.py` measures it on a
  shaved-head edit).
- In the **front** body view, `outline_{station}_w1` and `_w2`: the left and
  right edges of that body part (either order), straight across it (for a
  limb, across the limb, not horizontally).
- In a **side** body view, `outline_{station}_d1` and `_d2`: its front and
  back edges. For limbs use the side view that sees that side's limb
  (`body_y090` the character's left when `front_view` is `body_y000`).
- Both edges of a pair in the same view; a lone edge is ignored. Mark only
  stations you can judge (a limb hidden by the other is left out); the
  confidence says how sure you are of a hidden edge (0.4–0.6 under loose
  cloth, 0.8–1 for a visible or tight one).

**Bodysuit silhouettes** (measured, not marked): `scripts/mh-bodysuit.py
--edit --write` has an image model repaint the front and side views in a
skin-tight green suit with a shaved head, and `scripts/mh-suit-contours.py`
turns that edit's outline into `views/suit_contours.json`. Then
`scripts/mh-suit-ensemble.py --edit` asks for three more edits and writes
`views/suit_contours.ensemble.json` (the fit's default, Config
`suit_contours_file`): one consensus outline per view, each point weighted
by how far the edits agree there. When the edits disagree everywhere (more
than 3 px), it falls back to the single edit.

## 4. `garments`

One entry per separately worn item you want as its own mesh, including hair
and hair accessories:

```json
{"name": "robe", "label": "garment", "layer": 1, "tight": false,
 "regions": [{"view": "body_y000", "polygon": [[x,y], [x,y], ...]},
             {"view": "body_y180", "polygon": [...]}]}
```

- `label`: `garment` or `hair` (wigs, braids, buns, hair ties, hairpins: `hair`).
- **Hair goes in the head views too.** The head views are sharper and decide
  the hairline. Draw the hair in every `head_*` view where it is visible,
  including the fringe and any loose strands that hang over the forehead,
  temples, cheeks or neck.
- **Skin-coloured strands.** Generated scans often paint loose strands with
  the face's colour (beige strands in front of the cheeks, by the ears, over
  the neck), sometimes only their tips. Look for them in every `head_*` and
  `face_*` view: hair geometry (thin, long, hanging from the hair) whatever
  its colour. Mark each with a hair stroke along it (its own width), on the
  hair item. The pipeline also finds thin strands that float in front of the
  skin or hang from the hair by themselves, and `mh-guide-check` tints them
  (magenta: will become hair; cyan: a thin floating layer kept as skin). Check
  both: strands lying flat on the skin or fused into it are not found and
  need your stroke; an ear rim tinted magenta needs a `skin_strokes` stroke
  along it (a skin stroke is never relabelled). Beards, brows and lashes
  *painted* on the skin are skin, not strands.
- `layer`: 0 = worn on the skin (shirt, trousers, shoes), 1 = over that
  (robe, vest, armour), 2 = over that (cloak), and so on; use as many as the
  outfit has.
- `tight`: true for close-fitting items (T-shirt, leggings, gloves).
- `gap_mm` (optional): how far the skin sits inside the item's outer
  surface (material plus air), in millimetres, for a body the clothes hide:
  a few mm for a tight shirt, 15–30 for a loose tunic, 40–70 for a robe or
  armour over other layers. The hidden torso, arms and legs are fitted to
  the item's surface minus this gap. Leave it out when a cuff or hem shows
  skin just past the item: the gap is then measured there; without either a
  cautious default by `tight` and `layer` applies.
- `regions`: polygons drawn **inside** the item in any views where it is
  visible, up to about its edge. Each polygon's seed then **grows** over the
  connected surface until the colour changes (by up to ~24 px of that view),
  so a polygon a few pixels short of the edge still covers the item, and the
  surface between two polygons goes to the one whose colour it matches. A
  polygon forces its label where it is deep inside (≥ 8 px), where the colour
  matches the item, and on everything it grew onto; off-colour pixels near its
  edge (a strand at the edge of a face polygon) only choose between hair and
  garment. Where polygons overlap, the smaller one wins (in real size, across
  views; skin polygons and strokes included), so draw small items (a hairpin,
  a belt) over larger ones freely, and a face skin polygon inside a hair
  outline stays skin. Outline every visible side (front and back at least);
  unmarked garment parts are filled in from the nearest marked item; unmarked
  bare skin may end up as an extra piece, so mark all bare skin.
- **Necklines.** A top's polygons stop at its collar: the bare neck and the
  skin inside a V-neck or between crossed lapels are skin (mark them in the
  skin polygons), never part of the shirt's polygon. Polygon pixels that
  clearly show skin are not forced to the garment, but a wrong polygon still
  leaves the border unreliable. Around the neck the skin/garment border is
  snapped to the collar's crease or colour edge within about 2 cm, so a
  polygon drawn up to the collar is enough. Where a collar is nearly the skin
  colour (cream or white on pale skin) and the segmentation still puts the
  border on the wrong side of it, add a stroke along the collar band (its own
  width) to the garment: strokes keep their label exactly.
- `rigid` (optional, default false): a hard part attached to the body that
  must not bend — claws, horns, tusks, a beak tip, antlers, spikes, a horn
  ring. Each connected piece of it follows the single bone that dominates
  the body where it attaches (a finger claw the finger's last bone, a toe
  claw the toe, a horn the head) instead of blended cloth weights, and keeps
  its exact shape. Give claws, horns and tusks their own item (`"label":
  "garment"`, e.g. `claws`, `horns`) rather than leaving them unmarked:
  unmarked dark claws end up in the nearest garment and float off the
  fingers. Small claws can be marked with strokes, one short band along each.
- Mark each physical region that needs its own attachment as a separate item.
  On a helmet, outline the shell and draw separate strokes along loose ties.
  On a dress, mark the fitted bodice and the free skirt separately. Inspect
  the front, side and back views to locate the tie or waistband.
- `rig` selects the binding for that item's polygons and strokes. Exact item
  names `helmet`, `skirt` and `ribbon` supply defaults for older guides.
  An explicit mode takes precedence. Use `{"mode":"body"}` for close-fitting
  fabric, including a sewn-on ribbon or printed motif.
  - A hard shell can specify `{"mode":"rigid","bone":"head"}`. All its trim
    follows the same mount across UV and material boundaries. Omit `bone`
    for independently attached plates or claws to choose a bone per connected piece.
  - For a skirt panel assembly, use `{"mode":"skirt","bone":"root"}`.
    The waist stays attached. Set `cut` to `flared` for a four-chain A-line,
    `bell` for six chains around a full petticoat or ball gown, or `pleated`
    for eight around a pleated hem. Each chain has two weighted joints.
    Give a separate underskirt its own item and layer.
  - For a loose tail, use `{"mode":"ribbon","bone":"head","direction":[0,-1,0]}`.
    Set the bone to the actual attachment (e.g. `root` for a waist bow).
    Direction points from the knot to the tip in the aligned Y-up frame.
    Mark tails hanging in different directions separately. The end nearest
    the knot stays on the mount and the free end follows a two-joint chain.
  - For a cape hanging from the upper back use `{"mode":"cape"}`. For a
    veil attached to the head use `{"mode":"veil"}`. These spread three
    chains across the open width, each with its own pivot and free end.
    For an apron, open coat tail, sleeve flounce or asymmetric panel use
    `{"mode":"panel","bone":"root","direction":[0,-1,0],"width_axis":[1,0,0]}`.
    Set the attachment bone, fixed-edge direction (`width_axis`) and
    fixed-to-free direction for that individual piece. Annotate two sides of
    a split coat as separate items. Treat a narrow cord or tassel as a ribbon.
- Optional `fabric` on the **garment item** selects the baked response of the
  hanging section. Choose `woven` (default), `silk`, `chiffon`, `tulle`,
  `organza`, `lace`, `satin`, `knit`, `denim`, `wool`, `velvet` or
  `leather`. Presets control lag time, swing gain and maximum angle.
  Use `rigid` for metal plates, stiff plastic or a buckle. The same leather
  can form a rigid bracer, a fitted bodice, or a free coat panel.
  The response represents an art-directed approximation. Check the walking
  and turning motion for layer crossings, especially at the hem and floor.
- An agent can publish a reusable rig and fabric combination with
  `charloom mh-garment-preset --library LIB register manifest.json`.
  Use `schema` for the manifest format, `list` to discover definitions, and
  `get ID` to inspect one. Pass `-` instead of a filename to register from stdin.
  A manifest can define numerical motion directly:
  ```json
  {"id":"studio.organza-v1","rig":{"mode":"skirt","cut":{"panels":10}},"fabric":{"delay_s":0.07,"gain":0.48,"max_angle_rad":0.18}}
  ```
  In an item's annotation set `"preset":"studio.organza-v1"` alongside its
  name, label and marked regions. The default lookup directory is
  `garment-presets/` beside `guidance.json`. For a shared directory use
  `mh-guide SCAN --out GUIDE --preset-library LIB` or set the guidance's
  top-level `garment_preset_library` to a path relative to the guidance
  file or an absolute path. To populate the default, pass
  `--library GUIDE/garment-presets` to the registration command.
  Explicit `rig` and `fabric` fields each override that entire field from
  the preset. The transfer records the ID and the resolved settings in
  the dressed mesh's extras. Registering an identical definition is
  repeatable; publish a changed definition under a new ID to keep old
  guidance reproducible.
- Example wedding dress entries (add the actual image regions to each):
  ```json
  [
    {"name":"bodice","label":"garment","layer":0,"rig":{"mode":"body"}},
    {"name":"petticoat","label":"garment","layer":1,"fabric":"organza","rig":{"mode":"skirt","cut":"bell"}},
    {"name":"overdress","label":"garment","layer":2,"fabric":"tulle","rig":{"mode":"skirt","cut":"bell"}},
    {"name":"bridal_veil","label":"garment","layer":1,"fabric":"chiffon","rig":{"mode":"veil"}}
  ]
  ```
  Use a skin joint name from the selected MakeHuman prior. Inspect
  `stage4_garments.rigid_pieces`, `secondary_chains` and the dressed walk.
- `strokes` (optional): bands along items too thin for a polygon — a strap, a
  cord, a moustache, a single strand:
  `{"view": id, "points": [[x,y], [x,y], ...], "width_px": w}`. `points` is
  the centre line in order (one point = a dot), `width_px` the item's visible
  width. Everything under the band is seeded and forced, so keep the width to
  the item's own width.

## 5. `skin`

Polygons on clearly bare skin (face, neck, arms, hands, legs). These seed the
skin labels and define the skin colour (draw them up to the skin's edge; they
grow to it like garment polygons). `skin_strokes` (optional, same form as
garment strokes) mark thin bare strips, e.g. skin between a glove and a
sleeve. Anything inside a skin polygon is
treated as skin and its paint is copied onto the body. On the face, keep the
polygon a few pixels below the hairline and away from loose strands: cover
the brows, eyes, nose, cheeks, mouth and chin, but not the fringe or hair
lying on the temples and cheeks. A skin polygon does not keep a skin-coloured
strand in front of the face as skin (the strand check overrides it); only a
skin stroke does.

## 6. `skin_cleanup`

Polygons on bare skin whose painted texture should NOT be copied onto the new
body: hair strands painted on the cheek, a garment's shadow on the neck,
straps painted onto the arm. Leave empty if none. Thin painted marks can be
`skin_cleanup_strokes` (same form as garment strokes).

Painted strands: scans often paint hair onto the face skin itself — thin dark
lines on the forehead, cheeks or neck that continue a strand or start at the
hairline, with or without strand geometry over them. The bake removes thin,
dark, hanging lines that touch hair by itself; mark the ones it misses (they
stay visible in `3-skin` renders: short or nearly horizontal lines, lines that
do not reach the hair, pale or reddish paint) with a `skin_cleanup_strokes`
stroke along each, `width_px` the painted line's own width plus a pixel or
two, in the `face_*`/`head_*` view that sees it most frontally. The skin
around a cleanup stroke is restored from the skin beside it (a ~4 mm
feather), so a stroke may cross a brow without erasing it, but never mark
brows, eyelashes, eyeliner, the lip line, freckles or a painted beard or
stubble: they are the character's own and are copied.

Brows (`brow_strokes`, optional; mark them whenever the brows are painted
and visible): one stroke per brow in `face_y000` (or the frontal head view),
same form as garment strokes: `points` along the **middle** of the brow from
its head (by the nose) to its tail, 4–6 points following its curve,
`width_px` the brow's thickest part plus 2–4 px, so the stroke covers the
whole brow and a little skin round it. Stop the stroke where a strand
crosses the brow or hair covers its tail (put a second stroke past the
strand if the brow shows again): what lies under a brow stroke is copied as
painted, strands included. Inside the stroke the skin bake copies the scan
exactly, even where the segmentation called the brow hair: no feather from
the fringe or a strand next to it, no painted-strand or outlier removal, no
mirroring of the other brow; and the fill around it never takes the brow's
dark colour (unmarked, the brows came out thin, broken and smeared into
the forehead next to a fringe). Check the violet bands in
`views/check/face_y000.png`: each must cover its brow from head to tail and
nothing else dark (no strand, no eyeliner, no hairline).

## 7. Close-ups (`charloom mh-guide-zoom`)

When a spot is small or hidden in the views — an eye corner behind a strand,
a wrist under a cuff, the fingers, a strap — ask for a close-up:

```sh
charloom mh-guide-zoom guidance.json --landmark eye_r_inner
charloom mh-guide-zoom guidance.json --box body_y000 100 500 220 640
```

`--landmark` centres on a point you already marked; `--box VIEW X0 Y0 X1 Y1`
on a pixel box of a view. Each adds two calibrated views to `views/` and
`views.json`: `zoom_<name>` from the same direction as the view it came from
and `zoom_<name>_o30`, turned 30° towards that side of the model. Mark
points, polygons or strokes in them like in any other view, replacing the
coarser observation of the same landmark.

## 8. Self-check

Use the Python command from the repository root with the current source and guide.

```sh
"$PY" scripts/scan-guide.py "$SCAN" --check "$GUIDE/guidance.json"
```

This validates source and image hashes, resolves marked pixels onto scan triangles, checks required transfer landmarks and prints resolved JSON. Inspect the returned coordinates and the marked images. Create overlays from the current guidance on its calibrated views when visual verification is needed. The current command emits resolved JSON to stdout. The older `charloom mh-guide-check` interface and its `views/check` image outputs belong to the retired transfer implementation.

Re-mark any image whose pixels changed after calibration. Verify anatomical left and right, each garment's coverage, joint depths, eye openings and lip endpoints before transfer.
