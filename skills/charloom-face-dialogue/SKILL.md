---
name: charloom-face-dialogue
description: "Bind the original scanned head, eyelids, lips and oral lining. Diagnose protrusions inside the mouth, incomplete lip closure and facial motion during dialogue."
---

# Bind the face and review dialogue

English · [简体中文](SKILL.zh-CN.md)

Bind the eyelids and mouth shape to the visible triangular surface of the original scan head. Retain the neutral surface, UV, textures, and the existing dental arch. First, identify the drawn facial features, open eyelid sheets, individual eyeballs, and the fragmented oral surface, then decide the binding method for each component. Cut the upper and lower lips along the actual lip seams, and the inner wall of the mouth follows the actual lip edges. [Original Head Configuration, Action Embedding, and Verification Commands](references/face-build.md)

**Thin layers appear outside the mouth during speech**

1. Use the reported syllable, time and side camera. Hide the mouth lining, tongue, dental arches and lips one layer at a time. Identify which surface protrudes at that exact frame.
2. If the reconstructed lining protrudes, locate the upper and lower lip edges segment by segment on the final face mesh. Bind the corresponding edges and interpolate across the width of the mouth. Drive the lining from adjacent lip-edge deformation. Measure front-to-back distances in the moving head’s local frame. Inspect the mouth corners, off-centre samples and any interpolation outside a projected triangle.
3. Play dialogue, expressions and head turns on both exported GLBs. Inspect peak opening in profile and tooth-to-lower-lip contact from an oblique front view. Check M/B/P closure, F/V contact and open vowels against the actual speech. Record the lip aperture and the maximum forward offset of the mouth lining relative to the lip edge, with the syllable, frame and side view.
