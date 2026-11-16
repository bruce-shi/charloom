# Annotating a humanoid scan

Copy `guidance.template.json` to `guidance.json`. Keep the scan and image hashes from the template. Use the supplied schema for field names and supported values.

Open the rendered views. Specify the forward-facing body view and use image pixel coordinates measured from the top-left. Left and right refer to the character. Mark visible face, body, hand and foot landmarks. Add garment, hair and skin regions on the surfaces visible in each view. Give occluded landmarks their visibility and an estimate of confidence.

Run `charloom mh-guide-refine guidance.json` for face close-ups. Mark the inner and outer eye corners and both lid margins. Use `mh-guide-zoom` for small features and `mh-guide-check` to inspect the overlays. Supply the resulting file to `mh-transfer --guidance`.
