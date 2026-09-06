# L16 Resource Audit

Status: complete; deterministic 192-tile alternate-skin candidate critically inspected and explicitly approved under the user's 2026-09-05 autonomous-review authorization.

## Scene and provenance

- Terrain is the unchanged authoritative 16x12 BIQ viewport inherited from L13-L15.
- `fixtures/l16_resources_192.csv` is a separate deterministic source-independent Lab augmentation. It adds 34 visible map-resource witnesses and one hidden witness without changing any BIQ terrain or gameplay field.
- Eight normalized resource bodies cover strategic (horses, iron, uranium), luxury (gold, dyes), bonus (wheat, cattle), and aquatic bonus (fish) reads.
- All bodies and base-color materials come from `ResourceNormalized`. No procedural body, smoke, fire, bloom, or invented light is present.

## Critical visual review

- The first candidate was rejected because crops and dye vegetation dominated their cells, wheat collapsed into rectangular walls, and fish disappeared at reduced scale.
- The corrected candidate reduces crop, plant, and mineral body scale/count while retaining source-authored animal single-subject presentation. Fish receives only a waterline/scale calibration around its unchanged bind-pose body.
- Resources now remain subordinate to mountains and canopy, retain distinct silhouettes beside roads and rails, and remain readable at both Civ III zooms.
- Minerals and crops use compact deterministic clusters; horses and cattle remain single subjects; fish sit visually within the water material without terrestrial cast shadows.
- Every raised land body uses the accepted L13A face, contact, and shared-direction cast lighting. The hidden and no-resource controls are byte-identical to the approved L15 scene.

## Deterministic evidence

Two corrected `python3 Renderer/tools/renderer_dev.py lab` runs produced byte-identical outputs:

- complete: `3c112b32e677df37950498d681637e8e5c1a668710bccac9f2a10b0089533a9a`
- reduced: `0714943d96b15d6ebe71e2fd6b415d89327f58ea9b6e9551ff4be12a2c71f90a`
- no resources: `69c6f30ebe70ec47eafd208f2ab0ff70cd30423589924e3a32cef39f6f72242f`
- resources only: `06b9aedcf212d4d6ed2da3dd602d4b1defd03edf9e061bc3f242d340a4dbe3f6`
- hidden resources: `69c6f30ebe70ec47eafd208f2ab0ff70cd30423589924e3a32cef39f6f72242f`
- Lab scenario: `4bee2c83af59ffdbb076796c7747f4c067b5b3e964aa30d9687b9a5751c527ee`
- resource runtime: `4e66ed69d8483544b1d24ccb4f78a0c106480b7ca4bf762843563671d2d5c3f2`

The no-resources and hidden controls are byte-identical to the approved L15 native render.
