# L19 Farm And Tundra Audit

Status: complete; deterministic 192-tile alternate-skin candidate critically inspected and explicitly approved under the user's 2026-09-06 autonomous-review authorization.

## Scene and provenance

- The accepted L18 scene is preserved byte-for-byte when farms are disabled.
- `fixtures/l19_farms_192.csv` adds thirty-seven deterministic Lab-only farm witnesses over unchanged authoritative BIQ terrain. The physical blocks cover continuous interiors, edges, corners, junctions, isolated cells, every mask from 0 through 15, all four eras, and every irrigable terrain family; one hidden record proves suppression.
- `farm_runtime.bin` recursively compacts normalized source graphs into three era families. Its eight-texture ABI preserves all five crop materials, one sparse boundary material, and the two source-authored emissive channels.
- Crop geometry is the primary field read. Source buildings and windbreak/boundary pieces are sparse deterministic accents, so a connected field remains cultivated land instead of multiplying complete farm compounds at every tile.
- `fixtures/l19_tundra_witness_192.csv` is a separate Lab-only material witness. It changes only authoritative base-terrain fields in fifty-six mixed farm/unirrigated cells; farm rendering does not create or select tundra.

## Critical visual review

- Four intermediate farm candidates were rejected: complete source bases per connected edge produced dark prop sprawl; complete bases per tile remained tree-heavy; crop-first composition initially lost green/gold crop materials during global texture compaction; and frequency-based material selection produced connected brown outlines.
- The accepted compaction reserves all crop materials before sparse accents. Connected fields now read as varied green and gold cultivated blocks at native and reduced Civ III scales without hiding roads, rails, resources, cities, rivers, relief, or shorelines.
- Every farm component uses existing normalized source art. No invented crop resource, smoke, fire, bloom, analytic light, or terrain deformation was added.
- Sparse raised accents use the shared face/contact/cast-lighting contract. Only source-authored emissive channels are eligible at night.
- Civ III base terrain 3 now has an independent alternate-skin tundra base-color, height-normal, and specular path. The accepted witness reads as broad cold gray-white ground across irrigated and unirrigated cells and blends continuously into grassland, plains, and desert without hard tile seams.

## Deterministic evidence

Two final unchanged `python3 Renderer/tools/renderer_dev.py lab` runs produced byte-identical outputs:

- noon complete: `99a82cff7891d946481a4d96ee5eb80e42b71baa2457a118dedf1a25ee276e2c`
- midnight complete: `4534ed052fc2294979981262a17070eb24d9a8f7cff81c4588213703ca519355`
- reduced: `9fdc3426d5fef96e72e6b6869763ba8631e5d7eb3b3f0f4087f83a44e92e761f`
- no farms: `934cb7fe2dc9e620357682c11416f12652266f2f1fa1749c010401d3b569a129`
- terrain+farms isolation: `79a1595ad2299048edfeac0b802b052a88edc45c5600874dc94d6bd9a9ea8968`
- tundra witness: `871e463269cb3607d81eb1b600267c89e6fe86ad06f63bd00e523bf473d79873`
- farm scenario: `8ab4f84e86b6778283c193def06ee681560026b3f56bb41895f0b5fd9085bb9d`
- tundra scenario: `6da892dd9807905c93e38c01cf6e4f40d5c4977e2a06bb8ff04352caf057e8e6`
- farm runtime: `c82b8d17ad7256f11f1807ca2ccf3e7ced402a63a4d99e275e67d22ee6ed769a`

The no-farm control is byte-identical to the approved L18 noon render.
