# L18 Mine Audit

Status: complete; deterministic 192-tile alternate-skin candidate critically inspected and explicitly approved under the user's 2026-09-06 autonomous-review authorization.

## Scene and provenance

- The accepted L17 terrain/object scene remains unchanged when mines are disabled.
- `fixtures/l18_mines_192.csv` adds twenty deterministic Lab-only mine witnesses without changing BIQ terrain or gameplay fields: all four Civ III eras, all three variants per era family, nine mineral-associated cells, and four relief cells.
- `mine_runtime.bin` recursively flattens the normalized source component graph. It retains the six dominant base materials (356 of 366 source draw parts) and both confirmed emissive channels; ten tiny rare-prop material parts are omitted to fit the generic eight-texture proof ABI.
- Resource-conditioned ore children remain excluded so L16 resource bodies are never duplicated. No smoke, fire, bloom, invented excavation, or analytic light is present.

## Critical visual review

- The first candidate was rejected because its strategic-scale read was too timid and its isolation path failed to create the feature shader.
- The corrected scale stays within one Civ III tile while making the excavation footprint, carts, rails, rocks, and era-specific structures readable at both zooms.
- One compound shadow is retained per mine; redundant shadows from recursively flattened child material groups are discarded.
- Source decals are alpha-clipped, remain terrain-following, and do not create rectangular black quads. Relief, shore, road, rail, resource, vegetation, and city adjacency remain legible.
- The terrain+mine isolation view exposes the source-authored warm excavation footprint clearly. Source emissives deactivate at noon and remain restrained at midnight.

## Deterministic evidence

Two final unchanged `python3 Renderer/tools/renderer_dev.py lab` runs produced byte-identical outputs:

- noon complete: `934cb7fe2dc9e620357682c11416f12652266f2f1fa1749c010401d3b569a129`
- midnight complete: `1beca612d2cf2f4f6e6c7923c4456f5013a92a76ffc07124a54fdc7cdc33db0c`
- reduced: `62d1b8f4730d562c7463bb378b369857ae8e310a04a1793fb71102eeaa29d6db`
- no mines: `53d9717775789cee0e3a67fd936447817818c90672c7c8945c6b8a72ae5a2ed6`
- terrain+mines isolation: `8e62bd3eba1d61052524d6c12763ac8092feabb89a7301a09dfce100390e3e2d`
- Lab scenario: `3cf0b60852e1caa4934febbacc649d18c7953e2da07b72f823b4ec1ffafe62a7`
- mine runtime: `d833335ec1a4b1e4c9c4da637f81b46f7828ff038a722a9a5a4673c76ba751f1`

The no-mine control is byte-identical to the approved L17 noon render.
