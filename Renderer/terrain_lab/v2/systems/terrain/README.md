# Q2 continuous terrain candidate

Standalone base-material provider; not a complete map renderer. It reuses
normalized alternate-skin color/height/specular channels, adapts only the base
sampling lattice, and renders through Q0's Mac packet runner. Relief, water,
vegetation, clutter, route/object layers and shared occlusion are external.

From the repository root:

```sh
python3 Renderer/tools/lab_v2.py prompt Q2-terrain
python3 -B -m unittest discover -s Renderer/terrain_lab/v2/tests/terrain -p 'test_*.py'
python3 Renderer/terrain_lab/v2/systems/terrain/fixture_matrix.py
python3 Renderer/terrain_lab/v2/fixtures/terrain/prepare_context.py
python3 Renderer/terrain_lab/v2/app/runner.py quick --fixture Renderer/terrain_lab/v2/fixtures/terrain/context-detail.fixture.json --candidate context-quick
python3 Renderer/terrain_lab/v2/systems/terrain/checkpoint.py controls
python3 Renderer/terrain_lab/v2/systems/terrain/checkpoint.py pairs
python3 Renderer/terrain_lab/v2/systems/terrain/checkpoint.py real
python3 Renderer/terrain_lab/v2/systems/terrain/checkpoint.py regions
python3 Renderer/terrain_lab/v2/systems/terrain/material_audit.py
python3 Renderer/terrain_lab/v2/systems/terrain/evidence.py
```

The 14-family recipe has 105 unordered pairs including homogeneous controls,
1,260 pair orientations/aliases, 60 junction cases, and 20 conservative base/real
stress states. Those twenty states are not all asserted legal in vanilla maps.
The rendered base matrix deduplicates to 15 actual material pairs, both axes and
ownership directions, four hours and both zooms. Shore/relief/wetland composition
must still prove the larger family matrix. Generated source-derived renders
remain ignored under the owned audit `out/` directory.

`surface.h` has a continuous base datum and periodic, deterministic material
field. Both material texture coordinates and domain warp respect map-width
aliases. `builder.cpp` is a narrow `cpp_packet` client, with a generated flat
base grid and fixed Civ III pixel basis. It does not copy/retune the frozen
terrain monolith. Normal detail and roughness are material response, not new
macro geometry. `baseline` in the fixture ID selects the recorded v1-style
raw-noise/edge-band equation inside this same flat diagnostic provider; this is
an ablation proxy, NOT a rerender of or replacement for immutable L21.

The current controls retain one macro source-color sample; detail-on adds
subordinate repeated samples of the SAME authored source height at 3x and 8x,
bounded normals, restrained height-correlated albedo and roughness breakup.
The specular-to-roughness interpretation is C3X-authored and explicit; it is not
claimed as decoded Firaxis shader behavior. No original-art fallback is selected.

See `../../audits/terrain/INTERFACE_REQUESTS.md`, source materials, metrics and
candidate handoff for scope, consumed hashes, exact evidence, and pending gates.

The full-source scene adapter uses Q0 `TerrainHooksV1` through
`composed-complete.module.json`; this adds no duplicate ground plane. Replay one
bounded real-source witness with:

```sh
python3 Renderer/terrain_lab/v2/app/runner.py quick --fixture Renderer/terrain_lab/v2/fixtures/terrain/composed-complete-cold.fixture.json --candidate composed-complete-cold-review
python3 Renderer/terrain_lab/v2/tests/terrain/verify_evidence.py
```

The complete scene's empty augmentation layers add zero objects. These diagnostic
views preserve actual terrain, vegetation and relief. The supplemental scene
material include has two opt-in Q6 calls; it is reviewed separately from the
isolated full material response in r14 on/off controls. Do not infer full linear-scene acceptance
from the frozen complete-scene adapter witness.

Current composed candidate: `composed-hydro-on.module.json` consumes Q2 material
weights/UV, Q3 shoreline and Q6 linear shading. Full acceptance intentionally
fails until the published incident-normal mismatch is resolved:

```sh
python3 Renderer/terrain_lab/v2/app/runner.py check --fixture Renderer/terrain_lab/v2/fixtures/terrain/composed-hydro-wet-on.fixture.json --candidate wet-hydro-review
python3 Renderer/terrain_lab/v2/systems/terrain/composed_surface_audit.py wet
python3 Renderer/terrain_lab/v2/systems/terrain/composed_surface_audit.py dry
python3 Renderer/terrain_lab/v2/tests/terrain/acceptance_gate.py
```
