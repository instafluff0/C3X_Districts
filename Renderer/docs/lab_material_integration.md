# Renderer64 Lab material integration

The user requested Mountains and Terrain first, followed by other Lab imports
before a combined in-game regression pass. Full gameplay testing is deferred
for this batch; each import still needs focused build and rendering checks.

## Mountains and Terrain

| Change | Live integration |
| --- | --- |
| Lower mountain shape | 68% height and 108% spans; preserves five authored variants and connected ridge union. Shared by Lab and runtime mesh generation. |
| Ground at mountain fringes | Existing mesh handoff paired with the current terrain material and ground lighting classification. |
| Grass texture repetition | Preserves the fine sample; averages only its coarse color field. Both ordinary ground and mountain ground use it. |
| Ground detail | Current Terrain Lab broad/detail response, with gentler grass normals than plains. |
| Hard grass triangles | Already omitted by the shared surface mesh; plains/desert surface patches remain. |

The mountain canopy study remains an isolated proposal, separate from the
combined lower-mountain/terrain candidate. Other Lab proposals were not part of that first import. The Coast pass below
follows it. Native HUD, camera logic and unit rendering are
unchanged by this batch.

## Reproduce a material import

Renderer64 uses `Renderer/packs/Renderer64CutoverControl` at runtime, rather
than the mutable generated Lab shader directory. Updating only the shared Lab
HLSL therefore does not update the game. Prepare selected materials explicitly:

```sh
python3 Renderer/tools/prepare_renderer64_materials.py \
  --out Renderer/native/build/mountains-terrain/candidate-shaders terrain mountain
```

Use a fresh output directory. The tool runs current adapters in a disposable
mirror and overlays only the selected city-profile materials on the runtime
bundle. Other shaders remain byte-identical. `material-preparation.json` records
source, base and output hashes. Compiled caches are derived from shader bytes
and rebuilt by D3D. No art pack or reference image is modified.

The normal category dispatcher currently detects edited generated City shader
files and preserves them. Do not clear that protection or overwrite another
Lab's generated work to import these materials. The private mirror gives this
candidate current bindings without changing those files.

The GPU fixture accepts `--shader-root` with this repository-local directory;
the source tree and selected runtime bundle are included in its input receipt.
After verification, stage the matching bridge, x64 DLL and helper together,
plus the selected HLSL files, while Civ III is closed. Verify all hashes. A
Renderer-only change does not need injected-code compilation or installation.

Task-local builds, captures and receipts are under
`Renderer/native/build/mountains-terrain/`. Fixed references remain unchanged.

## Verification and staging

The matched bridge, x64 DLL, helper and selected shader files are staged.
Their hashes match the checked candidate; the startup probe reports healthy.
The build's animation/skin checks, 22 natural/material tests and five GPU
projection/detail/shadow checks pass. Provider equations were checked against
the private current adapter output because of the generated-file guard above.

Four 1600×900 captures compare the previous staged build and this candidate at
the same ridge and grassland cameras in `test.biq`. They use the actual x64
Renderer64 DLL through the bounded sandbox client, with zero fallback. The
comparison sheets are `mountains-comparison.png` and `terrain-comparison.png`.
These are renderer captures, not Civ III gameplay or FPS measurements. The
combined in-game regression and performance check remains pending by request.

## Coast and water

The next import uses the Coast Lab's selected `desert-broad-mosaic` appearance:
clear aquamarine coast water, fewer repeating brown bed stamps, and broad
world-stable regions of irregular sand/hill source-height relief. The three
bed functions match the selected Lab snapshot exactly. This is material relief;
the rejected displaced-bed geometry is excluded. Coast, sea and ocean retain
their existing family coordinates and distinct color response. The later
water-boundary randomization experiment was explicitly rolled back by the user
and is excluded.

Two separate lighting changes accompany this import:

- Renderer64's water eye no longer selects a nearest wrapped world copy per
  pixel with `round()`. It recovers the continuous water-plane offset from the
  interpolated world-coordinate derivatives and the actual viewport center.
  The orthographic projected basis accounts for scrolling, wrapped draw copies
  and zoom without a second camera approximation. Repeating wave textures
  retain wrapped coordinates. Sun, twilight and moon glint remain.
- The final scene's warm sun-ray brightening and its unused CPU preparation are
  removed. Ordinary material lighting, shadows, tone mapping and water glint
  remain. The water-ray correction applies to the active Renderer64 water
  surface; the older standalone hydrology optical path is separate.

The water constants now include the guarded viewport center. Stage the matched
DLL and water shader together. There is no new injected hook, asset format,
readback or game-thread wait.

Prepare the selected material pair with:

```sh
python3 Renderer/tools/prepare_renderer64_materials.py \
  --out Renderer/native/build/coast-water/candidate-shaders coast
```

Only city-profile hydrology and the Renderer64 water-surface HLSL change in the
86-shader runtime bundle. The other 84 shaders retain the preceding staged
Mountains/Terrain baseline. Category metadata includes the water surface and
its GPU continuity regression.

Task-local evidence lives under `Renderer/native/build/coast-water/`. The
bounded capture script uses the production x64 DLL through the sandbox client;
it does not operate Civ III or replace reference images. Full gameplay and FPS
regression remains deferred until the requested Lab imports are combined.

### Shoreline blending and wetland relief (October 5)

The material follow-up from commit `25221b37` is now active in Renderer64.
`q3_beach_coverage` keeps sand beneath the grass/plains partial-coverage band,
and `coast_edge_coverage` uses the stronger texture breakup.

- **How it was applied.** The commit's own diff for its nine runtime shader
  copies was applied to the active `Renderer/packs/Renderer64ResidentRuntime`
  pack, with no regeneration from the Lab.
  - The coast, city-light and resident-submission layers are byte-preserved.
  - `shoreline-overlay.json` in the pack records the patch hash, the changed
    files and their before/after hashes.
  - The prior pack is kept as `Renderer64ResidentRuntime-before-shoreline-20261005`.
- **What else changed.** The gentler coastal shoulder (`coast_join.h`,
  `surface_mesh_body.h`) and the wetland collar (`queries.h`, commit
  `d43d4517`) are compiled into the x64 DLL, so they arrive with the rebuilt
  trio.
- **Shader verification.** All 19 runtime pixel entries compile against the
  patched pack (`C3X_RENDERER_GPU_TESTS=1 python3 -m unittest
  Renderer.native.test_runtime_shader_programs`).
- **Staging.** The bridge, DLL and helper were staged with hashes matching the
  build output.
- **Game check.** Bounded `near` runs on the busy and light saves show no
  failures or fallback. Coasts show the broken-up sand-to-grass edge.

### Coast verification and staging

The matching bridge, x64 DLL, helper and two selected HLSL files are staged;
all hashes match their checked candidate. Civ III was closed and the startup
probe reports healthy. Nine focused tests pass: water motion, projected water
view, scene projection, shore grain/coverage, material adaptation and the shared
light/river contract. The category checker validates all 28 definitions.

Seven production-DLL captures have zero fallback:

- A uniform 16×64 wrapped ocean with the old and new water view equations,
  using the same candidate DLL and materials, reproduces the two vertical
  jumps at columns 288 and 1312. Their mean column-color jumps fall from
  23.86 to 0.04 and 0.39 on the 0–255 scale. The entire central 1024-column
  region, including its glint, is pixel-identical. See `seam-comparison.png`
  and `seam-measurements.json`.
- The matched 2240×1260 BIQ coast comparison at tile width 128 shows the
  selected shallow material and existing coast/sea/ocean bands. The far-ocean
  control's left 300 columns are pixel-identical. A width-192 close capture
  checks the material at the current closest game scale. See
  `coast-comparison.png` and `captures/after/close/frame.png`.
- Matched hour-18 captures use the same control shaders and compare the old
  and new DLLs. They isolate removal of the final warm ray overlay while the
  water glint remains visible. See `sunshine-comparison.png`.

`build-evidence.json`, `material-preparation.json` in `candidate-shaders/`,
capture receipts and `stage-evidence.json` bind these results to source and
binary hashes. Capture BMPs are disposable; lossless PNGs retain the inspected
frames. A failed Parallels dispatch was retried only after Windows confirmed
that no fixture process remained. These checks do not establish live-game FPS.
