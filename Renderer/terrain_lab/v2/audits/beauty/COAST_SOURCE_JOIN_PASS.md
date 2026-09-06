# Selected cliff source, coastal shoulder and water normals

2026-09-06. `coast-pass-rocks-r8` is the latest retained work in progress.
This is an incremental combined visual improvement, not visual acceptance.
The previous `rocks-r4` checkpoint, intermediate renders, and synthetic
`volcano-witness-r2` remain preserved. No milestone or Integration gate advances.

## What changed in the pixels

The matched long-coast crop shows gray connected rock faces where `rocks-r4`
had separate brown stones. Grass rises behind the rocks instead of lying below
their tops. The clearest locations in the 1616x888 image are x628–967/y499–599,
x938–1068/y649–728, and the upper headland x618–722/y123–181. These move toward
the grassy shoulders and gray rock groups in canonical `sea_and_shore.png`.
They remain less naturally integrated than the reference, particularly around
the small pond. No new shoreline profile was used in this continuation: the
retained articulated Civ III-compatible contour stays fixed.

Source water normals replace the perfectly flat surface response with fine
directional detail. It is subtle in daylight and more visible in moonlight.
The new coast's broad open water exposes this change at actual zoom 2; the
shore remains readable. Broad blurry shallow bands remain unresolved. The
static normal field does not claim source-engine animation or recovered LEAN
rendering equations.

All three original 100-tile regions, the 128-tile long coast, and a newly selected
100-tile coast were rendered at noon/midnight and both fixed gameplay zooms:
20 combined frames. Matched source terrain, empty object scenarios, cameras,
output sizes, and sampling settings were verified. Full zoom-2 day/night views
and matched native crops were directly inspected. The new region at raw
`[20,30]`, extent 10x10, halo 6 was chosen by source coverage before rendering
this candidate there. It contains 17 land tiles, 83 water tiles and 10 native
land/water edges. It received no region-specific tuning and is now a regression
witness. Its own preserved `rocks-r4` baseline uses the identical camera and
terrain. Fixed benchmark definitions remain unchanged.

## Why the previous attempts missed the reference

1. **r5 — source material correction.** The old normalized cliff pack contained
   Base-game materials, although the selected skin overrides all four channels.
   Extraction from the selected skin found identical vertices and indices for
   all six supported cliff bodies, but different base, LEAN0, LEAN1 and gloss
   payloads. Rebinding only these source channels changes 7,541 noon pixels at
   zoom 1 (2,155 at zoom 2), with identical instance transforms. The selected
   gray material is visibly closer to the reference. This correction is retained.
2. **r6 — rejected buried join.** Coupling rock tops to the existing low hill
   edge buried most of the source bodies. The result left scattered pebbles.
   It was rejected, not compensated with more random scale changes.
3. **r7 — coastal shoulder.** Source ArtDef investigation revealed a separate
   cliff height control and cliff clutter independent of terrain height. The
   C3X join now supplies a small grassy shoulder behind the source meshes,
   retaining a flat water collar. Source rock bodies use uniform transforms,
   preserved UVs, tighter overlap and placement derived from that shoulder.
   This changes 53,295 noon pixels relative to r5 at zoom 1. The result has
   connected banks; some transitions remain abrupt and need further inspection.
4. **r8 — combined source water normals.** Four already-bound source water
   normal/moment textures were unused by the combined Q3 surface branch.
   Their static large/small response now participates in shared sun/moon
   illumination and roughness attenuation. Full composition follows a separate
   shader-only diagnostic. Comparing r7/r8 preserves all source instances and
   changes 135,181 noon pixels at zoom 1. This includes small lighting changes
   over the water surface, not a quality score.

## Source evidence and adaptation boundary

Exact selected source extraction is reproduced by
`systems/relief/selected_coast_source.py`. The environment override
`C3X_CIV6_ENVIRONMENT_SKIN` selects its installed root; output manifests use
portable paths. `fixtures/beauty/selected-coast-source-v1/provenance.json`
records source entries, payload hashes, mesh parity and output hashes. Licensed
meshes/textures and compiled bundles stay ignored local artifacts.

- Selected `environment/clutter.blp` SHA-256:
  `44e1b0993cf563a8912b913c0a8e73844afae8904876422096dd34f04f4df15c`.
- Cliff base entry: `TEXTURE_DiffuseTint_CivV_Cliff_Rocks_B_null`.
- `ArtDefs/TerrainStyle.artdef` SHA-256:
  `97aa31e96f4216ff704c90317ca441df50344f54d00dcf9cae64c8d2c2f68db3`.
  Confirmed Cliff controls include Height=5, IsCliff=true, position offsets=1.1,
  height offsets=1.3, and position/height roughness=.5. Separate large, small,
  and upper small-rock clutter sets are present.
- `ArtDefs/Clutter.artdef` SHA-256:
  `0bf3201ff7c51adc8003b8b00754ce24c8c556498297f08b82d01bd4edd5cc7a`.
  Confirmed large cliff clutter uses TerrainHeight=false, FixedHeight=0,
  RotateZ, ScaleVariation=.1 and AllowOverlap=true.

These values do not prove source-engine world units or placement equations.
The existing normalizer divides source mesh positions by 12. The provisional
C3X grassy shoulder uses Height 5 / 12, a uniform .5 placement calibration,
the fixed 112-pixel vertical basis, and a bounded coastal envelope. This is
`source_adaptation`: subordinate topology behind actual source rock bodies,
not a recovered source mesh or an invented replacement rock mass. Source hill
height sampling and all rock UVs remain intact. The cliff LEAN interpretation
and water normal/moment interpretation remain C3X lighting adaptations.

Actual composed draw metadata confirms cliff base/LEAN0/LEAN1/gloss views
`[72,83,80,71]` and water large/small normal/moment views `[11,35,11,35]`.
Gloss uses a compression-compatible linear view of unchanged source payload.
Water textures come from the selected terrain pack. Runtime consumes generic
compiled geometry/material data; source-specific extraction remains offline.

## Supporting checks and remaining work

`COAST_rocks-r8_EVIDENCE.json` compares all 20 frames with retained r4, including
the untouched region. `COAST_SOURCE_BINDING_DIAGNOSTICS.json` separates the
material correction, rejected burial, shoulder, water and build-only changes.
Neither record asserts visual acceptance.

The Q6 postprocessor discards the old kind-10 terrain shadow surface before
building its real world shadows. An opt-in skips that redundant CPU build;
the r5 control rendered byte-identical BMPs at both zooms. The runner permits
this only with the Q6 world-shadow fixture. Actual source-mesh shadows remain.

Passed: Lab workflow (132 Python tests, 12 Node tests, campaign validation),
18 platform checks, source-coordinate and varied-coast regressions, and the
actual height-kernel shared-edge/flat-water regression extended to the new
shoulder. Local Metal composition compiled and rendered every retained frame.
No full/VM/injected verification is claimed for this unfinished Lab pass.

The next three visible gaps are: abrupt grass/rock joins and pond-bank shape;
soft shallow-water structure and absent shoreline water interaction; inherited
diagnostic dunes plus unresolved mountain/volcano physical extent and side
projection. The synthetic volcano remains separate because verified test.biq
contains zero volcano tiles. No source-fidelity exception or human approval
has been invented.

```sh
python3 Renderer/terrain_lab/v2/qa/coastal_pass.py --region all --revision rocks-r8 --hours 12 0 --output-root Renderer/terrain_lab/v2/audits/beauty/out/replay-coast-r8
python3 Renderer/terrain_lab/v2/qa/coastal_pass.py --region freshcoast --revision rocks-r8 --hours 12 0 --output-root Renderer/terrain_lab/v2/audits/beauty/out/replay-coast-r8
python3 Renderer/terrain_lab/v2/qa/verify_coastal_pass.py --revision rocks-r8 --previous rocks-r4 --regions coastal inland wilderness longcoast freshcoast
python3 Renderer/terrain_lab/v2/qa/present_coastal_pass.py --revision rocks-r8 --previous rocks-r4 --regions coastal inland wilderness longcoast freshcoast
```
