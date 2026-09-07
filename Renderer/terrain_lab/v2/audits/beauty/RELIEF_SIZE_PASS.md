# Larger mountain and volcano bodies

2026-09-06. Retained incremental candidate: `relief-size-r3`. The user explicitly
requested larger mountains and volcanoes, including width and modest overlap
into neighboring tiles. No visual gate, human approval, or Integration promotion
is recorded. The preserved previous best is `coast-pass-rocks-r8`.

## Visible result and fixed comparison

Mountains use a 1.30 uniform source-body scale; the volcano uses 1.60. Both
horizontal sampling coordinates and vertical displacement change together before
topology joins. Their foothills can extend at most .25 tile beyond each ownership
edge and fade before the neighboring tile center. This replaces the previous
hard restriction to relief cells and the narrow edge envelope. Source height
samples are unchanged. Coastal water joins and existing river-valley carving
remain; no camera or terrain classification was changed to make the bodies larger.

At native gameplay size, the inland central range has wider rock bases, larger
connected ridges, and more height relative to the unchanged forest crowns.
The isolated mountain near full-image x680–800/y320–410 is broader as well.
The synthetic volcano occupies a substantially larger footprint in this same
area and now reads clearly above the neighboring forest. Its lower source rock
continues across the tile edge instead of ending in a bare grassy face.
These changes move toward the imposing ridges and mountain-to-forest proportion
in canonical `mountain.png`. They do not establish correct source-engine units
or Civ VI-equivalent appearance; the volcano sides still look streaked.

The three original 100-tile regions, 128-tile long coast, and previous 100-tile
fresh coast remain fixed. Another 100-tile region at raw `[46,46]`, halo 6,
contains four mountains and 14 land/water edges. It was selected by coverage
before viewing this size pass there and received no region-specific tuning.
It is now a regression witness, not an untuned witness for later acceptance.

All six real regions were composed at noon/midnight and both original gameplay
zooms. A separate synthetic 100-tile inland copy contributes four additional
frames, for **28 retained combined frames**. It reuses the same one-tile
mountain-to-volcano replacement as the earlier witness and now includes the
current cliff/water/hill treatment. Its own matched baseline includes that same
composition. Verified test.biq still contains zero volcano tiles, and no real
benchmark is relabeled as synthetic or modified.

Vegetation XY anchors, source mesh IDs, scales, yaws and instance counts are
unchanged. Grounding heights are resampled where the enlarged terrain reaches
an existing plant. Cities, units and improvements remain deferred; this is not
evidence of their clearance acceptance. The camera, output dimensions, terrain,
scenario contents and source texture payloads match each corresponding baseline.

## Attempts and interface correction

- `r1`: 1.30 scale for both source families. Mountains gain visible mass, but
  the volcano remains comparatively small. Preserved as an intermediate result.
- `r2`: retain mountains at 1.30 and increase the volcano to 1.60. This exposes
  a material-ownership error: the height field crosses the tile boundary while
  volcano material classification and `frac(world)` coordinates stop/reset there.
  The bare grassy cut through the enlarged body is a regression; that volcano
  result is rejected.
- `r3`: an optional generic float4 terrain attribute carries source-owner UVs,
  coverage and activation through the existing extensible packet attributes.
  All volcano material channels use these coordinates across the skirt. Actual
  terrain classification is preserved. The expanded source body stays textured
  in the complete scene and is retained as incremental progress.

The optional attribute is gated by the module and requires the existing broad
relief, world and hydrology data. It does not change the wire version, historical
handoffs, native renderer, or injected code. A regression exercises the actual
CPU sampling and material handoff: uniform scale at corresponding source points,
nonzero neighboring skirt, zero contribution at the neighboring center, and
matching source UV/coverage/activation on both sides of a receiving-tile edge.

## Shadow investigation preserved separately

Before the size request, shader-only isolation identified the thin dune lines
as false shadowing rather than open geometric seams. Disabling all shadows
removes them; disabling only contact darkening does not. Removing the shading
normal offset makes the lines worse. None of these diagnostics is selected;
the actual cast shadows stay enabled. `RELIEF_SHADOW_DIAGNOSTICS.json` pins
the matching packets and shader identities. Next investigation should examine
receiver-plane filtering, shadow-depth quantization and adjacent-facet error,
not repeatedly tune dune color or remove grounding shadows.

## Evidence, limits and reproduction

`RELIEF_SIZE_r3_EVIDENCE.json` verifies the 28 matched frames and source/placement
invariants. Noon changed pixels at zoom 1 include 96,436 inland, 52,367 on the
long coast, 31,386 in the new region, and 107,778 in the combined volcano witness.
Counts include recomputed shadows and are not quality scores. Native crops and
full zoom-2 day/night comparisons are under `out/relief-size-r3/review`.

Passed: Lab workflow (132 Python tests, 12 Node tests and campaign validation),
18 platform checks, six lighting checks, actual source-mapping/size/material-edge
regression, and the shared-height/flat-water regression. Every retained frame
compiled and rendered on local Metal. No full/VM/injected verification or visual
promotion is claimed for this unfinished Lab pass.

Three largest remaining gaps: mountain/volcano surface projection and natural
shoulder integration; shadow-receiver lines plus unapproved analytic dune bodies;
soft shallow-water structure and abrupt cliff/grass joins. Larger scale does not
resolve source physical reconstruction, and the synthetic volcano's active
material is not an accepted eruption or emissive-light implementation.

```sh
python3 Renderer/terrain_lab/v2/qa/relief_size_pass.py --region all --revision r3 --output-root Renderer/terrain_lab/v2/audits/beauty/out/replay-relief-size-r3
python3 Renderer/terrain_lab/v2/qa/inspect_relief_size.py --revision r3 --present
```

The final presentation command requires Pillow. Replay outputs need a fresh
directory; preserved completed reports are never overwritten.
