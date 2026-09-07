# Water object reflections

The user said **“Water looks great”** about water-natural-r6 and asked for
reflections from objects above it. Preserve that water appearance. This feedback
is not milestone approval, approval of this new reflection candidate, or a
Civ VI-equivalence claim. Animation and coastal surf remain deferred.

**water-reflection-r5** adds a real planar-reflection pass to the standalone Lab.
It redraws the existing geometry mirrored across the common water plane, then
samples that scene-linear target in the water shader. Both passes run in one GPU
command buffer on Metal, with the same GPU pass ordering in D3D11. There is no intermediate image readback, baked reflection asset,
geometry rebuild, second presenter, or injected-code change.

## Visible result

Shoreline rocks acquire a muted inverted continuation below their water contact;
overhanging canopy edges acquire small dark/green reflections. These are actual
reflected scene silhouettes, distinct from the existing directional cast shadows.
The current wave normal slightly distorts the sample, and the existing Fresnel
response controls its contribution. Open water retains the r6 sky/ripple result.
The effect is intentionally subtle at the fixed gameplay scale.

- [Long coast before/after at native zoom 2](out/water-reflection-r5/review/longcoast-h12-z2-pan00.png)
- [Rock comparison at native gameplay pixels](out/water-reflection-r5/review/rocks-native.png)
- [Rock detail, explicitly enlarged 3x](out/water-reflection-r5/review/rocks-detail.png)
- [Canopy detail, explicitly enlarged 3x](out/water-reflection-r5/review/trees-detail.png)
- [Frozen coastal holdout at night](out/water-reflection-r5/review/freshwater-h00-z2-pan00.png)

Twenty combined frames cover coastal, inland, wilderness, longcoast and the
previously unseen freshwater-named coast, noon/midnight and both fixed zooms.
Cameras, terrain, object placement, output sizes and complete source packets
are unchanged. This is a reflection candidate, not a new complete scene promotion.

## Implementation and rejected diagnostics

The reflection adapter uses authoritative vertex world height and the existing
pixel projection. World z is authoring height /112, and projection height is
`.82 * half_width /112`; a reflected point therefore moves downward by twice
its original projected height. Depth is mirrored and compressed to retain order
within the diagnostic clip range. Original normals, material coordinates and
shadow coordinates remain unchanged so the object's original illumination is
reflected. Water and below-plane geometry cannot reflect themselves.

R1 incorrectly included positive-height terrain support beneath the water,
producing a false reflected land sheet. It is preserved and rejected. R2 also
clips main terrain by continuous optical shore distance, removing that support.
The two-render r2 diagnostic writes a linear RGBA16F target solely to establish
an independently inspectable reference. R3 moved the passes onto GPU and matched
the long coast exactly.

A guard in r4 caught an important namespace alias: t121 is unused in the main
terrain/water draws but holds an existing resource texture in feature draws.
R5 overrides t121 **only for main/water draws** and leaves feature bindings intact.
It rejects an occupied main slot. This preserves the earlier namespace contract;
it does not reserve that slot globally or change the packet format.

`backends/metal.mm` optionally accepts a reflection shader directory after the
existing linear-postprocess arguments. It allocates a linear reflection target,
runs the reflected pass first, resolves it before sampling, and runs the normal
scene. Reflection is absent by default. The diagnostic currently rejects multiple
shader namespaces and render scales other than 1. The two benchmark zooms use
the existing reconstruction/downsample path and are both supported.

`backends/d3d11.cpp` accepts an optional expanded reflection HLSL path after the
render-scale argument. It uses the same restrictions and main-only t121 binding,
resolves the mirrored RGBA16F target before sampling, and clears the shared depth
and validity attachments independently for each pass. The default path remains
reflection-free. This is the standalone Windows Lab backend, not the native DLL.

The new material input uses linear premultiplied color and coverage. Missing
reflection coverage retains the existing sky. Reflection sampling happens before
the existing final tone mapping and output conversion. No reflection is sampled
from the final display PNG. Pass counts and actual submitted draw counts are
included in backend cost records.

## Verification and cost

`WATER_REFLECTION_r5_EVIDENCE.json` records:

- 20 source-packet hash matches; finite linear output and unchanged validity masks.
- Eight offline-reference versus direct-GPU comparisons across longcoast and
  freshwater, with at most a one-level/one-pixel diagnostic tolerance.
- Four reflection-disabled captures exactly matching water-natural-r6.
- Eight repeated frames per camera for cost, run separately from other GPU jobs.
- Four matched viewport-shift probes move the shoreline trees above the top of
  the view while preserving their reflections. Subtracting each sky-only control
  isolates the reflection effect: its shifted crop matches within one code value
  (three frames exact), with zero error in the first three rows. This proves
  offscreen raster handling for captured geometry; it does not prove the scene
  provider captures every potential reflector beyond the viewport.
- Sixteen Metal/D3D11 comparisons across the rocky long coast and tree-covered
  coastal holdout: eight reflected views plus eight sky-only controls, covering
  noon/midnight and both zooms. All pass the unchanged parity limits; all sixteen
  Windows repeat renders are byte-identical. Maximum frame-average RGB error is
  0.281 code values and p99 is at most 5. These two regions are a focused backend
  witness, not the full promotion matrix. The enlarged rock/canopy comparisons
  were inspected beside their Metal counterparts.

One Windows run failed opening a shared packet resource before rendering. All
1,296 local resource hashes (535 MB) were verified; explicit resume preserved and
validated completed outputs, then finished the missing cases unchanged. The
incident remains recorded in the evidence rather than being hidden by retries.

For the 1616x888 internal long-coast target, the measured extra GPU time is
3.2–7.5 ms across the four day/night/zoom cases. Main-plus-reflection submits 14
draws versus 7; sampled allocation increases about 59–60 MB. These are local
Metal diagnostic measurements, not a native-game frame-rate guarantee or a
completed performance gate. The Lab workflow passes 132 Python and 12 Node tests.

Run `qa/water_reflection_pass.py --region REGION --revision NEW_NUMBER --gpu`
for a new preserved capture. The driver refuses existing destinations.
`qa/verify_water_reflections.py` regenerates the r5 review/evidence. Historical
expanded shader closures remain beside their output reports.

Run `qa/reflection_crop_probe.py --region freshwater --shift-y -120` for the
offscreen witness; the driver refuses to replace existing results. A new shift
creates a separate diagnostic without changing the fixed benchmark cameras.
`qa/reflection_d3d_probe.py --region REGION --build --repeat` builds and replays
through `renderer_dev.native_command_result` on the documented Windows VM.
Use `--resume` only for an incomplete run; it verifies existing packet, shader
and output hashes and skips completed cases. It never replaces a captured image.

## Integration implementation notes

The reusable operation is a second view of the same scene reflected about an
explicit water plane. Keep the original object lighting, material coordinates,
alpha cutouts and cast-shadow coordinates, while reflecting projected position
and depth. Exclude water, geometry below the plane, and hidden supporting terrain
under water. Resolve linear premultiplied color plus coverage, then sample it
only from the water material before tone mapping. Zero coverage retains the sky.

The Lab's `.82`, `/112` and mirrored-depth constants describe its recorded packet
projection. Integration must derive equivalent transforms from its authoritative
projection rather than transplanting those constants. Likewise, t121 is a
validated single-namespace Lab slot, not a new global pack binding. Reflection
visibility must include objects whose originals lie offscreen but whose mirrored
projection reaches visible water. Reflection-only contributors must never claim
native tile replacement ownership. Use the existing captured scene and shared
environment state; no second presenter or runtime Civ VI dependency is needed.

## Remaining scope and delivery

This is implemented in the standalone Metal and D3D11 Lab, **not yet the Windows
C3X renderer**. Before delivery, verify provider halo/culling coverage and cost
in the integration architecture. The current reflection
plane is common sea/pool z=0. Rivers at varying elevations and multiple water
planes need their own contract and are not silently reflected about sea level.
Roughness blur, reflection fade and a broader combined visual checkpoint remain.
No milestone gate or frozen pickup is changed.

The prior seabed audit also found that one height modulation samples different
ocean atlas patches/scales than its coastal color layer. It was diagnosed but
not changed before the user's reflection request; retain it as a later shallow
detail investigation.
