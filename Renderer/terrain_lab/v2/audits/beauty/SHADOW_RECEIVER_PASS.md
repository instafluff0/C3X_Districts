# Combined receiver correction

2026-09-06. Retained work in progress: `shadow-receiver-r1`, building on
`relief-size-r3`. No human approval, milestone closure or promotion is claimed.
All historical candidates and the larger source mountain/volcano bodies remain.

## Visible result

The native wilderness comparison shows substantially fewer thin dark lines
across the desert at full-frame x360–750/y260–450. Those lines previously traced
the tessellation instead of the visible dune form. Tree shadows around
x770–867/y368–443 and the mountain shadow near x450–477/y514–544 remain present.
Compare `out/shadow-receiver-r1/review/wilderness-h12-z1-comparison.png` with
canonical `Renderer/canonical/desert.png` and `hills.png`: removing the mesh-edge
pattern moves toward their continuous terrain lighting. This does not supply
their finer source dune shapes, rock detail or texture richness.

The same setting is composed across the three fixed 100-tile benchmarks,
longcoast128, previous freshcoast100/freshrelief100, a new freshshadow100 and the
separate synthetic combined volcano witness. Noon/midnight and both fixed zooms
give 32 frames. Full-size crops and native zoom-2 pairs are generated without
resampling. Night readability and visible cast shadows remain; no light or
material was disabled. Faint residual lines and hard facet patches on some
shoulders remain visible, so this is a reduction rather than complete removal.

The fresh region was selected by source coverage before viewing the candidate:
raw origin [54,64], 10x10 with halo 6, including 16 desert, 11 hill, 13 mountain
and 12 forest tiles. Its own unchanged baseline uses identical source terrain,
placements and cameras. Its registry request and camera are frozen in
`fixtures/beauty/shadow-receiver-foundation/freshshadow/BENCHMARKS.json`.
It is now a regression witness, not an untuned witness for a later pass.

## Diagnosis and retained implementation

The read-only `qa/shadow_receiver_probe.cpp` samples the actual retained packet,
its terrain triangles and its R16 shadow field. In the wilderness diagnostic
rectangle, the field span is 10.75 normalized units at 2048 texels. At triangle
centers the 99th-percentile maximum 3x3-tap excess depth is about .000205;
near edges it is .004779. These are measured depth discrepancies, not quality
scores. Some high values also include legitimate separate casters. The far
larger edge discrepancies explain why a constant .001 normal displacement
fails around adjacent facets. The old physical comparison bias is .00060.

With `Q6_TEXEL_RECEIVER_OFFSET=1`, the receiver moves along its normal by one
shadow texel, capped at 6/1024 normalized units. The wilderness offset is
10.75/2048 = .005249. Receiver-plane derivatives use the unshifted geometry so
material-normal variation cannot define the plane. The ordinary/default path
retains the old offset and derivatives. The depth field, all caster triangles,
3x3 filter, physical comparison bias and contact rule remain unchanged. The cap
prevents oversized scenes from continually widening the offset. This is a Lab
lighting adaptation, not a claim about Civ VI's shadow implementation.

`SHADOW_RECEIVER_r1_EVIDENCE.json` verifies byte-identical input packets for
every pair, including their geometry, source materials, shadow field and
placement; source metadata hashes are identical too. It separately hashes
changed output pixels. Visual acceptance and approval remain false/null.

## Rejected and diagnostic work

All outputs remain under `audits/beauty/out/` with the following names:

| Attempt | Finding | Disposition |
| --- | --- | --- |
| shadow-plane-diagnostic-r1 | Geometric derivatives alone leave the lines. | Diagnostic only. |
| shadow-center-diagnostic-r1 | One shadow tap narrows the lines, but loses filter softness. | Rejected. |
| shadow-geometric-normal-diagnostic-r1 | Facet-normal offset adds hard shoulder artifacts. | Rejected. |
| shadow-caster-plane-r1 | Storing caster gradients and reprojecting them produces new edge patterns. | Rejected. |
| shadow-caster-plane-r2 | Conservative use of both planes reduces lines but leaves an avoidable data change. | Not selected. |
| shadow-texel-offset-r1 | Same original packet with measured-footprint offset visibly cleans the sand. | Composed as shadow-receiver-r1. |

The caster-plane experiment is preserved as
`fixtures/beauty/shadow-caster-plane-r1/experiment.patch`; it is **not applied**
to the retained shadow builder. Its fixture requires that patch to reproduce
the experiment. Expanded source shaders are ignored diagnostic artifacts, not
selected source files. The previous `RELIEF_SHADOW_DIAGNOSTICS.json` remains
historical evidence; disabling all shadows was never selected.

## Volcano source-channel finding

The actual frozen loader binds `textures/water/volcano/height.dds` as BC5_UNORM.
The existing import report identifies `TEXTURE_Feature_Volcano_H`, 512x512,
source payload SHA `1bf210d121afa17fa89aadc17aaca3a00ebe56071a44e7f67d7d5f06aafaee4e`.
Direct channel inspection shows detailed relief centered on .5 in red and a
smooth footprint-like field returning to zero in green. This contradicts the
current use of both channels as signed XY normal components. The exact source
meaning of green remains unconfirmed. No source-engine equivalence is inferred.

`volcano-height-normal-diagnostic-r1` reconstructs finite-difference normals
from red, retains the existing normal strength and blends into the source-owned
coverage. Its four shader-only frames preserve all geometry and bindings.
Changed noon pixels are confined to x664–815/y307–416, but the gameplay pair
does not convincingly recover the missing face detail; it is **not selected**
or counted as a visual improvement. Texture stretching and source-height
physical reconstruction still need investigation. Do not resume by treating
green as a normal component or by simply increasing the same gain.

Channel PNGs and native comparisons are in
`out/volcano-source-channel-inspection/` and
`out/volcano-height-normal-diagnostic-r1/review/`. The base-color preview uses
the same BC3 payload, with an in-memory SRGB-to-UNORM header alias only because
Pillow does not recognize the SRGB DDS enum; no source file was modified.

## Checks and reproduction

Passed: Lab workflow (132 Python, 12 Node, campaign validation), six lighting
checks, all 32 matched frame/packet comparisons, and actual Metal composition.
No injected source or shared packet contract changed; no VM/full verification
or milestone closure is claimed. Canonical desert/hill images were inspected
directly. The three largest remaining gaps are source dune reconstruction and
residual facet artifacts, mountain/volcano projection and material detail, and
soft shallow-water structure with abrupt cliff/grass joins.

```sh
python3 Renderer/terrain_lab/v2/qa/shadow_receiver_pass.py --region all --output-root Renderer/terrain_lab/v2/audits/beauty/out/replay-shadow-receiver-r1
python3 Renderer/terrain_lab/v2/qa/shadow_receiver_pass.py --region freshshadow --baseline --output-root Renderer/terrain_lab/v2/audits/beauty/out/replay-shadow-receiver-baseline
python3 Renderer/terrain_lab/v2/qa/inspect_shadow_receiver.py
python3 -m unittest discover -s Renderer/terrain_lab/v2/tests/lighting -p 'test_*.py'
python3 Renderer/tools/renderer_dev.py lab
```

Rendering requires the local Metal device. Presentation/inspection requires
Pillow. Completed render directories are protected; choose fresh replay paths.
The inspector reads the retained paths by design. Licensed source-derived
payloads and images stay ignored and local.
