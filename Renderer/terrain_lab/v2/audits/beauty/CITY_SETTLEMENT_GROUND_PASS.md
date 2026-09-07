# Connected settlement ground r1–r2

Status: provisional **modern** ground improvement. Medieval extensions are
rejected. City quality, broader coverage, native delivery and all approval gates
remain open. One-era appearance, authoritative future capital state and the
deferred connecting-road scope are preserved.

## Visible result

The modern city now has a shared paved surface between its existing building
pads. The clearest gains are in the inland city and around the American palace:
small grass gaps become a coherent built site. Geometry, building placement,
terrain, shadows, source materials and the selected local night-light records
remain matched. The paving receives the existing shadows and facade spill.
The capital lake reflection ROI is pixel-identical to the preceding candidate.

The local selected candidates are `out/city-settlement-ground-r2/{inland,
wilderness,freshshadow,capital,small}/render`. Their matched comparison is
`out/city-settlement-ground-r2/selected-native-comparison.png`; the small/medium
wilderness growth witness is `growth-native.png` in the same directory.

| Modern case | Noon changed pixels >2/255 | Midnight changed pixels >2/255 |
| --- | ---: | ---: |
| Inland large | 847 | 828 |
| Wilderness medium | 429 | 424 |
| Freshshadow medium | 642 | 629 |
| American capital | 1,431 | 1,404 |
| Wilderness small | 519 | 513 |

These are gameplay-size contributions toward the connected building bases in
`Renderer/canonical/nightlights.jpg`. They do not prove its overall richness,
urban layout or material response. The untuned ground application to freshshadow
uses the same modern parameters; its crowded skyline remains unaccepted.

## Geometry and sampling

`systems/objects/settlement_ground.py` builds the union of ground-level building
rectangles, with a rounded 0.1-tile apron and 0.025-tile edge feather. A shared
0.025-tile grid samples the pinned terrain. Exact tile-cell clipping excludes
water and forest/jungle cells; the existing signed shoreline clipping further
restricts the mesh. This is an **authored footprint adapter**, not recovered
Civ VI generator geometry. Full route/river-mesh clearance remains unproven.

The underlay samples an inspected unmarked atlas interior, UV rectangle
[0.62,0.56,0.94,0.86], retaining the source alpha. It mirrors that interior using
explicit gradients and world-aligned coordinates. Existing compound triangles
and atlas UVs remain untouched. No atlas, texture or pack is duplicated.

r1 derived texel density from visible compounds, which would change the pattern
as buildings were added. r2 instead measures the entire configured source pool
at the uniform city scale. The small and medium stages have identical periods
[0.7104908582,0.6660851796] tiles and exactly matching first-four footprints.
Paving coordinates therefore remain fixed through that growth step.

`qa/append_settlement_ground.cpp` inserts one alpha receiver draw before existing
city decals. It adds one buffer and reuses an existing atlas resource. It adds no
caster, performs no shadow rebuild and changes no prior draw or texture. The Lab
wire encodes coverage in reserved material values 62–63. This is an explicit Lab
contract; it does not silently establish an approved native material ABI.

The small-city baseline is the preserved r47 four-body prefix, with the existing
facade-light gain 4 and era-ground replacement applied before this underlay.
It is not a newly tuned city layout or proof that its early tower selection is
visually appropriate.

## Rejected medieval treatment and investigation

Both r1's sharp edge and r2's 0.06-tile feather still read as a broad flat orange
platform. Neither replaces `out/city-ground-binding-r1/medieval/render`, the
previous best. See `out/city-settlement-ground-r2/medieval-compare.png`.

Rather than keep adjusting the feather or color, the follow-up inspects actual
installed ground-height materials. Both source height payloads already exist
in the normalized pack, verified by compressed payload identity:

- Modern: `height_8e9be2515c4165b5.dds` under the compound texture directory.
- Medieval study: `height_3a104c47da8d5dbc.dds` in the same directory.

They are BC5 RG inputs, not established tangent-space normal maps. In the selected
patch, red varies only 84–90 for modern and 91–92 for medieval. Green resembles
coverage: it matches modern base alpha in this patch, while the medieval patch
has a mean difference of about 0.05/255 and maximum 11/255. Full-atlas differences
are larger. Exact channel and terrain/decal blend semantics remain unproven.
Consequently, adding a generic bump response is not evidence that the missing
appearance will be recovered. Investigate source blending and footprint/UV usage
before another medieval material adjustment. `height-intake.json` and
`height-channels.png` under the r2 output preserve the measurements.

## Verification and continuation

Twelve Windows D3D11 comparisons pass for the selected modern cases, including
both capital zooms. Six analytic ground tests pass. Sixteen independent packet
checks (also covering rejected medieval output) verify that original geometry,
constant buffers, materials, draw state and shadow textures are unchanged.
The inland gain-zero control reproduces its preceding day/night images exactly.
Pixel differences are localized; the evidence records any isolated 1/255
outside-city variation rather than counting it as improvement.
Completed new HDR sidecars were removed after verification: 386.0 MiB across
48 files. Images, rejected trials, frozen shaders and shared packets remain.
See `CITY_SETTLEMENT_GROUND_CLEANUP.json`.

Reproduce a new candidate with `qa/settlement_ground_probe.py`, supplying an
existing `--source-render`, `--augmentation`, `--ground-parts`, `--binding`, and
fresh `--output` under the beauty audit output. Run it with the offline
Pillow/NumPy Python environment. `--gain 0` is the disabled control.
`qa/settlement_ground_evidence.py` verifies the saved case set, source hashes,
packet isolation, fixed ground coordinates, dry/vegetation cell exclusion,
localized image changes and Windows evidence. Connecting roads and injected
code are untouched; no full Lab or milestone pass is claimed.

The next larger visible problem is city growth hierarchy: the small stage already
contains the tallest towers, so it reads as an incomplete metropolis. Revisit
lower-to-taller component order using the newer constrained layout search, then
expand culture/era coverage. Preserve these ground candidates and fixed terrain
while comparing. Crowded freshshadow, wilderness eleven-body fit, facade
environment specular and the medieval ground treatment remain open.
