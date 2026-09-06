# Large terrain evaluation preview

Requested by the user for visual evaluation. This is an unaltered 128-tile
16x8 crop of the verified test.biq, raw origin (58,56), with six neighboring
halo tiles for source shadow queries. Twelve source terrain families and 21
river tiles are present. No cities, units, routes or other objects are added.

The overview uses Q2 continuous ground weights/source UV, Q3 corrected signed
shoreline and Q6 source-compatible linear shading, with anisotropy 8, zero mip
bias, 4x MSAA and no sharpening. Requested outputs are 2400x1600 and 1200x800.
Existing candidate geometry-normal and final caster-shadow gates remain pending;
this is a user review artifact, not a milestone promotion.

Replay from the repository root:

```sh
python3 Renderer/terrain_lab/v2/app/runner.py compose --fixture Renderer/terrain_lab/v2/fixtures/terrain/large-evaluation/overview.fixture.json --candidate large-map-evaluation-01
```

The initial large render exposed a Q0 source-metadata lookup defect: coastal
asset texture indexes are rebased for shader binding, but metadata used the
rebased value as an index into the bundle-local texture list. The exact failure
and proposed preservation of the original source index were sent to Q0.
No source assets or shared code were modified by Q2.

Q0 corrected the source texture metadata lookup. Both requested images now render successfully and are directly inspected. The current shoreline retains angular notches in the lower-right sandy area, and existing relief/shadow limitations remain visible for evaluation.
