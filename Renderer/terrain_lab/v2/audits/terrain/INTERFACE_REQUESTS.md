# Q2 convergence interfaces (r09)

These are pending convergence obligations, not launch gates. The owned base
provider already renders on the Mac through `cpp_packet`/packet v2.

## Q0: compose the real continuous field

Publish an optional, versioned base-field provider entry before terrain mesh
construction. Proposed call shape: `SurfaceSample sample_base(scene, lattice_x,
lattice_y)`, consuming immutable visible+halo tile records, canonical raw origin,
map dimensions and explicit wrap flags. Q2's concrete implementation is
`systems/terrain/surface.h`: `q2::Surface::sample(double x, double y)`.
Its coordinate mapping is raw X = origin X + x + y - 1, raw Y = origin Y + x - y;
cell centers are `(column + .5, row + .5)`. Five weight slots are grass, plains,
desert, marsh, tundra. Flood plains consume the normalized plains channels,
independently of their desert base (confirmed pack binding).

Q0 has now published and connected opt-in `TerrainHooksV1` in
`shared/scene_hooks.h`. `systems/terrain/scene_adapter.h` supplies initialize,
material_weights and material_uv. The complete scene calls weights before mesh
construction and source UV on ground/land passes only; no second plane is added.
`composed-complete.module.json` opts in, and the matching baseline leaves hooks
null. Source UV uses raw map coordinates with integral repeats; local crop UV
would reset its phase between neighboring captures. All 771 shared-edge samples
on the dry/cold/wet fixed neighboring recaptures now have matching weights/UV.
Complete cold and wet quick witnesses have been directly inspected. These use
frozen material lighting, not the isolated Q2 detail shader.

The former material-response blocker is resolved: Q6 exposes two optional
shader calls in the full source-compatible scene-linear adapter. `shaders/terrain/scene_material_v1.hlsl` supplies
`q2_material_form` before diffuse and `q2_material_specular` before highlight.
It preserves existing source elevated/relief/decal responses and adds only
subordinate 3x/8x source detail, bounded by a continuous relief/slope envelope.
This composed adaptation now has directly inspected r14 on/off review at both final zoom sizes. The pure
isolated response was extracted into `material_response_v1.hlsl`; r09 four
native scroll frames are byte-identical to the reviewed r06 images.

Q0 texture-array witness: r01 used `Texture2D textures[15]` which became one MSL
`[[id(0)]]` array. The backend bound only explicitly parsed IDs, leaving slots
1..14 absent and most terrain black. r02 replaced it with explicit t0..t14
bindings inside Q2. The corrected module needs no shared edit for this issue.

## Q1: material detail versus filtering

Consume the r06 context detail/off controls with no sharpening, anisotropy 8,
zero mip bias, one sample and native/reduced pixels. Source height is sampled at
1x/3x/8x; secondary amplitudes .22/.07; the normal's raw-X/raw-Y derivatives are
rotated into the two orthogonal lattice axes. Component slopes are capped at
.28. Texture repeats around raw X/Y extents are integral, so aliases preserve
phase. No new procedural texture is used. Compare the same exact fixtures with
Q1's sharpness policy and real linear attachment when available. A larger filter
contrast is not evidence of missing source-detail structure being repaired.

## Q3 and Q4: full pair geometry

`fixtures/terrain/matrix_v1.json` declares every unordered vanilla family pair,
axis, reversal, aliases and selected junctions. Water-facing transitions are
shore-mediated; relief-facing transitions are shoulders. Q2 owns the matching
base weights, not shore distance/depth, pools, rivers, dune/hill/mountain bodies
or their normals. Current all-water isolation deterministically supplies a
neutral grass underlay, as the frozen base routine does. This is NOT water art
or evidence that an ocean tile is land. Replace/cover that underlay with the Q3
bed/water contract before composed acceptance. Real-map river flags are preserved
in CSV inputs but this base-only module intentionally emits no river geometry.

## Q6 and Q8

Q2 includes Q6 `shaders/lighting/response_v1.hlsl` read-only, using the existing
shared environment evaluator. The independent r06 outputs remain single opaque passes on the provisional
display-encoded platform. Composed r14 uses Q6 scene-linear premultiplied
attachments and source-compatible material evaluation. Full shared occlusion
and final gameplay composition remain pending.
Q2 does not add contact shadows, lights or a separate clock.

Q8 can consume the Q0 terrain hooks in plausible gameplay compositions. The owned 512x256 crop is explicitly synthetic
open countryside, 128x64 tile basis, no added objects. It supplies matched
baseline/off/on evidence and does not claim captured gameplay or beauty closure.

## Actual-map coverage

The verified dataset has SHA-256
`a6a88d7fffcc567c3500bbd5aa947398dd48170d4f412aa1e518bb45ffe8453e`.
Initial mixed/holdout/wrap exports run unchanged and are directly inspected,
but mostly reduce to grass in base isolation. `REGION_REQUEST.json` supplies
read-only-discovered dry/cold/wet origins. Q0 subsequently registered all three
with fixed neighboring holdouts; all six exported regions now pass the base-only
checkpoint and were directly inspected. Volcano is absent and remains synthetic.
No source BIQ is modified.

Q1 independently rendered the r06 on/off controls and sampling A/B at both
final sizes. Its current recommendation is zero mip bias, anisotropy 8,
sharpening off, and 4x MSAA for geometry. Material gain is restrained and does
not justify negative bias or a sharpening pass. The real linear attachment
comparison remains a convergence gate.

## Decisive remaining Q4/Q0 convergence failure

The Q3 signed-shore hook is adopted via owned `hydrology_consumer.h` and fixes
the user-rejected pale inland rectangle and lip. Wet/cold/dry plus fixed wet
holdout pass 32 composed phase/zoom deterministic image checks and direct review.
Q1's existing sampling recommendation is used; two-zoom composed detail controls
are reviewed. See `composed_metrics_v1.json`.

The source geometry query now proves 145 wet and 153 dry incident-edge normal
failures, with maxima .421376169 and .826748308 respectively. Heights and shore
agree. Cold passes the .001 normal tolerance. Q4/Q0 must supply continuous
raised-surface derivative normals; Q2 cannot substitute its flat +Z base normal
for authored relief. Exact witnesses and executable query/gate are published.
This is a real cross-owner blocker, not an unstarted dependency or a claim that
all remaining visual criteria pass. No required user action is introduced.

## Updated caster-shape requirement

The coordinator's new user-directed requirement is consumed: every cast shadow
must use the actual transformed/posed source caster geometry and authored cutout
alpha, through Q6 shared lighting and real receivers. Generic foliage ribbons,
ovals or footprint substitutes cannot pass. Current frozen/composed diagnostic
shadows are not promoted as accepted shadow evidence. Q2 adds no shadow hack or
shadow suppression. Q2 publishes continuous base weights/UV and flat base datum;
Q4 owns the actual raised terrain and vegetation geometry, while Q0's exact
surface query exposes their current receiver geometry. Its normal failures remain
explicit. Final acceptance additionally requires Q6's actual-silhouette shared
shadow composition; physical placement clearance remains a separate contract.
