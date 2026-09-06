# Q6 lighting candidate audit

Status: active actual-caster-shape convergence, not package acceptance. Frozen L21 and Game
Integration are unchanged. This audit records directly inspected images as well
as executable evidence; hashes alone are not visual approval.

## Gameplay context first

Primary before/after: `out/Q1/Q6-lighting/context-before-after.png` is an exact
matched crop of the verified `test.biq` mixed region, comparing frozen output
with the first shared response adaptation. That intermediate adaptation retained
display-space blending and is superseded by `real-linear-mixed-11/review-z1.png`
and `review-z2.png` beneath the same output root. The latter use actual HDR scene
composition, 4x MSAA, anisotropy 8, bias 0, scale 1 and no sharpening.
No geometry or object scale was changed to beautify the before/after. These are
small real-terrain contexts with separate deterministic Lab city/route layers,
not captured game-state objects. The selected alternate environment skin is
inherited from the fixture; it is not silently relabeled Civ VI art.

The source BIQ hash is
`a6a88d7fffcc567c3500bbd5aa947398dd48170d4f412aa1e518bb45ffe8453e`.
Primary raw origin is (14,42), 4x4, halo 2; held-out mixed origin (18,46), same
extent/halo, horizontal wrapping. Owned fixtures preserve registry metadata,
terrain and augmentation digests. The source has no volcano; this is a recorded
coverage gap. The held-out region contains four source river tiles, but dense vegetation
obscures their presentation; this does not prove full river material acceptance.

Direct observations: coast/bed variation remains visible, noon has clear relief,
sunset is warmer than sunrise, midnight has cool relief and legible roads plus
localized city windows. Midnight does not erase unlit geometry. The first HDR
render (`real-linear-10`) exposed tree-shadow polygons outside the map diamond;
Q0 corrected validity writes so only opaque coverage establishes map validity.
The rebuilt `real-linear-mixed-11` matrix directly confirms the fringe is gone
at both zooms and all phases. Current inherited tree shadows and category form
adjustments remain provisional. Forest clearance can hide route/city portions
in the held-out context; Q4/Q5/Q7 own placement, not Q6 lighting.

## Focused source diagnostics and iteration

The initial analytic proxy (`proxy-01`) exposed an inverted vertical projection;
`proxy-02` corrected it to the 2:1 basis. The boxes, cutout lattice, water and
synthetic window mask are explicitly diagnostic proxies, excluded from beauty
acceptance. Shared native EnvironmentState supplies every phase; no category
clocks or invented local light attachments were introduced.

Initial inherited city diagnostics exposed detached source roof/slab pieces.
Q7 traced this to first-geometry/first-material extraction from compound assets.
`prepare_q7_city.py` now consumes Q7's complete source components and stable
layout read-only. Old city-01 through city_linear-07 remain historical failure
evidence, not accepted source completeness. Final `q7_city-09` uses complete
American modern components: four material/emission pairs and 4742 triangles
including the diagnostic receiver, at native 128x64 tile projection. Whole
components receive uniform scale/rigid transforms; source UVs remain unchanged.

The shared generic world-triangle shadow field evaluates actual receiver XYZ.
Initial rock-07 renders had visible slope stippling from fixed depth comparison.
Receiver-plane depth correction at each PCF texel removed that acne in rock-08
and final rocks-09 while preserving contact. This is the highest-impact lighting
correction in the source matrix. Contact uses a narrow blocker-gap interval,
not a painted footprint or blanket AO; source baked shading is not multiplied
by another ambient-occlusion field. Water is excluded from ambient contact.

Final source matrices: `q7_city-09`, `trees-09`, `rocks-09`, `units-09`,
`improvements-09`, with matching `_contact_off-09`, `_shadows_off-09`, and
`-scroll-09`. `q7_city_emissive_only-09` isolates source windows;
`q7_city_reverse-09` reverses opaque triangle order. Review sheets and the
`*-controls.png` matched actual-pixel crops were directly inspected.
Trees preserve their source crowns/trunks; contact is deliberately subtle and
has zero quantized effect at sunset in this representative tree fixture.
Rocks retain opposing face values without stippling. Infantry silhouettes retain
their source pose, and removing shadows visibly removes grounding. Source mine
emission stays on authored channels; bright gantry detail does not illuminate
nearby surfaces. Native city roofs/facades separate at noon; midnight retains
dark architecture and localized windows. No whole-city glow or clipped white
window blobs was observed. These flat-receiver source diagnostics do not approve
Q7 gameplay composition or prove the final real-terrain shadow integration.

## Contracts, metrics and reproducibility

Consumed boundaries: shared EnvironmentState; packet wire3/4 scene-linear
premultiplied branch; Q6 `COLOR_ALPHA_CONTRACT_V1.md`; Q0 `scene_exchange_v1`;
verified real-map registry v1. `scene_linear_provenance.json` pins immutable
source shader and generated linear adapter. Each rendered report pins shader,
geometry, tools, fixture and packs. Q2 source material hook is explicit and
read-only; it owns no exposure/tone conversion. An A/B with different Q0 backend
closure (`real-linear-10` versus `real-linear-q2-10`) is invalid for attributing
material improvement and is not used as such.

`EVIDENCE_METRICS.json` is generated by the strict evidence gate: 40 base source
variants (five categories, four phases, two zooms), all outputs unclipped,
per-backend byte repeat, city opaque-order invariance, return-to-origin identity,
phase luminance ordering and sunset/sunrise color ordering pass. Contact/shadow
control channel counts and RGB errors are recorded, not treated as beauty scores.
All 296 final source/control/scroll jobs completed before evidence compaction.
Six portable tests also pass, including shared shadow direction, receiver
intersection, alpha coverage, extent rejection and static-idle behavior.

Exact commands from the repository root:

```sh
python3 Renderer/terrain_lab/v2/systems/lighting/prepare_categories.py
python3 Renderer/terrain_lab/v2/systems/lighting/prepare_q7_city.py
python3 Renderer/terrain_lab/v2/tests/lighting/run_source_matrix.py
python3 Renderer/terrain_lab/v2/tests/lighting/test_lighting.py
python3 Renderer/terrain_lab/v2/tests/lighting/verify_evidence.py
python3 Renderer/terrain_lab/v2/systems/lighting/prepare_linear_scene.py
python3 Renderer/terrain_lab/v2/app/runner.py check --fixture Renderer/terrain_lab/v2/fixtures/lighting/real-mixed/linear.fixture.json --candidate real-linear-mixed-11
python3 Renderer/terrain_lab/v2/app/runner.py check --fixture Renderer/terrain_lab/v2/fixtures/lighting/real-holdout/linear.fixture.json --candidate real-linear-holdout-11
python3 Renderer/terrain_lab/v2/app/parity.py Renderer/terrain_lab/v2/audits/lighting/out/Q1/Q6-lighting/q7_city-09/report.json
```

Optional review generation uses Pillow with `tests/lighting/make_review.py`.
No injected/native gameplay source changed; an injected compile is inapplicable.
The final 192-tile promotion remains pending until convergence, not silently
substituted with a small synthetic matrix.

## Evidence retention and remaining work

A shared disk-full incident interrupted review-sheet writing. Q6 losslessly
compacted 2277 MiB across 1854 owned files: byte-identical repeats became hard
links; raw HDR/validity readbacks gained `.gz` suffixes, except one representative
city midnight raw pair retained. Primary BMPs, reports and recipes remain.
Uncompress with gzip when consuming the archived raw attachment. No other owner
files were removed. Review generation then succeeded.

Concrete remaining convergence items: Q0 is extending main/feature attributes
with authoritative world XYZ and a versioned shadow texture binding; existing
feature shadow polygons contain only screen XY and zero macro UV, so Q6 cannot
recover valid receivers without violating the projection contract. After that
interface lands, Q6 must bind its common shadow field, remove projected dark
layers, rerender real contexts and all applicable categories, and close water/
cutout/transparency ordering. Q0 owns those shared input/binding changes; Q6
continues independent work. Native parity initially encountered a stopped VM/path issue. After Q0 transport
recovery, a bounded render exposed Q6 selecting the wrong attribute semantic
layout on D3D: world coordinates were read as normals. The provider now selects
the feature layout. All eight Metal images are byte-identical after the fix;
`q7-city-nativefix-13/parity.json` passes all eight Metal/D3D pairs and native
repeats (maximum RGB MAE 0.000529 byte, p99 0). The native midnight image was
directly inspected and no longer has the incorrect broad receiver gradient.

Changed files are restricted to the five Q6 owned directories and its status
file. Source-normalized content remains local and generic at runtime. The scan of 138 publishable touched text files found no personal paths, user
identifiers or email addresses. Ignored local runtime reports are not
distributable source artifacts; repeat the scan when adding later files.

World-shadow shader staging: `scene_world_v1.hlsl` opts into Q0 world XYZ, main
TEXCOORD14 / feature TEXCOORD2, shared b1 (five float4 fields), main t25 / feature
t17 aliases. The disabled hook compiles and renders; it is not evidence that
the pending common field binding works. Q1 identified an additional shared box
reconstruction validity-masking defect in reduced controls; use Q1 contract2
reconstruction for final convergence after Q0 adopts its correction.

Final checkpoint: `INTERFACE_REQUEST_V1.json` pins the observed shared contracts
and exact missing interface. The ordinary complete shader including disabled
Q3/Q6 hooks renders byte-identically (`scene-hooks-off-15`) to base12. Native
parity and strict evidence gates pass. No remaining independent lighting edit
can replace the missing receiver data/binding; Q0 and the coordinator have the
request. No user action is required, and acceptance is explicitly false.

The interface-blocked checkpoint above is historical: Q0 then published wire5,
authoritative world attributes and the owned packet postprocessor. Q6 resumed,
implemented `scene_shadow.cpp`, removed all legacy projected shadow layers and
directly inspected `source-world-16`. The user's explicit shape requirement is
recorded in `ACTUAL_CASTER_SHAPE.md`; no generic shadow substitute is acceptable.
Current world-field control/matrix validation is still in progress.
