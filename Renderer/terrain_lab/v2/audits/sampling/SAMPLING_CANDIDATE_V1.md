# Q1 sampling candidate v1 — cross-owner acceptance blocked

The candidate choice is **8x anisotropy, bias 0, 4x MSAA, render scale 1,
linear area reduction, no sharpening**. The actual production policy requires
Q6 scene-linear premultiplied input. Legacy experiments remain comparison
evidence; they do not establish correct linear scene blending.

## Gameplay-scale evidence first

Before/after at reduced gameplay zoom, exact final pixels, identical placement:

![Source-linear mixed real terrain, four phases and unsharpened controls](out/Q1/Q1-sampling/c021linear-matrix/phase-comparison-z2.png)

![Source-linear held-out region, four phases and unsharpened controls](out/Q1/Q1-sampling/c022linear-holdout/phase-comparison-z2.png)

Actual `test.biq` SHA-256:
`a6a88d7fffcc567c3500bbd5aa947398dd48170d4f412aa1e518bb45ffe8453e`.
Dataset payload:
`4e5fa91a03586b9b36da1ec3d8ee8249d19e803fe710ecd6659a174697cee837`.
The named `mixed` crop is raw origin (14,42), 4x4, halo 2;
`mixed-holdout` is (18,46), 4x4, halo 2. Wrapping is X only.
Q0 registry, source terrain and source BIQ are read-only. The halo is pinned to
these witnesses; a larger neighbor-footprint requirement needs a new revision.
The holdout was selected before tuning. It contains dense vegetation and rivers,
but no water; mixed supplies coast/sea, hills, mountains, forest and jungle.

The city (era 3), one tank, roads and rails are separately hashed, legal-domain,
seed-731 **Lab augmentation**, not captured Civ III state. Source terrain hashes
are unchanged. `c013off-mixed` and `c014off-holdout` render augmentation-off
controls at both zooms with the same full source provider and map data.
`prepare_contexts.py` reproduces the exact two augmented fixture hashes used by
the phase matrices, plus off and existing attack-pose recipes.

Direct inspection: 4x MSAA smooths opaque silhouette stair steps, especially the
tank and city. It does not reconstruct absent normal/material structure or fix
cutout texture aliasing. Bright shores stay continuous; no new ringing or edge
halos are apparent in these unsharpened controls. Fine foliage and rail sleepers
remain fragile at reduced zoom. At night the objects remain legible, but the
legacy city emissive pattern is busy. The visibly broken tall city cluster,
terrain-adapted mountain faces, and routes hidden under holdout foliage prevent
calling this a finished beauty scene. They are preserved as cross-owner
regressions, not hidden by object scaling or sharpness changes.

## Actual baseline and one-variable tests

The original frozen pipeline uses anisotropy 8, bias 0, one sample, scale 1,
gamma-2.2 shader output, straight-alpha BGRA8 blending and byte-box reduction.
Native output is 768x512; reduced is 384x256. Both have 768x512 internal pixels.
The separate scale-2 control renders 1536x1024, four times baseline pixels at
either zoom. Existing reduced-zoom oversampling is not counted as this change.

`c006mixedab` is the cached 4x4 mixed diagnostic context: source ground/foliage,
modern city, tank, resource, routes/rails and water. It is synthetic terrain,
not the BIQ. Same scene, camera, phase, seed and source scales for every row;
one GPU/device batch, final sizes unchanged, deterministic repeats identical.
All selected settings are supported; capability reports expose sample counts
1/2/4. No unsupported count is silently substituted.

| Single change | Native RGB MAE vs baseline | Reduced RGB MAE | Observation |
|---|---:|---:|---|
| Anisotropy 16 | 0.0009 | 0.0009 | Negligible benefit over 8 |
| Bias -0.35 | 0.1241 | 0.0860 | Slight extra texture contrast |
| Bias -0.65 | 0.2335 | 0.1512 | More grain, no recovered object structure |
| Bias -1 | 0.3604 | 0.2289 | Stronger detail contrast, unjustified risk |
| MSAA 4 | 0.3329 | 0.2228 | Smoother geometric silhouettes |
| Scale 2 per axis | 0.7737 | 0.5013 | Largest cost; no basis to select by default |
| Gamma-2.2 linear box workaround | 0 | 0.3495 | Reduction corrected only after legacy blend |

MAE is in 8-bit RGB code values, not a perceptual-quality score; black background
dilutes full-frame means. Direct final-pixel crops decide visual interpretation.
`c002ab` and `c003post` retain historical crowded-gallery/crop diagnostics last,
not as the primary benchmark. The sharper Mitchell B=C=1/3 kernel and unsharp
0.3 controls did not provide a justified improvement. Unbounded unsharp creates
contrast overshoot in the portable step witness; the bounded control clamps
to local extrema and disables sharpening at alpha/viewport edges, but still
crisps existing noise. The selected policy keeps sharpening off.

Q2 r06 material detail was consumed read-only in `q2detail-ab` and
`q2off-control`: macro color stays similar; source detail adds restrained
small-scale structure. Negative bias strengthens grain. Q1 recommended bias 0,
anisotropy 8 and no sharpen to Q2. It remains a separate Q2 candidate input.

## Pan and existing animation

`c007pan` (synthetic) and `c019real-pan` (verified mixed real terrain) use offsets
(0,0), (.25,.25), (.5,.5), (1,1), (2,1), (4,2), (8,4), (16,8), (32,16), (64,32),
and (0,0), at both zooms. These cross tile-sized distances, include diagonal and
viewport-edge clipping, and hold animation/lighting fixed. Return to origin is
byte-identical. Aligned interior overlap is used; raw moving-frame differences
are not called shimmer. Integer registration is exact; fractional registration
is bilinear and necessarily includes low-pass/interpolation error.

| Real mixed pan | Native baseline / finalist | Reduced baseline / finalist |
|---|---:|---:|
| Registered MAE, (1,1) | 0.01446 / 0.01388 | 0.00207 / 0.00219 |
| Registered MAE, (.25,.25) | 1.24869 / 0.85024 | 1.38300 / 1.17274 |
| Registered MAE, (64,32) | 0.00001 / 0.00002 | 0.00001 / 0.00002 |

The fractional residual improves but is not proof of zero shimmer. Lossless
WebP comparisons were opened in the local browser and inspected during playback
at native/reduced final sizes, with phase/frame crops and registered metrics.
No trails or new border halos were observed; foliage remains finely textured
and existing route occlusion persists. This is a finite witness, not a claim
about every camera speed or final integrated scene.

`c015animation0` through `c018animation3` use the four existing normalized tank
attack poses, fixed anchor/time of day, both zooms, and all unsharpened controls.
The body/turret motion is visible; no Q1-induced scale or geometry change occurs.
Native crops and reduced playback are in `out/Q1/Q1-sampling/animation/`.
Hydrology stays static. Animation differences are not scored as pan shimmer.

## Texture and source fidelity audit

`c006mixedab/uploaded-textures.json` matches all 208 uploaded texture payloads
to normalized DDS data. Total supplied upload bytes are 229,602,816. Largest
dimension is 4096; six 4096 textures have eleven supplied levels through 4x4.
The loader uploads those supplied levels. The source compiler's CIVBIG extract
copies the declared payload without resizing; separate one-mip PNG conversion
is a preview. Neither fact proves every upstream original largest mip survived.

The hardware base-ground t0 query at (384,140) is about LOD 3.13 at both zooms,
because their internal raster is identical. `ground-lod01` colors this query;
it measures t0 and actual UVs, not every selected terrain material or feature.
`checker-hardware-lod02` confirms actual minification (~LOD 1.13) on the 4-repeat
diagnostic panel. Checker/displaced/16x-stretched negative controls are labeled
diagnostic proxies. No larger top-mip experiment is warranted by this evidence.

`source-mesh-metrics.json` audits 181 actual normalized assets: tank animation
parts, city components, resources and selected alternate-skin vegetation.
Their authored normals are unit length to float precision. Source UV Jacobian
distributions, density, degeneracies and hashes are recorded without changing
UVs. Source triangles above anisotropy 16: tank 9, city 20, resources 29,
vegetation 12. City has 277 geometric/UV degeneracies; resources 27. These are
authored baseline investigations, not permission to repair the source silently.
Runtime instance/source deltas and semantic normal/gloss bindings remain
required; screen-position-only output cannot prove preprojection fidelity.

## Linear policy and retained boundary

`systems/sampling/policy_v1.json` declares the order and candidate settings.
`reconstruction_v1.hlsl` is generic scene-linear premultiplied area reconstruction:
no exposure, tone map, transfer, negative lobes or sharpen. Contract-2 wrapper
uses t0 RGBA16F, u1 RGBA16F, t3 R8 validity, u4 R8 validity and b2 dimensions /
half-open output rectangle. Invalid hidden color is excluded; valid coverage is
averaged separately; premultiplied color is never multiplied by coverage again.

`linear01` ran Q0's actual multi-draw HDR/cutout/water/viewport fixture with 4x
MSAA and extra scale 2. Eight phase/zoom repeats and independent NumPy reference
passed, max final channel error 1. Raw prepost RGBA16F/validity captures prove
HDR above 1 and black/invalid pixels outside the map rectangle. Portable tests
also test hidden invalid color, half coverage, transfer, overshoot, mirrored
tangent handedness and nonuniform-transform rejection. Retained game UI is
outside this filter; actual live-game compositing remains Integration work.

## Cost and unresolved gates

Small mixed AB observed GPU times span roughly 6–31 ms under concurrent owners;
baseline native 19.48 ms versus MSAA 15.60 ms is noisy, not evidence MSAA is
faster. These times exclude uploads/readback. Sampled allocation high-water was
220.70 MiB baseline, 231.56 MiB MSAA, 230.05 MiB scale 2 in that batch. Later
mixed real runs include retained cached phase data and reach ~291 MiB. A sampled
API counter is not true driver transient peak; no such claim is made.

Q0/Q6 source-scene linear adoption has been exercised on actual source terrain.
D3D finalist parity is being coordinated with Q0.
Source alpha/channel completeness, per-instance transforms/tangents and source
UV exceptions require shared/Q7 material evidence. See `Q0_INTERFACE_REQUEST.md`.
The 192-tile/four-phase promotion suite and global closure verification are
pending those gates; none are waived. No v1 source, accepted handoff, Integration
gate or injected game code was edited. No injected compilation is applicable.

Exact recipes and commands: `REPRODUCE.md`. Images/reports/cache are local ignored
artifacts; source BIQ payload exports are ignored. Portable source and audit
records use repository-relative paths and contain no personal identifiers.

## Source-linear convergence checkpoint

Q6's generated `scene_linear_v1.hlsl`, Q0 `linear_adapter: 1`, declared
`shader_owner: Q6-lighting` and post contract 2 now feed the actual source-map
fixture. `c020linear-mixed` was inspected before broader runs. `c021linear-matrix`
and `c022linear-holdout` then completed all four phases, both zooms and identical
repeats. Their 32 custom-filter variants match the independent raw attachment
reference within one byte; no acceptance tolerance was widened. Direct reduced
phase sheets and native night/day views preserve the same known city/foliage
weaknesses. The source shader/depth classification change is intentional and
separate from one-variable sampling comparisons within each new run.

A concrete edge defect was exposed in Q0's built-in box control: it accumulated
color even for samples with zero R8 validity. The small mixed reduced control
has maximum error 18 versus the independently masked reference (full-frame
MAE 0.00104). Q1's reconstruction already excludes those samples and passes.
`linear-reference-metrics.json` records failing Q0 controls explicitly instead
of hiding them or raising the tolerance; Q0 has the exact reproducer.

`c023linear-pan` repeats the deterministic pan under the new input contract.
All 44 custom-filter pan comparisons pass the raw reference within one byte.
At (1,1), registered native MAE is 0.01129 baseline / 0.01131 finalist; reduced
0.00003 / 0.00010. At (.25,.25), native is 1.15586 / 0.91349 and reduced
1.48782 / 1.34321 (still includes bilinear registration error). Origin returns
are exact. The linear pan and source attack sequence were opened for direct
motion review; no new halo/trail is apparent. Fine foliage remains a finite
witness, not proof of global zero shimmer.

`c024linear-animation1`, `c026linear-animation0`, `c027linear-animation2` and
`c028linear-animation3` retain all existing source attack poses, fixed anchors
and both zooms; `linear-animation/` contains lossless films. `c025linear-off`
retains the augmentation-off control. The proposed filter is ready to consume
more complete source material bindings; it does not confer acceptance on the
legacy source alpha/normal/gloss or instance-transform path.


## Cross-owner evidence resolved and remaining

`c029linear-parity` passed actual-source Metal/D3D11 replay at both zooms with
samples 4, anisotropy 8, bias 0, scale 1 and Q1 post 2. Maximum channel delta is
1, p99 is 0 and silhouette IoU is 1. Native/reduced RGB MAE is 0.000001695 /
0.000003391; D3D repeats are exact. This is a bounded finalist parity witness,
not the pending 192-tile closure matrix. Q0's shared invalid-color reducer fix
was then verified in `c033validity-fixed`: the previously failing built-in
controls now also match the independent reference (zero failed controls).

The current Q2 source-linear material hook was consumed read-only in
`c030q2-linear-on` / `c031q2-linear-off`. On the real mixed context, native /
reduced detail deltas are RGB MAE 0.01424 / 0.01203, p99 1 and max 2. Direct
inspection sees stable macro color and no new ringing. Q2's separate Q4
geometry-normal blocker is not remedied or waived by this sampling evidence.

Q1 independently fitted normalized source vertices to Q7's published world
artifacts without reusing Q7 emitter functions. `all-pools-01-q7-world-metrics.json`
records 808 city part instances: all 748 full-rank parts have positive uniform
transforms, exact UVs and correctly rotated normals. Maximum singular-value
ratio is 1.00000000013. The 60 planar/degenerate parts cannot identify a complete
3D affine transform and are explicitly not called full-rank passes. An additional
ancient-earth witness passes all 44 parts. Published world data reconstructs the
actual serialized Q7 geometry with zero float32 error for both witnesses.

This reduces the transform uncertainty but exposes a concrete material blocker:
664 all-pool part instances declare `gloss`, `normal_0` and `normal_1`; 160 declare
ambient occlusion. Q7's current provider binds base color and emissive only.
All 808 audited city source materials declare opaque alpha; cutout completeness
for vegetation/other families remains separate. No secondary-normal semantic or
gloss transfer was guessed to manufacture sharpness. Full source/material
acceptance needs the actual declared channels and their confirmed interpretation.

The updated common contract also requires actual transformed/posed caster mesh
and authored cutout alpha, projected onto real receivers through Q6's shared
light/shadow path. Existing projected or generic legacy shadows cannot earn
acceptance. Q1 does not add a separate shadow implementation or claim success
by omitting shadows. These remaining ownership obligations are recorded in
`candidate_v1.json` and the executable acceptance gate.


## Final ownership boundary

The source metadata regression was isolated to a remapped runtime texture index
being used against the original bundle texture array. Q0 preserved the original
bundle-local source texture identity at load. `c034source-audit` then passed:
34 source mesh identities, 208 DDS identities/formats and 175 instance references
match independently parsed normalized packs at both zooms. Its rendered image
hashes are unchanged from the actual-source parity witness. All 175 instances
explicitly retain a nonuniform legacy vertical projection calibration, and the
source tangent stream is absent. Q1 does not hide that behind a uniform scalar.

Q7 directly confirmed that there is no alternative published runtime material
path: AO, normal_0, normal_1 and gloss remain unbound; secondary-normal/gloss
interpretation is unproven. The remaining blockers are therefore source material
and projection implementation owned across Q0/Q7, plus Q6's shared actual
caster/alpha/receiver shadows. They are not missing user permission or a reason
to alter someone else's renderer. No further sampling-only render or sharpen
adjustment can supply those missing channels or correct source-body geometry.

The owned status and candidate are marked blocked on these precise convergence
gates. `verify_evidence.py` passes all 536 core study records, repeats/retained
repeat hashes, fixture identity and pan returns; six portable tests pass. The
explicit `--accept` gate fails on the unclosed owner obligations. The 192-tile
suite and full project closure remain pending, not waived. All edits stayed in
Q1-owned paths and its status file; portable files were scanned with no personal
paths, emails or sensitive identifiers found.


Q7's final published source-instances v1 sidecars were also consumed. All 96
explicit row-major source-to-Q7-world matrices across ancient, modern and both
registered regions pass uniform-transform validation; their mesh/material/texture
identities match the normalized source files. The sidecars explicitly retain
unbound AO/normal/gloss channels and state that the pinned Q7 coordinate system
is not Q0's authoritative world lattice; real fixtures still add a separate
clip-depth grounding bias. `q7-sidecar-validation.json` preserves the exact pins
and exceptions. This resolves metadata availability, not the documented runtime
material/world integration blockers.
