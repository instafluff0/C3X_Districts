# Q1 platform and material convergence — final disposition

Q1 sampling implementation is ready for convergence. Full acceptance is blocked
on the concrete remaining owner obligations below. No shared implementation,
other-owner source, v1 handoff, or Integration gate was edited by Q1.

## Delivered, consumed and independently verified

Q0 supplies cached sparse source replay, one-device batches, 8/16x anisotropy,
shader mip bias, supported MSAA 1/2/4, separate render scale, final-pixel camera
offsets, deterministic repeats, GPU timing and sampled allocation high-water.
The verified BIQ registry and owned deterministic augmentation validator work.
MSL 2.2 supports the actual hardware LOD diagnostic. All launch interfaces are
available; there is no generic platform-start blocker.

Q0/Q6 actual-source `linear_adapter: 1` feeds scene-linear premultiplied RGBA16F
plus R8 validity. Q1 `linear_reconstruct.hlsl` uses post contract 2 before Q6's
single exposure/tone/transfer. Both real regions, all phases/zooms, off controls,
pans and existing source animation have actual image evidence. Raw-reference
comparisons pass within one byte. Sharpen remains off.

Actual-source D3D finalist controls are verified in `c029linear-parity`: 4x MSAA,
anisotropy 8, bias 0, scale 1, Q1 post 2. Both zooms pass with max channel delta 1,
p99 0, silhouette IoU 1 and exact D3D repeats. Q0's post2 HDR fixture also passed
its broader parity matrix. The 192-tile closure gate remains separate.

Two specific platform defects were isolated and fixed by Q0: built-in reduction
included hidden invalid color; source provenance indexed the original texture
array using rebased runtime slots. `c033validity-fixed` passes the independent
masked reference for the previously failing controls. `c034source-audit` verifies
34 mesh identities, 208 DDS identities and 175 instance references against source
packs. Source-metadata fix image hashes equal the parity images at both zooms.

## Confirmed remaining acceptance blockers

1. **Q7 with Q0/Q1 material semantics: actual runtime channels.** Q7 explicitly
   confirmed there is no additional runtime path beyond `systems/objects/provider.cpp`
   and `shaders/objects/presentation.hlsl`: base color t124 and emissive t116,
   authored rotated normals, and unmodified UVs. AO, normal_0, normal_1 and gloss
   are unbound; tangents are absent. Across the independently audited all-pool
   source world data, 664 part instances declare both normal channels and gloss,
   and 160 declare AO. Q7 source-instances v1 sidecars explicitly confirm the
   unbound states and uncertain secondary-normal/gloss meaning. The request is
   a real generic channel binding/interpretation path, preserving declared
   transfer functions and source evidence, followed by Q1 validation. Do not
   guess BC4 contents, flip gloss transfer or replace absent detail with sharpen.
2. **Q7 with Q0: source-body projection/world convergence.** Q0 source metadata
   marks all 175 current mixed-scene instances with
   `legacy_vertical_calibration_is_uniform_world_transform: false`. It records
   150 projected Z pixels per source unit versus the separately scaled terrain
   authoring basis. The source's scalar uniform scale does not prove a uniform
   final world transform. Q7's published world pipeline independently passes
   748 full-rank city part fits, 44 ancient part fits, exact source UVs and zero
   float32 world-to-serialized error; 60 planar parts remain full-3D ambiguous.
   All 96 explicit final Q7 matrix sidecars pass the uniform validator, but Q7
   explicitly states that its pinned coordinates are not the authoritative Q0
   lattice and real fixtures add a separate clip-depth grounding bias. Adopt
   the agreed world/projection bridge and preserve source body aspect. Q1's
   sampling settings must not move geometry or independently rescale objects.
3. **Q6/Q0/Q7: actual caster geometry and alpha on actual receivers.** The updated
   common rule requires transformed/posed caster meshes plus authored cutout
   alpha. Legacy oval/ribbon/projected shadow substitutes cannot pass. Missing
   shadows are not a pass by omission. Q1 supplies source identity, coverage and
   animation witnesses; Q6 owns the common light/shadow implementation and Q0
   its shared world plumbing. Full family alpha/material completeness also
   remains unproven by opaque city coverage or the provisional mine alpha clip.

Exact Q7 inputs consumed: `audits/objects/metadata/` source-instances v1 for
ancient-earth-02, modern-id-02, registered-mixed-v2 and registered-mixed-holdout-v2.
Owned independent results: `all-pools-01-q7-world-metrics.json`,
`ancient-earth-02-q7-world-metrics.json`, `q7-sidecar-validation.json` and
`c034source-audit/verified-source-metadata.json` under the owned output namespace.
All source/channel hashes and ownership exceptions are preserved.

## Limits and resumption

Metal telemetry is sampled allocation high-water, not undocumented driver peak.
The uploaded normalized DDS audit matches 208/208 textures, largest 4096;
original upstream top-mip history is not fully proven. Ground t0 hardware LOD
~3.13 does not warrant a speculative larger-mip experiment or cover other slots.

Resume by consuming the owners' versioned material/world/caster candidates,
re-running the smallest affected witness, then the relevant real-region matrix
and coordinated 192-tile/full closure. No gate is waived. The candidate's
`--accept` verification deliberately fails until these obligations close.
No required user action, speculative patch symbol, or manual screenshot request.
