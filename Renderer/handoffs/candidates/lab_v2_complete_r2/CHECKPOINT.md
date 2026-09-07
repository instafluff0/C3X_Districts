# Consolidation checkpoint and acceptance ledger

The deliverable is an implementation-preparation catalog with executable pin,
replay and shadow-coordinate checks. It does not claim a new combined visual
improvement or Civ VI-equivalent quality. Images remain the acceptance evidence;
test counts and hashes support reproducibility only.

## Verification at preparation

`python3 Renderer/tools/renderer_dev.py full --report Renderer/verification/consolidation_full.json`
reproduces the known L19A failure after 379 prerequisite unit tests pass. The
failure is in `terrain_lab/test_l19a_tile_object_contract.py` against
`packs/TileObjectsNormalized/tile_object_runtime.bin`:

```text
Frozen expected: 16e1acdb3835b25cd929ad51221e08ce8b303c0caff4050f98aded15bcb41ec3
Current runtime: 1bf64e5b3128d0deee1e74f0e28cd8e954f9e42961cb3f5bd0886c2c221e8238
```

The runtime's new content must be audited against the frozen L19A contract and
the later separate camp preparation. Rebuild the exact approved legacy subset
or prepare an explicitly versioned reviewed successor; do not rewrite the
expected hash or expand goody-hut approval to camps as a shortcut. No failing
threshold or historical handoff was changed. Full stops in prerequisites, so
this run does not supply new native/injected verification. The older frozen
profile incremental boundary parity failure also remains open per integration's
record, separately from passing pickup checks.

Current native evidence is `Renderer/verification/animation/candidate-checkpoint.json`
and the newest sections of `Renderer/docs/animation_integration_checkpoint.md`.
Integration reports 576 movement and 1,128 action draws, 951 unit pose samples,
104 resource pose comparisons, translated-cache and color-key/16-bit checks.
Those establish the current supported animation implementation; they do not
establish posed resource casting or common direction. The user confirms unit
shadows and clicks work; latest civilian/fortify/dirty-bound changes still await
the existing user-run checkpoint. No new human approval is invented here.

The new coordinate probe checks the actual Lab frame against the candidate
adapter and exposes current native unit direction drift across four phases and
two zooms. `package.py verify` checks selected source and optional frame/asset/
packet pins. It intentionally reports native drift separately so ongoing native
improvements are not overwritten. A verified current asset pin does not erase
the conflicting historical L19A expected hash.

## Existing selected image evidence

The manifest's 27 cases are explicit branches, not a claim of uniform coverage:
eight terrain references (one synthetic volcano), four river/canopy scenes,
five water scenes and ten conditional city cases. Main coastal, inland developed
and wilderness inputs remain fixed. Freshcanopy/freshwater and earlier terrain
holdouts are retained regression witnesses; once seen, they are no longer
untuned for a future acceptance pass.

Relevant visual summaries, without duplicating their image payloads:

- [Terrain receiver correction](../../../terrain_lab/v2/audits/beauty/SHADOW_RECEIVER_PASS.md)
- [River bank/source-rock placement](../../../terrain_lab/v2/audits/beauty/RIVER_BANK_ROCK_PASS_r3.md)
- [Natural water and reflected objects](../../../terrain_lab/v2/audits/beauty/WATER_OBJECT_REFLECTIONS.md)
- [Central, grid-aligned capital at gameplay size](../../../terrain_lab/v2/audits/beauty/out/city-central-capital-r2/inland-native.png)
- [Capital in the previously city-untuned region](../../../terrain_lab/v2/audits/beauty/out/city-central-capital-r2/holdout-native.png)
- [Paving receiving facade light](../../../terrain_lab/v2/audits/beauty/CITY_PALACE_FACADE_ALIGNMENT_PASS.md)
- [Historical complete object scene and border limitations](../../../terrain_lab/L21_COMPLETE_BEAUTY_AUDIT.md)

The central capital evidence records four Windows frames, twenty independent
composition checks and 33 focused tests. Selected water has sixteen focused
Metal/D3D comparisons, exact disabled controls and offscreen-object crop probes.
Reuse valid evidence, but rerender when the native composition or bindings
change. Historical L21 final-border hashes are not available; the retained
hashes precede that change and cannot substitute for a new border comparison.

## Required checkpoints after integration work

First reconcile the shared frame and posed shadow blockers S1–S4, then compose
the natural scene and conditional city pipeline using the implementation order.
Legacy object systems may be prepared in their own authorized paired gates;
per-system pickup does not require waiting for every other category to finish.

For each changed system, render the fixed coastal/inland/wilderness scenes at
the exact source/camera/output settings recorded in the manifest, noon and
midnight at both gameplay zooms. Add 06:00/18:00 for shared-light changes. Include
one newly selected, previously untuned test.biq region without changing the
existing reference regions. Use original placement and canonical Civ VI reference
images alongside the previous selected render; name the changed pixels and
what they improve. Keep closeups diagnostic and judge at gameplay size.

Before the meaningful combined visual checkpoint, include goody huts, colonies,
all raised/flat infrastructure, borders, animated resources and multiple units
in the same scene, with visibility/removal/ownership/overlap and shadow controls.
Keep the existing 192-tile combined and LQ convergence requirements. Check
translated/cold parity, wrap, clipped dirty regions, reflection caster halo,
input response, memory and cache behavior. None of those engineering checks
alone supplies visual approval.

Open visual blockers: coastal central-palace fit; latest central/grid preference
across other styles/sizes; incomplete ground-layer grit and source height/normal
interpretation; bank/pool fidelity; wall/new city envelope composition; animated
shadow coverage; exact source environment/variance; unapproved dunes and source-only
camp/effect coverage. Keep M9 natural wonders, M10 constructed wonders and M11
Districts deferred. No new road work, patch CSV changes, automatic installation,
game launch or milestone advancement is part of this preparation.

Human review should occur after a meaningful combined rendered checkpoint,
using one batched checklist and the pending integration checkpoint where useful.
Continue independent implementation while that review is pending. Never label
an unreviewed fallback, source-only asset library or passing shader test as a
new best-of-scene acceptance.
