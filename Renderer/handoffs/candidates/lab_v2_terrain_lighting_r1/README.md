# Lab v2 terrain and lighting — integration preparation

Prepared at the user's request on 2026-09-06. This is the pickup point for
implementing the retained Lab result in the actual C3X renderer.

**Candidate:** `shadow-receiver-r1`, built on `relief-size-r3` and
`coast-pass-rocks-r8`. **Status:** implementation preparation, not an approved
visual handoff. No native/injected code changes are made by this package.
Historical L9–L21 handoffs remain immutable. LQ0/LQ1/LQ2, D3D11 parity, explicit
visual approval and the deliberate Integration refresh gate remain pending.
The user's request to consolidate does not by itself claim those gates passed.

## Start here

1. Read [IMPLEMENTATION.md](IMPLEMENTATION.md) for the port map, coupled changes,
   exact coordinate/material contracts, cache implications and ordered checks.
2. Run the read-only verifier from the repository root:

   ```sh
   python3 Renderer/handoffs/candidates/lab_v2_terrain_lighting_r1/package.py verify --evidence
   ```

   `manifest.json` pins the actual Lab implementation, fixture inputs and all
   32 retained frame hashes. `source_snapshot.tar.gz` is a local recovery copy
   of the pinned implementation and fixture text; it contains no pack art or
   native implementation. Source drift must be reviewed, not silently repinned.
   This is not a self-contained redistributable renderer/asset package.

3. Inspect these native-size comparisons before editing the native renderer:
   - [Wilderness shadow correction](../../../terrain_lab/v2/audits/beauty/out/shadow-receiver-r1/review/wilderness-h12-z1-comparison.png)
   - [Mountain scale and source material coverage](../../../terrain_lab/v2/audits/beauty/out/relief-size-r3/review/inland-day-z1-comparison.png)
   - [Long coast and selected cliff materials](../../../terrain_lab/v2/audits/beauty/out/shadow-receiver-r1/review/longcoast-h12-z2-comparison.png)
   - [Current combined volcano — synthetic](../../../terrain_lab/v2/audits/beauty/out/shadow-receiver-r1/review/combinedvolcano-h12-z1-comparison.png)
   - [Fresh 100-tile holdout](../../../terrain_lab/v2/audits/beauty/out/shadow-receiver-r1/review/freshshadow-h12-z2-comparison.png)
4. Prepare an isolated native candidate and parity evidence through the existing
   Integration workflow. Preserve the ongoing production cache/worker work.
   A production refresh still needs the existing gates; this document is not
   permission to overwrite the current renderer with the Lab monolith.

## Retained scope

| System | Retained behavior | Essential companion changes |
| --- | --- | --- |
| Terrain | Continuous world material weights, selected skin, source hills, continuous normals | Correct coordinate conversion and common terrain queries |
| Coast | Articulated Civ III-compatible shoreline; rocky/sandy differentiation | Same contour for geometry, water, beach, cliff placement and material coverage |
| Coastal hills | Actual selected-source gray cliff bodies with a grassy shoulder | Full base/LEAN0/LEAN1/gloss bindings, uniform mesh transforms and UVs |
| Mountains | Source bodies at 1.30 scale with bounded neighboring foothills | XY and Z together, material coverage, shadow extent, vegetation grounding |
| Ordinary volcano | Source body at 1.60 scale; source-owner material coordinates across its skirt | UV/coverage/state float4 and all material channels; witness is synthetic |
| Water | Static source large/small normal response under shared sun/moon lighting | Complete source channels and scene-linear composition |
| Shadows | Actual source caster field with bounded texel-sized receiver offset | World positions, shared frame constants, alpha-cutout coverage, valid caster halo |

## What this evidence does and does not prove

The retained matrix covers seven unmodified real-map regions and one separate
synthetic volcano region at noon/midnight, both zooms and fixed cameras. The
three original coastal/inland/wilderness regions have 100 tiles each. All
current shadow comparisons have byte-identical geometry/material/placement/
shadow-field packets. Previous size/coast evidence separately records the
intentional geometry and material changes. These facts make comparisons valid;
they are not a visual-approval decision.

`test.biq` SHA-256 is
`a6a88d7fffcc567c3500bbd5aa947398dd48170d4f412aa1e518bb45ffe8453e`.
It contains **zero volcanoes**. `combinedvolcano` replaces exactly one mountain
in a separate inland fixture and must never be presented as unmodified BIQ
evidence. Cities, units and improvements were explicitly excluded, so their
clearance/occlusion is unproven. The 192-tile, four-phase convergence gate is
not satisfied by this 100-tile/two-phase matrix. Seasons and live scrolling are
not newly accepted by these images.

## Open defects and exclusions

- The analytic dune body is an inherited **unapproved diagnostic proxy**. Do
  not promote it as recovered source art. Cleaner shadows do not clear that gate.
- Volcano sides remain stretched. The existing BC5 height RG interpretation is
  wrong: red contains detailed height, green a smooth footprint-like field.
  The exact green semantics are unconfirmed. The first red-height normal test
  had insufficient visible benefit and is not selected.
- Mountain/volcano physical source reconstruction is unproven; the scales are
  provisional C3X presentation calibrations, not recovered Civ VI engine units.
- Shallows remain soft; some cliff/grass joins, pond rock rings and relief
  shoulder facets remain conspicuous. These need visual work, not hidden waivers.
- No water animation, volcano eruption/VFX/light, new object system, or M9–M11
  wonder/district implementation is included.
- No new `civ_prog_objects.csv` entry is required to prepare this handoff.
  No new hook is established as necessary for the proposed visual port.

Rejected experiments are indexed in
[SHADOW_RECEIVER_PASS.md](../../../terrain_lab/v2/audits/beauty/SHADOW_RECEIVER_PASS.md),
[RELIEF_SIZE_PASS.md](../../../terrain_lab/v2/audits/beauty/RELIEF_SIZE_PASS.md) and
[COAST_SOURCE_JOIN_PASS.md](../../../terrain_lab/v2/audits/beauty/COAST_SOURCE_JOIN_PASS.md).
Do not adopt expanded diagnostic shaders, caster-plane texture changes, shadow
disabling, buried cliff r6, volcano size r2's material cut, or the unselected
volcano normal reconstruction.

## Pickup prompt

> Implement a staged native terrain/lighting candidate from
> `Renderer/handoffs/candidates/lab_v2_terrain_lighting_r1/`. Verify its manifest
> first and inspect its retained images. Use IMPLEMENTATION.md to port the
> coupled terrain/shore/material/relief/lighting behavior into the existing
> off-screen C3X renderer while preserving current cache/worker improvements,
> authoritative anchors, native ownership and all milestone gates. Establish
> Metal-to-D3D11 and cold-to-cached parity before proposing a refresh. Do not
> silently promote diagnostic dunes, synthetic volcano evidence, rejected tests
> or new systems. Return concrete discrepancies to the Lab and record any truly
> necessary capture/patch dependency in the existing ledger.

This prompt describes the next implementation task; it does not dispatch it or
declare missing approvals. The current production snapshot is an advisory map,
not a source file to restore over later integration work.
