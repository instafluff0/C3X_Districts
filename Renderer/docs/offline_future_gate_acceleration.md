# Offline future-gate acceleration

Status: reusable source-independent preparation is complete; no Lab promotion,
Integration ownership transfer, or injected-code activation is implied.

The later cross-cutting pass adds deterministic generic effect graphs, automatic
eight-facing/two-zoom tile fitting, complete category-state provenance, a
content-addressed reference loader ABI, visual-QA metrics, and corrected strict
industrial-camp conversion. See `offline_crosscutting_preparation.md`.

## What future Labs receive

`tools/lab_promotion_kit.py` reads the checked
`tools/lab_promotion_profiles.json` catalog and emits a deterministic scaffold
for L16 through L20. Every kit has the canonical 16x12 / 192-tile layout,
normal and reduced zoom, category selector/state coverage, complete/category-
only/without-category/thumbnail variants, required evidence names, and a draft
handoff. Drafts contain no hashes or approval and cannot be mistaken for a
promotion.

The command is:

```sh
python3 Renderer/tools/lab_promotion_kit.py --gate L16
```

The owning Lab still supplies actual scene construction, rendering, visual
metrics, critical inspection, hashes, limitations, and approval.

## What future Integrations receive

`integration/category_contracts.json` freezes generic scene fields, cache-key
inputs, invalidation triggers, retained layers, and atomic ownership policy for
resources, cities, mines, farms, tile objects, infrastructure, and units.
`tools/integration_contract_compiler.py` proves every declared invalidation
field changes its deterministic cache identity and emits inactive fixtures.
Custom-on category failure is a visible hard failure without native replay;
custom-off leaves the whole category native-owned.

These are design/compiler contracts only. They do not modify capture structs,
native suppression, C3X state, or patch dependencies.

## Content-addressed packs

`tools/asset_compiler/pack_content_index.py` emits a deterministic object blob
and path index beside any normalized pack. Identical bytes are stored once,
while logical paths and per-top-level resident sets remain available. Current
offline results save 10,632 bytes in `ResourceAnimatedLab`, 213,078 bytes in
`TileObjectsNormalized`, 2,484,624 bytes in `UnitFamilyLab`, and 257,996 bytes
in `CompoundUnitLab`. The source-independent reference loader ABI is now frozen
and executable in `pack_loader_abi.py`; native activation remains deferred.

## Arbitrary-scenario proof

`samples/contracts/arbitrary_scenario_stress.json` uses deliberately unknown
resource, city-style, mine, farm, tile-object, infrastructure, unit, formation,
and owner-palette identifiers. Its executable proof covers custom resolution,
hidden omission, an enabled missing mapping that hard-fails without native
replay, a disabled category that stays native, owner palette row 31, and a
one-rule partial override. No source-name switch is involved.

```sh
python3 Renderer/tools/arbitrary_scenario_stress.py
```

## Source blockers and new candidates

The ignored `FutureGateCandidates` pack now contains four persistent crater
decals, two culture-specific palace compounds, and one observatory-derived
radar candidate. The palaces include normalized emissives and exact fire/smoke
socket transforms. The radar body's isolated calibrated sheet reads as a flat
plaza rather than a tower, so it remains unapproved. Pollution now has an
approved source direction and a bounded generic seven-particle tile-local
profile backed by five normalized radiation textures. Authoritative Civ III
pollution state—not source disaster timing—owns its lifetime. Victory Location
is explicitly set aside. The former industrial camp rejects were stale: the
corrected matrix order now passes all three roots strictly and the main pack
includes them without a static bake.

`attachment_identity_compiler.py` consolidates exact attachment names, model
sockets, transforms, semantics, and state hints across city, improvement,
tile-object, and candidate builds. The current catalog contains 2,620 identities,
including 88 VFX candidates and six analytic-light candidates. This proves the
join points but not the original resource scripts. Generic authored effect
behavior is now executable, while exact source-script equivalence remains
unclaimed.
