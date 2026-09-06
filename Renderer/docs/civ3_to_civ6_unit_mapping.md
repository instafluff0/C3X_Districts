# Civ III To Civ VI Unit Mapping Seed

## Purpose

`inventory/vanilla_conquests_to_civ6_units.json` is the first editable mapping from the 93 standard Conquests unit records to Civ VI ArtDef unit identifiers. It is a curation seed, not a claim that every choice is final and not yet proof of BIQ reachability. Scenario configuration will be able to replace any target without modifying importer or renderer code.

The mapping selects art only. Civ III remains authoritative for unit identity, ownership, combat statistics, actions, timing, direction, and gameplay. During M7.4 the selected Civ VI asset replaces only the animated unit body; Civ III keeps the selection cursor/ring, health bar, activity/status marks, stack indicators, and related HUD.

## Current Coverage

- 93 Civ III records.
- 48 direct matches.
- 28 close equivalents.
- 15 temporary stand-ins.
- 2 units intentionally deferred to the effects pipeline.
- 62 high-confidence, 19 medium-confidence, 10 low-confidence, and 2 no-model mappings.
- 91 chosen targets resolve against the current local Civ VI Base/DLC ArtDefs; none are accidentally unavailable.

The two no-model records are Tactical Nuke and ICBM. Civ VI treats their visible delivery as operations and VFX rather than ordinary unit bodies, so they retain vanilla fallback until M7.5 can map launch, projectile, and impact assets correctly.

## First Curation Queue

These low-confidence choices deserve the first visual review after unit extraction works:

| Civ III unit | Initial target | Why provisional |
| --- | --- | --- |
| Enkidu Warrior | `UNIT_WARRIOR` | No Civ VI Enkidu model. |
| Rider | `UNIT_MONGOLIAN_KESHIG` | Similar regional mounted silhouette, wrong civilization and role details. |
| Marine | `UNIT_SPEC_OPS` | No normal Civ VI Marine model. |
| WWII Paratrooper | `UNIT_SPEC_OPS` | Correct broad role, wrong period styling. |
| Cruise Missile | `UNIT_ROCKET_ARTILLERY` | Launcher stand-in; should become an M7.5 projectile/effect mapping. |
| Transport | `UNIT_MODERNEMBARK` | Civ VI has embarked formations instead of transport units. |
| F-15 | `UNIT_JET_FIGHTER` | Correct era and class, not an F-15-specific model. |
| Stealth Fighter | `UNIT_JET_FIGHTER` | No dedicated stealth-fighter model. |
| Army | composite: active member + `UNIT_GREAT_GENERAL` / `UNIT_GREAT_GENERAL_MODERN` commander | Uses Civ III's authoritative displayed member beside dedicated era-appropriate Great General art; see `army_rendering_strategy.md`. |
| Sipahi | `UNIT_CAVALRY` | Correct era and role, no Ottoman Sipahi model. |
| Tactical Nuke | vanilla fallback | Requires WMD operation/VFX extraction. |
| ICBM | vanilla fallback | Requires WMD operation/VFX extraction. |

## Validation

Validate the checked-in schema without a Civ VI installation:

```powershell
py Renderer\inventory\unit_mapping_inventory.py
```

Resolve every non-deferred target against a local Civ VI asset tree:

```powershell
py Renderer\inventory\unit_mapping_inventory.py `
  --civ6-assets-root "Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets" `
  --require-all-targets
```

`--json-report <path>` optionally emits the exact relative ArtDef file or files that define each selected target. Machine-specific source paths are never stored in the seed map.

## Next Integration

M6.0 must correlate these `PRTO_*` records with actual standard `conquests.biq` unit IDs and effective INI/FLC bindings. The Civ VI adapter can then resolve each `civ6_artdef` through the relevant unit member, model, material, animation, and cooked package records. The source-agnostic pack should retain the Civ III semantic key and contain normalized model/animation assets; runtime code must not branch on Civ VI names.
