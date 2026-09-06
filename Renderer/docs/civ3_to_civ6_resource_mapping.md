# Civ III To Civ VI Map-Resource Mapping Seed

## Scope

`inventory/vanilla_conquests_to_civ6_resources.json` is an editable first-pass mapping for all 26 resources in the vanilla Conquests BIQ. It selects candidate Civ VI map art only. Civ III continues to own Civilopedia, city-screen, trade-network, advisor, diplomacy, notification, and every other non-map resource icon.

Runtime matching begins with authoritative Civ III BIQ resource identity, class, and the actual map `resources.pcx` icon index. The Civ VI target supplies presentation only; it never changes discovery, visibility, yields, trade, prerequisites, depletion, placement, or AI behavior.

Validate the seed and its exact agreement with the committed BIQ semantics:

```powershell
py Renderer\inventory\resource_mapping_inventory.py
```

Resolve every candidate against installed Base/DLC ArtDefs:

```powershell
py Renderer\inventory\resource_mapping_inventory.py `
  --civ6-assets-root "Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets" `
  --require-all-targets
```

## Initial Results

Twenty-one entries are direct same-resource mappings. Five require an explicit semantic translation:

| Civ III | Initial target | Confidence | Rationale |
|---|---|---:|---|
| Saltpeter | `RESOURCE_NITER` | high | Niter is Civ VI's saltpeter strategic resource. |
| Game | `RESOURCE_DEER` | high | Civ III's Game artwork/identity is deer. |
| Tropical Fruit | `RESOURCE_BANANAS` | high | Its internal Civ III ID is `GOOD_Bananas`. |
| Oasis | `FEATURE_OASIS` | high | Civ VI represents the same map object as a feature. |
| Rubber | `RESOURCE_COCOA` | low | Temporary tropical tree/plantation silhouette; Civ VI has no installed Rubber ArtDef. |

Gems maps directly to `RESOURCE_DIAMONDS` because the Civ III internal identity is `GOOD_Diamonds`; Jade and Silver remain optional alternatives. Rubber is the only deliberately weak stand-in and should be the first item reviewed or replaced with original/permissioned art.

All 26 selected targets resolve against the documented installed Civ VI Base/DLC ArtDefs. Resolution proves that a logical ArtDef entry exists, not that its cooked model/material payload has already been extracted or normalized.

## M7.2 Runtime Contract

- One visible resource instance is keyed by stable tile/resource identity and anchored to Civ III's captured resource draw position.
- Player-specific visibility and fog remain authoritative in Civ III; an undiscovered or hidden strategic resource is never revealed by the renderer.
- Terrain-specific Civ VI clutter variants may be normalized under one generic resource asset, selected using captured terrain and deterministic seeds.
- Animated herds, fish, whales, or similar ambient motion use stable IDs and absolute presentation time only while visible.
- Replacement suppresses only the native map resource body/shadow after the complete 3D instance is ready.
- Missing model, material, animation, visibility state, or renderer capability restores the native map resource exactly once.
- Loading or mapping map art must not replace or mutate `resources.pcx` globally, because native non-map icons remain required.

Scenario resources are handled later by the same generic rules: inventory the loaded BIQ, use scenario/user renderer definitions to bind its IDs, and fall back per resource when no mapping exists.

No new Civ III function or `civ_prog_objects.csv` entry is currently required. M7.2 begins with the existing `Map_Renderer_m09_Draw_Tile_Resources` and `Sprite_draw_on_map` patch capabilities plus captured BIQ resource metadata.
