# City era-ground atlas binding r1

Status: provisional small material improvement. Full city quality, human review,
native delivery and all milestone gates remain open. One era per city is unchanged;
connecting roads remain deferred.

## Finding and visible result

The generator's `GroundingMaterials` collection selects the modern pavement
material or Classical stone material by art era. Many normalized compound pads
still reference the default ancient dirt atlas: nine of twelve modern components
and seven of twelve medieval-study components. The previous probe recovered
their exact triangles and UVs but did not apply an era grounding override.

The installed material records resolve to the **same compressed payloads already
in the normalized pack**, verified byte for byte. No textures or packs were copied.
The atlas sheets share the arrangement of their ground pieces. Replacing the
default dirt binding therefore supplies asphalt or stone without enlarging the
geometry or stretching its UVs. Existing era-specific and non-ground bindings
remain untouched.

This is a source-informed **binding hypothesis**. The source `HeightRange` of
10–100 and the exact engine normal/pillaged state application remain unproven.
The probe does not claim to reconstruct those selectors. The material's height
channel remains absent; the invalid optional specular slot is not repurposed.

At actual gameplay size the modern city gains dark paved edges beneath its front
blocks. The inland example is clearest; the medieval stone change is subtler.
These move toward the finished building bases visible in
`Renderer/canonical/nightlights.jpg`, but are much smaller than its connected
urban fabric. Source references with historical-era mixtures do not supersede
the user's single-era preference.

| Fixed case | Noon pixels changed >2/255 | Midnight pixels changed >2/255 |
| --- | ---: | ---: |
| Wilderness modern | 295 | 289 |
| Coastal medieval | 274 | 215 |
| Inland large modern | 682 | 630 |
| Freshshadow modern | 187 | 178 |
| Modern American capital | 544 | 516 |

The selected local candidate is `out/city-ground-binding-r1/`, composed directly
on facade-light r3. Earlier packets and images are preserved. Freshshadow uses
the same override without local tuning; its crowded skyline remains unaccepted.

## What stays matched

The packet adapter changes only the matching city-ground texture binding at
material 60. It preserves all geometry/constant buffers, source UVs, draw order,
blend/depth/caster state, terrain, building materials, palace placement and shadow
textures. Main, reflection and glow shader closures match the previous render
exactly, including the facade light records. The fixed wilderness shadow grid
survives. The 100-tile scenes, cameras and output sizes are unchanged.

`prepare_city_ground_probe.py --binding-override` also emits reusable normalized
ground parts with the same triangles/UVs and source descriptor hashes. This avoids
requiring a runtime packet patch for future Lab scene generation. The replay
adapter exists to isolate the pixels from layout and shadow rebuilding.

For eventual integration, compile the selected generic grounding material into
the city's era/style definition or normalized draw list. Do not ship this
fingerprint-replacement diagnostic as a source-specific runtime branch. Civ III
still supplies the era and authoritative capital state. No new native hooks or
CSV symbols are required by this offline material preparation.

Completed new HDR readback sidecars were removed after verification: 257.3 MiB
across 34 files. Original renders, review PNGs, shader closures and shared packets
remain available. See `CITY_GROUND_BINDING_CLEANUP.json`.

## Reproduction and engineering evidence

Verify the installed bindings and existing normalized payloads:

```sh
python3 Renderer/terrain_lab/v2/qa/prepare_city_ground_bindings.py \
  --output Renderer/terrain_lab/v2/fixtures/beauty/city-ground-binding-r1 --verify
```

For a new matched render, call `qa/city_ground_binding_probe.py` with
`--source-render`, `--mapping` and a fresh `--output` under `audits/beauty/out`.
The modern/medieval mapping files are in `fixtures/beauty/city-ground-binding-r1`.
Use `qa/city_d3d_probe.py` for its Windows check, then
`qa/city_ground_binding_evidence.py` with the offline Pillow/NumPy environment.
The latter checks frozen shader identity, localized pixel changes, source
fingerprints, disabled-image identity and backend evidence.

Fourteen Windows comparisons pass, including both zooms for medieval and capital
cases. The no-op atlas control matches the previous wilderness day/night images
and packets exactly. The capital lake reflection ROI stays pixel-identical.
Outside the local city region, medieval and freshshadow noon each have one pixel
with a 1/255 difference; all other frames match exactly there. These isolated
differences are not counted as visible improvement.

The initial no-op control unnecessarily rebound a byte-identical duplicate
texture. It differed by 1/255 in one noon pixel, while a direct original-packet
replay matched exactly. The adapter now retains the original resource ID when
the requested material is unchanged. `disabled-exact` is the valid control;
`disabled` preserves the earlier redundant-rebind diagnostic. The evidence checks
this distinction explicitly.

No injected
compilation or full Lab pass is implied by this standalone change; the previously
reported unrelated L19A source-hash gate is not resolved here.

## Next visible gaps

1. Most source pads remain occluded by the buildings. Broader coherent city ground
   needs an explicit settlement footprint and correct source texel density,
   constrained to dry land and existing vegetation clearance. Do not repeatedly
   scale whole atlas pieces or silently introduce connecting roads.
2. Several views still have crowded, repetitive tower silhouettes. The holdout
   remains the clearest failure, and eleven-body wilderness growth is unresolved.
3. Facades still lack the source's full environment specular/material richness.
   Local night spill remains an authored approximation with building-box blockers.

Broader culture/era/size/capital coverage is unfinished. The mapped American
palace is preserved; this pass does not establish the rest of the 47-root roster's
selection or resolve the Gran Colombian tree attachments.
