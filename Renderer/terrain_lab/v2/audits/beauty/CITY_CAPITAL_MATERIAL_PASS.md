# American capital material restoration

The old r22 capital was still using the earlier vertex/material layout. r101
restores source normal/tangent frames, auxiliary AO, cooked gloss and opacity
while preserving its eight instances and all 13,097 city triangles. Source
geometry, winding, base/emission UVs, camera and shadow frame match independently.
The previous 31 local-light proxies and settlement ground wire also match exactly.

Source-only restoration changes 3,106 day / 3,353 night gameplay pixels above
2/255, concentrating on roofs, openings and facade response. The additional
authored environment reflection gives cooler modern facades and changes 7,797 /
5,199 pixels relative to the old capital. These are modest material improvements,
not a claim of complete source shading or Civ VI-level quality. Outside the city
comparison region, pixels are exact.

r102 applies the restored recipe to the inland capital with connected four/seven
house growth. r103 uses the previously city-untuned `freshcanopy` 100-tile region,
BIQ origin [76,58], anchor [5,2], selected from raw flat, nonriver terrain before
rendering. Both fit without per-region recipe tuning or vegetation movement and
pass independent river/forest clearance. The legacy r101 layout retains its
earlier clearance policy; it does not claim the newer envelope.

A city-reflection-off control keeps main rendering and packets fixed. Enabling
reflection changes 572 day / 665 night water pixels above 2/255, with night maximum
73/255. The earlier restricted lake witness remains within 1/255. This verifies
preservation and isolation, not a new reflection-strength improvement.

Ten Windows comparisons and 30 independent packet/frame checks pass. Evidence:
[material evidence](CITY_CAPITAL_MATERIAL_EVIDENCE.json). Recheck using
`python3 Renderer/terrain_lab/v2/qa/city_capital_material_evidence.py` with Pillow
and NumPy available. The American palace comes from the broader 47-root library;
source normals and opacity intake are under `fixtures/beauty/city-capital-materials-r1`.

The user subsequently identified the incorrect diagonal paving under this palace.
The material candidates remain valid, but their bounding-box paving is superseded
by [the footprint correction](CITY_PALACE_GROUND_ALIGNMENT_PASS.md). Keep the
prior Asian dielectric appearance; its environment trial remains unselected.
Broader single-era culture/era/size coverage, richer open-ground composition,
source environment/LEAN1 reconstruction and unresolved coastal/river growth
remain open. Native delivery and all manual/milestone gates are unchanged.

[Cleanup](CITY_CAPITAL_MATERIAL_CLEANUP.json) removes only completed new linear
readbacks, preserving images, packets, shared textures and previous bests.
