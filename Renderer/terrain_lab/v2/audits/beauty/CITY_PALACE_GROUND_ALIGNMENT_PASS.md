# Palace paving follows the source footprint

The user identified a visibly misaligned ground patch beneath the American
capital. The normalized palace already has a rotated foundation, but the authored
settlement underlay used its axis-aligned bounding rectangle. That rectangle
filled two large diagonal wedges outside the foundation. The source building's
texture coordinates were not the cause.

`settlement_ground_probe.py --capital-footprint source-hull` now derives the
capital underlay from the convex footprint of the normalized opaque body after
the same centering, rotation, uniform scale and world-axis conversion as the
building. The rectangle stays in use for conservative grid bounds; coverage uses
the hull. Other houses retain their existing footprint unions. The atlas, texel
density, world-stable UVs, margin and feather are preserved. This is an authored
ground rule using generic pack geometry, not recovered Civ VI generator behavior.
The legacy `bounds` mode remains available for matched replay controls.

The hull removes 45.1% of the palace's unexpanded bounding rectangle, preserving
the source orientation. In the reported inland view, 656 day / 639 night pixels
above 2/255 change around the paving. The diagonal wedges become grass and the
remaining apron follows the foundation. Coastal changes are 269 / 264 pixels;
the previously city-untuned freshcanopy site changes 605 / 589. No buildings,
source UVs, materials, local lights, shared shadows, cameras or terrain move.
Pixels outside the local comparison region are exact except one coastal daylight
channel rounding by 1/255. The checked coastal lake reflection region is exact
at both hours.

- [Gameplay-size inland comparison](out/city-palace-ground-alignment-r1/inland-native.png)
- [Two-times diagnostic](out/city-palace-ground-alignment-r1/diagnostic.png)
- [Previously untuned region](out/city-palace-ground-alignment-r1/holdout-native.png)
- [Coastal regression comparison](out/city-palace-ground-alignment-r1/native.png)

All six Windows day/night comparisons pass. Twelve independent packet checks
preserve original city data and bound metalness transport. Every new underlay
vertex retains the original position/normal/UV/world values; coverage only
decreases. Eight focused hull/coverage/clipping tests and Lab v2 validation pass.
[Executable evidence](CITY_PALACE_GROUND_ALIGNMENT_EVIDENCE.json) is regenerated
with `python3 Renderer/terrain_lab/v2/qa/city_palace_ground_evidence.py` using a
Python environment with Pillow and NumPy. The generated `settlement.json` files
record complete input paths, hashes, parameters and source-footprint geometry.

The provisional combined candidates are `environment/render`,
`inland-environment/render` and `holdout-environment/render` beneath
`out/city-palace-ground-alignment-r1`. Previous material candidates remain intact.
The hull approach is validated here for this palace at three sites; it is not
blanket visual acceptance for all 47 palace styles. Future concave courtyards may
need an explicit pack footprint rather than a convex enclosure. Connecting roads,
native changes and all manual/milestone gates remain deferred.

[Cleanup](CITY_PALACE_GROUND_ALIGNMENT_CLEANUP.json) removes only completed new
linear readbacks. Images, packets, shared textures and preparation-failure
diagnostics remain available.
