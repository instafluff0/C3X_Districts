# Volcanoes

Accepted 2026-10-07 (Lab), replacing the sixteen hashed forms. Ordinary Civ III
volcano terrain (`real_terrain_type == 10`) is a stamp of the shared mountain
relief (`lab/shared/natural/mountain_shape.h`), so it joins neighbouring
mountains and volcanoes through the same saddles and routes and resources sit
on the rendered surface.

- Shape: one clean stratovolcano cone (concave flanks, shallow crater, foot
  easing into the ground about 0.64 tiles out), rim about 84 (mountains 112):
  at 1x the summit sits about 39 px above the tile centre, just past the tile
  diamond's top corner (accepted follow-up: "shouldn't extend much beyond the
  height of the tile"),
  carrying 55% of Civ VI's terrain-element gullies about their radial mean and
  its authored footprint. Each tile turns and mirrors it; height varies ±4%.
- Material: Civ VI's ash colour at element scale over ground (not grey stone);
  the ash meets the ground through a rise blend like mountain rock.
- Activity: world-topology bit 24 (an active tile effect) lights a crusted
  crater pool and its inner wall; bit 27 (eruption, `V[2] == AE_Eruption`)
  brightens it and adds narrow lava flows down the authored channels. With
  custom rendering on, the injected spawn patch hides Civ III's own volcano
  smoke/lava animation; the effect keeps its state.
- The retired ground provider (`render_core/relief_query.h`) contributes
  nothing for volcanoes under separate natural relief; one slot family used to
  raise a second cone there.
- Smoke: an active volcano carries a plume in the city effect layer
  (`city_fidelity::volcano_plume`, kind 93): a narrow, tall camera-facing quad
  over the crater, drawn by `q8_effect_plume` (shared with chimney smoke) as a
  darker, slower ash column, larger and lit orange from the crater during an
  eruption. It animates through the city effect pass and needs the city pack's
  effect material. Snow-specific art and natural-wonder volcanoes remain
  deferred.

Study, tools and review sheets: [mountain study notes](../../../studies/mountains/README.md)
(`ranges.py --layout volcanoes`). Game pack: `Renderer/tools/overlay_volcano_shading.py`
copies the volcano shader regions into the Renderer64 runtime pack.

Portable checks:

```sh
python3 -m unittest Renderer.lab.test_natural Renderer.lab.test_volcano_fixture Renderer.native.source_fidelity.test_contract Renderer.native.test_volcano_effect_suppression Renderer.native.test_native_view_identity Renderer.native.test_city_site Renderer.native.test_world_object_identity
```

Older isolated studies remain in the [volcano study notes](../../../studies/volcanoes/README.md);
their images are exploration history.
