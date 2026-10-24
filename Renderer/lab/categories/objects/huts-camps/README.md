# Goody huts and barbarian camps

Neutral, static site bodies consume the shared world light and paged shadows.
Civ III supplies viewer-conditioned presence and camp tribe identity. Removal
invalidates both geometry and shadow pages; guards and resources remain separate.
The native m19 map boundary already owns this draw layer when custom rendering
is enabled. Config-off retains Civ III rendering.

Three normalized hut variants use a stable eight-bucket selection; primitive
camps remain the default in every era. Source body textures and UVs are retained.
Sizes (user choice 2026-10-07, gallery option B) live in
`Renderer/tools/asset_compiler/tile_site_looks.json`. Sites keep Civ VI's own
proportions: the old 2.6x vertical stretch turned squat huts into thin spires,
and mines never had it. Each hut piece is 1.7x where it stands; camp pieces are
1.8x, with the ring pulled in 15%.
Ground decals and optional animated attachments are excluded from this static
site pass. Terrain surface height supplies the object anchor.

The fixture contains both sites, including a camp sharing a resource tile,
at four hours and two zooms. Game capture uses the current viewer's visibility
accessors; no omniscient site bits enter the visible scene.

`integration huts-camps` runs the production site lifecycle witness: removal
must clear ownership and shadows, match a cold render exactly, and restoration
must reproduce the original pixels. The fixture also exercises hillside
grounding and resource coexistence.

The tested API 17 DLL is staged in `Renderer/bin` with the existing 768 MiB
cache configuration. Run `INSTALL.bat` to update the matching injected game
executable. Installation and live-game verification are left to the user.
Individual visual categories are `goody-huts` and `barbarian-camps`.

Ground states share the site pack (user choices 2026-10-07; see
`Renderer/docs/city_ruins_and_crater_art.md`). They follow Civ III's tile
state exactly; nothing native is hidden for them:
- Pollution of every source (cities, eruptions, meltdowns, nukes) has one
  ash-and-char look. Civ III stores no eruption marker.
- Blast craters are drawn over the ash.
- A razed city leaves a dark rubble field, in one of Civ III's three sizes.
- All three lie over a farm's crops and the routes, as Civ III draws them over
  irrigation and roads. The site layer follows farms.

`ground_state_asset_importer.py` imports the Civ VI sources into the ignored
`GroundStatesNormalized` pack. `build_site_runtime.py` (with
`ground_state_composer.py`) bakes each look into one draped decal per tile.
Craters and rubble carry baked sunlit relief, so they never turn.
`Renderer/lab/studies/ground_states/study.py` renders the 1498 AD save with a
candidate pack (`C3X_RENDERER_SITE_PACK`). Tests: `Renderer.native.test_ground_states`.

Civ III's tile buildings share the site pack too (user picks 2026-10-23; see
`Renderer/lab/studies/tile_buildings/README.md`):
- Fortresses are the Civ VI Fort's keep. A barricade replaces its fortress with
  the keep in a ring of the era's walls, towers at its corners. The runtime
  lays the ring round the dry ground holding the keep: along the tile's edge,
  and along a river's or the coast's bank where water cuts through. A road,
  railroad or bridge crosses it through the era's gate, or an opening between
  two towers in eras without one.
- Airfields are the early airstrip and the modern Aerodrome.
- Outposts are era watchtowers (Medieval: the Great Wall tower).
- The radar is Amundsen's mast; the victory point is a white obelisk.
- Colonies are kit A, placed at run time around the tile's resource, never on it.
Fortress, barricade and airfield follow the viewer's era, as Civ III draws
them; outposts, radar and colonies follow their owner's. They meet their
ground as cities, farms and mines do (`objects::place_tile_buildings`):
nothing stands in water or on its bank (the city sites' test: shore .05 tile,
river 10 px); a building moves to the nearest dry spot, shrinking if it must,
and on a mountain takes the rock's foot toward the camera. They stand on the
drawn rock, rigid pieces settled on the lowest ground of their footprint and
wall sections draped per vertex, as the city walls are. The forest keeps off
their exact footprints in the tile compile (`geometry.h`), keyed by the
canopy clearing's `buildings`. `sitepack.py` freezes
the baked picks into `TileBuildingsFrozen`; `build_site_runtime.py` merges them
(`tile_buildings_runtime.py`), and the hut and camp textures share their first
atlas. A site pack without compositions, or a renderer from before them,
leaves tile buildings undrawn, and the bridge then requires none. Tests:
`Renderer.native.test_tile_buildings`.
