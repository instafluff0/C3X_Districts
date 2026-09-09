# Shared lighting and shadows

One captured environment supplies the stylized directional light for every
current production material and cast shadow. `native/scene_lighting.h` owns the
world basis, fixed projection slope, object height conversion and normal
conversion. The fixed slope is intentional: time rotates shadows while object
height sets their length. Source sun/moon metadata is evidence for the inputs;
this projection and combined light are C3X presentation policy.

The user-supplied four-panel Civ VI comparison sets the screen-space target:
noon casts west (left), 18:00 south (down), midnight east (right), and 06:00
north (up). A continuous 24-hour orbit connects these directions. The world
azimuth compensates for the 2:1 isometric projection; simply rotating equal
world XY components would not rotate uniformly on screen. Day/night color is
intensity-weighted across the sun/moon handover, without a dominant-light switch.

World `(x,y,z)` projects to `((x+y)*half_width, (x-y)*half_height-z*height_pixels)`.
Source objects use `(x,-y,z*object_height_to_world)` in that world. Normals use
the inverse transpose of the same transform, after pose skinning and rotation.
Geometry, face lighting, specular highlights and shadow projection must agree
in screen space; passing identical numbers between different bases is incorrect.

## Ownership and interactions

| Content | Casts | Receives | Update ownership |
| --- | --- | --- | --- |
| Terrain, relief, static vegetation, buildings, static resources | Actual opaque/cutout geometry into world pages | World shadows on their actual surfaces | Cached world geometry and paged invalidation |
| Animated resources | Posed silhouette onto the local ground plane | Static world field on the body | Existing bounded animation redraw |
| Units | Pose-local self occlusion and a local ground footprint | Own pose; surrounding world field is not currently supplied by the native sprite API | Native action cursor and bounded sprite cache |
| Retained native content, fog, labels and HUD | Native behavior | Native behavior | Civ III |

The unit boundary is deliberate and explicit: these sprites cannot yet receive
a mountain's shadow or project onto arbitrary neighboring geometry. Adding that
interaction requires captured receiver placement and environment revisions in
the unit cache, plus dirty-region proofs. This refactor does not expand live
ownership or change the native API. Natural wonders, constructed wonders and
Districts remain deferred.

Both dynamic ground paths use `shadow_policy.hlsl` for opacity, with the
captured environment's shadow strength. Actual self/world visibility attenuates
direct illumination and leaves the material's ambient contribution intact.
The shared paged receiver owns cross-page sampling, receiver-plane depth bias,
3-by-3 filtering and tight contact. Local unit filtering remains limited to its
128-square pose raster and native sprite rectangle; it is not a world receiver.

## Verification

- Compare actual world and object projections, normal transforms and shadow
  lengths across the day, seasons, rotations and zooms.
- Exercise a mixed production fixture with relief, foliage, city, static and
  animated resources and units; show both detail and gameplay context.
- Preserve source cutout coverage, shared receiver filtering, bounded caching,
  scrolling, wrapping, animation, native compositing and fallback tests.
- Present candidate comparisons for visual review. Reference replacement,
  staging and live-game acceptance remain separate from automated verification.

The shared contract passed 5,764 phase/season samples and 209 combined portable
regression tests. Sixteen production D3D captures cover the mixed fixture at
four hours, two zooms and two scene contexts. Scrolling at both zooms, wrapping,
resource playback/removal, and day/night unit action/compositing witnesses passed.
The terrain-edit witness also passed with an established coast (127 tiles rebuilt,
260 reused, exact warm/cold pixels). An all-land world creating its first coast
legitimately invalidates all nearest-coast certificates, so that scene cannot
assert partial mesh reuse.

All seven native behavior replays passed in one combined run. Its final freshness
check rejected current-checkout certification because concurrent terrain/geometry
edits changed compiled inputs during verification. The successful candidate
snapshot is recorded under `lab/out/shadows/verification-snapshot.json`; a final
current-checkout run remains pending until the shared sources are stable.
Candidate captures are not visual acceptance or live-game evidence; no reference
replacement or staging is authorized by them.
