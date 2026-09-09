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

Coastal cliffs already rotate directly into world XY, without the source-object
Y reflection. Their `CliffTransform` uses the same height metric and inverse
transpose; their generated faces and rock bodies receive the shared world pages.
Applying the object adapter again would reverse cliff normals. The corrected
source-origin meshes must accompany this placement code.

## Ownership and interactions

| Content | Casts | Receives | Update ownership |
| --- | --- | --- | --- |
| Terrain, relief, static vegetation, buildings, static resources, goody huts and barbarian camps | Actual opaque/cutout geometry into world pages | World shadows on their actual surfaces | Cached world geometry and paged invalidation |
| Animated resources | Posed silhouette onto the local ground plane | Static world field on the body | Existing bounded animation redraw |
| Units | Pose-local self occlusion and a local ground footprint | Own pose; surrounding world field is not currently supplied by the native sprite API | Native action cursor and bounded sprite cache |
| Retained native content, fog, labels and HUD | Native behavior | Native behavior | Civ III |

The unit boundary is deliberate and explicit: these sprites cannot yet receive
a mountain's shadow or project onto arbitrary neighboring geometry. Adding that
interaction requires captured receiver placement and environment revisions in
the unit cache, plus dirty-region proofs. The subsequent hut/camp integration adds viewer-conditioned site capture in
API 17 and static world geometry, using the existing map boundary. Natural
wonders, constructed wonders and Districts remain deferred.

Both dynamic ground paths use `shadow_policy.hlsl` for opacity, with the
captured environment's shadow strength directly; animated resources do not
apply the legacy feature shader's separate night-time minimum. Actual self/world visibility attenuates
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

The shared contract passed 5,764 phase/season samples and 214 affected portable
regressions. The exact high-memory deployment DLL passed native API/pixel-format
smoke, 28 D3D captures (16 mixed-object and 12 shoreline views), and all seven
behavior replays: normal/reduced scrolling, wrapping, terrain edits, resource
playback/removal, and day/night unit actions/compositing. The coastal edit rebuilt
127 tiles and reused 260 with exact warm/cold pixels. All 12 shoreline views are
pixel-identical across standard and high-memory cache configurations.

At the user's explicit production request, the combined DLL and corrected eight
cliff meshes were deployed together. Runtime hashes remained unchanged through
verification, and Windows confirmed the installed-game path exposes the tested
DLL. `lab/out/shadows/deployment/deployment.json` records the deployment and
rollback files; `verification.json` records captures and executed witnesses.
The corresponding source snapshot and build flags are retained under
`native/build/shadow-cliff-production/`. It preserves the selected 768 MiB GPU
cache tier and completed query/index optimizations. Subsequent ground-grid cache
experiments in the concurrent zoom task have their own verification cycle.

These are production-renderer diagnostics, not a live-game acceptance claim.
Fixed references remain unchanged. This renderer-only deployment requires no
new native patches; INSTALL and Civ III were not run.
