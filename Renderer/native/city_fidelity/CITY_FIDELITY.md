# Current city integration

The runtime uses the Cities Lab's selected culture recipes, replacing the older
selected/generic growth and procedural wall implementation completely. Build with
`Renderer/tools/prepare_city_recipes.py`; `Renderer/renderer.py prepare` uses the
same builder. No layout solver or source-format loader runs in the game.

## Selection and ownership

- Five cultures, four eras, three population tiers, capital/noncapital and
  town walls/nonwalls, with three deterministic variants: **480 compositions**.
- The captured map seed and canonical tile coordinates choose variation.
  Ownership, iteration order and population changes do not change that seed.
- Native city state selects art again when population, culture, era, palace or
  walls change. Capture, founding and razing invalidate the affected scene.
- Selected layouts use the legal native city anchor, including coastal sites.
  Each building sits on the terrain at its own anchor. No old city body, paving,
  invented masonry foundation or procedural wall is retained as a fallback.
- City-covered resource meshes and resource animations are suppressed. Resource
  state remains native, allowing the resource to reappear after razing.

## Materials and shadows

The pack includes full-resolution base, normal, roughness, auxiliary AO, opacity
and emissive channels where present in source materials. The selected derivatives
originally omitted auxiliary texture coordinates in 56 mesh parts. The offline
`recover_city_auxiliary_uv.py` tool recovers them against exact geometry witnesses;
`CityRecipeAuxiliaryUV/uv.json` is a preserved normalized input. The compiler now
reports **zero missing-coordinate material gaps**. Geometry and layouts are
unchanged. Tangent frames are derived from the normalized mesh and UV0; they are
not claimed to be recovered source packed tangents.

All four city material passes use the same current scene shadow field as terrain,
including reflections and emission passes. Receiver bias follows the actual
shadow texel size. This fixes buildings casting shadows but failing to receive
neighboring-building shadows because they were using the older regional lookup.

Facade light proxies are derived offline from emissive windows. They are an
approximation, not decoded source-engine light bindings. A growable GPU buffer
holds the visible scene's complete lights and blockers; the old regional
1,024-light/256-blocker limit no longer makes detailed walled-city views fail.
A conservative XY grid selects lights per receiver in the existing Y-inverted,
Z-scaled light metric. Stable per-light blocker lists retain every possible slab
intersection, including neighboring cities. Both lists share the complete light
buffer; resource/index failure uses the original complete scan. Exact serialized
content, counts and ordering govern reuse; night/emission constants update
separately. No borrowed pointers survive upload. The same shader adapter covers
main and reflection receivers and the frozen Renderer64 control bundle.
Daylight skips uploading inactive facade lights. Existing material shading,
normal maps, HDR reconstruction and analytic environment response remain shared.

## Verification and limits

- `test_city_recipes.py`: every selection combination, deterministic variation,
  coastal anchors, wrapping, incomplete libraries and transactional decode.
- `test_pickup.py`: selected Lab transforms, preserved source hashes and closed
  runtime texture dependencies.
- `test_city_lighting.py`: 2,400 GPU lights/1,260 blockers, owner indices,
  daylight/reset/reuse, and all city shadow-pass connections.
- `test_light_spatial_index.py`: deterministic/random CPU candidate and illumination
  comparisons, native D3D full-scan equality, mutation/lifetime and shader routes.
- `test_city_auxiliary_uv.py`: unchanged clipped geometry and interpolated UVs.
- `capture_city_border_examples.py`: actual Renderer64 standalone day/night
  examples, with binary, shader, pack and fixture hashes beside each capture.

The current standalone evidence does not certify live-game FPS, every city site,
or the complete source game's lighting quality. Broad gameplay testing is deferred
at the user's request while Lab systems are integrated. Reference images were not
replaced. This pass adds no native patch-table symbols.

## Current staged evidence

The matched API-20 bridge/renderer/helper and complete pack are staged together;
the command-line installer completed successfully with Civ III closed. The startup
probe reports healthy. The final injected compile/injection smoke passes without
compiler warnings. `Renderer/native/build/cities-borders/staged.json` records the
binary, shader, pack and injected-source hashes.

The pack contains 188 models, 84 materials and 480 compositions; `city.bin` is
21,657,041 bytes. Full binary validation rejects 207 truncations transactionally.
The focused native city/capture/invalidation/lighting group passes ten tests, the
border GPU probes pass at 1/2/4 samples, pack/source parity passes, and 35 shared
city-layout/ground/facade/wall tests pass. Four daytime captures and the modern
Asian night capture complete with zero terrain fallback. Closer 512-pixel tile
views use the same staged renderer through the standalone diagnostic client.

No live gameplay or FPS result is claimed for this pass. Reference images remain
unchanged. Normal game testing can proceed after the requested Lab import rounds.
