# M6.2 Evidence — First Native Real Terrain Art

The `m6_2_native_terrain_art` gate builds the 32-bit D3D11 DLL, generates a portable synthetic normalized pack, samples its BC3 grassland texture, runs the native source contract, and recompiles injected code. It rejects a missing pack and exposes `textured_tile_count` so a successful colored placeholder cannot satisfy the test accidentally.

The `m6_2_local_grassland_art` gate loads the ignored locally built `Renderer/packs/GrasslandNormalized` pack. Its output must be deterministic and have a different pixel hash from the synthetic texture. This proves the native path consumes the actual locally extracted grassland DDS without committing or redistributing that payload.

No new Civ III function or address is required. The only injected change resolves the versioned pack-configuration export and passes the local normalized-pack directory before rendering.

