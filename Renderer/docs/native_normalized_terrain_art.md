# Native Normalized Terrain Art

M6.2 is the first native real-art increment. `Renderer/native/c3x_renderer.cpp` now consumes the same normalized grassland pack proven by the M1/M4 offline tools instead of coloring every grassland tile with a flat placeholder.

The DLL accepts a pack root through the versioned `c3x_renderer_set_pack_path` ABI. It validates the generic `c3x.asset_pack.v0` manifest, the `c3x.normalized_mesh.v0` terrain mesh contract, the `c3x.material.v0` material, and a bounded BC3 DDS payload. The DDS mip chain becomes a D3D11 shader resource and is sampled for terrain type 2 (grassland). Sampling decodes the sRGB texture to linear space, applies hour/season/variant multipliers, and encodes the result for the BGRA readback target; omitting that final conversion produced the excessively dark first in-game image. Other terrain types deliberately retain the existing placeholder rendering until M6.3 supplies their normalized assets.

The M6.2 compatibility entry point still accepts `Renderer\packs\GrasslandNormalized`. M6.3 uses the layered definition entry point documented in `definition_driven_terrain.md`; a missing or malformed individual asset now leaves only that tile transparent so Civ III's own terrain remains visible.

The native smoke creates a redistributable synthetic four-by-four BC3 pack and proves manifest/mesh/material validation, shader-resource creation, texture sampling, deterministic output, ABI behavior, resize/reset, export, scheduling, and blit. The local gate repeats the same executable using the ignored licensed-source `GrasslandNormalized` pack and requires its native pixel hash to differ from the synthetic pack. No converted texture is added to version control.
