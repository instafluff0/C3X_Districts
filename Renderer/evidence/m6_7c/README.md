# M6.7c Authored Relief Evidence

## Focused Automated Gate

- The asset-compiler tests pass for authored channel extraction, 2:1 LOD analysis, generic relief-set compilation, and terrain-pack material output.
- The native bridge contract passes, including per-tile fallback and relief-ownership assertions.
- The 32-bit native build and smoke render pass with the production DLL copied to `Renderer/bin/C3XRenderer.dll`.
- The stable `test.biq` replay renders 413 visible tiles at 128px and 1,543 at 64px with zero fallback.
- Two independent 128px renders are byte-identical (`D533A65087AD4849EFD87DDD0E003D156B00A2613660BDDCA858705CF1D79CC9`).
- Equivalent views at horizontal centers 0 and 100 differ by only 2,394 color bytes out of 3,686,400, with maximum channel delta 1 and mean absolute delta 0.000649. This is sub-quantization floating-point variation rather than a visible seam or variant change.

## Visual Fixtures

- `samples/scenes/m6_7c3_authored_hills.csv` isolates connected hills across desert, plains, grassland, and tundra bases.
- `samples/scenes/m6_7c4_authored_mountains.csv` separates desert and standard mountain groups and includes both isolated and connected cases.
- `preview/out/m6_7c5_authored_materials_128.bmp` demonstrates desert strata versus standard rock/snow.
- `preview/out/m6_7c6_deterministic_a.bmp` and `preview/out/m6_7c6_relief_64.bmp` are the final close/far `test.biq` relief views for this automated pass.

## Remaining Checkpoint

The code and standalone visual gates are complete. M6.7c6 remains active until one strategic in-game check confirms that the rebuilt production DLL loads, custom hills/mountains replace the native `0x4010` pass without duplication, and scrolling across a live relief range remains visually stable. No new patch symbol or `civ_prog_objects.csv` entry is required.
