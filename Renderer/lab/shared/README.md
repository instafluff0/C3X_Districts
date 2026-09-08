# Shared visual implementation

Category directories are entry points, not separate renderer forks.

`Renderer/lab/dependencies.py` maps visual inputs to current fixture consumers.
It uses observed asset-builder inputs as well as shared source locations. A new
unclassified input is treated as global until its narrower scope is established.
Keep this mapping aligned with fixture content when adding categories or objects;
do not classify a global implementation by whichever category first uses it.

`shaders/` contains the 24 source modules currently needed by the production
natural, water, unit and city shader adapters. Their relative include closure
is intact. `hydrology/` holds the portable source corridor implementation and
one small numerical comparison fixture. `cities/` holds current source intake,
layout, ground coverage and material/light derivation.

`natural/data.h` is the current production natural-pack decoder, height sampling
and terrain/mountain/object lighting-frame calculation. The D3D runtime inherits
this CPU data and retains responsibility for GPU resource creation and cleanup;
Mac tooling can use the same decoder with its own texture-upload callback.
Decode into fresh or cleared data, and discard partially loaded data on failure.
The shared module does not own graphics resources or change the pack format.

`natural/vertex.h` owns the production 168-byte map vertex layout.
`natural/ground.h` owns its pixel-defined projection, material/normal surface
sampling and 16x16 terrain grid. The native compiler calls these same functions;
they also compile and execute on macOS without Windows or a GPU.
`natural/queries.h` supplies production neighborhood lookup, coastal sampling,
material weights and combined natural/relief height. Each owner tile gets a fresh
query scope over caller-owned world data and reusable coast scratch storage.
Native callers retain the world/coast dependency observations used to invalidate
cached meshes.

`natural/world.h` owns the production 16-page river cache on top of the natural
CPU data, including revision invalidation and native wrapped topology.
`natural/relief.h` owns CPU relief fields, normalized source sampling, and the
flat-ground/ground-cache/height-only query policy. D3D terrain textures inherit
the shared fields; they do not make those queries GPU-specific. Callers still
provide asset loading and river/dune/activity callbacks and own output layers.
This is shared geometry and query code, not yet a complete Metal scene.

Native binding adapters remain under `Renderer/native/`. The current numerical
relief implementation is maintained in `native/source_fidelity/kernels.h`;
asset rebuilding no longer extracts C++ from a historical Lab provider.

The category workflow automatically refreshes production shader bindings after
shared-source edits. To run preparation alone:

```sh
python3 Renderer/renderer.py prepare
```

This runs the existing adapters in dependency order in a disposable source
mirror, validates their results and publishes only changed generated files.
It skips unchanged work and rebuilds natural/hill/cliff/city/unit/resource assets only when their
source bytes or builder change. Edited generated files are preserved as conflicts;
make intended changes in shared source or the builder/adapter. Comparison and
approval require fresh shader and asset preparation.

The unit shader is embedded C++; category renders automatically select a candidate
build after its generated code changes. Resource clip-unit calibration lives in `resources/clip_units.json`;
it contains only the current normalized clip paths and translation scales.
Ordinary category commands are documented in `Renderer/lab/README.md`.
