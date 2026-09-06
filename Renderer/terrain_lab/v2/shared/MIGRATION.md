# Frozen v1 migration adapter

`frozen_scene.cpp` is a versioned copy of the accepted v1 Lab construction path,
not an edit of `terrain_lab.cpp`. `frozen_sources.json` records source hashes.
The shared environment evaluator is copied without visual changes; the shader
copy under `shaders/common/frozen_l21.hlsl` is byte-identical to v1.

Mechanical differences in the scene copy:

- Platform headers/resource calls use a CPU-only recording adapter.
- Windows file-opening/scanning/path joining use portable equivalents.
- Shader creation records entry identities rather than compiling on the CPU
  construction path; bitmap writing emits a versioned render packet instead.
- Small decoded viewports are admitted, with a bounded optional viewport and
  a fitted orthographic camera. The corrected fit includes the half-tile left
  margin. Without that override, the original complete-scene camera is used.

The adapter preserves existing construction, material aliases, draw order,
normal/self shading, terrain and body cast shadows, and environment constants.
Its packet stores immutable content-addressed resource payloads and explicit input layouts and
submission/depth state. Metal and D3D11 replay that packet independently.

This isolates the portability refactor and permits objective parity before
visual changes. Independent C++ providers can now emit packets and compose with explicit dependency
ordering without editing this snapshot. It is not the final modular scene core: source identities,
preprojection transforms, independent system builders and a dependency-aware
semantic render graph are still required. The typed semantic contracts remain
provisional and must not be represented as fully consumed by this adapter.

The baseline guard records actual files at the start of Q0. Historical L21
hashes remain untouched and retain their documented pre-territory scope.
