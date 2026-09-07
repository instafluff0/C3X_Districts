# Native pickup refresh — work in progress

The user explicitly requested this port on 2026-09-06, superseding the earlier
production pause awaiting logs. Do not launch Civ III. The final installed test
build must preserve production's bounded mesh, viewport and pixel-block caches,
immutable worker snapshots, foreground priority and native ownership boundary.

Source: [C3X implementation pickup package](../../handoffs/candidates/lab_v2_terrain_lighting_r1/README.md).
`reference/` is the required shader/query closure extracted from that package's
immutable archive. `provenance.json` records original and local hashes. Active
Lab river/forest/jungle work is outside this port and is already modifying the
live Lab sources. Do not repin or copy those changes into this refresh.

Implemented so far:

- Callback-based material, profile-2 shoreline and continuous-normal kernels,
  without fixture IO or viewport seeds. Differential tests exercise the actual
  port against pinned functions, translated crops and horizontal/vertical wrap.
- Reproducible shader adapter with native gameplay/environment constants,
  owner activity for volcano skirts, b1 viewport/b2 shadow/b3 world bindings,
  and world/hydrology/relief-owner vertex attributes. Shader model 5 is required
  by the extended input count; Windows compilation passes all four entries.
- Worker-owned RGBA16F MSAA4 scratch targets, premultiplied composition and
  resolve before exposure, shared shoulder and sRGB transfer. Off-screen Windows
  tests check HDR values, transparency and transfer across the entire target.
- Native candidate wiring, preserving the frozen profile's 120-byte GPU vertex
  stride and existing cache budgets. `BUILD.bat candidate-only` runs compilation
  and smoke checks without replacing the installed DLL.

The port is **not ready for in-game testing**. `C3X_RENDERER_VISUAL_PROFILE=pickup-r1`
currently selects an incomplete native candidate for engineering only; the
default and installed renderer retain the prior profile. Do not promote this
switch or infer visual acceptance from component compilation.

Remaining coupled work:

1. Connect the terrain query to authoritative production topology, with a
   stable nearest-coast domain and complete cache dependency accounting. The
   Lab's nearest-contour query can consume distant coast; a viewport crop or
   arbitrary fixed-radius truncation is not equivalent. Resolve this explicitly
   before using shoreline fields in published cached meshes.
2. Port selected cliff meshes/channels/placement v4 and grassy shoulders, broad
   mountain/volcano XY+Z scale and material owner coverage, continuous normals
   and grounding. Preserve authoritative owner activity, routes/cities/resources
   and all deferred-system contracts.
3. Populate terrain and feature world attributes and build shared opaque/cutout
   source caster fields. The current candidate still binds disabled shadow
   constants and retains the legacy shadow pass until the complete replacement
   exists. Field origins/content must participate in shaded-pixel dependencies.
4. Verify linear/native composition and source bounds at both zooms and four
   phases, then cold/warm multi-tile scrolls, wrap, edits, reloads and resets.
   Run integration/full and approved injected compilation if capture/API changes.
5. Stage the final build and extensive diagnostics for the user-run checkpoint.

Verification commands:

```sh
python3 -m unittest Renderer.native.test_profile_v2
python3 Renderer/native/profile_v2/generate_shaders.py
```

Use `renderer_dev.windows_command_result` to dispatch `call VERIFY.bat` in
`Renderer/native/profile_v2`. It runs shader and off-screen composition checks;
it does not start the game. Current local evidence is
`Renderer/verification/pickup_profile_d3d11.json`.

The first native candidate compile passed C++ and the portable native smoke.
The licensed frozen-profile replay failed its existing incremental boundary
comparison: 1,939 changed channels, absolute error 54,687 across 5,344,928 bytes.
Earlier production evidence already records intermittent boundary failures.
Investigate against an unchanged baseline; do not weaken the threshold or claim
the candidate has passed. The build was not copied to `Renderer/bin`.

No new Civ III patch symbol is currently required. Historical visual approval,
LQ gates and convergence requirements remain pending rather than fabricated.

The unchanged installed DLL subsequently passed the same licensed replay. After
preserving the frozen profile's original mesh signatures, the staged candidate
also passed: prepared jump p95 22.262 ms; foreground p95 2.375 ms (953 samples),
maximum 24.145 ms; pixel-block comparison 319 changed channels / 10,045 absolute
error. These are preservation evidence, not proof the initial intermittent
boundary issue is eliminated. The installed DLL was not replaced.

API 14 now transports optional packed world topology (four bytes per playable
tile), with immutable foreground and preparation copies. The injected candidate
captures current authoritative base/real/river/activity values and logs capture
cost, changed count and revision through `OutputDebugStringA`. The frozen
profile does not request this capture. The approved injected compile passed;
the new topology and sparse coast-index kernels pass portable parity/update
checks. Native index population and geometry use remain to be connected.
