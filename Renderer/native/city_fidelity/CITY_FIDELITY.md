# City composition checkpoint

The user requested ending this round once the implementation compiles and runs.
The default profile now selects `city-fidelity`; normal `INSTALL.bat` uses the
staged renderer DLL. Explicit `source-fidelity-r13` remains available for the
previous natural presentation. The installer and game were not run by the agent.

## Selected paths carried together

- `city_central_capital_probe.py`, `city_scene_pass.py` and
  `city_growth_layout.py`: source buildings, uniform transforms, single-era
  growth and the selected r111 inland/r112 freshcanopy central modern palace.
- `city_scene_material.hlsl`: authored texture coordinates, source frames,
  full-resolution material channels, opacity and emissive windows.
- `city_facade_light_probe.py`, `city_light_buffer_probe.py` and
  `local_facade_lights.hlsl`: offline authored facade sampling and bounded
  runtime light/blocker selection; the selected Asian medium keeps all 53 lights.
- `settlement_ground_probe.py` and `settlement_ground.hlsl`: connected paving,
  palace footprint alignment, source atlas coordinates and local illumination.
- `city_environment.hlsl`, retained environment-refresh reflections and
  `hdr_glow_tiled.hlsl`: selected modern environment response, water reflection,
  guarded linear reconstruction and HDR glow before the final display transfer.

The normalized pack contains 122 models, 37 materials and 72 templates. Twelve
templates preserve selected Lab layouts; 60 use the Lab's generic growth solver
across 20 culture/era pools and three sizes. Generic layouts are not additional
Lab visual witnesses. Asset normalization stays offline; runtime consumes the
generic binary and DDS material channels. `shader-provenance.json` pins sources.

## Evidence and retained behavior

All four pickup evidence validators passed before implementation. Six portable
tests compare source hashes, selected transforms, facade lights, palace paving
and growth prefixes. The binary contract rejects 915 truncated inputs without
replacing a valid loaded library. Windows C++ compilation uses `/W4 /WX`;
the D3D contract compiles and creates the city shaders. Inherited shader compiler
warnings remain; they are not claimed to be a warning-free shader validation.

Actual candidate-DLL day/night synthetic scenes render with zero terrain
fallback and correct native ownership. The night replay returns to byte-identical
pixels (zero changed bytes out of 1,228,800); its cached return takes about 3 ms.
Cold rendering remains measured in seconds. See `../../verification/city_fidelity/`
for run reports and the final release checkpoint.

Geometry stays world-anchored in the existing bounded caches. The city pack is
7.24 MB, full-mip source DDS data approximately 35 MB, and additional city render
scratch approximately 9.08 MB. The existing geometry/shadow/viewport budgets,
native animation bridge, unit sprite caches, dirty bounds, input, overlays,
fog and fallback contracts remain in place. No injected source or patch-table
change is needed. Diagnostics use the established OutputDebugStringA path;
file traces are explicitly enabled only by the headless harness.

## Remaining limits

- Only the selected modern American palace body is included. Other capital
  styles use ordinary city composition and the retained capital indicator.
- A constrained shore/river/vegetation site can reject every composition and
  retain the previous city appearance. An earlier real-map case demonstrated
  this fallback; full real-map placement coverage is not claimed.
- The full city/era/zoom/control matrix and wrapped local-light continuity are
  not closed. The user requested wrapping up rather than extending this round.
- Lab's coastal central-capital and inferred environment-response limitations
  remain. This is not a recovery claim about the source game's entire lighting.
- Existing frozen-profile boundary-parity and L19A fixture-hash workflow failures
  remain separate from passing focused city checks; gates are not weakened.
- Civ III launch, gameplay stability and visual acceptance remain user-run.

No global Lab gate or deferred renderer milestone is closed by this checkpoint.
