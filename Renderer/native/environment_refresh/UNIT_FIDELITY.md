# Current unit fidelity

The accepted C3X build uses `UnitAnimationFidelity`: 78 unit entries and 94
standard Conquests native aliases. Civ III owns the action lifecycle, cursor,
timing and placement; the renderer supplies the existing source bodies and
materials. Unit names and gameplay roles do not select shader behavior.

## Source and preparation

`prepare_units.py` combines `UnitAnimationRuntime` with the fingerprinted authored
normal records in `UnitNormalFidelity`. Imported components use recovered
octahedral SNORM8 normals; original procedural assets retain their own normals.
The source importers and `refresh_unit_normals.py` remain the offline authority
for those records. See `Renderer/docs/unit_asset_conversion.md` for source-format
findings. The normal catalog covers 375 imported mesh primitives.

Category commands run unit preparation automatically when consumed catalogs,
payloads, textures, normal records or the builder change. Generation occurs in
disposable output, followed by source and output-conflict checks. Runtime payloads
are independent copies, not hard links to editable source files. The standalone
builder also accepts `--output` for isolated diagnosis; it no longer publishes a
historical verification report. Neither path installs or launches Civ III.

The builder preserves native aliases and material channels, validates source
topology, UV0 and normal fingerprints, then replaces only the selected normal
fields. Positions, indices, other vertex fields and complete animation palettes
remain exact. Both material address axes retain their repeat/clamp settings.
Absent normalized catalogs distinguish procedural art from imported components
missing normal authority; that absence is a tracked build dependency.

## Rendering contracts

- The shared `lab/shared/shaders/objects/beauty_objects.hlsl` supplies the selected
  GGX, sky fill, AO, gloss, emission and rim equations. `prepare_unit_shader.py`
  generates the embedded native shader; material edits require a candidate build.
- Preserve source normals under the inverse-transpose skinner, source attachments,
  primitive order, uniform native fit scale and local palette remaps.
- After skinning and yaw, `scene_lighting.h` converts source normals into the
  world basis used by terrain and buildings. The unit self/ground projection
  uses that same frame light and height metric. This does not alter source
  vertex data. Units still receive only their own pose's shadow; the native
  sprite API does not supply a surrounding world shadow field.
- Retain complete DDS dimensions and mip chains, 16x anisotropy, zero mip bias,
  native sprite resolution and 4x MSAA. Natural-terrain sampling settings do not
  silently replace the unit settings. Sharing a texture does not share its sampler.
- Native civilization color modulates the source texture without replacing its
  detail. Optional AO/gloss/emission come from material metadata; the current
  roster has no emissive maps. Preserve paired LEAN inputs offline; their decode
  remains unresolved and is not guessed in the runtime shader.
- Preserve the 96 MiB payload budget, 8 MiB sprite cache, identity/dirty bounds,
  action cursor reuse, clipping, config-off path, separate unit fallback and
  retained terrain. Texture diagnostics use debugger output, not game-file logging.

## Verification and limits

`python3 Renderer/renderer.py test units` runs the selected preparation and
animation checks. Integration verification adds native behavior checks; it does
not assert that a live game was tested. The native nine-family matrix is not the
entire roster. Existing source-payload tests cover 4,622 poses and 1,611 attachment
samples for the imported reference subset; composed/original kits retain their
separate verification requirements.

The old isolated Warrior close-up had a separate noon LUT. Native sprites retain
their established display transfer, authoritative environment and projection;
the close-up does not promise equal visible detail at native unit height. No
sharpening or invented texture detail is introduced by this reorganization.
Optional visual references live in the units and animation categories. Keep
current limitations in those category or implementation notes.

The separate `lab/studies/units` candidate adds explicit anatomy-fit comparisons
and optional per-unit `sample_scale: 2` scratch supersampling. Production bindings
omit that option and retain scale 1. Its shader-archive probe now confirms the
object-family LEAN equations; exact unit tangent/binding/constant correspondence
is still unresolved, so no normal-map behavior changes here. The sizing study
requires larger native dirty bounds for long weapons before any promotion.
