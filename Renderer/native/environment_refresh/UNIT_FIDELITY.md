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
  legacy GGX/sky response; `unit_microfacet.hlsl` supplies the selected studio
  RGB dual-lobe material, first-map normal detail and positive-alpha tint. `prepare_unit_shader.py`
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
  detail. Optional AO/gloss/emission come from material metadata. Selected
  source kits use authored tangent frames and the first normal map. The second
  LEAN variance constants remain unresolved and are not guessed.
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

The user approved putting the latest unit findings into the game. The authored
`unit_quality.json` selects Warrior, Spearman, Pikeman, Archer, Settler and Worker
for anatomy fitting, 4x scratch sampling, 1536px pose shadows and the recovered
material model. Other families retain their existing geometry/material policy.
The game keeps its authoritative projection, shared sun/moon, exposure and
native sprite anchors; the independent studio camera/light/transfer are not
silently installed as a category-specific environment.

Offline `UnitFrameFidelity` preserves the expensive recovered 32-component
frames and normalized identity inputs. `prepare_units.py` verifies position,
normal and UV correspondence before emitting generic C3XANM2 payloads: the v1
32-byte header gains magic/version 2; each vertex retains its original 64 bytes
and appends float3 tangent plus float3 bitangent. Indices and all animation
palettes remain exact. V1 decoding remains supported. Runtime code contains no
source-game dispatch or source-format dependency.

The optional `c3x_renderer_unit_draw_expanded` API reports the complete rectangle
on success. Pack-authored minimum canvases preserve the native center at odd,
reduced and custom zoom sizes. The injected bridge unions the returned rectangle
into the parent display unit's existing dirty region; failed draws leave the
native fallback and rectangle intact. The renderer permits output up to 1024px
for the existing maximum 2x projection, with the existing 8 MiB sprite cache.
No new patch-table entry is required. The updated bridge must be installed
alongside the staged DLL; an old bridge cannot consume the expanded bounds.


Delivery verification: 204 focused regression tests, the approved injected-code
compile/injection smoke test, native day/night action/compositing checks, and
six-family moving-pose Lab renders at 128/192 tile widths passed. The exact tested
DLL and updated injected bridge were installed after the user closed Civ III.
The installed executable contains the expanded-bounds callback and the staged
DLL hash matches the tested candidate. Receipt and previous executable/DLL are
retained locally under `lab/out/units/game-promotion/`. Civ III was not launched;
no live-game visual acceptance or fixed-reference replacement is claimed.
