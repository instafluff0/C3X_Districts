# Complete-scene linear adapter v1

Generate with `python3 Renderer/terrain_lab/v2/systems/lighting/prepare_linear_scene.py`.
The source-compatible shader is `shaders/lighting/generated/scene_linear_v1.hlsl`
relative to Lab v2. Its source/output digests are in `scene_linear_provenance.json`.
This is an opt-in adaptation of immutable L21, not a replacement accepted reference.

All twelve explicit display gamma conversions and their outer display clamps are
removed before composition. Frame exposure and tone mapping are identity inside
the material shader. Q0 packet exposure must equal the original `exposure` times
`environment_exposure` when `l13a_layout > .5` (otherwise original exposure).
The shared final pass applies that exposure, shoulder, then exact sRGB once.
PSMain and PSFeature retain the source attributes and produce premultiplied
scene-linear SV_Target0 plus independent validity SV_Target1. Zero alpha discards.
Texture transfer flags remain authoritative; never decode the final legacy image
and call it scene-linear shading.

Q0 splits actual triangles (uniform category per triangle) using these states:

| Source class | Depth | Blend | Order |
| --- | --- | --- | --- |
| Main panel > .5, kinds 0–4: land/relief/beach/cliff/bed | 2 test/write | 0 opaque | First |
| Feature opaque or authored cutout, excluding ground state below | 2 test/write | 0 opaque | Opaque scene |
| Main kinds 5/6 water/foam, 8 debug, 9 river, 11 route, 13 border | 1 test only | 1 premul | Preserve source order initially |
| Main kinds 7/10/12/14 legacy projected shadows | 1 test only | 1 premul | Provisional source order only |
| Feature material [20.5,28.5), fractional part [.295,.320) ground state | 1 test only | 1 premul | After receiver |

Only the stated fractional range is translucent ground state; other fractional
feature codes denote raised mines, units, infrastructure and tile objects.
Main diagnostic panels outside panel > .5 are opaque. Reject unknown kinds rather
than silently blending opaque terrain. Clear depth once, never per split draw.
Water does not write depth. Transparent source layers still need stable geometric
ordering; sorting transparent draws by arbitrary submission is not an acceptance.

The inherited shadow polygons remain a documented provisional path. They do not
prove receiver-following mutual shadows and must be removed when the common world
shadow field is bound. The legacy feature source shader also has incomplete alpha
coverage in some families; the source-complete Q7/Q6 mesh path has the explicit
cutout witness. This adapter does not promote those legacy material bindings.

Q2 material hook: define `Q2_MATERIAL_RESPONSE` to 1. The explicit read-only include
`shaders/terrain/scene_material_v1.hlsl` sees frozen textures and PixelInput.
An explicit path lets the shared runner pin and expand the full shader closure.
`q2_material_form(input, world_position, geometry_normal, albedo, material_normal)`
runs before diffuse; `q2_material_specular(input, world_position, geometry_normal,
specular)` runs before highlight. Albedo, material_normal and specular are inout.
No output conversion belongs in a material hook.

## World receiver extension (staged)

`scene_world_v1.hlsl` defines Q6_WORLD_SHADOWS. Main world position+validity is
TEXCOORD14; feature is TEXCOORD2. Q0 supplies actual generated XYZ in the shared
lattice space, never inferred from depth. b1 contains five float4s in order:
U.xyz/span, V.xyz/resolution, L.xyz/0, origin.xyz/0, enabled/contact/0/0.
Main uses texture t25 (feature-only source alias), feature uses t17 (bed-only).
The same receiver query covers all seven material illumination calls. It
excludes water from tighter contact and leaves emission outside cast visibility.
The shader compiles all four entry points, and the disabled hook renders
byte-identically to its unhooked complete-source reference. Shared world attrs,
b1 binding and owner packet-postprocessor are pending Q0 implementation; this
staged shader is not a completed composed-world rendering witness.

## Q3 material hooks

An owner wrapper may define Q3_WATER_MATERIAL and/or Q3_SHORE_MATERIAL, include
the generated linear shader, then include the owner's function definitions.
Conditional prototypes avoid a competing full source fork. Water callback
`float4 q3_water_material(PixelInput input)` runs early for main panel kinds
4/5/6/9 and returns raw radiance plus coverage; final wrapper premultiplies.
Shore callback `void q3_shore_material(PixelInput input,float2 world_position,
inout float3 albedo)` runs after source albedo adjustments and before Q2 form
and shared illumination. Q3 supplies normalized source data; Q0 must explicitly
expose hydrology attributes. Missing fields are not inferred from terrain flags.
