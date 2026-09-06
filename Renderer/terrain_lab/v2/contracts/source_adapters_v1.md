# Source adapter and wire5 contract

These are explicit module opt-ins. Default frozen wire2 and HLSL remain unchanged.
Q6 owns illumination, shadow eligibility and alpha semantics; Q0 only transports
attributes, textures and buffers. All cast-shadow candidates must use actual
transformed caster geometry and authored cutout alpha. Historical generic shadow
ribbons and masks remain regression evidence only.

## Attachments and bindings

Packet readers accept wire1–5. Linear wire3 adds color branch, final half-open
valid rectangle, exposure, explicit depth (0 off, 1 read, 2 read/write) and blend
(0 off, 1 premultiplied over, 2 RGB additive with unchanged alpha). Wire4 adds
shader_count and per-draw shader_index. Compatible linear modules can use different
HLSL sources; composition rebases resources and shader namespaces. Exposure,
viewport, valid rectangle and color semantics must match exactly.

Wire5 is selected with `Packet.binding_contract=2`. It preserves wire4's header
and adds `Draw.frame_buffer` after shader_index. UINT32_MAX means absent; otherwise
this is a packet buffer index with a nonzero, 16-byte-aligned size up to 64 KiB.
Shared b1 is fragment-stage only: Metal argument ID131 and D3D PS b1. Missing
referenced bindings fail. Composition rebases this buffer too. The source shadow
adapter uses main t25 and feature t17 as explicitly agreed aliases; no 129th
texture slot is introduced. Q6's five float4s at b1 remain Q6-owned.

Module `linear_adapter:1`, `world_positions:1` exports float4 world values at
main TEXCOORD14 and feature TEXCOORD2. XYZ uses local lattice column+u,
row+(1-v), authored ground height/112. Feature world coordinates use the actual
source yaw, XY scale and explicitly inherited vertical projection calibration.
W=1 identifies actual positions; W=0 identifies legacy projected geometry for
which no world position is invented. The packet's attribute offsets/stride are
authoritative; do not borrow a migration C++ vertex type for an independent ABI.

Module `packet_postprocessor` is exactly
`{"source":"<owned>.cpp","owner":"Q6-lighting","contract":1}`.
The executable receives INPUT_PACKET OUTPUT_PACKET HOUR FIXTURE_JSON. It reads
the source packet (including external blobs), adds owned resources/bindings and
writes a standalone packet. Q0 compacts the result. Input, executable closure,
fixture and phase participate in cache identity. Source metadata is retained as
pre-postprocessing evidence; final bindings are in the actual resulting packet.
Use `tests/platform/frame.fixture.json` for a runnable b1/postprocessor example.

## Sampling and output

Linear RGBA16F is premultiplied Rec709/D65; R8 validity is independent. Opaque
writes establish validity; translucent/additive fragments cannot grow the map.
Box reconstruction skips invalid source samples and averages remaining values
over the complete footprint (invalid pixels contribute zero). The shared CPU
output reference unpremultiplies, applies Q6 exposure and max-channel shoulder,
then exact IEC sRGB and straight BGRA8. GPU timing excludes this CPU conversion.

Postprocess contract2 binds source RGBA16F t0, destination RGBA16F u1, source
validity R8 t3, destination validity R8 u4, and b2 containing uint2 input_size,
uint2 output_size, int4 valid_rect. Dispatch is 8x8. Contract1 remains the legacy
BGRA8 path. Q1 owns reconstruction policy. Metal and D3D support matching MSAA,
resolve, scale1/2/4, deterministic final-pixel offsets and contract2 postprocessing.
Native parity currently requires anisotropy8 and mip_bias0; unsupported settings
fail, without silently substituting another quality policy.

## Surface, hydrology and placement

`app/surface_query.py` returns the exact frozen authoring height, normal,
projection and terrain identity at declared points, without GPU or VM access.
Do not infer world height from clip depth.

Module `terrain_hooks` supplies owned header, initialize, material_weights and
optional material_uv. `hydrology_hooks` supplies initialize, signed_shore_distance
and optional shore_sample. Every function name is a qualified C++ identifier.
Callbacks default null; headers and transitive dependencies are cache inputs.
Initializer receives the exact terrain CSV. Material weights are five finite
normalized grass/plains/desert/marsh/tundra values. UV replacement only affects
land material UVs; macro/world/water coordinates remain separate.

Hydrology's compatibility shore distance is water-positive [-1,1]. Optional
shore_sample(float x,float y,float out[4]) supplies positive-land distance,
beach width, rocky fraction and depth. Module `hydrology_data:1` also requires
world_positions:1 and the callback; main TEXCOORD15 carries the sample from
every source terrain/water/bed vertex. The owner shader enables its matching
Q3_HYDROLOGY_DATA declaration; feature vertices do not receive this attribute.

`placement_hooks` supplies owned header, initialize and accept_vegetation.
Initialize receives terrain CSV and complete fixture JSON. The callback is:

```cpp
bool accept_vegetation(const char* group, const char* asset_id,
 unsigned seed, unsigned instance, const float* xyz, unsigned vertex_count);
```

It receives every actual transformed forest/jungle source vertex, before
geometry and shadows are inserted. Coordinates are civ3_raw_delta_pixels_v1:
X=64*(column+row), Y=64*(column-row), Z=64*normalized_height. Stable seed/instance
and source asset identity permit deterministic placement. Q4 owns footprint,
crown/overhang and corridor decisions; Q7 provides actual city/building/wall
polygons. Camera offset does not enter placement. Default null preserves history.

## Source provenance

Each frozen packet has a hash-verified `.source.json` auxiliary artifact. Reports
reference it by portable path and SHA256. It records normalized bundle identity,
source byte hash, interleaved source vertex/index and separate UV/normal hashes,
counts, base-color reference, DDS source/view formats and mip counts, source
texture-to-packet binding identities, and source draw texture slots. Each instance
records source scale/yaw/anchor and the explicit legacy vertical calibration.
Tangent data is absent from C3XVEG1 and is declared absent. This is source reuse
and calibration evidence, not proof that every source channel has been imported
or that inherited geometry is a uniform world transform. Material channel names
come from the exact consumed HLSL declarations and normalized pack manifest.

## Generic draw geometry and final composition (wire6)

Fixture-level `packet_postprocessor` applies the same owned CLI after final
composition. Omit child postprocessors when one final shared field is intended.
No metadata is inferred from independent shaders' vertex layouts.

`Packet.geometry_contract=1` opts into wire6, which preserves wire5 and adds
five uint32s plus one float after frame_buffer in each draw: world_attribute,
normal_attribute, uv_attribute, alpha_texture_slot, geometry_flags, alpha_cutoff.
Attribute values index Draw.attributes, not byte offsets or shader semantics.
Absent attribute/slot is UINT32_MAX. World must be float4 normalized tile XYZ
and validity; normal float3 and UV float2 are optional except UV for cutout.
Flags are bit1 caster, bit2 receiver, bit4 cutout. A cutout must bind its actual
source-alpha texture at alpha_texture_slot with finite cutoff in [0,1]. Split
material draws when alpha bindings differ. The flag-zero/absent-world defaults
preserve older source-specific adapters. Q6 reads this explicit contract for
generic source modules and owns whether/how they participate in its field.
Same coordinate-space and actual posed mesh requirements still apply. This
extension adds no new backend shading or invented shadow geometry.
