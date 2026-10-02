#ifndef C3X_RESIDENT_INSTANCE_INPUT
#define C3X_RESIDENT_INSTANCE_INPUT
// One immutable 64-byte world/occurrence placement owner. Selected passes send
// four-byte indices; the authored source mesh remains the vertex stream.
struct ResidentInstanceInput {
 float3 source_position:POSITION;float3 source_normal:NORMAL;float2 source_uv:TEXCOORD0;
 uint selection:TEXCOORD1;
};
struct ResidentPlacement {float4 place0;float4 place1;float4 projection;float4 placement_view;};
StructuredBuffer<ResidentPlacement> C3XResidentPlacements:register(t15);
#endif
