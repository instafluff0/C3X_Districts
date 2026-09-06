// Diagnostic wrapper exercising the Q1 published candidate, not a selected policy.
#include "../sampling/reconstruction_v1.hlsl"
Texture2D<float4> Scene : register(t0);
RWTexture2D<float4> Output : register(u1);
cbuffer Sizes : register(b2) { uint2 input_size; uint2 output_size; int4 valid_rect; };
Texture2D<float> Validity : register(t3);
RWTexture2D<float> OutputValidity : register(u4);
[numthreads(8,8,1)]
void CSPost(uint3 id:SV_DispatchThreadID) {
 if(any(id.xy>=output_size))return;
 Q1Reconstructed r=q1_reconstruct_box(Scene,Validity,input_size,output_size,valid_rect,id.xy);
 Output[id.xy]=r.rgba;OutputValidity[id.xy]=r.coverage;
}
