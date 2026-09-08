// Interface witness only. Q1 owns downsampling/sharpening quality policies.
// Dispatch covers exactly output_size. RGBA is the frozen display-encoded color;
// alpha is straight. No exposure, gamma, sharpening, or alpha policy is implied.
Texture2D<float4> source_color : register(t0);
RWTexture2D<float4> output_color : register(u1);
cbuffer PostSettings : register(b2) { uint2 input_size; uint2 output_size; };
[numthreads(8,8,1)]
void CSPost(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= output_size)) return;
    uint2 at = min(input_size - 1, id.xy * input_size / output_size);
    output_color[id.xy] = source_color.Load(int3(at,0));
}
