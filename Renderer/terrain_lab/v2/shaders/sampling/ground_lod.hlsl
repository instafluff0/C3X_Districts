// Hardware LOD witness for base_color_texture/t0 at the actual ground UVs.
// Other terrain material branches and feature meshes are NOT measured by t0.
#define PSMain FrozenPSMain
#include "../common/frozen_l21.hlsl"
#undef PSMain
float4 PSMain(PixelInput input):SV_TARGET {
    float4 original=FrozenPSMain(input);
    float lod=base_color_texture.CalculateLevelOfDetail(material_sampler,input.uv);
    return float4(saturate(lod/7),saturate(1-lod/7),0,original.a);
}
