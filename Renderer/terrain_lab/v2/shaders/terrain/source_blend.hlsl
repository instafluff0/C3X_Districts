#ifndef Q2_SOURCE_BLEND
#define Q2_SOURCE_BLEND
// Source cache-bake bytecode weights color/spec/height by color.a^2 times
// an interpolated contribution, then resolves accumulated channels by weight.
// This diagnostic uses the Lab's existing terrain contributions; their mapping
// to the source engine's vertex contribution remains a Lab interpretation.
void q2_source_blend(inout PixelInput input) {
    if(input.surface_kind<.75 || input.surface_kind>1.25)return;
    float4 w=max(input.material_weights,0);
    float t=max(input.material_tundra,0);
    float4 a=float4(base_color_texture.Sample(material_sampler,input.uv).a,
        plains_base_texture.Sample(material_sampler,input.uv).a,
        desert_base_texture.Sample(material_sampler,input.uv).a,
        marsh_base_texture.Sample(material_sampler,input.uv).a);
    float ta=feature_base_texture_4.Sample(material_sampler,input.uv).a;
    float4 weighted=w*a*a;
    float wt=t*ta*ta;
    float total=dot(weighted,1)+wt;
    if(total>1e-8) {
        input.material_weights=weighted/total;
        input.material_tundra=wt/total;
    }
}
#endif
