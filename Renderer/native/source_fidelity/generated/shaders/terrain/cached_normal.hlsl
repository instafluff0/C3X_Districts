#ifndef Q2_CACHED_NORMAL_IMPL
#define Q2_CACHED_NORMAL_IMPL
// Optional per-material normal cache diagnostic. Final pack semantics and the
// source engine's combined-height cache boundary remain pending.
void q2_cached_normal(PixelInput input,float3 n,inout float3 material_normal) {
    float envelope=q2_base_detail_envelope(input,n);
    if(envelope<=0)return;
    float4 w=max(input.material_weights,0);float t=max(input.material_tundra,0);
    if(marsh_enabled<.5) {w.x+=w.w;w.w=0;}
    float total=max(.001,dot(w,1)+t);w/=total;t/=total;
    float2 xy=resource_base_texture_0.Sample(material_sampler,input.uv).rg*w.x
        +resource_base_texture_1.Sample(material_sampler,input.uv).rg*w.y
        +resource_base_texture_2.Sample(material_sampler,input.uv).rg*w.z
        +resource_base_texture_3.Sample(material_sampler,input.uv).rg*w.w
        +resource_base_texture_4.Sample(material_sampler,input.uv).rg*t;
    xy=xy*2-1;
    float2 ux=ddx(input.uv),uy=ddy(input.uv);
    float3 px=ddx(input.q6_world.xyz),py=ddy(input.q6_world.xyz);
    float det=ux.x*uy.y-ux.y*uy.x;
    if(abs(det)<1e-9)return;
    float3 tangent=normalize((px*uy.y-py*ux.y)/det);
    float3 bitangent=normalize((py*ux.x-px*uy.x)/det);
    // Source encoding is (+height dx,-height dy). Convert through this scene's
    // actual UV basis so the perturbation points away from rising height.
    float3 rebuilt=normalize(n-tangent*xy.x+bitangent*xy.y);
    material_normal=normalize(lerp(material_normal,rebuilt,envelope));
}
#ifdef Q2_CACHED_OCCLUSION
float q2_cached_occlusion(PixelInput input,float3 n) {
    float envelope=q2_base_detail_envelope(input,n);
    if(envelope<=0)return 1;
    float4 w=max(input.material_weights,0);float t=max(input.material_tundra,0);
    if(marsh_enabled<.5) {w.x+=w.w;w.w=0;}
    float total=max(.001,dot(w,1)+t);w/=total;t/=total;
    float ao=resource_base_texture_0.Sample(material_sampler,input.uv).b*w.x
        +resource_base_texture_1.Sample(material_sampler,input.uv).b*w.y
        +resource_base_texture_2.Sample(material_sampler,input.uv).b*w.z
        +resource_base_texture_3.Sample(material_sampler,input.uv).b*w.w
        +resource_base_texture_4.Sample(material_sampler,input.uv).b*t;
    return lerp(1,saturate(ao),envelope);
}
#endif
#endif
