Texture2D<float4> checker:register(t0);
SamplerState material_sampler:register(s0);
struct Input {float3 position:POSITION;float2 uv:TEXCOORD0;float panel:TEXCOORD1;};
struct Pixel {float4 position:SV_POSITION;float2 uv:TEXCOORD0;float panel:TEXCOORD1;};
Pixel VSMain(Input i) {Pixel o;o.position=float4(i.position,1);o.uv=i.uv;o.panel=i.panel;return o;}
Pixel VSFeature(Input i) {return VSMain(i);}
float4 PSMain(Pixel i):SV_TARGET {
    if(i.panel>2.5) {
        // Diagnostic query is available once Q0 exposes MSL 2.2 compilation.
#ifdef Q1_HARDWARE_LOD
        float lod=checker.CalculateLevelOfDetail(material_sampler,i.uv);
#else
        // Explicit estimate, NOT selected anisotropic hardware LOD.
        float2 dx=ddx(i.uv)*128,dy=ddy(i.uv)*128;
        float lod=log2(max(length(dx),length(dy)));
#endif
        return float4(saturate(lod/7),saturate(1-lod/7),0,1);
    }
    return checker.Sample(material_sampler,i.uv);
}
float4 PSFeature(Pixel i):SV_TARGET {return PSMain(i);}
