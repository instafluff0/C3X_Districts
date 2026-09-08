// Provisional display attachment: adopt Q6's shared response, no private clock.
#include "../lighting/response_v1.hlsl"
cbuffer Frame:register(b0) { float4 Sun, SunColorExposure, Moon, MoonColorNight, AmbientShadow, Bounds; };
Texture2D Base:register(t0);
Texture2D Heights:register(t1);
Texture2D Sand:register(t2);
Texture2D Rock:register(t3);
Texture2D SourceLean:register(t4);
Texture2D SourceGloss:register(t5);
Texture2D SourceFootprint:register(t6);
SamplerState Wrap:register(s0);
struct V {float3 p:POSITION; float3 n:NORMAL; float2 uv:TEXCOORD0; float3 w:TEXCOORD1; float4 blend:TEXCOORD2;};
struct P {float4 p:SV_POSITION; float3 n:NORMAL; float2 uv:TEXCOORD0; float3 w:TEXCOORD1; float4 blend:TEXCOORD2;};
P VSMain(V v) { P p; p.p=float4(v.p,1);p.n=v.n;p.uv=v.uv;p.w=v.w;p.blend=v.blend;return p; }
P VSFeature(V v) {return VSMain(v);}
float4 source_triplanar(Texture2D tex,float3 position,float3 normal) {
    float3 weights=pow(abs(normal),4);weights/=max(dot(weights,1),.00001);
    float3 uv=position*1.5;
    return tex.Sample(Wrap,uv.yz)*weights.x+tex.Sample(Wrap,uv.xz)*weights.y+
           tex.Sample(Wrap,uv.xy)*weights.z;
}
float shadow(float3 pos) {
    float2 dir=normalize((Sun.xyz*Sun.w+Moon.xyz*Moon.w).xy);
    float occ=0;
    for(int i=1;i<=24;i++) {
        float d=i*.045;
        float2 xy=pos.xy+dir*d;
        float2 uv=(xy-Bounds.xy)/Bounds.zw;
        if(any(uv<0)||any(uv>1)) continue;
        float h=Heights.SampleLevel(Wrap,uv,0).r*2.0-.25;
        occ=max(occ,saturate((h-pos.z-d*1.35-.018)*35));
    }
    return 1-occ*.72;
}
float4 PSMain(P p):SV_TARGET {
    float4 base=Base.Sample(Wrap,p.uv);
    if(p.blend.w==4) clip(SourceFootprint.Sample(Wrap,p.uv).r-.025);
    if(frac(p.blend.w*.5)>.2) clip(base.a-.32);
    float3 n=normalize(p.n);
    if(p.blend.w>=2 && p.blend.w<4) {
        float2 lean=SourceLean.Sample(Wrap,p.uv).rg*2-1;
        float3 dx=ddx(p.w),dy=ddy(p.w);float2 ux=ddx(p.uv),uy=ddy(p.uv);
        float det=ux.x*uy.y-ux.y*uy.x;
        if(abs(det)>0.00000001) {
            float3 tangent=normalize((dx*uy.y-dy*ux.y)/det);
            float3 bitangent=normalize((dy*ux.x-dx*uy.x)/det);
            n=normalize(n+(tangent*lean.x+bitangent*lean.y)*.35);
        }
    }
    float3 rock=Rock.Sample(Wrap,float2((p.w.x+p.w.y)*.38,p.w.z*.85+p.w.y*.035)).rgb;
    float3 color=lerp(base.rgb,Sand.Sample(Wrap,p.blend.w==4?p.uv:p.w.xy*.48).rgb,p.blend.x);
    if(p.blend.w==4) {
        // Source height-map UVs still drive the silhouette and footprint. Source
        // tiling material uses a separate world projection to avoid stretching
        // one planar texel column over a steep face. Projection is an inferred
        // C3X material adaptation, not a recovered source-engine equation.
        color=lerp(source_triplanar(Base,p.w,n).rgb,source_triplanar(Sand,p.w,n).rgb,p.blend.x);
        float detail=source_triplanar(SourceLean,p.w,n).r;
        float3 dx=ddx(p.w),dy=ddy(p.w);
        float3 r1=cross(dy,n),r2=cross(n,dx);float determinant=dot(dx,r1);
        float3 gradient=(ddx(detail)*r1+ddy(detail)*r2)*sign(determinant)/max(abs(determinant),.000001);
        n=normalize(n-gradient*.025);
    }
    color=lerp(color,rock*.78,p.blend.y);
    if(p.blend.z>.5) color=float3(.052,.18,.23);
    float directShadow=shadow(p.w);
    float3 ambient=AmbientShadow.rgb*.46;
    float3 direct=SunColorExposure.rgb*Sun.w*(.16+.84*saturate(dot(n,Sun.xyz)))+
        MoonColorNight.rgb*Moon.w*(.20+.80*saturate(dot(n,Moon.xyz)))*.82;
    float3 radiance=color*(ambient+direct*directShadow);
    return float4(q6_srgb_encode(q6_display_linear(radiance,SunColorExposure.w)),1);
}
float4 PSFeature(P p):SV_TARGET {return PSMain(p);}
