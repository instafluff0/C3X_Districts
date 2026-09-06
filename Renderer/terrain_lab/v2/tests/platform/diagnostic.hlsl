struct V {float3 position:POSITION; float4 color:TEXCOORD0;};
struct P {float4 position:SV_POSITION; float4 color:TEXCOORD0;};
P VSMain(V v){P p;p.position=float4(v.position,1);p.color=v.color;return p;}
P VSFeature(V v){return VSMain(v);}
float4 PSMain(P p):SV_TARGET{return p.color;}
float4 PSFeature(P p):SV_TARGET{return p.color;}
