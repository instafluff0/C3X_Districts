struct V {float3 position:POSITION; float4 color:TEXCOORD0;};
struct P {float4 position:SV_POSITION; float4 color:TEXCOORD0;};
P VSMain(V v){P p;p.position=float4(v.position,1);p.color=v.color;return p;}
P VSFeature(V v){return VSMain(v);}
struct T {float4 color:SV_Target0;float validity:SV_Target1;};
T PSMain(P p){T t;t.color=p.color;t.color.rgb*=.7;t.validity=1;return t;}
T PSFeature(P p){return PSMain(p);}
