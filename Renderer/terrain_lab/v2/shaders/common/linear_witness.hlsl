// Hardware blend/depth/attachment diagnostic, not a visual-system shader.
struct In { float3 p:POSITION; float4 color:TEXCOORD0; float2 uv:TEXCOORD1; };
struct Out { float4 p:SV_Position; float4 color:TEXCOORD0; float2 uv:TEXCOORD1; };
Out VSMain(In i) { Out o;o.p=float4(i.p,1);o.color=i.color;o.uv=i.uv;return o; }
Out VSFeature(In i) { return VSMain(i); }
struct Targets { float4 color:SV_Target0; float validity:SV_Target1; };
Targets PSMain(Out i) {
  Targets o;
  if(i.color.a<0) { clip(frac(i.uv.x*6)-.45);i.color.a=1; }
  o.color=i.color;o.validity=1;return o;
}
Targets PSFeature(Out i) { return PSMain(i); }
