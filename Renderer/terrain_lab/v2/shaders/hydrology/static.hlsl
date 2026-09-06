#include "../lighting/response_v1.hlsl"
cbuffer Frame:register(b0){float4 Sun,SunColorExposure,Moon,MoonColorMode,AmbientLegacy,MaterialPeriod;}
Texture2D Grass:register(t0);Texture2D Sand:register(t1);Texture2D Bed:register(t2);Texture2D Rock:register(t3);
Texture2D BedHeight:register(t4);
SamplerState Linear:register(s0);
struct V {float3 p:POSITION;float2 uv:TEXCOORD0;float3 normal:TEXCOORD1;float4 shore:TEXCOORD2;float4 river:TEXCOORD3;};
struct P {float4 p:SV_POSITION;float2 uv:TEXCOORD0;float3 normal:TEXCOORD1;float4 shore:TEXCOORD2;float4 river:TEXCOORD3;};
P VSMain(V v){P p;p.p=float4(v.p,1);p.uv=v.uv;p.normal=v.normal;p.shore=v.shore;p.river=v.river;return p;}
P VSFeature(V v){return VSMain(v);}
float scale(float requested){return MaterialPeriod.x>0?round(requested*MaterialPeriod.x)/MaterialPeriod.x:requested;}
float4 PSMain(P p):SV_TARGET {
 float sd=p.shore.x,width=p.shore.y,rocky=p.shore.z,depth=p.shore.w,rd=p.river.x;
 float3 grass=Grass.Sample(Linear,p.uv*scale(.37)).rgb;
 float3 sand=Sand.Sample(Linear,p.uv*scale(.91)).rgb;
 float2 bedUV=p.uv*scale(.48);
 float3 bed=Bed.Sample(Linear,bedUV).rgb;
 float h0=BedHeight.Sample(Linear,bedUV).r;
 float hx=BedHeight.Sample(Linear,bedUV+float2(.003,0)).r-h0;
 float hy=BedHeight.Sample(Linear,bedUV+float2(0,.003)).r-h0;
 // Source material height affects shading only, never bathymetric depth.
 float3 bedNormal=normalize(float3(-hx*9,-hy*9,1));
 bed*=clamp(.74+.34*dot(bedNormal,normalize(Sun.xyz+Moon.xyz*Moon.w)),.53,1.12);
 float3 rock=Rock.Sample(Linear,p.uv*scale(.75)).rgb;
 float grain=dot(sand,float3(.2126,.7152,.0722));
 float blend=1-smoothstep(width*.25,width+.18,sd+(grain-.28)*.24);
 if(AmbientLegacy.w>.5){width=.20;blend=1-smoothstep(.19,.23,sd);rocky=0;}
 float3 land=lerp(grass,sand,blend*(1-rocky));
 land=lerp(land,rock,rocky*(1-smoothstep(.03,.22,sd)));
 float wet=1-smoothstep(-.02,.12,sd);
 land*=1-.28*wet;
 float3 bedColor=lerp(sand,bed,smoothstep(0,.40,-sd));
 bedColor=lerp(bedColor,lerp(rock,bed,smoothstep(0,.70,-sd)),rocky);
 // Wet sand must continue below the surface without a lighter permanent ribbon.
 bedColor*=lerp(.72,1.0,smoothstep(0,.32,-sd));
 float river=1-smoothstep(-.009,.009,rd);
 // Thin terrain-tinted river bank, unioned into the coast (no mouth plug).
 land=lerp(land,lerp(grass,sand,.30)*.78,(1-smoothstep(.01,.043,rd))*step(0,sd));
 float submerged=1-smoothstep(-.006,.006,sd);
 submerged=max(submerged,river);
 float optical=max(depth,river*.045);
 float3 waterColor=float3(.023,.074,.096);
 float transmission=exp(-optical*5.4);
 float3 underwater=lerp(waterColor,bedColor,transmission);
 underwater=lerp(underwater,float3(.065,.125,.155),river*.78*smoothstep(-.20,.08,sd));
 float3 albedo=lerp(land,underwater,submerged);
 if(MoonColorMode.w==1)albedo=lerp(land,float3(.025,.045,.055),submerged); // beach-only
 if(MoonColorMode.w==2)albedo=lerp(land,bedColor,submerged); // bed-only
 if(MoonColorMode.w==3)albedo=float3(rocky,1-rocky,0)*(1-smoothstep(.1,.35,abs(sd))); // edge class
 if(MoonColorMode.w==4)albedo=lerp(albedo,float3(.85,.05,.1),.65*(1-step(.083,rd))); // placement exclusion witness, not art
 float3 light=AmbientLegacy.rgb*.46+SunColorExposure.rgb*Sun.w*.9+MoonColorMode.rgb*Moon.w*.78;
 float3 color=q6_srgb_encode(q6_display_linear(albedo*light,SunColorExposure.w));
 return float4(color,1);
}
float4 PSFeature(P p):SV_TARGET{return PSMain(p);}
