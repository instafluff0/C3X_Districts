cbuffer Caster : register(b0) { float4 U;float4 V;float4 L;float4 page;float4 offset; };
struct Input { float2 uv:TEXCOORD0;float material:TEXCOORD1;float4 world:TEXCOORD2;float coverage:TEXCOORD3; };
struct Pixel { float4 position:SV_POSITION;float2 uv:TEXCOORD0;nointerpolation float material:TEXCOORD1;float depth:TEXCOORD2;float coverage:TEXCOORD3;float boundary:TEXCOORD4; };
Pixel VS(Input i) { i.world.xyz+=offset.xyz;Pixel o;o.position=float4((dot(i.world.xyz,U.xyz)/6-page.x)*2-1,
 1-(dot(i.world.xyz,V.xyz)/6-page.y)*2,.5,1);o.uv=i.uv;o.material=i.material;o.depth=dot(i.world.xyz,L.xyz);o.coverage=i.coverage;o.boundary=i.material;return o; }
Texture2D source0:register(t0);
Texture2D source1:register(t1);
Texture2D source2:register(t2);
Texture2D source3:register(t3);
Texture2D source4:register(t4);
Texture2D source5:register(t5);
Texture2D source6:register(t6);
Texture2D source7:register(t7);
Texture2D source8:register(t8);
Texture2D source9:register(t9);
Texture2D source10:register(t10);
Texture2D source11:register(t11);
Texture2D source12:register(t12);
Texture2D source13:register(t13);
Texture2D source14:register(t14);
Texture2D source15:register(t15);
Texture2D source16:register(t16);
Texture2D source17:register(t17);
Texture2D source18:register(t18);
Texture2D source19:register(t19);
Texture2D source20:register(t20);
Texture2D source21:register(t21);
Texture2D source22:register(t22);
Texture2D source23:register(t23);
Texture2D source24:register(t24);
Texture2D source25:register(t25);
Texture2D source26:register(t26);
Texture2D source27:register(t27);
Texture2D source28:register(t28);
Texture2D source29:register(t29);
Texture2D source30:register(t30);
Texture2D source31:register(t31);
Texture2D source32:register(t32);
Texture2D natural_opacity:register(t33);
float alpha(int slot,float2 uv) { uint w,h;
if(slot==0){source0.GetDimensions(w,h);return source0.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==1){source1.GetDimensions(w,h);return source1.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==2){source2.GetDimensions(w,h);return source2.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==3){source3.GetDimensions(w,h);return source3.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==4){source4.GetDimensions(w,h);return source4.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==5){source5.GetDimensions(w,h);return source5.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==6){source6.GetDimensions(w,h);return source6.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==7){source7.GetDimensions(w,h);return source7.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==8){source8.GetDimensions(w,h);return source8.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==9){source9.GetDimensions(w,h);return source9.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==10){source10.GetDimensions(w,h);return source10.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==11){source11.GetDimensions(w,h);return source11.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==12){source12.GetDimensions(w,h);return source12.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==13){source13.GetDimensions(w,h);return source13.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==14){source14.GetDimensions(w,h);return source14.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==15){source15.GetDimensions(w,h);return source15.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==16){source16.GetDimensions(w,h);return source16.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==17){source17.GetDimensions(w,h);return source17.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==18){source18.GetDimensions(w,h);return source18.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==19){source19.GetDimensions(w,h);return source19.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==20){source20.GetDimensions(w,h);return source20.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==21){source21.GetDimensions(w,h);return source21.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==22){source22.GetDimensions(w,h);return source22.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==23){source23.GetDimensions(w,h);return source23.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==24){source24.GetDimensions(w,h);return source24.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==25){source25.GetDimensions(w,h);return source25.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==26){source26.GetDimensions(w,h);return source26.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==27){source27.GetDimensions(w,h);return source27.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==28){source28.GetDimensions(w,h);return source28.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==29){source29.GetDimensions(w,h);return source29.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==30){source30.GetDimensions(w,h);return source30.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==31){source31.GetDimensions(w,h);return source31.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
if(slot==32){source32.GetDimensions(w,h);return source32.Load(int3(int2(frac(uv)*float2(w,h)),0)).a;}
return 1; }
float PSOpaque(Pixel i):SV_TARGET {return i.depth;}
float PSCutout(Pixel i):SV_TARGET {
 if(i.material<0) {clip(i.boundary+10-.001);return i.depth;}
 // Match the visible mountain silhouette, including its interpolated mask.
 if(i.material==42) {clip(smoothstep(.08,.72,i.coverage)-.015);return i.depth;}
 if(i.material>=40) {
  uint w,h;natural_opacity.GetDimensions(w,h);
  bool repeat=i.material>=41;
  float2 uv=repeat?frac(i.uv):saturate(i.uv);
  clip(natural_opacity.Load(int3(min(int2(uv*float2(w,h)),int2(w,h)-1),0)).r-.5);
  return i.depth;
 }
 float part=frac(i.material);
 if(i.material>20.5 && i.material<28.5 && part>=.25 && part<.4)discard;
 if(abs(i.material-.48)>.001)clip(alpha(clamp(int(floor(i.material+.001)),0,32),i.uv)-.5);
 return i.depth;
}
