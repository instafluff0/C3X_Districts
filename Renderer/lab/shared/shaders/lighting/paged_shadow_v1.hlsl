// World-aligned six-tile pages preserve the retained 6/1024 sampling density.
// Source depths use R32_FLOAT physical light distance, avoiding page-dependent
// normalization/quantization. Page identity never contains a screen anchor.


cbuffer C3XShadowPages : register(b4) { float4 pickup_pages[64]; };
int pickup_page(int2 page) {
 uint key=(uint(page.x)*73856093u ^ uint(page.y)*19349663u)&63u;
 [loop]for(int n=0;n<33;n++) {
  float4 entry=pickup_pages[(key+uint(n))&63u];
  if(entry.w<.5)return -1;
  if(all(page==int2(entry.xy)))return int(entry.z);
 }
 return -1;
}
float pickup_blocker(Texture2DArray field,int2 texel,int2 center_page,int center_slot) {
 int2 page=int2(floor(float2(texel)/1024.));
 int slot=all(page==center_page)?center_slot:pickup_page(page);
 return slot<0?-1e6:field.Load(int4(texel-page*1024,slot,0)).r;
}
float c3x_paged_visibility(Texture2DArray field,float4 world,float3 normal,bool water,
 float4 ShadowU,float4 ShadowV,float4 ShadowL,float4 ShadowFlags) {
 if(world.w<=.5 || ShadowFlags.x<=.5)return 1;
 const float texel=6./1024.;
 float3 offset=world.xyz+normal*texel;
 float2 uv=float2(dot(offset,ShadowU.xyz),dot(offset,ShadowV.xyz))/texel;
 float z=dot(offset,ShadowL.xyz);
 float2 plane=float2(dot(world.xyz,ShadowU.xyz),dot(world.xyz,ShadowV.xyz))/texel;
 float plane_z=dot(world.xyz,ShadowL.xyz);
 float2 ux=ddx(plane),uy=ddy(plane);float zx=ddx(plane_z),zy=ddy(plane_z);
 float determinant=ux.x*uy.y-ux.y*uy.x;
 float2 gradient=0;
 if(abs(determinant)>1e-12)gradient=float2(zx*uy.y-zy*ux.y,zy*ux.x-zx*uy.x)/determinant;
 int2 center=int2(floor(uv));int2 center_page=int2(floor(float2(center)/1024.));
 int center_slot=pickup_page(center_page);float sum=0,closest_delta=0;
 [unroll]for(int y=-1;y<=1;y++)[unroll]for(int x=-1;x<=1;x++) {
  int2 sample= center+int2(x,y);
  float blocker=pickup_blocker(field,sample,center_page,center_slot);
  float receiver=z+dot(gradient,float2(sample)+.5-uv);
  sum+=step(blocker,receiver+.00060);
  if(x==0 && y==0)closest_delta=blocker-receiver;
 }
 float soft=sum/9;
 if(!water && ShadowFlags.y>.5 && closest_delta>.0039 && closest_delta<.024)soft=min(soft,.15);
 return soft;
}
