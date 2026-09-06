// Shared Q6 receiver query for shadow_field_v1.h; no private clock or material bindings.
float q6_shadow_visibility(Texture2D field, float3 world, float3 normal,
 float4 ShadowU, float4 ShadowV, float4 ShadowL, bool shadows, bool contact){
 if(!shadows)return 1;
 float3 w=world+normal*.001;
 float2 uv=float2(dot(w,ShadowU.xyz),dot(w,ShadowV.xyz))/ShadowU.w+.5;
 float z=dot(w,ShadowL.xyz)/ShadowU.w+.5;
 // Match comparison depth to each sampled receiver-plane location. A fixed
 // bias on sloping roofs/rocks creates a stippled false self shadow.
 float2 ux=ddx(uv),uy=ddy(uv);float zx=ddx(z),zy=ddy(z);
 float determinant=ux.x*uy.y-ux.y*uy.x;
 float2 gradient=abs(determinant)>1e-12?float2(zx*uy.y-zy*ux.y,zy*ux.x-zx*uy.x)/determinant:0;
 int2 center=int2(uv*ShadowV.w);float sum=0,closest_delta=0;
 for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
   float blocker=field.Load(int3(clamp(center+int2(x,y),int2(0,0),int2(ShadowV.w-1,ShadowV.w-1)),0)).r;
   float2 sampled_uv=(float2(center+int2(x,y))+.5)/ShadowV.w;
   float receiver=z+dot(gradient,sampled_uv-uv);
   // Physical bias remains fixed when a larger scene needs a wider field.
   float bias=.00060/ShadowU.w;
   sum+=step(blocker,receiver+bias);
   if(x==0&&y==0)closest_delta=blocker-receiver;
 }
 float soft=sum/9;
 // Tighten only genuinely adjacent caster/receiver contact, no blanket AO.
 float world_gap=closest_delta*ShadowU.w;
 if(contact && world_gap>.0039 && world_gap<.024)soft=min(soft,.15);
 return soft;
}
