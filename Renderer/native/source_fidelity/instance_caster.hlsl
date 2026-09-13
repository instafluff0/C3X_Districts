#define C3X_INSTANCE_CASTER 1
// Shared source-to-world instance transform for color and shadow passes.
struct InstanceInput {
 float3 source_position:POSITION;float3 source_normal:NORMAL;float2 source_uv:TEXCOORD0;
 float4 place0:TEXCOORD1;float4 place1:TEXCOORD2;
 float4 projection:TEXCOORD3;float4 placement_view:TEXCOORD4;
};
float3 instance_world(InstanceInput i) {
 precise float co=i.place1.x,si=i.place1.y,scale=i.place1.z;
 precise float x=i.place0.x+i.place0.z+(i.source_position.x*co-i.source_position.y*si)*scale;
 precise float y=i.place0.y+1-i.place0.w-(i.source_position.x*si+i.source_position.y*co)*scale;
 precise float z_basis=150.f/(.82f*64.f);
 precise float height=i.place1.w+i.source_position.z*scale*z_basis*112;
 return float3(x,y,height/112);
}
#ifdef C3X_INSTANCE_CASTER
cbuffer Caster:register(b0){float4 U;float4 V;float4 L;float4 page;float4 offset;};
struct InstancePixel {float4 position:SV_POSITION;float2 uv:TEXCOORD0;nointerpolation float material:TEXCOORD1;
 float depth:TEXCOORD2;float coverage:TEXCOORD3;float boundary:TEXCOORD4;float4 volcano:TEXCOORD5;};
InstancePixel VSInstance(InstanceInput i){
 float3 world=instance_world(i)+i.placement_view.xyz;
 InstancePixel o;o.position=float4((dot(world,U.xyz)/6-page.x)*2-1,1-(dot(world,V.xyz)/6-page.y)*2,.5,1);
 o.uv=i.source_uv;o.material=i.placement_view.w;o.depth=dot(world,L.xyz);o.coverage=1;o.boundary=o.material;o.volcano=0;return o;
}
#else
cbuffer InstanceMaterial:register(b9){float4 instance_material;float4 instance_secondary;};
P VSInstance(InstanceInput i){
 V v=(V)0;v.world=float4(instance_world(i),1);v.uv=i.source_uv;
 precise float co=i.place1.x,si=i.place1.y;
 precise float3 n=float3(i.source_normal.x*co-i.source_normal.y*si,-(i.source_normal.x*si+i.source_normal.y*co),i.source_normal.z/(150.f/(.82f*64.f)));
 float length=sqrt(dot(n,n));v.normal=length<=1e-8?n:n/length;v.material=instance_material;v.secondary=instance_secondary.xy;
 float dx=v.world.x-i.projection.x,dy=v.world.y-i.projection.y;
 float h=v.world.z*112-2.5,base=(dx-dy+1)*i.projection.z*.25;
 v.position=float3((dx+dy)*i.projection.z*.5,base-h*(i.projection.z/224*.82),base+h*.0016*i.projection.w);
 P o=VSMain(v);
 o.position.xy=(floor(v.position.xy*256+.5)/256+i.placement_view.xy)*inverse_size*float2(2,-2)+float2(-1,1);
 o.position.z=clamp(.5-(floor(v.position.z*256+.5)/256+i.placement_view.z)/16384.,.001,.999);
 return o;
}
#endif
