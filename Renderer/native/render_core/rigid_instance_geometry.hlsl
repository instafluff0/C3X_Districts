// Generic rigid pack placement, matching objects::append_instance. Source mesh
// normals/UVs remain authored; the instance supplies its stable world placement.
struct RigidInput {
 float3 source_position:POSITION;float3 source_normal:NORMAL;float2 source_uv:TEXCOORD0;
 float4 place0:TEXCOORD1;float4 place1:TEXCOORD2;
 float4 projection:TEXCOORD3;float4 placement_view:TEXCOORD4;
};
struct RigidPoint {float3 world;float3 position;float3 normal;};
RigidPoint rigid_point(RigidInput i){
 precise float co=i.place1.x,si=i.place1.y,scale=i.place1.z,ground=i.place1.w;
 precise float x=(i.source_position.x*co-i.source_position.y*si)*scale;
 precise float y=(i.source_position.x*si+i.source_position.y*co)*scale;
 precise float z=i.source_position.z*scale;
 precise float height=z*150*(128.f/224.f)/(128.f/224.f*.82f);
 RigidPoint p;
 p.world=float3(i.place0.x+i.place0.z+x,i.place0.y+1-i.place0.w-y,(ground+2.5f+height)/112);
 precise float sx=64+(i.place0.z-i.place0.w)*64+(x-y)*64;
 precise float sy=((i.place0.z+i.place0.w)*32-ground*(128.f/224.f*.82f))+(x+y)*32-z*150*(128.f/224.f);
 p.position=float3(sx/128,sy/128,height);
 precise float nx=i.source_normal.x*co-i.source_normal.y*si;
 precise float ny=i.source_normal.x*si+i.source_normal.y*co;
 precise float3 n=float3(nx,-ny,i.source_normal.z/(150.f/(112.f*.82f)));
 precise float length=sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
 p.normal=length>1e-6f?n/length:float3(0,0,1);return p;
}
