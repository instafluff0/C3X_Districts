#include "../city_fidelity/feature.hlsl"
#include "rigid_instance_geometry.hlsl"
FeaturePixelInput VSSharedFeature(RigidInput i){
 RigidPoint p=rigid_point(i);PackedFeatureInput packed;
 packed.position=p.position;packed.world=p.world;packed.normal=p.normal;packed.uv=i.source_uv;packed.material=i.placement_view.w;
 FeaturePixelInput o=VSIntegratedFeature(packed);
 float3 position=project_world_content(p.position,p.world,i.projection,2);
 o.position.xy=(floor(position.xy*256+.5)/256+i.placement_view.xy)*c3x_inverse_viewport_size*float2(2,-2)+float2(-1,1);
 o.position.z=clamp(.5-(floor(position.z*256+.5)/256+i.placement_view.z)/16384.,.001,.999);
 return o;
}
FeaturePixelInput VSSharedFeatureReflection(RigidInput i){
 FeaturePixelInput o=VSSharedFeature(i);RigidPoint p=rigid_point(i);
 float h=max(0,p.world.z-NativeReflection.z);
 o.position.y-=h*NativeReflection.x*4*c3x_inverse_viewport_size.y;
 float base=project_world_content(p.position,p.world,i.projection,2).y+h*NativeReflection.x;
 o.position.z=clamp(.5-(floor((base-h*NativeReflection.y)*256+.5)/256+i.placement_view.z)/16384.,.001,.999);
 return o;
}
