#include "../city_fidelity/feature.hlsl"
#include "rigid_instance_geometry.hlsl"
#include "../../lab/shared/shaders/objects/resident_instance.hlsl"
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
// The immutable union stores occurrence-relative native anchors. Camera and
// guarded-region translations are small pass constants, shared by all ranges.
// Keep addition grouping equal to the original CPU-built instance values.
RigidInput resident_rigid_input(ResidentInstanceInput input){
 ResidentPlacement p=C3XResidentPlacements[input.selection];RigidInput i;
 i.source_position=input.source_position;i.source_normal=input.source_normal;i.source_uv=input.source_uv;
 i.place0=p.place0;i.place1=p.place1;i.projection=p.projection;i.placement_view=p.placement_view;return i;
}
FeaturePixelInput VSResidentSharedFeature(ResidentInstanceInput input){
 RigidInput i=resident_rigid_input(input);
 i.placement_view.xy=c3x_viewport_translation+i.placement_view.xy;
 i.placement_view.z=c3x_viewport_depth_translation+i.placement_view.z;
 return VSSharedFeature(i);
}
FeaturePixelInput VSResidentSharedFeatureReflection(ResidentInstanceInput input){
 RigidInput i=resident_rigid_input(input);
 i.placement_view.xy=c3x_viewport_translation+i.placement_view.xy;
 i.placement_view.z=c3x_viewport_depth_translation+i.placement_view.z;
 return VSSharedFeatureReflection(i);
}
