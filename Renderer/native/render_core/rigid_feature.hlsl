#include "../city_fidelity/feature.hlsl"
#include "rigid_instance_geometry.hlsl"
#include "../../lab/shared/shaders/objects/resident_instance.hlsl"
FeaturePixelInput VSSharedFeature(RigidInput i){
 RigidPoint p=rigid_point(i);PackedFeatureInput packed;
 packed.position=p.position;packed.world=p.world;packed.normal=p.normal;packed.uv=i.source_uv;packed.material=i.placement_view.w;
 FeaturePixelInput o=VSIntegratedFeature(packed);
 float3 position=project_world_content(p.position,p.world,i.projection,2);
 o.position.xy=(floor(position.xy*256+.5)/256+i.placement_view.xy)*c3x_inverse_viewport_size*float2(2,-2)+float2(-1,1);
 // Bridges (materials 13-20: road and railroad, normal and pillaged) span rivers. Use the natural height-depth
 // basis of the terrain and water beneath them, as resource_natural_depth does
 // for resource bodies; the feature basis weights ground height far less.
 bool bridge=i.projection.z!=0 && i.placement_view.w>12.5 && i.placement_view.w<20.5;
 // Farm kit trees and farmhouses stand on low relief and hills in the same way.
 bool farm=i.projection.z!=0 && farm_kit_material(i.placement_view.w);
 if(bridge || farm){float h=p.world.z*112-2.5;
  position.z=position.y+h*(i.projection.z/224*.82)+h*.0016*i.projection.w;}
 o.position.z=clamp(.5-(floor(position.z*256+.5)/256+i.placement_view.z)/16384.,.001,.999);
 // The river surface (kind 9) sorts over its bed and banks 0.025*reserved.x
 // nearer (translated_depth). A bridge spans it, so take slightly more: below
 // about 16 height units, the deck and arches lost to the river and only the
 // parapets showed.
 if(bridge)o.position.z=max(.001,o.position.z-.0255*c3x_viewport_reserved.x/16384.);
 // A bridge rests on its lower bank, which is the waterline. Its authored
 // piers and walls continue below that base; they are underwater, but the
 // shallow carved channel no longer hides them. Carry the height above the
 // base (2 + h/128, readable down to 64 units below; other features keep 1)
 // so PSIntegratedFeature clips them.
 if(bridge)o.q6_world.w=2+p.position.z/128;
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
