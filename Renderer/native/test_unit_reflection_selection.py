"""Current-palette reflection selection; host-only, no GPU/VM runner."""
from pathlib import Path
import unittest
from Renderer.native.test_unit_contribution_plan import BASE, direct_stub, host_cpp

CORE=BASE+r'''
#include "Renderer/native/render_core/skin_shadow_bounds.h"
UnitReflectionBounds certificate(AnimationMesh const& mesh,float const* palette,float angle,float scale,float offset){
 SkinShadowBounds boxes;boxes.prepare(mesh);std::vector<UnitShadow::Point> points;
 boxes.append(palette,angle,scale,offset,points);UnitReflectionBounds result;
 result.append(points,0,boxes.known,boxes.weight_low,boxes.weight_high,double(offset)*scale);return result;
}
'''

class UnitReflectionSelectionTests(unittest.TestCase):
    def test_weighted_current_vertices_inside_rotated_translated_scaled_bounds(self):
        host_cpp(CORE+r'''
int main(){
 AnimationMesh mesh;mesh.bones=2;mesh.vertices.resize(8);float palette[32]={};
 for(unsigned j=0;j<2;++j){auto* p=palette+j*16;p[0]=.7f+j*.2f;p[5]=1.2f;p[10]=.4f;p[15]=1;
  p[4]=.3f;p[9]=-.2f;p[12]=j?1.7f:-.8f;p[13]=j?-.9f:.4f;p[14]=j?1.2f:-.6f;}
 for(unsigned i=0;i<8;++i){auto& v=mesh.vertices[i];v.source.position[0]=(i&1)?1.f:-1.f;
  v.source.position[1]=(i&2)?.4f:-.7f;v.source.position[2]=(i&4)?.6f:-.3f;
  v.joints={0,1,0,0};v.weights={.25f,.75f+(i&1?8e-6f:-8e-6f),0,0};}
 for(float angle:{0.f,.4f,1.7f,3.14f})for(float scale:{-.7f,.2f,1.f,3.9f})for(float offset:{-.5f,0.f,.7f}){
  auto b=certificate(mesh,palette,angle,scale,offset);assert(b.complete&&b.has_points);
  for(auto const& v:mesh.vertices){float p[3]={};
   for(unsigned k=0;k<4;++k){auto* m=palette+v.joints[k]*16;
    for(unsigned a=0;a<3;++a)p[a]+=v.weights[k]*(v.source.position[0]*m[a]+v.source.position[1]*m[a+4]+v.source.position[2]*m[a+8]+m[a+12]);}
   float x=(p[0]*std::cos(angle)-p[1]*std::sin(angle))*scale;
   float y=(p[0]*std::sin(angle)+p[1]*std::cos(angle))*scale,z=(p[2]+offset)*scale;
   double sx=(x-y)*64,sy=(x+y)*32+z*(150.*128/224);
   assert(sx>=b.local.left&&sx<=b.local.right&&sy>=b.local.top&&sy<=b.local.bottom);
  }
 }
}
''')

    def test_native_placement_zoom_ground_guard_receivers_and_unknown_fallback(self):
        host_cpp(CORE+r'''
int main(){
 UnitReflectionBounds b;b.local={-10,-20,30,40};b.has_points=true;
 UnitContributionView v;v.width=1001;v.height=801;v.zoom=2;v.reflection=true;
 // Integer viewport center (500,400); native anchor (600,300), native scale .5,
 // ground +12 and reflection guard +8 put the padded rectangle at:
 // X = [694,742], Y = [208,276].
 v.receivers={{741.99,230,760,250}};assert(b.overlaps(v,600,300,.5,12));
 v.receivers={{742.01,230,760,250}};assert(!b.overlaps(v,600,300,.5,12));
 v.receivers={{700,275.99,720,285}};assert(b.overlaps(v,600,300,.5,12));
 v.receivers={{700,276.01,720,285}};assert(!b.overlaps(v,600,300,.5,12));
 // Multiple water/river rectangles, including filter/distortion reach supplied by caller.
 v.receivers={{0,0,10,10},{700,230,720,250}};assert(b.overlaps(v,600,300,.5,12));
 v.receivers={{0,0,10,10}};assert(!b.overlaps(v,600,300,.5,12));
 auto unknown=b;unknown.complete=false;assert(unknown.overlaps(v,600,300,.5,12));
 unknown={};assert(unknown.overlaps(v,600,300,.5,12));
 unknown=b;unknown.local.left=unknown.local.right+1;assert(unknown.overlaps(v,600,300,.5,12));
 assert(b.overlaps({},600,300,.5,12));assert(b.overlaps(v,NAN,300,.5,12));
 assert(b.overlaps(v,600,300,0,12));assert(b.overlaps(v,600,300,.5,NAN));
 v.receivers[0].left=NAN;assert(b.overlaps(v,600,300,.5,12));
 auto mesh=fixture();mesh.vertices[0].weights[0]=-1;SkinShadowBounds boxes;boxes.prepare(mesh);assert(!boxes.known);
 std::vector<UnitShadow::Point> empty;unknown.append(empty,0,false,1,1,0);assert(!unknown.complete);
}
''')

    def test_actual_prepare_refines_only_reflections_and_rechecks_cached_pose_at_new_anchor(self):
        host_cpp(direct_stub()+r'''
int main(){Direct direct;setup(direct);c3x_renderer_frame_v1 frame{};frame.target_width=1000;frame.target_height=800;
 frame.tile_width=128;frame.tile_height=64;frame.presentation_frequency=1000;frame.presentation_time_ticks=1;
 std::vector<ScenePose> poses={occurrence(1,500,400),occurrence(2,900,300),occurrence(1,-4620,400)};
 UnitContributionPlan plan;plan.entries={{0,7},{1,7},{2,7}};plan.view.width=1000;plan.view.height=800;plan.view.reflection=true;
 plan.view.receivers={{480,440,530,475}};
 assert(direct.prepare_real(frame,poses,12,plan));
 assert(direct.main_contributors==3&&direct.shadow_contributors==3&&direct.reflection_contributors==1);
 assert(direct.reflection_bounds_builds==1&&direct.reflection_bounds_reuses==2&&direct.reflection_bounds_rejected==2);
 assert(direct.prepared_units[0].reflected&&!direct.prepared_units[1].reflected&&!direct.prepared_units[2].reflected);
 Target target;assert(direct.draw_real(frame,poses,target,1,12,true,1));
 assert(renderer.context->bodies==1);assert(direct.draw_real(frame,poses,target,1,12,false,1));
 assert(renderer.context->bodies==4&&renderer.context->shadows==3); // all main/shadow draws intact
 // Camera/occurrence shifts must not reuse the previous screen-space decision.
 poses[0].draw.body_x+=400;poses[1].draw.body_x-=400;poses[1].draw.body_y+=100;
 ++frame.presentation_time_ticks;assert(direct.prepare_real(frame,poses,12,plan));
 assert(direct.reflection_bounds_builds==0&&direct.reflection_bounds_reuses==3&&direct.reflection_contributors==1);
 assert(!direct.prepared_units[0].reflected&&direct.prepared_units[1].reflected);
 plan.view.receivers.clear();++frame.presentation_time_ticks;assert(direct.prepare_real(frame,poses,12,plan));
 assert(direct.reflection_contributors==0&&direct.main_contributors==3&&direct.shadow_contributors==3);
}
''',sources=("Renderer/native/environment_runtime.cpp",))

    def test_actual_frame_attachment_scale_and_blended_palette_are_in_certificate(self):
        host_cpp(direct_stub()+r'''
int main(){Direct direct;setup(direct);c3x_renderer_frame_v1 frame{};frame.target_width=1000;frame.target_height=800;
 frame.tile_width=128;frame.tile_height=64;frame.presentation_frequency=1000;frame.presentation_time_ticks=1;
 // At frame 0 the body is inland; frame 1 translates to the receiver.
 auto mesh=fixture();mesh.palettes[28]=4;renderer.unit_bodies.meshes[0].animation=std::make_shared<AnimationMesh const>(mesh);
 direct.meshes[0].shadow_bounds.prepare(mesh);
 renderer.unit_bodies.units[0].actions[0].loop=false;
 std::vector<ScenePose> poses={occurrence(1,500,400)};poses[0].draw.direction=8;
 UnitContributionPlan plan;plan.entries={{0,7}};plan.view.width=1000;plan.view.height=800;plan.view.reflection=true;
 plan.view.receivers={{810,550,840,580}};
 assert(direct.prepare_real(frame,poses,12,plan));assert(direct.reflection_contributors==0);
 poses[0].draw.action_cursor=9;++frame.presentation_time_ticks;
 assert(direct.prepare_real(frame,poses,12,plan));assert(direct.reflection_contributors==1&&direct.reflection_bounds_builds==1);
 // A separate attachment reaches water even when the main part does not.
 poses[0].draw.action_cursor=0;renderer.unit_bodies.meshes.resize(2);direct.meshes.resize(2);
 auto attachment=fixture(4);renderer.unit_bodies.meshes[1].animation=std::make_shared<AnimationMesh const>(attachment);
 direct.meshes[1]=direct.meshes[0];direct.meshes[1].shadow_bounds.prepare(attachment);
 auto part=renderer.unit_bodies.units[0].actions[0].parts[0];part.mesh=1;
 renderer.unit_bodies.units[0].actions[0].parts.push_back(part);
 ++frame.presentation_time_ticks;assert(direct.prepare_real(frame,poses,12,plan));
 assert(direct.part_samples==2&&direct.reflection_contributors==1);
 // Scale/offset changes are part of the exact cached sample; they cannot reuse old coverage.
 renderer.unit_bodies.units[0].scale=.2f;++frame.presentation_time_ticks;
 assert(direct.prepare_real(frame,poses,12,plan));assert(direct.reflection_contributors==0&&direct.reflection_bounds_builds==1);
 renderer.unit_bodies.units[0].scale=1;renderer.unit_bodies.units[0].offset_z=5;++frame.presentation_time_ticks;
 assert(direct.prepare_real(frame,poses,12,plan));assert(direct.reflection_contributors==0&&direct.reflection_bounds_builds==1);
 // Blended joint matrices, not their authored endpoint frame, define current coverage.
 auto mixed=fixture();mixed.rig.binding[0]=9;mixed.rig.parents={-1};mixed.rig.skin_joints={0};
 mixed.rig.inverse_bind={JointMatrix{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1}};
 JointPose p;p.scale={1,0,0,0,1,0,0,0,1};p.rotation={0,0,0,1};mixed.rig.poses={p,p};mixed.rig.poses[1].position[0]=4;
 UnitPoseTransitions t;assert(!t.sample(7,1,1,0,1000,mixed,0,11));
 auto* palette=t.sample(7,1,2,1,1000,mixed,1,22);assert(palette);
 palette=t.sample(7,1,2,61,1000,mixed,1,22);assert(palette&&std::abs(palette[12]-2)<1e-4);
 SkinShadowBounds boxes;boxes.prepare(mixed);std::vector<UnitShadow::Point> points;boxes.append(palette,0,1,0,points);
 UnitReflectionBounds b;b.append(points,0,boxes.known,boxes.weight_low,boxes.weight_high,0);
 assert(b.local.left<192&&b.local.right>192); // interpolated X=(1+2)*64
}
''',sources=("Renderer/native/environment_runtime.cpp",))

    def test_actual_missing_certificate_overflow_and_slot_replacement_keep_valid_draws(self):
        host_cpp(direct_stub()+r'''
int main(){Direct direct;setup(direct);c3x_renderer_frame_v1 frame{};frame.target_width=1000;frame.target_height=800;
 frame.tile_width=128;frame.tile_height=64;frame.presentation_frequency=1000;frame.presentation_time_ticks=1;
 auto original=renderer.unit_bodies.units[0];renderer.unit_bodies.units.resize(65,original);
 std::vector<ScenePose> poses;UnitContributionPlan plan;plan.view.width=1000;plan.view.height=800;plan.view.reflection=true;
 plan.view.receivers={{0,0,10,10}};
 for(unsigned i=0;i<65;++i){auto p=occurrence(i+1,500,400);p.unit=i;poses.push_back(p);plan.entries.push_back({i,7});
  renderer.unit_bodies.units[i].scale=1+i*.01f;}
 assert(direct.prepare_real(frame,poses,12,plan));assert(direct.shadow_overflow==1&&direct.reflection_contributors==0);
 assert(direct.reflection_bounds_builds==65&&direct.reflection_bounds_rejected==65);
 // Unknown metadata retains reflections despite non-overlap, never invents a small bound.
 direct.meshes[0].shadow_bounds.known=false;++renderer.unit_bodies.catalogue_generation;
 ++frame.presentation_time_ticks;assert(direct.prepare_real(frame,poses,12,plan));
 assert(direct.reflection_contributors==65&&direct.reflection_bounds_rejected==0&&direct.shadow_overflow==1);
 Target target;assert(direct.draw_real(frame,poses,target,1,12,true,1));assert(renderer.context->bodies==65);
 // Replaced cache slots must not retain the unknown certificate or previous pose rectangle.
 direct.meshes[0].shadow_bounds.known=true;++renderer.unit_bodies.catalogue_generation;
 ++frame.presentation_time_ticks;assert(direct.prepare_real(frame,poses,12,plan));
 assert(direct.reflection_contributors==0&&direct.reflection_bounds_rejected==65);
 assert(direct.main_contributors==65&&direct.shadow_contributors==65);
}
''',sources=("Renderer/native/environment_runtime.cpp",))

    def test_actual_workload_witness_separates_authoritative_order_source_and_pass_masks(self):
        source=(Path(__file__).resolve().parents[2]/"Renderer/sandbox/resident_scene.cpp").read_text()
        begin=source.index("        std::uint64_t facts=14695981039346656037ull,poses=facts,ordered=facts,source=facts;")
        end=source.index("        char detail[896];",begin)
        witness=source[begin:end]
        host_cpp(BASE+r'''
#include "Renderer/native/render_core/unit_instances.h"
struct Record {c3x_renderer_unit_v1 draw{};UnitInstances::ScenePose instance{};bool main=true,shadow=true,reflected=true;};
struct {std::vector<Record> prepared_units;} sandbox_direct_units;
struct Witness {std::uint64_t facts,poses,ordered,source;bool complete;};
Witness emit(c3x_renderer_frame_v1 const& frame){
'''+witness+r'''
 return {facts,poses,ordered,source,source_complete};
}
int main(){
 c3x_renderer_tile_v1 tiles[2]{};tiles[0].tile_x=4;tiles[1].tile_x=6;
 c3x_renderer_frame_v1 frame{};frame.tiles=tiles;frame.tile_count=2;frame.tile_width=128;frame.tile_height=64;
 Record a;a.draw.unit_id=1;a.draw.action=1;a.draw.body_x=20;a.instance.unit=4;a.instance.action=3;
 Record b=a;b.draw.unit_id=2;b.draw.body_x=40;sandbox_direct_units.prepared_units={a,b};
 auto old=emit(frame);assert(old.complete);
 sandbox_direct_units.prepared_units[0].reflected=false;auto removed=emit(frame);
 assert(old.ordered==removed.ordered&&old.source==removed.source&&old.facts!=removed.facts);
 sandbox_direct_units.prepared_units[0].draw.action_cursor=9;
 sandbox_direct_units.prepared_units[0].draw.presentation_time_ticks=100;auto sampled=emit(frame);
 assert(sampled.ordered==removed.ordered&&sampled.poses!=removed.poses);
 sandbox_direct_units.prepared_units[0].instance.action=5;assert(emit(frame).ordered!=sampled.ordered);
 sandbox_direct_units.prepared_units[0].instance.action=3;
 std::swap(sandbox_direct_units.prepared_units[0],sandbox_direct_units.prepared_units[1]);assert(emit(frame).ordered!=sampled.ordered);
 c3x_renderer_tile_v1 copied[2]={tiles[0],tiles[1]};frame.tiles=copied;frame.presentation_time_ticks=500;
 frame.presentation_frequency=1000;frame.dirty_flags=4;frame.visible_animation_count=1;
 assert(emit(frame).source==sampled.source); // no address/clock/scheduling identity
 copied[0].anchor_x=1;assert(emit(frame).source!=sampled.source);copied[0].anchor_x=0;
 copied[0].road_mask=1;assert(emit(frame).source!=sampled.source);copied[0].road_mask=0;
 std::swap(copied[0],copied[1]);assert(emit(frame).source!=sampled.source);
 frame.tiles=nullptr;assert(!emit(frame).complete);
 frame.tile_count=0;frame.world_topology_count=1;assert(!emit(frame).complete);
 frame.world_topology_count=0;frame.tile_count=8193;assert(!emit(frame).complete);
}
''')

if __name__=='__main__':unittest.main()
