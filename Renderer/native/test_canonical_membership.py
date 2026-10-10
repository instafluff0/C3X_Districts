"""Exact canonical membership coverage, native representatives and source leases."""
import unittest
from Renderer.native.native_cpp_test import ROOT, run_cpp

class CanonicalMembershipTests(unittest.TestCase):
    def test_occurrence_diff_resource_retirement_and_native_depth_basis(self):
        run_cpp(r'''#include "Renderer/native/render_core/canonical_membership_diff.h"
#include "Renderer/native/render_core/scene_membership.h"
#include "Renderer/native/render_core/scene_depth.h"
#include <cassert>
#include <array>
#include <vector>
#include <algorithm>
#include <iostream>
using namespace c3x_renderer::render_core;
c3x_renderer_tile_v1 tile(int x,int y,int ax,int ay,unsigned flags=C3X_RENDERER_TILE_RENDER){
    c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;t.anchor_x=ax;t.anchor_y=ay;
    t.terrain_type=t.real_terrain_type=2;t.city_id=t.city_owner_id=t.resource_id=-1;
    t.tile_flags=flags|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;return t;
}
auto same=[](auto const& old,auto const& current){
    auto a=CapturedScene::content(old),b=CapturedScene::content(current);
    constexpr unsigned eligibility=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;
    return (old.tile_flags&eligibility)==(current.tile_flags&eligibility) && !std::memcmp(&a,&b,sizeof(a));
};
struct Chunk {struct Bounds {int left,top,right,bottom;} bounds{};int translation_x=0,translation_y=0;float natural_projection[4]={};};
struct Payload {unsigned* frees;~Payload(){++*frees;}};
void exact_diff(){
    std::vector<c3x_renderer_tile_v1> old={tile(0,0,-64,-32,C3X_RENDERER_TILE_PREFETCH),tile(2,0,64,-32),tile(4,0,192,-32),tile(6,0,320,-32,C3X_RENDERER_TILE_PREFETCH)};
    ForegroundSelection selection{256,128,128,64,2,0,true,false};
    auto admitted=[](unsigned){return true;};CanonicalMembershipDiff diff;
    std::vector<c3x_renderer_tile_v1> current={old[2],old[0],old[1]};
    for(auto& t:current){t.anchor_x+=17;t.anchor_y+=9;t.unit_type_id=123;t.unit_state=17;}
    auto frame=c3x_renderer_frame_v1{};frame.tiles=current.data();frame.tile_count=unsigned(current.size());
    assert(diff.build(old,frame,selection,same,admitted));
    assert(!diff.covered() && diff.required==3 && diff.leaving==1 && diff.translation_x==17 && diff.translation_y==9);
    assert(diff.previous[0]==2 && diff.previous[1]==0 && diff.previous[2]==1);
    // Identical owners in a changed native order cannot retain alpha pixels.
    std::rotate(current.begin(),current.begin()+1,current.end());
    assert(diff.build(old,frame,selection,same,admitted) && diff.covered());
    assert(diff.previous[0]==0 && diff.previous[1]==1 && diff.previous[2]==2);
    // RENDER/PREFETCH role changes do not alter immutable bodies; native flags
    // and dynamic representative admission still come from the current frame.
    current[1].tile_flags^=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH;
    assert(diff.build(old,frame,selection,same,admitted) && diff.covered());
    // Every selected new guard enters the exact world-caster membership,
    // even when its native screen anchor is outside the target rectangle.
    current.push_back(tile(8,0,465,-23,C3X_RENDERER_TILE_PREFETCH));frame.tiles=current.data();frame.tile_count=4;
    assert(diff.build(old,frame,selection,same,admitted) && !diff.covered());
    assert(diff.required==4 && diff.entering==1 && diff.leaving==1);
    current[0].city_size=2;assert(diff.build(old,frame,selection,same,admitted) && diff.entering==2);
    current[0]=old[0];current[0].anchor_x+=17;current[0].anchor_y+=9;
    current[0].tile_flags&=~C3X_RENDERER_TILE_EXPLORED;
    // Losing body permission retires the old occurrence; it cannot enter
    // merely because the native fog traversal still provides its anchor.
    assert(diff.build(old,frame,selection,same,admitted));
    assert(diff.required==3 && diff.entering==1 && diff.leaving==2);
    assert(diff.previous[0]==CanonicalMembershipDiff::absent && !diff.keep[0]);
    current[0]=old[0];current[0].anchor_x+=17;current[0].anchor_y+=10;
    assert(!diff.build(old,frame,selection,same,admitted)); // inconsistent native lattice.
    current[0].anchor_y-=1;
    assert(diff.build(old,frame,selection,same,admitted));diff.reject(0);assert(diff.entering==2 && !diff.keep[0]);
    // Unwrapped occurrences remain independent even when their canonical
    // identity would alias at the map seam.
    current[0].tile_x+=64;assert(diff.build(old,frame,selection,same,admitted) && diff.entering==2);
    current[0]=old[0];current[0].anchor_x+=17;current[0].anchor_y+=9;
    auto sparse=[](unsigned index){return index!=0;};
    assert(diff.build(old,frame,selection,same,sparse) && diff.entering==2);
    // A retained normalized basis remains valid after many boundary updates;
    // admitted metadata, rather than a stale screen-position filter, owns it.
    for(auto& t:old)t.anchor_x-=1024;
    for(auto& t:current)t.anchor_x-=1024;
    assert(diff.build(old,frame,selection,same,admitted));
}
void ownership_diff(){
    auto budget=std::make_shared<ResidentRetirement>();SceneMembership<Chunk,2> membership(budget);
    std::array<unsigned,3> frees{};std::array<std::shared_ptr<Payload>,3> meshes;
    std::array<Chunk,3> chunks;
    for(unsigned i=0;i<3;++i){meshes[i]=std::make_shared<Payload>();meshes[i]->frees=&frees[i];
        assert(membership.retain({i,i+1},meshes[i]));GeometryDrawRecord<Chunk> record(chunks[i]);
        record.owner={i,i+1};record.tile_x=int(i)*2;record.tile_y=0;record.ordinal=0;
        membership.edit(i%2).push_back(record);
    }
    auto old=membership.publish();auto revision=membership.revision();auto bytes=budget->bytes.load();
    // Covered camera updates read the same generation, with no new leases.
    assert(membership.publish()==old && membership.revision()==revision && budget->bytes==bytes);
    membership.retain_occurrences([](auto const& draw){return draw.tile_x!=0;});
    auto boundary=membership.publish();assert(boundary!=old && boundary->revision>revision);
    assert(old->records[0].size()==2 && boundary->records[0].size()==1 && boundary->content.size()==2);
    meshes[0].reset();assert(!frees[0]);old.reset();assert(frees[0]==1);
    // The entering source is inserted into the exact current native order;
    // saved readers retain the old order and distinct generation.
    GeometryDrawRecord<Chunk> enter(chunks[0]);enter.owner={4,4};enter.tile_x=0;
    auto new_mesh=std::make_shared<Payload>();new_mesh->frees=&frees[0];assert(membership.retain(enter.owner,new_mesh));
    membership.edit(0).push_back(enter);
    assert(membership.order_occurrences([](auto const& draw){return unsigned(draw.tile_x);}));
    auto ordered=membership.publish();assert(ordered->records[0][0].tile_x==0 && ordered->records[0][1].tile_x==4);
    assert(boundary->records[0][0].tile_x==4);
    boundary.reset();ordered.reset();membership.clear();for(auto& mesh:meshes)mesh.reset();new_mesh.reset();
    assert(frees[0]==2 && frees[1]==1 && frees[2]==1 && !budget->bytes);
}
void depth_basis(){
    // Geometry uses one normalized camera basis. After a boundary diff, new
    // records are normalized before insertion; camera/depth constants restore
    // the exact native projection. Exercise source/wrap origin changes too.
    for(int row:{-130,-1,0,25,129,130,4096})for(int dy:{-129,-64,-17,0,19,64,129}){
        int anchor=630+row*32;auto exact=scene_depth_basis(anchor+dy,row,64,1260,0);
        auto retained=scene_depth_basis(anchor+dy,row,64,1260,dy);
        assert(exact.world_origin==retained.world_origin);
        // Stored records retain their former depth; the normalized camera
        // constant difference equals the native anchor displacement exactly.
        assert(retained.translation-exact.translation==float(dy));
        int entering_anchor=anchor+dy+64,normalized=entering_anchor-dy;
        assert(normalized+dy==entering_anchor);
    }
}
int main(){exact_diff();ownership_diff();depth_basis();std::cout<<"CANONICAL_MEMBERSHIP_HOST_PASS diff,guard,wrapped,visibility,ownership,depth\n";}
''')

    def test_production_admission_keeps_exact_current_caster_membership(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        begin=source.index('        c3x_renderer::render_core::CanonicalMembershipDiff membership_diff;')
        end=source.index('        if(reuse_geometry && fresh_scene_path && !covered_membership){',begin)
        run_cpp(r'''
#include "Renderer/native/render_core/canonical_membership_diff.h"
#include <vector>
#include <array>
#include <memory>
#include <cassert>
#include <cstring>
#include <iostream>
using SceneTopology=c3x_renderer::render_core::CapturedScene;
using Selection=c3x_renderer::render_core::ForegroundSelection;
struct Rect {long left,top,right,bottom;};
struct LARGE_INTEGER {long long QuadPart=0;};
void QueryPerformanceCounter(LARGE_INTEGER* value){value->QuadPart=1;}
struct Handle {unsigned generation=0;};
struct Proof {unsigned scope=1,assets=2;bool valid=true;};
struct Mesh {std::shared_ptr<Proof> proof=std::make_shared<Proof>();};
struct Owner {bool world_ground=true,world_objects=true;unsigned invalid=0;std::array<std::uint64_t,20> compile_context{};
 std::shared_ptr<Mesh> mesh=std::make_shared<Mesh>();Handle natural_content{};std::vector<int> anchor_dependencies;};
struct Signature {unsigned camera=1,environment=1,wrap=1;};
struct Cache {bool valid=true;Selection selection;Signature signature;std::vector<Handle> tile_keys;std::vector<unsigned> content_replacement_flags;
 std::vector<c3x_renderer_tile_v1> tiles;Rect coverage_bounds={-128,-64,768,384};};
struct Scene {
 struct Record {std::uint64_t revision=1,recipe_revision=5;unsigned visibility_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;};
 std::unordered_map<std::uint64_t,Record> records;
 unsigned scope_sequence()const{return 1;}
 std::uint64_t world_appearance_revision(std::uint64_t key)const{auto f=records.find(key);return f==records.end()?0:f->second.recipe_revision;}
 std::uint64_t key(int x,int y)const{return c3x_renderer::render_core::CanonicalMembershipDiff::occurrence(x,y);}
 Record const* retained(std::uint64_t key)const{auto f=records.find(key);return f==records.end()?nullptr:&f->second;}
};
struct Harness {
 bool prewarming=false,fresh_scene_path=true,canonical_world_content=true,world_ground=true,world_objects=true;
 unsigned content_revision=2,device_generation=3;int geometry_translation_x=0,geometry_translation_y=0;
 Selection selection{640,320,128,64,2,0,true,false};Signature signature{};Cache geometry_cache;Scene topology_cache;
 struct Content {std::vector<Owner*> owners;Owner* resolve(Handle h){return h.generation && h.generation<owners.size()?owners[h.generation]:nullptr;}}resident_content;
 std::array<std::uint64_t,2> compile_quality={11,12};
 bool tile_content_valid(Owner const& owner,c3x_renderer_tile_v1 const&){return !owner.invalid && owner.mesh->proof->valid;}
 bool raster_content_valid(Proof const& proof){return proof.valid;}
 struct Result {bool covered,incremental;unsigned entering,leaving;};
 Result admit(c3x_renderer_frame_v1 const& frame){bool reuse_geometry=false;std::array<LARGE_INTEGER,8> setup_marks{};
''' + source[begin:end] + r'''
   return {covered_membership,incremental_membership,membership_diff.entering,membership_diff.leaving};
 }
};
c3x_renderer_tile_v1 tile(int x,int ax,unsigned flags=C3X_RENDERER_TILE_RENDER){
 c3x_renderer_tile_v1 t{};t.tile_x=x;t.anchor_x=ax;t.anchor_y=64;t.tile_flags=flags|C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBILITY_KNOWN;
 t.terrain_type=t.real_terrain_type=2;t.city_id=t.city_owner_id=t.resource_id=-1;return t;
}
int main(){
 Harness h;h.geometry_cache.selection=h.selection;std::array<Owner,4> owners;
 h.resident_content.owners={nullptr,&owners[0],&owners[1],&owners[2],&owners[3]};
 for(auto& owner:owners){owner.compile_context[10]=2;owner.compile_context[11]=3;owner.compile_context[14]=11;owner.compile_context[15]=12;owner.compile_context[17]=5;}
 h.geometry_cache.tiles={tile(0,0),tile(2,128),tile(4,256),tile(6,384)};
 h.geometry_cache.tile_keys={{1},{2},{3},{4}};h.geometry_cache.content_replacement_flags={1,1,1,1};
 std::vector<c3x_renderer_tile_v1> current=h.geometry_cache.tiles;
 for(auto const& t:current)h.topology_cache.records[h.topology_cache.key(t.tile_x,t.tile_y)]={};
 c3x_renderer_frame_v1 frame{};frame.tiles=current.data();frame.tile_count=current.size();frame.target_width=640;frame.target_height=320;
 auto result=h.admit(frame);assert(result.covered && result.incremental && !result.entering && !result.leaving);
 // A changed native order retires alpha pixels while preserving all owners.
 std::swap(current[0],current[1]);result=h.admit(frame);
 assert(!result.covered && result.incremental && !result.entering && !result.leaving);
 std::swap(current[0],current[1]);result=h.admit(frame);assert(result.covered);
 // Native dynamic/RENDER roles do not replace equal immutable body owners.
 current[1].tile_flags^=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH;
 current[1].unit_state=17;result=h.admit(frame);assert(result.covered);
 ++h.topology_cache.records[h.topology_cache.key(0,0)].revision;
 result=h.admit(frame);assert(result.covered);
 ++h.topology_cache.records[h.topology_cache.key(0,0)].recipe_revision;
 result=h.admit(frame);assert(!result.covered && result.entering==1 && result.leaving==1);
 --h.topology_cache.records[h.topology_cache.key(0,0)].recipe_revision;
 // Additional native facts may make a permitted-world identity zero; exact
 // captured content plus the matching complete owner still remain reusable.
 h.topology_cache.records[h.topology_cache.key(0,0)].recipe_revision=0;owners[0].compile_context[17]=0;
 result=h.admit(frame);assert(result.covered);
 h.topology_cache.records[h.topology_cache.key(0,0)].recipe_revision=5;owners[0].compile_context[17]=5;
 // Broad stored coverage still permits affine pans of the identical set.
 for(auto& t:current)t.anchor_x-=17;result=h.admit(frame);assert(result.covered);
 for(auto& t:current)t.anchor_x-=200;result=h.admit(frame);assert(!result.covered && result.incremental);
 for(auto& t:current)t.anchor_x+=200;
 // An off-screen selected guard always enters the world-caster membership.
 // A screen body miss cannot prove exclusion from sampled light-space pages.
 current.push_back(tile(8,785,C3X_RENDERER_TILE_PREFETCH));frame.tiles=current.data();frame.tile_count=current.size();
 result=h.admit(frame);assert(!result.covered && result.incremental && result.entering==1 && !result.leaving);
 for(auto& t:current)t.anchor_x-=17;result=h.admit(frame);assert(!result.covered && result.entering==1);
 current.back().tile_flags|=C3X_RENDERER_TILE_RENDER;result=h.admit(frame);assert(!result.covered && result.entering==1);
 for(auto& t:current)t.anchor_x+=17;
 current.pop_back();frame.tiles=current.data();frame.tile_count=current.size();result=h.admit(frame);assert(result.covered);
 // Matched generation proofs retain their existing scope/device/quality and
 // permitted-facts checks, independently of occurrence membership.
 owners[0].mesh->proof->valid=false;result=h.admit(frame);assert(!result.covered && result.entering==1);owners[0].mesh->proof->valid=true;
 ++owners[1].compile_context[11];result=h.admit(frame);assert(!result.covered);--owners[1].compile_context[11];
 ++owners[1].compile_context[14];result=h.admit(frame);assert(!result.covered);--owners[1].compile_context[14];
 ++owners[2].mesh->proof->scope;result=h.admit(frame);assert(!result.covered);--owners[2].mesh->proof->scope;
 h.topology_cache.records[h.topology_cache.key(0,0)].visibility_flags&=~C3X_RENDERER_TILE_EXPLORED;
 result=h.admit(frame);assert(!result.covered);
 h.topology_cache.records[h.topology_cache.key(0,0)].visibility_flags|=C3X_RENDERER_TILE_EXPLORED;
 // Returning removes the formerly selected guard even if its anchor is far
 // outside the viewport: exact caster membership must have no history.
 current.pop_back();frame.tiles=current.data();frame.tile_count=current.size();
 result=h.admit(frame);assert(!result.covered && result.incremental && !result.entering && result.leaving==1);
 h.geometry_cache.tiles[3].anchor_x=785;
 result=h.admit(frame);assert(!result.covered && result.incremental && !result.entering && result.leaving==1);
 h.signature.camera=2;result=h.admit(frame);assert(!result.incremental && !result.covered);
 std::cout<<"CANONICAL_ADMISSION_HOST_PASS exact_guards,affine_coverage,ordering,proofs,representatives,return_prune\n";
}
''')

if __name__=='__main__':unittest.main()
