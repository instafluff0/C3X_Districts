"""Executable invalidation and bounded multi-pass sample ownership contracts."""
import unittest
from Renderer.native.native_cpp_test import run_cpp
from Renderer.lab.platform import ROOT

class SharedPassOwnershipTests(unittest.TestCase):
    def test_production_selection_reopens_zoomed_out_edges_without_world_edits(self):
        source=(ROOT/"Renderer/sandbox/fresh_pipeline.h").read_text()
        import re
        declaration=re.search(r"    std::array<float,\d+> visibility_view_key\{\};",source).group()
        gate="        std::array<std::uint64_t,6> scene_key="+source.split("        std::array<std::uint64_t,6> scene_key=",1)[1].split("        static_visible={};",1)[0]
        run_cpp(r'''
#define C3X_RENDERER64_FRESH
#include "Renderer/native/scene_projection.h"
#include <array>
#include <cassert>
#include <cstdint>
struct ViewportShaderSettings {float translation[2]={};};
struct SandboxPassWorkload {enum {selection=0,screen=0};};
struct State {
 struct {bool enabled=false;enum {selection,screen};struct {unsigned reuses=0,rebuilds=0;}counts[1][1];}work;
 struct {bool water_scene_active=true;struct {float height_pixels=1260;}reflection;
  struct {unsigned visibility_sequence(){return 3;}}topology_cache;}renderer;
 std::uint64_t view_revision(){return 7;}
 bool visibility_valid=false;unsigned visible=1;
 std::array<std::uint64_t,6> visibility_scene_key{};
'''+declaration+r'''
 float projection_zoom=3.f;unsigned selections=0;int source_left=0;
 bool capture(ViewportShaderSettings settings,ViewportShaderSettings reflected,int width,int height,int next_wrap_pixels){
'''+gate+r'''
  struct Rect {int left,top,right,bottom;};
  source_left=c3x_renderer::SceneProjection(width,height,projection_zoom).source_rect(Rect{0,0,width,height}).left;
  visibility_scene_key=scene_key;visibility_view_key=view_key;visibility_valid=true;++selections;return true;
 }
};
int main(){State state;ViewportShaderSettings main{},mirror{};
 assert(state.capture(main,mirror,2240,1260,8192));assert(state.selections==1 && state.source_left>700);
 assert(state.capture(main,mirror,2240,1260,8192));assert(state.selections==1);
 // Same generation, camera, native visibility and wrap; a reversal opens
 // contributors near the viewport edge for both main and reflected passes.
 state.projection_zoom=1.f;assert(state.capture(main,mirror,2240,1260,8192));
 assert(state.selections==2 && state.source_left==0);
 assert(state.capture(main,mirror,2240,1260,8192));assert(state.selections==2);
 state.projection_zoom=2.f;assert(state.capture(main,mirror,2240,1260,8192));assert(state.selections==3);
}
''')

    def test_exact_pose_dependencies_pin_consumers_and_allow_bounded_eviction(self):
        run_cpp(r'''
#include "Renderer/native/render_core/frame_sample_cache.h"
#include <array>
#include <cassert>
using namespace c3x_renderer::render_core;
int main(){
 FrameSampleCache<std::array<int,4>,int,2> cache;
 // Key represents mesh, sampled palette, facing, and light, not native ID.
 std::array<int,4> a={10,1,90,12},b={11,2,45,12},c={12,3,180,12};
 cache.begin();auto first=cache.select(a);cache[first].value=73;cache[first].valid=true;
 auto second=cache.select(b);cache[second].value=81;cache[second].valid=true;
 assert(cache.select(a)==first && cache[first].value==73);
 assert(cache.select(c)==cache.unavailable && cache.size()==2);
 // Main/reflected consumers cannot lose a pinned result to another admission.
 assert(cache[first].valid && cache[second].valid);
 cache.begin();assert(cache.select(a)==first && cache[first].valid);
 auto replacement=cache.select(c);assert(replacement==second && !cache[replacement].valid);
 assert(cache[first].value==73);
 auto changed=a;changed[1]=2;assert(cache.select(changed)==cache.unavailable);
 cache.begin();auto new_pose=cache.select(changed);assert(new_pose!=cache.unavailable && !cache[new_pose].valid);
 auto new_light=changed;new_light[3]=0;auto night=cache.select(new_light);
 assert(night!=cache.unavailable && night!=new_pose && !cache[night].valid);
}
''')

    def test_retained_pixels_keep_resource_free_proofs_and_reject_local_visibility_changes(self):
        run_cpp(r'''
#include "Renderer/native/render_core/raster_contributors.h"
#include <cassert>
struct Proof {int revision;unsigned* frees;~Proof(){++*frees;}};
using namespace c3x_renderer::render_core;
int main(){
 RasterContributors<Proof> pixels;unsigned frees=0;int world=7;unsigned visibility=3;
 auto proof=std::make_shared<Proof>();proof->revision=world;proof->frees=&frees;
 RasterContributors<Proof>::Key existing{};existing[0]=17;existing[1]=8;
 assert(pixels.add(existing,proof,55,visibility));proof.reset();
 assert(pixels.valid([&](auto const& p){return p.revision==world;},[&](auto){return visibility;}));
 assert(pixels.contains(existing));auto new_strip=existing;new_strip[0]=18;
 // Camera membership outside a covered region is not a pixel dependency.
 assert(!pixels.contains(new_strip) && pixels.contains(existing) && frees==0);
 ++visibility;assert(!pixels.valid([&](auto const& p){return p.revision==world;},[&](auto){return visibility;}));
 --visibility;++world;assert(!pixels.valid([&](auto const& p){return p.revision==world;},[&](auto){return visibility;}));
 pixels.clear();assert(frees==1 && pixels.complete);
}
''')

    def test_production_pixel_proof_separates_native_basis_and_retained_world_authority(self):
        source=(ROOT/"Renderer/native/c3x_renderer.cpp").read_text()
        validate="    std::array<unsigned,7> raster_proof_rejections"+source.split("    std::array<unsigned,7> raster_proof_rejections",1)[1].split("    bool tile_content_valid(",1)[0]
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <array>
#include <vector>
using namespace c3x_renderer::render_core;
struct CachedGeometryProof {
 std::uint64_t scope=1,assets=7;
 std::vector<std::pair<std::uint64_t,std::uint64_t>> appearance_dependencies,dependencies,coast_dependencies,world_dependencies;
 std::vector<std::pair<std::uint64_t,std::array<int,2>>> anchor_dependencies;
 int river_dependencies=0;
};
struct State {
 CapturedScene topology_cache;unsigned content_revision=7;int shadow_tile_width=128,shadow_tile_height=64;
 std::vector<c3x_renderer_tile_v1> cached_tiles;
 unsigned tile_topology_signature(c3x_renderer_tile_v1 const& tile){return tile.road_mask+100;}
 struct World {unsigned node_revision(std::uint64_t){return 1;}World& world(){return *this;}unsigned at(std::uint64_t){return 2;}}world_coast;
 struct Natural {bool valid(int value){return value==0;}}natural;
'''+validate+r'''
};
int main(){
 State state;c3x_renderer_tile_v1 a{};a.tile_x=2;a.tile_y=2;a.anchor_x=135;a.anchor_y=55;
 a.tile_flags=C3X_RENDERER_TILE_RENDER;a.road_mask=5;
 auto b=a;b.tile_x=4;b.anchor_x+=128;
 c3x_renderer_frame_v1 frame{};frame.world_width_tiles=frame.world_height_tiles=64;
 frame.world_wrap_x=frame.world_wrap_y=1;c3x_renderer_camera_identity_v1 identity{};
 state.topology_cache.publication_scope(frame,identity,1);
 bool changed=false;assert(state.topology_cache.publish(a,changed));assert(state.topology_cache.publish(b,changed));
 std::vector<c3x_renderer_tile_v1> observed={a,b};
 auto submit=[&]{frame.tiles=observed.data();frame.tile_count=unsigned(observed.size());
  assert(state.topology_cache.begin(frame));for(auto const& tile:observed)
   assert(state.topology_cache.update(tile,2,2,2,tile.road_mask+100));state.topology_cache.finish();
  state.cached_tiles=observed;};submit();
 CachedGeometryProof proof;proof.scope=state.topology_cache.scope_sequence();
 auto neighbor=state.topology_cache.key(4,2);proof.anchor_dependencies={{neighbor,{128,0}}};proof.dependencies={{neighbor,CapturedScene::topology(b)}};
 assert(state.raster_content_valid(proof));
 // Camera motion and an alternate wrapped occurrence retain the same world basis.
 for(auto& tile:observed){tile.anchor_x+=37;tile.anchor_y-=21;}
 observed[1].tile_x-=64;observed[1].anchor_x-=4096;submit();assert(state.raster_content_valid(proof));
 // Pixel world proofs do not consult a different preparation camera. The
 // exact native anchor closure remains in tile_content_valid, and its new
 // generation/occurrence key is checked below.
 ++observed[1].anchor_x;submit();assert(state.raster_content_valid(proof));--observed[1].anchor_x;
 // The authoritative record proves an off-screen semantic input; departure
 // does not authorize a stale value after a local published edit.
 observed.pop_back();submit();assert(state.raster_content_valid(proof));
 ++b.road_mask;assert(state.topology_cache.publish(b,changed));assert(!state.raster_content_valid(proof));
 --b.road_mask;assert(state.topology_cache.publish(b,changed));assert(state.raster_content_valid(proof));
 auto halo=b;halo.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO;++halo.road_mask;
 assert(state.topology_cache.publish(halo,changed));assert(!state.raster_content_valid(proof));
 assert(state.topology_cache.publish(b,changed));assert(state.raster_content_valid(proof));
 observed.push_back(b);observed[1].anchor_x=observed[0].anchor_x+128;observed[1].anchor_y=observed[0].anchor_y;
 submit();assert(state.raster_content_valid(proof));
 proof.river_dependencies=1;assert(!state.raster_content_valid(proof));
 assert(state.raster_proof_rejections[2]==2 && state.raster_proof_rejections[5]==0 && state.raster_proof_rejections[6]==1);
}
''')

    def test_production_normalized_occurrence_key_rejects_changed_native_anchors_and_generations(self):
        source=(ROOT/"Renderer/sandbox/fresh_pipeline.h").read_text()
        key="    RasterInputs::Key contributor_key("+source.split("    RasterInputs::Key contributor_key(",1)[1].split("    bool raster_dependencies(",1)[0]
        run_cpp(r'''
#include "Renderer/native/render_core/raster_contributors.h"
#include <cassert>
#include <cstring>
struct GeometryDrawRecord {
 struct {std::uint64_t generation=9;}owner;unsigned ordinal=1;int tile_x=2,tile_y=4,translation_x=128,translation_y=64;
 float natural_projection[4]={3,-1,128,1260};struct {int left=0,top=0,right=10,bottom=20;}bounds;
 unsigned territory_rgb=17,territory_edges=3;bool water_dependent=false,water_visible=true;
};
struct Proof {};
struct State {using RasterInputs=c3x_renderer::render_core::RasterContributors<Proof>;
'''+key+r'''
};
int main(){State state;GeometryDrawRecord record;auto proof=std::make_shared<Proof>();
 State::RasterInputs pixels;auto original=state.contributor_key(4,record);
 assert(pixels.add(original,proof,1,2));assert(pixels.contains(state.contributor_key(4,record)));
 // Camera transforms are borrowed settings; a different native anchor basis
 // or recompiled content changes the actual contributing key.
 ++record.translation_x;assert(!pixels.contains(state.contributor_key(4,record)));--record.translation_x;
 ++record.owner.generation;assert(!pixels.contains(state.contributor_key(4,record)));--record.owner.generation;
 record.natural_projection[2]=64;assert(!pixels.contains(state.contributor_key(4,record)));
}
''')

if __name__=='__main__':unittest.main()
