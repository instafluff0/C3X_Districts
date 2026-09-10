"""Execute production region ownership and complete render dependency keys."""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class RenderRegionTests(unittest.TestCase):
    def test_linear_backdrop_lookup_requires_complete_dependencies_and_preserves_fast_hits(self):
        source=(ROOT/"Renderer/native/c3x_renderer.cpp").read_text()
        lookup="int key_x=rect.left-anchor_x,key_y=rect.top-anchor_y;"+source.split("int key_x=rect.left-anchor_x,key_y=rect.top-anchor_y;",1)[1].split("            context->OMSetRenderTargets",1)[0]
        run_cpp(r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#include <stdexcept>
namespace c3x_renderer {namespace render_core {using RenderRegionKey=std::vector<std::uint64_t>;}}
using Key=c3x_renderer::render_core::RenderRegionKey;
struct ViewportShaderSettings {float translation[2]={},inverse_size[2]={};};
struct Block {int x=0,y=0;std::uint64_t signature=1;Key dependencies;};
struct State {
 std::vector<Block> resource_backdrops={{0,0,1,{10,20,30}}};
 std::uint64_t backdrop_signature=2;
 bool backdrop_reuse_control=false,dependency_backdrops=true;
 unsigned backdrop_dependency_hits=0,backdrop_dependency_rejections=0,calls=0;
 int anchor_x=64,anchor_y=32,geometry_vertex_buffers=0,casters=0,prepared=0;
 int *animation_casters_ptr=&casters,*animation_prepared_ptr=&prepared;
 ViewportShaderSettings geometry_viewport_settings{{64,32},{}};
 Key next_key={10,20,30};int key_result=1;
 bool render_region_key(int,ViewportShaderSettings settings,int,int*,Key& key) {
  ++calls;assert(settings.translation[0]==4 && settings.translation[1]==4);
  assert(settings.inverse_size[0]==1.f/136 && settings.inverse_size[1]==1.f/136);
  key=next_key;if(key_result<0)throw std::bad_alloc();return key_result!=0;
 }
 bool lookup() {
  struct {int left=64,top=32;}rect;
''' + lookup + r'''
  return found!=resource_backdrops.end();
 }
};
int main() {
 State state;assert(state.lookup() && state.calls==1 && state.backdrop_dependency_hits==1);
 state.next_key={10,20};assert(!state.lookup()); // A matching prefix is insufficient.
 state.next_key={10,20,31};assert(!state.lookup()); // A local edit must miss.
 state.next_key={10,20,30};state.key_result=0;assert(!state.lookup() && state.backdrop_dependency_rejections==1);
 state.key_result=-1;assert(!state.lookup() && state.backdrop_dependency_rejections==2);
 state.key_result=1;state.backdrop_reuse_control=true;auto calls=state.calls;
 assert(!state.lookup() && state.calls==calls);
 state.backdrop_reuse_control=false;state.dependency_backdrops=false;assert(!state.lookup());
 state.backdrop_signature=1;assert(state.lookup() && state.calls==calls); // Existing unchanged-view fast path.
 state.resource_backdrops.clear();state.dependency_backdrops=true;assert(!state.lookup());
}
''')

    def test_center_samples_replay_dependencies_and_reject_edits(self):
        run_cpp(r'''
#include "Renderer/native/render_core/center_shore_cache.h"
#include "Renderer/lab/shared/natural/queries.h"
#include <cassert>
using namespace c3x_renderer::render_core;
int main(){
 std::vector<std::uint32_t> values(32*24/2);
 for(int y=0;y<24;++y)for(int x=y&1;x<32;x+=2){unsigned t=((x/8+y/6)%3)==0?11:2;values[(y*32+x)/2]=t|(t<<8);}
 WorldCoast coast;coast.update({32,24,true,true},values.data(),values.size(),1);
 CenterShoreCache cache;
 auto equal=[](ShoreSample a,ShoreSample b){assert(a.distance==b.distance && a.rocky==b.rocky && a.beach_width==b.beach_width && a.depth==b.depth);};
 for(int revision=1;revision<=3;++revision){
  if(revision>1){values[revision*3]=2u|(5u<<8);coast.update({32,24,true,true},values.data(),values.size(),revision);}
  for(int repeat=0;repeat<2;++repeat)for(int x=-4;x<36;x+=2){int y=8;
   std::map<std::size_t,std::uint32_t> observed_world,expected_world;
   std::map<std::uint64_t,std::uint64_t> observed_coast,expected_coast;
   auto w=[&](auto i,auto value){observed_world[i]=value;};auto c=[&](auto i,auto value){observed_coast[i]=value;};
   auto ew=[&](auto i,auto value){expected_world[i]=value;};auto ec=[&](auto i,auto value){expected_coast[i]=value;};
   auto result=cache.get(coast,x,y,w,c);
   equal(result,coast.sample({float(x+y)*.5f+.5f,float(x-y)*.5f+.5f},ec,ew));
   assert(observed_world==expected_world && observed_coast==expected_coast);
   ExactPointCache<ShoreSample> scratch_a,scratch_b;
   c3x_renderer::fidelity::SurfaceQueries a(coast,scratch_a,x,y,ew,ec);
   c3x_renderer::fidelity::SurfaceQueries b(coast,scratch_b,x,y,w,c);
   equal(a.shore(a.center_u,a.center_v),result);b.prime_center(result);
   for(float dx:{-.45f,0.f,.45f})for(float dy:{-.3f,0.f,.3f})equal(a.shore(a.center_u+dx,a.center_v+dy),b.shore(b.center_u+dx,b.center_v+dy));
  }
 }
 assert(cache.hits>0 && cache.bytes<=cache.budget && cache.entries.size()<=cache.entry_limit);
 cache.clear();assert(cache.bytes==0 && cache.entries.empty());
}
''')

    def test_region_index_is_conservative_ordered_and_bounded(self):
        run_cpp(r'''
#include "Renderer/native/render_core/region_contributor_index.h"
#include <cassert>
using Index=c3x_renderer::render_core::RegionContributorIndex;
int main(){
 for(int width:{128,160,192}){
  Index index;std::vector<std::array<double,4>> bounds;
  for(unsigned i=0;i<200;++i){
   double x=int(i%17)*width/2.-700,y=int(i/17)*width/4.-400;
   bounds.push_back({x,y-.03125,x+width+3.25,y+width*1.3});
   assert(index.add(i%2,{i%7,i},bounds.back()[0],bounds.back()[1],bounds.back()[2],bounds.back()[3]));
  }
  index.ready=true;
  for(unsigned pass=0;pass<2;++pass)for(int y=-520;y<650;y+=73)for(int x=-810;x<1000;x+=97){
   std::vector<Index::Item> found;assert(index.query(pass,x,y,144,found));
   assert(std::is_sorted(found.begin(),found.end()));
   assert(std::adjacent_find(found.begin(),found.end())==found.end());
   for(unsigned i=pass;i<bounds.size();i+=2){auto b=bounds[i];
    if(!(b[2]<=x || b[0]>=x+144 || b[3]<=y || b[1]>=y+144))
     assert(std::binary_search(found.begin(),found.end(),Index::Item{i%7,i}));
   }
  }
  assert(index.bytes<=Index::budget);index.clear();assert(!index.ready && index.bytes==0 && index.count==0);
  assert(!index.add(0,{0,0},0,0,1e8,1e8));
  std::vector<Index::Item> found;assert(!index.query(0,0,0,144,found));
 }
}
''')

    def test_animation_backdrop_cannot_hit_bitmap_only_cache(self):
        source=(ROOT/"Renderer/native/c3x_renderer.cpp").read_text()
        predicate=source.split("bool const region_path=",1)[1].split(";",1)[0]
        run_cpp(r'''
#include <cassert>
bool eligible(bool require_linear_backdrop) {
 bool world_regions=true,accumulate=false;int region_size=128;
 void *shadow_buffers_ptr=nullptr;int geometry_vertex_buffers=0;
 auto &buffers=geometry_vertex_buffers;
 return '''+predicate+r''';
}
int main(){assert(eligible(false));assert(!eligible(true));}
''')
        animation=source.split('unsigned backdrop_hits=0,backdrop_misses=0;',1)[1].split('++backdrop_misses;',1)[0]
        self.assertIn('animation_casters_ptr,animation_prepared_ptr,128,0,0,true)',animation)

    def test_support_ring_never_promotes_topology_only_or_uncaptured_inputs(self):
        source=(ROOT/"Renderer/native/c3x_renderer.cpp").read_text()
        predicate="            if (prewarming ?"+source.split("            if (prewarming ?",1)[1].split("            if (cancelled())",1)[0]
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <vector>
#include <cassert>
std::vector<unsigned> select(c3x_renderer_frame_v1 const& frame,int region_input_ring,bool prewarming=false,int prewarm_index=-1){
 bool pickup_profile=true;std::vector<unsigned> result;
 for(c3x_renderer_u32 index=0;index<frame.tile_count;++index){
  auto const& tile=frame.tiles[index];
''' + predicate + r'''
  result.push_back(index);
 }return result;
}
int main(){
 c3x_renderer_tile_v1 tiles[5]={};
 tiles[0].tile_flags=C3X_RENDERER_TILE_RENDER;
 tiles[1].tile_flags=C3X_RENDERER_TILE_PREFETCH;tiles[1].anchor_x=896;
 tiles[2].tile_flags=C3X_RENDERER_TILE_PREFETCH;tiles[2].anchor_x=1152;
 tiles[3].tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO;tiles[3].anchor_x=650;
 tiles[4].tile_flags=C3X_RENDERER_TILE_PREFETCH;tiles[4].anchor_x=768;
 c3x_renderer_frame_v1 frame={};frame.tiles=tiles;frame.tile_count=5;
 frame.target_width=frame.target_height=512;frame.tile_width=128;frame.tile_height=64;
 assert((select(frame,2)==std::vector<unsigned>{0,4}));
 assert((select(frame,4)==std::vector<unsigned>{0,1,4}));
 assert((select(frame,2,true,1)==std::vector<unsigned>{1}));
 assert(tiles[1].tile_flags==C3X_RENDERER_TILE_PREFETCH && tiles[3].tile_flags==C3X_RENDERER_TILE_TOPOLOGY_HALO);
}
''')

    def test_actual_projected_vertices_stay_inside_retained_bounds(self):
        ground=(ROOT/"Renderer/lab/shared/natural/ground.h").read_text()
        projection="struct GroundProjection {"+ground.split("struct GroundProjection {",1)[1].split("\n};",1)[0]+"\n};"
        run_cpp(r'''
#include "Renderer/native/render_core/projected_mesh_bounds.h"
#include <cassert>
#include <vector>
struct MapVertex {float x,y,z,world_x,world_y,world_z,world_valid,normal_z,u,v;};
''' + projection + r'''
int main(){
 using Bounds=c3x_renderer::render_core::ProjectedMeshBounds;
 for(int column:{-500,-1,0,23,500})for(int row:{-450,0,71}){
  Bounds bounds;std::vector<std::array<float,3>> points;
  for(int i=0;i<300;++i){float x=column+float(i%17)/8-1,y=row+float(i%23)/9-1,z=float(i%31)/13-.3f;
   points.push_back({x,y,z});bounds.include(x,y,z);}
  for(int width:{64,128,160,192}){
   auto rect=bounds.project(column,row,width);
   GroundProjection projection{column,row,width*.5f,width*.25f,width/224.f*.82f,1192};
   for(auto p:points){auto vertex=projection(p[0],p[1],p[2]*112.f);
    assert(vertex.x>=rect[0] && vertex.x<=rect[2] && vertex.y>=rect[1] && vertex.y<=rect[3]);}
  }
 }
 Bounds slope;slope.include(0,0,0);slope.include(1,0,.25f*224.f/.82f/112.f);
 auto tight=slope.project(0,0,128);assert(tight[3]-tight[1]<=5);
}
''')

    def test_diagnostic_pixels_match_world_region_after_clipped_scroll(self):
        from Renderer.native.analyze_region_dependencies import analyze_pixels
        from PIL import Image
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);lines=[]
            for step, (screen_x, right) in enumerate(((0,128),(-64,64),(0,128))):
                image=Image.new("RGB",(128,128),"white")
                if step==2:image.putpixel((3,4),(254,255,255))
                image.save(folder/f"zoom.bmp.resident{step}.bmp")
                lines.extend([f"stage=render-region-dependencies x=0 y=0 hit=0 screen_x={screen_x} screen_y=0 left=0 top=0 right={right} bottom=128",
                              "stage=render-region-cache hits=0 misses=1"])
            report=analyze_pixels(lines,folder,3)
            self.assertEqual(report["counts"], {"no_prior_full_region_image":1,"identical_visible_pixels":1,"changed_visible_pixels":1})
            self.assertEqual(report["changes"][0]["changed_pixels"],1)
            self.assertEqual(report["changes"][0]["maximum_channel_difference"],1)

    def test_diagnostic_misses_distinguish_new_regions_from_dependencies(self):
        from Renderer.native.analyze_region_dependencies import analyze, COMPONENTS
        def region(x, hit, shadow=1):
            parts = {name: 1 for name in COMPONENTS};parts["shadow"] = shadow
            return f"stage=render-region-dependencies x={x} y=0 hit={hit} parts=7 " + " ".join(f"{k}={v}" for k,v in parts.items())
        lines = [region(0, 0), "stage=render-region-cache hits=0 misses=1",
                 region(0, 0, 2), region(128, 0), "stage=render-region-cache hits=0 misses=2",
                 region(0, 0, 2), region(128, 1), "stage=render-region-cache hits=1 misses=1"]
        result = analyze(lines, 2)
        self.assertEqual(result["causes"], {"changed_dependencies": 1, "first_observed_region": 1, "unchanged_fingerprints": 1})
        self.assertEqual(result["changed_components"], {"shadow": 1})
        self.assertEqual(result["hits"], 1)
        with self.assertRaisesRegex(ValueError, "coverage"):
            analyze(lines[:-1] + ["stage=render-region-cache hits=1 misses=2"], 2)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            analyze(lines + [region(0, 0)], 2)

    def test_budgets_eviction_and_queued_copy_lifetime(self):
        run_cpp(r'''
#include "Renderer/native/render_core/render_region_cache.h"
#include "Renderer/native/render_core/region_contributor_index.h"
#include "Renderer/native/render_core/projected_mesh_bounds.h"
#include <cassert>
#include <deque>
struct Image {int refs=1;void Release(){assert(refs>0);--refs;}};
int main(){
 using Cache=c3x_renderer::render_core::RenderRegionCache<Image>;
 using Key=c3x_renderer::render_core::RenderRegionKey;
 constexpr std::size_t mib=1024u*1024u;
 Image a,b,c;Cache cache;
 assert(cache.insert({1},&a,160*mib));assert(cache.insert({2},&b,96*mib));
 assert(!cache.make_room({3},Cache::gpu_budget+1));assert(a.refs==1 && b.refs==1);
 assert(cache.find({1})==&a);b.refs++; // A queued GPU copy holds its own reference.
 assert(cache.insert({3},&c,1));assert(b.refs==1 && a.refs==1);
 assert(!cache.find({2}));b.Release();assert(b.refs==0);
 assert(!cache.insert({1},&b,1)); // Rejected ownership stays with the caller.
 Key huge;huge.reserve(Cache::metadata_budget/sizeof(std::uint64_t));huge.push_back(9);
 assert(!cache.make_room(huge,1));assert(cache.find({1})==&a);
 cache.clear();assert(a.refs==0 && c.refs==0 && cache.gpu_bytes==0 && cache.metadata_bytes==0);
 Key pressure;pressure.reserve(40*mib/sizeof(std::uint64_t));pressure.push_back(10);
 assert(cache.make_room(pressure,1));cache.metadata_limit=32*mib;
 assert(!cache.make_room(pressure,1));cache.metadata_limit=Cache::metadata_budget;
 std::deque<Image> images;
 for(unsigned i=0;i<Cache::entry_limit+1;++i){images.emplace_back();assert(cache.insert({i},&images.back(),0));}
 assert(cache.entries.size()==Cache::entry_limit && images.front().refs==0);
 cache.clear();for(auto const& image:images)assert(image.refs==0);
}
''')

    def test_key_tracks_shadow_reflection_lighting_and_current_draws(self):
        source=(ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        def struct(name):
            return "struct "+name+" {"+source.split("struct "+name+" {",1)[1].split("\n};",1)[0]+"\n};\n"
        viewport=struct("ViewportShaderSettings")
        chunk=struct("CachedVertexChunk")
        layers="enum GeometryLayer : std::size_t {"+source.split("enum GeometryLayer : std::size_t {",1)[1].split("};",1)[0]+"};\n"
        shadow=(ROOT / "Renderer/native/render_core/source_shadow.h").read_text()
        preparation="static std::array<float,4> project("+shadow.split("static std::array<float,4> project(",1)[1].split("    template<class Bind>",1)[0]
        methods="    static bool append_region_bytes("+source.split("    static bool append_region_bytes(",1)[1].split("    bool draw_cached_geometry(",1)[0]
        program=r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <set>
#include <map>
#include <vector>
#include "Renderer/native/render_core/render_region_cache.h"
#include "Renderer/native/render_core/region_contributor_index.h"
#include "Renderer/native/render_core/projected_mesh_bounds.h"
using UINT=unsigned;using DXGI_FORMAT=unsigned;constexpr unsigned DXGI_FORMAT_R32_UINT=42;
struct ID3D11Buffer{};struct ID3D11ShaderResourceView{};struct ID3D11Texture2D{void Release(){}};
struct D3D11_RECT{long left,top,right,bottom;};
namespace c3x_renderer { namespace city_fidelity {
struct Lighting {
 struct Light{float position[3]={},range=0,color[3]={},intensity=0,direction[3]={},owner=0;};
 struct Box{float low[4],high[4];};std::vector<Light> lights;std::vector<Box> blockers;
};
} namespace render_core {
struct SourceShadow {
 struct Bounds{float low[3]={},high[3]={};};
 struct Caster{void *vertices=nullptr,*indices=nullptr;unsigned count=0,stride=0;Bounds bounds;float offset[3]={};std::uint64_t version=1;unsigned layer=0,index_format=42,binding=0xffffffffu;};
 std::array<float,12> basis{};
''' + preparation + r'''
};}}
''' + viewport+chunk+layers+r'''
struct State{
 c3x_renderer::render_core::RenderRegionKey region_context{1,2,3,4};
 std::array<float,12> shadow_basis{1,0,0,0,0,1,0,0,0,0,1,0};
 struct {float height_pixels=40;bool enabled=true;}reflection;
 int height=1192,shadow_tile_width=128;
 bool region_receiver_shadows=false;
 std::array<double,3> frame_region_phase_ms{};
 c3x_renderer::render_core::RegionContributorIndex region_contributors;
''' + methods + r'''
};
int main(){
 using Shadow=c3x_renderer::render_core::SourceShadow;using Key=c3x_renderer::render_core::RenderRegionKey;
 State state;ViewportShaderSettings settings{};settings.inverse_size[0]=settings.inverse_size[1]=1.f/136;
 std::array<std::vector<CachedVertexChunk>,geometry_layer_count> buffers;
 CachedVertexChunk ground;ground.version=11;ground.bounds={0,0,128,128};ground.world_bounds={{0,0,0},{1,1,0}};
 buffers[geometry_underlay].push_back(ground);
 CachedVertexChunk distant=ground;distant.version=12;distant.bounds={1000,1000,1100,1100};
 buffers[geometry_feature].push_back(distant);
 std::vector<Shadow::Caster> casters(1);casters[0].version=77;casters[0].bounds={{0,0,1},{1,1,2}};
 Shadow::PreparedCasters prepared;
 auto key=[&](){Key result;assert(prepared.build(casters,state.shadow_basis,true));
   state.region_contributors.clear();state.prepare_region_contributors(buffers);assert(state.region_contributors.ready);
   assert(state.render_region_key(buffers,settings,casters,&prepared,result));
   Shadow::PreparedCasters independent;assert(independent.build(casters,state.shadow_basis));
   state.region_contributors.clear();
   Key expected;assert(state.render_region_key(buffers,settings,casters,&independent,expected));
   assert(result==expected);return result;};
 auto original=key();assert(key()==original);assert(prepared.receiver_hits>0);
 Key diagnostic;std::vector<std::size_t> sections;
 assert(state.render_region_key(buffers,settings,casters,&prepared,diagnostic,&sections));
 assert(diagnostic==original && sections.size()==7 && sections.front()==state.region_context.size() && sections.back()==original.size());
 buffers[geometry_underlay][0].translation_x+=7;settings.translation[0]-=7;
 buffers[geometry_feature][0].translation_x+=7;
 assert(key()==original); // New camera, same effective region geometry.
 buffers[geometry_feature][0].version++;assert(key()==original); // Unrelated off-region geometry.
 casters[0].version++;assert(key()!=original);casters[0].version--;
 casters[0].binding=4;assert(key()!=original);casters[0].binding=0xffffffffu;
 auto outside=casters[0];outside.bounds={{4,4,1},{5,5,2}};outside.version=300;casters.push_back(outside);
 auto broad=key();casters.back().version++;assert(key()!=broad);
 state.region_receiver_shadows=true;auto local_key=key();casters.back().version++;assert(key()==local_key);
 casters.back().bounds={{1.015f,0,1},{1.02f,1,2}};assert(key()!=local_key); // Filter/normal reach remains a dependency.
 auto near_key=key();casters.back().binding=8;assert(key()!=near_key);
 casters.pop_back();assert(key()==original);state.region_receiver_shadows=false;
 buffers[geometry_underlay][0].version++;assert(key()!=original);buffers[geometry_underlay][0].version--;
 auto reflected=ground;reflected.version=33;reflected.bounds={20,-40,40,-20};
 reflected.world_bounds={{0,0,.5f},{1,1,1}};buffers[geometry_feature].push_back(reflected);
 auto reflected_key=key();buffers[geometry_feature].back().version++;assert(key()!=reflected_key);
 buffers[geometry_feature].pop_back();assert(key()==original);
 auto city=ground;city.version=44;city.bounds={160,20,170,30};
 city.city_lighting=std::make_shared<c3x_renderer::city_fidelity::Lighting>();
 city.city_lighting->lights.resize(1);buffers[geometry_city].push_back(city);
 auto lit=key();city.city_lighting->lights[0].intensity=2;assert(key()!=lit);
 buffers[geometry_city].clear();assert(key()==original);
 auto second=buffers[geometry_underlay][0];second.version=55;buffers[geometry_underlay].push_back(second);
 auto ordered=key();std::reverse(buffers[geometry_underlay].begin(),buffers[geometry_underlay].end());assert(key()!=ordered);
 buffers[geometry_underlay].clear();assert(key()!=original); // Current capture no longer permits that surface.
 buffers[geometry_underlay].push_back(ground);settings.translation[0]=0;assert(key()==original);
 state.region_context[0]++;assert(key()!=original);state.region_context[0]--;
 ID3D11ShaderResourceView animation;buffers[geometry_underlay][0].animation_texture=&animation;
 Key rejected;assert(!state.render_region_key(buffers,settings,casters,&prepared,rejected));
 assert(!state.render_region_key(buffers,settings,casters,nullptr,rejected));
}
'''
        run_cpp(program)
