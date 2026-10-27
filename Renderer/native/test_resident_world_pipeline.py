"""The fresh pipeline reads the resident world store (restructure, increment 3).

With C3X_RENDERER_RESIDENT_WORLD=1 the pipeline's contributor queries read the
world-basis store instead of the camera window's lease and region index. On
equal data both paths must select the same records in the same draw order,
across the wrap seam and in the mirrored pass, and a raster proof's tiles are
resident when the store holds them.
"""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method


class ResidentWorldPipelineTests(unittest.TestCase):
    def test_store_contributors_equal_window_contributors(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        contributors = method(source, '    template<class Visit>void contributors(')
        each = method(source, '    template<class Visit>void each_resident(')
        resident = method(source, '    bool tile_resident(')
        run_cpp(r'''
#define C3X_RENDERER64_FRESH
#include "Renderer/native/render_core/resident_world.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <tuple>
#include <unordered_set>
#include <vector>
struct D3D11_RECT {long left,top,right,bottom;};
struct ViewportShaderSettings {float translation[2]={};};
constexpr unsigned geometry_layer_count=4,geometry_wave=3;
struct GeometryDrawRecord {
 int id=0,tile_x=0,tile_y=0,translation_x=0,translation_y=0;
 struct {int left=0,top=0,right=0,bottom=0;} bounds;float low_z=0,high_z=0;
};
using World=c3x_renderer::render_core::ResidentWorld<GeometryDrawRecord,geometry_layer_count>;
double const reflection_height=30;
double mirror(float z){return 2*reflection_height*std::max(0.,double(z)-2.5/112.);}
// The window's lease, region index and the store, as production holds them.
struct Lease {std::array<std::vector<GeometryDrawRecord>,geometry_layer_count> records;};
struct Renderer {
 struct {float height_pixels=float(reflection_height);} reflection;
 int resident_world_half[2]={64,32};
 Lease const* window=nullptr;
 bool query_region_inputs(unsigned pass,int left,int top,unsigned extent_x,unsigned extent_y,
   std::vector<std::pair<unsigned,unsigned>>& output)const{
  output.clear();
  for(unsigned layer=0;layer<geometry_layer_count;++layer)for(unsigned i=0;i<window->records[layer].size();++i){
   auto const& r=window->records[layer][i];double lo=pass?mirror(r.low_z):0,hi=pass?mirror(r.high_z):0;
   double l=r.bounds.left+r.translation_x,t=r.bounds.top+r.translation_y+lo,rr=r.bounds.right+r.translation_x,b=r.bounds.bottom+r.translation_y+hi;
   if(!(rr<=left || l>=left+double(extent_x) || b<=top || t>=top+double(extent_y)))output.push_back({layer,i});
  }
  return true;
 }
};
struct RasterInputs {using Key=std::array<std::uint64_t,3>;};
struct Pipeline {
 Renderer renderer;
 std::shared_ptr<Lease> resident_lease;
 using ResidentWorld=World;ResidentWorld::Lease world_lease;
 std::unordered_set<std::uint64_t> resident_tiles;
 float resident_basis_x=0,resident_basis_y=0;int wrap_pixels=0;
''' + resident + each + contributors + r'''
};
// The exact test the callers apply after a query.
bool meets(GeometryDrawRecord const& r,ViewportShaderSettings const& s,D3D11_RECT c,bool mirrored){
 int dx=r.translation_x+int(s.translation[0]),dy=r.translation_y+int(s.translation[1]);
 double lo=mirrored?mirror(r.low_z):0,hi=mirrored?mirror(r.high_z):0;
 return !(r.bounds.right+dx<=c.left || r.bounds.left+dx>=c.right || r.bounds.bottom+dy+hi<=c.top || r.bounds.top+dy+lo>=c.bottom);
}
int main(){
 std::mt19937 rng(11);
 int const columns=40,rows=30,wrap_tiles=columns,wrap=wrap_tiles*64;
 int const basis_x=-1700,basis_y=-900; // window records + basis = world records
 // World records, by tile, in append order; the window covers columns 30..49
 // (columns 0..9 occur past the seam) and rows 4..20.
 World world;Lease window;int next=0;
 struct Occurrence {int y,x,i;unsigned layer;GeometryDrawRecord window;};
 std::vector<Occurrence> occurrences;
 for(int y=0;y<rows;++y)for(int x=(y&1);x<columns;x+=2){
  World::Tile tile;tile.tile_x=x;tile.tile_y=y;
  for(unsigned layer=0;layer<geometry_layer_count;++layer){unsigned count=rng()%3;
   for(unsigned i=0;i<count;++i){GeometryDrawRecord r;r.id=next++;r.tile_x=x;r.tile_y=y;
    r.translation_x=x*64;r.translation_y=y*32;
    r.bounds={-40-int(rng()%40),-30-int(rng()%60),40+int(rng()%40),30+int(rng()%20)};
    r.low_z=float(rng()%100)/400.f;r.high_z=r.low_z+float(rng()%100)/200.f;
    tile.add(layer,r,{double(r.bounds.left)+r.translation_x,double(r.bounds.top)+r.translation_y,
     double(r.bounds.right)+r.translation_x,double(r.bounds.bottom)+r.translation_y,r.low_z,r.high_z});
    int occurrence_x=x<10?x+wrap_tiles:x;
    if(occurrence_x>=30 && y>=4 && y<=20){auto w=r;w.translation_x+=(occurrence_x-x)*64-basis_x;w.translation_y-=basis_y;
     occurrences.push_back({y,occurrence_x,int(i),layer,w});}
   }
  }
  world.set(std::move(tile));
 }
 std::stable_sort(occurrences.begin(),occurrences.end(),[](auto const& a,auto const& b){return std::tie(a.y,a.x,a.i)<std::tie(b.y,b.x,b.i);});
 for(auto const& o:occurrences)window.records[o.layer].push_back(o.window);
 Pipeline window_path,store_path;
 window_path.renderer.window=&window;window_path.resident_lease=std::make_shared<Lease>(window);
 window_path.resident_basis_x=basis_x;window_path.resident_basis_y=basis_y;window_path.wrap_pixels=wrap;
 store_path.renderer.window=&window;store_path.world_lease=world.publish();store_path.wrap_pixels=wrap;
 std::size_t compared=0,seam=0;
 for(int q=0;q<400;++q){
  bool mirrored=q&1;ViewportShaderSettings settings;
  settings.translation[0]=-float(rng()%1000);settings.translation[1]=-float(rng()%600);
  // Clips stay inside the window's coverage (columns 30..49, rows 4..20, less
  // the largest extents), so both paths see every contributor.
  long wl=2010+long(rng()%500),wt=200+long(rng()%150);
  D3D11_RECT clip;clip.left=wl+long(settings.translation[0]);clip.top=wt+long(settings.translation[1]);
  clip.right=clip.left+long(rng()%500)+1;clip.bottom=clip.top+long(rng()%200)+1;
  std::vector<std::tuple<unsigned,int,int,int>> a,b;
  window_path.contributors(settings,clip,mirrored,[&](unsigned layer,GeometryDrawRecord const& r){
   if(meets(r,settings,clip,mirrored))a.push_back({layer,r.id,r.translation_x,r.translation_y});});
  store_path.contributors(settings,clip,mirrored,[&](unsigned layer,GeometryDrawRecord const& r){
   assert(layer!=geometry_wave);
   if(meets(r,settings,clip,mirrored))b.push_back({layer,r.id,r.translation_x,r.translation_y});});
  std::vector<std::tuple<unsigned,int,int,int>> window_no_waves;
  for(auto const& item:a)if(std::get<0>(item)!=geometry_wave)window_no_waves.push_back(item);
  assert(window_no_waves==b);compared+=b.size();
  for(auto const& item:b)if(std::get<2>(item)>=wrap)++seam;
 }
 assert(compared>5000 && seam>100);
 // Raster proofs: a tile is resident when the store holds it.
 RasterInputs::Key held{},missing{};held[2]=(std::uint64_t(std::uint32_t(4))<<32)|6;missing[2]=(std::uint64_t(std::uint32_t(5))<<32)|6;
 assert(store_path.tile_resident(held) && !store_path.tile_resident(missing));
 std::printf("PASS store contributors: %zu records, %zu across the seam\n",compared,seam);
}
''')

    def test_store_caster_selection_follows_the_window_box(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        selection = method(source, '    Membership::Lease caster_selection()const{')
        box = method(source, '    bool update_caster_box(std::array<std::int64_t,4> const& window,std::int64_t margin_x,std::int64_t margin_y){')
        run_cpp(r"""
#define C3X_RENDERER64_FRESH
#include "Renderer/native/render_core/scene_membership.h"
#include "Renderer/native/render_core/resident_world.h"
#include <cassert>
#include <cstdio>
#include <unordered_set>
using namespace c3x_renderer::render_core;
struct Chunk {struct Bounds {int left,top,right,bottom;} bounds{};int translation_x=0,translation_y=0;float natural_projection[4]={};};
constexpr unsigned geometry_layer_count=2;
using Record=GeometryDrawRecord<Chunk>;
using World=ResidentWorld<Record,geometry_layer_count>;
struct Dimensions {int width=0,height=0;bool wrap_x=false,wrap_y=false;};
struct Renderer {
 bool resident_world_active=true;World resident_world;int resident_world_half[2]={64,32};
 struct {bool valid=true;std::array<std::int64_t,4> box{};} world_window;
 struct {Dimensions value;Dimensions world()const{return value;}} world_coast_value;
 struct Coast {Dimensions value;Dimensions dimensions()const{return value;}};
 struct {Coast coast;Coast const& world()const{return coast;}} world_coast;
 SceneMembership<Chunk,geometry_layer_count> geometry_vertex_buffers;
};
struct Shadow {
 using Membership=SceneMembership<Chunk,geometry_layer_count>;
 Renderer& renderer;
 std::array<std::int64_t,4> caster_box{};bool caster_box_valid=false;
""" + box + selection + r"""
};
int main(){
 Renderer renderer;renderer.world_coast.coast.value={20,40,true,false};
 Chunk chunk;std::vector<std::weak_ptr<int>> meshes;
 // Tiles at columns 0..19 (wrapping), rows 0..9; tile (x,y) holds one
 // record per layer with ordinal = 100*y+x.
 for(int y=0;y<10;++y)for(int x=(y&1);x<20;x+=2){
  World::Tile t;t.tile_x=x;t.tile_y=y;auto mesh=std::make_shared<int>(x);meshes.push_back(mesh);
  ContentHandle handle{std::size_t(y*20+x),std::uint64_t(1000+y*20+x)};t.owners.push_back({handle,mesh});
  for(unsigned layer=0;layer<geometry_layer_count;++layer){Record r(chunk);r.owner=handle;r.ordinal=unsigned(100*y+x);r.tile_x=x;r.tile_y=y;t.add(layer,r,{});}
  renderer.resident_world.set(std::move(t));}
 Shadow shadow{renderer};
 // A box across the seam: occurrence columns 16..23 (pixels 1024..1536), rows 2..5.
 // No caster box yet: no casters.
 assert(shadow.caster_selection()->records[0].empty());
 shadow.caster_box={16*64,2*32,24*64,6*32};shadow.caster_box_valid=true;
 auto lease=shadow.caster_selection();
 std::vector<unsigned> got;for(auto const& r:lease->records[1])got.push_back(r.ordinal);
 std::vector<unsigned> want;
 for(int y=2;y<6;++y)for(int x=16;x<24;++x)if(!((x+y)&1))want.push_back(unsigned(100*y+(x%20)));
 assert(got==want && lease->records[0].size()==want.size());
 assert(lease->content.size()==want.size() && lease->content.get(lease->records[1][0].owner));
 // A box wider than the world holds each canonical tile once.
 shadow.caster_box={-4*64,0,30*64,2*32};
 lease=shadow.caster_selection();assert(lease->records[0].size()==20);
 // The selection owns its meshes after the store forgets them.
 shadow.caster_box={0,0,64,32};lease=shadow.caster_selection();
 assert(lease->records[0].size()==1 && lease->records[0][0].ordinal==0);
 renderer.resident_world.clear();
 for(std::size_t i=0;i<meshes.size();++i)assert(meshes[i].expired()==(i!=0));
 lease.reset();assert(meshes[0].expired());
 // Without the store the window's own lease is the selection.
 renderer.resident_world_active=false;
 assert(shadow.caster_selection()==renderer.geometry_vertex_buffers.publish());
 // The caster box grows the window by a block and holds while the window
 // stays inside it, so a crossing inside it keeps the caster set.
 shadow.caster_box_valid=false;std::int64_t bx=8*64,by=8*32;
 std::array<std::int64_t,4> window{0,0,4*bx,3*by};
 assert(shadow.update_caster_box(window,bx,by) && (shadow.caster_box==std::array<std::int64_t,4>{-bx,-by,5*bx,4*by}));
 window={bx,0,5*bx,3*by};assert(!shadow.update_caster_box(window,bx,by));      // one block right: kept
 window={0,-by,4*bx,2*by};assert(!shadow.update_caster_box(window,bx,by));     // one block up: kept
 window={2*bx,0,6*bx,3*by};assert(shadow.update_caster_box(window,bx,by) &&    // two blocks: regrown
  (shadow.caster_box==std::array<std::int64_t,4>{bx,-by,7*bx,4*by}));
 // Without a margin the box is the window itself.
 window={0,0,4*bx,3*by};assert(shadow.update_caster_box(window,0,0) && shadow.caster_box==window);
 assert(!shadow.update_caster_box(window,0,0));
 window={bx,0,5*bx,3*by};assert(shadow.update_caster_box(window,0,0) && shadow.caster_box==window);
 std::puts("PASS store caster selection");
}
""")

    def test_a_moved_window_takes_ownership_from_stored_content(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        methods = '\n'.join(method(source, signature) for signature in (
            '    std::uint64_t terrain_content_key(c3x_renderer_tile_v1 const& tile)const{',
            '    c3x_renderer_u32 content_replacement_flags(c3x_renderer_tile_v1 const& tile)const{',
            '    bool resident_world_ownership(c3x_renderer::render_core::WorldWindow const& window,unsigned native_count,'))
        run_cpp(r"""
#include "Renderer/native/render_core/world_window.h"
#include "Renderer/native/render_core/resident_world.h"
#include <cassert>
#include <cstdio>
using namespace c3x_renderer::render_core;
struct Record {int id=0;};
using World=ResidentWorld<Record,2>;
struct Owner {bool replaces_resource=false;};
struct State {
 bool feature_assets_ready=true,marsh_assets_ready=true,volcano_assets_ready=true,dune_assets_ready=true,river_assets_ready=true,
  route_assets_ready=true,city_assets_ready=true,site_assets_ready=true,mine_assets_ready=true,farm_assets_ready=true;
 struct ResidentWorldRecords {using Tile=World::Tile;};
 World resident_world;std::vector<Owner> owners;
 struct {std::vector<int> compositions;} site_bundle;
 struct Content {std::vector<Owner>* owners;Owner* resolve(ContentHandle h)const{return h.slot<owners->size()?&(*owners)[h.slot]:nullptr;}} resident_content{&owners};
""" + methods + r"""
};
int main(){
 State state;state.owners.resize(3);state.owners[1].replaces_resource=true;
 auto tile=[](int x,int y){c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;t.city_id=-1;t.terrain_type=t.real_terrain_type=2;t.tile_flags=C3X_RENDERER_TILE_RENDER;return t;};
 auto plain=tile(0,0),road=tile(2,0),resource=tile(1,1);road.road_mask=3;
 auto store=[&](c3x_renderer_tile_v1 const& t,std::size_t owner){World::Tile s;s.tile_x=t.tile_x;s.tile_y=t.tile_y;
  s.owners.push_back({ContentHandle{owner,owner+1},std::make_shared<int>(0)});s.content=state.terrain_content_key(t);state.resident_world.set(std::move(s));};
 store(plain,0);store(road,2);store(resource,1);
 // The window lists plain, road, resource; Civ III's capture lists them in reverse.
 WorldWindow window;window.valid=true;window.tiles={plain,road,resource};window.window_count=3;window.native_index={2,1,0};
 std::vector<c3x_renderer_u32> out;unsigned missing=0,differ=0;
 assert(state.resident_world_ownership(window,3,out,missing,differ) && out.size()==3);
 assert(out[2]==C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED);
 assert(out[1]==(C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED|C3X_RENDERER_TILE_CUSTOM_ROAD_REPLACED));
 assert(out[0]==(C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED|C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED));
 // Placement and visibility flags are not content.
 window.tiles[0].tile_flags|=C3X_RENDERER_TILE_VISIBLE|C3X_RENDERER_TILE_PREFETCH;
 assert(state.resident_world_ownership(window,3,out,missing,differ));
 // Content the store was not built from, a tile it lacks, or site art not ready refuse.
 window.tiles[1].railroad_mask=1;assert(!state.resident_world_ownership(window,3,out,missing,differ) && differ==1);
 window.tiles[1]=road;window.tiles.push_back(tile(4,0));window.window_count=4;window.native_index.push_back(3);
 missing=differ=0;assert(!state.resident_world_ownership(window,4,out,missing,differ) && missing==1);
 window.tiles.pop_back();window.window_count=3;window.native_index.pop_back();
 window.tiles[0].improvement_flags=C3X_RENDERER_IMPROVEMENT_GOODY_HUT;store(window.tiles[0],0);state.site_assets_ready=false;
 missing=differ=0;assert(!state.resident_world_ownership(window,3,out,missing,differ) && differ==1);
 // Tile buildings: a site pack without compositions leaves them undrawn and
 // unclaimed (the bridge then requires none); one with them claims them.
 state.site_assets_ready=true;window.tiles[0]=plain;window.tiles[0].improvement_flags=C3X_RENDERER_IMPROVEMENT_FORTRESS;
 store(window.tiles[0],0);
 missing=differ=0;assert(state.resident_world_ownership(window,3,out,missing,differ) && out[2]==C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED);
 state.site_bundle.compositions.push_back(0);
 assert(state.resident_world_ownership(window,3,out,missing,differ) &&
  out[2]==(C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED|C3X_RENDERER_TILE_CUSTOM_TILE_BUILDING_REPLACED));
 std::puts("PASS resident world ownership");
}
""")

    def test_prefetch_renders_do_not_switch_the_draw_off_the_store(self):
        # Idle prefetch renders between camera steps recompute the per-render
        # feed flag as off; the draw and the deferral read a flag only gameplay
        # renders set, or the pipeline flipped between the store and the
        # window's lease (and refused moved-window steps).
        pipeline = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        renderer = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        self.assertNotIn('renderer.resident_world_feed', pipeline)
        self.assertIn('renderer.resident_world_active', pipeline)
        self.assertIn('if(!prewarming)resident_world_active=resident_world_feed;', renderer)
        # A window change retires the pipeline's borrowed selection only when
        # it borrows the window's records; with the store it owns its meshes.
        self.assertIn('if(!completed_scene_retained.load(std::memory_order_relaxed) && !resident_world_feed)\n'
                      '                c3x_renderer64_retire_geometry_selection();', renderer)
        defer = method(renderer, '    bool defer_scroll_step(PublishedMapFrame& ready){')
        self.assertIn('store_moves=state.resident_world_active', defer)
        self.assertIn('store_moves && state.resident_world_ownership(', defer)
        self.assertNotIn('resident_world_feed', defer)

    def test_window_basis_follows_the_window_not_the_camera(self):
        """Waves and resource poses stay on screen across the wrap seam.

        They are window records drawn once (no wrap copies) at the capture's
        camera: on screen at record + basis + camera, which must be the
        window's shift (the viewport translation). Across the seam Civ III
        re-bases the camera a world width while the window keeps its frame;
        a basis kept from before drew them a world width off screen until the
        window next changed, so waves and animals vanished until a turn ended
        (the user, October 27). Fails on the basis kept per window alone.
        """
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        update = method(source, '    void update_window_basis(){')
        run_cpp(r"""
#define C3X_RENDERER64_FRESH
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
struct Trace {unsigned rebases=0;void write(char const*,char const*,bool){++rebases;}};
struct Renderer {struct {float translation[2]={};} geometry_viewport_settings;std::uint64_t window_frame_serial=0,records_revision=0;Trace trace;};
struct Pipeline {
 Renderer renderer;int camera_x=0,camera_y=0,wrap_pixels=8320;
 float window_basis_x=0,window_basis_y=0,tracked_basis_x=0,tracked_basis_y=0;
 std::uint64_t window_signature=~std::uint64_t(0),window_serial=~std::uint64_t(0);
 std::uint64_t window_revision()const{return renderer.records_revision;}
""" + update + r"""
 // A window record (a wave ribbon or a posed body) as the dynamic pass draws it.
 float screen(float record){update_window_basis();return record+window_basis_x+float(camera_x);}
 float& translation(){return renderer.geometry_viewport_settings.translation[0];}
};
int main(){
 float const record=700;
 Pipeline p;p.translation()=-3000+5280;p.camera_x=-3000;
 assert(p.screen(record)==record+p.translation() && p.window_basis_x==5280 && p.tracked_basis_x==5280);
 // A covered step keeps the window: the viewport follows the camera.
 p.camera_x=-3128;p.translation()=-3128+5280;
 assert(p.screen(record)==record+p.translation() && p.window_basis_x==5280);
 // Across the wrap seam Civ III re-bases the camera a world width while the
 // window keeps its frame: the record stays where the window put it.
 p.camera_x=-3128+8320;
 assert(p.screen(record)==record+p.translation() && p.window_basis_x==5280-8320);
 p.camera_x=-3128+8320-64;p.translation()=-3128-64+5280; // a covered step past the seam
 assert(p.screen(record)==record+p.translation());
 p.camera_x=-3128-64;assert(p.screen(record)==record+p.translation()); // and back across it
 assert(p.window_basis_x==5280 && p.tracked_basis_x==5280 && p.renderer.trace.rebases==2);
 // Any smaller difference is the window's own: the kept basis decides.
 p.camera_x=-3128;assert(p.window_basis_x==5280 && p.screen(record)==record+5280-3128);
 // A changed window re-bases it.
 p.renderer.window_frame_serial=1;p.translation()=p.camera_x-3040;
 assert(p.screen(record)==record+p.translation() && p.window_basis_x==-3040);
 // Without wrapping the kept basis is never moved.
 p.wrap_pixels=0;p.camera_x+=8320;assert(p.screen(record)==record-3040+p.camera_x);
 std::puts("PASS window basis");
}
""")

    def test_casters_are_collected_once_per_store_block(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        begin = source.index('    using WorldRecords=c3x_renderer::render_core::ResidentWorld<GeometryDrawRecord,geometry_layer_count>;')
        cache = source[begin:source.index('#endif\n    bool refresh_casters(std::uint64_t scene) {', begin)]
        run_cpp(r"""
#include "Renderer/native/render_core/scene_membership.h"
#include "Renderer/native/render_core/resident_world.h"
#include <cassert>
#include <cstdio>
#include <map>
#include <set>
#include <unordered_set>
using namespace c3x_renderer;
struct Chunk {struct Bounds {int left,top,right,bottom;} bounds{};int translation_x=0,translation_y=0;float natural_projection[4]={};
 std::shared_ptr<std::vector<int>> instances;};
constexpr unsigned geometry_layer_count=2;
using GeometryDrawRecord=render_core::GeometryDrawRecord<Chunk>;
using GeometryDrawView=render_core::GeometryDrawView<Chunk,geometry_layer_count>;
struct Shadow {struct Caster {unsigned ordinal=0;int tile_x=0,tile_y=0;};};
struct AtlasInputs {using Key=std::array<std::uint64_t,2>;struct Hash{std::size_t operator()(Key const& k)const{return std::size_t(k[0]*31+k[1]);}};};
struct Dimensions {int width=0,height=0;bool wrap_x=false,wrap_y=false;};
struct Renderer {
 render_core::ResidentWorld<GeometryDrawRecord,geometry_layer_count> resident_world;int resident_world_half[2]={64,32};
 struct Coast {Dimensions value;Dimensions dimensions()const{return value;}};
 struct {Coast coast;Coast const& world()const{return coast;}} world_coast;
 unsigned collections=0;std::size_t collected=0;
 void collect_shadow_casters(GeometryDrawView view,std::vector<Shadow::Caster>& out){++collections;
  for(unsigned layer=0;layer<geometry_layer_count;++layer)for(auto const& r:view[layer]){out.push_back({r.occurrence->ordinal,r.occurrence->tile_x,r.occurrence->tile_y});++collected;}}
};
struct Pipeline {
 using Membership=render_core::SceneMembership<Chunk,geometry_layer_count>;
 Renderer& renderer;std::array<std::int64_t,4> caster_box{};bool caster_box_valid=true;
 AtlasInputs::Key caster_key(Shadow::Caster const& c)const{return {std::uint64_t(c.ordinal),std::uint64_t(c.tile_x)*1000+std::uint64_t(c.tile_y)};}
""" + cache + r"""
};
int main(){
 Renderer renderer;renderer.world_coast.coast.value={40,32,true,false};
 Chunk plain,forest;forest.instances=std::make_shared<std::vector<int>>(3,1);
 std::map<std::pair<int,int>,std::weak_ptr<int>> meshes;
 auto put=[&](int x,int y,unsigned ordinal){render_core::ResidentWorld<GeometryDrawRecord,geometry_layer_count>::Tile t;t.tile_x=x;t.tile_y=y;
  auto mesh=std::make_shared<int>(x);meshes[{x,y}]=mesh;render_core::ContentHandle h{std::size_t(y*40+x),std::uint64_t(1+y*40+x)};
  t.owners.push_back({h,mesh});GeometryDrawRecord r((x%4)?plain:forest);r.owner=h;r.ordinal=ordinal;r.tile_x=x;r.tile_y=y;
  t.add(0,r,{});renderer.resident_world.set(std::move(t));};
 for(int y=0;y<16;++y)for(int x=(y&1);x<40;x+=2)put(x,y,1);
 Pipeline p{renderer};
 // Blocks (row 0, columns 0 and 1): tiles x 0..15, y 0..7.
 p.caster_box={0,0,16*64,8*32};
 std::vector<Shadow::Caster> casters;std::vector<AtlasInputs::Key> keys;Pipeline::Membership::Lease lease;
 assert(p.world_casters(casters,keys,lease) && casters.size()==64 && keys.size()==64);
 assert(p.block_caster_builds==2 && p.block_caster_reuses==0 && renderer.collections==2);
 assert(lease->content.size()==64 && lease->records[0].size()==16); // owners of all; records only instanced (x%4==0)
 // Unchanged blocks are reused, with the same casters.
 auto first=keys;casters.clear();keys.clear();
 assert(p.world_casters(casters,keys,lease) && keys==first && p.block_caster_builds==2 && p.block_caster_reuses==2);
 // An edit recollects only its block.
 put(10,4,7);casters.clear();keys.clear();
 assert(p.world_casters(casters,keys,lease) && p.block_caster_builds==3 && p.block_caster_reuses==3 && keys.size()==64);
 // Moving the box a block collects only the entering block; the leaving
 // block's meshes are released once nothing borrows them.
 p.caster_box={8*64,0,24*64,8*32};casters.clear();keys.clear();
 assert(p.world_casters(casters,keys,lease) && p.block_caster_builds==4 && p.block_caster_reuses==4 && casters.size()==64);
 assert(p.block_casters.size()==2);
 renderer.resident_world.clear();
 assert((meshes[{0,0}].expired() && !meshes[{8,0}].expired())); // block 0 left; block 1 is still borrowed
 lease.reset();p.block_casters.clear();assert((meshes[{8,0}].expired()));
 // Across the seam the box's canonical blocks are found once each.
 for(int y=0;y<16;++y)for(int x=(y&1);x<40;x+=2)put(x,y,1);
 p.caster_box={32*64,0,48*64,8*32};casters.clear();keys.clear();
 assert(p.world_casters(casters,keys,lease) && casters.size()==64); // columns 32..39 and 0..7
 std::puts("PASS casters per block");
}
""")

    def test_a_scene_render_keeps_the_drawn_casters_for_a_one_block_move(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        frame_scope = method(source, '    std::uint64_t frame_caster_scope(){')
        run_cpp(r"""
#define C3X_RENDERER64_FRESH
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace c3x_renderer{namespace render_core{struct WorldWindow{static constexpr int block=8;};
 inline bool cached_environment(char const*,char*,unsigned){return false;}}}
struct Pipeline {
 struct {bool resident_world_active=true,scene_render=true;int resident_world_half[2]={64,32};
  struct {int level=0;void write(char const*,char const*,bool){}} trace;} renderer;
 struct {std::array<std::int64_t,4> caster_box{},selected_box{};bool atlas_complete=true;std::uint64_t caster_signature=1;} shadow;
 std::uint64_t current=1;std::uint64_t caster_scope()const{return current;}
 bool caster_refresh_owed=false;std::uint64_t caster_refreshes_deferred=0;
""" + frame_scope.replace('    bool caster_refresh_owed=false;std::uint64_t caster_refreshes_deferred=0;\n', '') + r"""
};
int main(){
 Pipeline p;std::int64_t bx=8*64,by=8*32;
 p.shadow.selected_box={0,0,4*bx,3*by};
 // A crossing adoption: the window moved a block; keep the drawn casters.
 p.shadow.caster_box={bx,0,5*bx,3*by};p.current=2;
 assert(p.frame_caster_scope()==1 && p.caster_refresh_owed);
 // The next frame (a display frame) collects the new set.
 p.renderer.scene_render=false;assert(p.frame_caster_scope()==2 && !p.caster_refresh_owed);
 // Without a display frame between them, the next scene render refreshes.
 p.renderer.scene_render=true;p.current=3;assert(p.frame_caster_scope()==1);assert(p.frame_caster_scope()==3);
 // A far jump, no drawn selection or a frame outside a scene render refresh at once.
 p.caster_refresh_owed=false;p.shadow.caster_box={3*bx,0,7*bx,3*by};assert(p.frame_caster_scope()==3);
 p.shadow.caster_box={bx,0,5*bx,3*by};p.shadow.caster_signature=~std::uint64_t(0);assert(p.frame_caster_scope()==3); // nothing drawn yet
 p.shadow.caster_signature=1;p.renderer.scene_render=false;assert(p.frame_caster_scope()==3);
 // Unchanged scope: nothing to defer.
 p.renderer.scene_render=true;p.current=1;assert(p.frame_caster_scope()==1 && !p.caster_refresh_owed);
 std::puts("PASS caster refresh deferral");
}
""")

if __name__ == '__main__':
    unittest.main()
