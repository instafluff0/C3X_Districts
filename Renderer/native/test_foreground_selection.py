"""Camera reuse must preserve the contributors that a cold view would select."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ForegroundSelectionTests(unittest.TestCase):
    def test_copied_body_permission_preserves_explored_fog_and_legacy_inputs(self):
        run_cpp(r'''
#include "Renderer/native/render_core/foreground_selection.h"
#include <cassert>
#include <cstring>
#include <initializer_list>
using namespace c3x_renderer::render_core;
int main(){
 constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED;
 c3x_renderer_tile_v1 tile{};tile.tile_flags=C3X_RENDERER_TILE_RENDER|known;
 tile.anchor_x=29;tile.anchor_y=47;auto original=tile;
 for(bool pickup:{false,true})for(bool offload:{false,true}){
  ForegroundSelection selection{2240,1260,128,64,4,2,pickup,offload};
  assert(!selection.selects(tile));assert(!std::memcmp(&tile,&original,sizeof tile));
  auto fog=tile;fog.tile_flags|=explored;
  assert(selection.selects(fog));assert(!selection.preserves(tile,fog));
  auto visible=fog;visible.tile_flags|=C3X_RENDERER_TILE_VISIBLE;
  assert(selection.selects(visible)&&selection.preserves(fog,visible));
  auto invalid=tile;invalid.tile_flags|=C3X_RENDERER_TILE_VISIBLE;assert(!selection.selects(invalid));
  auto legacy=tile;legacy.tile_flags&=~known;
  legacy.anchor_x=-99999;assert(selection.selects(legacy));
 }
 ForegroundSelection selection{2240,1260,128,64,4,2,true,false};
 tile.tile_flags=C3X_RENDERER_TILE_PREFETCH|known;assert(!selection.selects(tile));
 tile.tile_flags|=explored;assert(selection.selects(tile));
 tile.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO|known|explored;assert(!selection.selects(tile));
}
''')

    def test_wrapped_foreground_and_required_regions_share_permitted_bodies(self):
        run_cpp(r'''
#include "Renderer/native/render_core/foreground_selection.h"
#include "Renderer/native/render_core/world_preparation_region.h"
#include <cassert>
#include <cstring>
#include <set>
using namespace c3x_renderer::render_core;
int main(){
 c3x_renderer_frame_v1 frame{};frame.world_width_tiles=frame.world_height_tiles=60;
 frame.world_wrap_x=frame.world_wrap_y=1;frame.tile_width=128;frame.tile_height=64;
 frame.target_width=2240;frame.target_height=1260;
 std::vector<std::uint32_t> topology(1800,2u|(2u<<8));
 frame.world_topology=topology.data();frame.world_topology_count=unsigned(topology.size());
 frame.world_topology_revision=1;
 std::vector<c3x_renderer_tile_v1> records;
 for(int y=0;y<60;++y)for(int x=y&1;x<60;x+=2){
  c3x_renderer_tile_v1 tile{};tile.tile_x=x;tile.tile_y=y;
  tile.anchor_x=x*64-1376;tile.anchor_y=y*32-426;
  tile.terrain_type=tile.real_terrain_type=2;tile.city_id=tile.resource_id=-1;
  tile.variant_seed=unsigned(x*73856093u)^unsigned(y*19349663u);
  tile.tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|
   (records.size()<798?C3X_RENDERER_TILE_RENDER:C3X_RENDERER_TILE_TOPOLOGY_HALO);
  if(records.size()<11)tile.tile_flags|=C3X_RENDERER_TILE_EXPLORED;
  records.push_back(tile);
 }
 frame.tiles=records.data();frame.tile_count=unsigned(records.size());
 CapturedScene scene;scene.publication_scope(frame,{1,1,1,1},1);
 for(auto const& tile:records){bool changed=false;assert(scene.publish(tile,changed));}
 assert(scene.size()==1800&&scene.authoritative_size()==798);
 auto snapshot=scene.world_snapshot();auto appearance=scene.appearance_sequence();
 auto visibility=scene.visibility_sequence();auto scope=scene.scope_sequence();
 auto unknown_flags=snapshot->current(snapshot->key(records[797].tile_x,records[797].tile_y))->occurrence.tile_flags;
 auto original=records;auto original_topology=topology;
 ForegroundSelection selection{2240,1260,128,64,4,2,true,false};
 std::set<std::uint64_t> foreground,required;
 for(auto const& tile:records)if(selection.selects(tile))foreground.insert(scene.key(tile.tile_x,tile.tile_y));
 assert(foreground.size()==11);
 assert(WorldPreparationRegion::count(frame,true)==64);
 for(unsigned n=0;n<64;++n){
  WorldPreparationRegion region;assert(region.build(scene,frame,n,true));
  for(auto index:region.selected){auto const& tile=region.tiles[index];
   required.insert(scene.key(tile.tile_x,tile.tile_y));}
 }
 assert(required==foreground);
 // Wrapped native occurrences retain their independent screen placement.
 auto wrapped=records[0];wrapped.tile_x+=60;wrapped.tile_y-=60;
 wrapped.anchor_x+=3840;wrapped.anchor_y-=1920;
 assert(selection.selects(wrapped)&&foreground.count(scene.key(wrapped.tile_x,wrapped.tile_y)));
 auto unknown=records[797];unknown.tile_x-=60;unknown.tile_y+=60;
 assert(!selection.selects(unknown));
 // Selection cannot manufacture body facts, invalidate proofs or replace leases.
 assert(!std::memcmp(records.data(),original.data(),records.size()*sizeof(records[0])));
 assert(topology==original_topology&&scene.world_snapshot()==snapshot);
 assert(scene.appearance_sequence()==appearance&&scene.visibility_sequence()==visibility&&scene.scope_sequence()==scope);
 auto neighbor=snapshot->current(snapshot->key(59,-1));assert(neighbor);
 assert(neighbor->occurrence.terrain_type==2&&!(neighbor->occurrence.tile_flags&C3X_RENDERER_TILE_EXPLORED));
 auto promoted=records[797];promoted.tile_flags|=C3X_RENDERER_TILE_EXPLORED;
 assert(!selection.preserves(records[797],promoted));
 assert(snapshot->current(snapshot->key(promoted.tile_x,promoted.tile_y))->occurrence.tile_flags==unknown_flags);
}
''')

    def test_unknown_render_anchors_retain_wrapped_nine_cell_fog_feather(self):
        run_cpp(r'''
#include "Renderer/native/render_core/foreground_selection.h"
#include "Renderer/native/render_core/visibility_coverage.h"
#include <cassert>
#include <cstring>
using namespace c3x_renderer::render_core;
int main(){
 c3x_renderer_frame_v1 frame{};frame.world_width_tiles=frame.world_height_tiles=32;
 frame.world_wrap_x=frame.world_wrap_y=1;frame.target_width=128;frame.target_height=128;
 frame.tile_width=128;frame.tile_height=64;
 c3x_renderer_tile_v1 tiles[2]{};
 tiles[0].tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN;
 tiles[1]=tiles[0];tiles[1].tile_x=1;tiles[1].tile_y=31;
 tiles[1].anchor_x=64;tiles[1].anchor_y=-32;
 tiles[1].tile_flags|=C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBLE;
 frame.tiles=tiles;frame.tile_count=2;
 VisibilityCoverage before;assert(before.capture(frame));
 ForegroundSelection selection{128,128,128,64,4,2,true,false};
 assert(!selection.selects(tiles[0])&&selection.selects(tiles[1]));
 VisibilityCoverage after;assert(after.capture(frame));
 assert(before.states==after.states&&before.tiles.size()==after.tiles.size());
 for(unsigned n=0;n<before.tiles.size();++n){
  auto a=before.tiles[n],b=after.tiles[n];assert(a.x==b.x&&a.y==b.y&&a.cells==b.cells);
 }
 assert(after.state(0,0)==0&&after.state(1,-1)==2);
 auto unknown=std::find_if(after.tiles.begin(),after.tiles.end(),[](auto tile){return tile.x==0&&tile.y==0;});
 assert(unknown!=after.tiles.end());
 assert(VisibilityCoverage::coverage(unknown->cells,.5f,.5f,1)==0); // Opaque unknown center.
 assert(VisibilityCoverage::coverage(unknown->cells,.5f,.001f,1)>0); // Neighbor feather still exists.
 assert(VisibilityCoverage::coverage(unknown->cells,.5f,.001f,1)<1);
 // Body selection preserves the native fog RENDER anchors rather than clearing them.
 assert(tiles[0].tile_flags==(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN));
}
''')

    def test_camera_crosses_prefetch_boundary(self):
        run_cpp(r'''
#include "Renderer/native/render_core/foreground_selection.h"
#include <cassert>
#include <climits>
using namespace c3x_renderer::render_core;
int main(){
 ForegroundSelection select{2240,1260,128,64,4,0,true,false};
 c3x_renderer_tile_v1 old{};old.tile_flags=C3X_RENDERER_TILE_PREFETCH;
 old.anchor_x=-640;old.anchor_y=0;
 auto current=old;--current.anchor_x;
 assert(select.selects(old)&&!select.selects(current));
 assert(!select.preserves(old,current));
 old.anchor_x=2240+512;current=old;++current.anchor_x;
 assert(!select.preserves(old,current));
 old.anchor_x=0;old.anchor_y=-320;current=old;--current.anchor_y;
 assert(!select.preserves(old,current));
 old.anchor_y=1260+256;current=old;++current.anchor_y;
 assert(!select.preserves(old,current));
 old.anchor_x=old.anchor_y=0;current=old;current.anchor_x=57;
 assert(select.preserves(old,current)); // Ordinary interior reuse remains valid.
 old.tile_flags=current.tile_flags=C3X_RENDERER_TILE_RENDER;
 old.anchor_x=INT_MIN;current.anchor_x=INT_MAX;
 assert(select.preserves(old,current)); // Native visible ownership wins.
 old.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO;
 assert(!select.selects(old));
 old.tile_flags=C3X_RENDERER_TILE_PREFETCH;old.anchor_x=0;old.anchor_y=0;
 select.offload=true;assert(!select.selects(old));
 select.guard=2;assert(select.selects(old));
 current=old;current.anchor_x=2240+257;assert(!select.preserves(old,current));
 auto changed=select;changed.ring=2;assert(!(changed==select));
 changed=select;changed.width=1920;assert(!(changed==select));
}
''')
