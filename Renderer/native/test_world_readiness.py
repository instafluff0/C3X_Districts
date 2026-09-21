"""Whole-world pages and compiler leases preserve authority and visibility."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class WorldReadinessTests(unittest.TestCase):
    def test_returned_page_cannot_change_scope_or_replace_newer_publication(self):
        run_cpp(r'''
#include "Renderer/native/render_core/world_input_capture.h"
#include <cassert>
using namespace c3x_renderer::render_core;
int main(){
 ScenePublication journal;WorldInputCapture capture;std::vector<unsigned> topology(800,2);
 c3x_renderer_frame_v1 frame{};frame.world_width_tiles=40;frame.world_height_tiles=40;
 frame.world_topology=topology.data();frame.world_topology_count=unsigned(topology.size());
 c3x_renderer_camera_identity_v1 identity{1,2,3,4};assert(journal.capture(frame,identity));
 auto fill=[](c3x_renderer_world_page_v1& page){page.count=128;
  for(unsigned n=0;n<page.count;++n){auto& t=page.tiles[n];t={};t.tile_y=n/20;t.tile_x=2*(n%20)+(t.tile_y&1);t.tile_flags=C3X_RENDERER_TILE_PREFETCH;}};
 auto page=capture.page(*journal.state());fill(page);auto sequence=journal.state()->sequence;
 for(unsigned n=0;n<7;++n){auto bad=page;
  if(n==0)++bad.identity.map_epoch;if(n==1)++bad.identity.viewer_epoch;
  if(n==2)++bad.identity.visibility_epoch;if(n==3)++bad.identity.scene_epoch;
  if(n==4)++bad.frame.world_width_tiles;if(n==5)++bad.frame.world_topology_revision;
  if(n==6)++bad.frame.world_topology_count;
  assert(!capture.accept(bad,journal));assert(capture.cursor==0&&journal.state()->sequence==sequence);
 }
 assert(journal.capture(frame,identity)); // New publication, even in the same epoch.
 assert(!capture.accept(page,journal)&&capture.cursor==0);
 page=capture.page(*journal.state());fill(page);assert(capture.accept(page,journal)&&capture.cursor==128);
}
''')

    def test_paging_backpressure_scope_and_remote_removal(self):
        run_cpp(r'''
#include "Renderer/native/render_core/world_input_capture.h"
#include <cassert>
using namespace c3x_renderer::render_core;
int main(){
 ScenePublication journal;CapturedScene scene;WorldInputCapture capture;
 std::vector<unsigned> topology(800,2);
 c3x_renderer_frame_v1 frame{};frame.world_width_tiles=40;frame.world_height_tiles=40;
 frame.world_topology=topology.data();frame.world_topology_count=unsigned(topology.size());
 c3x_renderer_camera_identity_v1 identity{1,1,1,1};
 assert(journal.capture(frame,identity));bool changed=false;assert(journal.apply(scene,changed));
 auto fill=[](c3x_renderer_world_page_v1& page){
  page.count=std::min(page.capacity,page.frame.world_topology_count-page.first);
  for(unsigned i=0;i<page.count;++i){auto n=page.first+i;auto& t=page.tiles[i];t={};
   t.tile_y=int(n/20);t.tile_x=int(n%20)*2+(t.tile_y&1);t.city_id=int(n);t.resource_id=4;
   t.tile_flags=C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_VISIBILITY_KNOWN;
  }
 };
 auto page=capture.page(*journal.state());fill(page);
 ScenePublication full(128);assert(!capture.accept(page,full)&&capture.cursor==0);
 ++page.tiles[0].tile_x;assert(!capture.accept(page,journal));--page.tiles[0].tile_x;
 --page.count;assert(!capture.accept(page,journal));++page.count;
 assert(capture.accept(page,journal));page.tiles[0].city_id=999;
 assert(journal.apply(scene,changed)&&scene.retained(scene.key(0,0))->appearance.city_id==0);
 while(!capture.passes){page=capture.page(*journal.state());fill(page);assert(capture.accept(page,journal));assert(journal.apply(scene,changed));}
 assert(scene.authoritative_size()==800&&capture.pages==7&&capture.records==800);
 assert(!(scene.retained(scene.key(0,0))->visibility_flags&C3X_RENDERER_TILE_EXPLORED));
 auto sequence=scene.appearance_sequence();page=capture.page(*journal.state());fill(page);
 assert(capture.accept(page,journal)&&journal.apply(scene,changed)&&!changed&&sequence==scene.appearance_sequence());
 page=capture.page(*journal.state());fill(page);auto removed=scene.key(page.tiles[0].tile_x,page.tiles[0].tile_y);
 page.tiles[0].city_id=page.tiles[0].resource_id=-1;
 assert(capture.accept(page,journal)&&journal.apply(scene,changed)&&changed);
 assert(scene.retained(removed)->appearance.city_id==-1&&scene.retained(removed)->appearance.resource_id==-1);
 ++identity.viewer_epoch;assert(journal.capture(frame,identity)&&journal.apply(scene,changed));
 page=capture.page(*journal.state());assert(page.first==0&&!capture.passes&&!scene.authoritative_size());
 fill(page);assert(capture.accept(page,journal)&&journal.apply(scene,changed)&&scene.authoritative_size()==128);
 journal.reset();assert(journal.capture(frame,identity));page=capture.page(*journal.state());assert(page.first==0);
}
''')

    def test_regions_require_complete_authority_and_never_grant_pixels(self):
        run_cpp(r'''
#include "Renderer/native/render_core/world_preparation_region.h"
#include <cassert>
using namespace c3x_renderer::render_core;
int main(){
 CapturedScene scene;c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=40;
 f.world_wrap_x=f.world_wrap_y=1;
 f.tile_width=128;f.tile_height=64;f.target_width=1120;f.target_height=1192;
 scene.publication_scope(f,{1,1,1,1},1);WorldPreparationRegion region;
 assert(!region.build(scene,f,0));bool changed=false;
 for(int y=0;y<40;++y)for(int x=y&1;x<40;x+=2){c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;
  t.city_id=y*40+x;t.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO;assert(scene.publish(t,changed));}
 assert(!scene.authoritative_size()&&!region.build(scene,f,0));
 for(int y=0;y<40;++y)for(int x=y&1;x<40;x+=2){c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;
  t.city_id=y*40+x;t.tile_flags=C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_VISIBILITY_KNOWN;
  assert(scene.publish(t,changed));}
 assert(scene.authoritative_size()==800);
 for(unsigned n=0;n<region.count(f);++n){assert(region.build(scene,f,n));assert(region.selected.size()==32);
  for(auto const&t:region.tiles){auto key=scene.key(t.tile_x,t.tile_y);assert(t.city_id==int(std::uint32_t(key))*40+int(std::uint32_t(key>>32)));
   assert(!(t.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBLE)));}
 }
 f.world_wrap_x=f.world_wrap_y=1;assert(region.build(scene,f,0));
 assert(region.tiles.size()==512);bool wrapped=false;
 for(auto const&t:region.tiles)if(t.tile_x<0&&t.anchor_x<0)wrapped=true;
 assert(wrapped);assert(!region.build(scene,f,region.count(f)));
 // Compiler leases contain values, unaffected by later world publication.
 assert(region.build(scene,f,0));auto old=region.tiles.front();auto next=old;next.city_id=-1;
 assert(scene.publish(next,changed)&&region.tiles.front().city_id==old.city_id);
 f.world_width_tiles=f.world_height_tiles=16;assert(!region.build(scene,f,0));
}
''')


    def test_large_world_input_capacity(self):
        run_cpp(r'''
#include "Renderer/native/render_core/world_input_capture.h"
#include "Renderer/native/render_core/world_preparation_region.h"
#include <chrono>
#include <cstdio>
#include <cassert>
#include <set>
using namespace c3x_renderer::render_core;
int main(){for(int size:{100,160,332}){
 ScenePublication journal;CapturedScene scene;WorldInputCapture capture;WorldPreparationRegion region;
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=size;f.world_wrap_x=1;
 f.tile_width=128;f.tile_height=64;f.target_width=1120;f.target_height=1192;
 std::vector<unsigned> topology(size*size/2,2);f.world_topology=topology.data();f.world_topology_count=unsigned(topology.size());
 assert(journal.capture(f,{1,1,1,1}));bool changed=false;assert(journal.apply(scene,changed));
 auto start=std::chrono::steady_clock::now();double maximum=0;
 while(!capture.passes){auto begin=std::chrono::steady_clock::now();auto p=capture.page(*journal.state());
  p.count=std::min(p.capacity,f.world_topology_count-p.first);
  for(unsigned n=0;n<p.count;++n){auto i=p.first+n;auto& t=p.tiles[n];t={};t.tile_y=i/(size/2);t.tile_x=2*(i%(size/2))+(t.tile_y&1);
   t.city_id=i%31==0?int(i):-1;t.resource_id=i%7==0?4:-1;t.tile_flags=C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_VISIBILITY_KNOWN;}
  assert(capture.accept(p,journal)&&journal.apply(scene,changed));maximum=std::max(maximum,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());
 }
 auto captured=std::chrono::steady_clock::now();unsigned selected=0;std::set<std::uint64_t> identities;
 for(unsigned n=0;n<region.count(f);++n){assert(region.build(scene,f,n));selected+=unsigned(region.selected.size());
  for(auto i:region.selected)identities.insert(scene.key(region.tiles[i].tile_x,region.tiles[i].tile_y));}
 assert(identities.size()==topology.size()&&scene.authoritative_size()==identities.size()&&selected>identities.size());
 std::printf("WORLD_INPUT_CAPACITY tiles=%u occurrences=%u pages=%llu record_bytes=%zu journal_peak=%zu regions=%u copy_publish_ms=%.3f page_max_ms=%.3f region_lease_ms=%.3f record_limit=%zu\n",unsigned(identities.size()),selected,(unsigned long long)capture.pages,scene.bytes(),journal.peak,region.count(f),std::chrono::duration<double,std::milli>(captured-start).count(),maximum,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-captured).count(),scene.record_limit);
}}
''')


if __name__ == '__main__':
    unittest.main()
