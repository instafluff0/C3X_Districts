"""A camera job's required geometry may grow past the soft budget.

On the 1498 AD save, the world-residency sweep set the geometry budget to
436 MB. The first camera job's view needs about 785 MB. World geometry's
`future` reserve, for optional caches and attachments not yet allocated,
consumed all usable memory, so the budget could not grow. Every resident
entry belonged to that job, so nothing could be evicted. make_tile_cache_room
refused an 88 KB admission, the job failed, and the map stayed black for the
session.

With a 1.2 GB memory hold in the VM, a later failure showed: under a physical
shortfall, the soft budget is recomputed each frame as owned minus half the
shortfall. Evicting does not promptly raise available memory, so the budget
fell below the already resident view every frame (646, 621, ... 270 MB), and
42 of 47 camera jobs failed (performance review 4x).

Required geometry for a foreground camera job now has a second ceiling,
FrameWorkingSet::required_geometry. It keeps the system floor and compile
lanes, drops the optional future reserve, and never shrinks below what is
owned. Eviction of older content still comes first. Prewarming, loading and
prefetch keep the soft budget.
"""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp

MIB = 1024 * 1024


class RequiredGeometryAdmissionTests(unittest.TestCase):
    def test_current_view_grows_into_reserve_after_eviction_and_only_in_foreground(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        start = source.index('    bool make_tile_cache_room(std::size_t bytes) {')
        end = source.index('    bool cache_geometry_layer(', start)
        room = source[start:end]
        run_cpp(r'''
#include "Renderer/native/render_core/residency_candidates.h"
#include "Renderer/native/render_core/frame_working_set.h"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
using namespace c3x_renderer::render_core;
template<std::size_t N,class... A>void sprintf_s(char(&b)[N],char const* f,A... a){std::snprintf(b,N,f,a...);}
struct CachedTileGeometry {std::uint64_t signature=0;std::size_t byte_count=0;bool prefetched=false;
 std::uint64_t last_used=0,animation_epoch=0;ContentHandle binding,natural_content;bool world_ground=false;};
constexpr std::size_t tile_geometry_cache_capacity=8192;constexpr std::uint64_t viewport_cache_capacity=32;
struct Retired {std::atomic<std::size_t> bytes{0};};
struct Trace {std::vector<std::string> stages;void write(char const* stage,char const*,bool){stages.push_back(stage);}
 unsigned count(char const* stage)const{unsigned n=0;for(auto const& s:stages)n+=s==stage;return n;}};
struct Cache {
 std::unordered_multimap<std::uint64_t,CachedTileGeometry> tile_geometry_cache;
 ResidentContent<CachedTileGeometry> resident_content{tile_geometry_cache_capacity};
 ResidencyCandidates residency_candidates;
 std::shared_ptr<Retired> retired_content=std::make_shared<Retired>();
 std::size_t tile_geometry_cache_bytes=0,prefetched_geometry_bytes=0,terrain_patch_index_bytes=0;
 std::size_t tile_geometry_runtime_budget=0,tile_geometry_required_budget=0;
 std::uint64_t tile_geometry_epoch=1;bool loading_gpu_residency=false,world_gpu_capacity_refused=false;
 bool geometry_tier=false,world_gpu_residency_ready=false;
 unsigned frame_tiles_evicted=0,cache_evictions=0;
 struct {unsigned clears=0;void clear(){++clears;}} shared_instances;struct {void reset(){}} resource_visibility_membership;
 Trace trace;
 void release_resident_content(CachedTileGeometry& owner){resident_content.release(owner.binding);}
 void add(std::uint64_t key,std::size_t bytes,std::uint64_t used){
  auto it=tile_geometry_cache.emplace(key,CachedTileGeometry{key,bytes,false,used,0,{}});
  it->second.binding=resident_content.bind(it->second);tile_geometry_cache_bytes+=bytes;}
''' + room + r'''
};
int main(){
 constexpr std::size_t mib=1024u*1024u;
 // The VM's measurements at the failed job (roads profile1): 14 GiB,
 // 4.75 GB available, 3.29 GB adapter headroom. Any future reserve at least
 // the usable memory leaves growth at zero: the budget equals what loading
 // admitted, below the ~785 MB view.
 std::size_t available=4748341248u,physical=14336u*mib,gpu=3293000000u,owned=435947420u;
 auto soft=FrameWorkingSet::world_geometry(available,physical,owned,gpu,2048u*mib,8192u*mib,4,false);
 auto required=FrameWorkingSet::required_geometry(available,physical,owned,gpu,8192u*mib,4);
 assert(soft==owned);
 assert(required>=785166966u);
 // Under the 1.2 GB hold (run135, second job): 3.02 GB available with the
 // 785 MB view resident. The soft budget shrinks below the view; the
 // required ceiling keeps it and still admits the job's new tiles.
 std::size_t pressed=3019345920u,view=785166966u;
 assert(FrameWorkingSet::world_geometry(pressed,physical,view,gpu,1024u*mib,8192u*mib,4,false)<view);
 auto kept=FrameWorkingSet::required_geometry(pressed,physical,view,gpu,8192u*mib,4);
 assert(kept>=view+32u*mib);
 // It is bounded: by the system floor, the adapter and the ceiling.
 assert(FrameWorkingSet::required_geometry(2048u*mib,physical,view,gpu,8192u*mib,4)==view);
 assert(FrameWorkingSet::required_geometry(available,physical,view,0,8192u*mib,4)==view);
 assert(FrameWorkingSet::required_geometry(available,physical,view,gpu,view,4)==view);
 {
  // Every resident entry belongs to the current job (epoch 1).
  Cache c;c.tile_geometry_runtime_budget=436006339u;c.tile_geometry_required_budget=required;
  for(unsigned i=0;i<1218;++i)c.add(i,i<1217?357924u:owned-1217u*357924u,1);
  assert(c.tile_geometry_cache_bytes==owned);
  assert(c.make_tile_cache_room(87951));
  c.tile_geometry_cache_bytes+=87951;
  for(int i=0;i<20;++i){assert(c.make_tile_cache_room(357924));c.tile_geometry_cache_bytes+=357924;}
  assert(c.tile_geometry_cache.size()==1218 && c.frame_tiles_evicted==0);
  assert(c.trace.count("tile-cache-overflow")==1 && c.trace.count("tile-cache-budget")==0);
  // The second ceiling is still a ceiling.
  assert(!c.make_tile_cache_room(required) && c.trace.count("tile-cache-budget")==1);
 }
 {
  // Fixed geometry tier: view-specific owners (zoomed-out builds) leave
  // before the explored world's canonical content, even when newer.
  Cache c;c.tile_geometry_epoch=3;c.tile_geometry_runtime_budget=100u*mib;c.tile_geometry_required_budget=100u*mib;
  c.geometry_tier=c.world_gpu_residency_ready=true;
  for(unsigned i=0;i<6;++i){c.add(i,10u*mib,1);c.tile_geometry_cache.find(i)->second.world_ground=true;}
  for(unsigned i=6;i<10;++i)c.add(i,10u*mib,2);
  assert(c.make_tile_cache_room(30u*mib));
  unsigned canonical=0;for(auto const& entry:c.tile_geometry_cache)canonical+=entry.second.world_ground;
  assert(c.frame_tiles_evicted==3 && canonical==6);
  // Without the tier, least recently used goes first.
  Cache plain;plain.tile_geometry_epoch=3;plain.tile_geometry_runtime_budget=100u*mib;plain.tile_geometry_required_budget=100u*mib;
  for(unsigned i=0;i<6;++i){plain.add(i,10u*mib,1);plain.tile_geometry_cache.find(i)->second.world_ground=true;}
  for(unsigned i=6;i<10;++i)plain.add(i,10u*mib,2);
  assert(plain.make_tile_cache_room(30u*mib));
  unsigned kept=0;for(auto const& entry:plain.tile_geometry_cache)kept+=entry.second.world_ground;assert(kept==3);
 }
 {
  // Older content is evicted before any overflow.
  Cache c;c.tile_geometry_epoch=2;c.tile_geometry_runtime_budget=100u*mib;c.tile_geometry_required_budget=400u*mib;
  for(unsigned i=0;i<10;++i)c.add(i,10u*mib,i<4?1:2);
  assert(c.make_tile_cache_room(20u*mib));
  assert(c.frame_tiles_evicted==2 && c.tile_geometry_cache.size()==8 && c.trace.count("tile-cache-overflow")==0);
  unsigned old=0;for(auto const& entry:c.tile_geometry_cache)old+=entry.second.last_used==1;
  assert(old==2); // only older entries were evicted
 }
 {
  // Retired borrowers stay charged inside the second ceiling: the reclaim
  // runs once a call exceeds it, not on every admission (run136 traced it
  // 17,381 times while it freed nothing).
  Cache c;c.tile_geometry_runtime_budget=100u*mib;c.tile_geometry_required_budget=200u*mib;
  for(unsigned i=0;i<10;++i)c.add(i,10u*mib,1);
  c.retired_content->bytes=5u*mib;
  for(int i=0;i<50;++i){assert(c.make_tile_cache_room(1u*mib));c.tile_geometry_cache_bytes+=1u*mib;}
  assert(c.shared_instances.clears==0 && c.trace.count("tile-cache-reclaim")==0);
  assert(!c.make_tile_cache_room(200u*mib));
  assert(c.shared_instances.clears==1 && c.trace.count("tile-cache-reclaim")==1);
 }
 {
  // Prewarming, loading and missing measurements have no second ceiling.
  Cache c;c.tile_geometry_runtime_budget=100u*mib;
  for(unsigned i=0;i<10;++i)c.add(i,10u*mib,1);
  assert(!c.make_tile_cache_room(1) && c.trace.count("tile-cache-budget")==1);
  c.tile_geometry_required_budget=400u*mib;c.loading_gpu_residency=true;
  assert(!c.make_tile_cache_room(1) && c.world_gpu_capacity_refused);
 }
 std::puts("PASS required geometry admission: overflow after eviction, bounded, foreground only");
}
''')
        # The per-frame computation: foreground only, without the future
        # reserve, reset when unmeasured.
        budget = source[source.index('        tile_geometry_required_budget=0;\n        if(GlobalMemoryStatusEx(&content_memory)){'):]
        budget = budget[:budget.index('cpu_preparation_budget=')]
        self.assertIn('if(!prewarming && !loading_gpu_residency)\n'
                      '                    tile_geometry_required_budget=std::max(tile_geometry_runtime_budget,geometry_tier?Budget::world_tier(owned,gpu,0,ceiling):\n'
                      '                        Budget::required_geometry(', budget)
        self.assertEqual(budget.count('tile_geometry_required_budget='), 2)


    def test_tier_caps_the_budget_at_canonical_content_plus_view_allowance(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        self.assertTrue('if(geometry_tier && world_gpu_residency_ready && !loading_gpu_residency)\n'
                        '            tile_geometry_runtime_budget=std::min(tile_geometry_runtime_budget,canonical_geometry_bytes()+\n'
                        '                retired_content->bytes.load()+terrain_patch_index_bytes+view_geometry_allowance);' in source)
        self.assertTrue('static constexpr std::size_t view_geometry_allowance=384u*1024u*1024u;' in source)

    def test_whole_world_tier_follows_the_adapter_not_free_memory(self):
        """On a 12 GB VM the busy save (world 1.27 GB) loaded with no world
        geometry: the free-memory rule wanted about 4.7 GB free first (2.8 GB
        system reserve plus 1.9 GB of attachments not yet allocated) and the
        guest had 4.0 GB. After a debug reveal the camera jobs needed more
        than the free-memory ceiling allowed and every one failed (review 88).
        The geometry tier sizes the explored world by the adapter's budget."""
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        self.assertIn('tile_geometry_runtime_budget=geometry_tier?Budget::world_tier(owned,gpu,future,ceiling):Budget::world_geometry(', source)
        self.assertIn('tile_geometry_required_budget=std::max(tile_geometry_runtime_budget,geometry_tier?Budget::world_tier(owned,gpu,0,ceiling):', source)
        self.assertIn('auto recipes=geometry_tier && world_gpu_residency_ready?c3x_renderer::render_core::FrameWorkingSet::world_recipes(', source)
        self.assertIn('if(!world_backing.configure(recipes,', source)
        run_cpp(r'''
#include "Renderer/native/render_core/frame_working_set.h"
#include <cassert>
#include <cstdio>
using c3x_renderer::render_core::FrameWorkingSet;
int main(){
 constexpr std::size_t mib=1024u*1024u,ceiling=8192u*mib,world=1269u*mib;
 // Loading sweep on the 12 GB VM: 4.0 GB available, adapter budget 5.85 GB
 // with 0.67 GB used (headroom 3.72 GB after its 25% reserve), 1.9 GB future.
 std::size_t available=4000u*mib,physical=12288u*mib,headroom=3720u*mib,future=1900u*mib;
 assert(FrameWorkingSet::world_geometry(available,physical,0,headroom,future,ceiling,4,true)<world); // the old rule: none of it
 assert(FrameWorkingSet::world_tier(0,headroom,future,ceiling)>=world);
 // Debug reveal: 2.82 GB available, 0.77 GB owned, adapter headroom 2.16 GB.
 std::size_t owned=770u*mib;available=2820u*mib;headroom=2160u*mib;
 assert(FrameWorkingSet::required_geometry(available,physical,owned,headroom,ceiling,4)<owned+world); // old: no room for the revealed world
 assert(FrameWorkingSet::world_tier(owned,headroom,0,ceiling)>=owned+world);
 // An exhausted adapter admits nothing new but never falls below what is owned;
 // the ceiling still caps it.
 assert(FrameWorkingSet::world_tier(owned,0,future,ceiling)==owned);
 assert(FrameWorkingSet::world_tier(owned,headroom,0,owned+100u*mib)==owned+100u*mib);
 // The reveal's required world pass stores 618 MB of compressed recipes. With
 // 2.16 GB available the free-memory rule left the emptied store no capacity;
 // the commit limit (8.7 GB available) admits them.
 constexpr std::size_t recipes=618u*mib;
 auto ram=FrameWorkingSet::residency(2160u*mib,physical,owned,0,headroom,ceiling,4);
 assert(ram.prepared<recipes);
 assert(FrameWorkingSet::world_recipes(ram.prepared,0,8700u*mib)>=recipes);
 assert(FrameWorkingSet::world_recipes(ram.prepared,0,1024u*mib)==ram.prepared); // no commit headroom: unchanged
 std::puts("PASS world tier: the explored world is sized by the adapter and commit limit, not free memory");
}
''')

if __name__ == '__main__':
    unittest.main()
