"""Host-only contracts for the production bounded static-raster state helper.

These checks own fake targets and consume the production selector/writer policy.
They do not qualify D3D pixels, reflection writers, or publication behavior.
"""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class StaticRasterStateTests(unittest.TestCase):
    def test_two_slots_invalidation_writer_transactions_and_target_lifetime(self):
        run_cpp(r'''
#include "Renderer/sandbox/static_raster_state.h"
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <type_traits>
#include <vector>
using namespace c3x_renderer::render_core;
void require(bool value,char const* message){if(!value)throw std::runtime_error(message);}
struct FakeTarget {
 struct Token{unsigned id;std::size_t bytes;};
 inline static unsigned constructed=0,destroyed=0,allocated=0,released=0,live=0,peak=0;
 inline static std::size_t charged=0;
 Token* token=nullptr;
 FakeTarget(){++constructed;}
 FakeTarget(FakeTarget const&)=delete;
 FakeTarget& operator=(FakeTarget const&)=delete;
 ~FakeTarget(){reset();++destroyed;}
 void reset(){if(token){charged-=token->bytes;delete token;token=nullptr;++released;--live;}}
 void allocate(std::size_t bytes){
  require(!token,"fixture must not overwrite a live owner");
  token=new Token{++allocated,bytes};charged+=bytes;++live;if(live>peak)peak=live;
 }
 std::size_t bytes()const{return token?token->bytes:0;}
 unsigned id()const{return token?token->id:0;}
};
using Pair=StaticRasterStates<FakeTarget>;
static_assert(!std::is_copy_constructible<Pair>::value,"pair must not copy owning targets");
static_assert(!std::is_copy_assignable<Pair>::value,"pair must not copy owning targets");
static_assert(!std::is_copy_constructible<StaticRasterState<FakeTarget>>::value,"slot must not copy its target");
static_assert(!std::is_copy_assignable<StaticRasterState<FakeTarget>>::value,"slot must not copy its target");
std::size_t charge(Pair const& pair){return std::size_t(pair.layout[0]+640)*(pair.layout[1]+384)*12*pair.layout[2];}
auto& warm(Pair& pair,float zoom,int x=-64,int y=32,float depth=1408){
 auto& state=pair.select(zoom);pair.begin_write();
 if(!state.region.id())state.region.allocate(charge(pair));
 state.valid=true;++state.metrics.full_draws;pair.restored(x,y,depth);return state;
}
int main(){try{
 {
  Pair pair;
  require(FakeTarget::constructed==2&&FakeTarget::allocated==0&&pair.bytes()==0,"exactly two lazy target owners");
  require(pair.states[0].projection==1&&pair.current().projection==1,"canonical identity must start at 1x");
  require(!pair.states[0].valid&&!pair.states[1].valid&&pair.writer.slot==2,"cold raster and scratch must be invalid");
  pair.set_layout(2248,1268,1);
  auto& canonical=warm(pair,1);
  require(&canonical==&pair.states[0]&&pair.selected==0,"1x must select canonical");
  require(FakeTarget::live==1&&!pair.states[1].region.id(),"display allocation must remain lazy");
  auto canonical_id=canonical.region.id();auto canonical_revision=canonical.revision;
  auto& display=warm(pair,1.25f);auto display_id=display.region.id();auto display_revision=display.revision;
  require(&display==&pair.states[1]&&pair.selected==1,"non1x must use single display owner");
  require(FakeTarget::live==2&&pair.bytes()==114503424&&pair.bytes()==FakeTarget::charged,"both region bytes must be charged");
  // Same camera/depth at alternating projections is precisely the false-hit
  // case that caused canonical/display thrash in the actual renderer.
  for(unsigned i=0;i<100;++i){
   auto zoom=i%2?1.25f:1.f;auto& state=pair.select(zoom);
   require(state.valid,"selection must preserve a warm matching raster");
   require(pair.needs_restore(true,-64,32,1408),"different slot must miss shared viewport writer");
   pair.restored(-64,32,1408);
   require(!pair.needs_restore(true,-64,32,1408),"same slot/view/depth/revision should hit restored scratch");
   auto& again=pair.select(zoom);require(&again==&state&&!pair.needs_restore(true,-64,32,1408),"same-state selection should not evict scratch");
  }
  pair.select(1);pair.restored(-64,32,1408);pair.select(1.25f);
  pair.begin_restore(); // Simulate a partial failed write; no restored() commit.
  require(pair.needs_restore(true,-64,32,1408),"failed cross-slot restore must leave selected scratch invalid");
  pair.select(1);
  require(pair.needs_restore(true,-64,32,1408),"failed cross-slot restore must retire the previous writer too");
  require(canonical.revision==canonical_revision&&display.revision==display_revision,"alternation must not invalidate static generations");
  require(FakeTarget::allocated==2&&FakeTarget::peak==2,"alternation must not allocate a pipeline or zoom LRU");
  for(float zoom:{1.5f,3.f,1.25f,2.f,1.25f}){
   auto& selected=pair.select(zoom);
   require(&selected==&display&&!display.valid,"display zoom replacement must invalidate its single identity");
   require(canonical.valid&&canonical.revision==canonical_revision&&canonical.region.id()==canonical_id,"display replacement must preserve canonical raster/owner");
   require(display.region.id()==display_id,"projection replacement should reuse display target storage");
   warm(pair,zoom);
  }
  require(FakeTarget::allocated==2&&FakeTarget::live==2,"display replacement must stay bounded");
  // The invalidation arrives while one slot is inactive. No second lighting
  // or scene-change event is delivered when that inactive slot is selected.
  for(auto reason:{raster_environment,raster_lights,raster_scene,raster_classification,raster_explicit_reset}){
   for(unsigned active:{0u,1u}){
    warm(pair,1);warm(pair,1.25f);pair.select(active?1.25f:1.f);
    auto a=canonical.revision,b=display.revision;
    auto ca=canonical.metrics.reasons[reason],cb=display.metrics.reasons[reason];
    pair.invalidate_all(reason);
    require(!canonical.valid&&!display.valid&&pair.writer.slot==2,"common changes must retire active AND inactive slots");
    require(canonical.revision==a+1&&display.revision==b+1,"both changed identities must advance");
    require(canonical.metrics.reasons[reason]==ca+1&&display.metrics.reasons[reason]==cb+1,"invalidation attribution must cover both warm owners");
    require(!pair.select(active?1.f:1.25f).valid,"inactive return must remain invalid without another change event");
   }
  }
  // Other-slot scratch may already have the new layout. Both region owners
  // must retire old sample/layout storage, rather than trust scratch's size.
  for(auto next:std::vector<std::array<unsigned,3>>{{2248,1268,2},{2248,1268,4},{2248,1268,1},{2264,1268,1},{2264,1284,1}}){
   warm(pair,1);warm(pair,1.25f);auto before=FakeTarget::released;
   pair.set_layout(next[0],next[1],next[2]);
   require(pair.layout==next&&!canonical.valid&&!display.valid&&pair.writer.slot==2,"layout/sample change must retire both identities and scratch");
   require(pair.bytes()==0&&FakeTarget::live==0&&FakeTarget::released==before+2,"layout reset must release both old owning targets");
   warm(pair,1.25f);
   require(FakeTarget::live==1&&!canonical.region.id(),"inactive canonical must stay lazy after layout reset");
   require(!pair.select(1).valid,"inactive old-sample canonical must not become valid merely on selection");
   pair.select(1.25f);auto revision=display.revision;auto id=display.region.id();
   pair.set_layout(next[0],next[1],next[2]);
   require(display.valid&&display.revision==revision&&display.region.id()==id&&!pair.needs_restore(true,-64,32,1408),"unchanged layout must preserve target, raster and writer");
  }
  warm(pair,1);warm(pair,1.25f);pair.select(1);
  require(pair.needs_restore(true,-64,32,1408),"matching camera/depth from other slot must still miss");
  pair.restored(-64,32,1408);
  require(pair.needs_restore(false,-64,32,1408),"unready raster must always restore");
  require(pair.needs_restore(true,-63,32,1408)&&pair.needs_restore(true,-64,31,1408),"both camera coordinates participate in writer identity");
  require(pair.needs_restore(true,-64,32,1409),"actual native depth basis participates in writer identity");
  auto old_writer=pair.writer;
  // A stale writer record can carry identical slot/camera/depth but refer to
  // prior pixels. Revision proof must reject it even if the tag is retained.
  pair.invalidate(0,raster_scene);canonical.valid=true;pair.writer=old_writer;
  require(pair.needs_restore(true,-64,32,1408),"same-slot raster mutation must defeat stale writer revision");
  pair.restored(-64,32,1408);require(!pair.needs_restore(true,-64,32,1408),"current completed revision may restore scratch");
  // Region write failure: preserve the completed other-slot target and its
  // still-untouched scratch, while never committing the partially written slot.
  warm(pair,1.25f);auto other_id=display.region.id();auto other_revision=display.revision;
  pair.select(1);auto previous=canonical.revision;pair.begin_write();
  require(!canonical.valid&&canonical.revision==previous+1,"begin_write must retire raster before partial blending");
  require(display.valid&&display.region.id()==other_id&&display.revision==other_revision,"region failure must not destroy untouched other owner");
  require(pair.writer.slot==1,"region mutation must preserve an untouched other-slot viewport");
  pair.invalidate(0,raster_error);
  require(!canonical.valid&&pair.needs_restore(false,-64,32,1408),"failed region must not commit a usable raster or writer");
  pair.select(1.25f);require(!pair.needs_restore(true,-64,32,1408),"untouched completed other-slot scratch remains valid");
  warm(pair,1);previous=canonical.revision;pair.begin_write();
  require(!canonical.valid&&canonical.revision==previous+1&&pair.writer.slot==2,"mutating writer's own region must retire its scratch before failure");
  require(pair.needs_restore(true,-64,32,1408),"no restore commit may survive a failed own-region write");
  require(FakeTarget::peak<=2&&pair.bytes()==FakeTarget::charged,"target ownership and charged bytes remain bounded");
 }
 require(FakeTarget::destroyed==2&&FakeTarget::live==0&&FakeTarget::charged==0,"destructor must release both owner lifetimes exactly once");
 require(FakeTarget::released==FakeTarget::allocated,"all lazy allocations must be released once");
 std::printf("PASS static raster state: two_owners=1 lazy_allocation=1 alternations=100 canonical_preserved=1 inactive_invalidation=1 sample_layout_reset=1 writer_slot_camera_depth_revision=1 failed_region_no_commit=1 releases=%u allocations=%u peak_live=%u\n",FakeTarget::released,FakeTarget::allocated,FakeTarget::peak);return 0;
 }catch(std::exception const& error){std::fprintf(stderr,"FAIL static raster state: %s\n",error.what());return 1;}}
''')


if __name__ == '__main__':
    unittest.main()
