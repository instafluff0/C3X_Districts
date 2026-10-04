"""Host-only contracts for the production bounded static-raster state helper.

These checks own fake targets and consume the production lane/front/back policy.
They do not qualify D3D pixels, reflection writers, or publication behavior.
"""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class StaticRasterStateTests(unittest.TestCase):
    def test_lanes_front_back_stale_preview_and_target_lifetime(self):
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
 inline static unsigned constructed=0,destroyed=0,allocated=0,released=0,live=0;
 inline static std::size_t charged=0;
 Token* token=nullptr;
 FakeTarget(){++constructed;}
 FakeTarget(FakeTarget const&)=delete;
 FakeTarget& operator=(FakeTarget const&)=delete;
 ~FakeTarget(){reset();++destroyed;}
 void reset(){if(token){charged-=token->bytes;delete token;token=nullptr;++released;--live;}}
 void allocate(std::size_t bytes){require(!token,"fixture must not overwrite a live owner");
  token=new Token{++allocated,bytes};charged+=bytes;++live;}
 std::size_t bytes()const{return token?token->bytes:0;}
 unsigned id()const{return token?token->id:0;}
};
using States=StaticRasterStates<FakeTarget>;
static_assert(!std::is_copy_constructible<States>::value,"states must not copy owning targets");
static_assert(!std::is_copy_assignable<States>::value,"states must not copy owning targets");
static_assert(!std::is_copy_constructible<StaticRasterState<FakeTarget>>::value,"slot must not copy its target");
void complete(StaticRasterState<FakeTarget>& slot,StaticRasterKey key,float projection){
 if(!slot.region.id())slot.region.allocate(1000);
 slot.key=key;slot.projection=projection;slot.valid=true;slot.stale=false;slot.refining=false;
 slot.covered={0,0,100,100};++slot.metrics.full_draws;
}
int main(){try{
 StaticRasterKey key{{1,2,3,4,5,6}},other{{1,2,3,9,5,6}};
 {
  States states;
  require(FakeTarget::constructed==4&&FakeTarget::allocated==0&&states.bytes()==0,"four lazy slot owners");
  require(States::lane_of(1.f)==0&&States::lane_of(1.25f)==1&&States::lane_of(3.f)==1,"exact 1x is the canonical lane");
  require(&states.select_lane(0)==&states.states[0]&&&states.select_lane(1)==&states.states[2],"each lane starts at its first slot");
  require(states.back_index(0)==1&&states.back_index(1)==3,"back slot is the lane's other slot");
  // Lane 1 front and back alternate; lane 0 is never touched by lane 1 work.
  complete(states.front(0),key,1.f);
  for(unsigned i=0;i<10;++i){
   auto& back=states.back(1);back.refining=true;back.valid=false;
   complete(back,key,i%2?1.25f:2.f);back.refining=false;
   states.promote(1);
   require(&states.front(1)==&back&&states.front(1).fresh(key),"promoted back becomes the fresh front");
   require(states.back(1).stale&&!states.back(1).refining,"replaced front becomes a stale, idle back");
   require(states.front(0).fresh(key),"lane 1 promotion must not disturb the canonical lane");
  }
  require(FakeTarget::allocated==3&&FakeTarget::live==3,"slots allocate once and are reused");
  // Lighting changes keep pixels as a preview, never as reusable fresh pixels.
  auto revision=states.front(0).revision;
  states.invalidate_all(raster_environment);
  for(auto const& slot:states.states)require(!slot.fresh(key),"invalidated slots are not fresh");
  require(states.front(0).valid&&states.front(0).stale&&states.front(0).revision==revision+1,"stale slots stay displayable");
  complete(states.front(0),key,1.f);
  require(states.front(0).fresh(key)&&!states.front(0).fresh(other),"freshness requires the current content key");
  // A scene edit cannot reuse even a complete old preview beneath new water.
  auto pixels=states.front(0).region.id();
  states.invalidate(states.front_slot[0],raster_scene);
  require(!states.front(0).valid&&states.front(0).stale,"scene changes retire displayed pixels");
  require(states.front(0).region.id()==pixels,"retiring pixels keeps the bounded allocation");
  // Refinement bookkeeping is per lane.
  states.back(1).refining=true;
  require(states.refining()&&states.lane_refining(1)&&!states.lane_refining(0),"refinement is tracked per lane");
  // Errors discard pixels entirely.
  states.discard_all(raster_error);
  for(auto const& slot:states.states)require(!slot.valid&&!slot.refining&&slot.covered.empty(),"discarded slots are unusable");
  // Layout changes release every owning target; an unchanged layout keeps them.
  auto before=FakeTarget::released;
  states.set_layout(2248,1268,1);
  require(states.bytes()==0&&FakeTarget::live==0&&FakeTarget::released==before+3,"layout reset releases all region targets");
  states.front(1).region.allocate(10);states.set_layout(2248,1268,1);
  require(FakeTarget::live==1,"unchanged layout keeps region targets");
  // Rect helpers used by coverage extension.
  StaticRasterState<FakeTarget>::Rect a{0,0,10,10},b{2,2,5,5},empty{3,3,3,9};
  require(a.contains(b)&&!b.contains(a)&&a.contains(empty)&&empty.empty()&&a.area()==100&&empty.area()==0,"rect helpers");
 }
 require(FakeTarget::destroyed==4&&FakeTarget::live==0&&FakeTarget::charged==0,"destructor releases all owners exactly once");
 std::printf("PASS static raster lanes: lazy_slots=4 promotions=10 stale_preview=1 discard=1 layout_release=1\n");return 0;
 }catch(std::exception const& error){std::fprintf(stderr,"FAIL static raster lanes: %s\n",error.what());return 1;}}
''')


if __name__ == '__main__':
    unittest.main()
