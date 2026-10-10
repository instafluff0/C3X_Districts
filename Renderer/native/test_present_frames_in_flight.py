"""Native presents are bounded in flight, so Civ III cannot outrun the display.

Civ III's combat loop presented ~60 times a second; each present and its image
batches cost more transport time than that allows. Unbounded, the display fell
4.5 s behind and the publication overflowed (performance review, section 47).
"""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class PresentFramesInFlightTests(unittest.TestCase):
    def test_third_present_waits_for_the_first_and_faults_release_it(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_scene_client.h"
#include <cassert>
#include <chrono>
using namespace std::chrono_literals;
struct Shared {};
struct State {std::mutex mutex;std::condition_variable wake;bool open=false;int result=C3X_RENDERER_RESULT_OK;
 std::atomic<unsigned> entered{0},finished{0};};
struct Fake {
 State& state;explicit Fake(State& value):state(value){}
 void publication_pressure(std::size_t){}void supersede_pending_camera(){}
 int visual_policy(unsigned){return 1;}
 int present(c3x_renderer_gpu_present_v1 const&,Shared&){++state.entered;
  std::unique_lock<std::mutex> lock(state.mutex);state.wake.wait(lock,[&]{return state.open;});++state.finished;return state.result;}
};
void open(State& state){{std::lock_guard<std::mutex> lock(state.mutex);state.open=true;}state.wake.notify_all();}
template<class P>void until(P ready){auto end=std::chrono::steady_clock::now()+2s;
 while(!ready()){assert(std::chrono::steady_clock::now()<end);std::this_thread::sleep_for(1ms);}}
int main(){
 {State state;c3x_remote_scene::AsyncSceneClient<Fake> client(true,[](char const*){assert(false);},state);
  c3x_renderer_gpu_present_v1 value{sizeof(value)};value.action=1;Shared frame;
  // Two presents in flight return at once: one executing, one queued.
  assert(client.present(value,frame)==C3X_RENDERER_RESULT_OK);until([&]{return state.entered==1;});
  assert(client.present(value,frame)==C3X_RENDERER_RESULT_OK);
  // A third waits for the first to retire instead of queuing behind it.
  auto third=std::async(std::launch::async,[&]{return client.present(value,frame);});
  assert(third.wait_for(150ms)==std::future_status::timeout&&client.publication_status().accepted==2);
  open(state);assert(third.wait_for(2s)==std::future_status::ready&&third.get()==C3X_RENDERER_RESULT_OK);
  until([&]{return state.finished==3;});assert(client.publication_status().rejected==0);}
 {// A failed present faults the queue; a waiting present returns rather than hangs.
  State state;state.result=C3X_RENDERER_RESULT_DEVICE_ERROR;
  c3x_remote_scene::AsyncSceneClient<Fake> client(true,[](char const*){},state);
  c3x_renderer_gpu_present_v1 value{sizeof(value)};value.action=1;Shared frame;
  assert(client.present(value,frame)==C3X_RENDERER_RESULT_OK);until([&]{return state.entered==1;});
  assert(client.present(value,frame)==C3X_RENDERER_RESULT_OK);
  auto third=std::async(std::launch::async,[&]{return client.present(value,frame);});
  assert(third.wait_for(150ms)==std::future_status::timeout);
  open(state);assert(third.wait_for(2s)==std::future_status::ready&&third.get()!=C3X_RENDERER_RESULT_OK);}
}
''')


if __name__ == '__main__':
    unittest.main()
