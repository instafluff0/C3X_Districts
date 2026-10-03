"""Join actual source readiness after native setup, using owned ordered input."""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def fixture():
    client = (ROOT / 'Renderer/native/remote_renderer_client.h').read_text()
    start = client.index('    static c3x_inputs::Bytes definition_input(')
    methods = client[start:client.index('    int pack(', start)]
    runtime = (ROOT / 'Renderer/native/input_recording/runtime.h').read_text()
    start = runtime.index('inline void settings_fields(')
    settings = runtime[start:runtime.index('class Runtime', start)]
    backend = (ROOT / 'Renderer/native/remote_renderer_backend.h').read_text()
    start = backend.index('    int screen(c3x_native_images::ScreenSnapshot const* source)')
    screen = backend[start:backend.index('    int reset()', start)]
    return r'''
#include "Renderer/sandbox/async_scene_client.h"
#include "Renderer/native/input_recording/journal.h"
#include <cassert>
#include <chrono>
using namespace std::chrono_literals;
namespace c3x_inputs {
using Settings=std::vector<std::pair<std::string,std::string>>;
Settings caller_settings={{"C3X_RENDERER_PACK_ROOT","original-pack"}};
Settings input_settings(){return caller_settings;}
''' + settings + r'''
}
struct State {
 std::promise<void> entered,release;std::shared_future<void> held=release.get_future().share();
 std::atomic<unsigned> prepared{0},scopes{0},worlds{0},resets{0},errors{0},retired{0};
 std::atomic<bool> released{false},fail{false};
 std::vector<std::string> roots,settings;std::vector<bool> scenario_present;
 void unblock(){released=true;release.set_value();}
};
struct Fake {
 State& state;
 struct Transport {State& state;void retire_camera_receipt(){++state.retired;}} transport;
 struct Reply {unsigned code;} reply{};
 explicit Fake(State& s):state(s),transport{s}{}
 bool alive()const{return true;}
 void publication_pressure(std::size_t){}
 int stats(){return int(state.prepared.load());}
 Reply const& invoke(unsigned kind,unsigned subtype,unsigned char const* data,unsigned bytes){
  assert(kind==unsigned(c3x_inputs::Kind::native_bridge)&&subtype==8);
  if(state.prepared==0){state.entered.set_value();state.held.wait();}
  c3x_inputs::Bytes owned(data,data+bytes);c3x_inputs::Reader in{owned};bool present=false;
  state.roots.push_back(in.string(32768));assert(in.string(32768)=="default");
  auto scenario=in.string(32768,&present);state.scenario_present.push_back(present);
  assert(!present||scenario.empty());assert(in.string(32768,&present).empty()&&present);
  assert(in.u32()==1&&in.string(256)=="C3X_RENDERER_PACK_ROOT");
  state.settings.push_back(in.string(32768));in.done();
  ++state.prepared;reply.code=state.fail?C3X_RENDERER_RESULT_ERROR:C3X_RENDERER_RESULT_OK;return reply;
 }
''' + methods + r'''
 int seed_world_scope(c3x_renderer_camera_request_v1 const&,bool loading=false){
  assert(loading&&state.released&&state.prepared);++state.scopes;return C3X_RENDERER_RESULT_OK;
 }
 int prepare_world_loading(c3x_renderer_camera_identity_v1 const&){
  assert(state.scopes&&state.prepared);++state.worlds;return C3X_RENDERER_RESULT_OK;
 }
 int reset(){assert(state.released&&state.prepared);++state.resets;return C3X_RENDERER_RESULT_OK;}
};
using Client=c3x_remote_scene::AsyncSceneClient<Fake>;
namespace c3x_native_images {
struct ScreenSnapshot {
 struct Area {int left=0,top=0,right=640,bottom=480;} area;
 int width=640,height=480,native_format=1;void* window=nullptr;std::vector<unsigned short> pixels;
};
}
struct Menu {
 std::mutex gate;bool visual_active=false,direct_active=false;void* active_window=nullptr;
 unsigned uploads=0,presents=0;
 struct Cadence {void disable(){}} cadence;
 struct Device {void* Get(){return nullptr;}} device,context;
 struct Presenter {
  Menu& owner;bool initialized=false;
  bool preserve_display(void*){return true;}void release_native(){}
  bool prepare(void*,void*,unsigned,unsigned,bool){return true;}
  bool seed_bgra(void*,unsigned const*,unsigned,unsigned){return true;}
  bool upload_screen(void*,unsigned short const*,unsigned,unsigned,
                     c3x_native_images::ScreenSnapshot::Area const&,int){++owner.uploads;return true;}
  int present(){++owner.presents;return C3X_RENDERER_RESULT_OK;}
 } presenter{*this};
 bool graphics(){return true;}
 bool detach_direct(bool,std::vector<unsigned>* =nullptr,unsigned* =nullptr,unsigned* =nullptr){assert(false);return false;}
''' + screen + r'''
};
void await_records(Client& client,std::size_t wanted){
 auto limit=std::chrono::steady_clock::now()+2s;
 while(client.publication_status().records<wanted&&std::chrono::steady_clock::now()<limit)std::this_thread::yield();
 assert(client.publication_status().records>=wanted);
}
'''


class AsyncSourceLoadingTests(unittest.TestCase):
    def test_owned_definition_admission_leaves_ui_free_and_world_barriers_join(self):
        run_cpp(fixture() + r'''
int main(){
 State state;Client client(true,[&](char const*){++state.errors;},state);
 char root[]="original-root",fallback[]="default",empty[]="";
 assert(client.definitions(root,fallback,nullptr,empty)==C3X_RENDERER_RESULT_OK);
 assert(state.entered.get_future().wait_for(2s)==std::future_status::ready);
 // Native setup work runs on this caller while the real source command is held.
 Menu menu;c3x_native_images::ScreenSnapshot setup;
 assert(menu.screen(&setup)==C3X_RENDERER_RESULT_OK);
 assert(menu.uploads==1&&menu.presents==1&&!state.prepared);
 root[0]='X';fallback[0]='X';c3x_inputs::caller_settings[0].second="replacement-pack";
 c3x_renderer_frame_v1 frame{};c3x_renderer_camera_request_v1 request{};request.frame=&frame;
 auto scope=std::async(std::launch::async,[&]{return client.seed_world_loading_scope(request);});
 await_records(client,2);assert(scope.wait_for(20ms)==std::future_status::timeout&&!state.scopes);
 state.unblock();assert(scope.get()==C3X_RENDERER_RESULT_OK);
 assert(client.prepare_world_loading({})==C3X_RENDERER_RESULT_OK);
 assert(state.roots==std::vector<std::string>{"original-root"});
 assert(state.settings==std::vector<std::string>{"original-pack"});
 assert(state.scenario_present==std::vector<bool>{false});
 assert(state.worlds==1&&state.retired==1&&!state.errors&&client.alive());
 assert(client.publication_status().peak_bytes>=256);
}
''')

    def test_source_failure_refuses_queued_world_and_reset_recovers(self):
        run_cpp(fixture() + r'''
int main(){
 State state;state.fail=true;Client client(true,[&](char const*){++state.errors;},state);
 assert(client.definitions("original-root","default","","")==C3X_RENDERER_RESULT_OK);
 state.entered.get_future().wait();c3x_renderer_frame_v1 frame{};
 c3x_renderer_camera_request_v1 request{};request.frame=&frame;
 auto scope=std::async(std::launch::async,[&]{
  try{client.seed_world_loading_scope(request);return false;}catch(std::exception const&){return true;}
 });
 await_records(client,2);state.unblock();assert(scope.get());
 assert(!client.alive()&&state.errors==1&&state.scopes==0&&state.worlds==0);
 assert(state.scenario_present==std::vector<bool>{true});
 assert(client.reset()==C3X_RENDERER_RESULT_OK&&state.resets==1&&client.alive());
 auto stats=client.publication_status();assert(stats.abandoned==2&&stats.bytes==0&&stats.records==0);
}
''')

    def test_reset_joins_active_source_before_new_configuration(self):
        run_cpp(fixture() + r'''
int main(){
 State state;Client client(true,[&](char const*){++state.errors;},state);
 assert(client.definitions("old","default",nullptr,"")==C3X_RENDERER_RESULT_OK);
 state.entered.get_future().wait();
 auto reset=std::async(std::launch::async,[&]{return client.reset();});
 assert(reset.wait_for(20ms)==std::future_status::timeout&&state.resets==0);
 state.unblock();assert(reset.get()==C3X_RENDERER_RESULT_OK&&state.resets==1);
 c3x_inputs::caller_settings[0].second="new-pack";
 assert(client.definitions("new","default","","")==C3X_RENDERER_RESULT_OK);
 assert(client.stats()==2);
 assert(state.roots==std::vector<std::string>({"old","new"}));
 assert(state.settings==std::vector<std::string>({"original-pack","new-pack"}));
 assert(state.scenario_present==std::vector<bool>({false,true}));
 assert(!state.errors&&client.publication_status().superseded==0);
}
''')

    def test_synchronous_source_path_returns_actual_completion_or_failure(self):
        run_cpp(fixture() + r'''
int main(){
 State state;state.fail=true;Client client(false,[&](char const*){++state.errors;},state);
 auto configure=std::async(std::launch::async,[&]{return client.definitions("sync","default",nullptr,"");});
 state.entered.get_future().wait();
 assert(configure.wait_for(20ms)==std::future_status::timeout&&!state.prepared);
 state.unblock();assert(configure.get()==C3X_RENDERER_RESULT_ERROR);
 assert(state.prepared==1&&state.retired==1&&state.errors==0);
 assert(client.publication_status().accepted==0);
}
''')


if __name__ == '__main__':
    unittest.main()
