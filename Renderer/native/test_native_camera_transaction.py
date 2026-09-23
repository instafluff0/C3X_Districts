"""Execute the native owner's real request/poll/commit policy with a held worker."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp


class NativeCameraTransactionTests(unittest.TestCase):
    def test_pending_supersession_lifetime_and_explicit_commit(self):
        source=Path('Renderer/native/native_composition_owner.h').read_text()
        methods=source[source.index('    bool eligible('):source.index('    void set_tactical(')]
        methods+='\n'+source[source.index('    int map('):source.index('    int operation(')]
        run_cpp(r'''
#include <cassert>
#include <cstring>
#include <memory>
#include <stdexcept>
#include "Renderer/native/gpu_frame_api.h"
#include "Renderer/native/native_navigation.h"
using DWORD=unsigned;
DWORD caller_thread=1;
DWORD GetCurrentThreadId(){return caller_thread;}
using Id=unsigned long long;
struct Rect {int left=0,top=0,right=0,bottom=0;};
int creates=0,imports=0,inserts=0,flushes=0,polls=0,cancels=0;
bool eligible_native=true,ready=false,fail_begin=false,fail_poll=false,fail_adapter=false;
long long next_ticket=0,worker_ticket=0;
namespace c3x_gpu_images {
struct WorkerClient {
 bool empty=true;
 WorkerClient(c3x_renderer_gpu_images_fn,c3x_renderer_gpu_frame_v1 const&){++creates;}
 bool flushed()const{return empty;}
 void advance(c3x_renderer_gpu_frame_v1 const&){++imports;}
 void flush(){++flushes;empty=true;}
};
}
template<class Client> struct Adapter {
 Adapter(Client&,void*,void*,c3x_renderer_native_lifetime_fn){if(fail_adapter)throw std::bad_alloc();}
 bool admit(void*){return true;}
 bool insert_map(void*,Id,Rect,int,int,int,int){++inserts;return true;}
};
struct Owner {
 c3x_native_images::Navigation navigation;
 DWORD thread=1;
 c3x_renderer_gpu_render_fn render=nullptr;
 c3x_renderer_gpu_images_fn images=nullptr;
 c3x_renderer_gpu_present_fn present=nullptr;
 c3x_renderer_gpu_unit_fn unit=nullptr;
 c3x_renderer_native_lifetime_fn lifetime=nullptr;
 c3x_renderer_gpu_camera_begin_fn camera_begin=nullptr;
 c3x_renderer_gpu_camera_poll_view_fn camera_poll=nullptr;
 c3x_renderer_camera_cancel_fn camera_cancel=nullptr;
 c3x_renderer_i64 camera_ticket=0;void* camera_image=nullptr;int camera_width=0,camera_height=0;
 int route=0;void* route_image=nullptr;std::string route_text;
 void* bits=nullptr;void* release=nullptr;void* pending=nullptr;
 Rect area;int phase_x=0,phase_y=0;
 std::unique_ptr<c3x_gpu_images::WorkerClient> client;
 std::unique_ptr<Adapter<c3x_gpu_images::WorkerClient>> adapter;
 c3x_renderer_gpu_frame_v1 frame={sizeof(frame)};
 void check_thread(){}
 static int field(void* p,unsigned offset){return *reinterpret_cast<int*>(static_cast<char*>(p)+offset);}
''' + methods.replace('CompositionOwner(', 'Owner(') + r'''
};
int begin(c3x_renderer_camera_request_v1 const*,long long* ticket){if(fail_begin)throw std::bad_alloc();worker_ticket=*ticket=++next_ticket;return C3X_RENDERER_RESULT_PENDING;}
int poll(long long ticket,c3x_renderer_gpu_camera_view_v1* out){
 ++polls;if(fail_poll)throw std::bad_alloc();if(ticket!=worker_ticket)return C3X_RENDERER_RESULT_SUPERSEDED;
 if(!ready)return C3X_RENDERER_RESULT_PENDING;
 c3x_renderer_gpu_camera_view_v1 value={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(value)};
 value.image={sizeof(value)};value.image.ticket=ticket;value.image.session=1;value.image.map_image=123;
 value.camera.ticket=ticket;value.camera.frame.target_width=640;value.camera.frame.target_height=480;
 value.camera.output.clip_right=640;value.camera.output.clip_bottom=480;
 value.pixel_phase_x=19;value.pixel_phase_y=-7;*out=value;return C3X_RENDERER_RESULT_OK;
}
int cancel(long long){++cancels;return C3X_RENDERER_RESULT_OK;}
int life(int,void*,int){return eligible_native?1:0;}
int main(){
 Owner owner(nullptr,nullptr,nullptr,nullptr,life,nullptr,nullptr);owner.set_camera(begin,poll,cancel);
 alignas(int) char image[0x500]={},other[0x500]={};
 *reinterpret_cast<int*>(image+0x24)=16;*reinterpret_cast<int*>(image+0x38)=640;*reinterpret_cast<int*>(image+0x3c)=480;
 c3x_renderer_frame_v1 frame{};frame.target_width=640;frame.target_height=480;
 c3x_renderer_camera_request_v1 request{};request.frame=&frame;
 long long ticket=777;assert(owner.request_camera(image,request,ticket)==C3X_RENDERER_RESULT_PENDING && ticket==1);
 c3x_renderer_gpu_camera_view_v1 view{};view.image.ticket=333;auto unchanged=view;
 for(int n=0;n<100;++n){
  assert(owner.poll_camera(image,ticket,view)==C3X_RENDERER_RESULT_PENDING);
  assert(!std::memcmp(&view,&unchanged,sizeof(view)) && !creates && !imports && !inserts && !flushes);
  assert(owner.map(C3X_NATIVE_MAP_COMMIT,image,nullptr,nullptr)==C3X_RENDERER_RESULT_BAD_ARGUMENT);
 }
 assert(owner.poll_camera(other,ticket,view)==C3X_RENDERER_RESULT_SUPERSEDED && polls==100);
 long long newer=0;assert(owner.request_camera(image,request,newer)==C3X_RENDERER_RESULT_PENDING && newer>ticket);
 assert(owner.poll_camera(image,ticket,view)==C3X_RENDERER_RESULT_SUPERSEDED);
 eligible_native=false;assert(owner.poll_camera(image,newer,view)==C3X_RENDERER_RESULT_BAD_ARGUMENT && polls==100);
 eligible_native=true;*reinterpret_cast<int*>(image+0x38)=320;
 assert(owner.poll_camera(image,newer,view)==C3X_RENDERER_RESULT_BAD_ARGUMENT && polls==100);
 *reinterpret_cast<int*>(image+0x38)=640;ready=true;
 assert(owner.poll_camera(image,newer,view)==C3X_RENDERER_RESULT_OK && view.camera.ticket==newer && creates==1 && !inserts);
 assert(owner.phase_x==19 && owner.phase_y==-7);
 assert(owner.map(C3X_NATIVE_MAP_COMMIT,other,nullptr,nullptr)==C3X_RENDERER_RESULT_BAD_ARGUMENT);
 assert(owner.map(C3X_NATIVE_MAP_COMMIT,image,nullptr,nullptr)==C3X_RENDERER_RESULT_OK && inserts==1);
 owner.client->empty=false;long long untouched=999;
 assert(owner.request_camera(image,request,untouched)==C3X_RENDERER_RESULT_BAD_ARGUMENT && untouched==999 && flushes==1);
 owner.client->empty=true;ready=false;
 assert(owner.request_camera(image,request,ticket)==C3X_RENDERER_RESULT_PENDING);
 assert(owner.map(C3X_NATIVE_MAP_CANCEL,image,nullptr,nullptr)==C3X_RENDERER_RESULT_OK && cancels==1);
 assert(owner.poll_camera(image,ticket,view)==C3X_RENDERER_RESULT_SUPERSEDED && inserts==1);
 assert(owner.request_camera(image,request,ticket)==C3X_RENDERER_RESULT_PENDING);
 caller_thread=2;owner.retire_image(C3X_NATIVE_DESTROY,image);assert(owner.camera_ticket==ticket);
 caller_thread=1;owner.retire_image(C3X_NATIVE_INIT,image);
 ready=true;assert(owner.poll_camera(image,ticket,view)==C3X_RENDERER_RESULT_SUPERSEDED && cancels==2);
 assert(owner.request_camera(image,request,ticket)==C3X_RENDERER_RESULT_PENDING);
 assert(owner.poll_camera(image,ticket,view)==C3X_RENDERER_RESULT_OK);
 owner.retire_image(C3X_NATIVE_IMAGE_REINIT,image);
 assert(owner.map(C3X_NATIVE_MAP_COMMIT,image,nullptr,nullptr)==C3X_RENDERER_RESULT_BAD_ARGUMENT);

 // Navigation uses the same owner, pending rules and commit operation.
 custom_renderer_native_view displayed{};displayed.width=640;displayed.height=480;displayed.tile_width=128;displayed.native_width=128;
 auto target=displayed;target.camera_x=320;target.min_x=5;
 c3x_renderer_tile_v1 tile{};frame.tiles=&tile;frame.tile_count=1;
 unsigned topology[2]={1,2};frame.world_topology=topology;frame.world_topology_count=2;
 ready=false;
 assert(owner.navigate(C3X_NAV_REQUEST,image,target,&request)==C3X_RENDERER_RESULT_PENDING);
 auto begun=next_ticket;frame.presentation_time_ticks+=10;
 assert(owner.navigate(C3X_NAV_REQUEST,image,target,&request)==C3X_RENDERER_RESULT_PENDING && next_ticket==begun);
 for(int n=0;n<20;++n)assert(owner.navigate(C3X_NAV_POLL,image,displayed,nullptr)==C3X_RENDERER_RESULT_PENDING&&displayed.camera_x==0);
 ready=true;assert(owner.navigate(C3X_NAV_POLL,image,displayed,nullptr)==C3X_RENDERER_RESULT_OK && displayed.camera_x==320);
 assert(owner.map(C3X_NATIVE_MAP_COMMIT,image,nullptr,nullptr)==C3X_RENDERER_RESULT_BAD_ARGUMENT); // Fresh capture is mandatory.
 c3x_renderer_output_v1 output{};
 assert(owner.map(C3X_NATIVE_MAP_PREPARE,image,&request,&output)==C3X_RENDERER_RESULT_OK&&output.clip_right==640);
 assert(owner.map(C3X_NATIVE_MAP_COMMIT,image,nullptr,nullptr)==C3X_RENDERER_RESULT_OK);
 // Lifecycle changes after polling still prohibit commit and prepared reuse.
 assert(owner.navigate(C3X_NAV_REQUEST,image,target,&request)==C3X_RENDERER_RESULT_PENDING);
 assert(owner.navigate(C3X_NAV_POLL,image,displayed,nullptr)==C3X_RENDERER_RESULT_OK);
 owner.route=7;owner.route_image=image;owner.route_text="old";
 owner.retire_image(C3X_NATIVE_DESTROY,image);assert(!owner.navigation.available());
 assert(!owner.route&&!owner.route_image&&owner.route_text.empty());
 assert(owner.map(C3X_NATIVE_MAP_COMMIT,image,nullptr,nullptr)==C3X_RENDERER_RESULT_BAD_ARGUMENT);
 // Fresh capture checks include ordered anchors, visibility and topology.
 int exact_calls=0;owner.render=[](c3x_renderer_camera_request_v1 const*,c3x_renderer_gpu_frame_v1*,c3x_renderer_output_v1*)->int{return C3X_RENDERER_RESULT_ERROR;};
 for(int change=0;change<4;++change){
  assert(owner.navigate(C3X_NAV_REQUEST,image,target,&request)==C3X_RENDERER_RESULT_PENDING);
  assert(owner.navigate(C3X_NAV_POLL,image,displayed,nullptr)==C3X_RENDERER_RESULT_OK);
  if(change==0)++tile.anchor_y;if(change==1)++tile.visibility_mask;
  if(change==2)++topology[1];if(change==3)++request.identity.viewer_epoch;
  assert(owner.map(C3X_NATIVE_MAP_PREPARE,image,&request,&output)==C3X_RENDERER_RESULT_ERROR);
  assert(!owner.navigation.available()&&!owner.pending);++exact_calls;
 }
 assert(exact_calls==4);
 // A native action barrier keeps the destination, but grants no ready pixels.
 ready=false;target.camera_x=512;
 assert(owner.navigate(C3X_NAV_REQUEST,image,target,&request)==C3X_RENDERER_RESULT_PENDING);
 assert(owner.navigate(C3X_NAV_BARRIER,image,displayed,nullptr)==C3X_RENDERER_RESULT_OK&&displayed.camera_x==512);
 assert(!owner.navigation.active()&&!owner.pending);
 assert(owner.navigate(C3X_NAV_REQUEST,image,target,&request)==C3X_RENDERER_RESULT_PENDING);
 displayed.tile_width=160;
 assert(owner.navigate(C3X_NAV_POLL,image,displayed,nullptr)==C3X_RENDERER_RESULT_SUPERSEDED&&!owner.navigation.active());
 displayed.tile_width=128;
 // Lost lifetime (including cross-thread invalidation) also rejects barriers.
 assert(owner.navigate(C3X_NAV_REQUEST,image,target,&request)==C3X_RENDERER_RESULT_PENDING);
 eligible_native=false;displayed.camera_x=100;
 assert(owner.navigate(C3X_NAV_BARRIER,image,displayed,nullptr)==C3X_RENDERER_RESULT_SUPERSEDED&&displayed.camera_x==100);
 eligible_native=true;
 // Exceptions retire unpublished input and recover the intended camera exactly.
 assert(owner.navigate(C3X_NAV_REQUEST,image,target,&request)==C3X_RENDERER_RESULT_PENDING);
 fail_begin=true;
 try{owner.navigate(C3X_NAV_REQUEST,image,displayed,&request);assert(false);}catch(std::bad_alloc const&){}
 assert(!owner.navigation.active()&&!owner.camera_ticket&&!owner.pending);fail_begin=false;
 assert(owner.navigate(C3X_NAV_REQUEST,image,target,&request)==C3X_RENDERER_RESULT_PENDING);
 fail_poll=true;
 assert(owner.navigate(C3X_NAV_POLL,image,displayed,nullptr)==C3X_RENDERER_RESULT_OK&&displayed.camera_x==512);
 assert(!owner.navigation.active()&&!owner.pending);fail_poll=false;
 assert(owner.map(C3X_NATIVE_MAP_COMMIT,image,nullptr,nullptr)==C3X_RENDERER_RESULT_BAD_ARGUMENT);
 // A partially constructed client must not survive failed adapter allocation.
 Owner fresh(nullptr,nullptr,nullptr,nullptr,life,nullptr,nullptr);fresh.set_camera(begin,poll,cancel);
 ready=true;fail_adapter=true;
 assert(fresh.navigate(C3X_NAV_REQUEST,image,target,&request)==C3X_RENDERER_RESULT_PENDING);
 assert(fresh.navigate(C3X_NAV_POLL,image,displayed,nullptr)==C3X_RENDERER_RESULT_OK);
 assert(!fresh.client&&!fresh.adapter&&!fresh.pending&&!fresh.navigation.active());fail_adapter=false;
 assert(fresh.navigate(C3X_NAV_REQUEST,image,target,&request)==C3X_RENDERER_RESULT_PENDING);
 assert(fresh.navigate(C3X_NAV_POLL,image,displayed,nullptr)==C3X_RENDERER_RESULT_OK);
 assert(fresh.map(C3X_NATIVE_MAP_PREPARE,image,&request,&output)==C3X_RENDERER_RESULT_OK);
 assert(fresh.map(C3X_NATIVE_MAP_COMMIT,image,nullptr,nullptr)==C3X_RENDERER_RESULT_OK);
}
''')

    def test_unload_and_shared_reset_barriers_do_not_fail_open(self):
        injected=Path('injected_code.c').read_text()
        unload='void unload_custom_renderer ()'+injected.split('void\nunload_custom_renderer ()',1)[1].split('\tis->custom_renderer_module = NULL;',1)[0]+'}'
        renderer=Path('Renderer/native/c3x_renderer.cpp').read_text()
        drain='bool drain_native_composition(){'+renderer.split('bool drain_native_composition(){',1)[1].split('\n}\nint remote_draw_cpu_unit',1)[0]+'\n}'
        run_cpp(r'''
#include <cassert>
#include <stdexcept>
#include "Renderer/native/gpu_frame_api.h"
constexpr int IS_INIT_FAILED=2;
bool fail=true;int drains=0,resets=0,frees=0,detaches=0,settled=-1;
int image(int,void*,void*,void const*,void const*,unsigned){++drains;return fail?-1:0;}
void reset(){++resets;}
void FreeLibrary(void*){++frees;}
void settle_custom_renderer_navigation(int action){settled=action;}
void set_custom_renderer_native_probe(void*){++detaches;}
struct State {
 struct {bool enable_custom_rendering=false;}current_config;
 int custom_renderer_init_state=1;
 void* custom_renderer_module=this;
 c3x_renderer_native_image_fn custom_renderer_native_image=image;
 void (*custom_renderer_reset)()=reset;
} state;auto is=&state;
'''+unload+r'''
struct Composition {void drain(){++drains;if(fail)throw std::runtime_error("blocked");}};
struct Worker {int native_screen(void*){++resets;return fail?C3X_RENDERER_RESULT_ERROR:C3X_RENDERER_RESULT_OK;}};
Composition* native_composition=nullptr;Worker worker;Worker* renderer_worker=&worker;
struct Remote {int screen(void*){++resets;return fail?C3X_RENDERER_RESULT_ERROR:C3X_RENDERER_RESULT_OK;}};
Remote* remote_renderer=nullptr;
void OutputDebugStringA(char const*){}
namespace c3x_inputs {struct Assets{bool enabled=false;};Assets& replay_assets(){static Assets a;return a;}}
'''+drain+r'''
int main(){
 unload_custom_renderer();assert(drains==1&&!resets&&!frees&&!detaches&&settled==C3X_NAV_BARRIER&&state.custom_renderer_init_state==IS_INIT_FAILED);
 fail=false;unload_custom_renderer();assert(drains==2&&resets==1&&frees==1&&detaches==1);
 state.current_config.enable_custom_rendering=true;unload_custom_renderer();assert(settled==C3X_NAV_DISCARD);
 fail=true;native_composition=new Composition;
 assert(!drain_native_composition()&&native_composition); // failed readback keeps owner alive
 fail=false;assert(drain_native_composition()&&!native_composition);
 fail=true;assert(!drain_native_composition()); // CPU-source-only display failure is not swallowed
 fail=false;assert(drain_native_composition());
}
''')

    def test_navigation_copy_failure_retires_worker_ticket(self):
        run_cpp(r'''
#include <cassert>
#include <cstdlib>
#include <new>
#include "Renderer/native/native_navigation.h"
bool fail_allocation=false;
void* operator new(std::size_t n){if(fail_allocation){fail_allocation=false;throw std::bad_alloc();}auto p=std::malloc(n);if(!p)throw std::bad_alloc();return p;}
void operator delete(void* p) noexcept{std::free(p);}
struct Owner {
 c3x_native_images::Navigation navigation;int cancellations=0;
 int request_camera(void*,c3x_renderer_camera_request_v1 const&,long long& ticket){ticket=1;fail_allocation=true;return C3X_RENDERER_RESULT_PENDING;}
 int map(int action,void*,void*,void*){assert(action==C3X_NATIVE_MAP_CANCEL);navigation.clear();++cancellations;return C3X_RENDERER_RESULT_OK;}
};
int main(){
 Owner owner;c3x_renderer_tile_v1 tile{};c3x_renderer_frame_v1 frame{};frame.tiles=&tile;frame.tile_count=1;
 c3x_renderer_camera_request_v1 request{};request.frame=&frame;custom_renderer_native_view view{};
 try{owner.navigation.request(owner,&owner,view,request);assert(false);}catch(std::bad_alloc const&){}
 assert(!owner.navigation.active()&&!owner.navigation.available()&&owner.cancellations==1);
}
''')
