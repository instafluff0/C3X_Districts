"""Execute the native owner's real request/poll/commit policy with a held worker."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp


class NativeCameraTransactionTests(unittest.TestCase):
    def test_exact_optional_guard_and_required_paths(self):
        s=Path('Renderer/native/c3x_renderer.cpp').read_text()
        begin=s.index('            if(!has_job && !stop_requested && !camera_pending && !camera_paused && scene_changes_ok &&\n               !(camera_gpu')
        end=s.index('{',begin)
        expression=s[begin:end].strip()[3:-1]
        # Extract, rather than reproduce, the actual complete optional gate.
        program='''#include <cassert>
#include <cstring>
#include <memory>
struct Texture {};
struct Topology {unsigned scope=7;unsigned scope_sequence()const{return scope;}};
struct State {bool world_preparation=true,cache_valid=true;Topology topology_cache;bool pixel_work_pending()const{return false;}} renderer_state;
bool has_job=false,stop_requested=false,camera_pending=false,camera_paused=false,scene_changes_ok=true;
bool camera_gpu=false,world_content_turn=true,ready_ahead=false,ready_units=false;
long long gpu_camera_front_ticket=1,camera_ticket=2;
int camera_result=1;
constexpr int C3X_RENDERER_RESULT_OK=1;
struct {struct {std::shared_ptr<Texture> texture;}resident;}camera_ready;
bool ahead_pending(){return ready_ahead;}bool unit_preparation_pending(){return ready_units;}
bool optional(){return '''+expression+''';}
'''
        # Required initialization/authority routing cannot invoke the new predicate.
        guard='!(camera_gpu && camera_result==C3X_RENDERER_RESULT_OK && gpu_camera_front_ticket!=camera_ticket && camera_ready.resident.texture)'
        self.assertEqual(s.count(guard),1)
        actual_required=s
        loading=actual_required.index('prepare_required_world(state',actual_required.index('bool prepare_required_world(')+1)
        self.assertNotIn(guard,actual_required[loading:loading+300])
        begin=s.index('auto initial_world=world_input.passes')
        initial=s[begin:s.index(';',begin)+1]
        begin=s.index('                bool valid=scene_changes_ok',s.index('command==Command::prepare_world_loading'))
        valid=s[begin:s.index(';',begin)+1].strip()
        program+='''
struct Identity {int value=7;}job_camera_identity,job_required_world_identity;
struct Authority {bool topology=true;Identity identity;}authority;
struct Scene {Authority* state(){return &authority;}}scene_changes;
struct WorldInput {unsigned passes=1,cursor=0;}world_input;
unsigned world_initialization_scope=7;bool authority_changed=false;
bool required_world_changes_for(Identity const&){return authority_changed;}
bool required_camera(){
'''+initial+'''return initial_world!=nullptr;}
bool required_loading(){auto state=scene_changes.state();
'''+valid+'''return valid;}
int main(){
 assert(optional());camera_gpu=true;assert(optional()); // no ready owner
 camera_ready.resident.texture=std::make_shared<Texture>();assert(!optional());
 assert(required_loading());assert(!required_camera());
 world_initialization_scope=0;assert(required_camera()); // first scope initialization
 world_initialization_scope=7;authority_changed=true;assert(required_camera()); // changed copied authority
 authority_changed=false;world_input.passes=0;assert(!required_loading()&&!required_camera());world_input.passes=1;
 camera_result=0;assert(optional());camera_result=1;
 gpu_camera_front_ticket=camera_ticket;assert(optional()); // adopted ready camera
 gpu_camera_front_ticket=1;camera_gpu=false;assert(optional()); // CPU compatibility
 camera_gpu=true;camera_ready.resident.texture.reset();assert(optional());
 has_job=true;assert(!optional());has_job=false;camera_pending=true;assert(!optional());camera_pending=false;
 camera_paused=true;assert(!optional());camera_paused=false;scene_changes_ok=false;assert(!optional());
 scene_changes_ok=true;renderer_state.cache_valid=false;assert(!optional());
 // Required paths remain directly eligible by their existing authority/loading
 // checks, including while the optional ready-camera predicate is false.
 renderer_state.cache_valid=true;camera_ready.resident.texture=std::make_shared<Texture>();assert(!optional());
}
'''
        run_cpp(program)

    def test_pending_supersession_lifetime_and_explicit_commit(self):
        source=Path('Renderer/native/native_composition_owner.h').read_text()
        methods=source[source.index('    void clear_camera_capture()'):source.index('    void set_tactical(')]
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
std::vector<std::string> debug_lines;
void OutputDebugStringA(char const* line){debug_lines.emplace_back(line);}
using Id=unsigned long long;
struct Rect {int left=0,top=0,right=0,bottom=0;};
int creates=0,imports=0,inserts=0,flushes=0,polls=0,cancels=0;
bool eligible_native=true,ready=false,fail_begin=false,fail_poll=false,fail_adapter=false;
long long next_ticket=0,worker_ticket=0;
namespace c3x_gpu_images {
struct WorkerClient {
 bool empty=true;
 WorkerClient(c3x_renderer_gpu_images_fn,c3x_renderer_gpu_frame_v1 const&,bool=false){++creates;}
 bool flushed()const{return empty;}
 void advance(c3x_renderer_gpu_frame_v1 const&){assert(empty);++imports;}
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
 c3x_renderer_frame_v1 camera_capture={};c3x_renderer_camera_identity_v1 camera_identity={};
 std::vector<c3x_renderer_tile_v1> camera_tiles;std::vector<c3x_renderer_u32> camera_topology;
 int route=0;void* route_image=nullptr;std::string route_text;struct{std::vector<int> points;}route_anchors;
 void* bits=nullptr;void* release=nullptr;void* pending=nullptr;
 Rect area;int phase_x=0,phase_y=0;
 std::unique_ptr<c3x_gpu_images::WorkerClient> client;
 std::unique_ptr<Adapter<c3x_gpu_images::WorkerClient>> adapter;
 c3x_renderer_gpu_frame_v1 frame={sizeof(frame)};
 bool scene_units=false,trace_success=false,tactical=true;void* front_native=nullptr;void* display_native=nullptr;unsigned surface_copy_reports=0,surface_fill_reports=0,cold_stroke_reports=0;
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
 // Routine transaction status is opt-in; failures remain diagnosable even
 // when the native hot path is otherwise quiet.
 Owner quiet(nullptr,nullptr,nullptr,nullptr,life,nullptr,nullptr);
 for(int code:{C3X_RENDERER_RESULT_OK,C3X_RENDERER_RESULT_PENDING,
     C3X_RENDERER_RESULT_SUPERSEDED,C3X_RENDERER_RESULT_BUSY})quiet.trace_map("probe",code,nullptr);
 assert(debug_lines.empty());
 quiet.trace_map("probe",C3X_RENDERER_RESULT_ERROR,nullptr);
 quiet.trace_map("probe",C3X_RENDERER_RESULT_BAD_ARGUMENT,nullptr);
 assert(debug_lines.size()==2);
 Owner traced(nullptr,nullptr,nullptr,nullptr,life,nullptr,nullptr,false,true);
 traced.trace_map("probe",C3X_RENDERER_RESULT_PENDING,nullptr);
 assert(debug_lines.size()==3);
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
 // The scroll timer's in-flight query never changes the view or the job.
 custom_renderer_native_view probe{},untouched_probe{};
 assert(owner.navigate(C3X_NAV_PENDING,image,probe,nullptr)==C3X_RENDERER_RESULT_OK);
 assert(owner.navigate(C3X_NAV_REQUEST,image,target,&request)==C3X_RENDERER_RESULT_PENDING);
 auto begun=next_ticket;
 for(int n=0;n<5;++n)assert(owner.navigate(C3X_NAV_PENDING,image,probe,nullptr)==C3X_RENDERER_RESULT_PENDING&&
     next_ticket==begun&&owner.navigation.active()&&!std::memcmp(&probe,&untouched_probe,sizeof(probe)));
 frame.presentation_time_ticks+=10;
 assert(owner.navigate(C3X_NAV_REQUEST,image,target,&request)==C3X_RENDERER_RESULT_PENDING && next_ticket==begun);
 // Different timer steps must not starve the pending frame. A deliberate
 // camera jump retains ordinary supersession (exercised below).
 for(int n=1;n<20;++n){auto moving=target;moving.camera_x+=n;
  assert(owner.navigate(C3X_NAV_REQUEST_SCROLL,image,moving,&request)==C3X_RENDERER_RESULT_PENDING&&next_ticket==begun);}
 ++request.identity.viewer_epoch;
 assert(owner.navigate(C3X_NAV_REQUEST_SCROLL,image,target,&request)==C3X_RENDERER_RESULT_PENDING&&next_ticket>begun);
 begun=next_ticket;target.camera_x=322;
 assert(owner.navigate(C3X_NAV_REQUEST,image,target,&request)==C3X_RENDERER_RESULT_PENDING&&next_ticket>begun);
 for(int n=0;n<20;++n)assert(owner.navigate(C3X_NAV_POLL,image,displayed,nullptr)==C3X_RENDERER_RESULT_PENDING&&displayed.camera_x==0);
 ready=true;assert(owner.navigate(C3X_NAV_POLL,image,displayed,nullptr)==C3X_RENDERER_RESULT_OK && displayed.camera_x==322);
 assert(owner.navigate(C3X_NAV_PENDING,image,probe,nullptr)==C3X_RENDERER_RESULT_OK&&owner.navigation.available());
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
 // Renderer64 admission starts and polls copied scene work; the exact render
 // pointer is absent, so any synchronous legacy call fails this fixture.
 Owner async(nullptr,nullptr,nullptr,nullptr,life,nullptr,nullptr);async.scene_units=true;
 async.set_camera(begin,poll,cancel);
 c3x_renderer_frame_v1 async_frame{};async_frame.struct_size=sizeof(async_frame);
 async_frame.target_width=640;async_frame.target_height=480;
 c3x_renderer_tile_v1 async_tile{};async_frame.tiles=&async_tile;async_frame.tile_count=1;
 c3x_renderer_camera_request_v1 async_request{};async_request.frame=&async_frame;
 ready=false;
 assert(async.map(C3X_NATIVE_MAP_PREPARE,image,&async_request,&output)==C3X_RENDERER_RESULT_PENDING);
 *reinterpret_cast<int*>(other+0x24)=16;*reinterpret_cast<int*>(other+0x38)=640;*reinterpret_cast<int*>(other+0x3c)=480;
 c3x_renderer_native_stroke stroke{3,3,20,3,1,0,0x80ffffffu};
 assert(async.defer_cold_stroke(other,&stroke)&&!async.adapter&&async.cold_stroke_reports==1);
 eligible_native=false;assert(!async.defer_cold_stroke(other,&stroke));eligible_native=true;
 *reinterpret_cast<int*>(other+0x38)=320;assert(!async.defer_cold_stroke(other,&stroke));
 *reinterpret_cast<int*>(other+0x38)=640;stroke.width=0;assert(!async.defer_cold_stroke(other,&stroke));stroke.width=1;
 auto first_async_ticket=async.camera_ticket;
 assert(async.map(C3X_NATIVE_MAP_PREPARE,image,&async_request,&output)==C3X_RENDERER_RESULT_PENDING &&
        async.camera_ticket==first_async_ticket && !async.pending);
 async_tile.anchor_x=8;
 assert(async.map(C3X_NATIVE_MAP_PREPARE,image,&async_request,&output)==C3X_RENDERER_RESULT_PENDING &&
        async.camera_ticket!=first_async_ticket && !async.pending);
 ready=true;
 assert(async.map(C3X_NATIVE_MAP_PREPARE,image,&async_request,&output)==C3X_RENDERER_RESULT_OK &&
        output.clip_right==640);
 assert(!async.defer_cold_stroke(other,&stroke));
 assert(async.map(C3X_NATIVE_MAP_COMMIT,image,nullptr,nullptr)==C3X_RENDERER_RESULT_OK);
 // A movement sight commit starts preparation without drawing or adopting.
 // The later native redraw must poll that exact request, never supersede it.
 ready=false;long long early=0;
 assert(async.request_camera(image,async_request,early)==C3X_RENDERER_RESULT_PENDING);
 assert(!async.pending&&early==async.camera_ticket);
 auto early_serial=next_ticket;
 async_frame.presentation_time_ticks+=100;
 assert(async.map(C3X_NATIVE_MAP_PREPARE,image,&async_request,&output)==C3X_RENDERER_RESULT_PENDING);
 assert(async.camera_ticket==early&&next_ticket==early_serial&&!async.pending);
 ready=true;
 assert(async.map(C3X_NATIVE_MAP_PREPARE,image,&async_request,&output)==C3X_RENDERER_RESULT_OK);
 assert(async.map(C3X_NATIVE_MAP_COMMIT,image,nullptr,nullptr)==C3X_RENDERER_RESULT_OK);
 // Real Civ III begins map redraws with unpresented native clear/HUD commands.
 // Publish that old-ticket batch without waiting before starting or polling
 // the next camera. Pending work is ordinary ordering, not bad arguments.
 ready=false;async.client->empty=false;auto previous_flushes=flushes;
 assert(async.map(C3X_NATIVE_MAP_PREPARE,image,&async_request,&output)==C3X_RENDERER_RESULT_PENDING);
 assert(async.client->empty&&flushes==previous_flushes+1);
 auto queued_ticket=async.camera_ticket;
 async.client->empty=false;
 assert(async.map(C3X_NATIVE_MAP_PREPARE,image,&async_request,&output)==C3X_RENDERER_RESULT_PENDING);
 assert(async.camera_ticket==queued_ticket&&async.client->empty&&flushes==previous_flushes+2);
 ready=true;async.client->empty=false;
 assert(async.map(C3X_NATIVE_MAP_PREPARE,image,&async_request,&output)==C3X_RENDERER_RESULT_OK);
 assert(async.client->empty&&flushes==previous_flushes+3);
 assert(async.map(C3X_NATIVE_MAP_COMMIT,image,nullptr,nullptr)==C3X_RENDERER_RESULT_OK);
}
''')

    def test_unload_and_shared_reset_barriers_do_not_fail_open(self):
        injected=Path('injected_code.c').read_text()
        unload='void unload_custom_renderer ()'+injected.split('void\nunload_custom_renderer ()',1)[1].split('\tis->custom_renderer_module = NULL;',1)[0]+'}'
        unload=unload.replace('BOOL (WINAPI * kill_timer) (HWND, UINT_PTR) = (void *)(*p_GetProcAddress) (is->user32, "KillTimer");', 'auto kill_timer=test_kill_timer;')
        unload=unload.replace('(void *)(*p_GetProcAddress)', '(int (*)(void))(*p_GetProcAddress)')
        renderer=Path('Renderer/native/c3x_renderer.cpp').read_text()
        drain='bool drain_native_composition(){'+renderer.split('bool drain_native_composition(){',1)[1].split('\n}\nint remote_draw_cpu_unit',1)[0]+'\n}'
        run_cpp(r'''
#include <cassert>
#include <memory>
#include <stdexcept>
#include <cstdlib>
#include "Renderer/native/gpu_frame_api.h"
constexpr int IS_INIT_FAILED=2;
constexpr int __=0;
int background_destroys=0;
struct PCX_Image;
void destroy_background(PCX_Image*,int,int){++background_destroys;}
struct PCX_VTable {void(*destruct)(PCX_Image*,int,int)=destroy_background;} background_vtable;
struct PCX_Image {PCX_VTable* vtable;};
bool fail=true;int drains=0,resets=0,frees=0,detaches=0,settled=-1,kills=0;
bool test_kill_timer(void*,unsigned id){assert(id==17);++kills;return true;}
int image(int,void*,void*,void const*,void const*,unsigned){++drains;return fail?-1:0;}
void reset(){++resets;}
int end_scene(){++resets;return C3X_RENDERER_RESULT_OK;}
void* get_proc(void*,char const*){return reinterpret_cast<void*>(end_scene);}auto p_GetProcAddress=get_proc;
void log_custom_renderer_event(char const*,int){}
void FreeLibrary(void*){++frees;}
void settle_custom_renderer_navigation(int action){settled=action;}
void set_custom_renderer_native_probe(void*){++detaches;}
struct State {
 struct {bool enable_custom_rendering=false;}current_config;
 unsigned custom_renderer_view_timer=17;
 PCX_Image* custom_renderer_combat_odds_background=nullptr;
 bool combat_odds_hud_rect_drawn=true;void* combat_odds_hud_background_canvas=this;
 int custom_renderer_init_state=1;
 int custom_renderer_zoom_tile_width=192,custom_renderer_zoom_target_width=192,custom_renderer_zoom_wheel_remainder=80;
 void* custom_renderer_hud_canvas=this;
 void* custom_renderer_module=this;
 c3x_renderer_native_image_fn custom_renderer_native_image=image;
 void (*custom_renderer_reset)()=reset;
} state;auto is=&state;
'''+unload+r'''
struct Composition {void drain(){++drains;if(fail)throw std::runtime_error("blocked");}void abandon(){++detaches;}};
struct Worker {int native_screen(void*){++resets;return fail?C3X_RENDERER_RESULT_ERROR:C3X_RENDERER_RESULT_OK;}};
Composition* native_composition=nullptr;Worker worker;Worker* renderer_worker=&worker;
struct Remote {bool healthy()const{return !fail;}void abandon(){++detaches;}int screen(void*){++resets;return fail?C3X_RENDERER_RESULT_ERROR:C3X_RENDERER_RESULT_OK;}};
std::unique_ptr<Remote> remote_renderer;
void OutputDebugStringA(char const*){}
namespace c3x_inputs {struct Assets{bool enabled=false;};Assets& replay_assets(){static Assets a;return a;}}
'''+drain+r'''
int main(){
 state.custom_renderer_combat_odds_background=(PCX_Image*)std::malloc(sizeof(PCX_Image));
 state.custom_renderer_combat_odds_background->vtable=&background_vtable;
 unload_custom_renderer();assert(drains==1&&!resets&&!frees&&!detaches&&settled==C3X_NAV_BARRIER&&state.custom_renderer_init_state==IS_INIT_FAILED);
 assert(background_destroys==1&&!state.custom_renderer_combat_odds_background&&
        !state.combat_odds_hud_rect_drawn&&!state.combat_odds_hud_background_canvas);
 assert(kills==1&&!state.custom_renderer_view_timer);
 fail=false;unload_custom_renderer();assert(kills==1);assert(drains==2&&resets==1&&frees==1&&detaches==1);
 assert(background_destroys==1);
 assert(state.custom_renderer_zoom_tile_width==0&&state.custom_renderer_zoom_target_width==128&&
        state.custom_renderer_zoom_wheel_remainder==0&&!state.custom_renderer_hud_canvas);
 state.current_config.enable_custom_rendering=true;unload_custom_renderer();assert(settled==C3X_NAV_DISCARD);
 fail=true;native_composition=new Composition;
 assert(!drain_native_composition()&&native_composition); // failed readback keeps owner alive
 fail=false;assert(drain_native_composition()&&!native_composition);
 fail=true;assert(!drain_native_composition()); // CPU-source-only display failure is not swallowed
 fail=false;assert(drain_native_composition());
 remote_renderer=std::make_unique<Remote>();native_composition=new Composition;
 int before=detaches;fail=true;
 assert(drain_native_composition()&&!native_composition&&!remote_renderer&&detaches==before+2);
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
