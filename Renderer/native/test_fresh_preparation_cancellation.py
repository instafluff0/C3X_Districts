"""Execute actual FRESH preparation and worker publication cancellation gates.

D3D calls are recorded by host stubs; this checks control flow and temporary
resource release, not native GPU execution or map-jump timing.
"""
import hashlib
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def block_at(source, start):
    """Find a braced C++ block, ignoring comments and quoted literals."""
    opening = source.index('{', start)
    depth = 0
    state = 'code'
    i = opening
    while i < len(source):
        c, n = source[i], source[i:i + 2]
        if state == 'line':
            if c == '\n':
                state = 'code'
        elif state == 'comment':
            if n == '*/':
                state = 'code'
                i += 1
        elif state in ('"', "'"):
            if c == '\\':
                i += 1
            elif c == state:
                state = 'code'
        elif n == '//':
            state = 'line'
            i += 1
        elif n == '/*':
            state = 'comment'
            i += 1
        elif c in ('"', "'"):
            state = c
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
        i += 1
    raise ValueError('Unterminated production block')


def extract(source):
    anchor = source.index('// The authoritative scene has been prepared.')
    begin = source.rfind('if (fresh_scene_path) {', 0, anchor)
    if begin < 0:
        raise ValueError('Production fresh preparation branch missing')
    branch = block_at(source, source.index("if(fresh_scene_path && fresh_path_failed){")) + "\n" + block_at(source, begin)
    capture_begin = source.index('bool complete=renderer_state.render(job_frame,output,-1,&camera_cancelled')
    render_end_marker = '[this]{service_camera_preparation();});'
    render_end = source.index(render_end_marker, capture_begin) + len(render_end_marker)
    marker='capture_gpu(ready,output,job_frame,job_camera_identity);'
    capture_end = source.index(marker,capture_begin)+len(marker)
    gate_begin = source.rfind('complete=complete && !camera_cancelled.load', render_end, capture_end)
    if gate_begin < 0:
        raise ValueError('Production completed GPU capture gate missing')
    # Unit-source/loading-region owners have separate contracts. Execute the
    # actual render call and final capture gate without importing those owners
    # into this focused temporary-target/publication fixture.
    capture = source[capture_begin:render_end] + '\n' + source[gate_begin:capture_end]
    guard_begin = source.index('if(gpu_ticket==camera_ticket && camera_result==C3X_RENDERER_RESULT_PENDING && !camera_cancelled.load', capture_end)
    guard = block_at(source, guard_begin)
    return branch, capture, guard


PRELUDE = r'''
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>
using UINT=unsigned;
constexpr int DXGI_FORMAT_B8G8R8A8_UNORM=87;
constexpr int D3D11_BIND_RENDER_TARGET=1, D3D11_BIND_SHADER_RESOURCE=2;
constexpr int C3X_RENDERER_API_VERSION=1;
constexpr int C3X_RENDERER_RESULT_OK=0, C3X_RENDERER_RESULT_ERROR=-1,
    C3X_RENDERER_RESULT_PENDING=1;
bool FAILED(int value){return value<0;}
struct LARGE_INTEGER {long long QuadPart=0;};
void Sleep(unsigned){}
void QueryPerformanceCounter(LARGE_INTEGER* value){static long long now=1;value->QuadPart=now++;}
struct D3D11_TEXTURE2D_DESC {
    UINT Width=0,Height=0,MipLevels=0,ArraySize=0;
    struct {UINT Count=0;} SampleDesc;
    int Format=0,BindFlags=0;
};
struct Texture {
    D3D11_TEXTURE2D_DESC description;
    static int alive;
    explicit Texture(D3D11_TEXTURE2D_DESC const& desc):description(desc){++alive;}
    void GetDesc(D3D11_TEXTURE2D_DESC* out){*out=description;}
    void Release(){--alive;delete this;}
};
int Texture::alive=0;
struct ID3D11RenderTargetView {
    static int alive,releases;
    ID3D11RenderTargetView(){++alive;}
    void Release(){--alive;++releases;delete this;}
};
int ID3D11RenderTargetView::alive=0;
int ID3D11RenderTargetView::releases=0;
template<class T>void release(T*& value){if(value){value->Release();value=nullptr;}}
struct Frame {unsigned visible_animation_count=0,tile_count=0;int const* tiles=nullptr;};
struct Output {unsigned version=0,size=0;bool filled=false;};
struct Trace {int writes=0;void write(char const*,char const*,bool){++writes;}};
enum class CancelAt {none,before,wave,texture,rtv,selection,assets,meshes,draw,capture};
struct Harness;
struct Device {
    Harness* owner;
    int texture_calls=0,rtv_calls=0,rtv_created=0;
    int CreateTexture2D(D3D11_TEXTURE2D_DESC const*,void*,Texture**);
    int CreateRenderTargetView(Texture*,void*,ID3D11RenderTargetView**);
};
struct Coverage {int calls=0;bool capture(Frame const&){++calls;return true;}};
struct Published {
    int identity=0;
    std::shared_ptr<int> lease;
    bool fail_swap=false;
    void swap(Published& other){
        if(fail_swap)throw std::runtime_error("publication refused");
        std::swap(identity,other.identity);lease.swap(other.lease);
    }
};
struct Harness {
    std::atomic<bool> camera_cancelled{false};
    CancelAt cancel_at=CancelAt::none;
    bool wave_ok=true,draw_ok=true,texture_ok=true,rtv_ok=true;
    bool selection_ok=true,capture_ok=true;
    int assets_result=C3X_RENDERER_RESULT_OK,meshes_result=C3X_RENDERER_RESULT_OK;
    bool fresh_scene_path=true,visibility_pass=true,fresh_path_failed=false;
    bool gpu_map_valid=false,cpu_output_stale=false;
    bool cached_request_continuous_redraw=false,cache_valid=false;
    unsigned cached_signature=0,previous_signature=0,cached_rendered_tile_count=0,
        cached_fallback_tile_count=0,cached_textured_tile_count=0,cached_visible_animation_count=0;
    std::vector<int> cached_tiles,cached_replacement_tile_flags;
    struct Records {std::vector<int> layers[1];
        auto& operator[](unsigned layer){return layers[layer];}
        auto& edit(unsigned layer){return layers[layer];}
    } geometry_vertex_buffers;
    static constexpr int geometry_wave=0;
    Coverage visibility_coverage;
    Trace trace;
    Device owned_device{this};Device* device=&owned_device;
    Texture* gpu_map_texture=nullptr;
    long long frame_geometry_ticks=0,frame_draw_ticks=0;
    char const* frame_cache_path="prior";
    int waves=0,draws=0,fills=0,captures=0,selections=0,assets=0,meshes=0,consumes=0;
    std::uint64_t route_map_serial=0,gpu_serial=19;
    std::vector<int> fresh_unit_poses;
    bool select_frame_units(Frame const&,std::vector<int> const&,std::vector<int>&,float){
        ++selections;event(CancelAt::selection);return selection_ok;
    }
    int prepare_frame_unit_assets(std::vector<int> const&){++assets;event(CancelAt::assets);return assets_result;}
    int c3x_renderer64_prepare_unit_meshes(){++meshes;event(CancelAt::meshes);return meshes_result;}
    void consume_required_world_changes(unsigned,bool success){assert(success);++consumes;}
    // Resident data and the last completed map keep independent owners.
    std::shared_ptr<int> resident_lease=std::make_shared<int>(73);
    Published gpu_publication{41,std::make_shared<int>(41)};
    Published camera_ready{51,std::make_shared<int>(51)};
    unsigned gpu_ticket=2,camera_ticket=2;
    int camera_result=C3X_RENDERER_RESULT_PENDING;
    bool camera_ready_prepared=true;
    ~Harness(){release(gpu_map_texture);}
    void event(CancelAt stage){if(cancel_at==stage)camera_cancelled.store(true);}
    bool prepare_wave_chunks(Frame const&){++waves;event(CancelAt::wave);return wave_ok;}
    bool c3x_renderer64_render_fresh(Frame const&,ID3D11RenderTargetView*){++draws;event(CancelAt::draw);return draw_ok;}
    bool fill_output(Frame const&,Output& output,unsigned,long long){++fills;output.filled=true;return true;}
    bool capture_gpu(Published& ready,Output const&,Frame const&,unsigned){
        ++captures;event(CancelAt::capture);
        if(!capture_ok)return false;
        ready={61,std::make_shared<int>(61)};return true;
    }
    void service_camera_preparation(){}
    bool render(Frame const& frame,Output& output,int,std::atomic<bool>* pending,unsigned=0,void const* =nullptr,unsigned=0,void const* =nullptr,std::function<void()> ={}){
        auto cancelled=[&]{return pending&&pending->load(std::memory_order_relaxed);};
        unsigned signature=97,textured_tile_count=2,fallback_tile_count=0,invalidations=0;
        int width=128,height=64;
        std::vector<int> replacement_tile_flags{1,1};
        LARGE_INTEGER started={};QueryPerformanceCounter(&started);
'''

MIDDLE = r'''
        return false;
    }
    bool worker(){
        Frame job_frame={};unsigned job_camera_identity=3;
        auto& renderer_state=*this;
        Output output={C3X_RENDERER_API_VERSION,sizeof(output)};
        Published ready;
'''

END = r'''
        return complete;
    }
};
int Device::CreateTexture2D(D3D11_TEXTURE2D_DESC const* desc,void*,Texture** output){
    ++texture_calls;owner->event(CancelAt::texture);
    if(!owner->texture_ok)return -1;
    *output=new Texture(*desc);return 0;
}
int Device::CreateRenderTargetView(Texture*,void*,ID3D11RenderTargetView** output){
    ++rtv_calls;owner->event(CancelAt::rtv);
    if(!owner->rtv_ok)return -1;
    *output=new ID3D11RenderTargetView();++rtv_created;return 0;
}
void cancelled_case(CancelAt stage,bool wave_ok=true,bool draw_ok=true,bool texture_ok=true,bool rtv_ok=true){
    assert(ID3D11RenderTargetView::alive==0);
    Harness h;h.cancel_at=stage;h.wave_ok=wave_ok;h.draw_ok=draw_ok;h.texture_ok=texture_ok;h.rtv_ok=rtv_ok;
    if(stage==CancelAt::before)h.camera_cancelled.store(true);
    auto resident=h.resident_lease,prior=h.gpu_publication.lease,ready=h.camera_ready.lease;
    int releases=ID3D11RenderTargetView::releases;
    bool complete=h.worker();
    assert(!complete&&"cancelled preparation must return incomplete");
    assert(h.camera_cancelled.load());
    assert(h.fills==0&&!h.gpu_map_valid&&!h.cpu_output_stale);
    assert(!h.fresh_path_failed&&"cancel is not a sticky renderer failure");
    assert(h.captures==0&&"actual worker capture expression must suppress cancelled publication");
    assert(h.camera_ready.identity==51&&h.camera_ready.lease==ready);
    assert(h.gpu_publication.identity==41&&h.gpu_publication.lease==prior);
    assert(h.resident_lease==resident&&*resident==73);
    assert(h.camera_result==C3X_RENDERER_RESULT_PENDING);
    assert(h.consumes==0);
    assert(ID3D11RenderTargetView::alive==0);
    assert(ID3D11RenderTargetView::releases-releases==h.owned_device.rtv_created);
    if(stage==CancelAt::before){
        assert(h.waves==0&&h.draws==0&&h.visibility_coverage.calls==0);
        assert(h.owned_device.texture_calls==0&&h.owned_device.rtv_calls==0);
        assert(!h.cache_valid&&h.cached_signature==0);
    }else if(stage==CancelAt::wave){
        assert(h.waves==1&&h.draws==0);
        assert(h.owned_device.texture_calls==0&&h.owned_device.rtv_calls==0);
    }else if(stage==CancelAt::texture||stage==CancelAt::rtv){
        assert(h.waves==1&&h.draws==0);
    }else if(stage==CancelAt::draw){assert(h.waves==1&&h.draws==1);}
}
int main(){
    cancelled_case(CancelAt::before);
    cancelled_case(CancelAt::wave,true);
    cancelled_case(CancelAt::wave,false);
    cancelled_case(CancelAt::texture);
    cancelled_case(CancelAt::rtv);
    cancelled_case(CancelAt::texture,true,true,false);
    cancelled_case(CancelAt::rtv,true,true,true,false);
    cancelled_case(CancelAt::selection);
    cancelled_case(CancelAt::assets);
    cancelled_case(CancelAt::meshes);
    cancelled_case(CancelAt::draw,true,true);
    cancelled_case(CancelAt::draw,true,false);
    {
        Harness h;assert(h.worker());
        assert(h.waves==1&&h.draws==1&&h.fills==1&&h.captures==1);
        assert(h.gpu_map_valid&&h.cpu_output_stale&&!h.fresh_path_failed);
        assert(h.camera_ready.identity==61&&h.camera_result==C3X_RENDERER_RESULT_OK);
        assert(!h.camera_ready_prepared&&ID3D11RenderTargetView::alive==0);
        assert(h.selections==1&&h.assets==1&&h.meshes==1&&h.consumes==1);
        assert(h.route_map_serial==h.gpu_serial+1);
    }
    {
        Harness h;h.wave_ok=false;assert(!h.worker());
        assert(h.fresh_path_failed&&h.waves==1&&h.draws==0&&h.fills==0);
    }
    {
        Harness h;h.draw_ok=false;assert(!h.worker());
        assert(h.fresh_path_failed&&h.waves==1&&h.draws==1&&h.fills==0);
        assert(ID3D11RenderTargetView::alive==0);
    }
    {
        Harness h;h.texture_ok=false;assert(!h.worker());
        assert(h.fresh_path_failed&&h.draws==0&&h.fills==0);
    }
    {
        Harness h;h.rtv_ok=false;assert(!h.worker());
        assert(h.fresh_path_failed&&h.draws==0&&h.fills==0);
    }
    // A failed request leaves the old publication intact and a subsequent
    // request can recover through the same fresh path without restarting.
    for(int failure=0;failure<4;++failure){
        Harness h;auto prior=h.camera_ready.lease;
        if(failure==0)h.wave_ok=false;
        if(failure==1)h.texture_ok=false;
        if(failure==2)h.rtv_ok=false;
        if(failure==3)h.draw_ok=false;
        assert(!h.worker()&&h.fresh_path_failed&&h.camera_ready.lease==prior);
        h.wave_ok=h.texture_ok=h.rtv_ok=h.draw_ok=true;
        ++h.gpu_ticket;h.camera_ticket=h.gpu_ticket;h.camera_result=C3X_RENDERER_RESULT_PENDING;
        assert(h.worker()&&!h.fresh_path_failed&&h.camera_ready.identity==61);
        assert(h.camera_result==C3X_RENDERER_RESULT_OK&&h.consumes==1);
    }
    // Real downstream refusal cannot publish or consume the required-world
    // marker, even though the earlier draw itself completed successfully.
    for(int failure=0;failure<4;++failure){
        Harness h;auto prior=h.camera_ready.lease;
        if(failure==0)h.selection_ok=false;
        if(failure==1)h.assets_result=C3X_RENDERER_RESULT_ERROR;
        if(failure==2)h.meshes_result=C3X_RENDERER_RESULT_ERROR;
        if(failure==3)h.capture_ok=false;
        assert(!h.worker());assert(h.camera_ready.identity==51&&h.camera_ready.lease==prior);
        assert(h.camera_result==C3X_RENDERER_RESULT_ERROR&&h.consumes==0);
        assert(ID3D11RenderTargetView::alive==0);
    }
    {
        Harness h;h.camera_ready.fail_swap=true;auto prior=h.camera_ready.lease;
        assert(h.worker()); // GPU capture completed; the publication swap failed.
        assert(h.camera_result==C3X_RENDERER_RESULT_ERROR&&h.consumes==0);
        assert(h.camera_ready.identity==51&&h.camera_ready.lease==prior);
    }
    for(int rejected_owner=0;rejected_owner<3;++rejected_owner){
        Harness h;auto prior=h.camera_ready.lease;
        if(rejected_owner==0)h.camera_ticket=h.gpu_ticket+1;
        if(rejected_owner==1)h.cancel_at=CancelAt::capture;
        if(rejected_owner==2)h.camera_result=C3X_RENDERER_RESULT_OK;
        auto prior_result=h.camera_result;
        assert(h.worker()); // Final owner gate must still reject adoption.
        assert(h.captures==1&&h.camera_result==prior_result&&h.consumes==0);
        assert(h.camera_ready.identity==51&&h.camera_ready.lease==prior);
    }
    assert(Texture::alive==0&&ID3D11RenderTargetView::alive==0);
    std::puts("fresh preparation cancellation: PASS (12 cancelled stages, success, 9 failures, 3 owner rejections)");
}
'''


class FreshPreparationCancellationTests(unittest.TestCase):
    def test_supersession_skips_work_and_preserves_publication_and_real_failure(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        branch, capture, guard = extract(source)
        code = PRELUDE + branch + MIDDLE + capture + '\nint result=complete?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_ERROR;\n' + guard + END
        print('production_branch_sha256=' + hashlib.sha256(branch.encode()).hexdigest())
        print('production_capture_sha256=' + hashlib.sha256(capture.encode()).hexdigest())
        print('production_publication_sha256=' + hashlib.sha256(guard.encode()).hexdigest())
        run_cpp(code)


if __name__ == '__main__':
    unittest.main()
