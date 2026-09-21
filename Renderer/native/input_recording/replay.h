#pragma once
#include "runtime.h"
#include "witness.h"
#include "canvas.h"
#include "pixel_delta.h"
#include <map>
namespace c3x_inputs {
// Replay is an explicit tool entry, never selected by installed rendering. All
// scene/unit/tactical work below re-enters the existing production owners.
inline int replay_world_callback(c3x_renderer_world_page_v1*){return C3X_RENDERER_RESULT_PENDING;}
struct ReplayState {
    std::map<std::int64_t,std::int64_t> images,tickets,cameras;std::map<std::int64_t,int> formats;
    c3x_renderer_gpu_present_v1 displayed={};c3x_native_images::Lifetimes lifetimes;CanvasReplay canvases;
    ScreenReplay native_pixels;NativeValueStream native_values;bool allow_unknown=false;
    std::int64_t id(std::map<std::int64_t,std::int64_t> const& values,std::int64_t key){
        if(!key)return 0;auto it=values.find(key);if(it==values.end()){require(allow_unknown,"replay missing lifecycle identity");return INT64_MAX;}return it->second;
    }
    void target(c3x_renderer_gpu_unit_v1& value){value.struct_size=sizeof(value);value.ticket=id(tickets,value.ticket);
        value.destination=id(images,value.destination);value.background=id(images,value.background);value.detail=id(images,value.detail);value.background_detail=id(images,value.background_detail);}
    void map_result(Reader& expected,c3x_renderer_gpu_frame_v1 const& actual,c3x_renderer_output_v1 const& metadata,int code){
        auto ticket=std::int64_t(expected.u64()),image=std::int64_t(expected.u64());expected.u64();
        check_gpu(expected,actual,metadata,code);
        if(code==C3X_RENDERER_RESULT_OK){require(ticket&&image&&actual.ticket&&actual.map_image,"replay missing map identity");tickets[ticket]=actual.ticket;images[image]=actual.map_image;formats[actual.map_image]=C3X_GPU_BGRA32;}
    }
    int call(Kind kind,unsigned subtype,Reader& in,Reader& expected,HWND window){
        auto token=in.u64();in.u64();require(expected.u64()==token,"replay call/result mismatch");int wanted=0;expected(wanted);allow_unknown=wanted!=C3X_RENDERER_RESULT_OK&&wanted!=C3X_RENDERER_RESULT_PENDING;
        int actual=C3X_RENDERER_RESULT_ERROR;bool performance=replay_execution().performance;
        if(kind==Kind::configuration){
            if(subtype==1){bool present;auto path=in.string(32768,&present);actual=c3x_renderer_set_pack_path(present?path.c_str():nullptr);}
            else if(subtype==2){bool present[4]={};std::string paths[4];for(unsigned n=0;n<4;++n)paths[n]=in.string(32768,&present[n]);
                actual=c3x_renderer_set_definition_paths(present[0]?paths[0].c_str():nullptr,present[1]?paths[1].c_str():nullptr,present[2]?paths[2].c_str():nullptr,present[3]?paths[3].c_str():nullptr);}
            else if(subtype==3){int enabled=0;in(enabled);actual=c3x_renderer_set_unit_rendering(enabled);}
            else if(subtype==4){auto enabled=in.u32();require(enabled<=1,"invalid world capture lifecycle");actual=c3x_renderer_set_world_capture(enabled?replay_world_callback:nullptr);}
            else throw std::runtime_error("unknown configuration input");
            if(subtype==1||subtype==2){images.clear();tickets.clear();cameras.clear();}
        }else if(kind==Kind::scene){
            c3x_renderer_camera_identity_v1 identity={};if(subtype!=1)c3x_renderer_camera_identity_v1_fields(in,identity);
            Frame owned;frame(in,owned);c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&owned.value,identity};
            c3x_renderer_output_v1 output={C3X_RENDERER_API_VERSION,sizeof(output)};
            if(subtype==3){c3x_renderer_gpu_frame_v1 view={sizeof(view)};actual=c3x_renderer_gpu_render(&request,&view,&output);map_result(expected,view,output,actual);}
            else if(subtype==1)actual=c3x_renderer_render(&owned.value,&output);
            else if(subtype==2)actual=c3x_renderer_render_view(&request,&output);
            else throw std::runtime_error("unknown scene input");
            if(subtype==1||subtype==2)check_output(expected,output,actual);
        }else if(kind==Kind::native_bridge){
            NativeValuesProvider native;native.replay=true;native.target=window;native.values=native_values.decode(expected);
            // Reconstruct external GDI/input storage outside production timing.
            for(auto const& entry:native.values){auto type=std::get<0>(entry.first);auto object=NativeOperationInput::object(std::get<1>(entry.first));if(type==NativeValuesProvider::context)native.dc(object);else if(type==NativeValuesProvider::write_words)native.words(object,nullptr,true);}
            struct Scope {c3x_native_access::Provider* prior=c3x_native_access::provider();Scope(NativeValuesProvider& p){c3x_native_access::provider()=&p;}~Scope(){c3x_native_access::provider()=prior;}} scope(native);
            if(subtype==1){NativeOperationInput operation;operation.decode(in);
                if(operation.operation==C3X_NATIVE_IMAGE_PRESENT){auto w=native.field(operation.image,0x38),h=native.field(operation.image,0x3c);require(w>0&&w<=2240&&h>0&&h<=1260,"native presentation extent");SetWindowPos(window,nullptr,0,0,w,h,SWP_NOZORDER|SWP_NOACTIVATE);}
                actual=measure_replay([&]{return c3x_renderer_native_image(operation.operation,operation.image,operation.source,operation.from,operation.to,operation.color);});
                if(operation.operation==C3X_NATIVE_DESTROY)native_values.retire(unsigned(reinterpret_cast<std::uintptr_t>(operation.image)));
                if(operation.operation==C3X_NATIVE_UNIT_DRAW&&operation.to&&wanted==1)for(auto n:operation.b){auto old=expected.u32();if(!performance)require(std::uint32_t(n)==old,"native unit bounds differ");}
            }else if(subtype==2){int action=0;in(action);auto image=NativeOperationInput::object(in.u32());Frame owned;c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request)};bool present=in.u32()!=0;
                if(present){c3x_renderer_camera_identity_v1_fields(in,request.identity);frame(in,owned);request.frame=&owned.value;}
                c3x_renderer_output_v1 output={C3X_RENDERER_API_VERSION,sizeof(output)};
                actual=measure_replay([&]{return c3x_renderer_native_map(action,image,present?&request:nullptr,action==C3X_NATIVE_MAP_PREPARE?&output:nullptr);});

                if(performance)expected.at=expected.bytes.size();else check_output(expected,output,action==C3X_NATIVE_MAP_PREPARE?actual:0);
            }else if(subtype==6){actual=measure_replay([&]{c3x_renderer_reset();return native_reset_outcome();});images.clear();tickets.clear();cameras.clear();formats.clear();displayed={};canvases.reset();native_pixels={};
            }else if(subtype==7){bool present;auto path=in.string(32768,&present);actual=measure_replay([&]{return c3x_renderer_set_pack_path(present?path.c_str():nullptr);});images.clear();tickets.clear();cameras.clear();
            }else if(subtype==8){bool present[4]={};std::string paths[4];for(unsigned n=0;n<4;++n)paths[n]=in.string(32768,&present[n]);actual=measure_replay([&]{return c3x_renderer_set_definition_paths(present[0]?paths[0].c_str():nullptr,present[1]?paths[1].c_str():nullptr,present[2]?paths[2].c_str():nullptr,present[3]?paths[3].c_str():nullptr);});images.clear();tickets.clear();cameras.clear();
            }else if(subtype==3||subtype==5){int action=0;if(subtype==5)in(action);auto image=NativeOperationInput::object(in.u32());custom_renderer_native_view native_view_value={};if(subtype==5)native_view(in,native_view_value);
                Frame owned;c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request)};bool present=in.u32()!=0;
                if(present){c3x_renderer_camera_identity_v1_fields(in,request.identity);frame(in,owned);request.frame=&owned.value;}
                if(subtype==3){c3x_renderer_i64 ticket=0;actual=measure_replay([&]{return c3x_renderer_native_camera_request(image,present?&request:nullptr,&ticket);});auto old=std::int64_t(expected.u64());if(old)cameras[old]=ticket;}
                else {
                    actual=measure_replay([&]{auto deadline=GetTickCount64()+60000;do{
                        actual=c3x_renderer_native_navigation(action,image,&native_view_value,present?&request:nullptr);
                        if(!performance||action!=C3X_NAV_POLL||wanted!=C3X_RENDERER_RESULT_OK||actual!=C3X_RENDERER_RESULT_PENDING)break;
                        require(GetTickCount64()<deadline,"performance native navigation timeout");Sleep(1);
                    }while(true);return actual;});
                    Writer value;native_view(value,native_view_value);expected.available(value.bytes.size());if(!performance)require(std::equal(value.bytes.begin(),value.bytes.end(),expected.bytes.begin()+expected.at),"native navigation adoption differs");expected.at+=value.bytes.size();
                }
            }else if(subtype==4){auto image=NativeOperationInput::object(in.u32());auto ticket=id(cameras,std::int64_t(in.u64()));c3x_renderer_gpu_camera_view_v1 view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
                actual=measure_replay([&]{auto deadline=GetTickCount64()+60000;do{actual=c3x_renderer_native_camera_poll(image,ticket,&view);
                    if(!performance||wanted!=C3X_RENDERER_RESULT_OK||actual!=C3X_RENDERER_RESULT_PENDING)break;
                    require(GetTickCount64()<deadline,"performance native camera timeout");Sleep(1);
                }while(true);return actual;});
                if(performance)expected.at=expected.bytes.size();else check_adoption(expected,view.camera,actual);
            }else throw std::runtime_error("unsupported native bridge input");
        }else if(kind==Kind::native_operation){
            require(subtype==1,"unsupported native input operation");int operation=0,context=0;in(operation);auto object=in.u32();in(context);auto thread=in.u32();require(thread&&thread<=64,"invalid native input thread");
            auto invoke=[&]{c3x_renderer_native_lifetime(operation,NativeOperationInput::object(object),context);actual=native_lifetime_outcome();};
            if(thread==1)invoke();else{std::thread other(invoke);other.join();}
        }else if(kind==Kind::native_snapshot&&subtype>=2&&subtype<=4){canvases.event(subtype,in);actual=1;
        }else if(kind==Kind::native_snapshot){
            require(subtype==0,"unknown native snapshot");auto present=in.u32();require(present<=1,"invalid native screen presence");c3x_native_images::ScreenSnapshot screen;
            if(present){in(screen.width);in(screen.height);in(screen.native_format);std::int32_t area[4]={};for(auto& x:area)in(x);screen.area={area[0],area[1],area[2],area[3]};screen.window=window;
                native_pixels.decode(in);auto count=native_pixels.pixels.size();require(screen.width>0&&screen.width<=2240&&screen.height>0&&screen.height<=1260&&screen.native_format>=1&&screen.native_format<=2&&
                    count==((unsigned(screen.width)+1)&~1u)*unsigned(screen.height),"invalid native screen extent");
                screen.pixels=native_pixels.pixels;
                require(window!=nullptr,"native screen requires replay window");SetWindowPos(window,nullptr,0,0,screen.width,screen.height,SWP_NOZORDER|SWP_NOACTIVATE);}else native_pixels={};
            actual=get_renderer_worker().native_screen(present?&screen:nullptr);
        }else if(kind==Kind::world_page){
            c3x_renderer_world_page_v1 page={};page.struct_size=sizeof(page);in(page.first);in(page.capacity);in(page.count);c3x_renderer_camera_identity_v1_fields(in,page.identity);
            Frame owned;frame(in,owned);page.frame=owned.value;int callback_result=0;in(callback_result);
            require(page.count<=page.capacity&&page.count<=128,"replay world page bounds");std::vector<c3x_renderer_tile_v1> records(page.count);
            for(auto& record:records)c3x_renderer_tile_v1_fields(in,record);page.tiles=records.data();
            actual=get_renderer_worker().replay_world_page(page,callback_result);
        }else if(kind==Kind::image_commands){
            require(subtype==0,"unknown image input");Images owned;c3x_inputs::images(in,owned);auto& request=owned.value;
            auto old_image=request.image;request.ticket=id(tickets,request.ticket);request.image=id(images,request.image);
            for(auto& c:owned.commands){c.destination=id(images,c.destination);c.source=id(images,c.source);c.background=id(images,c.background);
                c.detail=id(images,c.detail);c.background_detail=id(images,c.background_detail);c.program=id(images,c.program);}
            std::vector<unsigned> pixels;if(request.action==C3X_GPU_READBACK)pixels.resize(request.pixel_count);
            c3x_renderer_gpu_result_v1 result={sizeof(result)};actual=c3x_renderer_gpu_images(&request,&result,pixels.empty()?nullptr:pixels.data(),unsigned(pixels.size()));
            auto old_result=std::int64_t(expected.u64());auto count=expected.u32(),witness=expected.u32();require(witness<=1,"invalid output witness");
            if(witness){require(actual==C3X_RENDERER_RESULT_OK&&result.pixel_count==count&&count<=pixels.size(),"replay readback extent differs");
                auto hash=c3x_renderer::asset_content_hash(reinterpret_cast<unsigned char const*>(pixels.data()),std::size_t(count)*4);
                for(auto part:hash)require(part==expected.u32(),"replay pixel witness differs");}
            if(actual==C3X_RENDERER_RESULT_OK){if(request.action==C3X_GPU_CREATE){require(old_result&&result.image,"replay missing created identity");images[old_result]=result.image;formats[result.image]=request.format;}
                else if(request.action==C3X_GPU_DESTROY)images.erase(old_image);}
        }else if(kind==Kind::unit&&subtype==2){
            auto variant=in.u32();c3x_renderer_unit_v1 value={};unit(in,value);auto flags=in.u32(),dest_id=in.u32(),back_id=in.u32();auto& dest=canvases.get(dest_id);Canvas* back=back_id?&canvases.get(back_id):nullptr;int bounds[4]={};
            if(variant==1)actual=c3x_renderer_unit_draw(&value,dest.dc);
            else if(variant==2)actual=c3x_renderer_unit_draw_background(&value,dest.dc,back?back->dc:nullptr);
            else if(variant==3)actual=c3x_renderer_unit_draw_expanded(&value,dest.dc,back?back->dc:nullptr,bounds);
            else if(variant==4)actual=c3x_renderer_unit_draw_playback(&value,dest.dc,back?back->dc:nullptr,bounds,flags);
            else throw std::runtime_error("unknown CPU unit entry");
            GdiFlush();auto has_bounds=expected.u32();require(has_bounds<=1,"invalid CPU unit bounds witness");if(has_bounds)for(unsigned n=0;n<4;++n){int recorded=0;expected(recorded);if(wanted==1)require(bounds[n]==recorded,"CPU unit bounds differ");}
            canvases.check(expected,dest);if(back&&back!=&dest)canvases.check(expected,*back);
        }else if(kind==Kind::unit){
            require(subtype==1,"unsupported unit input");c3x_renderer_unit_v1 value={};unit(in,value);c3x_renderer_gpu_unit_v1 dest={};target_fields(in,dest);target(dest);
            int bounds[4]={};actual=c3x_renderer_gpu_unit(&value,&dest,bounds);
            for(unsigned n=0;n<4;++n){int recorded=0;expected(recorded);if(wanted==C3X_RENDERER_RESULT_OK)require(bounds[n]==recorded,"replay unit bounds differ");}
        }else if(kind==Kind::unit_forget){int unit_id=0;in(unit_id);c3x_renderer_unit_forget(unit_id);actual=1;
        }else if(kind==Kind::tactical){c3x_renderer_gpu_unit_v1 dest={};target_fields(in,dest);target(dest);
            c3x_renderer::tactical::Input value;tactical(in,value);actual=get_renderer_worker().draw_tactical(value,dest);
        }else if(kind==Kind::presentation){
            c3x_renderer_gpu_present_v1 value={sizeof(value)};in(value.action);in(value.ticket);in(value.image);in(value.width);in(value.height);for(auto& x:value.area)in(x);auto owner=in.u32();require(owner<=1,"invalid presentation caller role");
            value.ticket=id(tickets,value.ticket);value.image=id(images,value.image);value.window=window;
            if(value.action==0){require(window!=nullptr,"replay requires presentation window");SetWindowPos(window,nullptr,0,0,value.width,value.height,SWP_NOZORDER|SWP_NOACTIVATE);}
            if(owner)actual=c3x_renderer_gpu_present(&value);
            else{require(replay_clock()->values.empty(),"nonowner presentation consumed a visual clock");
                std::thread caller([&]{actual=c3x_renderer_gpu_present(&value);});caller.join();}
            if(actual==1)displayed=value;
        }else if(kind==Kind::reset){actual=drain_native_composition()?1:0;if(actual)destroy_renderer_worker();images.clear();tickets.clear();cameras.clear();formats.clear();displayed={};canvases.reset();native_pixels={};
        }else if(kind==Kind::visual){
            if(subtype==1){auto automatic=in.u32();require(automatic<=1,"invalid ambient offer");
                // No clock consumed means this offer was rejected before work.
                // Such decisions are evidence, not forced render durations.
                actual=measure_replay([&]{return performance?get_renderer_worker().visual_frame(automatic!=0,true):(replay_clock()->values.empty()?C3X_RENDERER_RESULT_PENDING:c3x_renderer_gpu_visual_frame());});
            }else if(subtype==3)actual=get_renderer_worker().visual_policy(in.u32());
            else if(subtype==4){auto ticks=c3x_renderer_visual_clock();require(std::uint64_t(ticks)==expected.u64(),"exported visual clock differs");actual=1;}
            else throw std::runtime_error("unknown visual input");
        }else if(kind==Kind::camera){
            if(subtype==1){c3x_renderer_camera_identity_v1 identity={};c3x_renderer_camera_identity_v1_fields(in,identity);Frame owned;frame(in,owned);
                c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&owned.value,identity};c3x_renderer_i64 ticket=0;
                actual=c3x_renderer_gpu_camera_begin(&request,&ticket);auto old=std::int64_t(expected.u64());if(old)cameras[old]=ticket;
            }else if(subtype==2||subtype==3){
                auto ticket=id(cameras,std::int64_t(in.u64()));c3x_renderer_gpu_frame_v1 view={sizeof(view)};
                c3x_renderer_output_v1 metadata={C3X_RENDERER_API_VERSION,sizeof(metadata)};c3x_renderer_gpu_camera_view_v1 adopted={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(adopted)};
                if(wanted==C3X_RENDERER_RESULT_PENDING)actual=wanted; // forensic readiness barrier, never a performance result
                else{auto deadline=GetTickCount64()+60000;
                    do{if(subtype==2)actual=c3x_renderer_gpu_camera_poll(ticket,&view,&metadata);
                        else{actual=c3x_renderer_gpu_camera_poll_view(ticket,&adopted);if(actual==1){view=adopted.image;metadata=adopted.camera.output;}}
                        if(actual!=C3X_RENDERER_RESULT_PENDING)break;Sleep(1);
                    }while(GetTickCount64()<deadline);}
                map_result(expected,view,metadata,actual);if(subtype==3){check_adoption(expected,adopted.camera,actual);if(actual==1){int x=0,y=0;expected(x);expected(y);require(x==adopted.pixel_phase_x&&y==adopted.pixel_phase_y,"GPU adopted pixel phase differs");}}
            }else if(subtype==4)actual=c3x_renderer_camera_cancel(id(cameras,std::int64_t(in.u64())));
            else if(subtype>=5&&subtype<=7){int query=0;if(subtype==6)in(query);c3x_renderer_camera_identity_v1 identity={};c3x_renderer_camera_identity_v1_fields(in,identity);Frame owned;frame(in,owned);
                c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&owned.value,identity};
                if(subtype==5){c3x_renderer_camera_view_v1 view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};actual=c3x_renderer_camera_present_view(&request,&view);check_adoption(expected,view,actual);}
                else if(subtype==6)actual=c3x_renderer_prepare_view(&request,query);else actual=c3x_renderer_prepare_nearby_view(&request);
            }else if(subtype==8||subtype==9){
                c3x_renderer_camera_identity_v1 identity={};if(subtype==9)c3x_renderer_camera_identity_v1_fields(in,identity);Frame owned;frame(in,owned);
                c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&owned.value,identity};c3x_renderer_i64 ticket=0;
                actual=subtype==8?c3x_renderer_camera_begin(&owned.value,&ticket):c3x_renderer_camera_begin_view(&request,&ticket);
                auto old=std::int64_t(expected.u64());if(old)cameras[old]=ticket;
            }else if(subtype==10||subtype==11){
                auto ticket=id(cameras,std::int64_t(in.u64()));c3x_renderer_output_v1 output={C3X_RENDERER_API_VERSION,sizeof(output)};
                c3x_renderer_camera_view_v1 view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
                if(wanted==C3X_RENDERER_RESULT_PENDING)actual=wanted;
                else{auto deadline=GetTickCount64()+60000;do{
                    actual=subtype==10?c3x_renderer_camera_poll(ticket,&output):c3x_renderer_camera_poll_view(ticket,&view);
                    if(actual!=C3X_RENDERER_RESULT_PENDING)break;Sleep(1);
                }while(GetTickCount64()<deadline);}
                if(subtype==10)check_output(expected,output,actual);else check_adoption(expected,view,actual);
            }else throw std::runtime_error("unsupported camera transition");
        }else throw std::runtime_error("unsupported production replay family");
        in.done();expected.done();replay_execution().result=actual;
        bool variable=performance&&kind==Kind::visual&&subtype==1;
        if(variable&&actual!=wanted)require((wanted==C3X_RENDERER_RESULT_OK||wanted==C3X_RENDERER_RESULT_PENDING||wanted==C3X_RENDERER_RESULT_SUPERSEDED)&&(actual==C3X_RENDERER_RESULT_OK||actual==C3X_RENDERER_RESULT_PENDING||actual==C3X_RENDERER_RESULT_SUPERSEDED),"performance replay producer failed");
        if(actual!=wanted&&!variable)throw std::runtime_error("production replay return code differs: expected="+std::to_string(wanted)+" actual="+std::to_string(actual));return actual;
    }
    void write_frame(wchar_t const* path){
        std::vector<unsigned> pixels;unsigned width=0,height=0;
        require(renderer_worker&&renderer_worker->replay_display(pixels,width,height),"cannot export retained replay display");
        display_bitmap(path,pixels,width,height);
    }
};
inline ReplayState& replay_state(){static ReplayState state;return state;}
}
// The tool passes an explicitly encoded envelope. No callbacks, structures or
// game pointers cross this optional diagnostic entry. Normal DLL users ignore it.
extern "C" __declspec(dllexport) int c3x_renderer_input_replay(
    unsigned char const* bytes,unsigned size,void* window,char* error,unsigned error_size){
    c3x_inputs::ReplayClock clock;
    auto& execution=c3x_inputs::replay_execution();execution.service_ms=0;execution.result=0;execution.reused_adoption=false;
    try{
        c3x_inputs::require(bytes&&size<=2*c3x_inputs::payload_limit&&error&&error_size,"invalid replay envelope");
        c3x_inputs::require(!c3x_inputs::runtime().active(),"capture cannot run inside input replay");
        c3x_inputs::Bytes envelope(bytes,bytes+size);c3x_inputs::Reader in{envelope};
        auto kind=c3x_inputs::Kind(in.u32());auto subtype=in.u32();auto count=in.u32();in.available(count);
        c3x_inputs::Bytes input(envelope.begin()+in.at,envelope.begin()+in.at+count);in.at+=count;
        count=in.u32();in.available(count);c3x_inputs::Bytes expected(envelope.begin()+in.at,envelope.begin()+in.at+count);in.at+=count;
        count=in.u32();c3x_inputs::require(count<=64,"excessive replay clock samples");
        for(unsigned n=0;n<count;++n){auto ticks=std::int64_t(in.u64()),frequency=std::int64_t(in.u64());clock.values.emplace_back(ticks,frequency);}in.done();
        struct ClockScope {ClockScope(c3x_inputs::ReplayClock& clock){c3x_inputs::replay_clock()=&clock;}~ClockScope(){c3x_inputs::replay_clock()=nullptr;}} scope(clock);
        c3x_inputs::Reader payload{input},result{expected};
        c3x_inputs::replay_state().call(kind,subtype,payload,result,static_cast<HWND>(window));
        c3x_inputs::require(!clock.failure,clock.failure?clock.failure:"replay clock failure");
        c3x_inputs::replay_assets().check();if(!execution.performance)c3x_inputs::require(clock.at==clock.values.size(),"replay has unconsumed clock inputs");error[0]=0;return 1;
    }catch(std::exception const& e){if(error&&error_size){std::snprintf(error,error_size,"%s",clock.failure?clock.failure:e.what());error[error_size-1]=0;}return 0;}
}

extern "C" __declspec(dllexport) int c3x_renderer_input_replay_frame(wchar_t const* path){
    try{c3x_inputs::require(path!=nullptr,"missing frame path");c3x_inputs::replay_state().write_frame(path);return 1;}
    catch(...){return 0;}
}

extern "C" __declspec(dllexport) int c3x_renderer_input_replay_fingerprint(unsigned* extent,unsigned* hash){
    try{
        c3x_inputs::require(extent&&hash,"missing display fingerprint output");
        std::vector<unsigned> pixels;unsigned width=0,height=0;
        c3x_inputs::require(renderer_worker&&renderer_worker->replay_display(pixels,width,height),"cannot fingerprint retained replay display");
        auto value=c3x_renderer::asset_content_hash(reinterpret_cast<unsigned char const*>(pixels.data()),pixels.size()*4);
        extent[0]=width;extent[1]=height;std::copy(value.begin(),value.end(),hash);return 1;
    }catch(...){return 0;}
}

extern "C" __declspec(dllexport) int c3x_renderer_input_replay_asset(unsigned char const* bytes,unsigned size){
    try{if(!bytes&&!size){c3x_inputs::replay_assets().begin();return 1;}
        c3x_inputs::require(bytes&&size<=65536,"invalid replay asset envelope");c3x_inputs::Bytes data(bytes,bytes+size);c3x_inputs::Reader in{data};std::string path;
        auto asset=c3x_inputs::asset_fields(in,path);in.done();c3x_inputs::replay_assets().add(std::move(path),asset);return 1;
    }catch(...){return 0;}
}

// A failed replay owns only proxy identities. Teardown never materializes pixels
// into a native game surface that does not exist in this process.
extern "C" __declspec(dllexport) void c3x_renderer_input_replay_shutdown(){
    delete native_composition;native_composition=nullptr;destroy_renderer_worker();
}

extern "C" __declspec(dllexport) int c3x_renderer_input_replay_execution(unsigned performance,double* service_ms,int* result,unsigned* reused){
    if(performance>1)return 0;auto& state=c3x_inputs::replay_execution();state.performance=performance!=0;
    if(service_ms)*service_ms=state.service_ms;if(result)*result=state.result;if(reused)*reused=state.reused_adoption?1u:0u;return 1;
}
