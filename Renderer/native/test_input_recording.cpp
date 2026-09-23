#define NOMINMAX
#include "input_recording/journal.h"
#include "input_recording/pixel_delta.h"
#include "input_recording/inspect.h"
#include <sstream>
#include <iostream>
#include <cassert>
#include <algorithm>
#ifdef _WIN32
#include "input_recording/assets.h"
#endif
using namespace c3x_inputs;
int main(int argc,char** argv){try{
    require(argc==2,"test requires disposable root");auto root=std::filesystem::path(argv[1]);std::filesystem::create_directories(root);
#ifdef _WIN32
    {auto file=root/"path-contract.txt";std::ofstream(file)<<"stable filesystem identity";
        auto missing=root/"absent-directory"/"optional.bin";
        for(auto path:{file,missing}){
            auto stable=asset_path(path.c_str());auto resolved=std::filesystem::u8path(stable);
            require(asset_path(resolved.c_str())==stable,"filesystem path is not idempotent");
            auto relative=std::filesystem::relative(path,std::filesystem::current_path());
            require(asset_path(relative.c_str())==stable,"relative and stable asset identities differ");
        }
        require(asset_file(asset_path(file.c_str())).exists&&!asset_file(asset_path(missing.c_str())).exists,"stable optional asset presence differs");}
#endif
    c3x_renderer_tile_v1 tile={};tile.tile_x=-8;tile.tile_y=3;tile.tile_flags=458752;tile.anchor_x=-71;tile.resource_id=29;
    std::memcpy(tile.resource_name,"cattle",7);std::memcpy(tile.city_owner,"test-owner",11);tile.barbarian_tribe_id=18;
    c3x_renderer_u32 topology[]={0x12345678,0xffffffff};c3x_renderer_frame_v1 frame={};frame.api_version=C3X_RENDERER_API_VERSION;frame.struct_size=sizeof(frame);
    frame.target_width=2240;frame.target_height=1260;frame.clip_right=2240;frame.clip_bottom=1260;frame.tile_count=1;frame.tiles=&tile;
    frame.presentation_time_ticks=1234567890123;frame.presentation_frequency=24000000;frame.world_topology_count=2;frame.world_topology=topology;
    Writer data;c3x_inputs::frame(data,frame);auto encoded=data.bytes;
    Frame restored;Reader parser{encoded};c3x_inputs::frame(parser,restored);parser.done();
    require(restored.value.tiles!=&tile&&restored.value.world_topology!=topology,"replay must own arrays");
    require(restored.value.presentation_time_ticks==frame.presentation_time_ticks&&restored.tiles[0].tile_x==-8&&restored.topology[1]==0xffffffff,"scene values changed");
    Writer roundtrip;c3x_inputs::frame(roundtrip,restored.value);require(roundtrip.bytes==encoded,"canonical scene roundtrip differs");
    auto bad=encoded;bad.pop_back();try{Reader r{bad};c3x_inputs::frame(r,restored);throw 1;}catch(std::runtime_error const&){}
    c3x_renderer_unit_v1 actor={};actor.struct_size=sizeof(actor);actor.unit_id=41;actor.action=8;actor.queued_action=3;
    actor.action_cursor=19;actor.direction=7;actor.presentation_time_ticks=UINT64_C(1234567890123);actor.presentation_frequency=24000000;
    std::memcpy(actor.unit_key,"PRTO_Worker",12);Writer actor_wire;unit(actor_wire,actor);c3x_renderer_unit_v1 actor_copy={};Reader actor_in{actor_wire.bytes};unit(actor_in,actor_copy);actor_in.done();
    require(actor_copy.action==8&&actor_copy.queued_action==3&&actor_copy.action_cursor==19&&actor_copy.presentation_time_ticks==actor.presentation_time_ticks,"unit action/time changed");
    auto changed=actor;changed.action_cursor++;Writer changed_wire;unit(changed_wire,changed);require(changed_wire.bytes!=actor_wire.bytes,"action mutation lost");
    c3x_renderer_unit_visual_v1 visual={sizeof(visual)};visual.unit_id=41;visual.action=2;
    visual.pixel_x=127;visual.pixel_y=-2;visual.target_x=215;visual.target_y=42;
    visual.body_x=640;visual.body_y=320;visual.projection_scale_milli=750;
    visual.damage=2;visual.max_hp=4;visual.flags=C3X_RENDERER_UNIT_STATE_CAPTURED;
    visual.presentation_time_ticks=1234567890123;visual.presentation_frequency=24000000;
    Writer visual_wire;unit_visual_fields(visual_wire,visual);
    c3x_renderer_unit_visual_v1 visual_copy={sizeof(visual_copy)};
    Reader visual_in{visual_wire.bytes};unit_visual_fields(visual_in,visual_copy);visual_in.done();
    require(visual_copy.pixel_y==-2&&visual_copy.target_x==215&&visual_copy.damage==2&&visual_copy.max_hp==4&&
        visual_copy.presentation_time_ticks==visual.presentation_time_ticks,"unit visual observation changed");
    c3x_renderer_unit_move_v1 accepted={sizeof(accepted)};accepted.unit_id=41;
    accepted.old_x=4;accepted.old_y=4;accepted.new_x=5;accepted.new_y=5;
    accepted.action=2;accepted.source_visible=0;accepted.target_visible=1;
    accepted.map_epoch=17;accepted.viewer_epoch=3;
    accepted.presentation_time_ticks=1234567890123;accepted.presentation_frequency=24000000;
    Writer move_wire;unit_move_fields(move_wire,accepted);
    c3x_renderer_unit_move_v1 move_copy={sizeof(move_copy)};
    Reader move_in{move_wire.bytes};unit_move_fields(move_in,move_copy);move_in.done();
    require(move_copy.unit_id==41&&move_copy.old_x==4&&move_copy.new_x==5&&
        move_copy.source_visible==0&&move_copy.target_visible==1&&move_copy.viewer_epoch==3,
        "accepted unit move changed in replay wire");
    c3x_renderer_unit_spawn_v1 born={sizeof(born)};born.unit_id=41;born.tile_x=5;born.tile_y=5;
    born.unit_type_id=2;born.owner_id=1;born.visible=1;born.map_epoch=17;born.viewer_epoch=3;
    born.presentation_time_ticks=1234567890123;born.presentation_frequency=24000000;
    Writer spawn_wire;unit_spawn_fields(spawn_wire,born);
    c3x_renderer_unit_spawn_v1 spawn_copy={sizeof(spawn_copy)};
    Reader spawn_in{spawn_wire.bytes};unit_spawn_fields(spawn_in,spawn_copy);spawn_in.done();
    require(spawn_copy.unit_id==41&&spawn_copy.tile_x==5&&spawn_copy.owner_id==1&&
        spawn_copy.map_epoch==17,"accepted unit spawn changed in replay wire");
    c3x_renderer_unit_state_v1 state={sizeof(state)};state.kind=C3X_RENDERER_UNIT_STATE_OBSERVE;
    state.unit_id=41;state.tile_x=5;state.tile_y=5;state.unit_type_id=2;state.owner_id=1;
    state.action=3;state.damage=2;state.max_hp=4;state.visible=1;
    state.map_epoch=17;state.viewer_epoch=3;
    state.presentation_time_ticks=1234567890124;state.presentation_frequency=24000000;
    Writer state_wire;unit_state_fields(state_wire,state);
    c3x_renderer_unit_state_v1 state_copy={sizeof(state_copy)};
    Reader state_in{state_wire.bytes};unit_state_fields(state_in,state_copy);state_in.done();
    require(state_copy.kind==C3X_RENDERER_UNIT_STATE_OBSERVE&&state_copy.unit_id==41&&
        state_copy.damage==2&&state_copy.viewer_epoch==3,"accepted unit state changed in replay wire");
    unsigned upload[]={0,0xffffffffu,0x12345678u};c3x_renderer_gpu_images_v1 native={};native.struct_size=sizeof(native);native.action=C3X_GPU_UPLOAD;
    native.ticket=9;native.image=21;native.revision=3;native.pixel_count=3;native.pixels=upload;Writer upload_wire;images(upload_wire,native);Images native_copy;Reader upload_in{upload_wire.bytes};images(upload_in,native_copy);upload_in.done();
    upload[1]=0;require(native_copy.value.pixels!=upload&&native_copy.pixels[1]==0xffffffffu,"native CPU source must be immutable");
    c3x_renderer::tactical::Input overlay;overlay.ring(200,90,128,true);overlay.line(90,70,180,200);Writer overlay_wire;tactical(overlay_wire,overlay);
    c3x_renderer::tactical::Input overlay_copy;Reader overlay_in{overlay_wire.bytes};tactical(overlay_in,overlay_copy);overlay_in.done();require(overlay_copy.animated&&overlay_copy.primitives.size()==2,"selection/path input lost");
    ScreenInput screen_encoder;ScreenReplay screen_decoder;std::vector<unsigned short> native_pixels(40000,0x5a31);
    Writer initial;screen_encoder.encode(initial,native_pixels);Reader initial_in{initial.bytes};screen_decoder.decode(initial_in);initial_in.done();
    require(screen_decoder.pixels==native_pixels,"native screen initial input differs");
    Writer unchanged;screen_encoder.encode(unchanged,native_pixels);require(unchanged.bytes.size()==40,"unchanged screen must use content reference");
    Reader unchanged_in{unchanged.bytes};screen_decoder.decode(unchanged_in);unchanged_in.done();
    native_pixels[17001]^=73;Writer delta;screen_encoder.encode(delta,native_pixels);require(delta.bytes.size()<20000,"screen delta must contain only changed block");
    Reader delta_in{delta.bytes};screen_decoder.decode(delta_in);delta_in.done();require(screen_decoder.pixels==native_pixels,"native CPU write lost in delta");
    auto corrupt_delta=delta.bytes;corrupt_delta.back()^=1;
    try{ScreenReplay wrong;Reader r{initial.bytes};wrong.decode(r);Reader changed_input{corrupt_delta};wrong.decode(changed_input);throw 1;}catch(std::runtime_error const&){}
    try{ScreenReplay missing;Reader r{delta.bytes};missing.decode(r);throw 1;}catch(std::runtime_error const&){}
    // Small segments force cross-file reconstruction; no render output is an input.
    Limits limits;limits.segment_bytes=2048;limits.total_bytes=1u<<20;
    {Journal journal(root/"roundtrip",1000,limits);for(unsigned n=0;n<120;++n)require(journal.emit(Kind::scene,n*5000,encoded),"input enqueue");journal.finish(Stop::duration);}
    SegmentReader reader(root/"roundtrip");Event event;unsigned count=0;
    while(!reader.footer&&reader.next(event)){if(event.kind==Kind::footer)break;
        require(event.kind==Kind::scene&&event.payload==encoded&&event.ticks==count*5000,"input event/order/time changed");++count;}
    require(count==120&&reader.footer&&reader.reason==Stop::duration&&reader.segment>1,"segmented duration reconstruction");
    {Journal journal(root/"unit-events",1000,limits);
        require(journal.emit(Kind::unit_move,1,move_wire.bytes),"move event not recorded");
        require(journal.emit(Kind::unit_spawn,2,spawn_wire.bytes),"spawn event not recorded");
        require(journal.emit(Kind::unit_state,3,state_wire.bytes),"state event not recorded");
        journal.finish(Stop::closed);}
    {SegmentReader stream(root/"unit-events");
        require(stream.next(event)&&event.kind==Kind::unit_move&&event.payload==move_wire.bytes&&event.sequence==1,"move event order lost");
        require(stream.next(event)&&event.kind==Kind::unit_spawn&&event.payload==spawn_wire.bytes&&event.sequence==2,"spawn event order lost");
        require(stream.next(event)&&event.kind==Kind::unit_state&&event.payload==state_wire.bytes&&event.sequence==3,"state event order lost");
        require(stream.next(event)&&event.kind==Kind::footer,"unit event journal did not close");}
    // These timestamps exercise indexing, not ten-minute endurance.
    limits.writer_delay_ms=20;limits.queue_bytes=encoded.size()+48;
    {Journal journal(root/"slow",1000,limits);require(journal.emit(Kind::scene,1,encoded),"first bounded enqueue");
        require(!journal.emit(Kind::scene,2,encoded),"slow disk must reject before growing queue");journal.finish(Stop::closed);require(journal.stop_reason()==Stop::queue_full,"queue failure reason lost");}
    {SegmentReader replay(root/"slow");while(!replay.footer&&replay.next(event)){}require(replay.footer&&replay.reason==Stop::queue_full,"overflow cannot certify completion");}
    std::filesystem::copy(root/"roundtrip",root/"gap",std::filesystem::copy_options::recursive);
    std::filesystem::remove(root/"gap"/"segment-000001.c3xi");
    try{SegmentReader replay(root/"gap");throw 1;}catch(std::runtime_error const&){}
    try{Journal overwrite(root/"roundtrip",1000,limits);throw 1;}catch(std::runtime_error const&){}
    {Journal journal(root/"disk-error",1000,limits);std::filesystem::create_directory(root/"disk-error"/"segment-000000.c3xi");
        journal.emit(Kind::scene,1,encoded);journal.finish(Stop::closed);require(journal.stop_reason()==Stop::io_error,"disk error must stop input capture");}
    auto corrupt=root/"roundtrip"/"segment-000000.c3xi";
    {std::fstream file(corrupt,std::ios::binary|std::ios::in|std::ios::out);file.seekg(32+24);char byte;file.read(&byte,1);byte^=1;file.seekp(32+24);file.write(&byte,1);}
    try{SegmentReader replay(root/"roundtrip");replay.next(event);throw 1;}catch(std::runtime_error const&){}
    limits.writer_delay_ms=0;limits.queue_bytes=1u<<20;limits.segment_bytes=2048;limits.total_bytes=2048;
    {Journal journal(root/"quota",1000,limits);for(unsigned n=0;n<16;++n)journal.emit(Kind::scene,n,encoded);journal.finish(Stop::closed);require(journal.stop_reason()==Stop::byte_limit,"quota reason");}
    {SegmentReader replay(root/"quota");while(!replay.footer&&replay.next(event)){}require(replay.footer&&replay.reason==Stop::byte_limit,"quota cannot certify completion");}
    limits.queue_bytes=1u<<20;limits.segment_bytes=1u<<16;limits.total_bytes=1u<<20;
    {
        std::mutex admission;std::condition_variable checked;bool busy=true;unsigned checks=0;
        Journal journal(root/"idle-deadline",1000,limits,[&](Journal& writer){
            std::lock_guard<std::mutex> lock(admission);++checks;
            if(!busy)writer.stop(Stop::duration);checked.notify_one();
        });
        std::unique_lock<std::mutex> lock(admission);
        require(checked.wait_for(lock,std::chrono::seconds(2),[&]{return checks>=2;}),"idle deadline watchdog did not run");
        require(journal.active(),"idle deadline interrupted an admitted call");busy=false;
        require(checked.wait_for(lock,std::chrono::seconds(2),[&]{return !journal.active();}),"idle deadline requires another producer call");
        lock.unlock();journal.finish(Stop::closed);require(journal.stop_reason()==Stop::duration,"idle deadline reason lost");
    }
    {Journal journal(root/"timeline",1000,limits);Writer manifest;manifest.u32(protocol_version);manifest.u32(C3X_RENDERER_API_VERSION);manifest.u64(0);clock_origin(manifest,{10000,20000,10002,1});manifest.u32(0);
        journal.emit(Kind::manifest,0,std::move(manifest.bytes));
        for(unsigned i=1;i<=3;++i){Writer input;input.u64(i);input.u64(0);input.u32(i==2?1:0);
            journal.emit(i==1?Kind::presentation:i==2?Kind::native_snapshot:Kind::visual,i*1000,std::move(input.bytes),i==3?1:0);
            Writer result;result.u64(i);result.u32(1);journal.emit(Kind::result,i*1000+20,std::move(result.bytes));}journal.finish(Stop::closed);}
    {InputInspection inspection;std::ostringstream timeline;inspection.read(root/"timeline",timeline);
        require(inspection.complete&&inspection.frames==3&&inspection.calls==3,"input timeline must include native and ambient presentations");
        require(inspection.origin.qpc==10000&&inspection.origin.utc_filetime==20000&&inspection.origin.qpc_after_utc==10002&&inspection.origin.precise_utc==1,"absolute clock correlation lost");
        require(timeline.str().find("\"seconds\":3.02")!=std::string::npos,"input timeline clock changed");}
    try{Writer invalid;clock_origin(invalid,{10000,20000,9999,1});throw 1;}catch(std::runtime_error const&){}
    std::filesystem::copy(root/"timeline",root/"crash",std::filesystem::copy_options::recursive);
    auto crash=root/"crash"/"segment-000000.c3xi";std::filesystem::resize_file(crash,std::filesystem::file_size(crash)-8);
    {InputInspection inspection;std::ostringstream timeline;inspection.read(root/"crash",timeline);
        require(!inspection.complete&&inspection.verified&&inspection.frames==3,"torn footer must preserve verified frame prefix without certifying completion");}
    std::cout<<"PASS renderer input codec and bounded segmented writer: owned arrays, exact values, rollover, slow disk, quota, disk error, segment gap, header corruption, actions, CPU inputs, tactical overlays\n";return 0;
}catch(std::exception const& error){std::cerr<<error.what()<<'\n';return 1;}catch(...){std::cerr<<"expected rejection did not occur\n";return 1;}}
