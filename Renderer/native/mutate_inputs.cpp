#include "input_recording/journal.h"
#include <iostream>
#include <set>
using namespace c3x_inputs;
int main(int argc,char** argv){try{
    require(argc==4,"usage: mutate_inputs SOURCE NEW_DIRECTORY MUTATION");
    std::string mode=argv[3];std::set<std::string> modes={"missing-clock","missing-asset","missing-unit","missing-reset","alter-visibility","alter-action","alter-cpu-write","alter-configuration","alter-world-range","alter-world-scope","alter-world-topology"};
    require(modes.count(mode)!=0,"unknown mutation");
    // Offline transformation bursts are not paced by rendering. Give this test
    // tool a separate bounded queue; the live recorder keeps its 32 MiB limit.
    Limits mutation_limits;mutation_limits.queue_bytes=128u*1024u*1024u;
    SegmentReader reader(argv[1]);Journal output(argv[2],reader.frequency,mutation_limits);Event event;bool changed=false;std::set<std::uint64_t> omitted;
    while(!reader.footer&&reader.next(event)){if(event.kind==Kind::footer)break;
        if(!changed&&mode.rfind("alter-world-",0)==0&&event.kind==Kind::world_page){
            Reader in{event.payload};Writer out;out.u64(in.u64());out.u64(in.u64());
            auto first=in.u32(),capacity=in.u32(),count=in.u32();c3x_renderer_camera_identity_v1 identity={};
            c3x_renderer_camera_identity_v1_fields(in,identity);Frame owned;frame(in,owned);int result=0;in(result);
            require(count<=128&&count<=capacity,"invalid world mutation page");std::vector<c3x_renderer_tile_v1> records(count);
            for(auto& tile:records)c3x_renderer_tile_v1_fields(in,tile);in.done();
            if(result==C3X_RENDERER_RESULT_OK&&count){
                if(mode=="alter-world-range")first=UINT32_MAX;
                if(mode=="alter-world-scope")identity.viewer_epoch^=1;
                if(mode=="alter-world-topology"){require(!owned.topology.empty(),"world mutation needs topology");owned.topology[0]^=1;}
                out(first);out(capacity);out(count);c3x_renderer_camera_identity_v1_fields(out,identity);frame(out,owned.value);out(result);
                for(auto& tile:records)c3x_renderer_tile_v1_fields(out,tile);event.payload=std::move(out.bytes);changed=true;
            }
        }
        if(!changed&&mode=="alter-configuration"&&event.kind==Kind::manifest){
            Reader in{event.payload};Writer out;out.u32(in.u32());out.u32(in.u32());out.u64(in.u64());ClockOrigin origin;clock_origin(in,origin);clock_origin(out,origin);
            auto count=in.u32();out.u32(count);for(unsigned n=0;n<count;++n){auto key=in.string(256),value=in.string(32768);
                if(key=="C3X_RENDERER_WATER_MOTION"&&value=="1"){value="0";changed=true;}
                out.string(key.c_str(),256);out.string(value.c_str(),32768);}
            in.done();event.payload=std::move(out.bytes);
        }
        if(!changed&&mode=="alter-visibility"&&event.kind==Kind::scene&&event.flags==3){
            Reader in{event.payload};Writer out;out.u64(in.u64());out.u64(in.u64());c3x_renderer_camera_identity_v1 identity={};
            c3x_renderer_camera_identity_v1_fields(in,identity);c3x_renderer_camera_identity_v1_fields(out,identity);
            Frame scene;frame(in,scene);in.done();require(!scene.tiles.empty(),"visibility control needs tiles");
            for(auto& tile:scene.tiles)tile.tile_flags=(tile.tile_flags&~C3X_RENDERER_TILE_VISIBILITY_BITS)|C3X_RENDERER_TILE_VISIBILITY_KNOWN;
            frame(out,scene.value);event.payload=std::move(out.bytes);changed=true;
        }
        if(!changed&&mode=="alter-action"&&event.kind==Kind::unit&&event.flags==1){
            Reader in{event.payload};Writer out;out.u64(in.u64());out.u64(in.u64());c3x_renderer_unit_v1 actor={};unit(in,actor);
            actor.action=2;actor.action_cursor=7;unit(out,actor);c3x_renderer_gpu_unit_v1 target={};target_fields(in,target);target_fields(out,target);in.done();
            event.payload=std::move(out.bytes);changed=true;
        }
        if(!changed&&mode=="alter-cpu-write"&&event.kind==Kind::native_snapshot&&event.flags==3){
            Reader in{event.payload};in.u64();in.u64();in.u32();in.u32();require(in.u32()>0,"CPU control needs pixel rows");in.available(4);
            event.payload[in.at]^=1;changed=true; // Re-encoded journal checksum remains valid.
        }
        if(mode=="missing-asset"&&!changed&&event.kind==Kind::asset&&event.flags==0){Reader in{event.payload};auto path=in.string(32768);
            if(path.find("default.custom_rendering.txt")!=std::string::npos){changed=true;continue;}}
        if(mode=="missing-clock"&&!changed&&event.kind==Kind::visual&&event.flags==2){Reader in{event.payload};if(in.u64()){changed=true;continue;}}
        if(mode=="missing-unit"||mode=="missing-reset"){
            auto target=mode=="missing-unit"?Kind::unit:Kind::reset;
            if(!changed&&event.kind==target){Reader in{event.payload};omitted.insert(in.u64());changed=true;continue;}
            if(!omitted.empty()&&event.kind!=Kind::manifest&&event.kind!=Kind::asset&&event.kind!=Kind::settings){
                Reader in{event.payload};auto token=in.u64();
                if(omitted.count(token))continue;
                if(event.kind!=Kind::result&&!(event.kind==Kind::visual&&event.flags==2)&&omitted.count(in.u64())){omitted.insert(token);continue;}
            }}
        require(output.emit(event.kind,event.ticks,std::move(event.payload),event.flags),"mutation output capacity exceeded");
    }
    require(reader.footer&&(reader.reason==Stop::closed||reader.reason==Stop::duration),"mutation requires complete capture");require(changed,"requested input mutation not found");output.finish(Stop::closed);
    require(output.stop_reason()==Stop::closed,"mutation output failed");std::cout<<"PASS generated semantic negative control: "<<mode<<'\n';return 0;
}catch(std::exception const& error){std::cerr<<error.what()<<'\n';return 1;}}
