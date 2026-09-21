#pragma once
#include "journal.h"
#include <map>
#include <ostream>
namespace c3x_inputs {
struct InputInspection {
    struct Open {Kind kind;unsigned subtype;std::uint64_t ticks,parent;bool display;};
    std::uint64_t events=0,bytes=0,frames=0,calls=0,last_sequence=0,last_ticks=0,frequency=1,peak_second_bytes=0;
    bool complete=false,verified=true;unsigned stop=unsigned(Stop::unsupported);std::string error;
    ClockOrigin origin;
    std::map<unsigned,std::uint64_t> family_bytes,family_events;
    std::map<std::uint64_t,Open> pending;
    void read(std::filesystem::path const& directory,std::ostream& timeline){
        bool manifest=false;std::map<std::uint64_t,std::uint64_t> second_bytes;
        try{SegmentReader reader(directory);frequency=reader.frequency;Event event;bool truncated=false;
            while(!reader.footer&&reader.next_verified(event,true,truncated)){
                Reader payload{event.payload};auto family=unsigned(event.kind);
                ++events;bytes+=48+event.payload.size();family_bytes[family]+=48+event.payload.size();++family_events[family];
                last_sequence=event.sequence;last_ticks=std::max(last_ticks,event.ticks);
                if(event.kind==Kind::footer){stop=unsigned(reader.reason);break;}
                auto current=event.ticks/frequency;require(second_bytes.size()<86400||second_bytes.count(current),"input timeline duration budget");
                auto& burst=second_bytes[current];burst+=48+event.payload.size();peak_second_bytes=std::max(peak_second_bytes,burst);
                if(event.kind==Kind::manifest){require(!manifest&&event.sequence==1,"duplicate/misplaced input manifest");
                    require(payload.u32()==protocol_version&&payload.u32()==C3X_RENDERER_API_VERSION,"input API mismatch");payload.u64();clock_origin(payload,origin);
                    auto count=payload.u32();require(count<=512,"input settings limit");for(unsigned n=0;n<count;++n){payload.string(256);payload.string(32768);}payload.done();manifest=true;continue;}
                require(manifest,"missing input manifest");
                if(event.kind==Kind::asset||event.kind==Kind::settings)continue;
                if(event.kind==Kind::visual&&event.flags==2){auto parent=payload.u64();require(!parent||pending.count(parent),"clock has missing parent");continue;}
                auto token=payload.u64();
                if(event.kind!=Kind::result){auto parent=payload.u64();require(token&&pending.size()<256&&!pending.count(token),"invalid input call token");
                    bool display=(event.kind==Kind::native_bridge&&event.flags==1&&payload.u32()==C3X_NATIVE_IMAGE_PRESENT)||
                        (event.kind==Kind::visual&&event.flags==1)||
                        (event.kind==Kind::presentation&&payload.u32()==0)||
                        (event.kind==Kind::native_snapshot&&event.flags==0&&payload.u32()==1);
                    pending.emplace(token,Open{event.kind,event.flags,event.ticks,parent,display});continue;}
                auto found=pending.find(token);require(found!=pending.end(),"result has missing input");auto input=found->second;pending.erase(found);
                int result=0;payload(result);++calls;bool accepted=input.display&&result==C3X_RENDERER_RESULT_OK;if(accepted)++frames;
                timeline<<"{\"call\":"<<token<<",\"parent\":"<<input.parent<<",\"family\":"<<unsigned(input.kind)<<",\"subtype\":"<<input.subtype
                    <<",\"input_ticks\":"<<input.ticks<<",\"result_ticks\":"<<event.ticks<<",\"seconds\":"<<double(event.ticks)/double(frequency)
                    <<",\"result\":"<<result<<",\"frame\":"<<(accepted?frames:0)<<",\"sequence\":"<<event.sequence<<"}\n";
            }
            complete=manifest&&reader.footer&&(reader.reason==Stop::closed||reader.reason==Stop::duration)&&pending.empty();
            if(!complete)error=truncated?"truncated input segment":!reader.footer?"missing footer":!pending.empty()?"unfinished calls":"capture stopped incomplete";
        }catch(std::exception const& e){error=e.what();verified=false;}
    }
    void report(std::ostream& out)const{
        out<<"{\"schema\":1,\"qualified_for_gameplay\":false,\"complete\":"<<(complete?"true":"false")
            <<",\"verified_prefix\":"<<(verified?"true":"false")<<",\"events\":"<<events<<",\"calls\":"<<calls<<",\"accepted_presentations\":"<<frames
            <<",\"last_verified_sequence\":"<<last_sequence<<",\"duration_seconds\":"<<double(last_ticks)/double(frequency)
            <<",\"frequency\":"<<frequency<<",\"event_bytes\":"<<bytes<<",\"peak_second_bytes\":"<<peak_second_bytes
            <<",\"qpc_origin\":"<<origin.qpc<<",\"utc_filetime_100ns\":"<<origin.utc_filetime
            <<",\"qpc_utc_bracket_ticks\":"<<(origin.qpc_after_utc-origin.qpc)<<",\"precise_utc\":"<<(origin.precise_utc?"true":"false")
            <<",\"unfinished_calls\":"<<pending.size()<<",\"stop_reason\":"<<stop<<",\"error\":\"";
        for(auto c:error){if(c=='"'||c=='\\')out<<'\\';if(c>=32)out<<c;}out<<"\",\"families\":[";bool first=true;
        for(auto const& item:family_bytes){if(!first)out<<',';first=false;out<<"{\"kind\":"<<item.first<<",\"events\":"<<family_events.at(item.first)<<",\"bytes\":"<<item.second<<'}';}out<<"]}\n";
    }
};
}
