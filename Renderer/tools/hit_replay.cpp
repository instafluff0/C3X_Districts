// Replays an input-coverage trace (C3X_RENDERER_HIT_TRACE=1, written as
// renderer-core.log.hit by HitWorker::record) into the production coverage
// model, timing each operation. Optional probes hash pixel answers so a
// change to native_hit_scene.h can be checked for identical results.
//
// Build: clang++ -std=c++17 -O2 -I <repo> Renderer/tools/hit_replay.cpp -o hit_replay
// Usage: hit_replay TRACE.hit [--probe N]
// Recorded queries (kind 4) are replayed and their answers compared; exempt
// canvases (kind 5) are replayed in order.
//   --probe N  after every N operations, query a fixed 16-point lattice on the
//              most recently written destination and fold the answers into a
//              hash printed at the end.
#include "Renderer/native/native_hit_scene.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace c3x_gpu_images;

namespace {
struct Reader {
    std::FILE* file;
    template<class T> bool get(T& value){return std::fread(&value,sizeof(value),1,file)==1;}
};
struct Total {std::uint64_t count=0;double us=0,max_us=0;};
}

int main(int argc,char** argv){
    if(argc<2){std::fprintf(stderr,"usage: hit_replay TRACE.hit [--probe N]\n");return 2;}
    unsigned probe_every=0;
    for(int i=2;i+1<argc;++i)if(!std::strcmp(argv[i],"--probe"))probe_every=unsigned(std::atoi(argv[i+1]));
    std::FILE* file=std::fopen(argv[1],"rb");
    if(!file){std::perror(argv[1]);return 1;}
    Reader in{file};
    c3x_native_hit::Scene scene;
    std::map<unsigned,Total> totals;          // operation kind, or 100 + command kind
    std::map<unsigned,std::pair<unsigned,unsigned>> sizes;
    std::vector<unsigned> pixels;
    std::uint64_t operations=0,hash=1469598103934665603ull,probes=0,failures=0,queries=0,mismatches=0;
    std::map<unsigned,unsigned> queried;
    Id last_destination=0;
    double total_us=0;
    for(;;){
        std::uint32_t kind=0,micros=0;std::uint64_t began=0;
        if(!in.get(kind))break;
        if(!in.get(began)||!in.get(micros))break; // the game can exit mid-record
        auto start=std::chrono::steady_clock::now();
        unsigned key=kind;
        try{
            if(kind==0){std::uint64_t id;std::uint32_t w,h,format;in.get(id);in.get(w);in.get(h);in.get(format);
                start=std::chrono::steady_clock::now();scene.create(id,w,h,Format(format));sizes[unsigned(id)]={w,h};}
            else if(kind==1){std::uint64_t id;in.get(id);start=std::chrono::steady_clock::now();scene.destroy(id);}
            else if(kind==2){std::uint64_t id;std::uint32_t n;in.get(id);in.get(n);pixels.resize(n);
                if(n&&std::fread(pixels.data(),sizeof(unsigned),n,file)!=n)break;
                start=std::chrono::steady_clock::now();scene.upload(id,pixels.data(),n);}
            else if(kind==3){
                Command c{};std::uint32_t command_kind,color;std::uint64_t destination,source,background,detail,background_detail,program;
                std::int32_t v[10],sw,sh;
                in.get(command_kind);in.get(destination);in.get(source);for(auto& x:v)in.get(x);
                in.get(color);in.get(background);in.get(detail);in.get(background_detail);in.get(sw);in.get(sh);in.get(program);
                c.kind=Kind(command_kind);c.destination=destination;c.source=source;
                c.area={v[0],v[1],v[2],v[3]};c.clip={v[4],v[5],v[6],v[7]};c.source_x=v[8];c.source_y=v[9];
                c.color=color;c.background=background;c.detail=detail;c.background_detail=background_detail;
                c.source_width=sw;c.source_height=sh;c.program=program;
                key=100+command_kind;last_destination=destination;
                start=std::chrono::steady_clock::now();scene.submit(c);
            }else if(kind==4){std::uint64_t id;std::int32_t x,y;std::uint32_t found,value;
                if(!in.get(id)||!in.get(x)||!in.get(y)||!in.get(found)||!in.get(value))break;
                unsigned answer=0;start=std::chrono::steady_clock::now();bool hit=scene.pixel(id,x,y,answer);
                ++queries;queried[unsigned(id)]++;
                if(hit!=bool(found)||(hit&&answer!=value)){if(mismatches++<8)std::printf("  mismatch id=%llu x=%d y=%d recorded=%u/%08x replayed=%u/%08x\n",
                    (unsigned long long)id,x,y,found,value,unsigned(hit),answer);}
            }else if(kind==5){std::uint64_t id;if(!in.get(id))break;start=std::chrono::steady_clock::now();scene.exempt(id);
            }else{std::fprintf(stderr,"unknown record kind %u\n",kind);return 1;}
        }catch(std::exception const&){++failures;}
        double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
        auto& t=totals[key];++t.count;t.us+=us;if(us>t.max_us)t.max_us=us;total_us+=us;
        ++operations;
        if(probe_every&&operations%probe_every==0&&last_destination){
            auto found=sizes.find(unsigned(last_destination));
            if(found!=sizes.end()){auto [w,h]=found->second;
                for(unsigned j=0;j<16;++j){int x=int((j%4*2+1)*w/8),y=int((j/4*2+1)*h/8);unsigned value=0;
                    bool hit=scene.pixel(last_destination,x,y,value);
                    hash=(hash^(hit?value:0xffffffffu))*1099511628211ull;++probes;}}
        }
    }
    std::printf("operations=%llu replay_ms=%.1f failures=%llu probes=%llu hash=%016llx queries=%llu mismatches=%llu\n",
        (unsigned long long)operations,total_us/1000,(unsigned long long)failures,(unsigned long long)probes,(unsigned long long)hash,
        (unsigned long long)queries,(unsigned long long)mismatches);
    for(auto const& [id,n]:queried)std::printf("  queried image %u: %u\n",id,n);
    for(auto const& [key,t]:totals)
        std::printf("  %s%-3u n=%9llu ms=%10.1f mean_us=%8.2f max_us=%9.1f\n",key>=100?"command ":"op ",key>=100?key-100:key,
            (unsigned long long)t.count,t.us/1000,t.count?t.us/t.count:0.,t.max_us);
    return 0;
}
