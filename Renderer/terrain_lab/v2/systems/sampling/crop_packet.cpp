// Camera-only micro witness from a cached frozen scene; no model/UV changes.
#include "../../contracts/packet_v1.h"
#include <cstdlib>
#include <cstring>
#include <set>
int main(int argc,char**argv) {
    if(argc!=9)return 2;
    try {
        auto p=labv2::read_packet(argv[1]);
        float x=std::stof(argv[3]), y=std::stof(argv[4]);
        unsigned w=std::stoul(argv[5]),h=std::stoul(argv[6]);
        unsigned zoom=std::stoul(argv[7]);
        float margin=std::stof(argv[8]);
        std::set<unsigned> done;
        for(auto& d:p.draws) {
            if(!done.insert(d.vertex_buffer).second)continue;
            auto& b=p.buffers.at(d.vertex_buffer);
            for(size_t i=0;i<b.size();i+=d.stride) {
                float xy[2];std::memcpy(xy,b.data()+i,8);
                xy[0]=(((xy[0]+1)*p.width*.5f-x)/w)*2-1;
                xy[1]=1-(((1-xy[1])*p.height*.5f-y)/h)*2;
                std::memcpy(b.data()+i,xy,8);
            }
        }
        // Whole-triangle rejection only, with room for the declared pan sequence.
        for(auto& d:p.draws) {
            auto& b=p.buffers[d.vertex_buffer];std::vector<uint8_t> kept;
            for(unsigned i=0;i<d.count;i+=3) {
                float lo[2]={1e30f,1e30f},hi[2]={-1e30f,-1e30f};
                for(unsigned j=0;j<3;j++) {float xy[2];memcpy(xy,b.data()+(i+j)*d.stride,8);
                    for(unsigned k=0;k<2;k++){lo[k]=std::min(lo[k],xy[k]);hi[k]=std::max(hi[k],xy[k]);}}
                if(hi[0]<-1-2*margin/w||lo[0]>1+2*margin/w||hi[1]<-1-2*margin/h||lo[1]>1+2*margin/h)continue;
                kept.insert(kept.end(),b.begin()+i*d.stride,b.begin()+(i+3)*d.stride);
            }
            d.count=unsigned(kept.size()/d.stride);
            // Metal does not allocate zero-byte buffers. Zero-count draws keep
            // inert storage and the original depth-clear ordering.
            if(kept.empty())kept.resize(d.stride);
            b=std::move(kept);
        }
        p.width=w;p.height=h;p.downsample=zoom;
        return labv2::write_packet(argv[2],p)?0:1;
    }catch(const std::exception& e){fprintf(stderr,"sampling crop: %s\n",e.what());return 1;}
}
