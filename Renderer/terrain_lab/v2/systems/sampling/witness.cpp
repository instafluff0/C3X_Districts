// Procedural checker and UV diagnostic_proxy only, never replacement map art.
#include "../../contracts/packet_v1.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
struct Vertex { float x,y,z,u,v,panel; };
int main(int argc,char**argv) {
    if(argc!=7)return 2;
    labv2::Packet p;p.width=atoi(argv[2]);p.height=atoi(argv[3]);p.downsample=atoi(argv[5]);
    labv2::Texture t;t.width=t.height=128;t.format=11;
    std::vector<uint16_t> pixels(128*128*4);
    for(int y=0;y<128;y++)for(int x=0;x<128;x++) {
        float c=((x/8+y/8)&1)?.85f:.1f;
        for(int k=0;k<4;k++)pixels[(y*128+x)*4+k]=uint16_t((k==3?1:c)*65535);
    }
    for(unsigned n=128;n;n/=2){
        labv2::Mip m;m.pitch=n*8;m.bytes.resize(pixels.size()*2);memcpy(m.bytes.data(),pixels.data(),m.bytes.size());t.mips.push_back(m);
        if(n==1)break;
        std::vector<uint16_t> next(n*n);
        for(unsigned y=0;y<n/2;y++)for(unsigned x=0;x<n/2;x++)for(unsigned k=0;k<4;k++) {
            uint32_t sum=0;for(unsigned dy=0;dy<2;dy++)for(unsigned dx=0;dx<2;dx++)sum+=pixels[((2*y+dy)*n+2*x+dx)*4+k];
            next[(y*n/2+x)*4+k]=sum/4;
        }pixels=next;
    }p.textures.push_back(t);
    std::vector<Vertex> vertices;
    for(int panel=0;panel<4;panel++)for(int j=0;j<16;j++)for(int i=0;i<16;i++) {
        Vertex q[4];int dx[]={0,1,0,1},dy[]={0,0,1,1};
        for(int k=0;k<4;k++) {
            float x=(i+dx[k])/16.f,y=(j+dy[k])/16.f;
            float z=panel==1?.035f*std::sin(x*6.283f)*std::sin(y*6.283f):0;
            float cx=panel%2?.5f:-.5f,cy=panel<2?.48f:-.45f;
            float repeat=panel==3?4.f:1.f;
            q[k]={cx+(x-y)*.43f,cy-(x+y-1)*.28f+z,.5f-z,x*(panel==2?16:repeat),y*repeat,float(panel)};
        }
        for(int k:{0,1,2,2,1,3})vertices.push_back(q[k]);
    }
    p.buffers.resize(1);p.buffers[0].resize(vertices.size()*sizeof(Vertex));memcpy(p.buffers[0].data(),vertices.data(),p.buffers[0].size());
    labv2::Draw d;d.stride=sizeof(Vertex);d.count=vertices.size();d.depth=1;d.attributes={{3,0},{2,12},{1,20}};d.textures[0]=1;p.draws.push_back(d);
    return labv2::write_packet(argv[1],p)?0:1;
}
