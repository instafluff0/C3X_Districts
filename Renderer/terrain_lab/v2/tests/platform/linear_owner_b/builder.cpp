#include "../../../contracts/packet_v1.h"
#include <cstdlib>
#include <cstring>
int main(int argc,char**argv){
 if(argc!=7)return 2;labv2::Packet p;p.width=unsigned(atoi(argv[2]));p.height=unsigned(atoi(argv[3]));p.downsample=unsigned(atoi(argv[5]));
 float vertices[]={0.3f-.45f,-.4f,.5f,.1,.4,1,1, 0.3f+.45f,-.4f,.5f,.1,.4,1,1, 0.3f,.5f,.5f,.1,.4,1,1};
 p.color_branch=1;p.valid_rect={0,0,p.width/p.downsample,p.height/p.downsample};p.buffers.resize(1);p.buffers[0].resize(sizeof(vertices));memcpy(p.buffers[0].data(),vertices,sizeof(vertices));
 labv2::Draw d;d.count=3;d.stride=28;d.depth=1;d.clear_depth=0;d.attributes={{3,0},{4,12}};p.draws.push_back(d);
 return labv2::write_packet(argv[1],p)?0:1;
}
