#include "../../contracts/packet_v1.h"
#include <cstring>
int main(int argc,char**argv){
 auto p=labv2::read_packet(argv[1]);p.binding_contract=2;
 float color[4]={.5f,1.f,.25f,1.f};p.buffers.emplace_back(sizeof(color));
 std::memcpy(p.buffers.back().data(),color,sizeof(color));
 for(auto&d:p.draws)d.frame_buffer=p.buffers.size()-1;
 return labv2::write_packet(argv[2],p)?0:1;
}
