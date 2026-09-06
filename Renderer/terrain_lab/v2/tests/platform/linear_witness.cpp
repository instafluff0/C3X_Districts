#include "../../contracts/packet_v1.h"
#include <cstring>
struct V {float x,y,z,r,g,b,a,u,v;};
int main(int argc,char** argv) {
  labv2::Packet p; p.width=std::stoul(argv[2]);p.height=std::stoul(argv[3]);
  p.downsample=std::stoul(argv[5]); p.color_branch=1;
  p.valid_rect={0,0,p.width/p.downsample,p.height/p.downsample};
  p.buffers.push_back(std::vector<uint8_t>(16));
  auto rect=[&](float x,float y,float w,float h,float z,float r,float g,float b,float a,unsigned depth,unsigned blend){
    V v[]={{x,y,z,r,g,b,a,0,0},{x+w,y,z,r,g,b,a,1,0},{x,y+h,z,r,g,b,a,0,1},{x,y+h,z,r,g,b,a,0,1},{x+w,y,z,r,g,b,a,1,0},{x+w,y+h,z,r,g,b,a,1,1}};
    p.buffers.emplace_back(sizeof(v));std::memcpy(p.buffers.back().data(),v,sizeof(v));
    labv2::Draw d;d.vertex_buffer=p.buffers.size()-1;d.constant_buffer=0;d.count=6;d.stride=sizeof(V);d.depth_mode=depth;d.blend_mode=blend;d.attributes={{3,0},{4,12},{2,28}};p.draws.push_back(d);
  };
  // Explicit diagnostic geometry. No claim of source-backed beauty art.
  rect(-.9f,-.8f,1.8f,1.6f,.9f,.18f,.18f,.18f,1,2,0);
  rect(-.75f,-.55f,.7f,1.1f,.4f,1,0,0,1,2,0);
  rect(-.45f,-.4f,.7f,.8f,.3f,0,1,0,1,2,0);
  // Water over green; premultiplied blue, depth read only.
  rect(-.3f,-.55f,.85f,.65f,.2f,0,0,.5f,.5f,1,1);
  // Cutout uses UV-dependent discard; holes preserve earlier depth and color.
  rect(.1f,.15f,.7f,.5f,.2f,1,.4f,0,-1,2,0);
  rect(.4f,-.6f,.3f,.4f,.1f,12,6,1,1,2,0);
  // Submitted late behind the green body: must remain occluded.
  rect(-.4f,-.3f,.6f,.6f,.6f,1,0,1,1,2,0);
  return labv2::write_packet(argv[1],p)?0:1;
}
