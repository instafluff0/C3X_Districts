// Read-only numerical probe of retained packet receiver planes and shadow depth.
#include "../systems/lighting/shadow_field_v1.h"
#include <iostream>
#include <iomanip>
float read_float(std::vector<uint8_t> const& b,size_t n){float v;memcpy(&v,b.data()+n,4);return v;}
int main(int argc,char**argv){
 if(argc!=2)return 2;
 auto p=labv2::read_packet(argv[1]);
 std::cout<<"x,y,kind,edge,center_gap,max_gap,span,resolution\n";
 for(auto const& d:p.draws){
  if(d.feature || d.world_attribute!=UINT32_MAX || d.attributes.size()<17)continue;
  auto const& b=p.buffers[d.vertex_buffer];auto const& f=p.buffers[d.frame_buffer];
  float span=read_float(f,12);int S=int(read_float(f,28));
  auto const& tex=p.textures[d.textures[25]-1];
  for(unsigned i=0;i<d.count;i+=3){
   float kind=read_float(b,i*d.stride+d.attributes[6].offset);
   if(kind<.75f||kind>3.5f)continue;
   float uvz[3][3],screen[3][2];
   for(int j=0;j<3;j++){
    auto at=size_t(i+j)*d.stride;
    screen[j][0]=(read_float(b,at)+1)*p.width*.5f;
    screen[j][1]=(1-read_float(b,at+4))*p.height*.5f;
    q6::F3 w;for(int k=0;k<3;k++)w[k]=read_float(b,at+d.attributes[16].offset+4*k)-read_float(f,48+4*k);
    for(int k=0;k<3;k++)uvz[j][k]=(w[0]*read_float(f,k*16)+w[1]*read_float(f,k*16+4)+w[2]*read_float(f,k*16+8))/span+.5f;
   }
   float ux=uvz[1][0]-uvz[0][0],uy=uvz[1][1]-uvz[0][1],zx=uvz[1][2]-uvz[0][2];
   float vx=uvz[2][0]-uvz[0][0],vy=uvz[2][1]-uvz[0][1],zy=uvz[2][2]-uvz[0][2];
   float det=ux*vy-uy*vx;if(fabs(det)<1e-12)continue;
   float gx=(zx*vy-zy*uy)/det,gy=(zy*ux-zx*vx)/det;
   for(int edge=0;edge<4;edge++){
    float weights[3]={1.f/3,1.f/3,1.f/3};
    if(edge){weights[0]=weights[1]=weights[2]=.49f;weights[edge-1]=.02f;}
    float q[3]={},s[2]={};for(int j=0;j<3;j++){for(int k=0;k<3;k++)q[k]+=weights[j]*uvz[j][k];for(int k=0;k<2;k++)s[k]+=weights[j]*screen[j][k];}
    if(s[0]<150||s[0]>750||s[1]<260||s[1]>450)continue;
    int cx=int(q[0]*S),cy=int(q[1]*S);float center=0,maximum=-1e10;
    for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
     int sx=std::clamp(cx+x,0,S-1),sy=std::clamp(cy+y,0,S-1);uint16_t raw;
     memcpy(&raw,tex.mips[0].bytes.data()+size_t(sy)*tex.mips[0].pitch+sx*8,2);
     float receiver=q[2]+gx*((sx+.5f)/S-q[0])+gy*((sy+.5f)/S-q[1]);
     float gap=(raw/65535.f-receiver)*span;
     if(!x&&!y)center=gap;maximum=std::max(maximum,gap);
    }
    std::cout<<std::setprecision(8)<<s[0]<<','<<s[1]<<','<<kind<<','<<edge<<','<<center<<','<<maximum<<','<<span<<','<<S<<'\n';
   }
  }
 }
}
