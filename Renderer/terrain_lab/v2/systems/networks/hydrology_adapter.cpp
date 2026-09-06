// Offline Q5 consumer of Q3's published field. Does not modify source topology.
#include "../hydrology/field.h"
#include <iomanip>
int main(int argc,char**argv){
 try {
  if(argc!=4)return 2;hydro::Field field(argv[1]);field.wraps=std::atoi(argv[3])!=0;
  // CSV legacy header x is half-raw; tile coordinates are authoritative.
  field.origin_x=field.tile(0,0).raw_x;field.origin_y=field.tile(0,0).raw_y;field.build();
  std::ofstream out(argv[2]);out<<std::setprecision(14);out<<"{\"schema\":\"c3x.q5.crossing_adaptation.v1\",\"source\":\"Q3 Field v1\",\"edges\":[";
  bool first=true;
  for(int r=-1;r<=field.rows;r++)for(int c=-1;c<=field.cols;c++){
   if(field.tile(c,r).base>=11)continue;
   for(auto delta:std::vector<std::pair<int,int>>{{1,0},{0,1}}){
    int x=c+delta.first,y=r+delta.second;
    if(x>field.cols||y>field.rows||field.tile(x,y).base>=11)continue;
    auto crossings=field.crossings({double(c),double(r)},{double(x),double(y)});
    if(!first)out<<',';first=false;
    out<<"{\"from\":["<<c<<','<<r<<"],\"to\":["<<x<<','<<y<<"],\"crossings\":[";
    bool fc=true;
    for(auto const&a:crossings){
     if(!fc)out<<',';fc=false;
     double tx=64*(delta.first+delta.second),ty=64*(delta.first-delta.second),len=std::hypot(tx,ty);tx/=len;ty/=len;
     double wx=a.tangent.x+a.tangent.y,wy=a.tangent.x-a.tangent.y,wl=std::hypot(wx,wy);wx/=wl;wy/=wl;
     double sine=std::abs(tx*wy-ty*wx);if(sine<.1)throw std::runtime_error("grazing crossing requires hydrology span policy");
     out<<"{\"stable_id\":\""<<a.stable_id<<"\",\"hydrology_edge\":\""<<a.edge_id<<"\",\"world_xy\":["<<64*(a.point.x+a.point.y)<<','<<64*(a.point.x-a.point.y)<<"],\"route_tangent\":["<<tx<<','<<ty<<"],\"water_tangent\":["<<wx<<','<<wy<<"],\"span_width\":"<<a.width*64*std::sqrt(2.)/sine<<",\"deck_height\":null}";
    }
    out<<"]}";
   }
  }
  out<<"]}\n";return out?0:3;
 }catch(std::exception const&e){fprintf(stderr,"Q5 crossing adapter: %s\n",e.what());return 1;}
}
