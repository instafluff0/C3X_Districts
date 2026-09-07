// Export a conservative river-water footprint from the rendered terrain mesh.
// The packet profile is a Lab adapter; output polygons are generic tile XY.
#include "../contracts/packet_v1.h"
#include <cstring>

struct Point { float x,y,d; };
float value(std::vector<uint8_t> const& b,size_t offset){float v;std::memcpy(&v,b.data()+offset,4);return v;}
int main(int argc,char**argv){
 try {
  if(argc!=6)throw std::runtime_error("usage: river_city_exclusion packet anchor_x anchor_y threshold output");
  auto p=labv2::read_packet(argv[1]);float x=std::stof(argv[2])+.5f,y=std::stof(argv[3])+.5f,threshold=std::stof(argv[4]);
  if(!std::isfinite(threshold)||threshold<0||threshold>16)throw std::runtime_error("invalid river threshold");
  FILE* out=labv2::open_path(argv[5],"wb");if(!out)throw std::runtime_error("cannot write river footprint");
  std::fprintf(out,"{\"threshold_pixels\":%.9g,\"polygons\":[",threshold);size_t count=0,matched=0,river_triangles=0;
  for(auto const& draw:p.draws){
   if(draw.feature || draw.attributes.size()<17 || draw.attributes[16].offset!=116 || draw.attributes[16].components!=4 ||
      draw.attributes[6].offset!=48 || draw.attributes[14].offset!=96 || draw.stride<132)continue;
   matched++;
   auto const& vertices=p.buffers.at(draw.vertex_buffer);
   for(unsigned i=0;i+2<draw.count;i+=3){
    size_t at=size_t(i)*draw.stride;if(value(vertices,at+48)!=9)continue;river_triangles++;
    std::vector<Point> polygon;
    for(unsigned j=0;j<3;++j){auto offset=at+j*draw.stride;
     if(value(vertices,offset+48)!=9)throw std::runtime_error("mixed river surface triangle");
     polygon.push_back({value(vertices,offset+116)-x,y-value(vertices,offset+120),value(vertices,offset+96)});
    }
    std::vector<Point> clipped;
    for(unsigned j=0;j<polygon.size();++j){auto a=polygon[j],b=polygon[(j+1)%polygon.size()];
     if(a.d<=threshold)clipped.push_back(a);
     if((a.d<=threshold)!=(b.d<=threshold)){float t=(threshold-a.d)/(b.d-a.d);clipped.push_back({a.x+(b.x-a.x)*t,a.y+(b.y-a.y)*t,threshold});}
    }
    if(clipped.size()<3)continue;
    float lo_x=1000,lo_y=1000,hi_x=-1000,hi_y=-1000;
    for(auto a:clipped){lo_x=std::min(lo_x,a.x);lo_y=std::min(lo_y,a.y);hi_x=std::max(hi_x,a.x);hi_y=std::max(hi_y,a.y);}
    if(lo_x>1 || lo_y>1 || hi_x<-1 || hi_y<-1)continue;
    if(count++)std::fputc(',',out);std::fputc('[',out);
    for(unsigned j=0;j<clipped.size();++j){if(j)std::fputc(',',out);std::fprintf(out,"[%.9g,%.9g]",clipped[j].x,clipped[j].y);}
    std::fputc(']',out);
   }
  }
  std::fprintf(out,"],\"polygon_count\":%zu,\"matched_terrain_draws\":%zu,\"river_triangles\":%zu}\n",count,matched,river_triangles);fclose(out);
  if(!matched)throw std::runtime_error("unsupported terrain vertex profile");
  std::printf("river footprint polygons=%zu\n",count);return 0;
 }catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
