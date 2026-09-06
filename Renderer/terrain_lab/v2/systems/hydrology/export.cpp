#include "field.h"
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("usage: export tile.csv wrap_x");hydro::Field f(argv[1]);f.wraps=std::stoi(argv[2])!=0;f.build();
 std::cout<<std::setprecision(12)<<"{\"schema\":\"c3x.hydrology_candidate.v1\",\"coordinate_space\":\"local_tile_centers\",\"positive_shore\":\"land\",\"animation\":false,\"shore_segments\":[";
 bool comma=false;for(auto const&s:f.coast){if(comma)std::cout<<',';comma=true;std::cout<<"{\"a\":["<<s.a.x<<','<<s.a.y<<"],\"b\":["<<s.b.x<<','<<s.b.y<<"],\"rocky\":"<<s.rocky<<",\"rocky_a\":"<<f.shore_rockiness(s.a)<<",\"rocky_b\":"<<f.shore_rockiness(s.b)<<'}';}
 std::cout<<"],\"river_edges\":[";comma=false;for(auto const&r:f.rivers){if(comma)std::cout<<',';comma=true;std::cout<<"{\"id\":\""<<r.id<<"\",\"width\":"<<r.width<<",\"points\":[";bool sep=false;for(auto const&p:r.points){if(sep)std::cout<<',';sep=true;std::cout<<'['<<p.x<<','<<p.y<<']';}std::cout<<"]}";}
 std::cout<<"],\"exclusion_capsules\":[";comma=false;
 for(auto const&c:f.exclusions()){if(comma)std::cout<<',';comma=true;std::cout<<"{\"edge_id\":\""<<c.edge_id<<"\",\"a\":["<<c.a.x<<','<<c.a.y<<"],\"b\":["<<c.b.x<<','<<c.b.y<<"],\"water_radius\":"<<c.water_radius<<",\"bank_radius\":"<<c.bank_radius<<",\"clearance_radius\":"<<c.clearance_radius<<'}';}
 std::cout<<"],\"crossing_witnesses\":[";comma=false;
 for(int y=0;y<f.rows;y++)for(int x=0;x<f.cols;x++)for(hydro::P d:std::vector<hydro::P>{{1,0},{0,1}}){hydro::P a{double(x),double(y)},b=a+d;
  for(auto const&c:f.crossings(a,b)){if(comma)std::cout<<',';comma=true;std::cout<<"{\"route\":[["<<a.x<<','<<a.y<<"],["<<b.x<<','<<b.y<<"]],\"edge_id\":\""<<c.edge_id<<"\",\"position\":["<<c.point.x<<','<<c.point.y<<"],\"tangent\":["<<c.tangent.x<<','<<c.tangent.y<<"],\"width\":"<<c.width<<'}';}}
 std::cout<<"]}\n";
 }catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}}
