#include "../../systems/hydrology/scene_adapter.h"
#include <cassert>
#include <iostream>
int main(){
 const std::string root="Renderer/terrain_lab/v2/fixtures/hydrology/";
 hydro::Field coast(root+"coast.csv"),river(root+"rivers.csv");
 hydro::Field islands(root+"islands.csv");assert(islands.signed_coverage({2.5,2.5})<0);
 // Regression for frozen isolated-water max(result, 0) outside basin support.
 auto inland=coast;for(auto& kv:inland.tiles){kv.second.base=2;kv.second.real=2;kv.second.river=0;}
 inland.tiles[{4,1}].base=11;inland.tiles[{4,1}].real=11;inland.build();
 assert(inland.sample({3,1}).shore_distance>.35);
 assert(inland.sample({3,1}).height>.07);
 assert(std::abs(inland.sample({2.5-1e-5,1}).shore_distance-inland.sample({2.5+1e-5,1}).shore_distance)<1e-3);
 // Shared callback converts corner/center lattices and water/land signs exactly.
 q3_scene::field=inland;
 assert(q3_scene::signed_shore_distance(3.5f,1.5f)<-.5f);
 assert(q3_scene::signed_shore_distance(4.5f,1.5f)>0.f);
 for(int y=0;y<10;y++)for(int x=0;x<10;x++){
  hydro::P p{x*.5,y*.5};
  auto sample=inland.sample(p);
  float data[4];q3_scene::shore_sample(float(p.x+.5),float(p.y+.5),data);
  assert(std::abs(data[0]-sample.shore_distance)<1e-6);
  assert(std::abs(data[1]-sample.beach_width)<1e-6);
  assert(std::abs(data[2]-sample.rocky)<1e-6);
  assert(std::abs(data[3]-sample.depth)<1e-6);
  double expected=std::clamp(-sample.shore_distance/.65,-1.,1.);
  assert(std::abs(q3_scene::signed_shore_distance(float(p.x+.5),float(p.y+.5))-expected)<1e-6);
 }
 // All water types qualify, hill-only receiver classification leaves lowland sand.
 auto low=coast.sample({3.45,0}),hill=coast.sample({3.45,4});
 assert(hill.rocky>.95 && hill.beach_width<.015);assert(low.rocky<.05 && low.beach_width>.12);
 // Rocky/sandy class is sampled continuously at the contour foot, not quantized
 // to a marching segment owner. Infinitesimal displacements cannot jump a band.
 for(int i=0;i<100;i++){double y=2+i*.015;auto a=coast.sample({3.45,y}),b=coast.sample({3.45,y+1e-5});assert(std::abs(a.rocky-b.rocky)<.001);}
 // Depth along one normal ray is strictly nondecreasing into water.
 double previous=0;for(int i=0;i<100;i++){auto s=coast.sample({3.6+i*.015,0});assert(s.depth>=previous-1e-6);previous=s.depth;}
 // Remove every reciprocal flag. Unique geometry and actual crossings stay exact.
 auto single=river;for(auto&kv:single.tiles)kv.second.river&=~(32u|128u);single.build();
 assert(single.rivers.size()==river.rivers.size());
 for(size_t i=0;i<single.rivers.size();i++){assert(single.rivers[i].id==river.rivers[i].id);assert(hydro::length(single.rivers[i].points[12]-river.rivers[i].points[12])<1e-12);}
 auto anchors=river.crossings({1,1},{2,1});assert(anchors.size()==1);assert(std::abs(anchors[0].width-.095)<1e-9);
 auto reverse=river.crossings({2,1},{1,1});assert(reverse.size()==1);assert(hydro::length(anchors[0].point-reverse[0].point)<1e-10);assert(anchors[0].stable_id==reverse[0].stable_id);
 // Extents matter: an origin clear of the channel can have an intrusive overhang.
 assert(!river.intersects_footprint({{.1,.1},{.3,.1},{.3,.3},{.1,.3}}));
 assert(river.intersects_footprint({{1.10,.8},{1.45,.8},{1.45,1.2},{1.10,1.2}}));
 assert(river.intersects_footprint({{1.2,2.3},{1.8,2.3},{1.8,2.7},{1.2,2.7}}));
 for(auto const& c:river.exclusions()){assert(c.clearance_radius>c.bank_radius);assert(c.bank_radius>c.water_radius);}
 // Neighbor missing is a hard failure, not implicit water on a crop edge.
 auto missing=coast;missing.tiles.clear();bool rejected=false;try{missing.occupancy({1,1});}catch(std::runtime_error const&){rejected=true;}assert(rejected);
 // World-periodic noise and IDs survive a raw-X wrap representation change.
 river.wraps=true;river.build();auto wrapped=river;wrapped.origin_x+=wrapped.map_width;wrapped.build();
 assert(std::abs(wrapped.noise({2.23,1.71},123)-river.noise({2.23,1.71},123))<1e-12);
 assert(wrapped.rivers.size()==river.rivers.size());for(size_t i=0;i<river.rivers.size();i++)assert(wrapped.rivers[i].id==river.rivers[i].id);
 // Coast is evaluated outside the visible crop and has no closed crop-edge pond.
 assert(coast.signed_coverage({-.5,0})>0);assert(coast.signed_coverage({5.5,0})<0);
 // Junction shares exact endpoints, and channel valleys stay below water.
 for(auto const&r:river.rivers){assert(hydro::length(r.points.front()-r.a)<1e-12);assert(hydro::length(r.points.back()-r.b)<1e-12);assert(river.sample(r.points[12]).height<0);}
 std::cout<<"PASS: rocky classification, monotone depth, reciprocal deduplication, exact crossings, crop halo, wrap, carving\n";
}
