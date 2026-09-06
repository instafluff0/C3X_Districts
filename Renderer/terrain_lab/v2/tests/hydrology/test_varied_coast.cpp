#include "../../systems/hydrology/field.h"
#include <cassert>
#include <iostream>
int main(){
 const std::string base="Renderer/terrain_lab/v2/fixtures/beauty/";
 unsigned changed=0;
 for(int profile:{1,2})for(auto name:{"gameplay-100-v1/coastal/terrain.csv","gameplay-100-v1/inland/terrain.csv",
                "gameplay-100-v1/wilderness/terrain.csv","coast-pass-foundation/longcoast/terrain.csv"}){
  hydro::Field f;f.load(base+name);f.wraps=true;f.shoreline_profile=profile;
  auto old=f;old.shoreline_profile=0;
  for(int y=0;y<f.rows;y++)for(int x=0;x<f.cols;x++){
   bool wet=hydro::water(f.tile(x,y).base);
   // A protected radius retains authoritative center/domain readability.
   for(auto d:{hydro::P{0,0},hydro::P{.15,0},hydro::P{-.15,0},hydro::P{0,.15},hydro::P{0,-.15}})
    assert((f.signed_coverage(hydro::P{double(x),double(y)}+d)<0)==wet);
   // Adjacent navigable cells retain their center-to-center connection.
   for(auto d:{hydro::P{1,0},hydro::P{0,1}}){
    if(hydro::water(f.tile(x+int(d.x),y+int(d.y)).base)!=wet)continue;
    for(int i=0;i<=16;i++)assert((f.signed_coverage(hydro::P{double(x),double(y)}+d*(i/16.))<0)==wet);
   }
   for(int j=0;j<8;j++)for(int i=0;i<8;i++){
    hydro::P p{x+i/8.,y+j/8.};changed+=(f.signed_coverage(p)<0)!=(old.signed_coverage(p)<0);
   }
  }
  auto wrap=f;wrap.origin_x+=f.map_width;
  auto crop=f;crop.origin_x+=3;crop.origin_y+=1;crop.tiles.clear();
  for(auto const&[key,t]:f.tiles)crop.tiles[{key.first-2,key.second-1}]=t;
  for(int i=0;i<100;i++){
   hydro::P p{2.123+i*.037,2.34+i*.017};
   assert(std::abs(wrap.signed_coverage(p)-f.signed_coverage(p))<1e-11);
   assert(std::abs(crop.signed_coverage(p-hydro::P{2,1})-f.signed_coverage(p))<1e-11);
  }
 }
 assert(changed>100);
 std::cout<<"PASS varied coast: preserved centers/connections, crop/wrap continuity; "<<changed<<" changed occupancy probes\n";
}
