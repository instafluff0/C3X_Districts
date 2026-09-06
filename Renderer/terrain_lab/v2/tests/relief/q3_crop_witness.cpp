#include "../../systems/hydrology/field.h"
#include <cstdio>
int main(){
 hydro::Field a("Renderer/terrain_lab/v2/fixtures/relief/real-primary/terrain.csv");
 hydro::Field b("Renderer/terrain_lab/v2/fixtures/relief/real-holdout/terrain.csv");
 double worst=0;
 // Both positions refer to raw BIQ (18,46), plus the same fractional offset.
 for(int i=0;i<10;i++){
  hydro::P p{3.5+i*.05,.1},q{p.x-4,p.y};
  double diff=std::abs(a.noise(p,127)-b.noise(q,127));worst=std::max(worst,diff);
 }
 printf("same_raw_coordinate_noise_max_delta=%.9f\n",worst);
 return worst<1e-9?0:1;
}
