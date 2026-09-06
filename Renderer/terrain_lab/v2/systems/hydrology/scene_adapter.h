#pragma once
// Opt-in HydrologyHooksV1 adapter. A null shared hook retains frozen v1.
#include "field.h"
namespace q3_scene {
inline hydro::Field field;
// Current verified dataset has horizontal wrap only. A generic caller can
// assign field.wraps from its dataset policy before the final field.build().
inline void initialize(char const* csv){field=hydro::Field(csv);field.wraps=true;field.build();}
inline void initialize_no_wrap(char const* csv){field=hydro::Field(csv);field.wraps=false;field.build();}
inline void shore_sample(float corner_x,float corner_y,float out[4]){
 auto sample=field.sample({double(corner_x)-.5,double(corner_y)-.5});
 out[0]=float(sample.shore_distance);out[1]=float(sample.beach_width);
 out[2]=float(sample.rocky);out[3]=float(sample.depth);
}
// Proposed additive source-scene river channel, not bound by the current Q0 ABI.
inline void river_sample(float corner_x,float corner_y,float out[4]){
 auto sample=field.sample({double(corner_x)-.5,double(corner_y)-.5});
 out[0]=float(sample.river_distance-sample.river_width*.5);
 out[1]=float(sample.river_width);out[2]=float(sample.height);
 out[3]=float(hydro::smooth(.025,.80,sample.river_distance-sample.river_width*.5));
}
inline float signed_shore_distance(float corner_x,float corner_y){
 // Frozen convention uses tile-corner lattice and water-positive normalized
 // coverage, whereas Q3's reusable kernel uses center lattice and land-positive
 // true distance. This adapter is explicit, not an implicit unit/sign change.
 hydro::P p{double(corner_x)-.5,double(corner_y)-.5};double distance=1e6;
 // Frozen shadow construction calls this often; avoid unrelated river/material
 // sampling while returning the identical zero contour and signed distance.
 for(auto const& e:field.coast)distance=std::min(distance,field.segment_distance(p,e.a,e.b));
 double water_positive=field.signed_coverage(p)>=0?-distance:distance;
 return float(std::clamp(water_positive/.65,-1.,1.));
}
}
