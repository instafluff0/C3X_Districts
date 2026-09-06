#pragma once
#include <cmath>
#include <stdexcept>
namespace placement_witness {
inline void initialize(const char* terrain,const char* fixture) {if(!terrain||!fixture)throw std::runtime_error("missing placement context");}
inline bool accept(const char* group,const char* asset,unsigned seed,unsigned instance,const float* xyz,unsigned count) {
 if(!group||!asset||count<3)throw std::runtime_error("missing source placement geometry");
 for(unsigned i=0;i<count*3;i++)if(!std::isfinite(xyz[i]))throw std::runtime_error("nonfinite transformed source placement");
 return true;
}
}
