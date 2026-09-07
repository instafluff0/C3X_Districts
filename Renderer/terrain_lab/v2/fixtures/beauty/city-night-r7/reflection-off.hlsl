#define VSMain Q3OriginalVSMain
#define VSFeature Q3OriginalVSFeature
#define PSMain Q3OriginalPSMain
#define Q8_CITY_FEATURE_ENTRY Q3OriginalPSFeature
#include "city-off.hlsl"
#undef VSMain
#undef VSFeature
#undef PSMain
#define Q3_REFLECTION_HEIGHT_NDC 0.262400000000
#include "../../../shaders/hydrology/planar_reflection_pass.hlsl"
