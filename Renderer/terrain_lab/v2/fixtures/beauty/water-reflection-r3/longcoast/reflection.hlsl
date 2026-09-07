#define VSMain Q3OriginalVSMain
#define VSFeature Q3OriginalVSFeature
#define PSMain Q3OriginalPSMain
#define PSFeature Q3OriginalPSFeature
#include "../../shadow-receiver-r1/longcoast/combined.hlsl"
#undef VSMain
#undef VSFeature
#undef PSMain
#undef PSFeature
#define Q3_REFLECTION_HEIGHT_NDC 0.236396396396
#include "../../../../shaders/hydrology/planar_reflection_pass.hlsl"
