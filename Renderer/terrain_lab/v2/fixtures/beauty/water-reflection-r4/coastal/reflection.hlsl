#define VSMain Q3OriginalVSMain
#define VSFeature Q3OriginalVSFeature
#define PSMain Q3OriginalPSMain
#define PSFeature Q3OriginalPSFeature
#include "../../river-corridor-r3/coastal/combined.hlsl"
#undef VSMain
#undef VSFeature
#undef PSMain
#undef PSFeature
#define Q3_REFLECTION_HEIGHT_NDC 0.262400000000
#include "../../../../shaders/hydrology/planar_reflection_pass.hlsl"
