#define Q3_NATURAL_WATER 1
#define PSFeature Q8LegacyPSFeature
#include "../../river-corridor-r3/freshcanopy/combined.hlsl"
#undef PSFeature
#define Q8_CITY_CHANNELS 0
#define Q8_CITY_SURFACE_DETAIL 0
#define Q8_CITY_WORLD_Z_TO_SOURCE 0.648266978876
#define Q8_CITY_SEPARATE_EMISSION 1
#define Q8_CITY_EMISSIVE_GAIN 8.0
#include "../../../../shaders/objects/city_scene_material.hlsl"
