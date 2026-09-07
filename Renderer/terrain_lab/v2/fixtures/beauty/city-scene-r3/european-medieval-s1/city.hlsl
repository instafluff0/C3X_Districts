#define Q3_NATURAL_WATER 1
#define PSFeature Q8LegacyPSFeature
#include "../../river-corridor-r3/coastal/combined.hlsl"
#undef PSFeature
#define Q8_CITY_CHANNELS 0
#define Q8_CITY_EMISSIVE_GAIN 1.45
#include "../../../../shaders/objects/city_scene_material.hlsl"
