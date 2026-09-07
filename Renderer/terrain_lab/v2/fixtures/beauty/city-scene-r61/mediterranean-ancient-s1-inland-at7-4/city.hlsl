#define Q3_NATURAL_WATER 1
#define PSFeature Q8LegacyPSFeature
#include "terrain-auxiliary.hlsl"
#undef PSFeature
#define Q8_CITY_AUXILIARY_AO 1
#define Q8_CITY_AO_STRENGTH 1.000000000
#define Q8_CITY_EXTRA_MATERIALS 1
#define Q8_CITY_SOURCE_SURFACE 1
#define Q8_CITY_SOURCE_SPECULAR 1
#define Q8_CITY_VIEW_DIRECTION normalize(float3(1,1,0.790569494147))
#define Q8_CITY_CHANNELS 0
#define Q8_CITY_SURFACE_DETAIL 0
#define Q8_CITY_WORLD_Z_TO_SOURCE 0.648266978876
#define Q8_CITY_SEPARATE_EMISSION 1
#define Q8_CITY_EMISSIVE_GAIN 8.0
#include "../../../../shaders/objects/city_scene_material.hlsl"
