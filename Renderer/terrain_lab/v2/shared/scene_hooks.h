#pragma once
// Explicit opt-in adapters. Default null callbacks preserve frozen construction.
namespace labv2 {
struct TerrainHooksV1 {
  void (*initialize)(const char* terrain_csv) = nullptr;
  // Continuous local lattice column+u,row+(1-v); grass/plains/desert/marsh/tundra.
  void (*material_uv)(float x,float y,float uv_scale,float uv[2]) = nullptr;
  void (*material_weights)(float x,float y,float weights[5]) = nullptr;
};
inline TerrainHooksV1 terrain_hooks;
struct HydrologyHooksV1 {
  void (*initialize)(const char* terrain_csv)=nullptr;
  void (*shore_sample)(float x,float y,float data[4])=nullptr; // positive-land distance, width, rocky fraction, depth
  float (*signed_shore_distance)(float x,float y)=nullptr; // water-positive [-1,1]
};
inline HydrologyHooksV1 hydrology_hooks;
struct PlacementHooksV1 {
  void (*initialize)(const char* terrain_csv,const char* fixture_json)=nullptr;
  // Actual transformed source vertices in civ3_raw_delta_pixels_v1; xyz triplets.
  bool (*accept_vegetation)(const char* group,const char* asset_id,unsigned seed,unsigned instance,const float* xyz,unsigned vertex_count)=nullptr;
};
inline PlacementHooksV1 placement_hooks;
}
