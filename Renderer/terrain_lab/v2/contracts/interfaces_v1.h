#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>
namespace labv2 {
constexpr unsigned interface_version = 1;
using Float3 = std::array<float, 3>;
struct SurfaceSample {
  Float3 position{}, normal{0, 0, 1};
  std::array<float, 5> material_weights{};
  uint32_t terrain_owner = 0;
  float shore_distance = 0, water_depth = 0, wetness = 0;
};
enum class Layer : uint32_t {
  opaque,
  decal,
  route,
  object,
  cutout,
  shadow,
  water,
  transparent,
  emissive,
  effect
};
enum class DepthMode : uint32_t { read_write, read_only, disabled };
struct Renderable {
  std::string stable_id, system_id, mesh_id, material_id;
  std::array<float, 16> transform{};
  Float3 bounds_min{}, bounds_max{};
  Layer layer = Layer::opaque;
  DepthMode depth = DepthMode::read_write;
  bool caster = false, receiver = true, visible = true;
  uint64_t presentation_ticks = 0;
};
struct RouteNode {
  uint64_t stable_id = 0;
  Float3 position{};
};
struct CrossingAnchor {
  uint64_t stable_id = 0, hydrology_edge = 0;
  Float3 position{}, tangent{};
  float water_width = 0;
};
struct RouteEdge {
  uint64_t stable_id = 0, node_a = 0, node_b = 0;
  std::vector<SurfaceSample> grade_samples;
  std::vector<CrossingAnchor> crossings;
  uint32_t style = 0;
};
struct RouteGraph {
  std::vector<RouteNode> nodes;
  std::vector<RouteEdge> edges;
};
struct FrameEnvironment {
  float hour = 12, transition = 0;
  uint32_t season = 0;
  uint64_t presentation_ticks = 0;
  Float3 sun_direction{}, sun_color{}, moon_direction{}, moon_color{},
      ambient{};
  float sun_intensity = 0, moon_intensity = 0, exposure = 1,
        shadow_strength = 0, water_fresnel = 0, water_specular = 0,
        emissive_scale = 0;
};
struct PresentationProfile {
  std::string id;
  std::array<float, 2> tile_basis{128, 64};
  std::array<float, 2> zooms{1, .5f};
  std::array<float, 2> footprint_band{}, height_band{};
  std::vector<float> allowed_yaws;
  float grounding_offset = 0;
};
struct RenderPass {
  std::string id, owner;
  Layer layer;
  DepthMode depth;
  std::vector<std::string> reads, writes, after;
};
struct RenderGraph {
  std::vector<RenderPass> passes;
};
// Settings are data, never implicit backend policy. Baseline is v1.
struct SamplingSettings {
  uint32_t anisotropy = 8, samples = 1, render_scale = 1;
  float mip_bias = 0;
  std::string postprocess = "box";
  std::vector<std::array<float, 2>> camera_offsets{{0, 0}};
};
} // namespace labv2
