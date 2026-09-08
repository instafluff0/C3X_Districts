#pragma once
// Production relief inputs and query-cache policy, independent of GPU resources.
#include "../../../native/render_core/relief_query.h"
#include "../../../native/render_core/exact_point_cache.h"
namespace c3x_renderer { namespace fidelity {
struct ReliefFields {
    std::vector<std::uint8_t> height_pixels;
    std::vector<std::uint8_t> blend_pixels;
    std::array<std::vector<std::uint8_t>, 5> relief_height_variants;
    std::array<std::vector<std::uint8_t>, 5> relief_blend_variants;
    std::array<std::uint32_t, 5> relief_variant_widths = {};
    std::array<std::uint32_t, 5> relief_variant_heights = {};
    std::array<float, 5> relief_height_minimum = {};
    std::array<float, 5> relief_height_maximum = {};
    std::array<float, 5> relief_blend_minimum = {};
    std::array<float, 5> relief_blend_maximum = {};
    std::uint32_t height_width = 0;
    std::uint32_t height_height = 0;
    float height_minimum = 0.0f;
    float height_maximum = 1.0f;
    float blend_minimum = 0.0f;
    float blend_maximum = 1.0f;
};
inline float sample_normalized_field(std::vector<std::uint8_t> const & pixels,
                                         std::uint32_t width, std::uint32_t height,
                                         float minimum, float maximum,
                                         float u, float v) {
        // This is the Lab HeightField::sample implementation. Authored relief
        // fields are normalized to their own observed range and sampled with
        // wrapped bilinear coordinates, including at the source border.
        if (pixels.empty() || width == 0 || height == 0)
            return 0.0f;
        u -= std::floor(u);
        v -= std::floor(v);
        float px = u * static_cast<float>(width);
        float py = v * static_cast<float>(height);
        std::uint32_t x0 = static_cast<std::uint32_t>(std::floor(px)) % width;
        std::uint32_t y0 = static_cast<std::uint32_t>(std::floor(py)) % height;
        std::uint32_t x1 = (x0 + 1) % width;
        std::uint32_t y1 = (y0 + 1) % height;
        float tx = px - std::floor(px);
        float ty = py - std::floor(py);
        auto value = [&pixels, width, minimum, maximum](std::uint32_t x,
                                                        std::uint32_t y) {
            float raw = static_cast<float>(
                pixels[static_cast<std::size_t>(y) * width + x]) / 255.0f;
            return (raw - minimum) / std::max(0.0001f, maximum - minimum);
        };
        float top = value(x0, y0) * (1.0f - tx) + value(x1, y0) * tx;
        float bottom = value(x0, y1) * (1.0f - tx) + value(x1, y1) * tx;
        return top * (1.0f - ty) + bottom * ty;
    }


template<class Assets>
float relief_source(Assets const& terrain_textures,bool fidelity_profile,
                    int kind,unsigned variant,int channel,float u,float v) {
                if(fidelity_profile && (kind==5 || kind==6))return 0.f; // replaced exact natural providers

                auto const & asset = terrain_textures[kind];
                if (kind == 6) {
                    auto const & pixels = channel == 0 ? asset.relief_height_variants[variant] : asset.relief_blend_variants[variant];
                    return sample_normalized_field(pixels, asset.relief_variant_widths[variant],
                        asset.relief_variant_heights[variant], channel == 0 ? asset.relief_height_minimum[variant] : asset.relief_blend_minimum[variant],
                        channel == 0 ? asset.relief_height_maximum[variant] : asset.relief_blend_maximum[variant], u, v);
                }
                return sample_normalized_field(channel == 0 ? asset.height_pixels : asset.blend_pixels,
                    asset.height_width, asset.height_height,
                    channel == 0 ? asset.height_minimum : asset.blend_minimum,
                    channel == 0 ? asset.height_maximum : asset.blend_maximum, u, v);
            }

// Scratch is owned and cleared per tile by the caller, including reused tiles.
// Height-only normal samples intentionally bypass the ground/material cache.
template<class Lookup,class Source,class Shore,class River,class Dune,class Activity>
class ReliefSurface {
    render_core::ReliefQuery<Lookup,Source,Shore,River,Dune,Activity> query;
    render_core::FlatGroundRegion flat;
    render_core::ExactPointCache<render_core::GroundSample>& scratch;
    std::size_t& height_queries;
public:
    ReliefSurface(render_core::World world,int c,int r,double center_distance,
                  Lookup lookup,Source source,Shore shore,River river,Dune dune,Activity activity,
                  render_core::ExactPointCache<render_core::GroundSample>& samples,std::size_t& counter)
        :query(world,lookup,source,shore,river,dune,activity),
         flat(c,r,center_distance,lookup),
         scratch(samples),height_queries(counter) {}
    render_core::GroundSample sample(float u,float v) {
        if(flat.contains(u,v))return render_core::GroundSample{};
        return scratch.get(u,v,[&](){return query.sample(u,v);});
    }
    float height(float u,float v) {
        if(flat.contains(u,v))return 0.f;
        ++height_queries;
        return query.sample(u,v,false).height;
    }
};
} }
