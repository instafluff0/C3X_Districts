// Preserve the old relief source sampling and flat/cache query policy exactly.
#include "relief.h"
#include <cassert>
#include <iostream>
using namespace c3x_renderer;
using namespace c3x_renderer::fidelity;
float original_normalized_field(std::vector<std::uint8_t> const & pixels,
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


template<class Assets>float original_source(Assets const&terrain_textures,bool fidelity_profile,
    int kind,unsigned variant,int channel,float u,float v){
                if(fidelity_profile && (kind==5 || kind==6))return 0.f; // replaced exact natural providers

                auto const & asset = terrain_textures[kind];
                if (kind == 6) {
                    auto const & pixels = channel == 0 ? asset.relief_height_variants[variant] : asset.relief_blend_variants[variant];
                    return original_normalized_field(pixels, asset.relief_variant_widths[variant],
                        asset.relief_variant_heights[variant], channel == 0 ? asset.relief_height_minimum[variant] : asset.relief_blend_minimum[variant],
                        channel == 0 ? asset.relief_height_maximum[variant] : asset.relief_blend_maximum[variant], u, v);
                }
                return original_normalized_field(channel == 0 ? asset.height_pixels : asset.blend_pixels,
                    asset.height_width, asset.height_height,
                    channel == 0 ? asset.height_minimum : asset.blend_minimum,
                    channel == 0 ? asset.height_maximum : asset.blend_maximum, u, v);
            }
struct Report {std::vector<double> values,events;std::size_t hits=0,misses=0,heights=0;};
template<class Surface>void exercise(Surface&surface,Report&out) {
    for(float y:{-.005f,.5f,1.005f,1.1f})for(float x:{-.005f,.5f,1.005f}) {
        for(unsigned repeat=0;repeat<2;repeat++) {
            auto value=surface.sample(x,y);
            out.values.insert(out.values.end(),{value.height,value.authored_height,value.authored_blend});
            for(auto owner:value.owner)out.values.push_back(owner);
            out.values.push_back(surface.height(x,y));
        }
    }
}
template<class Query>struct OriginalSurface {
    Query&query;profile_v2::FlatGroundRegion&flat;
    profile_v2::ExactPointCache<profile_v2::GroundSample>&scratch;std::size_t&counter;
    profile_v2::GroundSample sample(float u,float v) {
        if(flat.contains(u,v))return profile_v2::GroundSample{};
        return scratch.get(u,v,[&](){return query.sample(u,v);});
    }
    float height(float u,float v) {
        if(flat.contains(u,v))return 0.f;
        ++counter;return query.sample(u,v,false).height;
    }
};
template<bool Shared>Report run(unsigned scene,bool fidelity,std::array<ReliefFields,11>const&assets) {
    Report out;profile_v2::ExactPointCache<profile_v2::GroundSample> scratch;
    auto lookup=[&](int c,int r) {
        out.events.insert(out.events.end(),{1.,double(c),double(r)});
        int real=2;
        if(scene==1)real=c==0 && r==0?10:c==1?6:c==-1?5:0;
        if(scene==2)real=c<0?12:5;
        if(scene==3 && c==0 && r==0)return profile_v2::Tile{};
        return profile_v2::Tile{real==5 || real==6 || real==10?2:real,real,true};
    };
    auto source=[&](int kind,unsigned variant,int channel,float u,float v) {
        out.events.insert(out.events.end(),{2.,double(kind),double(variant),double(channel),u,v});
        if constexpr(Shared)return relief_source(assets,fidelity,kind,variant,channel,u,v);
        else return original_source(assets,fidelity,kind,variant,channel,u,v);
    };
    auto shore=[&](float u,float v) {
        out.events.insert(out.events.end(),{3.,u,v});
        return profile_v2::ShoreSample{scene==2?double(u)*.4:3.,.06,.9,0};
    };
    auto river=[&](int c,int r,float u,float v) {
        out.events.insert(out.events.end(),{4.,double(c),double(r),u,v});return scene==1?8.f:1000.f;
    };
    auto dune=[&](float u,float v){out.events.insert(out.events.end(),{5.,u,v});return 2.f+u+v;};
    auto activity=[&](int c,int r){out.events.insert(out.events.end(),{6.,double(c),double(r)});return 1.f;};
    profile_v2::World world{64,64,true,true};
    for(unsigned owner=0;owner<2;owner++) {
        scratch.clear();double center=owner?0:shore(.5f,.5f).distance;
        if constexpr(Shared) {
            ReliefSurface surface(world,0,0,center,lookup,source,shore,river,dune,activity,scratch,out.heights);
            exercise(surface,out);
        } else {
            profile_v2::ReliefQuery query(world,lookup,source,shore,river,dune,activity);
            profile_v2::FlatGroundRegion flat(0,0,center,lookup);
            OriginalSurface<decltype(query)> surface{query,flat,scratch,out.heights};exercise(surface,out);
        }
    }
    out.hits=scratch.hits;out.misses=scratch.misses;return out;
}
int main() {
    std::array<ReliefFields,11> assets;
    for(unsigned kind=0;kind<assets.size();kind++) {
        auto&a=assets[kind];a.height_width=a.height_height=4;
        a.height_minimum=.1f;a.height_maximum=.9f;a.blend_minimum=.05f;a.blend_maximum=.8f;
        for(unsigned i=0;i<16;i++){a.height_pixels.push_back(std::uint8_t((i*17+kind*7)%256));a.blend_pixels.push_back(std::uint8_t(255-i*11));}
        for(unsigned v=0;v<5;v++) {
            a.relief_variant_widths[v]=a.relief_variant_heights[v]=4;
            a.relief_height_variants[v]=a.height_pixels;a.relief_height_variants[v][v]=std::uint8_t(v*11);
            a.relief_blend_variants[v]=a.blend_pixels;a.relief_blend_variants[v][v]=std::uint8_t(v*37);
            a.relief_height_minimum[v]=.03f*v;a.relief_height_maximum[v]=.4f+.1f*v;
            a.relief_blend_minimum[v]=.01f*v;a.relief_blend_maximum[v]=.2f+.15f*v;
        }
    }
    unsigned fields=0,scopes=0;std::size_t observations=0;
    for(bool fidelity:{false,true})for(int kind:{5,6,10})for(unsigned variant=0;variant<5;variant++)
    for(int channel:{0,1})for(float v:{-.25f,0.f,.125f,1.f})for(float u:{-1.f,-.25f,0.f,.125f,.875f,1.f}) {
        assert(original_source(assets,fidelity,kind,variant,channel,u,v)==relief_source(assets,fidelity,kind,variant,channel,u,v));fields++;
    }
    for(bool fidelity:{false,true})for(unsigned scene=0;scene<4;scene++) {
        auto a=run<false>(scene,fidelity,assets),b=run<true>(scene,fidelity,assets);
        assert(a.values==b.values && a.events==b.events);
        assert(a.hits==b.hits && a.misses==b.misses && a.heights==b.heights);
        assert(a.hits>0 && a.misses>0 && a.heights>0);observations+=a.events.size();scopes+=2;
    }
    assert(sample_normalized_field({},4,4,0,1,0,0)==0);
    assert(sample_normalized_field({255},0,1,0,1,0,0)==0);
    assert(sample_normalized_field({255},1,0,0,1,0,0)==0);
    assert(sample_normalized_field({128},1,1,.5f,.5f,0,0)==original_normalized_field({128},1,1,.5f,.5f,0,0));
    std::cout<<"PASS shared relief: "<<fields<<" exact field samples, "<<scopes<<" scopes and "<<observations<<" ordered query observations; flat/cache/height parity\n";
}
