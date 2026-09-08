#include "patterns.h"
#include <cassert>
#include <cstring>
#include <iostream>
namespace reference {
float smoothstep(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    return value * value * (3.0f - 2.0f * value);
}

std::uint32_t feature_hash(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}
float dune_height(float world_x, float world_y, float desert_weight) {
    if (desert_weight <= 0.0f)
        return 0.0f;
    constexpr float angle = 0.300001f;
    constexpr float dune_width = 4.0f;
    constexpr float dune_noise = 0.6f;
    float along = world_x * std::cos(angle) + world_y * std::sin(angle);
    float across = -world_x * std::sin(angle) + world_y * std::cos(angle);
    float broad_bend = std::sin(across * 1.05f + 0.8f) * dune_noise * 3.65f +
        std::sin(across * 2.35f - 0.6f) * dune_noise * 0.90f;
    float phase_noise = broad_bend +
        std::sin(across * 6.7f + along * 0.43f) * dune_noise * 0.32f +
        std::sin(across * 11.9f - along * 0.31f + 1.7f) * dune_noise * 0.13f;
    float wave = 0.5f + 0.5f * std::sin(along * 6.28318530718f /
                                        (dune_width * 0.24f) + phase_noise);
    float windward = smoothstep(wave);
    float crest = windward * windward * (1.18f - 0.18f * windward);
    float fine_wave = 0.5f + 0.5f * std::sin(along * 13.1f + across * 1.4f + 0.9f);
    return desert_weight * (crest * 17.0f + fine_wave * 1.6f);
}

}
int main() {
    unsigned values=0;
    for(std::uint32_t v=0;v<100000;v++)for(auto input:{v,v*0x873abe19u}){
        assert(c3x_renderer::patterns::feature_hash(input)==reference::feature_hash(input));
        float a=c3x_renderer::patterns::stable_random(input);
        float b=static_cast<float>(reference::feature_hash(input)&0x00ffffffu)/16777215.0f;
        assert(!std::memcmp(&a,&b,4));values++;
    }
    unsigned samples=0;
    for(int x=-128;x<=128;x++)for(int y=-64;y<=64;y++)for(float weight:{-1.f,0.f,.17f,1.f}){
        float a=c3x_renderer::patterns::dune_height(x*.137f,y*.213f,weight);
        float b=reference::dune_height(x*.137f,y*.213f,weight);
        assert(!std::memcmp(&a,&b,4));samples++;
    }
    std::cout<<"PASS production patterns: "<<values<<" hashes/random values, "<<samples<<" exact dune samples\n";
}
