#pragma once
namespace c3x_renderer {
// Shared map-only HDR display transfer. The curve changes neither geometry nor
// native UI and adds no intermediate image. The neutral control is exact legacy
// max-channel Reinhard; the filmic mixture adds a restrained toe and shoulder.
inline char const* scene_display_shader(){return R"(
float3 scene_display_linear(float3 radiance,float filmic_amount) {
 float3 x=max(0,radiance);
 float3 neutral=x/(1+max(x.r,max(x.g,x.b)));
 // Narkowicz's CC0 ACES-like fit, not a full ACES color transform:
 // https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
 // Preserve room for bright water and emissives; avoid overflow in FP32.
 float3 f=min(x*.65,64);
 f=saturate((f*(2.51*f+.03))/(f*(2.43*f+.59)+.14));
 return lerp(neutral,f,saturate(filmic_amount));
}
float3 scene_display_srgb(float3 radiance,float filmic_amount) {
 float3 rgb=scene_display_linear(radiance,filmic_amount);
 return float3(rgb.r<=.0031308?rgb.r*12.92:1.055*pow(rgb.r,1/2.4)-.055,
               rgb.g<=.0031308?rgb.g*12.92:1.055*pow(rgb.g,1/2.4)-.055,
               rgb.b<=.0031308?rgb.b*12.92:1.055*pow(rgb.b,1/2.4)-.055);
}
)";}
}
