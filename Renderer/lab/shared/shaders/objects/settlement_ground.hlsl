// Authored ground underlay. Crop and texel density are normalized offline data.
// Mirror inside an unmarked atlas interior rather than stretch the whole sheet.
float4 q8_settlement_ground_sample(FeaturePixelInput p) {
 float2 folded=frac(p.uv*.5)*2-1;
 float2 extent=Q8_SETTLEMENT_ATLAS.zw-Q8_SETTLEMENT_ATLAS.xy;
 float2 uv=Q8_SETTLEMENT_ATLAS.xy+(1-abs(folded))*extent;
 float2 direction=-sign(folded)*extent;
 float4 ground=city_base_texture_0.SampleGrad(decal_sampler,uv,ddx(p.uv)*direction,ddy(p.uv)*direction);
 ground.a*=saturate(p.material_index-62)*Q8_SETTLEMENT_GAIN;
 return ground;
}
