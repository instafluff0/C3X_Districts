// Authored ground underlay. Crop and texel density are normalized offline data.
// Mirror inside an unmarked atlas interior rather than stretch the whole sheet.
float2 q8_settlement_ground_uv(FeaturePixelInput p,float2 folded) {
 float2 extent=Q8_SETTLEMENT_ATLAS.zw-Q8_SETTLEMENT_ATLAS.xy;
 return Q8_SETTLEMENT_ATLAS.xy+(1-abs(folded))*extent;
}
float2 q8_settlement_ground_gradient(FeaturePixelInput p,float2 folded,float2 uv_gradient) {
 float2 extent=Q8_SETTLEMENT_ATLAS.zw-Q8_SETTLEMENT_ATLAS.xy;
 return uv_gradient*(-sign(folded)*extent);
}
float4 q8_settlement_ground_sample(FeaturePixelInput p) {
 float2 folded=frac(p.uv*.5)*2-1;
 float2 uv=q8_settlement_ground_uv(p,folded);
 float2 ux=q8_settlement_ground_gradient(p,folded,ddx(p.uv));
 float2 uy=q8_settlement_ground_gradient(p,folded,ddy(p.uv));
 float4 ground=city_base_texture_0.SampleGrad(decal_sampler,uv,ux,uy);
 ground.a*=saturate(p.material_index-(p.material_index>63.5?64:62))*Q8_SETTLEMENT_GAIN;
 return ground;
}
float3 q8_settlement_ground_normal(FeaturePixelInput p,
 out float3 tangent,out float3 bitangent,out float2 slope) {
 float3 n=normalize(p.geometry_normal);
 float2 folded=frac(p.uv*.5)*2-1;
 float2 uv=q8_settlement_ground_uv(p,folded);
 float2 ux=q8_settlement_ground_gradient(p,folded,ddx(p.uv));
 float2 uy=q8_settlement_ground_gradient(p,folded,ddy(p.uv));
 slope=resource_base_texture_3.SampleGrad(decal_sampler,uv,ux,uy).rg*2-1;
 tangent=float3(1,0,0);bitangent=float3(0,1,0);
 float3 world=p.q6_world.xyz*float3(1,-1,Q8_CITY_WORLD_Z_TO_SOURCE);
 float3 dx=ddx(world),dy=ddy(world);
 float determinant=ux.x*uy.y-uy.x*ux.y;
 if(abs(determinant)>1e-9) {
  float3 t=(dx*uy.y-dy*ux.y)/determinant;
  float3 b=(dy*ux.x-dx*uy.x)/determinant;
  t-=n*dot(t,n);b-=n*dot(b,n);
  if(dot(t,t)>1e-10 && dot(b,b)>1e-10) {
   tangent=normalize(t);bitangent=normalize(b);
   n=normalize(n+tangent*slope.x+bitangent*slope.y);
  }
 }
 return n;
}
