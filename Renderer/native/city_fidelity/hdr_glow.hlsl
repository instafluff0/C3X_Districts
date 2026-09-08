// Same two-scale HDR lens response as hdr_glow.hlsl, with separable workgroup
// filtering. Opt-in Lab diagnostic for the fixed 1x/2x gameplay reconstruction.
// Q1 candidate, consumes q6_scene_linear_premultiplied_v1.
// No transfer, exposure, tone map, body scaling, or sharpening is performed.
// Integer scale only (the platform currently offers 1/2/4); equal-area box
// preserves HDR radiance, premultiplied coverage and constant fields.
struct Q1Reconstructed { float4 rgba; float coverage; };
Q1Reconstructed q1_reconstruct_box(
    Texture2D<float4> scene_linear, Texture2D<float4> map_validity,
    uint2 input_size, uint2 output_size, int4 valid_output_rect, int2 pixel) {
    Q1Reconstructed result; result.rgba=0; result.coverage=0;
    if(any(pixel<valid_output_rect.xy)||any(pixel>=valid_output_rect.zw))return result;
    int2 ratio=int2(input_size/output_size),start=pixel*ratio;
    for(int y=0;y<ratio.y;y++)for(int x=0;x<ratio.x;x++) {
        int2 at=start+int2(x,y);
        float validity=map_validity.Load(int3(at,0)).a;
        // Color already contains opacity/coverage; never premultiply it again.
        if(validity>0)result.rgba+=scene_linear.Load(int3(at,0));
        result.coverage+=validity;
    }
    float area=ratio.x*ratio.y;
    result.rgba/=area;result.coverage/=area;
    return result;
}

Texture2D<float4> Scene:register(t0);
RWTexture2D<float4> Output:register(u1);
Texture2D<float4> Validity:register(t3);
RWTexture2D<float> OutputValidity:register(u4);
cbuffer PostSettings:register(b2) {uint2 input_size;uint2 output_size;int4 valid_rect;float4 NativeGlow;};
#ifndef Q8_GLOW_GAIN
#define Q8_GLOW_GAIN NativeGlow.x
#endif
groupshared float3 highlights[1024];
groupshared float3 near_rows[256];
groupshared float3 far_rows[256];
float3 q8_highlight(int2 p) {
 if(any(p<0)||any(p>=int2(input_size)))return 0;
 float4 c=Scene.Load(int3(p,0));
 if(Validity.Load(int3(p,0)).a<=0||c.a<=0)return 0;
 float peak=max(c.r,max(c.g,c.b));
 return c.rgb*(max(peak-1.0,0)/max(peak,0.0001));
}
[numthreads(8,8,1)]
void CSPost(uint3 id:SV_DispatchThreadID,uint3 group:SV_GroupID,uint3 local:SV_GroupThreadID) {
 uint step=input_size.x/output_size.x;
 bool supported=step>=1&&step<=2&&all(input_size==output_size*step);
 // Keep all lanes alive through both barriers, including output-edge lanes.
 // FXC rejects a varying early return even inside a uniform fallback branch.
 step=clamp(step,1u,2u);
 uint side=8*step+16;uint lane=local.y*8+local.x;
 int2 base=int2(group.xy*8*step)+int(step/2)-8;
 for(uint load_index=lane;load_index<side*side;load_index+=64)
  highlights[load_index]=q8_highlight(base+int2(load_index%side,load_index/side));
 GroupMemoryBarrierWithGroupSync();
 for(uint row_index=lane;row_index<side*8;row_index+=64) {
  uint x=(row_index%8)*step+8;uint y=row_index/8;
  float3 n=0,f=0;
  for(int dx=-8;dx<=8;dx++) {
   float3 c=highlights[y*side+x+dx];float d=float(dx*dx);
   n+=c*exp(-d/5.0);f+=c*exp(-d/28.0);
  }
  near_rows[row_index]=n;far_rows[row_index]=f;
 }
 GroupMemoryBarrierWithGroupSync();
 if(any(id.xy>=output_size))return;
 float3 near_sum=0,far_sum=0;float nw=0,fw=0;
 uint y=local.y*step+8;
 for(int dy=-8;dy<=8;dy++) {
  float d=float(dy*dy),n=exp(-d/5.0),f=exp(-d/28.0);
  near_sum+=near_rows[(y+dy)*8+local.x]*n;
  far_sum+=far_rows[(y+dy)*8+local.x]*f;nw+=n;fw+=f;
 }
 float3 glow=(near_sum+0.12*far_sum)/(nw*nw+0.12*fw*fw);
 Q1Reconstructed r=q1_reconstruct_box(Scene,Validity,input_size,output_size,valid_rect,int2(id.xy));
 if(supported)r.rgba.rgb+=glow*Q8_GLOW_GAIN*r.rgba.a;
 Output[id.xy]=r.rgba;OutputValidity[id.xy]=r.coverage;
}
