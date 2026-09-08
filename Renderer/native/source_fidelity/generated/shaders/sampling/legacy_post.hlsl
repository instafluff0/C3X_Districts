// Provisional compatibility experiment ONLY: input is frozen gamma-2.2 encoded
// straight-alpha BGRA8. This cannot repair prior gamma-space scene blending.
// Q6 scene-linear RGBA16_FLOAT must use reconstruction_v1.hlsl instead.
Texture2D<float4> source_color : register(t0);
RWTexture2D<float4> output_color : register(u1);
cbuffer PostSettings : register(b2) { uint2 input_size; uint2 output_size; };
float4 linear_sample(int2 p) {
    float4 s=source_color.Load(int3(clamp(p,int2(0,0),int2(input_size)-1),0));
    return float4(pow(max(s.rgb,0),2.2)*s.a,s.a);
}
float4 box_at(int2 p) {
    p=clamp(p,int2(0,0),int2(output_size)-1);
    int2 ratio=int2(input_size/output_size),start=p*ratio;
    float4 value=0;
    for(int y=0;y<ratio.y;y++)for(int x=0;x<ratio.x;x++)value+=linear_sample(start+int2(x,y));
    return value/(ratio.x*ratio.y);
}
float mitchell_weight(float x) {
    x=abs(x);
    if(x<1)return ((7*x-12)*x*x+16.0/3.0)/6;
    if(x<2)return (((-7.0/3.0*x+12)*x-20)*x+32.0/3.0)/6;
    return 0;
}
float4 reconstruct(int2 p) {
    float4 center=box_at(p);
#if Q1_MODE == 1
    float2 ratio=float2(input_size)/output_size;
    if(all(ratio==1))return center;
    float2 at=(p+.5)*ratio-.5;
    int2 lo=int2(floor(at-2*ratio)),hi=int2(ceil(at+2*ratio));
    float4 sum=0;float weight=0;
    for(int y=lo.y;y<=hi.y;y++)for(int x=lo.x;x<=hi.x;x++) {
        float w=mitchell_weight((x-at.x)/ratio.x)*mitchell_weight((y-at.y)/ratio.y);
        sum+=linear_sample(int2(x,y))*w;weight+=w;
    }
    return sum/weight; // Deliberately unbounded diagnostic, never promoted.
#elif Q1_MODE >= 2
    float4 n=box_at(p+int2(0,-1)),s=box_at(p+int2(0,1));
    float4 e=box_at(p+int2(1,0)),w=box_at(p+int2(-1,0));
    float3 sharpened=center.rgb+.3*(center.rgb-(n.rgb+s.rgb+e.rgb+w.rgb)*.25);
#if Q1_MODE == 3
    // Reject overshoot and protect coverage/viewport boundaries. This limits
    // the filter, but direct pan inspection still decides crawling/halo quality.
    float3 low=min(center.rgb,min(min(n.rgb,s.rgb),min(e.rgb,w.rgb)));
    float3 high=max(center.rgb,max(max(n.rgb,s.rgb),max(e.rgb,w.rgb)));
    sharpened=clamp(sharpened,low,high);
    if(any(p<1)||any(p>=int2(output_size)-1)||min(min(n.a,s.a),min(e.a,w.a))<.999||center.a<.999)
        sharpened=center.rgb;
#endif
    return float4(sharpened,center.a);
#else
    return center;
#endif
}
[numthreads(8,8,1)]
void CSPost(uint3 id:SV_DispatchThreadID) {
    if(any(id.xy>=output_size))return;
    float4 c=reconstruct(int2(id.xy));
    output_color[id.xy]=float4(pow(saturate(c.rgb/max(c.a,1e-6)),1/2.2),saturate(c.a));
}
