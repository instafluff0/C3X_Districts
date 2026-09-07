// Source tiling rock projected onto three planes, adopted from Q4's witness.
// Source height/footprint, placement and source skin remain unchanged.
// Define only in candidate fixtures until visual and backend gates pass.
float3 q4_relief_color(Texture2D source, PixelInput input) {
#ifdef Q4_COMBINED_ROCK_PROJECTION
    float3 weights=pow(abs(normalize(input.geometry_normal)),4);
    weights/=max(dot(weights,1),.00001);
    // q6_world supplies authoritative geometry; macro UV alone has no height
    // and stretches a narrow row of texels over every steep mountain face.
    float3 p=input.q6_world.xyz*1.5;
    return source.Sample(material_sampler,p.yz).rgb*weights.x
        +source.Sample(material_sampler,p.xz).rgb*weights.y
        +source.Sample(material_sampler,p.xy).rgb*weights.z;
#else
    return source.Sample(material_sampler,input.uv).rgb;
#endif
}

#ifdef Q4_COHERENT_ROCK_CHANNELS
#ifndef Q4_ROCK_HEIGHT_AMPLITUDE
#define Q4_ROCK_HEIGHT_AMPLITUDE .025
#endif
// Filtered height gradients share the color projection. They are transformed
// into the actual surface tangent plane; steep faces retain their response.
// The amplitude is an explicit C3X material calibration, not an engine value.
float2 q4_rock_plane_gradient(Texture2D source,float2 uv) {
    float2 dx=ddx(uv),dy=ddy(uv);
    uint width,height;source.GetDimensions(width,height);
    float2 step_uv=max(1.0/float2(width,height),.5*(abs(dx)+abs(dy)));
    return float2(
        source.SampleGrad(material_sampler,uv+float2(step_uv.x,0),dx,dy).r-
        source.SampleGrad(material_sampler,uv-float2(step_uv.x,0),dx,dy).r,
        source.SampleGrad(material_sampler,uv+float2(0,step_uv.y),dx,dy).r-
        source.SampleGrad(material_sampler,uv-float2(0,step_uv.y),dx,dy).r)/(2*step_uv);
}
float3 q4_rock_gradient(Texture2D source,PixelInput input) {
    float3 n=normalize(input.geometry_normal),w=pow(abs(n),4);
    w/=max(dot(w,1),.00001);
    float3 p=input.q6_world.xyz*1.5;
    float2 x=q4_rock_plane_gradient(source,p.yz);
    float2 y=q4_rock_plane_gradient(source,p.xz);
    float2 z=q4_rock_plane_gradient(source,p.xy);
    float3 gradient=(float3(0,x.x,x.y)*w.x+float3(y.x,0,y.y)*w.y+
        float3(z.x,z.y,0)*w.z)*1.5;
    gradient-=n*dot(n,gradient);
    return gradient;
}
float3 q4_rock_material_normal(Texture2D source,PixelInput input) {
    return normalize(normalize(input.geometry_normal)-q4_rock_gradient(source,input)*Q4_ROCK_HEIGHT_AMPLITUDE);
}
#ifdef Q4_COMPLETE_ROCK_CHANNELS
// Slots 108..115 are unused in these terrain draws. Feature draws keep their
// original bindings. The copied-packet adapter refuses occupied terrain slots.
#define q4_snow_height road_bridge_base_texture_0
#define q4_snow_specular road_bridge_base_texture_1
#define q4_stripe1_height road_bridge_base_texture_2
#define q4_stripe1_specular road_bridge_base_texture_3
#define q4_stripe2_height road_bridge_base_texture_4
#define q4_stripe2_specular road_bridge_base_texture_5
#define q4_stripe3_height road_bridge_base_texture_6
#define q4_stripe3_specular road_bridge_base_texture_7
float4 q4_rock_layer_weights(PixelInput input,bool desert) {
    float h=input.authored_relief.x,u=smoothstep(.22,.62,h);
    float4 w=float4(1-u,u,0,0);
    if(desert) {
        float second=smoothstep(.55,.76,h),third=smoothstep(.76,.90,h);
        w=lerp(w,float4(0,0,1,0),second);
        w=lerp(w,float4(0,0,0,1),third);
    } else {
        float snow=smoothstep(.70,.84,h)*smoothstep(.18,.72,input.geometry_normal.z);
        w=lerp(w,float4(0,0,1,0),snow);
    }
    return w;
}
float3 q4_complete_rock_normal(PixelInput input,bool desert) {
    float4 w=q4_rock_layer_weights(input,desert);
    float3 g;
    if(desert) {
        g=q4_rock_gradient(desert_mountain_height_texture,input)*w.x;
        if(w.y>0)g+=q4_rock_gradient(q4_stripe1_height,input)*w.y;
        if(w.z>0)g+=q4_rock_gradient(q4_stripe2_height,input)*w.z;
        if(w.w>0)g+=q4_rock_gradient(q4_stripe3_height,input)*w.w;
    } else {
        // Normalized base and top height/specular payloads are identical.
        g=q4_rock_gradient(mountain_height_texture,input)*(w.x+w.y);
        if(w.z>0)g+=q4_rock_gradient(q4_snow_height,input)*w.z;
    }
    return normalize(normalize(input.geometry_normal)-g*Q4_ROCK_HEIGHT_AMPLITUDE);
}
float q4_complete_rock_specular(PixelInput input,bool desert) {
    float4 w=q4_rock_layer_weights(input,desert);
    if(desert)return dot(w,float4(q4_relief_color(desert_mountain_specular_texture,input).r,
        q4_relief_color(q4_stripe1_specular,input).r,q4_relief_color(q4_stripe2_specular,input).r,
        q4_relief_color(q4_stripe3_specular,input).r));
    return q4_relief_color(mountain_specular_texture,input).r*(w.x+w.y)+
        q4_relief_color(q4_snow_specular,input).r*w.z;
}
#endif
#endif
