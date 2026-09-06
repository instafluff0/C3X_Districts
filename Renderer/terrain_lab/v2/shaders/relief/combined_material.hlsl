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
