// Q1 candidate, consumes q6_scene_linear_premultiplied_v1.
// No transfer, exposure, tone map, body scaling, or sharpening is performed.
// Integer scale only (the platform currently offers 1/2/4); equal-area box
// preserves HDR radiance, premultiplied coverage and constant fields.
struct Q1Reconstructed { float4 rgba; float coverage; };
Q1Reconstructed q1_reconstruct_box(
    Texture2D<float4> scene_linear, Texture2D<float> map_validity,
    uint2 input_size, uint2 output_size, int4 valid_output_rect, int2 pixel) {
    Q1Reconstructed result; result.rgba=0; result.coverage=0;
    if(any(pixel<valid_output_rect.xy)||any(pixel>=valid_output_rect.zw))return result;
    int2 ratio=int2(input_size/output_size),start=pixel*ratio;
    for(int y=0;y<ratio.y;y++)for(int x=0;x<ratio.x;x++) {
        int2 at=start+int2(x,y);
        float validity=map_validity.Load(int3(at,0));
        // Color already contains opacity/coverage; never premultiply it again.
        if(validity>0)result.rgba+=scene_linear.Load(int3(at,0));
        result.coverage+=validity;
    }
    float area=ratio.x*ratio.y;
    result.rgba/=area;result.coverage/=area;
    return result;
}
