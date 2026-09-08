// Q6 proposal v1: Rec.709/D65 scene-linear radiance, no category exposure.
// This function runs ONCE AFTER scene-linear composition/reconstruction.
float3 q6_display_linear(float3 radiance, float exposure)
{
    float3 c = max(0.0, radiance * exposure);
    // Shared max-channel shoulder preserves RGB ratios and never clips windows.
    return c / (1.0 + max(c.r, max(c.g, c.b)));
}
float3 q6_srgb_encode(float3 c)
{
    return float3(c.r <= 0.0031308 ? 12.92*c.r : 1.055*pow(c.r,1.0/2.4)-0.055,
                  c.g <= 0.0031308 ? 12.92*c.g : 1.055*pow(c.g,1.0/2.4)-0.055,
                  c.b <= 0.0031308 ? 12.92*c.b : 1.055*pow(c.b,1.0/2.4)-0.055);
}
float4 q6_over(float4 front, float4 back)
{
    // Both arguments contain linear RGB already multiplied by coverage/opacity.
    return front + back * (1.0-front.a);
}
