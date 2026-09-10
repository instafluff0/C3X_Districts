// Lab diagnosis: differentiate each height projection before blending it.
// Differentiating the blended height also differentiates projection weights,
// so unrelated texture values can introduce false gradients as a face turns.
float2 study_height_derivatives(Texture2D texture_map, float3 p, float3 n) {
    float3 weight = pow(abs(n), 5);
    weight /= max(dot(weight, 1), 0.00001);
    p *= Quality.y;
    float x = texture_map.Sample(Wrap, p.yz).r;
    float y = texture_map.Sample(Wrap, p.xz).r;
    float z = texture_map.Sample(Wrap, p.xy).r;
    return float2(ddx(x), ddy(x)) * weight.x +
           float2(ddx(y), ddy(y)) * weight.y +
           float2(ddx(z), ddy(z)) * weight.z;
}

float3 study_detail_normal(float3 geometric, float3 world, float2 detail, float strength) {
    float3 dx = ddx(world), dy = ddy(world);
    float3 r1 = cross(dy, geometric), r2 = cross(geometric, dx);
    float determinant = dot(dx, r1);
    float3 gradient = (detail.x * r1 + detail.y * r2) *
        sign(determinant) / max(abs(determinant), 0.000001);
    return normalize(geometric - gradient * strength);
}
