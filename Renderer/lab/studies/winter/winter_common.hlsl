// ---- Lab winter study (Renderer/lab/studies/winter) -----------------------
// Colour work in OKLab so source texture detail survives; snow cover is a
// world-space field over the authored material height, never a tile stamp.
float3 winter_to_oklab(float3 c) {
    c = max(c, 0);
    float3 lms = float3(dot(c, float3(0.4122214708, 0.5363325363, 0.0514459929)),
                        dot(c, float3(0.2119034982, 0.6806995451, 0.1073969566)),
                        dot(c, float3(0.0883024619, 0.2817188376, 0.6299787005)));
    lms = pow(max(lms, 1e-9), 1.0 / 3.0);
    return float3(dot(lms, float3(0.2104542553, 0.7936177850, -0.0040720468)),
                  dot(lms, float3(1.9779984951, -2.4285922050, 0.4505937099)),
                  dot(lms, float3(0.0259040371, 0.7827717662, -0.8086757660)));
}
float3 winter_from_oklab(float3 lab) {
    float3 lms = float3(lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z,
                        lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z,
                        lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z);
    lms = lms * lms * lms;
    return max(float3(dot(lms, float3(4.0767416621, -3.3077115913, 0.2309699292)),
                      dot(lms, float3(-1.2684380046, 2.6097574011, -0.3413193965)),
                      dot(lms, float3(-0.0041960863, -0.7034186147, 1.7076147010))), 0);
}
// x = lightness, y = chroma, z = hue in degrees
float3 winter_lch(float3 c) {
    float3 lab = winter_to_oklab(c);
    return float3(lab.x, length(lab.yz), atan2(lab.z, lab.y) * 57.2957795);
}
float3 winter_from_lch(float3 lch) {
    float h = lch.z / 57.2957795;
    return winter_from_oklab(float3(lch.x, lch.y * cos(h), lch.y * sin(h)));
}
float winter_luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }
float winter_hash(float2 p) { return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453); }
float winter_noise(float2 x) {
    float2 i = floor(x), f = frac(x);
    f = f * f * (3 - 2 * f);
    return lerp(lerp(winter_hash(i), winter_hash(i + float2(1, 0)), f.x),
                lerp(winter_hash(i + float2(0, 1)), winter_hash(i + 1), f.x), f.y);
}
float winter_fbm(float2 x) {
    return winter_noise(x) * 0.55 + winter_noise(x * 2.13 + 7.1) * 0.28 + winter_noise(x * 4.71 + 3.7) * 0.17;
}
// Concept measurements (winter-icy-blue, OKLab): lit snow L .92 C .025 h 261;
// snow in shade keeps L ~.85 with a clearly blue cast; desert snow is a more
// neutral pale grey (C .011); forests read dark navy-green under white boughs.
// Snow colour: cool, very slightly blue white; lows a touch bluer.
float3 winter_snow_albedo(float detail, float2 world) {
    float speck = winter_noise(world * 41.0) * 0.5 + winter_noise(world * 97.0 + 3.3) * 0.5;
    float3 snow = lerp(WINTER_SNOW_LOW, WINTER_SNOW_HIGH, saturate(detail));
    return snow * (1 + (speck - 0.5) * WINTER_SNOW_SPECK);
}
// Dormant ground showing through: straw/grey-brown, desaturated, darker.
float3 winter_dormant(float3 albedo) {
    float3 lch = winter_lch(albedo);
    lch.y *= WINTER_DORMANT_CHROMA;
    lch.z = lerp(lch.z, WINTER_DORMANT_HUE, 0.6);
    lch.x *= WINTER_DORMANT_LIGHT;
    return winter_from_lch(lch);
}
// Snow cover over ground: settles in material-height lows first, sheds on
// steep faces, thins in broad world-space wind-scoured patches. depth = 1 is
// the study's normal cover; 0 none. height is the authored material height.
float winter_cover(float depth, float height, float slope, float2 world) {
    float scour = winter_fbm(world * WINTER_SCOUR_SCALE);
    float level = depth * WINTER_DEPTH - (scour - 0.5) * WINTER_SCOUR;
    float cover = smoothstep(-0.12, 0.12, level - height);
    cover *= 1 - smoothstep(WINTER_SHED_LOW, WINTER_SHED_HIGH, slope);
    return saturate(cover);
}
