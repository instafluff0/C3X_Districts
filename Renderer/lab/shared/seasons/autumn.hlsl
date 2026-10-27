// Generated from the approved Lab look (Renderer/lab/studies); see seasons.py.
#define AUTUMN_PATCH_SCALE 0.45
#define AUTUMN_GROUND_SHIFT -20
#define AUTUMN_PATCH_HUE 18
#define AUTUMN_STRAW_CHROMA 1.12
#define AUTUMN_GRASS_CHROMA 0.84
#define AUTUMN_GRASS_LIGHT 0.92
#define AUTUMN_PATCH_LIGHT 0.18
#define AUTUMN_LEAF_LIGHT 1.0
#define AUTUMN_LEAF_DETAIL 0.8
#define AUTUMN_BROADLEAF_GB 3.2
#define AUTUMN_CROWN 0.85
#define AUTUMN_WRAP 0.2
#define AUTUMN_FILL 0.85
#define AUTUMN_CROWN_BASE 0.5
#define AUTUMN_CROWN_TOP 1.08
#define AUTUMN_LEAF_CHROMA 0.92
#define AUTUMN_SAND_FLOOR 70
#define AUTUMN_LEAF_BIAS 1.0
#define AUTUMN_HILL_TURF 1.32
#define AUTUMN_GREEN_PULL 0.6
#define AUTUMN_TRANSLUCENCY 0.8
#define AUTUMN_PALE_KEEP 0.85
#define AUTUMN_HILL_FLOOR 0.055
#define AUTUMN_FIELD_CELLS 2.5
#define AUTUMN_LARCH_SHARE 0
#define AUTUMN_SHRUB_SHARE 0
#define AUTUMN_ROAD_DARK float3(0.80, 0.76, 0.74)
#define AUTUMN_FIELD autumn_farm
#define AUTUMN_FARM_HUE 86
#define AUTUMN_FARM_PULL 0.85
#define AUTUMN_FARM_CHROMA 0.85
#define AUTUMN_FARM_LIGHT 1.02
#define AUTUMN_LITTER_END autumn_keep
#define AUTUMN_LEAF_COVER 0.75
#define AUTUMN_LEAF_SHADE 0.74
#define AUTUMN_LEAF_PATCH 18
#define AUTUMN_LEAF_GOLD 0.28
#define AUTUMN_LEAF_ORANGE 0.62
#define AUTUMN_LEAF_RED 0.78
#define AUTUMN_LEAF_SAT 0.76
#define AUTUMN_LEAF_LIGHTNESS 0.93
#define AUTUMN_LEAF_FALLOFF 0
#define AUTUMN_LEAF_GAPS 0.22
#define AUTUMN_LEAF_SCATTER 0
#define AUTUMN_LEAF_SCATTER_CELLS 40
#define AUTUMN_LEAF_DECAL_RADIUS 0
#define AUTUMN_TURF 1.1
#define AUTUMN_HILL_CHROMA 1.2
// ---- Lab autumn study (Renderer/lab/studies/autumn) ----------------------
// Season colour work happens in OKLab lightness/chroma/hue so source texture
// detail survives: only hue/chroma move, lightness keeps the authored grain.
float3 autumn_to_oklab(float3 c) {
    c = max(c, 0);
    float3 lms = float3(dot(c, float3(0.4122214708, 0.5363325363, 0.0514459929)),
                        dot(c, float3(0.2119034982, 0.6806995451, 0.1073969566)),
                        dot(c, float3(0.0883024619, 0.2817188376, 0.6299787005)));
    lms = pow(max(lms, 1e-9), 1.0 / 3.0);
    return float3(dot(lms, float3(0.2104542553, 0.7936177850, -0.0040720468)),
                  dot(lms, float3(1.9779984951, -2.4285922050, 0.4505937099)),
                  dot(lms, float3(0.0259040371, 0.7827717662, -0.8086757660)));
}
float3 autumn_from_oklab(float3 lab) {
    float3 lms = float3(lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z,
                        lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z,
                        lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z);
    lms = lms * lms * lms;
    return max(float3(dot(lms, float3(4.0767416621, -3.3077115913, 0.2309699292)),
                      dot(lms, float3(-1.2684380046, 2.6097574011, -0.3413193965)),
                      dot(lms, float3(-0.0041960863, -0.7034186147, 1.7076147010))), 0);
}
// x = lightness, y = chroma, z = hue in degrees
float3 autumn_lch(float3 c) {
    float3 lab = autumn_to_oklab(c);
    return float3(lab.x, length(lab.yz), atan2(lab.z, lab.y) * 57.2957795);
}
float3 autumn_from_lch(float3 lch) {
    float h = lch.z / 57.2957795;
    return autumn_from_oklab(float3(lch.x, lch.y * cos(h), lch.y * sin(h)));
}
float autumn_hash(float2 p) { return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453); }
float autumn_noise(float2 x) {
    float2 i = floor(x), f = frac(x);
    f = f * f * (3 - 2 * f);
    return lerp(lerp(autumn_hash(i), autumn_hash(i + float2(1, 0)), f.x),
                lerp(autumn_hash(i + float2(0, 1)), autumn_hash(i + 1), f.x), f.y);
}
// Broad dry/olive patches, world-defined.
float autumn_patches(float2 world) {
    return autumn_noise(world * AUTUMN_PATCH_SCALE) * 0.65 +
           autumn_noise(world * AUTUMN_PATCH_SCALE * 3.4 + 17.3) * 0.35;
}
// Greens turn olive-gold, straw warms toward honey; blue, grey and red stay.
// Measured on the concept view (OKLab): grass h 111 -> 90, C x.82, L x.91;
// plains h 92 -> 75, C x1.12; tundra h 98 -> 76; hills h 110 -> 89.
float3 autumn_ground(float3 albedo, float2 world) {
    float3 lch = autumn_lch(albedo);
    float patch = autumn_patches(world) - 0.5;
    float band = smoothstep(52, 72, lch.z) * (1 - smoothstep(150, 172, lch.z));
    float green = smoothstep(95, 108, lch.z);
    float gate = band * smoothstep(0.006, 0.022, lch.y);
    // Pale desert sand keeps its own cream hue; only turf and straw turn.
    gate *= 1 - AUTUMN_PALE_KEEP * smoothstep(0.76, 0.86, lch.x) * (1 - smoothstep(0.06, 0.10, lch.y));
    float hue = lch.z;
    lch.z += (AUTUMN_GROUND_SHIFT + patch * AUTUMN_PATCH_HUE) * gate;
    // The greenest turf (some hill and floodplain textures) is pulled further.
    lch.z -= max(0, hue - 112) * AUTUMN_GREEN_PULL * gate;
    // Pale sand and straw stay honey-coloured; they must not drift to salmon.
    float sand = smoothstep(0.62, 0.78, lch.x) * (1 - smoothstep(0.07, 0.12, lch.y));
    lch.z = lerp(lch.z, max(lch.z, min(hue, AUTUMN_SAND_FLOOR)), sand);
    lch.y *= lerp(1.0, lerp(AUTUMN_STRAW_CHROMA, AUTUMN_GRASS_CHROMA, green), gate);
    lch.x *= lerp(1.0, lerp(0.975, AUTUMN_GRASS_LIGHT, green) * (1 + patch * AUTUMN_PATCH_LIGHT), gate);
    return autumn_from_lch(lch);
}
// Fallen leaves over the authored forest-floor dirt.
float3 autumn_litter(float3 albedo, float2 world) {
    float3 lch = autumn_lch(albedo);
    float n = autumn_noise(world * 9.0);
    float leaf = smoothstep(0.25, 0.75, n);
    lch.z = lerp(lch.z, lerp(52, 72, autumn_noise(world * 23.0 + 5.1)), leaf * 0.85);
    lch.y = lerp(lch.y, max(lch.y, 0.11), leaf);
    lch.x *= lerp(1.0, 1.18, leaf);
    return autumn_from_lch(lch);
}
float3 autumn_keep(float3 albedo, float2 world) { return albedo; }
// Fallen leaves: a top-down leaf-clump pattern (the jungle-floor decal) takes
// small world patches of gold, orange, russet and brown, keeping each leaf's
// own light and dark shape.
float3 autumn_fallen(float3 leaf, float average_light, float2 world) {
    float3 lch = autumn_lch(leaf);
    float n = autumn_noise(world * AUTUMN_LEAF_PATCH) * 0.7 + autumn_noise(world * AUTUMN_LEAF_PATCH * 2.7 + 9.2) * 0.3;
    float3 pal = n < AUTUMN_LEAF_GOLD ? float3(0.70, 0.150, 84) : n < AUTUMN_LEAF_ORANGE ? float3(0.63, 0.160, 60) :
                 n < AUTUMN_LEAF_RED ? float3(0.53, 0.150, 42) : float3(0.45, 0.075, 58);
    return autumn_from_lch(float3(max(pal.x * AUTUMN_LEAF_LIGHTNESS + (lch.x - average_light) * 1.1, 0.05),
                                  pal.y * AUTUMN_LEAF_SAT, pal.z));
}
// Single fallen leaves sprinkled over the turf: one small, randomly placed
// leaf-shaped fleck per world cell, kept with a chance that falls with r
// (0 at a pile's centre, 1 at its rim).
float autumn_sprinkle(float2 world, float r) {
    float2 p = world * AUTUMN_LEAF_SCATTER_CELLS;
    float2 cell = floor(p), f = frac(p) - 0.5;
    float2 offset = float2(autumn_hash(cell + 3.3), autumn_hash(cell + 7.7)) - 0.5;
    float angle = autumn_hash(cell + 11.9) * 6.2831853;
    float2 q = f - offset * 0.5;
    q = float2(q.x * cos(angle) - q.y * sin(angle), q.x * sin(angle) + q.y * cos(angle)) * float2(1.0, 1.8);
    float leaf = 1 - smoothstep(0.17, 0.24, length(q));
    return leaf * step(autumn_hash(cell + 0.71), AUTUMN_LEAF_SCATTER * saturate(1.15 - r));
}
// Harvest: planted plots become golden stubble, ripe amber, a few dark
// ploughed or still-green fields, chosen per plot-sized world cell.
float3 autumn_field(float3 albedo, float2 world) {
    float3 lch = autumn_lch(albedo);
    float pick = autumn_hash(floor(world * AUTUMN_FIELD_CELLS) + 0.37);
    float3 target = pick < 0.50 ? float3(lch.x * 1.10, 0.090, 86) :      // stubble
                    pick < 0.74 ? float3(lch.x * 1.02, 0.110, 72) :      // ripe
                    pick < 0.90 ? float3(lch.x * 0.72, 0.045, 62) :      // ploughed
                                  float3(lch.x, lch.y * 0.85, lch.z - 8); // late green
    // Keep the plot's own row texture: shift around its lightness, not a flat fill.
    return autumn_from_lch(float3(target.x, target.y * saturate(lch.y / 0.05 + 0.4), target.z));
}
// Farms, simply: planted crops take one late-season golden tint over their
// own row texture (no per-plot states).
float3 autumn_farm(float3 albedo, float2 world) {
    float3 lch = autumn_lch(albedo);
    lch.z = lerp(lch.z, AUTUMN_FARM_HUE, AUTUMN_FARM_PULL);
    lch.y *= AUTUMN_FARM_CHROMA;
    lch.x *= AUTUMN_FARM_LIGHT;
    return autumn_from_lch(lch);
}
// ---------------------------------------------------------------------------
// Per-tree palette; the atlas keeps its own light/dark leaf structure.
float3 autumn_leaf_color(float3 albedo, float3 average, float seed) {
    float3 pal;
    if (seed < 0.34) pal = float3(0.75, 0.15, 84);
    else if (seed < 0.48) pal = float3(0.79, 0.14, 92);
    else if (seed < 0.6) pal = float3(0.7, 0.15, 70);
    else if (seed < 0.82) pal = float3(0.67, 0.155, 59);
    else if (seed < 0.88) pal = float3(0.6, 0.13, 54);
    else if (seed < 0.93) pal = float3(0.51, 0.125, 47);
    else pal = float3(0.62, 0.09, 104);
    pal.z += (frac(seed * 7.31) - 0.5) * 10;
    pal.x += (frac(seed * 13.7) - 0.5) * 0.06;
    float3 src = autumn_lch(albedo), avg = autumn_lch(average);
    float L = pal.x * AUTUMN_LEAF_LIGHT + (src.x - avg.x) * AUTUMN_LEAF_DETAIL;
    float C = pal.y * AUTUMN_LEAF_CHROMA * saturate(0.45 + 0.75 * src.x / max(avg.x, 0.05));
    float h = pal.z + clamp(src.z - avg.z, -25, 25) * 0.3;
    return autumn_from_lch(float3(max(L, 0.05), C, h));
}
