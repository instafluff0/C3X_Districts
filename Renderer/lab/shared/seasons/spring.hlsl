// Generated from the approved Lab look (Renderer/lab/studies); see seasons.py.
#define SPRING_GRASS_SHIFT 6
#define SPRING_STRAW_SHIFT 0
#define SPRING_GRASS_CHROMA 1.18
#define SPRING_STRAW_CHROMA 1.02
#define SPRING_GRASS_LIGHT 1.04
#define SPRING_LUSH_SCALE 1.1
#define SPRING_LUSH_HUE 10
#define SPRING_LUSH_CHROMA 0.25
#define SPRING_LUSH_LIGHT 0.16
#define SPRING_HILL_LIGHT 1.12
#define SPRING_HILL_CHROMA 1.15
#define SPRING_DRIFT_STRETCH 1.6
#define SPRING_DRIFT_SCALE 3.6
#define SPRING_DRIFT_THRESHOLD 0.54
#define SPRING_DRIFT_SOFT 0.14
#define SPRING_PALETTE_SCALE 2.4
#define SPRING_DENSITY 0.42
#define SPRING_SECOND_LAYER 0.6
#define SPRING_CELL 0.045
#define SPRING_BLOOM_MIN 0.012
#define SPRING_BLOOM_MAX 0.026
#define SPRING_BLOOM_LIGHT 1.0
#define SPRING_BLOOM_CHROMA 1.25
#define SPRING_AA 0.35
#define SPRING_PETAL_MODE 2
#define SPRING_JITTER 1.0
#define SPRING_UPRIGHT_CELL 0.8
#define SPRING_NOTCH 0.25
#define SPRING_RIM 0.78
#define SPRING_SIBLING 0.5
#define SPRING_SIBLING_GAP 3.4
#define SPRING_GRID_TURN 0.3
#define SPRING_SLOPE 1
#define SPRING_BLOOM_UP 0.6
#define SPRING_BLOOM_NORMAL 1
#define SPRING_STEM 0.9
#define SPRING_TOP_LIGHT 0.16
#define SPRING_CORAL 1
#define SPRING_SNAP 1
#define SPRING_BUNCH 0
#define SPRING_BUNCH_GAP 1.9
#define SPRING_PIXEL_SIZES 0
#define SPRING_STRAW_COLOR 0
#define SPRING_PLUS_FROM 0.85
#define SPRING_SHARE_YELLOW 0.24
#define SPRING_SHARE_WHITE 0.5
#define SPRING_SHARE_PINK 0.64
#define SPRING_SHARE_BLUE 0.8
#define SPRING_PLAINS_DENSITY 0.45
#define SPRING_TUNDRA_DENSITY 0
#define SPRING_HILL_DENSITY 0.5
#define SPRING_LEAF_SHADE 0.72
#define SPRING_LEAF_DRIFT 0.7
#define SPRING_TUFT 0.9
#define SPRING_SPRINKLE 0.004
#define SPRING_WORLEY_PALETTE 2
#define SPRING_STRAY 0.2
#define SPRING_CLUMP 0.5
#define SPRING_CLUMP_SCALE 22
#define SPRING_WOODLAND 0
#define SPRING_WHITE float3(0.95, 0.035, 92)
#define SPRING_YELLOW float3(0.88, 0.14, 97)
#define SPRING_TURF 1.2
#define SPRING_TURF_BIAS 3.0
#define SPRING_TURF_HUE 8
#define SPRING_HILL_HUE 107
#define SPRING_HILL_PULL 0.5
#define SPRING_BROADLEAF_GB 3.2
#define SPRING_CROWN 0.85
#define SPRING_WRAP 0.2
#define SPRING_FILL 0.85
#define SPRING_CROWN_BASE 0.58
#define SPRING_CROWN_TOP 1.06
#define SPRING_LEAF_LIGHT 1.0
#define SPRING_LEAF_CHROMA 1.0
#define SPRING_LEAF_DETAIL 0.7
#define SPRING_TRANSLUCENCY 0.5
#define SPRING_LEAF_BIAS 1.0
#define SPRING_SUN_HUE 10
#define SPRING_TREE_HUE_SPREAD 5
#define SPRING_BLOSSOM_SHARE 0.12
#define SPRING_BLOSSOM_COVER 0.8
// ---- Lab spring study (Renderer/lab/studies/spring) -----------------------
// Colour work happens in OKLab lightness/chroma/hue so source texture detail
// survives. Flowers are world-anchored marks composed into the ground albedo
// before lighting, so they receive the same sun, shadow and night as the turf.
float3 spring_to_oklab(float3 c) {
    c = max(c, 0);
    float3 lms = float3(dot(c, float3(0.4122214708, 0.5363325363, 0.0514459929)),
                        dot(c, float3(0.2119034982, 0.6806995451, 0.1073969566)),
                        dot(c, float3(0.0883024619, 0.2817188376, 0.6299787005)));
    lms = pow(max(lms, 1e-9), 1.0 / 3.0);
    return float3(dot(lms, float3(0.2104542553, 0.7936177850, -0.0040720468)),
                  dot(lms, float3(1.9779984951, -2.4285922050, 0.4505937099)),
                  dot(lms, float3(0.0259040371, 0.7827717662, -0.8086757660)));
}
float3 spring_from_oklab(float3 lab) {
    float3 lms = float3(lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z,
                        lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z,
                        lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z);
    lms = lms * lms * lms;
    return max(float3(dot(lms, float3(4.0767416621, -3.3077115913, 0.2309699292)),
                      dot(lms, float3(-1.2684380046, 2.6097574011, -0.3413193965)),
                      dot(lms, float3(-0.0041960863, -0.7034186147, 1.7076147010))), 0);
}
// x = lightness, y = chroma, z = hue in degrees
float3 spring_lch(float3 c) {
    float3 lab = spring_to_oklab(c);
    return float3(lab.x, length(lab.yz), atan2(lab.z, lab.y) * 57.2957795);
}
float3 spring_from_lch(float3 lch) {
    float h = lch.z / 57.2957795;
    return spring_from_oklab(float3(lch.x, lch.y * cos(h), lch.y * sin(h)));
}
// Integer hash: world cells reach thousands, where sin() hashes band.
float spring_hash(int2 cell, uint salt) {
    uint v = uint(cell.x) * 1597334677u ^ uint(cell.y) * 3812015801u ^ salt * 2654435769u;
    v = v * 747796405u + 2891336453u;
    v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return float((v >> 22u) ^ v) / 4294967295.0;
}
float spring_noise(float2 x, uint salt) {
    int2 i = int2(floor(x));
    float2 f = frac(x);
    f = f * f * (3 - 2 * f);
    return lerp(lerp(spring_hash(i, salt), spring_hash(i + int2(1, 0), salt), f.x),
                lerp(spring_hash(i + int2(0, 1), salt), spring_hash(i + int2(1, 1), salt), f.x), f.y);
}
float spring_fbm(float2 x, uint salt) {
    return spring_noise(x, salt) * 0.57 + spring_noise(x * 2.13 + 7.1, salt + 1) * 0.29 +
           spring_noise(x * 4.37 + 3.7, salt + 2) * 0.14;
}
// Meadow-scale field in world tiles: drifts lie a little along the screen
// horizontal (world 1,1), as the concept's do.
float2 spring_meadow_space(float2 world) {
    float2 axis = float2(world.x + world.y, world.y - world.x) * 0.70710678;
    return float2(axis.x / SPRING_DRIFT_STRETCH, axis.y);
}
// Fresh turf: greens brighten toward young yellow-green; lush hollows deepen.
// Measured on the concept (OKLab): grass h 111 -> 117, L x1.04, C x1.08 with
// roughly twice the mid-scale lightness variation; plains stays straw-led.
float3 spring_ground(float3 albedo, float2 world) {
    float3 lch = spring_lch(albedo);
    float2 m = spring_meadow_space(world);
    float lush = spring_fbm(m * SPRING_LUSH_SCALE + 41.0, 11u) - 0.5;
    float band = smoothstep(60, 80, lch.z) * (1 - smoothstep(150, 172, lch.z));
    float green = smoothstep(96, 108, lch.z);
    float gate = band * smoothstep(0.012, 0.03, lch.y);
    // Pale sand keeps its own cream; straw only freshens slightly.
    float pale = smoothstep(0.74, 0.84, lch.x) * (1 - smoothstep(0.06, 0.10, lch.y));
    gate *= 1 - pale;
    float shift = lerp(SPRING_STRAW_SHIFT, SPRING_GRASS_SHIFT, green) + lush * SPRING_LUSH_HUE * green;
    lch.z += shift * gate;
    lch.y *= lerp(1.0, lerp(SPRING_STRAW_CHROMA, SPRING_GRASS_CHROMA, green) * (1 + lush * SPRING_LUSH_CHROMA), gate);
    lch.x *= lerp(1.0, lerp(1.01, SPRING_GRASS_LIGHT, green) * (1 - lush * SPRING_LUSH_LIGHT), gate);
    return spring_from_lch(lch);
}

// One flower layer: a jittered grid of small blooms whose presence follows the
// drift density. Returns premultiplied colour and coverage.
// Plains share at the pixel being shaded (set by the meadow caller): white and
// yellow blooms vanish against straw, so plains lean to pink, blue and lilac.
static float spring_straw = 0;
float3 spring_bloom_color(float pick, float palette, float accent) {
    // Drift palettes: buttercup gold, daisy white, clover pink, cornflower blue
    // and lilac, each with a white or yellow companion bloom.
    float3 lch;
    float p = frac(palette);
    if (frac(palette * 7.13 + accent * 3.1) < spring_straw * SPRING_STRAW_COLOR)
        p = SPRING_SHARE_WHITE + frac(p * 3.31) * (1 - SPRING_SHARE_WHITE);
    // Woodland edge (palette 2): wood anemone, bluebell, primrose.
    if (palette > 1.5) {
        lch = pick < 0.5 ? SPRING_WHITE : (pick < 0.85 ? float3(0.60, 0.13, 285) : float3(0.92, 0.09, 100));
        lch.x += (accent - 0.5) * 0.06;
        return spring_from_lch(lch * float3(SPRING_BLOOM_LIGHT, SPRING_BLOOM_CHROMA, 1));
    }
    // White petals need a warm cast, or blue skylight turns them lavender.
    float3 white = SPRING_WHITE, yellow = SPRING_YELLOW;
    if (p < SPRING_SHARE_YELLOW)      lch = pick < 0.78 ? yellow : white;
    else if (p < SPRING_SHARE_WHITE)  lch = pick < 0.70 ? white : yellow;
    else if (p < SPRING_SHARE_PINK)   lch = pick < 0.66 ? float3(0.79, 0.105, 355) : (pick < 0.88 ? float3(0.94, 0.03, 20) :
                                            (SPRING_CORAL ? float3(0.75, 0.13, 38) : float3(0.72, 0.13, 345)));
    else if (p < SPRING_SHARE_BLUE)   lch = pick < 0.62 ? float3(0.74, 0.105, 252) : (pick < 0.84 ? float3(0.70, 0.11, 300) : white);
    else                              lch = pick < 0.55 ? float3(0.70, 0.115, 305) : (pick < 0.80 ? float3(0.80, 0.09, 350) : yellow);
    lch.x += (accent - 0.5) * 0.06;
    lch.z += (frac(accent * 7.3) - 0.5) * 12;
    return spring_from_lch(lch * float3(SPRING_BLOOM_LIGHT, SPRING_BLOOM_CHROMA, 1));
}
// Drift identity: the nearest of jittered feature points in meadow space, so
// each drift takes one palette with an even share of every palette.
float spring_drift_palette(float2 m) {
    float2 q = m * SPRING_PALETTE_SCALE;
    int2 base = int2(floor(q));
    float best = 1e9, pick = 0;
    [unroll] for (int y = -1; y <= 1; y++) [unroll] for (int x = -1; x <= 1; x++) {
        int2 cell = base + int2(x, y);
        float2 site = float2(cell) + float2(spring_hash(cell, 61u), spring_hash(cell, 67u));
        float2 delta = site - q;
        float distance2 = dot(delta, delta);
        // Mode 2: a low-discrepancy (R2) pick, so every neighbourhood holds
        // each palette in its share; a uniform hash leaves small views lopsided.
        float r2 = frac(float(cell.x) * 0.7548776662 + float(cell.y) * 0.5698402910 + spring_hash(cell, 71u) * 0.1);
        if (distance2 < best) { best = distance2; pick = SPRING_WORLEY_PALETTE > 1.5 ? r2 : spring_hash(cell, 71u); }
    }
    return pick;
}
// One bloom: `d` from its centre and `radius`, both in the layer's units.
// `sun.xy` is the light direction, `sun.zw` the layer's screen-down direction
// for upright blooms (zero on ground grids). `tuft` keeps the strongest leafy
// shade or contact shadow seen so far.
float4 spring_bloom_shape(float2 d, float radius, float footprint, float h1, float h2, float palette,
                          float4 sun, inout float tuft) {
    // Small upright blooms snap to exact pixel shapes: radii between about .6
    // and 1.2 px spill partial colour into neighbouring pixels (a soft halo).
    // Below 1.4 px a bloom is either one solid pixel or a crisp five-pixel
    // plus with a warm centre pixel.
    bool plus = false;
    if (SPRING_PIXEL_SIZES && dot(sun.zw, sun.zw) > 0.25) {
        float pixels = radius / footprint;
        if (pixels < 1.4) {
            plus = pixels >= SPRING_PLUS_FROM;
            radius = (plus ? 1.15 : 0.55) * footprint;
        }
    }
    // Never thinner than a pixel: keep a bloom's energy instead of letting it
    // sparkle in and out between pixels.
    float shown = max(radius, footprint * 0.55);
    float angle = atan2(d.y, d.x);
    // Mode 1 notches five distinct petals; mode 0 is the earlier soft lobing.
    float petals = plus ? radius :
                   SPRING_PETAL_MODE ? radius * (1 - SPRING_NOTCH + SPRING_NOTCH * abs(cos(2.5 * angle + h1 * 6.28)))
                                     : radius * (0.84 + 0.16 * cos(5 * angle + h1 * 6.28));
    float r = length(d);
    float aa = footprint * SPRING_AA + 1e-4;  // edge softness in pixels
    // The petal outline scaled up to the pixel minimum. (Until v13 this read
    // max(petals, shown): shown >= radius >= petals, so every bloom was a disc.)
    float outline = petals * shown / radius;
    float cover = saturate((outline - r) / aa + 0.5) * min(1, (radius * radius) / (shown * shown));
    // Mode 1: a hard-edged contact shadow cast away from the sun, instead of
    // the soft leafy halo that reads as fluff once blooms are several pixels.
    float2 leafy = d + sun.xy * radius * (SPRING_PETAL_MODE ? 0.45 : 0.7);
    tuft = max(tuft, SPRING_TUFT * (SPRING_PETAL_MODE ? saturate((max(radius, footprint * 0.55) - length(leafy)) / aa + 0.5)
                                                      : saturate(1 - length(leafy) / (max(radius, footprint * 0.5) * 2.0 + aa))));
    // Upright heads stand on a short dark-green stem, drawn screen-down.
    float2 down = sun.zw;
    bool upright = SPRING_STEM > 0 && dot(down, down) > 0.25;
    float along = upright ? dot(d, down) : 0;
    float stem = 0;
    if (upright) {
        float across = abs(dot(d, float2(-down.y, down.x)));
        float width = max(radius * 0.22, footprint * 0.5);
        stem = saturate((width - across) / aa + 0.5) * saturate((along - radius * 0.3) / aa + 0.5) *
               saturate((radius * (0.3 + SPRING_STEM) - along) / aa + 0.5);
    }
    if (cover <= 0 && stem <= 0)
        return 0;
    float4 stalk = float4(spring_from_lch(float3(0.40, 0.09, 132)) * stem, stem);
    if (cover <= 0)
        return stalk;
    float3 color = spring_bloom_color(h2, palette, h1);
    if (SPRING_PETAL_MODE) {
        // Each petal its own shade, tips a little lighter than the heart.
        float petal = floor(frac(angle / 6.2831853 + h1) * 5);
        color *= (1 + (frac(petal * 0.37 + h2) - 0.5) * 0.12) * lerp(0.86, 1.04, saturate(r / max(petals, 1e-5)));
        // A shaded petal rim separates each bloom cleanly from the turf.
        color *= lerp(1, SPRING_RIM, smoothstep(0.62, 1.0, r / max(outline, 1e-5)) * smoothstep(1.5, 3.0, outline / footprint));
    }
    // A warm eye on pale and blue blooms once they are large enough to read.
    float eye = plus ? saturate((footprint * 0.45 - r) / aa + 0.5) :
                saturate((radius * (SPRING_PETAL_MODE == 1 ? 0.27 : 0.32) - r) / aa + 0.5) *
                smoothstep(0.9, 2.2, radius / max(footprint, 1e-4));
    float3 lch = spring_lch(color);
    color = lerp(color, lch.z > 85 && lch.z < 110 && lch.y > 0.1 ? color * float3(0.85, 0.6, 0.35)
                                                                     : spring_from_lch(float3(0.82, 0.15, 88)),
                 eye * (SPRING_PETAL_MODE ? 1.0 : 0.85));
    // Lit upper petals, a softer lower lip; upright heads catch the sky on top.
    color *= upright ? 1 + SPRING_TOP_LIGHT * clamp(-along / radius, -1, 1)
                     : 1 + 0.10 * saturate(dot(normalize(d + 1e-5), sun.xy));
    return float4(color * cover, cover) + stalk * (1 - cover);
}
// Snap a bloom centre to the nearest pixel centre. `q` (this pixel, cell units)
// is a pixel centre and upright grids are exactly proportional to the screen,
// so every pixel finds the same snapped point. A small bloom then fills whole
// pixels instead of smearing partial colour over two to four of them.
float2 spring_snap(float2 centre, float2 q, float cell, float2 down, float snap) {
    if (snap <= 0)
        return centre;
    float2 across = float2(down.y, -down.x);
    float2 delta = (centre - q) * cell;
    float2 steps = floor(float2(dot(delta, across), dot(delta, down)) / snap + 0.5) * snap;
    return q + (across * steps.x + down * steps.y) / cell;
}
// One grid cell's bloom (and its occasional sibling) seen from `q`, the pixel
// in cell units. A forced edge/palette (>= 0) replaces the meadow drift field
// and its palette. `snap` is the pixel size in layer units (0: no snapping).
float4 spring_bloom_cell(int2 id, float2 q, float cell, float footprint, float density_scale, uint salt,
                         float2x2 to_world, float2 shift, float4 sun, float forced_edge, float forced_palette,
                         float snap, inout float tuft) {
    float h0 = spring_hash(id, salt), h1 = spring_hash(id, salt + 17u), h2 = spring_hash(id, salt + 31u);
    float h3 = spring_hash(id, salt + 47u);
    // The drift field is sampled at the cell centre so a bloom is all-or-nothing.
    float2 centre_world = mul(to_world, (float2(id) + 0.5) * cell - shift);
    float2 m = spring_meadow_space(centre_world);
    float drift = spring_fbm(m * SPRING_DRIFT_SCALE, 3u);
    float edge = forced_edge >= 0 ? forced_edge : smoothstep(SPRING_DRIFT_THRESHOLD, SPRING_DRIFT_THRESHOLD + SPRING_DRIFT_SOFT, drift);
    // Drift cores thin toward a ragged fringe; lone blooms dot the open turf.
    float density = (edge * SPRING_DENSITY + (forced_edge >= 0 ? 0 : SPRING_SPRINKLE)) * density_scale;
    // Blooms gather in small clumps inside a drift rather than an even spread.
    float clump = smoothstep(0.3, 0.8, spring_noise(centre_world * SPRING_CLUMP_SCALE, 83u)) * 1.8;
    density *= lerp(1, clump, SPRING_CLUMP);
    if (h0 >= density)
        return 0;
    float palette = SPRING_WORLEY_PALETTE ? spring_drift_palette(m) : spring_fbm(m * SPRING_PALETTE_SCALE + 19.0, 7u) * 1.6 - 0.3;
    // A few blooms stray from a neighbouring drift's palette.
    if (frac(h3 * 91.7) < SPRING_STRAY) palette = frac(h3 * 13.1);
    if (forced_palette >= 0) palette = forced_palette;
    // The centre may sit anywhere in its cell (jitter 1); the 2x2 neighbour
    // search in spring_bloom_layer keeps blooms near an edge whole.
    float2 centre = spring_snap(float2(id) + 0.5 + (float2(h1, h2) - 0.5) * SPRING_JITTER, q, cell, sun.zw, snap);
    float radius = lerp(SPRING_BLOOM_MIN, SPRING_BLOOM_MAX, h3) * 0.5;
    float4 bloom = spring_bloom_shape((q - centre) * cell, radius, footprint, h1, h2, palette, sun, tuft);
    // Meadow flowers grow in little groups: some blooms have a sibling beside them.
    float h4 = spring_hash(id, salt + 59u);
    if (SPRING_BUNCH > 0) {
        // Posies: up to SPRING_BUNCH more heads packed around the first on a
        // golden-angle spiral, just touching, so a bunch reads as one small
        // cluster of crisp heads. Members stay within a cell of their own cell
        // (the 3x3 search in spring_bloom_layer).
        int count = int(h4 * (SPRING_BUNCH + 1));
        float turn = spring_hash(id, salt + 67u) * 6.2831853;
        [unroll] for (int k = 1; k <= 4; k++) {
            if (k > count)
                break;
            float spin = turn + k * 2.39996323;
            float2 offset = float2(cos(spin), sin(spin)) * radius * SPRING_BUNCH_GAP * sqrt(float(k)) / cell;
            float2 member = spring_snap(clamp(centre + offset, float2(id) - 0.9, float2(id) + 1.9), q, cell, sun.zw, snap);
            float hk = spring_hash(id, salt + 71u + uint(k) * 13u);
            float4 head = spring_bloom_shape((q - member) * cell, radius * lerp(0.75, 1.05, hk), footprint,
                                             frac(h1 + k * 0.37), hk, palette, sun, tuft);
            bloom += head * (1 - bloom.a);
        }
    } else if (h4 < SPRING_SIBLING) {
        float h5 = spring_hash(id, salt + 67u), turn = h5 * 6.2831853;
        float2 offset = float2(cos(turn), sin(turn)) * radius * SPRING_SIBLING_GAP / cell;
        float2 twin = spring_snap(clamp(centre + offset, float2(id) + 0.02, float2(id) + 0.98), q, cell, sun.zw, snap);
        float4 second = spring_bloom_shape((q - twin) * cell, radius * lerp(0.7, 1.0, frac(h5 * 7.7)), footprint,
                                           frac(h1 + 0.37), frac(h4 * 13.3 / max(SPRING_SIBLING, 1e-3)), palette, sun, tuft);
        bloom += second * (1 - bloom.a);
    }
    return bloom;
}
// `p` is the layer's own (rotated) grid space; to_world maps a cell centre back.
// `tuft` returns the leafy shade beneath the blooms, offset away from the sun.
float4 spring_bloom_layer(float2 p, float cell, float footprint, float density_scale, uint salt,
                          float2x2 to_world, float2 shift, float4 sun, float forced_edge, float forced_palette,
                          float snap, out float tuft) {
    tuft = 0;
    float2 q = p / cell;
    // The 2x2 cells around the nearest cell corner hold every bloom that can
    // reach this pixel (centres stay inside their cell, radii under half a cell).
    // Posies reach up to a cell beyond their own, so they search 3x3.
    int span = SPRING_BUNCH > 0 ? 3 : 2;
    int2 base = SPRING_BUNCH > 0 ? int2(floor(q)) - 1 : int2(floor(q - 0.5));
    float4 sum = 0;
    [unroll] for (int y = 0; y < 3; y++) [unroll] for (int x = 0; x < 3; x++) {
        if (x >= span || y >= span)
            continue;
        float4 bloom = spring_bloom_cell(base + int2(x, y), q, cell, footprint, density_scale, salt, to_world, shift,
                                         sun, forced_edge, forced_palette, snap, tuft);
        sum += bloom * (1 - sum.a);
    }
    return sum;
}
// Combined meadow: two offset bloom layers (fine and slightly larger) plus a
// darker leafy ground under each drift. `density_scale` scales per biome;
// `height` is the ground's world z, which lifts it on screen.
float4 spring_meadow(float2 world, float height, float density_scale, float2 sun_direction, out float leaf) {
    float footprint = max(length(fwidth(world)), 1e-5) * 0.7071;
    float2 sun = normalize(sun_direction + 1e-5);
    float2 m = spring_meadow_space(world);
    float drift = spring_fbm(m * SPRING_DRIFT_SCALE, 3u);
    leaf = smoothstep(SPRING_DRIFT_THRESHOLD - 0.06, SPRING_DRIFT_THRESHOLD + SPRING_DRIFT_SOFT, drift) * density_scale;
    float tuft_a, tuft_b;
    // The second grid is rotated 37 degrees so the two never line up.
    float2x2 turn = float2x2(0.80, -0.60, 0.60, 0.80);
    float2 shift = float2(0.37, 0.53);
    float4 a, b;
    if (SPRING_PETAL_MODE > 1.5) {
        // Upright blooms facing the camera. Flat on the ground the isometric
        // view squashed them 2:1 into soft smudges; here the bloom grid lives
        // in screen-proportional units (one unit = one pixel in either screen
        // axis), so blooms are round and edges are exactly half a pixel soft.
        float2x2 to_screen = float2x2(0.70710678, 0.70710678, 0.35355339, -0.35355339);
        float2x2 to_ground = float2x2(0.70710678, 1.41421356, 0.70710678, -1.41421356);
        float2 s = mul(to_screen, world);
        // Height lifts ground on screen by 0.41 tile widths per world z (0.58
        // screen units). Folding it in keeps blooms round and evenly spread on
        // slopes too; flat-ground mapping stretched them on camera-facing hills
        // and squashed them on hills facing away.
        float2 lift = float2(0, SPRING_SLOPE ? 0.57983 * height : 0);
        s -= lift;
        float pixel = max(max(fwidth(s.x), fwidth(s.y)), 1e-5);
        // Flowers grow per patch of ground, not per pixel: scale the screen-grid
        // density by the ground area under each pixel against flat ground's.
        float2 wx = ddx(world), wy = ddy(world);
        density_scale *= SPRING_SLOPE ? clamp(abs(wx.x * wy.y - wx.y * wy.x) / (2 * pixel * pixel), 0.4, 2.5) : 1;
        // Tilt both grids off the screen axes, where a lattice reads as rows.
        float2x2 tilt = float2x2(cos(SPRING_GRID_TURN), -sin(SPRING_GRID_TURN), sin(SPRING_GRID_TURN), cos(SPRING_GRID_TURN));
        float2x2 tilt_b = mul(turn, tilt);
        float2 sun_s = normalize(mul(to_screen, sun) + 1e-5);
        // A screen cell of k*CELL covers 2k^2 CELL^2 of ground: keep blooms per tile.
        float cell = SPRING_CELL * SPRING_UPRIGHT_CELL;
        float per_tile = 2 * SPRING_UPRIGHT_CELL * SPRING_UPRIGHT_CELL;
        // Cell centres map back to ground through the same lift (local height).
        a = spring_bloom_layer(mul(tilt, s), cell, pixel, density_scale * per_tile, 101u, mul(to_ground, transpose(tilt)),
                               -mul(tilt, lift), float4(mul(tilt, sun_s), mul(tilt, float2(0, 1))), -1, -1,
                               SPRING_SNAP ? pixel : 0, tuft_a);
        b = spring_bloom_layer(mul(tilt_b, s) + shift, cell * 1.37, pixel, density_scale * per_tile * SPRING_SECOND_LAYER,
                               211u, mul(to_ground, transpose(tilt_b)), shift - mul(tilt_b, lift),
                               float4(mul(tilt_b, sun_s), mul(tilt_b, float2(0, 1))), -1, -1,
                               SPRING_SNAP ? pixel : 0, tuft_b);
    } else {
        a = spring_bloom_layer(world, SPRING_CELL, footprint, density_scale, 101u,
                               float2x2(1, 0, 0, 1), 0, float4(sun, 0, 0), -1, -1, 0, tuft_a);
        b = spring_bloom_layer(mul(turn, world) + shift, SPRING_CELL * 1.37, footprint,
                               density_scale * SPRING_SECOND_LAYER, 211u, transpose(turn), shift, float4(sun, 0, 0), -1, -1, 0, tuft_b);
    }
    // `leaf` carries both the drift's general shade and each bloom's tuft.
    leaf = saturate(leaf * SPRING_LEAF_DRIFT + max(tuft_a, tuft_b));
    return a + b * (1 - a.a);
}
// Woodland flowers along a forest floor's thinning edge (`fringe` 0..1).
float4 spring_woodland(float2 world, float fringe, float2 sun_direction) {
    float footprint = max(length(fwidth(world)), 1e-5) * 0.7071;
    float2 sun = normalize(sun_direction + 1e-5);
    float tuft;
    return spring_bloom_layer(world + 0.21, SPRING_CELL, footprint, SPRING_WOODLAND, 307u,
                              float2x2(1, 0, 0, 1), -0.21, float4(sun, 0, 0), fringe, 2, 0, tuft);
}
// ---------------------------------------------------------------------------
// Per-tree palette; the atlas keeps its own light/dark leaf structure.
float3 spring_leaf_color(float3 albedo, float3 average, float seed) {
    float3 pal;
    if (seed < 0.45) pal = float3(0.6, 0.13, 127);
    else if (seed < 0.85) pal = float3(0.58, 0.125, 130);
    else pal = float3(0.63, 0.135, 124);
    pal.z += (frac(seed * 7.31) - 0.5) * SPRING_TREE_HUE_SPREAD;
    pal.x += (frac(seed * 13.7) - 0.5) * 0.05;
    float3 src = spring_lch(albedo), avg = spring_lch(average);
    float L = pal.x * SPRING_LEAF_LIGHT + (src.x - avg.x) * SPRING_LEAF_DETAIL;
    float C = pal.y * SPRING_LEAF_CHROMA * saturate(0.45 + 0.75 * src.x / max(avg.x, 0.05));
    float h = pal.z + clamp(src.z - avg.z, -25, 25) * 0.3;
    return spring_from_lch(float3(max(L, 0.05), C, h));
}
