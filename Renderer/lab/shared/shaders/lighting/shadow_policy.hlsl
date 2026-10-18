// Shared scalar policy; also included inside the CPU lighting namespace.
// Dynamic ground footprints (units, animated resources) blend over a cached
// background. They remove the share of open flat-ground light that a world
// shadow removes, so terrain, forest, city, unit and resource shadows match.
// Inputs are the luminance of frame_illumination with and without shadow.
inline float c3x_dynamic_shadow_opacity(float open_ground, float shaded_ground) {
    return open_ground > .0001f ? 1 - shaded_ground / open_ground : 0;
}
