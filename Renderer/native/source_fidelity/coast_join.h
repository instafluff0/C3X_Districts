#pragma once
#include <algorithm>
namespace c3x_renderer { namespace fidelity {
inline float coast_ramp(float value) {
    value=std::clamp(value,0.f,1.f);
    return value*value*(3-2*value);
}
inline float coast_coverage(float distance,float beach_width) {
    // Begin within the dry sand instead of waiting until the beach material
    // has already disappeared. Keep the inland endpoint before relief rises,
    // and a sand-only optical edge even on the narrowest lowland beaches.
    float start=std::max(.025f,beach_width-.10f);
    return coast_ramp((distance-start)/(beach_width+.22f-start));
}
// Desert is already an authored sand surface. Keep it visible to the optical
// coast instead of exposing the broad generic beach join used by vegetation.
inline float desert_coast_coverage(float distance) {
    return coast_ramp((distance-.01f)/.11f);
}
// The retained beach is flat. Finish the material fade before raising the
// replacement surface, so alpha clipping cannot expose an elevated dune edge.
// Both endpoint slopes are zero; authored inland relief is unchanged.
inline float coast_relief(float distance,float beach_width) {
    return coast_ramp((distance-beach_width-.22f)/.36f);
}
// Mountains and volcanoes meet water as a rocky coast where their rock stands
// (stone: the stamp's own rise in units, before coast or river scaling), as
// hills do behind a rocky shore: the cover is complete .04 tiles from the
// water and the rise follows over .22 tiles. Behind the beach ramp a one-tile
// island's shore (~.33 tiles out) left its peak none of its height; the hills'
// .14-tile cliff kept it but stood it up as a sheer column. Their low foot keeps the beach.
// Equal ends return exactly, so inland rock and full cover stay bit-identical.
inline float mountain_coast_rock(float stone) {
    return coast_ramp((stone-8.f)/24.f);
}
inline float mountain_coast_relief(float distance,float beach_width,float stone) {
    float beach=coast_relief(distance,beach_width);
    return beach+(coast_ramp((distance-.04f)/.22f)-beach)*mountain_coast_rock(stone);
}
inline float mountain_coast_coverage(float coverage,float distance,float stone) {
    return coverage+(coast_ramp(distance/.04f)-coverage)*mountain_coast_rock(stone);
}
} }
