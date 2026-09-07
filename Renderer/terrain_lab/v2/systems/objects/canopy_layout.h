#pragma once
#include <cstdint>
#include <numeric>
#include <vector>
#include <utility>

namespace canopy {
// Only canonical source tile identity and stand size enter this layout. A
// permutation preserves stratified canopy coverage while moving fixed source
// anchors (and missing anchors) out of their repeated first-row grid slots.
inline std::vector<unsigned> slots(std::uint32_t seed, unsigned count) {
    std::vector<unsigned> result(count);
    std::iota(result.begin(), result.end(), 0u);
    for (unsigned i = count; i > 1; --i) {
        std::uint32_t key = seed ^ (i * 0x9e3779b9u) ^ 0xa17c9e53u;
        key ^= key >> 16; key *= 0x7feb352du;
        key ^= key >> 15; key *= 0x846ca68bu;
        key ^= key >> 16;
        std::swap(result[i - 1], result[key % i]);
    }
    return result;
}
}
