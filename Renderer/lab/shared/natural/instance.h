#pragma once
#include <array>
#include <vector>
namespace c3x_renderer { namespace fidelity {
// Immutable world placement; the last two vectors are filled by selected passes.
struct MeshInstance {float place[8]={};float projection[4]={};float view[4]={};};
static_assert(sizeof(MeshInstance)==64,"instance stream ABI");
} }
