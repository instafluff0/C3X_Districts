#pragma once
// Test-only captured native demands shared by the existing replay/visual harness.
#include "c3x_renderer_api.h"
#include <vector>
#include <climits>
struct NativeFrameSample {
    int workload=0,step=0;
    c3x_renderer_frame_v1 frame={};
    std::vector<c3x_renderer_tile_v1> tiles;
};
