#pragma once
#include <d3d11.h>
#include <d3dcompiler.h>
#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>
#include "render_core/linear_target.h"
#include "render_core/scene_provenance.h"
#include "gpu_image_commands.h"

namespace c3x_renderer {
// Immutable raw scene samples, borrowed from one exact published map. Only
// selected rectangles are retained; working/finishing attachments remain shared.
struct UnitSceneRegion {
    render_core::LinearTarget base;
    std::shared_ptr<std::size_t> charge;
    std::size_t bytes=0;
    int width=0,height=0;
    ~UnitSceneRegion(){if(charge)*charge-=bytes;}
};
struct UnitSceneSource {
    using Rect=c3x_gpu_images::Rect;
    std::function<std::shared_ptr<UnitSceneRegion>(Rect)> capture;
};
using UnitSceneProvenance=render_core::SceneProvenance<UnitSceneSource,c3x_gpu_images::Rect>;
}
