#ifndef C3X_RENDERER_RIVER_NODE_LOCALITY_H
#define C3X_RENDERER_RIVER_NODE_LOCALITY_H
#include <cmath>
namespace c3x_renderer {
// All three frozen shader node effects are zero by 24 pixels. If any point
// responds, its node is within 24 + the tile diameter of every vertex. Nodes
// beyond this expanded bounding box therefore cannot change an interpolated
// response, even when a far vertex's raw nearest-node distance changes.
struct RiverNodeWindow {
    float half_width, half_height, radius;
    RiverNodeWindow(int width, int height)
        : half_width(width*0.5f), half_height(height*0.5f),
          radius(25.0f + std::sqrt(static_cast<float>(width)*width + static_cast<float>(height)*height)) {}
    bool contains(int delta_x, int delta_y) const {
        return std::abs(static_cast<float>(delta_x))*half_width <= half_width+radius &&
            std::abs(static_cast<float>(delta_y))*half_height <= half_height+radius;
    }
};
}
#endif
