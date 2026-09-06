#include "scroll_damage.h"
#include <cassert>
using namespace c3x_renderer;
int main() {
    std::vector<TileFootprint> old = {
        {1, 1, 64, 64, {-20, -40, 160, 90}},
        {2, 2, 320, 128, {-10, -50, 140, 80}},
        {3, 3, 700, 256, {-10, -40, 140, 80}}};
    for (auto delta : {PixelRect{256,0,0,0}, PixelRect{-256,0,0,0},
                       PixelRect{0,128,0,0}, PixelRect{0,-128,0,0},
                       PixelRect{256,128,0,0}, PixelRect{0,0,0,0}}) {
        auto current = old;
        for (auto & tile : current) { tile.anchor_x += delta.left; tile.anchor_y += delta.top; }
        current.erase(current.begin()); // removed overhanging feature
        current[0].mesh = 4; // terrain/neighbor mutation
        current.push_back({4, 5, 550, 320, {-40,-80,160,100}});
        int dx = 0, dy = 0;
        std::vector<PixelRect> rectangles;
        assert(scroll_damage(old, current, 1024, 512, dx, dy, rectangles));
        assert(dx == delta.left && dy == delta.top);
        int unchanged = 0;
        for (int y = 0; y < 512; ++y) for (int x = 0; x < 1024; ++x) {
            int coverage = 0;
            for (auto rect : rectangles)
                coverage += x >= rect.left && x < rect.right && y >= rect.top && y < rect.bottom;
            assert(coverage <= 1); // transparency may be composited only once
            bool needed = x-dx < 0 || x-dx >= 1024 || y-dy < 0 || y-dy >= 512;
            auto inside = [&](TileFootprint tile, int offset_x, int offset_y) {
                return x >= tile.anchor_x + offset_x + tile.bounds.left &&
                    x < tile.anchor_x + offset_x + tile.bounds.right &&
                    y >= tile.anchor_y + offset_y + tile.bounds.top &&
                    y < tile.anchor_y + offset_y + tile.bounds.bottom;
            };
            needed = needed || inside(old[0],dx,dy) || inside(old[1],dx,dy) ||
                inside(current[0],0,0) || inside(current[2],0,0);
            assert(!needed || coverage == 1);
            unchanged += coverage == 0;
        }
        assert(unchanged > 0);
    }
    auto current = old;
    int dx=0,dy=0; std::vector<PixelRect> rectangles;
    current[0].anchor_x++;
    assert(!scroll_damage(old,current,1024,512,dx,dy,rectangles));
    current = old; current.push_back(old[0]);
    assert(!scroll_damage(old,current,1024,512,dx,dy,rectangles));
    current = old;
    for (auto & tile : current) tile.anchor_x += 1024;
    assert(!scroll_damage(old,current,1024,512,dx,dy,rectangles));
}
