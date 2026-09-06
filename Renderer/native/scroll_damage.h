#ifndef C3X_RENDERER_SCROLL_DAMAGE_H
#define C3X_RENDERER_SCROLL_DAMAGE_H

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace c3x_renderer {

struct PixelRect { int left, top, right, bottom; };
struct TileFootprint {
    std::uint64_t coordinate = 0, mesh = 0;
    int anchor_x = 0, anchor_y = 0;
    PixelRect bounds = {};
};

// Tile jumps change the captured set. Compare the actual compiled versions,
// including their neighbor dependencies, then repaint every old/new changed
// footprint. Unchanged pixels can move even when an entire tile strip enters.
// Rectangles are disjoint: blending the same shadow twice would darken seams.
inline bool scroll_damage(std::vector<TileFootprint> const & previous,
                          std::vector<TileFootprint> const & current,
                          int width, int height, int & dx, int & dy,
                          std::vector<PixelRect> & rectangles) {
    if (width <= 0 || height <= 0 || previous.empty() || current.empty()) return false;
    std::unordered_map<std::uint64_t, TileFootprint const *> old_tiles, new_tiles;
    for (auto const & tile : previous)
        if (!old_tiles.emplace(tile.coordinate, &tile).second) return false;
    bool found_translation = false;
    for (auto const & tile : current) {
        if (!new_tiles.emplace(tile.coordinate, &tile).second) return false;
        auto old = old_tiles.find(tile.coordinate);
        if (old == old_tiles.end()) continue;
        int x = tile.anchor_x - old->second->anchor_x;
        int y = tile.anchor_y - old->second->anchor_y;
        if (found_translation && (x != dx || y != dy)) return false;
        dx = x; dy = y; found_translation = true;
    }
    if (!found_translation || dx <= -width || dx >= width || dy <= -height || dy >= height)
        return false;
    // A coarse coverage grid bounds bookkeeping and conservatively includes
    // derivative quads at damage boundaries; no persistent per-pixel metadata.
    constexpr int cell = 32;
    int columns = (width + cell - 1) / cell, rows = (height + cell - 1) / cell;
    std::vector<unsigned char> dirty(static_cast<std::size_t>(columns) * rows, 0);
    auto mark = [&](PixelRect rect) {
        rect.left = std::clamp(rect.left, 0, width);
        rect.right = std::clamp(rect.right, 0, width);
        rect.top = std::clamp(rect.top, 0, height);
        rect.bottom = std::clamp(rect.bottom, 0, height);
        if (rect.left >= rect.right || rect.top >= rect.bottom) return;
        for (int y = rect.top / cell; y <= (rect.bottom - 1) / cell; ++y)
            for (int x = rect.left / cell; x <= (rect.right - 1) / cell; ++x)
                dirty[static_cast<std::size_t>(y) * columns + x] = 1;
    };
    auto footprint = [](TileFootprint const & tile, int x, int y) {
        return PixelRect{tile.bounds.left + tile.anchor_x + x, tile.bounds.top + tile.anchor_y + y,
            tile.bounds.right + tile.anchor_x + x, tile.bounds.bottom + tile.anchor_y + y};
    };
    mark({0, 0, std::max(0, dx), height});
    mark({std::min(width, width + dx), 0, width, height});
    mark({0, 0, width, std::max(0, dy)});
    mark({0, std::min(height, height + dy), width, height});
    for (auto const & old : previous) {
        auto next = new_tiles.find(old.coordinate);
        if (next == new_tiles.end() || next->second->mesh != old.mesh)
            mark(footprint(old, dx, dy));
    }
    for (auto const & tile : current) {
        auto old = old_tiles.find(tile.coordinate);
        if (old == old_tiles.end() || old->second->mesh != tile.mesh)
            mark(footprint(tile, 0, 0));
    }
    rectangles.clear();
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < columns;) {
            if (!dirty[static_cast<std::size_t>(y) * columns + x]) { ++x; continue; }
            int start = x++;
            while (x < columns && dirty[static_cast<std::size_t>(y) * columns + x]) ++x;
            PixelRect rect{start * cell, y * cell, std::min(width, x * cell), std::min(height, (y+1) * cell)};
            bool merged = false;
            for (auto & earlier : rectangles)
                if (earlier.bottom == rect.top && earlier.left == rect.left && earlier.right == rect.right) {
                    earlier.bottom = rect.bottom; merged = true; break;
                }
            if (!merged) rectangles.push_back(rect);
        }
    }
    return true;
}

} // namespace c3x_renderer
#endif
