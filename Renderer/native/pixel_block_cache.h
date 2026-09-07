#pragma once

#include "scroll_damage.h"

namespace c3x_renderer {

constexpr int pixel_block_side = 128;

struct BlockContributor {
    std::uint64_t mesh;
    int x, y;
    bool operator==(BlockContributor const & other) const {
        return mesh == other.mesh && x == other.x && y == other.y;
    }
};
using BlockKey = std::vector<BlockContributor>;

inline BlockKey block_key(std::vector<TileFootprint> const & tiles, PixelRect rect) {
    BlockKey key;
    for (auto const & tile : tiles) {
        if (tile.bounds.right + tile.anchor_x <= rect.left ||
            tile.bounds.left + tile.anchor_x >= rect.right ||
            tile.bounds.bottom + tile.anchor_y <= rect.top ||
            tile.bounds.top + tile.anchor_y >= rect.bottom) continue;
        // Preserve draw order as well as every contributor, including overhangs.
        key.push_back({tile.mesh, tile.anchor_x-rect.left, tile.anchor_y-rect.top});
    }
    return key;
}

inline int block_floor(int value, int phase) {
    int remainder = (value-phase) % pixel_block_side;
    if (remainder < 0) remainder += pixel_block_side;
    return value-remainder;
}

inline int power_of_two_extent(int value) {
    int extent = 1;
    while (extent < value) extent *= 2;
    return extent;
}

// Input rectangles are disjoint. Two sorted passes coalesce shared full edges
// without filling holes, changing coverage, or a quadratic pair search.
template<typename Rect>
inline void merge_adjacent_rectangles(std::vector<Rect> & rectangles) {
    std::sort(rectangles.begin(),rectangles.end(),[](auto a,auto b) {
        if(a.top!=b.top) return a.top<b.top;
        if(a.bottom!=b.bottom) return a.bottom<b.bottom;
        return a.left<b.left;
    });
    std::size_t count=0;
    for(auto rect:rectangles) {
        if(count && rectangles[count-1].top==rect.top && rectangles[count-1].bottom==rect.bottom && rectangles[count-1].right==rect.left)
            rectangles[count-1].right=rect.right;
        else rectangles[count++]=rect;
    }
    rectangles.resize(count);
    std::sort(rectangles.begin(),rectangles.end(),[](auto a,auto b) {
        if(a.left!=b.left) return a.left<b.left;
        if(a.right!=b.right) return a.right<b.right;
        return a.top<b.top;
    });
    count=0;
    for(auto rect:rectangles) {
        if(count && rectangles[count-1].left==rect.left && rectangles[count-1].right==rect.right && rectangles[count-1].bottom==rect.top)
            rectangles[count-1].bottom=rect.bottom;
        else rectangles[count++]=rect;
    }
    rectangles.resize(count);
}

struct PixelBlock {
    BlockKey key;
    std::vector<std::uint32_t> pixels;
    std::uint64_t age = 0;
    std::size_t bytes() const {
        return key.capacity()*sizeof(BlockContributor) + pixels.capacity()*sizeof(std::uint32_t);
    }
};

class PixelBlockCache {
public:
    constexpr static std::size_t budget = 16u*1024u*1024u;
    std::vector<PixelBlock> blocks;
    std::size_t bytes = 0;
    std::uint64_t age = 0;

    int find(BlockKey const & key) {
        for (std::size_t i=0; i<blocks.size(); ++i) if (blocks[i].key == key) {
            blocks[i].age = ++age;
            return static_cast<int>(i);
        }
        return -1;
    }
    void insert(PixelBlock block) {
        if (block.pixels.size() != pixel_block_side*pixel_block_side || find(block.key) >= 0) return;
        if (blocks.capacity() < 256u) {
            blocks.reserve(256u);
            bytes = blocks.capacity()*sizeof(PixelBlock);
        }
        std::size_t size = block.bytes();
        if (size > budget) return;
        while (bytes+size > budget && !blocks.empty()) {
            auto oldest = std::min_element(blocks.begin(), blocks.end(), [](auto const & a, auto const & b) { return a.age < b.age; });
            bytes -= oldest->bytes();
            blocks.erase(oldest);
        }
        block.age = ++age;
        blocks.push_back(std::move(block));
        bytes += size;
    }
    void clear() { blocks.clear(); bytes = blocks.capacity()*sizeof(PixelBlock); age = 0; }
};

} // namespace c3x_renderer
