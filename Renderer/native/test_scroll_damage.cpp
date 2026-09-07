#include "scroll_damage.h"
#include "river_node_locality.h"
#include "pixel_block_cache.h"
#include <cassert>
using namespace c3x_renderer;
int main() {
    {
        std::vector<TileFootprint> source = {{1,11,0,0,{-2,-10,130,130}}, {2,22,128,0,{-2,0,130,130}}};
        PixelRect block{0,0,128,128};
        auto key = block_key(source,block);
        assert(key.size()==2); // neighboring overhang is a dependency
        auto shifted=source;
        for(auto & tile:shifted) { tile.anchor_x-=256; tile.anchor_y+=128; }
        assert(key==block_key(shifted,{-256,128,-128,256}));
        shifted=source; shifted[1].mesh++;
        assert(key!=block_key(shifted,block));
        shifted=source; shifted.pop_back();
        assert(key!=block_key(shifted,block));
        shifted=source; std::reverse(shifted.begin(),shifted.end());
        assert(key!=block_key(shifted,block)); // alpha/depth ties preserve draw order
        assert(block_floor(-1,0)==-128 && block_floor(127,0)==0 && block_floor(130,2)==130);
        assert(power_of_two_extent(1121)==2048 && power_of_two_extent(128)==128);
        std::vector<PixelRect> parts={{0,0,64,64},{64,0,128,64},{0,64,128,128},{128,64,192,128}};
        merge_adjacent_rectangles(parts);
        int area=0;
        for(auto rect:parts) {
            area+=(rect.right-rect.left)*(rect.bottom-rect.top);
            assert(!(rect.left<=128 && rect.right>128 && rect.top<64)); // hole stays empty
        }
        assert(area==128*128+64*64);
        PixelBlockCache cache;
        for(unsigned i=0;i<512;++i) {
            PixelBlock image;
            image.key={{i+1,0,0}};
            image.pixels.assign(128*128,i);
            cache.insert(std::move(image));
            assert(cache.bytes<=PixelBlockCache::budget && cache.blocks.size()<=256);
        }
        assert(cache.find({{1,0,0}})<0 && cache.find({{512,0,0}})>=0);
        auto bytes=cache.bytes;
        PixelBlock duplicate; duplicate.key={{512,0,0}}; duplicate.pixels.assign(128*128,0);
        cache.insert(std::move(duplicate));
        assert(cache.bytes==bytes);
        cache.clear();
        assert(cache.find({{512,0,0}})<0 && cache.bytes<=256*sizeof(PixelBlock));
    }
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
    // Compare full-world and culled nearest-node fields after interpolation,
    // including nodes just outside the cutoff and triangles at the tile edge.
    for (int width : {64,128,256}) {
        RiverNodeWindow window(width,width/2);
        unsigned seed = 7117;
        for (int sample = 0; sample < 100; ++sample) {
            std::vector<std::pair<int,int>> nodes;
            for (int n = 0; n < 64; ++n) {
                seed = seed*1664525u+1013904223u; int x = static_cast<int>((seed>>16)%33)-16;
                seed = seed*1664525u+1013904223u; int y = static_cast<int>((seed>>16)%33)-16;
                nodes.push_back({x,y});
            }
            auto nearest = [&](float u, float v, bool culled) {
                float distance = 1000.0f;
                for (auto node : nodes) {
                    if (culled && !window.contains(node.first,node.second)) continue;
                    float x = (u-v-node.first)*window.half_width;
                    float y = (u+v-1-node.second)*window.half_height;
                    distance = std::min(distance,std::sqrt(x*x+y*y));
                }
                return distance;
            };
            for (int row=0; row<8; ++row) for (int col=0; col<8; ++col) {
                float full[3]={nearest(col/8.0f,row/8.0f,false),nearest((col+1)/8.0f,row/8.0f,false),nearest((col+1)/8.0f,(row+1)/8.0f,false)};
                float local[3]={nearest(col/8.0f,row/8.0f,true),nearest((col+1)/8.0f,row/8.0f,true),nearest((col+1)/8.0f,(row+1)/8.0f,true)};
                for (int i=0;i<=4;++i) for(int j=0;j<=4-i;++j) {
                    float a=i/4.0f,b=j/4.0f,c=1-a-b;
                    // Below the largest effect cutoff, the actual interpolated
                    // distance must be identical; above it both responses vanish.
                    float d0=full[0]*a+full[1]*b+full[2]*c;
                    float d1=local[0]*a+local[1]*b+local[2]*c;
                    assert((d0>=24 && d1>=24) || std::abs(d0-d1)<0.0001f);
                }
            }
        }
    }

}
