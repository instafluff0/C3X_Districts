"""The route witness hashes a publication's tiles once, not on every frame.

With C3X_RENDERER_ROUTE_WITNESS=1 every displayed frame hashed every captured
tile byte (about 2.5 MB with the world window's capture margin), which cost
1-3 ms a frame and made the window look 10-18% slower at idle than it is
(performance review, section 52). The digest itself must not change.
"""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class RouteSourceDigestTests(unittest.TestCase):
    def test_tiles_are_hashed_once_per_publication_with_the_same_digest(self):
        source = (ROOT / 'Renderer/sandbox/resident_scene.cpp').read_text()
        start = source.index('        auto captured=frame;captured.tiles=nullptr;')
        end = source.index('        auto const& records=sandbox_direct_units.prepared_units;', start)
        block = source[start:end]
        run_cpp(r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include "Renderer/native/c3x_renderer_api.h"
struct RouteSource {std::int64_t serial=0;c3x_renderer_frame_v1 key{};std::uint64_t digest=0;bool valid=false;};
struct Renderer {std::int64_t route_map_serial=1;RouteSource route_source;} renderer;
std::size_t hashed=0;
std::uint64_t digest_of(c3x_renderer_frame_v1 const& frame){
 std::uint64_t source=14695981039346656037ull;
 auto hash=[](std::uint64_t& value,void const* data,std::size_t size){hashed+=size;
  auto bytes=static_cast<unsigned char const*>(data);for(std::size_t i=0;i<size;++i){value^=bytes[i];value*=1099511628211ull;}};
''' + block + r'''
 return source;}
int main(){
 std::vector<c3x_renderer_tile_v1> tiles(300);for(unsigned i=0;i<tiles.size();++i){tiles[i].tile_x=int(i);tiles[i].terrain_type=int(i%7);}
 c3x_renderer_frame_v1 frame{};frame.api_version=C3X_RENDERER_API_VERSION;frame.struct_size=sizeof(frame);
 frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());frame.target_width=800;
 auto first=digest_of(frame);auto bytes=hashed;assert(bytes>=tiles.size()*sizeof(tiles[0]));
 // Later frames of the same publication (only presentation fields move).
 for(int n=0;n<10;++n){frame.presentation_time_ticks=1000+n;frame.dirty_flags=n;assert(digest_of(frame)==first);}
 assert(hashed==bytes);
 // A new publication, a changed view or new storage hashes again.
 renderer.route_map_serial=2;assert(digest_of(frame)==first);assert(hashed==2*bytes);
 frame.target_width=801;auto other=digest_of(frame);assert(other!=first&&hashed>2*bytes);
 auto copy=tiles;frame.tiles=copy.data();frame.target_width=800;auto before=hashed;
 assert(digest_of(frame)==first&&hashed>before);
}
''')


if __name__ == '__main__':
    unittest.main()
