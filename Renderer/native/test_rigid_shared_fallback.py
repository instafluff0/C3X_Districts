"""A rigid draw outside the shared-instance union falls back, not fails.

The resident shared-instance union is built from the frame's geometry draws.
The dynamic scene pass can submit a rigid feature (an authored bridge over a
river) with its own occurrence translation, whose key that union never
registered. The rigid flush rejected the whole frame
(C3X_RENDERER_RESULT_DEVICE_ERROR). Such a draw now uses the explicit instance
stream, as every rigid draw did before shared submission. Found draws keep the
shared ranges, per-draw parameters are uploaded when any draw needs them, and
a merged instanced run never mixes the two sources.
"""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class RigidSharedFallbackTests(unittest.TestCase):
    def test_missing_shared_key_uses_explicit_stream_and_keeps_found_draws_shared(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        start = source.index('            // Draws found in the resident shared-instance union use it.')
        end = source.index('            if(!rigid_instances.empty() && !rigid_sources.stream.upload', start)
        routing = source[start:end]
        run_cpp(r'''
#include <array>
#include <cassert>
#include <cstdio>
#include <memory>
#include <vector>
struct Instance {float projection[4]{},view[4]{};};
struct Mesh {bool rigid_source=false;std::shared_ptr<std::vector<Instance>> instances;float instance_material=0;};
struct Draw {Mesh mesh;float projection[4]{};int key=0;
 Mesh const& content()const{return mesh;}float const (&natural_projection()const)[4]{return projection;}};
struct Range {unsigned first=0,count=0;explicit operator bool()const{return count!=0;}};
struct Front {std::vector<std::pair<int,Range>> ranges;
 Range find(int key)const{for(auto const& r:ranges)if(r.first==key)return r.second;return {};}};
struct Parameters {static constexpr unsigned limit=8;};
struct Settings {float translation[2]{};float depth_translation=0;};
namespace c3x_renderer {namespace fidelity {using MeshInstance=Instance;}}
struct Harness {
 int layer=8;bool streamed=true;std::vector<Draw> selected;std::array<Settings,Parameters::limit> parameters{};
 std::array<int,Parameters::limit> packets{};std::shared_ptr<Front> shared_front;
 unsigned shared_instance_fallbacks=0,rejections=0,uploads=0;
 struct {bool upload(Settings const*,unsigned){return ++*count,true;}unsigned* count;}draw_parameters{&uploads};
 int shared_instance_draw_key(unsigned,Draw const& draw)const{return draw.key;}
 bool reject_shared_instance_range(unsigned,Draw const&,std::shared_ptr<Front> const&,Range){++rejections;return false;}
 std::vector<Instance> rigid_instances;std::vector<unsigned> rigid_indices;std::array<unsigned,Parameters::limit> rigid_offsets{};
 bool route(std::array<bool,Parameters::limit>& out){
''' + routing + r'''
  out=rigid_shared;return true;
 }
};
Draw rigid(int key,float x){Draw d;d.mesh.rigid_source=true;d.mesh.instances=std::make_shared<std::vector<Instance>>(1);
 d.mesh.instance_material=13;d.key=key;d.projection[0]=x;return d;}
int main(){
 Harness h;h.shared_front=std::make_shared<Front>();h.shared_front->ranges={{1,{5,1}},{3,{9,1}}};
 h.selected={rigid(1,.1f),rigid(2,.2f),rigid(3,.3f)};
 h.parameters[1].translation[0]=-64;h.parameters[1].translation[1]=48;h.parameters[1].depth_translation=7;
 std::array<bool,Parameters::limit> shared{};
 // The bridge (key 2) is outside the union: the frame continues.
 assert(h.route(shared));
 assert(shared[0]&&!shared[1]&&shared[2]);
 assert((h.rigid_indices==std::vector<unsigned>{5,9})&&h.rigid_offsets[0]==0&&h.rigid_offsets[2]==1);
 assert(h.rigid_instances.size()==1&&h.rigid_offsets[1]==0);
 auto const& explicit_bridge=h.rigid_instances[0];
 assert(explicit_bridge.view[0]==-64&&explicit_bridge.view[1]==48&&explicit_bridge.view[2]==7&&explicit_bridge.view[3]==13);
 assert(explicit_bridge.projection[0]==.2f);
 assert(h.uploads==1&&h.rejections==1&&h.shared_instance_fallbacks==1); // explicit draws need per-draw parameters
 // All found: shared ranges only, and no per-draw parameter upload.
 Harness all;all.shared_front=h.shared_front;all.selected={rigid(1,0),rigid(3,0)};
 assert(all.route(shared)&&shared[0]&&shared[1]&&all.rigid_instances.empty()&&all.uploads==0&&all.rejections==0);
 // No resident union at all: every rigid draw is explicit, as before.
 Harness none;none.selected={rigid(1,0),rigid(2,0)};
 assert(none.route(shared)&&!shared[0]&&!shared[1]&&none.rigid_instances.size()==2&&none.uploads==1);
 std::puts("PASS rigid shared fallback: missing key explicit, found keys shared, parameters only when needed");
}
''')
        # Issue and merge follow each draw's own route.
        flush = source[end:source.index('            selected.clear();return true;', end)]
        self.assertIn('bool shared=shared_front && chunk.content().rigid_source && rigid_shared[index];', flush)
        self.assertIn('rigid_shared[end]!=rigid_shared[i]', flush)
        self.assertNotIn('IASetInputLayout(shared_front?', flush)


if __name__ == '__main__':
    unittest.main()
