"""Execute production phase uploads across flushes and authored phase changes."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method


def production_harness():
    source = (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()
    issue = method(source, "    bool issue_records(")
    cache_start = issue.index("        using WaterSample=")
    cache_end = issue.index("        std::array<ViewportShaderSettings", cache_start)
    assert cache_end < issue.index("        auto flush=")
    phase_start = issue.index("                if(renderer.environment_profile && (layer==geometry_water")
    phase_end = issue.index("                context->DrawIndexed(mesh.index_count", phase_start)
    counts = method(source, "    struct PhaseConstantCounts {") + ";"
    return r'''
#include <array>
#include <cassert>
#include <cstring>
#include <vector>
#include "Renderer/native/render_core/water_material_frame.h"
using c3x_renderer::render_core::WaterMaterialFrame;
enum {geometry_water,geometry_river,geometry_wave};
''' + counts + r'''
PhaseConstantCounts phase_constant_counts;
struct Context {
 WaterMaterialFrame water{};std::array<float,4> wave{};
 unsigned water_updates=0,wave_updates=0,water_bindings=0;
 void UpdateSubresource(int buffer,int,void*,void const* input,int,int){
  if(buffer==1){std::memcpy(&water,input,sizeof(water));++water_updates;}
  else {assert(buffer==2);std::memcpy(wave.data(),input,sizeof(wave));++wave_updates;}
 }
 void PSSetConstantBuffers(int slot,int count,int* buffer){
  assert(slot==10&&count==1&&*buffer==1);++water_bindings;
 }
} context_value;
struct Renderer {
 bool environment_profile=true,water_scene_active=true;
 WaterMaterialFrame water_material;float wave_time_seconds=19.75f;
 int water_frame=1,wave_frame=2;
} renderer;
struct Work {
 unsigned water_uploads=0,wave_uploads=0;
 void upload_buffer(int buffer){if(buffer==1)++water_uploads;else {assert(buffer==2);++wave_uploads;}}
} work;
struct Record {
 struct Mesh {float visual_time=-1;} mesh;
 bool visible=true,active=true;
 WaterMaterialFrame material;
 bool water_visible()const{return visible;}
};
void submit(std::vector<Record> const& records,int layer,unsigned flush_size){
 auto* context=&context_value;
''' + issue[cache_start:cache_end] + r'''
 // The cache remains outside the actual production flush body. Execute that
 // production phase body with different boundaries and inspect every draw's
 // constants against an independent unoptimized submission oracle.
 for(unsigned first=0;first<records.size();first+=flush_size)
  for(unsigned i=first;i<records.size()&&i<first+flush_size;++i){
   auto const& chunk=records[i];auto const& mesh=chunk.mesh;
   renderer.water_scene_active=chunk.active;renderer.water_material=chunk.material;
''' + issue[phase_start:phase_end] + r'''
   if(layer==geometry_wave){
    std::array<float,4> expected={mesh.visual_time<0?renderer.wave_time_seconds:mesh.visual_time,0,0,0};
    assert(!std::memcmp(expected.data(),context->wave.data(),sizeof(expected)));
   }else{
    WaterMaterialFrame expected=chunk.material;
    if(!chunk.active||!chunk.visible||mesh.visual_time>=0){expected.time=0;for(auto& v:expected.drift)v=0;}
    assert(!std::memcmp(&expected,&context->water,sizeof(expected)));
   }
  }
}
Record animated(){Record r;r.material.time=19.75f;
 r.material.drift[0]=.1f;r.material.drift[1]=-.2f;r.material.drift[2]=.3f;
 r.material.camera[0]=45.5f;r.material.camera[1]=-14.f;r.material.camera[2]=130.f;r.material.camera[3]=130.f;return r;}
void reset(){context_value={};work={};phase_constant_counts={};renderer={};}
'''


class PhaseConstantSubmissionTests(unittest.TestCase):
    def test_identical_records_reuse_upload_across_flushes_and_rebind_each_layer(self):
        run_cpp(production_harness() + r'''
int main(){
 for(unsigned flush:{1u,7u,256u,4096u}){
  reset();std::vector<Record> water(10000,animated()),river(5000,animated());
  submit(water,geometry_water,flush);
  assert(context_value.water_updates==1&&context_value.water_bindings==1);
  // Another pass can replace the underlying buffer. A new layer invocation
  // must initialize it even when its first sample equals the prior layer.
  context_value.water.time=-87;
  submit(river,geometry_river,flush);
  assert(context_value.water_updates==2&&context_value.water_bindings==2);
  assert(work.water_uploads==2);
  assert(phase_constant_counts.water_records==15000&&phase_constant_counts.water_updates==2);
  assert(phase_constant_counts.water_hits==14998&&!phase_constant_counts.wave_records);
 }
 // No record means no constant binding or upload.
 reset();submit({},geometry_water,256);assert(!context_value.water_updates&&!context_value.water_bindings);
}
''')

    def test_animated_still_hidden_and_authored_water_transitions_preserve_each_draw(self):
        run_cpp(production_harness() + r'''
int main(){
 reset();auto live=animated(),still=live,hidden=live,disabled=live;
 still.mesh.visual_time=4.25f;hidden.visible=false;disabled.active=false;
 std::vector<Record> records;
 // Animated -> still -> animated has to restore time AND all drift members;
 // still water keeps the camera optics, even when supplied an authored time.
 for(auto const& phase:{live,still,live,hidden,live,disabled,live})
  for(unsigned i=0;i<513;++i)records.push_back(phase);
 submit(records,geometry_water,256);
 assert(context_value.water_updates==7&&context_value.water_bindings==1);
 assert(work.water_uploads==7&&phase_constant_counts.water_records==3591);
 assert(phase_constant_counts.water_updates==7&&phase_constant_counts.water_hits==3584);
 // All eight explicit members participate. Bit patterns (including signed
 // zero and NaN payloads) are preserved without reading struct padding.
 reset();records.clear();auto sample=animated();records.push_back(sample);records.push_back(sample);
 sample.material.time+=1;records.push_back(sample);records.push_back(sample);
 for(unsigned i=0;i<3;++i){sample.material.drift[i]+=1;records.push_back(sample);records.push_back(sample);}
 for(unsigned i=0;i<4;++i){sample.material.camera[i]+=1;records.push_back(sample);records.push_back(sample);}
 sample.material.camera[0]=0.f;records.push_back(sample);records.push_back(sample);
 sample.material.camera[0]=-0.f;records.push_back(sample);records.push_back(sample);
 unsigned nan_bits=0x7fc00123u;std::memcpy(&sample.material.camera[1],&nan_bits,sizeof(float));
 records.push_back(sample);records.push_back(sample);
 ++nan_bits;std::memcpy(&sample.material.camera[1],&nan_bits,sizeof(float));records.push_back(sample);records.push_back(sample);
 submit(records,geometry_river,3);
 assert(context_value.water_updates==13&&phase_constant_counts.water_hits==13);
}
''')

    def test_wave_fixed_times_and_current_clock_change_only_at_phase_boundaries(self):
        run_cpp(production_harness() + r'''
int main(){
 for(unsigned flush:{1u,7u,256u,4096u}){
  reset();auto live=animated(),fixed=live,other=live,zero=live;
  fixed.mesh.visual_time=4.25f;other.mesh.visual_time=8.5f;zero.mesh.visual_time=0;
  std::vector<Record> records;
  for(auto const& phase:{live,fixed,fixed,other,zero,live})
   for(unsigned i=0;i<513;++i)records.push_back(phase);
  submit(records,geometry_wave,flush);
  assert(context_value.wave_updates==5&&work.wave_uploads==5);
  assert(phase_constant_counts.wave_records==3078&&phase_constant_counts.wave_updates==5);
  assert(phase_constant_counts.wave_hits==3073&&!phase_constant_counts.water_records);
  // A later visual tick must upload its current time on the first draw.
  renderer.wave_time_seconds+=.125f;
  submit({live,live},geometry_wave,1);
  assert(context_value.wave[0]==19.875f&&context_value.wave_updates==6);
  assert(phase_constant_counts.wave_records==3080&&phase_constant_counts.wave_hits==3074);
 }
}
''')


if __name__ == "__main__":
    unittest.main()
