"""Execute prepared-world placement closure, source lifetime and bounded revisits."""
import unittest
import subprocess
from unittest.mock import patch

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method
from Renderer.native.test_shared_instance_submission import GPU_STUB


def run_cpp_quiet(program, **options):
    execute = subprocess.run

    def run(command, *args, **kwargs):
        if len(command) != 1:
            return execute(command, *args, **kwargs)
        kwargs["capture_output"] = True
        try:
            return execute(command, *args, **kwargs)
        except subprocess.CalledProcessError as error:
            output = (error.stdout or b"") + (error.stderr or b"")
            raise AssertionError(output.decode(errors="replace")[-6000:]) from error

    # Keep repeated production console telemetry quiet after success. Any
    # executable assertion/error still returns its complete captured output.
    with patch("Renderer.native.native_cpp_test.subprocess.run", side_effect=run):
        return run_cpp(program, **options)


class PreparedWorldPlacementUnionTests(unittest.TestCase):
    def test_exact_fresh_union_revisits_eviction_and_plateau(self):
        fresh = (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()
        cpp = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        key = method(cpp, "    c3x_renderer::render_core::SharedInstanceSubmission::Key shared_instance_draw_key(")
        methods = "\n".join(method(fresh, signature) for signature in [
            "    AtlasInputs::Key caster_key(",
            "    Submission::Key shared_caster_placement_key(",
            "    template<class BodyInputs> bool body_placements_covered(",
            "    template<class BodyInputs> bool append_body_placements(",
            "    template<class BodyInputs,class RetireCompletedPlans> bool prepare_instances("])
        run_cpp_quiet(GPU_STUB + r'''
#include <cstdio>
#include "Renderer/native/render_core/scene_membership.h"
#include "Renderer/native/render_core/body_placement_requirements.h"
struct CachedGeometryProof {bool valid=true;};
struct CachedMeshGeneration {
 std::shared_ptr<std::vector<Owner::Instance> const> instances;
 std::shared_ptr<CachedGeometryProof> proof=std::make_shared<CachedGeometryProof>();
};
struct CachedTileGeometry {std::shared_ptr<CachedMeshGeneration> mesh=std::make_shared<CachedMeshGeneration>();};
struct Mesh {
 std::array<int,4> bounds{};int translation_x=0,translation_y=0;float natural_projection[4]={};
 std::uint64_t version=0;std::shared_ptr<std::vector<Owner::Instance> const> instances;float instance_material=40;
};
using Membership=c3x_renderer::render_core::SceneMembership<Mesh,2>;
using GeometryDrawView=c3x_renderer::render_core::GeometryDrawView<Mesh,2>;
using GeometryDrawReference=GeometryDrawView::Reference;
using GeometryDrawRecord=GeometryDrawView::Record;
struct Caster {
 struct Bounds {float low[3]={},high[3]={};}bounds;
 Owner::Instance const* unused=nullptr;
 std::vector<Owner::Instance> const* instances=nullptr;float instance_material=40,offset[3]={};
 std::uint64_t content_generation=0,version=0;unsigned layer=1;
 ID3D11Buffer *vertices=nullptr,*indices=nullptr;
 unsigned vertex_offset=0,index_offset=0,count=0,index_format=3,binding=0xffffffffu,stride=32;bool rigid=false;
};
struct Work {std::size_t bytes=0;void upload(std::size_t value){bytes+=value;}};
struct Renderer {
 Owner shared_instances;ID3D11Device* device=nullptr;ID3D11DeviceContext* context=nullptr;Membership geometry_vertex_buffers;
 c3x_renderer::render_core::ResidentContent<CachedTileGeometry> resident_content{32};
 unsigned device_generation=42,content_revision=7,frame_content_uploads=0;std::size_t frame_upload_bytes=0;
 bool raster_content_valid(CachedGeometryProof const& proof){return proof.valid;}
''' + key + r'''
};
struct Harness {
 using Submission=Owner;Renderer& renderer;Work* work=nullptr;
 struct Shadow {using Caster=::Caster;};
 struct InstanceGroup {Caster source;struct Part {Caster::Bounds bounds;unsigned first=0,count=0;};std::vector<Part> parts;};
 std::vector<Caster> casters;std::vector<InstanceGroup> instance_groups;
 Owner::Lease shared_front;Owner::CpuLease shared_metadata;Membership::Lease caster_lease;
 std::uint64_t caster_signature=0,submission_generation=0;
 struct AtlasInputs {using Key=std::array<std::uint64_t,20>;};
''' + methods + r'''
};
int main(){
 ID3D11Device device;ID3D11DeviceContext context{&device};Renderer renderer;renderer.device=&device;renderer.context=&context;Work work;Harness harness{renderer,&work};
 unsigned retire_completed_calls=0;
 std::array<CachedTileGeometry,3> tiles;std::array<Mesh,3> meshes;
 std::array<c3x_renderer::render_core::ContentHandle,3> handles;
 unsigned counts[]={503,631,521};
 for(unsigned camera=0;camera<3;++camera){
  auto values=std::make_shared<std::vector<Owner::Instance>>(counts[camera]);
  for(unsigned n=0;n<values->size();++n){(*values)[n].place[0]=float(n+camera*1000);(*values)[n].projection[2]=128;(*values)[n].projection[3]=1260;}
  tiles[camera].mesh->instances=values;meshes[camera].instances=values;meshes[camera].version=100+camera;
  meshes[camera].natural_projection[2]=128;meshes[camera].natural_projection[3]=1260;
  handles[camera]=renderer.resident_content.bind(tiles[camera],tiles[camera].mesh);
 }
 auto request=[&](unsigned camera){
  renderer.geometry_vertex_buffers.clear();assert(renderer.geometry_vertex_buffers.retain(handles[camera],tiles[camera].mesh));
  GeometryDrawRecord draw(meshes[camera]);draw.owner=handles[camera];draw.translation_x=int(camera*1234);draw.translation_y=int(camera*61);
  renderer.geometry_vertex_buffers.edit(1).push_back(draw);harness.caster_lease=renderer.geometry_vertex_buffers.publish();
  harness.casters.clear();Caster caster;caster.content_generation=handles[camera].generation;caster.version=meshes[camera].version;
  caster.instances=meshes[camera].instances.get();caster.vertices=reinterpret_cast<ID3D11Buffer*>(std::uintptr_t(512+camera));caster.count=64;
  harness.casters.push_back(caster);harness.instance_groups.clear();++harness.caster_signature;
  c3x_renderer::render_core::BodyPlacementRequirements<Mesh> inputs;
  assert(inputs.add(renderer.shared_instances,1,GeometryDrawReference(draw),renderer.shared_instance_draw_key(1,GeometryDrawReference(draw))));
  // This fixture owns no cached per-pass selection plans. Count the actual
  // production retirement boundary; lease/cache retirement is covered by
  // test_instance_selection_retirement using the production callback.
  auto retire_completed_plans=[&]{++retire_completed_calls;};
  renderer.frame_upload_bytes=0;assert(harness.prepare_instances(inputs,retire_completed_plans));
  auto body=harness.shared_front->find(renderer.shared_instance_draw_key(1,GeometryDrawReference(draw)));
  auto shadow=harness.shared_front->find(harness.shared_caster_placement_key(caster));
  assert(body.count==counts[camera] && shadow.count==counts[camera]);
  Owner::Instance actual;std::memcpy(&actual,harness.shared_front->buffer->data.data()+body.first*64,64);
  assert(!std::memcmp(actual.place,meshes[camera].instances->front().place,sizeof(actual.place)));
  assert(!std::memcmp(actual.projection,draw.natural_projection,sizeof(actual.projection)));
  assert(actual.view[0]==draw.translation_x && actual.view[1]==draw.translation_y && actual.view[2]==draw.translation_y && actual.view[3]==40);
  auto canonical=harness.shared_front->source(meshes[camera].instances.get(),40,counts[camera]);
  assert(canonical && canonical.count==counts[camera]);
  Owner::Instance canonical_packed;
  std::memcpy(&canonical_packed,harness.shared_front->buffer->data.data()+canonical.first*64,64);
  // Canonical caster shaders use place0/place1 and material, replacing the
  // occurrence view.xyz and ignoring its body projection. A validated carried
  // caster representative is therefore as exact as the original body range.
  assert(!std::memcmp(canonical_packed.place,meshes[camera].instances->front().place,sizeof(canonical_packed.place)));
  assert(canonical_packed.view[3]==meshes[camera].instance_material);
  assert(!harness.shared_front->content);
  assert(renderer.shared_instances.bytes()<=Owner::budget);
  return renderer.frame_upload_bytes;
 };
 assert(request(0)==64384);assert(request(1)==80768);assert(request(2)==66688);
 // Each replacement uploads its new request only. The unchanged resident
 // ranges incur charged GPU copies, never another CPU placement packing.
 assert(renderer.shared_instances.copied_bytes>0 && renderer.shared_instances.carried_ranges==6);
 assert(renderer.shared_instances.packed_records==2*(503+631+521));
 assert(retire_completed_calls==3);
 assert(harness.shared_front->records==2*(503+631+521));
 auto uploads=renderer.shared_instances.uploads,creates=device.creates;auto plateau=renderer.shared_instances.bytes();
 for(unsigned sweep=0;sweep<1000;++sweep){for(unsigned camera=0;camera<3;++camera)assert(request(camera)==0);
  assert(renderer.shared_instances.uploads==uploads && device.creates==creates && renderer.shared_instances.bytes()==plateau);
  assert(retire_completed_calls==3);}
 // A still-live retired mesh cannot pass the resident slot/generation proof.
 auto retired=tiles[0].mesh;auto old_handle=handles[0];renderer.resident_content.release(old_handle);
 tiles[0].mesh=std::make_shared<CachedMeshGeneration>();tiles[0].mesh->instances=meshes[0].instances;
 handles[0]=renderer.resident_content.bind(tiles[0],tiles[0].mesh);
 assert(handles[0].generation!=old_handle.generation && request(0)>0);
 for(auto const& entry:harness.shared_front->retained_sources)assert(entry.second.owner[1]!=old_handle.generation);
 // Unchanged source residency alone cannot admit changed dependency proof.
 tiles[1].mesh->proof->valid=false;tiles[0].mesh->proof.reset();++meshes[2].translation_y;
 renderer.content_revision=8;auto allocation=renderer.shared_instances.allocated_bytes;
 auto copied=renderer.shared_instances.copied_bytes;
 assert(request(2)==0 && renderer.shared_instances.allocated_bytes>allocation && renderer.shared_instances.copied_bytes>copied);
 for(auto const& entry:harness.shared_front->retained_sources)assert(entry.second.owner[1]!=handles[1].generation && entry.second.owner[1]!=handles[0].generation);
 // Source residency remains alive only through the cache/current required
 // selection, never through carried weak placement entries.
 std::weak_ptr<CachedMeshGeneration> weak=tiles[1].mesh;
 renderer.resident_content.release(handles[1]);tiles[1].mesh.reset();assert(weak.expired());
 renderer.geometry_vertex_buffers.clear();harness.caster_lease.reset();harness.shared_front.reset();harness.shared_metadata.reset();
 renderer.shared_instances.clear();assert(!renderer.shared_instances.bytes());
 assert(device.freed==device.creates);
}
''', timeout=60)


if __name__ == "__main__":
    unittest.main()
