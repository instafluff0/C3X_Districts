"""Execute production indexing and projection against independent witnesses."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ZoomMeshTests(unittest.TestCase):
    def test_evidence_storage_preflight_keeps_reserve_before_output_creation(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from Renderer.native.record_navigation_evidence import storage_preflight
        with patch("Renderer.native.record_navigation_evidence.shutil.disk_usage",return_value=SimpleNamespace(free=40*1024**3)):
            report=storage_preflight(ROOT,2240,1192,100)
            self.assertGreater(report["estimated_output_bytes"],100*2240*1192*4)
            self.assertEqual(report["reserve_bytes"],8*1024**3)
        with patch("Renderer.native.record_navigation_evidence.shutil.disk_usage",return_value=SimpleNamespace(free=8*1024**3)):
            with self.assertRaisesRegex(RuntimeError,"Insufficient evidence disk space"):
                storage_preflight(ROOT,2240,1192,100)

    def test_dirty_block_clip_keeps_glow_and_reflection_sample_coverage(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        method = "    D3D11_RECT guarded_block_rectangle(" + source.split(
            "    D3D11_RECT guarded_block_rectangle(", 1)[1].split("    void collect_shadow_casters(", 1)[0]
        program = r'''
#include <algorithm>
#include <cassert>
using LONG=int;struct D3D11_RECT {LONG left,top,right,bottom;};
''' + method + r'''
int main(){
 for(int guard:{0,4})for(int origin:{0,128,2176})for(int l=0;l<128;++l)
 for(int width:{1,2,7,64,128}){
  int r=std::min(128,l+width),extent=128+2*guard;
  auto clip=guarded_block_rectangle({origin+l,origin+l,origin+r,origin+r},guard-origin,guard-origin,guard,extent);
  assert(clip.left>=0 && clip.top>=0 && clip.right<=extent && clip.bottom<=extent);
  assert(clip.left<=l+guard && clip.right>=r+guard);
  // Independent enumeration of every high-resolution glow input used by
  // the copied output: +/-8 taps around the 2x pixel's second sample.
  for(int p=l+guard;p<r+guard;++p)for(int tap=-2*guard;tap<=2*guard;++tap){
   int sample=2*p+1+tap;
   if(sample>=0 && sample<2*extent)assert(sample>=2*clip.left && sample<2*clip.right);
  }
  auto mirror=guarded_block_rectangle(clip,4,4,4,extent+8);
  // Native-pixel bound includes both high-res distortion and filter taps.
  for(int p=clip.left;p<clip.right;++p)for(int offset=-2;offset<=2;++offset){
   int sample=p+4+offset;
   if(sample>=0 && sample<extent+8)assert(sample>=mirror.left && sample<mirror.right);
  }
  auto full=guarded_block_rectangle({0,0,extent,extent},4,4,4,extent+8);
  assert(full.left==0 && full.top==0 && full.right==extent+8 && full.bottom==extent+8);
 }
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-dirty-block-") as directory:
            cpp, binary = Path(directory) / "test.cpp", Path(directory) / "test"
            cpp.write_text(program)
            subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_resident_navigation_requires_fresh_complete_independent_evidence(self):
        from Renderer.native.analyze_resident_navigation import analyze, checksum
        import struct

        def fixture(root, mode, steps=14):
            root.mkdir(exist_ok=True)
            (root / "completion.txt").write_text("0\n")
            (root / "C3XRenderer.dll").write_bytes(b"synthetic identity only; never executable")
            lines = []
            for cycle in range(2):
                for step in range(6):
                    lines.append(f"NAV cycle={cycle} step={step} result=1")
                    if cycle:
                        lines.append(f"NAV parity step={step} status=pass")
            pattern = "tile-v1" if steps == 14 else "pixel-v1"
            lines.append(f"RESIDENT_BEGIN mode={mode} steps={steps} width=16 height=8 tile_width=128 pattern={pattern}")
            for step in range(steps):
                cold = mode == "cold"
                pixel_y = (step+1)*64 if steps == 14 else (step+1)*960//(steps+1)
                lines.append(f"RESIDENT_NAV step={step} x=35 y={39 + (pixel_y//64)*2} result=1 built={3 if cold else 0} reused={0 if cold else 3} upload_bytes={168 if cold else 0} ms=20 capture_ms=1 geometry_ms=5 draw_ms=7 readback_ms=7 pixel_y={pixel_y}")
                body = bytes((step, 100, 200, 255)) * (16 * 8)
                header = bytearray(54)
                header[:2] = b"BM"
                struct.pack_into("<I", header, 2, len(header) + len(body))
                struct.pack_into("<I", header, 10, len(header))
                struct.pack_into("<IiiHH", header, 14, 40, 16, -8, 1, 32)
                (root / f"zoom.bmp.resident{step}.bmp").write_bytes(header + body)
                lines.append(f"RESIDENT_IMAGE step={step} bytes={len(body)} fnv64={checksum(body)} saved=1")
            lines.extend(("RESIDENT_END status=pass", "BIQ 16x8 viewport: 3 visible tiles, 0 fallback"))
            (root / "benchmark.log").write_text("\n".join(lines) + "\n")

        with tempfile.TemporaryDirectory(prefix="c3x-resident-evidence-") as directory:
            root = Path(directory)
            cold, warm = root / "cold", root / "warm"
            fixture(cold, "cold")
            fixture(warm, "retained")
            report = analyze(cold, warm)
            self.assertTrue(report["pass"])
            self.assertEqual(report["max_ms"], 20)
            self.assertEqual(len(report["steps"]), 14)
            self.assertFalse(report["discrete_sample_requirement_met"])
            good = (warm / "benchmark.log").read_text()
            # Success markers alone cannot certify missing/duplicated evidence,
            # camera identity, NaN timings, or stale output files.
            for old, new in (
                ("RESIDENT_END status=pass", ""),
                ("RESIDENT_BEGIN mode=retained", "RESIDENT_BEGIN mode=cold"),
                ("step=0 x=35 y=41", "step=0 x=35 y=39"),
                ("ms=20 capture_ms=1", "ms=nan capture_ms=1"),
                ("ms=20 capture_ms=1", "ms=0 capture_ms=1"),
                ("upload_bytes=0", "upload_bytes=-1"),
                ("saved=1", "saved=0"),
                ("RESIDENT_NAV step=1 ", "RESIDENT_NAV step=0 "),
                ("width=16 height=8", "width=32 height=8"),
            ):
                with self.subTest(new=new):
                    (warm / "benchmark.log").write_text(good.replace(old, new))
                    with self.assertRaises(ValueError):
                        analyze(cold, warm)
            (warm / "benchmark.log").write_text(good)
            (warm / "completion.txt").write_text("1\n")
            with self.assertRaises(ValueError):
                analyze(cold, warm)
            (warm / "completion.txt").write_text("0\n")
            (warm / "C3XRenderer.dll").write_bytes(b"different renderer")
            with self.assertRaises(ValueError):
                analyze(cold, warm)
            fixture(warm, "retained")
            image = warm / "zoom.bmp.resident0.bmp"
            damaged = bytearray(image.read_bytes());damaged[-1] ^= 1;image.write_bytes(damaged)
            with self.assertRaises(ValueError):
                analyze(cold, warm)
            # A matching receipt proves which pixels were saved, not that the
            # renderer produced correct pixels. Independent reference parity
            # must reject a completed but visually different result too.
            original_body = bytes((0, 100, 200, 255)) * (16 * 8)
            (warm / "benchmark.log").write_text(good.replace(
                f"fnv64={checksum(original_body)}", f"fnv64={checksum(damaged[54:])}"))
            report = analyze(cold, warm)
            self.assertFalse(report["pixels_exact"])
            self.assertFalse(report["pass"])
            self.assertTrue(report["no_rebuilds_or_uploads"])
            self.assertTrue(report["latency_target_met"])
            self.assertFalse(report["steps"][0]["pixel_exact"])
            self.assertTrue(all(row["pixel_exact"] for row in report["steps"][1:]))
            image.write_bytes(b"BM")
            with self.assertRaisesRegex(ValueError, "Truncated resident image header"):
                analyze(cold, warm)
            fixture(warm, "retained")
            for old, new, flag in (("built=0", "built=1", "no_rebuilds_or_uploads"),
                                   ("upload_bytes=0", "upload_bytes=168", "no_rebuilds_or_uploads"),
                                   ("ms=20 capture_ms=1", "ms=101 capture_ms=1", "latency_target_met")):
                (warm / "benchmark.log").write_text(good.replace(old, new))
                report = analyze(cold, warm)
                self.assertFalse(report[flag])
                self.assertFalse(report["pass"])
            fixture(warm, "retained")
            cold_log = cold / "benchmark.log"
            cold_log.write_text(cold_log.read_text().replace("built=3", "built=0"))
            with self.assertRaises(ValueError):
                analyze(cold, warm)
            fixture(cold, "cold", 100)
            fixture(warm, "retained", 100)
            report = analyze(cold, warm)
            self.assertTrue(report["pass"])
            self.assertTrue(report["discrete_sample_requirement_met"])
            self.assertEqual(len({r["pixel_y"] for r in report["steps"]}), 100)
            self.assertEqual(report["p95_ms"], 20)
            self.assertEqual(report["p99_ms"], 20)
            log = warm / "benchmark.log"
            log.write_text(log.read_text().replace("pixel_y=9\n", "pixel_y=0\n"))
            with self.assertRaisesRegex(ValueError, "Unexpected pixel camera"):
                analyze(cold, warm)
            fixture(warm, "retained", 100)
            import hashlib, json
            for folder in (cold, warm):
                (folder / "biq_preview.exe").write_bytes(b"frozen witness")
                receipt={"invocation":folder.name,"inputs":{"fixture":"same"},
                         "args":{"waves":"0","world_grid":True},"environment":{},
                         "binaries":{name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                                     for name in ("C3XRenderer.dll","biq_preview.exe")}}
                (folder / "inputs.json").write_text(json.dumps(receipt))
                (folder / "evidence.json").write_text(json.dumps({"invocation":folder.name,"returncode":0,"inputs_unchanged":True}))
            self.assertTrue(analyze(cold,warm)["pass"])
            original=(warm/"inputs.json").read_text()
            for field, value in (("args", {"region_diagnostics":True}),
                                 ("environment", {"C3X_RENDERER_REGION_DIAGNOSTICS":"1"})):
                changed=json.loads(original);changed[field]=value
                (warm/"inputs.json").write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,"Per-region diagnostic logging"):
                    analyze(cold,warm)
            for field, value in (("quality_mode", "diagnostic_reflections_disabled"),
                                 ("args", {"waves":"0", "world_grid":True, "reflection_ablation":True}),
                                 ("environment", {"C3X_RENDERER_REFLECTION_CONTROL":"1"})):
                changed=json.loads(original);changed[field]=value
                (warm/"inputs.json").write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,"Diagnostic pass ablation"):
                    analyze(cold,warm)
            changed=json.loads(original);changed["args"]["waves"]="1"
            (warm/"inputs.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"quality settings"):
                analyze(cold,warm)
            changed=json.loads(original);changed["args"]["world_regions"]=True
            (warm/"inputs.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"Completed-region raster recipe"):
                analyze(cold,warm)
            (warm/"inputs.json").write_text(original)
            (warm/"biq_preview.exe").write_bytes(b"changed after execution")
            with self.assertRaisesRegex(ValueError,"binary identity"):
                analyze(cold,warm)

    def test_exact_height_cache_keeps_support_dependencies_and_owner_scope(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        body = "auto natural_height_at =" + source.split("auto natural_height_at =", 1)[1].split(
            "            auto relief_at_world", 1)[0]
        self.assertIn("natural_height_samples.clear();", source)
        self.assertIn("return natural_height_at(x,y,support)",
                      (ROOT / "Renderer/native/source_fidelity/geometry.h").read_text())
        program = r'''
#include <array>
#include <cassert>
#include <set>
#include "Renderer/native/render_core/exact_point_cache.h"
struct Queries {
 int version=1,calls=0;std::set<int> dependencies;
 float height(int,int,float u,float v,float* support){
  ++calls;dependencies.insert(int(u));dependencies.insert(int(v));
  *support=u*.25f+v*.125f;return u*2.f-v+float(version);
 }
};
int main(){
 c3x_renderer::render_core::ExactPointCache<std::array<float,2>> natural_height_samples;
 Queries queries;int natural=0,pickup_height_at=0;bool retain_height_samples=true;
''' + body + r'''
 auto verify=[&](float x,float y){
  float support=-99;float a=natural_height_at(x,y),b=natural_height_at(x,y,&support);
  assert(a==x*2.f-y+queries.version && b==a && support==x*.25f+y*.125f);
 };
 verify(1.5f,2.25f);assert(queries.calls==1);
 auto dependencies=queries.dependencies;
 verify(1.5f,2.25f);assert(queries.calls==1 && queries.dependencies==dependencies);
 verify(1.500001f,2.25f);assert(queries.calls==2); // No quantization.
 natural_height_samples.clear();queries.version++;queries.dependencies.clear();
 verify(1.5f,2.25f);assert(queries.calls==3 && queries.dependencies==dependencies);
 retain_height_samples=false;verify(1.5f,2.25f);assert(queries.calls==5);
 retain_height_samples=true;
 for(int i=0;i<20000;++i)verify(float(i),float(i)*.125f);
 assert(natural_height_samples.bytes()<=16384u*24u);
 // A point beyond bounded admission still evaluates correctly every time.
 auto before=queries.calls;verify(30000.f,17.f);assert(queries.calls==before+2);
 natural_height_samples.clear();queries.version++;verify(30000.f,17.f);
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-height-samples-") as directory:
            cpp, binary = Path(directory) / "test.cpp", Path(directory) / "test"
            cpp.write_text(program)
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_submission_casters_preserve_order_wrapping_and_content(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        method = "    void collect_shadow_casters(" + source.split(
            "    void collect_shadow_casters(", 1)[1].split("    c3x_renderer::render_core::SourceShadow::PreparedCasters* prepare_shadow_submission(", 1)[0]
        submission = source.split("    bool submit_geometry(", 1)[1].split("    bool ensure_block_targets", 1)[0]
        self.assertIn("shadow_buffers_ptr?shadow_buffers_ptr:buffers,submission_casters", submission)
        self.assertIn("shadow_buffers_ptr,false,shadow_casters_ptr,prepared_casters_ptr", submission)
        self.assertIn("shadow_buffers_ptr,reflection_pass,shadow_casters_ptr,prepared_casters_ptr", submission)
        self.assertIn("shadow_buffers_ptr,true,shadow_casters_ptr,prepared_casters_ptr", submission)
        self.assertEqual(submission.count("prepare_shadow_submission("), 1)
        program = r'''
#include <array>
#include <vector>
#include <cassert>
#include <cstdint>
namespace c3x_renderer {namespace render_core {struct SourceShadow {
 struct Bounds {float low[3]={1,2,3},high[3]={4,5,6};};
 struct Caster {int vertices=0,indices=0,count=0,index_format=0,stride=0;unsigned vertex_offset=0,index_offset=0;
  void const* instances=nullptr;float instance_material=40;
  unsigned layer=0,version=0,binding=0;Bounds bounds;float offset[2]={};};
};}}
enum {geometry_land=1,geometry_feature=2,geometry_natural_decal=3,geometry_layer_count=5};
struct CachedVertexChunk {
 struct {long left=0,top=0,right=0,bottom=0;} bounds;
 int translation_x=0,translation_y=0;float natural_projection[4]={}; unsigned projection_kind=0;int source_tile_width=128;
 struct {void const* value=nullptr;void const* get()const{return value;}} instances;float instance_material=40;
 int buffer=7,indices=8,index_count=9,index_format=16,vertex_stride=64;unsigned vertex_offset=128,index_offset=768;
 unsigned version=10,city_material=0xffffffffu;bool animation_texture=false;
 c3x_renderer::render_core::SourceShadow::Bounds world_bounds;
};
#include "Renderer/native/render_core/geometry_draws.h"
using GeometryDrawRecord=c3x_renderer::render_core::GeometryDrawRecord<CachedVertexChunk>;
using GeometryDrawView=c3x_renderer::render_core::GeometryDrawView<CachedVertexChunk,geometry_layer_count>;
using GeometryDrawReference=GeometryDrawView::Reference;
struct State {
 struct Dims {int width=100,height=80;bool wrap_x=false,wrap_y=false;} dims;
 struct World {Dims* dims;Dims dimensions()const{return *dims;}} world{&dims};
 struct Coast {World* value;World const& world()const{return *value;}} world_coast{&world};
 struct Material {bool ground=false;};
 struct {struct {std::vector<Material> materials{{true},{false}};} library;} cities;
''' + method + r'''
};
int main(){
 State s;std::array<std::vector<CachedVertexChunk>,geometry_layer_count> buffers;
 for(unsigned layer=0;layer<geometry_layer_count;++layer){
  CachedVertexChunk c;c.buffer=100+layer;c.version=200+layer;buffers[layer].push_back(c);
 }
 CachedVertexChunk ground;ground.city_material=0;buffers[2].push_back(ground);
 CachedVertexChunk building;building.city_material=1;building.buffer=500;buffers[2].push_back(building);
 CachedVertexChunk animated;animated.animation_texture=true;buffers[4].push_back(animated);
 using C=c3x_renderer::render_core::SourceShadow::Caster;
 for(bool wx:{false,true})for(bool wy:{false,true}){
  s.dims.wrap_x=wx;s.dims.wrap_y=wy;std::vector<C> out;s.collect_shadow_casters(buffers,out);
  int copies=(wx?3:1)*(wy?3:1);assert(out.size()==unsigned(4*copies));
  GeometryDrawView::Records occurrences;
  for(unsigned layer=0;layer<geometry_layer_count;++layer)for(auto const& chunk:buffers[layer]){
   occurrences[layer].emplace_back(chunk);
   occurrences[layer].back().translation_x=999;
   occurrences[layer].back().natural_projection[0]=-321;
  }
  std::vector<C> borrowed;s.collect_shadow_casters(occurrences,borrowed);
  assert(borrowed.size()==out.size());
  for(std::size_t i=0;i<out.size();++i){
   assert(borrowed[i].vertices==out[i].vertices && borrowed[i].version==out[i].version);
   assert(borrowed[i].offset[0]==out[i].offset[0] && borrowed[i].offset[1]==out[i].offset[1]);
   assert(borrowed[i].bounds.low[2]==out[i].bounds.low[2] && borrowed[i].bounds.high[2]==out[i].bounds.high[2]);
  }
  unsigned index=0;
  for(int body:{101,102,500,104})for(int y=wy?-1:0;y<=(wy?1:0);++y)
   for(int x=wx?-1:0;x<=(wx?1:0);++x){
    auto const& c=out[index++];assert(c.vertices==body && c.indices==8 && c.count==9);
    assert(c.index_format==16 && c.stride==64 && c.bounds.low[2]==3 && c.bounds.high[2]==6);
    assert(c.vertex_offset==128 && c.index_offset==768);
    assert(c.offset[0]==(x*100+y*80)*.5f && c.offset[1]==(x*100-y*80)*.5f);
    if(body==500)assert(c.binding==10001 && c.version==10);
    else assert(c.version==unsigned(body+100));
   }
  // A later submission sees content edits; the earlier request remains immutable.
  buffers[1][0].version++;std::vector<C> next;s.collect_shadow_casters(buffers,next);
  assert(next[0].version==out[0].version+1);buffers[1][0].version--;
 }
 for(auto& layer:buffers)layer.clear();std::vector<C> empty;s.collect_shadow_casters(buffers,empty);assert(empty.empty());
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-submission-casters-") as directory:
            cpp, binary = Path(directory) / "test.cpp", Path(directory) / "test"
            cpp.write_text(program)
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_retained_ground_dependencies_and_bounded_admission(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        # CachedGroundGrid now lives in source_fidelity/ground_compiler.h; the
        # cache-key lookup/admission logic around it is unmoved caller code.
        ground_compiler = (ROOT / "Renderer/native/source_fidelity/ground_compiler.h").read_text()
        retained = "struct CachedGroundGrid {" + ground_compiler.split("struct CachedGroundGrid {", 1)[1].split("struct GroundCompileInput", 1)[0]
        ground_tile = "struct CachedGroundTile {" + source.split("struct CachedGroundTile {", 1)[1].split("struct RiverNode", 1)[0]
        lookup = "auto ground_key=" + source.split("auto ground_key=", 1)[1].split("            auto append_feature_instance", 1)[0]
        admission = "if(!pending_ground_grids.empty()){\n" + source.split("if(!pending_ground_grids.empty()){\n", 1)[1].split("            QueryPerformanceCounter(&phase_end);ground_ticks", 1)[0]
        program = r'''
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "Renderer/lab/shared/natural/world.h"
#include "Renderer/lab/shared/natural/vertex.h"
using Vertex=c3x_renderer::fidelity::MapVertex;
using MapVertex=c3x_renderer::fidelity::MapVertex;
''' + retained + r'''
''' + ground_tile + r'''
struct State {
 struct Frame {int world_width_tiles=100,world_height_tiles=100,world_wrap_x=1,world_wrap_y=0;
  int world_topology_revision=1,tile_width=128,tile_height=64,target_width=128,target_height=64;} frame;
 struct Tile {int tile_x=0,tile_y=0,anchor_x=0,anchor_y=0;} tile;
 bool retained_world=true;
 c3x_renderer::fidelity::NaturalWorld natural;
 std::uint64_t content_revision=1,tile_geometry_epoch=1;
 std::unordered_map<std::uint64_t,CachedGroundTile> ground_grid_cache;
 std::size_t ground_grid_cache_bytes=0,natural_mesh_cache_budget=8192,natural_mesh_cache_capacity=4;
 struct Scene {struct Record {std::uint64_t semantic=11;} record;
  Record const* current(std::uint64_t key) const{return key==9?&record:nullptr;}} topology_cache;
 struct World {unsigned value=17;unsigned at(std::size_t) const{return value;}};
 struct Coast {World data;std::uint64_t revision=23;
  World const& world() const{return data;}
  std::uint64_t node_revision(std::uint64_t) const{return revision;}} world_coast;
 auto coordinate_key(int x,int y){return (std::uint64_t(std::uint32_t(x))<<32)|std::uint32_t(y);}
 std::uint64_t tile_content_signature(Tile const&) const{return 3;}
 bool run(int x,bool record){
  tile.tile_x=x;tile.anchor_x=x*64;
  bool world_ground=false,retain_ground_grids=true,prewarming=false;
  struct Node {int lattice_x,lattice_y,degree,touches_water;};
  std::vector<Node const*> local_river_nodes;
  c3x_renderer::fidelity::NaturalWorld::CellInputs river_dependencies;
  std::unordered_map<std::uint64_t,std::uint64_t> dependencies,coast_dependencies;
  std::unordered_map<std::size_t,std::uint32_t> world_dependencies;
''' + lookup + r'''
  bool hit=ground_hit;
  if(record){
   CachedGroundGrid grid;grid.divisions=1;grid.layer=.5f;grid.vertices.resize(4);grid.samples.resize(4);
   pending_ground_grids.push_back(std::move(grid));
   dependencies[9]=topology_cache.record.semantic;coast_dependencies[5]=world_coast.revision;world_dependencies[7]=world_coast.data.value;
  }else if(hit){assert(dependencies.at(9)==topology_cache.record.semantic);assert(coast_dependencies.at(5)==world_coast.revision);assert(world_dependencies.at(7)==world_coast.data.value);}
  struct PreparedProof {
   decltype(world_dependencies) world;
   decltype(coast_dependencies) coast,topology;
   decltype(river_dependencies) rivers;
  } proof{world_dependencies,coast_dependencies,dependencies,river_dependencies};
  PreparedProof ground_cache_proof;
  auto prepared_ground=&proof;
''' + admission + r'''
  std::size_t sum=0;for(auto const& entry:ground_grid_cache)sum+=entry.second.bytes;
  assert(sum==ground_grid_cache_bytes && sum<=natural_mesh_cache_budget && ground_grid_cache.size()<=natural_mesh_cache_capacity);
  return hit;
 }
};
int main(){
 State s;assert(!s.run(0,true));assert(s.run(0,false));
 assert(!s.run(100,false) && !s.run(-100,false));
 s.world_coast.revision++;assert(!s.run(0,false) && !s.ground_grid_cache_bytes);
 s.run(0,true);s.world_coast.data.value++;assert(!s.run(0,false));
 s.run(0,true);s.topology_cache.record.semantic++;assert(!s.run(0,false));
 s.run(0,true);s.content_revision++;assert(!s.run(0,false));
 s.run(0,true);s.frame.world_topology_revision++;assert(s.run(0,false)); // Local values unchanged.
 s.retained_world=false;s.run(0,true);s.frame.world_topology_revision++;assert(!s.run(0,false));
 s.retained_world=true;
 s.run(0,true);s.frame.world_width_tiles+=2;assert(!s.run(0,false));
 s.run(0,true);s.frame.world_wrap_x=0;assert(!s.run(0,false));
 s.run(0,true);s.frame.tile_width=192;s.frame.tile_height=96;assert(s.run(0,false));
 State bounded;for(int x:{-2,-1,0,1})bounded.run(x,true);
 assert(bounded.ground_grid_cache.size()==4);
 bounded.run(100,true);assert(bounded.ground_grid_cache.size()==4 && !bounded.run(100,false));
 State tiny;tiny.natural_mesh_cache_budget=1;tiny.run(0,true);assert(tiny.ground_grid_cache.empty());
 State extended;extended.run(0,true);auto before=extended.ground_grid_cache_bytes;
 assert(extended.run(0,true) && extended.ground_grid_cache_bytes>before);
 assert(extended.ground_grid_cache.begin()->second.grids->size()==2);
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-ground-cache-") as directory:
            cpp = Path(directory) / "test.cpp"
            cpp.write_text(program)
            binary = Path(directory) / "test"
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_ground_compile_call_site_uses_private_scratch_not_shared_natural(self):
        # Milestone 1.2: ground's compile_ground_surfaces call must read
        # through its own ground_query_scratch (source_fidelity/
        # surface_query_scratch.h), not the frame-level natural/queries/
        # pickup_surface that feature, forest exclusion and city generation
        # also read for this same tile -- otherwise two tiles' ground
        # compiles would race NaturalWorld's consumer/last_cell state exactly
        # as test_world_view_submission.py's
        # test_ground_and_cliff_private_scratches_never_share_consumer_state
        # demonstrates. This guards the call site wiring by source, since the
        # call site itself needs the real WorldCoast/ReliefSurface/D3D
        # dependencies to execute directly.
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        ground_compiler = (ROOT / "Renderer/native/source_fidelity/ground_compiler.h").read_text()
        # The compiler's own signature must accept the base NaturalWorld (its
        # only two uses, river_sample/river_affects, are both declared there),
        # not the GPU-owning Natural subclass -- otherwise a private
        # NaturalWorld-only scratch could never be substituted for it.
        signature = ground_compiler.split("void compile_ground_surfaces(", 1)[1].split(");", 1)[0]
        self.assertIn("NaturalWorld & natural", signature)
        self.assertNotIn("Natural & natural", signature)
        pipeline = (ROOT / "Renderer/native/source_fidelity/prepared_ground.h").read_text()
        call = "c3x_renderer::fidelity::compile_ground_surfaces(" + pipeline.split(
            "c3x_renderer::fidelity::compile_ground_surfaces(", 1)[1].split(");", 1)[0]
        self.assertIn("ground_query_scratch.rivers", call)
        self.assertIn("ground_relief_at_world", call)
        self.assertIn("ground_river_distance", call)
        self.assertIn("ground_water_family_depth", call)
        self.assertNotIn(", natural,", call)
        # Milestone 1.2 correction: ground_query_scratch must no longer be a
        # persistent renderer member shared by every ground compile (that
        # would let two future ground jobs race on its point caches and
        # NaturalWorld river-page/consumer state). It must instead be
        # declared fresh, right before this call, so each invocation gets
        # its own instance -- the call site's whole private-pipeline block
        # (anchored on its own explanatory comment, through to the call)
        # must contain the local declaration, not just a bare use of the
        # name, and that declaration must not also appear as a persistent
        # struct member anywhere in the file.
        pipeline_anchor = "Ground's own private query/height/river pipeline"
        pipeline_block = pipeline.split(pipeline_anchor, 1)[1].split(
            "c3x_renderer::fidelity::compile_ground_surfaces(", 1)[0]
        self.assertIn("ground_query_scratch.reset_tile();", pipeline_block)
        self.assertNotRegex(
            pipeline.split(pipeline_anchor, 1)[0],
            r"SurfaceQueryScratch\s+ground_query_scratch\s*;")
        # local_river_nodes holds pointers into topology_cache.rivers, which
        # is cleared/rebuilt whenever the frame's topology pre-pass reruns.
        # Ground must copy the pointed-to RiverNode values into its own
        # owned vector before compiling, and pass pointers into that OWNED
        # copy to compile_ground_surfaces, not the shared local_river_nodes
        # pointers directly.
        self.assertIn("ground_river_node_values", pipeline_block)
        self.assertIn("ground_river_nodes", call)
        self.assertNotIn("local_river_nodes", call)
        # CachedGroundTile::grids must be a shared_ptr so an in-flight
        # reader can keep the vector alive past this tile's map-entry being
        # evicted by a later tile's LRU admission.
        cached_ground_tile = source.split("struct CachedGroundTile {", 1)[1].split("};", 1)[0]
        self.assertIn("std::shared_ptr<std::vector<CachedGroundGrid>> grids", cached_ground_tile)
        # Every dependency accumulator ground's private pipeline records into
        # must be merged into the shared maps the outer cache-persistence and
        # future cache-hit validity checks (river_dependencies/dependencies/
        # coast_dependencies/world_dependencies) actually read.
        merge = source.split("auto prepared_ground=ground_task.take();", 1)[1].split("pending_ground_grids=", 1)[0]
        for field, shared_map in (("world", "world_dependencies"), ("coast", "coast_dependencies"),
                                  ("topology", "dependencies"), ("rivers", "river_dependencies")):
            self.assertIn(f"for(auto const& dependency:prepared_ground->{field}){shared_map}.emplace(", merge)
        self.assertIn("ground_grid_lease.get(), result->grid_hits", call)

    def test_cached_ground_grids_shared_ptr_survives_map_entry_erasure(self):
        # Milestone 1.2 correction: CachedGroundTile::grids must be a
        # shared_ptr, because the LRU admission logic above can erase ANY
        # map entry -- including one belonging to an earlier tile a future
        # worker job is still reading -- while picking a victim to make
        # room for a new entry. This proves the actual production struct's
        # grids field keeps the vector alive (readable, unchanged content)
        # through exactly that erasure, as long as a caller (standing in for
        # an in-flight job) still holds a copy of the shared_ptr.
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        ground_compiler = (ROOT / "Renderer/native/source_fidelity/ground_compiler.h").read_text()
        retained = "struct CachedGroundGrid {" + ground_compiler.split("struct CachedGroundGrid {", 1)[1].split("struct GroundCompileInput", 1)[0]
        ground_tile = "struct CachedGroundTile {" + source.split("struct CachedGroundTile {", 1)[1].split("struct RiverNode", 1)[0]
        self.assertIn("std::shared_ptr<std::vector<CachedGroundGrid>> grids", ground_tile)
        program = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include "Renderer/lab/shared/natural/world.h"
#include "Renderer/lab/shared/natural/vertex.h"
using Vertex=c3x_renderer::fidelity::MapVertex;
using MapVertex=c3x_renderer::fidelity::MapVertex;
''' + retained + r'''
''' + ground_tile + r'''
int main(){
 std::unordered_map<std::uint64_t,CachedGroundTile> cache;
 auto& entry=cache[1];
 std::vector<CachedGroundGrid> grids(2);
 grids[0].divisions=3;grids[0].layer=.25f;
 grids[1].divisions=5;grids[1].layer=.75f;
 entry.grids=std::make_shared<std::vector<CachedGroundGrid>>(std::move(grids));
 std::shared_ptr<std::vector<CachedGroundGrid>> lease=entry.grids; // simulates a job capturing the lease before its tile is evicted
 assert(cache.erase(1)==1); // a later tile's admission evicting this entry as its LRU victim
 assert(cache.empty());
 assert(lease && lease->size()==2);
 assert((*lease)[0].divisions==3 && (*lease)[0].layer==.25f);
 assert((*lease)[1].divisions==5 && (*lease)[1].layer==.75f);
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-ground-grid-lease-") as directory:
            cpp = Path(directory) / "test.cpp"
            cpp.write_text(program)
            binary = Path(directory) / "test"
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_retained_ground_grid_preserves_projection_and_raw_normals(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        # CachedGroundGrid now lives in source_fidelity/ground_compiler.h.
        source = (ROOT / "Renderer/native/source_fidelity/ground_compiler.h").read_text()
        retained = "struct CachedGroundGrid {" + source.split("struct CachedGroundGrid {", 1)[1].split("struct GroundCompileInput", 1)[0]
        program = r'''
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include "Renderer/lab/shared/natural/vertex.h"
using Vertex=c3x_renderer::fidelity::MapVertex;
using MapVertex=c3x_renderer::fidelity::MapVertex;
''' + retained + r'''
int main(){
 for(int divisions:{8,12,16,24,32})for(float layer:{.5f,1.f,9.f}){
  CachedGroundGrid grid;grid.divisions=divisions;grid.layer=layer;
  grid.vertices.resize((divisions+1)*(divisions+1));grid.samples.resize(grid.vertices.size());
  for(unsigned i=0;i<grid.vertices.size();i++){
   auto& vertex=grid.vertices[i];auto* fields=reinterpret_cast<float*>(&vertex);
   for(unsigned j=0;j<sizeof(Vertex)/sizeof(float);j++)fields[j]=float(i+j)*.123f;
   grid.samples[i]={layer==.5f?0.f:float(i%37)*.73f,float(int(i%7)-3)*.031f,float(int(i%13)-6)*.019f};
  }
  for(int width:{64,96,128,160,192})for(int height:{32,48,64,80,96})
  for(unsigned i=0;i<grid.vertices.size();i++){
   float u=float(i%unsigned(divisions+1))/divisions,v=float(i/unsigned(divisions+1))/divisions;
   float half_w=float(width)*.5f,half_h=float(height)*.5f;
   Vertex expected=grid.vertices[i];
   float h=grid.samples[i][0]*(float(width)/224.f*.82f);
   float gx=0.f+(half_w+(u-v)*half_w),gy=0.f+(u+v)*half_h;
   expected.x=gx;expected.y=gy-h;expected.z=gy+h*.75f;
   if(layer==1.f || layer==9.f){
    float su=grid.samples[i][1]*1.f/(2.f*.006f*float(width));
    float sv=grid.samples[i][2]*-1.f/(2.f*.006f*float(width));
    float length=std::sqrt(su*su+sv*sv+1.f);
    expected.normal_x=-su/length;expected.normal_y=-sv/length;expected.normal_z=1.f/length;
   }
   auto actual=grid.project(i,width,height);
   assert(!std::memcmp(&actual,&expected,sizeof(Vertex)));
  }
 }
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-ground-reprojection-") as directory:
            cpp = Path(directory) / "test.cpp"
            cpp.write_text(program)
            binary = Path(directory) / "test"
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_viewport_draw_identity_rejects_eviction_and_stale_frames(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        caches = "struct CachedViewport {" + source.split("struct CachedViewport {", 1)[1].split("struct CachedVertexChunk", 1)[0]
        restore = "    bool restore_viewport_geometry(" + source.split(
            "    bool restore_viewport_geometry(", 1)[1].split("    static bool append_region_bytes(", 1)[0]
        program = r'''
#include <cassert>
#include <cstdint>
#include "Renderer/native/render_core/resident_content.h"
using Handle=c3x_renderer::render_core::ContentHandle;
#include <unordered_map>
#include <vector>
using c3x_renderer_u32=unsigned;
namespace c3x_renderer {struct TerrainFrameSignature {std::uint64_t complete=0;};}
struct c3x_renderer_tile_v1 {int x=0;};
struct c3x_renderer_frame_v1 {unsigned tile_count=2;c3x_renderer_tile_v1 const* tiles;int world_topology_revision=9;};
''' + caches + r'''
struct CachedTileGeometry {bool shared_natural=false;std::uint64_t version=0;Handle binding,natural_content;};
struct State {
 c3x_renderer::render_core::ResidentContent<CachedTileGeometry> resident_content{8};
 CachedGeometry geometry_cache;int geometry_world_revision=0,appends=0;
 void append_tile_geometry(CachedTileGeometry&,c3x_renderer_tile_v1 const&,bool animated){assert(animated);++appends;}
''' + restore + r'''
};
int main(){
 State state;CachedViewport view;view.signature.complete=10;
 view.tiles={{3},{5}};
 view.replacement_flags={1,0};view.fallback_indices={1};view.rendered_tile_count=1;
 view.fallback_tile_count=1;view.textured_tile_count=1;
 c3x_renderer_frame_v1 frame={2,view.tiles.data(),9};
 CachedTileGeometry camera,world,second;world.shared_natural=true;
 world.binding=state.resident_content.bind(world);camera.natural_content=world.binding;
 camera.binding=state.resident_content.bind(camera);second.binding=state.resident_content.bind(second);
 view.tile_keys={camera.binding,{7,99999}};
 assert(!state.restore_viewport_geometry(view,frame,{10}) && !state.appends);
 view.tile_keys[1]=second.binding;
 assert(!state.restore_viewport_geometry(view,frame,{11}) && !state.appends);
 assert(state.restore_viewport_geometry(view,frame,{10}) && state.appends==2);
 assert(state.geometry_cache.valid && state.geometry_cache.tile_keys==view.tile_keys);
 assert(state.geometry_cache.replacement_flags==view.replacement_flags);
 assert(state.geometry_cache.fallback_indices==view.fallback_indices && state.geometry_world_revision==9);
 state.resident_content.release(world.binding);world.binding=state.resident_content.bind(world);state.appends=0;
 assert(!state.restore_viewport_geometry(view,frame,{10}) && !state.appends);
 state.resident_content.release(world.binding);state.appends=0;state.geometry_cache.clear();
 assert(!state.restore_viewport_geometry(view,frame,{10}) && !state.appends && !state.geometry_cache.valid);
 frame.tile_count=1;assert(!state.restore_viewport_geometry(view,frame,{10}));
 // Zero identities are intentionally non-rendered records, not missing owners.
 frame.tile_count=2;view.tile_keys={{},second.binding};
 assert(state.restore_viewport_geometry(view,frame,{10}) && state.appends==1);
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-viewport-identities-") as directory:
            cpp, binary = Path(directory) / "test.cpp", Path(directory) / "test"
            cpp.write_text(program)
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_resident_draw_records_borrow_active_owners_and_grow_amortized(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        append = "    void append_tile_geometry(" + source.split(
            "    void append_tile_geometry(", 1)[1].split("    c3x_renderer::fidelity::TerrainCompileInput terrain_compile_input(", 1)[0]
        program = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include "Renderer/native/render_core/resident_content.h"
using Handle=c3x_renderer::render_core::ContentHandle;
#include <unordered_map>
#include <vector>
constexpr int geometry_layer_count=2,C3X_RENDERER_TILE_RENDER=1;
struct Ref {int references=1;void AddRef(){++references;}void Release(){--references;}};
struct CachedVertexChunk {
 CachedVertexChunk(Ref* b=nullptr,Ref* i=nullptr):buffer(b),indices(i){}
 CachedVertexChunk(CachedVertexChunk const&)=delete;
 CachedVertexChunk& operator=(CachedVertexChunk const&)=delete;
 CachedVertexChunk(CachedVertexChunk&&)=default;
 Ref *buffer=nullptr,*indices=nullptr;int translation_x=0,translation_y=0;struct {long left=0,top=0,right=0,bottom=0;} bounds;float natural_projection[4]={}; unsigned projection_kind=0;int source_tile_width=128;};
#include "Renderer/native/render_core/geometry_draws.h"
using GeometryDrawRecord=c3x_renderer::render_core::GeometryDrawRecord<CachedVertexChunk>;
using GeometryDrawView=c3x_renderer::render_core::GeometryDrawView<CachedVertexChunk,geometry_layer_count>;
using GeometryDrawReference=GeometryDrawView::Reference;

struct c3x_renderer_tile_v1 {int tile_flags=1,anchor_x=0,anchor_y=0;};
struct Anchor {int anchor_x=0,anchor_y=0;};
struct CachedTileGeometry {
 bool prefetched=false,shared_natural=false;std::size_t byte_count=0;
 std::uint64_t last_used=0,animation_epoch=0;Handle binding,natural_content;
 int anchor_x=0,anchor_y=0;std::vector<Anchor> resource_anchors;
 std::array<std::vector<CachedVertexChunk>,geometry_layer_count> buffers;
};
struct State {
 std::size_t prefetched_geometry_bytes=20;std::uint64_t tile_geometry_epoch=5;
 std::unordered_map<int,CachedTileGeometry> tile_geometry_cache;
 c3x_renderer::render_core::ResidentContent<CachedTileGeometry> resident_content{4};
 struct Scene {void attach(c3x_renderer_tile_v1 const&,Handle){}} topology_cache;
 std::vector<Anchor> resource_anchors;std::vector<int> geometry_footprints;
 GeometryDrawView::Records geometry_vertex_buffers;
 void index_world_content(CachedTileGeometry const&){}
 int tile_footprint(CachedTileGeometry const&,c3x_renderer_tile_v1 const&){return 1;}
 GeometryDrawRecord project_natural_chunk(GeometryDrawRecord chunk,c3x_renderer_tile_v1 const&){chunk.natural_projection[0]=1;return chunk;}
''' + append + r'''
};
int main(){
 Ref ground,world,indices;State state;CachedTileGeometry tile;
 tile.prefetched=true;tile.byte_count=20;tile.binding=state.resident_content.bind(tile);
 tile.buffers[0].push_back({&ground,&indices});tile.resource_anchors.push_back({1,2});
 auto& owner=state.tile_geometry_cache[99];owner.shared_natural=true;owner.byte_count=100;
 owner.buffers[1].push_back({&world,&indices});
 tile.natural_content=state.resident_content.bind(owner);
 std::size_t copied_capacity=0;
 for(int i=0;i<4000;i++){
  auto old=state.geometry_vertex_buffers[0].capacity();
  state.append_tile_geometry(tile,{1,i*2,40},true);
  if(state.geometry_vertex_buffers[0].capacity()!=old)copied_capacity+=old;
 }
 assert(copied_capacity<16000); // exact per-tile reserve is quadratic, ~8 million.
 assert(!tile.prefetched && !state.prefetched_geometry_bytes);
 assert(tile.last_used==5 && owner.last_used==5 && owner.animation_epoch==5);
 assert(ground.references==1 && world.references==1 && indices.references==1);
 assert(state.resource_anchors.size()==4000 && state.resource_anchors.back().anchor_x==7999);
 assert(state.geometry_vertex_buffers[1].back().translation_x==7998);
 assert(state.geometry_vertex_buffers[1].back().natural_projection[0]==1 && !owner.buffers[1][0].natural_projection[0]);
 assert(state.geometry_vertex_buffers[1].back().source==&owner.buffers[1][0]);
 GeometryDrawView view(state.geometry_vertex_buffers);assert(view.is(state.geometry_vertex_buffers));
 int count=0;for(auto const& draw:view[1]){assert(&draw.content()==&owner.buffers[1][0]);++count;}
 assert(count==4000 && view[0].size()==4000);
 GeometryDrawView direct(owner.buffers);assert(direct.is(owner.buffers));
 assert(direct[0].empty() && direct[1][0].content().buffer==&world);
 assert(!GeometryDrawView{});
 for(auto& layer:state.geometry_vertex_buffers)layer.clear();
 assert(ground.references==1 && world.references==1 && indices.references==1);
 // Caster-only records must not publish resource anchors or pin indefinitely.
 state.tile_geometry_epoch=6;state.append_tile_geometry(tile,{0,0,0},false);
 assert(state.resource_anchors.size()==4000 && owner.last_used==6 && owner.animation_epoch==5);
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-shared-draws-") as directory:
            cpp, binary = Path(directory) / "test.cpp", Path(directory) / "test"
            cpp.write_text(program)
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_pixel_prefetch_borrows_both_camera_and_world_layers(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        borrow = "        for (auto const & contributor : pending_pixel_block.key) {" + source.split(
            "        for (auto const & contributor : pending_pixel_block.key) {", 1)[1].split(
            "        ViewportShaderSettings settings={};", 1)[0]
        program = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include "Renderer/native/render_core/resident_content.h"
using Handle=c3x_renderer::render_core::ContentHandle;
#include <unordered_map>
#include <vector>
constexpr int geometry_layer_count=3,geometry_natural_terrain=1;
struct CachedVertexChunk {int id,translation_x=0,translation_y=0;struct {long left=0,top=0,right=0,bottom=0;} bounds;float natural_projection[4]={}; unsigned projection_kind=0;int source_tile_width=128;};
#include "Renderer/native/render_core/geometry_draws.h"
using GeometryDrawRecord=c3x_renderer::render_core::GeometryDrawRecord<CachedVertexChunk>;
using GeometryDrawView=c3x_renderer::render_core::GeometryDrawView<CachedVertexChunk,geometry_layer_count>;
using GeometryDrawReference=GeometryDrawView::Reference;

struct c3x_renderer_tile_v1 {int tile_x=0,tile_y=0;};
struct Entry {
 bool shared_natural=false;std::uint64_t version=0;Handle binding,natural_content;
 int tile_x=0,tile_y=0;
 std::array<std::vector<CachedVertexChunk>,geometry_layer_count> buffers;
};
struct State {
 std::unordered_multimap<int,Entry> tile_geometry_cache;
 c3x_renderer::render_core::ResidentContent<Entry> resident_content{4};
 struct Contributor {std::uint64_t mesh;int x,y;};
 struct {std::vector<Contributor> key;} pending_pixel_block;
 int pixel_prepare_cursor=0,submissions=0;
 GeometryDrawView::Records buffers;
 GeometryDrawRecord project_natural_chunk(GeometryDrawRecord chunk,c3x_renderer_tile_v1 record){
  chunk.natural_projection[0]=float(record.tile_x*100+record.tile_y);return chunk;
 }
 bool prepare(){
''' + borrow + r'''
  ++submissions;return true;
 }
};
int main(){
 State s;Entry camera;camera.version=7;
 camera.tile_x=15;camera.tile_y=47;camera.buffers[0].push_back({1});
 Entry world;world.version=7;world.shared_natural=true;world.buffers[1].push_back({2});
 s.tile_geometry_cache.emplace(42,std::move(camera));s.tile_geometry_cache.emplace(99,std::move(world));
 s.tile_geometry_cache.find(99)->second.binding=s.resident_content.bind(s.tile_geometry_cache.find(99)->second);
 s.tile_geometry_cache.find(42)->second.natural_content=s.tile_geometry_cache.find(99)->second.binding;
 s.pending_pixel_block.key.push_back({7,120,60});
 assert(s.prepare() && s.submissions==1 && !s.pixel_prepare_cursor);
 assert(s.buffers[0].size()==1 && s.buffers[1].size()==1 && s.buffers[2].empty());
 assert(s.buffers[0][0].content().id==1 && !s.buffers[0][0].natural_projection[0]);
 assert(s.buffers[1][0].content().id==2 && s.buffers[1][0].natural_projection[0]==1547);
 assert(s.buffers[1][0].translation_x==120 && s.buffers[1][0].translation_y==60);
 assert(!s.tile_geometry_cache.find(99)->second.buffers[1][0].natural_projection[0]);
 // An evicted owner must cancel preparation, never cache incomplete pixels.
 s.resident_content.release(s.tile_geometry_cache.find(99)->second.binding);
 s.tile_geometry_cache.erase(99);s.buffers={};
 assert(s.prepare() && s.submissions==1 && s.pixel_prepare_cursor==1);
 // A shared owner's coincident version is not itself a camera contributor.
 s.resident_content.clear();s.tile_geometry_cache.clear();Entry orphan;orphan.version=7;orphan.shared_natural=true;
 s.tile_geometry_cache.emplace(99,std::move(orphan));
 assert(s.prepare() && s.submissions==1 && s.pixel_prepare_cursor==2);
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-world-prefetch-") as directory:
            cpp, binary = Path(directory) / "test.cpp", Path(directory) / "test"
            cpp.write_text(program)
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_shared_world_bounds_cover_every_zoom_and_reflected_height(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        project = "GeometryDrawRecord project_natural_chunk(" + source.split(
            "GeometryDrawRecord project_natural_chunk(", 1)[1].split(
            "    void index_world_content", 1)[0]
        program = r'''
#include <cassert>
#include <climits>
#include "Renderer/lab/shared/natural/ground.h"
#include "Renderer/native/render_core/projected_mesh_bounds.h"
using LONG=long;
struct c3x_renderer_tile_v1 {int tile_x,tile_y;};
struct CachedVertexChunk {
 int translation_x=0,translation_y=0;
 c3x_renderer::render_core::ProjectedMeshBounds projected_bounds;
 float natural_projection[4]={}; unsigned projection_kind=0;int source_tile_width=128;
 struct {float low[3],high[3];} world_bounds;
 struct {LONG left,top,right,bottom;} bounds;
};
constexpr int geometry_layer_count=1;
#include "Renderer/native/render_core/geometry_draws.h"
using GeometryDrawRecord=c3x_renderer::render_core::GeometryDrawRecord<CachedVertexChunk>;
using GeometryDrawView=c3x_renderer::render_core::GeometryDrawView<CachedVertexChunk,geometry_layer_count>;
using GeometryDrawReference=GeometryDrawView::Reference;
struct State {int content_view_height=900;int shadow_tile_width=128,shadow_tile_height=64,height=1192;bool tight_natural_bounds=false;
''' + project + r'''
};
int main(){
 State state;CachedVertexChunk stored={};
 stored.world_bounds={{30.5f,-17.25f,-.03f},{33.5f,-14.7f,1.73f}};
 for(int width:{64,96,128,160,192})for(int height:{480,1192,2160}){
  state.shadow_tile_width=width;state.shadow_tile_height=width/2;state.height=height;
  auto chunk=state.project_natural_chunk(stored,{15,47});
  assert(chunk.natural_projection[0]==31 && chunk.natural_projection[1]==-16);
  assert(chunk.natural_projection[2]==width && chunk.natural_projection[3]==height);
  stored.projection_kind=4;auto authored=state.project_natural_chunk(stored,{15,47});
  assert(authored.natural_projection[3]==state.content_view_height);
  assert(authored.bounds.left==chunk.bounds.left && authored.bounds.bottom==chunk.bounds.bottom);
  stored.projection_kind=0;
  c3x_renderer::fidelity::GroundProjection project{31,-16,width*.5f,width*.25f,width/224.f*.82f,float(height)};
  for(int i=0;i<10000;i++){
   float x=30.5f+(i%101)/100.f*3,y=-17.25f+(i%103)/102.f*2.55f,z=-.03f+(i%107)/106.f*1.76f;
   auto p=project(x,y,z*112.f);
   assert(p.x>=chunk.bounds.left && p.x<chunk.bounds.right);
   assert(p.y>=chunk.bounds.top && p.y<chunk.bounds.bottom);
   float reflection=2*(112.f*.82f*width/224.f)*std::max(0.f,z-2.5f/112.f);
   float max_reflection=2*(112.f*.82f*width/224.f)*std::max(0.f,stored.world_bounds.high[2]-2.5f/112.f);
   assert(p.y+reflection<chunk.bounds.bottom+max_reflection);
  }
 }
 // Computing a camera view must not mutate the cached world's projection.
 assert(stored.natural_projection[2]==0);
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-shared-world-") as directory:
            cpp, binary = Path(directory) / "test.cpp", Path(directory) / "test"
            cpp.write_text(program)
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_benchmark_validates_centered_and_historical_ladders(self):
        from Renderer.native.compare_zoom_benchmark import run
        with tempfile.TemporaryDirectory(prefix="c3x-zoom-ladder-") as directory:
            root = Path(directory)
            for levels in ((128,112,96,80,64), (128,96,64,192,160), (128,64,96,192,160)):
                lines = []
                for cycle in range(2):
                    for width in levels:
                        lines.append(f"ZOOM cycle={cycle} width={width} result=1")
                        if cycle:
                            lines.append(f"ZOOM parity width={width} changed=0 error=0 status=pass")
                lines.append("BIQ 100x100 viewport: 100 visible tiles, 0 fallback")
                (root / "benchmark.log").write_text("\n".join(lines) + "\n")
                if levels[1] == 64:
                    with self.assertRaisesRegex(ValueError, "unexpected zoom ladder"):
                        run("test", root, "zoom")
                else:
                    self.assertEqual(len(run("test", root, "zoom")[1]), 10)

    def test_backdrop_budget_preserves_current_view_and_evicts_lru(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        eviction = "bool make_resource_backdrop_room(" + source.split("bool make_resource_backdrop_room(", 1)[1].split("    void reset_resource_buffers", 1)[0]
        program = r'''
#include <cassert>
#include <vector>
#include <cstdint>
struct State {
 struct Block {std::uint64_t signature,used;std::size_t bytes;int color=1,depth=1;};
 std::vector<Block> resource_backdrops={{1,1,30},{2,2,30},{3,3,30}};
 std::size_t resource_backdrop_bytes=90,resource_backdrop_cache_budget=100;
 unsigned releases=0;
 void release(int& resource){assert(resource);resource=0;++releases;}
''' + eviction + r'''
};
int main(){
 State state;
 assert(!state.make_resource_backdrop_room(101,2) && !state.releases);
 assert(state.make_resource_backdrop_room(20,2));
 assert(state.resource_backdrops.size()==2 && state.resource_backdrops[0].signature==2);
 assert(state.resource_backdrop_bytes==60 && state.releases==2);
 assert(state.make_resource_backdrop_room(60,2));
 assert(state.resource_backdrops.size()==1 && state.resource_backdrop_bytes==30 && state.releases==4);
 assert(!state.make_resource_backdrop_room(80,2) && state.resource_backdrop_bytes==30);
 // A changed static signature cannot keep using a stale background.
 assert(state.make_resource_backdrop_room(80,4) && state.resource_backdrop_bytes==0);
 assert(state.resource_backdrops.empty() && state.releases==6);
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-backdrop-budget-") as directory:
            cpp = Path(directory) / "test.cpp"
            cpp.write_text(program)
            binary = Path(directory) / "test"
            subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_direct_grid_preserves_complete_triangle_stream(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        # append_ground_layer and CachedGroundGrid now live in
        # source_fidelity/ground_compiler.h, compiled from there directly.
        source = (ROOT / "Renderer/native/source_fidelity/ground_compiler.h").read_text()
        grid = "auto append_ground_layer = " + source.split("auto append_ground_layer = ", 1)[1].split(
            "    append_ground_layer(destination.underlay_vertices", 1)[0]
        retained = "struct CachedGroundGrid {" + source.split("struct CachedGroundGrid {", 1)[1].split("struct GroundCompileInput", 1)[0]
        program = r'''
#include <array>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <unordered_map>
#include <vector>
#include "Renderer/lab/shared/natural/world.h"
#include "Renderer/lab/shared/natural/vertex.h"
using Vertex=c3x_renderer::fidelity::MapVertex;
using MapVertex=c3x_renderer::fidelity::MapVertex;
using UINT=unsigned;
''' + retained + r'''
int main(){
 struct Input {
  bool world_ground=false,retain_ground_grids=false,reuse_nested_ground_grids=true,prewarming=false;
 } input;
 struct Destination { std::vector<CachedGroundGrid> pending_grids; } destination;
 struct Frame {int tile_width=128,tile_height=64;} frame;
 std::vector<CachedGroundGrid> grids;
 std::vector<CachedGroundGrid> const* cached_grid_source=&grids;
 unsigned ground_grid_hit_counter=0;
 auto river_node_distance=[](float,float,unsigned){return 1000.f;};
 struct Point {float relief[3]={},normal_delta[2]={};} point;
 auto ground_point_at=[&](float,float)->Point&{return point;};
 bool stop=false;unsigned samples=0;
 auto cancelled=[&](){return stop;};
 auto make_ground_vertex=[&](float u,float v,float layer){
  ++samples;Vertex out={};out.x=u*137;out.y=v*71;out.z=layer+u*v;
  out.u=u;out.v=v;out.normal_x=u-.5f;out.normal_y=v-.5f;out.normal_z=1;
  out.world_x=53+u;out.world_y=17-v;out.material_grass=layer;
  out.material_plains=u==0?-0.f:u;return out;
 };
''' + grid + r'''
 for(int n:{1,8,12,16,24,32})for(float layer:{.5f,1.f,4.f,5.f,9.f,10.f}){
  std::vector<Vertex> old,packed;std::vector<UINT> indices;
  append_ground_layer(old,layer,n);
  samples=0;append_ground_layer(packed,layer,n,&indices);
  assert(samples==unsigned((n+1)*(n+1)) && packed.size()==samples);
  assert(indices.size()==old.size() && indices.size()==unsigned(n*n*6));
  for(std::size_t i=0;i<indices.size();++i){
   assert(indices[i]<packed.size());
   assert(!std::memcmp(&packed[indices[i]],&old[i],sizeof(Vertex)));
  }
  // Reuse a larger scratch allocation for a smaller grid: no stale triangles.
  append_ground_layer(packed,layer,1,&indices);
  assert(packed.size()==4 && indices==std::vector<UINT>({0,1,3,0,3,2}));
 }
 // Execute the actual cached-grid selection/indexing against every supported
 // fine/coarse pair. Compare all channels, not only the projected position.
 for(int fine:{8,12,16,24,32})for(int coarse:{8,12,16,24,32})
 for(float layer:{.5f,1.f,9.f})for(int width:{64,96,128,160,192}){
  frame.tile_width=width;frame.tile_height=width/2;
  grids.clear();grids.emplace_back();
  auto& cached=grids.back();cached.divisions=fine;cached.layer=layer;
  for(int y=0;y<=fine;++y)for(int x=0;x<=fine;++x){
   cached.vertices.push_back(make_ground_vertex(float(x)/fine,float(y)/fine,layer));
   cached.samples.push_back({float(x+y)*.25f,float(x)*.017f,float(y)*-.023f});
  }
  int stride=cached.sample_stride(coarse);
  assert(bool(stride)==(fine>=coarse && fine%coarse==0));
  assert(!cached.sample_stride(0) && !cached.sample_stride(-1));
  std::vector<Vertex> packed;std::vector<UINT> indices;samples=0;
  append_ground_layer(packed,layer,coarse,&indices);
  assert(indices.size()==unsigned(coarse*coarse*6));
  if(!stride){assert(samples==unsigned((coarse+1)*(coarse+1)));continue;}
  assert(!samples);
  CachedGroundGrid expected;expected.divisions=coarse;expected.layer=layer;
  for(int y=0;y<=coarse;++y)for(int x=0;x<=coarse;++x){
   auto from=(y*stride)*(fine+1)+x*stride;
   expected.vertices.push_back(cached.vertices[from]);expected.samples.push_back(cached.samples[from]);
  }
  for(unsigned i=0;i<packed.size();++i){
   auto vertex=expected.project(i,width,width/2);
   if(layer==9.f)vertex.river_branch_count=1000.f;
   assert(!std::memcmp(&packed[i],&vertex,sizeof(Vertex)));
  }
  input.reuse_nested_ground_grids=false;samples=0;append_ground_layer(packed,layer,coarse,&indices);
  assert(samples==(fine==coarse?0u:unsigned((coarse+1)*(coarse+1))));
  input.reuse_nested_ground_grids=true;
 }
 stop=true;samples=0;std::vector<Vertex> aborted;std::vector<UINT> empty;
 append_ground_layer(aborted,.5f,32,&empty);assert(!samples && empty.empty());
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-indexed-grid-") as directory:
            cpp = Path(directory) / "test.cpp"
            cpp.write_text(program)
            binary = Path(directory) / "test"
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_memory_receipt_distinguishes_missing_samples(self):
        from Renderer.native.compare_zoom_benchmark import memory_samples
        with tempfile.TemporaryDirectory(prefix="c3x-camera-memory-") as directory:
            root = Path(directory)
            log = root / "benchmark.log"
            log.write_text("ZOOM cycle=0\n")
            self.assertIsNone(memory_samples(root))
            log.write_text("CAMERA memory available_virtual=2000 largest_free_region=1500 total_virtual=4000\n"
                           "CAMERA memory available_virtual=1800 largest_free_region=1200 total_virtual=4000\n")
            self.assertEqual(memory_samples(root), {"samples": 2,
                "minimum_available_virtual_bytes": 1800,
                "minimum_largest_free_region_bytes": 1200,
                "total_virtual_bytes": 4000})

    def test_budget_eviction_preserves_active_frame(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        eviction = "    bool make_tile_cache_room(" + source.split("    bool make_tile_cache_room(", 1)[1].split("    bool cache_geometry_layer(", 1)[0]
        program = r'''
#include <cassert>
#include <cstdio>
#include <cstdint>
#include "Renderer/native/render_core/resident_content.h"
using Handle=c3x_renderer::render_core::ContentHandle;
#include <unordered_map>
template<std::size_t N,class... A> int sprintf_s(char(&out)[N],char const*format,A...args){return std::snprintf(out,N,format,args...);}
struct State {
    struct Item {std::size_t byte_count;unsigned last_used;bool prefetched;int buffers=0;std::uint64_t animation_epoch=0;Handle binding;};
    using CachedTileGeometry=Item;
    c3x_renderer::render_core::ResidentContent<Item> resident_content{2048};
    unsigned viewport_cache_capacity=32;
    struct Trace {void write(char const*,char const*,bool){}} trace;
    std::unordered_map<int,Item> tile_geometry_cache;
    std::size_t terrain_patch_index_bytes=0;
    std::size_t tile_geometry_cache_bytes=60,prefetched_geometry_bytes=10;
    std::size_t tile_geometry_runtime_budget=100,tile_geometry_cache_capacity=2048;
    unsigned tile_geometry_epoch=3,frame_tiles_evicted=0,cache_evictions=0;
    void release_geometry_vertex_buffers(int&){}
    void release_resident_content(Item& owner){resident_content.release(owner.binding);}
''' + eviction + r'''
};
int main(){
 State s;s.tile_geometry_cache={{1,{10,1,true}},{2,{20,2,false}},{3,{30,3,false}}};
 for(auto& item:s.tile_geometry_cache)item.second.binding=s.resident_content.bind(item.second);
 auto expired=s.tile_geometry_cache.at(1).binding;
 assert(s.make_tile_cache_room(10) && s.tile_geometry_cache.size()==3);
 assert(s.make_tile_cache_room(50) && s.tile_geometry_cache.size()==2);
 assert(!s.tile_geometry_cache.count(1) && s.tile_geometry_cache.count(3));
 assert(!s.resident_content.resolve(expired));
 assert(s.tile_geometry_cache_bytes==50 && !s.prefetched_geometry_bytes && s.frame_tiles_evicted==1);
 assert(!s.make_tile_cache_room(80) && s.tile_geometry_cache.size()==1);
 assert(s.tile_geometry_cache_bytes==30 && s.tile_geometry_cache.count(3));
 assert(!s.make_tile_cache_room(71) && s.tile_geometry_cache_bytes==30);
 assert(s.make_tile_cache_room(70) && s.tile_geometry_cache.size()==1);
 s.terrain_patch_index_bytes=10;assert(!s.make_tile_cache_room(70));assert(s.make_tile_cache_room(60));
 State favored;favored.tile_geometry_cache={{1,{10,1,true}},{2,{20,2,false}},{3,{30,3,false}}};
 favored.tile_geometry_cache.at(1).animation_epoch=1;
 assert(favored.make_tile_cache_room(50));
 assert(favored.tile_geometry_cache.count(1) && !favored.tile_geometry_cache.count(2));
 assert(favored.tile_geometry_cache.count(3) && favored.tile_geometry_cache_bytes==40);
 // Preference expires and never pins unused geometry against the hard cap.
 favored.tile_geometry_epoch=40;favored.tile_geometry_cache.at(3).last_used=40;
 assert(favored.make_tile_cache_room(70) && !favored.tile_geometry_cache.count(1));
 State bounded;bounded.tile_geometry_cache={{1,{10,1,true}},{2,{20,2,false}},{3,{30,3,false}}};
 bounded.tile_geometry_cache.at(1).animation_epoch=1;
 bounded.tile_geometry_cache.at(2).animation_epoch=2;
 assert(bounded.make_tile_cache_room(70) && bounded.tile_geometry_cache.size()==1);
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-cache-pressure-") as directory:
            cpp = Path(directory) / "test.cpp"
            cpp.write_text(program)
            binary = Path(directory) / "test"
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_exact_indexing_and_world_reprojection(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        program = r'''
#include <array>
#include <atomic>
#include <cassert>
#include <climits>
#include <cstring>
#include <unordered_map>
#include "Renderer/lab/shared/natural/ground.h"
#include "Renderer/native/render_core/prepared_mesh.h"
using namespace c3x_renderer::render_core;
using UINT=unsigned;
unsigned index_at(PreparedMesh const& mesh,unsigned i){
 unsigned result=0;std::memcpy(&result,mesh.indices.data()+i*mesh.index_stride,mesh.index_stride);return result;
}
bool check(bool feature) {
    std::vector<Vertex> vertices;
    for(unsigned i=0;i<30000;i++) {
        Vertex v={};unsigned key=(i*73)%1789;
        v.x=float(key%41);v.y=float(key/41);v.z=float(key%13);
        v.u=key*.125f;v.normal_z=1;v.world_x=key*.25f;
        v.material_grass=float(i%3);v.v=i%7==0?-0.f:0.f;
        vertices.push_back(v);
    }
    std::vector<Vertex> expected;std::vector<UINT> expected_indices;
    std::unordered_map<Vertex,UINT,VertexHash,VertexEqual> old(0,VertexHash{sizeof(Vertex),feature},VertexEqual{sizeof(Vertex),feature});
    for(auto const& v:vertices){auto at=old.emplace(v,UINT(expected.size()));
        if(at.second)expected.push_back(v);expected_indices.push_back(at.first->second);}
    for(unsigned projection:{0u,2u,3u}){
        MeshFormat format;format.feature=feature;format.projection_kind=projection;
        PreparedMesh mesh;assert(prepare_mesh(vertices,nullptr,format,mesh,[]{return false;}));
        assert(mesh.index_count==expected_indices.size() && mesh.vertices.size()/mesh.vertex_stride==expected.size());
        for(unsigned i=0;i<mesh.index_count;++i)assert(index_at(mesh,i)==expected_indices[i]);
        for(unsigned i=0;i<expected.size();++i){auto v=expected[i];
            if(projection){v.x/=128.f;v.y/=128.f;if(projection==3){v.z/=128.f;v.river_branch_count/=128.f;}}
            float compact[]={v.x,v.y,v.z,v.u,v.v,v.normal_x,v.normal_y,v.normal_z,v.base_terrain,v.world_x,v.world_y,v.world_z};
            assert(!std::memcmp(mesh.vertices.data()+i*mesh.vertex_stride,feature?static_cast<void*>(compact):&v,mesh.vertex_stride));
        }
        auto previous=mesh.vertices;
        assert(!prepare_mesh(vertices,nullptr,format,mesh,[]{return true;}));assert(mesh.vertices==previous);
        unsigned calls=0;assert(!prepare_mesh(vertices,nullptr,format,mesh,[&]{return ++calls==3;}));assert(mesh.vertices==previous);
    }
    return true;
}
void check_index_width(std::size_t vertex_count) {
    std::vector<Vertex> vertices(vertex_count);
    std::vector<UINT> indices={0,1,UINT(vertex_count-1),0,UINT(vertex_count-1),1};
    PreparedMesh mesh;MeshFormat format;
    assert(prepare_mesh(vertices,&indices,format,mesh,[]{return false;}));
    assert(mesh.index_stride==(vertex_count<=65535?2:4));
    assert(mesh.vertices.size()==vertex_count*sizeof(Vertex));
    for(unsigned i=0;i<indices.size();++i)assert(index_at(mesh,i)==indices[i]);
    indices.push_back(unsigned(vertex_count));assert(!prepare_mesh(vertices,&indices,format,mesh,[]{return false;}));
}
void check_formats(){
    std::vector<Vertex> vertices(3);float values[42];
    for(unsigned i=0;i<42;++i)values[i]=float(i)+.125f;
    for(auto& v:vertices)std::memcpy(&v,values,sizeof(v));
    vertices[1].x=-10.25f;vertices[1].world_z=25;vertices[2].y=100.25f;
    std::vector<unsigned> indices={2,0,1};
    for(bool natural:{false,true}){
        MeshFormat format;format.natural=natural;format.pickup=natural;
        PreparedMesh mesh;assert(prepare_mesh(vertices,&indices,format,mesh,[]{return false;}));
        assert(mesh.bounds[0]==-13 && mesh.bounds[3]==103);
        for(unsigned i=0;i<3;++i){auto const& v=vertices[i];
            float data[]={v.x,v.y,v.z,v.world_x,v.world_y,v.world_z,v.world_valid,v.normal_x,v.normal_y,v.normal_z,v.u,v.v,
                v.material_grass,v.material_plains,v.material_desert,v.material_marsh,v.authored_relief_height,v.authored_relief_blend,v.base_terrain,
                v.relief_owner_u,v.relief_owner_v,v.relief_owner_coverage,v.relief_owner_state};
            assert(mesh.vertex_stride==(natural?92:120));
            assert(!std::memcmp(mesh.vertices.data()+i*mesh.vertex_stride,natural?static_cast<void*>(data):&v,mesh.vertex_stride));
        }
        if(natural)assert(mesh.projected_bounds.valid && mesh.world_high[2]==32.125f && mesh.world_low[2]==25);
    }
}
int main(){
    check_formats();
    for(auto count:{3u,65535u,65536u,70000u})check_index_width(count);
    assert(check(false) && check(true));
    // IEEE float grids share low mantissa bits. A power-of-two table must
    // disperse high bits too, or every regular vertex starts in one bucket.
    std::vector<UINT> slots(8192,UINT_MAX);std::size_t probes=0;
    VertexHash hash;
    for(unsigned i=0;i<4096;++i){Vertex v={};v.x=(i%64)*.125f;v.y=(i/64)*.125f;
        auto slot=hash(v)&8191u;++probes;
        while(slots[slot]!=UINT_MAX){++probes;slot=(slot+1)&8191u;}
        slots[slot]=i;
    }
    assert(probes<4096*4);
    using c3x_renderer::fidelity::GroundProjection;
    for(int original:{64,80,96,112,128,160,192})for(int next:{64,80,96,112,128,160,192})
    for(int target:{480,640,1080})for(int i=0;i<10000;i++){
        float x=31.f+(i%65)/64.f,y=-17.f+(i%67)/64.f;
        float h=2.5f+(i%3001)*.1253f;
        GroundProjection before{31,-17,original*.5f,original*.25f,original/224.f*.82f,640};
        GroundProjection after{31,-17,next*.5f,next*.25f,next/224.f*.82f,float(target)};
        auto cached=before(x,y,h),fresh=after(x,y,h),restored=after(cached.world_x,cached.world_y,cached.world_z*112.f);
        assert(std::abs(fresh.x-restored.x)<1.f/256 && std::abs(fresh.y-restored.y)<1.f/256 && std::abs(fresh.z-restored.z)<1.f/256);
    }
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-zoom-mesh-") as directory:
            cpp = Path(directory) / "test.cpp"
            cpp.write_text(program)
            binary = Path(directory) / "test"
            compiled = subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
