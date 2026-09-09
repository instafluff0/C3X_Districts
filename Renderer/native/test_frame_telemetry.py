"""Execute the production delayed-query owner with deterministic D3D query outcomes."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from Renderer.lab.platform import ROOT, native_command_result


class FrameTelemetryTests(unittest.TestCase):
    def test_query_lifetime_pressure_disjoint_and_errors(self):
        source = (ROOT / "Renderer/native/render_core/frame_telemetry.h").read_text()
        body = "struct GpuFrameTelemetry {" + source.split("struct GpuFrameTelemetry {", 1)[1].split("struct AddressSpaceSample", 1)[0]
        program = r'''
#include <array>
#include <cstdint>
#include <cassert>
#include <cstring>
using HRESULT=int;using UINT64=std::uint64_t;
constexpr int S_OK=0,S_FALSE=1,D3D11_QUERY_TIMESTAMP_DISJOINT=1,D3D11_QUERY_TIMESTAMP=2;
constexpr int D3D11_ASYNC_GETDATA_DONOTFLUSH=2;
bool FAILED(int value){return value<0;}
struct D3D11_QUERY_DESC{int Query,MiscFlags;};
struct D3D11_QUERY_DATA_TIMESTAMP_DISJOINT{UINT64 Frequency;bool Disjoint;};
int allocated=0;
struct ID3D11Query{int kind;UINT64 stamp=0;void Release(){--allocated;delete this;}};
struct ID3D11Device{
    int creates=0,fail_at=-1;
    int CreateQuery(D3D11_QUERY_DESC* desc,ID3D11Query** query){
        if(creates++==fail_at)return -1;
        *query=new ID3D11Query{desc->Query};++allocated;return 0;
    }
};
struct ID3D11DeviceContext{
    int status=S_FALSE;bool disjoint=false;unsigned begins=0,ends=0,reads=0;UINT64 time=0;
    void Begin(ID3D11Query*){++begins;}
    void End(ID3D11Query* q){q->stamp=(time+=10);++ends;}
    int GetData(ID3D11Query* q,void* data,unsigned bytes,unsigned flags){
        assert(flags==D3D11_ASYNC_GETDATA_DONOTFLUSH);++reads;
        if(status!=S_OK)return status;
        if(q->kind==D3D11_QUERY_TIMESTAMP_DISJOINT){
            D3D11_QUERY_DATA_TIMESTAMP_DISJOINT value{1000,disjoint};assert(bytes==sizeof(value));std::memcpy(data,&value,bytes);
        }else{assert(bytes==sizeof(q->stamp));std::memcpy(data,&q->stamp,bytes);}
        return S_OK;
    }
};
''' + body + r'''
int main(){
    GpuFrameTelemetry owner;ID3D11Device device;ID3D11DeviceContext context;
    for(unsigned i=0;i<8;++i){assert(owner.begin(&device,&context,100+i));owner.draw_end(&context);owner.end(&context);}
    assert(allocated==32);assert(!owner.begin(&device,&context,108));assert(owner.skipped==1);
    unsigned reports=0;
    auto valid=[&](std::uint64_t sequence,bool ok,double draw,double copy,unsigned skipped){
        assert(sequence==100+reports);assert(ok);assert(draw==10 && copy==10 && skipped==1);++reports;
    };
    owner.poll(&context,valid);assert(reports==0 && context.reads==8); // Never spins on pending data.
    context.status=S_OK;owner.poll(&context,valid);assert(reports==8);
    assert(owner.begin(&device,&context,200));{GpuFrameTelemetry::Scope cancelled{owner,&context};}
    unsigned invalid=0;auto reject=[&](std::uint64_t,bool ok,double draw,double copy,unsigned){assert(!ok && draw==0 && copy==0);++invalid;};
    owner.poll(&context,reject);assert(invalid==1);
    assert(owner.begin(&device,&context,201));owner.draw_end(&context);owner.end(&context);
    context.disjoint=true;owner.poll(&context,reject);assert(invalid==2);
    assert(owner.begin(&device,&context,202));owner.draw_end(&context);owner.end(&context);
    context.status=-1;owner.poll(&context,reject);assert(invalid==3);
    owner.reset();assert(allocated==0 && owner.skipped==0);
    device.creates=0;device.fail_at=2;assert(!owner.begin(&device,&context,203));assert(owner.failed);
    assert(!owner.begin(&device,&context,204));owner.reset();assert(allocated==0);
    device.fail_at=-1;assert(owner.begin(&device,&context,205));owner.draw_end(&context);owner.end(&context);
    owner.reset();assert(allocated==0); // Reset discards pending queries without polling or waiting.
}
'''
        self.run_program(program)

    def test_prepared_casters_preserve_bounds_and_scope(self):
        source = (ROOT / "Renderer/native/render_core/source_shadow.h").read_text()
        body = source.split("    static std::array<float,4> project(", 1)[1].split("    template<class Bind>", 1)[0]
        program = r'''
#include <algorithm>
#include <array>
#include <vector>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <set>
#include <cmath>
struct Shadow {
    struct Bounds{float low[3]={},high[3]={};};
    struct Caster{Bounds bounds;float offset[3]={};std::uint64_t version=1;unsigned layer=0,index_format=42,binding=0xffffffffu;};
    std::array<float,12> basis{};
    static std::array<float,4> project(''' + body + r'''
};
int main(){
    std::vector<Shadow::Caster> casters(50);
    std::array<float,12> basis={2,-3,4,0,-5,6,-7,0,0,0,0,0};
    for(unsigned i=0;i<casters.size();++i)for(unsigned axis=0;axis<3;++axis){
        casters[i].bounds.low[axis]=float(i+axis);casters[i].bounds.high[axis]=float(i+axis+3);
        casters[i].offset[axis]=float(int(i)%3-1)*100;
    }
    Shadow::PreparedCasters prepared;assert(prepared.build(casters,basis));assert(prepared.matches(casters,basis));
    for(unsigned i=0;i<casters.size();++i){
        std::array<float,4> expected{};
        for(unsigned row=0;row<2;++row)for(unsigned axis=0;axis<3;++axis){
            float k=basis[row*4+axis],a=(casters[i].bounds.low[axis]+casters[i].offset[axis])*k;
            float b=(casters[i].bounds.high[axis]+casters[i].offset[axis])*k;
            expected[row]+=std::min(a,b);expected[row+2]+=std::max(a,b);
        }
        assert(expected==prepared.bounds[i]);
    }
    auto other=casters;assert(!prepared.matches(other,basis));
    auto changed=basis;changed[0]=-2;assert(!prepared.matches(casters,changed));
    auto previous=prepared.bounds[0];casters[0].bounds.high[2]+=10;
    assert(prepared.build(casters,basis));assert(prepared.bounds[0]!=previous);
    // Exact inclusive page boundaries, stable draw order and content identity.
    casters.resize(4);for(auto& c:casters)c.offset[0]=c.offset[1]=c.offset[2]=0;
    std::vector<std::array<float,4>> box{{-1,-1,0,0},{6,6,7,7},{7,0,8,1},{1,1,2,2}};
    auto selection=Shadow::PreparedCasters::select(casters,box,{0,0});
    assert((selection.indices==std::vector<std::size_t>{0,1,3}));
    auto hash=selection.hash;
    auto cached=prepared.admit({0,0},std::move(selection));assert(cached && cached==prepared.find({0,0}));
    casters[0].version=2;assert(Shadow::PreparedCasters::select(casters,box,{0,0}).hash!=hash);
    casters[0].version=1;casters[1].binding=17;assert(Shadow::PreparedCasters::select(casters,box,{0,0}).hash!=hash);
    casters[1].binding=0xffffffffu;casters[1].index_format=57;assert(Shadow::PreparedCasters::select(casters,box,{0,0}).hash!=hash);
    Shadow::PreparedCasters::Selection too_large;too_large.indices.resize(8u*1024u*1024u/sizeof(std::size_t)+1);
    assert(!prepared.admit({1,1},std::move(too_large)));assert(!too_large.indices.empty());
    assert(prepared.build(casters,basis));assert(!prepared.find({0,0}) && prepared.selection_bytes==0);
    casters.resize(8u*1024u*1024u/sizeof(std::array<float,4>)+1);
    assert(!prepared.build(casters,basis));assert(!prepared.matches(casters,basis));
    casters.clear();assert(prepared.build(casters,basis));assert(prepared.bounds.empty());
}
'''
        self.run_program(program)

    def run_program(self, program):
        build = ROOT / "Renderer/native/build"
        build.mkdir(exist_ok=True)
        # Windows scanners can briefly retain the just-exited executable. Its
        # disposable directory is not a test result and need not block cleanup.
        with tempfile.TemporaryDirectory(dir=build, ignore_cleanup_errors=True) as directory:
            path = Path(directory)
            cpp = path / "query_test.cpp"
            cpp.write_text(program)
            if os.name == "nt":
                batch = path / "test.bat"
                batch.write_text(r'''@echo off
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "C3X_TEST_VS=%%I"
if not defined C3X_TEST_VS exit /b 2
call "%C3X_TEST_VS%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 exit /b 2
cd /d "%~dp0"
cl /nologo /std:c++17 /EHsc /W4 /WX query_test.cpp /Fe:query_test.exe
if errorlevel 1 exit /b 1
query_test.exe
exit /b %errorlevel%
''')
                result = native_command_result("Renderer/native", f'call "{batch}"')
                self.assertEqual(result["returncode"], 0, result["output_tail"])
            else:
                compiler = shutil.which("clang++") or shutil.which("g++")
                if not compiler:
                    self.skipTest("C++ compiler unavailable")
                executable = path / "query_test"
                subprocess.run([compiler, "-std=c++17", str(cpp), "-o", str(executable)], check=True)
                subprocess.run([str(executable)], check=True)

    def test_raster_grid_keeps_world_coordinates_under_camera_translation(self):
        source = (ROOT / "Renderer/native/render_core/raster_grid.h").read_text().replace("#pragma once", "")
        self.run_program(source + r'''
#include <cassert>
int main(){
 using namespace c3x_renderer::render_core;
 for(int extent:{128,256,512})for(int origin=-1100;origin<1100;origin+=13)
 for(int pixel=-20;pixel<800;pixel+=7)for(int pan=-131;pan<=131;++pan){
   int old_floor=raster_region_floor(pixel,raster_phase(origin,extent),extent);
   int moved_floor=raster_region_floor(pixel+pan,raster_phase(origin+pan,extent),extent);
   assert(old_floor<=pixel && old_floor+extent>pixel);
   assert(moved_floor==old_floor+pan);
   assert(pixel-old_floor==(pixel+pan)-moved_floor);
   assert(raster_translation_matches(raster_phase(origin,extent),raster_phase(origin+pan,extent),pan,extent));
 }
 // A canonical wrap may change phase at zooms whose map period is not
 // divisible by the raster extent. Such old pixels must not be translated.
 for(int tile_width:{64,96,128,160,192})for(int map_width:{32,100,102}){
   int before=raster_anchor_phase(17,0,tile_width,128);
   int after=raster_anchor_phase(17+3,map_width,tile_width,128);
   bool same=(map_width*tile_width/2)%128==0;
   assert(raster_translation_matches(before,after,3,128)==same);
 }
}
'''.replace("#include <cassert>", "#include <cassert>\n#include <initializer_list>"))


if __name__ == "__main__":
    unittest.main()
