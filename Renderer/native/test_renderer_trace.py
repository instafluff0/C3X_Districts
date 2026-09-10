"""Execute the production debug trace with a deterministic OS clock and sink."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class RendererTraceTests(unittest.TestCase):
    def test_timestamps_usage_counts_results_and_disabled_output(self):
        run_cpp(r'''
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include "Renderer/native/c3x_renderer_api.h"
using DWORD=unsigned long;
struct LARGE_INTEGER {long long QuadPart=0;};
struct FILETIME {std::uint32_t dwLowDateTime=0,dwHighDateTime=0;};
constexpr unsigned MAX_PATH=260;
std::vector<std::string> lines;
char const* trace_level="2";long long clock_ticks=0;
DWORD GetEnvironmentVariableA(char const* name,char* out,unsigned size){
 if(std::strcmp(name,"C3X_RENDERER_TRACE"))return 0;
 assert(std::strlen(trace_level)<size);std::strcpy(out,trace_level);return std::strlen(out);
}
void QueryPerformanceFrequency(LARGE_INTEGER* out){out->QuadPart=1000000;}
void QueryPerformanceCounter(LARGE_INTEGER* out){out->QuadPart=clock_ticks+=1000;}
void GetSystemTimeAsFileTime(FILETIME* out){std::uint64_t ticks=116444736000000000ull+12340000ull;out->dwLowDateTime=std::uint32_t(ticks);out->dwHighDateTime=std::uint32_t(ticks>>32);}
DWORD GetCurrentThreadId(){return 7;} DWORD GetCurrentProcessId(){return 8;}
void OutputDebugStringA(char const* line){lines.emplace_back(line);}
int fopen_s(FILE** out,char const* path,char const* mode){*out=std::fopen(path,mode);return *out?0:1;}
#include "Renderer/native/renderer_trace.h"
int main(){
 RendererTrace trace;assert(lines.size()==1);
 assert(lines.back().find("qpc_frequency=1000000 utc_unix_ms=1234")!=std::string::npos);
 assert(lines.back().find("process=8 thread=7")!=std::string::npos);
 c3x_renderer_tile_v1 tiles[2]={};
 for(auto& t:tiles){t.tile_flags=C3X_RENDERER_TILE_RENDER;t.tile_x=3;t.tile_y=5;t.anchor_x=212;t.anchor_y=170;
  t.city_id=1;t.road_mask=15;t.railroad_mask=15;t.resource_id=5;t.unit_type_id=1;
  t.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION|C3X_RENDERER_IMPROVEMENT_MINE|C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP;}
 tiles[1].tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO;
 c3x_renderer_frame_v1 frame={};frame.tiles=tiles;frame.tile_count=2;frame.tile_width=128;frame.tile_height=64;
 auto id=trace.usage_view(frame);assert(id==1);
 assert(lines.back().find("origin_x=20 origin_y=10")!=std::string::npos);
 assert(lines.back().find("captured=2 visible=1 cities=1 roads=1 railroads=1 farms=1 mines=1 camps=1 resources=1 tile_units=1")!=std::string::npos);
 c3x_renderer_output_v1 output={};output.geometry_tiles_built=99;
 trace.usage_result(id,C3X_RENDERER_RESULT_OK,42000,output);
 assert(lines.back().find("call_ms=42.000 built=99")!=std::string::npos);
 trace.usage_result(id,C3X_RENDERER_RESULT_ERROR,42000,output);
 assert(lines.back().find("call_ms=42.000 built=0")!=std::string::npos);
 frame.tile_count=0;trace.usage_view(frame);assert(lines.back().find("origin_valid=0")!=std::string::npos);
 for(auto const& line:lines)assert(line.size()<1024 && line.back()=='\n');
 auto count=lines.size();trace_level="0";RendererTrace disabled;
 assert(disabled.usage_view(frame)==0);disabled.usage_result(1,1,0,output);disabled.write("test","",true);
 assert(lines.size()==count);
}
''')


if __name__ == '__main__':
    unittest.main()
