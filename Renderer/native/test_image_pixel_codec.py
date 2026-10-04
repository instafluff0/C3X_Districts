"""Fullscreen UI uploads retain the wire format without per-pixel growth."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ImagePixelCodecTests(unittest.TestCase):
    def test_bulk_words_match_legacy_stream_and_validate_extents(self):
        run_cpp(r'''
#include "Renderer/native/input_recording/codec.h"
#include <cassert>
#include <chrono>
#include <cstdio>
int main(){
 using namespace c3x_inputs;
 std::vector<unsigned> pixels(2240*1260);
 unsigned value=0x9e3779b9;
 for(auto& pixel:pixels){value=value*1664525u+1013904223u;pixel=value;}
 auto begin=std::chrono::steady_clock::now();
 Writer legacy;legacy.bytes.push_back(0x53);legacy.reserve(pixels.size()*4);
 for(auto pixel:pixels)legacy(pixel);
 auto middle=std::chrono::steady_clock::now();
 Writer bulk;bulk.bytes.push_back(0x53);bulk.pixels(pixels.data(),pixels.size());
 auto end=std::chrono::steady_clock::now();
 assert(bulk.bytes==legacy.bytes); // deliberately unaligned word payload
 Reader reader{bulk.bytes,1};std::vector<unsigned> decoded(pixels.size());
 reader.pixels(decoded.data(),decoded.size());reader.done();assert(decoded==pixels);
 Writer empty;empty.pixels(nullptr,0);Reader no_words{empty.bytes};no_words.pixels(nullptr,0);no_words.done();
 bool rejected=false;try{empty.pixels(nullptr,payload_limit/4+1);}catch(std::runtime_error const&){rejected=true;}
 assert(rejected&&empty.bytes.empty());
 rejected=false;Reader short_input{bulk.bytes,2};
 try{short_input.pixels(decoded.data(),decoded.size());}catch(std::runtime_error const&){rejected=true;}
 assert(rejected&&short_input.at==2);
 c3x_renderer_gpu_images_v1 request={sizeof(request)};request.action=C3X_GPU_UPLOAD;
 request.pixels=pixels.data();request.pixel_count=unsigned(pixels.size());
 Writer encoded;images(encoded,request);Images restored;Reader input{encoded.bytes};images(input,restored);input.done();
 assert(restored.pixels==pixels&&restored.value.pixels==restored.pixels.data());
 std::printf("PASS %zu pixel words: legacy_encode_ms=%.3f bulk_encode_ms=%.3f identical_bytes=%zu\n",pixels.size(),
  std::chrono::duration<double,std::milli>(middle-begin).count(),
  std::chrono::duration<double,std::milli>(end-middle).count(),bulk.bytes.size());
}
''')
