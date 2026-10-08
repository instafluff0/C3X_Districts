"""The runtime effect pack and sampler reproduce the reference effect graphs.

`write_runtime_pack` writes the flat pack; `render_core/effect_pack.h` must
load it (rejecting truncated or wrong-version packs) and
`render_core/effect_sampler.h` must place exactly the particles that
`effect_graph_compiler.sample_effect` defines for every authored profile:
same spawn ordinals, frames, positions, sizes, pivots, rotation and opacity, at both
zoom policies and across burst, delayed and continuous emitters.
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler import effect_graph_compiler as graphs

ROOT = Path(__file__).resolve().parents[2]

TIMES = (0, 1, 37, 150, 333, 600, 999, 1234, 1750, 2799, 2800)


def run_host(code, argument, timeout=120):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise unittest.SkipTest("C++ compiler unavailable")
    with tempfile.TemporaryDirectory() as directory:
        source, executable = Path(directory) / "parity.cpp", Path(directory) / "parity"
        source.write_text(code)
        subprocess.run([compiler, "-std=c++17", "-O1", "-I", str(ROOT), str(source), "-o", str(executable)], check=True)
        return subprocess.run([str(executable), str(argument)], check=True, capture_output=True, text=True,
                              timeout=timeout).stdout


class EffectSamplerParity(unittest.TestCase):
    def test_runtime_sampler_matches_reference_for_every_profile(self):
        compiled = graphs.compile_effect_graphs()
        aliases = graphs.runtime_aliases(compiled)
        names = sorted(compiled["profiles"]) + sorted(aliases)
        code = r'''
#include "Renderer/native/render_core/effect_pack.h"
#include <cstdio>
#include <fstream>
#include <iterator>
using namespace c3x_renderer::effects;
int main(int argc,char** argv){
 std::ifstream file(argv[1],std::ios::binary);std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)),{});
 Pack pack;if(!parse_pack(bytes.data(),bytes.size(),pack)){std::puts("REJECTED");return 1;}
 // A truncated or corrupted pack is rejected as a whole.
 if(parse_pack(bytes.data(),bytes.size()-1,pack)){std::puts("ACCEPTED TRUNCATED");return 1;}
 auto corrupt=bytes;corrupt[8]=2;if(parse_pack(corrupt.data(),corrupt.size(),pack)){std::puts("ACCEPTED VERSION");return 1;}
 parse_pack(bytes.data(),bytes.size(),pack);
 long long times[]={''' + ",".join(map(str, TIMES)) + r'''};
 for(auto const& profile:pack.profiles)for(int reduced=0;reduced<2;++reduced)for(long long t:times){
  std::vector<Particle> out;sample(profile,"impact/7",t,reduced!=0,out);
  for(auto const& p:out)std::printf("%s %d %lld %s %u %u %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
   profile.id.c_str(),reduced,t,profile.emitters[p.emitter].id.c_str(),p.ordinal,p.frame,p.position[0],p.position[1],
   p.position[2],p.size[0],p.size[1],p.atlas[0],p.atlas[1],p.atlas[2],p.atlas[3],p.opacity,p.rotation,p.pivot[0],p.pivot[1]);}
 return 0;}
'''
        with tempfile.TemporaryDirectory() as directory:
            summary = graphs.write_runtime_pack(compiled, Path(directory) / "pack")
            self.assertEqual(summary["profiles"], len(names))
            output = run_host(code, Path(directory) / "pack/effects.bin")
        compiled = {**compiled["profiles"], **{alias: compiled["profiles"][target] for alias, target in aliases.items()}}
        actual = {}
        for line in output.splitlines():
            fields = line.split()
            if len(fields) != 19:
                continue
            key = (fields[0], int(fields[1]), int(fields[2]), fields[3], int(fields[4]))
            actual[key] = [int(fields[5])] + [float(v) for v in fields[6:]]
        expected = {}
        for name in names:
            for reduced, zoom in ((0, "normal"), (1, "reduced")):
                for t in TIMES:
                    for p in graphs.sample_effect(name, compiled[name], "impact/7", t, zoom):
                        ordinal = int(p["id"].rsplit("/", 1)[1])
                        expected[(name, reduced, t, p["emitter"], ordinal)] = [p["frame"], *p["position_tile"],
                                                                                *p["size_tile"], *p["atlas_uv"],
                                                                                p["opacity"], p["rotation"], *p["pivot"]]
        self.assertGreater(len(expected), 100)
        self.assertEqual(sorted(expected), sorted(actual))
        for key, values in expected.items():
            self.assertEqual(values[0], actual[key][0], key)
            for a, b in zip(values[1:], actual[key][1:]):
                self.assertAlmostEqual(a, b, delta=2e-4 * max(1.0, abs(a)), msg=key)


if __name__ == "__main__":
    unittest.main()
