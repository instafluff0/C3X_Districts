"""Execute the benchmark's synthetic world-fixed object placement."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class NavigationFixtureTests(unittest.TestCase):
    def test_timing_mode_does_not_require_disabled_trace_records(self):
        import re
        import tempfile
        import textwrap
        from pathlib import Path
        from types import SimpleNamespace
        source=(ROOT/"Renderer/native/record_navigation_evidence.py").read_text()
        start=source.index('    completion["trace_path_validation"]')
        end=source.index('    if args.boundary_fixture:\n',start)
        check=textwrap.dedent(source[start:end])
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            for mode in ("diagnostic","timing"):
                args=SimpleNamespace(instrumentation=mode,local_region_revisions=True,
                                     shared_scene_surface=False,automatic_scene_surface=True,
                                     prepared_resource_pass=True,boundary_fixture=False)
                completion={};result={"returncode":0}
                exec(check,dict(args=args,out=out,completion=completion,result=result,re=re))
                self.assertEqual(1 if mode=="diagnostic" else 0,result["returncode"])
                if mode=="timing":
                    self.assertNotIn("shared_scene_surface_frames",completion)
                    self.assertEqual("unavailable-timing-mode",completion["trace_path_validation"])
            (out/"renderer.log").write_text("stage=render-region-cache local_revisions=1\nstage=shared-scene-surface valid=1\n")
            args.instrumentation="diagnostic";completion={};result={"returncode":0}
            exec(check,dict(args=args,out=out,completion=completion,result=result,re=re))
            self.assertEqual(0,result["returncode"])
            self.assertEqual(1,completion["local_region_revision_frames"])
            self.assertEqual(1,completion["shared_scene_surface_frames"])

    def test_compiled_city_texture_references_join_measurement_identity(self):
        import tempfile
        import struct
        from pathlib import Path
        from unittest.mock import patch
        from Renderer.native import record_navigation_evidence as evidence
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            texture=root/"Renderer/packs/Shared Materials/body texture.dds"
            texture.parent.mkdir(parents=True);texture.write_bytes(b"first payload")
            pack=root/"Renderer/packs/CityCompositionRuntime/city.bin"
            pack.parent.mkdir(parents=True)
            name=texture.relative_to(root).as_posix().encode()
            # Parse the production header/material table, not a filename regex.
            pack.write_bytes(b"C3XCITY2"+struct.pack("<6I",1,1,1,0,0,0)+
                             struct.pack("<I",len(name))+name+struct.pack("<6I",0,0,0,0,0,0))
            world=root/"Renderer/lab/.local/verification/world.csv"
            world.parent.mkdir(parents=True);world.write_text("fixture")
            with patch.object(evidence,"ROOT",root):
                first=evidence.inputs();key=texture.relative_to(root).as_posix()
                assert key in first
                texture.write_bytes(b"changed payload");second=evidence.inputs()
                assert first[key]["sha256"]!=second[key]["sha256"]
                texture.unlink()
                with self.assertRaises(FileNotFoundError):evidence.inputs()

    def test_dense_objects_keep_world_identity_and_draw_eligibility(self):
        source=(ROOT/"Renderer/native/biq_preview.cpp").read_text()
        seed="std::uint32_t preview_seed("+source.split("std::uint32_t preview_seed(",1)[1].split("\nbool preview_units",1)[0]
        body="if(dense_scene)for(auto & tile:tiles) {"+source.split("if(dense_scene)for(auto & tile:tiles) {",1)[1].split("    if (animate && !dense_scene",1)[0]
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
#include <cstring>
#include <cstdint>
#include <vector>
template<std::size_t N>void strcpy_s(char(&out)[N],char const* value){assert(std::strlen(value)<N);std::strcpy(out,value);}
'''+seed+r'''
void place(std::vector<c3x_renderer_tile_v1>& tiles,int map_width,bool dense_scene=true){
'''+body+r'''
}
int main(){
 std::vector<c3x_renderer_tile_v1> original;
 for(int y=0;y<48;++y)for(int x=y%2;x<48;x+=2){
  c3x_renderer_tile_v1 t={};t.tile_x=x;t.tile_y=y;t.real_terrain_type=y<36?2:11;
  t.tile_flags=x<24?C3X_RENDERER_TILE_RENDER:C3X_RENDERER_TILE_TOPOLOGY_HALO;
  t.resource_id=t.city_id=t.barbarian_tribe_id=-1;original.push_back(t);
 }
 auto first=original,shifted=original;place(first,48);
 for(auto& t:shifted){t.tile_x-=48;t.anchor_x+=317;t.anchor_y-=93;}
 place(shifted,48);unsigned cities=0,farms=0,mines=0,camps=0,land_resources=0,water_resources=0;
 for(std::size_t i=0;i<first.size();++i){
  auto a=first[i],b=shifted[i];assert(a.tile_flags==original[i].tile_flags && b.tile_flags==a.tile_flags);
  b.tile_x+=48;b.anchor_x-=317;b.anchor_y+=93;assert(std::memcmp(&a,&b,sizeof(a))==0);
  cities+=a.city_id>=0;farms+=(a.improvement_flags&C3X_RENDERER_IMPROVEMENT_IRRIGATION)!=0;
  mines+=(a.improvement_flags&C3X_RENDERER_IMPROVEMENT_MINE)!=0;
  camps+=(a.improvement_flags&C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP)!=0;
  if(a.resource_id>=0){land_resources+=a.real_terrain_type==2;water_resources+=a.real_terrain_type==11;}
  if(a.real_terrain_type==11)assert(!a.road_mask && !a.improvement_flags && a.city_id<0);
 }
 assert(cities && farms && mines && camps && land_resources && water_resources);
 auto disabled=original;place(disabled,48,false);
 assert(std::memcmp(disabled.data(),original.data(),original.size()*sizeof(original[0]))==0);
}
''')


if __name__ == "__main__":
    unittest.main()
