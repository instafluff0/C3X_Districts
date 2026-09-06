import copy
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'Renderer/terrain_lab/v2/app'))
sys.path.insert(0,str(ROOT/'Renderer/terrain_lab/v2/shared'))
import real_map
from scene_exchange import intersects,validate
from cache import file_hash

class RealMapTests(unittest.TestCase):
    def test_wrap_and_nonwrap_neighbors(self):
        data=dict(width=24,height=24,wrap_x=True,wrap_y=False,tiles=[dict(sourceX=x,sourceY=y,base=2,real=2,bonus=0,overlays=0,riverMask=2 if x==0 else 0) for y in range(24) for x in range(y%2,24,2)])
        real_map.prepare(data)
        r=real_map.region_tiles(data,[22,12],[4,4],2)
        self.assertEqual(len(r),64)
        self.assertTrue(any(t['sourceX']==0 and t['riverMask']==2 for t in r))
        with self.assertRaises(ValueError):real_map.region_tiles(data,[0,0],[4,4],2)
        data['wrap_x']=False
        with self.assertRaises(ValueError):real_map.region_tiles(data,[22,12],[4,4],2)

    def test_registered_source_and_overlay_rejection(self):
        if not real_map.REGISTRY.exists():self.skipTest('local BIQ registry absent')
        reg,data=real_map.load_registry()
        self.assertEqual(reg['source']['sha256'],'a6a88d7fffcc567c3500bbd5aa947398dd48170d4f412aa1e518bb45ffe8453e')
        self.assertEqual(len(data['tiles']),5000)
        for r in reg['regions']:
            self.assertEqual(real_map.digest(real_map.csv_bytes(data,r)),r['terrain_sha256'])
        region=real_map.resolved_region(reg,data,'mixed',6)
        self.assertEqual(len(real_map.region_tiles(data,region['origin'],region['extent'],6)),256)
        fpath=ROOT/'Renderer/terrain_lab/v2/tests/platform/out/real-mixed-augmented/fixture.json'
        if not fpath.exists():self.skipTest('local augmented witness absent')
        f=json.loads(fpath.read_text());real_map.validate_provenance(f)
        for key in ['source_sha256','dataset_payload_sha256','profile']:
            bad=copy.deepcopy(f);bad['real_map'][key]='stale'
            with self.assertRaises(ValueError):real_map.validate_provenance(bad)
        bad=copy.deepcopy(f);bad['real_map']['overlay_sha256']='0'*64
        with self.assertRaises(ValueError):real_map.validate_provenance(bad)

class ExchangeTests(unittest.TestCase):
    def test_full_extent_and_margin_not_origin_only(self):
        road=[[0,-.1],[2,-.1],[2,.1],[0,.1]]
        crown=[[.8,.08],[1.2,.08],[1.2,.5],[.8,.5]]
        self.assertTrue(intersects(crown,road))
        clear=[[.8,.2],[1.2,.2],[1.2,.5],[.8,.5]]
        self.assertFalse(intersects(clear,road))
        self.assertTrue(intersects(clear,road,.1))
        self.assertTrue(intersects(road,crown))
        diagonal=[[-.5,-.5],[.5,.5],[.6,.4],[-.4,-.6]]
        self.assertTrue(intersects(diagonal,road))

    def test_world_identity_and_uniform_transform(self):
        scene=dict(schema='c3x.lab_v2.world_scene.v1',coordinate_space='tile_xyz_v1',terrain_sha256='0'*64,region_id='synthetic',meshes=[dict(id='mesh',positions=[[0,0,0],[1,0,0],[0,1,0]],normals=[[0,0,1]]*3,uv=[[0,0],[1,0],[0,1]],indices=[0,1,2])],materials=[dict(id='material',alpha_mode='opaque',caster=True)],instances=[dict(id='body',mesh='mesh',material='material',scale=1,rotation=[1,0,0,0,1,0,0,0,1],translation=[0,0,0])])
        validate(scene)
        bad=copy.deepcopy(scene);bad['instances'][0]['rotation'][0]=2
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(scene);bad['materials'][0]['alpha_mode']='cutout'
        with self.assertRaises(ValueError):validate(bad)

    def test_linear_invalid_samples_and_wire5_bounds(self):
        code='#include "'+str(ROOT/'Renderer/terrain_lab/v2/shared/color_response.h')+'"\n'+r"""
int main(int argc,char**argv){
 labv2::Packet p;p.width=p.height=2;p.color_branch=1;p.valid_rect={0,0,1,1};p.downsample=2;
 std::vector<uint16_t> rgba(16,0x7c00);for(int k=0;k<4;k++)rgba[k]=0x3c00;
 auto pixels=labv2::display_pixels(rgba,{255,0,0,0},2,2,2,p);
 if(pixels!=std::vector<uint8_t>({188,188,188,64}))return 1;
 p.binding_contract=2;p.buffers={{0,0,0,0,0,0,0,0,0,0,0,0},std::vector<uint8_t>(16),std::vector<uint8_t>(16)};
 p.buffers[0].resize(36);labv2::Draw d;d.count=3;d.stride=12;d.constant_buffer=1;d.frame_buffer=2;d.attributes={{3,0}};p.draws.push_back(d);
 if(!labv2::write_packet(argv[1],p))return 2;
 auto q=labv2::read_packet(argv[1]);if(q.binding_contract!=2||q.draws[0].frame_buffer!=2)return 3;
 q.geometry_contract=1;q.draws[0].stride=28;q.buffers[0].resize(84);
 q.draws[0].attributes={{3,0},{4,12}};q.draws[0].world_attribute=1;q.draws[0].geometry_flags=3;
 if(!labv2::write_packet(argv[1],q))return 5;
 q=labv2::read_packet(argv[1]);if(q.geometry_contract!=1||q.draws[0].world_attribute!=1)return 6;
 q.draws[0].geometry_flags=7;try{labv2::write_packet(argv[2],q);return 7;}catch(const std::runtime_error&){}
 q.draws[0].geometry_flags=3;q.draws[0].frame_buffer=3;
 try{labv2::write_packet(argv[2],q);return 4;}catch(const std::runtime_error&){}
 return 0;
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);source=tmp/'contract.cpp';source.write_text(code);exe=tmp/'contract'
            subprocess.run(['clang++','-std=c++17',str(source),'-o',str(exe)],check=True,capture_output=True)
            subprocess.run([str(exe),str(tmp/'valid.packet'),str(tmp/'bad.packet')],check=True)

    def test_atomic_image_replacement_preserves_hardlink(self):
        code='#include "'+str(ROOT/'Renderer/terrain_lab/v2/contracts/packet_v1.h')+'"\nint main(int argc,char**argv){std::vector<uint8_t> p(16,17);return labv2::write_bmp(argv[1],p,2,2,1)?0:1;}\n'
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);source=tmp/'image.cpp';source.write_text(code);exe=tmp/'image';subprocess.run(['clang++','-std=c++17',str(source),'-o',str(exe)],check=True,capture_output=True)
            target=tmp/'image.bmp';target.write_bytes(b'prior evidence');alias=tmp/'repeat.bmp';os.link(target,alias)
            subprocess.run([str(exe),str(target)],check=True)
            self.assertEqual(alias.read_bytes(),b'prior evidence')
            self.assertEqual(target.read_bytes()[:2],b'BM')

if __name__=='__main__':unittest.main()
