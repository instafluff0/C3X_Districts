"""Evidence checks for the private six-unit material/anatomy study."""
import hashlib, shutil, struct, subprocess, tempfile, unittest
from pathlib import Path
import numpy as np
from .prepare import ROOT, SOURCE, read
from .material_study import HERE,SUBJECTS
OUT=ROOT/'Renderer/lab/out/units/material-roster-study'
PACK=ROOT/'Renderer/packs/UnitMaterialRosterStudy'
FRAMES=ROOT/'Renderer/lab/out/units/source-roster-frame-pack'

class MaterialRosterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT/'source/Renderer/native/animation_runtime.h').exists():raise unittest.SkipTest('Prepare roster material study first')

    def test_all_payloads_preserve_geometry_and_animation(self):
        source=read(SOURCE/'manifest.json')['units'];bindings=read(PACK/'bindings.json')
        total=0
        for column,(slug,_,_) in enumerate(SUBJECTS):
            for action,data in source['unit/'+slug]['actions'].items():
                for i,p in enumerate(data['parts']):
                    a=(SOURCE/p['mesh']).read_bytes()
                    for row in (1,2):
                        b=(PACK/bindings['unit'+str(row*6+column)][action]['part'+str(i)]['mesh']).read_bytes()
                        self.assertEqual(b[:8],b'C3XANM2\0');self.assertEqual(a[12:32],b[12:32])
                        n=struct.unpack_from('<I',a,12)[0]
                        for j in range(n):self.assertEqual(a[32+j*64:32+(j+1)*64],b[32+j*88:32+j*88+64])
                        self.assertEqual(a[32+n*64:],b[32+n*88:]);total+=1
        self.assertGreater(total,400)

    def test_authored_frame_pair_and_uv_orientation(self):
        frames=read(FRAMES/'frames.json')['components']
        self.assertEqual(len(frames),32);self.assertEqual(sum(v['vertices'] for v in frames.values()),9716)
        for aid,v in frames.items():
            self.assertGreater(v['uv_tangent_mean_dot'],.80,aid);self.assertLess(v['uv_bitangent_mean_dot'],-.80,aid)
            self.assertLess(v['tangent_pair_mean_abs_dot'],.015,aid)

    def test_anatomy_and_sampling_equal_across_material_rows(self):
        bindings=read(PACK/'bindings.json');fits=read(ROOT/'Renderer/packs/UnitQualityStudy/study.json')['subjects']
        for column,(slug,_,_) in enumerate(SUBJECTS):
            for row in range(3):
                b=bindings['unit'+str(row*6+column)];f=fits['unit/'+slug]
                self.assertEqual(b['scale'],f['scale']);self.assertEqual(b['offset_z'],f['offset_z']);self.assertEqual(b['sample_scale'],2)

    def test_native_decoder_all_six_idle_and_moving(self):
        compiler=shutil.which('clang++') or shutil.which('g++')
        if not compiler:self.skipTest('No C++ compiler')
        bindings=read(PACK/'bindings.json');source=read(SOURCE/'manifest.json')['units']
        with tempfile.TemporaryDirectory() as tmp:
            exe=Path(tmp)/'test_frame';subprocess.run([compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-I',str(OUT/'source/Renderer/native'),'-I',str(HERE),str(HERE/'test_frame.cpp'),'-o',str(exe)],check=True)
            for column,(slug,_,_) in enumerate(SUBJECTS):
                for action in ('idle','move'):
                    a=SOURCE/source['unit/'+slug]['actions'][action]['parts'][0]['mesh'];b=PACK/bindings['unit'+str(6+column)][action]['part0']['mesh']
                    subprocess.run([str(exe),str(a),str(b)],check=True,capture_output=True)

if __name__=='__main__':unittest.main()
