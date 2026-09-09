"""Optional checks against the prepared, private material-study snapshot."""
import hashlib, json, shutil, subprocess, tempfile, unittest
from pathlib import Path
import numpy as np
from .material_study import ROOT, HERE, SOURCE, SNAP, PACK, FRAME_PACK, OUT
from .prepare import read

class MaterialStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (SNAP/'native/animation_runtime.h').exists():raise unittest.SkipTest('Prepare private material study first')

    def test_native_frame_decode_and_rotation(self):
        compiler=shutil.which('clang++') or shutil.which('g++')
        if not compiler:self.skipTest('No C++ compiler')
        manifest=read(SOURCE/'manifest.json')['units']['unit/warrior']
        bindings=read(PACK/'bindings.json')
        baseline=SOURCE/manifest['actions']['idle']['parts'][0]['mesh']
        candidate=PACK/bindings['unit6']['idle']['part0']['mesh']
        with tempfile.TemporaryDirectory() as tmp:
            exe=Path(tmp)/'test_frame'
            subprocess.run([compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-I',str(SNAP/'native'),'-I',str(HERE),str(HERE/'test_frame.cpp'),'-o',str(exe)],check=True)
            subprocess.run([str(exe),str(baseline),str(candidate)],check=True)

    def test_source_frame_identity_and_orientation(self):
        frames=read(FRAME_PACK/'frames.json')['components']
        self.assertEqual(sum(c['vertices'] for c in frames.values()),1601)
        for aid,c in frames.items():
            self.assertEqual(c['offset'],20 if c['stride']==32 else 12)
            self.assertGreater(c['uv_tangent_mean_dot'],.89)
            self.assertLess(c['uv_bitangent_mean_dot'],-.89)
            t,b=np.array(c['tangents']),np.array(c['bitangents'])
            self.assertLess(float(np.max(np.abs(np.linalg.norm(t,axis=1)-1))),1e-12)
            self.assertLess(float(np.mean(np.abs(np.sum(t*b,axis=1)))),.006)

    def test_equal_size_sampling_and_unmodified_baseline(self):
        bindings=read(PACK/'bindings.json')
        manifest=read(SOURCE/'manifest.json')['units']['unit/warrior']
        production=next(v for v in read(SOURCE/'bindings.json').values() if isinstance(v,dict) and v.get('key0') in manifest['civ3_ids'])
        for i in range(18):
            unit=bindings['unit'+str(i)]
            for key in ('scale','offset_z','yaw_offset'):self.assertEqual(unit[key],production[key])
            self.assertEqual(unit['sample_scale'],2)
        for action,data in manifest['actions'].items():
            for i,p in enumerate(data['parts']):
                a=(SOURCE/p['mesh']).read_bytes();b=(PACK/bindings['unit0'][action]['part'+str(i)]['mesh']).read_bytes()
                self.assertEqual(a,b)
        for p,digest in read(PACK/'study.json')['files'].items():self.assertEqual(hashlib.sha256((PACK/p).read_bytes()).hexdigest(),digest)

    def test_production_artifacts_unchanged(self):
        before=read(OUT/'production-before.json')
        for p,digest in before.items():self.assertEqual(hashlib.sha256((ROOT/p).read_bytes()).hexdigest(),digest,p)

if __name__=='__main__':unittest.main()
