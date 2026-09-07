"""Actual-source crop/wrap witness for the optional continental ground field."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

V2=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(V2/'app'))
import real_map


@unittest.skipUnless((V2/'fixtures/beauty/source-continental-r1/map_context.h').exists(),
                     'local derived source fields are required')
class ContinentalGroundTests(unittest.TestCase):
    def test_actual_adjacent_crops_wrap_and_extended_context(self):
        _,data=real_map.load_registry()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);exe=p/'probe'
            subprocess.run(['clang++','-std=c++17','-O2',str(Path(__file__).with_name('continental_ground_probe.cpp')),
                            '-o',str(exe)],check=True)
            for origins in (([60,50],[62,50]),([98,40],[0,40])):
                files=[]
                for i,origin in enumerate(origins):
                    f=p/(str(i)+'.csv');f.write_bytes(real_map.csv_bytes(data,{'origin':origin,'extent':[10,10],'halo':6}));files.append(str(f))
                run=subprocess.run([str(exe),*files],capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                result=json.loads(run.stdout)
                self.assertEqual(result['samples'],1764)
                self.assertLessEqual(result['maximum_crop_or_wrap_delta_px'],1e-5)


if __name__=='__main__':unittest.main()
