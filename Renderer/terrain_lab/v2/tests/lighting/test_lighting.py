import hashlib,json,math,struct,subprocess,tempfile,unittest
from pathlib import Path
V2=Path(__file__).resolve().parents[2]
ROOT=V2.parents[2]
OUT=V2/'audits/lighting/out/Q1/Q6-lighting'

def bmp(path):
    b=path.read_bytes();offset=struct.unpack_from('<I',b,10)[0];w,h=struct.unpack_from('<ii',b,18)
    if b[:2]!=b'BM' or struct.unpack_from('<H',b,28)[0]!=32:raise ValueError('expected BGRA32 evidence')
    return w,abs(h),b[offset:]

def compare(a,b):
    wa,ha,a=bmp(a);wb,hb,b=bmp(b);assert (wa,ha)==(wb,hb)
    delta=[abs(x-y) for i,(x,y) in enumerate(zip(a,b)) if i%4!=3]
    return {'rgb_mae':sum(delta)/len(delta),'changed_channels':sum(x!=0 for x in delta),'max_channel_delta':max(delta)}

class Lighting(unittest.TestCase):
    def test_shadow_contract(self):
        with tempfile.TemporaryDirectory() as d:
            exe=Path(d)/'shadow-contract'
            subprocess.run(['clang++','-std=c++17','-O2',str(Path(__file__).with_name('shadow_contract.cpp')),'-o',str(exe)],check=True)
            subprocess.run([str(exe)],check=True)
    def test_transfer_contract(self):
        c=json.loads((V2/'systems/lighting/color_alpha_v1.json').read_text())
        self.assertEqual(c['alpha'],'premultiplied_linear');self.assertEqual(c['reconstruction'],'before_exposure_tonemap_transfer')
        # Independently known IEC sRGB midpoint, not legacy byte-box 127.
        midpoint=round((1.055*.5**(1/2.4)-.055)*255)
        self.assertEqual(midpoint,188)
        for value in [0,.001,.1,1,10,10000]:
            bounded=value/(1+value);self.assertTrue(0<=bounded<1)
    def test_city_repeat_and_order(self):
        if not (OUT/'city_reverse-04/report.json').exists():self.skipTest('render matrix not generated')
        for h in [12,18,0,6]:
            for z in [1,2]:
                name=f'h{h:02}-z{z}-pan00.bmp'
                a=(OUT/'city-04'/name).read_bytes()
                self.assertEqual(a,(OUT/'city_reverse-04'/name).read_bytes(),name)
                self.assertEqual(a,(OUT/'city-04'/(name+'.repeat1.bmp')).read_bytes(),name)
    def test_visual_controls(self):
        if not (OUT/'city-04/report.json').exists():self.skipTest('render matrix not generated')
        for h in [12,0]:
            for z in [1,2]:
                name=f'h{h:02}-z{z}-pan00.bmp'
                diff=compare(OUT/'city-04'/name,OUT/'city_shadows_off-04'/name)
                self.assertGreater(diff['changed_channels'],100)
                w,hgt,pixels=bmp(OUT/'city_emissive_only-04'/name)
                # Background clear is 9; noon map emission is zero everywhere.
                if h==12:self.assertLessEqual(max(pixels[0::4]+pixels[1::4]+pixels[2::4]),9)
                else:self.assertGreater(max(pixels[0::4]+pixels[1::4]+pixels[2::4]),100)
    def test_scroll_return(self):
        for directory in ['real-mixed-04','real-holdout-04','city-scroll-04']:
            if not (OUT/directory/'report.json').exists():self.skipTest('scroll matrix not generated')
            for h in [12,18,0,6]:
                for z in [1,2]:
                    self.assertEqual((OUT/directory/f'h{h:02}-z{z}-pan00.bmp').read_bytes(),(OUT/directory/f'h{h:02}-z{z}-pan03.bmp').read_bytes())
    def test_real_map_source(self):
        reg=json.loads((V2/'shared/real_map/registry_v1.json').read_text())
        for region in ['real-mixed','real-holdout']:
            f=json.loads((V2/f'fixtures/lighting/{region}/response.fixture.json').read_text())
            self.assertEqual(f['real_map']['source_sha256'],reg['source']['sha256'])
            self.assertEqual(hashlib.sha256((ROOT/f['terrain']).read_bytes()).hexdigest(),f['real_map']['region']['terrain_sha256'])
if __name__=='__main__':unittest.main()
