"""Generic source-fidelity gates over the complete roster and actual C++ skinner."""
import hashlib,json,math,struct,subprocess,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];PACKS=ROOT/'Renderer/packs'
def read(p):return json.loads(p.read_text())
class UnitFidelity(unittest.TestCase):
    def test_only_normals_and_material_addressing_change(self):
        old=PACKS/'UnitAnimationRuntime';new=PACKS/'UnitAnimationFidelity'
        bound=read(new/'bindings.json')
        a=read(old/'manifest.json');b=read(new/'manifest.json');self.assertEqual(a['units'].keys(),b['units'].keys())
        payloads=set();textures=set();components=set();samplers=set()
        authority=read(PACKS/'UnitNormalFidelity/manifest.json')['meshes']
        for uid,unit in a['units'].items():
            updated=b['units'][uid];self.assertEqual(unit['civ3_ids'],updated['civ3_ids'])
            self.assertEqual(unit['actions'].keys(),updated['actions'].keys())
            binding=next(v for v in bound.values() if isinstance(v,dict) and v.get('key0') in unit['civ3_ids'])
            for action,data in unit['actions'].items():
                for i,part in enumerate(updated['actions'][action]['parts']):
                    for channel,field in [('ambient_occlusion','ao_texture'),('gloss','gloss_texture'),('emissive','emissive_texture')]:
                        self.assertEqual(binding[action]['part'+str(i)].get(field),part['material']['channels'].get(channel,{}).get('texture'))
                self.assertEqual({k:v for k,v in data.items() if k!='parts'},{k:v for k,v in updated['actions'][action].items() if k!='parts'})
                for x,y in zip(data['parts'],updated['actions'][action]['parts']):
                    key=(x['mesh'],y['mesh'])
                    if key not in payloads:
                        original=(old/x['mesh']).read_bytes();fresh=(new/y['mesh']).read_bytes()
                        self.assertEqual(original[:32],fresh[:32]);count=struct.unpack_from('<I',original,12)[0]
                        self.assertEqual(original[32+count*64:],fresh[32+count*64:])
                        normal_key=unit['source_pack']+'/'+x.get('source_mesh','')
                        source=read(ROOT/authority[normal_key]) if normal_key in authority else None
                        for i in range(count):
                            start=32+i*64
                            self.assertEqual(original[start:start+12],fresh[start:start+12])
                            self.assertEqual(original[start+24:start+64],fresh[start+24:start+64])
                            if source:self.assertEqual(fresh[start+12:start+24],struct.pack('<3f',*source['normals'][i]))
                        payloads.add(key)
                    for channel,d in x['material']['channels'].items():
                        target=y['material']['channels'][channel];pair=d['texture'],target['texture']
                        if pair not in textures:
                            self.assertEqual((old/pair[0]).read_bytes(),(new/pair[1]).read_bytes())
                            textures.add(pair)
                    if y.get('normal_authority'):components.add(y['normal_authority'])
                    base=y['material']['channels']['base_color'];samplers.add((base.get('address_u'),base.get('address_v')))
        self.assertEqual(len(components),375);self.assertIn(('repeat','repeat'),samplers);self.assertIn(('clamp','clamp'),samplers)
        # Transform/fit, native aliases, loop policy, parts and owner colors stay exact.
        a=read(old/'bindings.json');b=read(new/'bindings.json')
        for k,u in b.items():
            if not isinstance(u,dict):continue
            for name,action in u.items():
                if not isinstance(action,dict) or 'part_count' not in action:continue
                for i in range(action['part_count']):
                    original_part=a[k][name]['part'+str(i)]
                    part=action['part'+str(i)];mode=part.pop('address_mode');self.assertIn(mode,range(4));part['mesh']=original_part['mesh']
                    # Current upstream bindings already carry these material
                    # channels. Compare their payloads before removing the
                    # allowed adapter fields from BOTH identity dictionaries.
                    original_part.pop('address_mode',None)
                    for field in ('ao_texture','gloss_texture','emissive_texture'):
                        if field in part:
                            relative=part.pop(field);self.assertEqual((old/relative).read_bytes(),(new/relative).read_bytes())
                            if field in original_part:self.assertEqual(original_part.pop(field),relative)
        self.assertEqual(a,b)
        print('Fidelity byte gates:',len(payloads),'payloads,',len(textures),'unchanged DDS chains,',len(components),'authored components')

    def test_inverse_transpose_across_families(self):
        import numpy as np
        root=PACKS/'UnitAnimationFidelity';m=read(root/'manifest.json');count=0;worst=0
        # Humanoid, civilian, mounted, mechanical, aircraft and naval witnesses.
        names=['warrior','archer','infantry','settler','worker','horseman','tank','fighter','galley']
        with tempfile.TemporaryDirectory(prefix='c3x-unit-normal-') as folder:
            exe=Path(folder)/'sample';out=Path(folder)/'vertices.bin'
            subprocess.run(['clang++','-std=c++17','-O2',str(ROOT/'Renderer/native/test_animation_runtime.cpp'),'-o',str(exe)],check=True)
            for name in names:
                unit=m['units']['unit/'+name]
                for action in ['idle','move','death']:
                    if action not in unit['actions']:continue
                    for part in unit['actions'][action]['parts']:
                        blob=(root/part['mesh']).read_bytes();_,n,ni,nb,nf,duration=struct.unpack_from('<5If',blob,8)
                        normal=np.array([struct.unpack_from('<3f',blob,44+i*64) for i in range(n)])
                        joints=np.array([struct.unpack_from('<4I',blob,64+i*64) for i in range(n)])
                        weight=np.array([struct.unpack_from('<4f',blob,80+i*64) for i in range(n)])
                        matrices=np.frombuffer(blob,dtype='<f4',offset=32+n*64+ni*4).reshape(nf,nb,4,4).astype(float)
                        for phase in [0,.417,1]:
                            frame=phase*(nf-1);a=int(frame);z=min(nf-1,a+1);t=frame-a
                            palette=matrices[a]+(matrices[z]-matrices[a])*t
                            linear=palette[:,:3,:3];det=np.linalg.det(linear);valid=np.abs(det)>=1e-10
                            if not np.all(valid[joints[weight>0]]):continue # authored hidden inventory
                            inv=np.zeros_like(linear);inv[valid]=np.linalg.inv(linear[valid]).transpose(0,2,1)
                            expected=sum(weight[:,i,None]*np.einsum('vi,vij->vj',normal,inv[joints[:,i]]) for i in range(4))
                            lengths=np.linalg.norm(expected,axis=1);mask=lengths>1e-8;expected[mask]/=lengths[mask,None]
                            subprocess.run([str(exe),str(root/part['mesh']),str(phase*duration),'0',str(out)],check=True)
                            actual=np.frombuffer(out.read_bytes(),dtype='<f4').reshape(n,8)[:,3:6]
                            error=float(np.max(np.abs(expected[mask]-actual[mask])));worst=max(worst,error)
                            self.assertLess(error,3e-4,(name,action,phase,error));count+=1
        print('Independent inverse-transpose gates:',count,'component poses; maximum error',worst)
if __name__=='__main__':unittest.main()
