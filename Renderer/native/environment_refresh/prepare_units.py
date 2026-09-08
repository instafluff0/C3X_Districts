"""Publish selected material/normal fidelity without changing animation palettes.

The full production roster and its native action timing remain authoritative.
Every fingerprinted normalized component uses its recovered authored normals.
No unit name or role selects rendering behavior.
Payloads are independent copies; generated output never aliases editable sources.
"""
from pathlib import Path
import argparse,hashlib,json,struct
ROOT=Path(__file__).resolve().parents[3]
PACKS=ROOT/'Renderer/packs'
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build_pack(target=None):
    source=PACKS/'UnitAnimationRuntime'
    target=Path(target or PACKS/'UnitAnimationFidelity').resolve()
    target.relative_to(ROOT/'Renderer')
    consumed={};copied=set();mesh_cache={}
    def local(path):
        path=Path(path).resolve();path.relative_to(ROOT)
        return path
    def content(path):
        path=local(path);data=path.read_bytes();key=path.relative_to(ROOT).as_posix()
        value=hashlib.sha256(data).hexdigest()
        if key in consumed and consumed[key]!=value:raise ValueError('Unit input changed during build: '+key)
        consumed[key]=value
        return data
    def read(path):return json.loads(content(path))
    def sha(path):return hashlib.sha256(content(path)).hexdigest()
    def separate(path):
        path=local(path)
        if target==path or target in path.parents or path in target.parents:
            raise ValueError('Unit output must not overlap source inputs')
    separate(source);separate(PACKS/'UnitNormalFidelity')
    manifest=read(source/'manifest.json');bindings=read(source/'bindings.json')
    normal_manifest=read(PACKS/'UnitNormalFidelity/manifest.json');pins={};updates={};modes={};poses=0;normal_cache={}
    for unit in manifest['units'].values():separate(PACKS/unit['source_pack'])
    def payload(root,relative):
        path=local(root/relative)
        if not path.is_relative_to(root.resolve()):raise ValueError('Unit payload path escapes its pack')
        return path
    def copy_payload(relative):
        if relative in copied:return
        a=payload(source,relative);b=payload(target,relative);data=content(a)
        b.parent.mkdir(parents=True,exist_ok=True)
        if not b.exists():b.write_bytes(data)
        elif digest(b)!=hashlib.sha256(data).hexdigest():raise ValueError('candidate payload collision')
        copied.add(relative)
    for index,(unit_id,unit) in enumerate(manifest['units'].items()):
        # The compiler preserves insertion order in bindings, while its manifest
        # is sorted. Resolve by native keys rather than assuming equal order.
        candidates=[b for k,b in bindings.items() if k.startswith('unit') and isinstance(b,dict)
                    and set(unit['civ3_ids'])=={b['key'+str(i)] for i in range(b['key_count'])}]
        if len(candidates)!=1:raise ValueError('ambiguous native unit binding '+unit_id)
        bound=candidates[0]
        for action,data in unit['actions'].items():
            for i,part in enumerate(data['parts']):
                for ch in part['material']['channels'].values():copy_payload(ch['texture'])
                for channel,field in [('ambient_occlusion','ao_texture'),('gloss','gloss_texture'),('emissive','emissive_texture')]:
                    if channel in part['material']['channels']:
                        bound[action]['part'+str(i)][field]=part['material']['channels'][channel]['texture']
                base=part['material']['channels']['base_color'];address=[base.get('address_'+a,'repeat') for a in ('u','v')]
                if any(a not in ('repeat','clamp') for a in address):raise ValueError('unsupported address')
                mode=sum(1<<a for a,v in enumerate(address) if v=='clamp')
                bound[action]['part'+str(i)]['address_mode']=mode
                relative=part['mesh'];blob=content(payload(source,relative))
                normal_key=unit['source_pack']+'/'+part.get('source_mesh','')
                if normal_key in normal_manifest['meshes']:
                    path=ROOT/normal_manifest['meshes'][normal_key]
                    if normal_key not in normal_cache:normal_cache[normal_key]=read(path)
                    recovered=normal_cache[normal_key];pins[str(path.relative_to(ROOT))]=consumed[local(path).relative_to(ROOT).as_posix()]
                    if recovered['normal_source']!='authored_octahedral_snorm8':raise ValueError('authored normal source missing')
                    source_path=payload(PACKS/unit['source_pack'],part['source_mesh'])
                    if source_path not in mesh_cache:mesh_cache[source_path]=read(source_path)
                    if consumed[source_path.relative_to(ROOT).as_posix()]!=recovered['normalized_mesh_sha256']:raise ValueError('normalized mesh changed')
                    mesh=mesh_cache[source_path]
                    mode=3 if recovered['address_mode']=='clamp' else 0
                    if recovered['address_mode'] not in ('repeat','clamp'):raise ValueError('unresolved primitive addressing')
                    bound[action]['part'+str(i)]['address_mode']=mode
                    for channel in ('base_color','ambient_occlusion','gloss'):
                        if channel in part['material']['channels']:
                            for axis in ('u','v'):part['material']['channels'][channel]['address_'+axis]=recovered['address_mode']
                    version,n,ni,nb,nf=struct.unpack_from('<5I',blob,8)
                    if version!=1 or n!=len(mesh['vertices']) or ni!=len(mesh['topology']['indices']):raise ValueError('source topology changed')
                    if tuple(mesh['topology']['indices'])!=struct.unpack_from('<'+str(ni)+'I',blob,32+n*64):raise ValueError('source index order changed')
                    out=bytearray(blob)
                    for j,v in enumerate(mesh['vertices']):
                        row=struct.unpack_from('<8f4I4f',blob,32+j*64)
                        # Rigid attachments may apply their authored uniform
                        # model scale to positions; normals stay in local space.
                        if max(abs(a-b) for a,b in zip(row[6:8],v['uv0']))>1e-6:raise ValueError('UV0 changed')
                        if max(abs(a-b) for a,b in zip(row[3:6],v['normal']))>1e-6:raise ValueError('normal basis changed')
                        struct.pack_into('<3f',out,32+j*64+12,*recovered['normals'][j])
                    assert out[32+n*64:]==blob[32+n*64:] # indices and every action palette exact
                    new=hashlib.sha256(out).hexdigest();dest=target/'clips'/f'{new}.bin';dest.parent.mkdir(parents=True,exist_ok=True)
                    if not dest.exists():dest.write_bytes(out)
                    elif digest(dest)!=new:raise ValueError('candidate normal payload collision')
                    updates[relative]=str(dest.relative_to(target));part['mesh']=updates[relative]
                    bound[action]['part'+str(i)]['mesh']=updates[relative];poses+=nf
                    part['normal_authority']=str(path.relative_to(ROOT))
                else:
                    catalog=local(PACKS/unit['source_pack']/'manifest.json')
                    if catalog.exists():raise ValueError('missing component normal authority '+normal_key)
                    # This absence distinguishes original procedural art from
                    # an imported component missing its authored normal record.
                    consumed[catalog.relative_to(ROOT).as_posix()]=None
                    copy_payload(relative)
                modes[mode]=modes.get(mode,0)+1
    target.mkdir(parents=True,exist_ok=True)
    for name,data in [('manifest.json',manifest),('bindings.json',bindings)]:
        (target/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    evidence={'status':'pass','source_manifest_sha256':sha(source/'manifest.json'),'bindings_sha256':digest(target/'bindings.json'),
      'unit_count':len(manifest['units']),'native_keys':sum(v['key_count'] for v in bindings.values() if isinstance(v,dict)),
      'address_mode_parts':modes,'normal_payloads':len(updates),'unchanged_palette_frames':poses,'source_sha256':pins,
      'settings':{'msaa':4,'anisotropy':16,'mip_bias':0,'render_scale':1},
      'limits':['Original generic assets preserve their authored normals; imported components use fingerprinted source octahedral normals.',
                'Native environment, team colors, projection and working self-shadow visibility adapt the selected Lab material response; the isolated witness LUT is not applied to the native sprite.']}
    return evidence,consumed

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Build a disposable pack without publishing a verification report')
    args=parser.parse_args()
    evidence,_=build_pack(ROOT/args.output if args.output else None)
    print(f"PASS {evidence['unit_count']} units, {evidence['native_keys']} native keys; animation palettes preserved")
if __name__=='__main__':main()
