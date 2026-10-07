"""Publish selected material/normal fidelity without changing animation palettes.

The full production roster and its native action timing remain authoritative.
Every fingerprinted normalized component uses its recovered authored normals.
No unit name or role selects rendering behavior.
Payloads are independent copies; generated output never aliases editable sources.
"""
from pathlib import Path
import argparse,hashlib,json,math,struct
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
PACKS=ROOT/'Renderer/packs'
Z_PIXELS=150*128/224 # live unit vertex shader height metric
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def idle_geometry(blob):
    """First-frame skinned positions and triangles of a C3XANM1/2 payload."""
    version,n,ni,nb,nf=struct.unpack_from('<5I',blob,8);stride=88 if version==2 else 64
    raw=np.frombuffer(blob,np.uint8,n*stride,32).reshape(n,stride)
    position=raw[:,:12].copy().view('<f4').reshape(n,3);joints=raw[:,32:48].copy().view('<u4').reshape(n,4)
    weights=raw[:,48:64].copy().view('<f4').reshape(n,4)
    triangles=np.frombuffer(blob,'<u4',ni,32+n*stride).reshape(-1,3)
    palette=np.frombuffer(blob,'<f4',nb*16,32+n*stride+ni*4).reshape(nb,4,4)
    points=np.zeros((n,3))
    for k in range(4):
        m=palette[joints[:,k]]
        points+=weights[:,k,None]*(position[:,0,None]*m[:,0,:3]+position[:,1,None]*m[:,1,:3]+position[:,2,None]*m[:,2,:3]+m[:,3,:3])
    return points,triangles
def silhouette(points,triangles,scale,offset,yaw,directions=range(8),supersample=2):
    """Median above-ground projected area/height/width in 128px-tile pixels."""
    rows=[]
    for direction in directions:
        angle=math.radians(yaw+direction*45);c,s=math.cos(angle),math.sin(angle)
        x=(points[:,0]*c-points[:,1]*s)*scale;y=(points[:,0]*s+points[:,1]*c)*scale;z=(points[:,2]+offset)*scale
        xs,ys=(x-y)*64*supersample,((x+y)*32-z*Z_PIXELS)*supersample
        tris=triangles[(z[triangles]>=0).any(axis=1)]
        x0,y0=int(np.floor(xs.min()))-1,int(np.floor(ys.min()))-1
        w,h=int(np.ceil(xs.max()))-x0+2,int(np.ceil(ys.max()))-y0+2
        mask=np.zeros((h,w),bool)
        for a,b,t in tris:
            ax,ay,bx,by,cx,cy=xs[a]-x0,ys[a]-y0,xs[b]-x0,ys[b]-y0,xs[t]-x0,ys[t]-y0
            area=(bx-ax)*(cy-ay)-(by-ay)*(cx-ax)
            if abs(area)<1e-9:continue
            left,right=int(max(0,np.floor(min(ax,bx,cx)))),int(min(w-1,np.ceil(max(ax,bx,cx))))
            top,bottom=int(max(0,np.floor(min(ay,by,cy)))),int(min(h-1,np.ceil(max(ay,by,cy))))
            px,py=np.meshgrid(np.arange(left,right+1)+.5,np.arange(top,bottom+1)+.5)
            u=((bx-px)*(cy-py)-(by-py)*(cx-px))/area;v=((cx-px)*(ay-py)-(cy-py)*(ax-px))/area
            mask[top:bottom+1,left:right+1]|=(u>=0)&(v>=0)&(u+v<=1)&(u*z[a]+v*z[b]+(1-u-v)*z[t]>=0)
        ys_,xs_=np.nonzero(mask)
        if len(ys_):rows.append(((np.ptp(ys_)+1)/supersample,(np.ptp(xs_)+1)/supersample,mask.sum()/supersample**2))
    if not rows:raise ValueError('unit has no visible idle silhouette')
    height,width,area=(float(np.median(column)) for column in zip(*rows))
    return {'height':height,'width':width,'area':area}
def ground_skinned_bodies(bindings,target):
    """Ground a skinned body that a small stray rigid primitive lifts.

    A rigid (single-bone) primitive with under 5% of a unit's idle vertices
    that reaches more than 10% of its height below the lowest skinned vertex is
    a mis-bound attachment standing at the root origin, not the ground: the
    skinned body then defines ground contact (the primitive is clipped below)."""
    grounded=[]
    for binding in bindings.values():
        if not isinstance(binding,dict) or 'idle' not in binding:continue
        idle=binding['idle'];skinned=[];rigid=[];heights=[]
        for i in range(idle['part_count']):
            blob=(target/idle['part'+str(i)]['mesh']).read_bytes()
            z=idle_geometry(blob)[0][:,2]+binding['offset_z'];heights.append(z)
            (skinned if struct.unpack_from('<I',blob,20)[0]>1 else rigid).append(z)
        if not skinned or not rigid:continue
        total=sum(len(z) for z in heights);span=float(np.ptp(np.concatenate(heights)))
        floor=min(float(z.min()) for z in skinned)
        stray=[z for z in rigid if len(z)<.05*total and float(z.min())<floor-.10*span]
        if stray and floor>.02*span and all(float(z.min())>=floor-.03*span or len(z)<.05*total for z in rigid):
            binding['offset_z']-=floor;binding['ground_policy']='skinned_body';grounded.append(binding['key0'])
    return grounded
def idle_extent(binding,target):
    """Lowest and highest model-space z and the horizontal length of a binding's idle parts."""
    idle=binding['idle'];points=np.concatenate([idle_geometry((target/idle['part'+str(i)]['mesh']).read_bytes())[0] for i in range(idle['part_count'])])
    return float(points[:,2].min()),float(points[:,2].max()),float(max(np.ptp(points[:,0]),np.ptp(points[:,1])))
def float_at_waterline(bindings,target,domains,afloat,draft):
    """Units of a floating domain sit at their authored waterline, the model
    origin, so the hull below it is clipped by the water plane. A model with
    nothing below its origin takes the median waterline (fraction of idle
    height) of authored hulls with a similar height/length, else `draft`."""
    authored,pending=[],[]
    for binding in bindings.values():
        if not isinstance(binding,dict) or 'idle' not in binding or domains.get(binding['key0']) not in afloat:continue
        low,high,length=idle_extent(binding,target);aspect=(high-low)/max(length,1e-9)
        if low<0:
            binding['offset_z']=0.0;binding['ground_policy']='authored_waterline'
            authored.append((aspect,-low/(high-low)))
        else:pending.append((binding,aspect,low,high))
    for binding,aspect,low,high in pending:
        near=[f for a,f in authored if abs(math.log(a/aspect))<math.log(1.25)] or [f for _,f in authored] or [draft]
        binding['offset_z']=-(low+float(np.median(near))*(high-low));binding['ground_policy']='shape_neighbor_waterline'
    return [binding['key0'] for binding in bindings.values() if isinstance(binding,dict) and binding.get('ground_policy') in ('authored_waterline','shape_neighbor_waterline')]
def hover_flying_units(bindings,target,sprites,factor,minimum):
    """A unit whose native sprite floats above its own shadow (aircraft,
    missiles) keeps its lowest idle point that many pixels above the ground."""
    hovering=[]
    for binding in bindings.values():
        if not isinstance(binding,dict) or 'idle' not in binding:continue
        sprite=next((sprites[binding['key'+str(i)]] for i in range(binding['key_count']) if binding['key'+str(i)] in sprites),None)
        if not sprite or sprite.get('lift',0)<minimum:continue
        low,_,_=idle_extent(binding,target)
        binding['offset_z']=sprite['lift']*factor/(Z_PIXELS*binding['scale'])-low
        binding['hover_policy']='native_sprite_lift';hovering.append(binding['key0'])
    return hovering
def fit_sizes(bindings,target,sprites,factor):
    """Match each idle silhouette to its native sprite's area. Units without a
    sprite take the median change of measured units with a similar shape."""
    measured,pending=[],[]
    for binding in bindings.values():
        if not isinstance(binding,dict) or 'idle' not in binding:continue
        idle=binding['idle'];points,triangles=[],[];base=0
        for i in range(idle['part_count']):
            p,t=idle_geometry((target/idle['part'+str(i)]['mesh']).read_bytes());points.append(p);triangles.append(t+base);base+=len(p)
        ours=silhouette(np.concatenate(points),np.concatenate(triangles),binding['scale'],binding['offset_z'],binding.get('yaw_offset',225.0))
        sprite=next((sprites[binding['key'+str(i)]] for i in range(binding['key_count']) if binding['key'+str(i)] in sprites),None)
        if sprite:
            change=math.sqrt(sprite['area']/ours['area']);measured.append((ours['height']/ours['width'],change))
            binding['scale']*=change*factor;binding['fit_policy']='native_sprite_area'
        else:pending.append((binding,ours['height']/ours['width']))
    for binding,aspect in pending:
        near=[c for a,c in measured if abs(math.log(a/aspect))<math.log(1.25)] or [c for _,c in measured] or [1.0]
        binding['scale']*=float(np.median(near))*factor;binding['fit_policy']='native_sprite_area_shape_neighbors'
    return len(measured),len(pending)
def build_pack(target=None, source=None):
    source=Path(source or PACKS/'UnitAnimationRuntime')
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
    quality=read(ROOT/'Renderer/native/environment_refresh/unit_quality.json')
    frame_root=PACKS/quality['frame_pack'];separate(frame_root)
    frames=read(frame_root/'frames.json')['components'] if quality['units'] else {}
    frame_manifest=read(frame_root/'manifest.json') if quality['units'] else {}

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
        selected=quality['units'].get(unit_id)
        if selected:bound.update(selected)

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
                if selected:
                    record=bound[action]['part'+str(i)]
                    frame=frames[part['asset']]
                    frame_mesh=read(frame_root/frame['normalized_mesh'])
                    component=read(frame_root/frame_manifest['assets'][part['asset']]['component'])
                    local_scale=component['model_scale'] if component['binding_mode']=='rigid_attachment' else 1
                    old=(target/record['mesh']).read_bytes()
                    n=struct.unpack_from('<I',old,12)[0]
                    if len(frame_mesh['vertices'])!=n:raise ValueError('tangent vertex count changed')
                    new=bytearray(old[:32]);new[:8]=b'C3XANM2\0';struct.pack_into('<I',new,8,2)
                    for j,v in enumerate(frame_mesh['vertices']):
                        row=old[32+j*64:32+(j+1)*64];values=struct.unpack_from('<8f',row)
                        expected=[x*local_scale for x in v['position']]+v['normal']+v['uv0']
                        if max(abs(a-b) for a,b in zip(values,expected))>1e-6:raise ValueError('tangent/geometry identity mismatch')
                        new+=row+struct.pack('<6f',*(frame['tangents'][j]+frame['bitangents'][j]))
                    new+=old[32+n*64:]
                    relative='clips/'+hashlib.sha256(new).hexdigest()+'.bin'
                    destination=target/relative
                    if not destination.exists():destination.write_bytes(new)
                    elif destination.read_bytes()!=new:raise ValueError('candidate frame payload collision')
                    record['mesh']=part['mesh']=relative;record['material_model']=1
                    normal=part['material']['channels'].get('normal_0')
                    if normal:record['normal_texture']=normal['texture']
                    part['frame_authority']=str((frame_root/'frames.json').relative_to(ROOT))
                modes[mode]=modes.get(mode,0)+1
    target.mkdir(parents=True,exist_ok=True)
    grounded=ground_skinned_bodies(bindings,target)
    waterline=quality.get('waterline');floated=None
    if waterline:
        # Recipe domains are pack metadata; procedural sources have no recipe.
        domains={}
        for unit in manifest['units'].values():
            recipe=PACKS/unit['source_pack']/unit.get('source_recipe','')
            if recipe.is_file():domains[unit['civ3_ids'][0]]=read(recipe).get('domain')
        domains={binding['key0']:next((domains[binding['key'+str(i)]] for i in range(binding['key_count']) if binding['key'+str(i)] in domains),None)
                 for binding in bindings.values() if isinstance(binding,dict) and 'idle' in binding}
        floated=float_at_waterline(bindings,target,domains,set(waterline['domains']),float(waterline['draft']))
    sizing=quality.get('sizing');fitted=None;sprites=None
    if sizing:
        # Floating hulls match the native sprite on their visible area.
        sprites=read(ROOT/sizing['sprites'])['units']
        fitted=fit_sizes(bindings,target,sprites,float(sizing.get('factor',1)))
    hover=quality.get('hover');hovering=None
    if hover:hovering=hover_flying_units(bindings,target,sprites or read(ROOT/hover['sprites'])['units'],float(hover['factor']),float(hover['minimum']))
    garments=quality.get('owner_garments');painted=None
    if garments:
        # Units whose measured owner colour is too sparse tint one garment component.
        from Renderer.tools.asset_compiler import unit_owner_coverage as coverage
        sha(ROOT/'Renderer/tools/asset_compiler/unit_owner_coverage.py')
        painted=coverage.paint_garments(bindings,target,coverage.components_by_binding(manifest,bindings),
            float(garments['minimum']),float(garments['garment_share']),float(garments['target']),float(garments['strength']))
    for name,value in (quality.get('look') or {}).items():
        if name not in ('gain','saturation','owner') or not 0<=float(value)<=2:raise ValueError('unit look values are gain, saturation, owner in [0,2]')
        bindings['look_'+name]=float(value)
    for name,data in [('manifest.json',manifest),('bindings.json',bindings)]:
        (target/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    evidence={'status':'pass','source_manifest_sha256':sha(source/'manifest.json'),'bindings_sha256':digest(target/'bindings.json'),
      'unit_count':len(manifest['units']),'native_keys':sum(v['key_count'] for v in bindings.values() if isinstance(v,dict)),
      'address_mode_parts':modes,'sizing':{'native_sprite':fitted[0],'shape_neighbors':fitted[1]} if fitted else None,'owner_garments':painted,'grounded':grounded,
      'waterline':floated,'hover':hovering,'normal_payloads':len(updates),'unchanged_palette_frames':poses,'source_sha256':pins,
      'settings':{'msaa':4,'anisotropy':16,'mip_bias':0,'render_scale':'pack selected 1 or 4'},
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
