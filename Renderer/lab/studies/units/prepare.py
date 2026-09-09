"""Build an isolated, opt-in unit sizing/sampling study from unchanged fidelity art.

Production packs are read-only. Explicit anatomy selections are study authoring,
not runtime dispatch by unit name. All generated payloads are independent copies.
"""
from pathlib import Path
import copy
import hashlib
import json
import math
import shutil
import struct
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'Renderer/packs/UnitAnimationFidelity'
OUTPUT = ROOT / 'Renderer/packs/UnitQualityStudy'
HEIGHT = 150 * 128 / 224


def read(path):
    return json.loads(path.read_text())


def poses(blob):
    version, n, ni, nb, nf = struct.unpack_from('<5I', blob, 8)
    if version not in (1,2) or blob[:8] != b'C3XANM'+str(version).encode()+b'\0':
        raise ValueError('Unsupported animation payload')
    stride=88 if version==2 else 64
    rows = [struct.unpack_from('<8f4I4f', blob, 32+i*stride) for i in range(n)]
    positions = np.array([v[:3]+(1.,) for v in rows])
    joints = np.array([v[8:12] for v in rows])
    weights = np.array([v[12:16] for v in rows])
    indices = np.frombuffer(blob, dtype='<u4', count=ni, offset=32+n*stride)
    matrices = np.frombuffer(blob, dtype='<f4', offset=32+n*stride+ni*4).reshape(nf, nb, 4, 4)
    for palette in matrices:
        yield sum(weights[:,i,None]*np.einsum('vi,vij->vj', positions, palette[joints[:,i]])
                  for i in range(4))[:,:3], indices


def anatomy_fit(points, reference_height, member_ratio):
    points = np.asarray(points)
    low, high = points[:,2].min(), points[:,2].max()
    span = float(high-low)
    if not math.isfinite(span) or span <= 0 or not 0 < member_ratio <= 4:
        raise ValueError('Invalid anatomy reference')
    return reference_height * member_ratio / span, -float(low)


def project(points, scale, offset, direction):
    angle = math.radians(225+direction*45)
    c, s = math.cos(angle), math.sin(angle)
    x = (points[:,0]*c-points[:,1]*s)*scale
    y = (points[:,0]*s+points[:,1]*c)*scale
    z = (points[:,2]+offset)*scale
    return np.stack(((x-y)*64, (x+y)*32-z*HEIGHT, z), axis=-1)


def visible_points(points, indices):
    """Include exact triangle/ground intersections, not buried inventory."""
    triangles = points[indices].reshape(-1,3,3)
    a = triangles.reshape(-1,3)
    b = np.roll(triangles,-1,axis=1).reshape(-1,3)
    crossing = (a[:,2] < 0) != (b[:,2] < 0)
    aa, bb = a[crossing], b[crossing]
    cut = aa+(bb-aa)*(-aa[:,2]/(bb[:,2]-aa[:,2]))[:,None]
    return np.concatenate((a[a[:,2]>=0],cut))


def build(source=SOURCE, output=OUTPUT, policy=None):
    if (output.resolve() == source.resolve() or source.resolve() in output.resolve().parents
            or output.resolve() in source.resolve().parents):
        raise ValueError('Study output must not overlap the production pack')
    policy = policy or read(HERE/'sizing.json')
    manifest, bindings = read(source/'manifest.json'), read(source/'bindings.json')
    units = manifest['units']
    selected = {}
    for uid, parts in policy['subjects'].items():
        assets = {uid+'/'+name for name in parts}
        samples = [next(poses((source/p['mesh']).read_bytes()))[0]
                   for p in units[uid]['actions']['idle']['parts'] if p['asset'] in assets]
        found = {p['asset'] for p in units[uid]['actions']['idle']['parts'] if p['asset'] in assets}
        if found != assets:
            raise ValueError('Missing anatomy selection: '+uid)
        selected[uid] = np.concatenate(samples)
    def binding(uid):
        return next(v for v in bindings.values() if isinstance(v,dict)
                    and v.get('key0') in units[uid]['civ3_ids'])
    ref = policy['reference']
    reference_height = float(np.ptp(selected[ref][:,2])) * binding(ref)['scale']
    member_reference = units[ref]['member_scale']
    study = {'schema':'c3x.unit_quality_study.v1','production_enabled':False,
             'source_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
             'reference_world_height':reference_height,'subjects':{},'files':{}}
    out_bindings = {'unit_count':len(selected)*3}
    out_manifest = {'schema':'c3x.unit_quality_study.v1','runtime_enabled':False,'units':{}}
    copies = set()
    for uid, anatomy in selected.items():
        old = binding(uid)
        scale, offset = anatomy_fit(anatomy, reference_height, units[uid]['member_scale']/member_reference)
        row = {'old_scale':old['scale'],'scale':scale,'ratio':scale/old['scale'],
               'old_offset_z':old['offset_z'],'offset_z':offset,
               'old_anatomy_height_px':float(np.ptp(anatomy[:,2]))*old['scale']*HEIGHT,
               'anatomy_height_px':float(np.ptp(anatomy[:,2]))*scale*HEIGHT,
               'actions':{}}
        for action, data in units[uid]['actions'].items():
            envelope = 0.; frame_count = 0
            for part in data['parts']:
                copies.add(part['mesh'])
                copies.update(c['texture'] for c in part['material']['channels'].values())
                for points, indices in poses((source/part['mesh']).read_bytes()):
                    frame_count += 1
                    for direction in range(1,9):
                        visible = visible_points(project(points,scale,offset,direction),indices)
                        if len(visible):envelope=max(envelope,float(np.max(np.abs(visible[:,:2]))))
            row['actions'][action] = {'component_frames':frame_count,
                'symmetric_sprite_side_px':2*math.ceil(envelope)+4}
        study['subjects'][uid]=row
        for mode in ('baseline','anatomy','ssaa'):
            key = 'PRTO_Lab_'+mode+'_'+uid.split('/')[-1]
            record = copy.deepcopy(old)
            for k in list(record):
                if k.startswith('key') and k != 'key_count':del record[k]
            record.update(key_count=1,key0=key,sample_scale=2 if mode=='ssaa' else 1)
            if mode!='baseline':record.update(scale=scale,offset_z=offset,fit_policy='explicit_anatomy_reference')
            out_bindings['unit'+str(len(out_manifest['units']))]=record
            clone=copy.deepcopy(units[uid]);clone['civ3_ids']=[key]
            out_manifest['units'][uid+'/'+mode]=clone
    output.mkdir(parents=True,exist_ok=True)
    for relative in sorted(copies):
        path=(source/relative).resolve()
        if not path.is_relative_to(source.resolve()):raise ValueError('Payload escapes pack')
        blob=path.read_bytes();digest=hashlib.sha256(blob).hexdigest()
        destination=output/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():shutil.copyfile(path,destination)
        elif hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:
            raise ValueError('Preserving conflicting study payload: '+relative)
        if destination.stat().st_ino==path.stat().st_ino:raise ValueError('Study aliases source')
        study['files'][relative]=digest
    for name,data in [('bindings.json',out_bindings),('manifest.json',out_manifest),('study.json',study)]:
        (output/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    return study


if __name__=='__main__':
    result=build()
    for uid,row in result['subjects'].items():
        print(uid, f"anatomy {row['old_anatomy_height_px']:.1f} -> {row['anatomy_height_px']:.1f} px; "
              f"scale x{row['ratio']:.3f}; idle canvas {row['actions']['idle']['symmetric_sprite_side_px']} px")
