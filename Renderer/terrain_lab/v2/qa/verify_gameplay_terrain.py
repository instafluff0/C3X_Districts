"""Verify matched real-map inputs, material closure and changed output pixels.

These are supporting checks. Actual image inspection and the existing human
visual approval gate remain mandatory; this script never marks beauty passed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'
AUDIT=V2/'audits/beauty'

def load(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def pixels(path):
    data=path.read_bytes();offset=struct.unpack_from('<I',data,10)[0]
    w,h=struct.unpack_from('<ii',data,18)
    bits=struct.unpack_from('<H',data,28)[0]
    assert bits==32 and w>0 and h!=0,'Expected renderer BGRA32 BMP'
    rows=[data[offset+y*w*4:offset+(y+1)*w*4] for y in range(abs(h))]
    if h>0:rows.reverse()
    return w,abs(h),b''.join(rows)

def difference(a,b):
    w,h,aa=pixels(a);ww,hh,bb=pixels(b);assert (w,h)==(ww,hh)
    count=0;x0=w;y0=h;x1=y1=-1
    for i in range(w*h):
        if aa[4*i:4*i+3]!=bb[4*i:4*i+3]:
            count+=1;x=i%w;y=i//w;x0=min(x0,x);x1=max(x1,x);y0=min(y0,y);y1=max(y1,y)
    return {'changed_pixels':count,'bounds':[x0,y0,x1+1,y1+1] if count else None,
            'output_size':[w,h]}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision',default='candidate-v5')
    args=parser.parse_args()
    results=[]
    sys_path=V2/'app'
    import sys
    sys.path.insert(0,str(sys_path))
    import real_map
    for region in ['coastal','inland','wilderness']:
        reports=[load(AUDIT/'out'/('gameplay-100-'+revision)/region/'report.json')
                 for revision in ['baseline-v1',args.revision]]
        previous=load(AUDIT/'out/gameplay-100-candidate-v2'/region/'report.json')
        before,after=[r['effective'] for r in reports]
        for key in ['settings','pack_hash','capabilities','postprocess_hash']:
            assert before[key]==after[key],(region,key)
        for key in ['real_map','terrain','scenarios','viewport','tile_count','packs']:
            assert before['fixture'][key]==after['fixture'][key],(region,key)
        assert before['module']['projection']==after['module']['projection']
        assert after['fixture']['tile_count']==100
        real_map.validate_provenance(after['fixture'])
        for r in reports+[previous]:
            for frame in r['outputs']:
                assert sha(ROOT/frame['image'])==frame['sha256']
                assert sha(ROOT/frame['source_metadata']['path'])==frame['source_metadata']['sha256']
        frames={};previous_frames={}
        for frame in reports[1]['outputs']:
            metadata=load(ROOT/frame['source_metadata']['path'])
            terrain=[d for d in metadata['draw_texture_bindings'] if not d['feature']]
            assert terrain and all(all(d['slots'][slot]>0 for slot in [52,53,54,57,58,59,60,61,62]) for d in terrain),region
            key=(frame['hour'],frame['zoom'],tuple(frame['offset']))
            match=next((x for x in reports[0]['outputs'] if (x['hour'],x['zoom'],tuple(x['offset']))==key),None)
            if match:
                delta=difference(ROOT/match['image'],ROOT/frame['image'])
                assert delta['changed_pixels']>0,(region,key,'unchanged output')
                frames[str(key)]=delta
            match=next((x for x in previous['outputs'] if (x['hour'],x['zoom'],tuple(x['offset']))==key),None)
            assert match,'Previous best needs the same day/night and zoom'
            previous_frames[str(key)]=difference(ROOT/match['image'],ROOT/frame['image'])
        assert len(frames)>=2,'Both gameplay zooms need a matched daytime pair'
        results.append({'region':region,'source_region':after['fixture']['real_map']['region'],
            'baseline_identity':reports[0]['render_identity'],'candidate_identity':reports[1]['render_identity'],
            'previous_best_identity':previous['render_identity'],
            'matched_frames':frames,'previous_best_frames':previous_frames,
            'all_desert_relief_materials_bound':True})
    record={'schema':'c3x.combined_terrain_checkpoint.v1','beauty_accepted':False,
        'gate_status':'no milestone closed; no Integration promotion',
        'intentional_changes':['mountain tiling material projection','river absorption and banks',
            'selected source hill height and source height-scale ratio','complete desert material bindings',
            'BIQ desert material coverage','continuous desert biome skirt and flat water join',
            'shared clock-driven night ambient and moon lighting'],
        'revision':args.revision,
        'results':results,'review':'Renderer/terrain_lab/v2/audits/beauty/CURRENT_VISUAL.md'}
    (AUDIT/'GAMEPLAY_TERRAIN_EVIDENCE.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
