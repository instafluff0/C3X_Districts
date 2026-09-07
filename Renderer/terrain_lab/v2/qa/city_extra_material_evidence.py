"""Check city mask controls and wilderness clearance; no visual acceptance implied."""
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import tempfile

from city_source_surface_evidence import ROOT, V2, OUT, FIX, read, sha, compare

MODERN='american-modern-s1-at7-5'
WILD='american-modern-s1-wilderness-at6-6'
SMALL='american-modern-s0-wilderness-at6-6'


def augmentation(revision,name):
    return read(FIX/f'city-scene-r{revision}'/name/'augmentation.json')


def forest_clearance(a):
    # Audit the stored placement against the frozen dense terrain witness.
    # Local bounds are already rotated. Account for placement padding and the
    # requested vegetation margin, including the conservative grid envelope.
    grid=read(FIX/'city-scene-foundation/wilderness-6-6/surface.json')['samples']
    result=[]
    for instance in a['instances']:
        bounds=instance['local_bounds'];offset=instance['offset'];margin=.012+.12
        box=[bounds[i]+offset[i%2]+(-margin if i<2 else margin) for i in range(4)]
        low=[math.floor((box[i]+1)/.04) for i in range(2)]
        high=[math.ceil((box[i+2]+1)/.04) for i in range(2)]
        assert min(low)>=0 and max(high)<=50
        hits=sum(grid[y*51+x]['real'] in (7,8)
                 for y in range(low[1],high[1]+1) for x in range(low[0],high[0]+1))
        result.append({'slot':instance['slot'],'vegetation_samples_in_margin':hits})
    return result


def main():
    baseline=augmentation(32,MODERN)
    for rev in (34,35,36):
        a=augmentation(rev,MODERN)
        for key in ('instances','anchor_tile','source_biq_sha256','projection','pool','size',
                    'uniform_scale_factor','source_normals','source_surface','compound_ground'):
            assert a[key]==baseline[key],key
    parity=[read(OUT/f'city-scene-r{rev}'/folder/'evidence.json') for rev,folder in
            ((34,'windows-opacity'),(37,'windows-wilderness-opacity'),(39,'windows-small-wilderness'))]
    assert [len(p['results']) for p in parity]==[4,2,2]
    assert all(r['metrics']['pass'] for p in parity for r in p['results'])
    contracts={}
    with tempfile.TemporaryDirectory(prefix='city-mask-audit-') as temporary:
        binary=Path(temporary)/'contract'
        subprocess.run(['clang++','-std=c++17','-O2',str(V2/'qa/city_cutout_contract.cpp'),'-o',str(binary)],check=True)
        for rev,name,state in ((34,MODERN,'on'),(36,MODERN,'off'),(39,SMALL,'on')):
            packet=OUT/f'city-scene-r{rev}'/name/'combined-0.packet'
            output=subprocess.check_output([str(binary),str(packet),state],text=True)
            contracts[str(rev)]={**json.loads(output),'packet_sha256':sha(packet)}
    old=augmentation(37,WILD);small=augmentation(39,SMALL)
    for key in ('source_biq_sha256','benchmark_region','anchor_tile','projection','uniform_scale_factor'):
        assert old[key]==small[key],key
    for name,rev in ((WILD,37),(SMALL,39)):
        surface=read(FIX/f'city-scene-r{rev}'/name/'surface.json')
        assert surface['region']['region']['extent']==[10,10]
        if rev==37:region=surface['region'];terrain=surface['terrain_sha256']
        else:assert surface['region']==region and surface['terrain_sha256']==terrain
    rejected=forest_clearance(old);legal=forest_clearance(small)
    assert any(r['vegetation_samples_in_margin'] for r in rejected)
    assert not any(r['vegetation_samples_in_margin'] for r in legal)
    failed=read(FIX/f'city-scene-r38/{WILD}/layout-attempts.json')
    assert len(failed['attempts'])==33 and all(a['status']=='no_fit' for a in failed['attempts'])
    overlay=FIX/'city-extra-materials-r1/modern'
    source=read(overlay/'source-evidence.json');mapping=read(overlay/'mapping.json')
    for row in source:assert sha(ROOT/row['texture'])==row['dds_sha256']
    evidence={
        'classification':'Partial roof opening improvement; metalness diagnostic and medium wilderness composition unselected; no milestone promotion',
        'opacity_pixels':compare(OUT/f'city-scene-r32/{MODERN}/combined',OUT/f'city-scene-r34/{MODERN}/combined',roi=(650,350,1050,610)),
        'disabled_transport':compare(OUT/f'city-scene-r32/{MODERN}/combined',OUT/f'city-scene-r36/{MODERN}/combined',roi=(650,350,1050,610),exact=True),
        'unselected_metalness':compare(OUT/f'city-scene-r34/{MODERN}/combined',OUT/f'city-scene-r35/{MODERN}/combined',zooms=(1,),roi=(650,350,1050,610)),
        'material_overlay':{'sha256':sha(overlay/'mapping.json'),'normalized_materials':len(mapping['materials']),
                            'bindings':dict(Counter(r['role'] for r in source)),
                            'unique_textures':len({r['texture'] for r in source})},
        'packet_mask_contracts':contracts,'standalone_windows_parity':parity,
        'wilderness':{'anchor':[6,6],'terrain_sha256':terrain,'rejected_r37':rejected,
                      'r38_bounded_failure':failed,'r39_small_clearance':legal,
                      'r39_extent':small['footprint_half_extent_tiles'],
                      'interpretation':'Same anchor and terrain; size 0 four bodies fit at extent 0.8 after the unrendered 0.65 attempt failed. Size 1 seven bodies remain unresolved; 33 failed bounded searches do not prove infeasibility.'},
        'remaining':['Filtered environment specular and local lights','Wilderness medium/large layout with stable growth and clearance',
                     'Full culture/era/size city matrix','General importer material intake beyond modern Lab overlay',
                     'Complete visual review and existing milestone gates']}
    target=V2/'audits/beauty/CITY_EXTRA_MATERIAL_r34_r39_EVIDENCE.json'
    target.write_text(json.dumps(evidence,indent=2)+'\n');print(target.relative_to(ROOT))


if __name__=='__main__':main()
