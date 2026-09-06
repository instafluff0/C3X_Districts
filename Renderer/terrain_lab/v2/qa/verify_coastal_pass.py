"""Check real-map composition identities and actual source-cliff draw bindings.

Changed pixels locate edits; neither these checks nor source reuse accepts art.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'
sys.path.insert(0,str(V2/'app'))
sys.path.insert(0,str(V2/'qa'))
import real_map
from verify_gameplay_terrain import load,sha,difference

def verify(region,revision,previous=None):
    out=V2/'audits/beauty/out'
    baseline=out/('coast-pass-baseline-v5' if region=='longcoast' else 'gameplay-100-candidate-v5')/region/'report.json'
    if previous:baseline=out/('coast-pass-'+previous)/region/'report.json'
    candidate=out/('coast-pass-'+revision)/region/'report.json'
    b,c=load(baseline),load(candidate)
    baseline_frames=b['outputs'][:]
    night=out/'coast-pass-baseline-v5-night/longcoast/report.json'
    if not previous and region=='longcoast' and night.exists():
        night_report=load(night)
        assert night_report['effective']['fixture']==b['effective']['fixture']
        baseline_frames+=night_report['outputs']
    bf,cf=b['effective']['fixture'],c['effective']['fixture']
    for key in ('real_map','terrain','scenarios','viewport','tile_count','packs','settings'):
        assert bf[key]==cf[key],(region,key)
    assert b['effective']['module']['projection']==c['effective']['module']['projection']
    real_map.validate_provenance(cf)
    frames=[]
    for frame in c['outputs']:
        assert sha(ROOT/frame['image'])==frame['sha256']
        assert sha(ROOT/frame['source_metadata']['path'])==frame['source_metadata']['sha256']
        m=load(ROOT/frame['source_metadata']['path'])
        textures={t['packet_texture']:t for t in m['textures']}
        water_channels=[]
        shader=(ROOT/c['effective']['module']['shader']).read_text()
        if '#define Q3_SOURCE_WATER_NORMALS 1' in shader:
            terrain_draw=next(d for d in m['draw_texture_bindings'] if not d['feature'])
            water_channels=[textures[terrain_draw['slots'][s]] for s in range(20,24)]
            assert [t['view_format'] for t in water_channels]==[11,35,11,35]
            assert all(t['source_format']==t['view_format'] for t in water_channels)
        cliff_draws=[]
        for d in m['draw_texture_bindings']:
            t=textures.get(d['slots'][25],{})
            if not d['feature'] or 'cliff_large_' not in t.get('path',''):continue
            channels=[textures[d['slots'][s]] for s in range(25,29)]
            assert [t['view_format'] for t in channels]==[72,83,80,71]
            cliff_draws.append([t['sha256'] for t in channels])
        instances=[i for i in m['instances'] if i.get('class')=='source_coastal_rock']
        assert bool(instances)==bool(cliff_draws)
        assert all(i['source_uv_preserved'] and i['uniform_world_scale']>0 for i in instances)
        matched=next((x for x in baseline_frames if all(x[k]==frame[k] for k in ('hour','zoom','offset'))),None)
        assert matched,'Missing preserved matched frame'
        frames.append({'hour':frame['hour'],'zoom':frame['zoom'],
            'cliff_instances':len(instances),'complete_cliff_material_draws':len(cliff_draws),
            'source_water_channel_sha256':[t['sha256'] for t in water_channels],
            'difference':difference(ROOT/matched['image'],ROOT/frame['image']) if matched else None})
    assert any(f['difference'] for f in frames)
    return {'region':region,'source_region':cf['real_map']['region_id'],
        'source_biq_sha256':cf['real_map']['source_sha256'],
        'baseline_identity':b['render_identity'],'candidate_identity':c['render_identity'],'frames':frames}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision',default='rocks-r4')
    p.add_argument('--previous',help='Compare with this preserved coastal revision')
    p.add_argument('--regions',nargs='+',default=['coastal','inland','wilderness','longcoast'])
    a=p.parse_args()
    record={'schema':'c3x.coastal_composition_evidence.v1','revision':a.revision,
        'visual_accepted':False,'approval':None,'previous':a.previous,
        'results':[verify(r,a.revision,a.previous) for r in a.regions]}
    path=V2/'audits/beauty'/('COAST_'+a.revision+'_EVIDENCE.json')
    path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()
