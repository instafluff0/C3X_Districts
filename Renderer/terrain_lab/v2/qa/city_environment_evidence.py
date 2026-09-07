"""Verify preserved city environment trials without treating parity as visual acceptance."""
import json
import subprocess
from PIL import Image, ImageDraw
import numpy as np
from city_growth_evidence import ROOT, V2, OUT, FIX, read, sha
from city_growth_hierarchy_evidence import difference
from city_scene_pass import executable, Cache

BASE = OUT/'city-environment-r2'
CASES = {'inland-large': (51, (630,280,1050,570)),
         'wilderness-medium': (55, (700,220,1080,490)),
         'asian-medium': (94, (630,280,1050,570))}


def main():
    contract = executable(V2/'qa/city_metalness_contract.cpp', Cache(V2/'app/.cache'))
    cases = {}
    for name, (revision, roi) in CASES.items():
        folder = BASE/name; experiment = read(folder/'experiment.json')
        source = ROOT/experiment['source']; render = folder/'render'
        assert sha(source/'report.json') == experiment['source_report_sha256']
        assert sha(V2/'shaders/lighting/city_environment.hlsl') == experiment['helper_sha256']
        a_path = next((FIX/f'city-scene-r{revision}').glob('*/augmentation.json'))
        a = read(a_path)
        assert a['generator_profile']['era_policy'] == 'single_current_era_user_preference'
        assert read(a_path.parent/'surface.json')['region']['region']['extent'] == [10,10]
        checks = []
        for r in experiment['packets']:
            for k in ('original','output'): assert sha(ROOT/r[k]) == r[k+'_sha256']
            checks.append(json.loads(subprocess.check_output([str(contract), str(ROOT/r['original']), str(ROOT/r['output'])], text=True)))
        report = read(render/'report.json'); old = read(source/'report.json')
        if not experiment['enabled_bound_metalness']:
            assert report['packets'] == old['packets']
        windows = read(BASE/f'windows-{name}/evidence.json')
        assert len(windows['results']) == 2
        for index, r in enumerate(windows['results']):
            assert r['metrics']['pass']
            for k, p in [('d3d11_sha256', BASE/f'windows-{name}'/r['frame']),
                         ('packet_sha256', ROOT/report['packets'][index]['path']),
                         ('shader_sha256', render/'shaders/source.hlsl'),
                         ('reflection_sha256', render/'shaders/reflection/source.hlsl'),
                         ('post_sha256', render/'postprocess/source.hlsl')]: assert r[k] == sha(p)
        assert sha(render/'postprocess/source.hlsl') == sha(source/'postprocess/source.hlsl')
        pixels = [difference(source/f'h{h:02}-z1-pan00.png', render/f'h{h:02}-z1-pan00.png', roi) for h in (12,0)]
        cases[name] = {'source': experiment['source'], 'augmentation_sha256': sha(a_path),
                       'source_biq_sha256': a['source_biq_sha256'], 'tile_count': 100,
                       'source_body_count': len(a['instances']), 'material_packet_checks': checks,
                       'matched_pixels': pixels, 'windows': windows,
                       'selection': 'Provisional modern material candidate' if name!='asian-medium' else 'Unselected; roof hue still drifts toward gray'}
    disabled = BASE/'disabled/render'; original = ROOT/read(BASE/'disabled/experiment.json')['source']
    controls = []
    for h in (12,0):
        first = original/f'h{h:02}-z1-pan00.png'; last = disabled/f'h{h:02}-z1-pan00.png'
        assert np.array_equal(np.asarray(Image.open(first)), np.asarray(Image.open(last)))
        controls.append({'hour': h, 'pixel_exact': True, 'original_sha256': sha(first), 'disabled_sha256': sha(last)})
    disabled_windows = read(BASE/'windows-disabled/evidence.json')
    assert len(disabled_windows['results']) == 2 and all(r['metrics']['pass'] for r in disabled_windows['results'])
    disabled_report = read(disabled/'report.json')
    for index,r in enumerate(disabled_windows['results']):
        for key,path in [('d3d11_sha256',BASE/'windows-disabled'/r['frame']),
                         ('packet_sha256',ROOT/disabled_report['packets'][index]['path']),
                         ('shader_sha256',disabled/'shaders/source.hlsl'),
                         ('reflection_sha256',disabled/'shaders/reflection/source.hlsl'),
                         ('post_sha256',disabled/'postprocess/source.hlsl')]: assert r[key] == sha(path)
    direct = OUT/'city-environment-r1/inland-direct-only/render'
    assert [r['sha256'] for r in read(direct/'report.json')['packets']] == [r['sha256'] for r in read(BASE/'inland-large/render/report.json')['packets']]
    isolated = [difference(direct/f'h{h:02}-z1-pan00.png', BASE/'inland-large/render'/f'h{h:02}-z1-pan00.png', CASES['inland-large'][1]) for h in (12,0)]
    source_evidence = V2/'audits/beauty/CITY_ENVIRONMENT_SOURCE.json'
    se = read(source_evidence); assert sha(ROOT/se['disassembly']) == se['disassembly_sha256']
    evidence = {'classification': 'Partial modern material gain; no general environment, city or milestone acceptance',
                'cases': cases, 'disabled_control': controls, 'disabled_windows': disabled_windows,
                'isolated_environment_vs_direct_only': isolated, 'source_findings_sha256': sha(source_evidence),
                'rejected_trials': ['r1 analytic reflection omitted source roughness attenuation and washed out Asian roofs',
                    'r2 Asian roof hue remains unconvincing; retain the earlier palace appearance',
                    'The early American capital has no bound metalness in its old material layout; attempted preparation rejected before rendering'],
                'generalization': 'Wilderness medium was not used to choose sky colors or roughness attenuation. Identical recipe, source scale, placement, terrain and lighting; no regional adjustment.',
                'remaining': ['Recovered or authored environment probe calibration instead of the analytic hemisphere fallback',
                    'Full older American capital material intake; preserve its existing lake reflection evidence meanwhile',
                    'LEAN1 variance and exact active source material/environment constants',
                    'Dielectric roof response, facade variety, open ground, broader culture/era/size coverage',
                    'No new strong water-reflection claim; all human/native/milestone gates remain open']}
    target=V2/'audits/beauty/CITY_ENVIRONMENT_EVIDENCE.json';target.write_text(json.dumps(evidence,indent=2)+'\n')
    canvas=Image.new('RGB',(760,520));draw=ImageDraw.Draw(canvas)
    source=ROOT/read(BASE/'inland-large/experiment.json')['source']
    for column,(path,label) in enumerate([(source,'Previous'),(BASE/'inland-large/render','Modern material candidate')]):
        for row,h in enumerate((12,0)):
            canvas.paste(Image.open(path/f'h{h:02}-z1-pan00.png').crop((640,290,1020,530)),(column*380,row*260+20))
            draw.text((column*380+5,row*260+4),label,fill='white')
    canvas.save(BASE/'selected-modern-native.png')
    print(target.relative_to(ROOT))


if __name__=='__main__':main()
