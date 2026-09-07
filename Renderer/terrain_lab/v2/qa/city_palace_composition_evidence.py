"""Recheck preserved single-era capital growth, matched pixels and packet composition."""
import json
import math
import subprocess
from PIL import Image, ImageDraw
from city_growth_evidence import ROOT, V2, OUT, FIX, read, sha, placement
from city_growth_hierarchy_evidence import difference
from city_connected_growth_evidence import augmentation, boxes, clipped_area
from city_scene_pass import executable, Cache
from city_growth_layout import connected, shares_frontage
from city_light_buffer_probe import payload
from mesh_fingerprint import geometry_digest

BASE = OUT / 'city-palace-composition-r2'
CASES = {'asian-small': 93, 'asian-medium': 94, 'asian-large': 92,
         'asian-palace-off': 95, 'ancient-medium': 77, 'ancient-coast': 98}


def dry_clearance(a):
    site = a['benchmark_region']
    if a['anchor_tile'] != [3, 2]:
        site += '-' + '-'.join(map(str, a['anchor_tile']))
    grid = read(FIX / 'city-scene-foundation' / site / 'surface.json')['samples']
    rows = []
    all_boxes = boxes(a)
    for index, (i, box) in enumerate(zip(a['instances'], all_boxes)):
        pad = .024 if i['slot'] == 'capital' else .012
        padded = [v + (-pad if j < 2 else pad) for j, v in enumerate(box)]
        assert max(abs(v) for v in padded) <= a['footprint_half_extent_tiles'] + 1e-8
        for j, other in enumerate(all_boxes):
            if j != index:
                assert not (padded[0] < other[2] and padded[2] > other[0]
                            and padded[1] < other[3] and padded[3] > other[1])
        low = [math.floor((padded[j] - .12 + 1) / .04) for j in range(2)]
        high = [math.ceil((padded[j+2] + .12 + 1) / .04) for j in range(2)]
        assert min(low) >= 0 and max(high) <= 50
        vegetation = sum(grid[y*51+x]['real'] in (7, 8)
                         for y in range(low[1], high[1]+1) for x in range(low[0], high[0]+1))
        assert vegetation == 0
        inside = [s for k, s in enumerate(grid)
                  if box[0] <= -1+(k % 51)*.04 <= box[2]
                  and box[1] <= -1+(k // 51)*.04 <= box[3]]
        assert inside and all(s['base'] < 11 and s['shore_distance'] <= -.02 for s in inside)
        assert max(i['ground_height_range']) - min(i['ground_height_range']) <= 3
        rows.append({'slot': i['slot'], 'vegetation_samples_in_margin': vegetation,
                     'dense_interior_samples': len(inside), 'all_sampled_body_interior_dry': True,
                     'maximum_shore_distance': max(s['shore_distance'] for s in inside),
                     'samples_nearer_than_nominal_five_point_margin_0_05':
                         sum(s['shore_distance'] > -.05 for s in inside)})
    return rows


def main():
    cache = Cache(V2 / 'app/.cache')
    terrain = executable(V2 / 'qa/city_terrain_contract.cpp', cache)
    frame = executable(V2 / 'qa/city_shadow_frame_contract.cpp', cache)
    light = executable(V2 / 'qa/frame_data_contract.cpp', cache)
    intake = {}
    pack = ROOT / 'Renderer/packs/CityPalacesNormalized'
    manifest = read(pack / 'manifest.json')
    for name, asset in [('asian', 'city/palace/root/079e88fbd22dd1ce'),
                        ('ancient', 'city/palace/root/cbb941762b70fc8b')]:
        normal_path = FIX / 'city-palace-materials-r1' / (name+'-normals.json')
        normals = read(normal_path)
        landmark = read(pack / manifest['assets'][asset]['landmark'])
        for filename in landmark['components']['geometry']:
            mesh = read(pack / filename)
            assert geometry_digest(mesh) == normals['meshes'][mesh['asset_id']]['geometry_digest']
        opacity = read(FIX / 'city-palace-materials-r1' / (name+'-extra/source-evidence.json'))
        for binding in opacity:
            assert sha(ROOT / binding['texture']) == binding['dds_sha256']
        intake[name] = {'asset': asset, 'normalized_geometry_and_all_uvs_match': True,
                        'normal_mapping_sha256': sha(normal_path),
                        'mesh_count': len(normals['meshes']), 'source_primitive_count': len(normals['evidence']),
                        'opacity_bindings': opacity}
    cases, all_a = {}, {}
    for name, revision in CASES.items():
        path, a = augmentation(revision); all_a[name] = a
        base = OUT / 'city-palace-composition-r1' if name == 'ancient-medium' else BASE
        folder = base / name; render = folder / 'render'
        binding = read(folder / 'binding.json'); report = read(render / 'report.json')
        lights = read(ROOT / binding['lights'])
        assert sha(ROOT / binding['lights']) == binding['lights_sha256']
        assert lights['augmentation_sha256'] == sha(path)
        assert (folder / 'lights.bin').read_bytes() == payload(lights)
        assert sha(folder / 'lights.bin') == binding['payload_sha256']
        assert a['generator_profile']['era_policy'] == 'single_current_era_user_preference'
        assert a['uniform_scale_factor'] == 1.5 and a['graduated_growth']
        assert read(path.parent / 'surface.json')['region']['region']['extent'] == [10, 10]
        for key in ('source_normals', 'extra_materials'):
            assert sha(ROOT / a[key]['mapping']) == a[key]['sha256']
        assert not a['source_normals']['unmapped']
        b = boxes(a)
        house_boxes = [box for i, box in zip(a['instances'], b) if i['slot'] != 'capital']
        palace_boxes = [box for i, box in zip(a['instances'], b) if i['slot'] == 'capital']
        prefixes = {str(n): connected(house_boxes[:n], .08)
                    for n in (8, 16, 24) if n <= len(house_boxes)}
        assert all(prefixes.values())
        for palace in palace_boxes:
            assert any(shares_frontage(palace, house, .08) for house in house_boxes[:8])
            assert abs(max(palace[2]-palace[0], palace[3]-palace[1]) - .6) < 1e-8
        ex = read(ROOT / a['river_exclusion']['path'])
        assert sha(ROOT / a['river_exclusion']['path']) == a['river_exclusion']['sha256']
        assert sha(ROOT / a['river_exclusion']['terrain_packet']) == a['river_exclusion']['terrain_packet_sha256']
        hits = []
        for i, box in zip(a['instances'], b):
            pad = .024 if i['slot'] == 'capital' else .012
            padded = [v + (-pad if j < 2 else pad) for j, v in enumerate(box)]
            if any(clipped_area(p, padded) > 1e-12 for p in ex['polygons']): hits.append(i['slot'])
        assert not hits
        natural = read(OUT / 'river-corridor-r3' / a['benchmark_region'] / 'report.json')
        source_rows = [r for r in natural['outputs'] if r['zoom'] == 1]
        checks = []
        for index, record in enumerate(binding['packets']):
            for field in ('original', 'output'):
                assert sha(ROOT / record[field]) == record[field+'_sha256']
            checks.append(json.loads(subprocess.check_output([str(terrain), str(ROOT/source_rows[index]['packet']), str(ROOT/record['original'])], text=True)))
            checks.append(json.loads(subprocess.check_output([str(light), str(ROOT/record['original']), str(ROOT/record['output']), str(folder/'lights.bin')], text=True)))
            ref = read(ROOT/a['shadow_frame_report']['path'])['outputs'][index]['packet']
            checks.append(json.loads(subprocess.check_output([str(frame), str(ROOT/record['original']), str(ROOT/ref), str(ROOT/record['original'])], text=True)))
        windows = read(base / f'windows-{name}/evidence.json')
        assert len(windows['results']) == 2
        for index, row in enumerate(windows['results']):
            assert row['metrics']['pass']
            for field, p in [('d3d11_sha256', base/f'windows-{name}'/row['frame']),
                             ('packet_sha256', ROOT/report['packets'][index]['path']),
                             ('shader_sha256', render/'shaders/source.hlsl'),
                             ('reflection_sha256', render/'shaders/reflection/source.hlsl'),
                             ('post_sha256', render/'postprocess/source.hlsl')]:
                assert row[field] == sha(p)
        cases[name] = {'revision': revision, 'augmentation_sha256': sha(path),
                       'body_count': len(b), 'single_era': True, 'capital': a['capital'],
                       'connected_house_prefixes': prefixes, 'river_bank_overlap_slots': hits,
                       'dry_clearance': dry_clearance(a), 'lights': len(lights['lights']),
                       'packet_checks': checks, 'windows': windows}
    large = all_a['asian-large']
    houses = lambda a: [placement(i) for i in a['instances'] if i['slot'] != 'capital']
    palace = lambda a: [placement(i) for i in a['instances'] if i['slot'] == 'capital']
    for name in ('asian-small', 'asian-medium'):
        a = all_a[name]
        assert houses(a) == houses(large)[:len(houses(a))]
        assert palace(a) == palace(large)
        for key in ('source_biq_sha256', 'anchor_tile', 'projection', 'pool'):
            assert a[key] == large[key]
    assert houses(all_a['asian-palace-off']) == houses(all_a['asian-medium'])
    assert not palace(all_a['asian-palace-off'])
    comparisons = {}
    for before, after in [('asian-palace-off', 'asian-medium')]:
        comparisons['palace_visibility_control'] = [difference(BASE/before/'render'/f'h{h:02}-z1-pan00.png', BASE/after/'render'/f'h{h:02}-z1-pan00.png', (630, 280, 1050, 570)) for h in (12, 0)]
    comparisons['small_connected_palace'] = [difference(OUT/'city-palace-composition-r1/asian-small/render'/f'h{h:02}-z1-pan00.png', BASE/'asian-small/render'/f'h{h:02}-z1-pan00.png', (630, 280, 1050, 570)) for h in (12, 0)]
    before = augmentation(82)[1]
    for key in ('source_biq_sha256', 'anchor_tile', 'projection', 'pool', 'source_normals', 'extra_materials', 'emissive_gain', 'compound_ground'):
        assert before[key] == all_a['asian-small'][key], key
    assert [(i['asset'], i['scale']) for i in before['instances']] == [(i['asset'], i['scale']) for i in all_a['asian-small']['instances']]
    plan = lambda r: read(next((FIX/f'city-scene-r{r}').glob('*/layout-plan.json')))['instances']
    assert plan(100) == plan(92)
    assert plan(91) == plan(86)
    failed = {str(r): read(next((FIX/f'city-scene-r{r}').glob('*/capital-growth-search.json'))) for r in (96, 97, 99)}
    assert all(d['status'] == 'no_remaining_separated_palace_sites' for d in failed.values())
    evidence = {'classification': 'Provisional palace composition gain; no human approval or native promotion',
                'palace_source_intake': intake, 'cases': cases, 'matched_pixels': comparisons, 'exact_growth_prefixes': [8, 16, 24],
                'exact_palace_off_house_control': True,
                'dry_land_centroid_inland_r100_equals_r92': True,
                'frontage_only_r91_equals_r86_not_counted_as_gain': True,
                'preserved_failed_searches': failed,
                'previously_capital_untuned_holdout': {'region': 'freshwater', 'anchor': [8, 3],
                    'terrain_origin': [52, 30], 'tile_count': 100, 'revision': 99,
                    'result': 'No fitted capital; six separated sites, 6259 total nodes. This does not prove infeasibility.'},
                'limitations': ['Coastal r98 has six dense body samples inside the nominal .05 shoreline margin; all remain dry at the .02 clipping boundary. Sampling is not an exact continuous shoreline proof.',
                    'Ancient inland r77 predates final staged planning; independent prefix/frontage checks pass.',
                    'Source BRDF/environment response remains partial; local light spill is authored approximation.',
                    'No new water-reflection quality claim; earlier capital-lake witness retained.',
                    'Broader culture/era/size coverage and all existing milestone/manual gates remain open.']}
    target = V2/'audits/beauty/CITY_PALACE_COMPOSITION_EVIDENCE.json'
    target.write_text(json.dumps(evidence, indent=2)+'\n')
    canvas = Image.new('RGB', (760, 520)); draw = ImageDraw.Draw(canvas)
    for col, name in enumerate(('asian-palace-off', 'asian-medium')):
        for row, h in enumerate((12, 0)):
            canvas.paste(Image.open(BASE/name/'render'/f'h{h:02}-z1-pan00.png').crop((640, 290, 1020, 530)), (380*col, 260*row+20))
            draw.text((380*col+5, 260*row+4), name+' | '+('day' if h else 'night'), fill='white')
    canvas.save(BASE/'palace-off-on-native.png')
    print(target.relative_to(ROOT))


if __name__ == '__main__': main()
