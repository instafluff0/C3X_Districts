"""Preserve city growth, era distribution and grounding metadata for Lab use.

This reads authored parameters, not the source engine's placement algorithm.
No roads or native renderer code are changed by this probe.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def fields(element):
    result = {}
    for field in element.findall('./m_Fields/m_Values/Element'):
        name = field.find('m_ParamName').attrib['text']
        if name in result:
            raise ValueError('duplicate city-generator field: ' + name)
        value = {x.tag: x.get('text', x.text) for x in field if x.tag != 'm_ParamName'}
        result[name] = value
    return result


def record(element):
    children = {}
    collection_merge = {}
    for collection in element.findall('./m_ChildCollections/Element'):
        name = collection.find('m_CollectionName').attrib['text']
        children[name] = [record(x) for x in collection.findall('Element')]
        collection_merge[name] = collection.findtext('m_ReplaceMergedCollectionElements', default='false') == 'true'
    return {'name': element.find('m_Name').attrib['text'],
            'fields': fields(element), 'children': children,
            'replace_merged_collections': collection_merge,
            'append_merged_parameter_collections': element.findtext('m_AppendMergedParameterCollections', default='false') == 'true'}


def parse(path):
    root = ET.parse(path).getroot()
    return {c.find('m_CollectionName').attrib['text']: [record(x) for x in c.findall('Element')]
            for c in root.findall('./m_RootCollections/Element')
            if c.find('m_CollectionName').attrib['text'] in ('Generator', 'GroundingMaterials')}


def normalized_profile(generator, era_ids):
    values = generator['fields']
    scalar = lambda key: float(values[key]['m_fValue'])
    pair = lambda key: [float(values[key][axis]) for axis in ('m_x', 'm_y')]
    growth = []
    for item in generator['children']['GrowthStage']:
        f = item['fields']
        growth.append({'population': int(f['Var_Population']['m_nValue']),
                       'area_source_units': float(f['Var_CityArea']['m_fValue']),
                       'filler_ratio': float(f['Var_FillerRatio']['m_fValue']),
                       'filler_occupancy': float(f['Var_FillerOccupancy']['m_fValue']),
                       'scatter_grouping': float(f['Var_ScatterGrouping']['m_fValue'])})
    growth.sort(key=lambda x: x['population'])
    if len({x['population'] for x in growth}) != len(growth):
        raise ValueError('duplicate growth population')
    eras = {}
    for item in generator['children']['EraDistribution']:
        source_era = item['fields']['Era']['m_ElementName']
        if source_era not in era_ids:
            continue
        layers = []
        for distribution in item['children']['Distribution']:
            f = distribution['fields']
            source_layer = f['Era']['m_ElementName']
            if source_layer not in era_ids:
                raise ValueError('unmapped contributing art era: ' + source_layer)
            layers.append({'era': era_ids[source_layer],
                           'order_from_center': int(f['OrderFromCenter']['m_nValue']),
                           'weight': float(f['Weight']['m_fValue'])})
        if not layers or any(x['weight'] <= 0 for x in layers):
            raise ValueError('invalid era distribution weights')
        eras[era_ids[source_era]] = sorted(layers, key=lambda x: x['order_from_center'])
    return {'schema': 'c3x.lab.city_generator_parameters.v1',
            'model_scale': scalar('Var_ModelScale'),
            'block_variation': scalar('Var_BlockMaxVariation'),
            'block_height_range': pair('Var_BlockHeightRange'),
            'spines': {'enabled': values['Var_EnableSpines']['m_bValue'] == 'true',
                       'width_range': pair('Var_SpineWidthRange'),
                       'length_range': pair('Var_SpineLengthRange')},
            'growth': growth, 'era_layers': eras,
            'status': 'authored_parameters_recovered; placement_algorithm_not_recovered',
            'adaptation_required': ['Civ III diamond topology and authoritative anchors',
                                    'Source area-unit calibration',
                                    'Stable growth and source block footprints',
                                    'Population interpolation and filler selection semantics']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root', type=Path, default=Path.home() /
        "Library/Application Support/Steam/steamapps/common/Sid Meier's Civilization VI/Civ6.app/Contents/Assets")
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    strategy = json.loads(Path(__file__).with_name('city_render_strategy.json').read_text())
    era_ids = {x['source_art_era']: x['id'] for x in strategy['eras']}
    # Expansion2 changes the selection tag on its modern distribution and
    # adds a future distribution. Retain both instead of dropping the override.
    era_ids.update({'ARTERA_MOD_NO_FUTURE': 'modern', 'ARTERA_FUTURE': 'future'})
    sources = []
    for path in sorted(args.assets_root.rglob('CityGenerators*.artdef')):
        data = parse(path)
        if not any(data.values()):
            continue
        relative = path.relative_to(args.assets_root).as_posix()
        sources.append({'path': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **data})
        for generator in data.get('Generator', []):
            if generator['name'] == 'Gen_CityCenter':
                profile = normalized_profile(generator, era_ids)
                name = 'base' if relative.startswith('Base/') else relative.split('/')[1].lower()
                (args.output / (name + '-city-center.json')).write_text(json.dumps(profile, indent=2) + '\n')
    report = {'classification': 'source_parameter_evidence; no engine behavior or promotion claim',
              'sources': sources,
              'grounding_note': 'Material references are recorded only; no road topology or rendering work.',
              'prior_import_gap': 'City asset importer consumed tagged blocks but omitted Generator and GroundingMaterials.'}
    (args.output / 'source-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'source_documents': len(sources), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
