"""Offline selection records for house-pool and palace material investigations."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'


def selection(pool=None,palace=None):
    if bool(pool)==bool(palace):raise ValueError('select one house pool or palace')
    if pool:
        report=json.loads((V2/'audits/beauty/out/city-source-expanded-r1/build.json').read_text())
        selected=next(p['selected'] for p in report['pools'] if p['pool']=='city/pool/'+pool)
        return selected,ROOT/'Renderer/packs/CityStudyAuxiliaryUV',100,report
    report=json.loads((ROOT/'Renderer/preview/out/cities/palace_build.json').read_text())
    strategy=ROOT/'Renderer/tools/asset_compiler/palace_import_strategy.json'
    if hashlib.sha256(strategy.read_bytes()).hexdigest()!=report['strategy']['sha256']:
        raise ValueError('palace source normalization strategy changed')
    units=json.loads(strategy.read_text())['source_units_per_tile']
    root=next(p for p in report['root_catalog'] if p['asset_id']==palace)
    selected=[{'asset_id':palace,'package':root['source_package'],'entry':root['source_entry']}]
    return selected,ROOT/'Renderer/packs/CityPalacesNormalized',units,report
