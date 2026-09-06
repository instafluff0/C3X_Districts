"""Audit selected installed skin against normalized relief; extract owned inputs.

Source-specific offline tooling only. The generic renderer reads packet data.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[5]
V2=ROOT/'Renderer/terrain_lab/v2'
sys.path.insert(0,str(ROOT/'Renderer/tools/asset_compiler'))
import terrain_relief_builder as reader

def main():
    skin=Path(os.environ.get('C3X_CIV6_ENVIRONMENT_SKIN',str(Path.home()/'Library/Application Support/Steam/steamapps/workshop/content/289070/1702339134')))
    package=skin/'Platforms/Windows/BLPs/terrain/TerrainElementSet_Base.blp'
    resources,elements,report=reader.inspect_terrain_element_package(package)
    output=V2/'fixtures/relief/selected-source'
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    with package.open('rb') as source:
        for name,element in elements.items():
            hill=name=='ART_DEF_TERRAIN_ELEMENT_HILL'
            if not hill and 'SINGLEMOUNTAIN00' not in name:continue
            for role,lods in element['channels'].items():
                for lod in lods:
                    source.seek(report['big_data_offset']+lod['relative_offset'])
                    payload=source.read(lod['bytes'])
                    if len(payload)!=lod['bytes']:raise ValueError('short source read')
                    if hill:
                        normalized=ROOT/f'Renderer/packs/Civ5EnvironmentSkin/textures/relief/hills/standard/{role}_lod{lod["level"]}.dds'
                        destination=output/f'hills_{role}_lod{lod["level"]}.dds'
                        destination.write_bytes(reader.make_r8_dds(lod['width'],lod['height'],payload,62 if role=='region_ids' else 61))
                    else:
                        variant=int(name[-3:])
                        normalized=ROOT/f'Renderer/packs/Civ5EnvironmentSkin/textures/relief/mountains/standard/variant_{variant:02d}/{role}_lod{lod["level"]}.dds'
                        destination=None
                    records.append(dict(entry=name,channel=role,lod=lod['level'],parameters=element['parameters'],source_payload_sha256=hashlib.sha256(payload).hexdigest(),normalized_matches_selected_source=normalized.is_file() and normalized.read_bytes()[148:]==payload,owned_output=destination.relative_to(ROOT).as_posix() if destination else None))
    style=skin/'ArtDefs/TerrainStyle.artdef'
    styles={}
    for collection in ET.parse(style).findall('./m_RootCollections/Element'):
        name=collection.find('m_CollectionName').get('text')
        if name not in ('RidgelineMountain','StandardHills','DuneDesertHills'):continue
        styles[name]={v.find('m_ParamName').get('text'):float(v.findtext('m_fValue')) for v in collection.findall('./Element/m_Fields/m_Values/Element') if v.find('m_fValue') is not None}
    audit=dict(schema='c3x.q4.selected_relief_source_audit.v1',source_locator='C3X_CIV6_ENVIRONMENT_SKIN or installed Steam workshop item 1702339134',package_sha256=hashlib.sha256(package.read_bytes()).hexdigest(),style_sha256=hashlib.sha256(style.read_bytes()).hexdigest(),style_values=styles,channels=records,interpretation='Selected skin overrides hill pixels and height_scale (10 versus base 14). Standard mountain payloads match; scalar ArtDef values do not prove our coordinate interpretation or source-engine composition.')
    target=V2/'audits/relief/SELECTED_SOURCE_AUDIT.json'
    target.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(channels=len(records),mismatches=sum(not r['normalized_matches_selected_source'] for r in records),audit=target.relative_to(ROOT).as_posix())))

if __name__=='__main__':main()
