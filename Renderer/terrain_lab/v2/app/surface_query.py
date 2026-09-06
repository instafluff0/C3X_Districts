#!/usr/bin/env python3
"""Query the exact pinned terrain builder at owner-supplied tile-local points."""
import argparse
import json
from pathlib import Path
from runner import ROOT, APP, fixture, owned, executables, pack_identity, packet, relative
from cache import Cache, canonical, file_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--points',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--zoom',type=int,choices=[1,2],default=1);a=p.parse_args()
    f,m=fixture(a.fixture.resolve()); owned(a.points.resolve(),f['track']);owned(a.output.resolve(),f['track'])
    if m['provider']!='frozen_l21': raise ValueError('surface query requires a declared frozen terrain provider; other surface owners must publish their own versioned sampler')
    cache=Cache(APP/'.cache');scene,_=executables(cache);packs=pack_identity(cache,f['packs'])
    result=packet(cache,f,m,scene,12,a.zoom,packs,query=a.points.resolve())
    lines=result.read_text().splitlines();header=lines[0].split(',')
    if header[0]!='C3X_SURFACE_QUERY_V1': raise ValueError('surface contract drift')
    names=['column','row','u','v','height','screen_x','screen_y','depth','normal_x','normal_y','normal_z','shore_distance','base','real']
    samples=[dict(zip(names,map(float,l.split(',')))) for l in lines[1:]]
    for sample in samples:
        sample['screen_x']/=a.zoom;sample['screen_y']/=a.zoom
    report=dict(schema='c3x.lab_v2.surface_query.v1',provider='frozen_l21',classification='source_adaptation',fixture=relative(a.fixture),fixture_sha256=file_hash(a.fixture),points_sha256=file_hash(a.points),terrain_sha256=file_hash(ROOT/f['terrain']),pack_identity=packs,builder_sha256=file_hash(scene),region=f.get('real_map'),zoom=a.zoom,projection=dict(zip(['width','height','origin_x','origin_y','half_width','half_height','vertical_scale'],map(float,header[1:]))),coordinate_contract='tile-local column,row,u,v; u/v in [0,1]; source height in frozen authoring pixel units; geometric normal matches pinned terrain shader; screen_x/y are final output pixels',samples=samples)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(canonical(report));print(relative(a.output))
if __name__=='__main__':main()
