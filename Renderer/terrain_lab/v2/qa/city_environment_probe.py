"""Isolate an authored environment reflection term on immutable city packets."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from city_scene_pass import ROOT, V2, rel, save, file_hash, executable, Cache, compact_packet


def shader(source, enabled):
    if 'q8_city_environment_lobe' in source:
        raise ValueError('source already contains an environment trial; use the preserved original scene')
    marker='Q6SceneOutput Q8_CITY_FEATURE_ENTRY(FeaturePixelInput p) {'
    if source.count(marker)!=1: raise ValueError('city material entry changed')
    helper=(V2/'shaders/lighting/city_environment.hlsl').read_text()
    source=source.replace(marker,helper+'\n'+marker)
    marker='  float3 roughness=q8_surface_sample(resource_base_texture_1,p.uv,repeat_uv).rgb;'
    if source.count(marker)!=1: raise ValueError('city specular material entry changed')
    if enabled:
        source=source.replace(marker,marker+'\n  lit+=q8_city_environment_specular(n,roughness,base,metalness,ao);')
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--disabled',action='store_true')
    p.add_argument('--enable-bound-metalness',action='store_true',help='Enable only existing nonempty generic metalness bindings; retain all original textures')
    a=p.parse_args();source=a.source.resolve();out=a.output.resolve()
    out.relative_to(V2/'audits/beauty/out')
    if out.exists():raise ValueError('preserve existing environment trial')
    if shutil.disk_usage(V2).free<8*2**30:raise ValueError('preserve 8 GiB free')
    report=json.loads((source/'report.json').read_text())
    out.mkdir(parents=True)
    source_report=ROOT/report['source_report'];changes=[]
    if a.enable_bound_metalness:
        base=json.loads(source_report.read_text());jobs=json.loads((source_report.parent/'batch.json').read_text())
        exe=executable(V2/'qa/city_enable_bound_metalness.cpp',Cache(V2/'app/.cache'))
        for index,(row,job) in enumerate(zip(base['outputs'],jobs)):
            original=ROOT/row['packet'];target=out/f'combined-{index}.packet'
            details=json.loads(subprocess.check_output([str(exe),str(original),str(target)],text=True))
            compact_packet(target,V2/'app/.cache/content')
            changes.append({'original':rel(original),'original_sha256':file_hash(original),
                'output':rel(target),'output_sha256':file_hash(target),**details})
            row['packet']=rel(target);job[0]=str(target)
        source_report=out/'report.json';save(source_report,base);save(out/'batch.json',jobs)
    for name,path in [('combined',source/'shaders/source.hlsl'),('reflection',source/'shaders/reflection/source.hlsl')]:
        (out/(name+'.hlsl')).write_text(shader(path.read_text(),not a.disabled))
    save(out/'experiment.json',{'classification':'Authored analytic sky/ground reflection; source cubearray and exact normalization remain unresolved',
        'source':rel(source),'source_report_sha256':file_hash(source/'report.json'),
        'helper_sha256':file_hash(V2/'shaders/lighting/city_environment.hlsl'),'enabled':not a.disabled,
        'enabled_bound_metalness':a.enable_bound_metalness,'packets':changes})
    subprocess.run([sys.executable,str(V2/'qa/replay_shader.py'),'--report',str(source_report),
        '--shader',str(out/'combined.hlsl'),'--reflection-shader',str(out/'reflection.hlsl'),
        '--post-shader',str(source/'postprocess/source.hlsl'),'--output',str(out/'render')],check=True)


if __name__=='__main__':main()
