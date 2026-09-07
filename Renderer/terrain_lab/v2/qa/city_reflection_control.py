"""Remove only city draws from the reflected scene; preserve the visible city."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from city_scene_pass import ROOT, V2, rel, save, file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=a.source.resolve();out=a.output.resolve()
    out.relative_to(V2/'audits/beauty/out')
    if out.exists():raise ValueError('preserve existing reflection control')
    text=(source/'shaders/reflection/source.hlsl').read_text()
    marker='Q6SceneOutput PSFeature(FeaturePixelInput input) {\n clip(input.q6_world.w-.5);'
    if text.count(marker)!=1:raise ValueError('reflected feature entry changed')
    text=text.replace(marker,marker+'\n if(input.material_index>=39.5)clip(-1);')
    out.mkdir(parents=True);(out/'reflection.hlsl').write_text(text)
    save(out/'control.json',{'classification':'City-only reflected-draw removal; visible scene and all packets unchanged',
         'source':rel(source),'source_report_sha256':file_hash(source/'report.json')})
    report=json.loads((source/'report.json').read_text())
    subprocess.run([sys.executable,str(V2/'qa/replay_shader.py'),'--report',str(ROOT/report['source_report']),
        '--shader',str(source/'shaders/source.hlsl'),'--reflection-shader',str(out/'reflection.hlsl'),
        '--post-shader',str(source/'postprocess/source.hlsl'),'--output',str(out/'render')],check=True)


if __name__=='__main__':main()
