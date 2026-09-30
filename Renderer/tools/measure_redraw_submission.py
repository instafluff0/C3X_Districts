"""Bounded C15/control redraw comparisons; no staging or reference replacement."""
from pathlib import Path
import gzip
import hashlib
import json
import argparse
from Renderer.tools import measure_redraw_navigation as navigation
from Renderer.lab.platform import windows_root

OUT=Path(__file__).resolve().parents[1]/'.cache/redraw-submission-step'


def run(label,arm,*,hour=12,frames=120,diagnostic=False,captures=False,scenario='capture_zoom',controls=None):
    navigation.OUT=OUT
    options={'C3X_SANDBOX_PASS_COUNTS':'1' if diagnostic else '0',
             'C3X_RENDERER_TRACE':'2' if diagnostic else '0',
             'C3X_RENDERER_PROFILE':'1' if diagnostic else '0',
             'C3X_SANDBOX_WITNESS_SCENARIO':'zoom' if scenario=='capture_zoom' else 'resident128',
             'C3X_SANDBOX_GPU_TIMESTAMPS':'0',
             'C3X_SANDBOX_FRAME_TIMINGS':'1'}
    directory=windows_root()/OUT.relative_to(navigation.ROOT).as_posix()/label
    options.update(C3X_RENDERER_TRACE_FILE=str(directory/'renderer.log') if diagnostic else '',
                   C3X_RENDERER_TRACE_BUFFERED='0',
                   C3X_SANDBOX_WORLD_CONTENT_DIR=str(directory) if diagnostic else '')
    options.update(controls or {})
    result=navigation.run(label,arm,'navigation' if scenario in ('capture_zoom','navigation') else scenario,
                          hour,frames,options,captures,'full_guest','client')
    archive_outputs(OUT/label)
    return result


def archive_outputs(directory):
    # Preserve complete pixels losslessly. Only this invocation's generated
    # BMP/depth outputs are compressed, never earlier evidence or asset inputs.
    manifest=directory/'lossless-images.json'
    record=json.loads(manifest.read_text()) if manifest.exists() else {}
    for path in [*directory.glob('*.bmp'),*directory.glob('*.depth')]:
        original=path.read_bytes();destination=path.with_suffix(path.suffix+'.gz')
        destination.write_bytes(gzip.compress(original,compresslevel=6))
        if gzip.decompress(destination.read_bytes())!=original:raise ValueError('Archive verification failed')
        record[path.name]={'sha256':hashlib.sha256(original).hexdigest(),'bytes':len(original),
                          'compressed_sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
        path.unlink()
    manifest.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('label');p.add_argument('--arm',default='candidate');p.add_argument('--hour',type=int,default=12)
    p.add_argument('--frames',type=int,default=120);p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--captures',action='store_true');p.add_argument('--scenario',default='capture_zoom',choices=['capture_zoom','navigation','zoom','stationary'])
    p.add_argument('--control',action='append',default=[])
    a=p.parse_args();run(a.label,a.arm,hour=a.hour,frames=a.frames,diagnostic=a.diagnostic,captures=a.captures,
        scenario=a.scenario,controls=dict(x.split('=',1) for x in a.control))
