"""Bounded ownership comparison using one corrected standalone client.

Detailed traces are separate from production timing. This reuses the earlier
measurement receipt/display parser and preserves both earlier evidence trees.
"""
from pathlib import Path
import argparse
from Renderer.tools import measure_redraw_navigation as navigation
from Renderer.lab.platform import windows_root

OUT=Path(__file__).resolve().parents[1]/'.cache/world-ownership-step'


def run(label,arm,*,hour=12,frames=30,diagnostic=False,captures=False,scenario='navigation',controls=None):
    navigation.OUT=OUT
    options={'C3X_SANDBOX_PASS_COUNTS':'1' if diagnostic else '0',
             'C3X_RENDERER_TRACE':'2' if diagnostic else '0',
             'C3X_RENDERER_PROFILE':'1' if diagnostic else '0'}
    directory=windows_root()/OUT.relative_to(navigation.ROOT).as_posix()/label
    options.update(C3X_RENDERER_TRACE_FILE=str(directory/'renderer.log') if diagnostic else '',
                   C3X_RENDERER_TRACE_BUFFERED='0',
                   C3X_SANDBOX_WORLD_CONTENT_DIR=str(directory) if diagnostic else '')
    options.update(controls or {})
    return navigation.run(label,arm,scenario,hour,frames,options,captures,'full_guest','client')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('label');p.add_argument('--arm',default='candidate');p.add_argument('--hour',type=int,default=12)
    p.add_argument('--frames',type=int,default=30);p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--captures',action='store_true');p.add_argument('--scenario',default='navigation',choices=['navigation','zoom','stationary'])
    p.add_argument('--control',action='append',default=[])
    a=p.parse_args();run(a.label,a.arm,hour=a.hour,frames=a.frames,diagnostic=a.diagnostic,captures=a.captures,
        scenario=a.scenario,controls=dict(x.split('=',1) for x in a.control))
