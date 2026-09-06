"""Bounded process wrapper around the approved renderer_dev native dispatcher."""
import contextlib
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'Renderer/tools'))
import renderer_dev

if __name__=='__main__':
    captured=io.StringIO()
    with contextlib.redirect_stdout(captured):
        result=renderer_dev.native_command_result(sys.argv[1],sys.argv[2])
    result['output_tail']=captured.getvalue()[-4000:]
    print(json.dumps(result))
