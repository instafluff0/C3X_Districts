#!/usr/bin/env python3
"""Generate an owner-local Q6 response adapter from the read-only frozen shader.
Keep every material/shadow body intact; replace only tone and display transfer.
"""
import hashlib,json,re
from pathlib import Path
V2=Path(__file__).resolve().parents[2]
def prepare():
    source=V2/'shaders/common/frozen_l21.hlsl';s=source.read_text()
    old='return l13a_layout > 0.5 ? color / (1.0 + color * 0.30) : color;'
    if s.count(old)!=1:raise ValueError('frozen tone-map extraction drift')
    s=s.replace(old,'return l13a_layout > 0.5 ? q6_display_linear(color, 1.0) : color;')
    # Balanced call parser: only the explicit gamma-2.2 DISPLAY conversions.
    edits=[]
    for m in re.finditer(r'\bpow\(',s):
        depth=1;i=m.end();comma=None
        while depth and i<len(s):
            if s[i]=='(':depth+=1
            elif s[i]==')':depth-=1
            elif s[i]==',' and depth==1:comma=i
            i+=1
        if comma is not None and re.fullmatch(r'\s*1\.0\s*/\s*2\.2\s*',s[comma+1:i-1]):
            edits.append((m.start(),i,'q6_srgb_encode('+s[m.end():comma]+')'))
    if len(edits)!=12:raise ValueError('frozen output extraction drift: '+str(len(edits)))
    for start,end,value in reversed(edits):s=s[:start]+value+s[end:]
    out=V2/'shaders/lighting/generated';out.mkdir(exist_ok=True)
    target=out/'response_adapter.hlsl';target.write_text('#include "../response_v1.hlsl"\n'+s)
    record={'source':str(source.relative_to(V2.parents[2])),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'display_conversions':len(edits),'scope':'shared tone map and exact display transfer only; legacy display-space alpha blending and inherited projected shadows remain provisional'}
    (V2/'audits/lighting/response_adapter_provenance.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':prepare()
