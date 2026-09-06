#!/usr/bin/env python3
"""Consume Q7's declared matrix/material identities without changing them."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'Renderer/terrain_lab/v2/systems/sampling'))
from quality import uniform_transform

def main():
    base=ROOT/'Renderer/terrain_lab/v2/audits/objects/metadata';rows=[];hashes={}
    def digest(path):
        if path not in hashes:hashes[path]=hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes[path]
    for name in ['ancient-earth-02','modern-id-02','registered-mixed-v2','registered-mixed-holdout-v2']:
        path=base/name/'source-instances.json';d=json.loads(path.read_text());unbound=set();channels=0
        for asset in d['assets'].values():
            for part in asset['parts']:
                for kind in ['mesh','material']:
                    if digest(ROOT/part[kind])!=part[kind+'_sha256']:raise ValueError('source identity drift')
                for semantic,c in part['channels'].items():
                    if digest(ROOT/c['texture'])!=c['sha256']:raise ValueError('channel identity drift')
                    channels+=1
                    if c['binding_state']=='unbound':unbound.add(semantic)
        for instance in d['instances']:
            scale=uniform_transform(instance['source_to_preprojection_world_4x4'])
            if abs(scale-instance['source_uniform_scale'])>1e-8:raise ValueError('declared scale mismatch')
        rows.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=digest(path),
            verified_matrices=len(d['instances']),verified_channel_identities=channels,
            unbound_semantics=sorted(unbound),coordinate_space=d['coordinate_space'],exceptions=d['exceptions']))
    out=ROOT/'Renderer/terrain_lab/v2/audits/sampling/q7-sidecar-validation.json'
    out.write_text(json.dumps(rows,indent=2)+'\n')
    print('PASS',sum(r['verified_matrices'] for r in rows),'declared uniform matrices and source channel identities; unbound channels/world exceptions remain acceptance blockers')

if __name__=='__main__':main()
