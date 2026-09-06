"""Large source-art demonstration; explicit synthetic layout, not captured BIQ.

Exact source height samples are placed as overlapping undeformed contributions.
Physical mountain calibration/material assembly remain diagnostic.
"""
import argparse
import math
import numpy as np
from build_fixture import Fixture, ROOT, SKIN, height_dds, rel, digest, sample, smooth

class LargeMap(Fixture):
    def __init__(self,revision=1,source_extent='full'):
        super().__init__('large-source-map',revision)
        self.n=11;self.viewport=(1920,1280)
        self.projection=(960,160,76,38,107.48)
        self.bounds=(-.5,-.5,12,12)
        self.masses=[];self.mass_sources=[]
        # Synthetic topology only. Main visible mountain shapes remain the five
        # source height variants; overlap uses translation, yaw and uniform scale.
        chains=[[(3+i*.65,4+i*.40) for i in range(8)],
                [(9+i*.52,7+i*.59) for i in range(7)],
                [(5+i*.65,11-i*.16) for i in range(5)],
                [(11.8,3.2),(12.4,3.7),(13,4.1)]]
        for i,(x,y) in enumerate(p for chain in chains for p in chain):
            path=SKIN/f'textures/relief/mountains/standard/variant_0{i%5+1}/height_lod0.dds'
            self.sources[rel(path)]=digest(path)
            self.masses.append((x*.67,y*.67,1.273,(i%4)*math.pi/2,height_dds(path),.75471735))
            self.mass_sources.append(path)
        if source_extent=='radius':
            # Diagnostic of the unproven source coordinate interpretation:
            # MountainWidth may map UV[-1,+1], not UV[-.5,+.5]. Preserve every
            # HM sample and peak height; this is not an accepted body calibration.
            self.masses=[(x,y,s*2,a,h,aspect/2) for x,y,s,a,h,aspect in self.masses]
        self.source_extent=source_extent

    def fields(self,x,y):
        # Selected installed hill source, one continuous map-space sample.
        hill=sample(self.hillfield,(x*.085+.11)%1,(y*.085+.19)%1)*self.hill_amplitude
        # A wide lowland corridor is layout, not a replacement relief shape.
        lowland=smooth(1.3,2.4,np.abs(y-.45*x-1.6))
        height=.025+hill*(.18+.82*lowland)
        return height,np.full_like(height,10),np.zeros_like(height),np.zeros_like(height)

    def dressing(self):
        rng=np.random.default_rng(8103)
        for _ in range(720):
            x,y=rng.uniform(.4,10.6,2)
            if (x-5)**2+(y-3)**2>8 and (x-9)**2+(y-9)**2>8 and (x-3)**2+(y-9)**2>5:continue
            ground=float(self.ground(np.array(x),np.array(y)))
            if float(self.top(np.array(x),np.array(y)))>ground+.10:continue
            name='forest_leafy_v1_01' if rng.random()<.5 else 'forest_leafy_v4_02'
            self.source_mesh(ROOT/'Renderer/packs/Civ5EnvironmentVegetation',name,x,y,rng.uniform(.75,.95),rng.uniform(0,6.28))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',type=int,default=1)
    p.add_argument('--source-extent',choices=['full','radius'],default='full')
    args=p.parse_args();scene=LargeMap(args.revision,args.source_extent);scene.write()
    import json
    from build_fixture import OWN
    path=OWN/f'large-source-map-r{args.revision}/provenance.json'
    provenance=json.loads(path.read_text())
    provenance['source_coordinate_hypothesis']=dict(MountainWidth=args.source_extent,confirmed=False,classification='diagnostic: physical source extent interpretation unresolved',height_pixels_unchanged=True)
    path.write_text(json.dumps(provenance,indent=2)+'\n')
