"""Compile local source footprint channels into generic radial pool profiles.

This is a documented presentation adaptation, not a recovered water simulation
or a claim that the source green channel directly defines a native water edge.
Requires Pillow; derived art remains local and is excluded from version control.
"""
import hashlib
import json
import math
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[5]
V2=ROOT/'Renderer/terrain_lab/v2'

def main():
    source=ROOT/'Renderer/packs/Civ5EnvironmentSkin/textures/water/river/source_decal_height.dds'
    green=Image.open(source).getchannel('G')
    count=64; profiles=[]; records=[]
    for index,rect in enumerate(((0,0,256,256),(256,0,512,256))):
        field=green.crop(rect)
        # The observed atlas has two connected footprint-like regions in its
        # upper half. Center within each region, preserving its actual outline.
        points=[(x+.5,y+.5) for y in range(256) for x in range(256) if field.getpixel((x,y))>=128]
        cx=sum(p[0] for p in points)/len(points);cy=sum(p[1] for p in points)/len(points)
        radii=[]
        for i in range(count):
            theta=2*math.pi*i/count;r=.5
            while r<125:
                x=int(cx+r*math.cos(theta));y=int(cy+r*math.sin(theta))
                if not (0<=x<256 and 0<=y<256) or field.getpixel((x,y))<115:break
                r+=.5
            radii.append(r)
        # Keep source-scale lobes, suppress narrow mask spikes at gameplay size.
        filtered=[sum(radii[(i+j)%count]*w for j,w in ((-2,1),(-1,2),(0,3),(1,2),(2,1)))/9 for i in range(count)]
        rms=math.sqrt(sum(r*r for r in filtered)/count)
        normalized=[r/rms for r in filtered]
        profiles.append(normalized)
        records.append({'index':index,'atlas_pixel_rect':list(rect),'center_pixels':[cx,cy],
                        'radius_range_normalized':[min(normalized),max(normalized)]})
    out=V2/'fixtures/beauty/source-river-pools-r1';out.mkdir(parents=True,exist_ok=True)
    data=('C3X_RIVER_POOL_PROFILE_V1,2,64\n'+'\n'.join(','.join(f'{x:.9f}' for x in profile) for profile in profiles)+'\n').encode()
    target=out/'pool_profiles.csv'
    if target.exists() and target.read_bytes()!=data:raise ValueError('Preserved profile differs')
    target.write_bytes(data)
    (out/'.gitignore').write_text('pool_profiles.csv\n')
    record={'schema':'c3x.source_pool_profile.v1','classification':'source_footprint_presentation_adaptation',
            'source':source.relative_to(ROOT).as_posix(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_channel':'green of BC5; observed footprint-like coverage, native meaning unproven',
            'threshold':115/255,'angular_samples':count,'angular_filter':[1,2,3,2,1],
            'normalization':'unit RMS radius preserves approximate area at the existing pool scale',
            'profiles':records,'output':target.relative_to(ROOT).as_posix(),'output_sha256':hashlib.sha256(data).hexdigest(),
            'redistribution':'local source-derived art; not included in integration preparation archive'}
    (out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
