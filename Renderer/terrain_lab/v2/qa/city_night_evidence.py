"""Verify the preserved r8 city-light controls and build matched review crops.

Requires Pillow and NumPy. These are supporting checks; visual approval stays
pending. Source images remain unchanged and both gameplay zooms are measured.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2';OUT=V2/'audits/beauty/out';R8=OUT/'city-scene-r8'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pixels(p):return np.asarray(Image.open(p).convert('RGB')).astype(np.int16)
def delta(a,b):
    d=np.abs(a-b)
    return {'changed_pixels':int(np.any(d,axis=2).sum()),'maximum_channel_delta':int(d.max()),'mean_channel_delta':float(d.mean())}

def main():
    result={'promotion':False,'human_visual_approval':None,'frames':[],'source_geometry':{}}
    old=ROOT/'Renderer/packs/CityStudyExpanded';new=ROOT/'Renderer/packs/CityStudyAuxiliaryUV'
    count=0;vertices=0
    for p in sorted((old/'meshes').rglob('*.json')):
        q=new/p.relative_to(old);a=json.loads(p.read_text());b=json.loads(q.read_text())
        for v in b['vertices']:
            for channel in ('uv1','uv2'):
                assert len(v[channel])==2 and np.isfinite(v[channel]).all()
                v.pop(channel)
            vertices+=1
        assert a==b, f'source geometry changed: {p.name}'
        count+=1
    result['source_geometry']={'identical_meshes_excluding_new_uv_channels':count,'vertices_with_preserved_auxiliary_coordinates':vertices}
    for hour in ('12','00'):
        for zoom in (1,2):
            frame=f'h{hour}-z{zoom}-pan00.png'
            path=R8/'modern-glow-portable'/frame;a=pixels(path)
            row={'frame':frame,'image_sha256':sha(path),'controls':{}}
            for control in ('modern-no-glow','modern-lights-off','modern-no-reflection'):
                row['controls'][control]=delta(a,pixels(R8/control/frame))
            if hour=='12':
                # Existing Metal replay witness permits isolated 1/255 rounding.
                assert row['controls']['modern-no-glow']['maximum_channel_delta']<=1
                assert row['controls']['modern-no-glow']['changed_pixels']<=2
            else:assert row['controls']['modern-no-glow']['changed_pixels']>100
            # Entirely inside the visible lake, separated from direct city/glow.
            scale=zoom;box=(858//scale,491//scale,885//scale,501//scale)
            x0,y0,x1,y1=box
            row['lake_box']=box
            row['lake_lights_off_delta']=delta(a[y0:y1,x0:x1],pixels(R8/'modern-lights-off'/frame)[y0:y1,x0:x1])
            if hour=='00':assert row['lake_lights_off_delta']['maximum_channel_delta']>10
            original=pixels(R8/'american-modern-s1-at7-5/combined'/frame)
            row['portable_shader_delta']=delta(a,original)
            assert row['portable_shader_delta']['maximum_channel_delta']<=1
            assert row['portable_shader_delta']['changed_pixels']<=1
            result['frames'].append(row)
    windows=R8/'windows-modern-portable/evidence.json'
    result['windows']=json.loads(windows.read_text()) if windows.exists() else {'status':'pending'}
    if 'results' in result['windows']:assert all(x['metrics']['pass'] for x in result['windows']['results'])
    review=R8/'review';review.mkdir(exist_ok=True)
    # Native-size contextual crops first; enlarged diagnostics in separate files.
    for name,box,before,after in (
        ('modern',(770,325,1000,545),OUT/'city-scene-r5/american-modern-s1-at7-5/combined',R8/'modern-glow-portable'),
        ('medieval',(340,300,560,480),OUT/'city-scene-r4/european-medieval-s1/combined',R8/'european-medieval-s1/combined')):
        w,h=box[2]-box[0],box[3]-box[1]
        sheet=Image.new('RGB',(w*2+16,h+34),'#202428');draw=ImageDraw.Draw(sheet)
        for i,(source,title) in enumerate(((before,'Before'),(after,'Corrected lights + glow'))):
            im=Image.open(source/'h00-z1-pan00.png').convert('RGB').crop(box)
            sheet.paste(im,(i*(w+16),34));draw.text((i*(w+16)+5,10),title,fill='white')
        sheet.save(review/(name+'-night-native.png'))
    target=V2/'audits/beauty/CITY_NIGHT_r8_EVIDENCE.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS city source preservation, glow controls, reflected lights and portable shader comparison')

if __name__=='__main__':main()
