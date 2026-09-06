"""Compare Q1's GPU reconstruction + Q6 display against independent float reference."""
from pathlib import Path
import argparse,json,struct,sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'Renderer/terrain_lab/v2/systems/sampling'))
from quality import box_linear,srgb_encode

def main():
    ap=argparse.ArgumentParser();ap.add_argument('report',type=Path);ap.add_argument('--source-scene',action='store_true');a=ap.parse_args()
    p=a.report;r=json.loads(p.read_text());f=r.get('fixture_manifest',r.get('effective',{}).get('fixture'));rows=[]
    for e in r['outputs']:
        scale=r.get('effective',{}).get('settings',{}).get('render_scale',1)
        w,h=e.get('internal_size',[x*scale for x in f['viewport']])
        image=ROOT/e['image'];z=e['zoom'];ow,oh=[x//z for x in f['viewport']]
        header=(ROOT/e['packet']).read_bytes()[:44]
        magic,version=struct.unpack_from('<2I',header)
        if magic!=0x32514c43 or version not in (3,4):raise ValueError('linear packet required')
        rect=struct.unpack_from('<4I',header,24);exposure=struct.unpack_from('<f',header,40)[0]
        rgba=np.fromfile(str(image)+'.linear.rgba16f',dtype='<f2').astype(float).reshape(h,w,4)
        validity=np.fromfile(str(image)+'.validity.r8',dtype=np.uint8).reshape(h,w)/255.
        c,v=box_linear(rgba,validity,w//ow,rect)
        custom=e.get('post') not in ('off',None) if 'post' in e else isinstance(r['effective']['settings']['postprocess'],dict)
        if custom:c=c.astype(np.float16).astype(float)
        rgb=np.maximum(c[...,:3]/np.maximum(c[...,3:],1e-6)*exposure,0)
        mapped=rgb/(1+rgb.max(axis=2,keepdims=True));expected=np.floor(np.clip(srgb_encode(mapped),0,1)*255+.5)
        expected[(v==0)|(c[...,3]<=1e-6)]=0
        actual=np.asarray(Image.open(image).convert('RGB'),float);delta=np.abs(actual-expected)
        row={'image':e['image'],'max_rgb_error_bytes':float(delta.max()),'mae_bytes':float(delta.mean()),
             'hdr_input_max':float(rgba[...,:3].max()),'outside_validity_black':bool((actual[v==0]==0).all()),
             'finite':bool(np.isfinite(rgba).all()),'exposure':exposure,'valid_rect':rect}
        row['q1_contract2']=custom
        row['reference_matches']=bool(delta.max()<=1 and row['outside_validity_black'] and row['finite'])
        # Preserve Q0's built-in reducer as an explicit control. It currently
        # accumulates hidden invalid color; do not silently call that a pass.
        if (custom or not a.source_scene) and (not row['reference_matches'] or (not a.source_scene and row['hdr_input_max']<=1)):raise AssertionError(row)
        rows.append(row)
    (p.parent/'linear-reference-metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    selected=[x for x in rows if x['q1_contract2']]
    print('PASS',len(selected),'Q1 GPU/CPU reconstruction comparisons; max error',max(x['max_rgb_error_bytes'] for x in selected),'; differing Q0 controls',sum(not x['reference_matches'] for x in rows if not x['q1_contract2']))

if __name__=='__main__':main()
