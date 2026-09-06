#!/usr/bin/env python3
"""Strict Q6 rendered evidence gate. No missing-image skips or beauty claims."""
import hashlib,json,struct,sys
from pathlib import Path
from test_lighting import bmp,compare,V2,ROOT,OUT

def stats(path):
 w,h,p=bmp(path);rgb=[p[i:i+3][::-1] for i in range(0,len(p),4)]
 active=[x for x in rgb if max(x)>12]
 return dict(width=w,height=h,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mean_rgb=[sum(x[c] for x in active)/max(1,len(active)) for c in range(3)],luminance=sum(.2126*x[0]+.7152*x[1]+.0722*x[2] for x in active)/max(1,len(active)),clipped_channels=sum(c==255 for x in rgb for c in x))

def verify():
 report={'schema':'c3x.q6.evidence.v1','acceptance':'partial convergence evidence; not complete package approval','source_categories':{},'real_map':{},'invariants':{},'perceptual_metrics_are_not_beauty_approval':True}
 count=0
 for system in ['q7_city','trees','rocks','units','improvements']:
  variants={}
  for h in [12,18,0,6]:
   for z in [1,2]:
    name=f'h{h:02}-z{z}-pan00.bmp';base=OUT/(system+'-09')/name
    assert base.read_bytes()==Path(str(base)+'.repeat1.bmp').read_bytes()
    st=stats(base);st['shadows_off']=compare(base,OUT/(system+'_shadows_off-09')/name);st['contact_off']=compare(base,OUT/(system+'_contact_off-09')/name)
    assert st['shadows_off']['changed_channels']>0,(system,h,z,'shadow absent')
    assert st['clipped_channels']==0,(system,h,z,'clipped output')
    variants[f'{h}-{z}']=st;count+=1
  assert variants['12-1']['luminance']>variants['6-1']['luminance']>variants['0-1']['luminance']
  assert variants['12-1']['luminance']>variants['18-1']['luminance']>variants['0-1']['luminance']
  # Neutral ground makes dawn/dusk color ordering measurable independent of asset hue.
  dusk=variants['18-1']['mean_rgb'];dawn=variants['6-1']['mean_rgb']
  assert dusk[0]/dusk[2]>dawn[0]/dawn[2]
  for h in [12,18,0,6]:
   for z in [1,2]:
    folder=OUT/(system+'-scroll-09');zero=(folder/f'h{h:02}-z{z}-pan00.bmp').read_bytes();assert zero==(folder/f'h{h:02}-z{z}-pan03.bmp').read_bytes()
  report['source_categories'][system]=variants
 for h in [12,18,0,6]:
  for z in [1,2]:
   name=f'h{h:02}-z{z}-pan00.bmp';assert (OUT/'q7_city-09'/name).read_bytes()==(OUT/'q7_city_reverse-09'/name).read_bytes(),('opaque order',h,z)
   _,_,pixels=bmp(OUT/'q7_city_emissive_only-09'/name)
   peak=max(pixels[0::4]+pixels[1::4]+pixels[2::4]);assert (peak==0 if h==12 else peak>25),('emission activation',h,z,peak)
 for region in ['real-mixed','real-holdout']:
  f=json.loads((V2/f'fixtures/lighting/{region}/response.fixture.json').read_text());meta=f['real_map'];assert hashlib.sha256((ROOT/f['terrain']).read_bytes()).hexdigest()==meta['region']['terrain_sha256']
  variants={}
  for h in [12,18,0,6]:
   for z in [1,2]:
    folder=OUT/(region+'-04');a=folder/f'h{h:02}-z{z}-pan00.bmp';assert a.read_bytes()==(folder/f'h{h:02}-z{z}-pan03.bmp').read_bytes();variants[f'{h}-{z}']=stats(a)
  report['real_map'][region]={'provenance':meta,'variants':variants,'coverage_limit':'legacy terrain/object shadow composition retained; source-category world shadow field not yet composed into this real scene'}
 report['complete_source_linear']={}
 for region in ['mixed','holdout']:
  folder=OUT/('real-linear-'+region+'-11');variants={}
  for h in [12,18,0,6]:
   for z in [1,2]:
    a=folder/f'h{h:02}-z{z}-pan00.bmp'
    assert a.read_bytes()==Path(str(a)+'.repeat1.bmp').read_bytes()
    variants[f'{h}-{z}']=stats(a)
    assert variants[f'{h}-{z}']['clipped_channels']==0
  report['complete_source_linear'][region]={'variants':variants,'report':str((folder/'report.json').relative_to(V2)), 'visual_review':'Both zooms/four phases directly inspected; validity fringe fixed, inherited projected shadows still provisional.'}
 report['invariants']={'four_phase_two_zoom_source_variants':count,'complete_scene_linear_variants':16,'per_backend_repeat':True,'opaque_triangle_order':True,'return_to_origin':True,'all_source_outputs_unclipped':True,'phase_luminance_order':True,'sunset_warmer_than_sunrise':True}
 native=json.loads((OUT/'q7-city-nativefix-13/parity.json').read_text())
 assert native['pass'] and len(native['results'])==8
 for row in native['results']:
  assert row['deterministic'] and row['metrics']['pass']
  assert hashlib.sha256((ROOT/row['d3d11']).read_bytes()).hexdigest()==row['d3d11_sha256']
 for h in [12,18,0,6]:
  for z in [1,2]:
   name=f'h{h:02}-z{z}-pan00.bmp'
   assert (OUT/'q7-city-nativefix-13'/name).read_bytes()==(OUT/'q7_city-09'/name).read_bytes()
 report['native_city_parity']=native
 path=V2/'audits/lighting/EVIDENCE_METRICS.json';path.write_text(json.dumps(report,indent=2)+'\n');print('PASS strict Q6 evidence:',count,'source variants + two real-map regions; partial acceptance only')
if __name__=='__main__':verify()
