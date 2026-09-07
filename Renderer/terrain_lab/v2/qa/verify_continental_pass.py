"""Record continental diagnostics and exact disabled-provider controls."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageChops

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'
OUT=V2/'audits/beauty/out'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline=OUT/'river-corridor-r3/inland'
    original=json.loads((baseline/'report.json').read_text())['outputs']
    rows=[]
    for name in ['continental-ground-r1','continental-ground-r2','continental-material-r1',
                 'continental-ground-control','source-blend-r1','source-blend-control']:
        folder=OUT/name/'inland'
        outputs=json.loads((folder/'report.json').read_text())['outputs']
        jobs=json.loads((folder/'batch.json').read_text())
        assert len(outputs)==4
        for index,(old,new) in enumerate(zip(original,outputs)):
            assert Path(old['image']).name==Path(new['image']).name
            packet=ROOT/new['packet'] if 'packet' in new else Path(jobs[index][0])
            a,b=ROOT/old['image'],ROOT/new['image']
            ia,ib=Image.open(a).convert('RGB'),Image.open(b).convert('RGB')
            assert ia.size==ib.size
            diff=ImageChops.difference(ia,ib)
            packets_equal=sha(ROOT/old['packet'])==sha(packet)
            same=a.read_bytes()==b.read_bytes()
            if name.endswith('control'):
                assert same and packets_equal
            if name=='source-blend-r1':assert packets_equal
            rows.append({'candidate':name,'hour':old['hour'],'zoom':old['zoom'],
                'baseline_bmp_sha256':sha(a),'candidate_bmp_sha256':sha(b),
                'baseline_packet_sha256':sha(ROOT/old['packet']),
                'candidate_packet_sha256':sha(packet),
                'same_packet':packets_equal,'same_bmp':same,
                'changed_pixels':sum(p!=(0,0,0) for p in diff.get_flattened_data()),
                'changed_bounds':diff.getbbox(),'size':list(ia.size)})
    source=json.loads((V2/'fixtures/beauty/source-continental-r1/provenance.json').read_text())
    paired=[]
    for entry in sorted({r['entry'] for r in source['records']}):
        records=[r for r in source['records'] if r['entry']==entry]
        assert len(records)==2
        a,b=records
        assert a['parameters']==b['parameters']
        for x,y in zip(a['channels'],b['channels']):
            assert (x['role'],x['lod'],x['payload_sha256'])==(y['role'],y['lod'],y['payload_sha256'])
        paired.append({'entry':entry,'base_overlay_parameters_and_payloads_identical':True})
    result={'classification':'diagnostic evidence; no overall visual acceptance or promotion',
        'promotion':False,'frames':rows,'continental_source_pairs':paired,
        'decisions':{'continental-ground-r1':'reject: changed river routing during preparation',
            'continental-ground-r2':'unpromoted: corrected preparation guard, subtle ground relief; river path parity needs explicit probe',
            'continental-material-r1':'reject: pale broad cloud-like fields do not improve reference surface grit',
            'source-blend-r1':'unpromoted: source-informed transition correction; single-region effect, not overall grit acceptance'}}
    (V2/'audits/beauty/CONTINENTAL_PASS_EVIDENCE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'verified_frames':len(rows),'byte_exact_controls':sum(r['same_bmp'] and r['candidate'].endswith('control') for r in rows)}))


if __name__=='__main__':main()
