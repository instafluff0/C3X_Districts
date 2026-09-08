"""Run preserved Lab city gates, redirecting only derived review outputs."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3];LAB=ROOT/'Renderer/terrain_lab/v2';OUT=ROOT/'Renderer/verification/city_fidelity'
sys.path.insert(0,str(LAB/'qa'))
def main():
    from PIL import Image
    original=Image.Image.save
    Image.Image.save=lambda *args,**kwargs:None
    records=[]
    try:
        for name,report in [('city_capital_material_evidence','CITY_CAPITAL_MATERIAL_EVIDENCE.json'),('city_palace_ground_evidence','CITY_PALACE_GROUND_ALIGNMENT_EVIDENCE.json'),('city_central_capital_evidence','CITY_CENTRAL_CAPITAL_EVIDENCE.json'),('city_light_buffer_evidence','CITY_LIGHT_BUFFER_EVIDENCE.json')]:
            path=LAB/'qa'/f'{name}.py';source=path.read_text()
            source=source.replace("V2/'audits/beauty/"+report+"'","PICKUP_OUTPUT/'"+report+"'")
            scope={'__file__':str(path),'__name__':'pickup_gate','PICKUP_OUTPUT':OUT}
            exec(compile(source,str(path),'exec'),scope);scope['main']()
            records.append({'validator':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':'pass','report':report})
        (OUT/'pickup-validation.json').write_text(json.dumps(records,indent=2)+'\n')
    finally:Image.Image.save=original
if __name__=='__main__':main()
