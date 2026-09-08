"""Execute selected Lab evidence gates without rewriting preserved witnesses."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3]
LAB=ROOT/'Renderer/terrain_lab/v2'
OUT=ROOT/'Renderer/verification/environment_refresh'
sys.path.insert(0,str(LAB/'qa'))
def main():
    OUT.mkdir(exist_ok=True)
    records=[]
    for name in ['verify_water_reflections','city_light_buffer_evidence','city_central_capital_evidence']:
        path=LAB/'qa'/f'{name}.py';source=path.read_text()
        # All original assertions and independent executable contracts run.
        # Only derived report/image writes are redirected or suppressed.
        targets={'verify_water_reflections':'WATER_REFLECTION_r5_EVIDENCE.json',
                 'city_light_buffer_evidence':'CITY_LIGHT_BUFFER_EVIDENCE.json',
                 'city_central_capital_evidence':'CITY_CENTRAL_CAPITAL_EVIDENCE.json'}
        report=targets[name]
        source=source.replace("V2/'audits/beauty/"+report+"'", "PICKUP_OUTPUT/'"+report+"'")
        if name=='city_central_capital_evidence':
            source=source.replace("canvas.save(BASE/(name+'-native.png'))",'None # preserve Lab review image')
        scope={'__file__':str(path),'__name__':'pickup_gate','PICKUP_OUTPUT':OUT}
        exec(compile(source,str(path),'exec'),scope)
        if name=='verify_water_reflections':scope['sheet']=lambda *args,**kwargs:None
        scope['main']()
        records.append({'validator':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':'pass','report':report})
    (OUT/'pickup-validation.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
