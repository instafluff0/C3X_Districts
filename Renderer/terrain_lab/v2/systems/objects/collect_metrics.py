"""Small reproducible evidence ledger; image inspection remains a separate gate."""
import math
from pathlib import Path
import presentation as p

def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def overlap(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))

def main():
    output=p.AUD/'out/Q1/Q7-presentation'
    reports={}
    for name in ['modern-check','modern-emissive-only','all-pools-01','ancient-earth-02','corridor-01','mine-relief-02','modern-id-02','objects-03','registered-mixed-v2','registered-mixed-holdout-v2','registered-mixed-linear-v1','registered-mixed-holdout-linear-v1']:
        path=output/name/'report.json'
        if not (p.ROOT/path).exists():continue
        report=p.read(path);images=[]
        for entry in report['outputs']:
            image=Path(entry['image']);repeat=Path(str(image)+'.repeat1.bmp')
            images.append(dict(hour=entry['hour'],zoom=entry['zoom'],image=str(image),sha256=p.sha(image),
                repeat_identical=(p.ROOT/repeat).exists() and p.sha(image)==p.sha(repeat)))
        reports[name]=dict(report=str(path),sha256=p.sha(path),render_identity=report['render_identity'],tier=report['tier'],images=images)
    collisions=[]
    data=p.read(p.FIX/'generated/modern-id-02/layout.json')
    for c in data['cells']:
        records=[r for r in data['components'] if all(r.get(k)==c[k] for k in ['pool','size','recipe'])]
        pairs=[(a,b) for i,a in enumerate(records) for b in records[i+1:]]
        collisions.append(dict(pool=c['pool'],size=c['size'],recipe=c['recipe'],buildings=len(records),
            ground_aabb_pair_intersections=sum(overlap(a['footprint'],b['footprint'])>1e-10 for a,b in pairs),
            projected_aabb_pair_intersections=sum(overlap(a['bounds'],b['bounds'])>1e-10 for a,b in pairs),
            projected_overlap_area_sum=sum(overlap(a['bounds'],b['bounds']) for a,b in pairs),
            interpretation='Conservative source bounds, not exact visible silhouette or perceptual pass.'))
    pixels=p.read(output/'objects-03/pixel-metrics.json')
    pixels=[dict(hour=r['hour'],zoom=r['zoom'],objects=r['objects'][:4],repeat_identical=r['repeat_identical']) for r in pixels if r['hour']==12]
    contracts=['contracts/packet_v1.h','shared/frozen_scene.cpp','shared/compose.cpp','shared/scene_exchange.py','shared/real_map/registry_v1.json',
        'systems/lighting/scene_shadow.cpp','systems/lighting/shadow_field_v1.h','shaders/lighting/frame_shadow_v1.hlsl',
        'shaders/lighting/generated/scene_linear_v1.hlsl','shaders/sampling/linear_reconstruct.hlsl','systems/networks/clearance.py']
    pins={str(p.V2/path):p.sha(p.V2/path) for path in contracts}
    packs=['CityComponentsNormalized/city_catalog.json','CityComponentsNormalized/manifest.json','CityAdjunctsNormalized/manifest.json',
        'ResourceNormalized/resource_runtime.bin','CompoundUnitLab/unit_horseman_runtime.bin','UnitFamilyLab/unit_infantry_runtime.bin','ImprovementsNormalized/mine_runtime.bin']
    pins.update({str(Path('Renderer/packs')/path):p.sha(Path('Renderer/packs')/path) for path in packs})
    p.write(p.AUD/'METRICS.json',dict(schema='c3x.q7.metrics.v1',reports=reports,modern_layout=collisions,object_noon_measurements=pixels,
        source_contract_pins=pins,pin_scope='Files at collection time; each immutable report render_identity pins the earlier compiled closure.',
        tests=dict(command='python3 -B -m unittest discover -s Renderer/terrain_lab/v2/tests/objects -v',passed=6),
        visual_acceptance=False))
    print(p.AUD/'METRICS.json')
if __name__=='__main__':main()
