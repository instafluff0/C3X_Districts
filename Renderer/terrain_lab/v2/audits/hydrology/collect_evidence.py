"""Pin Q3 source/contract provenance and inspected output metrics; never approve."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
HERE=Path(__file__).resolve().parent
V2=ROOT/'Renderer/terrain_lab/v2'
OUT=HERE/'out/Q1/Q3-hydrology'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(ROOT))
files=[V2/'shared/scene_hooks.h',V2/'contracts/scene_exchange_v1.md',V2/'systems/hydrology/scene_adapter.h',V2/'systems/hydrology/publish_corridors.py',V2/'contracts/packet_v1.h',V2/'contracts/interfaces_v1.h',V2/'contracts/platform_v1.json',V2/'contracts/scene_exchange_v1.md',V2/'shaders/lighting/response_v1.hlsl',V2/'shared/environment_runtime.cpp',V2/'shared/environment_runtime.h',V2/'systems/hydrology/field.h',V2/'systems/hydrology/packet.cpp',V2/'shaders/hydrology/static.hlsl',ROOT/'Renderer/handoffs/L21_complete_beauty_scene.json']
assets=[]
for name,kind,role in [('grassland_base_color','diagnostic_proxy','Q2 ground receiver using actual alternate-skin material'),('cliff_base_color','diagnostic_proxy','Q4 cliff receiver material on generated smooth slope; NOT source cliff geometry'),('beach_base_color','source_adaptation','source sand on generated Civ III shoreline topology'),('shallows_base_color','source_adaptation','source shallow bed on monotone generic bathymetry'),('shallows_height','source_reuse','unaltered authored height texture for material normals only')]:
 p=ROOT/f'Renderer/packs/Civ5EnvironmentSkin/provenance/terrain_textures/{name}.dds';assets.append({'asset':name,'pack':'Civ5EnvironmentSkin','path':rel(p),'sha256':sha(p),'classification':kind,'use':role,'uv':'world lattice, repeat scale rounded to integral world-wrap periods; no texture edits','geometry':'generated shore/bed topology; generated smooth hill receiver is diagnostic only'})
source={'schema':'c3x.q3.source_provenance.v1','assets':assets,'contracts':{rel(p):sha(p) for p in files},'packet_wire_version':2,'semantic_interface_version':1,'hydrology_interface_version':1,'source_evidence':['Renderer/terrain_lab/CANONICAL_REFERENCE_AUDIT.md','Renderer/terrain_lab/L9_5_SHORELINE_AUDIT.md','Renderer/terrain_lab/L13_RIVER_AUDIT.md'],'confirmed':'Normalized alternate-skin DDS roles and source ArtDef records documented in existing audits.','inferred':'B-spline shoreline, bounded erosion, bathymetry/absorption calibration and Bezier topology are native adaptations, not recovered source engine behavior.','deferred':'Wave/surf/foam animation, caustics and river flow; no time-driven hydrology redraws.','original_art_fallback_selected':False}
(HERE/'SOURCE_PROVENANCE.json').write_text(json.dumps(source,indent=2)+'\n')
reports=sorted(OUT.glob('*static-07/report.json'))+[OUT/'context-07/report.json',OUT/'classification-07/report.json']
summary=[];repeats=0
for p in reports:
 if not p.exists():continue
 r=json.loads(p.read_text());entries=[]
 for e in r['outputs']:
  image=ROOT/e['image'];cost=json.loads((ROOT/e['cost']).read_text());repeat=Path(str(image)+'.repeat1.bmp')
  # Retained original plus report records the earlier two-render equality check.
  passed=image.exists() and sha(image)==e['sha256'];rep=repeat.exists() and sha(repeat)==e['sha256']
  if r['tier']=='check' and passed:repeats+=1
  entries.append({'image':e['image'],'sha256':e['sha256'],'hour':e['hour'],'zoom':e['zoom'],'offset':e['offset'],'retained_original_matches':passed,'repeat_retained_and_matches':rep,'runner_repeat_pass':r['tier']=='check','gpu_ms_mean':cost.get('gpu_ms_mean'),'allocated_bytes':cost.get('allocated_bytes')})
 summary.append({'report':rel(p),'run_identity':r['render_identity'],'effective_contract':r['effective']['contract'],'outputs':entries})
context=json.loads((OUT/'context-07/report.json').read_text());returns=[]
for h in [12,18,0,6]:
 for z in [1,2]:
  es=[x for x in context['outputs'] if x['hour']==h and x['zoom']==z];returns.append({'hour':h,'zoom':z,'byte_identical_return_to_origin':es[0]['sha256']==es[-1]['sha256']})
metrics={'schema':'c3x.q3.evidence.v1','candidate':'static-07','full_current_revision_matrix_complete':False,'previous_complete_topology_matrix':'static-06 (PNG evidence preserved via ARCHIVED_OUTPUTS.json)','storage_incident':'Final broad rerun stopped by full filesystem; coordinator requested a heavy-matrix pause. Partial rows are not counted as passing.','current_revision_checked_variants':repeats,'reports':summary,'scroll_returns':returns,'perceptual_metrics':'Per-output inspection.json records mean displayed luminance, standard deviation and horizontal luminance differences. These are descriptive, not beauty thresholds.','visual_review':'Directly inspected iterations and property/isolation/contextual images; see REVIEW.md. Overall art approval remains false.'}
(HERE/'EVIDENCE.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps({'checked_current_variants':repeats,'return_to_origin_passes':sum(x['byte_identical_return_to_origin'] for x in returns),'source_assets':len(assets),'reports':len(summary)},indent=2))
