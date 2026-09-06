"""Read-only real-map hydrology coverage and proposed additional region witnesses."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'Renderer/terrain_lab/v2/app'))
from real_map import load_registry,region_tiles,metrics,lookup
reg,data=load_registry()
def details(origin):
 tiles=region_tiles(data,origin,[4,4],2);visible=[t for t in tiles if t['visible']];rock=mouth=0
 for t in visible:
  ns=[lookup(data,t['sourceX']+dx,t['sourceY']+dy) for dx,dy in [(1,1),(1,-1),(-1,1),(-1,-1)]]
  coast=any(n and (n['base']>=11)!=(t['base']>=11) for n in ns)
  rock+=t['real']==5 and coast;mouth+=bool(t['riverMask']) and coast
 return dict(metrics(tiles),rocky_hill_tiles=rock,river_coast_adjacent_tiles=mouth)
coverage={r['id']:details(r['origin']) for r in reg['regions']}
candidates=[]
for t in data['tiles']:
 if not t['riverMask']:continue
 origin=[t['sourceX'],t['sourceY']]
 try:m=details(origin)
 except ValueError:continue
 if m['river_coast_adjacent_tiles'] and m['land_water_edges']:candidates.append((m['river_coast_adjacent_tiles']*10+m['river_tiles'],origin,m))
candidates.sort(reverse=True)
result={'source_sha256':reg['source']['sha256'],'regions':coverage,'requested_mouth_region':candidates[0] if candidates else None,'note':'River/coast adjacency is a candidate mouth witness, not inferred flow direction. Shared registry remains unchanged.'}
print(json.dumps(result,indent=2))
