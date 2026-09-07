"""Bounded terrain-constrained placement for normalized city bodies.

Select the most constrained remaining body first and backtrack across earlier
new placements. Existing growth-prefix instances are immutable. The search
changes translation and quarter-turn orientation, never source proportions.
"""
import math


def bounds(asset, rotation, scale):
    c,s=math.cos(rotation),math.sin(rotation)
    cx=(asset['lo'][0]+asset['hi'][0])/2
    cy=(asset['lo'][1]+asset['hi'][1])/2
    points=[((v['position'][0]-cx)*c-(v['position'][1]-cy)*s,
             (v['position'][0]-cx)*s+(v['position'][1]-cy)*c)
            for mesh,_ in asset['parts'] for v in mesh['vertices']]
    return [min(p[0] for p in points)*scale,min(p[1] for p in points)*scale,
            max(p[0] for p in points)*scale,max(p[1] for p in points)*scale]


def expanded(box, padding=.012):
    return [v+(-padding if i<2 else padding) for i,v in enumerate(box)]


def overlaps(a,b):
    return a[0]<b[2] and a[2]>b[0] and a[1]<b[3] and a[3]>b[1]


def gap(a,b):
    return math.hypot(max(0,a[0]-b[2],b[0]-a[2]),max(0,a[1]-b[3],b[1]-a[3]))


def connected(boxes,maximum_gap):
    reached={0}
    while True:
        added={j for j in range(len(boxes)) if j not in reached and
               any(gap(boxes[i],boxes[j])<=maximum_gap+1e-10 for i in reached)}
        if not added:return len(reached)==len(boxes)
        reached.update(added)


def shares_frontage(a,b,maximum_gap):
    """Require a visible shared side, rather than only a courtyard corner."""
    shared_x=min(a[2],b[2])-max(a[0],b[0]);shared_y=min(a[3],b[3])-max(a[1],b[1])
    x_gap=max(0,a[0]-b[2],b[0]-a[2]);y_gap=max(0,a[1]-b[3],b[1]-a[3])
    return (shared_x>=max(.04,.2*min(a[2]-a[0],b[2]-b[0])) and y_gap<=maximum_gap) or \
           (shared_y>=max(.04,.2*min(a[3]-a[1],b[3]-b[1])) and x_gap<=maximum_gap)


def solve(order, count, scale, extent, buildable, preserved=(), node_limit=20000, grid_step=.07, slot_extents=None, neighbor_gap=None, connected_prefixes=(), fixed_neighbors=(), candidate_cost=None, staged_connection=False):
    if not 0<count<=32 or not 0<node_limit<=200000:
        raise ValueError('invalid bounded city growth search')
    if grid_step not in (.035,.07):raise ValueError('unsupported city placement grid')
    if neighbor_gap is not None and (not math.isfinite(neighbor_gap) or not .02<=neighbor_gap<=.3):
        raise ValueError('invalid city neighbor gap')
    if any(not isinstance(n,int) or not 1<=n<=count for n in connected_prefixes):
        raise ValueError('invalid connected growth prefix')
    if any(len(b)!=4 or any(not math.isfinite(v) for v in b) or b[0]>=b[2] or b[1]>=b[3] for b in fixed_neighbors):
        raise ValueError('invalid fixed neighborhood footprint')
    limits=[extent]*count if slot_extents is None else list(slot_extents)
    if len(limits)!=count or any(not math.isfinite(v) or not 0<v<=extent for v in limits):
        raise ValueError('invalid per-growth-slot footprint limits')
    chosen={};domains={};stats={'nodes':0,'backtracks':0,'candidate_counts':{},'grid_step':grid_step,
                              'preserved_prefix':len(preserved),'node_limit':node_limit}
    if slot_extents is not None:stats['slot_extents']=limits
    if neighbor_gap is not None:stats.update(neighbor_gap=neighbor_gap,connected_prefixes=list(connected_prefixes),ranking='compact envelope, nearby houses, center, remaining freedom')
    if fixed_neighbors:stats['fixed_neighbors']=list(fixed_neighbors)
    if candidate_cost is not None:stats['candidate_cost']='authored projected landmark visibility preference'
    if staged_connection:
        if neighbor_gap is None:raise ValueError('staged connection requires a neighbor gap')
        stats['staged_connection']=True
    if len(preserved)>count:raise ValueError('growth stage cannot discard preserved bodies')
    for i,item in enumerate(preserved):
        asset=order[i%len(order)]
        if item['slot']!=i or item['asset']!=asset['id'] or abs(item['scale']-scale)>1e-10:
            raise ValueError('preserved growth prefix differs from requested source bodies/scale')
        local=bounds(asset,item['rotation'],scale);x,y=item['offset']
        if max(abs(a-b) for a,b in zip(local,item['local_bounds']))>1e-8:
            raise ValueError('preserved source body geometry changed')
        box=[local[j]+(x,y)[j%2] for j in range(4)]
        padded=expanded(box)
        if any(abs(v)>limits[i]+1e-10 for v in padded) or not buildable(padded):
            raise ValueError('preserved body violates requested terrain clearance')
        if any(overlaps(padded,c['box']) for c in chosen.values()):
            raise ValueError('preserved city bodies overlap')
        chosen[i]={'x':x,'y':y,'rotation':item['rotation'],'box':box,'padded':padded}
    steps=math.ceil(extent/grid_step)
    for i in range(len(preserved),count):
        asset=order[i%len(order)];candidates=[]
        for turn in (i%2,1-i%2):
            rotation=turn*math.pi/2;local=bounds(asset,rotation,scale)
            for ix in range(-steps,steps+1):
                for iy in range(-steps,steps+1):
                    x,y=ix*grid_step,iy*grid_step
                    box=[local[j]+(x,y)[j%2] for j in range(4)];padded=expanded(box)
                    if any(abs(v)>limits[i] for v in padded) or not buildable(padded):continue
                    if any(overlaps(padded,c['box']) for c in chosen.values()):continue
                    cost=candidate_cost(asset,rotation,x,y,scale) if candidate_cost else 0
                    if not math.isfinite(cost) or cost<0:raise ValueError('invalid placement cost')
                    candidates.append({'x':x,'y':y,'rotation':rotation,'box':box,'padded':padded,'cost':cost,
                                       'score':x*x+y*y+(x+y)*.025+(turn!=i%2)*.004})
        domains[i]=sorted(candidates,key=lambda c:(c['score'],c['rotation'],c['x'],c['y']))
        stats['candidate_counts'][str(i)]=len(candidates)
    def search(remaining):
        if neighbor_gap is not None:
            for n in set(connected_prefixes)|{count}:
                if all(i in chosen for i in range(n)):
                    houses=[chosen[i]['box'] for i in range(n)]
                    # A wide courtyard must not disguise isolated house groups
                    # in a smaller growth stage. Later houses cannot repair a
                    # prefix that is already fully assigned.
                    if not connected(houses,neighbor_gap) or not connected(list(fixed_neighbors)+houses,neighbor_gap):return None
                    if any(not any(shares_frontage(f,b,neighbor_gap) for b in houses) for f in fixed_neighbors):return None
        if not remaining:
            return dict(chosen)
        if stats['nodes']>=node_limit:return None
        stage=min((n for n in connected_prefixes if any(i<n for i in remaining)),default=count) if staged_connection else count
        slot=min((i for i in remaining if i<stage),key=lambda i:(len(remaining[i]),i))
        proposals=[]
        for rank,candidate in enumerate(remaining[slot]):
            if staged_connection:
                if chosen and not any(gap(candidate['box'],c['box'])<=neighbor_gap for c in chosen.values()):continue
                if not chosen and fixed_neighbors and not any(shares_frontage(candidate['box'],f,neighbor_gap) for f in fixed_neighbors):continue
            following={i:[c for c in domain if not overlaps(candidate['padded'],c['box'])]
                       for i,domain in remaining.items() if i!=slot}
            if any(not domain for domain in following.values()):continue
            # Leave room for the remaining bodies instead of filling the first
            # central opening. This is especially important for shoreline sites.
            freedom=sum(math.log(len(domain)) for domain in following.values())
            score=-freedom+.1*candidate['score']+candidate['cost']
            if neighbor_gap is not None:
                boxes=list(fixed_neighbors)+[c['box'] for c in chosen.values()]+[candidate['box']]
                width=max(b[2] for b in boxes)-min(b[0] for b in boxes)
                height=max(b[3] for b in boxes)-min(b[1] for b in boxes)
                nearest=min((gap(candidate['box'],b) for b in list(fixed_neighbors)+[c['box'] for c in chosen.values()]),default=0)
                score=(width*height+candidate['cost'],nearest,candidate['score'],-freedom)
            proposals.append((score,rank,candidate,following))
        for _,_,candidate,following in sorted(proposals,key=lambda p:p[:2]):
            if stats['nodes']>=node_limit:return None
            stats['nodes']+=1
            chosen[slot]=candidate
            result=search(following)
            if result is not None:return result
            del chosen[slot];stats['backtracks']+=1
        return None
    solution=search(domains)
    stats['status']='fit' if solution is not None else 'budget_exhausted' if stats['nodes']>=node_limit else 'no_grid_solution'
    if solution is None:return None,stats
    return [dict(asset=order[i%len(order)],slot=i,x=solution[i]['x'],y=solution[i]['y'],
                 rotation=solution[i]['rotation'],scale=scale) for i in range(count)],stats
