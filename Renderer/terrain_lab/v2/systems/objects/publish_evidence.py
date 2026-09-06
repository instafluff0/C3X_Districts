"""Publish small source/transform/corridor contracts from preserved Q7 layouts."""
import argparse, hashlib, json, math, sys
from pathlib import Path
import presentation as p
sys.path.insert(0,str(p.ROOT/p.V2/'shared'))
from scene_exchange import validate

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def source_parts(asset):
    manifest=p.read(p.PACK/'manifest.json')
    landmark_path=p.PACK/manifest['assets'][asset]['landmark']
    landmark=p.read(landmark_path)
    parts=[]
    for b in landmark['draw_bindings']:
        if 'worked' not in b['states']:continue
        meshpath=p.PACK/landmark['components']['geometry'][b['geometry']]
        matpath=p.PACK/landmark['components']['materials'][b['material']]
        mesh=p.read(meshpath);mat=p.read(matpath)
        channels={}
        for key,value in mat['channels'].items():
            c=dict(value)
            if 'texture' in c:
                c['texture']=str(p.PACK/c['texture']);c['sha256']=p.sha(Path(c['texture']))
            c['runtime_slot']={'base_color':124,'emissive':116}.get(key)
            c['binding_state']='bound' if c['runtime_slot'] is not None else 'unbound'
            c['interpretation']='normalized declaration; secondary normal/gloss engine meaning unconfirmed' if key in ('normal_1','gloss') else 'normalized declaration'
            channels[key]=c
        parts.append(dict(mesh=str(meshpath),mesh_sha256=p.sha(meshpath),material=str(matpath),material_sha256=p.sha(matpath),
            source_uv_sha256=digest([v['uv0'] for v in mesh['vertices']]),source_normal_sha256=digest([v['normal'] for v in mesh['vertices']]),
            tangent='absent_in_normalized_mesh_and_runtime',uv_operation='identity',normal_operation='yaw_rotation_only',
            alpha_mode=mat['alpha_mode'],channels=channels,vertices=len(mesh['vertices']),triangles=len(mesh['topology']['indices'])//3))
    return dict(landmark=str(landmark_path),landmark_sha256=p.sha(landmark_path),parts=parts,
        sockets=landmark.get('attachment_points',[]),socket_runtime='not_attached; no inferred Light/VFX'),landmark

def publish(name):
    folder=p.FIX/'generated'/name
    layout=p.read(folder/'layout.json');fixture=p.read(folder/'fixture.json')
    materials={};instances=[]
    for i,r in enumerate(layout['components']):
        if 'pool' not in r:continue
        asset=r['asset'];a=p.component(asset)
        if asset not in materials:materials[asset]=source_parts(asset)[0]
        if 'anchor' in layout:center=[layout['anchor']['screen_x'],layout['anchor']['screen_y']]
        else:center=next(c['center'] for c in layout['cells'] if all(c[k]==r[k] for k in ['pool','size','recipe']))
        origin=p.world_at([r['translation'][0],r['translation'][1],0],center,fixture['viewport'])
        mid=[(a['lo'][0]+a['hi'][0])/2,(a['lo'][1]+a['hi'][1])/2,a['lo'][2]]
        c,s=math.cos(r['rotation']),math.sin(r['rotation']);sc=r['scale']
        rotation=[[c,-s,0],[s,c,0],[0,0,1]]
        matrix=[[sc*rotation[row][col] for col in range(3)]+[origin[row]-sc*sum(rotation[row][col]*mid[col] for col in range(3))] for row in range(3)]+[[0,0,0,1]]
        instances.append(dict(id=f'city:{i}',asset=asset,pool=r['pool'],size=r['size'],slot=r['slot'],source_to_preprojection_world_4x4=matrix,
            matrix_convention='row-major; column homogeneous position',source_uniform_scale=sc,yaw_radians=r['rotation'],
            front_mapping='source +X provisionally treated as front; symmetric/ambiguous source fronts not art-verified',
            output_front='southeast' if abs(r['rotation'])<1e-8 else 'southwest',
            screen_bounds=r['bounds'],footprint=r['footprint']))
    sidecar=dict(schema='c3x.q7.source_instances.v1',classification='source_adaptation',fixture=str(folder/'fixture.json'),
        layout_sha256=p.sha(folder/'layout.json'),geometry_sha256=p.sha(folder/'geometry.bin'),coordinate_space='q7_pinned_preprojection_tile_world_v1',
        projection=dict(tile_pixels=[128,64],z_pixels_per_source_world_unit=80.9543),
        exceptions=['This pinned lab coordinate system is not the Q0 authoritative world-position attribute.',
            'Real fixtures add a clip-depth grounding bias; source matrix remains uniform, but Q0 world conversion is pending.',
            'Source UVs are unchanged, including authored high anisotropy/degenerate exceptions audited by Q1.',
            'Base/emissive display path only; no tangent/normal-map/AO/gloss/owner-color material closure.',
            'No source-equivalent lamp point lights or animated Light/VFX are inferred.'],
        assets=materials,instances=instances)
    out=p.AUD/'metadata'/name
    p.write(out/'source-instances.json',sidecar)
    if 'anchor' in layout:
        surface=p.read(folder/'surface.json');projection=surface['projection'];anchor=layout['anchor']
        world_draws=[]
        for record in layout['components']:
            a=p.component(record['asset']);sc=record['scale'];yaw=record['rotation']
            for part_index,(mesh,mat) in enumerate(a['parts']):
                vertices=[]
                for v in mesh['vertices']:
                    local=[v['position'][0]-(a['lo'][0]+a['hi'][0])/2,v['position'][1]-(a['lo'][1]+a['hi'][1])/2,v['position'][2]-a['lo'][2]]
                    local=[x*sc for x in p.rotate(local,yaw)]
                    local[0]+=record['translation'][0];local[1]+=record['translation'][1]
                    dx,dy=p.project(local);sx=anchor['screen_x']+dx;sy=anchor['screen_y']+dy
                    depth=.94-(anchor['screen_y']+(local[0]+local[1])*32)/fixture['viewport'][1]*.75-local[2]*.20732+layout['ground_depth_bias']
                    world=[anchor['column']+anchor['u']+local[0],anchor['row']+1-anchor['v']-local[1],(anchor['height']+local[2]*80.9543/projection['vertical_scale'])/112]
                    normal=p.rotate(v['normal'],yaw)
                    vertices.append([*world,*v['uv0'],*normal,sx,sy,depth])
                source=materials[record['asset']]['parts'][part_index]
                world_draws.append(dict(asset=record['asset'],slot=record['slot'],mesh=source['mesh'],mesh_sha256=source['mesh_sha256'],material=source['material'],
                    channels=source['channels'],alpha_mode=source['alpha_mode'],caster=True,receiver=True,current_state='worked_rigid',
                    vertices=[vertices[i] for i in mesh['topology']['indices']]))
        p.write(folder/'source-world-v1.json',dict(schema='c3x.q7.registered_source_world.v1',coordinate_space='Q0_lattice_xyz_height_div112',
            fixture=str(folder/'fixture.json'),terrain_sha256=p.sha(Path(fixture['terrain'])),surface_query_sha256=p.sha(folder/'surface.json'),
            fields=['world_x','world_y','world_z','u','v','authored_yaw_normal_x','authored_yaw_normal_y','authored_yaw_normal_z','screen_x','screen_y','depth'],
            projection=projection,draws=world_draws,grounding='five Q0 samples share one flat height; no depth-derived camera',
            calibration=dict(source_transform='uniform scale and yaw before Q0 projection conversion',world_y='row+1-v-local_y',
                world_z='(anchor_height+source_local_z*80.9543/vertical_scale)/112',
                normal='authored normal after yaw, same frozen shading convention; not an inverse-transpose Q0 metric normal',
                legacy_vertical_calibration_is_uniform_world_transform=False)))
        witness=next(r for r in p.read(p.AUD/'CITY_VEGETATION_WITNESS.json')['regions'] if r['region']==layout['region'])
        envelopes=[]
        for record in layout['components']:
            a=p.component(record['asset']);x0,y0,x1,y1=record['footprint']
            center=[(v-o)*64 for v,o in zip(witness['raw_city_anchor'],witness['origin_raw'])]
            polygon=[[center[0]+(x-y)*64,center[1]+(x+y)*64] for x,y in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]]
            envelopes.append(dict(id='city:0:building:'+str(record['slot']),kind='building',polygon=polygon,
                height_range=[layout['anchor']['height'],layout['anchor']['height']+(a['hi'][2]-a['lo'][2])*record['scale']*80.9543/projection['vertical_scale']],
                clearance=1.0,source_geometry_sha256=p.sha(folder/'geometry.bin'),source_asset=record['asset'],
                envelope_method='conservative full transformed source mesh XY bounds, including roof overhangs; not a cast shadow'))
        corridors=dict(schema='c3x.lab_v2.corridors.v1',coordinate_space='civ3_raw_delta_pixels_v1',terrain_sha256=p.sha(Path(fixture['terrain'])),
            source_sha256=layout['source_sha256'],region_id=layout['region'],origin_raw=layout['source_coordinates'],wrap_period=[6400,0],
            provider='Q7-presentation',revision=1,classification='source_adaptation',envelopes=envelopes,
            source_geometry=dict(path=str(folder/'geometry.bin'),sha256=p.sha(folder/'geometry.bin')),
            height_units='Q0 authoring-height pixels; local projected feature Z divided by Q0 vertical_scale',
            pending='Q4 placement adoption and composed re-render; Q5/Q3 exact real corridors')
        validate(corridors);p.write(out/'corridors.json',corridors)
    print(out/'source-instances.json')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('names',nargs='+')
    for name in ap.parse_args().names:publish(name)
