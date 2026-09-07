"""Bake node-local compound unit motion into the generic runtime skin format."""
from __future__ import annotations
import json
from pathlib import Path
from Renderer.tools.asset_compiler import normalized_animation as animation, normalized_pose_cache as poses, normalized_skin as skin
from Renderer.tools.asset_compiler.build_resource_animation_runtime import encode
from Renderer.tools.asset_compiler.unit_family_action_validator import _best_group, SOCKET_PROFILE
from Renderer.preview.render_unit_turntable import _multiply

IDENTITY = (1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.)


def compile_unit(pack: Path, manifest: dict, recipe: dict, publish) -> dict:
    from Renderer.tools.asset_compiler.build_unit_animation_runtime import fortify_transition, SOURCE_TINTS
    def read(relative):
        return json.loads((pack / relative).read_text())
    nodes = recipe['nodes']
    skeletons = {key: skin.load_skeleton(pack / node['skeleton']) for key,node in nodes.items()}
    root = recipe['root_node']
    root_index = next(i for i,b in enumerate(skeletons[root]['bones']) if b['parent'] == -1)
    root_rest = skin.world_matrices(skeletons[root])[root_index]
    unit = {'actions':{}, 'civ3_ids':recipe['civ3_ids'], 'member_scale':1.,
            'source_pack':pack.name, 'source_recipe':manifest['units'][recipe['unit_id']]['recipe']}
    for action,binding in recipe['actions'].items():
        clips = {key:animation.load_clip(pack / manifest['animations'][clip]['clip']) for key,clip in binding['node_clips'].items()}
        count = max(c.frame_count for c in clips.values())
        duration = max(c.duration for c in clips.values())
        transition = action == 'fortify' and count <= 2
        if transition:
            count, duration = 16, .5
        # Every node uses the same normalized native action phase. Local source
        # durations may differ, but never create a second simulation clock.
        def cache(skeleton, node):
            clip=clips[node]
            if transition:
                idle=animation.load_clip(pack / manifest['animations'][recipe['actions']['idle']['node_clips'][node]]['clip'])
                return fortify_transition(skeleton,idle,clip)
            group=_best_group(clip,{b['name'] for b in skeleton['bones']})[0]
            values=[]
            for frame in range(count):
                pose=skin.sample_pose(skeleton,clip,group,clip.duration*frame/(count-1),False)
                worlds=list(skin.world_matrices(skeleton,pose))
                rest=skin.world_matrices(skeleton)
                for i,bone in enumerate(skeleton['bones']):
                    if bone['name'] in nodes[node].get('disabled_socket_bones',[]):worlds[i]=rest[i]
                values.extend(v for matrix in worlds for v in matrix)
            return poses.PoseCache(duration,(count-1)/duration,count,tuple(b['name'] for b in skeleton['bones']),tuple(values))
        drivers={key:cache(skeletons[key],key) for key in nodes}
        worlds=[]
        def scale(value):
            return (value,0.,0.,0.,0.,value,0.,0.,0.,0.,value,0.,0.,0.,0.,1.)
        for frame in range(count):
            composed={root:scale(nodes[root]['variation_scale'])}
            pending=list(recipe['joints'])
            while pending:
                previous=len(pending)
                for joint in list(pending):
                    parent,child=joint['parent'],joint['child']
                    if parent not in composed:continue
                    index=drivers[parent].bone_names.index(joint['parent_bone'])
                    begin=(frame*len(drivers[parent].bone_names)+index)*16
                    socket=drivers[parent].matrices[begin:begin+16]
                    if action in nodes[child].get('parent_space_actions',[]):
                        # These source clips already contain the animated
                        # parent socket transform, including its translation.
                        socket=IDENTITY
                    transform=_multiply(scale(nodes[child]['variation_scale']),joint['local_transform'])
                    composed[child]=_multiply(_multiply(transform,socket),composed[parent])
                    pending.remove(joint)
                if len(pending)==previous:raise ValueError('cyclic compound animation graph')
            worlds.append(composed)
        parts=[]
        for node_id,node in nodes.items():
            draws=[]
            for record in node['components']:
                component=read(manifest['assets'][record['asset']]['component'])
                for draw in component['draw_bindings']:
                    draws.append((record,{**component,'binding_mode':draw.get('binding_mode',component['binding_mode'])},draw))
            for record,component,draw in draws:
                skeleton=skin.load_skeleton(pack / component['skeleton'])
                disabled=set(node.get('disabled_socket_bones',[]))
                if component['binding_mode']=='vertex_skin' and disabled:
                    mesh=read(component['meshes'][draw['mesh']])
                    names=[b['name'] for b in skeleton['bones']]
                    if any(weight and names[joint] in disabled for v in mesh['vertices']
                           for joint,weight in zip(v['joints'],v['weights'])):
                        raise ValueError('disabled inventory socket influences visible skin')
                local=cache(skeleton,node_id) if component['binding_mode']=='vertex_skin' else drivers[node_id]
                attachment=record.get('attachment_bone',SOCKET_PROFILE.get(component['attachment_point'],{}).get('bone'))
                if component['binding_mode']=='rigid_attachment':
                    if attachment in disabled:raise ValueError('visible attachment uses disabled socket')
                    if attachment not in local.bone_names:
                        raise ValueError(f"unresolved compound socket {recipe['unit_id']}/{record['asset']}: {attachment}")
                    bone=local.bone_names.index(attachment)
                    skeleton={'bones':[{'name':attachment,'parent':-1,'inverse_bind_matrix':IDENTITY,
                        'local':{'position':[0.,0.,0.],'orientation':[0.,0.,0.,1.],
                                 'scale_shear':[1.,0.,0.,0.,1.,0.,0.,0.,1.]}}]}
                composed=[]
                for frame in range(count):
                    offset=(frame*len(drivers[root].bone_names)+root_index)*16
                    travel=[(drivers[root].matrices[offset+12+a]-root_rest[12+a])*nodes[root]['variation_scale'] for a in (0,1)]
                    indices=[bone] if component['binding_mode']=='rigid_attachment' else range(len(local.bone_names))
                    for i in indices:
                        start=(frame*len(local.bone_names)+i)*16
                        matrix=list(_multiply(local.matrices[start:start+16],worlds[frame][node_id]))
                        matrix[12]-=travel[0];matrix[13]-=travel[1]
                        composed.extend(matrix)
                bound=poses.PoseCache(duration,(count-1)/duration,count,tuple(b['name'] for b in skeleton['bones']),tuple(composed))
                mesh_path=component['meshes'][draw['mesh']]
                material_path=component['materials'][draw['material']]
                mesh=read(mesh_path)
                if component['binding_mode']=='rigid_attachment':
                    for v in mesh['vertices']:
                        v['position']=[p*component['model_scale'] for p in v['position']]
                        v['joints']=[0,0,0,0];v['weights']=[1.,0.,0.,0.]
                material=read(material_path)
                channels={key:{**value,'texture':publish((pack/value['texture']).read_bytes(),'textures','dds')} for key,value in material['channels'].items()}
                payload=encode(mesh,skeleton,bound)
                parts.append({'asset':record['asset'],'mesh':publish(payload,'clips','bin'),'bytes':len(payload),
                    'material':{'alpha_mode':material.get('alpha_mode','opaque'),'channels':channels,
                        'source_tint':component.get('tint'),'tint_rgb':component.get('tint_rgb'),
                        'owner_color':component.get('owner_color')},'source_mesh':mesh_path,'source_material':material_path})
        unit['actions'][action]={'duration':duration,'frames':count,'presentation':'composed_node_clips',
            'allow_exit_clip':action=='death','loop':binding['loop'],'parts':parts}
    return unit
