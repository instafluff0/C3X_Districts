"""Small original C3X missile bodies where source games supply only VFX.

Geometry, materials and pose curves are authored here; no licensed source art is
embedded. Native Civ III still owns launch effects, action completion and removal.
"""
import math
import struct
from Renderer.tools.asset_compiler.normalized_pose_cache import PoseCache
from Renderer.tools.asset_compiler.build_resource_animation_runtime import encode

IDENTITY=(1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.)


def white_dds():
    header=bytearray(148)
    header[:4]=b'DDS '
    for offset,value in {4:124,8:0x81007,12:4,16:4,20:8,28:1,76:32,80:4,108:0x1000,128:72,132:3,140:1}.items():
        struct.pack_into('<I',header,offset,value)
    header[84:88]=b'DX10'
    return bytes(header)+struct.pack('<HHI',0xffff,0xffff,0)


def geometry(radius,height,fins):
    vertices=[];indices=[]
    def triangle(a,b,c):
        ab=[b[i]-a[i] for i in range(3)];ac=[c[i]-a[i] for i in range(3)]
        n=[ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0]]
        length=math.sqrt(sum(v*v for v in n))
        if length<1e-12:return
        start=len(vertices)
        for p in (a,b,c):vertices.append({'position':p,'normal':[v/length for v in n], 'uv0':[.5,.5],'joints':[0,0,0,0],'weights':[1.,0.,0.,0.]})
        indices.extend((start,start+1,start+2))
    for k in range(20):
        t=k*math.tau/20;u=(k+1)*math.tau/20
        a=[radius*math.cos(t),radius*math.sin(t),0.]
        b=[radius*math.cos(u),radius*math.sin(u),0.]
        c=[b[0],b[1],height*.78];d=[a[0],a[1],height*.78]
        triangle(a,b,c);triangle(a,c,d);triangle(d,c,[0.,0.,height]);triangle(b,a,[0.,0.,0.])
    if fins:
        for k in range(4):
            t=k*math.tau/4;c,s=math.cos(t),math.sin(t)
            points=[[radius*c,radius*s,height*.25],[radius*2.6*c,radius*2.6*s,0.],[radius*c,radius*s,0.]]
            triangle(*points);triangle(*reversed(points))
    return {'vertices':vertices,'topology':{'primitive':'triangles','indices':indices}}


def compile_units(publish):
    texture=publish(white_dds(),'textures','dds')
    skeleton={'bones':[{'name':'body','parent':-1,'inverse_bind_matrix':IDENTITY,
        'local':{'position':[0.,0.,0.],'orientation':[0.,0.,0.,1.],'scale_shear':[1.,0.,0.,0.,1.,0.,0.,0.,1.]}}]}
    result={}
    for slug,key,radius,height,fins in [('cruise_missile','PRTO_Cruise_Missile',.075,1.0,True),
                                      ('tactical_nuke','PRTO_Nuke',.12,1.,True),('icbm','PRTO_ICBM',.11,1.4,False)]:
        mesh=geometry(radius,height,fins)
        unit={'civ3_ids':[key],'actions':{},'member_scale':1.,'source_pack':'C3XGenericMissiles',
              'source_recipe':'original_geometry_and_pose_curves'}
        for action in ('idle','move','attack','death','fortify','fidget','victory'):
            frames=[]
            for frame in range(16):
                phase=frame/15;angle=0.;z=0.;scale=1.
                if action=='move':angle=.035*math.sin(phase*math.tau)
                if action=='fidget':angle=.09*math.sin(phase*math.tau)
                if action=='fortify':angle=.06*(1-phase)
                if action=='attack':z=.25*phase;scale=max(.04,1-phase**4)
                if action=='death':z=-height*phase;scale=1-.8*phase
                c,s=math.cos(angle)*scale,math.sin(angle)*scale
                frames.append((c,0.,s,0.,0.,scale,0.,0.,-s,0.,c,0.,0.,0.,z,1.))
            cache=PoseCache(.5,30.,16,('body',),tuple(v for f in frames for v in f))
            payload=encode(mesh,skeleton,cache)
            unit['actions'][action]={'duration':.5,'frames':16,'presentation':'original_c3x_motion',
                'loop':action in ('idle','move'),'parts':[{'asset':'unit/'+slug+'/body','mesh':publish(payload,'clips','bin'),
                'bytes':len(payload),'material':{'alpha_mode':'opaque','source_tint':None,'tint_rgb':[.8,.83,.86],
                'owner_color':{'mode':'none','strength':0.,'mask_source':'constant_one'},
                'channels':{'base_color':{'texture':texture}}},'source_mesh':'original','source_material':'original'}]}
        result['unit/'+slug]=unit
    return result
