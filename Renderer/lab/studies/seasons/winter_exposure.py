"""Blender read-only exposure bake for snow on existing normalized tree meshes.

No models are created or modified. A BVH measures upper-surface visibility,
including existing opacity masks; the result is a small generic UV-space mask.
Overlapping source UVs store an averaged exposure, not a unique surface unwrap.
"""
import argparse
import json
from pathlib import Path
import struct
import sys
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def opacity_at(pixels, size, uv):
    width, height = size
    x = int((uv[0] % 1)*width) % width
    y = int((1-(uv[1] % 1))*height) % height
    return pixels[(y*width+x)*4]


def interpolate(vertices, face, point):
    a,b,c = [Vector(vertices[i]["position"]) for i in face]
    v,w,p = b-a,c-a,point-a
    vv,vw,ww,pv,pw = v.dot(v),v.dot(w),w.dot(w),p.dot(v),p.dot(w)
    determinant = vv*ww-vw*vw
    if abs(determinant)<1e-12:
        return vertices[face[0]]["uv0"]
    y,z = (ww*pv-vw*pw)/determinant,(vv*pw-vw*pv)/determinant
    return [sum(vertices[face[k]]["uv0"][axis]*weight for k,weight in enumerate((1-y-z,y,z))) for axis in range(2)]


def bake(row, size=128):
    source = json.loads(Path(row["mesh"]).read_text())
    vertices = source["vertices"]
    indices = source["topology"]["indices"]
    faces = [indices[i:i+3] for i in range(0,len(indices),3)]
    points = [Vector(v["position"]) for v in vertices]
    bvh = BVHTree.FromPolygons(points, faces, all_triangles=True)
    image = bpy.data.images.load(row["opacity"], check_existing=False) if row["opacity"] else None
    if image:
        image.colorspace_settings.name = "Non-Color"
        pixels, dimensions = list(image.pixels[:]), tuple(image.size)
    values, samples = np.zeros((size,size)),np.zeros((size,size))
    axis = Vector((0,0,1));rays=0
    for face in faces:
        tri = [vertices[i] for i in face]
        area = (points[face[1]]-points[face[0]]).cross(points[face[2]]-points[face[0]]).length
        if area<1e-10:
            continue
        # Barycentric sampling avoids huge UV bounds in wrapping foliage art.
        for i in range(9):
            for j in range(9-i):
                weights = (1-(i+j)/8,i/8,j/8)
                p = sum((points[face[k]]*weights[k] for k in range(3)),Vector())
                uv = [sum(tri[k]["uv0"][a]*weights[k] for k in range(3)) for a in range(2)]
                if image and opacity_at(pixels, dimensions, uv)<.5:
                    continue
                normal = sum((Vector(tri[k]["normal"])*weights[k] for k in range(3)),Vector())
                # Same inverse vertical basis as the current forest emitter.
                normal.z /= 150/(.82*64)
                normal.normalize()
                slope = max(0,min(1,(normal.z-.06)/.70))
                visibility=1
                if slope>.001:
                    start=p+axis*.003
                    for _ in range(6):
                        hit,_,index,_ = bvh.ray_cast(start,axis,2)
                        rays+=1
                        if hit is None:
                            break
                        hit_uv=interpolate(vertices,faces[index],hit)
                        if not image or opacity_at(pixels, dimensions, hit_uv)>.5:
                            visibility=0
                            break
                        start=hit+axis*.003
                x,y = int((uv[0]%1)*size)%size,int((uv[1]%1)*size)%size
                # Area weighting prevents tiny decorative faces dominating
                # an atlas region shared with a large canopy face.
                values[y,x]+=slope*(.12+.88*visibility)*area
                samples[y,x]+=area
    # Normalized small convolution fills sample gaps without inventing UV seams.
    numerator,denominator=np.zeros_like(values),np.zeros_like(samples)
    for dy in range(-3,4):
        for dx in range(-3,4):
            weight=np.exp(-(dx*dx+dy*dy)/7)
            numerator+=np.roll(values,(dy,dx),(0,1))*weight
            denominator+=np.roll(samples,(dy,dx),(0,1))*weight
    exposure=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>1e-12)
    # This is a supplemental shelter mask; shader slope and leaf masks remain
    # authoritative and retain detail where UVs are shared or sparsely sampled.
    result=np.round(np.clip(.25+exposure*.75,0,1)*255).astype(np.uint8)
    if image:
        bpy.data.images.remove(image)
    return result, {"triangles":len(faces),"rays":rays,"mean_exposure":float(exposure.mean()),"maximum_exposure":float(exposure.max())}


def write_dds(path, image):
    height,width=image.shape
    levels=[]
    for _ in range(5):
        levels.append(image.tobytes())
        image=np.round(image.reshape(image.shape[0]//2,2,image.shape[1]//2,2).mean((1,3))).astype(np.uint8)
    words=[124,0x2100f,height,width,width,0,5,*([0]*11),32,4,int.from_bytes(b"DX10","little"),0,0,0,0,0,0x401008,0,0,0,0]
    path.write_bytes(b"DDS "+struct.pack("<31I",*words)+struct.pack("<5I",61,3,0,1,0)+b"".join(levels))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for row in json.loads(args.inputs.read_text()):
        mask,metrics=bake(row)
        name=f"body-{row['body']:02d}.dds"
        write_dds(args.output/name,mask)
        results.append({"body":row["body"],"path":name,**metrics})
    (args.output/"bake.json").write_text(json.dumps({"blender_version":bpy.app.version_string,"masks":results},indent=2)+"\n")
    print(f"PASS Blender read-only exposure masks: {len(results)} bodies; version={bpy.app.version_string}")


if __name__=="__main__":
    main()
