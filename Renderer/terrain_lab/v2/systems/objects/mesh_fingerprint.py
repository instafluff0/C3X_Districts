"""Fingerprint geometry and coordinate sets independently of shading normals."""
import hashlib
import json


def material_digest(material):
    return hashlib.sha256(json.dumps(material,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def geometry_digest(mesh):
    data={'vertices':[{k:v[k] for k in ('position','uv0','uv1','uv2') if k in v} for v in mesh['vertices']],
          'topology':mesh['topology']}
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
