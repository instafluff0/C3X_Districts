"""Portable sampling invariants. Requires numpy; no game or backend imports."""
import numpy as np


def srgb_decode(x):
    x=np.asarray(x,dtype=float)
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def srgb_encode(x):
    x=np.asarray(x,dtype=float)
    return np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-.055)


def box_linear(rgba, validity, ratio, rect):
    rgba=np.asarray(rgba,dtype=float);validity=np.asarray(validity,dtype=float)
    h,w=rgba.shape[:2]
    if ratio not in (1,2,4) or h%ratio or w%ratio:raise ValueError('unsupported scale')
    if rgba.shape!=(h,w,4) or validity.shape!=(h,w):raise ValueError('shape mismatch')
    if not np.isfinite(rgba).all() or not np.isfinite(validity).all():raise ValueError('nonfinite input')
    if np.any((validity<0)|(validity>1)):raise ValueError('invalid coverage')
    rh,rw=h//ratio,w//ratio
    x0,y0,x1,y1=rect
    if not (0<=x0<=x1<=rw and 0<=y0<=y1<=rh):raise ValueError('invalid rectangle')
    c=np.where((validity>0)[...,None],rgba,0).reshape(rh,ratio,rw,ratio,4).mean((1,3))
    a=validity.reshape(rh,ratio,rw,ratio).mean((1,3))
    mask=np.zeros((rh,rw),bool);mask[y0:y1,x0:x1]=True
    return np.where(mask[...,None],c,0),np.where(mask,a,0)


def sharpen(rgb,strength=.3,bounded=True):
    x=np.asarray(rgb,dtype=float);p=np.pad(x,((1,1),(1,1),(0,0)),mode='edge')
    n,s,e,w=p[:-2,1:-1],p[2:,1:-1],p[1:-1,2:],p[1:-1,:-2]
    candidate=x+strength*(x-(n+s+e+w)/4)
    if bounded:
        stack=np.stack([x,n,s,e,w]);candidate=np.clip(candidate,stack.min(0),stack.max(0))
        candidate[[0,-1]]=x[[0,-1]];candidate[:,[0,-1]]=x[:,[0,-1]]
    return candidate


def uniform_transform(matrix,tolerance=1e-5):
    m=np.asarray(matrix,dtype=float)
    if m.shape!=(4,4) or not np.isfinite(m).all():raise ValueError('invalid transform')
    if not np.allclose(m[3],[0,0,0,1],atol=tolerance):raise ValueError('nonaffine transform')
    s=np.linalg.svd(m[:3,:3],compute_uv=False)
    if min(s)<=1e-8 or max(s)/min(s)>1+tolerance:raise ValueError('nonuniform transform')
    if np.linalg.det(m[:3,:3])<=0:raise ValueError('unexplained reflection')
    return float(s.mean())


def uv_metric(position,uv):
    """Geometric-to-UV singular values, independent of camera foreshortening."""
    p=np.asarray(position,float);uv=np.asarray(uv,float)
    if p.shape!=(3,3) or uv.shape!=(3,2) or not np.isfinite(p).all() or not np.isfinite(uv).all():
        raise ValueError('invalid triangle')
    edges=(p[1:]-p[0]).T
    q,r=np.linalg.qr(edges)
    if abs(np.linalg.det(r))<1e-10:raise ValueError('degenerate geometry')
    jacobian=(uv[1:]-uv[0]).T @ np.linalg.inv(r)
    s=np.linalg.svd(jacobian,compute_uv=False)
    if min(s)<1e-10:raise ValueError('degenerate UV')
    return {'anisotropy':float(max(s)/min(s)),'density':float(np.sqrt(np.prod(s)))}


def tangent_frame(position,uv):
    p=np.asarray(position,float);uv=np.asarray(uv,float)
    uv_metric(p,uv) # rejects degeneracies before inverse
    e=p[1:]-p[0];d=uv[1:]-uv[0]
    tb=np.linalg.solve(d,e);n=np.cross(e[0],e[1]);n/=np.linalg.norm(n)
    t=tb[0]-np.dot(tb[0],n)*n;t/=np.linalg.norm(t)
    sign=1 if np.dot(np.cross(n,t),tb[1])>=0 else -1
    return t,np.cross(n,t)*sign,n,sign
