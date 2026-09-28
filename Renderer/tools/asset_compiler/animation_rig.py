"""Optional generic joint hierarchy/local-pose trailer for baked skin palettes.

This is offline work. Runtime skinning remains on the GPU. Recover local affine
poses from the final, anchor-normalized world cache so attachment placement and
authored scale/shear survive exactly, including reflected and collapsed bones.
"""
import hashlib
import json
import struct

import numpy as np


def encode(skeleton, cache, skin_joints=None, inverse_bind=None, identity=""):
    bones = skeleton["bones"]
    count = len(bones)
    parents = [b["parent"] for b in bones]
    if any(p < -1 or p >= i for i, p in enumerate(parents)):
        raise ValueError("non-canonical animation hierarchy")
    joints = list(range(count)) if skin_joints is None else list(skin_joints)
    binds = [b["inverse_bind_matrix"] for b in bones] if inverse_bind is None else inverse_bind
    if len(joints) != len(binds) or any(j < 0 or j >= count for j in joints):
        raise ValueError("invalid skin-to-rig binding")
    world = np.asarray(cache.matrices, dtype=np.float64).reshape(cache.frame_count, count, 4, 4)
    local = world.copy()
    for i, parent in enumerate(parents):
        if parent >= 0:
            # Visibility tracks can deliberately collapse a whole subtree.
            # A pseudoinverse recovers its observable transform; reconstruction
            # below rejects any loss in the original, final world-space pose.
            local[:, i] = world[:, i] @ np.linalg.pinv(world[:, parent], rcond=1e-12)
            local[:, i, :3, 3] = 0
            local[:, i, 3, 3] = 1
    u, _, vh = np.linalg.svd(local[:, :, :3, :3])
    rotation = u @ vh
    negative = np.linalg.det(rotation) < 0
    u[negative, :, 2] *= -1
    rotation = u @ vh
    scale = local[:, :, :3, :3] @ rotation.swapaxes(-1, -2)
    # Convert row-vector rotations to unit quaternions. Largest-component
    # extraction is stable at 180 degrees and across mirrored/zero scales.
    quaternions = np.empty((cache.frame_count, count, 4))
    for index in np.ndindex(cache.frame_count, count):
        r = rotation[index].T
        values = [1+r[0,0]-r[1,1]-r[2,2], 1-r[0,0]+r[1,1]-r[2,2],
                  1-r[0,0]-r[1,1]+r[2,2], 1+np.trace(r)]
        k = int(np.argmax(values))
        q = np.zeros(4); q[k] = np.sqrt(max(0., values[k])) / 2
        divisor = 4*q[k]
        if k == 3:
            q[:3] = [(r[2,1]-r[1,2])/divisor, (r[0,2]-r[2,0])/divisor, (r[1,0]-r[0,1])/divisor]
        else:
            a, b = (k+1)%3, (k+2)%3
            q[a] = (r[a,k]+r[k,a])/divisor
            q[b] = (r[b,k]+r[k,b])/divisor
            q[3] = (r[b,a]-r[a,b])/divisor
        quaternions[index] = q / np.linalg.norm(q)
    reconstructed = local.copy()
    reconstructed[:, :, :3, :3] = scale @ rotation
    for i, parent in enumerate(parents):
        if parent >= 0:
            reconstructed[:, i] = reconstructed[:, i] @ reconstructed[:, parent]
    if not np.isfinite(local).all() or np.max(np.abs(reconstructed-world)) > 2e-5:
        raise ValueError("local animation reconstruction changed an authored world pose")
    binding = {"bones": [{k:b[k] for k in ("name", "parent", "inverse_bind_matrix")} for b in bones],
               "skin_joints": joints, "inverse_bind": binds, "identity": identity}
    fingerprint = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).digest()
    poses = np.concatenate((local[:, :, 3, :3], quaternions, scale.reshape(cache.frame_count, count, 9)), axis=2)
    result = bytearray(struct.pack("<8sII32s", b"C3XRIG1\0", count, 0, fingerprint))
    result += struct.pack(f"<{count}i", *parents)
    result += struct.pack(f"<{len(joints)}I", *joints)
    result += np.asarray(binds, dtype="<f4").tobytes()
    result += poses.astype("<f4").tobytes()
    return bytes(result)
