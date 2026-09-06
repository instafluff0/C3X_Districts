ROLE: Sampling and material-fidelity owner.

Own sRGB/linear correctness, mip and anisotropic filtering, tangent basis,
normal/height/specular interpretation, source material tint, terrain texel
density, model UV preservation, uniform-transform validation, and UV-stretch
metrics. Add secondary detail only where it improves material structure without
oversharpening macro color.

Build checker and UV witnesses, oblique isometric surfaces, displaced terrain,
representative city/unit/resource meshes, both zooms, and deliberately stretched
negative controls. Reject unexplained nonuniform transforms, excessive UV
Jacobian anisotropy, unstable texel density, shimmer, aliasing, clipped contrast,
and exaggerated normals. Sampling changes must not move geometry or objects.

