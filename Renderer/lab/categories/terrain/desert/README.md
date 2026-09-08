# Desert

Production desert channels now use continuous source surface-detail shading plus sparse,
world-stable projected dune accents with exact source triangles and atlas UVs. The fidelity profile no longer applies the
old analytic dune field across every desert tile; the base source material keeps
its fine sand ripple while the regional patches add broader variation.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.
