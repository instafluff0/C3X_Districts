# Legacy Mac packet runner

This is a migration dependency, not the current user-facing workflow.
Use `Renderer/renderer.py`; see `Renderer/lab/MIGRATION.md` for remaining work.
The old runner still contains campaign ownership/fixture plumbing, which must
not become authority for new work.

Reusable backend behavior:

- Headless Metal replay; no window or second presenter.
- HLSL compiled through glslang and SPIRV-Cross to Metal, with D3D11 using HLSL
  directly. Shader tools live in ignored `.local/`; `C3X_LAB_SHADER_TOOLS`
  can select an equivalent installation.
- Separate geometry, shader and texture caches; generic vertex/constant-buffer
  packets, explicit depth/blend state and shared frame bindings.
- The newer linear branch supports per-draw shader modules and world geometry
  semantics. Packet readers validate bounds and required resources.
- A restricted shell may expose no Metal device; report that failure rather
  than substituting an image or claiming a GPU pass.

Important migration limit: this backend is not yet equivalent to current C3X
production. Its older binding/texture support and isolated providers do not
implement the complete production shadow pages, continuous terrain transitions,
reflection, city lighting and reconstruction path. Old Mac pictures must not be
silently relabeled as the current production gold standard.

The runner's real-map importer and normalized pack access may remain useful
while preserving current candidate inputs. Its campaign validation and historical
promotion commands are retired as user workflows.
