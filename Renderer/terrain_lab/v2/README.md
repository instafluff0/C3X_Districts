# Legacy Lab implementation — migration only

This tree is no longer workflow authority. Do not start Q tracks, generate
campaign prompts or treat historical pass/handoff files as current approval.
Use `Renderer/renderer.py` and `Renderer/lab/README.md`.

The current production build is the baseline. Current city source algorithms
and the production shader source closure have moved to `Renderer/lab/shared`.
The remaining tree temporarily preserves the older Mac backend/providers and
unmigrated candidate inputs. See `Renderer/lab/MIGRATION.md` before deleting
dependencies; ignored geometry/assets are not backed up by Git.
