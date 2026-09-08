# Resources

Current normalized resource bodies and southeast-facing animation; retain source family limitations.

Land studies show Iron, Cattle, Horses, Wheat, Gold and Dyes. Water studies show
Fish and Whales. Both have detail and surrounding-terrain cases. This is a small
go-to sample, not a claim that every source resource is replaced in C3X. In
particular the current production mapping does not replace Silks.

The fixed references preserve the earlier black-backdrop defect around animated
resources. Current code fixes guarded scene-linear backdrop accumulation.
Comparison will show that intended difference; it does not block Integration.
Replace the fixed reference only if the user wants the corrected appearance to
become the new visual comparison point.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.

Category commands prepare the static bundle and animation pack when their source
inputs or compilers change. Current inputs are `ResourceNormalized` and
`ResourceAnimatedLab`, with clip units in `Renderer/lab/shared/resources/clip_units.json`.
Builders use disposable output and preserve source bytes, animated root deltas,
marine school facing and the accepted fish surface offset. Historical runtime
enablement/checkpoint flags and release numbers are not workflow state.
