# Resources

Current normalized resource bodies and southeast-facing animation; retain source family limitations.

Land studies show Iron, Cattle, Horses, Wheat, Gold and Dyes. Water studies show
Fish and Whales. Both have detail and surrounding-terrain cases. This is a small
go-to sample, not a claim that every source resource is replaced in C3X. In
particular the current production mapping does not replace Silks.

Known baseline witness issue: animated resources produce black rectangular
background patches in the current headless output. Keep this evidence visible;
do not repair the reference image or claim verified in-game equivalence until
the capture/compositing behavior has been investigated.

The current build is the approved revision 1. `standard.json` identifies the
shared implementation, dependencies, fixture recipe and focused regression tests.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. New visual changes need explicit
approval before replacing these references. See `Renderer/docs/visual_fidelity_playbook.md`.

Category commands prepare the static bundle and animation pack when their source
inputs or compilers change. Current inputs are `ResourceNormalized` and
`ResourceAnimatedLab`, with clip units in `Renderer/lab/shared/resources/clip_units.json`.
Builders use disposable output and preserve source bytes, animated root deltas,
marine school facing and the approved fish surface offset. Historical runtime
enablement/checkpoint flags are not approval state; category revisions are.
