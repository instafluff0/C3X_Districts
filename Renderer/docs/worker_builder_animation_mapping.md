# Worker and Builder Animation Mapping

## Decision

Civ III's authoritative worker `Job_ID` selects the custom work presentation.
The current FLC animation type is supporting evidence only. This is necessary,
not merely preferable: the native job-to-animation function aliases road and
railroad to `AT_ROAD`, fortress and barricade to `AT_FORTRESS`, irrigation and
pollution cleanup to `AT_IRRIGATE`, and airfield, radar tower, and outpost to
`AT_DEFAULT`.

The source-specific mapping lives in
`tools/asset_compiler/worker_builder_action_strategy.json`. The compiler checks
the installed Civ III INIs, Civ VI ArtDefs, Builder families, and animation
payloads, then emits an ignored `c3x.worker_action_mapping.v0` file containing
only generic C3X IDs. Production code must consume that normalized contract and
must not branch on Civ VI asset names.

Applicability is capability-driven, not name-driven. Any scenario, enslaved,
or modded unit with an authoritative worker job and a pack-authored worker
action set uses the same mapping while retaining its own body. The four
Builder-era bodies are defaults for the ordinary Worker profile, not a test for
whether a unit is allowed to perform work.

## Checked Mapping

| Civ III `Job_ID` | Native FLC slot | Generic action | Selected tool | Optional attached VFX |
| ---: | --- | --- | --- | --- |
| 0, mine | `MINE` | heavy ground work | pickaxe | stone chips/dust |
| 1, irrigate | `IRRIGATE` | light ground work | shovel | soil dust |
| 2, fortress | `FORTRESS` | light ground work | shovel | soil dust |
| 3, road | `ROAD` | light ground work | shovel | soil dust |
| 4, railroad | `ROAD` | light ground work | shovel | soil dust |
| 5, plant forest | `PLANT` | heavy ground work | pickaxe | soil dust |
| 6, clear forest | `FOREST` | cutting | axe | wood chips |
| 7, clear wetlands | `JUNGLE` | cutting | axe | leaf debris |
| 8, clean damage/pollution | `IRRIGATE` | heavy ground work | pickaxe | cleanup dust |
| 9, airfield | `DEFAULT` | light ground work | shovel | soil dust |
| 10, radar tower | `DEFAULT` | light ground work | shovel | soil dust |
| 11, outpost | `DEFAULT` | light ground work | shovel | soil dust |
| 12, barricade | `FORTRESS` | light ground work | shovel | soil dust |

The first three motions are exact Civ VI Builder action families:

- `ACTIVITY_BUILD -> ACTION_1 -> Builder_BuildAction01_Shovel` becomes
  `animation/unit/worker/work_ground`.
- `ACTIVITY_DIG -> ACTION_2 -> Builder_BuildAction02_2H` becomes
  `animation/unit/worker/work_heavy`.
- `ACTIVITY_CUT -> ACTION_3 -> Builder_BuildAction03_Axe` becomes
  `animation/unit/worker/work_cut`.

Those activity and clip bindings are confirmed ArtDef/package data. Shovel for
ACTION_1 and Axe for ACTION_3 are also named by the source clips. Pickaxe for
the generically named two-hand ACTION_2 is an inferred initial presentation
choice and remains subject to L20's rendered tool/socket proof; it is recorded
as inference in the strategy rather than presented as a Firaxis fact.

Civ VI explicitly overrides Farm BUILD to `ACTIVITY_BUILD` and Mine BUILD to
`ACTIVITY_DIG`. Its operation defaults also support BUILD for improvements and
routes, DIG for contamination cleanup and planting forest, and CUT for removing
features. The table keeps those source-backed choices even when a different
tool might sound semantically tempting. L20 may approve a pack-authored
alternative after rendered comparison; the runtime contract does not change.

Four one-hand repair clips are converted and retained as optional authored
alternatives. Civ VI maps repair to `ACTION_A`, but the ordinary Civ III worker
job inventory has no distinct repair job. They are therefore not mislabeled as
one of the thirteen persistent jobs.

## Capture and Generic BUILD

Civ III workers have a real `CAPTURE` slot and Civ VI has four dedicated
Builder captured clips. Select one deterministically from the stable body
variation (`variation % 4`), so redraw and cache rebuilds cannot make the same
worker jump between performances.

The vanilla Worker INIs leave `BUILD` empty. Decompiled call sites use
`AT_BUILD` for one-shot transitions such as founding a city and other generic
build/despawn presentation, not as the persistent worker-job identity. Map it
to one clamped ground-work action only when Civ III actually requests BUILD;
never use it to infer the job or its completion.

## Tools Are an Exclusive Attachment Group

The Builder member's single `Tool` attachment lists Axe, Pickaxe, Shovel, and
Sledgehammer bins. Those alternatives must not become four simultaneously
visible components. The generic pack contract declares an
`exclusive_attachment_group` at the `Tool` socket. Each work action selects
exactly one logical tool component; non-work actions hide the group.

This is a generic parent/child/socket feature. Any arbitrary unit or mod pack
can declare an action-selected equipment group without a Builder-specific
runtime branch. The existing source resolver's ordinary additive-bin behavior
remains valid for attachments that genuinely compose multiple parts; Builder
tool import must opt into the exclusive group.

## Timing, Effects, and Ownership

Persistent motions loop only while the authoritative job remains active.
Where a worker has a meaningful native work FLC, the renderer samples the
custom clip from normalized native progress. For airfield, radar tower, and
outpost, whose native presentation is DEFAULT, Integration must maintain a
stable observed-job presentation clock. That clock is visual only and cannot
change worker strength, turn completion, tile state, movement, or orders.

The body clip owns worker and tool pose. Dust, stone or wood chips, leaves,
sparks, and cleanup particles belong to the attached generic effect pipeline.
Their release points use authored normalized markers rather than inferred
rendered frames. Civ III remains authoritative for job completion and audio;
native worker sounds must not be duplicated.

Era selects one of four generic body profiles corresponding to the located
Ancient, Medieval, Industrial, and Modern Builder families. The source graph
has four member recipes per era. Industrial and Modern member import still
needs a generic missing-bin/candidate-resolution fix before model-bound pose
caches can be baked; that is an explicit L20 visual/body intake gap, not a
reason to discard their animation clips.

## Converted Evidence and Remaining Gate

The Windows converter successfully normalized eleven installed payloads:
three primary work motions, four optional repair motions, and four capture
motions. All have one `Root` group, 41 tracks, 81--141 frames, and positive
2.67--4.67 second durations. Generated `.c3anim` files, hashes, source reports,
and runtime maps are ignored local evidence and cannot be redistributed.

This preparation does not enable unit rendering. L20 still owns Builder body
and tool model conversion, skeleton binding, socket calibration, eight-facing
and two-zoom rendered sheets, owner-color coverage, effect restraint, and final
visual approval. I20 then owns native `Job_ID`/state capture, stable event
clocks, suppression/fallback, and runtime invalidation.

## Reproduction

```sh
python3 Renderer/tools/asset_compiler/worker_builder_action_compiler.py
# Windows 11 VM/shared checkout:
Renderer\tools\asset_compiler\CONVERT_WORKER_BUILDER_ANIMATIONS.bat
python3 Renderer/tools/asset_compiler/worker_builder_clip_validator.py
python3 -m unittest Renderer.tools.asset_compiler.test_worker_builder_action_compiler
```
