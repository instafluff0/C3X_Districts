# Storage and evidence retention

## Policy

Keep the current source, licensed/unique asset inputs, fixed comparison references,
staged DLL and rollback. Preserve manifests, build identities, measurements and
representative visual evidence. Do not infer disposability from Git ignore rules.
Some historical snapshot assets share hard links with live packs; deleting a
snapshot path does not necessarily reclaim the bytes reported by `du`.

The default is to **delete unnecessary generated output**, not compress it or
keep another archive. The user explicitly requested this after repeated disk
exhaustion. Complete each experiment with a retention pass:

- Keep the current candidate and matched control complete until the comparison
  closes, plus any reproducer for an unresolved failure. Identify these paths
  before cleaning up.
- After closure, retain measurements, source/build identities, necessary replay
  inputs and representative/event frames. Delete the remaining generated frame
  sequence and obsolete compiler intermediates. A completed experiment does not
  need every frame from every superseded candidate.
- Delete closed guest capture copies once their complete host copies have been
  verified by relative path, size and hash. Do this after each run rather than
  waiting for the guest disk to fill. Keep the original input saves and required
  replay journals.
- Reuse immutable pack inputs. Avoid another source/pack tree per measurement.
  Do not overwrite hard-linked source art or add hard links to editable files.
- Bound recording duration and check free space on both host and guest before
  capture. Maintain the 8 GiB reserve; an explicitly bounded recovery run may
  use its documented lower threshold, followed immediately by duplicate cleanup.
- Review exact paths before deletion, recheck file identities, exclude symlinks
  and tracked inputs, and record deletions in a small local receipt. Age and Git
  ignore status alone do not establish that a file is disposable.

Unique historical inputs remain protected until their consumers and replacement
coverage have been checked. An old summary is not a lossless replacement for a
required replay journal. Conversely, obsolete generated output should not be
kept indefinitely merely because a summary mentions it.

## Legacy maintenance tool

The preview-first tool `Renderer/native/maintain_storage.py` only selects old,
unshared, untracked generated files beneath `Renderer/native/build` and
`Renderer/lab/out`. It excludes current/latest candidates, promotion/rollback,
verified/reference output, source and pack trees. Default minimum age is two days;
age alone does not override those exclusions. It does not manage Git.

- Its legacy BMP action creates `.bmp.gz`, with full decompressed SHA-256 verification before
  removal of the original. Existing archives must match. Preserve originals when
  compression does not save space. This retains every selected image losslessly.
- Old `.obj`, `.exp` and `.lib` compiler intermediates are removed. Rebuild the
  corresponding isolated candidate to recover them. DLLs, executables, source
  snapshots, logs and JSON receipts remain.
- Recent candidates remain available. Its exclusions do not replace the explicit
  current candidate/control ownership and completed-output retirement above.
- Do not overwrite hard-linked source art or deduplicate editable files by adding
  hard links. Preserve unique historical inputs until their dependency/recovery
  contract has been established.

Do not use the legacy compression action for the user's delete-only cleanup.
The commands below describe that tool for existing archives, not the default
retention workflow.

### Preview and apply

From the repository root:

```sh
python3 Renderer/native/maintain_storage.py --plan Renderer/lab/out/maintenance/storage-plan.json
python3 Renderer/native/maintain_storage.py --plan Renderer/lab/out/maintenance/storage-plan.json --apply
```

Choose a fresh manifest filename for each cleanup; existing plans and receipts
are never overwritten. Inspect the manifest before applying. Stop relevant builds/benchmarks first.
Application rechecks file identity and age, refuses symlink paths, and journals
each action in a sibling `.receipt.json`. Local receipts are ignored; do not
commit machine-specific metadata. A changed file aborts application rather than
being removed. Interrupted runs preserve verified archives and journal entries;
inspect the receipt and generate a preview with a fresh filename before continuing.

The navigation analyzer reads `.bmp.gz` when the original BMP is absent. Other
tools or direct image viewers may need restoration first:

```sh
gzip -dk path/to/generated-image.bmp.gz
```

Restore only needed inputs from an existing archive. After checking its consumers,
delete an archive of obsolete generated output; preserve required inputs first.
Neither restoration nor cleanup belongs in timed rendering measurements.

## Explicit generated-output cleanup

### October 4 recovery

A delete-only pass removed 25,800 unnecessary generated files (16.56 GB):
superseded window sequences, verbose renderer traces, compiler intermediates,
and completed color/depth comparison outputs. Host free space increased from
4.02 GB to 20.54 GB. No replacement archive was created.

All tracked and uncommitted source, runtime packs, unique art, saves, input
journals, staged/rollback binaries, the current failed research capture, and its
light/busy controls remain. Changed-source and staged-binary hashes were
verified unchanged. The VM was stopped and was not restarted; guest captures
were not cleaned in this pass. Per-file deletion receipts are under
`Renderer/.cache/disk-cleanup-20261004/`.

Older retained result/timing receipts describe their original runs; most no
longer have complete raw window sequences or verbose traces. First/last frames
and existing contact sheets remain. Do not report those directories as complete
recordings. New experiments must finish with the retention pass above instead
of accumulating another full copy of each superseded capture.

### October 2 recovery

A reviewed delete-only cleanup removed 24.29 GiB of host allocations: repeated
historical window frames, obsolete compiler intermediates and diagnostic
recordings, an old raster/log archive, stale Git temporary packs, and two
unreachable single-blob asset ZIPs. Required baseline inputs, recovery logs and
113 changed historical asset files were preserved separately (0.446 GiB total),
for a targeted net allocation reduction of 23.85 GiB. The host had 25.32 GiB
free at completion; concurrent VM/build activity can change that figure.

Both asset ZIPs were fully streamed and each payload checked with SHA-256
against a live pack file or a preserved historical file before disposal. Fresh
reachability checks covered five worktrees. Full Git integrity passed after
deletion and refs were unchanged. The user explicitly approved retrying archive
disposal after automatic review initially rejected it.

Current candidate/control work, live packs, source, saves, rollback binaries and
the complete unresolved R4 failure recording remain. Closed R2/R6 diagnostic
prefixes were retired while their reports and change facts remain; their
directories explicitly say raw replay is no longer available. Per-file plans,
preservation checks and receipts are local under
`Renderer/.cache/disk-cleanup-20261002/`; `final-cleanup-summary.json` summarizes
the exact completed actions. No new compressed archive was created.

### Earlier generated-output retirement

At the user's request, a reviewed cleanup removed 5,079 generated files from
`native/build`: redundant replay frame sequences, repeated diagnostic BMPs and
compiler intermediates. Its logical size fell from 52.22 to 19.47 GiB; measured
filesystem free space increased by 32.77 GiB. Original recording journals,
live captures, source assets, active short-capture evidence, tools and staged/
rollback DLLs remain. SHA-256 checks verified 1,542 retained recording and binary
files. Per-test control images, sequence samples and the latest examples of named
diagnostics remain; historical output directories no longer contain every BMP.
Replay those preserved inputs only when a particular missing image is needed.
The ignored plan and deletion receipt are under
`lab/out/maintenance/build-cleanup-20260921.*.json`.

## Git maintenance

Git is a separate maintenance operation, with no history rewrite or ref deletion.
First check active Git processes and locks, run `git fsck --full --no-dangling`,
and save ref identities. Routine maintenance removes only individually verified
stale temporary files or unmatched indexes reported by Git. Never remove a pack
because of its size or remove an unknown lock.
Use conservative repacking with pruning and reflog expiration disabled; verify
integrity and unchanged refs afterward. Do not use `prune=now`, aggressive GC or
delete `.git` contents based on size alone. Git metadata may require filesystem
approval even though generated Renderer output is writable.

Explicit disposal of an unwanted archive stored as an unreachable Git blob is a
separate reviewed action. Identify its complete contents and retained live inputs;
check every branch, stash, index and reflog, including linked worktrees. A pack
may be removed only when every contained object has been individually established
as disposable. Record exact object/file identities, recheck before deletion, and
verify Git integrity and unchanged refs afterward. This does not authorize pruning
unrelated unreachable commits or recovery history.

## Documentation retention

[The documentation index](README.md) separates current entry points, durable
contracts/findings and historical evidence. Keep the active scorecard short;
archive completed experiment narratives under `docs/history/` and retain old
entry links where needed. Historical next-step instructions are inactive. Do not
remove source-art findings, integration contracts or deferred feature contracts
as a documentation cleanup shortcut.

## Initial maintenance result

The first pass archived 1,375 generated BMPs losslessly and removed 682 old
compiler intermediates. Git integrity passed before and after removal of 11
stale temporary/unmatched files and repacking with object pruning and reflog
expiration disabled. All refs remained unchanged; Git reported zero garbage.
Observed filesystem free space increased by approximately 5 GiB overall.

The current renderer, staged/rollback binaries, accepted references, unique source
inputs and cliff-audit asset tree were preserved. The audit's 26,995 hard-linked
files already share storage with live packs, so their apparent copied size is not
fully reclaimable. Per-file receipts and restoration hashes are local under
`Renderer/lab/out/maintenance/`; no renderer benchmark or game launch was run.

Final October 4 verification kept bounded exploration and busy-zoom evidence,
then removed verified duplicate guest captures and 84.7 MB of rebuildable
compiler intermediates. Source saves, current binaries, regression witnesses,
licensed inputs and reference images remain. Receipts are under
`Renderer/.cache/navigation-quality-20261004/`; the final two captures include
input hashes, full timing logs, reviewed contact sheets and analysis reports.
