# Storage and evidence retention

## Policy

Keep the current source, licensed/unique asset inputs, fixed comparison references,
staged DLL and rollback. Preserve manifests, build identities, measurements and
representative visual evidence. Do not infer disposability from Git ignore rules.
Some historical snapshot assets share hard links with live packs; deleting a
snapshot path does not necessarily reclaim the bytes reported by `du`.

The preview-first tool `Renderer/native/maintain_storage.py` only selects old,
unshared, untracked generated files beneath `Renderer/native/build` and
`Renderer/lab/out`. It excludes current/latest candidates, promotion/rollback,
verified/reference output, source and pack trees. Default minimum age is two days;
age alone does not override those exclusions. It does not manage Git.

- Old BMPs become `.bmp.gz`, with full decompressed SHA-256 verification before
  removal of the original. Existing archives must match. Preserve originals when
  compression does not save space. This retains every selected image losslessly.
- Old `.obj`, `.exp` and `.lib` compiler intermediates are removed. Rebuild the
  corresponding isolated candidate to recover them. DLLs, executables, source
  snapshots, logs and JSON receipts remain.
- Recent candidates remain available. At the end of an optimization, designate
  the current candidate and its comparison/rollback explicitly; archive old image
  sequences and remove intermediates rather than another entire source/pack copy.
- Do not overwrite hard-linked source art or deduplicate editable files by adding
  hard links. Preserve unique historical inputs until their dependency/recovery
  contract has been established.

## Preview and apply

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

Keep the archive, restore only needed images, and retain the existing 8 GiB free
space reserve for evidence generation. Compression is maintenance, never part of
timed rendering or a reason to recreate every archived frame.

## Git maintenance

Git is a separate maintenance operation, with no history rewrite or ref deletion.
First check active Git processes and locks, run `git fsck --full --no-dangling`,
and save ref identities. Remove only individually verified stale temporary files
or unmatched indexes reported by Git, never a valid pack or an unknown lock.
Use conservative repacking with pruning and reflog expiration disabled; verify
integrity and unchanged refs afterward. Do not use `prune=now`, aggressive GC or
delete `.git` contents based on size alone. Git metadata may require filesystem
approval even though generated Renderer output is writable.

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
