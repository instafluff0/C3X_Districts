#!/usr/bin/env bash
# Audit and remove only byte-identical leftovers from the retired Renderer Lab v2
# tree. The source snapshot is pinned so every removed file remains recoverable:
#
#   git show daf8b8082cbf4fe6bfe440ed7c1575b62c05cea2:path/to/file > restored-file
#
# The default is a dry run limited to JSON, Markdown, and Python files. Use
# --all-legacy to include every byte-identical historical file under the retired
# tree, and --apply to perform deletion. Modified, new, symlinked, currently
# tracked, or non-historical files are always preserved.

set -euo pipefail

mode=dry-run
scope=metadata
verbose=0
for argument in "$@"; do
  case "$argument" in
    --apply) mode=apply ;;
    --all-legacy) scope=all ;;
    --verbose) verbose=1 ;;
    --help|-h)
      sed -n '1,17p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $argument" >&2
      exit 2
      ;;
  esac
done

script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(git -C "$script_directory" rev-parse --show-toplevel)
source_revision=daf8b8082cbf4fe6bfe440ed7c1575b62c05cea2

if [[ ! -f "$repository_root/Renderer/README.md" || ! -f "$repository_root/Renderer/renderer.py" ]]; then
  echo "Refusing to run: repository root does not contain the current Renderer." >&2
  exit 1
fi
if ! git -C "$repository_root" cat-file -e "$source_revision^{commit}"; then
  echo "Refusing to run: recovery revision $source_revision is unavailable." >&2
  exit 1
fi
if [[ $(git -C "$repository_root" rev-parse --show-object-format) != sha1 ]]; then
  echo "Refusing to run: this audited script expects a SHA-1 Git repository." >&2
  exit 1
fi

cd "$repository_root"
export C3X_CLEANUP_MODE=$mode
export C3X_CLEANUP_SCOPE=$scope
export C3X_CLEANUP_VERBOSE=$verbose
export C3X_CLEANUP_REVISION=$source_revision

python3 -B - <<'PY'
from __future__ import annotations

from collections import Counter
import hashlib
import os
from pathlib import Path
import subprocess
import sys

root = Path.cwd().resolve()
legacy = (root / "Renderer/terrain_lab/v2").resolve()
mode = os.environ["C3X_CLEANUP_MODE"]
scope = os.environ["C3X_CLEANUP_SCOPE"]
verbose = os.environ["C3X_CLEANUP_VERBOSE"] == "1"
revision = os.environ["C3X_CLEANUP_REVISION"]
metadata_extensions = {".json", ".md", ".py"}

if legacy == root or root not in legacy.parents:
    raise SystemExit("Refusing to run: legacy path escaped the repository")
if legacy.is_symlink():
    raise SystemExit("Refusing to run: legacy root is a symlink")

# Current asset/shader preparation receipts are the authoritative read closures.
# A legacy path here means the migration is not complete enough for deletion.
receipt_paths = list((root / "Renderer/lab/.cache/assets").glob("*.json"))
shader_receipt = root / "Renderer/lab/.cache/shader-preparation.json"
if shader_receipt.is_file():
    receipt_paths.append(shader_receipt)
legacy_receipts = [path for path in receipt_paths
                   if b"Renderer/terrain_lab/v2" in path.read_bytes()]
if legacy_receipts:
    print("Refusing to run: active preparation receipts still name legacy inputs:", file=sys.stderr)
    for path in legacy_receipts:
        print("  " + path.relative_to(root).as_posix(), file=sys.stderr)
    raise SystemExit(1)

# The city migration specifically required old DDS payloads until their current
# copies were verified. Refuse the broad mode if a consumed legacy DDS path has
# reappeared in the curated material input.
city_materials = root / "Renderer/packs/CityFidelitySources/current/palace-materials.json"
if scope == "all" and (not city_materials.is_file() or
                       b"Renderer/terrain_lab/v2" in b"\n".join(
                           line for line in city_materials.read_bytes().splitlines()
                           if b".dds" in line.lower())):
    raise SystemExit("Refusing --all-legacy: curated city materials still name a legacy DDS")

tracked = set(filter(None, subprocess.check_output(
    ["git", "ls-files", "-z", "--", "Renderer/terrain_lab/v2"]
).decode("utf-8", "surrogateescape").split("\0")))

tree = subprocess.check_output([
    "git", "ls-tree", "-rz", revision, "--", "Renderer/terrain_lab/v2"
])
historical = []
for record in tree.split(b"\0"):
    if not record:
        continue
    metadata, raw_name = record.split(b"\t", 1)
    file_mode, kind, object_id = metadata.decode("ascii").split()
    name = raw_name.decode("utf-8", "surrogateescape")
    if kind == "blob" and file_mode in {"100644", "100755"}:
        historical.append((name, object_id))

def blob_id(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1()
    digest.update(b"blob " + str(size).encode("ascii") + b"\0")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

eligible = []
preserved = Counter()
for name, expected in historical:
    path = root / name
    if scope == "metadata" and path.suffix.lower() not in metadata_extensions:
        preserved["outside requested extensions"] += 1
        continue
    if name in tracked:
        preserved["currently tracked"] += 1
        continue
    if path.is_symlink():
        preserved["symlink"] += 1
        continue
    if not path.is_file():
        preserved["already absent"] += 1
        continue
    resolved = path.resolve()
    if resolved == legacy or legacy not in resolved.parents:
        preserved["path escaped legacy root"] += 1
        continue
    if blob_id(path) != expected:
        preserved["locally modified"] += 1
        continue
    eligible.append((path, expected))

counts = Counter(path.suffix.lower() or "[no extension]" for path, _ in eligible)
bytes_total = sum(path.stat().st_size for path, _ in eligible)
print(f"Mode: {mode}; scope: {scope}")
print(f"Eligible byte-identical retired files: {len(eligible):,}")
print(f"Eligible logical size: {bytes_total / (1024 ** 2):,.2f} MiB")
print("Eligible types: " + ", ".join(f"{key}={value:,}" for key, value in sorted(counts.items())))
if preserved:
    print("Preserved: " + ", ".join(f"{key}={value:,}" for key, value in sorted(preserved.items())))
if verbose:
    for path, _ in eligible:
        print(("DELETE " if mode == "apply" else "WOULD DELETE ") + path.relative_to(root).as_posix())

if mode != "apply":
    print("Dry run only. Re-run with --apply to delete exactly this audited set.")
    raise SystemExit(0)

deleted = 0
deleted_bytes = 0
changed_after_audit = 0
for path, expected in eligible:
    # Close the audit/delete race: a file changed while the audit was running is
    # preserved even in apply mode.
    if not path.is_file() or path.is_symlink() or blob_id(path) != expected:
        changed_after_audit += 1
        continue
    size = path.stat().st_size
    path.unlink()
    deleted += 1
    deleted_bytes += size

# Remove only directories made empty by the file-by-file deletions. Never use a
# recursive directory delete; local-only evidence causes rmdir to fail harmlessly.
if legacy.is_dir():
    for directory, _, _ in os.walk(legacy, topdown=False):
        path = Path(directory)
        try:
            path.rmdir()
        except OSError:
            pass
print(f"Deleted {deleted:,} recoverable retired files "
      f"({deleted_bytes / (1024 ** 2):,.2f} MiB logical).")
if changed_after_audit:
    print(f"Preserved {changed_after_audit:,} files that changed during the audit.")
PY
