#!/usr/bin/env bash
# Remove obsolete generated navigation images; keep recent runs and result records.
set -euo pipefail

case "${1:-}" in
  ""|--preview) apply=false ;;
  --apply) apply=true ;;
  *) printf 'Usage: bash %s [--preview|--apply]\n' "$0" >&2; exit 2 ;;
esac
if (( $# > 1 )); then printf 'Too many arguments.\n' >&2; exit 2; fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root="$script_dir/build/navigation-evidence-20260909"
if [[ ! -d "$root" || -L "$script_dir/build" || -L "$root" ]]; then
  printf 'Missing or linked evidence directory: %s\n' "$root" >&2
  exit 1
fi
resolved_root=$(cd -- "$root" && pwd -P)
if [[ "$resolved_root" != "$root" ]]; then
  printf 'Evidence directory resolves outside the expected location.\n' >&2
  exit 1
fi

shopt -s nullglob
images=()
for run in "$root"/*; do
  [[ -d "$run" && ! -L "$run" ]] || continue
  case "${run##*/}" in
    region-input-ring4-100|region-input-ring4-clean-100|tight-natural-bounds-100|receiver-shadows-independent-100) continue ;;
  esac
  for file in "$run"/*.bmp; do
    [[ -f "$file" && ! -L "$file" ]] || continue
    case "${file##*/}" in
      zoom.bmp|zoom.bmp.resident0.bmp|zoom.bmp.resident47.bmp|zoom.bmp.resident99.bmp) continue ;;
    esac
    [[ "${file%/*}" == "$run" && "${run%/*}" == "$root" ]] || exit 1
    images+=("$file")
  done
done

printf 'Selected %s obsolete generated BMP files.\n' "${#images[@]}"
printf 'Sources, assets, logs, reports and recent reference runs are preserved.\n'
if [[ "$apply" == false ]]; then
  printf 'Preview only. Run with --apply to delete these images.\n'
  exit 0
fi
if (( ${#images[@]} == 0 )); then exit 0; fi

printf 'Disk space before cleanup:\n'
df -h "$root"
manifest="$root/image-cleanup-files.txt"
if [[ -L "$manifest" ]]; then printf 'Refusing a linked cleanup manifest.\n' >&2; exit 1; fi
printf '%s\n' "${images[@]}" > "$manifest"
# Small argument batches work in Git Bash/MSYS2 and tolerate paths with spaces.
for ((i=0; i<${#images[@]}; i+=50)); do
  rm -f -- "${images[@]:i:50}"
done
printf 'Deleted %s obsolete images. Disk space after cleanup:\n' "${#images[@]}"
df -h "$root"
