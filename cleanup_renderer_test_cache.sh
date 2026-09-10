#!/usr/bin/env bash
# Reclaim generated renderer-test storage while preserving source assets,
# production binaries, rollback DLLs, reports, logs, and representative images.
set -euo pipefail

apply=false
include_snapshots=false

usage() {
  cat <<'EOF'
Usage: bash cleanup_renderer_test_cache.sh [--preview] [--apply] [--include-snapshots]

  --preview            Show what would be removed (the default).
  --apply              Perform the cleanup after safety checks.
  --include-snapshots  Also remove explicitly listed historical Lab checkout
                       snapshots. These are large but can contain local copies,
                       so they are excluded by default.

The standard cleanup removes:
  * generated native benchmark BMP/BMP.GZ frames, while retaining one base
    image per run and all current production-verification runs;
  * Renderer/lab/.cache; and
  * Renderer/lab/out/integration/replays.

JSON reports, logs, DLLs, executables, patches, approved references, local
verification inputs, runtime/source packs, promotions, and rollback files stay.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --preview) apply=false ;;
    --apply) apply=true ;;
    --include-snapshots) include_snapshots=true ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
native_build="$repo_root/Renderer/native/build"
lab_cache="$repo_root/Renderer/lab/.cache"
integration_replays="$repo_root/Renderer/lab/out/integration/replays"
lab_out="$repo_root/Renderer/lab/out"

if [[ ! -f "$repo_root/Renderer/README.md" || ! -f "$repo_root/Renderer/lab/catalog.json" ]]; then
  printf 'This script must remain in the C3X_Districts repository root.\n' >&2
  exit 1
fi
for required_dir in "$native_build" "$lab_out"; do
  if [[ ! -d "$required_dir" || -L "$required_dir" ]]; then
    printf 'Missing or linked required directory: %s\n' "$required_dir" >&2
    exit 1
  fi
done

# These runs are the compact current-production/default equivalence evidence.
# Their frame sets are small enough to retain in full.
is_protected_run() {
  case "$1" in
    navigation-production-explicit-128|navigation-production-explicit-160|navigation-production-explicit-192|\
    navigation-production-defaults-128|navigation-production-defaults-160|navigation-production-defaults-192|\
    navigation-usage-animation-128)
      return 0
      ;;
    *) return 1 ;;
  esac
}

native_images=()
native_image_count=0
if [[ -d "$native_build" ]]; then
  while IFS= read -r -d '' file; do
    [[ -f "$file" && ! -L "$file" ]] || continue
    relative=${file#"$native_build"/}
    run=${relative%%/*}
    base=${file##*/}
    is_protected_run "$run" && continue
    case "$base" in
      zoom.bmp|zoom.bmp.gz) continue ;;
    esac
    native_images+=("$file")
    native_image_count=$((native_image_count + 1))
  done < <(find "$native_build" -type f \( -name '*.bmp' -o -name '*.bmp.gz' \) -print0)
fi

snapshot_dirs=(
  "$lab_out/mountains/promotion/root"
  "$lab_out/mountains/zebra-study/root"
  "$lab_out/mountains/crevice-correction/root"
  "$lab_out/shorelines/surface-transition/root"
  "$lab_out/waves/snapshot"
  "$lab_out/waves-quality/snapshot"
  "$lab_out/waves-spacing/snapshot"
  "$lab_out/waves-shoaling/snapshot"
)

selected_trees=()
selected_tree_count=0
for tree in "$lab_cache" "$integration_replays"; do
  if [[ -e "$tree" || -L "$tree" ]]; then
    selected_trees+=("$tree")
    selected_tree_count=$((selected_tree_count + 1))
  fi
done
if [[ "$include_snapshots" == true ]]; then
  for tree in "${snapshot_dirs[@]}"; do
    if [[ -e "$tree" || -L "$tree" ]]; then
      selected_trees+=("$tree")
      selected_tree_count=$((selected_tree_count + 1))
    fi
  done
fi

image_bytes=0
if (( native_image_count > 0 )); then
  for file in "${native_images[@]}"; do
    if size=$(stat -f '%z' "$file" 2>/dev/null); then
      :
    else
      size=$(stat -c '%s' "$file")
    fi
    image_bytes=$((image_bytes + size))
  done
fi

tree_kib=0
if (( selected_tree_count > 0 )); then
  # One du invocation avoids counting the same hard-linked payload once for
  # every selected snapshot that names it.
  tree_kib=$(du -sk "${selected_trees[@]}" | awk '{sum += $1} END {print sum + 0}')
fi

awk -v count="$native_image_count" -v bytes="$image_bytes" -v trees="$selected_tree_count" -v kib="$tree_kib" '
  BEGIN {
    printf "Selected %d generated native frames (%.2f GiB logical).\n", count, bytes / 1073741824
    printf "Selected %d generated trees (%.2f GiB, with in-selection hard links counted once).\n", trees, kib / 1048576
  }
'
printf 'Protected production-verification runs and per-run zoom.bmp base images stay.\n'
printf 'Reports, logs, binaries, patches, assets, local inputs and references stay.\n'
if [[ "$include_snapshots" == false ]]; then
  printf 'Historical Lab snapshots are excluded; add --include-snapshots to select them.\n'
fi

if [[ "$apply" == false ]]; then
  printf '\nPreview only. Re-run with --apply to delete the selected output.\n'
  exit 0
fi

# Compression, benchmark, and Integration jobs update the selected outputs.
# Refuse to race any of them.
recent_output=$(find "$native_build" -type f \( -name '*.bmp' -o -name '*.bmp.gz' \) -mmin -2 -print -quit)
if [[ -z "$recent_output" && -d "$lab_cache" ]]; then
  recent_output=$(find "$lab_cache" -type f -mmin -2 -print -quit)
fi
if [[ -z "$recent_output" && -d "$integration_replays" ]]; then
  recent_output=$(find "$integration_replays" -type f -mmin -2 -print -quit)
fi
if [[ -n "$recent_output" ]]; then
  printf 'Refusing cleanup: selected output changed within the last two minutes.\n' >&2
  printf 'Wait for the renderer, Integration, or compression job to finish, then run this command again.\n' >&2
  exit 1
fi

timestamp=$(date '+%Y%m%d-%H%M%S')
manifest="$native_build/cache-cleanup-$timestamp.txt"
if [[ -e "$manifest" || -L "$manifest" ]]; then
  printf 'Refusing to overwrite cleanup manifest: %s\n' "$manifest" >&2
  exit 1
fi
{
  printf 'C3X renderer generated-output cleanup\n'
  printf 'Timestamp: %s\n' "$timestamp"
  printf 'Native frames:\n'
  if (( native_image_count > 0 )); then printf '%s\n' "${native_images[@]}"; fi
  printf 'Generated trees:\n'
  if (( selected_tree_count > 0 )); then printf '%s\n' "${selected_trees[@]}"; fi
} > "$manifest"

printf '\nDisk space before cleanup:\n'
df -h "$repo_root"

# Small argument batches avoid command-line length limits.
if (( native_image_count > 0 )); then
  for ((i=0; i<native_image_count; i+=50)); do
    rm -f -- "${native_images[@]:i:50}"
  done
fi

if (( selected_tree_count > 0 )); then
  for tree in "${selected_trees[@]}"; do
    case "$tree" in
      "$lab_cache"|"$integration_replays"|"$lab_out"/*/snapshot|"$lab_out"/*/*/snapshot|\
      "$lab_out/mountains/promotion/root"|"$lab_out/mountains/zebra-study/root"|\
      "$lab_out/mountains/crevice-correction/root"|"$lab_out/shorelines/surface-transition/root")
        rm -rf -- "$tree"
        ;;
      *)
        printf 'Refusing unexpected cleanup target: %s\n' "$tree" >&2
        exit 1
        ;;
    esac
  done
fi

printf '\nCleanup complete. Manifest retained at:\n%s\n' "$manifest"
printf 'Disk space after cleanup:\n'
df -h "$repo_root"
