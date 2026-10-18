#!/usr/bin/env python3
"""Renderer64 memory peak and world-tile churn of one scripted run.

Reads the CSV written by sample_memory.ps1 during a scenario (any sampler with
the columns t, process, private_mb and gpu_shared_mb) and, optionally, that
run's renderer log. In the Parallels VM every D3D allocation is charged to the
helper's private bytes, so private_mb is the helper's whole footprint and
gpu_shared_mb the part the adapter reports.

Fails (exit 1) when the helper's peak private or GPU memory exceeds the
limits, or when the log shows world tiles restored from the RAM copy or
compiled again after loading (the eviction churn of performance review 87).
Defaults are the busy 1498 AD save's `near` targets: 4.5 GB private, 2.5 GB
GPU.

    python3 Renderer/tools/memory_peak_report.py memory.csv [--log renderer.log]
"""
import argparse
import csv
import re
import statistics
import sys


def helper_rows(path):
    with open(path, encoding='utf-8', errors='ignore') as handle:
        rows = [row for row in csv.DictReader(handle) if row.get('process') == 'C3XRendererHelper64']
    return [(float(row['t']), float(row['private_mb']), float(row['gpu_shared_mb'])) for row in rows]


def world_churn(path):
    """World tiles restored or compiled again after the loading sweep."""
    released, loaded, restores, compiles = False, False, 0, 0
    with open(path, encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            if 'stage=world-gpu-residency ' in line:
                loaded = True
            elif 'stage=world-backing-released' in line:
                released = True
            elif loaded and 'stage=world-backing records=' in line:
                match = re.search(r'compiler_calls=(\d+) restore_calls=(\d+)', line)
                if match:
                    compiles += int(match.group(1))
                    restores += int(match.group(2))
    return released, restores, compiles


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('samples')
    parser.add_argument('--log', help='renderer.log of the same run')
    parser.add_argument('--max-private-mb', type=float, default=4500)
    parser.add_argument('--max-gpu-mb', type=float, default=2500)
    parser.add_argument('--max-churn', type=int, default=100, help='world tiles restored or compiled after loading')
    args = parser.parse_args()
    rows = helper_rows(args.samples)
    if not rows:
        print('no helper samples')
        return 1
    peak = max(rows, key=lambda row: row[1])
    gpu = max(rows, key=lambda row: row[2])
    end = rows[-1][0]
    late = [row for row in rows if row[0] >= end - 60] or rows
    print(f'helper peak private {peak[1]:.0f} MB at t={peak[0]:.0f} s; peak GPU {gpu[2]:.0f} MB at t={gpu[0]:.0f} s')
    print(f'last 60 s median: private {statistics.median(r[1] for r in late):.0f} MB, GPU {statistics.median(r[2] for r in late):.0f} MB')
    failed = peak[1] > args.max_private_mb or gpu[2] > args.max_gpu_mb
    if args.log:
        released, restores, compiles = world_churn(args.log)
        print(f'world RAM copy released: {"yes" if released else "no"}; after loading: '
              f'{restores} tiles restored, {compiles} compiled')
        failed = failed or restores + compiles > args.max_churn
    print('FAIL' if failed else 'PASS')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
