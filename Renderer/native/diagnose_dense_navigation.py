"""One fixed causal batch selecting the next retained renderer implementation.

Pixel ablations are diagnostics, never production correctness or gameplay passes.
All arms use the production DLL and the existing world/capture/evidence owners.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from Renderer.lab.platform import ROOT
from Renderer.native.analyze_navigation_run import (read_dense_diagnostic_case as read_case,
    compare_dense_diagnostic_runs as compare_runs, summarize_dense_diagnostic as summarize)

OFFSETS = [1, 2, 4, 8, 4, 2, 1, 0, -2, -4, -8, -4, -2, 0]
ARMS = {
    "full": [],
    "route_draws_omitted": ["--diagnostic-routes", "draw"],
    "route_surfaces_omitted": ["--diagnostic-routes", "all"],
    "prepared_content": ["--case-reset", "prepared_resident"],
    "half_geometry_pixels": ["--diagnostic-half-pixels"],
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, choices=(2, 3), default=2)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.relative_to(ROOT / "Renderer/native/build")
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "viewport": [2240, 1192], "tile_width": 128, "center": [75, 39],
                "hour": 12, "clock": 1000000, "dense_scene": True, "waves": False, "reflections": False,
                "fixture": "Renderer/lab/.local/verification/world.csv", "offsets": OFFSETS,
                "arms": ARMS, "repetitions": args.repetitions, "minimum_useful_mean_transition_ms": 20,
                "minimum_useful_fraction": .1, "cache_tier": "normal", "budgets": "unchanged production retained defaults",
                "limits": ["Standalone synthetic population on world terrain, not a saved game or live presentation",
                           "Route controls cover road/rail surfaces; bridge and improvement objects remain",
                           "Prepared arm warms all scene content including route buffers; not route-only attribution",
                           "Half-pixel arm retains capture/geometry/selection/submission, halves geometry scissor width; finishing/readback unchanged",
                           "First exposure and revisits are separate; cached destination revisits cannot establish the navigation gate",
                           "Two repetitions detect large effects; no tail or gameplay performance claim"]}
    write(out / "manifest.json", manifest)
    runs = []
    identity = None
    try:
        for repeat in range(args.repetitions):
            arms = list(ARMS) if repeat % 2 == 0 else list(reversed(ARMS))
            for workload, offsets in (("four_columns", [4]), ("reversal", OFFSETS)):
                for arm in arms:
                    folder = out / f"r{repeat}-{workload}-{arm}"
                    command = [sys.executable, "-m", "Renderer.native.record_navigation_evidence",
                               "--binaries", str(args.binaries), "--out", str(folder), "--scenario", "scroll",
                               "--tier", "normal", "--width", "2240", "--height", "1192", "--tile-width", "128",
                               "--dense-scene", "--waves", "0", "--reflection-ablation", "--profile",
                               "--world-grid", "--world-regions", "--production-defaults",
                               "--case-repeats", "1", "--case-time-limit", "180", "--exclusive-gpu", *ARMS[arm]]
                    if workload == "reversal":
                        command.append("--scroll-sequence")
                    print(f"Dense diagnostic: repetition {repeat+1}, {workload}, {arm}", flush=True)
                    subprocess.run(command, cwd=ROOT, check=True)
                    result = read_case(folder, offsets)
                    current_identity = result.pop("identity")
                    if identity is not None and identity != current_identity:
                        raise ValueError("Batch source/assets/binary identity changed between arms")
                    if runs and (result["budgets"] != runs[0]["budgets"] or result["fixture"] != runs[0]["fixture"]):
                        raise ValueError("Batch budgets or fixture changed between arms")
                    identity = current_identity
                    result.update(arm=arm, workload=workload, repeat=repeat)
                    runs.append(result)
                    write(out / "results.json", {"status": "running", "runs": runs})
        comparisons = summarize(runs)
        write(out / "results.json", {"status": "complete", "runs": runs, "comparisons": comparisons,
                                      "performance_claim": "causal diagnosis only; no retained optimization or native navigation pass"})
        print(json.dumps(comparisons, indent=2), flush=True)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        write(out / "results.json", {"status": "invalid_or_incomplete", "reason": str(error), "runs": runs})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
