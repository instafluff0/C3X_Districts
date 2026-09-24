"""Pin the exact Renderer64 diagnostic capture binaries after controlled checks.

This qualifies one short recording attempt, not live FPS or visual parity.
The real game's display timeline must still be calibrated against its samples.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from Renderer.lab.platform import ROOT


BUILD = ROOT / "Renderer/native/build"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def summary(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.startswith("{")]
    if not rows:
        raise ValueError(f"Missing replay summary: {path}")
    return rows[-1]


def machine(path):
    data = path.read_bytes()[:4096]
    offset = struct.unpack_from("<I", data, 60)[0]
    if data[:2] != b"MZ" or data[offset:offset + 4] != b"PE\0\0":
        raise ValueError(f"Invalid PE binary: {path}")
    return struct.unpack_from("<H", data, offset + 4)[0]


def qualify(fixture, controls, startup):
    for path in (fixture, controls, startup):
        path.resolve().relative_to(BUILD)
    staged = ROOT / "Renderer/bin/renderer64"
    bridge = staged / "C3XRenderer.dll"
    core = staged / "C3XRenderer_x64.dll"
    helper = staged / "C3XRendererHelper64.exe"
    if [machine(path) for path in (bridge, core, helper)] != [0x14c, 0x8664, 0x8664]:
        raise ValueError("Renderer64 capture needs one x86 bridge and two x64 companions")

    recorded = read(fixture / "receipt.json")
    if not recorded.get("inputs_unchanged") or not recorded["input_recording"]["closed"]:
        raise ValueError("Controlled recording did not close with unchanged inputs")
    for path in (bridge, core, helper):
        key = path.relative_to(ROOT).as_posix()
        if recorded["inputs"].get(key) != digest(path):
            raise ValueError(f"Controlled recording used another binary: {path.name}")
    if recorded.get("status") != "fail" or "every native final displayed pixel exact" not in (
            fixture / "test.log").read_text(errors="replace"):
        raise ValueError("Expected direct-surface visual-oracle limitation changed")

    checks = read(controls / "receipt.json")
    if checks.get("status") != "pass" or checks.get("game_launched") is not False or checks.get("sampled_frames", 0) < 1:
        raise ValueError("Launcher/window collector control did not pass")
    if "renderer64-bootstrap select=1 definitions=1 healthy=1" not in startup.read_text(encoding="utf-8-sig"):
        raise ValueError("Fresh-process Renderer64 startup probe did not pass")
    for name, log in (("exact", "renderer64-exact-replay.log"),
                      ("realtime", "renderer64-realtime.log")):
        replay = summary(fixture / log)
        if (replay.get("status") != "development_input_replay" or not replay.get("complete")
                or not replay.get("binary_matches_capture") or not replay.get("direct_surface_trial")
                or replay.get("unfinished_calls") != 0 or replay.get("accepted_presentations", 0) < 1
                or replay.get("calls", 0) < 100):
            raise ValueError(f"Renderer64 {name} replay did not complete")

    tools = ROOT / "Renderer/native/build/input-recording"
    receipt = {
        "status": "pass", "scope": "renderer64-short-diagnostic-capture",
        "dll_sha256": digest(bridge), "renderer64_dll_sha256": digest(core),
        "renderer64_helper_sha256": digest(helper),
        "window_witness_sha256": digest(BUILD / "window-witness/window_witness.exe"),
        "inspector_sha256": digest(tools / "inspect_inputs.exe"),
        "replay_sha256": digest(tools / "replay_inputs.exe"),
        "launcher_sha256": digest(ROOT / "Renderer/tools/capture_game.ps1"),
        "batch_sha256": digest(ROOT / "Renderer/CAPTURE_DIAGNOSTIC.bat"),
        "frames_launcher_sha256": digest(ROOT / "Renderer/tools/capture_frames.ps1"),
        "startup_probe_sha256": digest(BUILD / "renderer64_startup_probe.exe"),
        "evidence": {"fixture": fixture.relative_to(ROOT).as_posix(),
                     "launcher_controls": controls.relative_to(ROOT).as_posix(),
                     "startup_log": startup.relative_to(ROOT).as_posix()},
        "recorded_direct_calls": summary(fixture / "renderer64-exact-replay.log")["calls"],
        "recorded_direct_presentations": summary(fixture / "renderer64-exact-replay.log")["accepted_presentations"],
        "live_fps_qualified": False, "live_replay_calibrated": False,
        "native_pixel_parity_qualified": False,
        "limitations": ["The controlled direct-surface fixture fails the legacy x86-pixel oracle; recording and same-route replay pass.",
                        "PresentMon captures Renderer64 helper presents; sampled window images cover the full Civ III composite.",
                        "Recorder/window observer overhead and live gameplay scheduling require calibration before FPS claims."],
    }
    target = tools / "renderer64-short-capture-ready.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--launcher-controls", type=Path, required=True)
    parser.add_argument("--startup-log", type=Path, required=True)
    args = parser.parse_args()
    print(qualify(args.fixture.resolve(), args.launcher_controls.resolve(), args.startup_log.resolve()))
