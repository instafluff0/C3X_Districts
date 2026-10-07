#!/usr/bin/env python3
"""Private source and shader tree for a Lab candidate.

Several sessions build and stage from this one checkout. A visual proposal
that is not yet accepted must not leak into their builds, so this tool copies
the candidate DLL's include closure, the Lab preview tool's closure and the
runtime shader tree into `Renderer/lab/out/mountains/tree-NAME/`, compiles the
DLL and preview there on the VM, and regenerates shaders from the private
shader sources. The checkout itself is never written.

Files listed in the tree's `owned.txt` are edits under study: `sync` keeps
them and records the checkout version they started from under `.base/`, so a
promotion can merge them into whatever the checkout holds by then.

    python3 Renderer/lab/studies/mountains/private_tree.py sync NAME
    python3 Renderer/lab/studies/mountains/private_tree.py own NAME PATH...
    python3 Renderer/lab/studies/mountains/private_tree.py shaders NAME
    python3 Renderer/lab/studies/mountains/private_tree.py build NAME
    python3 Renderer/lab/studies/mountains/private_tree.py diff NAME
    python3 Renderer/lab/studies/mountains/private_tree.py root NAME VARIANT KNOB=VALUE...

`root` copies the tree's runtime shaders to `ranges/root-VARIANT` and prepends
the given `#define` knobs to the generated mountain shaders.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer import renderer
from Renderer.lab import preparation

OUT = ROOT / "Renderer/lab/out/mountains"
SKIP_PARTS = {"build", ".cache", "__pycache__"}


def tree(name: str) -> Path:
    return OUT / f"tree-{name}"


def owned(root: Path) -> set[str]:
    path = root / "owned.txt"
    return {line.strip() for line in path.read_text().splitlines() if line.strip()} if path.is_file() else set()


def closure() -> set[str]:
    sources = set(renderer.native_inputs())
    sources |= set(renderer.source_inputs([renderer.LAB / "native_preview.cpp", renderer.LAB / "build_native_preview.bat",
                                           ROOT / "Renderer/native/biq_preview.cpp", ROOT / "Renderer/native/c3x_renderer_api.h"]))
    sources |= set(preparation.input_paths(ROOT))
    # Runtime shader tree: every non-code file the renderer may compile or read.
    for path in (ROOT / "Renderer/native").rglob("*"):
        if path.is_file() and not SKIP_PARTS & set(path.relative_to(ROOT).parts) and \
                path.suffix.lower() in (".hlsl", ".hlsli", ".json", ".h"):
            sources.add(path.relative_to(ROOT).as_posix())
    for path in (ROOT / "Renderer/lab/shared").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            sources.add(path.relative_to(ROOT).as_posix())
    return sources


def sync(name: str) -> None:
    root = tree(name)
    keep = owned(root)
    copied = 0
    for relative in sorted(closure()):
        if relative in keep:
            continue
        source, target = ROOT / relative, root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.read_bytes() != source.read_bytes():
            shutil.copy2(source, target)
            copied += 1
    print(f"{root.relative_to(ROOT)}: {copied} files refreshed, {len(keep)} owned")


def own(name: str, paths: list[str]) -> None:
    root = tree(name)
    keep = owned(root)
    for relative in paths:
        relative = Path(relative).as_posix()
        source = ROOT / relative
        base = root / ".base" / relative
        base.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            shutil.copy2(source, base)
            if not (root / relative).is_file():
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, root / relative)
        keep.add(relative)
    (root / "owned.txt").write_text("\n".join(sorted(keep)) + "\n")
    print("owned:", ", ".join(sorted(keep)))


def shaders(name: str) -> None:
    """Regenerate prepared shaders from the private sources into the tree."""
    root = tree(name)
    names = preparation.input_paths(root)
    with tempfile.TemporaryDirectory(prefix="mountain-prepare-") as directory:
        mirror = Path(directory)
        for relative in names:
            target = mirror / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
        preparation.generate(mirror)
        produced = 0
        for path in mirror.rglob("*"):
            relative = path.relative_to(mirror).as_posix()
            if path.is_file() and "__pycache__" not in path.parts and relative not in names:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.is_file() or target.read_bytes() != path.read_bytes():
                    shutil.copyfile(path, target)
                    produced += 1
    print(f"{produced} generated shader files updated in {root.relative_to(ROOT)}")


def build(name: str) -> None:
    from Renderer.lab.platform import native_command_result
    # `sync` refreshes generated shaders from the checkout; regenerate them
    # from the private sources so a build never pairs new code with old shaders.
    shaders(name)
    root = tree(name)
    native = root / "Renderer/native"
    script = native / "BUILD_PRIVATE.bat"
    command = None
    for line in (ROOT / "Renderer/native/BUILD.bat").read_text().splitlines():
        if "/Fe:build\\candidate\\C3XRenderer.dll" in line:
            command = line.strip()
    if not command:
        raise SystemExit("Candidate compile command not found in BUILD.bat")
    setup = (ROOT / "Renderer/native/BUILD.bat").read_text().split("if not exist \"..\\bin\"")[0]
    script.write_text(setup.replace("\n", "\r\n") +
                      "if not exist \"build\\candidate\" mkdir \"build\\candidate\"\r\n" +
                      command.replace("%C3X_RENDERER_ORACLE_FLAGS%", "") + "\r\nif errorlevel 1 exit /b 1\r\n"
                      "pushd \"%~dp0..\\lab\"\r\nif not exist \".cache\" mkdir \".cache\"\r\n"
                      "cl /nologo /std:c++17 /EHsc /O2 /W4 /WX native_preview.cpp /Fo:.cache\\native_preview.obj "
                      "/Fe:.cache\\native_preview.exe /link /LARGEADDRESSAWARE gdi32.lib user32.lib\r\n"
                      "set \"C3X_PRIVATE_RESULT=%errorlevel%\"\r\npopd\r\npopd\r\nexit /b %C3X_PRIVATE_RESULT%\r\n")
    relative = native.relative_to(ROOT).as_posix()
    result = native_command_result(relative, "call BUILD_PRIVATE.bat")
    if result["status"] != "pass":
        raise SystemExit("Private build failed")
    dll = native / "build/candidate/C3XRenderer.dll"
    preview = root / "Renderer/lab/.cache/native_preview.exe"
    print(dll.relative_to(ROOT), renderer.checksum(dll))
    print(preview.relative_to(ROOT), renderer.checksum(preview))


MOUNTAIN_SHADERS = ("Renderer/native/city_fidelity/mountain.hlsl", "Renderer/native/source_fidelity/mountain.hlsl",
                    "Renderer/native/environment_refresh/mountain.hlsl")


def variant_root(name: str, variant: str, knobs: list[str]) -> None:
    source, root = tree(name), OUT / "ranges" / f"root-{variant}"
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(source, root, ignore=shutil.ignore_patterns("build", ".cache", ".base", "*.cpp", "*.bat"))
    head = "".join("#define {} {}\n".format(*knob.split("=", 1)) for knob in knobs)
    for relative in MOUNTAIN_SHADERS:
        path = root / relative
        path.write_text(head + path.read_text())
    print(root.relative_to(ROOT), head.strip().replace("\n", "; "))


def diff(name: str) -> None:
    root = tree(name)
    for relative in sorted(owned(root)):
        base = root / ".base" / relative
        subprocess.run(["git", "--no-pager", "diff", "--no-index", "--stat",
                        str(base) if base.is_file() else "/dev/null", str(root / relative)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("sync", "own", "shaders", "build", "diff", "root"))
    parser.add_argument("name")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    if args.command == "sync":
        sync(args.name)
    elif args.command == "own":
        own(args.name, args.paths)
    elif args.command == "shaders":
        shaders(args.name)
    elif args.command == "build":
        build(args.name)
    elif args.command == "root":
        variant_root(args.name, args.paths[0], args.paths[1:])
    else:
        diff(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
