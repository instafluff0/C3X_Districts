#!/usr/bin/env python3
"""A private build of the checkout's constructed-wonder port, for Lab parity.

The port landed in the checkout (docs/constructed_wonder_rendering.md); this tree owns
nothing and only copies the checkout, so prod_parity.py renders with binaries built in
isolation while other sessions compile the shared tree.

    python3 Renderer/lab/studies/wonders/prod_tree.py prepare   # copy the checkout
    python3 Renderer/lab/studies/wonders/prod_tree.py build     # VM compile of that tree
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.mountains import private_tree

NAME = "prod"
private_tree.OUT = ROOT / "Renderer/lab/out/wonders"


def prepare() -> None:
    (private_tree.tree(NAME) / "owned.txt").unlink(missing_ok=True)
    private_tree.sync(NAME)


def binaries() -> tuple[Path, Path]:
    root = private_tree.tree(NAME)
    return root / "Renderer/native/build/candidate/C3XRenderer.dll", root / "Renderer/lab/.cache/native_preview.exe"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("prepare", "build"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        private_tree.build(NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
