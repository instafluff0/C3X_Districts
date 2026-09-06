#!/usr/bin/env bash
set -euo pipefail

cd /c
MSYS2_ARG_CONV_EXCL='*' cmd.exe /d /c 'cd /d "C:\Program Files (x86)\GOG Galaxy\Games\Civilization III Complete\Conquests\C3X_Districts" && call TEST_INJECTED_CODE_COMPILE.bat'
