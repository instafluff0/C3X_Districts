"""Run extracted production C++ contracts with the host's native compiler."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from Renderer.lab.platform import ROOT, native_command_result, windows_root


def run_cpp(program, *, sources=(), timeout=30):
    build = ROOT / "Renderer/native/build"
    build.mkdir(exist_ok=True)
    # Python 3.9 is still the system interpreter on the Mac fast path;
    # ignore_cleanup_errors was only added in 3.10.
    with tempfile.TemporaryDirectory(dir=build) as directory:
        path = Path(directory)
        cpp = path / "contract.cpp"
        cpp.write_text(program)
        needs_windows = any('#include <windows.h>' in (ROOT / source).read_text()
                            for source in sources)
        if os.name == "nt" or needs_windows:
            target_root = ROOT if os.name == "nt" else windows_root()
            # Keep generated paths as quoted batch arguments; no shell expansion
            # is allowed in caller-provided source names.
            paths = [str(target_root), *(str(target_root / source) for source in sources)]
            if any(any(char in value for char in '\"%\r\n') for value in paths):
                raise ValueError("Unsupported native test path")
            additional = " ".join(f'"{value}"' for value in paths[1:])
            batch = path / "test.bat"
            batch.write_text(r'''@echo off
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "C3X_TEST_VS_RECORD=%TEMP%\c3x-renderer-test-vs-path.txt"
"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath >"%C3X_TEST_VS_RECORD%"
set /p C3X_TEST_VS=<"%C3X_TEST_VS_RECORD%"
if not defined C3X_TEST_VS "%VSWHERE%" -all -products * -property installationPath >"%C3X_TEST_VS_RECORD%"
if not defined C3X_TEST_VS set /p C3X_TEST_VS=<"%C3X_TEST_VS_RECORD%"
del "%C3X_TEST_VS_RECORD%" >nul 2>nul
if not defined C3X_TEST_VS exit /b 2
if not exist "%C3X_TEST_VS%\VC\Auxiliary\Build\vcvars32.bat" exit /b 2
call "%C3X_TEST_VS%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 exit /b 2
pushd "%~dp0"
''' + f'cl /nologo /std:c++17 /EHsc /O1 /W3 /I "{target_root}" contract.cpp {additional} /Fe:contract.exe /link /LARGEADDRESSAWARE\n'
                + 'if errorlevel 1 exit /b 1\ncontract.exe\nexit /b %errorlevel%\n')
            target_batch = target_root / batch.relative_to(ROOT).as_posix()
            result = native_command_result("Renderer/native", f'call "{target_batch}"', timeout_seconds=timeout+60)
            if result["returncode"] != 0:
                raise AssertionError(result["output_tail"])
        else:
            compiler = shutil.which("clang++") or shutil.which("g++")
            if not compiler:
                raise unittest.SkipTest("C++ compiler unavailable")
            executable = path / "contract"
            subprocess.run([compiler, "-std=c++17", "-O1", "-pthread", "-I", str(ROOT),
                            str(cpp), *(str(ROOT / source) for source in sources), "-o", str(executable)], check=True)
            subprocess.run([str(executable)], check=True, timeout=timeout)
