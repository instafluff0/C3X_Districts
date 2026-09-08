"""Portable compiler helpers shared by the current Lab and retained source tools."""
from pathlib import Path
import os
import re
import subprocess
from .cache import file_hash

ROOT = Path(__file__).resolve().parents[3]
SHADER_TOOLS = ROOT / "Renderer/lab/.local/shader-tools"

def run(args, **kw):
    result = subprocess.run(
        [str(x) for x in args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **kw,
    )
    if result.returncode:
        raise ValueError(f"command failed: {Path(str(args[0])).name}\n{result.stdout}")
    return result.stdout


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def closure(path, active=()):
    path = Path(path).resolve()
    if path in active:
        raise ValueError("cyclic shader include")
    result = {relative(path): file_hash(path)}
    for name in re.findall(r'^\s*#include\s+"([^"]+)"', path.read_text(), re.M):
        child = (path.parent / name).resolve()
        if not child.is_file():
            raise ValueError("missing shader include: " + name)
        result.update(closure(child, active + (path,)))
    return result


def shader_source(path, active=()):
    path = Path(path).resolve()
    if path in active:
        raise ValueError("cyclic shader include")

    def include(match):
        return shader_source(path.parent / match.group(1), active + (path,))

    return re.sub(
        r'^\s*#include\s+"([^"]+)"[^\n]*', include, path.read_text(), flags=re.M
    )


def tool_libraries(env):
    return {
        str(index) + ":" + p.name: file_hash(p)
        for index, directory in enumerate(env["DYLD_LIBRARY_PATH"].split(":"))
        for p in sorted(Path(directory).glob("*.dylib"))
    }


def compile_cpp(cache, source, objc=False):
    flags = ["-std=c++17", "-O2"] + (["-fobjc-arc"] if objc else [])
    dependencies = run(["clang++", *flags, "-MM", source])
    # clang escapes spaces in repository paths. Match escaped tokens before resolving.
    words = re.findall(
        r"(?:\\.|[^\s])+", dependencies.replace("\\\n", " ").split(":", 1)[1]
    )
    deps = [Path(x.replace("\\ ", " ")) for x in words]
    identity = {
        "compiler": run(["clang++", "--version"]),
        "flags": flags,
        "dependencies": {relative(p): file_hash(p) for p in deps},
    }
    return cache.artifact(
        "cpp", identity, lambda out: run(["clang++", *flags, "-c", source, "-o", out])
    )


def tools():
    root = Path(os.environ.get("C3X_LAB_SHADER_TOOLS", str(SHADER_TOOLS)))
    glslang = root / "glslang/16.5.0/bin/glslang"
    cross = root / "spirv-cross/1.4.357.0/bin/spirv-cross"
    if not glslang.is_file() or not cross.is_file():
        raise ValueError(
            "shader tools missing; run Renderer/lab/backends/bootstrap_tools.py or configure C3X_LAB_SHADER_TOOLS"
        )
    env = dict(
        os.environ,
        DYLD_LIBRARY_PATH=str(root / "glslang/16.5.0/lib")
        + ":"
        + str(root / "spirv-tools/1.4.357.0/lib"),
    )
    return glslang, cross, env


def executable(cache, source, *, objc=False, libraries=()):
    obj=compile_cpp(cache,source,objc)
    target=cache.artifact("executable", {"object":file_hash(obj),"libraries":list(libraries)},
                         lambda out:run(["clang++",obj,*libraries,"-o",out]))
    target.chmod(0o755)
    return target


def apply_mip_bias(source, bias):
    if bias == 0:
        return source
    result = []
    start = 0
    for m in re.finditer(r"\.Sample\(", source):
        pos = m.end()
        depth = 1
        while depth and pos < len(source):
            depth += (source[pos] == "(") - (source[pos] == ")")
            pos += 1
        if depth:
            raise ValueError("malformed texture sample call")
        result.append(
            source[start : m.start()]
            + ".SampleBias("
            + source[m.end() : pos - 1]
            + ","
            + str(float(bias))
            + ")"
        )
        start = pos
    result.append(source[start:])
    return "".join(result)


def shaders(cache, path, bias, msl_version=20100, *, entrypoints=None, entries=None):
    if msl_version not in (20100,20200):
        raise ValueError("unsupported MSL capability version")
    glslang, cross, env = tools()
    identity = {
        "closure": closure(path),
        "glslang": file_hash(glslang),
        "spirv_cross": file_hash(cross),
        "tool_libraries": tool_libraries(env),
        "mip_bias": bias,
        "msl": msl_version,
        "bindings": {"textures": 0, "samplers": 128, "constants": 130},
        "compiler_options": "auto-map, strict runtime Metal math",
        "entry_adapter": 2,
    }
    outputs = {}
    for entry in entries or ("VSMain", "VSFeature", "PSMain", "PSFeature"):
        if not re.fullmatch(r"(?:VS|PS)[A-Za-z0-9_]*", entry):
            raise ValueError("Invalid vertex/pixel shader entry point")
        stage = "vert" if entry.startswith("VS") else "frag"
        source_entry=(entrypoints or {}).get(entry,entry)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",source_entry):
            raise ValueError("Invalid shader entry point")

        def build(out):
            source = out.parent / "input.hlsl"
            # Resolve includes through the original shader directory; no source-specific behavior.
            text=shader_source(path)
            if not re.search(r"\b"+source_entry+r"\s*\(",text):
                raise ValueError("Missing shader entry point: "+source_entry)
            if source_entry!=entry:
                # Keep an existing helper with the destination name callable.
                # glslang otherwise renames the selected entry onto that body.
                helper="c3x_source_"+entry
                while re.search(r"\b"+helper+r"\b",text):helper="c3x_"+helper
                text=re.sub(r"\b"+entry+r"\b",helper,text)
                text=re.sub(r"\b"+source_entry+r"\b",entry,text)
            source.write_text(apply_mip_bias(text, bias))
            spv = out.parent / "shader.spv"
            run(
                [
                    glslang,
                    "-D",
                    "-V",
                    "-S",
                    stage,
                    "-e",
                    entry,
                    "--source-entrypoint",
                    entry,
                    "--auto-map-bindings",
                    "--auto-map-locations",
                    "--shift-texture-binding",
                    stage,
                    "0",
                    "--shift-sampler-binding",
                    stage,
                    "128",
                    "--shift-UBO-binding",
                    stage,
                    "130",
                    "-I" + str(path.parent),
                    source,
                    "-o",
                    spv,
                ],
                env=env,
            )
            run(
                [
                    cross,
                    spv,
                    "--msl",
                    "--msl-version",
                    str(msl_version),
                    "--msl-argument-buffers",
                    "--msl-decoration-binding",
                    "--output",
                    out,
                ]
            )

        outputs[entry] = cache.artifact("shader", dict(identity, entry=entry, source_entry=source_entry), build)
    return outputs
