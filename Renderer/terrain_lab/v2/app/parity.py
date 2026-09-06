#!/usr/bin/env python3
"""Compare controlled BGRA8 packet renders and dispatch D3D through renderer_dev."""
import argparse
import os
import signal
import subprocess
import array
import json
from pathlib import Path
import struct
import sys
import time
from cache import canonical, file_hash
from runner import ROOT, V2, CONTRACT, local, relative, run


def pixels(path):
    b = Path(path).read_bytes()
    if b[:2] != b"BM" or len(b) < 54:
        raise ValueError("not a BMP")
    offset = struct.unpack_from("<I", b, 10)[0]
    w, h = struct.unpack_from("<ii", b, 18)
    bits = struct.unpack_from("<H", b, 28)[0]
    if bits != 32 or w <= 0 or h >= 0 or len(b) != offset + w * (-h) * 4:
        raise ValueError("unsupported BMP layout")
    return w, -h, b[offset:]


def compare(a, b):
    wa, ha, a = pixels(a)
    wb, hb, b = pixels(b)
    if (wa, ha) != (wb, hb):
        raise ValueError("parity image size mismatch")
    alpha_mask = 0 in a[3::4] or 0 in b[3::4]
    histogram = [0] * 256
    total = 0
    changed = 0
    intersection = union = 0
    luma_a = luma_b = 0
    for i in range(0, len(a), 4):
        ma = a[i+3]>0 if alpha_mask else any(a[i + c] != 9 for c in range(3))
        mb = b[i+3]>0 if alpha_mask else any(b[i + c] != 9 for c in range(3))
        intersection += ma and mb
        union += ma or mb
        for c in range(3):
            d = abs(a[i + c] - b[i + c])
            total += d
            histogram[d] += 1
            changed += d != 0
        luma_a += 0.0722 * a[i] + 0.7152 * a[i + 1] + 0.2126 * a[i + 2]
        luma_b += 0.0722 * b[i] + 0.7152 * b[i + 1] + 0.2126 * b[i + 2]
    n = wa * ha * 3
    count = 0
    p99 = 255
    for value, c in enumerate(histogram):
        count += c
        if count >= n * 0.99:
            p99 = value
            break
    metrics = {
        "rgb_mean_absolute": total / n,
        "rgb_p99_absolute": p99,
        "max_channel_delta": max(i for i, n in enumerate(histogram) if n),
        "changed_channels": changed,
        "silhouette_iou": intersection / max(1, union),
        "luminance_mean_delta": abs(luma_a - luma_b) / (wa * ha),
    }
    tolerance = json.loads(CONTRACT.read_text())["parity_v1"]
    metrics["pass"] = (
        metrics["rgb_mean_absolute"] <= tolerance["rgb_mean_absolute_max"]
        and p99 <= tolerance["rgb_p99_absolute_max"]
        and metrics["silhouette_iou"] >= tolerance["silhouette_iou_min"]
        and metrics["luminance_mean_delta"] <= tolerance["luminance_mean_delta_max"]
    )
    return metrics


def native_dispatch(relative_cwd,command):
    timeout=float(os.environ.get("C3X_LAB_NATIVE_TIMEOUT","120"))
    if not 1<=timeout<=300:raise ValueError("native timeout must be1..300seconds")
    proc=subprocess.Popen([sys.executable,str(V2/'app/native_dispatch.py'),relative_cwd,command],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    try:
        output,_=proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.communicate()
        return {"returncode":124,"status":"fail","output_tail":"approved native dispatcher exceeded bounded transport timeout"}
    if proc.returncode:return {"returncode":proc.returncode,"status":"fail","output_tail":output[-4000:]}
    return json.loads(output)


def d3d(report_path):

    root = V2 / "backends"
    build = root / "build"
    build.mkdir(exist_ok=True)
    identity = {
        relative(p): file_hash(p)
        for p in [
            root / "d3d11.cpp",
            root / "build_d3d11.bat",
            V2 / "contracts/packet_v1.h",
        V2 / "shared/content_hash.h",
        V2 / "shared/color_response.h",
        ]
    }
    record = build / "identity.json"
    exe = build / "d3d11.exe"
    if (
        not record.exists()
        or json.loads(record.read_text()).get("sources") != identity
        or not exe.is_file()
        or json.loads(record.read_text()).get("binary") != file_hash(exe)
    ):
        r = native_dispatch(
            "Renderer/terrain_lab/v2/backends", "call build_d3d11.bat"
        )
        if r["returncode"]:
            raise ValueError("D3D11 build failed")
        record.write_bytes(canonical({"sources": identity, "binary": file_hash(exe)}))

    def dispatch(command):
        for attempt in range(3):
            result = native_dispatch(".", command)
            if not result["returncode"]:
                return result
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
        raise ValueError("D3D11 dispatch/transport failed after three attempts")

    report = json.loads(report_path.read_text())
    results = []
    for row in report["outputs"]:
        source = local(row["packet"])
        target = local(row["image"]).with_suffix(".d3d11.bmp")
        settings=report['effective']['settings']
        if settings['mip_bias']!=0 or settings['anisotropy']!=8:
            raise ValueError('D3D parity currently supports anisotropy8/mip_bias0 only')
        from runner import shader_source
        post='box'
        if isinstance(settings['postprocess'],dict):
            if settings['postprocess']['contract']!=2:raise ValueError('D3D custom post requires contract2')
            dest=report_path.parent/'parity-post.hlsl';dest.write_text(shader_source(local(settings['postprocess']['shader'])));post=relative(dest)
        shaderpath=report_path.parent/'parity-source.hlsl'
        shaderpath.write_text(shader_source(local(report['effective']['module']['shader'])))
        args = [
            relative(exe),
            relative(source),
            relative(report_path.parent/"shaders") if report["effective"]["module"].get("provider")=="compose" and all(m.get("color_branch")=="q6_scene_linear_premultiplied_v1" for m in report["effective"]["module"]["modules"]) else relative(shaderpath),
            relative(target),str(settings["samples"]),post,
            str(row.get("offset",[0,0])[0]),str(row.get("offset",[0,0])[1]),str(settings["render_scale"]),
        ]
        command = " ".join('"' + a.replace("/", "\\") + '"' for a in args)
        result = dispatch(command)
        if result["returncode"]:
            raise ValueError("D3D11 replay failed")
        repeat = target.with_suffix(".repeat.bmp")
        args[3] = relative(repeat)
        result = dispatch(" ".join('"' + a.replace("/", "\\") + '"' for a in args))
        if file_hash(target) != file_hash(repeat):
            raise ValueError("D3D11 repeat differs")
        metrics = compare(local(row["image"]), target)
        results.append(
            {
                "metal": row["image"],
                "d3d11": relative(target),
                "d3d11_sha256": file_hash(target),
                "deterministic": True,
                "metrics": metrics,
            }
        )
    output = report_path.parent / "parity.json"
    output.write_bytes(
        canonical(
            {
                "schema": "c3x.lab_v2.parity.v1",
                "tolerance": json.loads(CONTRACT.read_text())["parity_v1"],
                "results": results,
                "pass": all(r["metrics"]["pass"] for r in results),
            }
        )
    )
    if not all(r["metrics"]["pass"] for r in results):
        raise ValueError("parity tolerance exceeded: " + relative(output))
    print("PASS Metal/D3D11 parity: " + relative(output))
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    args = p.parse_args()
    try:
        d3d(args.report.resolve())
    except (ValueError, OSError, KeyError) as e:
        print("parity: " + str(e), file=sys.stderr)
        raise SystemExit(1)
