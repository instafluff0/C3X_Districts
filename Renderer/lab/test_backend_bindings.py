"""Portable packet validation; the matching HLSL also runs on both real GPUs."""
from pathlib import Path
import subprocess
import tempfile
import unittest
import shutil
import struct
from Renderer.lab.backends.packet_store import compact_packet

ROOT = Path(__file__).resolve().parents[2]


class Bindings(unittest.TestCase):
    def test_array_and_extended_buffer_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            binary=Path(directory)/"bindings"
            subprocess.run(["clang++", "-std=c++17", "-O2", str(ROOT/"Renderer/lab/backends/test_bindings.cpp"),
                            "-o", str(binary)], check=True, capture_output=True)
            result=subprocess.run([str(binary), str(Path(directory)/"test.packet")],
                                  check=True, capture_output=True, text=True)
            self.assertIn("PASS array mip layout, b2/b7 transport", result.stdout)
            packet=Path(directory)/"test.packet"
            compact_packet(packet,Path(directory)/"content")
            subprocess.run([str(binary),"--read",str(packet)],check=True,capture_output=True)
            blob=next(Path(str(packet)+".blobs").iterdir())
            blob.write_bytes(b"corrupt")
            rejected=subprocess.run([str(binary),"--read",str(packet)],capture_output=True,text=True)
            self.assertNotEqual(rejected.returncode,0)
            self.assertIn("corrupt packet content",rejected.stderr)


def gpu_check(*, windows=False):
    """Opt-in actual GPU transport check; never a production-scene approval."""
    from Renderer.lab.backends.cache import Cache
    from Renderer.lab.backends.compiler import executable,run,shaders
    from Renderer import renderer
    lab=ROOT/"Renderer/lab";out=lab/"out/backend-bindings"
    out.mkdir(parents=True,exist_ok=True)
    receipt=out/"result.json"
    renderer.write(receipt,{"status":"running"})
    try:
        cache=Cache(lab/".cache/backend")
        builder=executable(cache,lab/"backends/test_bindings.cpp")
        packet=out/"test.packet"
        run([builder,packet])
        metal=executable(cache,lab/"backends/metal.mm",objc=True,
                         libraries=("-framework","Metal","-framework","Foundation"))
        shader_dir=out/"shaders";shader_dir.mkdir(exist_ok=True)
        for entry,path in shaders(cache,lab/"backends/test_bindings.hlsl",0).items():
            shutil.copyfile(path,shader_dir/(entry+".msl"))
        run([metal,packet,shader_dir,out/"metal.bmp",out/"metal.cost.json"])
        raw=(out/"metal.bmp.linear.rgba16f").read_bytes()
        if len(raw)!=64*64*8:raise ValueError("Unexpected linear attachment size")
        for x in (1,7,8,32,62):
            color=struct.unpack_from("<4e",raw,(32*64+x)*8)
            expected=(0,0,0,0) if x<8 else (128/255,.25,.75,1)
            if any(abs(a-b)>.0003 for a,b in zip(color,expected)):
                raise ValueError("Vertex translation, array mip or constant-buffer result differs")
        if windows:
            from Renderer.lab.platform import native_command_result
            result=native_command_result("Renderer/lab/backends",
                r"call build_d3d11.bat && ..\.cache\d3d11.exe ..\out\backend-bindings\test.packet test_bindings.hlsl ..\out\backend-bindings\d3d11.bmp")
            if result["status"]!="pass":raise ValueError("D3D11 binding witness failed")
            if (out/"d3d11.bmp.linear.rgba16f").read_bytes()!=raw:
                raise ValueError("Metal/D3D11 linear binding results differ")
            if (out/"metal.bmp").read_bytes()!=(out/"d3d11.bmp").read_bytes():
                raise ValueError("Metal/D3D11 final binding images differ")
        from Renderer.lab.backends.production import prepare
        production=prepare(cache,lab/"out/production-shaders",metal=metal)
        renderer.write(receipt,{"status":"pass","metal":True,"d3d11":windows,
            "checks":["vertex b1 translation","pixel b0/b2/b7","texture array layer 1 mip 1"],
            "production_shader_compile":sorted(production),
            "production_scene_parity":False})
        print("PASS backend bindings"+("; Metal/D3D11 pixels are identical" if windows else " on Metal"))
    except Exception as error:
        renderer.write(receipt,{"status":"fail","reason":str(error)})
        raise


if __name__ == "__main__":
    unittest.main()
