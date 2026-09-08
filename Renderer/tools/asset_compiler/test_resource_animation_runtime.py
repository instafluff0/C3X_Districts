import tempfile
import unittest
import hashlib
import json
import struct
from pathlib import Path
from unittest.mock import patch

from Renderer.tools.asset_compiler import normalized_animation as animation
from Renderer.tools.asset_compiler.build_resource_animation_runtime import calibrated_resource_poses, clip_translation_scales, build
from Renderer.tools.asset_compiler import build_resource_runtime as static


class ResourceAnimationCalibrationTests(unittest.TestCase):
    def test_current_clip_recipe_preserves_translation_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);clip=root/'motion.c3anim';clip.write_bytes(b'normalized clip bytes')
            recipe=root/'units.json';recipe.write_text(json.dumps({'schema':'c3x.resource_clip_units.v1',
                'clips':[{'path':'motion.c3anim','translation_scale':1/12}]}))
            consumed=[]
            def read(path):consumed.append(path);return path.read_bytes()
            result=clip_translation_scales(root,recipe,read_bytes=read)
            self.assertEqual(result,{hashlib.sha256(clip.read_bytes()).hexdigest():1/12})
            self.assertEqual(consumed,[recipe,clip.resolve()])

    def test_clip_recipe_rejects_invalid_units_and_path_escapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);recipe=root/'units.json';(root/'clip').write_bytes(b'clip')
            for value in (0,-1,2,True,None,float('inf')):
                recipe.write_text(json.dumps({'schema':'c3x.resource_clip_units.v1',
                    'clips':[{'path':'clip','translation_scale':value}]}))
                with self.assertRaisesRegex(ValueError,'translation scale'):
                    clip_translation_scales(root,recipe)
            recipe.write_text(json.dumps({'schema':'c3x.resource_clip_units.v1',
                'clips':[{'path':'../outside','translation_scale':.1}]}))
            with self.assertRaisesRegex(ValueError,'escapes'):
                clip_translation_scales(root,recipe)

    def test_identical_clip_bytes_cannot_have_conflicting_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);recipe=root/'units.json'
            for name in ('a','b'):(root/name).write_bytes(b'same clip')
            recipe.write_text(json.dumps({'schema':'c3x.resource_clip_units.v1','clips':[
                {'path':'a','translation_scale':.1},{'path':'b','translation_scale':.2}]}))
            with self.assertRaisesRegex(ValueError,'Conflicting units'):
                clip_translation_scales(root,recipe)

    def test_animation_output_cannot_overlap_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for output in (root,root/'animated',root/'animated/child'):
                with self.assertRaisesRegex(ValueError,'overlap'):
                    build(root/'animated',root/'landmarks',output)
            self.assertEqual(list(root.iterdir()),[])

    def test_translation_units_root_placement_and_missing_tracks(self):
        def bone(name, parent, position):
            return {"name": name, "parent": parent, "local": {
                "position": position, "orientation": [0, 0, 0, 1],
                "scale_shear": [1, 0, 0, 0, 1, 0, 0, 0, 1]}}
        skeleton = {"bones": [bone("root", -1, [.5, 0, 0]),
                              bone("animated_child", 0, [0, .2, 0]),
                              bone("untracked_child", 0, [0, 0, .3])]}
        def track(name, positions):
            return animation.TransformTrack(name, 0,
                animation.Channel(animation.SAMPLED, 3, positions),
                animation.Channel(animation.IDENTITY, 4, ()),
                animation.Channel(animation.IDENTITY, 9, ()))
        clip = animation.AnimationClip(1, 1, 2, (animation.TrackGroup("body", (
            track("root", (30, 0, 0, 40, 0, 0)),
            track("animated_child", (0, 2, 0, 0, 2, 0)))),))
        with tempfile.TemporaryDirectory() as directory:
            cache = calibrated_resource_poses(skeleton, clip, 0, .1, Path(directory)/"pose.bin")
        start, end = cache.sample(0, False), cache.sample(1, False)
        self.assertAlmostEqual(start[0][12], .5)  # Preserve normalized anchor, not source-scene x=30.
        self.assertAlmostEqual(end[0][12], 1.5)  # Preserve authored root motion after unit conversion.
        for pose in (start, end):
            self.assertAlmostEqual(pose[1][13], .2)  # Authored child uses converted units.
            self.assertAlmostEqual(pose[2][14], .3)  # Missing track already uses normalized rest units.


class StaticResourcePreparation(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.pack=self.root/'source';self.pack.mkdir()
        self.mesh={'vertices':[{'position':[x,y,0],'normal':[0,0,1],'uv0':[x,y]}
                               for x,y in ((0,0),(1,0),(0,1))],'topology':{'indices':[0,1,2]}}
        (self.pack/'manifest.json').write_text(json.dumps({'assets':{'fish/body':{'mesh':'mesh.json','material':'material.json'}}}))
        (self.pack/'mesh.json').write_text(json.dumps(self.mesh))
        (self.pack/'material.json').write_text(json.dumps({'base_color':{'texture':'color.dds'}}))
        (self.pack/'color.dds').write_bytes(b'unchanged mip chain')
        replacement=patch.object(static,'SELECTIONS',(('fish','fish/body',1.75,1),))
        replacement.start();self.addCleanup(replacement.stop)

    def test_disposable_bundle_preserves_fields_and_surface_offset(self):
        target=self.root/'candidate/bundle.bin'
        (self.pack/'resource_runtime.bin').write_bytes(b'prior runtime')
        before=(self.pack/'mesh.json').read_bytes()
        static.build(self.pack,target)
        def string(value):return struct.pack('<I',len(value))+value.encode()
        expected=b'C3XVEG1\0'+struct.pack('<4I',1,1,1,1)+string('color.dds')+string('fish/body')+struct.pack('<3I',0,3,3)
        for v in self.mesh['vertices']:
            expected+=struct.pack('<8f',*v['position'][:2],.060,*v['normal'],*v['uv0'])
        expected+=struct.pack('<3I',0,1,2)+string('fish')+struct.pack('<I',1)+struct.pack('<IffIIIIff',0,1.75,.03,1,1,5,0,0,0)
        self.assertEqual(target.read_bytes(),expected)
        self.assertEqual((self.pack/'mesh.json').read_bytes(),before)
        self.assertEqual((self.pack/'resource_runtime.bin').read_bytes(),b'prior runtime')

    def test_input_file_cannot_be_selected_as_output(self):
        target=self.pack/'manifest.json';before=target.read_bytes()
        with self.assertRaisesRegex(ValueError,'overwrite source'):
            static.build(self.pack,target)
        self.assertEqual(target.read_bytes(),before)

    def test_texture_path_cannot_escape_the_pack(self):
        (self.pack/'material.json').write_text(json.dumps({'base_color':{'texture':'../outside.dds'}}))
        with self.assertRaisesRegex(ValueError,'escapes'):
            static.build(self.pack,self.root/'candidate.bin')


if __name__ == "__main__":
    unittest.main()
