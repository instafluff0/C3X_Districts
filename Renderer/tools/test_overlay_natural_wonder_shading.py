"""The natural-wonder overlay inserts only its regions into a runtime pack shader, once."""
import unittest

from Renderer.tools import overlay_natural_wonder_shading as overlay

PACK = ("float4 q8_effect_plume(float2 q,float4 shape,float3 look){\n  clip(alpha-.004);\n  float grey=0;\n}\n"
        "/*other overlay*/\nfloat4 q8_city_effect(FeaturePixelInput p){\n"
        " float2 pixel=float2(max(abs(ddx(p.uv.x)),1e-5),max(abs(ddy(p.uv.y)),1e-5));\n"
        " if(kind>2.5){\n  // A volcano's plume (93)\n }\n}\n")
EFFECTS = ("// Natural wonders (docs/natural_wonder_rendering.md): a geyser (94).\nfloat4 q8_effect_geyser(){\n"
           "  clip(result.a-.004);\n  return result;\n}\n// Natural wonders: spray (99)\nfloat4 q8_effect_spray(){\n"
           "  clip(result.a-.004);\n  return result;\n}\n")
DISPATCH = (" if(kind>3.5 && kind<4.5)return q8_effect_geyser(q,pixel,t,seed,strength,night);\n"
            " if(kind>8.5 && kind<9.5)return q8_effect_spray(q,pixel,t,seed,strength,night);\n")
SOURCE = ("float4 q8_effect_plume(float2 q,float4 shape,float3 look,bool keep=false){\n  if(!keep)clip(alpha-.004);\n"
          "  float grey=0;\n}\n" + EFFECTS + "float4 q8_city_effect(FeaturePixelInput p){\n"
          " float2 pixel=float2(max(length(float2(ddx(p.uv.x),ddy(p.uv.x))),1e-5),\n"
          "                     max(length(float2(ddx(p.uv.y),ddy(p.uv.y))),1e-5));\n"
          + DISPATCH + " if(kind>2.5){\n  // A volcano's plume (93)\n }\n}\n")


class NaturalWonderOverlayTests(unittest.TestCase):
    def test_regions_insert_once_and_keep_other_bytes(self):
        result, applied = overlay.wonder_overlay(PACK, SOURCE)
        self.assertEqual(set(applied), {"plume_keep", "plume_clip", "effect_pixel", "wonder_effects", "wonder_dispatch"})
        self.assertIn(EFFECTS + "float4 q8_city_effect", result)          # every effect function, up to its anchor
        self.assertIn(DISPATCH + " if(kind>2.5){", result)
        self.assertIn("/*other overlay*/", result)
        again, more = overlay.wonder_overlay(result, SOURCE)
        self.assertEqual((again, more), (result, []))                      # idempotent

    def test_water_shadow_region_inserts_before_the_return(self):
        tail = ("  return visibility;\n }\n#endif\n return legacy_shadow;\n}\n"
                "float3 q6_receiver_illumination(PixelInput input,float3 normal,\n")
        pack = "  float visibility=q6_world_visibility(pickup_shadow_terrain,input.q6_world,normal,water);\n" + tail
        region = ("  // Natural wonders (docs/natural_wonder_rendering.md): a wonder's lake or fjord channel\n"
                  "  // (shape_visibility.x 2) keeps a third.\n"
                  "  if(water && input.shape_visibility.x>1.5)visibility=lerp(1.0,visibility,.35);\n")
        source = "  float visibility=q6_world_visibility(pickup_shadow_terrain,input.q6_world,normal,water);\n" + region + tail
        result, applied = overlay.wonder_overlay(pack, source)
        self.assertEqual(applied, ["wonder_water_shadow"])
        self.assertEqual(result, source)
        self.assertEqual(overlay.wonder_overlay(result, source), (result, []))

    def test_missing_source_regions_change_nothing(self):
        self.assertEqual(overlay.wonder_overlay(PACK, PACK), (PACK, []))


if __name__ == "__main__":
    unittest.main()
