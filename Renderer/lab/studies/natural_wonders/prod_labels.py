"""Renderer-drawn natural-wonder names (prod_tree.py merges these): the label pass after the
city-site overlay and before the fog (docs/natural_wonder_rendering.md, Labels)."""

PATCHES = {
    "Renderer/sandbox/fresh_pipeline.h": [
        ('#include "../native/gpu_city_site_overlay.h"\n',
         '#include "../native/gpu_city_site_overlay.h"\n#include "../native/gpu_wonder_labels.h"\n'),
        ("    c3x_renderer::GpuCitySiteOverlay city_site_overlay;\n",
         "    c3x_renderer::GpuCitySiteOverlay city_site_overlay;\n    c3x_renderer::GpuWonderLabels wonder_labels;\n")],
    "Renderer/sandbox/resident_scene.cpp": [
        ("    if(!presented && !failed_step)failed_step=\"site_overlay\";\n",
         "    if(!presented && !failed_step)failed_step=\"site_overlay\";\n"
         "    // Natural-wonder names: map text under the fog, as Civ III draws its fog last.\n"
         "    // A failed label pass leaves the frame without names rather than unpublished.\n"
         "    if(presented){\n"
         "        Microsoft::WRL::ComPtr<ID3D11Resource> resource;\n"
         "        Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;\n"
         "        target->GetResource(&resource);\n"
         "        if(FAILED(resource.As(&texture)) ||\n"
         "           !sandbox_fresh.wonder_labels.draw(renderer.device,renderer.context,texture.Get(),frame,zoom))\n"
         "            renderer.trace.write(\"wonder-labels\",\"label pass failed\",true);\n"
         "    }\n")],
}
