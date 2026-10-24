"""C3X side of the natural-wonder integration (prod_tree.py merges these): facts only.

C3X is much more than Renderer64, so injected code stays minimal and keeps no
renderer logic (the user, 2026-10-23):
- the tile record reports the wonder's config name and C3X's show-name setting (API 23);
- the native draw_map_tile_text label is skipped while the renderer DLL is loaded
  (C3X keeps the module only when its API version matches, so it draws the names).
Everything else (kits, relief slots, bodies, effects, labels) lives in Renderer64.
No C3X.h state and no new civ_prog_objects.csv entry.
"""

RECORD = """				strncpy (record->resource_name, resource->Name, sizeof record->resource_name);
				record->resource_name[(sizeof record->resource_name) - 1] = '\\0';
			}
			// API 23: the natural wonder here, by its config name (Renderer64 draws its kit and name).
			struct district_instance * wonder = is->current_config.enable_natural_wonders ? get_district_instance (tile) : NULL;
			struct natural_wonder_district_config const * nw = ((wonder != NULL) && (wonder->district_id == NATURAL_WONDER_DISTRICT_ID)) ?
				get_natural_wonder_config_by_id (wonder->natural_wonder_info.natural_wonder_id) : NULL;
			if ((nw != NULL) && (nw->name != NULL)) {
				strncpy (record->natural_wonder, nw->name, sizeof record->natural_wonder);
				record->natural_wonder[(sizeof record->natural_wonder) - 1] = '\\0';
				record->natural_wonder_flags = is->current_config.show_natural_wonder_name_on_map ? C3X_RENDERER_NATURAL_WONDER_NAME : 0;
			}
"""

LABEL = """	// Renderer64 draws natural-wonder names itself, under its fog.
	bool draw_natural_wonders = is->current_config.enable_natural_wonders &&
	                            is->current_config.show_natural_wonder_name_on_map &&
	                            ! (is->current_config.enable_custom_rendering && (is->custom_renderer_module != NULL));
"""

PATCHES = {
    "injected_code.c": [
        ("				strncpy (record->resource_name, resource->Name, sizeof record->resource_name);\n"
         "				record->resource_name[(sizeof record->resource_name) - 1] = '\\0';\n			}\n", RECORD),
        ("	bool draw_natural_wonders = is->current_config.enable_natural_wonders &&\n"
         "	                            is->current_config.show_natural_wonder_name_on_map;\n", LABEL)],
}
