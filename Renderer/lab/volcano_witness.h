// Included only by the standalone Lab harness, with the actual production API.
char volcano_study[8]={};
if(ok && pickup && GetEnvironmentVariableA("C3X_LAB_VOLCANO_STUDY",volcano_study,sizeof(volcano_study))) {
    auto verify_volcano=[&]() {
        auto saved_tiles=tiles;auto saved_world=world;
        auto pixels=[&](){auto p=static_cast<unsigned char const*>(output.bgra_pixels);
            return std::vector<unsigned char>(p,p+output.stride_bytes*output.height);};
        auto draw=[&](){return render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0;};
        auto cold=[&](){reset();return set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK && draw();};
        auto original=pixels();
        // Mutate the same captured topology and occurrences used in the game.
        auto set_terrain=[&](int x,int y,int real) {
            auto& value=world[(std::size_t(y)*map_width+x)/2];
            value=(value&~0xff00u)|(unsigned(real)<<8);
            for(auto& tile:tiles)if(((tile.tile_x%map_width)+map_width)%map_width==x && tile.tile_y==y)
                tile.real_terrain_type=real;
            ++frame.world_topology_revision;
        };
        auto volcano=std::find_if(tiles.begin(),tiles.end(),[](auto const&t){return t.real_terrain_type==10;});
        if(volcano==tiles.end())return false;
        int x=((volcano->tile_x%map_width)+map_width)%map_width,y=volcano->tile_y;
        set_terrain(x,y,volcano->terrain_type);
        if(!draw() || !output.geometry_tiles_built || !output.geometry_tiles_reused)return false;
        auto removed=pixels();
        if(removed==original || !cold() || pixels()!=removed)return false;
        std::printf("PASS volcano removal and exact cold parity\n");
        // Add a different captured placement: no fixed Lab-center material mask.
        int other_x=(x+2)%map_width;
        set_terrain(other_x,y,10);
        if(!draw())return false;
        auto moved=pixels();
        if(moved==removed || moved==original || !cold() || pixels()!=moved)return false;
        std::printf("PASS volcano placement and exact cold parity\n");
        set_terrain(x,y,10);
        if(!draw())return false;
        auto multiple=pixels();
        if(multiple==moved || !cold() || pixels()!=multiple)return false;
        std::printf("PASS multiple volcanoes and exact cold parity\n");
        // Restore in place so the frame's topology/occurrence pointers stay valid.
        std::copy(saved_world.begin(),saved_world.end(),world.begin());
        std::copy(saved_tiles.begin(),saved_tiles.end(),tiles.begin());
        ++frame.world_topology_revision;
        if(!draw() || pixels()!=original)return false;
        if(!draw() || output.geometry_tiles_built || output.geometry_upload_bytes || pixels()!=original)return false;
        std::printf("PASS volcano reappearance and cached repeat\n");
        int original_x=center_x,original_y=center_y;
        for(int shift:{4,map_width}) {
            center_x=original_x+shift;tiles=capture_view();
            frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
            if(!draw())return false;
            auto translated=pixels();
            if(!cold() || pixels()!=translated)return false;
        }
        center_x=original_x;center_y=original_y;tiles=capture_view();
        frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
        if(!draw() || pixels()!=original)return false;
        std::printf("PASS volcano scroll and wrapped occurrence cold parity\n");
        return true;
    };
    ok=verify_volcano();
    std::printf("%s volcano lifecycle\n",ok?"PASS":"FAIL");
}
