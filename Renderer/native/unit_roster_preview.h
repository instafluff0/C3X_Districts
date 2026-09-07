// Headless verification only: explicit keys/actions, the actual DLL body export,
// both zooms and four facing directions. No game process or input is involved.
bool preview_unit_roster(HMODULE module,char const* picture_path,int hour) {
    char cases_path[4*MAX_PATH]={};
    if(!GetEnvironmentVariableA("C3X_RENDERER_UNIT_CASES",cases_path,sizeof(cases_path)))return true;
    FILE* file=nullptr;if(fopen_s(&file,cases_path,"r") || !file)return false;
    auto draw=reinterpret_cast<c3x_renderer_unit_draw_fn>(GetProcAddress(module,"c3x_renderer_unit_draw"));
    BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);info.bmiHeader.biWidth=1024;
    info.bmiHeader.biHeight=-1152;info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;
    HDC dc=CreateCompatibleDC(nullptr);void* bits=nullptr;
    HBITMAP bitmap=CreateDIBSection(dc,&info,DIB_RGB_COLORS,&bits,nullptr,0);
    if(!dc || !bitmap || !draw){if(bitmap)DeleteObject(bitmap);if(dc)DeleteDC(dc);std::fclose(file);return false;}
    auto old=SelectObject(dc,bitmap);unsigned draws=0,failures=0,rows=0;char key[64]={};unsigned mask=0;
    std::vector<std::uint32_t> sheet(1024*1152,0xff565b62u);
    while(fscanf_s(file,"%63s %u",key,unsigned(sizeof(key)),&mask)==2) {
        c3x_renderer_unit_v1 unit={};unit.struct_size=sizeof(unit);strcpy_s(unit.unit_key,key);unit.unit_id=int(rows);
        unit.sprite_width=unit.sprite_height=191;unit.frame_count=16;unit.hour=hour;unit.display_color_rgb=0x20a6b0;
        for(int zoom=0;zoom<2;++zoom)for(int action=1;action<=18;++action)if(mask&(1u<<action))
            for(int direction=1;direction<=8;direction+=2)for(int cursor:{0,7,15}) {
                std::fill_n(static_cast<std::uint32_t*>(bits),1024*1152,0xff565b62u);
                unit.reduced=zoom;unit.action=action;unit.direction=direction;unit.action_cursor=cursor;
                unit.body_x=unit.body_y=100;
                int result=draw(&unit,dc);GdiFlush();++draws;
                if(result!=C3X_RENDERER_RESULT_OK) {
                    ++failures;
                    if(failures<100)std::printf("FAIL ROSTER key=%s action=%d direction=%d cursor=%d zoom=%d result=%d\n",key,action,direction,cursor,zoom,result);
                }
                if(zoom==0 && action<=9 && direction==3 && cursor==7) {
                    unsigned column=unsigned(action-1),row=rows%8;
                    // Native anchor at 195,195. Preserve a 112px body/shadow crop.
                    for(unsigned y=0;y<112;++y)for(unsigned x=0;x<112;++x)
                        sheet[(row*144+y+24)*1024+column*112+x]=static_cast<std::uint32_t*>(bits)[(y+139)*1024+x+139];
                }
            }
        ++rows;
        if(rows%8==0) {
            c3x_renderer_output_v1 output={};output.width=1024;output.height=1152;output.stride_bytes=4096;output.bgra_pixels=sheet.data();
            std::string path=std::string(picture_path)+".roster-"+std::to_string((rows-1)/8)+".bmp";
            if(hour==12 && !write_bmp(path.c_str(),output))++failures;
            std::fill(sheet.begin(),sheet.end(),0xff565b62u);
        }
        std::printf("ROSTER checked key=%s cumulative_draws=%u failures=%u\n",key,draws,failures);
    }
    if(rows%8) {
        c3x_renderer_output_v1 output={};output.width=1024;output.height=1152;output.stride_bytes=4096;output.bgra_pixels=sheet.data();
        std::string path=std::string(picture_path)+".roster-"+std::to_string((rows-1)/8)+".bmp";
        if(hour==12 && !write_bmp(path.c_str(),output))++failures;
    }
    std::fclose(file);SelectObject(dc,old);DeleteObject(bitmap);DeleteDC(dc);
    std::printf("ROSTER complete keys=%u draws=%u failures=%u status=%s\n",rows,draws,failures,rows && !failures?"pass":"FAIL");
    return rows && !failures;
}
