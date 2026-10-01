#pragma once

// Private standalone workload only. It feeds the existing admitted body/action
// owner and draw_real path; it never observes or changes an actual game session.
// Include after resident_scene's existing direct_units/fresh_pipeline headers.
struct SandboxZoomPreviewBusyMetrics {
    unsigned admitted=0,visible=0,travelling=0,parts=0,legs=0,updates=0;
    bool complete=false;
};
struct SandboxZoomPreviewBusyUnits {
    struct Actor {
        c3x_renderer_unit_v1 body{};
        c3x_renderer_unit_state_v1 state{};
        int ax=0,ay=0,bx=0,by=0;
        c3x_renderer_i64 last_leg=-1;
        bool at_b=false,admitted=false;
    };
    c3x_renderer::render_core::UnitInstances owner{16};
    Actor actors[16]{};
    SandboxZoomPreviewBusyMetrics metrics{};
    bool attempted=false;
    int world_width=0,world_height=0;
    unsigned wrap_x=0,wrap_y=0;
    std::size_t catalog_size=0;
    int canonical(int value,int extent,bool wrap)const {
        return wrap?(value%extent+extent)%extent:value;
    }
    bool land(c3x_renderer_tile_v1 const& tile)const {
        unsigned required=C3X_RENDERER_TILE_VISIBLE|C3X_RENDERER_TILE_RENDER;
        return (tile.tile_flags&required)==required&&tile.terrain_type>=0&&
            tile.terrain_type<11&&tile.real_terrain_type>=0&&tile.real_terrain_type<11;
    }
    bool used(int x,int y)const {
        for(auto const& actor:actors)if(actor.admitted&&
            ((actor.ax==x&&actor.ay==y)||(actor.bx==x&&actor.by==y)))return true;
        return false;
    }
    c3x_renderer_tile_v1 const* tile_at(c3x_renderer_frame_v1 const& frame,int x,int y)const {
        for(unsigned n=0;n<frame.tile_count;++n){auto const& tile=frame.tiles[n];
            if(land(tile)&&canonical(tile.tile_x,world_width,wrap_x!=0)==x&&
                canonical(tile.tile_y,world_height,wrap_y!=0)==y)return &tile;
        }return nullptr;
    }
    bool initialize(c3x_renderer_frame_v1 const& frame){
        attempted=true;world_width=frame.world_width_tiles;world_height=frame.world_height_tiles;
        wrap_x=frame.world_wrap_x;wrap_y=frame.world_wrap_y;
        auto const& catalog=renderer.unit_bodies.units;catalog_size=catalog.size();
        if(catalog.empty()){
            std::printf("ZOOM_BUSY_GAP reason=empty-unit-catalog prerequisite=unit-rendering-enabled-before-pack-load\n");return false;
        }
        char const* keys[]={"PRTO_Warrior","PRTO_Archer","PRTO_Spearman"};
        std::vector<char const*> available;
        for(auto key:keys){
            auto unit=std::find_if(catalog.begin(),catalog.end(),[&](auto const& item){
                return std::find(item.keys.begin(),item.keys.end(),key)!=item.keys.end();});
            if(unit==catalog.end()){std::printf("ZOOM_BUSY_GAP reason=missing-body key=%s\n",key);continue;}
            auto idle=std::find_if(unit->actions.begin(),unit->actions.end(),[](auto const& action){return action.name=="idle";});
            auto move=std::find_if(unit->actions.begin(),unit->actions.end(),[](auto const& action){return action.name=="move";});
            if(idle==unit->actions.end()||move==unit->actions.end()||idle->parts.empty()||
                move->parts.empty()||move->duration<=0){
                std::printf("ZOOM_BUSY_GAP reason=missing-idle-or-timed-move-parts key=%s\n",key);continue;
            }
            available.push_back(key);
        }
        if(available.empty())return false;
        std::vector<c3x_renderer_tile_v1 const*> candidates;
        for(unsigned n=0;n<frame.tile_count;++n){auto const& tile=frame.tiles[n];
            int x=tile.anchor_x+frame.tile_width/2,y=tile.anchor_y+frame.tile_height/2;
            if(land(tile)&&x>=128&&x<frame.target_width-128&&y>=96&&y<frame.target_height-96)candidates.push_back(&tile);
        }
        std::stable_sort(candidates.begin(),candidates.end(),[&](auto a,auto b){
            return std::abs(a->anchor_x+64-frame.target_width/2)+std::abs(a->anchor_y+32-frame.target_height/2)<
                std::abs(b->anchor_x+64-frame.target_width/2)+std::abs(b->anchor_y+32-frame.target_height/2);});
        int const neighbors[8][2]={{1,1},{1,-1},{-1,1},{-1,-1},{2,0},{-2,0},{0,2},{0,-2}};
        for(auto start:candidates){
            if(metrics.admitted==16)break;
            int ax=canonical(start->tile_x,world_width,wrap_x!=0),ay=canonical(start->tile_y,world_height,wrap_y!=0);
            if(ax<0||ay<0||ax>=world_width||ay>=world_height||((ax+ay)&1)||used(ax,ay))continue;
            c3x_renderer_tile_v1 const* end=nullptr;int bx=0,by=0;
            for(auto const& offset:neighbors){
                bx=canonical(ax+offset[0],world_width,wrap_x!=0);by=canonical(ay+offset[1],world_height,wrap_y!=0);
                if(bx<0||by<0||bx>=world_width||by>=world_height||used(bx,by))continue;
                auto candidate=tile_at(frame,bx,by);
                if(candidate){int x=candidate->anchor_x+64,y=candidate->anchor_y+32;
                    if(x>=128&&x<frame.target_width-128&&y>=96&&y<frame.target_height-96){end=candidate;break;}}
            }
            if(!end)continue;
            auto& actor=actors[metrics.admitted];actor.ax=ax;actor.ay=ay;actor.bx=bx;actor.by=by;
            auto& state=actor.state;state.struct_size=sizeof(state);state.kind=C3X_RENDERER_UNIT_STATE_OBSERVE;
            state.unit_id=100000+int(metrics.admitted);state.tile_x=ax;state.tile_y=ay;
            state.unit_type_id=int(metrics.admitted%available.size());state.owner_id=0;
            state.action=1;state.max_hp=3;state.visible=1;
            state.presentation_time_ticks=frame.presentation_time_ticks;state.presentation_frequency=frame.presentation_frequency;
            auto& body=actor.body;body.struct_size=sizeof(body);body.unit_id=state.unit_id;
            body.action=1;body.direction=1+int(metrics.admitted%8);body.frame_count=16;
            body.action_cursor=int(metrics.admitted*3%16);body.sprite_width=body.sprite_height=191;
            body.body_x=start->anchor_x+64-95;body.body_y=start->anchor_y+32-95;
            body.projection_scale_milli=1000;body.hour=frame.hour;body.season=frame.season;
            body.display_color_rgb=0x205bdd;body.presentation_time_ticks=frame.presentation_time_ticks;
            body.presentation_frequency=frame.presentation_frequency;strcpy_s(body.unit_key,available[metrics.admitted%available.size()]);
            c3x_renderer::render_core::UnitInstances::Selection selected;
            if(!owner.state(state)||!owner.capture(body,C3X_RENDERER_UNIT_STATE_CAPTURED,catalog,c3x_renderer::native_unit_action,selected)){
                owner.forget(state.unit_id);std::printf("ZOOM_BUSY_GAP reason=body-state-admission key=%s tile=%d,%d\n",body.unit_key,ax,ay);continue;
            }
            actor.admitted=true;++metrics.admitted;
        }
        if(metrics.admitted<16)std::printf("ZOOM_BUSY_GAP reason=distinct-visible-land-pairs requested=16 admitted=%u\n",metrics.admitted);
        metrics.complete=metrics.admitted==16;
        return metrics.admitted!=0;
    }
    bool update(c3x_renderer_frame_v1 const& frame){
        if(!frame.tiles||!frame.tile_count||frame.tile_width!=128||frame.tile_height!=64||
            frame.world_width_tiles<=0||frame.world_height_tiles<=0||frame.presentation_frequency<=0||frame.presentation_time_ticks<0){
            std::printf("ZOOM_BUSY_GAP reason=invalid-native128-frame\n");renderer.fresh_unit_poses.clear();metrics.complete=false;return false;
        }
        bool first=!attempted;
        if(first&&!initialize(frame)){renderer.fresh_unit_poses.clear();return false;}
        if(frame.world_width_tiles!=world_width||frame.world_height_tiles!=world_height||
            frame.world_wrap_x!=wrap_x||frame.world_wrap_y!=wrap_y||renderer.unit_bodies.units.size()!=catalog_size){
            std::printf("ZOOM_BUSY_GAP reason=changed-world-or-catalog\n");renderer.fresh_unit_poses.clear();metrics.complete=false;return false;
        }
        auto const& catalog=renderer.unit_bodies.units;
        auto poses=owner.scene_poses(frame,frame.presentation_time_ticks,frame.presentation_frequency,catalog);
        bool changed=false;
        for(unsigned n=0;n<16;++n){auto& actor=actors[n];if(!actor.admitted)continue;
            if(!tile_at(frame,actor.ax,actor.ay)||!tile_at(frame,actor.bx,actor.by)){
                owner.forget(actor.body.unit_id);actor.admitted=false;metrics.complete=false;changed=true;
                std::printf("ZOOM_BUSY_GAP reason=movement-pair-no-longer-visible-land id=%d\n",actor.body.unit_id);continue;
            }
            double elapsed=actor.last_leg<0?1.:double(frame.presentation_time_ticks-actor.last_leg)/double(frame.presentation_frequency);
            if(owner.motion_count(actor.body.unit_id)||elapsed<.8+.037*(n%6))continue;
            c3x_renderer_unit_move_v1 event{};event.struct_size=sizeof(event);event.unit_id=actor.body.unit_id;
            event.old_x=actor.at_b?actor.bx:actor.ax;event.old_y=actor.at_b?actor.by:actor.ay;
            event.new_x=actor.at_b?actor.ax:actor.bx;event.new_y=actor.at_b?actor.ay:actor.by;
            event.action=2;event.source_visible=event.target_visible=1;
            event.presentation_time_ticks=frame.presentation_time_ticks;event.presentation_frequency=frame.presentation_frequency;
            auto next=actor.state;next.tile_x=event.new_x;next.tile_y=event.new_y;next.action=2;
            next.presentation_time_ticks=frame.presentation_time_ticks;next.presentation_frequency=frame.presentation_frequency;
            if(!owner.begin_motion(event,world_width,world_height,wrap_x!=0,wrap_y!=0)||!owner.move(event)||!owner.state(next)){
                owner.forget(actor.body.unit_id);actor.admitted=false;metrics.complete=false;changed=true;
                std::printf("ZOOM_BUSY_GAP reason=adjacent-motion-admission id=%d\n",actor.body.unit_id);continue;
            }
            actor.state=next;actor.at_b=!actor.at_b;actor.last_leg=frame.presentation_time_ticks;
            ++metrics.legs;changed=true;
        }
        if(changed)poses=owner.scene_poses(frame,frame.presentation_time_ticks,frame.presentation_frequency,catalog);
        renderer.fresh_unit_poses=std::move(poses);metrics.visible=unsigned(renderer.fresh_unit_poses.size());
        metrics.travelling=metrics.parts=0;++metrics.updates;
        for(auto const& pose:renderer.fresh_unit_poses){metrics.travelling+=unsigned(pose.travelling);
            if(pose.unit<catalog.size()&&pose.action<catalog[pose.unit].actions.size())
                metrics.parts+=unsigned(catalog[pose.unit].actions[pose.action].parts.size());
        }
        if(first)report();return metrics.complete;
    }
    void report()const {
        std::printf("ZOOM_BUSY actors_admitted=%u visible_poses=%u travelling_poses=%u admitted_parts=%u legs=%u updates=%u complete_16=%u source_ticks_from_full_draw=1\n",
            metrics.admitted,metrics.visible,metrics.travelling,metrics.parts,metrics.legs,metrics.updates,unsigned(metrics.complete));
    }
};
SandboxZoomPreviewBusyUnits& sandbox_zoom_preview_busy_state(){
    static SandboxZoomPreviewBusyUnits state;return state;
}
bool sandbox_zoom_preview_busy_update(c3x_renderer_frame_v1 const& frame){
    return sandbox_zoom_preview_busy_state().update(frame);
}
SandboxZoomPreviewBusyMetrics sandbox_zoom_preview_busy_metrics(bool print=true){
    auto& state=sandbox_zoom_preview_busy_state();if(print)state.report();return state.metrics;
}
