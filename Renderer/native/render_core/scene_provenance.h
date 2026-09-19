#pragma once
#include <algorithm>
#include <memory>
#include <vector>

namespace c3x_renderer { namespace render_core {
// A proof that pixels still represent an unmodified rectangle of one map
// publication. Coordinates move with copies; any other write cuts the proof.
// This is rendering provenance, never visibility or gameplay authority.
template<class Source,class Rect> class SceneProvenance {
public:
    struct Part {Rect area;std::shared_ptr<Source> source;int x=0,y=0;};
    std::vector<Part> parts;
    static Rect intersect(Rect a,Rect b){return {std::max(a.left,b.left),std::max(a.top,b.top),std::min(a.right,b.right),std::min(a.bottom,b.bottom)};}
    static bool empty(Rect a){return a.left>=a.right||a.top>=a.bottom;}
    void erase(Rect area){
        std::vector<Part> next;
        for(auto const& p:parts){auto cut=intersect(area,p.area);if(empty(cut)){next.push_back(p);continue;}
            Rect pieces[]={{p.area.left,p.area.top,p.area.right,cut.top},{p.area.left,cut.bottom,p.area.right,p.area.bottom},
                {p.area.left,cut.top,cut.left,cut.bottom},{cut.right,cut.top,p.area.right,cut.bottom}};
            for(auto r:pieces)if(!empty(r))next.push_back({r,p.source,p.x,p.y});
        }
        // Losing an optional proof selects the compatibility path. It cannot
        // grant a new rectangle, grow without bound, or change current pixels.
        parts=next.size()<=256?std::move(next):std::vector<Part>{};
    }
    void copy(SceneProvenance const& input,Rect area,int sx,int sy){
        auto source=input.parts; // self-copy observes the before-image
        erase(area);
        for(auto const& p:source){auto r=intersect(area,{p.area.left-sx,p.area.top-sy,p.area.right-sx,p.area.bottom-sy});
            if(!empty(r))parts.push_back({r,p.source,p.x+sx,p.y+sy});}
        if(parts.size()>256)parts.clear();
    }
    Part select(Rect area)const{
        if(empty(area))return {};
        for(auto const& p:parts)if(area.left>=p.area.left&&area.top>=p.area.top&&area.right<=p.area.right&&area.bottom<=p.area.bottom)return p;
        Part selected{};std::vector<Rect> remaining={area};
        for(auto const& p:parts){if(empty(intersect(area,p.area)))continue;
            if(!selected.source)selected=p;
            if(p.source!=selected.source||p.x!=selected.x||p.y!=selected.y)return {};
            std::vector<Rect> next;
            for(auto r:remaining){auto cut=intersect(r,p.area);if(empty(cut)){next.push_back(r);continue;}
                Rect pieces[]={{r.left,r.top,r.right,cut.top},{r.left,cut.bottom,r.right,r.bottom},
                    {r.left,cut.top,cut.left,cut.bottom},{cut.right,cut.top,r.right,cut.bottom}};
                for(auto piece:pieces)if(!empty(piece))next.push_back(piece);}
            if(next.size()>256)return {};remaining=std::move(next);
        }
        return remaining.empty()?selected:Part{};
    }
};
} }
