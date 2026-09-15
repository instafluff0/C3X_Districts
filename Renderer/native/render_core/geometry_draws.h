#pragma once
#include <array>
#include <cstddef>
#include <vector>

namespace c3x_renderer { namespace render_core {
// The active view borrows immutable content from its epoch-protected owner.
// Only projection belongs to an occurrence; copying this record acquires no
// resource or material ownership. Clear records before releasing their owner.
template<class Chunk> struct GeometryDrawRecord {
    Chunk const* source=nullptr;
    decltype(((Chunk*)nullptr)->bounds) bounds={};
    int translation_x=0,translation_y=0;
    float natural_projection[4]={};

    GeometryDrawRecord()=default;
    GeometryDrawRecord(Chunk const& chunk):source(&chunk),bounds(chunk.bounds),
        translation_x(chunk.translation_x),translation_y(chunk.translation_y) {
        for(unsigned i=0;i<4;++i)natural_projection[i]=chunk.natural_projection[i];
    }
    Chunk const& content() const {return *source;}
};

// One read interface for resident occurrences and the existing owned dynamic
// chunks. It neither assembles a second list nor resolves/copies mesh metadata.
template<class Chunk,std::size_t Layers> class GeometryDrawView {
public:
    using Record=GeometryDrawRecord<Chunk>;
    using Records=std::array<std::vector<Record>,Layers>;
    using Chunks=std::array<std::vector<Chunk>,Layers>;
    // Traversal copies two pointers, never projection fields or mesh metadata.
    struct Reference {
        Chunk const* source;
        Record const* occurrence;
        Reference(Chunk const& value):source(&value),occurrence(nullptr){}
        Reference(Record const& value):source(value.source),occurrence(&value){}
        Chunk const& content() const {return *source;}
        auto const& bounds() const {return occurrence?occurrence->bounds:source->bounds;}
        int translation_x() const {return occurrence?occurrence->translation_x:source->translation_x;}
        int translation_y() const {return occurrence?occurrence->translation_y:source->translation_y;}
        float const (&natural_projection() const)[4] {
            if(occurrence)return occurrence->natural_projection;
            return source->natural_projection;
        }
    };
    struct Layer {
        Chunk const* chunks;
        Record const* records;
        std::size_t count;
        Reference operator[](std::size_t i) const {return chunks?Reference(chunks[i]):Reference(records[i]);}
        std::size_t size() const {return count;}
        bool empty() const {return count==0;}
        struct Iterator {
            Layer const* layer;
            std::size_t index;
            Reference operator*() const {return (*layer)[index];}
            Iterator& operator++(){++index;return *this;}
            bool operator!=(Iterator const& other) const {return index!=other.index;}
        };
        Iterator begin() const {return {this,0};}
        Iterator end() const {return {this,count};}
    };
    // Immutable pass membership for one borrowed view. It lives within the
    // owner's lease; no resource references or visibility survive view teardown.
    struct Pass {
        std::array<bool,Layers> selected{};
        std::size_t size=0;
        bool any()const{return size!=0;}
        bool has(std::size_t layer)const{return selected[layer];}
        std::size_t count()const{return size;}
    };
    Pass pass()const{
        Pass result;
        if(*this)for(std::size_t i=0;i<Layers;++i){result.selected[i]=!(*this)[i].empty();result.size+=result.selected[i];}
        return result;
    }
    GeometryDrawView(std::nullptr_t=nullptr){}
    GeometryDrawView(Records const& value):records(&value){}
    GeometryDrawView(Chunks const& value):chunks(&value){}
    explicit operator bool() const {return records || chunks;}
    bool is(Records const& value) const {return records==&value;}
    bool is(Chunks const& value) const {return chunks==&value;}
    Layer operator[](std::size_t i) const {
        if(chunks)return {(*chunks)[i].data(),nullptr,(*chunks)[i].size()};
        return {nullptr,(*records)[i].data(),(*records)[i].size()};
    }
private:
    Records const* records=nullptr;
    Chunks const* chunks=nullptr;
};
} }
