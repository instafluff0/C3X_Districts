#include "animation_runtime.h"
#include "unit_animation_runtime.h"
#include <cassert>
#include <fstream>
#include <iterator>
#include <limits>
#include <iostream>

void append_u32(std::vector<std::uint8_t> & data, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) data.push_back(static_cast<std::uint8_t>(value >> (i * 8)));
}
void append_float(std::vector<std::uint8_t> & data, float value) {
    std::uint32_t bits; std::memcpy(&bits, &value, 4); append_u32(data, bits);
}
void replace_u32(std::vector<std::uint8_t> & data, std::size_t offset, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) data[offset+i] = static_cast<std::uint8_t>(value >> (i*8));
}
std::vector<std::uint8_t> fixture() {
    std::vector<std::uint8_t> data = {'C','3','X','A','N','M','1',0};
    for (auto value : {1u,3u,3u,1u,2u}) append_u32(data, value);
    append_float(data, 2);
    for (unsigned i = 0; i < 3; ++i) {
        for (float v : {1.f, float(i), 0.f, 1.f, 1.f, 0.f, 0.25f, 0.75f}) append_float(data, v);
        for (unsigned j = 0; j < 4; ++j) append_u32(data, 0);
        for (float v : {1.f,0.f,0.f,0.f}) append_float(data, v);
    }
    for (auto value : {0u,1u,2u}) append_u32(data, value);
    for (unsigned i = 0; i < 2; ++i)
        for (float v : {1.f+2.f*i,0.f,0.f,0.f,0.f,1.f,0.f,0.f,
                        0.f,0.f,1.f,0.f,4.f*i,0.f,0.f,1.f}) append_float(data, v);
    return data;
}

int main(int argc, char ** argv) {
    using namespace c3x_renderer;
    if (argc == 5) {
        std::ifstream stream(argv[1], std::ios::binary);
        std::vector<std::uint8_t> data{std::istreambuf_iterator<char>(stream), {}};
        AnimationMesh mesh;
        if (!decode_animation_mesh(data, mesh)) return 2;
        std::vector<FeatureSourceVertex> vertices;
        if (!sample_animation_mesh(mesh, std::stod(argv[2]), std::stoi(argv[3]) != 0, vertices)) return 3;
        std::ofstream result(argv[4], std::ios::binary);
        result.write(reinterpret_cast<char const *>(vertices.data()),
                     static_cast<std::streamsize>(vertices.size()*sizeof(FeatureSourceVertex)));
        return result ? 0 : 4;
    }
    auto data = fixture();
    AnimationMesh mesh;
    assert(decode_animation_mesh(data, mesh));
    std::vector<FeatureSourceVertex> vertices;
    assert(sample_animation_mesh(mesh, 1, false, vertices));
    assert(std::abs(vertices[0].position[0] - 4.f) < 1e-6f);
    // Nonuniform animated scale must inverse-transform normals.
    assert(std::abs(vertices[0].normal[0] - 1/std::sqrt(5.f)) < 1e-6f);
    assert(std::abs(vertices[0].normal[1] - 2/std::sqrt(5.f)) < 1e-6f);
    assert(vertices[0].uv[0] == .25f && vertices[0].uv[1] == .75f);
    assert(sample_animation_mesh(mesh, 20, false, vertices));
    assert(vertices[0].position[0] == 7.f); // Native one-shot endpoint remains final pose.
    assert(sample_animation_mesh(mesh, 20, true, vertices));
    assert(vertices[0].position[0] == 1.f);
    assert(sample_animation_mesh(mesh, -1, true, vertices));
    assert(vertices[0].position[0] == 4.f);
    assert(!sample_animation_mesh(mesh, std::numeric_limits<double>::quiet_NaN(), false, vertices));
    {
        auto hidden=mesh;
        for(unsigned frame=0;frame<hidden.frames;++frame) {
            auto p=hidden.palettes.data()+frame*16;
            p[0]=p[5]=p[10]=0;
        }
        assert(sample_animation_mesh(hidden,1,false,vertices));
        for(auto const& v:vertices) {
            assert(v.position[0]==2.f && v.position[1]==0.f && v.position[2]==0.f);
            for(float n:v.normal)assert(std::isfinite(n));
        }
        assert(sample_animation_mesh(mesh,1,false,vertices));
        assert(vertices[0].position[0]==4.f); // Visible pose unaffected.
    }
    double late = ambient_animation_time(9000000123ll, 1000, 2.0, 17);
    assert(late == ambient_animation_time(9000002123ll, 1000, 2.0, 17));
    assert(ambient_animation_time(1, 0, 2, 0) == 0);
    // Ambient source timing is independent of a shorter Civ III cursor span.
    assert(ambient_animation_frame(1500,1000,4.0,121,0)==45);
    assert(ambient_animation_frame(5500,1000,4.0,121,0)==45);
    // Repeated draws and skipped draws sample exactly the same absolute phase.
    assert(sample_animation_mesh(mesh, late, true, vertices));
    auto snapshot = vertices;
    assert(sample_animation_mesh(mesh, late + 1234, true, vertices));
    assert(sample_animation_mesh(mesh, late, true, vertices));
    assert(std::memcmp(snapshot.data(), vertices.data(), vertices.size()*sizeof(FeatureSourceVertex)) == 0);
    auto reject = [&](std::vector<std::uint8_t> const & bad) {
        assert(!decode_animation_mesh(bad, mesh)); assert(mesh.vertices.size() == 3 && mesh.duration == 2);
    };
    for (std::size_t length = 0; length < data.size(); ++length)
        reject(std::vector<std::uint8_t>(data.begin(), data.begin()+static_cast<std::ptrdiff_t>(length)));
    for (auto field : {8u, 12u, 16u, 20u, 24u}) {
        auto bad = data; replace_u32(bad, field, 0xffffffffu); reject(bad);
    }
    for (auto field : {28u,32u,44u,56u,80u,236u}) {
        auto bad = data; replace_u32(bad, field, 0x7fc00000u); reject(bad);
    }
    { auto bad = data; replace_u32(bad, 64, 1); reject(bad); } // Joint outside palette.
    { auto bad = data; replace_u32(bad, 224, 3); reject(bad); } // Index outside mesh.
    { auto bad = data; bad.push_back(0); reject(bad); }
    { auto bad = data; replace_u32(bad, 248, 0x3f800000u); reject(bad); } // Non-affine matrix.
    for(int scale:{250,500,625,1000,1500,2000})for(int width:{127,191,320,401}) {
        int x=-101,y=63,w=width,h=width+2;
        int ax=x+w*scale/2000,ay=y+h*scale/2000;
        assert(expand_unit_canvas(x,y,w,h,scale,320));
        assert(x+w*scale/2000==ax && y+h*scale/2000==ay);
        assert(w>=320 && h>=320 && w>=width && h>=width+2);
    }
    {int x=INT32_MAX,y=0,w=191,h=191;assert(!expand_unit_canvas(x,y,w,h,2000,320));assert(x==INT32_MAX && w==191);}
    NativeUnitDraw draw;
    draw.expected_sprite = draw.sprite = 17; draw.expected_canvas = draw.canvas = 29;
    draw.unit_id = 5; draw.action = 2; draw.direction = 3;
    draw.frame_count = 8; draw.action_cursor = 3;
    draw.body_x = -11; draw.body_y = 73; draw.sprite_width = 131; draw.sprite_height = 127;
    UnitAnimationPose pose;
    assert(prepare_native_unit_pose(draw, true, pose));
    assert(pose.anchor_x == 54 && pose.anchor_y == 136 && pose.phase == 3.0/8);
    assert(std::strcmp(pose.action, "move") == 0 && pose.direction == 3);
    auto retained = pose;
    // Camera translation changes only placement, and a repeated callback does
    // not restart the pose. Native direction changes are forwarded unchanged.
    draw.body_x -= 256; draw.body_y += 128; draw.direction = 7;
    assert(prepare_native_unit_pose(draw, true, pose));
    assert(pose.anchor_x == retained.anchor_x-256 && pose.anchor_y == retained.anchor_y+128);
    assert(pose.phase == retained.phase && pose.direction == 7);
    draw.reduced = true;
    assert(prepare_native_unit_pose(draw, true, pose));
    assert(pose.anchor_x == draw.body_x+32 && pose.anchor_y == draw.body_y+31);
    assert(pose.projection_scale == .5f);
    draw.projection_scale_milli = 625;
    assert(prepare_native_unit_pose(draw, true, pose));
    assert(pose.anchor_x == draw.body_x+40 && pose.anchor_y == draw.body_y+39);
    assert(pose.projection_scale == .625f);
    draw.projection_scale_milli = 249;
    assert(!prepare_native_unit_pose(draw, true, pose));
    draw.projection_scale_milli = 625;
    draw.action = 6; draw.action_cursor = 999;
    assert(prepare_native_unit_pose(draw, false, pose));
    assert(pose.phase == 1 && std::strcmp(pose.action, "death") == 0);
    draw.frame_count = 1;
    assert(prepare_native_unit_pose(draw, false, pose) && pose.phase == 1);
    assert(prepare_native_unit_pose(draw, true, pose) && pose.phase == 0);
    retained = pose;
    for (auto bad : {0, 19, -1}) {
        draw.action = bad; assert(!prepare_native_unit_pose(draw, false, pose));
        assert(pose.anchor_x == retained.anchor_x && pose.phase == retained.phase);
    }
    draw.action = 1; draw.sprite = 42;
    assert(!prepare_native_unit_pose(draw, true, pose)); // Unrelated Sprite remains native.
    draw.sprite = 17; draw.canvas = 42;
    assert(!prepare_native_unit_pose(draw, true, pose)); // Another canvas remains native.
    draw.canvas = 29; draw.frame_count = 0;
    assert(!prepare_native_unit_pose(draw, true, pose));
    draw.frame_count = 8; draw.body_x = INT32_MAX;
    assert(!prepare_native_unit_pose(draw, true, pose));
    std::cout << "animation runtime: timing, skinning, normals, endpoints, malformed payloads passed\n";
}
