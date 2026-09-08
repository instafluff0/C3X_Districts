Texture2DArray<float> Layers : register(t0);
cbuffer Base : register(b0) { float4 base; };
cbuffer Position : register(b1) { float4 movement; };
cbuffer Shadow : register(b2) { float4 shadow; };
cbuffer Material : register(b7) { float4 material; };
struct Vertex {float3 position:POSITION;};
struct Pixel {float4 position:SV_Position;};
struct Color {float4 color:SV_Target0;float validity:SV_Target1;};
Pixel VSMain(Vertex v){Pixel p;p.position=float4(v.position.xy+movement.xy,v.position.z,1);return p;}
Pixel VSFeature(Vertex v){return VSMain(v);}
Color PSMain(Pixel p){Color c;c.color=float4(Layers.Load(int4(0,0,1,1))*base.x,shadow.y,material.z,1);c.validity=1;return c;}
Color PSFeature(Pixel p){return PSMain(p);}
