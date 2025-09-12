#version 450

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
} params;

#pragma stage fragment
layout(location = 0) in vec2 v_tex_coord;
layout(location = 0) out vec4 out_color;
layout(binding = 0) uniform sampler2D Source;

void main()
{
    vec4 color = texture(Source, v_tex_coord);
    out_color = vec4(1.0 - color.rgb, color.a);
}

#pragma stage vertex
layout(location = 0) in vec2 in_vert;
layout(location = 0) out vec2 v_tex_coord;

void main()
{
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_tex_coord = in_vert * 0.5 + 0.5;
}
