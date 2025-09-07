#version 330 core

// Input from the vertex shader
in vec2 v_uv;

// The texture we're processing
uniform sampler2D u_texture;

// Output color
out vec4 f_color;

void main() {
    // Sample the texture at the given texture coordinate
    // and write it to the output color
    f_color = texture(u_texture, v_uv);
}
