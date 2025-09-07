#version 330 core

// Input texture coordinate from the vertex shader
in vec2 v_uv;

// Output color for the current fragment
out vec4 f_color;

void main() {
    // Create a simple color pattern based on the texture coordinates.
    // The red component will be based on the horizontal position (v_uv.x).
    // The green component will be based on the vertical position (v_uv.y).
    // The blue component will be fixed at 0.2.
    // This creates a colorful gradient that is easy to verify.
    f_color = vec4(v_uv.x, v_uv.y, 0.2, 1.0);
}
