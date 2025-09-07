#version 330 core

// Input vertex attribute (position) from the VBO
in vec2 in_vert;

// Output texture coordinate to the fragment shader
out vec2 v_uv;

void main() {
    // The input vertices are already in Normalized Device Coordinates (-1 to 1),
    // so we can pass them directly to gl_Position.
    gl_Position = vec4(in_vert, 0.0, 1.0);

    // Convert the vertex position to texture coordinates (0 to 1 range)
    // and pass it to the fragment shader.
    v_uv = in_vert * 0.5 + 0.5;
}
