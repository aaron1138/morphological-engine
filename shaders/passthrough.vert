#version 330 core

// Input vertex attributes from the VBO
in vec2 in_vert;
in vec2 in_uv;

// Output to the fragment shader
out vec2 v_uv;

void main() {
    // Pass the texture coordinate to the fragment shader
    v_uv = in_uv;
    // Set the position of the vertex
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
