#version 430 core

// This is a dummy vertex shader.
// It is required by the current simplified `load_shader_program` method
// when creating a program that primarily uses a compute shader.
// For pure compute operations, this shader does not perform any meaningful work.

void main() {
    // No-op
}
