#version 430 core

// Define the size of the work group.
// These values are often tuned for performance.
layout (local_size_x = 8, local_size_y = 8, local_size_z = 4) in;

// Input data is a dense 3D texture.
layout (r32f, binding = 0) uniform readonly image3D input_texture;

// Output data is another dense 3D texture.
layout (r32f, binding = 1) uniform writeonly image3D output_texture;

// Uniforms can be used to pass in parameters from the Python side.
uniform float add_value;

void main() {
    // Get the unique ID of this invocation.
    ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);

    // Read a voxel value from the input texture at the given coordinate.
    float voxel_value = imageLoad(input_texture, gid).r;

    // Perform a simple operation.
    float processed_value = voxel_value + add_value;

    // Write the result to the same coordinate in the output texture.
    imageStore(output_texture, gid, vec4(processed_value, 0.0, 0.0, 0.0));
}
