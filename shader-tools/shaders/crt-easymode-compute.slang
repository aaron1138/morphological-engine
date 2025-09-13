// crt-easymode-compute.slang

struct Push
{
    float BRIGHT_BOOST;
    float DILATION;
    float GAMMA_INPUT;
    float GAMMA_OUTPUT;
    float MASK_SIZE;
    float MASK_STAGGER;
    float MASK_STRENGTH;
    float MASK_DOT_HEIGHT;
    float MASK_DOT_WIDTH;
    float SCANLINE_CUTOFF;
    float SCANLINE_BEAM_WIDTH_MAX;
    float SCANLINE_BEAM_WIDTH_MIN;
    float SCANLINE_BRIGHT_MAX;
    float SCANLINE_BRIGHT_MIN;
    float SCANLINE_STRENGTH;
    float SHARPNESS_H;
    float SHARPNESS_V;
};

struct UBO
{
    float4x4 MVP;
    float4 OutputSize;
    float4 OriginalSize;
    float4 SourceSize;
};

Texture2D<float4> inputTexture;
RWTexture2D<float4> outputTexture;
uniform Push params;
uniform UBO globals;
SamplerState smp;

#define PI 3.141592653589
#define TEX2D(c) dilate(inputTexture.SampleLevel(smp, c, 0))

float4 dilate(float4 col)
{
    float4 x = lerp(float4(1.0), col, params.DILATION);
    return col * x;
}

float curve_distance(float x, float sharp)
{
    float x_step = step(0.5, x);
    float curve = 0.5 - sqrt(0.25 - (x - x_step) * (x - x_step)) * sign(0.5 - x);
    return lerp(x, curve, sharp);
}

float4x4 get_color_matrix(float2 co, float2 dx)
{
    return float4x4(TEX2D(co - dx), TEX2D(co), TEX2D(co + dx), TEX2D(co + 2.0 * dx));
}

float3 filter_lanczos(float4 coeffs, float4x4 color_matrix)
{
    float4 col        = mul(color_matrix, coeffs);
    float4 sample_min = min(color_matrix[1], color_matrix[2]);
    float4 sample_max = max(color_matrix[1], color_matrix[2]);
    col = clamp(col, sample_min, sample_max);
    return col.rgb;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float2 vTexCoord = float2(dispatchThreadID.xy) / globals.OutputSize.xy;

    float2 dx     = float2(globals.SourceSize.z, 0.0);
    float2 dy     = float2(0.0, globals.SourceSize.w);
    float2 pix_co = vTexCoord * globals.SourceSize.xy - float2(0.5, 0.5);
    float2 tex_co = (floor(pix_co) + float2(0.5, 0.5)) * globals.SourceSize.zw;
    float2 dist   = frac(pix_co);
    float curve_x;
    float3 col, col2;

    curve_x = curve_distance(dist.x, params.SHARPNESS_H * params.SHARPNESS_H);
    float4 coeffs = PI * float4(1.0 + curve_x, curve_x, 1.0 - curve_x, 2.0 - curve_x);
    coeffs = max(abs(coeffs), 1e-5);
    coeffs = 2.0 * sin(coeffs) * sin(coeffs * 0.5) / (coeffs * coeffs);
    coeffs /= dot(coeffs, float4(1.0));

    col  = filter_lanczos(coeffs, get_color_matrix(tex_co, dx));
    col2 = filter_lanczos(coeffs, get_color_matrix(tex_co + dy, dx));

    col = lerp(col, col2, curve_distance(dist.y, params.SHARPNESS_V));
    col = pow(col, float3(params.GAMMA_INPUT / (params.DILATION + 1.0)));

    float luma        = dot(float3(0.2126, 0.7152, 0.0722), col);
    float bright      = (max(col.r, max(col.g, col.b)) + luma) * 0.5;
    float scan_bright = clamp(bright, params.SCANLINE_BRIGHT_MIN, params.SCANLINE_BRIGHT_MAX);
    float scan_beam   = clamp(bright * params.SCANLINE_BEAM_WIDTH_MAX, params.SCANLINE_BEAM_WIDTH_MIN, params.SCANLINE_BEAM_WIDTH_MAX);
    float scan_weight = 1.0 - pow(cos(vTexCoord.y * 2.0 * PI * globals.SourceSize.y) * 0.5 + 0.5, scan_beam) * params.SCANLINE_STRENGTH;

    float mask   = 1.0 - params.MASK_STRENGTH;
    float2 mod_fac = floor(vTexCoord * globals.OutputSize.xy * globals.SourceSize.xy / (globals.SourceSize.xy * float2(params.MASK_SIZE, params.MASK_DOT_HEIGHT * params.MASK_SIZE)));
    int dot_no   = int((mod_fac.x + (float(int(mod_fac.y) % 2)) * params.MASK_STAGGER) / params.MASK_DOT_WIDTH) % 3;
    float3 mask_weight;

    if      (dot_no == 0) mask_weight = float3(1.0,  mask, mask);
    else if (dot_no == 1) mask_weight = float3(mask, 1.0,  mask);
    else                  mask_weight = float3(mask, mask, 1.0);

    if (globals.SourceSize.y >= params.SCANLINE_CUTOFF)
        scan_weight = 1.0;

    col2 = col.rgb;
    col *= float3(scan_weight);
    col  = lerp(col, col2, scan_bright);
    col *= mask_weight;
    col  = pow(col, float3(1.0 / params.GAMMA_OUTPUT));

    outputTexture[dispatchThreadID.xy] = float4(col * params.BRIGHT_BOOST, 1.0);
}
