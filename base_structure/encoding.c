#include "encoding.h"
#include <string.h>

/* Cortex-M4 uses little-endian IEEE 754 float32. memcpy accepts any alignment. */
static float read_float(const uint8_t *data)
{
    float value;
    memcpy(&value, data, sizeof(value));
    return value;
}

static const float edge_kernels[5][9] = {
    {-1, 0, 1, -2, 0, 2, -1, 0, 1},
    {-1, -2, -1, 0, 0, 0, 1, 2, 1},
    {0, 1, 2, -1, 0, 1, -2, -1, 0},
    {2, 1, 0, 1, 0, -1, 0, -1, -2},
    {0, 1, 0, 1, -4, 1, 0, 1, 0}
};

static float input_value(const uint8_t *input, InputType type, size_t index)
{
    return type == INPUT_FLOAT32 ? read_float(input + 4u * index)
                                : (float)input[index] / 255.0f;
}

void encoding_forward(const Encoding *encoding, const void *input,
                      uint8_t *output, float *scratch)
{
    const uint8_t *raw = input;
    const size_t channels = encoding->raw_shape.channels;
    const size_t height = encoding->raw_shape.height;
    const size_t width = encoding->raw_shape.width;
    const size_t bits = encoding->bits;
    const size_t positions = height * width;
    const size_t stride = bitplane_bytes((uint32_t)positions);
    memset(output, 0, (channels * bits + encoding->edge_channels) * stride);

    for (size_t c = 0; c < channels; ++c) {
        for (size_t b = 0; b < bits; ++b) {
            uint8_t *plane = output + (c * bits + b) * stride;
            for (size_t p = 0; p < positions; ++p) {
                size_t t = encoding->threshold_layout == THRESHOLDS_CHANNEL ? c * bits + b :
                           encoding->threshold_layout == THRESHOLDS_ROW ? (p / width) * bits + b : b;
                float threshold = read_float(encoding->thermo_thresholds + 4u * t);
                if (input_value(raw, encoding->input_type, c * positions + p) > threshold)
                    bitplane_set(plane, (uint32_t)p);
            }
        }
    }

    if (!encoding->edge_channels) return;
    for (size_t p = 0; p < positions; ++p) {
        float sum = 0.0f;
        for (size_t c = 0; c < channels; ++c)
            sum += input_value(raw, encoding->input_type, c * positions + p);
        scratch[p] = sum / (float)channels;
    }

    for (size_t d = 0; d < 5; ++d) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                float acc = 0.0f;
                for (int ky = 0; ky < 3; ++ky) {
                    ptrdiff_t yy = (ptrdiff_t)y + ky - 1;
                    if (yy < 0 || (size_t)yy >= height) continue;
                    for (int kx = 0; kx < 3; ++kx) {
                        ptrdiff_t xx = (ptrdiff_t)x + kx - 1;
                        if (xx < 0 || (size_t)xx >= width) continue;
                        acc += scratch[(size_t)yy * width + (size_t)xx]
                             * edge_kernels[d][ky * 3 + kx];
                    }
                }
                for (size_t t = 0; t < 2; ++t) {
                    float threshold = read_float(encoding->edge_thresholds + 4u * (2u * d + t));
                    if (acc > threshold)
                        bitplane_set(output + (channels * bits + 2u * d + t) * stride,
                                       (uint32_t)(y * width + x));
                }
            }
        }
    }
}
