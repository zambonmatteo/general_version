#include "encode.h"
#include <float.h>
#include <string.h>

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
               FLT_MAX_EXP == 128, "IEEE 754 binary32 float required");

static const float kernels[5][9] = {
    {-1,0,1,-2,0,2,-1,0,1},
    {-1,-2,-1,0,0,0,1,2,1},
    {0,1,2,-1,0,1,-2,-1,0},
    {2,1,0,1,0,-1,0,-1,-2},
    {0,1,0,1,-4,1,0,1,0}
};

static float value(const void *input, int float_input, size_t index)
{
    return float_input ? ((const float *)input)[index]
                       : (float)((const uint8_t *)input)[index] / 255.0f;
}

static void set_bit(uint8_t *plane, size_t position)
{
    plane[position / 8] |= (uint8_t)(1u << (position % 8));
}

void encode_input(const void *input, int float_input, size_t channels,
                  size_t height, size_t width, size_t bits, int layout,
                  const float *thresholds, int edges,
                  const float *edge_thresholds, float *scratch, uint8_t *out)
{
    const size_t positions = height * width;
    const size_t stride = (positions + 7) / 8;
    memset(out, 0, (channels * bits + (edges ? 10 : 0)) * stride);
    for (size_t c = 0; c < channels; ++c) {
        for (size_t b = 0; b < bits; ++b) {
            uint8_t *plane = out + (c * bits + b) * stride;
            for (size_t p = 0; p < positions; ++p) {
                size_t t = layout == 1 ? c * bits + b :
                           layout == 2 ? (p / width) * bits + b : b;
                if (value(input, float_input, c * positions + p) > thresholds[t])
                    set_bit(plane, p);
            }
        }
    }
    if (!edges) return;
    for (size_t p = 0; p < positions; ++p) {
        float sum = 0.0f;
        for (size_t c = 0; c < channels; ++c)
            sum += value(input, float_input, c * positions + p);
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
                             * kernels[d][ky * 3 + kx];
                    }
                }
                for (size_t t = 0; t < 2; ++t)
                    if (acc > edge_thresholds[2 * d + t])
                        set_bit(out + (channels * bits + 2 * d + t) * stride,
                                y * width + x);
            }
        }
    }
}
