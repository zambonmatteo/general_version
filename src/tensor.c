#include "tensor.h"
#ifdef MODEL_USE_BITSLICE
#include "bitslice.h"
#endif
#include <float.h>
#include <string.h>

static size_t plane_bytes(uint32_t positions)
{
    return positions / 8u + (positions % 8u != 0u);
}

size_t tensor_bytes(TensorShape shape)
{
    const uint64_t positions = (uint64_t)shape.height * shape.width;
    if (!shape.channels || !positions || positions > UINT32_MAX ||
        shape.channels > UINT32_MAX / positions) return 0;
    return (size_t)shape.channels * plane_bytes((uint32_t)positions);
}

int tensor_shape_equal(TensorShape a, TensorShape b)
{
    return a.channels == b.channels && a.height == b.height && a.width == b.width;
}

int tensor_ranges_overlap(const void *a, size_t a_bytes, const void *b, size_t b_bytes)
{
    if (!a_bytes || !b_bytes) return 0;
    const uintptr_t aa = (uintptr_t)a, bb = (uintptr_t)b;
    return aa >= bb ? aa - bb < b_bytes : bb - aa < a_bytes;
}

static uint32_t get_bit(const uint8_t *data, uint32_t position)
{
    return (uint32_t)((data[position / 8u] >> (position % 8u)) & 1u);
}

static void set_bit(uint8_t *data, uint32_t position)
{
    data[position / 8u] |= (uint8_t)(1u << (position % 8u));
}

static int buffers_valid(const uint8_t *input, size_t input_bytes, size_t need_input,
                         uint8_t *output, size_t output_bytes, size_t need_output)
{
    return input && output && need_input && need_output &&
           input_bytes >= need_input && output_bytes >= need_output &&
           !tensor_ranges_overlap(input, need_input, output, need_output);
}

ModelStatus tensor_pool_shape(TensorShape input, PoolGeometry g, TensorShape *output)
{
    if (!output) return MODEL_INVALID_ARGUMENT;
    *output = (TensorShape){0};
    if (!tensor_bytes(input) || !g.kernel_h || !g.kernel_w ||
        !g.stride_h || !g.stride_w || input.height < g.kernel_h || input.width < g.kernel_w)
        return MODEL_INVALID_SPEC;
    *output = (TensorShape){input.channels,
        (input.height - g.kernel_h) / g.stride_h + 1u,
        (input.width - g.kernel_w) / g.stride_w + 1u};
    return MODEL_OK;
}

ModelStatus tensor_or_pool(const uint8_t *input, size_t input_bytes, TensorShape shape,
                           PoolGeometry g, uint8_t *output, size_t output_bytes)
{
    TensorShape result;
    ModelStatus status = tensor_pool_shape(shape, g, &result);
    if (status != MODEL_OK) return status;
    const size_t need_input = tensor_bytes(shape), need_output = tensor_bytes(result);
    if (!buffers_valid(input, input_bytes, need_input, output, output_bytes, need_output))
        return MODEL_INVALID_ARGUMENT;
    const size_t input_plane = plane_bytes(shape.height * shape.width);
    const size_t output_plane = plane_bytes(result.height * result.width);
    memset(output, 0, need_output);
#ifdef MODEL_USE_BITSLICE
    if (bitslice_pool_supported(shape, g)) {
        bitslice_pool_forward(input, shape, g, output);
        return MODEL_OK;
    }
#endif
    for (uint32_t c = 0; c < shape.channels; ++c) {
        const uint8_t *src = input + (size_t)c * input_plane;
        uint8_t *dst = output + (size_t)c * output_plane;
        for (uint32_t y = 0; y < result.height; ++y) {
            for (uint32_t x = 0; x < result.width; ++x) {
                uint32_t value = 0;
                for (uint32_t ky = 0; ky < g.kernel_h && !value; ++ky)
                    for (uint32_t kx = 0; kx < g.kernel_w && !value; ++kx)
                        value |= get_bit(src, (y * g.stride_h + ky) * shape.width +
                                              x * g.stride_w + kx);
                if (value) set_bit(dst, y * result.width + x);
            }
        }
    }
    return MODEL_OK;
}

ModelStatus tensor_flatten(const uint8_t *input, size_t input_bytes, TensorShape shape,
                           uint8_t *output, size_t output_bytes)
{
    const size_t need_input = tensor_bytes(shape);
    if (!need_input) return MODEL_INVALID_SPEC;
    const uint32_t positions = shape.height * shape.width;
    const uint32_t bits = shape.channels * positions;
    const size_t need_output = plane_bytes(bits), stride = plane_bytes(positions);
    if (!buffers_valid(input, input_bytes, need_input, output, output_bytes, need_output))
        return MODEL_INVALID_ARGUMENT;
    memset(output, 0, need_output);
    for (uint32_t c = 0; c < shape.channels; ++c)
        for (uint32_t p = 0; p < positions; ++p)
            if (get_bit(input + (size_t)c * stride, p)) set_bit(output, c * positions + p);
    return MODEL_OK;
}

ModelStatus tensor_group_argmax(const uint8_t *input, size_t input_bytes,
                                uint32_t bits, uint32_t classes, float tau, uint32_t *prediction)
{
    if (!input || !prediction || !bits || !classes || bits % classes ||
        !(tau > 0.0f && tau <= FLT_MAX) || input_bytes < plane_bytes(bits))
        return MODEL_INVALID_ARGUMENT;
    const uint32_t group = bits / classes;
    uint32_t winner = 0;
    float best = 0;
    for (uint32_t c = 0; c < classes; ++c) {
        uint32_t count = 0;
        for (uint32_t j = 0; j < group; ++j) count += get_bit(input, c * group + j);
        const float score = (float)count / tau;
        if (score > best) { best = score; winner = c; }
    }
    *prediction = winner;
    return MODEL_OK;
}
