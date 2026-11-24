#include "layer.h"
#include "lut.h"
#ifdef USE_BITSLICE
#include "bitslice.h"
#endif
#include <string.h>

uint32_t layer_read_index(const uint8_t *data, uint32_t size_bytes)
{
    if (size_bytes == 2u) {
        uint16_t value;
        memcpy(&value, data, sizeof(value));
        return value;
    }
    uint32_t value;
    memcpy(&value, data, sizeof(value));
    return value;
}

static uint32_t fc_address(const LutLayer *layer, const uint8_t *indices,
                           const uint8_t *input)
{
    uint32_t address = 0;
    for (uint32_t j = 0; j < layer->input_bits; ++j) {
        const uint32_t position = layer_read_index(indices, layer->index_size_bytes);
        address |= bitplane_get(input, position) << j;
        indices += layer->index_size_bytes;
    }
    return address;
}

typedef struct {
    const uint8_t *plane;
    uint32_t row, column;
} ConvTap;

static inline uint32_t conv_address(const ConvTap *taps, uint32_t count, Shape shape,
                                    int64_t row, int64_t column, int narrow)
{
    uint32_t address = 0;
    if (narrow) {
        for (uint32_t j = 0; j < count; ++j) {
            const int32_t iy = (int32_t)row + (int32_t)taps[j].row;
            const int32_t ix = (int32_t)column + (int32_t)taps[j].column;
            if ((uint32_t)iy >= shape.height || (uint32_t)ix >= shape.width) continue;
            address |= bitplane_get(taps[j].plane, (uint32_t)iy * shape.width + (uint32_t)ix) << j;
        }
    } else {
        for (uint32_t j = 0; j < count; ++j) {
            const int64_t iy = row + taps[j].row, ix = column + taps[j].column;
            if (iy < 0 || ix < 0 || iy >= shape.height || ix >= shape.width) continue;
            address |= bitplane_get(taps[j].plane, (uint32_t)iy * shape.width + (uint32_t)ix) << j;
        }
    }
    return address;
}

static void forward_fc(const Layer *layer, const uint8_t *input, uint8_t *output)
{
    const LutLayer *data = &layer->parameters.lut;
    const size_t index_stride = (size_t)data->input_bits * data->index_size_bytes;
    for (uint32_t k = 0; k < data->lut_count; ++k) {
        const Lut lut = {data->tables + (size_t)k * data->table_stride_bytes,
                         UINT32_C(1) << data->input_bits};
        const uint8_t *indices = data->indices + (size_t)k * index_stride;
        if (lut_read(&lut, fc_address(data, indices, input))) bitplane_set(output, k);
    }
}

static void forward_conv(const Layer *layer, const uint8_t *input, uint8_t *output)
{
    const LutLayer *data = &layer->parameters.lut;
    const ConvParams g = data->convolution;
    const Shape in = layer->input_shape, out = layer->output_shape;
    const size_t input_plane = bitplane_bytes(in.height * in.width);
    const size_t output_plane = bitplane_bytes(out.height * out.width);
    const size_t index_stride = (size_t)data->input_bits * data->index_size_bytes;
    const uint32_t kernel_positions = g.kernel_h * g.kernel_w;
    /* Check once per layer; retain wide coordinates for larger valid shapes. */
    const int narrow = in.height <= INT32_MAX && in.width <= INT32_MAX &&
        g.padding_h <= INT32_MAX && g.padding_w <= INT32_MAX &&
        (uint64_t)(out.height - 1u) * g.stride_h + g.kernel_h - 1u <= INT32_MAX &&
        (uint64_t)(out.width - 1u) * g.stride_w + g.kernel_w - 1u <= INT32_MAX;
    ConvTap taps[31];  /* The generator validates 1..31 inputs per LUT. */
    for (uint32_t k = 0; k < data->lut_count; ++k) {
        const Lut lut = {data->tables + (size_t)k * data->table_stride_bytes,
                         UINT32_C(1) << data->input_bits};
        const uint8_t *indices = data->indices + (size_t)k * index_stride;
        /* Routing is constant over this LUT's complete output plane. */
        for (uint32_t j = 0; j < data->input_bits; ++j) {
            const uint32_t index = layer_read_index(indices + (size_t)j * data->index_size_bytes,
                                                     data->index_size_bytes);
            const uint32_t position = index % kernel_positions;
            taps[j] = (ConvTap){input + (size_t)(index / kernel_positions) * input_plane,
                                position / g.kernel_w, position % g.kernel_w};
        }
        uint8_t *plane = output + (size_t)k * output_plane;
        for (uint32_t y = 0; y < out.height; ++y) {
            const int64_t row = (int64_t)y * g.stride_h - g.padding_h;
            for (uint32_t x = 0; x < out.width; ++x) {
                const int64_t column = (int64_t)x * g.stride_w - g.padding_w;
                if (lut_read(&lut, conv_address(taps, data->input_bits, in, row, column, narrow)))
                    bitplane_set(plane, y * out.width + x);
            }
        }
    }
}

static void forward_pool(const Layer *layer, const uint8_t *input, uint8_t *output)
{
    const Shape in = layer->input_shape, out = layer->output_shape;
    const PoolParams g = layer->parameters.pool;
    const size_t input_plane = bitplane_bytes(in.height * in.width);
    const size_t output_plane = bitplane_bytes(out.height * out.width);
    for (uint32_t c = 0; c < in.channels; ++c) {
        const uint8_t *src = input + (size_t)c * input_plane;
        uint8_t *dst = output + (size_t)c * output_plane;
        for (uint32_t y = 0; y < out.height; ++y) {
            for (uint32_t x = 0; x < out.width; ++x) {
                uint32_t bit = 0;
                for (uint32_t ky = 0; ky < g.kernel_h && !bit; ++ky)
                    for (uint32_t kx = 0; kx < g.kernel_w && !bit; ++kx)
                        bit |= bitplane_get(src, (y * g.stride_h + ky) * in.width + x * g.stride_w + kx);
                if (bit) bitplane_set(dst, y * out.width + x);
            }
        }
    }
}

static void forward_flatten(const Layer *layer, const uint8_t *input, uint8_t *output)
{
    const Shape shape = layer->input_shape;
    const uint32_t positions = shape.height * shape.width;
    const size_t plane_bytes = bitplane_bytes(positions);
    for (uint32_t c = 0; c < shape.channels; ++c)
        for (uint32_t p = 0; p < positions; ++p)
            if (bitplane_get(input + (size_t)c * plane_bytes, p))
                bitplane_set(output, c * positions + p);
}

void layer_forward(const Layer *layer, const uint8_t *input, uint8_t *output)
{
    const Shape shape = layer->output_shape;
    memset(output, 0, shape.channels * bitplane_bytes(shape.height * shape.width));
    switch (layer->type) {
    case LAYER_LUT_CONV:
#ifdef USE_BITSLICE
        if (bitslice_conv_supported(layer)) {
            bitslice_conv_forward(layer, input, output);
            break;
        }
#endif
        forward_conv(layer, input, output);
        break;
    case LAYER_LUT_FC:
        forward_fc(layer, input, output);
        break;
    case LAYER_OR_POOL:
#ifdef USE_BITSLICE
        if (bitslice_pool_supported(layer->input_shape, layer->parameters.pool)) {
            bitslice_pool_forward(input, layer->input_shape, layer->parameters.pool, output);
            break;
        }
#endif
        forward_pool(layer, input, output);
        break;
    case LAYER_FLATTEN:
        forward_flatten(layer, input, output);
        break;
    default:
        break;
    }
}

uint32_t layer_group_sum(const uint8_t *input, GroupSumParams parameters)
{
    uint32_t winner = 0;
    float best = 0;
    for (uint32_t c = 0; c < parameters.classes; ++c) {
        uint32_t count = 0;
        for (uint32_t j = 0; j < parameters.bits_per_class; ++j)
            count += bitplane_get(input, c * parameters.bits_per_class + j);
        const float score = (float)count / parameters.tau;
        if (score > best) {
            best = score;
            winner = c;
        }
    }
    return winner;
}
