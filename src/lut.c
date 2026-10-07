#include "lut.h"
#ifdef MODEL_USE_BITSLICE
#include "bitslice.h"
#endif
#include <string.h>

static uint32_t input_bit(const uint8_t *input, uint32_t position)
{
    return (uint32_t)((input[position / 8u] >> (position % 8u)) & 1u);
}

uint32_t lut_table_bit(const Lut *lut, uint32_t address)
{
    return (uint32_t)((lut->table[address / 8u] >> (address % 8u)) & 1u);
}

uint32_t lut_eval_fc(const Lut *lut, const uint8_t *input)
{
    uint32_t address = 0;
    for (uint32_t j = 0; j < lut->n_inputs; ++j) {
        const uint32_t position = lut_index(lut, j);
        /* Preserve mapper order: selected input j becomes address bit j. */
        address |= input_bit(input, position) << j;
    }
    return lut_table_bit(lut, address);
}

uint32_t lut_eval_conv(const Lut *lut, const LutLayerSpec *spec,
                       const uint8_t *input, uint32_t output_y, uint32_t output_x)
{
    const ConvGeometry *g = &spec->convolution;
    const uint32_t height = spec->input_shape.height;
    const uint32_t width = spec->input_shape.width;
    const uint32_t positions = height * width;
    /* Avoid positions + 7, which could overflow for a large valid shape.
     * Round each channel separately: its padding is not part of the next one. */
    const size_t plane_bytes = positions / 8u + (positions % 8u != 0u);
    const uint32_t kernel_positions = g->kernel_h * g->kernel_w;

    /* Signed coordinates represent the negative side of zero padding.
     * Use 64 bits so large validated dimensions cannot wrap an intermediate. */
    const int64_t origin_y = (int64_t)output_y * g->stride_h - g->padding_h;
    const int64_t origin_x = (int64_t)output_x * g->stride_w - g->padding_w;
    uint32_t address = 0;
    for (uint32_t j = 0; j < lut->n_inputs; ++j) {
        const uint32_t index = lut_index(lut, j);
        const uint32_t channel = index / kernel_positions;
        const uint32_t kernel_position = index % kernel_positions;
        const uint32_t kernel_y = kernel_position / g->kernel_w;
        const uint32_t kernel_x = kernel_position % g->kernel_w;
        const int64_t y = origin_y + kernel_y;
        const int64_t x = origin_x + kernel_x;

        /* Out-of-bounds taps contribute zero; skip them before forming a
         * pointer or position into the input buffer. */
        if (y < 0 || x < 0 || y >= height || x >= width) continue;
        const uint8_t *plane = input + (size_t)channel * plane_bytes;
        const uint32_t position = (uint32_t)y * width + (uint32_t)x;
        address |= input_bit(plane, position) << j;
    }
    return lut_table_bit(lut, address);
}

static int layer_ready(const LutLayer *layer)
{
    return layer && layer->spec && layer->tables && layer->indices &&
           (layer->spec->type == LUT_FULLY_CONNECTED ||
            layer->spec->type == LUT_CONVOLUTION);
}

/* Shapes have already been checked during binding. A channel is a packed
 * plane, while an FC vector uses one channel and one row. */
static size_t shape_bytes(TensorShape shape)
{
    const uint32_t positions = shape.height * shape.width;
    const size_t plane_bytes = positions / 8u + (positions % 8u != 0u);
    return (size_t)shape.channels * plane_bytes;
}

size_t lut_layer_input_bytes(const LutLayer *layer)
{
    return layer_ready(layer) ? shape_bytes(layer->spec->input_shape) : 0;
}

size_t lut_layer_output_bytes(const LutLayer *layer)
{
    return layer_ready(layer) ? shape_bytes(layer->spec->output_shape) : 0;
}

/* Output is cleared before evaluation, so zero results need no write. */
static void set_output_bit(uint8_t *output, uint32_t position)
{
    output[position / 8u] |= (uint8_t)(1u << (position % 8u));
}

static void forward_fc(const LutLayer *layer, const uint8_t *input, uint8_t *output)
{
    for (uint32_t k = 0; k < layer->spec->n_luts; ++k) {
        Lut lut;
        /* Binding and this loop's bounds guarantee that the view is valid. */
        (void)lut_layer_get(layer, k, &lut);
        if (lut_eval_fc(&lut, input)) set_output_bit(output, k);
    }
}

static void forward_conv(const LutLayer *layer, const uint8_t *input, uint8_t *output)
{
    const LutLayerSpec *spec = layer->spec;
    const ConvGeometry *g = &spec->convolution;
    const uint32_t height = spec->output_shape.height;
    const uint32_t width = spec->output_shape.width;
    const uint32_t positions = height * width;
    const size_t plane_bytes = positions / 8u + (positions % 8u != 0u);

    const uint32_t input_height = spec->input_shape.height;
    const uint32_t input_width = spec->input_shape.width;
    const uint32_t input_positions = input_height * input_width;
    const size_t input_plane_bytes = input_positions / 8u + (input_positions % 8u != 0u);
    const uint32_t kernel_positions = g->kernel_h * g->kernel_w;

    /* Binding limits LUTs to 31 inputs. This cache belongs only to the current
     * LUT: decode routing once, then reuse it over the complete output plane.
     * Keeping source pointers also avoids repeating channel-stride products. */
    struct {
        const uint8_t *plane;
        uint32_t row, column;
    } taps[31];

    /* Ordinary model dimensions fit signed 32-bit coordinates. Prove every
     * intermediate safe once per layer, outside the pixel loop. Preserve the
     * wide path for valid geometries that need more than 32 signed bits. */
    const int use_narrow = input_height <= INT32_MAX && input_width <= INT32_MAX &&
        g->padding_h <= INT32_MAX && g->padding_w <= INT32_MAX &&
        (uint64_t)(height - 1u) * g->stride_h + g->kernel_h - 1u <= INT32_MAX &&
        (uint64_t)(width - 1u) * g->stride_w + g->kernel_w - 1u <= INT32_MAX;

    for (uint32_t k = 0; k < spec->n_luts; ++k) {
        Lut lut;
        (void)lut_layer_get(layer, k, &lut);
        for (uint32_t j = 0; j < lut.n_inputs; ++j) {
            const uint32_t index = lut_index(&lut, j);
            const uint32_t channel = index / kernel_positions;
            const uint32_t position = index % kernel_positions;
            taps[j].plane = input + (size_t)channel * input_plane_bytes;
            taps[j].row = position / g->kernel_w;
            taps[j].column = position % g->kernel_w;
        }
        /* Each convolution LUT produces one complete output channel. */
        uint8_t *plane = output + (size_t)k * plane_bytes;
        if (use_narrow) {
            for (uint32_t y = 0; y < height; ++y) {
                const int32_t origin_y = (int32_t)(y * g->stride_h) - (int32_t)g->padding_h;
                for (uint32_t x = 0; x < width; ++x) {
                    const int32_t origin_x = (int32_t)(x * g->stride_w) - (int32_t)g->padding_w;
                    uint32_t address = 0;
                    for (uint32_t j = 0; j < lut.n_inputs; ++j) {
                        const int32_t iy = origin_y + (int32_t)taps[j].row;
                        const int32_t ix = origin_x + (int32_t)taps[j].column;
                        /* Unsigned bounds also reject negative coordinates. */
                        if ((uint32_t)iy >= input_height || (uint32_t)ix >= input_width) continue;
                        address |= input_bit(taps[j].plane, (uint32_t)iy * input_width + (uint32_t)ix) << j;
                    }
                    if (lut_table_bit(&lut, address)) set_output_bit(plane, y * width + x);
                }
            }
        } else {
            for (uint32_t y = 0; y < height; ++y) {
                const int64_t origin_y = (int64_t)y * g->stride_h - g->padding_h;
                for (uint32_t x = 0; x < width; ++x) {
                    const int64_t origin_x = (int64_t)x * g->stride_w - g->padding_w;
                    uint32_t address = 0;
                    for (uint32_t j = 0; j < lut.n_inputs; ++j) {
                        const int64_t iy = origin_y + taps[j].row;
                        const int64_t ix = origin_x + taps[j].column;
                        if (iy < 0 || ix < 0 || iy >= input_height || ix >= input_width) continue;
                        address |= input_bit(taps[j].plane, (uint32_t)iy * input_width + (uint32_t)ix) << j;
                    }
                    if (lut_table_bit(&lut, address)) set_output_bit(plane, y * width + x);
                }
            }
        }
    }
}

ModelStatus lut_layer_forward(const LutLayer *layer,
                              const uint8_t *input, size_t input_bytes,
                              uint8_t *output, size_t output_bytes)
{
    if (!layer_ready(layer) || !input || !output) return MODEL_INVALID_ARGUMENT;
    const size_t required_input = lut_layer_input_bytes(layer);
    const size_t required_output = lut_layer_output_bytes(layer);
    if (input_bytes < required_input || output_bytes < required_output)
        return MODEL_INVALID_ARGUMENT;

    /* Compare address distances instead of forming potentially overflowing end
     * addresses. Check only the byte ranges that evaluation will actually use.
     * In-place evaluation would clear input bits before the mapper reads them. */
    const uintptr_t input_address = (uintptr_t)input;
    const uintptr_t output_address = (uintptr_t)output;
    if (output_address >= input_address ? output_address - input_address < required_input
                                        : input_address - output_address < required_output)
        return MODEL_INVALID_ARGUMENT;

    memset(output, 0, required_output);
#ifdef MODEL_USE_BITSLICE
    if (bitslice_conv_supported(layer->spec)) {
        bitslice_conv_forward(layer, input, output);
        return MODEL_OK;
    }
#endif
    if (layer->spec->type == LUT_FULLY_CONNECTED)
        forward_fc(layer, input, output);
    else
        forward_conv(layer, input, output);
    return MODEL_OK;
}
