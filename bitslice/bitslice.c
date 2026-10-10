#include "bitslice.h"
#include <string.h>

/* Evaluate convolution and OR pooling on 32-bit words of one encoded sample.
 * CHW planes use LSB-first bits and independent byte padding. Layer descriptors
 * supply convolution geometry and LUT routing indices. */
static uint32_t low_mask(uint32_t bits)
{
    return bits >= BS_WORD_BITS ? UINT32_MAX : (UINT32_C(1) << bits) - 1u;
}

static uint32_t plane_bytes(uint32_t positions)
{
    return (positions + 7u) / 8u;
}

static uint32_t load_word(const uint8_t *plane, int32_t word, uint32_t bytes)
{
    if (word < 0) return 0;
    const uint32_t start = (uint32_t)word * 4u;
    if (start >= bytes) return 0;
    if (bytes - start >= 4u) {
        uint32_t value;
        /* Word loads require little-endian byte order. memcpy accepts unaligned
         * plane addresses caused by independent per-channel byte padding. */
        memcpy(&value, plane + start, sizeof(value));
        return value;
    }
    uint32_t value = 0;
    for (uint32_t b = 0; b < bytes - start; ++b)
        value |= (uint32_t)plane[start + b] << (8u * b);
    return value;
}

static uint32_t word_at(const uint8_t *plane, int32_t bit, uint32_t bytes)
{
    /* Explicit floor division avoids relying on signed negative right shifts. */
    const int32_t word = bit >= 0 ? bit / 32 : -((-bit + 31) / 32);
    const uint32_t shift = (uint32_t)(bit - word * 32);
    const uint32_t lo = load_word(plane, word, bytes);
    return shift ? (lo >> shift) | (load_word(plane, word + 1, bytes) << (32u - shift)) : lo;
}

static void store_word(uint8_t *plane, uint32_t word, uint32_t value, uint32_t bytes)
{
    const uint32_t start = word * 4u;
    if (bytes - start >= 4u) {
        memcpy(plane + start, &value, sizeof(value));
        return;
    }
    for (uint32_t b = 0; b < bytes - start; ++b)
        plane[start + b] = (uint8_t)(value >> (8u * b));
}

static uint32_t evaluate_word(const Lut *lut, const uint32_t *taps, uint32_t *values)
{
    const uint32_t select[4] = {0, ~taps[0], taps[0], UINT32_MAX};
    uint32_t count = 1u << (lut->n_inputs - 1u);
    for (uint32_t a = 0; a < count; ++a)
        values[a] = select[(lut->table[a / 4u] >> ((a % 4u) * 2u)) & 3u];
    for (uint32_t j = 1; j < lut->n_inputs; ++j) {
        count /= 2u;
        for (uint32_t a = 0; a < count; ++a)
            values[a] = (values[2u * a] & ~taps[j]) | (values[2u * a + 1u] & taps[j]);
    }
    return values[0];
}

int bitslice_conv_supported(const LutLayerSpec *s)
{
    if (!s || s->type != LUT_CONVOLUTION || !s->n_inputs || s->n_inputs > BS_MAX_INPUTS)
        return 0;
    const ConvGeometry *g = &s->convolution;
    
    return g->stride_h == 1u && g->stride_w == 1u &&
        (uint64_t)s->input_shape.height * s->input_shape.width <= BS_MAX_POSITIONS &&
        (uint64_t)s->output_shape.height * s->output_shape.width <= BS_MAX_POSITIONS &&
        (uint64_t)g->kernel_h * g->kernel_w <= BS_MAX_POSITIONS &&
        g->padding_h <= BS_MAX_POSITIONS && g->padding_w <= BS_MAX_POSITIONS;
}

/* Mask bit shifts that cross a row or tensor boundary. A 3x3 convolution with
 * one-cell padding uses nine masks per word, one for each kernel position. */
static uint32_t same_mask(uint32_t word, uint32_t h, uint32_t w, int32_t dr, int32_t dc)
{
    const uint32_t base = word * BS_WORD_BITS, positions = h * w;
    uint32_t mask = low_mask(positions - base);
    if (dc < 0)
        for (uint32_t b = (w - base % w) % w; b < BS_WORD_BITS; b += w)
            mask &= ~(UINT32_C(1) << b);
    else if (dc > 0)
        for (uint32_t b = (2u * w - 1u - base % w) % w; b < BS_WORD_BITS; b += w)
            mask &= ~(UINT32_C(1) << b);
    if (dr < 0 && base < w) mask &= ~low_mask(w - base);
    if (dr > 0) {
        const uint32_t last_row = (h - 1u) * w;
        if (base + BS_WORD_BITS > last_row)
            mask &= low_mask(last_row > base ? last_row - base : 0u);
    }
    return mask;
}

static void forward_same3(const LutLayer *layer, const uint8_t *input, uint8_t *output)
{
    const LutLayerSpec *s = layer->spec;
    const uint32_t h = s->input_shape.height, w = s->input_shape.width;
    const uint32_t positions = h * w, bytes = plane_bytes(positions);
    const uint32_t words = (positions + BS_WORD_BITS - 1u) / BS_WORD_BITS;
    uint32_t masks[9][BS_MAX_POSITIONS / BS_WORD_BITS];
    uint32_t values[1u << (BS_MAX_INPUTS - 1u)], taps[BS_MAX_INPUTS];
    for (int32_t dr = -1; dr <= 1; ++dr)
        for (int32_t dc = -1; dc <= 1; ++dc)
            for (uint32_t word = 0; word < words; ++word)
                masks[(dr + 1) * 3 + dc + 1][word] = same_mask(word, h, w, dr, dc);
    for (uint32_t k = 0; k < s->n_luts; ++k) {
        Lut lut;
        (void)lut_layer_get(layer, k, &lut);
        const uint8_t *planes[BS_MAX_INPUTS];
        const uint32_t *tap_masks[BS_MAX_INPUTS];
        int32_t offsets[BS_MAX_INPUTS];
        for (uint32_t j = 0; j < lut.n_inputs; ++j) {
            const uint32_t index = lut_index(&lut, j), position = index % 9u;
            const int32_t dr = (int32_t)(position / 3u) - 1;
            const int32_t dc = (int32_t)(position % 3u) - 1;
            planes[j] = input + (size_t)(index / 9u) * bytes;
            offsets[j] = dr * (int32_t)w + dc;
            tap_masks[j] = masks[position];
        }
        for (uint32_t word = 0; word < words; ++word) {
            for (uint32_t j = 0; j < lut.n_inputs; ++j)
                taps[j] = word_at(planes[j], (int32_t)(word * BS_WORD_BITS) + offsets[j], bytes)
                    & tap_masks[j][word];
            store_word(output + (size_t)k * bytes, word,
                evaluate_word(&lut, taps, values) & masks[4][word], bytes);
        }
    }
}

/* Append a row/chunk to a byte-padded plane even when rows straddle words. */
typedef struct {
    uint8_t *plane;
    uint32_t bytes, word, pending, used;
} PackedWriter;

static void append_bits(PackedWriter *writer, uint32_t value, uint32_t bits)
{
    writer->pending |= value << writer->used;
    const uint32_t total = writer->used + bits;
    if (total >= BS_WORD_BITS) {
        store_word(writer->plane, writer->word++, writer->pending, writer->bytes);
        const uint32_t remaining = total - BS_WORD_BITS;
        writer->pending = remaining ? value >> (bits - remaining) : 0;
        writer->used = remaining;
    } else writer->used = total;
}

static void finish_bits(PackedWriter *writer)
{
    if (writer->used) store_word(writer->plane, writer->word, writer->pending, writer->bytes);
}

/* Row chunks support rectangular kernels, per-axis padding and valid output
 * shapes without assuming equal input/output widths or a word-sized row. */
static void forward_rows(const LutLayer *layer, const uint8_t *input, uint8_t *output)
{
    const LutLayerSpec *s = layer->spec;
    const ConvGeometry *g = &s->convolution;
    const uint32_t ih = s->input_shape.height, iw = s->input_shape.width;
    const uint32_t oh = s->output_shape.height, ow = s->output_shape.width;
    const uint32_t in_bytes = plane_bytes(ih * iw), out_bytes = plane_bytes(oh * ow);
    const uint32_t kernel_positions = g->kernel_h * g->kernel_w;
    uint32_t values[1u << (BS_MAX_INPUTS - 1u)], taps[BS_MAX_INPUTS];
    for (uint32_t k = 0; k < s->n_luts; ++k) {
        Lut lut;
        (void)lut_layer_get(layer, k, &lut);
        const uint8_t *planes[BS_MAX_INPUTS];
        int32_t rows[BS_MAX_INPUTS], columns[BS_MAX_INPUTS];
        for (uint32_t j = 0; j < lut.n_inputs; ++j) {
            const uint32_t index = lut_index(&lut, j), position = index % kernel_positions;
            planes[j] = input + (size_t)(index / kernel_positions) * in_bytes;
            rows[j] = (int32_t)(position / g->kernel_w) - (int32_t)g->padding_h;
            columns[j] = (int32_t)(position % g->kernel_w) - (int32_t)g->padding_w;
        }
        PackedWriter writer = {output + (size_t)k * out_bytes, out_bytes, 0, 0, 0};
        for (uint32_t y = 0; y < oh; ++y) {
            for (uint32_t x = 0; x < ow; x += BS_WORD_BITS) {
                const uint32_t chunk = ow - x < BS_WORD_BITS ? ow - x : BS_WORD_BITS;
                for (uint32_t j = 0; j < lut.n_inputs; ++j) {
                    const int32_t row = (int32_t)y + rows[j], col = (int32_t)x + columns[j];
                    int32_t lo = col < 0 ? -col : 0;
                    int32_t hi = (int32_t)iw - col;
                    if (hi > (int32_t)chunk) hi = (int32_t)chunk;
                    taps[j] = 0;
                    if ((uint32_t)row < ih && hi > lo) {
                        const uint32_t mask = low_mask((uint32_t)(hi - lo)) << (uint32_t)lo;
                        taps[j] = word_at(planes[j], row * (int32_t)iw + col, in_bytes) & mask;
                    }
                }
                append_bits(&writer, evaluate_word(&lut, taps, values) & low_mask(chunk), chunk);
            }
        }
        finish_bits(&writer);
    }
}

void bitslice_conv_forward(const LutLayer *layer, const uint8_t *input, uint8_t *output)
{
    const LutLayerSpec *s = layer->spec;
    const ConvGeometry *g = &s->convolution;
    if (g->kernel_h == 3u && g->kernel_w == 3u &&
        g->padding_h == 1u && g->padding_w == 1u &&
        s->input_shape.height == s->output_shape.height &&
        s->input_shape.width == s->output_shape.width)
        forward_same3(layer, input, output);
    else forward_rows(layer, input, output);
}

/* Map each byte to four OR results from adjacent bit pairs. Repeated lookup
 * compacts groups of 2, 4 or 8 columns for horizontal OR pooling. */
#define P4(b) ((uint8_t)((((b) | ((b) >> 1)) & 1u) \
    | (((((b) >> 2) | ((b) >> 3)) & 1u) << 1) \
    | (((((b) >> 4) | ((b) >> 5)) & 1u) << 2) \
    | (((((b) >> 6) | ((b) >> 7)) & 1u) << 3)))
#define P4_4(b) P4((b)+0), P4((b)+1), P4((b)+2), P4((b)+3)
#define P4_16(b) P4_4((b)+0), P4_4((b)+4), P4_4((b)+8), P4_4((b)+12)
#define P4_64(b) P4_16((b)+0), P4_16((b)+16), P4_16((b)+32), P4_16((b)+48)
static const uint8_t pool_pairs[256] = {P4_64(0), P4_64(64), P4_64(128), P4_64(192)};

int bitslice_pool_supported(TensorShape shape, PoolGeometry g)
{
    return (uint64_t)shape.height * shape.width <= BS_MAX_POSITIONS &&
        g.stride_w == g.kernel_w && g.kernel_w >= 2u && g.kernel_w <= 8u &&
        !(g.kernel_w & (g.kernel_w - 1u));
}

static uint32_t halve_columns(uint32_t value, uint32_t bits)
{
    uint32_t result = 0;
    for (uint32_t b = 0; b < (bits + 7u) / 8u; ++b)
        result |= (uint32_t)pool_pairs[(value >> (8u * b)) & 255u] << (4u * b);
    return result;
}

void bitslice_pool_forward(const uint8_t *input, TensorShape shape, PoolGeometry g, uint8_t *output)
{
    const uint32_t oh = (shape.height - g.kernel_h) / g.stride_h + 1u;
    const uint32_t ow = (shape.width - g.kernel_w) / g.stride_w + 1u;
    const uint32_t in_bytes = plane_bytes(shape.height * shape.width), out_bytes = plane_bytes(oh * ow);
    const uint32_t used_columns = ow * g.kernel_w;
    for (uint32_t c = 0; c < shape.channels; ++c) {
        const uint8_t *src = input + (size_t)c * in_bytes;
        PackedWriter writer = {output + (size_t)c * out_bytes, out_bytes, 0, 0, 0};
        for (uint32_t y = 0; y < oh; ++y) {
            for (uint32_t x = 0; x < used_columns; x += BS_WORD_BITS) {
                uint32_t bits = used_columns - x < BS_WORD_BITS ? used_columns - x : BS_WORD_BITS;
                uint32_t value = 0;
                for (uint32_t ky = 0; ky < g.kernel_h; ++ky)
                    value |= word_at(src, (int32_t)((y * g.stride_h + ky) * shape.width + x), in_bytes);
                value &= low_mask(bits);
                for (uint32_t factor = g.kernel_w; factor > 1u; factor /= 2u) {
                    value = halve_columns(value, bits);
                    bits /= 2u;
                }
                append_bits(&writer, value, bits);
            }
        }
        finish_bits(&writer);
    }
}
