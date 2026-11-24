#ifndef LAYER_H
#define LAYER_H

#include "bitplane.h"

typedef struct {
    uint32_t kernel_h, kernel_w;
    uint32_t padding_h, padding_w;
    uint32_t stride_h, stride_w;
} ConvParams;

typedef struct {
    uint32_t kernel_h, kernel_w;
    uint32_t stride_h, stride_w;
} PoolParams;

typedef struct {
    uint32_t classes;
    uint32_t bits_per_class;
    float tau;
} GroupSumParams;

typedef enum {
    LAYER_LUT_CONV, LAYER_LUT_FC, LAYER_OR_POOL, LAYER_FLATTEN, LAYER_GROUP_SUM
} LayerType;

/* Tables are consecutive packed rows; routing indices belong to the layer. */
typedef struct {
    const uint8_t *tables;
    const uint8_t *indices;
    uint32_t lut_count;
    uint32_t input_bits;
    uint32_t table_stride_bytes;
    uint32_t index_size_bytes;
    ConvParams convolution;  /* Zero for FC. */
} LutLayer;

typedef struct {
    LayerType type;
    Shape input_shape;
    Shape output_shape;  /* GroupSum describes class scores, not bits. */
    union {
        LutLayer lut;
        PoolParams pool;
        GroupSumParams groupsum;
    } parameters;
} Layer;

/* Parameters come from the validated generated model. Buffers are separate.
 * GroupSum uses layer_group_sum() because its result is a class index. */
void layer_forward(const Layer *layer, const uint8_t *input, uint8_t *output);

/* Read one mapper entry, accepting unaligned little-endian ARM data. */
uint32_t layer_read_index(const uint8_t *data, uint32_t size_bytes);

/* Requires validated class groups and a positive finite tau. */
uint32_t layer_group_sum(const uint8_t *input, GroupSumParams parameters);

#endif
