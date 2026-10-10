/* Helpers and operations for packed binary tensors: storage sizes, shape and
 * buffer checks, OR pooling, CHW flattening and GroupSum class selection.
 * Implementations are in tensor.c. */

#ifndef GENERAL_TENSOR_H
#define GENERAL_TENSOR_H

#include "model.h"

/* Binary CHW storage: every channel is independently byte-padded, LSB first.
 * Invalid or overflowing shapes return zero. Flat vectors use {1, 1, bits}. */
size_t tensor_bytes(TensorShape shape);
int tensor_shape_equal(TensorShape a, TensorShape b);
int tensor_ranges_overlap(const void *a, size_t a_bytes, const void *b, size_t b_bytes);

ModelStatus tensor_pool_shape(TensorShape input, PoolGeometry geometry, TensorShape *output);

/* OR pooling is binary max pooling, with explicit rectangular kernel/stride,
 * no spatial padding and floor output geometry. Partial trailing windows are
 * discarded. Input and output must be separate caller-owned buffers. */
ModelStatus tensor_or_pool(const uint8_t *input, size_t input_bytes, TensorShape shape,
                           PoolGeometry geometry, uint8_t *output, size_t output_bytes);

/* Repack CHW planes into one flat vector, removing per-channel byte padding
 * while preserving channel/row/column bit order. */
ModelStatus tensor_flatten(const uint8_t *input, size_t input_bytes, TensorShape shape,
                           uint8_t *output, size_t output_bytes);

/* Sum consecutive equal-size class groups, divide each count by float32 tau
 * and return argmax. Equal float32 scores choose the lowest class index,
 * including ties caused by float32 rounding or division overflow. */
ModelStatus tensor_group_argmax(const uint8_t *input, size_t input_bytes,
                                uint32_t bits, uint32_t classes, float tau,
                                uint32_t *prediction);

#endif
