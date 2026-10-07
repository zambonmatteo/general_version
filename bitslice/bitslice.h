#ifndef GENERAL_BITSLICE_H
#define GENERAL_BITSLICE_H

#include "model.h"

/* The fast kernels operate on 32 spatial positions of ONE encoded sample.
 * Bounds keep their temporary storage small. Unsupported layers use scalar
 * evaluation; these limits do not limit the model format. */
#define BS_WORD_BITS 32u
#define BS_MAX_INPUTS 8u
#define BS_MAX_POSITIONS 1024u

int bitslice_conv_supported(const LutLayerSpec *spec);
int bitslice_pool_supported(TensorShape shape, PoolGeometry geometry);

/* Internal kernels: callers have validated descriptors, non-overlapping
 * buffers and capacities, and cleared the required output bytes. Call only
 * after the corresponding support predicate succeeds. */
void bitslice_conv_forward(const LutLayer *layer, const uint8_t *input, uint8_t *output);
void bitslice_pool_forward(const uint8_t *input, TensorShape shape,
                           PoolGeometry geometry, uint8_t *output);

#endif
