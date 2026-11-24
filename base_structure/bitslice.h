#ifndef BITSLICE_H
#define BITSLICE_H

#include "layer.h"

/* Up to 32 spatial positions of one sample are evaluated together.
 * Larger/unsupported configurations keep the scalar layer implementation. */
#define BS_WORD_BITS 32u
#define BS_MAX_INPUTS 8u
#define BS_MAX_POSITIONS 1024u

int bitslice_conv_supported(const Layer *layer);
int bitslice_pool_supported(Shape shape, PoolParams geometry);
void bitslice_conv_forward(const Layer *layer, const uint8_t *input, uint8_t *output);
void bitslice_pool_forward(const uint8_t *input, Shape shape,
                           PoolParams geometry, uint8_t *output);

#endif
