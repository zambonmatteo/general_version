#ifndef GENERAL_ENCODE_H
#define GENERAL_ENCODE_H
#include <stddef.h>
#include <stdint.h>

/* layout: 0=shared, 1=channel, 2=row. scratch: H*W floats if edges=1. */
void encode_input(const void *input, int float_input, size_t channels,
                  size_t height, size_t width, size_t bits, int layout,
                  const float *thresholds, int edges,
                  const float *edge_thresholds, float *scratch, uint8_t *out);
#endif
