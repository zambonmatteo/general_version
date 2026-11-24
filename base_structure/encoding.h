#ifndef ENCODING_H
#define ENCODING_H

#include "bitplane.h"

typedef enum {
    INPUT_UINT8, INPUT_FLOAT32
} InputType;

typedef enum {
    THRESHOLDS_SHARED, THRESHOLDS_CHANNEL, THRESHOLDS_ROW
} ThresholdLayout;

typedef struct {
    Shape raw_shape;
    InputType input_type;
    uint32_t bits;
    ThresholdLayout threshold_layout;
    uint32_t edge_channels;  /* Zero or ten monochrome edge channels. */
    const uint8_t *thermo_thresholds;
    const uint8_t *edge_thresholds;
} Encoding;

/* Parameters come from the generated model. Scalars use little-endian
 * float32; uint8 pixels are normalized by 255. Output is packed CHW.
 * Scratch needs height * width floats for edges; otherwise it may be null. */
void encoding_forward(const Encoding *encoding, const void *input,
                      uint8_t *output, float *scratch);

#endif
