#ifndef GENERAL_LUT_H
#define GENERAL_LUT_H

#include "model.h"

/* Read the table bit at the computed address. */
uint32_t lut_table_bit(const Lut *lut, uint32_t address);

/* Use the selected input bits as a table address and return the stored bit. */
uint32_t lut_eval_fc(const Lut *lut, const uint8_t *input);

/* Use selected convolution-window bits as a table address and return its bit.
 * Currently unused. */
uint32_t lut_eval_conv(const Lut *lut, const LutLayerSpec *spec,
                       const uint8_t *input, uint32_t output_y, uint32_t output_x);

/* Compute the input bytes required by the layer. */
size_t lut_layer_input_bytes(const LutLayer *layer);

/* Compute the output bytes required by the layer. */
size_t lut_layer_output_bytes(const LutLayer *layer);

/* Check buffers, clear the output and execute the LUT layer. */
ModelStatus lut_layer_forward(const LutLayer *layer,
                              const uint8_t *input, size_t input_bytes,
                              uint8_t *output, size_t output_bytes);

#endif
