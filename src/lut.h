#ifndef GENERAL_LUT_H
#define GENERAL_LUT_H

#include "model.h"

/* Read a packed table bit at address < 2^n_inputs; table/input packing is LSB-first. */
uint32_t lut_table_bit(const Lut *lut, uint32_t address);

/* Evaluate one FC LUT on a packed flat vector; mapper indices are absolute bit positions. */
uint32_t lut_eval_fc(const Lut *lut, const uint8_t *input);

/* Evaluate one convolution LUT at (output_y, output_x). Mapper order is
 * channel/kernel row/kernel column. CHW input planes are byte-padded separately;
 * spatial coordinates outside the input tensor contribute zero padding. */
uint32_t lut_eval_conv(const Lut *lut, const LutLayerSpec *spec,
                       const uint8_t *input, uint32_t output_y, uint32_t output_x);

/* Evaluators require validated LUT views and an input buffer matching the layer
 * shape. Convolution coordinates must be within the output shape, with the LUT
 * and descriptor from the same layer. Each evaluator returns one bit. */

/* Return packed bytes per input/output sample, or zero for an unbound layer or
 * unsupported layer type. Bound descriptors must remain immutable. */
size_t lut_layer_input_bytes(const LutLayer *layer);
size_t lut_layer_output_bytes(const LutLayer *layer);

ModelStatus lut_layer_forward(const LutLayer *layer,
                              const uint8_t *input, size_t input_bytes,
                              uint8_t *output, size_t output_bytes);

#endif
