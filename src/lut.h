#ifndef GENERAL_LUT_H
#define GENERAL_LUT_H

#include "model.h"

/* Read a packed table bit at address < 2^n_inputs; table/input packing is LSB-first. */
uint32_t lut_table_bit(const Lut *lut, uint32_t address);

/* Evaluate one FC LUT on a packed flat vector; mapper indices are absolute bit positions. */
uint32_t lut_eval_fc(const Lut *lut, const uint8_t *input);

/* Evaluate one convolution LUT at (output_y, output_x). Mapper order is
 * channel/kernel row/kernel column. CHW input planes are byte-padded separately;
 * out-of-bounds spatial reads return zero, matching the reference model. */
uint32_t lut_eval_conv(const Lut *lut, const LutLayerSpec *spec,
                       const uint8_t *input, uint32_t output_y, uint32_t output_x);

/* Evaluators require bound views/spec-sized input; conv uses in-range coordinates
 * and same-layer LUT/spec. Layer checks at init only; return one bit, input/blob unchanged. */

/* Bytes/sample (not batch); zero if unbound/unsupported. Keep bound descriptors
 * immutable; inference skips dimension checks. */
size_t lut_layer_input_bytes(const LutLayer *layer);
size_t lut_layer_output_bytes(const LutLayer *layer);

/* Run a bound layer: FC -> flat bits; conv -> CHW, byte-padded per channel.
 * Clear output before writing; padding stays zero. Larger buffers allowed,
 * tail bytes untouched. Conv routing decoded once/LUT, reused via bounded local
 * cache; no persistent cache or dynamic allocation. Caller-owned buffers cannot
 * overlap; writable output must be separate from immutable blob/descriptors.
 * Invalid args or short/overlapping buffers: MODEL_INVALID_ARGUMENT, output unchanged. */
ModelStatus lut_layer_forward(const LutLayer *layer,
                              const uint8_t *input, size_t input_bytes,
                              uint8_t *output, size_t output_bytes);

#endif
