#ifndef GENERAL_MODEL_H
#define GENERAL_MODEL_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t channels;
    uint32_t height;
    uint32_t width;
} TensorShape;

/* A view of one LUT. Both pointers refer to the immutable model blob.
 * Input j contributes bit j of the table address (least significant first).
 * The mapper's interpretation belongs to the layer, not to this view. */
typedef struct {
    uint32_t n_inputs;
    uint32_t index_bytes;
    const uint8_t *table;
    const uint8_t *indices;
} Lut;

typedef enum {
    LUT_FULLY_CONNECTED,
    LUT_CONVOLUTION
} LutLayerType;

typedef struct {
    uint32_t offset;
    uint32_t bytes;
} BlobSection;

typedef struct {
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t padding_h;
    uint32_t padding_w;
    uint32_t stride_h;
    uint32_t stride_w;
} ConvGeometry;

/* Generated from JSON on the PC; constant descriptors use read-only storage.
 * FC vectors use shape {1, 1, number_of_bits}; convolution uses CHW.
 * FC convolution parameters are zero and are never used. */
typedef struct {
    LutLayerType type;
    uint32_t n_luts;
    uint32_t n_inputs;
    uint32_t index_bytes;
    BlobSection tables;
    BlobSection indices;
    /* No IDs for one shared table or one table per LUT; otherwise IDs are
     * unsigned little-endian values selecting rows in the table pool. */
    uint32_t n_tables;
    uint32_t table_id_bytes;
    BlobSection table_ids;
    TensorShape input_shape;
    TensorShape output_shape;
    ConvGeometry convolution;
} LutLayerSpec;

/* Only the bound pointers need RAM: shared parameters stay in the spec.
 * The spec and blob must outlive this layer and all Lut views obtained from it. */
typedef struct {
    const LutLayerSpec *spec;
    const uint8_t *tables;
    const uint8_t *indices;
    const uint8_t *table_ids;
} LutLayer;

typedef struct {
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
} PoolGeometry;

typedef struct {
    uint32_t classes;
    float tau;  /* Positive finite float32 divisor for class population counts. */
} GroupSumHead;

typedef enum {
    OP_LUT,
    OP_OR_POOL,
    OP_FLATTEN,
    OP_GROUP_SUM
} OperationType;

/* The execution sequence is generated on the PC. GroupSum's output shape
 * describes class scores, but only the winning class is returned: scores
 * are not materialized as a packed tensor or a floating-point array. */
typedef struct {
    OperationType type;
    TensorShape input_shape;
    TensorShape output_shape;
    union {
        const LutLayerSpec *lut;
        PoolGeometry pool;
        GroupSumHead head;
    } parameters;
} Operation;

typedef struct {
    size_t blob_bytes;
    TensorShape input_shape;
    const Operation *operations;
    uint32_t operation_count;
} ModelSpec;

typedef struct {
    const ModelSpec *spec;
    const uint8_t *blob;
    size_t blob_bytes;
    size_t activation_bytes;  /* Capacity of each of the two working buffers. */
} Model;

typedef enum {
    MODEL_OK = 0,
    MODEL_INVALID_ARGUMENT,
    MODEL_INVALID_SPEC,
    MODEL_INVALID_BLOB,
    MODEL_INVALID_INDEX
} ModelStatus;

/* Check the geometry, byte ranges and mapper indices, then bind without copying.
 * Call once before inference timing. On failure, layer is cleared.
 * Indices in the blob are unsigned little-endian uint16 or uint32. */
ModelStatus lut_layer_bind(LutLayer *layer, const LutLayerSpec *spec,
                           const uint8_t *blob, size_t blob_bytes);

/* Populate a LUT view with pointers to its table and routing indices.
 * The layer must be bound and k must be below its LUT count.
 * On failure, lut is cleared. */
ModelStatus lut_layer_get(const LutLayer *layer, uint32_t k, Lut *lut);

/* Read one mapper entry without assuming pointer alignment or host endianness.
 * lut must be a valid view and j must be smaller than lut->n_inputs. */
uint32_t lut_index(const Lut *lut, uint32_t j);

/* Validate the complete graph and every mapper once, before inference timing.
 * Descriptors and blob are immutable and must outlive the initialized model.
 * A failed initialization clears model. */
ModelStatus model_init(Model *model, const ModelSpec *spec,
                       const uint8_t *blob, size_t blob_bytes);

size_t model_workspace_bytes(const Model *model);

/* Execute one already encoded sample, returning its predicted class.
 * Workspace is caller-owned writable memory of model_workspace_bytes().
 * Workspace, input and prediction must not overlap each other or overwrite
 * the immutable model data/descriptors. No allocation or input encoding occurs.
 * The output prediction remains unchanged on error. */
ModelStatus model_forward(const Model *model, const uint8_t *input, size_t input_bytes,
                          uint8_t *workspace, size_t workspace_bytes,
                          uint32_t *prediction);

#endif
